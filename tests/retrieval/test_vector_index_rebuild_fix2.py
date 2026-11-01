"""AC-40 coverage gap: a **real** vector-index rebuild from stored facts.

Codex round-2: "projection rebuild 用例不能证明 vector index rebuild", and F11
showed stale vectors can still be written by a late inline embedding. This
module therefore exercises the actual ``claim_vectors`` generation lifecycle:

* 旧向量不能指向新正文 - after a correction changes the display text, the cached
  old vector is never served for the new text (text-hash generation check), and a
  hand-crafted stale row is ignored;
* pending 可查并自动补算 - a missing vector is visible as pending work in the
  status pipeline and is filled by the real worker / bounded inline collection;
* 混用维度明确拒绝 - vectors from another dimension generation are never loaded or
  ranked together, and writing a wrong-length vector is rejected;
* 索引可从事实重建 - a snapshot import reports ``rebuild_required.vectors``,
  rebuilds nothing during ``rebuild_projection``, and the vector index is then
  reconstructed from the imported claim revisions through the real embed job
  queue + worker (not from the projection only).

Frozen expected (docs/baseline-20261030/acceptance-cases.json AC-40):
"旧向量不能指向新正文；pending可查并自动补算；混用维度明确拒绝；索引可从事实重建。"
"""

from __future__ import annotations

import pytest

from tests.retrieval.conftest import recall_request
from tests.state.conftest import record
from tests.store.conftest import make_event, make_batch

from knowledge_weaver.embeddings.index import (
    HashEmbeddingProvider,
    configure_model,
    load_vectors,
    run_embed_worker_once,
    store_vector,
    text_hash,
)
from knowledge_weaver.jobs import enqueue, status_counts
from knowledge_weaver.portability import export_snapshot, import_snapshot
from knowledge_weaver.retrieval import execute_recall
from knowledge_weaver.retrieval.embeddings_index import collect_vectors
from knowledge_weaver.store import Store, rebuild_projection, receive_batch, status as status_view

BINDING = {
    "client_id": "client:test",
    "workspace_id": "owner:demo",
    "project_ids": ["project:songta"],
    "actor_ids": ["*"],
}
MODEL_ID = "test:ac40-hash"
GEN = {"model_revision": "1", "normalization": "l2"}


@pytest.fixture
def store(tmp_path):
    configure_model(None)
    return Store(tmp_path / "kw" / "knowledge.db")


def _provider(dimension=4, model_id=MODEL_ID):
    return HashEmbeddingProvider(dimension=dimension, model_id=model_id)


def _load(store, revision_ids, provider):
    return load_vectors(
        store,
        set(revision_ids),
        model_id=provider.model_id,
        model_revision=provider.model_revision,
        normalization=provider.normalization,
        dimension=int(provider.dimension),
    )


def _vectors(store, revision_id):
    with store.read() as conn:
        return [
            dict(row)
            for row in conn.execute(
                "SELECT revision_id, text_hash, model_id, model_revision, dimension, normalization "
                "FROM claim_vectors WHERE revision_id = ? ORDER BY dimension",
                (revision_id,),
            ).fetchall()
        ]


def _pipeline(store, operation):
    return next(entry for entry in status_view(store)["pipelines"] if entry["operation"] == operation)


def _recall(store, query, *, embedder=None, intent="current"):
    return execute_recall(
        store, recall_request(query=query, intent=intent), binding=BINDING, embedder=embedder
    )


def test_old_vector_never_points_to_corrected_text(store):
    """Clause: 旧向量不能指向新正文."""
    provider = _provider()
    first = record(store, display_text="旧端口4100", value=4100)
    collect_vectors(store, {first["revision_id"]}, provider)
    assert len(_vectors(store, first["revision_id"])) == 1

    corrected = record(
        store,
        display_text="新端口4200",
        value=4200,
        supersedes=[first["revision_id"]],
        event_time={"value": "2026-10-30T11:00:00Z", "precision": "instant", "timezone": "UTC"},
    )
    # The new revision text has no vector yet: the old vector must never be
    # served for the new text.
    assert _load(store, {corrected["revision_id"]}, provider) == {}

    # Even a stale row (e.g. from a crash between write and text change) that
    # claims the new revision id but carries the old text hash is ignored.
    from knowledge_weaver.embeddings.index import _pack, sqlite_blob

    with store.transaction() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO claim_vectors "
            "(revision_id, text_hash, model_id, model_revision, dimension, normalization, vector, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                corrected["revision_id"],
                text_hash("旧端口4100"),
                provider.model_id,
                provider.model_revision,
                provider.dimension,
                provider.normalization,
                sqlite_blob(_pack(_provider().embed(["旧端口4100"])[0])),
                "2026-10-30T00:00:00Z",
            ),
        )
    assert _load(store, {corrected["revision_id"]}, provider) == {}, "stale text hash must be rejected"

    # Real auto-recompute fills the new generation from the new text.
    collect_vectors(store, {corrected["revision_id"]}, provider)
    loaded = _load(store, {corrected["revision_id"]}, provider)
    assert list(loaded) == [corrected["revision_id"]]
    with store.read() as conn:
        row = conn.execute(
            "SELECT text_hash FROM claim_vectors WHERE revision_id = ? AND dimension = ?",
            (corrected["revision_id"], provider.dimension),
        ).fetchone()
    assert row["text_hash"] == text_hash("新端口4200")

    # The corrected fact is retrieved with the new text/value; the old value is
    # not the current truth anymore.
    result = _recall(store, "端口4200", embedder=provider)
    assert result["status"] == "ok", result
    assert [item["value"] for item in result["items"]] == [4200], result["items"]
    old_result = _recall(store, "旧端口4100", embedder=provider)
    assert all(item["value"] != 4100 for item in old_result["items"]), old_result["items"]


def test_pending_embedding_is_visible_and_automatically_computed(store, tmp_path):
    """Clause: pending可查并自动补算."""
    provider = _provider()
    configure_model(MODEL_ID, model_revision=provider.model_revision, dimension=provider.dimension)
    recorded = record(store, display_text="端口4200", value=4200)

    pending = _pipeline(store, "embed")
    assert pending["pending"] == 1, pending
    assert pending["state"] == "idle", pending
    assert pending["oldest_pending_at"] is not None, pending
    assert pending["last_success_at"] is None, pending
    assert pending["last_error_code"] is None, pending
    assert _vectors(store, recorded["revision_id"]) == []

    # The recall path fills a bounded missing vector automatically (inline
    # collection) from the stored fact text, and does not degrade to keyword-only.
    inline = _recall(store, "端口4200", embedder=provider)
    assert inline["status"] == "ok", inline
    assert "embedding_unavailable" not in inline["incomplete_reasons"], inline
    assert len(_vectors(store, recorded["revision_id"])) == 1

    # The durable embed job is still pending until a worker processes it, then
    # the pipeline records last_success and clears the queue depth.
    outcome = run_embed_worker_once(store, provider)
    assert outcome["state"] == "succeeded", outcome
    idle = _pipeline(store, "embed")
    assert idle["pending"] == 0 and idle["state"] == "idle", idle
    assert idle["last_success_at"] is not None, idle

    # A restart (fresh Store instance on the same file) keeps both the fact and
    # the computed vector usable: no re-embed, no lost state.
    restarted = Store(store.db_path)
    assert _pipeline(restarted, "embed")["last_success_at"] is not None
    assert _load(restarted, {recorded["revision_id"]}, provider), "vector must survive restart"
    assert _recall(restarted, "端口4200", embedder=provider)["items"]


def test_mixed_dimension_generations_are_rejected_not_mixed(store):
    """Clause: 混用维度明确拒绝."""
    provider_2d = _provider(dimension=2, model_id=MODEL_ID)
    provider_3d = _provider(dimension=3, model_id=MODEL_ID)
    recorded = record(store, display_text="端口4200", value=4200)
    collect_vectors(store, {recorded["revision_id"]}, provider_2d)

    assert _vectors(store, recorded["revision_id"])[0]["dimension"] == 2
    # A different-dimension generation never mixes with the stored one.
    assert _load(store, {recorded["revision_id"]}, provider_3d) == {}
    # Writing a vector whose length does not match the provider is rejected.
    with pytest.raises(ValueError):
        store_vector(
            store,
            revision_id=recorded["revision_id"],
            provider=provider_3d,
            vector=[1.0, 0.0],
            display_text="端口4200",
        )
    # The new dimension becomes its own generation (the cache row is keyed by
    # revision, so switching generations replaces the old vector instead of
    # mixing vectors of different dimensions in one load).
    collect_vectors(store, {recorded["revision_id"]}, provider_3d)
    rows = _vectors(store, recorded["revision_id"])
    assert [row["dimension"] for row in rows] == [3], rows
    assert _load(store, {recorded["revision_id"]}, provider_2d) == {}, "old dimension must not be mixed"
    assert _load(store, {recorded["revision_id"]}, provider_3d)
    assert _recall(store, "端口4200", embedder=provider_3d)["items"]


def test_vector_index_is_rebuilt_from_facts_after_snapshot_import(store, tmp_path):
    """Clause: 索引可从事实重建 (not just the projection)."""
    provider = _provider()
    configure_model(MODEL_ID, model_revision=provider.model_revision, dimension=provider.dimension)
    recorded = []
    for index in range(3):
        recorded.append(
            record(
                store,
                display_text=f"端口{4100 + index}",
                value=4100 + index,
                predicate=f"server.port.{index}",
                native_id=f"message:ac40-rebuild-{index}",
            )
        )
    # Build vectors once with the real worker so the snapshot has a full index.
    while run_embed_worker_once(store, provider)["claimed"]:
        pass
    built = len(_vectors(store, recorded[0]["revision_id"]))
    assert built == 1

    snapshot = tmp_path / "snapshot"
    export_snapshot(store, snapshot)
    report = import_snapshot(snapshot, (tmp_path / "imported").resolve())
    assert report["rebuild_required"]["vectors"] is True, report
    imported = Store(report["target_db"])
    with imported.read() as conn:
        revision_ids = [
            row["revision_id"]
            for row in conn.execute("SELECT revision_id FROM claim_revisions ORDER BY revision_id")
        ]
        assert len(revision_ids) == 3
        assert conn.execute("SELECT COUNT(*) AS c FROM claim_vectors").fetchone()["c"] == 0

    # A projection-only rebuild is explicitly NOT a vector index rebuild.
    assert rebuild_projection(imported) == 3
    with imported.read() as conn:
        assert conn.execute("SELECT COUNT(*) AS c FROM claim_vectors").fetchone()["c"] == 0

    # Rebuild the vector index from the imported facts/revisions through the
    # real persistent embed job queue + worker (new generation version).
    for revision_id in revision_ids:
        enqueue(imported, "embed", revision_id, input_version=2)
    processed = 0
    while True:
        outcome = run_embed_worker_once(imported, provider)
        if not outcome["claimed"]:
            break
        assert outcome["state"] == "succeeded", outcome
        processed += 1
    assert processed == 3, status_counts(imported)

    with imported.read() as conn:
        rebuilt = conn.execute("SELECT COUNT(*) AS c FROM claim_vectors").fetchone()["c"]
    assert rebuilt == 3, rebuilt
    for revision_id in revision_ids:
        assert _load(imported, {revision_id}, provider), revision_id
    recall = _recall(imported, "端口4100", embedder=provider)
    assert recall["status"] == "ok", recall
    assert recall["items"], recall
    assert all(item["state"] == "current" for item in recall["items"]), recall["items"]
