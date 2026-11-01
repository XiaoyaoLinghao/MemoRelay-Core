"""F11-R3 / AC-40: a retained conclusion must be able to rebuild its vector.

Migrated from the Codex round-3 followup counterexample
``test_F11_valid_retained_conclusion_can_rebuild_vector_after_source_expiry``:
retention expired only the cited *source detail* while the conclusion body stays
retained, and the shared late-commit guard used to refuse rebuilding because the
evidence links were no longer ``full``.  The fix tolerates retention expiry of
the source detail for a still-retained revision, while every invalidation of the
revision itself (redact/withheld, retract, expiry, superseded, tombstone, text
change) keeps rejecting late results inside the same write transaction.

Only synthetic pytest ``tmp_path`` data is used; no network, no real host data.
"""

from __future__ import annotations

import pytest

from knowledge_weaver.embeddings.index import (
    HashEmbeddingProvider,
    configure_model,
    run_embed_worker_once,
)
from knowledge_weaver.lifecycle import classify_claim_revisions, register_lifecycle, sweep
from knowledge_weaver.portability import export_snapshot, import_snapshot
from knowledge_weaver.retrieval import execute_recall
from knowledge_weaver.retrieval.embeddings_index import collect_vectors
from knowledge_weaver.store import Store, current_items, fetch_source, receive_batch

from ..state.conftest import record
from ..store.conftest import make_batch, make_event
from .conftest import recall_request

SOURCE_ANCHOR = "2020-01-01T00:00:00Z"
AFTER_EXPIRY = "2026-10-31T00:00:00Z"


def _vector_count(store, revision_id: str) -> int:
    with store.read() as conn:
        return conn.execute(
            "SELECT COUNT(*) AS c FROM claim_vectors WHERE revision_id = ?", (revision_id,)
        ).fetchone()["c"]


def _revision_row(store, revision_id: str):
    with store.read() as conn:
        return conn.execute(
            "SELECT display_text, status FROM claim_revisions WHERE revision_id = ?",
            (revision_id,),
        ).fetchone()


def _retained_conclusion(store):
    """A still-retained conclusion whose cited source detail expires under TTL."""
    event = make_event(
        native_id="message:fix3-retained", text="来源过程详情：端口配置4200"
    )
    receive_batch(store, make_batch([event]))
    conclusion = record(
        store, display_text="端口配置4200", value=4200, evidence_event=event
    )
    register_lifecycle(
        store,
        subject_kind="event_part",
        subject_id=f"{event['event_id']}#1@0",
        class_="completed_task",
        business_anchor=SOURCE_ANCHOR,
    )
    swept = sweep(store, now=AFTER_EXPIRY)
    assert swept["event_parts"] == 1, swept
    # The conclusion stays explainable but its source detail is gone.
    item = next(
        item
        for item in current_items(store, workspace_id="owner:demo")
        if item["revision_id"] == conclusion["revision_id"]
    )
    assert item["value"] == 4200 and item["source_detail_expired"] is True, item
    source = fetch_source(store, event["event_id"], 1, 0)
    assert source["availability"] == "expired" and source["text"] is None, source
    return event, conclusion


class _ActDuringEmbed(HashEmbeddingProvider):
    """Run one real concurrent store operation while the provider is embedding."""

    def __init__(self, action, *, dimension: int = 3) -> None:
        super().__init__(dimension=dimension)
        self._action = action
        self.fired = False

    def embed(self, texts):
        if not self.fired:
            self.fired = True
            self._action()
        return super().embed(texts)


def _redact_event(event: dict) -> dict:
    return make_event(
        native_id=event["source"]["native_event_id"],
        text="synthetic sanitized",
        revision=2,
        operation="redact",
        parent_revision=1,
    )


def _retract_event(event: dict) -> dict:
    return make_event(
        native_id=event["source"]["native_event_id"],
        text="",
        revision=2,
        operation="retract",
        parent_revision=1,
    )


# ---------------------------------------------------------------------------
# Positive: retained conclusion rebuilds from its own retained body
# ---------------------------------------------------------------------------


def test_F11_retained_conclusion_rebuilds_vector_after_source_detail_expiry(store):
    """Codex counterexample: source expired, conclusion retained -> rebuild works."""
    _event, conclusion = _retained_conclusion(store)
    revision_id = conclusion["revision_id"]
    assert _vector_count(store, revision_id) == 0

    rebuilt = collect_vectors(store, {revision_id}, HashEmbeddingProvider(dimension=3))
    assert revision_id in rebuilt, rebuilt
    assert _vector_count(store, revision_id) == 1
    # The rebuilt index is actually usable and still reports the expired source.
    result = execute_recall(
        store,
        recall_request(query="端口配置4200"),
        binding={
            "client_id": "client:test",
            "workspace_id": "owner:demo",
            "project_ids": ["project:songta"],
            "actor_ids": ["*"],
        },
        embedder=HashEmbeddingProvider(dimension=3),
    )
    items = [item for item in result["items"] if item["revision_id"] == revision_id]
    assert items and items[0]["source_detail_expired"] is True, result["items"]


def test_F11_retained_conclusion_rebuilds_vector_after_snapshot_restore(store, tmp_path):
    """Index loss via restore: the imported retained conclusion rebuilds too."""
    _event, conclusion = _retained_conclusion(store)
    revision_id = conclusion["revision_id"]
    snapshot = tmp_path / "snapshot"
    export_snapshot(store, snapshot)
    report = import_snapshot(snapshot, tmp_path / "restored")
    restored = Store(report["target_db"])
    # Restore applied the retention clock: the source detail is still expired.
    with restored.read() as conn:
        row = conn.execute(
            "SELECT availability FROM evidence_links WHERE revision_id = ?", (revision_id,)
        ).fetchone()
    assert row["availability"] == "expired", dict(row)
    rebuilt = collect_vectors(restored, {revision_id}, HashEmbeddingProvider(dimension=3))
    assert revision_id in rebuilt, rebuilt


def test_F11_embed_worker_rebuilds_retained_conclusion_after_source_expiry(store):
    """The shared guard change applies to the durable embed worker as well."""
    configure_model("test:ac40-fix3", model_revision="1", dimension=2)
    _event, conclusion = _retained_conclusion(store)
    revision_id = conclusion["revision_id"]
    assert _vector_count(store, revision_id) == 0
    result = run_embed_worker_once(
        store, HashEmbeddingProvider(dimension=2, model_id="test:ac40-fix3")
    )
    assert result["state"] == "succeeded", result
    assert _vector_count(store, revision_id) == 1


# ---------------------------------------------------------------------------
# Negative: invalidations of the revision itself keep rejecting late results
# ---------------------------------------------------------------------------


def test_F11_late_redact_during_rebuild_is_still_rejected(store):
    event, conclusion = _retained_conclusion(store)
    revision_id = conclusion["revision_id"]
    embedder = _ActDuringEmbed(
        lambda: receive_batch(
            store, make_batch([_redact_event(event)], batch_id="batch:fix3:redact")
        )
    )
    rebuilt = collect_vectors(store, {revision_id}, embedder)
    assert embedder.fired is True
    assert revision_id not in rebuilt, rebuilt
    assert _vector_count(store, revision_id) == 0
    # The real redaction scrubbed the derived body: no retained text remains.
    assert _revision_row(store, revision_id)["display_text"] == ""


def test_F11_late_retract_during_rebuild_is_still_rejected(store):
    event, conclusion = _retained_conclusion(store)
    revision_id = conclusion["revision_id"]
    embedder = _ActDuringEmbed(
        lambda: receive_batch(
            store, make_batch([_retract_event(event)], batch_id="batch:fix3:retract")
        )
    )
    rebuilt = collect_vectors(store, {revision_id}, embedder)
    assert embedder.fired is True
    assert revision_id not in rebuilt, rebuilt
    assert _vector_count(store, revision_id) == 0
    assert _revision_row(store, revision_id)["status"] == "revoked"


def test_F11_redacted_conclusion_cannot_be_rebuilt_after_source_expiry(store):
    """A redaction that lands after retention expiry still forbids a rebuild."""
    event, conclusion = _retained_conclusion(store)
    revision_id = conclusion["revision_id"]
    receive_batch(store, make_batch([_redact_event(event)], batch_id="batch:fix3:late-redact"))
    rebuilt = collect_vectors(store, {revision_id}, HashEmbeddingProvider(dimension=3))
    assert revision_id not in rebuilt, rebuilt
    assert _vector_count(store, revision_id) == 0


def test_F11_retracted_conclusion_cannot_be_rebuilt_after_source_expiry(store):
    event, conclusion = _retained_conclusion(store)
    revision_id = conclusion["revision_id"]
    receive_batch(store, make_batch([_retract_event(event)], batch_id="batch:fix3:late-retract"))
    rebuilt = collect_vectors(store, {revision_id}, HashEmbeddingProvider(dimension=3))
    assert revision_id not in rebuilt, rebuilt
    assert _vector_count(store, revision_id) == 0


def test_F11_expired_evidence_does_not_revive_superseded_revision(store):
    """Expired *source detail* tolerance never resurrects a superseded revision."""
    event = make_event(native_id="message:fix3-superseded", text="旧端口4100过程详情")
    receive_batch(store, make_batch([event]))
    old = record(store, display_text="旧端口4100", value=4100, evidence_event=event)
    new = record(
        store,
        display_text="新端口4200",
        value=4200,
        supersedes=[old["revision_id"]],
        event_time={"value": "2025-01-01T00:00:00Z", "precision": "instant", "timezone": "UTC"},
    )
    register_lifecycle(
        store,
        subject_kind="event_part",
        subject_id=f"{event['event_id']}#1@0",
        class_="completed_task",
        business_anchor=SOURCE_ANCHOR,
    )
    # This clock expires the 2020 source detail but not the 2025 replacement
    # history (which would only expire in 2026), so the old revision is still
    # merely superseded when the rebuild is attempted.
    sweep(store, now="2020-07-01T00:00:00Z")
    assert _revision_row(store, old["revision_id"])["status"] == "superseded"
    rebuilt = collect_vectors(store, {old["revision_id"]}, HashEmbeddingProvider(dimension=3))
    assert rebuilt == {}, rebuilt
    assert _vector_count(store, old["revision_id"]) == 0
    # The superseding retained revision still rebuilds normally.
    kept = collect_vectors(store, {new["revision_id"]}, HashEmbeddingProvider(dimension=3))
    assert new["revision_id"] in kept, kept


def test_F11_expired_revision_itself_cannot_be_rebuilt(store):
    """A revision expired by a real sweep is scrubbed and stays unrebuildable."""
    first = record(
        store,
        display_text="旧端口4100",
        value=4100,
        event_time={"value": "2024-01-01T00:00:00Z", "precision": "instant", "timezone": "UTC"},
    )
    record(
        store,
        display_text="新端口4200",
        value=4200,
        supersedes=[first["revision_id"]],
        event_time={"value": "2024-06-01T00:00:00Z", "precision": "instant", "timezone": "UTC"},
    )
    with store.transaction() as conn:
        conn.execute(
            "UPDATE claim_revisions SET status = 'superseded', status_changed_at = ? "
            "WHERE revision_id = ?",
            ("2024-06-02T00:00:00Z", first["revision_id"]),
        )
    classify_claim_revisions(store, now="2025-01-01T00:00:00Z")
    sweep(store, now="2030-01-01T00:00:00Z")
    row = _revision_row(store, first["revision_id"])
    assert row["status"] == "expired" and row["display_text"] == "", dict(row)
    rebuilt = collect_vectors(store, {first["revision_id"]}, HashEmbeddingProvider(dimension=3))
    assert rebuilt == {}, rebuilt
    assert _vector_count(store, first["revision_id"]) == 0
