"""F02 regression: an explicit redact must remove *derived* secret payloads.

The source body purge alone used to leave the old fact text/value in
``claim_revisions`` and the FTS index, so recall and the portable export still
leaked a newly-classified secret. All canaries below are synthetic.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from knowledge_weaver import contracts
from knowledge_weaver.embeddings.index import (
    HashEmbeddingProvider,
    configure_model,
    enqueue_embed_in_transaction,
    load_vectors,
    run_embed_worker_once,
)
from knowledge_weaver.portability import export_snapshot
from knowledge_weaver.portability.export import ExportRefused, export_snapshot as export_snapshot_fn
from knowledge_weaver.retrieval import execute_recall
from knowledge_weaver.store import receive_batch

from ..retrieval.conftest import recall_request
from ..store.conftest import make_batch, make_event
from .conftest import record


CANARY = "T20_NEWLY_CLASSIFIED_SECRET_7129"


def _binding():
    return {
        "client_id": "client:test",
        "workspace_id": "owner:demo",
        "project_ids": ["project:songta"],
        "actor_ids": ["*"],
    }


def _redact_revision(event, *, text="历史配置 [REDACTED:PASSWORD_ASSIGNMENT]"):
    return make_event(
        native_id=event["source"]["native_event_id"],
        revision=2,
        operation="redact",
        parent_revision=1,
        text=text,
        content_revision=event["body"]["content_revision"],
    )


@pytest.fixture
def redacted_store(store):
    event = make_event(native_id="message:f02", text="历史配置 密码: " + CANARY)
    receive_batch(store, make_batch([event]))
    record(
        store,
        display_text=event["body"]["text"],
        value=CANARY,
        evidence_event=event,
    )
    configure_model("test:hash-embedding", dimension=4)
    try:
        with store.transaction() as conn:
            revision_id = conn.execute("SELECT revision_id FROM claim_revisions").fetchone()[
                "revision_id"
            ]
            enqueue_embed_in_transaction(conn, revision_id)
        assert run_embed_worker_once(
            store, HashEmbeddingProvider(dimension=4, model_id="test:hash-embedding")
        )["claimed"] == 1
        assert load_vectors(
            store, {revision_id}, model_id="test:hash-embedding", dimension=4
        ), "the vector must exist before the redaction"
        result = receive_batch(store, make_batch([_redact_revision(event)], batch_id="batch:redact"))
        assert result["receipts"][0]["disposition"] == "accepted"
        return store, event, revision_id
    finally:
        configure_model(None)


def test_redact_scrubs_derived_text_value_index_and_vectors(redacted_store):
    store, event, revision_id = redacted_store
    with store.read() as conn:
        revision = conn.execute(
            "SELECT display_text, value_json FROM claim_revisions WHERE revision_id = ?",
            (revision_id,),
        ).fetchone()
        fts = conn.execute(
            "SELECT COUNT(*) AS c FROM claim_fts WHERE revision_id = ?", (revision_id,)
        ).fetchone()["c"]
        vectors = conn.execute(
            "SELECT COUNT(*) AS c FROM claim_vectors WHERE revision_id = ?", (revision_id,)
        ).fetchone()["c"]
        links = conn.execute(
            "SELECT availability FROM evidence_links WHERE revision_id = ?", (revision_id,)
        ).fetchall()
    assert revision["display_text"] == ""
    assert revision["value_json"] == "null"
    assert fts == 0
    assert vectors == 0
    assert {link["availability"] for link in links} == {"withheld"}
    # Lineage is preserved (no secret, still explainable).
    with store.read() as conn:
        assert conn.execute(
            "SELECT claim_id FROM claim_revisions WHERE revision_id = ?", (revision_id,)
        ).fetchone()["claim_id"]


def test_redact_makes_recall_and_export_secret_free(redacted_store, tmp_path):
    store, event, revision_id = redacted_store
    result = execute_recall(store, recall_request(query="历史配置"), binding=_binding())
    assert CANARY not in json.dumps(result, ensure_ascii=False)
    snapshot = tmp_path / "snapshot"
    export_snapshot(store, snapshot)
    leaked = [
        path.relative_to(snapshot).as_posix()
        for path in snapshot.rglob("*")
        if path.is_file() and CANARY in path.read_text(encoding="utf-8", errors="ignore")
    ]
    assert leaked == []


def test_redact_leaves_no_secret_bytes_in_database_or_wal(redacted_store):
    store, event, revision_id = redacted_store
    needle = CANARY.encode("utf-8")
    files = [store.db_path]
    files.extend(Path(str(store.db_path) + suffix) for suffix in ("-wal", "-shm"))
    residues = {
        path.name: needle in path.read_bytes()
        for path in files
        if path.exists()
    }
    assert not any(residues.values()), residues


def test_export_refuses_logical_records_still_containing_secrets(store, tmp_path):
    """Defence in depth: a row that escaped the scrub must not be exported."""
    with store.transaction() as conn:
        conn.execute(
            "INSERT INTO claims (claim_id, workspace_id, project_id, subject_id, predicate, "
            "facet, qualifiers_json, created_at) VALUES "
            "('claim:leaky', 'owner:demo', 'project:songta', 'subject:leaky', 'note', "
            "'reported', '{}', '2026-10-30T00:00:00Z')"
        )
        conn.execute(
            "INSERT INTO claim_revisions (revision_id, claim_id, revision_number, value_json, "
            "display_text, authority, assertion_kind, event_time_value, event_time_precision, "
            "event_time_timezone, status, supersedes_json, correction_of, source_actors_json, created_at) "
            "VALUES ('rev:leaky', 'claim:leaky', 1, '\"x\"', ?, 'reported', 'explicit_statement', "
            "NULL, 'unknown', 'UTC', 'candidate', '[]', NULL, '[]', '2026-10-30T00:00:00Z')",
            ("密码: " + CANARY,),
        )
    with pytest.raises(ExportRefused) as excinfo:
        export_snapshot_fn(store, tmp_path / "leaky-snapshot")
    assert excinfo.value.code == "REDACTION_REQUIRED"
    assert not (tmp_path / "leaky-snapshot").exists()
