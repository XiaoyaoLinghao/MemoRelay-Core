"""T20 round-4 regression: F02 (a redaction must also clean source-derived claim
metadata, not only the revision payload).

Codex round-3 counterexample (``test_independent_v2.py::
test_F02_redact_cleans_qualifier_payload_without_deleting_safe_lineage``): the
explicit ``redact`` path scrubbed ``claim_revisions`` payloads (display text,
value, FTS, vectors) but left the same secret in ``claims.qualifiers_json``.
``claims/claims.jsonl`` in the logical export therefore still carried the newly
classified secret, so "ordinary memory contains no secrets" / export claims were
false even though recall happened to be empty.

Round-4 rule: for every claim lineage touched by a redaction downgrade, the
source-derived claim metadata (qualifiers) is cleared together with the revision
payload, in the same inbox transaction, and no store/recall/index/export/restore
surface may still serve it. Safe lineage is retained: claim id, revision number,
authority, business time, evidence ids/hashes/spans and survival semantics of
unrelated claims. Only synthetic canaries are used.
"""

from __future__ import annotations

import json

from knowledge_weaver.portability import export_snapshot, import_snapshot
from knowledge_weaver.retrieval import execute_recall
from knowledge_weaver.store import get_current, receive_batch

from ..retrieval.conftest import recall_request
from ..store.conftest import make_batch, make_event
from .conftest import create_user_event, record

CANARY = "R4_NEW_SECRET_METADATA_7291"
WORKSPACE = "owner:demo"
PROJECT = "project:songta"
BINDING = {
    "client_id": "client:test",
    "workspace_id": WORKSPACE,
    "project_ids": [PROJECT],
    "actor_ids": ["*"],
}


def _redaction_event(event, *, revision=2, text="历史配置 [REDACTED:PASSWORD_ASSIGNMENT]"):
    return make_event(
        native_id=event["source"]["native_event_id"],
        revision=revision,
        operation="redact",
        parent_revision=revision - 1,
        text=text,
        content_revision=event["body"]["content_revision"],
        event_time=event["event_time"],
    )


def _secret_claim(store, native_id="message:fix3-f02-secret"):
    event = create_user_event(store, "configuration " + CANARY, native_id=native_id)
    claimed = record(
        store,
        display_text=event["body"]["text"],
        value=CANARY,
        qualifiers={"connection_note": CANARY},
        evidence_event=event,
    )
    return event, claimed


def _safe_claim(store):
    return record(
        store,
        display_text="当前端口4200",
        value=4200,
        subject_id="subject:unrelated-clean",
        qualifiers={"env": "prod", "region": "cn"},
        native_id="message:fix3-f02-safe",
    )


def _redact(store, event, *, batch_id="batch:fix3-f02-redact"):
    result = receive_batch(store, make_batch([_redaction_event(event)], batch_id=batch_id))
    assert result["receipts"][0]["disposition"] == "accepted", result
    return result


def _one(store, sql, params=()):
    with store.read() as conn:
        return conn.execute(sql, params).fetchone()


def _canary_hits(root) -> list[str]:
    hits = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            try:
                if CANARY in path.read_text(encoding="utf-8", errors="ignore"):
                    hits.append(path.as_posix())
            except OSError:  # pragma: no cover - unreadable synthetic artifact
                hits.append(path.as_posix())
    return hits


def _canary_on_disk_hits(store) -> dict[str, bool]:
    """SEC-02 at-rest check: no canary bytes left in the database/WAL/shm files."""
    from pathlib import Path

    needle = CANARY.encode("utf-8")
    candidates = [
        store.db_path,
        Path(f"{store.db_path}-wal"),
        Path(f"{store.db_path}-shm"),
    ]
    return {
        path.name: needle in path.read_bytes() for path in candidates if path.exists()
    }


def test_F02_redact_cleans_qualifier_payload_without_deleting_safe_lineage(store, tmp_path):
    """Original Codex counterexample, kept verbatim in semantics."""
    event, claimed = _secret_claim(store)
    safe = _safe_claim(store)
    _redact(store, event)

    recall = execute_recall(
        store, recall_request(query="configuration"), binding=BINDING
    )
    assert CANARY not in json.dumps(recall, ensure_ascii=False)

    destination = tmp_path / "export-f02-qualifiers"
    export_snapshot(store, destination)
    assert _canary_hits(destination) == [], _canary_hits(destination)

    # The evidence lineage itself is retained (no evidence link is deleted).
    links = _one(
        store,
        "SELECT COUNT(*) AS c FROM evidence_links WHERE revision_id = ?",
        (claimed["revision_id"],),
    )
    assert links["c"] == 1

    # An unrelated, non-secret fact is untouched and still current.
    assert get_current(store, safe["claim_id"])["state"] == "current"


def test_F02_redact_scrubs_claim_qualifiers_and_payload_across_db_and_index(store):
    """DB/index/recall: only the payload goes, the non-secret lineage stays."""
    event, claimed = _secret_claim(store, native_id="message:fix3-f02-db")
    revision_id = claimed["revision_id"]

    # Precondition: the claim metadata really holds the derived secret.
    row = _one(
        store, "SELECT qualifiers_json FROM claims WHERE claim_id = ?", (claimed["claim_id"],)
    )
    assert CANARY in row["qualifiers_json"]

    _redact(store, event, batch_id="batch:fix3-f02-db")

    row = _one(
        store, "SELECT * FROM claims WHERE claim_id = ?", (claimed["claim_id"],)
    )
    assert row is not None, "claim identity must survive (only its payload is secret)"
    assert CANARY not in row["qualifiers_json"]
    assert json.loads(row["qualifiers_json"]) == {}

    revision = _one(
        store, "SELECT * FROM claim_revisions WHERE revision_id = ?", (revision_id,)
    )
    assert revision["display_text"] == ""
    assert revision["value_json"] == "null"
    # Non-secret lineage is retained and explainable.
    assert revision["claim_id"] == claimed["claim_id"]
    assert revision["revision_number"] == 1
    assert revision["authority"] == "user_confirmed"
    assert revision["event_time_value"] == event["event_time"]["value"]

    assert _one(
        store, "SELECT COUNT(*) AS c FROM claim_fts WHERE revision_id = ?", (revision_id,)
    )["c"] == 0
    assert _one(
        store, "SELECT COUNT(*) AS c FROM claim_vectors WHERE revision_id = ?", (revision_id,)
    )["c"] == 0
    part = _one(
        store,
        "SELECT body_text FROM event_parts WHERE event_id = ? AND revision = 1 AND part_index = 0",
        (event["event_id"],),
    )
    assert part["body_text"] is None

    link = _one(
        store,
        "SELECT * FROM evidence_links WHERE revision_id = ?",
        (revision_id,),
    )
    assert link["body_sha256"] == event["body"]["sha256"]
    assert (link["event_id"], link["part_index"]) == (event["event_id"], 0)

    # Recall no longer exposes the secret anywhere in its payload.
    recall = execute_recall(store, recall_request(query="configuration"), binding=BINDING)
    assert CANARY not in json.dumps(recall, ensure_ascii=False)
    # SEC-02 at-rest: secure_delete/checkpoint leaves no secret bytes behind,
    # including the cleared claim metadata.
    assert not any(_canary_on_disk_hits(store).values()), _canary_on_disk_hits(store)


def test_F02_redact_scrubbed_snapshot_restores_without_secret(store, tmp_path):
    """Export -> import (restore) must not reintroduce the cleared metadata."""
    event, claimed = _secret_claim(store, native_id="message:fix3-f02-restore")
    safe = _safe_claim(store)
    _redact(store, event, batch_id="batch:fix3-f02-restore")

    destination = tmp_path / "export-f02-restore"
    export_snapshot(store, destination)
    assert _canary_hits(destination) == [], _canary_hits(destination)

    restored = import_snapshot(destination, tmp_path / "restored-f02")
    db_path = restored["target_db"]
    raw = open(db_path, "rb").read()
    assert CANARY.encode("utf-8") not in raw
    # Safe lineage survives the restore: the claim row still exists and an
    # unrelated fact is usable again.
    from knowledge_weaver.store import Store

    restored_store = Store(db_path)
    row = _one(
        restored_store,
        "SELECT qualifiers_json FROM claims WHERE claim_id = ?",
        (claimed["claim_id"],),
    )
    assert row is not None
    assert json.loads(row["qualifiers_json"]) == {}
    assert get_current(restored_store, safe["claim_id"])["state"] == "current"


def test_F02_redact_keeps_unrelated_claim_metadata(store):
    """Fail-closed must not over-delete: unrelated facts keep their qualifiers."""
    secret_event, _ = _secret_claim(store, native_id="message:fix3-f02-scope-secret")
    safe = _safe_claim(store)
    _redact(store, secret_event, batch_id="batch:fix3-f02-scope")

    row = _one(
        store, "SELECT qualifiers_json FROM claims WHERE claim_id = ?", (safe["claim_id"],)
    )
    assert json.loads(row["qualifiers_json"]) == {"env": "prod", "region": "cn"}
    assert get_current(store, safe["claim_id"])["state"] == "current"
