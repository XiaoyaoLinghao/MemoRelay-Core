"""T02: durable event receipt, duplicates/conflicts, chunks, retract/redact, exclusion."""
from __future__ import annotations

import sqlite3

import pytest

from knowledge_weaver import contracts
from knowledge_weaver.store import Store, StoreRejection, fetch_source, receive_batch

from .conftest import assert_schema, clone, make_batch, make_event


def test_receive_creates_durable_receipts(store):
    event = make_event()
    result = receive_batch(store, make_batch([event]))
    assert_schema("BatchResult", result)
    assert len(result["receipts"]) == 1
    receipt = result["receipts"][0]
    assert_schema("Receipt", receipt)
    assert receipt["disposition"] == "accepted"
    assert receipt["durable_remote"] is True
    assert result["watermark"] >= 1
    # State is committed: a second independent handle sees it.
    reopened = Store(store.db_path)
    source = fetch_source(reopened, event["event_id"], 1)
    assert_schema("SourceResult", source)
    assert source["text"] == event["body"]["text"]
    assert source["availability"] == "full"


def test_duplicate_returns_same_receipt_without_new_write(store):
    event = make_event()
    first = receive_batch(store, make_batch([event]))
    watermark_after_first = first["watermark"]
    second = receive_batch(store, make_batch([event], batch_id="batch:test:2"))
    receipt = second["receipts"][0]
    assert receipt["disposition"] == "duplicate"
    assert receipt["receipt_id"] == first["receipts"][0]["receipt_id"]
    assert second["watermark"] == watermark_after_first
    assert store.count("SELECT COUNT(*) FROM event_parts") == 1


def test_same_key_different_payload_is_conflict_not_overwrite(store):
    event = make_event(text="端口4200")
    receive_batch(store, make_batch([event]))
    conflicting = make_event(text="端口4100")
    result = receive_batch(store, make_batch([conflicting], batch_id="batch:test:2"))
    receipt = result["receipts"][0]
    assert receipt["disposition"] == "rejected"
    assert receipt["error_code"] == "EVENT_CONFLICT"
    assert receipt["durable_remote"] is False
    source = fetch_source(store, event["event_id"], 1)
    assert source["text"] == "端口4200"


def test_incomplete_group_is_stored_but_not_complete(store):
    part0 = make_event(text="第0片内容", part_index=0, part_count=2)
    result = receive_batch(store, make_batch([part0]))
    assert result["receipts"][0]["disposition"] == "accepted"
    with store.read() as conn:
        row = conn.execute("SELECT group_state FROM event_revisions").fetchone()
    assert row["group_state"] == "incomplete"

    part1 = make_event(text="第1片内容", part_index=1, part_count=2)
    result = receive_batch(store, make_batch([part1], batch_id="batch:test:2"))
    assert result["receipts"][0]["disposition"] == "accepted"
    with store.read() as conn:
        row = conn.execute("SELECT group_state FROM event_revisions").fetchone()
    assert row["group_state"] == "complete"


def test_metadata_disagreement_rejects_whole_batch(store):
    part0 = make_event(text="a", part_index=0, part_count=2)
    part1 = make_event(text="b", part_index=1, part_count=2, project_ids=("project:other",))
    with pytest.raises(StoreRejection) as excinfo:
        receive_batch(store, make_batch([part0, part1]))
    assert excinfo.value.code == "INVALID_REQUEST"
    assert store.count("SELECT COUNT(*) FROM event_parts") == 0


def test_forged_event_id_is_rejected(store):
    event = make_event()
    event["event_id"] = "evt:" + "0" * 64
    with pytest.raises(StoreRejection) as excinfo:
        receive_batch(store, make_batch([event]))
    assert excinfo.value.code == "INVALID_REQUEST"
    assert store.count("SELECT COUNT(*) FROM events") == 0


def test_unsupported_schema_version_is_rejected(store):
    event = make_event()
    event["schema_version"] = "99.0"
    with pytest.raises(StoreRejection) as excinfo:
        receive_batch(store, make_batch([event]))
    assert excinfo.value.code == "UNSUPPORTED_SCHEMA"


def test_retract_marks_prior_parts_retracted(store):
    create = make_event()
    receive_batch(store, make_batch([create]))
    retract = make_event(
        revision=2,
        operation="retract",
        parent_revision=1,
        text="",
    )
    retract["body"]["sha256"] = contracts.EMPTY_BODY_SHA256
    result = receive_batch(store, make_batch([retract], batch_id="batch:test:2"))
    assert result["receipts"][0]["disposition"] == "accepted"
    with store.read() as conn:
        revision = conn.execute(
            "SELECT lifecycle FROM event_revisions WHERE revision = 1"
        ).fetchone()
        part = conn.execute("SELECT availability FROM event_parts WHERE revision = 1").fetchone()
    assert revision["lifecycle"] == "retracted"
    assert part["availability"] == "retracted"
    source = fetch_source(store, create["event_id"], 1)
    assert source["availability"] == "retracted"
    assert source["text"] is None


def test_redact_purges_old_body_transactionally(store):
    create = make_event(text="API key CANARY-API-91b2")
    receive_batch(store, make_batch([create]))
    redact = make_event(
        revision=2,
        operation="redact",
        parent_revision=1,
        text="API key [REDACTED:API_KEY]",
        content_revision="native:1",
    )
    result = receive_batch(store, make_batch([redact], batch_id="batch:test:2"))
    assert result["receipts"][0]["disposition"] == "accepted"
    old = fetch_source(store, create["event_id"], 1)
    assert old["availability"] == "withheld"
    assert old["text"] is None
    with store.read() as conn:
        text = conn.execute("SELECT body_text FROM event_parts WHERE revision = 1").fetchone()["body_text"]
    assert text is None
    new = fetch_source(store, create["event_id"], 2)
    assert new["text"] == "API key [REDACTED:API_KEY]"
    assert new["redaction"]["status"] == "passed"


def test_redact_must_keep_native_content_revision_and_time(store):
    create = make_event()
    receive_batch(store, make_batch([create]))
    redact = make_event(
        revision=2,
        operation="redact",
        parent_revision=1,
        text="sanitized",
        content_revision="native:2",
    )
    result = receive_batch(store, make_batch([redact], batch_id="batch:test:2"))
    receipt = result["receipts"][0]
    assert receipt["disposition"] == "rejected"
    assert receipt["error_code"] == "INVALID_REQUEST"


def test_pending_dependency_stored_then_promoted(store):
    edit = make_event(
        native_id="message:2", revision=2, operation="edit", parent_revision=1, text="修改后"
    )
    first = receive_batch(store, make_batch([edit]))
    receipt = first["receipts"][0]
    assert receipt["disposition"] == "accepted"
    assert receipt["error_code"] == "PARENT_PENDING"
    with store.read() as conn:
        row = conn.execute("SELECT group_state FROM event_revisions").fetchone()
    assert row["group_state"] == "pending_dependency"
    # Parts are durable even though the dependency is missing.
    assert store.count("SELECT COUNT(*) FROM event_parts") == 1

    create = make_event(native_id="message:2", revision=1, operation="create")
    second = receive_batch(store, make_batch([create], batch_id="batch:test:2"))
    assert second["receipts"][0]["disposition"] == "accepted"
    with store.read() as conn:
        rows = conn.execute(
            "SELECT revision, group_state FROM event_revisions ORDER BY revision"
        ).fetchall()
    assert [(row["revision"], row["group_state"]) for row in rows] == [
        (1, "complete"),
        (2, "complete"),
    ]


def test_withheld_event_stores_no_body(store):
    text = ""
    withheld = make_event(
        text=text,
        redaction={"policy_version": "redact:1", "status": "withheld", "replacements": 0, "credential_refs": []},
    )
    withheld["body"]["sha256"] = contracts.EMPTY_BODY_SHA256
    result = receive_batch(store, make_batch([withheld]))
    receipt = result["receipts"][0]
    assert receipt["disposition"] == "withheld"
    assert receipt["durable_remote"] is True
    with store.read() as conn:
        part = conn.execute("SELECT body_text, availability FROM event_parts").fetchone()
    assert part["body_text"] is None
    assert part["availability"] == "withheld"
    source = fetch_source(store, withheld["event_id"], 1)
    assert source["text"] is None
    assert source["availability"] == "withheld"


def test_entry_secret_check_withholds_content(store):
    event = make_event(text="密码 CANARY-PW-7f3a")
    result = receive_batch(store, make_batch([event]), secret_check=lambda text: "CANARY-PW" in text)
    receipt = result["receipts"][0]
    assert receipt["disposition"] == "withheld"
    assert receipt["error_code"] == "REDACTION_REQUIRED"
    with store.read() as conn:
        part = conn.execute("SELECT body_text FROM event_parts").fetchone()
    assert part["body_text"] is None


def test_memory_injection_is_excluded_not_semantically_ingested(store):
    injected = make_event(origin_kind="memory_injection", role="system", text="历史记忆注入")
    result = receive_batch(store, make_batch([injected]))
    receipt = result["receipts"][0]
    assert receipt["disposition"] == "excluded"
    assert receipt["processing"] == "not_applicable"
    with store.read() as conn:
        row = conn.execute("SELECT lifecycle FROM event_revisions").fetchone()
    assert row["lifecycle"] == "excluded"


def test_watermark_is_monotonic(store):
    e1 = make_event(native_id="message:1")
    e2 = make_event(native_id="message:2")
    first = receive_batch(store, make_batch([e1]))
    second = receive_batch(store, make_batch([e2], batch_id="batch:test:2"))
    assert second["watermark"] > first["watermark"]
    assert second["backend_instance_id"] == first["backend_instance_id"]
    assert second["restore_epoch"] == first["restore_epoch"]


def test_batch_rejects_non_object_and_bad_events(store):
    with pytest.raises(StoreRejection):
        receive_batch(store, {"schema_version": "1.0", "batch_id": "batch:1", "events": [1]})
    with pytest.raises(StoreRejection):
        receive_batch(store, [])


# ---------------------------------------------------------------------------
# F06: chunk metadata agreement must hold across requests, not only in a batch
# ---------------------------------------------------------------------------


def _job_keys(store):
    with store.read() as conn:
        return [
            row["input_key"]
            for row in conn.execute("SELECT input_key FROM jobs WHERE kind = 'extract' ORDER BY input_key")
        ]


def test_cross_request_part_count_change_is_rejected(store):
    first = make_event(text="片0", part_index=0, part_count=2)
    assert receive_batch(store, make_batch([first]))["receipts"][0]["disposition"] == "accepted"
    second = make_event(text="片1", part_index=1, part_count=3)
    result = receive_batch(store, make_batch([second], batch_id="batch:test:2"))
    receipt = result["receipts"][0]
    assert receipt["disposition"] == "rejected", receipt
    assert receipt["error_code"] == "EVENT_CONFLICT"
    with store.read() as conn:
        state = conn.execute("SELECT group_state FROM event_revisions").fetchone()["group_state"]
        parts = conn.execute("SELECT COUNT(*) AS c FROM event_parts").fetchone()["c"]
    assert state == "incomplete"
    assert parts == 1


def test_cross_request_project_scope_change_is_rejected(store):
    first = make_event(text="片0", part_index=0, part_count=2)
    assert receive_batch(store, make_batch([first]))["receipts"][0]["disposition"] == "accepted"
    second = make_event(
        text="片1", part_index=1, part_count=2, project_ids=("project:other",)
    )
    result = receive_batch(store, make_batch([second], batch_id="batch:test:2"))
    receipt = result["receipts"][0]
    assert receipt["disposition"] == "rejected", receipt
    assert receipt["error_code"] == "EVENT_CONFLICT"
    with store.read() as conn:
        project_ids = conn.execute(
            "SELECT project_ids_json FROM event_revisions"
        ).fetchone()["project_ids_json"]
    assert project_ids == '["project:songta"]'


def test_cross_request_content_revision_change_is_rejected(store):
    first = make_event(text="片0", part_index=0, part_count=2)
    receive_batch(store, make_batch([first]))
    second = make_event(text="片1", part_index=1, part_count=2, content_revision="native:2")
    receipt = receive_batch(store, make_batch([second], batch_id="batch:test:2"))["receipts"][0]
    assert receipt["disposition"] == "rejected"
    assert receipt["error_code"] == "EVENT_CONFLICT"


def test_out_of_order_parts_complete_the_group_once(store):
    part1 = make_event(text="片1", part_index=1, part_count=2)
    part0 = make_event(text="片0", part_index=0, part_count=2)
    assert receive_batch(store, make_batch([part1]))["receipts"][0]["disposition"] == "accepted"
    result = receive_batch(store, make_batch([part0], batch_id="batch:test:2"))
    assert result["receipts"][0]["disposition"] == "accepted"
    with store.read() as conn:
        state = conn.execute("SELECT group_state FROM event_revisions").fetchone()["group_state"]
    assert state == "complete"
    assert _job_keys(store) == [f"{part0['event_id']}@1"]


# ---------------------------------------------------------------------------
# F07: a dependency becoming complete must queue its extraction job atomically
# ---------------------------------------------------------------------------


def test_pending_child_gets_extraction_job_when_parent_arrives(store):
    child = make_event(
        revision=2, operation="edit", parent_revision=1, content_revision="native:2", text="端口4300"
    )
    first = receive_batch(store, make_batch([child]))
    assert first["receipts"][0]["error_code"] == "PARENT_PENDING"
    parent = make_event(text="端口4200")
    second = receive_batch(store, make_batch([parent], batch_id="batch:test:2"))
    assert second["receipts"][0]["disposition"] == "accepted"
    with store.read() as conn:
        child_state = conn.execute(
            "SELECT group_state FROM event_revisions WHERE revision = 2"
        ).fetchone()["group_state"]
    assert child_state == "complete"
    assert sorted(_job_keys(store)) == sorted(
        [f"{parent['event_id']}@1", f"{child['event_id']}@2"]
    )


def test_duplicate_receipt_does_not_duplicate_jobs(store):
    event = make_event(text="端口4200")
    receive_batch(store, make_batch([event]))
    receive_batch(store, make_batch([event], batch_id="batch:test:2"))
    receive_batch(store, make_batch([event], batch_id="batch:test:3"))
    assert _job_keys(store) == [f"{event['event_id']}@1"]


def test_reconcile_extraction_jobs_is_idempotent(store):
    from knowledge_weaver.extraction.pipeline import reconcile_extraction_jobs

    event = make_event(text="端口4200")
    receive_batch(store, make_batch([event]))
    # Simulate the crash window: the revision is complete/active but its job row
    # is missing (e.g. written by an older build).
    with store.transaction() as conn:
        conn.execute("DELETE FROM jobs WHERE kind = 'extract'")
    assert _job_keys(store) == []
    first = reconcile_extraction_jobs(store)
    assert first["queued"] == 1
    assert _job_keys(store) == [f"{event['event_id']}@1"]
    second = reconcile_extraction_jobs(store)
    assert second["queued"] == 0
    assert _job_keys(store) == [f"{event['event_id']}@1"]
