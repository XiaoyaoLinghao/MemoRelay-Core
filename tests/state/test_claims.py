"""T05: fact-level claims, versions/conflicts, corrections, invalidation."""
from __future__ import annotations

import pytest

from knowledge_weaver import contracts
from knowledge_weaver.store import (
    ClaimError,
    current_items,
    get_current,
    rebuild_projection,
    receive_batch,
)
from knowledge_weaver.store.claims import apply_correction, record_claim_revision

from ..store.conftest import make_batch, make_event
from .conftest import create_user_event, current_item, evidence_ref, record


def _claims(store):
    with store.read() as conn:
        return conn.execute("SELECT COUNT(*) AS c FROM claims").fetchone()["c"]


def _revision_statuses(store, claim_id):
    with store.read() as conn:
        rows = conn.execute(
            "SELECT revision_number, status FROM claim_revisions WHERE claim_id = ? ORDER BY revision_number",
            (claim_id,),
        ).fetchall()
    return [(row["revision_number"], row["status"]) for row in rows]


def test_independent_properties_never_overwrite(store):
    port = record(store, display_text="松塔端口为4100", value=4100)
    backup = record(
        store,
        display_text="备份周期为每周",
        value="weekly",
        predicate="backup_interval",
    )
    assert port["claim_id"] != backup["claim_id"]
    assert _claims(store) == 2
    record(store, display_text="松塔端口改为4200", value=4200, supersedes=[port["revision_id"]])
    item = current_item(store)
    assert item["value"] == 4200
    backup_item = current_item(store, predicate="backup_interval")
    assert backup_item["value"] == "weekly"  # untouched by the port update (AC-35)


def test_desired_and_observed_are_separate_facets(store):
    record(store, display_text="用户要求端口4200", value=4200, facet="desired")
    record(store, display_text="实测端口4100", value=4100, facet="observed", authority="tool_verified")
    desired = current_item(store, facet="desired")
    observed = current_item(store, facet="observed")
    assert desired["value"] == 4200 and desired["state"] == "current"
    assert observed["value"] == 4100 and observed["state"] == "current"
    assert desired["claim_id"] != observed["claim_id"]


def test_explicit_supersede_and_history(store):
    first = record(store, display_text="端口4100", value=4100)
    second = record(
        store,
        display_text="端口改为4200",
        value=4200,
        supersedes=[first["revision_id"]],
        event_time={"value": "2026-10-30T02:00:00+00:00", "precision": "instant", "timezone": "UTC"},
    )
    current = get_current(store, first["claim_id"])
    assert current["state"] == "current"
    assert current["selected_revision_ids"] == [second["revision_id"]]
    assert _revision_statuses(store, first["claim_id"]) == [(1, "superseded"), (2, "current")]


def test_late_older_information_does_not_roll_back_current(store):
    newer = record(
        store,
        display_text="端口4200",
        value=4200,
        event_time={"value": "2026-10-30T10:00:00+00:00", "precision": "instant", "timezone": "UTC"},
    )
    record(
        store,
        display_text="端口4100",
        value=4100,
        event_time={"value": "2026-10-30T09:00:00+00:00", "precision": "instant", "timezone": "UTC"},
    )
    current = get_current(store, newer["claim_id"])
    item = current_item(store)
    assert item["value"] == 4200
    assert current["selected_revision_ids"] == [newer["revision_id"]]
    assert _revision_statuses(store, newer["claim_id"])[1][1] == "superseded"


def test_inferred_cannot_override_user_confirmed(store):
    confirmed = record(store, display_text="端口4200", value=4200)
    record(
        store,
        display_text="助手猜测端口4100",
        value=4100,
        authority="inferred",
        assertion_kind="inference",
        event_time={"value": "2026-10-30T11:00:00+00:00", "precision": "instant", "timezone": "UTC"},
    )
    item = current_item(store)
    assert item["value"] == 4200
    assert get_current(store, confirmed["claim_id"])["selected_revision_ids"] == [confirmed["revision_id"]]


def test_conflict_keeps_both_candidates(store):
    same_time = {"value": "2026-10-30T10:00:00+00:00", "precision": "instant", "timezone": "UTC"}
    first = record(store, display_text="端口4200", value=4200, event_time=same_time)
    second = record(store, display_text="端口4100", value=4100, event_time=same_time)
    current = get_current(store, first["claim_id"])
    assert current["state"] == "conflict"
    assert set(current["selected_revision_ids"]) == {first["revision_id"], second["revision_id"]}
    item = current_item(store)
    assert item["state"] == "conflict"
    assert set(item["conflict_revision_ids"]) == {second["revision_id"]}
    with store.read() as conn:
        statuses = {
            row["revision_id"]: row["status"]
            for row in conn.execute(
                "SELECT revision_id, status FROM claim_revisions WHERE claim_id = ?",
                (first["claim_id"],),
            )
        }
    assert set(statuses.values()) == {"conflict"}


def test_unknown_business_time_never_becomes_current(store):
    known = record(
        store,
        display_text="端口4200",
        value=4200,
        event_time={"value": "2026-10-30T10:00:00+00:00", "precision": "instant", "timezone": "UTC"},
    )
    unknown = record(
        store,
        display_text="端口4100（时间未知）",
        value=4100,
        event_time={"value": None, "precision": "unknown", "timezone": "UTC"},
    )
    current = get_current(store, known["claim_id"])
    assert current["selected_revision_ids"] == [known["revision_id"]]
    assert _revision_statuses(store, known["claim_id"]) == [(1, "current"), (2, "uncertain")]


def _correction(claim_id, expected_revision, generation, evidence, *, operation_id="op:corr:1"):
    return {
        "schema_version": "1.0",
        "operation_id": operation_id,
        "claim_id": claim_id,
        "expected_revision_id": expected_revision,
        "expected_projection_generation": generation,
        "new_value": 4200,
        "display_text": "端口改为4200",
        "evidence": [evidence],
        "supersedes_revision_ids": [expected_revision],
        "reason": "用户明确更正",
    }


def test_correction_applied_with_user_evidence(store):
    first = record(store, display_text="端口4100", value=4100)
    current = get_current(store, first["claim_id"])
    user_event = create_user_event(store, "端口改为4200，请更新")
    correction = _correction(
        first["claim_id"], first["revision_id"], current["projection_generation"], evidence_ref(user_event)
    )
    result = apply_correction(store, correction)
    assert contracts.validate_instance("CorrectionResult", result) == []
    assert result["status"] == "applied"
    item = current_item(store)
    assert item["value"] == 4200
    assert item["evidence_grade"] == "user_confirmed"
    assert get_current(store, first["claim_id"])["projection_generation"] > current["projection_generation"]


def test_correction_stale_generation_conflicts(store):
    first = record(store, display_text="端口4100", value=4100)
    current = get_current(store, first["claim_id"])
    user_event = create_user_event(store, "端口改为4200")
    apply_correction(
        store,
        _correction(first["claim_id"], first["revision_id"], current["projection_generation"], evidence_ref(user_event)),
    )
    stale = _correction(
        first["claim_id"], first["revision_id"], current["projection_generation"], evidence_ref(user_event), operation_id="op:stale"
    )
    result = apply_correction(store, stale)
    assert result["status"] == "conflict"


def test_correction_requires_native_user_evidence(store):
    first = record(store, display_text="端口4100", value=4100)
    current = get_current(store, first["claim_id"])
    assistant = make_event(native_id="message:assistant-1", text="端口改为4200", role="assistant", origin_kind="assistant_reply")
    receive_batch(store, make_batch([assistant]))
    result = apply_correction(
        store,
        _correction(first["claim_id"], first["revision_id"], current["projection_generation"], evidence_ref(assistant)),
    )
    assert result["status"] == "pending_evidence"
    assert result["revision_id"] is None
    assert current_item(store)["value"] == 4100


def test_correction_value_must_be_supported_by_quote(store):
    first = record(store, display_text="端口4100", value=4100)
    current = get_current(store, first["claim_id"])
    unrelated = create_user_event(store, "今天天气不错")
    result = apply_correction(
        store,
        _correction(first["claim_id"], first["revision_id"], current["projection_generation"], evidence_ref(unrelated)),
    )
    assert result["status"] == "pending_evidence"


def test_correction_operation_replay_is_stable(store):
    first = record(store, display_text="端口4100", value=4100)
    current = get_current(store, first["claim_id"])
    user_event = create_user_event(store, "端口改为4200")
    correction = _correction(
        first["claim_id"], first["revision_id"], current["projection_generation"], evidence_ref(user_event)
    )
    applied = apply_correction(store, correction)
    replay = apply_correction(store, correction)
    assert replay["status"] == "duplicate"
    assert replay["revision_id"] == applied["revision_id"]
    assert _revision_statuses(store, first["claim_id"]) == [(1, "superseded"), (2, "current")]
    assert len(_revision_statuses(store, first["claim_id"])) == 2


def test_retract_invalidates_dependent_claim(store):
    event = create_user_event(store, "端口4200")
    recorded = record_claim_revision(
        store,
        workspace_id="owner:demo",
        project_id="project:songta",
        subject_id="subject:songta-server",
        predicate="server.port",
        facet="desired",
        qualifiers={},
        value=4200,
        display_text="端口4200",
        authority="user_confirmed",
        assertion_kind="explicit_statement",
        event_time=event["event_time"],
        evidence=[evidence_ref(event)],
    )
    retract = make_event(native_id=event["source"]["native_event_id"], revision=2, operation="retract", parent_revision=1, text="")
    retract["body"]["sha256"] = contracts.EMPTY_BODY_SHA256
    receive_batch(store, make_batch([retract], batch_id="batch:retract"))
    current = get_current(store, recorded["claim_id"])
    assert current["state"] == "revoked"
    assert current_items(store, workspace_id="owner:demo") == []
    with store.read() as conn:
        availability = conn.execute("SELECT availability FROM evidence_links").fetchone()["availability"]
    assert availability == "retracted"


def test_redact_degrades_dependent_claim(store):
    event = create_user_event(store, "端口4200，含密码 CANARY-PW-7f3a")
    recorded = record_claim_revision(
        store,
        workspace_id="owner:demo",
        project_id="project:songta",
        subject_id="subject:songta-server",
        predicate="server.port",
        facet="desired",
        qualifiers={},
        value=4200,
        display_text="端口4200",
        authority="user_confirmed",
        assertion_kind="explicit_statement",
        event_time=event["event_time"],
        evidence=[evidence_ref(event, end_byte=len("端口4200".encode("utf-8")))],
    )
    redact = make_event(
        native_id=event["source"]["native_event_id"],
        revision=2,
        operation="redact",
        parent_revision=1,
        text="端口4200，含密码 [REDACTED:PASSWORD_ASSIGNMENT]",
        content_revision="native:1",
    )
    receive_batch(store, make_batch([redact], batch_id="batch:redact"))
    current = get_current(store, recorded["claim_id"])
    assert current["state"] == "uncertain"
    items = current_items(store, workspace_id="owner:demo", include_states=("uncertain",))
    assert items and items[0]["source_detail_expired"] is True
    with store.read() as conn:
        availability = conn.execute("SELECT availability FROM evidence_links").fetchone()["availability"]
    assert availability == "withheld"


def test_memory_item_matches_contract_schema(store):
    record(store, display_text="端口4200", value=4200)
    items = current_items(store, workspace_id="owner:demo")
    assert len(items) == 1
    assert contracts.validate_instance("MemoryItem", items[0]) == []


def test_rebuild_projection_is_idempotent(store):
    recorded = record(store, display_text="端口4200", value=4200)
    before = get_current(store, recorded["claim_id"])
    assert rebuild_projection(store) == 1
    assert rebuild_projection(store) == 1
    after = get_current(store, recorded["claim_id"])
    assert after["selected_revision_ids"] == before["selected_revision_ids"]
    assert after["projection_generation"] == before["projection_generation"]


def test_unsupported_facet_is_rejected(store):
    event = create_user_event(store, "端口4200")
    with pytest.raises(ClaimError):
        record_claim_revision(
            store,
            workspace_id="owner:demo",
            project_id="project:songta",
            subject_id="subject:songta-server",
            predicate="server.port",
            facet="guess",
            qualifiers={},
            value=4200,
            display_text="端口4200",
            authority="user_confirmed",
            assertion_kind="explicit_statement",
            event_time=event["event_time"],
            evidence=[evidence_ref(event)],
        )


# ---------------------------------------------------------------------------
# F05: unknown / abnormal-future business time must not crash or dominate
# ---------------------------------------------------------------------------


def test_two_unknown_times_do_not_crash_projection(store):
    for index in range(2):
        record(
            store,
            display_text=f"端口{4200 + index}（时间未知）",
            value=4200 + index,
            event_time={"value": None, "precision": "unknown", "timezone": "UTC"},
        )
    items = current_items(store, workspace_id="owner:demo")
    assert all(item["state"] != "current" for item in items), items


def test_single_unknown_time_is_not_current(store):
    recorded = record(
        store,
        display_text="端口（时间未知）",
        value=4100,
        event_time={"value": None, "precision": "unknown", "timezone": "UTC"},
    )
    current = get_current(store, recorded["claim_id"])
    assert current["state"] == "uncertain"
    assert current["selected_revision_ids"] == []


def test_abnormal_future_time_does_not_dominate_current(store):
    known = record(store, display_text="端口4200", value=4200)
    future = record(
        store,
        display_text="端口4300",
        value=4300,
        event_time={
            "value": "2099-01-01T00:00:00Z",
            "precision": "instant",
            "timezone": "UTC",
        },
    )
    current = get_current(store, known["claim_id"])
    assert current["selected_revision_ids"] == [known["revision_id"]]
    statuses = dict(_revision_statuses(store, known["claim_id"]))
    assert statuses[2] == "uncertain"
    items = current_items(store, workspace_id="owner:demo")
    assert not any(item["value"] == 4300 and item["state"] == "current" for item in items)


def test_abnormal_future_time_alone_is_uncertain_not_current(store):
    recorded = record(
        store,
        display_text="端口2099",
        value=2099,
        event_time={
            "value": "2099-01-01T00:00:00Z",
            "precision": "instant",
            "timezone": "UTC",
        },
    )
    current = get_current(store, recorded["claim_id"])
    assert current["state"] == "uncertain"
    assert current["selected_revision_ids"] == []
