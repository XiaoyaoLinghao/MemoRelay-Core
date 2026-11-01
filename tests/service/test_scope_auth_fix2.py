"""F21 round-3 regression: corrections authorize every evidence reference's scope.

Migrated from the independent round-2 counterexample: a correction targeting a
claim in one workspace must not accept evidence stored under another workspace,
even when the evidence project matches the binding allow-list. The real
HTTP-over-UDS service and local client binding are exercised; cross-scope data is
seeded through the trusted store layer (its ingestion would be rejected, which is
exactly the asymmetry the defect had).
"""

from __future__ import annotations

from knowledge_weaver.store import get_current, receive_batch
from knowledge_weaver.store.claims import record_claim_revision

from ..state.conftest import evidence_ref, record
from ..store.conftest import make_batch, make_event


def _owned_claim(store, *, native_id: str, text: str = "端口4100", value: int = 4100):
    event = make_event(native_id=native_id, text=text)
    receive_batch(store, make_batch([event]))
    claim = record(store, display_text=text, value=value, evidence_event=event)
    return event, claim


def _correction_body(claim, current, evidence_events, *, operation_id: str):
    return {
        "schema_version": "1.0",
        "operation_id": operation_id,
        "claim_id": claim["claim_id"],
        "expected_revision_id": claim["revision_id"],
        "expected_projection_generation": current["projection_generation"],
        "new_value": 4200,
        "display_text": "端口改为4200",
        "evidence": [evidence_ref(event) for event in evidence_events],
        "supersedes_revision_ids": [claim["revision_id"]],
        "reason": "synthetic correction",
    }


def _revision_count(store, claim_id: str) -> int:
    with store.read() as conn:
        return conn.execute(
            "SELECT COUNT(*) AS c FROM claim_revisions WHERE claim_id = ?", (claim_id,)
        ).fetchone()["c"]


def _operation_count(store, operation_id: str) -> int:
    with store.read() as conn:
        return conn.execute(
            "SELECT COUNT(*) AS c FROM correction_operations WHERE operation_id = ?",
            (operation_id,),
        ).fetchone()["c"]


def _assert_denied_without_writes(store, client, claim, current, body):
    status, payload = client.request("POST", "/v1/corrections", body)
    assert status == 403, payload
    assert payload["error"]["code"] == "SCOPE_DENIED", payload
    assert get_current(store, claim["claim_id"]) == current, "a denied correction moved the projection"
    assert _revision_count(store, claim["claim_id"]) == 1
    assert _operation_count(store, body["operation_id"]) == 0, "denied before any write"


def test_correction_denies_cross_workspace_evidence(store, client):
    """F21: a foreign-workspace source is 403 for GET *and* unusable as evidence."""
    _own_event, claim = _owned_claim(store, native_id="message:fix2-own", text="端口4100")
    current = get_current(store, claim["claim_id"])
    foreign = make_event(
        native_id="message:fix2-foreign",
        text="端口改为4200",
        workspace_id="owner:other",
        instance_id="instance:other",
    )
    receive_batch(store, make_batch([foreign]))
    assert client.get_source(foreign["event_id"], 1)[0] == 403
    body = _correction_body(claim, current, [foreign], operation_id="op:fix2:cross-workspace")
    _assert_denied_without_writes(store, client, claim, current, body)


def test_correction_denies_evidence_from_other_project(store, client):
    """F21: same workspace, evidence project outside the binding allow-list -> 403."""
    _own_event, claim = _owned_claim(store, native_id="message:fix2-own-project", text="端口4100")
    current = get_current(store, claim["claim_id"])
    foreign = make_event(
        native_id="message:fix2-other-project",
        text="端口改为4200",
        project_ids=("project:forbidden",),
    )
    receive_batch(store, make_batch([foreign]))
    body = _correction_body(claim, current, [foreign], operation_id="op:fix2:cross-project")
    _assert_denied_without_writes(store, client, claim, current, body)


def test_correction_applies_authorized_evidence(store, client):
    """F21: same workspace + authorized project evidence still applies normally."""
    _own_event, claim = _owned_claim(store, native_id="message:fix2-legal", text="端口4100")
    current = get_current(store, claim["claim_id"])
    fix = make_event(native_id="message:fix2-legal-fix", text="端口改为4200")
    assert client.post_batch(make_batch([fix]))[0] == 200
    status, payload = client.request(
        "POST",
        "/v1/corrections",
        _correction_body(claim, current, [fix], operation_id="op:fix2:legal"),
    )
    assert status == 200, payload
    assert payload["status"] == "applied", payload
    after = get_current(store, claim["claim_id"])
    assert after["selected_revision_ids"] == [payload["revision_id"]]
    assert _revision_count(store, claim["claim_id"]) == 2


def test_correction_denies_cross_workspace_target_claim(store, client):
    """F21: the target claim's own workspace check is unchanged (still 403)."""
    foreign_event = make_event(
        native_id="message:fix2-foreign-claim",
        text="端口4100",
        workspace_id="owner:other",
        instance_id="instance:other",
    )
    receive_batch(store, make_batch([foreign_event]))
    foreign_claim = record_claim_revision(
        store,
        workspace_id="owner:other",
        project_id="project:songta",
        subject_id="subject:fix2-other-server",
        predicate="server.port",
        facet="desired",
        qualifiers={},
        value=4100,
        display_text="端口4100",
        authority="user_confirmed",
        assertion_kind="explicit_statement",
        event_time=foreign_event["event_time"],
        evidence=[evidence_ref(foreign_event)],
    )
    current = get_current(store, foreign_claim["claim_id"])
    fix = make_event(native_id="message:fix2-target-fix", text="端口改为4200")
    assert client.post_batch(make_batch([fix]))[0] == 200
    body = _correction_body(
        foreign_claim, current, [fix], operation_id="op:fix2:cross-workspace-target"
    )
    status, payload = client.request("POST", "/v1/corrections", body)
    assert status == 403, payload
    assert payload["error"]["code"] == "SCOPE_DENIED", payload
    assert get_current(store, foreign_claim["claim_id"]) == current
    assert _revision_count(store, foreign_claim["claim_id"]) == 1
    assert _operation_count(store, body["operation_id"]) == 0


def test_correction_unknown_evidence_keeps_pending_semantics(store, client):
    """F21: unknown/missing references are left to the store, not guessed as 403."""
    _own_event, claim = _owned_claim(store, native_id="message:fix2-unknown", text="端口4100")
    current = get_current(store, claim["claim_id"])
    unknown_ref = {
        "event_id": "evt:" + "a" * 64,
        "event_revision": 1,
        "part_index": 0,
        "start_byte": 0,
        "end_byte": 3,
        "body_sha256": "b" * 64,
        "availability": "full",
    }
    body = {
        "schema_version": "1.0",
        "operation_id": "op:fix2:unknown-evidence",
        "claim_id": claim["claim_id"],
        "expected_revision_id": claim["revision_id"],
        "expected_projection_generation": current["projection_generation"],
        "new_value": 4200,
        "display_text": "端口改为4200",
        "evidence": [unknown_ref],
        "supersedes_revision_ids": [claim["revision_id"]],
        "reason": "synthetic correction",
    }
    status, payload = client.request("POST", "/v1/corrections", body)
    assert status == 200, payload
    assert payload["status"] == "pending_evidence", payload
    assert get_current(store, claim["claim_id"]) == current
    assert _revision_count(store, claim["claim_id"]) == 1
