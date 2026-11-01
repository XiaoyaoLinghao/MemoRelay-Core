"""F21 regression: source reads and corrections must honor project scope.

These tests use the real HTTP-over-UDS service and the real client binding; the
"other project" data is seeded through the trusted store layer (a batch from the
bound client would be rejected at ingestion, which is exactly the asymmetry the
defect had).
"""
from __future__ import annotations

from knowledge_weaver.store import get_current, receive_batch, register_client

from ..state.conftest import evidence_ref, record
from ..store.conftest import make_batch, make_event


def _foreign_event(native_id="message:foreign"):
    return make_event(
        native_id=native_id,
        text="project B private source",
        project_ids=("project:forbidden",),
    )


def test_source_read_denies_other_project_in_same_workspace(store, client):
    permitted = make_event(native_id="message:permitted", text="allowed source")
    assert client.post_batch(make_batch([permitted]))[0] == 200
    assert client.get_source(permitted["event_id"], 1)[0] == 200
    foreign = _foreign_event()
    assert client.post_batch(make_batch([foreign]))[0] == 403
    receive_batch(store, make_batch([foreign]))
    status, payload = client.get_source(foreign["event_id"], 1)
    assert status == 403, payload
    assert payload["error"]["code"] == "SCOPE_DENIED"
    assert "project B private source" not in str(payload)


def test_source_read_denies_cross_workspace(store, client):
    other_workspace = make_event(
        native_id="message:other-ws",
        text="other workspace source",
        workspace_id="owner:other",
    )
    receive_batch(store, make_batch([other_workspace]))
    status, payload = client.get_source(other_workspace["event_id"], 1)
    assert status == 403, payload
    assert payload["error"]["code"] == "SCOPE_DENIED"
    assert "other workspace source" not in str(payload)


def _correction_body(claim, current, fix_event, *, operation_id="op:t20:scope"):
    return {
        "schema_version": "1.0",
        "operation_id": operation_id,
        "claim_id": claim["claim_id"],
        "expected_revision_id": claim["revision_id"],
        "expected_projection_generation": current["projection_generation"],
        "new_value": 4200,
        "display_text": "端口改为4200",
        "evidence": [evidence_ref(fix_event)],
        "supersedes_revision_ids": [claim["revision_id"]],
        "reason": "synthetic correction",
    }


def test_correction_denies_other_project_in_same_workspace(store, client):
    foreign = _foreign_event()
    receive_batch(store, make_batch([foreign]))
    claim = record(
        store,
        display_text=foreign["body"]["text"],
        value=4100,
        project_id="project:forbidden",
        evidence_event=foreign,
    )
    before = get_current(store, claim["claim_id"])
    fix = make_event(native_id="message:fix", text="端口改为4200")
    assert client.post_batch(make_batch([fix]))[0] == 200
    status, payload = client.request(
        "POST", "/v1/corrections", _correction_body(claim, before, fix)
    )
    after = get_current(store, claim["claim_id"])
    assert status == 403, payload
    assert payload["error"]["code"] == "SCOPE_DENIED"
    assert after == before, "a denied correction must not move the projection"


def test_correction_denies_cross_project_evidence_for_authorized_claim(store, socket_path):
    """A multi-project binding still may not use project B evidence for A."""
    register_client(
        store,
        client_id="client:multi",
        workspace_id="owner:demo",
        host="hermes",
        instance_id="instance:hermes-test",
        actor_ids=["*"],
        project_ids=["project:songta", "project:forbidden"],
    )
    from knowledge_weaver.service import ServiceApp
    from knowledge_weaver.service.client import ServiceClient

    app = ServiceApp(store, socket_path).start()
    try:
        multi = ServiceClient(app.socket_path, client_id="client:multi")
        claim_event = make_event(native_id="message:claim", text="端口4100")
        assert multi.post_batch(make_batch([claim_event]))[0] == 200
        claim = record(
            store,
            display_text="端口4100",
            value=4100,
            evidence_event=claim_event,
        )
        current = get_current(store, claim["claim_id"])
        foreign_fix = _foreign_event(native_id="message:foreign-fix")
        receive_batch(store, make_batch([foreign_fix]))
        body = _correction_body(claim, current, foreign_fix)
        status, payload = multi.request("POST", "/v1/corrections", body)
        after = get_current(store, claim["claim_id"])
        assert status == 403, payload
        assert payload["error"]["code"] == "SCOPE_DENIED"
        assert after == current
    finally:
        app.stop()


def test_authorized_project_correction_still_applies(store, client):
    claim_event = make_event(native_id="message:claim-ok", text="端口4100")
    assert client.post_batch(make_batch([claim_event]))[0] == 200
    claim = record(store, display_text="端口4100", value=4100, evidence_event=claim_event)
    current = get_current(store, claim["claim_id"])
    fix = make_event(native_id="message:fix-ok", text="端口改为4200")
    assert client.post_batch(make_batch([fix]))[0] == 200
    status, payload = client.request(
        "POST", "/v1/corrections", _correction_body(claim, current, fix, operation_id="op:ok:1")
    )
    assert status == 200, payload
    assert payload["status"] == "applied"
    after = get_current(store, claim["claim_id"])
    assert after["selected_revision_ids"] == [payload["revision_id"]]
