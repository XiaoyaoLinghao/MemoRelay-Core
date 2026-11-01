"""T02: HTTP-over-UDS service behavior and contract-shaped responses."""
from __future__ import annotations

import os
import stat

import pytest

from knowledge_weaver import contracts

from .conftest import make_batch, make_event


def test_health_over_unix_socket(service, client):
    status, payload = client.health()
    assert status == 200
    issues = contracts.validate_instance("Health", payload)
    assert issues == []
    assert payload["status"] == "ready"


def test_socket_permissions_and_no_tcp(service):
    info = os.stat(service.socket_path)
    assert stat.S_ISSOCK(info.st_mode)
    assert stat.S_IMODE(info.st_mode) == 0o600
    assert stat.S_IMODE(os.stat(service.socket_path.parent).st_mode) == 0o700
    assert service._server is not None
    assert service._server.socket.family.name == "AF_UNIX"


def test_batch_roundtrip_and_duplicate(client):
    event = make_event()
    status, result = client.post_batch(make_batch([event], batch_id="batch:http:1"))
    assert status == 200
    assert contracts.validate_instance("BatchResult", result) == []
    receipt = result["receipts"][0]
    assert receipt["disposition"] == "accepted"
    assert result["watermark"] >= 1
    status, duplicate = client.post_batch(make_batch([event], batch_id="batch:http:2"))
    assert status == 200
    assert duplicate["receipts"][0]["disposition"] == "duplicate"
    assert duplicate["receipts"][0]["receipt_id"] == receipt["receipt_id"]
    assert duplicate["watermark"] == result["watermark"]


def test_source_read_requires_revision_and_scopes(client):
    event = make_event()
    client.post_batch(make_batch([event]))
    status, source = client.get_source(event["event_id"], 1)
    assert status == 200
    assert contracts.validate_instance("SourceResult", source) == []
    assert source["text"] == event["body"]["text"]
    status, missing_revision = client.request("GET", f"/v1/sources/{event['event_id']}")
    assert status == 400
    assert missing_revision["error"]["code"] == "INVALID_REQUEST"
    status, unknown = client.get_source("evt:" + "a" * 64, 1)
    assert status == 404
    assert unknown["error"]["code"] == "SOURCE_NOT_FOUND"


def test_unregistered_client_is_denied(store, service):
    from knowledge_weaver.service.client import ServiceClient

    anonymous = ServiceClient(service.socket_path)
    status, payload = anonymous.post_batch(make_batch([make_event()]))
    assert status == 403
    assert payload["error"]["code"] == "UNREGISTERED_CLIENT"
    assert contracts.validate_instance("Error", payload) == []
    revoked = ServiceClient(service.socket_path, client_id="client:does-not-exist")
    status, payload = revoked.health()
    assert status == 200  # health is readable; writes require a binding
    status, payload = revoked.post_batch(make_batch([make_event()]))
    assert status == 403
    assert payload["error"]["code"] == "UNREGISTERED_CLIENT"


def test_scope_denied_for_other_workspace_and_actor(service, client):
    other_workspace = make_event(workspace_id="owner:other", native_id="message:other")
    status, payload = client.post_batch(make_batch([other_workspace], batch_id="batch:http:2"))
    assert status == 403
    assert payload["error"]["code"] == "SCOPE_DENIED"
    other_actor = make_event(actor_id="intruder", native_id="message:intruder")
    status, payload = client.post_batch(make_batch([other_actor], batch_id="batch:http:3"))
    assert status == 403
    assert payload["error"]["code"] == "SCOPE_DENIED"
    other_project = make_event(
        native_id="message:proj", project_ids=("project:forbidden",)
    )
    status, payload = client.post_batch(make_batch([other_project], batch_id="batch:http:4"))
    assert status == 403
    assert payload["error"]["code"] == "SCOPE_DENIED"


def test_read_scope_checked_against_stored_workspace(store, service, binding):
    from knowledge_weaver.service.client import ServiceClient
    from knowledge_weaver.store import register_client

    register_client(
        store,
        client_id="client:kw-second",
        workspace_id="owner:other",
        host="hermes",
        instance_id="instance:other",
        actor_ids=["*"],
        project_ids=[],
    )
    other = ServiceClient(service.socket_path, client_id="client:kw-second")
    event = make_event(
        workspace_id="owner:other", instance_id="instance:other", native_id="message:other"
    )
    status, _ = other.post_batch(make_batch([event], batch_id="batch:http:5"))
    assert status == 200
    first = ServiceClient(service.socket_path, client_id="client:hermes-main")
    status, payload = first.get_source(event["event_id"], 1)
    assert status == 403
    assert payload["error"]["code"] == "SCOPE_DENIED"


def test_invalid_json_and_oversized_payload(service, client):
    status, payload = client.request_raw(
        "POST",
        "/v1/events:batch",
        b"{not json",
        headers={"Content-Type": "application/json"},
    )
    assert status == 400
    assert payload["error"]["code"] == "INVALID_REQUEST"
    from knowledge_weaver.service.app import MAX_REQUEST_BYTES
    from knowledge_weaver.service.client import UnixHTTPConnection

    connection = UnixHTTPConnection(service.socket_path)
    try:
        connection.putrequest("POST", "/v1/events:batch")
        connection.putheader("Content-Length", str(MAX_REQUEST_BYTES + 1))
        connection.putheader("X-DMA-Client", client.client_id)
        connection.endheaders()
        response = connection.getresponse()
        raw = response.read()
        assert response.status == 413
        assert b"PAYLOAD_TOO_LARGE" in raw
    finally:
        connection.close()
    assert service.store.count("SELECT COUNT(*) FROM events") == 0


def test_unsupported_schema_and_unknown_route(service, client):
    event = make_event()
    event["schema_version"] = "99.0"
    status, payload = client.post_batch(make_batch([event]))
    assert status == 400
    assert payload["error"]["code"] == "UNSUPPORTED_SCHEMA"
    status, payload = client.request("GET", "/v1/nope")
    assert status == 400
    assert payload["error"]["code"] == "INVALID_REQUEST"
    status, payload = client.request("POST", "/v1/recall", {})
    assert status == 400
    assert payload["error"]["code"] == "INVALID_REQUEST"


def test_request_id_is_echoed(service, client):
    from knowledge_weaver.service.client import UnixHTTPConnection
    import json

    connection = UnixHTTPConnection(service.socket_path)
    try:
        connection.request(
            "GET",
            "/v1/health",
            headers={"X-Request-Id": "req:test-fixed-1"},
        )
        response = connection.getresponse()
        payload = json.loads(response.read())
        assert response.status == 200
        assert response.getheader("X-Request-Id") == "req:test-fixed-1"
        assert payload["status"] == "ready"
    finally:
        connection.close()


def test_error_never_echoes_body(service, client):
    event = make_event(text="password CANARY-PW-7f3a")
    event["schema_version"] = "bad-version"
    status, payload = client.post_batch(make_batch([event]))
    assert status == 400
    assert "CANARY" not in payload["error"]["message"]
    assert "password" not in payload["error"]["message"]


def test_stop_removes_socket(store, socket_path):
    from knowledge_weaver.service import ServiceApp

    app = ServiceApp(store, socket_path)
    app.start()
    assert socket_path.exists()
    app.stop()
    assert not socket_path.exists()


def test_entry_recheck_withholds_secret_canary(service, client, store):
    event = make_event(text="数据库密码: CANARY-PW-7f3a")
    status, result = client.post_batch(make_batch([event]))
    assert status == 200
    receipt = result["receipts"][0]
    assert receipt["disposition"] == "withheld"
    assert receipt["error_code"] == "REDACTION_REQUIRED"
    assert receipt["durable_remote"] is True
    with store.read() as conn:
        part = conn.execute("SELECT body_text, availability FROM event_parts").fetchone()
        raw_db = service.store.db_path.read_bytes()
    assert part["body_text"] is None
    assert part["availability"] == "withheld"
    assert b"CANARY-PW-7f3a" not in raw_db  # secret never touches the database file


def test_corrections_endpoint_applies_and_validates(store, service, binding, client):
    from knowledge_weaver.store import current_items, get_current
    from knowledge_weaver.store.claims import record_claim_revision

    created = make_event(text="端口4100")
    status, _ = client.post_batch(make_batch([created]))
    assert status == 200
    ref = {
        "event_id": created["event_id"],
        "event_revision": 1,
        "part_index": 0,
        "start_byte": 0,
        "end_byte": len("端口4100".encode("utf-8")),
        "body_sha256": created["body"]["sha256"],
        "availability": "full",
    }
    claim = record_claim_revision(
        store,
        workspace_id="owner:demo",
        project_id="project:songta",
        subject_id="subject:songta-server",
        predicate="server.port",
        facet="desired",
        qualifiers={},
        value=4100,
        display_text="端口4100",
        authority="user_confirmed",
        assertion_kind="explicit_statement",
        event_time=created["event_time"],
        evidence=[ref],
    )
    current = get_current(store, claim["claim_id"])
    user_fix = make_event(native_id="message:correction", text="端口改为4200")
    status, _ = client.post_batch(make_batch([user_fix], batch_id="batch:http:corr"))
    assert status == 200
    correction = {
        "schema_version": "1.0",
        "operation_id": "op:http:1",
        "claim_id": claim["claim_id"],
        "expected_revision_id": claim["revision_id"],
        "expected_projection_generation": current["projection_generation"],
        "new_value": 4200,
        "display_text": "端口改为4200",
        "evidence": [
            {
                "event_id": user_fix["event_id"],
                "event_revision": 1,
                "part_index": 0,
                "start_byte": 0,
                "end_byte": len("端口改为4200".encode("utf-8")),
                "body_sha256": user_fix["body"]["sha256"],
                "availability": "full",
            }
        ],
        "supersedes_revision_ids": [claim["revision_id"]],
        "reason": "用户明确更正",
    }
    status, result = client.request("POST", "/v1/corrections", correction)
    assert status == 200
    assert contracts.validate_instance("CorrectionResult", result) == []
    assert result["status"] == "applied"
    items = current_items(store, workspace_id="owner:demo")
    assert items and items[0]["value"] == 4200

    # Replay is stable, not a second revision.
    status, replay = client.request("POST", "/v1/corrections", correction)
    assert status == 200 and replay["status"] == "duplicate"


def test_corrections_endpoint_rejects_foreign_claim_workspace(store, service, binding):
    from knowledge_weaver.service.client import ServiceClient
    from knowledge_weaver.store import register_client
    from knowledge_weaver.store.claims import record_claim_revision

    register_client(
        store,
        client_id="client:other-ws",
        workspace_id="owner:other",
        host="hermes",
        instance_id="instance:other",
        actor_ids=["*"],
        project_ids=[],
    )
    other = ServiceClient(service.socket_path, client_id="client:other-ws")
    event = make_event(workspace_id="owner:other", instance_id="instance:other", native_id="message:other")
    status, _ = other.post_batch(make_batch([event], batch_id="batch:other"))
    assert status == 200
    claim = record_claim_revision(
        store,
        workspace_id="owner:other",
        project_id=None,
        subject_id="subject:other",
        predicate="server.port",
        facet="desired",
        qualifiers={},
        value=80,
        display_text="端口80",
        authority="user_confirmed",
        assertion_kind="explicit_statement",
        event_time=event["event_time"],
        evidence=[
            {
                "event_id": event["event_id"],
                "event_revision": 1,
                "part_index": 0,
                "start_byte": 0,
                "end_byte": len("松塔项目端口改为4200。".encode("utf-8")),
                "body_sha256": event["body"]["sha256"],
                "availability": "full",
            }
        ],
    )
    first = ServiceClient(service.socket_path, client_id="client:hermes-main")
    correction = {
        "schema_version": "1.0",
        "operation_id": "op:foreign:1",
        "claim_id": claim["claim_id"],
        "expected_revision_id": claim["revision_id"],
        "expected_projection_generation": 1,
        "new_value": 81,
        "display_text": "端口81",
        "evidence": [
            {
                "event_id": event["event_id"],
                "event_revision": 1,
                "part_index": 0,
                "start_byte": 0,
                "end_byte": 5,
                "body_sha256": event["body"]["sha256"],
                "availability": "full",
            }
        ],
        "supersedes_revision_ids": [claim["revision_id"]],
        "reason": "forged",
    }
    status, payload = first.request("POST", "/v1/corrections", correction)
    assert status == 403
    assert payload["error"]["code"] == "SCOPE_DENIED"



def test_recall_endpoint_returns_contract_result(store, service, binding, client):
    from ..retrieval.conftest import recall_request
    from ..state.conftest import record

    record(store, display_text="松塔项目服务器端口为4200", value=4200, predicate="server.port")
    status, payload = client.request("POST", "/v1/recall", recall_request())
    assert status == 200
    assert contracts.validate_instance("RecallResult", payload) == []
    assert payload["items"]
    assert payload["items"][0]["project_id"] == "project:songta"
    # No embedder is wired in this service fixture: degradation is explicit.
    assert "embedding_unavailable" in payload["incomplete_reasons"]


def test_recall_endpoint_denies_unauthorized_scope(store, service, binding, client):
    from ..retrieval.conftest import recall_request

    status, payload = client.request(
        "POST", "/v1/recall", recall_request(project_ids=["project:other"])
    )
    assert status == 403
    assert payload["error"]["code"] == "SCOPE_DENIED"


def test_capabilities_and_resources_endpoints(store, service, binding, client):
    from knowledge_weaver.resource_registry import register_capability, register_resource

    capability = register_capability(
        store,
        workspace_id="owner:demo",
        actor_id="actor:hermes",
        name="kw-skill",
        source_uri="https://example.invalid/skill",
        version="1.0.0",
        content_hash="c" * 64,
        install_locator="~/.hermes/skills/kw-skill",
        state="available",
    )
    register_resource(
        store,
        workspace_id="owner:demo",
        project_id="project:songta",
        name="nas",
        endpoint="smb://nas.example.invalid/songta",
        account_label="readonly",
        credential_ref="credref:nas",
        recipe_id=capability["capability_id"],
        state="needs_os_authorization",
    )
    status, payload = client.request("GET", "/v1/capabilities")
    assert status == 200
    assert contracts.validate_instance("CapabilitiesResult", payload) == []
    assert payload["items"][0]["name"] == "kw-skill"
    status, payload = client.request("GET", "/v1/resources")
    assert status == 200
    assert contracts.validate_instance("ResourcesResult", payload) == []
    assert len(payload["items"]) == 1
    anonymous = client.request("GET", "/v1/capabilities")[0]  # still bound via fixture client
    assert anonymous == 200
    from knowledge_weaver.service.client import ServiceClient

    unbound = ServiceClient(service.socket_path)
    status, error = unbound.request("GET", "/v1/capabilities")
    assert status == 403
    assert error["error"]["code"] == "UNREGISTERED_CLIENT"
