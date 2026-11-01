"""Shared fixtures/builders for store tests (synthetic data only)."""
from __future__ import annotations

import copy

import pytest

from knowledge_weaver import contracts
from knowledge_weaver.store import Store


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "kw" / "knowledge.db")


def assert_schema(name, value):
    issues = contracts.validate_instance(name, value)
    assert issues == [], [str(issue) for issue in issues]


def make_event(
    *,
    native_id="message:1",
    workspace_id="owner:demo",
    instance_id="instance:hermes-test",
    host="hermes",
    actor_id="jun-test",
    session_id="session:test",
    role="user",
    origin_kind="user_input",
    revision=1,
    operation="create",
    parent_revision=None,
    text="松塔项目端口改为4200。",
    part_index=0,
    part_count=1,
    content_revision="native:1",
    event_time=None,
    project_ids=("project:songta",),
    redaction=None,
    derived_from=None,
):
    event = {
        "schema_version": "1.0",
        "event_id": contracts.canonical_event_id(workspace_id, instance_id, native_id),
        "event_revision": revision,
        "workspace_id": workspace_id,
        "operation": operation,
        "source": {
            "instance_id": instance_id,
            "host": host,
            "actor_id": actor_id,
            "session_id": session_id,
            "native_event_id": native_id,
            "native_schema": "hermes-snapshot-20261030",
            "role": role,
            "origin_kind": origin_kind,
            "locator": f"hermes://{session_id}/{native_id}",
        },
        "event_time": event_time
        or {"value": "2026-10-30T09:00:00+08:00", "precision": "instant", "timezone": "Asia/Shanghai"},
        "body": {
            "text": text,
            "sha256": contracts.body_digest(text),
            "part_index": part_index,
            "part_count": part_count,
            "content_revision": content_revision,
            "format": "text/plain",
        },
        "redaction": redaction
        or {"policy_version": "redact:1", "status": "passed", "replacements": 0, "credential_refs": []},
        "project_ids": list(project_ids),
        "parent_revision": parent_revision,
        "derived_from": list(derived_from or []),
    }
    return event


def make_batch(events, batch_id="batch:test:1"):
    return {"schema_version": "1.0", "batch_id": batch_id, "events": events}


def clone(value):
    return copy.deepcopy(value)


def fetch(store, event, revision=None, part=0):
    from knowledge_weaver.store import fetch_source

    return fetch_source(
        store, event["event_id"], int(revision if revision is not None else event["event_revision"]), part
    )
