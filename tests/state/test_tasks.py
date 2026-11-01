"""AC-09/36: reported completion vs verified evidence vs failure, and recall tasks."""
from __future__ import annotations

from knowledge_weaver.retrieval import execute_recall
from knowledge_weaver.store import list_task_views

from .conftest import create_user_event, record


def _task_claim(
    store,
    *,
    value,
    facet="reported",
    authority="reported",
    assertion_kind="completion_report",
    native_id=None,
    text=None,
):
    event = create_user_event(store, text or f"任务状态 {value} 说明", native_id=native_id)
    return record(
        store,
        display_text=text or f"任务状态 {value}",
        value=value,
        predicate="task.status",
        facet=facet,
        authority=authority,
        assertion_kind=assertion_kind,
        subject_id="subject:task:deploy",
        native_id=native_id,
        evidence_event=event,
    )


def _view(store):
    views = list_task_views(store, workspace_id="owner:demo")
    return views[0] if views else None


def test_reported_done_is_not_verified(store):
    _task_claim(store, value="reported_done", facet="reported", authority="reported")
    view = _view(store)
    assert view["status"] == "reported_done"
    assert view["reported_revision_id"] is not None
    assert view["verified_revision_id"] is None


def test_tool_failure_overrides_reported_completion(store):
    _task_claim(store, value="reported_done", facet="reported", authority="reported")
    _task_claim(
        store,
        value={"status": "failed", "conditions": ["functional-test"]},
        facet="observed",
        authority="tool_verified",
        assertion_kind="observation",
        text="功能测试失败",
    )
    view = _view(store)
    assert view["status"] == "failed"
    assert view["failed_revision_id"] is not None


def test_command_success_does_not_verify_whole_task(store):
    _task_claim(
        store,
        value={"status": "passed", "conditions": ["command"], "satisfied": ["command"]},
        facet="observed",
        authority="tool_verified",
        assertion_kind="observation",
        text="命令 exit 0",
    )
    _task_claim(
        store,
        value={"status": "failed", "conditions": ["functional"], "satisfied": []},
        facet="observed",
        authority="tool_verified",
        assertion_kind="observation",
        text="功能测试失败",
    )
    view = _view(store)
    assert view["status"] == "failed"
    assert view["verified_revision_id"] is None


def test_verified_requires_all_conditions_and_evidence(store):
    _task_claim(
        store,
        value={"status": "verified", "conditions": ["a", "b"], "satisfied": ["a", "b"]},
        facet="observed",
        authority="tool_verified",
        assertion_kind="observation",
        text="验收 a,b 通过",
    )
    view = _view(store)
    assert view["status"] == "verified_done"
    assert view["verified_revision_id"] is not None
    assert view["verification_conditions"] == ["a", "b"]
    assert view["satisfied_conditions"] == ["a", "b"]


def test_partial_conditions_are_not_verified(store):
    _task_claim(
        store,
        value={"status": "verified", "conditions": ["a", "b"], "satisfied": ["a"]},
        facet="observed",
        authority="tool_verified",
        assertion_kind="observation",
        text="只通过 a",
    )
    view = _view(store)
    assert view["status"] == "open"
    assert view["verified_revision_id"] is None


def test_retracted_evidence_degrades_task_view(store):
    from knowledge_weaver import contracts
    from knowledge_weaver.store import receive_batch

    from ..store.conftest import make_batch, make_event

    event = make_event(native_id="message:task-evidence", text="验收 a,b 通过")
    receive_batch(store, make_batch([event]))
    from knowledge_weaver.store.claims import record_claim_revision

    claim = record_claim_revision(
        store,
        workspace_id="owner:demo",
        project_id="project:songta",
        subject_id="subject:task:deploy",
        predicate="task.status",
        facet="observed",
        qualifiers={},
        value={"status": "verified", "conditions": ["a"], "satisfied": ["a"]},
        display_text="验收 a,b 通过",
        authority="tool_verified",
        assertion_kind="observation",
        event_time=event["event_time"],
        evidence=[
            {
                "event_id": event["event_id"],
                "event_revision": 1,
                "part_index": 0,
                "start_byte": 0,
                "end_byte": len("验收 a,b 通过".encode("utf-8")),
                "body_sha256": event["body"]["sha256"],
                "availability": "full",
            }
        ],
    )
    assert _view(store)["status"] == "verified_done"
    retract = make_event(native_id="message:task-evidence", revision=2, operation="retract", parent_revision=1, text="")
    retract["body"]["sha256"] = contracts.EMPTY_BODY_SHA256
    receive_batch(store, make_batch([retract], batch_id="batch:retract-task"))
    view = _view(store)
    assert view is None or view["status"] in ("unknown", "open")
    assert (view or {}).get("verified_revision_id") is None


def test_recall_tasks_intent_filters_task_claims(store):
    _task_claim(store, value="reported_done", facet="reported", authority="reported")
    record(store, display_text="端口4200", value=4200, predicate="server.port")
    binding = {
        "client_id": "client:test",
        "workspace_id": "owner:demo",
        "project_ids": ["project:songta"],
        "actor_ids": ["*"],
    }
    request = {
        "schema_version": "1.0",
        "request_id": "req:tasks:1",
        "session_id": "session:test",
        "turn_id": "turn:1",
        "query": "任务状态",
        "project_ids": ["project:songta"],
        "intent": "tasks",
        "budget_tokens": 2000,
        "limit": 8,
        "deadline_ms": 7000,
        "cursor": None,
    }
    result = execute_recall(store, request, binding=binding)
    assert result["items"]
    assert all(item["predicate"] == "task.status" for item in result["items"])


def test_verified_level_from_self_reported_authority_is_not_trusted(store):
    """F04 defence in depth: even a stored 'verified' value cannot verify alone."""
    _task_claim(
        store,
        value={
            "status": "verified",
            "conditions": ["functional"],
            "satisfied": ["functional"],
        },
        facet="observed",
        authority="reported",
        assertion_kind="completion_report",
        text="助手自报已完成",
    )
    view = _view(store)
    assert view["status"] == "reported_done"
    assert view["verified_revision_id"] is None


def test_completion_report_assertion_never_verifies_even_with_tool_authority(store):
    _task_claim(
        store,
        value={"status": "verified", "conditions": ["a"], "satisfied": ["a"]},
        facet="observed",
        authority="tool_verified",
        assertion_kind="completion_report",
        text="工具报告声称任务完成",
    )
    view = _view(store)
    assert view["status"] != "verified_done"
    assert view["verified_revision_id"] is None
