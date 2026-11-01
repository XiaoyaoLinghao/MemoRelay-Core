"""T20 round-3 (fix2) F04 defence in depth in the task projection.

The proposal validation layer refuses to grant ``verified`` to a task completion
whose acceptance conditions have no grounding in the stored sanitized evidence.
The projection layer must independently refuse a value the backend itself marked
as downgraded, while keeping direct ``record()`` fixtures (tests/state/
test_tasks.py, tests/lifecycle/test_retention.py) untouched: those are already
backend-recorded evidence and are intentionally not re-grounded against text.

Counterexample source (independent round-2 review):
``artifacts/T20-round2/run-20261030-codex/test_variants.py``.
"""

from __future__ import annotations

from knowledge_weaver.store import list_task_views

from .conftest import create_user_event, record


def _record_task(
    store,
    value,
    *,
    authority="tool_verified",
    assertion_kind="observation",
    facet="observed",
    text="验收 a,b 通过",
):
    event = create_user_event(store, text)
    return record(
        store,
        display_text=text,
        value=value,
        predicate="task.status",
        facet=facet,
        authority=authority,
        assertion_kind=assertion_kind,
        subject_id="subject:task:deploy",
        evidence_event=event,
    )


def _only_view(store):
    views = list_task_views(store, workspace_id="owner:demo")
    assert views, "expected a task view"
    return views[0]


def test_backend_downgraded_verified_value_never_becomes_verified_done(store):
    """F04 defence in depth: a backend-downgraded value cannot verify."""
    _record_task(
        store,
        {
            "status": "verified",
            "conditions": ["security_review"],
            "satisfied": ["security_review"],
            "backend_downgraded": "conditions_not_grounded_in_evidence",
        },
    )
    view = _only_view(store)
    assert view["status"] == "reported_done", view
    assert view["verified_revision_id"] is None


def test_backend_downgraded_marker_wins_over_satisfied_conditions(store):
    """Every structural verified_done requirement is met yet the marker blocks it."""
    _record_task(
        store,
        {
            "status": "verified",
            "conditions": ["a", "b"],
            "satisfied": ["a", "b"],
            "backend_downgraded": "completion_report_cannot_verify",
        },
        authority="tool_verified",
        assertion_kind="observation",
    )
    view = _only_view(store)
    assert view["status"] != "verified_done", view
    assert view["verified_revision_id"] is None


def test_direct_verified_fixture_without_backend_marker_stays_verified(store):
    """Compatibility: direct record() fixtures are not re-grounded against text."""
    _record_task(
        store,
        {"status": "verified", "conditions": ["a", "b"], "satisfied": ["a", "b"]},
    )
    view = _only_view(store)
    assert view["status"] == "verified_done", view
    assert view["verified_revision_id"] is not None
    assert view["verification_conditions"] == ["a", "b"]
    assert view["satisfied_conditions"] == ["a", "b"]


def test_downgraded_marker_on_reported_value_stays_reported(store):
    _record_task(
        store,
        {
            "status": "reported_done",
            "conditions": [],
            "satisfied": [],
            "backend_downgraded": "conditions_not_grounded_in_evidence",
        },
    )
    view = _only_view(store)
    assert view["status"] == "reported_done", view
    assert view["verified_revision_id"] is None


def test_verified_level_without_conditions_stays_open(store):
    """verified_done still requires a non-empty acceptance-condition set."""
    _record_task(store, {"status": "verified", "conditions": [], "satisfied": []})
    view = _only_view(store)
    assert view["status"] == "open", view
    assert view["verified_revision_id"] is None
