"""T20 round-4 (fix3) F04 regression tests: condition grounding must be explicit.

Round-3 independent review kept F04 open: the backend accepted a condition name
merely appearing anywhere in the stored sanitized part text as if the review had
succeeded, and it used the whole part text even when the proposal referenced a
narrow ``support_quote`` span. These tests pin the round-4 contract (AC-09/36):

* condition grounding is computed only from the backend-recorded text span of
  each referenced ``support_quote`` (never from the rest of the part);
* every ``conditions``/``satisfied`` entry needs an explicit successful review
  result bound to its name in that span; FAILED / NOT RUN / pending / TODO /
  skipped / checklist-only mentions remain unverified;
* one ungrounded condition downgrades the whole completion to
  ``reported_done`` with an auditable ``backend_downgraded`` reason, while the
  successful tool observation itself stays;
* explicit successes (e.g. ``security_review passed`` / ``security_review 通过``)
  still project as ``verified_done``.

Counterexample source (Codex round-3 independent review, rewritten with repo
fixtures and the real extraction/projection pipeline):
``artifacts/T20-codex-round3/run-20261031-independent/test_independent_v2.py::
test_F04_mentions_do_not_prove_condition_success``.
"""

from __future__ import annotations

import json

import pytest

from knowledge_weaver.extraction import (
    load_event_group,
    run_extraction_job,
    run_worker_once,
    sanitized_parts,
)
from knowledge_weaver.jobs import claim
from knowledge_weaver.store import list_task_views, receive_batch

from ..store.conftest import make_batch, make_event
from .test_extraction import MockExtractor, _claim, _proposal

ACCEPTANCE_CONDITIONS = ("security_review", "restore_validation")


# ---------------------------------------------------------------------------
# helpers (real receive -> sanitized parts -> worker -> task projection)
# ---------------------------------------------------------------------------


def _receive_tool(store, text, *, native_id):
    event = make_event(native_id=native_id, text=text, role="tool", origin_kind="tool_execution")
    received = receive_batch(store, make_batch([event], batch_id=f"batch:{native_id}"))
    assert received["receipts"][0]["disposition"] == "accepted", received
    return event


def _only_part(store, event):
    parts = sanitized_parts(load_event_group(store, event["event_id"], 1))
    assert len(parts) == 1, parts
    return parts[0]


def _verified_value(conditions=ACCEPTANCE_CONDITIONS):
    names = list(conditions)
    return {"status": "verified", "conditions": names, "satisfied": list(names)}


def _task_claim(part, value, *, quote=None):
    return _claim(
        part,
        quote=quote,
        predicate="task.status",
        facet="observed",
        value=value,
        display_text=part.text,
        assertion_kind="observation",
    )


def _propose_task(store, part, value, *, quote=None):
    result = run_worker_once(store, MockExtractor(_proposal(_task_claim(part, value, quote=quote))))
    assert result["state"] == "succeeded", result
    return result


def _only_view(store):
    views = list_task_views(store, workspace_id="owner:demo")
    assert len(views) == 1, views
    return views[0]


def _stored_task_value(store):
    with store.read() as conn:
        row = conn.execute("SELECT value_json FROM claim_revisions").fetchone()
    assert row is not None
    return json.loads(row["value_json"])


def _assert_downgraded_to_report(store):
    """A completion without grounded successes stays an auditable report."""
    view = _only_view(store)
    assert view["status"] == "reported_done", view
    assert view["verified_revision_id"] is None, view
    value = _stored_task_value(store)
    assert value["status"] == "reported_done", value
    assert value.get("backend_downgraded") == "conditions_not_grounded_in_evidence", value
    return view


# ---------------------------------------------------------------------------
# F04: the round-3 independent counterexample, same pipeline semantics
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "quote"),
    [
        ("security_review FAILED; restore_validation NOT RUN", None),
        (
            "exit_code=0; requested checks: security_review, restore_validation. No reviews executed.",
            "exit_code=0",
        ),
    ],
    ids=["negative-results", "names-outside-quoted-exit0"],
)
def test_F04_mentions_do_not_prove_condition_success(store, text, quote):
    """A condition name mention is not proof that the condition succeeded."""
    event = _receive_tool(store, text, native_id="message:fix3:f04:mentions")
    part = _only_part(store, event)
    _propose_task(store, part, _verified_value(), quote=quote)
    _assert_downgraded_to_report(store)


@pytest.mark.parametrize(
    "text",
    [
        "security_review failed; restore_validation failed",
        "security_review NOT RUN; restore_validation NOT RUN",
        "security_review 失败；restore_validation 未执行",
        "security_review 待执行；restore_validation pending",
        "security_review TODO; restore_validation skipped",
        "security_review not executed; restore_validation not run",
        "security_review 未通过；restore_validation 未成功",
    ],
    ids=[
        "failed",
        "not-run",
        "zh-failed-not-executed",
        "zh-pending",
        "todo-skipped",
        "not-executed",
        "zh-not-passed",
    ],
)
def test_F04_negative_or_unexecuted_review_cannot_verify(store, text):
    """FAILED/NOT RUN/待执行/pending/TODO/skip are never successful reviews."""
    event = _receive_tool(store, text, native_id="message:fix3:f04:negative")
    part = _only_part(store, event)
    _propose_task(store, part, _verified_value())
    _assert_downgraded_to_report(store)


@pytest.mark.parametrize(
    "text",
    [
        "security_review passed but then FAILED; restore_validation passed",
        "security_review passed but cannot be verified; restore_validation passed",
        "security_review passed with failures; restore_validation passed",
        "security_review passed but not ok; restore_validation passed",
    ],
    ids=["mixed-failed", "negated-success", "success-with-failures", "negated-ok"],
)
def test_F04_negative_mention_overrides_success_word_in_same_clause(store, text):
    """A mixed clause must not be turned into success by an adjacent success word."""
    event = _receive_tool(store, text, native_id="message:fix3:f04:mixed-clause")
    part = _only_part(store, event)
    _propose_task(store, part, _verified_value())
    _assert_downgraded_to_report(store)


def test_F04_checklist_names_without_success_result_cannot_verify(store):
    """Only listing the requested checks is not a recorded result."""
    text = "requested checks: security_review, restore_validation"
    event = _receive_tool(store, text, native_id="message:fix3:f04:checklist")
    part = _only_part(store, event)
    _propose_task(store, part, _verified_value())
    _assert_downgraded_to_report(store)


def test_F04_mixed_positive_and_negative_results_downgrade_whole_item(store):
    """One ungrounded condition downgrades the whole completion (fail closed)."""
    text = "security_review passed; restore_validation NOT RUN"
    event = _receive_tool(store, text, native_id="message:fix3:f04:mixed")
    part = _only_part(store, event)
    _propose_task(store, part, _verified_value())
    view = _assert_downgraded_to_report(store)
    assert view["verification_conditions"] == [], view


def test_F04_condition_outside_quoted_span_cannot_verify(store):
    """A success quoted for one condition cannot ground a condition outside the span."""
    text = "security_review passed; restore_validation NOT RUN"
    event = _receive_tool(store, text, native_id="message:fix3:f04:outside-span")
    part = _only_part(store, event)
    _propose_task(store, part, _verified_value(), quote="security_review passed")
    _assert_downgraded_to_report(store)


# ---------------------------------------------------------------------------
# F04: legitimate explicit successes keep projecting as verified_done
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "security_review passed; restore_validation passed",
        "security_review PASS; restore_validation PASS",
        "security_review succeeded; restore_validation succeeded",
        "security_review 通过；restore_validation 通过",
        "security_review 已完成；restore_validation 已完成",
    ],
    ids=["passed", "upper-pass", "succeeded", "zh-passed", "zh-completed"],
)
def test_F04_explicit_success_results_still_verify(store, text):
    """The fail-closed rule must not reject genuinely recorded review results."""
    event = _receive_tool(store, text, native_id="message:fix3:f04:positive")
    part = _only_part(store, event)
    _propose_task(store, part, _verified_value())
    view = _only_view(store)
    assert view["status"] == "verified_done", view
    assert view["verified_revision_id"] is not None, view
    assert view["verification_conditions"] == list(ACCEPTANCE_CONDITIONS), view
    assert view["satisfied_conditions"] == list(ACCEPTANCE_CONDITIONS), view


def test_F04_ungrounded_condition_downgrade_is_warned_in_job_result(store):
    """The downgrade is reported as a warning, so it is auditable, not silent."""
    text = "security_review FAILED; restore_validation NOT RUN"
    event = _receive_tool(store, text, native_id="message:fix3:f04:warn")
    part = _only_part(store, event)
    value = _verified_value()
    job = claim(store, worker_id="worker:fix3", kinds=("extract",), limit=1)[0]
    result = run_extraction_job(store, job, MockExtractor(_proposal(_task_claim(part, value))))
    assert result["claims_written"] == 1, result
    assert any("grounding" in warning for warning in result.get("warnings") or []), result
    _assert_downgraded_to_report(store)
