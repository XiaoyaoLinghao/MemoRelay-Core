"""T20 round-5 (fix4) F04 regression tests: success must be the condition's own result.

Round-4 independent review kept F04 open (review/T20-CODEX-ROUND4.md §2
"F04-R4 / AC-09、AC-36"): the round-4 rule only required a condition name and a
success word in the same clause, so the following all produced a false
``verified_done``:

* an unrelated action/object success after the name
  (``security_review log upload succeeded while review result is red``) or an
  unrelated saving step (``security_review output saved OK but the review has
  not occurred``);
* negation/future intent/false assignment/quantified half-results that were not
  in the round-4 failure word list (``not yet passed``, ``passed = false``,
  ``has zero passed checks``, ``scheduled to be verified tomorrow``, ``待通过``,
  ``并非通过``);
* an unresolved conflict inside one referenced span (``security_review passed;
  security_review FAILED`` and ``security_review FAIL; security_review
  passed``), because grounding stopped at the first occurrence / first success.

These tests pin the round-5 contract through the real product path only:
``receive_batch -> sanitized_parts(load_event_group) -> run_worker_once /
run_extraction_job(MockExtractor) -> list_task_views``. The model proposal is
untrusted input; the backend decides the verification level.

Contract (AC-09/AC-36):

* a success word only grounds a condition when it is *bound* to that condition
  as its own predicate/conclusion (``security_review passed`` /
  ``security_review 已通过`` / ``security_review: PASS`` /
  ``security_review has passed``); only neutral function words may stand
  between the name and the predicate;
* a referenced span that records any failure / NOT RUN / pending / future /
  negated / false / quantified / unknown result vetoes the completion, even when
  another success word appears there: an unresolved conflict is not proof;
* every occurrence of the condition in every referenced span is inspected, so
  the conflict cannot be missed by looking at the first occurrence only;
* unknown wording fails closed: the task becomes ``reported_done`` with an
  auditable ``backend_downgraded`` reason in the stored claim value and a
  warning in the worker/job result, while the real tool observation (event,
  evidence span, source) is preserved;
* genuine explicit results (passed / PASS / succeeded / 通过 / 已完成 /
  ``security_review: passed`` / ``已通过``) still project as ``verified_done``.

Counterexample source (read-only): Codex round-4 independent review
``review/T20-CODEX-ROUND4.md`` §2 F04-R4 and
``artifacts/T20-codex-round4/run-20261031-independent/repros/``
(``test_new_variants.py::test_F04_negative_or_unrelated_success_must_not_verify``,
``test_independent_v2.py::test_F04_mentions_do_not_prove_condition_success``),
adapted to repo fixtures and the real pipeline in
``work/t20-fix4/repros/test_new_variants.py``.
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
from knowledge_weaver.store import current_items, list_task_views, receive_batch

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


def _evidence_ref(part, quote=None):
    return {
        "event_id": part.event_id,
        "event_revision": part.event_revision,
        "part_index": part.part_index,
        "support_quote": quote or part.text,
    }


def _task_claim(part, value, *, quotes=None):
    """A model proposal for ``task.status``; quotes select the referenced spans."""
    quotes = [None] if quotes is None else list(quotes)
    return _claim(
        part,
        quote=quotes[0],
        predicate="task.status",
        facet="observed",
        value=value,
        display_text=part.text,
        assertion_kind="observation",
        evidence=[_evidence_ref(part, quote) for quote in quotes],
    )


def _propose_task(store, part, value, *, quotes=None):
    result = run_worker_once(store, MockExtractor(_proposal(_task_claim(part, value, quotes=quotes))))
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


def _task_item(store):
    items = [item for item in current_items(store, workspace_id="owner:demo") if item["predicate"] == "task.status"]
    assert len(items) == 1, items
    return items[0]


def _evidence_links(store):
    with store.read() as conn:
        return [dict(row) for row in conn.execute("SELECT * FROM evidence_links ORDER BY rowid")]


def _assert_downgraded_to_report(store):
    """An ungrounded/conflicted completion stays an auditable report, never verified."""
    view = _only_view(store)
    assert view["status"] == "reported_done", view
    assert view["status"] != "verified_done", view
    assert view["verified_revision_id"] is None, view
    assert view["verification_conditions"] == [], view
    value = _stored_task_value(store)
    assert value["status"] == "reported_done", value
    assert value.get("backend_downgraded") == "conditions_not_grounded_in_evidence", value
    return view


def _assert_tool_observation_preserved(store, event, *, expected_links: int = 1):
    """The downgrade only changes the verification level, never the tool evidence."""
    item = _task_item(store)
    assert item["facet"] == "observed", item
    assert item["evidence_grade"] == "tool_verified", item
    assert item["sources"], item
    assert item["sources"][0]["event_id"] == event["event_id"], item
    links = _evidence_links(store)
    assert len(links) == expected_links, links
    for link in links:
        assert link["event_id"] == event["event_id"], link
        assert link["body_sha256"] == event["body"]["sha256"], link
        assert link["availability"] == "full", link
        assert 0 <= link["start_byte"] < link["end_byte"], link
    return item


# ---------------------------------------------------------------------------
# F04 round-5: the 10 false-positive Codex round-4 variants (+ NOT RUN sibling)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        # Codex round-4 F04-R4 variants (artifacts/T20-codex-round4/.../repros)
        "security_review not yet passed; restore_validation passed",
        "security_review passed = false; restore_validation passed",
        "security_review has zero passed checks; restore_validation passed",
        "security_review output saved OK but the review has not occurred; restore_validation passed",
        "security_review log upload succeeded while review result is red; restore_validation passed",
        "security_review is scheduled to be verified tomorrow; restore_validation passed",
        "security_review 待通过；restore_validation 通过",
        "security_review 并非通过；restore_validation 通过",
        "security_review passed; security_review FAILED; restore_validation passed",
        "security_review FAIL; security_review passed; restore_validation passed",
        # sibling form the round-4 rule already caught; kept so the fix does not regress
        "security_review (NOT RUN), log upload succeeded; restore_validation passed",
    ],
    ids=[
        "not-yet-passed",
        "passed-false-assignment",
        "zero-passed-checks",
        "saved-ok-review-not-occurred",
        "upload-succeeded-review-red",
        "scheduled-to-be-verified-tomorrow",
        "zh-pending-passed",
        "zh-not-passed",
        "passed-then-failed",
        "fail-then-passed",
        "not-run-with-upload-success",
    ],
)
def test_F04_fix4_codex_negative_variants_must_not_verify(store, text):
    """Unrelated/negated/future/false/quantified/conflicted results never verify."""
    event = _receive_tool(store, text, native_id="message:fix4:f04:codex")
    part = _only_part(store, event)
    _propose_task(store, part, _verified_value())
    _assert_downgraded_to_report(store)
    _assert_tool_observation_preserved(store, event)


# ---------------------------------------------------------------------------
# F04 round-5: other expression forms and state transitions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        # Chinese not-yet / not-passed / future-intent state transitions
        "security_review 尚未通过；restore_validation 通过",
        "security_review 未通过；restore_validation 通过",
        "security_review 预计下月验证；restore_validation 通过",
        # unknown / non-predicate wording ("whether passed" is not a result)
        "security_review 是否通过；restore_validation 通过",
        "security_review: unknown; restore_validation passed",
        # negation and success words in different clauses of the same span
        "security_review passed; however the scheduled re-check has not happened yet. restore_validation passed",
        # quantified half-result: a partial count is not a full pass
        "security_review passed 1 of 3 checks; restore_validation passed",
    ],
    ids=[
        "zh-not-yet-passed",
        "zh-not-passed",
        "zh-expected-next-month-verification",
        "zh-whether-passed",
        "unknown-result",
        "same-span-other-clause-negation",
        "partial-passed-count",
    ],
)
def test_F04_fix4_state_and_expression_variants_must_not_verify(store, text):
    """One negative/unresolved clause in the quoted span vetoes the whole item."""
    event = _receive_tool(store, text, native_id="message:fix4:f04:variants")
    part = _only_part(store, event)
    _propose_task(store, part, _verified_value())
    _assert_downgraded_to_report(store)


def test_F04_fix4_conflicting_quotes_one_pass_one_fail_must_not_verify(store):
    """Two referenced spans, one ``passed`` and one ``FAILED``: unresolved conflict.

    The value declares only ``security_review`` so the test isolates the
    conflict rule: a first-occurrence/first-success implementation would verify
    from the ``security_review passed`` quote; the pass+fail conflict must not.
    """
    text = "security_review passed; security_review FAILED; restore_validation passed"
    event = _receive_tool(store, text, native_id="message:fix4:f04:conflict")
    part = _only_part(store, event)
    _propose_task(
        store,
        part,
        _verified_value(("security_review",)),
        quotes=["security_review passed", "security_review FAILED"],
    )
    view = _assert_downgraded_to_report(store)
    assert view["verified_revision_id"] is None, view
    _assert_tool_observation_preserved(store, event, expected_links=2)


def test_F04_fix4_opposite_occurrence_order_is_also_refused(store):
    """``FAIL; passed`` order is refused too: the rule scans every occurrence."""
    text = "security_review FAIL; security_review passed; restore_validation passed"
    event = _receive_tool(store, text, native_id="message:fix4:f04:order")
    part = _only_part(store, event)
    _propose_task(store, part, _verified_value(("security_review",)))
    _assert_downgraded_to_report(store)


# ---------------------------------------------------------------------------
# F04 round-5: auditable downgrade (reason + warnings + real tool observation)
# ---------------------------------------------------------------------------


def test_F04_fix4_downgrade_reason_and_warnings_are_auditable(store):
    """The downgrade is explicit in the claim value and in the job warnings.

    The task projection (``list_task_views``) intentionally exposes only the
    derived status/conditions, not ``backend_downgraded``; the reason is
    therefore asserted on the stored claim value, and the warning on the
    ``run_extraction_job`` result — both are real product outputs of the same
    pipeline run.
    """
    text = "security_review log upload succeeded while review result is red; restore_validation passed"
    event = _receive_tool(store, text, native_id="message:fix4:f04:audit")
    part = _only_part(store, event)
    value = _verified_value()
    job = claim(store, worker_id="worker:fix4", kinds=("extract",), limit=1)[0]
    result = run_extraction_job(store, job, MockExtractor(_proposal(_task_claim(part, value))))
    assert result["claims_written"] == 1, result
    warnings = result.get("warnings") or []
    assert any("grounding" in warning for warning in warnings), result
    assert any("downgraded" in warning for warning in warnings), result

    view = _only_view(store)
    assert view["status"] == "reported_done", view
    assert view["verified_revision_id"] is None, view
    stored = _stored_task_value(store)
    assert stored["status"] == "reported_done", stored
    assert stored.get("backend_downgraded") == "conditions_not_grounded_in_evidence", stored
    assert stored.get("conditions") == list(ACCEPTANCE_CONDITIONS), stored
    _assert_tool_observation_preserved(store, event)


# ---------------------------------------------------------------------------
# F04 round-5: genuine explicit results keep verifying (controls)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "security_review passed; restore_validation passed",
        "security_review PASS; restore_validation PASS",
        "security_review succeeded; restore_validation succeeded",
        "security_review 通过；restore_validation 通过",
        "security_review 已完成; restore_validation 已完成",
        "security_review: passed; restore_validation: passed",
        "security_review has passed; restore_validation has passed",
        "security_review 已通过；restore_validation 已通过",
        "security_review checks passed; restore_validation checks passed",
        "security_review passed 3 of 3 checks; restore_validation passed",
    ],
    ids=[
        "passed",
        "upper-pass",
        "succeeded",
        "zh-passed",
        "zh-completed",
        "colon-pass",
        "has-passed",
        "zh-yi-passed",
        "checks-passed",
        "full-count-passed",
    ],
)
def test_F04_fix4_explicit_predicate_success_still_verifies(store, text):
    """The binding grammar must not reject genuinely recorded review results."""
    event = _receive_tool(store, text, native_id="message:fix4:f04:positive")
    part = _only_part(store, event)
    _propose_task(store, part, _verified_value())
    view = _only_view(store)
    assert view["status"] == "verified_done", view
    assert view["verified_revision_id"] is not None, view
    assert view["verification_conditions"] == list(ACCEPTANCE_CONDITIONS), view
    assert view["satisfied_conditions"] == list(ACCEPTANCE_CONDITIONS), view


def test_F04_fix4_positive_projection_and_evidence_are_auditable(store):
    """A genuine explicit result keeps its tool_execution source and evidence span."""
    text = "security_review: passed; restore_validation: passed"
    event = _receive_tool(store, text, native_id="message:fix4:f04:positive-audit")
    part = _only_part(store, event)
    job = claim(store, worker_id="worker:fix4", kinds=("extract",), limit=1)[0]
    result = run_extraction_job(
        store,
        job,
        MockExtractor(_proposal(_task_claim(part, _verified_value()))),
    )
    assert result["claims_written"] == 1, result
    assert result.get("warnings") == [], result

    view = _only_view(store)
    assert view["status"] == "verified_done", view
    assert view["verified_revision_id"] is not None, view
    assert view["verification_conditions"] == list(ACCEPTANCE_CONDITIONS), view
    assert view["satisfied_conditions"] == list(ACCEPTANCE_CONDITIONS), view

    item = _task_item(store)
    assert item["facet"] == "observed", item
    assert item["evidence_grade"] == "tool_verified", item
    assert item["sources"][0]["event_id"] == event["event_id"], item
    links = _evidence_links(store)
    assert len(links) == 1, links
    assert links[0]["body_sha256"] == event["body"]["sha256"], links
    assert links[0]["start_byte"] == 0, links
    assert links[0]["end_byte"] == len(text.encode("utf-8")), links
    stored = _stored_task_value(store)
    assert stored["status"] == "verified", stored
    assert "backend_downgraded" not in stored, stored
