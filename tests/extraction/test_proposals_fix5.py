"""T20 round-6 (fix5) F04 regression tests: the *complete conclusion* must ground.

Round-5 independent review kept F04 open (``review/T20-CODEX-ROUND5.md`` §3
"F04-R5 / AC-09、AC-36", P1): the round-5 binding only constrained the gap
between the condition name and the success word. Any wording *after* the success
word was unconstrained, so the following real
``receive_batch -> sanitized_parts -> run_worker_once -> list_task_views``
inputs still projected ``verified_done``:

* ``security_review 通过了吗；restore_validation 通过`` (question),
* ``security_review 通过是假设；restore_validation 通过`` (hypothesis),
* ``security_review passed is hypothetical; restore_validation passed``
  (hypothesis),
* ``security_review passed = 0; restore_validation passed`` (false assignment).

The fix is not a longer negative word list. The backend now requires the *whole
conclusion*, not just the name-to-success-word prefix: after a bound success
predicate, the rest of its clause may only belong to the result (neutral
completion fillers, a full review count such as ``3 of 3 checks``). A question
(``passed?`` / ``通过？``), a hypothesis or scoping modifier (``is
hypothetical`` / ``per a log upload`` / ``only in theory`` / ``了吗`` /
``是假设``) and a false assignment (``= 0`` / ``0 of 0``) are not a completed
asserted result and fail closed. A same-sentence qualifier clause
(``passed, hypothetically``) also fails, while a same-sentence clause that
coordinates another proposed acceptance conclusion (``passed,
restore_validation passed``) is still a complete conclusion.

Contract (AC-09/AC-36):

* a condition is grounded only by an explicit, *complete* success conclusion
  bound to its own name in the backend-recorded ``support_quote`` span; a
  question, hypothesis, false/empty result or any unproven modifier downgrades
  the whole completion instead of verifying it;
* ungrounded/qualified results become ``reported_done`` with the auditable
  ``backend_downgraded = "conditions_not_grounded_in_evidence"`` reason and a
  worker warning, while the real tool observation (event, evidence span, body
  hash) and the quoted span are preserved;
* genuine explicit results (passed / PASS / succeeded / 通过 / 已完成 / 已通过 /
  has passed / ``3 of 3 checks`` / ``2/2 checks`` / 通过了 / passed
  successfully) still project as ``verified_done``.

Counterexample sources (read-only): Codex round-5 independent review
``review/T20-CODEX-ROUND5.md`` §3 F04-R5 and
``artifacts/T20-codex-round5/run-20261031-independent/repros/
test_round5_variants.py::test_F04_unproven_suffix_or_question_must_not_verify``,
adapted to the repo fixtures and the real product pipeline. Round-4/round-5
regression suites stay in ``test_proposals_fix3.py`` / ``test_proposals_fix4.py``.

These tests pin the round-6 contract through the real product path only:
``receive_batch -> sanitized_parts(load_event_group) -> run_worker_once /
run_extraction_job(MockExtractor) -> list_task_views``. The model proposal is
untrusted input; the backend decides the verification level.
"""

from __future__ import annotations

import json

import pytest

from knowledge_weaver import contracts
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


def _task_claim(part, value, *, quote=None):
    """A model proposal for ``task.status``; ``quote`` selects the referenced span."""
    return _claim(
        part,
        quote=quote,
        predicate="task.status",
        facet="observed",
        value=value,
        display_text=part.text,
        assertion_kind="observation",
        evidence=[_evidence_ref(part, quote)],
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


def _task_item(store):
    items = [item for item in current_items(store, workspace_id="owner:demo") if item["predicate"] == "task.status"]
    assert len(items) == 1, items
    return items[0]


def _evidence_links(store):
    with store.read() as conn:
        return [dict(row) for row in conn.execute("SELECT * FROM evidence_links ORDER BY rowid")]


def _assert_downgraded_to_report(store):
    """A qualified/unproven completion stays an auditable report, never verified."""
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
# F04 round-6: the four blocking Codex round-5 real-entry inputs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        # Codex round-5 F04-R5 blocking inputs (review/T20-CODEX-ROUND5.md §3):
        # question, Chinese hypothesis, English hypothesis, false assignment.
        "security_review 通过了吗；restore_validation 通过",
        "security_review 通过是假设；restore_validation 通过",
        "security_review passed is hypothetical; restore_validation passed",
        "security_review passed = 0; restore_validation passed",
    ],
    ids=[
        "zh-question-le-ma",
        "zh-hypothesis-suffix",
        "en-hypothetical-suffix",
        "false-assignment-zero",
    ],
)
def test_F04_R5_codex_round5_blocking_inputs_must_not_verify(store, text):
    """A success word whose clause is a question/hypothesis/false assignment is not a result."""
    event = _receive_tool(store, text, native_id="message:fix5:f04:blocking")
    part = _only_part(store, event)
    _propose_task(store, part, _verified_value())
    _assert_downgraded_to_report(store)
    _assert_tool_observation_preserved(store, event)


# ---------------------------------------------------------------------------
# F04 round-6: the remaining Codex round-5 diagnostic inputs (all fail closed)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "security_review passed? restore_validation passed?",
        "security_review passed per a log upload; restore_validation passed",
        "security_review passed only in theory; restore_validation passed",
        "security_review passed 0 of 0 checks; restore_validation passed",
        "security_review allegedly passed; restore_validation passed",
        "security_review reportedly passed; restore_validation passed",
        "security_review 是否通过；restore_validation 通过",
    ],
    ids=[
        "question-mark",
        "per-log-upload",
        "only-in-theory",
        "empty-count-zero-of-zero",
        "allegedly",
        "reportedly",
        "zh-whether-passed",
    ],
)
def test_F04_R5_codex_round5_diagnostic_inputs_must_not_verify(store, text):
    """The other round-5 diagnostics also stay unverified (conservative fail closed)."""
    event = _receive_tool(store, text, native_id="message:fix5:f04:diagnostics")
    part = _only_part(store, event)
    _propose_task(store, part, _verified_value())
    _assert_downgraded_to_report(store)
    _assert_tool_observation_preserved(store, event)


# ---------------------------------------------------------------------------
# F04 round-6: adjacent punctuation / qualifier / partial variants
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        # punctuation forms of the same result classes
        "security_review passed? restore_validation passed",
        "security_review passed; restore_validation passed？",
        "security_review passed: 0; restore_validation passed",
        "security_review passed 2/3 checks; restore_validation passed",
        "security_review passed 0/0 checks; restore_validation passed",
        # a qualifier attached through a following clause of the same sentence
        "security_review passed, hypothetically; restore_validation passed",
        "security_review passed, which is hypothetical; restore_validation passed",
        "security_review 通过（假设）；restore_validation 通过",
        "security_review 通过了吗；restore_validation 通过？",
        "security_review passed but is hypothesized; restore_validation passed",
    ],
    ids=[
        "question-mark-first-only",
        "question-mark-second-only",
        "colon-zero-assignment",
        "partial-slash-count",
        "empty-slash-count",
        "comma-hypothetically",
        "comma-which-is-hypothetical",
        "zh-parenthetical-hypothesis",
        "zh-both-question",
        "but-hypothesized",
    ],
)
def test_F04_R5_adjacent_qualified_variants_must_not_verify(store, text):
    """Punctuation and trailing/attached qualifiers are still not a complete result."""
    event = _receive_tool(store, text, native_id="message:fix5:f04:adjacent")
    part = _only_part(store, event)
    _propose_task(store, part, _verified_value())
    _assert_downgraded_to_report(store)
    _assert_tool_observation_preserved(store, event)


# ---------------------------------------------------------------------------
# F04 round-6: the downgrade keeps reason + warnings + tool/quote evidence
# ---------------------------------------------------------------------------


def test_F04_R5_downgrade_reason_warnings_and_quote_span_are_auditable(store):
    """The qualified result is an auditable report; the tool evidence/span stays.

    A narrow ``support_quote`` span holds the interrogative clause; the
    downgrade must keep that exact backend-recorded span (start/end byte,
    body sha) and the real tool observation, and must be explained by the stored
    ``backend_downgraded`` reason plus the worker warning.
    """
    text = "security_review 通过了吗；restore_validation 通过"
    quote = "security_review 通过了吗"
    event = _receive_tool(store, text, native_id="message:fix5:f04:audit")
    part = _only_part(store, event)
    job = claim(store, worker_id="worker:fix5", kinds=("extract",), limit=1)[0]
    result = run_extraction_job(
        store,
        job,
        MockExtractor(_proposal(_task_claim(part, _verified_value(), quote=quote))),
    )
    assert result["claims_written"] == 1, result
    warnings = result.get("warnings") or []
    assert any("grounding" in warning for warning in warnings), result
    assert any("downgraded" in warning for warning in warnings), result

    view = _only_view(store)
    assert view["status"] == "reported_done", view
    assert view["verified_revision_id"] is None, view
    assert view["verification_conditions"] == [], view
    stored = _stored_task_value(store)
    assert stored["status"] == "reported_done", stored
    assert stored.get("backend_downgraded") == "conditions_not_grounded_in_evidence", stored
    # The declared acceptance set is preserved for audit; only the level dropped.
    assert stored.get("conditions") == list(ACCEPTANCE_CONDITIONS), stored

    item = _assert_tool_observation_preserved(store, event)
    links = _evidence_links(store)
    expected_start, expected_end = contracts.utf8_span(part.text, quote)
    assert item["sources"][0]["event_id"] == event["event_id"], item
    assert links[0]["start_byte"] == expected_start, links
    assert links[0]["end_byte"] == expected_end, links
    assert part.text.encode("utf-8")[expected_start:expected_end].decode("utf-8") == quote


# ---------------------------------------------------------------------------
# F04 round-6: genuine complete explicit results still verify (controls)
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
        "security_review passed 2/2 checks; restore_validation passed",
        "security_review 通过了；restore_validation 通过了",
        "security_review passed successfully; restore_validation passed",
        "security_review 通过检查；restore_validation 通过检查",
        # same sentence, comma-coordinated complete conclusions are still complete
        "security_review passed, restore_validation passed",
        "security_review passed, and restore_validation passed",
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
        "full-count-of",
        "full-count-slash",
        "zh-completion-particle",
        "trailing-successfully",
        "zh-artifact-noun",
        "comma-coordinated",
        "comma-and-coordinated",
    ],
)
def test_F04_R5_complete_explicit_success_still_verifies(store, text):
    """The complete-conclusion rule must not reject genuine recorded review results."""
    event = _receive_tool(store, text, native_id="message:fix5:f04:positive")
    part = _only_part(store, event)
    _propose_task(store, part, _verified_value())
    view = _only_view(store)
    assert view["status"] == "verified_done", view
    assert view["verified_revision_id"] is not None, view
    assert view["verification_conditions"] == list(ACCEPTANCE_CONDITIONS), view
    assert view["satisfied_conditions"] == list(ACCEPTANCE_CONDITIONS), view


def test_F04_R5_positive_projection_and_evidence_are_auditable(store):
    """A genuine complete result keeps its tool_execution source and evidence span."""
    text = "security_review passed 3 of 3 checks; restore_validation passed"
    event = _receive_tool(store, text, native_id="message:fix5:f04:positive-audit")
    part = _only_part(store, event)
    job = claim(store, worker_id="worker:fix5", kinds=("extract",), limit=1)[0]
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


# ---------------------------------------------------------------------------
# F04 round-6 hardening: a hypothesis stated in a *later sentence/clause* of the
# same quoted span (the round-5 span veto class completion; Codex's most likely
# adjacent counterexample)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "conditions"),
    [
        ("security_review passed. This is only hypothetical.", ("security_review",)),
        ("security_review 通过。这只是假设。", ("security_review",)),
        ("security_review passed; restore_validation passed is hypothetical.", ACCEPTANCE_CONDITIONS),
    ],
    ids=[
        "en-cross-sentence-hypothetical",
        "zh-cross-sentence-hypothesis",
        "en-cross-clause-hypothetical",
    ],
)
def test_F04_R5_cross_sentence_hypothesis_must_not_verify(store, text, conditions):
    """A hypothesis qualifier anywhere in the quoted span vetoes the whole item.

    The bound success conclusion may be a clean clause; the Round-5 span veto
    classes cover unknown/conditional/hearsay wording, and the hypothesis class
    completion also covers the qualifier when it is stated in the next sentence
    or a later clause of the same span. The declared condition set matches the
    span, so the downgrade can only come from the hypothesis qualifier, never
    from a condition that is merely absent from the span.
    """
    event = _receive_tool(store, text, native_id="message:fix5:f04:cross-sentence")
    part = _only_part(store, event)
    _propose_task(store, part, _verified_value(conditions))
    _assert_downgraded_to_report(store)
    _assert_tool_observation_preserved(store, event)


def test_F04_R5_ordinary_cross_sentence_description_now_reports(store):
    """Round-7 contract tightening: conclusion + ordinary prose is not a complete record.

    Before round-7 a later sentence without hypothesis/conditional wording was
    treated as neutral and the item still verified (the old test asserted
    ``verified_done``). The round-7 rule requires *every* sentence/clause of the
    referenced span to be a complete success conclusion of a declared acceptance
    condition; an ordinary description sentence such as ``The log upload
    succeeded.`` cannot be consumed, so the same real input now stays an
    auditable ``reported_done``. This is a deliberate contract tightening, not a
    weakened assertion: the tool observation, the exact quoted span, the body
    sha and the declared conditions are all still preserved for audit.
    """
    text = "security_review passed. The log upload succeeded."
    event = _receive_tool(store, text, native_id="message:fix5:f04:cross-sentence-positive")
    part = _only_part(store, event)
    _propose_task(store, part, _verified_value(("security_review",)))
    _assert_downgraded_to_report(store)
    _assert_tool_observation_preserved(store, event)
