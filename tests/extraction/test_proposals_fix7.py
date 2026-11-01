"""T20 round-7 (fix7) F04 regression tests: verified only from a complete result record.

Round-6 independent review kept F04 open
(``review/T20-CODEX-ROUND6.md`` §3 "F04-R6 / AC-09、AC-36", P1): the round-6
binding constrained the condition name and the success clause, but any material
*before* the condition name or *outside* the completed clause was unconstrained.
The following real ``receive_batch -> sanitized_parts -> run_worker_once /
run_extraction_job -> list_task_views`` inputs still projected ``verified_done``
with no ``backend_downgraded`` and no warning:

* ``Example output only: security_review passed; restore_validation passed``
* ``Fictional transcript: security_review passed; restore_validation passed``
* ``security_review passed; restore_validation passed. These statements are
  fictional examples.``
* ``示例输出：security_review 通过；restore_validation 通过。以上只是示例。``
* ``security_review passed; restore_validation passed. Both reviews remain
  unfinished.``
* ``security_review passed; restore_validation passed. 实际两项均不及格。``
* ``Did security_review pass; restore_validation passed``
* ``security_review passed 3 of 2 checks; restore_validation passed``

The round-7 fix is a structural completeness rule, not a longer negative word
list. The trusted fact is no longer "a success word appears inside the
referenced span" but "the referenced span itself is a complete review-result
record", and the trusted *scope* is the complete stored part record:

* the *whole* backend-recorded ``support_quote`` span must be consumable: every
  sentence and every comma-delimited clause must be a complete success
  conclusion of one of the declared acceptance conditions (optionally led by
  ``and``/``then``/``plus``);
* a full result count must satisfy ``n == d > 0`` (``3 of 3 checks``); a partial
  (``2/3``), over-counted (``3 of 2``) or empty (``0 of 0``) count is not a
  result;
* a question-terminated sentence (``?``/``？``) never grounds a result, and a
  prefix label / trailing scoping declaration / ordinary prose / unknown wording
  cannot be consumed, so the whole span fails closed;
* the negative result markers stay as a defence-in-depth veto;
* **scope rule**: every referenced ``support_quote`` must cover the complete
  stored sanitized part record (normalized equality with the part text). A
  narrow excerpt -- even a clean one that itself reads like a complete
  conclusion, e.g. quoting ``security_review passed`` out of ``Example output
  only: security_review passed`` -- cannot grant ``verified``; it stays
  ``reported_done`` with a ``scope incomplete`` warning and the exact cited span
  (start/end byte, body sha) is still recorded for audit. The veto scan still
  covers every referenced span; only *granting* is limited to complete-part
  references.

Contract (AC-09/AC-36):

* a condition is grounded only by a complete success conclusion for its own name
  inside a span that is itself entirely a review-result record;
* the cited evidence must be the complete stored part record, not an excerpt;
* ungrounded/unconsumable/narrow-scope results stay ``reported_done`` with the
  auditable ``backend_downgraded = "conditions_not_grounded_in_evidence"``
  marker and a worker warning, while the real tool observation (event, exact
  quoted span, body sha, declared conditions) is preserved;
* a plain completion self-report stays ``reported_done`` (never verified);
* genuine clean explicit records whose ``support_quote`` is the complete part
  (``security_review passed; restore_validation passed`` / ``已通过`` /
  ``passed 3 of 3 checks`` / comma-coordinated conclusions) still project as
  ``verified_done``.

Counterexample sources (read-only): Codex round-6 independent review
``review/T20-CODEX-ROUND6.md`` §3 F04-R6 and its §3 minimal inputs. Round-4..6
regression suites stay in ``test_proposals_fix3.py`` / ``test_proposals_fix4.py``
/ ``test_proposals_fix5.py``.

These tests pin the round-7 contract through the real product path only:
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


def _receive_any(store, event):
    received = receive_batch(store, make_batch([event], batch_id=f"batch:{event['body']['sha256'][:12]}"))
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


def _task_claim(part, value, *, quote=None, assertion_kind="observation"):
    """A model proposal for ``task.status``; ``quote`` selects the referenced span."""
    return _claim(
        part,
        quote=quote,
        predicate="task.status",
        facet="observed",
        value=value,
        display_text=part.text,
        assertion_kind=assertion_kind,
        evidence=[_evidence_ref(part, quote)],
    )


def _propose_task(store, part, value, *, quote=None):
    result = run_worker_once(store, MockExtractor(_proposal(_task_claim(part, value, quote=quote))))
    assert result["state"] == "succeeded", result
    return result


def _run_task_job(store, part, value, *, quote=None, assertion_kind="observation", worker_id="worker:fix7"):
    """Run the real extraction job (not just the worker helper) to collect warnings."""
    job = claim(store, worker_id=worker_id, kinds=("extract",), limit=1)[0]
    result = run_extraction_job(
        store,
        job,
        MockExtractor(_proposal(_task_claim(part, value, quote=quote, assertion_kind=assertion_kind))),
    )
    assert result["claims_written"] == 1, result
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
    """An unconsumable completion stays an auditable report, never verified."""
    view = _only_view(store)
    assert view["status"] == "reported_done", view
    assert view["status"] != "verified_done", view
    assert view["verified_revision_id"] is None, view
    assert view["verification_conditions"] == [], view
    value = _stored_task_value(store)
    assert value["status"] == "reported_done", value
    assert value.get("backend_downgraded") == "conditions_not_grounded_in_evidence", value
    return view


def _assert_verified(store, conditions=ACCEPTANCE_CONDITIONS):
    view = _only_view(store)
    assert view["status"] == "verified_done", view
    assert view["verified_revision_id"] is not None, view
    assert view["verification_conditions"] == list(conditions), view
    assert view["satisfied_conditions"] == list(conditions), view
    value = _stored_task_value(store)
    assert value["status"] == "verified", value
    assert "backend_downgraded" not in value, value
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
# F04 round-7: the eight blocking Codex round-6 real-entry inputs (verbatim)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        # review/T20-CODEX-ROUND6.md §3 F04-R6 minimal inputs, support_quote is
        # the complete original text (the real product path stores it that way).
        "Example output only: security_review passed; restore_validation passed",
        "Fictional transcript: security_review passed; restore_validation passed",
        "security_review passed; restore_validation passed. These statements are fictional examples.",
        "示例输出：security_review 通过；restore_validation 通过。以上只是示例。",
        "security_review passed; restore_validation passed. Both reviews remain unfinished.",
        "security_review passed; restore_validation passed. 实际两项均不及格。",
        "Did security_review pass; restore_validation passed",
        "security_review passed 3 of 2 checks; restore_validation passed",
    ],
    ids=[
        "example-output-prefix-label",
        "fictional-transcript-prefix-label",
        "post-fiction-declaration",
        "zh-example-output-prefix-label",
        "post-unfinished-declaration",
        "post-actual-failure-declaration",
        "interrogative-prefix",
        "invalid-count-three-of-two",
    ],
)
def test_F04_R6_codex_round6_blocking_inputs_must_not_verify(store, text):
    """Round-6 blocking inputs: only a whole complete record can ground verified."""
    event = _receive_tool(store, text, native_id="message:fix7:f04:round6")
    part = _only_part(store, event)
    _propose_task(store, part, _verified_value())
    _assert_downgraded_to_report(store)
    _assert_tool_observation_preserved(store, event)


# ---------------------------------------------------------------------------
# F04 round-7: a clean complete record still verifies (positive controls)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "security_review passed; restore_validation passed",
        "security_review PASS; restore_validation PASS",
        "security_review succeeded; restore_validation succeeded",
        "security_review 通过；restore_validation 通过",
        "security_review 已通过；restore_validation 已通过",
        "security_review passed 3 of 3 checks; restore_validation passed",
        "security_review passed 2/2 checks; restore_validation passed",
        "security_review passed, restore_validation passed",
        "security_review passed, and restore_validation passed",
        "security_review: passed; restore_validation: passed",
        "security_review has passed; restore_validation has passed",
        "security_review 通过检查；restore_validation 通过检查",
        "security_review 通过、restore_validation 通过",
    ],
    ids=[
        "en-semicolon",
        "en-upper-pass",
        "en-succeeded",
        "zh-passed",
        "zh-yi-passed",
        "full-count-of",
        "full-count-slash",
        "comma-coordinated",
        "comma-and-coordinated",
        "colon-pass",
        "has-passed",
        "zh-artifact-noun",
        "zh-enumeration-comma",
    ],
)
def test_F04_R7_clean_complete_record_still_verifies(store, text):
    """The whole-record rule must not reject genuine clean explicit results."""
    event = _receive_tool(store, text, native_id="message:fix7:f04:positive")
    part = _only_part(store, event)
    _propose_task(store, part, _verified_value())
    _assert_verified(store)
    _assert_tool_observation_preserved(store, event)


def test_F04_R7_pure_single_conclusion_record_still_verifies_without_warning(store):
    """A single pure conclusion record is a complete record: verified, no warning."""
    text = "security_review passed 3 of 3 checks"
    event = _receive_tool(store, text, native_id="message:fix7:f04:positive-pure")
    part = _only_part(store, event)
    result = _run_task_job(store, part, _verified_value(("security_review",)))
    assert result.get("warnings") == [], result
    _assert_verified(store, ("security_review",))
    item = _task_item(store)
    assert item["facet"] == "observed", item
    assert item["evidence_grade"] == "tool_verified", item
    links = _evidence_links(store)
    assert len(links) == 1, links
    assert links[0]["body_sha256"] == event["body"]["sha256"], links
    assert links[0]["start_byte"] == 0, links
    assert links[0]["end_byte"] == len(text.encode("utf-8")), links


# ---------------------------------------------------------------------------
# F04 round-7: ordinary completion self-report stays reported, never verified
# ---------------------------------------------------------------------------


def test_F04_R7_assistant_completion_report_stays_reported(store):
    """An ordinary assistant completion report is a report, not a verified result."""
    event = make_event(
        native_id="message:fix7:f04:self-report",
        text="security_review passed; restore_validation passed",
        role="assistant",
        origin_kind="assistant_reply",
    )
    _receive_any(store, event)
    part = _only_part(store, event)
    job = claim(store, worker_id="worker:fix7", kinds=("extract",), limit=1)[0]
    result = run_extraction_job(
        store,
        job,
        MockExtractor(_proposal(_task_claim(part, _verified_value(), assertion_kind="completion_report"))),
    )
    assert result["claims_written"] == 1, result
    warnings = result.get("warnings") or []
    assert any("downgraded" in warning for warning in warnings), result

    view = _only_view(store)
    assert view["status"] == "reported_done", view
    assert view["verified_revision_id"] is None, view
    assert view["verification_conditions"] == [], view
    stored = _stored_task_value(store)
    assert stored["status"] == "reported_done", stored
    assert stored.get("backend_downgraded") == "verified_requires_verified_evidence", stored
    item = _task_item(store)
    assert item["facet"] == "reported", item
    assert item["evidence_grade"] == "reported", item
    assert item["sources"][0]["event_id"] == event["event_id"], item


def test_F04_R7_completion_report_assertion_stays_reported_even_on_tool_evidence(store):
    """A completion_report assertion on real tool text still cannot grant verified."""
    text = "security_review passed; restore_validation passed"
    event = _receive_tool(store, text, native_id="message:fix7:f04:completion-report-tool")
    part = _only_part(store, event)
    result = _run_task_job(store, part, _verified_value(), assertion_kind="completion_report")
    warnings = result.get("warnings") or []
    assert any("completion report" in warning for warning in warnings), result

    view = _only_view(store)
    assert view["status"] == "reported_done", view
    assert view["verified_revision_id"] is None, view
    stored = _stored_task_value(store)
    assert stored["status"] == "reported_done", stored
    assert stored.get("backend_downgraded") == "completion_report_cannot_verify", stored
    _assert_tool_observation_preserved(store, event)


def test_F04_R7_plain_user_completion_self_description_stays_reported(store):
    """A plain "I finished the review" user description is not a review result record."""
    event = make_event(
        native_id="message:fix7:f04:user-self-description",
        text="我已经完成安全复核，两项验收都通过了。",
        role="user",
        origin_kind="user_input",
    )
    _receive_any(store, event)
    part = _only_part(store, event)
    _propose_task(store, part, _verified_value())
    _assert_downgraded_to_report(store)
    item = _task_item(store)
    assert item["sources"][0]["event_id"] == event["event_id"], item


# ---------------------------------------------------------------------------
# F04 round-7: explicit failure / not-run results stay reported (controls)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "security_review failed; restore_validation passed",
        "security_review FAILED; restore_validation FAILED",
        "security_review NOT RUN; restore_validation passed",
        "security_review 失败；restore_validation 通过",
        "security_review 未执行；restore_validation 通过",
        "security_review not reviewed; restore_validation passed",
    ],
    ids=[
        "en-failed",
        "both-failed",
        "not-run",
        "zh-failed",
        "zh-not-executed",
        "not-reviewed",
    ],
)
def test_F04_R7_explicit_failure_still_downgrades(store, text):
    """Explicit failure/not-executed records are never a complete success record."""
    event = _receive_tool(store, text, native_id="message:fix7:f04:failure")
    part = _only_part(store, event)
    _propose_task(store, part, _verified_value())
    _assert_downgraded_to_report(store)
    _assert_tool_observation_preserved(store, event)


# ---------------------------------------------------------------------------
# F04 round-7: unknown status / question forms
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "security_review unknown; restore_validation passed",
        "security_review: unknown; restore_validation passed",
        "security_review status: TBD; restore_validation passed",
        "Can we say security_review passed; restore_validation passed",
        "security_review passed? restore_validation passed",
        "security_review passed; restore_validation passed？",
        "security_review 是否通过；restore_validation 通过",
        "security_review passed, but it is unclear; restore_validation passed",
    ],
    ids=[
        "unknown",
        "colon-unknown",
        "tbd",
        "interrogative-opener-can-we-say",
        "question-terminal-first",
        "question-terminal-second",
        "zh-whether-passed",
        "comma-unclear",
    ],
)
def test_F04_R7_unknown_or_question_forms_must_not_verify(store, text):
    """Unknown status and interrogative forms are not completed asserted results."""
    event = _receive_tool(store, text, native_id="message:fix7:f04:unknown-question")
    part = _only_part(store, event)
    _propose_task(store, part, _verified_value())
    _assert_downgraded_to_report(store)
    _assert_tool_observation_preserved(store, event)


# ---------------------------------------------------------------------------
# F04 round-7: invalid / partial result counts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "security_review passed 3 of 2 checks; restore_validation passed",
        "security_review passed 2/3 checks; restore_validation passed",
        "security_review passed 0 of 0 checks; restore_validation passed",
        "security_review passed 1 of 3 checks; restore_validation passed",
        "security_review passed 2 of 3 checks; restore_validation passed",
    ],
    ids=[
        "over-count-three-of-two",
        "partial-slash-two-of-three",
        "empty-count-zero-of-zero",
        "partial-of-one-of-three",
        "partial-of-two-of-three",
    ],
)
def test_F04_R7_invalid_result_counts_must_not_verify(store, text):
    """Only a full non-empty count (n == d > 0) is a completed result."""
    event = _receive_tool(store, text, native_id="message:fix7:f04:count")
    part = _only_part(store, event)
    _propose_task(store, part, _verified_value())
    _assert_downgraded_to_report(store)
    _assert_tool_observation_preserved(store, event)


# ---------------------------------------------------------------------------
# F04 round-7: prefix labels, trailing declarations and ordinary cross-sentence
# prose cannot be consumed by the complete-record grammar
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "security_review passed. The log upload succeeded.",
        "security_review passed; restore_validation passed. Example only.",
        "security_review passed; restore_validation passed. Please confirm.",
        "security_review passed; restore_validation passed; see attached log.",
        "security_review passed, but the review is still open; restore_validation passed",
        "security_review passed; restore_validation passed. Note: fictional.",
        "安全复核通过；恢复验证通过。以上为示例，请勿作为验收依据。",
        "security_review 通过。以上只是示例。",
    ],
    ids=[
        "cross-sentence-ordinary-prose",
        "post-example-only",
        "post-please-confirm",
        "trailing-see-attached-log",
        "comma-but-open",
        "post-note-fictional",
        "zh-post-example-note",
        "zh-post-example-only",
    ],
)
def test_F04_R7_unconsumable_span_material_must_not_verify(store, text):
    """Any material that is not a declared-condition conclusion fails the whole span."""
    event = _receive_tool(store, text, native_id="message:fix7:f04:unconsumable")
    part = _only_part(store, event)
    conditions = ACCEPTANCE_CONDITIONS if "security_review" in text else ("安全复核", "恢复验证")
    _propose_task(store, part, _verified_value(conditions))
    _assert_downgraded_to_report(store)
    _assert_tool_observation_preserved(store, event)


# ---------------------------------------------------------------------------
# F04 round-7 scope rule: every cited support_quote must cover the complete
# stored sanitized part record; a narrow excerpt cannot grant "verified"
# ---------------------------------------------------------------------------


def test_F04_R7_narrow_quote_cannot_verify(store):
    """A clean excerpt of a stored part cannot verify: the evidence scope is incomplete.

    The backend still records the cited narrow span (start/end byte, body sha)
    and the tool observation for audit, but *granting* verification requires
    every ``support_quote`` to cover the complete stored part record, so the
    claim stays ``reported_done`` with the auditable downgrade marker and a
    ``scope incomplete`` warning.
    """
    text = "tool: compress log. security_review passed; restore_validation passed"
    quote = "security_review passed; restore_validation passed"
    event = _receive_tool(store, text, native_id="message:fix7:f04:scope-narrow")
    part = _only_part(store, event)
    result = _run_task_job(store, part, _verified_value(), quote=quote)
    warnings = result.get("warnings") or []
    assert any("scope incomplete" in warning for warning in warnings), result

    _assert_downgraded_to_report(store)
    stored = _stored_task_value(store)
    assert stored.get("conditions") == list(ACCEPTANCE_CONDITIONS), stored

    item = _assert_tool_observation_preserved(store, event)
    links = _evidence_links(store)
    expected_start, expected_end = contracts.utf8_span(part.text, quote)
    assert item["sources"][0]["event_id"] == event["event_id"], item
    assert links[0]["start_byte"] == expected_start, links
    assert links[0]["end_byte"] == expected_end, links
    assert (links[0]["start_byte"], links[0]["end_byte"]) != (
        0,
        len(part.text.encode("utf-8")),
    ), links
    assert links[0]["body_sha256"] == event["body"]["sha256"], links


@pytest.mark.parametrize(
    ("text", "quote"),
    [
        (
            "Example output only: security_review passed; restore_validation passed",
            "security_review passed; restore_validation passed",
        ),
        (
            "示例输出：security_review 通过；restore_validation 通过。以上只是示例。",
            "security_review 通过；restore_validation 通过。",
        ),
    ],
    ids=["en-example-prefix-clean-excerpt", "zh-example-prefix-clean-excerpt"],
)
def test_F04_R7_narrow_quote_from_example_label_cannot_verify(store, text, quote):
    """A clean excerpt of an example-labeled record cannot verify (scope rule).

    The excerpt itself reads like a complete successful record, so this pins the
    scope rule rather than the whole-span completeness rule: the stored part is
    an explicit example, and the model may not quote only the clean conclusion
    out of it.
    """
    event = _receive_tool(store, text, native_id="message:fix7:f04:scope-example")
    part = _only_part(store, event)
    result = _run_task_job(store, part, _verified_value(), quote=quote)
    warnings = result.get("warnings") or []
    assert any("scope incomplete" in warning for warning in warnings), result
    _assert_downgraded_to_report(store)
    item = _assert_tool_observation_preserved(store, event)
    links = _evidence_links(store)
    expected_start, expected_end = contracts.utf8_span(part.text, quote)
    assert item["sources"][0]["event_id"] == event["event_id"], item
    assert links[0]["start_byte"] == expected_start, links
    assert links[0]["end_byte"] == expected_end, links


def test_F04_R7_narrow_quote_from_clean_text_cannot_verify(store):
    """Even from an otherwise clean record, only the complete part text can verify."""
    text = "security_review passed; restore_validation passed; exit_code=0"
    quote = "security_review passed; restore_validation passed"
    event = _receive_tool(store, text, native_id="message:fix7:f04:scope-clean-excerpt")
    part = _only_part(store, event)
    result = _run_task_job(store, part, _verified_value(), quote=quote)
    warnings = result.get("warnings") or []
    assert any("scope incomplete" in warning for warning in warnings), result
    _assert_downgraded_to_report(store)


def test_F04_R7_complete_part_quote_still_verifies(store):
    """A support_quote equal to the complete stored part is still a full record."""
    text = "security_review passed; restore_validation passed"
    event = _receive_tool(store, text, native_id="message:fix7:f04:scope-complete")
    part = _only_part(store, event)
    result = _run_task_job(store, part, _verified_value(), quote=text)
    assert result.get("warnings") == [], result
    _assert_verified(store)
    links = _evidence_links(store)
    assert links[0]["start_byte"] == 0, links
    assert links[0]["end_byte"] == len(text.encode("utf-8")), links


def test_F04_R7_any_narrow_reference_downgrades_the_whole_claim(store):
    """One complete reference plus one narrow reference is still scope-incomplete."""
    text = "security_review passed; restore_validation passed"
    full_quote = text
    narrow_quote = "security_review passed"
    event = _receive_tool(store, text, native_id="message:fix7:f04:scope-mixed-refs")
    part = _only_part(store, event)
    multi_evidence_claim = _claim(
        part,
        quote=full_quote,
        predicate="task.status",
        facet="observed",
        value=_verified_value(),
        display_text=part.text,
        assertion_kind="observation",
        evidence=[_evidence_ref(part, full_quote), _evidence_ref(part, narrow_quote)],
    )
    job = claim(store, worker_id="worker:fix7", kinds=("extract",), limit=1)[0]
    result = run_extraction_job(store, job, MockExtractor(_proposal(multi_evidence_claim)))
    assert result["claims_written"] == 1, result
    warnings = result.get("warnings") or []
    assert any("scope incomplete" in warning for warning in warnings), result

    _assert_downgraded_to_report(store)
    _assert_tool_observation_preserved(store, event, expected_links=2)


# ---------------------------------------------------------------------------
# F04 round-7: the downgrade keeps reason + warnings + tool/quote evidence +
# declared conditions for audit
# ---------------------------------------------------------------------------


def test_F04_R7_downgrade_reason_warnings_quotes_and_conditions_are_auditable(store):
    """A fictional post-declaration keeps every audit fact but loses verified level."""
    text = (
        "security_review passed; restore_validation passed. "
        "These statements are fictional examples."
    )
    quote = text
    event = _receive_tool(store, text, native_id="message:fix7:f04:audit")
    part = _only_part(store, event)
    result = _run_task_job(store, part, _verified_value(), quote=quote)
    warnings = result.get("warnings") or []
    assert any("grounding" in warning for warning in warnings), result
    assert any("downgraded" in warning for warning in warnings), result

    view = _only_view(store)
    assert view["status"] == "reported_done", view
    assert view["status"] != "verified_done", view
    assert view["verified_revision_id"] is None, view
    assert view["verification_conditions"] == [], view
    stored = _stored_task_value(store)
    assert stored["status"] == "reported_done", stored
    assert stored.get("backend_downgraded") == "conditions_not_grounded_in_evidence", stored
    # The declared acceptance set is preserved for audit; only the level dropped.
    assert stored.get("conditions") == list(ACCEPTANCE_CONDITIONS), stored
    assert stored.get("satisfied") == list(ACCEPTANCE_CONDITIONS), stored

    item = _assert_tool_observation_preserved(store, event)
    links = _evidence_links(store)
    expected_start, expected_end = contracts.utf8_span(part.text, quote)
    assert item["sources"][0]["event_id"] == event["event_id"], item
    assert links[0]["start_byte"] == expected_start, links
    assert links[0]["end_byte"] == expected_end, links
    assert links[0]["body_sha256"] == event["body"]["sha256"], links
    assert part.text.encode("utf-8")[expected_start:expected_end].decode("utf-8") == quote


def test_F04_R7_prefix_label_inside_quoted_span_is_audited(store):
    """A prefix label inside the quoted span downgrades and keeps the exact span."""
    text = "Example output only: security_review passed"
    quote = text
    event = _receive_tool(store, text, native_id="message:fix7:f04:audit-prefix")
    part = _only_part(store, event)
    result = _run_task_job(store, part, _verified_value(("security_review",)), quote=quote)
    warnings = result.get("warnings") or []
    assert any("grounding" in warning for warning in warnings), result

    _assert_downgraded_to_report(store)
    stored = _stored_task_value(store)
    assert stored.get("conditions") == ["security_review"], stored

    item = _assert_tool_observation_preserved(store, event)
    links = _evidence_links(store)
    expected_start, expected_end = contracts.utf8_span(part.text, quote)
    assert item["sources"][0]["event_id"] == event["event_id"], item
    assert links[0]["start_byte"] == expected_start, links
    assert links[0]["end_byte"] == expected_end, links
