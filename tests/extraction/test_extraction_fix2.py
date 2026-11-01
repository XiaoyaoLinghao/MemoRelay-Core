"""T20 round-3 (fix2) F04/F07 regression tests.

F04: a native tool/tool_execution observation whose stored body only says
``exit_code=0`` must not become a whole-task ``verified_done`` just because the
model proposed invented acceptance conditions. Every proposed condition must be
grounded in the referenced sanitized evidence text; ungrounded ones downgrade
the completion to ``reported_done`` while the successful tool observation stays.

F07: reconciliation of complete/active revisions that lack an extraction job
must page over the missing set only and advance a stable cursor, so later gaps
are never starved by earlier rows and excluded origins do not block the queue.

Counterexample source (independent round-2 review, rewritten with repo
fixtures):
``artifacts/T20-round2/run-20261030-codex/test_variants.py::
test_F04_tool_exit_zero_cannot_invent_all_acceptance`` and
``::test_F07_reconcile_limit_does_not_starve_later_missing_job``.
"""

from __future__ import annotations

import json

from knowledge_weaver import contracts
from knowledge_weaver.extraction import (
    load_event_group,
    reconcile_extraction_jobs,
    run_extraction_job,
    run_worker_once,
    sanitized_parts,
)
from knowledge_weaver.jobs import claim
from knowledge_weaver.store import current_items, list_task_views, receive_batch

from ..store.conftest import make_batch, make_event
from .test_extraction import MockExtractor, _claim, _proposal


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _receive(
    store,
    text,
    *,
    native_id="message:fix2:1",
    role="user",
    origin_kind="user_input",
    batch_id=None,
    **kwargs,
):
    event = make_event(native_id=native_id, text=text, role=role, origin_kind=origin_kind, **kwargs)
    result = receive_batch(store, make_batch([event], batch_id=batch_id or f"batch:{native_id}"))
    assert result["receipts"][0]["disposition"] == "accepted", result
    return event


def _receive_tool(store, text, *, native_id="message:fix2:tool:1", **kwargs):
    return _receive(store, text, native_id=native_id, role="tool", origin_kind="tool_execution", **kwargs)


def _only_part(store, event):
    parts = sanitized_parts(load_event_group(store, event["event_id"], 1))
    assert len(parts) == 1, parts
    return parts[0]


def _task_claim(part, value, *, assertion_kind="observation", facet="observed", display_text=None, evidence=None):
    claim_value = _claim(
        part,
        predicate="task.status",
        facet=facet,
        value=value,
        display_text=display_text or part.text,
        assertion_kind=assertion_kind,
    )
    if evidence is not None:
        claim_value["evidence"] = evidence
    return claim_value


def _task_views(store):
    return list_task_views(store, workspace_id="owner:demo")


def _only_view(store):
    views = _task_views(store)
    assert views, "expected a task view"
    return views[0]


def _task_items(store):
    return [item for item in current_items(store, workspace_id="owner:demo") if item["predicate"] == "task.status"]


def _task_status(store):
    views = _task_views(store)
    return views[0]["status"] if views else None


# ---------------------------------------------------------------------------
# F04: completion must be grounded in the stored sanitized evidence
# ---------------------------------------------------------------------------


def test_F04_tool_exit_zero_cannot_invent_all_acceptance(store):
    """Round-2 counterexample: exit_code=0 + invented conditions != verified_done."""
    event = _receive_tool(store, "exit_code=0", native_id="message:fix2:f04:exit0")
    part = _only_part(store, event)
    proposal = _proposal(
        _task_claim(
            part,
            {
                "status": "verified",
                "conditions": ["security_review", "restore_validation"],
                "satisfied": ["security_review", "restore_validation"],
            },
        )
    )
    result = run_worker_once(store, MockExtractor(proposal))
    assert result["state"] == "succeeded", result
    view = _only_view(store)
    assert view["status"] == "reported_done", view
    assert view["verified_revision_id"] is None
    # The successful tool observation is preserved: same claim, observed facet,
    # tool-verified evidence grade and the original source span.
    items = _task_items(store)
    assert len(items) == 1, items
    assert items[0]["facet"] == "observed"
    assert items[0]["evidence_grade"] == "tool_verified"
    assert items[0]["sources"] and items[0]["sources"][0]["event_id"] == event["event_id"]
    # The downgrade is explicit and auditable in the stored value.
    with store.read() as conn:
        raw = conn.execute("SELECT value_json FROM claim_revisions").fetchone()["value_json"]
    value = json.loads(raw)
    assert value["status"] == "reported_done", value
    assert value.get("backend_downgraded") == "conditions_not_grounded_in_evidence", value


def test_F04_ungrounded_conditions_are_warned_in_job_result(store):
    """The proposal-validation downgrade is reported as a warning, not silent."""
    event = _receive_tool(store, "exit_code=0", native_id="message:fix2:f04:warn")
    part = _only_part(store, event)
    proposal = _proposal(
        _task_claim(
            part,
            {"status": "verified", "conditions": ["security_review"], "satisfied": ["security_review"]},
        )
    )
    job = claim(store, worker_id="worker:fix2", kinds=("extract",), limit=1)[0]
    result = run_extraction_job(store, job, MockExtractor(proposal))
    assert result["claims_written"] == 1, result
    assert any("grounding" in warning for warning in result.get("warnings") or []), result


def test_F04_tool_body_can_ground_every_acceptance_condition(store):
    """Legitimate case: the tool body explicitly records every acceptance result."""
    event = _receive_tool(
        store,
        "security_review 通过；restore_validation 通过",
        native_id="message:fix2:f04:grounded",
    )
    part = _only_part(store, event)
    proposal = _proposal(
        _task_claim(
            part,
            {
                "status": "verified",
                "conditions": ["security_review", "restore_validation"],
                "satisfied": ["security_review", "restore_validation"],
            },
        )
    )
    result = run_worker_once(store, MockExtractor(proposal))
    assert result["state"] == "succeeded", result
    view = _only_view(store)
    assert view["status"] == "verified_done", view
    assert view["verified_revision_id"] is not None
    assert view["verification_conditions"] == ["security_review", "restore_validation"]
    assert view["satisfied_conditions"] == ["security_review", "restore_validation"]


def test_F04_conditions_grounded_across_evidence_parts_verify(store):
    """Every condition may be grounded by a different referenced sanitized part."""
    native_id = "message:fix2:f04:parts"
    first = make_event(
        native_id=native_id,
        text="security_review 通过",
        part_index=0,
        part_count=2,
        role="tool",
        origin_kind="tool_execution",
    )
    second = make_event(
        native_id=native_id,
        text="restore_validation 通过",
        part_index=1,
        part_count=2,
        role="tool",
        origin_kind="tool_execution",
    )
    received = receive_batch(store, make_batch([first, second], batch_id="batch:fix2:f04:parts"))
    assert [receipt["disposition"] for receipt in received["receipts"]] == ["accepted", "accepted"], received
    parts = sanitized_parts(load_event_group(store, first["event_id"], 1))
    assert len(parts) == 2, parts
    evidence = [
        {
            "event_id": part.event_id,
            "event_revision": part.event_revision,
            "part_index": part.part_index,
            "support_quote": part.text,
        }
        for part in parts
    ]
    proposal = _proposal(
        _task_claim(
            parts[0],
            {
                "status": "verified",
                "conditions": ["security_review", "restore_validation"],
                "satisfied": ["security_review", "restore_validation"],
            },
            display_text="任务验收完成",
            evidence=evidence,
        )
    )
    result = run_worker_once(store, MockExtractor(proposal))
    assert result["state"] == "succeeded", result
    view = _only_view(store)
    assert view["status"] == "verified_done", view
    assert view["verified_revision_id"] is not None


def test_F04_self_report_with_grounded_conditions_still_not_verified(store):
    """Self-report semantics are unchanged even when the words are present."""
    event = _receive(
        store,
        "security_review 通过；restore_validation 通过",
        native_id="message:fix2:f04:self",
        role="assistant",
        origin_kind="assistant_reply",
    )
    part = _only_part(store, event)
    proposal = _proposal(
        _task_claim(
            part,
            {
                "status": "verified",
                "conditions": ["security_review", "restore_validation"],
                "satisfied": ["security_review", "restore_validation"],
            },
            assertion_kind="completion_report",
        )
    )
    result = run_worker_once(store, MockExtractor(proposal))
    assert result["state"] == "succeeded", result
    view = _only_view(store)
    assert view["status"] == "reported_done", view
    assert view["verified_revision_id"] is None
    items = _task_items(store)
    assert items and items[0]["facet"] == "reported", items


def test_F04_retraction_still_degrades_grounded_verification(store):
    """Retraction of the evidence still revokes a previously grounded verification."""
    event = _receive_tool(
        store,
        "security_review 通过",
        native_id="message:fix2:f04:retract",
    )
    part = _only_part(store, event)
    proposal = _proposal(
        _task_claim(
            part,
            {"status": "verified", "conditions": ["security_review"], "satisfied": ["security_review"]},
        )
    )
    assert run_worker_once(store, MockExtractor(proposal))["state"] == "succeeded"
    assert _only_view(store)["status"] == "verified_done"
    retract = make_event(
        native_id="message:fix2:f04:retract",
        revision=2,
        operation="retract",
        parent_revision=1,
        text="",
        role="tool",
        origin_kind="tool_execution",
    )
    retract["body"]["sha256"] = contracts.EMPTY_BODY_SHA256
    received = receive_batch(store, make_batch([retract], batch_id="batch:fix2:f04:retract"))
    assert received["receipts"][0]["disposition"] == "accepted", received
    assert _task_status(store) != "verified_done"


# ---------------------------------------------------------------------------
# F07: reconcile pages over the missing-job set with a stable forward cursor
# ---------------------------------------------------------------------------


def _receive_events(store, events, *, batch_id):
    received = receive_batch(store, make_batch(events, batch_id=batch_id))
    assert all(receipt["disposition"] == "accepted" for receipt in received["receipts"]), received


def _extract_job_keys(store):
    with store.read() as conn:
        return sorted(
            row["input_key"]
            for row in conn.execute("SELECT input_key FROM jobs WHERE kind = 'extract'").fetchall()
        )


def _job_exists(store, input_key):
    with store.read() as conn:
        return (
            conn.execute(
                "SELECT 1 FROM jobs WHERE kind = 'extract' AND input_key = ? LIMIT 1",
                (input_key,),
            ).fetchone()
            is not None
        )


def _delete_extract_job(store, event, revision=1):
    input_key = f"{event['event_id']}@{int(revision)}"
    with store.transaction() as conn:
        conn.execute("DELETE FROM jobs WHERE kind = 'extract' AND input_key = ?", (input_key,))
    assert not _job_exists(store, input_key)
    return input_key


def _assert_honest_counts(result):
    assert result["scanned"] == result["queued"] + result["skipped"], result
    assert "cursor" in result and "wrapped" in result, result


def test_F07_reconcile_limit_does_not_starve_later_missing_job(store):
    """Round-2 counterexample: 3 events, last (sorted) job deleted, limit=2, 3 calls."""
    events = [make_event(native_id=f"message:fix2:f07:starve:{index}") for index in range(3)]
    _receive_events(store, events, batch_id="batch:fix2:f07:starve")
    target = max(events, key=lambda event: event["event_id"])
    target_key = _delete_extract_job(store, target)
    results = [reconcile_extraction_jobs(store, limit=2) for _ in range(3)]
    assert _job_exists(store, target_key), results
    assert any(result["queued"] == 1 for result in results), results
    for result in results:
        _assert_honest_counts(result)


def test_F07_many_existing_jobs_do_not_starve_later_missing_jobs(store):
    """The scan must page over the missing set, not over every complete row."""
    events = [make_event(native_id=f"message:fix2:f07:many:{index}") for index in range(30)]
    _receive_events(store, events, batch_id="batch:fix2:f07:many")
    ordered = sorted(events, key=lambda event: event["event_id"])
    missing_keys = [_delete_extract_job(store, event) for event in ordered[-2:]]
    before = len(_extract_job_keys(store))
    first = reconcile_extraction_jobs(store, limit=5)
    assert first["queued"] == 2, first
    _assert_honest_counts(first)
    keys_after = _extract_job_keys(store)
    assert len(keys_after) == before + 2, keys_after
    assert all(key in keys_after for key in missing_keys)


def test_F07_reconcile_cursor_covers_the_whole_gap_set(store):
    """Bounded pages keep progressing over the missing set and then wrap."""
    events = [make_event(native_id=f"message:fix2:f07:cursor:{index}") for index in range(8)]
    _receive_events(store, events, batch_id="batch:fix2:f07:cursor")
    ordered = sorted(events, key=lambda event: event["event_id"])
    gaps = [_delete_extract_job(store, event) for event in ordered[:5]]
    results = [reconcile_extraction_jobs(store, limit=2)]
    # A periodic scheduler opens its own handle on every run: the cursor must
    # persist in the store, not in process memory.
    reopened = type(store)(store.db_path)
    results.extend(reconcile_extraction_jobs(reopened, limit=2) for _ in range(3))
    for result in results:
        _assert_honest_counts(result)
    assert sum(result["queued"] for result in results) == 5, results
    assert results[0]["cursor"] is not None, results
    assert results[1]["cursor"] is not None, results
    # The fifth gap is the last row: the page is partial, so the cursor resets
    # for the next cycle and the result reports the wrap honestly.
    assert results[2]["cursor"] is None, results
    assert results[2]["wrapped"] is True, results
    assert results[3]["queued"] == 0, results
    keys_after = _extract_job_keys(store)
    assert len(keys_after) == 8, keys_after
    assert all(key in keys_after for key in gaps)
    # Repeated periodic runs stay idempotent and never duplicate work.
    for _ in range(2):
        again = reconcile_extraction_jobs(store, limit=2)
        _assert_honest_counts(again)
        assert again["queued"] == 0, again
    assert _extract_job_keys(store) == keys_after


def test_F07_excluded_origins_advance_the_cursor_instead_of_blocking(store):
    """Excluded maintenance/injection rows are skipped but must not clog the head."""
    eligible = make_event(native_id="message:fix2:f07:excluded:eligible", text="端口4200")
    result = receive_batch(store, make_batch([eligible], batch_id="batch:fix2:f07:excluded:eligible"))
    assert result["receipts"][0]["disposition"] == "accepted", result
    excluded = []
    before_eligible = []
    for index in range(40):
        event = make_event(
            native_id=f"message:fix2:f07:excluded:{index}",
            text=f"维护事件 {index}",
            role="system",
            origin_kind="maintenance",
        )
        received = receive_batch(
            store,
            make_batch([event], batch_id=f"batch:fix2:f07:excluded:{index}"),
        )
        # Excluded origins are durably received but deliberately not ingested;
        # their receipt disposition says so instead of pretending work will run.
        assert received["receipts"][0]["disposition"] in ("accepted", "excluded"), received
        # Legacy databases can contain excluded-origin revisions that are still
        # marked active; that is exactly the rows the reconciler must skip past.
        with store.transaction() as conn:
            conn.execute(
                "UPDATE event_revisions SET lifecycle = 'active' WHERE event_id = ?",
                (event["event_id"],),
            )
        excluded.append(event)
        before_eligible = [item for item in excluded if item["event_id"] < eligible["event_id"]]
        if len(before_eligible) >= 2:
            break
    assert len(before_eligible) >= 2, [item["event_id"] for item in excluded]
    target_key = _delete_extract_job(store, eligible)
    results = [reconcile_extraction_jobs(store, limit=2) for _ in range(30)]
    for result in results:
        _assert_honest_counts(result)
    assert _job_exists(store, target_key), results
    assert any(result["skipped"] > 0 for result in results), results
    # Excluded origins are never semantically ingested.
    for event in excluded:
        assert not _job_exists(store, f"{event['event_id']}@1"), event["event_id"]


def test_F07_reconcile_repeat_is_idempotent(store):
    """Repeated calls queue each missing revision exactly once."""
    events = [make_event(native_id=f"message:fix2:f07:idempotent:{index}") for index in range(4)]
    _receive_events(store, events, batch_id="batch:fix2:f07:idempotent")
    ordered = sorted(events, key=lambda event: event["event_id"])
    for event in ordered[:2]:
        _delete_extract_job(store, event)
    first = reconcile_extraction_jobs(store, limit=1)
    second = reconcile_extraction_jobs(store, limit=1)
    assert first["queued"] == 1, first
    assert second["queued"] == 1, second
    keys_after = _extract_job_keys(store)
    assert len(keys_after) == len(set(keys_after)) == 4, keys_after
    for _ in range(2):
        again = reconcile_extraction_jobs(store, limit=1)
        _assert_honest_counts(again)
        assert again["queued"] == 0, again
        assert again["skipped"] == 0, again
    assert _extract_job_keys(store) == keys_after
