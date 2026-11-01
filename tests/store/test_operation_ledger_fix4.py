"""AC52-R4/R5: failure recovery follows actual completion causality (F16).

Codex round-4 counterexample (AC52-R4): two threads run the *real*
``export_snapshot``.  The first export (A) starts first and is held in
``_write_jsonl``; the second export (B) starts later and finishes successfully;
only after B completed does A fail with ``ExportRefused('REDACTION_REQUIRED')``.
The old status query treated a same-operation failure as recovered when a
succeeded ledger row had a *later rowid* — but rowid is start-registration
order, so A's just-recorded failure was hidden (failed=0, state=idle,
last_error_code=None) by a success that had completed *before* the failure even
happened.

Fix (round 5): recovery is an explicit, durable association written inside the
transaction that completes a successful run.  That transaction stamps every
same-operation operation-ledger ``failed`` row that is *already recorded* at
that moment with ``recovered_by_job_id`` / ``recovered_at``.  Consequences this
module pins down with real entry points:

* a success that finished before a failure was recorded never recovers it
  (the round-4 counterexample, two real concurrent exports);
* a failure that finished before a success — even if that success *started*
  first — is recovered by the success, because recovery follows completion
  causality ("run in flight when the failure is recorded" is not yet a
  recovery; it counts only once it completes);
* a later success recovers earlier recorded failures of the same operation, and
  a single success recovers several old failures of that operation only;
* another operation's success (F16) and queue-backed failures are never
  auto-recovered;
* the association lives on the ledger row, so it survives a process restart and
  travels through snapshot export/import;
* opening a round-4 database (without the recovery columns) migrates
  idempotently and never resurrects or hides a failure.

Only synthetic ``tmp_path`` data is used; no network, no real host data.
"""

from __future__ import annotations

import json
import sqlite3
import threading

import pytest

import knowledge_weaver.lifecycle.retention as retention
import knowledge_weaver.portability.export as exporting
import knowledge_weaver.store.claims as claims_module
from knowledge_weaver import contracts
from knowledge_weaver.jobs import claim, commit, enqueue, fail
from knowledge_weaver.lifecycle import register_lifecycle, sweep
from knowledge_weaver.portability import export_snapshot, import_snapshot
from knowledge_weaver.portability.export import ExportRefused
from knowledge_weaver.store import Store, rebuild_projection, status as status_view
from knowledge_weaver.store.status import (
    OPERATION_LEDGER_VERSION,
    record_operation_finish,
    record_operation_start,
)

from ..state.conftest import record
from ..store.conftest import make_batch, make_event

DUE_NOW = "2026-10-31T00:00:00Z"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pipelines(store) -> dict:
    return {entry["operation"]: entry for entry in status_view(store)["pipelines"]}


def _assert_status_contract(store) -> None:
    view = status_view(store)
    assert contracts.validate_instance("Status", view) == [], view
    for pipeline in view["pipelines"]:
        assert set(pipeline) == {
            "operation",
            "state",
            "pending",
            "failed",
            "oldest_pending_at",
            "last_success_at",
            "last_error_code",
        }, pipeline


def _ledger_rows(store, kind: str) -> list[dict]:
    with store.read() as conn:
        return [
            dict(row)
            for row in conn.execute(
                "SELECT rowid, job_id, kind, state, model_version, last_error_code, "
                "recovered_by_job_id, recovered_at FROM jobs WHERE kind = ? "
                "ORDER BY created_at, rowid",
                (kind,),
            )
        ]


def _operation_rows(store, kind: str) -> list[dict]:
    return [
        row for row in _ledger_rows(store, kind) if row["model_version"] == OPERATION_LEDGER_VERSION
    ]


def _due_event_part(store):
    """A real event part whose completed-task lifecycle anchor is due (cleanup)."""
    event = make_event(native_id="message:ac52-fix4", text="已完成任务的过程详情")
    from knowledge_weaver.store import receive_batch

    receive_batch(store, make_batch([event]))
    register_lifecycle(
        store,
        subject_kind="event_part",
        subject_id=f"{event['event_id']}#1@0",
        class_="completed_task",
        business_anchor="2020-01-01T00:00:00Z",
    )
    return event


def _refuse_export(table, rows):
    raise ExportRefused("REDACTION_REQUIRED", "synthetic export refusal")


def _raise_synthetic(*args, **kwargs):
    raise RuntimeError("synthetic operation failure")


# ---------------------------------------------------------------------------
# The Codex round-4 counterexample: a success that completed first must not
# hide a failure that completes later
# ---------------------------------------------------------------------------


def test_AC52_real_concurrent_export_success_that_completed_first_cannot_hide_later_failure(
    store, tmp_path, monkeypatch
):
    """AC52-R4 exact counterexample: rowid order != completion order.

    A (thread, starts first, smaller rowid) is held inside ``_write_jsonl`` until
    B (main thread, real export) has completely finished (success committed).
    Only then does A raise ``ExportRefused``.  The just-recorded failure must stay
    visible; B's earlier-completed success must not be recorded as its recovery.
    """
    record(store, display_text="端口4200", value=4200)
    late_failure_pending = threading.Event()
    early_success_finished = threading.Event()
    errors: list[BaseException] = []
    original = exporting._write_jsonl

    def controlled(path, rows):
        if threading.current_thread().name == "ac52-late-failure":
            late_failure_pending.set()
            assert early_success_finished.wait(15), "the early success never finished"
            raise ExportRefused(
                "REDACTION_REQUIRED",
                "synthetic refused after the concurrent export already succeeded",
            )
        return original(path, rows)

    monkeypatch.setattr(exporting, "_write_jsonl", controlled)

    def late_failure():
        try:
            export_snapshot(store, tmp_path / "late-failed-export")
        except BaseException as exc:  # noqa: BLE001 - reported below
            errors.append(exc)

    thread = threading.Thread(target=late_failure, name="ac52-late-failure")
    thread.start()
    assert late_failure_pending.wait(15), "the late-failing export never started"
    try:
        export_snapshot(store, tmp_path / "early-succeeded-export")
    finally:
        early_success_finished.set()
    thread.join(15)
    assert not thread.is_alive()
    assert len(errors) == 1 and isinstance(errors[0], ExportRefused), errors

    pipeline = _pipelines(store)["export"]
    assert pipeline["failed"] == 1, pipeline
    assert pipeline["state"] == "degraded", pipeline
    assert pipeline["last_error_code"] == "REDACTION_REQUIRED", pipeline
    assert pipeline["pending"] == 0, pipeline
    # The succeeded export already completed before the failure, so it is a real
    # success (visible as last_success_at) but not a recovery.
    assert pipeline["last_success_at"] is not None, pipeline
    _assert_status_contract(store)

    rows = _operation_rows(store, "export")
    failed = [row for row in rows if row["state"] == "failed"]
    succeeded = [row for row in rows if row["state"] == "succeeded"]
    assert len(failed) == 1 and len(succeeded) == 1, rows
    # The counterexample is exactly the case where the success has the larger
    # rowid (started later) but completed earlier.
    assert failed[0]["rowid"] < succeeded[0]["rowid"], rows
    assert failed[0]["last_error_code"] == "REDACTION_REQUIRED", failed[0]
    assert failed[0]["recovered_by_job_id"] is None, failed[0]
    assert failed[0]["recovered_at"] is None, failed[0]
    assert succeeded[0]["recovered_by_job_id"] is None, succeeded[0]

    # A fresh session on the same database keeps the unrecovered failure.
    restarted = _pipelines(Store(store.db_path))["export"]
    assert restarted["failed"] == 1, restarted
    assert restarted["state"] == "degraded", restarted
    assert restarted["last_error_code"] == "REDACTION_REQUIRED", restarted

    # A genuinely later real export recovers the recorded failure.
    monkeypatch.undo()
    export_snapshot(store, tmp_path / "recovering-export")
    recovered = _pipelines(store)["export"]
    assert recovered["failed"] == 0 and recovered["state"] == "idle", recovered
    assert recovered["last_error_code"] is None and recovered["last_success_at"], recovered
    rows = _operation_rows(store, "export")
    failure_rows = [row for row in rows if row["state"] == "failed"]
    assert len(failure_rows) == 1, rows
    assert failure_rows[0]["recovered_by_job_id"] is not None, failure_rows
    assert failure_rows[0]["recovered_at"] is not None, failure_rows


# ---------------------------------------------------------------------------
# Completion-causal transitions: started-first success, failure recorded
# first, success completes last
# ---------------------------------------------------------------------------


def test_AC52_success_started_first_but_completed_after_a_recorded_failure_recovers_it(
    store, tmp_path, monkeypatch
):
    """Recovery follows completion order: while the success is still in flight
    it is not a recovery; when it completes after the failure was recorded it is.

    The success run starts first (smaller rowid) and is blocked in
    ``_write_jsonl``; a later-started export fails; only then is the in-flight
    success released.  The failure is recorded before the success completes, so
    the completing success recovers it — the mirror image of the counterexample.
    """
    record(store, display_text="端口4200", value=4200)
    success_started = threading.Event()
    failure_recorded = threading.Event()
    errors: list[BaseException] = []
    original = exporting._write_jsonl

    def controlled(path, rows):
        if threading.current_thread().name == "ac52-early-start-success":
            success_started.set()
            assert failure_recorded.wait(15), "the failure was never recorded"
            return original(path, rows)
        raise ExportRefused("REDACTION_REQUIRED", "synthetic early failure")

    monkeypatch.setattr(exporting, "_write_jsonl", controlled)

    def early_start_success():
        try:
            export_snapshot(store, tmp_path / "started-first-succeeded-last")
        except BaseException as exc:  # noqa: BLE001 - reported below
            errors.append(exc)

    thread = threading.Thread(target=early_start_success, name="ac52-early-start-success")
    thread.start()
    assert success_started.wait(15), "the early-started success never started"

    with pytest.raises(ExportRefused):
        export_snapshot(store, tmp_path / "started-later-failed-first")

    # The failure is durably recorded while the success is still running.  A
    # pending/in-flight success is not a recovery yet: pending=1, failed=1,
    # degraded, and the visible error is kept.
    in_flight = _pipelines(store)["export"]
    assert in_flight["failed"] == 1, in_flight
    assert in_flight["pending"] == 1, in_flight
    assert in_flight["state"] == "degraded", in_flight
    assert in_flight["last_error_code"] == "REDACTION_REQUIRED", in_flight
    assert in_flight["last_success_at"] is None, in_flight
    rows = _operation_rows(store, "export")
    failed_before = [row for row in rows if row["state"] == "failed"]
    assert len(failed_before) == 1 and failed_before[0]["recovered_by_job_id"] is None, rows

    failure_recorded.set()
    thread.join(15)
    assert not thread.is_alive(), "the success export did not finish"
    assert errors == [], errors

    recovered = _pipelines(store)["export"]
    assert recovered["failed"] == 0 and recovered["pending"] == 0, recovered
    assert recovered["state"] == "idle", recovered
    assert recovered["last_error_code"] is None, recovered
    assert recovered["last_success_at"] is not None, recovered
    rows = _operation_rows(store, "export")
    failed_rows = [row for row in rows if row["state"] == "failed"]
    succeeded_rows = [row for row in rows if row["state"] == "succeeded"]
    assert len(failed_rows) == 1 and len(succeeded_rows) == 1, rows
    # The success started first (smaller rowid) but completed last; the explicit
    # association was written by its completion transaction.
    assert succeeded_rows[0]["rowid"] < failed_rows[0]["rowid"], rows
    assert failed_rows[0]["recovered_by_job_id"] == succeeded_rows[0]["job_id"], rows
    assert failed_rows[0]["recovered_at"] is not None, failed_rows

    _assert_status_contract(store)
    restarted = _pipelines(Store(store.db_path))["export"]
    assert restarted["failed"] == 0 and restarted["state"] == "idle", restarted
    assert restarted["last_error_code"] is None, restarted


def test_AC52_later_success_recovers_recorded_failure_and_the_association_is_durable(
    store, tmp_path, monkeypatch
):
    """A later successful run of the same operation recovers the earlier failure;
    the explicit association persists across a restart (existing fix3 semantics
    now backed by the durable connection, not a rowid heuristic)."""
    record(store, display_text="端口4200", value=4200)
    monkeypatch.setattr(exporting, "_assert_rows_secret_free", _refuse_export)
    with pytest.raises(ExportRefused):
        export_snapshot(store, tmp_path / "failed-export")
    failed = _pipelines(store)["export"]
    assert failed["failed"] == 1 and failed["state"] == "degraded", failed
    assert failed["last_error_code"] == "REDACTION_REQUIRED", failed
    assert failed["last_success_at"] is None, failed

    monkeypatch.undo()
    export_snapshot(store, tmp_path / "recovering-export")
    recovered = _pipelines(store)["export"]
    assert recovered["failed"] == 0 and recovered["state"] == "idle", recovered
    assert recovered["last_error_code"] is None, recovered
    assert recovered["last_success_at"] is not None, recovered

    rows = _operation_rows(store, "export")
    failed_rows = [row for row in rows if row["state"] == "failed"]
    succeeded_rows = [row for row in rows if row["state"] == "succeeded"]
    assert len(failed_rows) == 1 and len(succeeded_rows) == 1, rows
    assert failed_rows[0]["recovered_by_job_id"] == succeeded_rows[0]["job_id"], rows
    assert failed_rows[0]["recovered_at"] is not None, failed_rows
    assert failed_rows[0]["last_error_code"] == "REDACTION_REQUIRED", failed_rows

    # New session: the durable association is what the status view reads.
    restarted = Store(store.db_path)
    durable = _pipelines(restarted)["export"]
    assert durable["failed"] == 0 and durable["state"] == "idle", durable
    assert durable["last_error_code"] is None, durable
    rows_after_restart = _operation_rows(restarted, "export")
    recovered_row = next(row for row in rows_after_restart if row["state"] == "failed")
    assert recovered_row["recovered_by_job_id"] == succeeded_rows[0]["job_id"], rows_after_restart


# ---------------------------------------------------------------------------
# One success recovers several old failures, but only of its own operation
# ---------------------------------------------------------------------------


def test_AC52_one_success_recovers_multiple_old_failures_of_its_operation_only(
    store, tmp_path, monkeypatch
):
    """A single later success recovers two recorded export failures; the cleanup
    failure (F16, another operation) and a queue-backed export unit stay visible."""
    _due_event_part(store)
    record(store, display_text="端口4200", value=4200)

    # Failure 1: real export refusal.
    monkeypatch.setattr(exporting, "_assert_rows_secret_free", _refuse_export)
    with pytest.raises(ExportRefused):
        export_snapshot(store, tmp_path / "failed-one")
    monkeypatch.undo()

    # Failure 2: a different real export failure code.
    monkeypatch.setattr(exporting, "_write_jsonl", _raise_synthetic)
    with pytest.raises(RuntimeError):
        export_snapshot(store, tmp_path / "failed-two")
    monkeypatch.undo()

    # F16: a real cleanup failure of another operation must stay unrecovered.
    monkeypatch.setattr(retention, "_tombstone", _raise_synthetic)
    with pytest.raises(RuntimeError):
        sweep(store, now=DUE_NOW)
    monkeypatch.undo()

    # A queue-backed export unit fails too; queue work is never auto-recovered.
    enqueue(store, "export", "queue:export:synthetic:1")
    queue_job = claim(store, worker_id="worker:test", kinds=("export",))[0]
    fail(store, queue_job["job_id"], queue_job["lease_token"], "EXPORT_QUEUE_FAILED", retryable=False)

    before = _pipelines(store)
    assert before["export"]["failed"] == 3, before["export"]
    assert before["export"]["state"] == "degraded", before["export"]
    assert before["cleanup"]["failed"] == 1, before["cleanup"]
    assert before["cleanup"]["last_error_code"] == "CLEANUP_FAILED", before["cleanup"]

    # One real successful export recovers both operation-ledger failures only.
    export_snapshot(store, tmp_path / "recovering-export")
    after = _pipelines(store)
    assert after["export"]["failed"] == 1, after["export"]
    assert after["export"]["state"] == "degraded", after["export"]
    assert after["export"]["last_error_code"] == "EXPORT_QUEUE_FAILED", after["export"]
    assert after["export"]["last_success_at"] is not None, after["export"]
    assert after["cleanup"]["failed"] == 1, after["cleanup"]
    assert after["cleanup"]["state"] == "degraded", after["cleanup"]
    assert after["cleanup"]["last_error_code"] == "CLEANUP_FAILED", after["cleanup"]

    rows = _ledger_rows(store, "export")
    operation_failed = [
        row
        for row in rows
        if row["state"] == "failed" and row["model_version"] == OPERATION_LEDGER_VERSION
    ]
    succeeded = [
        row
        for row in rows
        if row["state"] == "succeeded" and row["model_version"] == OPERATION_LEDGER_VERSION
    ]
    queue_failed = [
        row
        for row in rows
        if row["state"] == "failed" and row["job_id"] == queue_job["job_id"]
    ]
    assert len(operation_failed) == 2 and len(succeeded) == 1 and len(queue_failed) == 1, rows
    recovered_by = {row["recovered_by_job_id"] for row in operation_failed}
    assert recovered_by == {succeeded[0]["job_id"]}, operation_failed
    assert queue_failed[0]["recovered_by_job_id"] is None, queue_failed[0]
    _assert_status_contract(store)

    # Exactly one more success recovers... nothing new (idempotent association),
    # and the queue failure is still the visible one.
    export_snapshot(store, tmp_path / "second-recovering-export")
    final = _pipelines(store)["export"]
    assert final["failed"] == 1 and final["last_error_code"] == "EXPORT_QUEUE_FAILED", final


def test_AC52_queue_job_success_does_not_recover_operation_ledger_failure(
    store, tmp_path, monkeypatch
):
    """F16 symmetric boundary: a successful queue job of the same operation is a
    different unit of work and never recovers a recorded operation-ledger
    failure; only a later successful *operation* run does."""
    record(store, display_text="端口4200", value=4200)
    monkeypatch.setattr(exporting, "_assert_rows_secret_free", _refuse_export)
    with pytest.raises(ExportRefused):
        export_snapshot(store, tmp_path / "failed-export")
    monkeypatch.undo()
    failed = _pipelines(store)["export"]
    assert failed["failed"] == 1 and failed["state"] == "degraded", failed
    assert failed["last_error_code"] == "REDACTION_REQUIRED", failed

    enqueue(store, "export", "queue:export:synthetic:success")
    success_job = claim(store, worker_id="worker:test", kinds=("export",))[0]
    commit(
        store,
        success_job["job_id"],
        success_job["lease_token"],
        expected_input_version=int(success_job["input_version"]),
    )
    after_queue_success = _pipelines(store)["export"]
    assert after_queue_success["failed"] == 1, after_queue_success
    assert after_queue_success["state"] == "degraded", after_queue_success
    assert after_queue_success["last_error_code"] == "REDACTION_REQUIRED", after_queue_success
    assert after_queue_success["last_success_at"] is not None, after_queue_success
    operation_failed = [
        row
        for row in _operation_rows(store, "export")
        if row["state"] == "failed"
    ]
    assert len(operation_failed) == 1 and operation_failed[0]["recovered_by_job_id"] is None

    # Only the real export operation recovering run clears it.
    export_snapshot(store, tmp_path / "recovering-export")
    recovered = _pipelines(store)["export"]
    assert recovered["failed"] == 0 and recovered["state"] == "idle", recovered
    assert recovered["last_error_code"] is None, recovered


# ---------------------------------------------------------------------------
# The recovery association travels with the ledger row through a snapshot
# ---------------------------------------------------------------------------

def test_AC52_snapshot_export_import_keeps_recovered_and_unrecovered_failures(
    store, tmp_path, monkeypatch
):
    """The explicit association is part of the exported jobs ledger row: after
    import the recovered export failure stays recovered, while an unrecovered
    cleanup failure is still degraded (a restore never turns a failure green)."""
    _due_event_part(store)
    record(store, display_text="端口4200", value=4200)

    # Record and then recover an export failure.
    monkeypatch.setattr(exporting, "_assert_rows_secret_free", _refuse_export)
    with pytest.raises(ExportRefused):
        export_snapshot(store, tmp_path / "failed-export")
    monkeypatch.undo()
    export_snapshot(store, tmp_path / "recovering-export")

    # Record an unrecovered cleanup failure after the recovery.
    monkeypatch.setattr(retention, "_tombstone", _raise_synthetic)
    with pytest.raises(RuntimeError):
        sweep(store, now=DUE_NOW)
    monkeypatch.undo()

    before = _pipelines(store)
    assert before["export"]["failed"] == 0 and before["export"]["state"] == "idle", before["export"]
    assert before["cleanup"]["failed"] == 1, before["cleanup"]
    assert before["cleanup"]["last_error_code"] == "CLEANUP_FAILED", before["cleanup"]

    snapshot = tmp_path / "snapshot"
    export_snapshot(store, snapshot)
    jobs_rows = [
        json.loads(line)
        for line in (snapshot / "records" / "jobs.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    exported_failed = [
        row
        for row in jobs_rows
        if row["kind"] == "export" and row["state"] == "failed"
    ]
    exported_succeeded = [
        row
        for row in jobs_rows
        if row["kind"] == "export" and row["state"] == "succeeded"
    ]
    assert len(exported_failed) == 1 and len(exported_succeeded) == 1, jobs_rows
    assert exported_failed[0]["recovered_by_job_id"] == exported_succeeded[0]["job_id"], jobs_rows
    assert exported_failed[0]["recovered_at"] is not None, exported_failed
    # The snapshot still never exports the export's own in-flight ledger row.
    assert not [
        row
        for row in jobs_rows
        if row["kind"] == "export" and row["state"] == "pending"
    ], jobs_rows

    report = import_snapshot(snapshot, tmp_path / "restored")
    restored = Store(report["target_db"])
    after = _pipelines(restored)
    assert after["export"]["failed"] == 0 and after["export"]["state"] == "idle", after["export"]
    assert after["export"]["last_error_code"] is None, after["export"]
    assert after["export"]["last_success_at"] is not None, after["export"]
    assert after["cleanup"]["failed"] == 1, after["cleanup"]
    assert after["cleanup"]["state"] == "degraded", after["cleanup"]
    assert after["cleanup"]["last_error_code"] == "CLEANUP_FAILED", after["cleanup"]
    rows = _operation_rows(restored, "export")
    failed_rows = [row for row in rows if row["state"] == "failed"]
    succeeded_rows = [row for row in rows if row["state"] == "succeeded"]
    assert len(failed_rows) == 1 and len(succeeded_rows) == 1, rows
    assert failed_rows[0]["recovered_by_job_id"] == succeeded_rows[0]["job_id"], rows
    assert failed_rows[0]["recovered_at"] is not None, failed_rows


# ---------------------------------------------------------------------------
# Existing database migration: a round-4 database opens safely and is repaired
# idempotently; its recorded failures are neither hidden nor duplicated
# ---------------------------------------------------------------------------


def test_AC52_round4_database_migrates_recovery_columns_idempotently(tmp_path):
    """Opening a DB created before the recovery columns adds them idempotently,
    keeps existing ledger history and its unrecovered failure, and the next real
    success can then recover it."""
    db_path = tmp_path / "round4" / "knowledge.db"
    store = Store(db_path)
    record(store, display_text="端口4200", value=4200)
    failure = record_operation_start(store, "export")
    recorded = record_operation_finish(store, failure, error_code="EXPORT_FAILED")
    assert recorded["state"] == "failed"
    failed_before = _pipelines(store)["export"]
    assert failed_before["failed"] == 1 and failed_before["state"] == "degraded", failed_before

    # Simulate the round-4 schema: the recovery columns did not exist yet.
    with sqlite3.connect(db_path) as conn:
        for column in ("recovered_at", "recovered_by_job_id"):
            conn.execute(f"ALTER TABLE jobs DROP COLUMN {column}")
        conn.commit()
    with sqlite3.connect(db_path) as conn:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(jobs)")}
    assert "recovered_by_job_id" not in columns and "recovered_at" not in columns

    # The first open applies the compatibility migration; repeated opens are no-ops.
    migrated = Store(db_path)
    for _ in range(2):
        Store(db_path)
    with migrated.read() as conn:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(jobs)")}
    assert {"recovered_by_job_id", "recovered_at"} <= columns, columns

    # Existing ledger history and the unrecovered failure survive migration.
    failure_rows = [row for row in _operation_rows(migrated, "export") if row["state"] == "failed"]
    assert len(failure_rows) == 1, failure_rows
    assert failure_rows[0]["job_id"] == recorded["job_id"], failure_rows
    assert failure_rows[0]["last_error_code"] == "EXPORT_FAILED", failure_rows
    assert failure_rows[0]["recovered_by_job_id"] is None, failure_rows
    migrated_status = _pipelines(migrated)["export"]
    assert migrated_status["failed"] == 1 and migrated_status["state"] == "degraded", migrated_status
    assert migrated_status["last_error_code"] == "EXPORT_FAILED", migrated_status

    # A genuinely later real success records the new association and recovers it.
    export_snapshot(migrated, tmp_path / "after-migration")
    recovered = _pipelines(Store(db_path))["export"]
    assert recovered["failed"] == 0 and recovered["state"] == "idle", recovered
    assert recovered["last_error_code"] is None, recovered
    rows = _operation_rows(Store(db_path), "export")
    failed_rows = [row for row in rows if row["state"] == "failed"]
    succeeded_rows = [row for row in rows if row["state"] == "succeeded"]
    assert len(failed_rows) == 1, rows
    assert failed_rows[0]["recovered_by_job_id"] == succeeded_rows[0]["job_id"], rows


# ---------------------------------------------------------------------------
# Every real synchronous operation wired in round 4 recovers the same way
# (stable parametrized node ids: [operation-expected-code])
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("operation", "expected_code"),
    [
        ("export", "REDACTION_REQUIRED"),
        ("cleanup", "CLEANUP_FAILED"),
        ("project", "PROJECT_FAILED"),
    ],
)
def test_AC52_real_operation_success_recovers_its_recorded_failure(
    store, tmp_path, monkeypatch, operation, expected_code
):
    """Real failure entry point -> visible degraded failure; real success entry
    point of the same operation -> explicit durable recovery (all round-4 wired
    operations, not just hand-built job rows)."""
    _due_event_part(store)
    record(store, display_text="端口4200", value=4200)
    runs = {"count": 0}

    def run_real_operation():
        runs["count"] += 1
        if operation == "export":
            return export_snapshot(store, tmp_path / f"export-run-{runs['count']}")
        if operation == "cleanup":
            return sweep(store, now=DUE_NOW)
        return rebuild_projection(store)

    if operation == "export":
        monkeypatch.setattr(exporting, "_assert_rows_secret_free", _refuse_export)
    elif operation == "cleanup":
        monkeypatch.setattr(retention, "_tombstone", _raise_synthetic)
    else:
        monkeypatch.setattr(claims_module, "recompute_claim", _raise_synthetic)

    with pytest.raises((ExportRefused, RuntimeError)):
        run_real_operation()
    failed = _pipelines(store)[operation]
    assert failed["failed"] == 1, (operation, failed)
    assert failed["state"] == "degraded", (operation, failed)
    assert failed["last_error_code"] == expected_code, (operation, failed)
    assert failed["pending"] == 0 and failed["last_success_at"] is None, (operation, failed)
    failed_rows = [
        row for row in _operation_rows(store, operation) if row["state"] == "failed"
    ]
    assert len(failed_rows) == 1 and failed_rows[0]["recovered_by_job_id"] is None, failed_rows

    # The real operation runs again and its completion transaction recovers the
    # recorded failure explicitly.
    monkeypatch.undo()
    run_real_operation()
    recovered = _pipelines(store)[operation]
    assert recovered["failed"] == 0 and recovered["pending"] == 0, (operation, recovered)
    assert recovered["state"] == "idle", (operation, recovered)
    assert recovered["last_error_code"] is None, (operation, recovered)
    assert recovered["last_success_at"] is not None, (operation, recovered)
    rows = _operation_rows(store, operation)
    failed_rows = [row for row in rows if row["state"] == "failed"]
    succeeded_rows = [row for row in rows if row["state"] == "succeeded"]
    assert len(failed_rows) == 1 and len(succeeded_rows) == 1, (operation, rows)
    assert failed_rows[0]["recovered_by_job_id"] == succeeded_rows[0]["job_id"], rows
    assert failed_rows[0]["recovered_at"] is not None, failed_rows
    _assert_status_contract(store)
