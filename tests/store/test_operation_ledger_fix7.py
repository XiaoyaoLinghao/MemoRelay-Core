"""AC52-R6/R7 / F16, AC-31, AC-52: completion order must be durable evidence.

Codex round-6 counterexamples (``review/T20-CODEX-ROUND6.md`` §3 AC52-R6):

* variant A: two *real* export operations complete in the same UTC second.  The
  successful one completed first; a hand-edited snapshot claimed the recorded
  failure was recovered by that earlier success, and the round-6 importer trusted
  ``t.updated_at >= j.updated_at`` (equal timestamps pass) and hid the failure;
* variant B: mixed UTC-offset spellings.  A success at
  ``2026-10-31T09:00:00+08:00`` (01:00Z, actually earlier) compares as *greater*
  text than a failure at ``2026-10-31T02:00:00Z``, so a text comparison turned a
  later failure into "recovered by an earlier success".

The round-7 fix makes completion order a persisted fact instead of timestamp
text:

* ``jobs.completed_seq`` (NULL for rows written before the upgrade) and
  ``backend_meta.ledger_seq`` are added by the idempotent compatibility
  migration.  The transaction that finishes an operation row (success *or*
  failure) increments the ledger sequence and stamps the row in the same
  transaction, so equal-second completions and offset spellings no longer order
  each other by accident;
* ``store.status.VALID_RECOVERY_LINK_SQL`` requires both rows to carry a
  ``completed_seq`` and the recovering success to be *strictly* later
  (``t.completed_seq > j.completed_seq``); timestamp text is no longer consulted;
* an old database's already-recorded links are converted on first open
  (failed row -> 0, recorded target success -> next sequence values); anything
  that cannot be proven stays NULL, i.e. fail-safe visible;
* snapshot import normalizes the ``jobs`` timestamp columns to canonical UTC
  seconds, refuses present recovery links without a provable completion order
  (``SnapshotError('INVALID_REQUEST', ...)``, nothing published), and seeds the
  restored ``ledger_seq`` with ``max(completed_seq)`` so a success after the
  restore still completes strictly later;
* status derivation shows the *last completed* unrecovered failure
  (``completed_seq`` descending, NULL last) instead of the lexicographically
  largest timestamp text.

Every node exercises a real product entry point (``export_snapshot`` /
``import_snapshot`` / ``sweep``) or the durable store API; synthetic tmp_path
data only, no network, no real host data.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import sqlite3
import threading

import pytest

import knowledge_weaver.lifecycle.retention as retention
import knowledge_weaver.portability.export as exporting
from knowledge_weaver import contracts
from knowledge_weaver.lifecycle import register_lifecycle, sweep
from knowledge_weaver.portability import SnapshotError, export_snapshot, import_snapshot
from knowledge_weaver.portability.export import ExportRefused
from knowledge_weaver.store import Store, receive_batch, status as status_view
from knowledge_weaver.store.status import (
    OPERATION_LEDGER_VERSION,
    record_operation_finish,
    record_operation_start,
)

# ``knowledge_weaver.store.status`` is shadowed by the exported ``status``
# function in the package namespace, so resolve the module explicitly to
# monkeypatch its clock.
status_module = importlib.import_module("knowledge_weaver.store.status")

from ..state.conftest import record  # noqa: E402
from ..store.conftest import make_batch, make_event  # noqa: E402

DUE_NOW = "2026-10-31T00:00:00Z"
SAME_SECOND = "2026-10-31T02:00:00Z"


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
                "recovered_by_job_id, recovered_at, updated_at, created_at, completed_seq "
                "FROM jobs WHERE kind = ? ORDER BY created_at, rowid",
                (kind,),
            )
        ]


def _operation_rows(store, kind: str) -> list[dict]:
    return [
        row
        for row in _ledger_rows(store, kind)
        if row["model_version"] == OPERATION_LEDGER_VERSION
    ]


def _job_row(store, job_id: str) -> dict:
    with store.read() as conn:
        row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
    assert row is not None, job_id
    return dict(row)


def _update_job(store, job_id: str, **columns) -> None:
    assignments = ", ".join(f"{name} = ?" for name in columns)
    with store.transaction() as conn:
        cursor = conn.execute(
            f"UPDATE jobs SET {assignments} WHERE job_id = ?",
            (*columns.values(), job_id),
        )
        assert cursor.rowcount == 1, job_id


def _ledger_seq(store) -> int:
    with store.read() as conn:
        row = conn.execute("SELECT ledger_seq FROM backend_meta WHERE id = 1").fetchone()
    return int(row["ledger_seq"])


def _max_completed_seq(store) -> int:
    with store.read() as conn:
        return int(
            conn.execute("SELECT COALESCE(MAX(completed_seq), 0) FROM jobs").fetchone()[0]
        )


def _freeze_clock(monkeypatch, moment: str = SAME_SECOND) -> None:
    """Make every operation ledger finish use the same virtual timestamp."""

    monkeypatch.setattr(status_module, "utc_now", lambda: moment)


def _scripted_clock(monkeypatch, values: list[str]) -> None:
    """Deterministic finishing timestamps; the last value repeats afterwards."""
    state = {"index": 0}
    last = values[-1]

    def fake() -> str:
        index = state["index"]
        state["index"] = index + 1
        return values[index] if index < len(values) else last

    monkeypatch.setattr(status_module, "utc_now", fake)


def _raise_synthetic(*args, **kwargs):
    raise RuntimeError("synthetic operation failure")


def _refuse_export(table, rows):
    raise ExportRefused("REDACTION_REQUIRED", "synthetic export refusal")


def _fail_export(store, tmp_path, monkeypatch, name: str) -> dict:
    """Run the real export entry point and record one failed export ledger row."""
    with monkeypatch.context() as patch:
        patch.setattr(exporting, "_write_jsonl", _raise_synthetic)
        with pytest.raises(RuntimeError):
            export_snapshot(store, tmp_path / name)
    failed = [row for row in _operation_rows(store, "export") if row["state"] == "failed"]
    assert failed, "the real export failure must be recorded in the ledger"
    return failed[-1]


def _succeed_export(store, tmp_path, name: str) -> dict:
    """Run the real export entry point successfully and return its ledger row."""
    export_snapshot(store, tmp_path / name)
    succeeded = [row for row in _operation_rows(store, "export") if row["state"] == "succeeded"]
    assert succeeded
    return succeeded[-1]


def _due_event_part(store):
    """A real event part whose completed-task lifecycle anchor is due (cleanup)."""
    event = make_event(native_id="message:ac52-fix7", text="已完成任务的过程详情")
    receive_batch(store, make_batch([event]))
    register_lifecycle(
        store,
        subject_kind="event_part",
        subject_id=f"{event['event_id']}#1@0",
        class_="completed_task",
        business_anchor="2020-01-01T00:00:00Z",
    )
    return event


def _rehash(snapshot) -> None:
    """Recompute manifest file hashes and SHA256SUMS after a synthetic edit."""
    manifest_path = snapshot / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name, entry in manifest["files"].items():
        payload = (snapshot / name).read_bytes()
        entry["sha256"] = hashlib.sha256(payload).hexdigest()
        entry["size"] = len(payload)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    sums = "".join(
        hashlib.sha256(path.read_bytes()).hexdigest()
        + "  "
        + path.relative_to(snapshot).as_posix()
        + "\n"
        for path in sorted(snapshot.rglob("*"))
        if path.is_file() and path.name != "SHA256SUMS"
    )
    (snapshot / "SHA256SUMS").write_text(sums, encoding="utf-8")


def _snapshot_jobs(snapshot) -> list[dict]:
    path = snapshot / "records" / "jobs.jsonl"
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_snapshot_jobs(snapshot, rows: list[dict]) -> None:
    path = snapshot / "records" / "jobs.jsonl"
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    _rehash(snapshot)


def _export_snapshot_rows(snapshot) -> tuple[dict, dict]:
    """The failed and succeeded export ledger rows written into the snapshot."""
    rows = _snapshot_jobs(snapshot)
    failed = [
        row
        for row in rows
        if row["kind"] == "export" and row["state"] == "failed"
    ]
    succeeded = [
        row
        for row in rows
        if row["kind"] == "export"
        and row["state"] == "succeeded"
        and row["model_version"] == OPERATION_LEDGER_VERSION
    ]
    assert len(failed) == 1, failed
    assert succeeded, succeeded
    return failed[0], succeeded[-1]


# ---------------------------------------------------------------------------
# Same-second real concurrent exports: completion order, not the clock, decides
# ---------------------------------------------------------------------------


def test_AC52_same_second_success_completed_first_cannot_hide_later_failure(
    store, tmp_path, monkeypatch
):
    """Codex round-6 variant A, real entry point, equal-second timestamps.

    The successful export completes first; the second export then fails in the
    same virtual second.  The failure must stay visible (failed=1/degraded with
    its error code) and the earlier-completed success must not be recorded as its
    recovery, because ``completed_seq`` proves the failure completed later.
    """
    record(store, display_text="端口4200", value=4200)
    _freeze_clock(monkeypatch)
    late_failure_pending = threading.Event()
    early_success_finished = threading.Event()
    errors: list[BaseException] = []

    with monkeypatch.context() as patch:
        original = exporting._write_jsonl

        def controlled(path, rows):
            if threading.current_thread().name == "ac52-r7-late-failure":
                late_failure_pending.set()
                assert early_success_finished.wait(15), "the early success never finished"
                raise ExportRefused(
                    "REDACTION_REQUIRED",
                    "synthetic refusal after the same-second success already completed",
                )
            return original(path, rows)

        patch.setattr(exporting, "_write_jsonl", controlled)

        def late_failure():
            try:
                export_snapshot(store, tmp_path / "late-failed-export")
            except BaseException as exc:  # noqa: BLE001 - reported below
                errors.append(exc)

        thread = threading.Thread(target=late_failure, name="ac52-r7-late-failure")
        thread.start()
        assert late_failure_pending.wait(15), "the late-failing export never started"
        try:
            early_success = _succeed_export(store, tmp_path, "early-succeeded-export")
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

    rows = _operation_rows(store, "export")
    failed = [row for row in rows if row["state"] == "failed"][0]
    succeeded = [row for row in rows if row["state"] == "succeeded"][0]
    # Both really completed in the same stored second; only the durable sequence
    # records which one was later.
    assert failed["updated_at"] == succeeded["updated_at"] == SAME_SECOND, rows
    assert succeeded["completed_seq"] is not None and failed["completed_seq"] is not None, rows
    assert failed["completed_seq"] > succeeded["completed_seq"], rows
    assert failed["recovered_by_job_id"] is None, failed
    assert failed["recovered_at"] is None, failed
    _assert_status_contract(store)

    # The sequence is durable: a fresh session keeps the failure visible.
    restarted = _pipelines(Store(store.db_path))["export"]
    assert restarted["failed"] == 1 and restarted["state"] == "degraded", restarted
    assert restarted["last_error_code"] == "REDACTION_REQUIRED", restarted

    # A genuinely later real export completes strictly later and recovers it.
    recovering = _succeed_export(store, tmp_path, "recovering-export")
    recovered = _pipelines(store)["export"]
    assert recovered["failed"] == 0 and recovered["state"] == "idle", recovered
    assert recovered["last_error_code"] is None, recovered
    assert recovering["completed_seq"] > failed["completed_seq"], (recovering, failed)
    repaired = _job_row(store, failed["job_id"])
    assert repaired["recovered_by_job_id"] == recovering["job_id"], repaired
    assert repaired["recovered_at"] == SAME_SECOND, repaired


def test_AC52_same_second_failure_recorded_first_is_recovered_by_later_completion(
    store, tmp_path, monkeypatch
):
    """The mirror order in the same virtual second: a success that started first
    but completes after the recorded failure recovers it (completion, not start
    order or clock text, decides)."""
    record(store, display_text="端口4200", value=4200)
    _freeze_clock(monkeypatch)
    success_started = threading.Event()
    failure_recorded = threading.Event()
    errors: list[BaseException] = []

    with monkeypatch.context() as patch:
        original = exporting._write_jsonl

        def controlled(path, rows):
            if threading.current_thread().name == "ac52-r7-early-start-success":
                success_started.set()
                assert failure_recorded.wait(15), "the failure was never recorded"
                return original(path, rows)
            raise ExportRefused("REDACTION_REQUIRED", "synthetic early failure")

        patch.setattr(exporting, "_write_jsonl", controlled)

        def early_start_success():
            try:
                export_snapshot(store, tmp_path / "started-first-succeeded-last")
            except BaseException as exc:  # noqa: BLE001 - reported below
                errors.append(exc)

        thread = threading.Thread(
            target=early_start_success, name="ac52-r7-early-start-success"
        )
        thread.start()
        assert success_started.wait(15), "the early-started success never started"
        with pytest.raises(ExportRefused):
            export_snapshot(store, tmp_path / "started-later-failed-first")

        # The failure is durable while the success still runs: pending success is
        # not a recovery yet.
        in_flight = _pipelines(store)["export"]
        assert in_flight["failed"] == 1 and in_flight["pending"] == 1, in_flight
        assert in_flight["state"] == "degraded", in_flight
        assert in_flight["last_error_code"] == "REDACTION_REQUIRED", in_flight

        failure_recorded.set()
        thread.join(15)
        assert not thread.is_alive(), "the success export did not finish"
    assert errors == [], errors

    recovered = _pipelines(store)["export"]
    assert recovered["failed"] == 0 and recovered["pending"] == 0, recovered
    assert recovered["state"] == "idle", recovered
    assert recovered["last_error_code"] is None, recovered

    rows = _operation_rows(store, "export")
    failed = [row for row in rows if row["state"] == "failed"][0]
    succeeded = [row for row in rows if row["state"] == "succeeded"][0]
    assert failed["updated_at"] == succeeded["updated_at"] == SAME_SECOND, rows
    assert succeeded["completed_seq"] > failed["completed_seq"], rows
    assert failed["recovered_by_job_id"] == succeeded["job_id"], failed
    assert failed["recovered_at"] == SAME_SECOND, failed
    assert succeeded["rowid"] < failed["rowid"], rows  # started first, completed last
    _assert_status_contract(store)
    restarted = _pipelines(Store(store.db_path))["export"]
    assert restarted["failed"] == 0 and restarted["state"] == "idle", restarted


# ---------------------------------------------------------------------------
# Offset spellings and timestamp text no longer decide recovery causality
# ---------------------------------------------------------------------------


def test_AC52_offset_spelling_does_not_change_completion_causality(store):
    """``2026-10-31T09:00:00+08:00`` (01:00Z) completed before
    ``2026-10-31T02:00:00Z``; text comparison says the opposite.  Completion is
    ordered by the durable sequence, so a failure that completed later stays
    visible, and the later success that follows recovers it."""
    first = record_operation_start(store, "export")
    first_done = record_operation_finish(
        store, first, now="2026-10-31T09:00:00+08:00"
    )
    second = record_operation_start(store, "export")
    second_done = record_operation_finish(
        store, second, error_code="EXPORT_FAILED", now="2026-10-31T02:00:00Z"
    )
    # The timestamp *text* would order the success after the failure; the durable
    # completion sequence correctly orders it before.
    assert first_done["completed_seq"] < second_done["completed_seq"]
    visible = _pipelines(store)["export"]
    assert visible["failed"] == 1, visible
    assert visible["state"] == "degraded", visible
    assert visible["last_error_code"] == "EXPORT_FAILED", visible
    failed_row = _job_row(store, second_done["job_id"])
    assert failed_row["recovered_by_job_id"] is None, failed_row

    third = record_operation_start(store, "export")
    third_done = record_operation_finish(store, third, now="2026-10-31T02:00:01Z")
    recovered = _pipelines(store)["export"]
    assert recovered["failed"] == 0 and recovered["state"] == "idle", recovered
    assert third_done["completed_seq"] > failed_row["completed_seq"]
    recovered_row = _job_row(store, second_done["job_id"])
    assert recovered_row["recovered_by_job_id"] == third_done["job_id"], recovered_row


def test_AC52_last_error_code_follows_completion_order_not_timestamp_text(store):
    """When two unrecovered failures are visible, the reported error is the one
    that completed last (``completed_seq``), even if an earlier completion has a
    lexicographically larger timestamp."""
    first = record_operation_start(store, "export")
    record_operation_finish(
        store, first, error_code="FIRST_FAILURE", now="2026-10-31T02:00:05Z"
    )
    second = record_operation_start(store, "export")
    record_operation_finish(
        store, second, error_code="SECOND_FAILURE", now="2026-10-31T02:00:00Z"
    )
    pipeline = _pipelines(store)["export"]
    assert pipeline["failed"] == 2, pipeline
    assert pipeline["state"] == "degraded", pipeline
    assert pipeline["last_error_code"] == "SECOND_FAILURE", pipeline
    restarted = _pipelines(Store(store.db_path))["export"]
    assert restarted["last_error_code"] == "SECOND_FAILURE", restarted


# ---------------------------------------------------------------------------
# Snapshot export/import round trip keeps the durable completion ledger
# ---------------------------------------------------------------------------


def test_AC52_snapshot_roundtrip_keeps_recovered_and_unrecovered_failures(
    store, tmp_path, monkeypatch
):
    """A new-format snapshot carries ``completed_seq`` through export/import:
    a genuinely recovered export failure stays recovered, an unrecovered cleanup
    failure stays visible, and the restored ledger sequence is seeded above every
    restored completion order."""
    _due_event_part(store)
    record(store, display_text="端口4200", value=4200)

    with monkeypatch.context() as patch:
        patch.setattr(exporting, "_assert_rows_secret_free", _refuse_export)
        with pytest.raises(ExportRefused):
            export_snapshot(store, tmp_path / "failed-export")
    recovering = _succeed_export(store, tmp_path, "recovering-export")

    with monkeypatch.context() as patch:
        patch.setattr(retention, "_tombstone", _raise_synthetic)
        with pytest.raises(RuntimeError):
            sweep(store, now=DUE_NOW)

    before = _pipelines(store)
    assert before["export"]["failed"] == 0 and before["export"]["state"] == "idle", before
    assert before["cleanup"]["failed"] == 1 and before["cleanup"]["state"] == "degraded", before
    assert before["cleanup"]["last_error_code"] == "CLEANUP_FAILED", before

    snapshot = tmp_path / "snapshot"
    export_snapshot(store, snapshot)
    failed_record, succeeded_record = _export_snapshot_rows(snapshot)
    assert failed_record["recovered_by_job_id"] == succeeded_record["job_id"], failed_record
    assert failed_record["completed_seq"] is not None, failed_record
    assert succeeded_record["completed_seq"] is not None, succeeded_record
    assert succeeded_record["completed_seq"] > failed_record["completed_seq"], (
        failed_record,
        succeeded_record,
    )
    cleanup_rows = [
        row for row in _snapshot_jobs(snapshot) if row["kind"] == "cleanup" and row["state"] == "failed"
    ]
    assert len(cleanup_rows) == 1, cleanup_rows
    assert cleanup_rows[0]["completed_seq"] is not None, cleanup_rows
    assert cleanup_rows[0]["recovered_by_job_id"] is None, cleanup_rows

    report = import_snapshot(snapshot, tmp_path / "restored")
    assert report["referential_integrity"] == "ok", report
    restored = Store(report["target_db"])
    after = _pipelines(restored)
    assert after["export"]["failed"] == 0 and after["export"]["state"] == "idle", after["export"]
    assert after["cleanup"]["failed"] == 1 and after["cleanup"]["state"] == "degraded", after["cleanup"]
    assert after["cleanup"]["last_error_code"] == "CLEANUP_FAILED", after["cleanup"]

    restored_rows = _operation_rows(restored, "export")
    restored_failed = [row for row in restored_rows if row["state"] == "failed"][0]
    restored_succeeded = [row for row in restored_rows if row["state"] == "succeeded"][0]
    assert restored_failed["completed_seq"] == failed_record["completed_seq"], restored_rows
    assert restored_succeeded["completed_seq"] == succeeded_record["completed_seq"], restored_rows
    assert restored_failed["recovered_by_job_id"] == restored_succeeded["job_id"], restored_rows
    # The local ledger sequence is not exported; it is re-seeded above every
    # restored completion order so post-restore completions are strictly later.
    assert _ledger_seq(restored) == _max_completed_seq(restored), (
        _ledger_seq(restored),
        _max_completed_seq(restored),
    )
    assert _ledger_seq(restored) >= succeeded_record["completed_seq"]
    _assert_status_contract(restored)

    again = _pipelines(Store(report["target_db"]))
    assert again["export"]["failed"] == 0 and again["cleanup"]["failed"] == 1, again


def test_AC52_restored_ledger_sequence_lets_later_success_recover_old_failure(
    store, tmp_path, monkeypatch
):
    """After a restore, a real success must still complete *strictly later* than a
    restored failure: importing with ``ledger_seq`` left at 0 would tie the new
    sequence with the restored one and silently fail to recover."""
    _due_event_part(store)
    with monkeypatch.context() as patch:
        patch.setattr(retention, "_tombstone", _raise_synthetic)
        with pytest.raises(RuntimeError):
            sweep(store, now=DUE_NOW)
    failed_cleanup = [row for row in _operation_rows(store, "cleanup") if row["state"] == "failed"][0]
    assert failed_cleanup["completed_seq"] is not None, failed_cleanup

    snapshot = tmp_path / "snapshot"
    export_snapshot(store, snapshot)
    report = import_snapshot(snapshot, tmp_path / "restored")
    restored = Store(report["target_db"])
    visible = _pipelines(restored)["cleanup"]
    assert visible["failed"] == 1 and visible["state"] == "degraded", visible
    assert visible["last_error_code"] == "CLEANUP_FAILED", visible
    # The restored database only knows the snapshot's completions; the seed is the
    # max restored sequence, not the source's later export-ledger value.
    assert _ledger_seq(restored) == failed_cleanup["completed_seq"], (
        _ledger_seq(restored),
        failed_cleanup["completed_seq"],
    )

    sweep(restored, now=DUE_NOW)
    recovered = _pipelines(restored)["cleanup"]
    assert recovered["failed"] == 0 and recovered["state"] == "idle", recovered
    assert recovered["last_error_code"] is None, recovered
    restored_failed = [
        row for row in _operation_rows(restored, "cleanup") if row["state"] == "failed"
    ][0]
    restored_succeeded = [
        row for row in _operation_rows(restored, "cleanup") if row["state"] == "succeeded"
    ][-1]
    assert restored_succeeded["completed_seq"] > restored_failed["completed_seq"], (
        restored_failed,
        restored_succeeded,
    )
    assert restored_failed["recovered_by_job_id"] == restored_succeeded["job_id"], restored_failed


def test_AC52_import_normalizes_jobs_timestamps_to_utc_seconds(store, tmp_path):
    """``jobs`` timestamps are parsed as instants and stored as canonical UTC
    seconds (``...Z``), not kept as offset spellings; the recovery association and
    status survive the round trip."""
    failed = record_operation_finish(
        store,
        record_operation_start(store, "export"),
        error_code="EXPORT_FAILED",
        now="2026-10-31T09:00:00+08:00",
    )
    succeeded = record_operation_finish(
        store,
        record_operation_start(store, "export"),
        now="2026-10-31T02:00:00Z",
    )
    # The success finishes later despite the lexicographically larger failure
    # spelling (AC52-R7 completion order is the sequence).
    assert succeeded["completed_seq"] > failed["completed_seq"]
    assert _pipelines(store)["export"]["failed"] == 0
    snapshot = tmp_path / "snapshot"
    export_snapshot(store, snapshot)
    raw = _snapshot_jobs(snapshot)
    raw_failed = next(row for row in raw if row["job_id"] == failed["job_id"])
    assert raw_failed["updated_at"] == "2026-10-31T09:00:00+08:00", raw_failed

    report = import_snapshot(snapshot, tmp_path / "restored")
    restored = Store(report["target_db"])
    restored_failed = _job_row(restored, failed["job_id"])
    restored_succeeded = _job_row(restored, succeeded["job_id"])
    assert restored_failed["created_at"].endswith("Z"), restored_failed
    assert restored_failed["updated_at"] == "2026-10-31T01:00:00Z", restored_failed
    assert restored_failed["recovered_at"] == "2026-10-31T02:00:00Z", restored_failed
    assert restored_succeeded["updated_at"] == "2026-10-31T02:00:00Z", restored_succeeded
    assert _pipelines(restored)["export"]["failed"] == 0, _pipelines(restored)["export"]


def test_AC52_old_format_snapshot_without_links_still_imports(
    store, tmp_path, monkeypatch
):
    """Compatibility control: a pre-round-7 snapshot that contains *no* recovery
    association still imports (its ``jobs`` rows simply have no
    ``completed_seq``), the failure stays visible, and a later real success
    recovers it by stamping the legacy failed row ``0``."""
    record(store, display_text="端口4200", value=4200)
    failed = _fail_export(store, tmp_path, monkeypatch, "failed-export")
    assert failed["completed_seq"] is not None, failed

    snapshot = tmp_path / "snapshot"
    export_snapshot(store, snapshot)
    rows = _snapshot_jobs(snapshot)
    exported_failed = [row for row in rows if row["job_id"] == failed["job_id"]]
    assert len(exported_failed) == 1, exported_failed
    # The export's own successful completion happens after its records are
    # written, so the snapshot still shows the failure unrecovered.
    assert exported_failed[0].get("recovered_by_job_id") is None, exported_failed
    for row in rows:  # simulate the pre-round-7 jobs record format
        row.pop("completed_seq", None)
    _write_snapshot_jobs(snapshot, rows)

    report = import_snapshot(snapshot, tmp_path / "restored")
    assert report["referential_integrity"] == "ok", report
    restored = Store(report["target_db"])
    restored_failed = _job_row(restored, failed["job_id"])
    assert restored_failed["completed_seq"] is None, restored_failed
    visible = _pipelines(restored)["export"]
    assert visible["failed"] == 1 and visible["state"] == "degraded", visible
    assert visible["last_error_code"] == "EXPORT_FAILED", visible
    assert _ledger_seq(restored) == 0, _ledger_seq(restored)

    # The first success after the restore completes at sequence 1 and recovers
    # the legacy failure, which is stamped 0 (it existed before the transaction).
    recovering = _succeed_export(restored, tmp_path, "recovering-after-old-snapshot")
    assert recovering["completed_seq"] == 1, recovering
    recovered_row = _job_row(restored, failed["job_id"])
    assert recovered_row["completed_seq"] == 0, recovered_row
    assert recovered_row["recovered_by_job_id"] == recovering["job_id"], recovered_row
    assert _pipelines(restored)["export"]["failed"] == 0, _pipelines(restored)["export"]


# ---------------------------------------------------------------------------
# Forged recovery links in a snapshot are refused (round-6 variants A and B)
# ---------------------------------------------------------------------------

FORGED_LINK_VARIANTS = (
    "same-second-earlier-success",
    "mixed-offset-earlier-success",
    "missing-completion-order",
    "equal-completion-order",
)


def _prepare_forged_snapshot(store, tmp_path, monkeypatch, variant: str):
    """Real ledger history (an earlier success, a later failure, same operation),
    export it, then forge the failure's recovery link to the earlier success."""
    record(store, display_text="端口4200", value=4200)
    if variant == "mixed-offset-earlier-success":
        # One clock value per record_operation_* call (start and finish): the
        # earlier success *finishes* at an offset spelling whose text sorts after
        # the later failure's UTC spelling, although its instant is earlier.
        _scripted_clock(
            monkeypatch,
            [
                "2026-10-31T09:00:00+08:00",
                "2026-10-31T09:00:00+08:00",
                "2026-10-31T02:00:00Z",
                "2026-10-31T02:00:00Z",
            ],
        )
    else:
        _scripted_clock(monkeypatch, [SAME_SECOND])
    early_success = _succeed_export(store, tmp_path, "earlier-success")
    late_failure = _fail_export(store, tmp_path, monkeypatch, "later-failure")
    assert late_failure["completed_seq"] > early_success["completed_seq"]
    assert _pipelines(store)["export"]["failed"] == 1

    snapshot = tmp_path / "snapshot"
    export_snapshot(store, snapshot)
    failed_record, succeeded_record = _export_snapshot_rows(snapshot)
    assert failed_record["job_id"] == late_failure["job_id"], failed_record
    assert succeeded_record["job_id"] == early_success["job_id"], succeeded_record

    rows = _snapshot_jobs(snapshot)
    for row in rows:
        if row["job_id"] == failed_record["job_id"]:
            row["recovered_by_job_id"] = succeeded_record["job_id"]
            row["recovered_at"] = succeeded_record["updated_at"]
            if variant == "equal-completion-order":
                row["completed_seq"] = succeeded_record["completed_seq"]
        elif row["job_id"] == succeeded_record["job_id"]:
            if variant == "missing-completion-order":
                row["completed_seq"] = None
        if variant == "missing-completion-order":
            row["completed_seq"] = None
    _write_snapshot_jobs(snapshot, rows)
    return snapshot, failed_record, succeeded_record


@pytest.mark.parametrize("variant", FORGED_LINK_VARIANTS)
def test_AC52_forged_recovery_link_snapshot_is_refused(
    store, tmp_path, monkeypatch, variant
):
    """A snapshot cannot claim an earlier/same-second success recovered a later
    failure: without a strictly greater ``completed_seq`` it is logically corrupt
    and refused with ``INVALID_REQUEST``; nothing is published."""
    snapshot, failed_record, succeeded_record = _prepare_forged_snapshot(
        store, tmp_path, monkeypatch, variant
    )
    if variant == "mixed-offset-earlier-success":
        # The old text comparison would have trusted this: "09:00:00+08:00" is a
        # larger *string* than "02:00:00Z", although it is an earlier instant.
        assert succeeded_record["updated_at"] == "2026-10-31T09:00:00+08:00", succeeded_record
        assert failed_record["updated_at"] == "2026-10-31T02:00:00Z", failed_record
        assert succeeded_record["updated_at"] > failed_record["updated_at"]  # text order
    else:
        assert failed_record["updated_at"] == succeeded_record["updated_at"] == SAME_SECOND

    target = tmp_path / "restored"
    with pytest.raises(SnapshotError) as excinfo:
        import_snapshot(snapshot, target)
    assert type(excinfo.value) is SnapshotError
    assert excinfo.value.code == "INVALID_REQUEST"
    message = str(excinfo.value)
    assert "referential integrity failed" in message, message
    assert "invalid_recovery_links" in message, message
    assert not target.exists(), "a refused import must not leave a restored target behind"
    assert not (target / "knowledge.db").exists()


def test_AC52_forged_same_second_link_in_live_database_stays_visible(store, tmp_path, monkeypatch):
    """The status view fails safe on its own: a same-second link written directly
    into a legacy-style database (no strictly greater completion order) never
    hides the failure."""
    record(store, display_text="端口4200", value=4200)
    _freeze_clock(monkeypatch)
    early_success = _succeed_export(store, tmp_path, "earlier-success")
    late_failure = _fail_export(store, tmp_path, monkeypatch, "later-failure")
    _update_job(
        store,
        late_failure["job_id"],
        recovered_by_job_id=early_success["job_id"],
        recovered_at=late_failure["updated_at"],
    )
    pipeline = _pipelines(store)["export"]
    assert pipeline["failed"] == 1, pipeline
    assert pipeline["state"] == "degraded", pipeline
    assert pipeline["last_error_code"] == "EXPORT_FAILED", pipeline
    restarted = _pipelines(Store(store.db_path))["export"]
    assert restarted["failed"] == 1 and restarted["last_error_code"] == "EXPORT_FAILED", restarted


# ---------------------------------------------------------------------------
# Old database upgrade: recorded links converted once, unprovable links fail safe
# ---------------------------------------------------------------------------


def _drop_completion_columns(db_path) -> None:
    """Simulate a pre-round-7 database (no completion-order columns)."""
    with sqlite3.connect(db_path) as conn:
        conn.execute("ALTER TABLE jobs DROP COLUMN completed_seq")
        conn.execute("ALTER TABLE backend_meta DROP COLUMN ledger_seq")
        conn.commit()
    with sqlite3.connect(db_path) as conn:
        job_columns = {row[1] for row in conn.execute("PRAGMA table_info(jobs)")}
        meta_columns = {row[1] for row in conn.execute("PRAGMA table_info(backend_meta)")}
    assert "completed_seq" not in job_columns, job_columns
    assert "ledger_seq" not in meta_columns, meta_columns


def _completion_state(store) -> tuple:
    rows = []
    with store.read() as conn:
        for row in conn.execute(
            "SELECT job_id, state, completed_seq FROM jobs ORDER BY rowid"
        ):
            rows.append((row["job_id"], row["state"], row["completed_seq"]))
        ledger = conn.execute("SELECT ledger_seq FROM backend_meta WHERE id = 1").fetchone()
    return tuple(rows), int(ledger["ledger_seq"])


def test_AC52_old_database_upgrade_backfills_recorded_recovery_links(store):
    """Opening a pre-round-7 database converts the links the old code had already
    recorded (failed -> 0, target success -> next sequences) so an already
    recovered failure stays recovered; an unrecovered old failure stays visible
    and its next success recovers it.  Repeated opens are no-ops."""
    record(store, display_text="端口4200", value=4200)
    failed_one = record_operation_finish(
        store, record_operation_start(store, "export"), error_code="EXPORT_FAILED"
    )
    failed_two = record_operation_finish(
        store, record_operation_start(store, "export"), error_code="EXPORT_REFUSED"
    )
    success = record_operation_finish(store, record_operation_start(store, "export"))
    unrecovered = record_operation_finish(
        store, record_operation_start(store, "cleanup"), error_code="CLEANUP_FAILED"
    )
    assert _pipelines(store)["export"]["state"] == "idle"
    assert _pipelines(store)["cleanup"]["state"] == "degraded"

    _drop_completion_columns(store.db_path)
    migrated = Store(store.db_path)
    first_state = _completion_state(migrated)
    # Idempotent: repeated opens do not reassign sequences.
    for _ in range(2):
        Store(store.db_path)
    assert _completion_state(Store(store.db_path)) == first_state

    after_migration = _pipelines(migrated)
    assert after_migration["export"]["failed"] == 0, after_migration["export"]
    assert after_migration["export"]["state"] == "idle", after_migration["export"]
    assert after_migration["cleanup"]["failed"] == 1, after_migration["cleanup"]
    assert after_migration["cleanup"]["state"] == "degraded", after_migration["cleanup"]
    assert after_migration["cleanup"]["last_error_code"] == "CLEANUP_FAILED", after_migration

    for failed_job in (failed_one, failed_two):
        row = _job_row(migrated, failed_job["job_id"])
        assert row["completed_seq"] == 0, row
        assert row["recovered_by_job_id"] == success["job_id"], row
    migrated_success = _job_row(migrated, success["job_id"])
    assert migrated_success["completed_seq"] is not None, migrated_success
    assert migrated_success["completed_seq"] > 0, migrated_success
    assert _ledger_seq(migrated) == migrated_success["completed_seq"], _ledger_seq(migrated)
    # The unproven unrecovered failure keeps NULL (fail-safe visible).
    assert _job_row(migrated, unrecovered["job_id"])["completed_seq"] is None

    # A genuinely later success still completes above the migrated sequence and
    # recovers the old unrecovered failure.
    later = record_operation_finish(Store(store.db_path), record_operation_start(Store(store.db_path), "cleanup"))
    assert later["completed_seq"] > migrated_success["completed_seq"], (later, migrated_success)
    recovered = _pipelines(Store(store.db_path))["cleanup"]
    assert recovered["failed"] == 0 and recovered["state"] == "idle", recovered
    recovered_row = _job_row(Store(store.db_path), unrecovered["job_id"])
    assert recovered_row["recovered_by_job_id"] == later["job_id"], recovered_row
    assert recovered_row["completed_seq"] == 0, recovered_row


def test_AC52_old_database_unprovable_link_stays_fail_safe_visible(store):
    """A legacy link whose target cannot be proven is left without a completion
    order: the upgrade never guesses, so the failure stays visible."""
    record(store, display_text="端口4200", value=4200)
    failed = record_operation_finish(
        store, record_operation_start(store, "export"), error_code="EXPORT_FAILED"
    )
    with store.transaction() as conn:
        conn.execute(
            "UPDATE jobs SET recovered_by_job_id = 'job:R7missing', recovered_at = ? "
            "WHERE job_id = ?",
            (DUE_NOW, failed["job_id"]),
        )
    _drop_completion_columns(store.db_path)
    migrated = Store(store.db_path)

    row = _job_row(migrated, failed["job_id"])
    assert row["completed_seq"] is None, row
    assert row["recovered_by_job_id"] == "job:R7missing", row
    pipeline = _pipelines(migrated)["export"]
    assert pipeline["failed"] == 1, pipeline
    assert pipeline["state"] == "degraded", pipeline
    assert pipeline["last_error_code"] == "EXPORT_FAILED", pipeline
    assert _pipelines(Store(store.db_path))["export"]["failed"] == 1
