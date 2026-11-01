"""AC52-R5 / F16, AC-31: a non-NULL ``recovered_by_job_id`` is not a recovery.

Migrated from the Codex round-5 counterexamples
(``artifacts/T20-codex-round5/run-20261031-independent/repros/test_recovery_links.py``):
a real export failure was recorded, a normal export produced a snapshot still
containing that failed ledger row, and only the synthetic snapshot's ``jobs``
record was edited so the failed export's ``recovered_by_job_id`` pointed at
(a) a job that does not exist, (b) the failed row itself, or (c) a really
succeeded *cleanup* job (another operation).  All three imported successfully
and ``status(export)`` became ``failed=0 / idle / last_error=None``: the
snapshot hid an unrecovered failure.

The round-5 status view trusted ``recovered_by_job_id IS NOT NULL`` and
``importing._validate_referential_integrity`` did not validate the new recovery
reference.  The fix is two-layered:

* snapshot import validates every present recovery association and refuses a
  logically corrupt snapshot with ``SnapshotError('INVALID_REQUEST', ...)``
  (the module's existing integrity-error contract);
* status derivation only subtracts a failure when the same validity predicate
  (``store.status.VALID_RECOVERY_LINK_SQL``) holds, so an invalid relation that
  reaches a database through an old/upgraded schema or any other path still
  keeps the failure visible instead of silently greening the pipeline.

Valid recovery must name an existing, different, same-operation
operation-ledger job that actually succeeded and completed no earlier than the
failure was recorded; the link must be complete (``recovered_at`` present).

Preserved controls (real entry points only): normal same-operation recovery by
a success that completed after the failure, persistence across restart, the
snapshot export/import round trip of a valid recovery, the old-schema upgrade
path, and the original real concurrent-export counterexample.

Only synthetic ``tmp_path`` data is used; no network, no real host data.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading

import pytest

import knowledge_weaver.portability.export as exporting
from knowledge_weaver import contracts
from knowledge_weaver.jobs import claim, commit, enqueue
from knowledge_weaver.lifecycle import sweep
from knowledge_weaver.portability import SnapshotError, export_snapshot, import_snapshot
from knowledge_weaver.portability.export import ExportRefused
from knowledge_weaver.store import Store, status as status_view
from knowledge_weaver.store.status import OPERATION_LEDGER_VERSION

from ..state.conftest import record

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
                "recovered_by_job_id, recovered_at, updated_at, completed_seq "
                "FROM jobs WHERE kind = ? "
                "ORDER BY created_at, rowid",
                (kind,),
            )
        ]


def _operation_rows(store, kind: str) -> list[dict]:
    return [
        row for row in _ledger_rows(store, kind) if row["model_version"] == OPERATION_LEDGER_VERSION
    ]


def _job_row(store, job_id: str) -> dict:
    with store.read() as conn:
        row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
    assert row is not None, job_id
    return dict(row)


def _update_job(store, job_id: str, **columns) -> None:
    """Corrupt one durable ledger row directly (a non-import/old-database path)."""
    assignments = ", ".join(f"{name} = ?" for name in columns)
    with store.transaction() as conn:
        cursor = conn.execute(
            f"UPDATE jobs SET {assignments} WHERE job_id = ?",
            (*columns.values(), job_id),
        )
        assert cursor.rowcount == 1, job_id


def _raise_synthetic(*args, **kwargs):
    raise RuntimeError("synthetic operation failure")


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


def _cleanup_success(store) -> str:
    """A real successful *cleanup* operation ledger row (another operation)."""
    sweep(store, now=DUE_NOW)
    succeeded = [
        row for row in _operation_rows(store, "cleanup") if row["state"] == "succeeded"
    ]
    assert succeeded
    return succeeded[-1]["job_id"]


def _queue_export_success(store) -> str:
    """A real succeeded *queue-backed* export job (not an operation ledger row)."""
    enqueue(store, "export", "queue:ac52-fix5:succeeded")
    job = claim(store, worker_id="worker:ac52-fix5", kinds=("export",))[0]
    commit(
        store,
        job["job_id"],
        job["lease_token"],
        expected_input_version=int(job["input_version"]),
    )
    return job["job_id"]


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
        hashlib.sha256(path.read_bytes()).hexdigest() + "  " + path.relative_to(snapshot).as_posix() + "\n"
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


# ---------------------------------------------------------------------------
# Import must refuse a logically corrupt recovery association (AC52-R5)
# ---------------------------------------------------------------------------

INVALID_LINK_RELATIONS = (
    "missing-target",
    "self-reference",
    "other-operation-succeeded",
    "target-failed",
    "target-not-operation-ledger",
    "missing-recovered-at",
    "recovered-at-without-id",
    "target-completed-before-failure",
    "link-on-succeeded-row",
)


def _prepare_tampered_snapshot(store, tmp_path, monkeypatch, relation: str):
    """Real ledger history + one synthetic, logically-invalid recovery link.

    The snapshot is exported through the real entry point from a real store that
    already contains a valid recovery (two failed exports recovered by a later
    successful export).  Only the ``jobs`` record of the snapshot is edited for
    the parametrized invalid relation, and hashes are recomputed so byte damage
    cannot be the reason for a refusal.
    """
    record(store, display_text="端口4200", value=4200)
    cleanup_job = _cleanup_success(store)
    failed_one = _fail_export(store, tmp_path, monkeypatch, "failed-export-one")
    failed_two = _fail_export(store, tmp_path, monkeypatch, "failed-export-two")
    success = _succeed_export(store, tmp_path, "recovering-export")
    queue_job = _queue_export_success(store)

    snapshot = tmp_path / "snapshot"
    export_snapshot(store, snapshot)
    rows = _snapshot_jobs(snapshot)
    failed = sorted(
        (row for row in rows if row["kind"] == "export" and row["state"] == "failed"),
        key=lambda row: row["job_id"],
    )
    assert len(failed) == 2, failed
    target = failed[0]
    other_failed = failed[1]
    succeeded = next(row for row in rows if row["job_id"] == success["job_id"])
    # Baseline: the real later success did recover both failures before export.
    assert target["recovered_by_job_id"] == success["job_id"], target
    assert other_failed["recovered_by_job_id"] == success["job_id"], other_failed

    if relation == "missing-target":
        target["recovered_by_job_id"] = "job:C5missing"
    elif relation == "self-reference":
        target["recovered_by_job_id"] = target["job_id"]
    elif relation == "other-operation-succeeded":
        target["recovered_by_job_id"] = cleanup_job
    elif relation == "target-failed":
        target["recovered_by_job_id"] = other_failed["job_id"]
    elif relation == "target-not-operation-ledger":
        target["recovered_by_job_id"] = queue_job
    elif relation == "missing-recovered-at":
        target["recovered_at"] = None
    elif relation == "recovered-at-without-id":
        target["recovered_by_job_id"] = None
    elif relation == "target-completed-before-failure":
        # Same operation, operation-ledger, succeeded target — but its durable
        # completion order does not prove it finished *after* the failure it
        # claims to recover (AC52-R7 uses completed_seq, not timestamp text).
        target["completed_seq"] = succeeded["completed_seq"]
    elif relation == "link-on-succeeded-row":
        succeeded["recovered_by_job_id"] = other_failed["job_id"]
        succeeded["recovered_at"] = DUE_NOW
    else:  # pragma: no cover - defensive
        raise AssertionError(relation)
    _write_snapshot_jobs(snapshot, rows)
    return snapshot, (failed_one, failed_two, success, queue_job)


@pytest.mark.parametrize("relation", INVALID_LINK_RELATIONS)
def test_AC52_invalid_recovery_link_snapshot_is_refused(
    store, tmp_path, monkeypatch, relation
):
    """Every invalid recovery relation is refused before anything is published."""
    snapshot, _ids = _prepare_tampered_snapshot(store, tmp_path, monkeypatch, relation)
    target = tmp_path / "restored"
    with pytest.raises(SnapshotError) as excinfo:
        import_snapshot(snapshot, target)
    assert type(excinfo.value) is SnapshotError
    assert excinfo.value.code == "INVALID_REQUEST"
    message = str(excinfo.value)
    assert "referential integrity failed" in message, message
    assert "invalid_recovery_links" in message, message
    # A refused snapshot publishes nothing: the import target created by this
    # failed import is cleaned up again (no partially restored database).
    assert not target.exists(), "a refused import must not leave a restored target behind"
    assert not (target / "knowledge.db").exists()


# ---------------------------------------------------------------------------
# Positive control: a real valid recovery round-trips through export/import
# ---------------------------------------------------------------------------


def test_AC52_valid_recovery_survives_restart_and_snapshot_roundtrip(
    store, tmp_path, monkeypatch
):
    """A real failure recovered by a later real success stays recovered across a
    process restart and a real snapshot export/import round trip."""
    record(store, display_text="端口4200", value=4200)
    failed = _fail_export(store, tmp_path, monkeypatch, "failed-export")
    visible = _pipelines(store)["export"]
    assert visible["failed"] == 1 and visible["state"] == "degraded", visible
    assert visible["last_error_code"] == "EXPORT_FAILED", visible

    success = _succeed_export(store, tmp_path, "recovering-export")
    recovered = _pipelines(store)["export"]
    assert recovered["failed"] == 0 and recovered["state"] == "idle", recovered
    assert recovered["last_error_code"] is None and recovered["last_success_at"], recovered
    rows = _operation_rows(store, "export")
    failure_row = next(row for row in rows if row["state"] == "failed")
    success_row = next(row for row in rows if row["state"] == "succeeded")
    assert failure_row["job_id"] == failed["job_id"], rows
    assert failure_row["recovered_by_job_id"] == success_row["job_id"], rows
    assert failure_row["recovered_at"] is not None, rows
    # AC52-R7: the contract is a durable, strictly greater completion order (the
    # round-5 ``updated_at >=`` timestamp-text check is superseded; equal-second
    # timestamps and mixed UTC offsets cannot prove causality).
    assert success_row["completed_seq"] > failure_row["completed_seq"], rows
    _assert_status_contract(store)

    # Restart: the durable association, not in-memory state, keeps it recovered.
    restarted = Store(store.db_path)
    durable = _pipelines(restarted)["export"]
    assert durable["failed"] == 0 and durable["state"] == "idle", durable
    assert durable["last_error_code"] is None, durable

    snapshot = tmp_path / "snapshot"
    export_snapshot(store, snapshot)
    report = import_snapshot(snapshot, tmp_path / "restored")
    assert report["referential_integrity"] == "ok", report
    assert report["cleanup_warnings"] == [], report
    restored = Store(report["target_db"])
    after = _pipelines(restored)["export"]
    assert after["failed"] == 0 and after["state"] == "idle", after
    assert after["last_error_code"] is None, after
    restored_rows = _operation_rows(restored, "export")
    restored_failure = next(row for row in restored_rows if row["state"] == "failed")
    assert restored_failure["recovered_by_job_id"] == success_row["job_id"], restored_rows
    assert restored_failure["recovered_at"] is not None, restored_rows
    _assert_status_contract(restored)
    again = _pipelines(Store(report["target_db"]))["export"]
    assert again["failed"] == 0 and again["state"] == "idle", again


# ---------------------------------------------------------------------------
# Status derivation itself must fail safe on an invalid relation
# ---------------------------------------------------------------------------

STATUS_INVALID_RELATIONS = (
    "missing-target",
    "self-reference",
    "other-operation-succeeded",
    "target-failed",
    "target-not-operation-ledger",
    "missing-recovered-at",
    "impossible-completion-order",
)


@pytest.mark.parametrize("relation", STATUS_INVALID_RELATIONS)
def test_AC52_invalid_relation_in_existing_database_never_hides_failure(
    store, tmp_path, monkeypatch, relation
):
    """Status derivation is conservative on its own: a corrupt/legacy row whose
    ``recovered_by_job_id`` is non-NULL but invalid keeps the failure visible
    and keeps its error code (no import validation involved)."""
    record(store, display_text="端口4200", value=4200)
    cleanup_job = _cleanup_success(store)
    failed_one = _fail_export(store, tmp_path, monkeypatch, "failed-export-one")
    failed_two = _fail_export(store, tmp_path, monkeypatch, "failed-export-two")
    success = _succeed_export(store, tmp_path, "recovering-export")
    queue_job = _queue_export_success(store)
    # Sanity: the real recovery is valid, so the pipeline is healthy before the
    # synthetic corruption.
    assert _pipelines(store)["export"]["failed"] == 0

    if relation == "missing-target":
        _update_job(
            store,
            failed_one["job_id"],
            recovered_by_job_id="job:C5missing",
            recovered_at=DUE_NOW,
        )
    elif relation == "self-reference":
        _update_job(
            store,
            failed_one["job_id"],
            recovered_by_job_id=failed_one["job_id"],
            recovered_at=DUE_NOW,
        )
    elif relation == "other-operation-succeeded":
        _update_job(
            store,
            failed_one["job_id"],
            recovered_by_job_id=cleanup_job,
            recovered_at=DUE_NOW,
        )
    elif relation == "target-failed":
        _update_job(
            store,
            failed_one["job_id"],
            recovered_by_job_id=failed_two["job_id"],
            recovered_at=DUE_NOW,
        )
    elif relation == "target-not-operation-ledger":
        _update_job(
            store,
            failed_one["job_id"],
            recovered_by_job_id=queue_job,
            recovered_at=DUE_NOW,
        )
    elif relation == "missing-recovered-at":
        # The association target is real and valid, but the link is incomplete.
        _update_job(store, failed_one["job_id"], recovered_at=None)
    elif relation == "impossible-completion-order":
        # Same operation, operation-ledger, succeeded target — but the durable
        # completion order does not prove it finished *after* the failure it
        # claims to recover (AC52-R7: completed_seq, not timestamp text).
        _update_job(
            store, failed_one["job_id"], completed_seq=success["completed_seq"]
        )
    else:  # pragma: no cover - defensive
        raise AssertionError(relation)

    pipeline = _pipelines(store)["export"]
    assert pipeline["failed"] == 1, (relation, pipeline)
    assert pipeline["state"] == "degraded", (relation, pipeline)
    assert pipeline["last_error_code"] == "EXPORT_FAILED", (relation, pipeline)
    assert pipeline["pending"] == 0, pipeline
    assert pipeline["last_success_at"] is not None, pipeline
    _assert_status_contract(store)

    # The failure stays visible after a restart too (durable ledger read).
    restarted = _pipelines(Store(store.db_path))["export"]
    assert restarted["failed"] == 1, (relation, restarted)
    assert restarted["state"] == "degraded", (relation, restarted)
    assert restarted["last_error_code"] == "EXPORT_FAILED", (relation, restarted)
    # The corrupt association is never rewritten by the status view.
    row = _job_row(store, failed_one["job_id"])
    assert row["state"] == "failed"
    assert row["recovered_by_job_id"] is not None
    assert _job_row(store, success["job_id"])["state"] == "succeeded"


# ---------------------------------------------------------------------------
# Old-schema upgrade path
# ---------------------------------------------------------------------------


def _drop_recovery_columns(db_path) -> None:
    """Simulate a pre-round-5 database (no recovery association columns)."""
    with sqlite3.connect(db_path) as conn:
        for column in ("recovered_at", "recovered_by_job_id"):
            conn.execute(f"ALTER TABLE jobs DROP COLUMN {column}")
        conn.commit()
    with sqlite3.connect(db_path) as conn:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(jobs)")}
    assert "recovered_by_job_id" not in columns and "recovered_at" not in columns


def test_AC52_old_schema_upgrade_keeps_failure_visible_and_new_success_recovers(
    store, tmp_path, monkeypatch
):
    """Opening a pre-round-5 database adds the columns idempotently, its failure
    stays unrecovered, and a genuinely later real success recovers it."""
    record(store, display_text="端口4200", value=4200)
    failed = _fail_export(store, tmp_path, monkeypatch, "failed-export")
    assert _pipelines(store)["export"]["failed"] == 1

    _drop_recovery_columns(store.db_path)
    migrated = Store(store.db_path)
    for _ in range(2):  # repeated opens are no-ops
        Store(store.db_path)
    with migrated.read() as conn:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(jobs)")}
    assert {"recovered_by_job_id", "recovered_at"} <= columns, columns

    after_migration = _pipelines(migrated)["export"]
    assert after_migration["failed"] == 1, after_migration
    assert after_migration["state"] == "degraded", after_migration
    assert after_migration["last_error_code"] == "EXPORT_FAILED", after_migration
    failure_row = _job_row(migrated, failed["job_id"])
    assert failure_row["state"] == "failed"
    assert failure_row["recovered_by_job_id"] is None

    _succeed_export(migrated, tmp_path, "after-migration")
    recovered = _pipelines(Store(store.db_path))["export"]
    assert recovered["failed"] == 0 and recovered["state"] == "idle", recovered
    assert recovered["last_error_code"] is None, recovered
    rows = _operation_rows(Store(store.db_path), "export")
    failure_rows = [row for row in rows if row["state"] == "failed"]
    succeeded_rows = [row for row in rows if row["state"] == "succeeded"]
    assert len(failure_rows) == 1 and succeeded_rows, rows
    assert failure_rows[0]["recovered_by_job_id"] == succeeded_rows[-1]["job_id"], rows
    assert failure_rows[0]["recovered_at"] is not None, failure_rows


def test_AC52_upgraded_old_schema_invalid_relation_cannot_hide_failure(
    store, tmp_path, monkeypatch
):
    """After the compatibility migration an invalid relation (self-reference) is
    still not a recovery: the upgraded database keeps the failure visible, a
    later success does not overwrite the corrupt association, and repairing the
    association restores normal recovery."""
    record(store, display_text="端口4200", value=4200)
    failed = _fail_export(store, tmp_path, monkeypatch, "failed-export")
    _drop_recovery_columns(store.db_path)
    migrated = Store(store.db_path)

    _update_job(
        migrated,
        failed["job_id"],
        recovered_by_job_id=failed["job_id"],
        recovered_at=DUE_NOW,
    )
    upgraded = _pipelines(Store(store.db_path))["export"]
    assert upgraded["failed"] == 1, upgraded
    assert upgraded["state"] == "degraded", upgraded
    assert upgraded["last_error_code"] == "EXPORT_FAILED", upgraded

    # A genuine later success must not silently turn the corrupt association
    # green; the recorded failure remains visible until the ledger is repaired.
    _succeed_export(migrated, tmp_path, "after-corrupt-upgrade")
    still_visible = _pipelines(Store(store.db_path))["export"]
    assert still_visible["failed"] == 1, still_visible
    assert still_visible["state"] == "degraded", still_visible
    assert still_visible["last_error_code"] == "EXPORT_FAILED", still_visible

    # Repairing the invalid association lets a real same-operation success
    # recover the recorded failure normally.
    _update_job(migrated, failed["job_id"], recovered_by_job_id=None, recovered_at=None)
    _succeed_export(migrated, tmp_path, "after-repair")
    recovered = _pipelines(Store(store.db_path))["export"]
    assert recovered["failed"] == 0 and recovered["state"] == "idle", recovered
    assert recovered["last_error_code"] is None, recovered


# ---------------------------------------------------------------------------
# Preserved control: the original real concurrent-export counterexample
# ---------------------------------------------------------------------------


def test_AC52_real_concurrent_export_original_counterexample_stays_visible_and_importable(
    store, tmp_path, monkeypatch
):
    """A success that completed before a late failure cannot hide it, and the
    snapshot of that unrecovered failure imports as still visible."""
    record(store, display_text="端口4200", value=4200)
    late_failure_pending = threading.Event()
    early_success_finished = threading.Event()
    errors: list[BaseException] = []
    original = exporting._write_jsonl

    def controlled(path, rows):
        if threading.current_thread().name == "ac52-fix5-late-failure":
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

    thread = threading.Thread(target=late_failure, name="ac52-fix5-late-failure")
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
    failed_rows = [row for row in _operation_rows(store, "export") if row["state"] == "failed"]
    assert len(failed_rows) == 1, failed_rows
    assert failed_rows[0]["recovered_by_job_id"] is None, failed_rows

    # The snapshot is read while the failure is still unrecovered; the snapshot
    # export's own later success only changes the live database, never the
    # already-written snapshot.  Importing it must keep the failure visible.
    monkeypatch.undo()
    snapshot = tmp_path / "snapshot"
    export_snapshot(store, snapshot)
    report = import_snapshot(snapshot, tmp_path / "restored")
    assert report["referential_integrity"] == "ok", report
    restored = Store(report["target_db"])
    after = _pipelines(restored)["export"]
    assert after["failed"] == 1, after
    assert after["state"] == "degraded", after
    assert after["last_error_code"] == "REDACTION_REQUIRED", after
    restored_rows = _operation_rows(restored, "export")
    restored_failed = [row for row in restored_rows if row["state"] == "failed"]
    assert len(restored_failed) == 1, restored_rows
    assert restored_failed[0]["recovered_by_job_id"] is None, restored_rows
    _assert_status_contract(restored)
