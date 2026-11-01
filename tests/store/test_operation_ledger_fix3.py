"""AC52-R3: real product operations update their own durable status ledger.

Migrated from the Codex round-3 counterexample
``test_AC52_actual_export_updates_its_pipeline_ledger``: a real
``export_snapshot`` succeeded but ``status(store)`` still reported
``export.state=idle`` and ``last_success_at=None`` because only hand-made
generic jobs were covered.  The fix adds shared ``record_operation_start`` /
``record_operation_finish`` helpers and wires the real synchronous entry points
(export, retention sweep/cleanup; projection wiring is added by the Lead in
``claims.py``) to them:

* pending is visible while the operation runs;
* success updates ``last_success_at``;
* a real injected failure records ``failed`` + a visible ``last_error_code``;
* another operation's success never clears an unrecovered failure (F16);
* a later successful run of the *same* operation recovers its earlier failure;
* the frozen ``Status`` schema and the exported snapshot's jobs table stay
  consistent (an export does not export its own in-flight ledger row).

Only synthetic pytest ``tmp_path`` data is used; no network, no real host data.
"""

from __future__ import annotations

import json

import pytest

import knowledge_weaver.lifecycle.retention as retention
import knowledge_weaver.portability.export as exporting
import knowledge_weaver.store.claims as claims_module
from knowledge_weaver import contracts
from knowledge_weaver.lifecycle import register_lifecycle, sweep
from knowledge_weaver.portability import export_snapshot, import_snapshot
from knowledge_weaver.portability.export import ExportRefused
from knowledge_weaver.store import Store, rebuild_projection, status as status_view
from knowledge_weaver.store.status import OPERATION_LEDGER_VERSION

from ..state.conftest import record
from ..store.conftest import make_batch, make_event


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


def _due_event_part(store):
    event = make_event(native_id="message:ac52-fix3", text="已完成任务的过程详情")
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


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def test_AC52_real_export_updates_export_ledger(store, tmp_path):
    """Codex counterexample: real export success is visible as last_success."""
    record(store, display_text="端口4200", value=4200)
    result = export_snapshot(store, tmp_path / "actual-export")
    assert result["target"]
    pipeline = _pipelines(store)["export"]
    assert pipeline["last_success_at"] is not None, pipeline
    assert pipeline["pending"] == 0 and pipeline["failed"] == 0, pipeline
    assert pipeline["state"] == "idle" and pipeline["last_error_code"] is None, pipeline
    _assert_status_contract(store)
    # A fresh session on the same database sees the same durable ledger.
    restarted = _pipelines(Store(store.db_path))["export"]
    assert restarted["last_success_at"] == pipeline["last_success_at"], restarted


def test_AC52_export_is_pending_while_it_runs(store, tmp_path, monkeypatch):
    record(store, display_text="端口4200", value=4200)
    observed: list[dict] = []
    original = exporting._write_jsonl

    def observe(path, rows):
        if not observed:
            observed.append(_pipelines(store)["export"])
        return original(path, rows)

    monkeypatch.setattr(exporting, "_write_jsonl", observe)
    export_snapshot(store, tmp_path / "pending-export")
    assert observed, "the export must still be running when the state is read"
    assert observed[0]["pending"] == 1, observed[0]
    assert observed[0]["state"] == "idle", observed[0]
    finished = _pipelines(store)["export"]
    assert finished["pending"] == 0 and finished["last_success_at"] is not None, finished


def test_AC52_export_failure_records_failed_and_visible_error(store, tmp_path, monkeypatch):
    record(store, display_text="端口4200", value=4200)

    def refuse(table, rows):
        raise ExportRefused("REDACTION_REQUIRED", "synthetic export refusal")

    monkeypatch.setattr(exporting, "_assert_rows_secret_free", refuse)
    with pytest.raises(ExportRefused):
        export_snapshot(store, tmp_path / "failed-export")
    failed = _pipelines(store)["export"]
    assert failed["failed"] == 1, failed
    assert failed["state"] == "degraded", failed
    assert failed["last_error_code"] == "REDACTION_REQUIRED", failed
    assert failed["pending"] == 0 and failed["last_success_at"] is None, failed
    # The failed snapshot was not published.
    assert not (tmp_path / "failed-export").exists()
    _assert_status_contract(store)

    # A later real export of the same operation recovers the visible failure.
    monkeypatch.undo()
    export_snapshot(store, tmp_path / "recovered-export")
    recovered = _pipelines(store)["export"]
    assert recovered["failed"] == 0 and recovered["state"] == "idle", recovered
    assert recovered["last_error_code"] is None, recovered
    assert recovered["last_success_at"] is not None, recovered


def test_AC52_export_does_not_export_its_own_pending_ledger_row(store, tmp_path):
    record(store, display_text="端口4200", value=4200)
    first = tmp_path / "first"
    export_snapshot(store, first)
    first_jobs = [
        json.loads(line)
        for line in (first / "records" / "jobs.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert not [row for row in first_jobs if row.get("model_version") == OPERATION_LEDGER_VERSION], (
        "the in-flight export row must not be written into its own snapshot"
    )
    # The previous, completed export appears as ordinary operation history.
    second = tmp_path / "second"
    export_snapshot(store, second)
    second_jobs = [
        json.loads(line)
        for line in (second / "records" / "jobs.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    ledger_rows = [
        row for row in second_jobs if row.get("model_version") == OPERATION_LEDGER_VERSION
    ]
    assert len(ledger_rows) == 1 and ledger_rows[0]["state"] == "succeeded", ledger_rows


# ---------------------------------------------------------------------------
# Cleanup (retention sweep, the real CLI cleanup entry point)
# ---------------------------------------------------------------------------


def test_AC52_real_cleanup_updates_cleanup_ledger_and_shows_pending(store, monkeypatch):
    _due_event_part(store)
    observed: list[dict] = []
    original = retention._tombstone

    def observe(*args, **kwargs):
        if not observed:
            observed.append(_pipelines(store)["cleanup"])
        return original(*args, **kwargs)

    monkeypatch.setattr(retention, "_tombstone", observe)
    result = sweep(store, now="2026-10-31T00:00:00Z")
    assert result["event_parts"] == 1, result
    assert observed and observed[0]["pending"] == 1, observed
    clean = _pipelines(store)["cleanup"]
    assert clean["pending"] == 0 and clean["failed"] == 0, clean
    assert clean["last_success_at"] is not None and clean["state"] == "idle", clean
    _assert_status_contract(store)


def test_AC52_cleanup_failure_records_failed_and_recovers(store, monkeypatch):
    _due_event_part(store)
    original = retention._tombstone
    fired = {"done": False}

    def boom(*args, **kwargs):
        if not fired["done"]:
            fired["done"] = True
            raise RuntimeError("synthetic cleanup failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(retention, "_tombstone", boom)
    with pytest.raises(RuntimeError):
        sweep(store, now="2026-10-31T00:00:00Z")
    failed = _pipelines(store)["cleanup"]
    assert failed["failed"] == 1 and failed["state"] == "degraded", failed
    assert failed["last_error_code"] == "CLEANUP_FAILED", failed
    assert failed["pending"] == 0 and failed["last_success_at"] is None, failed

    # The real operation runs again and its success recovers the failure.
    result = sweep(store, now="2026-10-31T00:00:00Z")
    assert result["event_parts"] == 1, result
    recovered = _pipelines(store)["cleanup"]
    assert recovered["failed"] == 0 and recovered["state"] == "idle", recovered
    assert recovered["last_error_code"] is None and recovered["last_success_at"], recovered


def test_AC52_other_operation_success_does_not_clear_cleanup_failure(store, tmp_path, monkeypatch):
    """Clause: 空闲/他工序成功不清除未恢复失败."""
    _due_event_part(store)
    original = retention._tombstone

    def boom(*args, **kwargs):
        raise RuntimeError("synthetic cleanup failure")

    monkeypatch.setattr(retention, "_tombstone", boom)
    with pytest.raises(RuntimeError):
        sweep(store, now="2026-10-31T00:00:00Z")
    failed_before = _pipelines(store)["cleanup"]
    assert failed_before["failed"] == 1 and failed_before["state"] == "degraded", failed_before

    # A different real operation succeeds, and plain idle status reads happen.
    record(store, display_text="端口4200", value=4200)
    export_snapshot(store, tmp_path / "other-op-export")
    for _ in range(2):
        status_view(store)
    after = _pipelines(store)
    assert after["cleanup"]["failed"] == failed_before["failed"], after["cleanup"]
    assert after["cleanup"]["state"] == "degraded", after["cleanup"]
    assert after["cleanup"]["last_error_code"] == "CLEANUP_FAILED", after["cleanup"]
    assert after["export"]["last_success_at"] is not None, after["export"]


# ---------------------------------------------------------------------------
# Projection rebuild: the real entry point records its own run
# ---------------------------------------------------------------------------


def test_AC52_projection_rebuild_is_recorded_by_the_operation_ledger(store, monkeypatch):
    """The real ``rebuild_projection`` entry records its own pending/success."""
    record(store, display_text="端口4200", value=4200)
    observed: list[dict] = []
    original = claims_module.recompute_claim

    def observe(conn, claim_id):
        if not observed:
            observed.append(_pipelines(store)["project"])
        return original(conn, claim_id)

    monkeypatch.setattr(claims_module, "recompute_claim", observe)
    rebuilt = rebuild_projection(store)
    assert rebuilt == 1, rebuilt
    assert observed, "the projection must still be running when the state is read"
    assert observed[0]["pending"] == 1, observed[0]
    project = _pipelines(store)["project"]
    assert project["pending"] == 0 and project["failed"] == 0, project
    assert project["last_success_at"] is not None, project
    _assert_status_contract(store)


def test_AC52_projection_failure_is_recorded_with_visible_error(store, monkeypatch):
    """A real projection failure is recorded once, by the real entry point."""
    record(store, display_text="端口4200", value=4200)

    def boom(conn, claim_id):
        raise RuntimeError("synthetic projection failure")

    monkeypatch.setattr(claims_module, "recompute_claim", boom)
    with pytest.raises(RuntimeError):
        rebuild_projection(store)
    failed = _pipelines(store)["project"]
    assert failed["failed"] == 1 and failed["state"] == "degraded", failed
    assert failed["last_error_code"] == "PROJECT_FAILED", failed
    assert failed["pending"] == 0 and failed["last_success_at"] is None, failed


# ---------------------------------------------------------------------------
# Restore keeps the operation diagnostics
# ---------------------------------------------------------------------------


def test_AC52_operation_ledger_survives_snapshot_restore(store, tmp_path):
    record(store, display_text="端口4200", value=4200)
    # The first completed export is ordinary operation history; the second
    # snapshot contains it (but never its own in-flight ledger row).
    export_snapshot(store, tmp_path / "first")
    snapshot = tmp_path / "snapshot"
    export_snapshot(store, snapshot)
    report = import_snapshot(snapshot, tmp_path / "restored")
    restored = Store(report["target_db"])
    pipeline = _pipelines(restored)["export"]
    assert pipeline["last_success_at"] is not None, pipeline
    assert pipeline["pending"] == 0 and pipeline["failed"] == 0, pipeline
