#!/usr/bin/env python3
"""Focused tests for the DMA runtime-status consumer and KW report contract."""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import dma_status
import kw_health_check


class TestDmaStatusValidator(unittest.TestCase):
    def setUp(self) -> None:
        self.now = datetime(2026, 9, 9, 4, 0, tzinfo=timezone.utc)
        self.tmpdir = Path(tempfile.mkdtemp(prefix="kw-status-test-"))
        self.status_path = self.tmpdir / "runtime_status.json"
        self.log_path = self.tmpdir / "dma.log"
        self.checkpoint_path = self.tmpdir / "checkpoint.json"
        self.memory_dir = self.tmpdir / "memory"
        self.memory_dir.mkdir()
        self._saved = {
            "log": kw_health_check.DMA_LOG_PATH,
            "checkpoint": kw_health_check.DMA_CHECKPOINT_PATH,
            "memory": kw_health_check.MEMORY_DIR,
            "status": kw_health_check.DMA_STATUS_PATH,
        }
        kw_health_check.DMA_LOG_PATH = str(self.log_path)
        kw_health_check.DMA_CHECKPOINT_PATH = str(self.checkpoint_path)
        kw_health_check.MEMORY_DIR = str(self.memory_dir)
        kw_health_check.DMA_STATUS_PATH = str(self.status_path)
        os.environ.pop("KW_DMA_STATUS_PATH", None)

    def tearDown(self) -> None:
        kw_health_check.DMA_LOG_PATH = self._saved["log"]
        kw_health_check.DMA_CHECKPOINT_PATH = self._saved["checkpoint"]
        kw_health_check.MEMORY_DIR = self._saved["memory"]
        kw_health_check.DMA_STATUS_PATH = self._saved["status"]
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def payload(
        self,
        *,
        outcome: str = "idle",
        observed: datetime | None = None,
        pending_count: int | None = 0,
        pending_oldest: datetime | None = None,
        reconcile_count: int | None = 0,
        reconcile_oldest: datetime | None = None,
        summary_failures: int = 0,
        status_errors: list[str] | None = None,
    ) -> dict:
        seen = observed or self.now - timedelta(minutes=5)
        started = seen - timedelta(minutes=1)
        finished = None if outcome == "running" else seen
        timestamp = lambda value: value.isoformat() if value is not None else None
        return {
            "schema_version": dma_status.SCHEMA_VERSION,
            "run_id": "dma-test-run",
            "started_at": timestamp(started),
            "observed_at": timestamp(seen),
            "finished_at": timestamp(finished),
            "outcome": outcome,
            "reason": "test",
            "last_completed_at": timestamp(finished),
            "last_archived_at": timestamp(finished if outcome == "archived" else None),
            "pending_count": pending_count,
            "oldest_pending_at": timestamp(pending_oldest),
            "pending_reconcile_count": reconcile_count,
            "oldest_reconcile_at": timestamp(reconcile_oldest),
            "checkpoint_progress_at": timestamp(seen),
            "failures": {
                "storage": {"consecutive": 0, "last_failed_at": None, "last_recovered_at": None},
                "summary": {
                    "consecutive": summary_failures,
                    "last_failed_at": timestamp(seen - timedelta(hours=1)) if summary_failures else None,
                    "last_recovered_at": None,
                },
                "archive": {"consecutive": 0, "last_failed_at": None, "last_recovered_at": None},
            },
            "status_errors": status_errors or [],
        }

    def write_status(self, payload: dict) -> None:
        self.status_path.write_text(json.dumps(payload), encoding="utf-8")

    def test_valid_idle_without_memory_is_healthy(self) -> None:
        self.write_status(self.payload())
        result = kw_health_check.check_dma_health(now=self.now, status_path=str(self.status_path))
        self.assertEqual(result["status"], "ok")
        self.assertTrue(result["evidence_complete"])
        self.assertFalse(result["memory_file_exists"])
        self.assertIn("没有待归档消息", result["message"])

    def test_historical_log_errors_are_information_only(self) -> None:
        self.write_status(self.payload(outcome="archived"))
        line = f"[{self.now.strftime('%Y-%m-%d %H:%M:%S')}] [ERROR] old failure"
        self.log_path.write_text("\n".join([line] * 11) + "\n", encoding="utf-8")
        result = kw_health_check.check_dma_health(now=self.now, status_path=str(self.status_path))
        self.assertEqual(result["error_count_24h"], 11)
        self.assertEqual(result["status"], "ok")
        self.assertIn("历史信息", result["message"])

    def test_fresh_running_is_info_and_incomplete(self) -> None:
        self.write_status(self.payload(outcome="running"))
        result = kw_health_check.check_dma_health(now=self.now, status_path=str(self.status_path))
        self.assertEqual(result["status"], "info")
        self.assertEqual(result["runtime_state"], "running")
        self.assertFalse(result["evidence_complete"])

    def test_missing_status_is_rollout_warning(self) -> None:
        result = kw_health_check.check_dma_health(now=self.now, status_path=str(self.status_path))
        self.assertEqual(result["status"], "warning")
        self.assertEqual(result["runtime_state"], "unknown")
        self.assertEqual(result["runtime_status_error"], "dma_status_missing")
        self.assertFalse(result["evidence_complete"])

    def test_malformed_status_is_checker_alert(self) -> None:
        self.status_path.write_text("{\"schema_version\": \"wrong\"}", encoding="utf-8")
        result = kw_health_check.check_dma_health(now=self.now, status_path=str(self.status_path))
        self.assertEqual(result["status"], "alert")
        self.assertEqual(result["runtime_status_error"], "dma_status_malformed")
        self.assertFalse(result["evidence_complete"])

    def test_stale_valid_status_is_service_alert(self) -> None:
        self.write_status(self.payload(observed=self.now - timedelta(hours=2)))
        result = kw_health_check.check_dma_health(
            now=self.now, status_path=str(self.status_path), status_timeout_seconds=90 * 60
        )
        self.assertEqual(result["status"], "alert")
        self.assertEqual(result["runtime_status_error"], "dma_status_stale")
        self.assertFalse(result["evidence_complete"])
        self.assertEqual(result["status_errors"], [])

    def test_stale_status_errors_are_checker_evidence_failures(self) -> None:
        self.write_status(
            self.payload(
                observed=self.now - timedelta(hours=2),
                status_errors=["status_unreadable_during_run"],
            )
        )
        result = kw_health_check.check_dma_health(
            now=self.now, status_path=str(self.status_path), status_timeout_seconds=90 * 60
        )
        self.assertEqual(result["runtime_status_error"], "dma_status_stale")
        self.assertEqual(result["runtime_status_errors"], ["status_unreadable_during_run"])
        self.assertEqual(result["status_errors"], ["status_unreadable_during_run"])

    def test_failed_terminal_status_with_unknown_pending_is_valid_alert(self) -> None:
        self.write_status(
            self.payload(outcome="failed", pending_count=None, summary_failures=1)
        )
        result = kw_health_check.check_dma_health(now=self.now, status_path=str(self.status_path))
        self.assertEqual(result["status"], "alert")
        self.assertEqual(result["runtime_status_error"], None)
        self.assertFalse(result["evidence_complete"])

    def test_terminal_unknown_pending_cannot_look_healthy(self) -> None:
        self.write_status(self.payload(pending_count=None))
        result = kw_health_check.check_dma_health(now=self.now, status_path=str(self.status_path))
        self.assertEqual(result["status"], "warning")
        self.assertFalse(result["evidence_complete"])
        self.assertIn("dma_pending_count_unknown", result["warning_codes"])

    def test_status_env_override(self) -> None:
        self.write_status(self.payload())
        with patch.dict(os.environ, {"KW_DMA_STATUS_PATH": str(self.status_path)}):
            result = kw_health_check.check_dma_health(now=self.now)
        self.assertEqual(result["runtime_outcome"], "idle")

    def test_validator_rejects_timezone_less_and_future_values(self) -> None:
        payload = self.payload()
        payload["observed_at"] = payload["observed_at"].replace("+00:00", "")
        with self.assertRaises(dma_status.DmaStatusMalformedError):
            dma_status.validate_status(payload, now=self.now)
        payload = self.payload(observed=self.now + timedelta(minutes=10))
        with self.assertRaises(dma_status.DmaStatusMalformedError):
            dma_status.validate_status(payload, now=self.now)

    def test_validator_rejects_running_finished_timestamp(self) -> None:
        payload = self.payload(outcome="running")
        payload["finished_at"] = payload["observed_at"]
        with self.assertRaises(dma_status.DmaStatusMalformedError):
            dma_status.validate_status(payload, now=self.now)

    def test_validator_keeps_failure_recovery_sequence(self) -> None:
        payload = self.payload(outcome="failed", summary_failures=1)
        observed = datetime.fromisoformat(payload["observed_at"])
        payload["failures"]["summary"]["last_recovered_at"] = (
            observed - timedelta(hours=1)
        ).isoformat()
        dma_status.validate_status(payload, now=self.now)

    def test_validator_rejects_recovery_before_latest_resolved_failure(self) -> None:
        payload = self.payload(outcome="idle")
        observed = datetime.fromisoformat(payload["observed_at"])
        payload["failures"]["summary"] = {
            "consecutive": 0,
            "last_failed_at": observed.isoformat(),
            "last_recovered_at": (observed - timedelta(hours=1)).isoformat(),
        }
        with self.assertRaises(dma_status.DmaStatusMalformedError):
            dma_status.validate_status(payload, now=self.now)

    def test_validator_rejects_cumulative_timestamp_after_snapshot(self) -> None:
        payload = self.payload()
        observed = datetime.fromisoformat(payload["observed_at"])
        payload["last_completed_at"] = (observed + timedelta(seconds=1)).isoformat()
        with self.assertRaises(dma_status.DmaStatusMalformedError):
            dma_status.validate_status(payload, now=self.now)

    def test_validator_rejects_non_string_outcome(self) -> None:
        payload = self.payload()
        payload["outcome"] = []
        with self.assertRaises(dma_status.DmaStatusMalformedError):
            dma_status.validate_status(payload, now=self.now)

    def test_validator_rejects_source_timestamp_after_snapshot(self) -> None:
        payload = self.payload(pending_count=1)
        observed = datetime.fromisoformat(payload["observed_at"])
        # Keep it before the consumer's ``now`` so this specifically checks
        # snapshot ordering rather than the separate future-clock guard.
        payload["oldest_pending_at"] = (observed + timedelta(minutes=1)).isoformat()
        with self.assertRaises(dma_status.DmaStatusMalformedError):
            dma_status.validate_status(payload, now=self.now)

    def test_terminal_unknown_reconcile_count_cannot_look_healthy(self) -> None:
        self.write_status(
            self.payload(pending_count=0, reconcile_count=None)
        )
        result = kw_health_check.check_dma_health(now=self.now, status_path=str(self.status_path))
        self.assertEqual(result["status"], "warning")
        self.assertFalse(result["evidence_complete"])
        self.assertIn("dma_reconcile_count_unknown", result["warning_codes"])


class TestHealthReportContract(unittest.TestCase):
    """Use an isolated SQLite fixture to test report publication semantics."""

    def setUp(self) -> None:
        self.tmpdir = Path(tempfile.mkdtemp(prefix="kw-report-test-"))
        self.db_path = self.tmpdir / "knowledge.db"
        self.reports_dir = self.tmpdir / "reports"
        self.status_path = self.tmpdir / "runtime_status.json"
        self.now = datetime(2026, 9, 9, 4, 0, tzinfo=timezone.utc)
        self._saved = {
            "db": kw_health_check.DB_PATH,
            "reports": kw_health_check.REPORTS_DIR,
            "status": kw_health_check.DMA_STATUS_PATH,
            "log": kw_health_check.DMA_LOG_PATH,
            "checkpoint": kw_health_check.DMA_CHECKPOINT_PATH,
            "memory": kw_health_check.MEMORY_DIR,
        }
        kw_health_check.DB_PATH = str(self.db_path)
        kw_health_check.REPORTS_DIR = str(self.reports_dir)
        kw_health_check.DMA_STATUS_PATH = str(self.status_path)
        kw_health_check.DMA_LOG_PATH = str(self.tmpdir / "missing.log")
        kw_health_check.DMA_CHECKPOINT_PATH = str(self.tmpdir / "missing.cp")
        kw_health_check.MEMORY_DIR = str(self.tmpdir / "memory")
        self._create_db()
        self.status_path.write_text(json.dumps(self._status_payload()), encoding="utf-8")

    def tearDown(self) -> None:
        for key, value in self._saved.items():
            setattr(
                kw_health_check,
                {"db": "DB_PATH", "reports": "REPORTS_DIR", "status": "DMA_STATUS_PATH",
                 "log": "DMA_LOG_PATH", "checkpoint": "DMA_CHECKPOINT_PATH", "memory": "MEMORY_DIR"}[key],
                value,
            )
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _status_payload(self) -> dict:
        seen = self.now - timedelta(minutes=5)
        stamp = seen.isoformat()
        return {
            "schema_version": dma_status.SCHEMA_VERSION,
            "run_id": "report-test-run",
            "started_at": (seen - timedelta(minutes=1)).isoformat(),
            "observed_at": stamp,
            "finished_at": stamp,
            "outcome": "idle",
            "reason": "test",
            "last_completed_at": stamp,
            "last_archived_at": None,
            "pending_count": 0,
            "oldest_pending_at": None,
            "pending_reconcile_count": 0,
            "oldest_reconcile_at": None,
            "checkpoint_progress_at": stamp,
            "failures": {
                name: {"consecutive": 0, "last_failed_at": None, "last_recovered_at": None}
                for name in dma_status.FAILURE_OPERATIONS
            },
            "status_errors": [],
        }

    def _create_db(self) -> None:
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.executescript(
            """
            CREATE TABLE entities (
                id TEXT PRIMARY KEY, type TEXT NOT NULL, name TEXT NOT NULL,
                summary TEXT NOT NULL, importance REAL DEFAULT 0.0,
                first_seen TEXT NOT NULL, last_seen TEXT NOT NULL,
                day_count INTEGER DEFAULT 1, source_lines TEXT DEFAULT '[]',
                metadata TEXT DEFAULT '{}', created_at TEXT DEFAULT (datetime('now')),
                updated_at TEXT DEFAULT (datetime('now'))
            );
            CREATE TABLE relations (
                id TEXT PRIMARY KEY, from_entity TEXT NOT NULL, to_entity TEXT NOT NULL,
                rel_type TEXT NOT NULL, weight REAL DEFAULT 0.5, evidence TEXT NOT NULL,
                created_at TEXT DEFAULT (datetime('now'))
            );
            CREATE TABLE embedding_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE entity_vectors (entity_id TEXT PRIMARY KEY, embedding TEXT NOT NULL);
            CREATE TABLE merge_review (
                id INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT DEFAULT 'merge',
                new_entity_id TEXT NOT NULL, candidate_id TEXT, entity_type TEXT,
                score REAL, reason TEXT, status TEXT DEFAULT 'pending',
                created_at TEXT DEFAULT (datetime('now')), resolved_at TEXT
            );
            CREATE TABLE daily_manifest (
                date TEXT NOT NULL, source TEXT DEFAULT 'default', file_path TEXT NOT NULL,
                file_hash TEXT NOT NULL, entity_count INTEGER DEFAULT 0,
                processed_at TEXT NOT NULL, status TEXT DEFAULT 'ok',
                PRIMARY KEY (date, source)
            );
            INSERT INTO embedding_meta VALUES ('model', 'test');
            INSERT INTO embedding_meta VALUES ('dimension', '2');
            INSERT INTO entities VALUES ('e1','tech','SQLite','test',0.5,'2026-09-01','2026-09-09',1,'[]','{}',datetime('now'),datetime('now'));
            INSERT INTO entity_vectors VALUES ('e1','[0.1,0.2]');
            INSERT INTO daily_manifest VALUES ('2026-09-09','default','/tmp/m.md','hash',1,datetime('now'),'ok');
            """
        )
        conn.commit()
        conn.close()

    def test_dry_run_does_not_create_reports_directory(self) -> None:
        self.assertFalse(self.reports_dir.exists())
        report = json.loads(
            kw_health_check.run_health_check(dry_run=True, run_id="kw-run-1", now=self.now)
        )
        self.assertFalse(self.reports_dir.exists())
        self.assertEqual(report["schema_version"], "kw-health-report-v1")
        self.assertEqual(report["run_id"], "kw-run-1")
        self.assertTrue(report["checks_complete"])
        self.assertIn("day_count 不具备跨日统计语义", report["checks"]["cross_day_aggregation"]["message"])
        self.assertEqual(report["checks"]["cross_day_aggregation"]["status"], "info")

    def test_output_path_is_invocation_owned_and_atomic(self) -> None:
        output = self.tmpdir / "one-run" / "report.json"
        report = json.loads(
            kw_health_check.run_health_check(
                dry_run=False, run_id="kw-run-2", output_path=str(output), now=self.now
            )
        )
        self.assertTrue(output.is_file())
        disk = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(disk, report)
        self.assertEqual(report["report_date"], "2026-09-09")
        self.assertTrue(datetime.fromisoformat(report["generated_at"]).tzinfo)

    def test_cli_status_errors_are_checker_failure(self) -> None:
        # The subprocess uses the wall clock, so make this fixture fresh for
        # its independent process rather than the deterministic report clock.
        self.now = datetime.now(timezone.utc)
        payload = self._status_payload()
        payload["status_errors"] = ["reconcile_snapshot_unknown"]
        self.status_path.write_text(json.dumps(payload), encoding="utf-8")
        output = self.tmpdir / "cli-report.json"
        environment = dict(
            os.environ,
            KNOWLEDGE_WEAVER_DB_PATH=str(self.db_path),
            KW_HEALTH_REPORTS_DIR=str(self.reports_dir),
            KW_DMA_STATUS_PATH=str(self.status_path),
            KW_DMA_LOG_PATH=str(self.tmpdir / "missing.log"),
            KW_DMA_CHECKPOINT_PATH=str(self.tmpdir / "missing.cp"),
            KNOWLEDGE_WEAVER_MEMORY_DIR=str(self.tmpdir / "memory"),
        )
        result = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name("kw_health_check.py")),
                "--run-id",
                "cli-run",
                "--output",
                str(output),
            ],
            env=environment,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=30,
        )
        self.assertEqual(result.returncode, 1)
        report = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(report["run_id"], "cli-run")
        self.assertFalse(report["checks_complete"])
        self.assertIn("dma_status_observation", report["checker_errors"])


if __name__ == "__main__":
    unittest.main()
