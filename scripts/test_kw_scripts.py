#!/usr/bin/env python3
"""
Unit tests for Knowledge Weaver health check scripts.

Run with:
    python3 -m unittest test_kw_scripts.py
"""

import json
import os
import sqlite3
import shutil
import sys
import tempfile
import time
import unittest

# Ensure the scripts directory is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import kw_health_check
import kw_weekly_trend


class TestHelpers(unittest.TestCase):
    """Pure helper function tests (no DB needed)."""

    def test_calc_entropy_empty(self):
        self.assertEqual(kw_weekly_trend.calc_entropy({}), 0.0)

    def test_calc_entropy_uniform(self):
        e = kw_weekly_trend.calc_entropy({"a": 1, "b": 1, "c": 1, "d": 1})
        self.assertAlmostEqual(e, 2.0, places=2)

    def test_calc_entropy_single_type(self):
        e = kw_weekly_trend.calc_entropy({"a": 10})
        self.assertEqual(e, 0.0)

    def test_calc_entropy_skewed(self):
        e = kw_weekly_trend.calc_entropy({"a": 100, "b": 1})
        self.assertLess(e, 0.2)

    def test_compare_with_previous_none(self):
        result = kw_health_check.compare_with_previous(100, None)
        self.assertEqual(result["trend"], "N/A")
        self.assertEqual(result["delta"], 0)

    def test_compare_with_previous_stable(self):
        prev = {"checks": {"entity_total": {"total_entities": 100}}}
        result = kw_health_check.compare_with_previous(105, prev)
        self.assertEqual(result["trend"], "stable")
        self.assertEqual(result["delta"], 5)
        self.assertEqual(result["delta_pct"], 5.0)

    def test_compare_with_previous_growing(self):
        prev = {"checks": {"entity_total": {"total_entities": 100}}}
        result = kw_health_check.compare_with_previous(115, prev)
        self.assertEqual(result["trend"], "growing")

    def test_compare_with_previous_shrinking(self):
        prev = {"checks": {"entity_total": {"total_entities": 100}}}
        result = kw_health_check.compare_with_previous(85, prev)
        self.assertEqual(result["trend"], "shrinking")


class TestHealthCheckDB(unittest.TestCase):
    """Integration tests using a temporary DB seeded with test data."""

    @classmethod
    def setUpClass(cls):
        """Create a temporary on-disk DB with controlled test data."""
        cls.tmpdir = tempfile.mkdtemp(prefix="kw_test_")
        cls.db_path = os.path.join(cls.tmpdir, "test_knowledge.db")

        conn = sqlite3.connect(cls.db_path)
        cur = conn.cursor()

        # ── Schema ──
        cur.execute("""CREATE TABLE entities (
            id TEXT PRIMARY KEY, type TEXT NOT NULL, name TEXT NOT NULL,
            summary TEXT NOT NULL, importance REAL DEFAULT 0.0,
            first_seen TEXT NOT NULL, last_seen TEXT NOT NULL,
            day_count INTEGER DEFAULT 1, source_lines TEXT DEFAULT '[]',
            metadata TEXT DEFAULT '{}', created_at TEXT DEFAULT (datetime('now')),
            updated_at TEXT DEFAULT (datetime('now')))""")

        cur.execute("""CREATE TABLE relations (
            id TEXT PRIMARY KEY, from_entity TEXT NOT NULL,
            to_entity TEXT NOT NULL, rel_type TEXT NOT NULL,
            weight REAL DEFAULT 0.5, evidence TEXT NOT NULL,
            created_at TEXT DEFAULT (datetime('now')))""")

        cur.execute("""CREATE TABLE embedding_meta (
            key TEXT PRIMARY KEY, value TEXT NOT NULL)""")

        cur.execute("""CREATE TABLE entity_vectors (
            entity_id TEXT PRIMARY KEY, embedding TEXT NOT NULL)""")

        cur.execute("""CREATE TABLE merge_review (
            id INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT DEFAULT 'merge',
            new_entity_id TEXT NOT NULL, candidate_id TEXT,
            entity_type TEXT, score REAL, reason TEXT,
            status TEXT DEFAULT 'pending',
            created_at TEXT DEFAULT (datetime('now')),
            resolved_at TEXT)""")

        cur.execute("""CREATE TABLE daily_manifest (
            date TEXT NOT NULL, source TEXT DEFAULT 'default',
            file_path TEXT NOT NULL, file_hash TEXT NOT NULL,
            entity_count INTEGER DEFAULT 0, processed_at TEXT NOT NULL,
            status TEXT DEFAULT 'ok',
            PRIMARY KEY (date, source))""")

        # ── Seed entities ──
        # 10 entities: types tech/decision/risk/preference/task/idea/fact/project
        # e3, e7 = orphans (no relations)
        # e3, e4, e9 = day_count=2
        # "Python" appears as tech (e1) + decision (e2) → cross-type conflict
        entities = [
            ("e1",  "tech",       "Python",        "S1", 0.5, "2026-05-01", "2026-06-01", 5),
            ("e2",  "decision",   "Python",        "S2", 0.3, "2026-05-02", "2026-06-02", 7),  # cross-type conflict with e1
            ("e3",  "decision",   "GoLang",        "S3", 0.7, "2026-05-03", "2026-06-03", 2),  # orphan + day_count=2
            ("e4",  "risk",       "Security Risk", "S4", 0.9, "2026-05-04", "2026-06-04", 2),  # day_count=2
            ("e5",  "preference", "Dark Mode",     "S5", 0.4, "2026-05-05", "2026-06-05", 1),
            ("e6",  "task",       "Refactor X",    "S6", 0.6, "2026-05-06", "2026-06-06", 3),
            ("e7",  "idea",       "New Feature",   "S7", 0.2, "2026-05-07", "2026-06-07", 1),  # orphan
            ("e8",  "fact",       "Server IP",     "S8", 0.8, "2026-05-08", "2026-06-08", 1),
            ("e9",  "project",    "Project Alpha", "S9", 0.1, "2026-05-09", "2026-06-08", 2),  # day_count=2
            ("e10", "tech",       "SQLite",        "S10", 0.5, "2026-05-10", "2026-06-08", 1),
        ]
        for e in entities:
            cur.execute(
                "INSERT INTO entities VALUES (?,?,?,?,?,?,?,?,'[]','{}',datetime('now'),datetime('now'))",
                e,
            )

        # ── Seed relations ──
        # e3 and e7 have NO relations → orphans
        relations = [
            ("r1", "e1", "e2", "related_to", 0.5, "evidence"),
            ("r2", "e1", "e5", "related_to", 0.3, "evidence"),
            ("r3", "e4", "e6", "depends_on", 0.8, "evidence"),
            ("r4", "e6", "e9", "related_to", 0.6, "evidence"),
            ("r5", "e8", "e1", "documents", 0.4, "evidence"),
            ("r6", "e2", "e10", "related_to", 0.5, "evidence"),
        ]
        for r in relations:
            cur.execute(
                "INSERT INTO relations VALUES (?,?,?,?,?,?,datetime('now'))", r
            )

        # ── embedding_meta ──
        cur.execute("INSERT INTO embedding_meta VALUES ('model', 'bge-m3')")
        cur.execute("INSERT INTO embedding_meta VALUES ('dimension', '1024')")

        # ── entity_vectors (8 of 10 entities) ──
        for eid in ["e1", "e2", "e3", "e4", "e5", "e6", "e7", "e8"]:
            vec = json.dumps([0.1] * 1024)
            cur.execute("INSERT INTO entity_vectors VALUES (?,?)", (eid, vec))

        # ── merge_review: 2 pending, 1 merged ──
        cur.execute(
            "INSERT INTO merge_review (new_entity_id,candidate_id,status) VALUES ('e11','e1','pending')"
        )
        cur.execute(
            "INSERT INTO merge_review (new_entity_id,candidate_id,status) VALUES ('e12','e3','pending')"
        )
        cur.execute(
            "INSERT INTO merge_review (new_entity_id,candidate_id,status) VALUES ('e13','e5','merged')"
        )

        # Dangling relation for referential integrity testing
        cur.execute(
            "INSERT INTO relations VALUES ('r_dangle','e99','e1','related_to',0.5,'test',datetime('now'))",
        )

        # ── daily_manifest: 3 days ──
        for d, h, n in [
            ("2026-06-01", "hash1", 5),
            ("2026-06-02", "hash2", 3),
            ("2026-06-03", "hash3", 4),
        ]:
            cur.execute(
                "INSERT INTO daily_manifest VALUES (?, 'default', '/tmp/x.md', ?, ?, datetime('now'), 'ok')",
                (d, h, n),
            )

        conn.commit()
        conn.close()

        # Override module-level paths for testing
        cls._save = {
            "hp": kw_health_check.DB_PATH,
            "wp": kw_weekly_trend.DB_PATH,
            "hr": kw_health_check.REPORTS_DIR,
            "wr": kw_weekly_trend.REPORTS_DIR,
            "ww": kw_weekly_trend.WEEKLY_DIR,
            "hh": kw_weekly_trend.HISTORY_PATH,
        }
        kw_health_check.DB_PATH = cls.db_path
        kw_weekly_trend.DB_PATH = cls.db_path
        kw_health_check.REPORTS_DIR = os.path.join(cls.tmpdir, "health_reports")
        kw_weekly_trend.REPORTS_DIR = os.path.join(cls.tmpdir, "health_reports")
        kw_weekly_trend.WEEKLY_DIR = os.path.join(cls.tmpdir, "weekly_reports")
        kw_weekly_trend.HISTORY_PATH = os.path.join(cls.tmpdir, "health_history.jsonl")

    @classmethod
    def tearDownClass(cls):
        kw_health_check.DB_PATH = cls._save["hp"]
        kw_weekly_trend.DB_PATH = cls._save["wp"]
        kw_health_check.REPORTS_DIR = cls._save["hr"]
        kw_weekly_trend.REPORTS_DIR = cls._save["wr"]
        kw_weekly_trend.WEEKLY_DIR = cls._save["ww"]
        kw_weekly_trend.HISTORY_PATH = cls._save["hh"]
        shutil.rmtree(cls.tmpdir, ignore_errors=True)

    def setUp(self):
        for d in [
            kw_health_check.REPORTS_DIR,
            kw_weekly_trend.WEEKLY_DIR,
        ]:
            if os.path.isdir(d):
                for f in os.listdir(d):
                    os.remove(os.path.join(d, f))
        hp = kw_weekly_trend.HISTORY_PATH
        if os.path.isfile(hp):
            os.remove(hp)

    # ── Individual check tests ──

    def test_check_entity_total(self):
        conn = kw_health_check.get_db_connection()
        c = conn.cursor()
        self.assertEqual(kw_health_check.check_entity_total(c)["total_entities"], 10)
        conn.close()

    def test_check_orphan_ratio(self):
        conn = kw_health_check.get_db_connection()
        c = conn.cursor()
        r = kw_health_check.check_orphan_ratio(c)
        self.assertEqual(r["orphan_entities"], 2)  # e3, e7
        self.assertAlmostEqual(r["orphan_ratio"], 0.2, places=2)
        self.assertEqual(r["status"], "ok")  # 20% < 40%
        conn.close()

    def test_check_name_conflicts(self):
        conn = kw_health_check.get_db_connection()
        c = conn.cursor()
        r = kw_health_check.check_name_conflicts(c)
        self.assertEqual(r["conflict_count"], 1)  # "Python" = tech+decision
        self.assertEqual(r["conflicts"][0]["name"], "Python")
        self.assertEqual(r["conflicts"][0]["type_count"], 2)
        conn.close()

    def test_check_cross_day_aggregation(self):
        conn = kw_health_check.get_db_connection()
        c = conn.cursor()
        r = kw_health_check.check_cross_day_aggregation(c)
        self.assertEqual(r["day2_entities"], 3)  # e3, e4, e9
        self.assertAlmostEqual(r["day2_ratio"], 0.3, places=2)
        self.assertEqual(r["status"], "info")
        self.assertFalse(r["affects_overall"])
        self.assertIn("day_count 不具备跨日统计语义", r["message"])
        conn.close()

    def test_check_type_distribution(self):
        conn = kw_health_check.get_db_connection()
        c = conn.cursor()
        r = kw_health_check.check_type_distribution(c)
        td = r["type_distribution"]
        self.assertEqual(td["tech"], 2)  # e1, e10
        self.assertEqual(td["decision"], 2)  # e2, e3
        self.assertEqual(td["risk"], 1)
        self.assertEqual(td["preference"], 1)
        self.assertEqual(td["task"], 1)
        self.assertEqual(td["idea"], 1)
        self.assertEqual(td["fact"], 1)
        self.assertEqual(td["project"], 1)
        self.assertEqual(td["other"], 0)
        conn.close()

    def test_check_embedding_coverage(self):
        conn = kw_health_check.get_db_connection()
        c = conn.cursor()
        r = kw_health_check.check_embedding_coverage(c)
        self.assertEqual(r["total_entities"], 10)
        self.assertEqual(r["vector_count"], 8)
        self.assertAlmostEqual(r["coverage_ratio"], 0.8, places=2)
        self.assertEqual(r["status"], "warning")  # 80% < 90%
        self.assertEqual(r["embedding_model"], "bge-m3")
        self.assertEqual(r["dimension"], 1024)
        conn.close()

    def test_check_new_entity_inflow(self):
        conn = kw_health_check.get_db_connection()
        c = conn.cursor()
        r = kw_health_check.check_new_entity_inflow(c)
        self.assertEqual(r["new_entities_7d"], 0)  # all test entities have old dates
        self.assertEqual(r["status"], "alert")  # 0 → alert
        self.assertIsNotNone(r["last_consolidation"])
        conn.close()

    def test_check_relation_integrity(self):
        conn = kw_health_check.get_db_connection()
        c = conn.cursor()
        r = kw_health_check.check_relation_integrity(c)
        self.assertEqual(r["broken_relations"], 1)  # r_dangle references e99
        self.assertEqual(r["status"], "warning")  # 1 > 0
        conn.close()

    def test_check_pending_merges(self):
        conn = kw_health_check.get_db_connection()
        c = conn.cursor()
        r = kw_health_check.check_pending_merges(c)
        self.assertEqual(r["pending_merges"], 2)
        self.assertEqual(r["pending_by_kind"]["merge"], 2)
        conn.close()

    def test_check_coverage_days(self):
        conn = kw_health_check.get_db_connection()
        c = conn.cursor()
        r = kw_health_check.check_coverage_days(c)
        self.assertEqual(r["coverage_days"], 3)
        self.assertEqual(r["first_coverage"], "2026-06-01")
        self.assertEqual(r["last_coverage"], "2026-06-03")
        conn.close()

    # ── End-to-end tests ──

    def test_full_health_check_dry_run(self):
        output = kw_health_check.run_health_check(dry_run=True)
        report = json.loads(output)
        self.assertIn("report_date", report)
        self.assertIn("overall_status", report)
        self.assertIn("checks", report)
        self.assertIn("thresholds_evaluated", report)
        checks = report["checks"]
        for key in [
            "entity_total", "dma_health", "new_entity_inflow", "embedding_coverage",
            "orphan_ratio", "cross_day_aggregation", "type_distribution",
            "name_conflicts", "relation_integrity",
            "pending_merges", "coverage_days",
        ]:
            self.assertIn(key, checks)
            self.assertNotIn("error", checks[key],
                             f"check {key} raised: {checks[key].get('error')}")

    def test_full_health_check_writes_file(self):
        kw_health_check.run_health_check(dry_run=False)
        from datetime import date
        today = date.today().isoformat()
        path = os.path.join(kw_health_check.REPORTS_DIR, f"{today}.json")
        self.assertTrue(os.path.isfile(path))
        with open(path) as f:
            data = json.load(f)
        self.assertEqual(data["report_date"], today)

    def test_weekly_report_dry_run(self):
        output = kw_weekly_trend.generate_weekly_report(dry_run=True)
        for fragment in ["Knowledge Weaver", "总览", "类型分布", "活动摘要", "聚合健康"]:
            self.assertIn(fragment, output)

    def test_weekly_report_writes_files(self):
        kw_weekly_trend.generate_weekly_report(dry_run=False)
        from datetime import date
        iso_year, iso_week, _ = date.today().isocalendar()
        label = f"{iso_year}-W{iso_week:02d}"
        wpath = os.path.join(kw_weekly_trend.WEEKLY_DIR, f"{label}.md")
        self.assertTrue(os.path.isfile(wpath))
        self.assertTrue(os.path.isfile(kw_weekly_trend.HISTORY_PATH))
        with open(kw_weekly_trend.HISTORY_PATH) as hf:
            lines = hf.readlines()
        self.assertGreaterEqual(len(lines), 1)
        entry = json.loads(lines[-1])
        self.assertEqual(entry["week"], label)
        self.assertIn("metrics", entry)

    def test_weekly_trend_collect_metrics(self):
        m = kw_weekly_trend.collect_current_metrics()
        self.assertEqual(m["entity_total"], 10)
        self.assertEqual(m["relations_total"], 7)  # 6 normal + 1 dangling
        self.assertEqual(m["coverage_days"], 3)
        self.assertEqual(m["pending_merges"], 2)
        self.assertIn("tech", m["type_distribution"])
        self.assertIn("embedding_coverage", m)
        self.assertIn("new_entity_inflow_rate", m)
        self.assertIn("prev_week_inflow_rate", m)

    def test_weekly_trend_type_drift_insufficient(self):
        drift = kw_weekly_trend.type_drift([])
        self.assertEqual(drift["entropy_delta"], None)
        drift2 = kw_weekly_trend.type_drift([
            {"report_date": "2026-06-01",
             "checks": {"type_distribution": {"type_distribution": {"tech": 5}}}},
        ])
        self.assertEqual(drift2["entropy_delta"], None)

    def test_weekly_trend_type_drift_normal(self):
        r1 = {"report_date": "2026-06-01", "checks": {"type_distribution": {"type_distribution": {"a": 5, "b": 5}}}}
        r2 = {"report_date": "2026-06-07", "checks": {"type_distribution": {"type_distribution": {"a": 9, "b": 1}}}}
        drift = kw_weekly_trend.type_drift([r1, r2])
        self.assertIsNotNone(drift["entropy_delta"])
        self.assertLess(drift["new_entropy"], drift["old_entropy"])


class TestDMAHealthCheck(unittest.TestCase):
    """Tests for DMA health check — uses temp files, no DB."""

    @classmethod
    def setUpClass(cls):
        """Create temp directories and files for all DMA tests."""
        cls.tmpdir = tempfile.mkdtemp(prefix="kw_dma_test_")
        cls.log_path = os.path.join(cls.tmpdir, "dma.log")
        cls.checkpoint_path = os.path.join(cls.tmpdir, "checkpoint.json")
        cls.status_path = os.path.join(cls.tmpdir, "runtime_status.json")
        cls.memory_dir = os.path.join(cls.tmpdir, "memory")
        os.makedirs(cls.memory_dir, exist_ok=True)

        # Save original paths
        cls._save_paths = {
            "log": kw_health_check.DMA_LOG_PATH,
            "cp": kw_health_check.DMA_CHECKPOINT_PATH,
            "status": kw_health_check.DMA_STATUS_PATH,
            "mem": kw_health_check.MEMORY_DIR,
        }

        # Override to temp paths
        kw_health_check.DMA_LOG_PATH = cls.log_path
        kw_health_check.DMA_CHECKPOINT_PATH = cls.checkpoint_path
        kw_health_check.DMA_STATUS_PATH = cls.status_path
        kw_health_check.MEMORY_DIR = cls.memory_dir

    @classmethod
    def tearDownClass(cls):
        kw_health_check.DMA_LOG_PATH = cls._save_paths["log"]
        kw_health_check.DMA_CHECKPOINT_PATH = cls._save_paths["cp"]
        kw_health_check.DMA_STATUS_PATH = cls._save_paths["status"]
        kw_health_check.MEMORY_DIR = cls._save_paths["mem"]
        shutil.rmtree(cls.tmpdir, ignore_errors=True)

    def setUp(self):
        """Clean up temp files before each test."""
        for p in [self.log_path, self.checkpoint_path]:
            if os.path.isfile(p):
                os.remove(p)
        if os.path.isfile(self.status_path):
            os.remove(self.status_path)
        for f in os.listdir(self.memory_dir):
            os.remove(os.path.join(self.memory_dir, f))
        self._write_status()

    def _write_log(self, *lines: str, mtime_hours_ago: float | None = None) -> None:
        """Write lines to the test DMA log.

        If mtime_hours_ago is given, the file modification time is backdated.
        """
        with open(self.log_path, "w") as f:
            for line in lines:
                f.write(line + "\n")
        if mtime_hours_ago is not None:
            from datetime import datetime, timedelta
            old = datetime.now() - timedelta(hours=mtime_hours_ago)
            ts = old.timestamp()
            os.utime(self.log_path, (ts, ts))

    def _write_checkpoint(self) -> None:
        """Write a minimal checkpoint file."""
        with open(self.checkpoint_path, "w") as f:
            f.write('{"key": "value"}')

    def _write_status(self, *, outcome: str = "idle", pending_count: int | None = 0) -> None:
        """Write a fresh minimal DMA runtime-status-v1 fixture."""
        from datetime import datetime, timedelta, timezone
        observed = datetime.now(timezone.utc) - timedelta(minutes=1)
        finished = None if outcome == "running" else observed.isoformat()
        timestamp = observed.isoformat()
        payload = {
            "schema_version": "dma-runtime-status-v1",
            "run_id": "legacy-test-run",
            "started_at": (observed - timedelta(minutes=1)).isoformat(),
            "observed_at": timestamp,
            "finished_at": finished,
            "outcome": outcome,
            "reason": "test",
            "last_completed_at": finished,
            "last_archived_at": None,
            "pending_count": pending_count,
            "oldest_pending_at": None,
            "pending_reconcile_count": 0,
            "oldest_reconcile_at": None,
            "checkpoint_progress_at": timestamp,
            "failures": {
                operation: {
                    "consecutive": 0,
                    "last_failed_at": None,
                    "last_recovered_at": None,
                }
                for operation in ("storage", "summary", "archive")
            },
            "status_errors": [],
        }
        with open(self.status_path, "w", encoding="utf-8") as f:
            json.dump(payload, f)

    def _write_today_memory(self, content: str = "x" * 200) -> str:
        """Write today's memory file and return its path."""
        from datetime import date
        path = os.path.join(self.memory_dir, f"{date.today().isoformat()}.md")
        with open(path, "w") as f:
            f.write(content)
        return path

    # ── No files at all ──

    def test_all_files_missing(self):
        os.remove(self.status_path)
        r = kw_health_check.check_dma_health()
        self.assertEqual(r["status"], "warning")
        self.assertIsNone(r["log_freshness_hours"])
        self.assertEqual(r["error_count_24h"], 0)
        self.assertFalse(r["memory_file_exists"])
        self.assertEqual(r["memory_file_size"], 0)
        self.assertIsNone(r["checkpoint_freshness_hours"])
        self.assertEqual(r["runtime_status_error"], "dma_status_missing")
        self.assertIn("runtime status 尚未提供", r["message"])

    # ── Healthy state ──

    def test_all_healthy(self):
        now_ts = time.strftime("%Y-%m-%d %H:%M:%S")
        self._write_log(f"[{now_ts}] [INFO] All good")
        self._write_checkpoint()
        self._write_today_memory("x" * 500)

        r = kw_health_check.check_dma_health()
        self.assertEqual(r["status"], "ok")
        self.assertLess(r["log_freshness_hours"], 1)
        self.assertEqual(r["error_count_24h"], 0)
        self.assertTrue(r["memory_file_exists"])
        self.assertGreaterEqual(r["memory_file_size"], 500)
        self.assertLess(r["checkpoint_freshness_hours"], 1)
        self.assertIn("通过", r["message"])

    # ── Log freshness ──

    def test_log_stale_13h_warning(self):
        now_ts = time.strftime("%Y-%m-%d %H:%M:%S")
        self._write_log(f"[{now_ts}] [INFO] old entry", mtime_hours_ago=13)
        self._write_checkpoint()
        self._write_today_memory()

        r = kw_health_check.check_dma_health()
        self.assertEqual(r["status"], "ok")
        self.assertGreater(r["log_freshness_hours"], 12)
        self.assertNotIn("可能延迟", r["message"])

    def test_log_stale_25h_alert(self):
        now_ts = time.strftime("%Y-%m-%d %H:%M:%S")
        self._write_log(f"[{now_ts}] [INFO] very old entry", mtime_hours_ago=25)
        self._write_checkpoint()
        self._write_today_memory()

        r = kw_health_check.check_dma_health()
        self.assertEqual(r["status"], "ok")
        self.assertGreater(r["log_freshness_hours"], 24)
        self.assertNotIn("静默停摆", r["message"])

    # ── ERROR scanning ──

    def test_error_count_warning(self):
        now_ts = time.strftime("%Y-%m-%d %H:%M:%S")
        lines = [f"[{now_ts}] [ERROR] Something broke"]
        lines += [f"[{now_ts}] [INFO] normal"] * 5
        self._write_log(*lines)
        self._write_checkpoint()
        self._write_today_memory()

        r = kw_health_check.check_dma_health()
        self.assertEqual(r["error_count_24h"], 1)
        self.assertEqual(r["status"], "ok")
        self.assertIn("历史信息", r["message"])

    def test_error_count_alert(self):
        now_ts = time.strftime("%Y-%m-%d %H:%M:%S")
        lines = [f"[{now_ts}] [ERROR] broke {i}" for i in range(11)]
        self._write_log(*lines)
        self._write_checkpoint()
        self._write_today_memory()

        r = kw_health_check.check_dma_health()
        self.assertEqual(r["error_count_24h"], 11)
        self.assertEqual(r["status"], "ok")
        self.assertIn("历史信息", r["message"])

    def test_chinese_error_keywords_detected(self):
        now_ts = time.strftime("%Y-%m-%d %H:%M:%S")
        self._write_log(
            f"[{now_ts}] [INFO] API 错误: quota exceeded",
            f"[{now_ts}] [INFO] 操作失败: timeout",
            f"[{now_ts}] [INFO] 无法连接",
        )
        self._write_checkpoint()
        self._write_today_memory()

        r = kw_health_check.check_dma_health()
        self.assertEqual(r["error_count_24h"], 3)
        self.assertEqual(r["status"], "ok")

    def test_old_errors_excluded(self):
        """Errors older than 7 days should not be counted."""
        from datetime import datetime, timedelta
        old = (datetime.now() - timedelta(days=8)).strftime("%Y-%m-%d %H:%M:%S")
        recent = time.strftime("%Y-%m-%d %H:%M:%S")
        self._write_log(
            f"[{old}] [ERROR] ancient error",
            f"[{recent}] [INFO] all good now",
        )
        self._write_checkpoint()
        self._write_today_memory()

        r = kw_health_check.check_dma_health()
        self.assertEqual(r["error_count_24h"], 0)

    # ── Memory file ──

    def test_memory_file_missing_with_fresh_checkpoint(self):
        now_ts = time.strftime("%Y-%m-%d %H:%M:%S")
        self._write_log(f"[{now_ts}] [INFO] ok")
        self._write_checkpoint()
        # No memory file written — but checkpoint is fresh → warning, not alert

        r = kw_health_check.check_dma_health()
        self.assertEqual(r["status"], "ok")
        self.assertFalse(r["memory_file_exists"])
        self.assertEqual(r["memory_file_size"], 0)
        self.assertIn("没有待归档消息", r["message"])

    def test_memory_file_missing_with_stale_checkpoint(self):
        from datetime import datetime, timedelta
        now_ts = time.strftime("%Y-%m-%d %H:%M:%S")
        self._write_log(f"[{now_ts}] [INFO] ok")
        self._write_checkpoint()
        # Backdate checkpoint to > 24h ago
        old = datetime.now() - timedelta(hours=25)
        old_ts = old.timestamp()
        os.utime(self.checkpoint_path, (old_ts, old_ts))
        # No memory file written + stale checkpoint → alert

        r = kw_health_check.check_dma_health()
        self.assertEqual(r["status"], "ok")
        self.assertFalse(r["memory_file_exists"])
        self.assertEqual(r["memory_file_size"], 0)
        self.assertIn("没有待归档消息", r["message"])

    def test_memory_file_too_small(self):
        now_ts = time.strftime("%Y-%m-%d %H:%M:%S")
        self._write_log(f"[{now_ts}] [INFO] ok")
        self._write_checkpoint()
        self._write_today_memory("tiny")  # 4 bytes < 100

        r = kw_health_check.check_dma_health()
        self.assertEqual(r["status"], "warning")
        self.assertTrue(r["memory_file_exists"])
        self.assertEqual(r["memory_file_size"], 4)
        self.assertIn("较小", r["message"])

    # ── Checkpoint freshness ──

    def test_checkpoint_stale_25h_warning(self):
        from datetime import datetime, timedelta
        now_ts = time.strftime("%Y-%m-%d %H:%M:%S")
        self._write_log(f"[{now_ts}] [INFO] ok")
        self._write_today_memory()

        # Write checkpoint then backdate its mtime
        self._write_checkpoint()
        old = datetime.now() - timedelta(hours=25)
        ts = old.timestamp()
        os.utime(self.checkpoint_path, (ts, ts))

        r = kw_health_check.check_dma_health()
        self.assertEqual(r["status"], "ok")
        self.assertGreater(r["checkpoint_freshness_hours"], 24)
        self.assertLess(r["checkpoint_freshness_hours"], 49)

    def test_checkpoint_stale_49h_alert(self):
        from datetime import datetime, timedelta
        now_ts = time.strftime("%Y-%m-%d %H:%M:%S")
        self._write_log(f"[{now_ts}] [INFO] ok")
        self._write_today_memory()

        self._write_checkpoint()
        old = datetime.now() - timedelta(hours=49)
        ts = old.timestamp()
        os.utime(self.checkpoint_path, (ts, ts))

        r = kw_health_check.check_dma_health()
        self.assertEqual(r["status"], "ok")
        self.assertGreater(r["checkpoint_freshness_hours"], 48)
        self.assertNotIn("检查点卡死", r["message"])

    # ── Worst-case: all problems at once ──

    def test_all_problems_combined(self):
        from datetime import datetime, timedelta
        # Log with errors (all within 7 days) but mtime backdated 25h
        now_ts = time.strftime("%Y-%m-%d %H:%M:%S")
        errors = [f"[{now_ts}] [ERROR] critical failure {i}" for i in range(12)]
        self._write_log(*errors, mtime_hours_ago=25)

        self._write_checkpoint()
        old_cp = datetime.now() - timedelta(hours=49)
        ts = old_cp.timestamp()
        os.utime(self.checkpoint_path, (ts, ts))

        # No memory file

        r = kw_health_check.check_dma_health()
        self.assertEqual(r["status"], "ok")
        self.assertGreater(r["log_freshness_hours"], 24)
        self.assertEqual(r["error_count_24h"], 12)
        self.assertFalse(r["memory_file_exists"])
        self.assertGreater(r["checkpoint_freshness_hours"], 48)


if __name__ == "__main__":
    unittest.main()
