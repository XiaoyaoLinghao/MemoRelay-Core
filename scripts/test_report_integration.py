"""Real subprocess reports with isolated SQLite; optional sibling-DMA contract test.

Set KW_TEST_DMA_ROOT to a local DMA checkout to exercise its real status producer.
This never invokes archive, the network, or a deployed database.
"""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def environment(tmp_path):
    database = tmp_path / "fixture.db"
    conn = sqlite3.connect(database)
    conn.executescript('''
        CREATE TABLE entities(id TEXT PRIMARY KEY,type TEXT,name TEXT,summary TEXT,
          importance REAL,first_seen TEXT,last_seen TEXT,day_count INTEGER,
          source_lines TEXT,metadata TEXT,created_at TEXT,updated_at TEXT);
        CREATE TABLE relations(id TEXT,from_entity TEXT,to_entity TEXT);
        CREATE TABLE entity_vectors(entity_id TEXT,embedding TEXT);
        CREATE TABLE embedding_meta(key TEXT,value TEXT);
        CREATE TABLE merge_review(kind TEXT,status TEXT);
        CREATE TABLE daily_manifest(date TEXT,processed_at TEXT,status TEXT);
    ''')
    today = datetime.now(timezone.utc).date().isoformat()
    for i in range(10):
        conn.execute("INSERT INTO entities VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                     (str(i), "tech", "Entity " + str(i), "fixture", 0.5, today,
                      today, 2, "[]", "{}", today, today))
        conn.execute("INSERT INTO relations VALUES(?,?,?)", (str(i), str(i), str((i+1) % 10)))
        conn.execute("INSERT INTO entity_vectors VALUES(?,?)", (str(i), "[0.1,0.2]"))
    conn.execute("INSERT INTO daily_manifest VALUES(?,?,?)", (today, today, "ok"))
    conn.commit()
    conn.close()
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8",
               KNOWLEDGE_WEAVER_DB_PATH=str(database),
               KNOWLEDGE_WEAVER_MEMORY_DIR=str(tmp_path / "memory"),
               KW_DMA_STATUS_PATH=str(tmp_path / "dma.json"),
               KW_DMA_LOG_PATH=str(tmp_path / "dma.log"),
               KW_DMA_CHECKPOINT_PATH=str(tmp_path / "cursor.json"),
               KW_HEALTH_SCRIPT_DIR=str(ROOT / "scripts"),
               KW_HEALTH_REPORTS_DIR=str(tmp_path / "reports"),
               KW_WEEKLY_REPORTS_DIR=str(tmp_path / "weekly"),
               KW_HEALTH_HISTORY_PATH=str(tmp_path / "history.jsonl"),
               KW_ALERT_FLAG=str(tmp_path / "daily.flag"),
               KW_WEEKLY_ALERT_FLAG=str(tmp_path / "weekly.flag"))
    return tmp_path, env, hashlib.sha256(database.read_bytes()).hexdigest()


def run_wrapper(env, mode="daily"):
    return subprocess.run([sys.executable, str(ROOT / "deploy" / "kw_health_wrapper.py"), mode],
                          env=env, capture_output=True, text=True, encoding="utf-8", timeout=30)


def publish_idle(env):
    now = datetime.now(timezone.utc).isoformat()
    status = {"schema_version": "dma-runtime-status-v1", "run_id": "fixture-run",
              "started_at": now, "observed_at": now, "finished_at": now,
              "outcome": "idle", "reason": "no_messages", "last_completed_at": now,
              "last_archived_at": None, "pending_count": 0, "oldest_pending_at": None,
              "pending_reconcile_count": 0, "oldest_reconcile_at": None,
              "checkpoint_progress_at": None, "status_errors": [],
              "failures": {name: {"consecutive": 0, "last_failed_at": None,
                                  "last_recovered_at": None}
                           for name in ("storage", "summary", "archive")}}
    Path(env["KW_DMA_STATUS_PATH"]).write_text(json.dumps(status), encoding="utf-8")


def test_actual_checker_wrapper_idle_and_weekly_leave_database_unchanged(environment):
    tmp, env, original = environment
    publish_idle(env)
    Path(env["KW_ALERT_FLAG"]).write_text("old-alert", encoding="utf-8")
    result = run_wrapper(env)
    assert result.returncode == 0, result.stderr
    reports = list((tmp / "reports").glob("*.json"))
    assert len(reports) == 1
    report = json.loads(reports[0].read_text(encoding="utf-8"))
    assert report["overall_status"] == "ok", report
    assert report["checks"]["cross_day_aggregation"]["status"] == "info"
    assert not Path(env["KW_ALERT_FLAG"]).exists()
    assert not (tmp / "memory").exists()
    result = run_wrapper(env, "weekly")
    assert result.returncode == 0, result.stderr
    assert "聚合引擎可能失效" not in next((tmp / "weekly").glob("*.md")).read_text(encoding="utf-8")
    assert hashlib.sha256(Path(env["KNOWLEDGE_WEAVER_DB_PATH"]).read_bytes()).hexdigest() == original


def test_actual_checker_schema_error_becomes_checker_alert(environment):
    tmp, env, _ = environment
    publish_idle(env)
    conn = sqlite3.connect(env["KNOWLEDGE_WEAVER_DB_PATH"])
    conn.execute("DROP TABLE relations")
    conn.close()
    result = run_wrapper(env)
    assert result.returncode == 1
    flag = json.loads(Path(env["KW_ALERT_FLAG"]).read_text(encoding="utf-8"))
    assert flag["alerts"][0]["name"] == "health_check_failure"


def test_real_dma_producer_failure_idle_and_recovery(environment):
    dma_root = os.environ.get("KW_TEST_DMA_ROOT")
    if not dma_root:
        pytest.skip("Set KW_TEST_DMA_ROOT for the two-repository contract check")
    producer = Path(dma_root) / "scripts" / "runtime-status.py"
    assert producer.is_file()
    tmp, env, _ = environment
    checkpoint = tmp / "cursor.json"
    checkpoint.write_text("{}", encoding="utf-8")
    def emit(run_id, outcome, summary):
        def command(*args):
            result = subprocess.run([sys.executable, str(producer), *args], env=env,
                                    capture_output=True, text=True, encoding="utf-8", timeout=15)
            assert result.returncode == 0, result.stderr
            return result.stdout.strip()
        before = command("fingerprint", "--file", str(checkpoint))
        command("begin", "--path", env["KW_DMA_STATUS_PATH"], "--run-id", run_id)
        command("finish", "--path", env["KW_DMA_STATUS_PATH"], "--run-id", run_id,
                "--outcome", outcome, "--reason", "summary" if outcome == "failed" else "substantive" if outcome == "archived" else "no_input",
                "--pending-count", "0", "--reconcile-dir", str(tmp / "pending"),
                "--checkpoint-path", str(checkpoint), "--checkpoint-before", before,
                "--storage-result", "success", "--summary-result", summary,
                "--archive-result", "success" if outcome == "archived" else "unknown",
                "--archived", "1" if outcome == "archived" else "0")
        result = run_wrapper(env)
        assert result.returncode == 0, result.stderr
        return json.loads(next((tmp / "reports").glob("*.json")).read_text(encoding="utf-8"))
    assert emit("failed-run", "failed", "failed")["overall_status"] == "alert"
    assert emit("idle-run", "idle", "unknown")["overall_status"] == "alert"
    assert emit("recovered-run", "archived", "success")["overall_status"] == "ok"
    assert not Path(env["KW_ALERT_FLAG"]).exists()
    assert emit("failed-again", "failed", "failed")["overall_status"] == "alert"
    assert Path(env["KW_ALERT_FLAG"]).exists()
