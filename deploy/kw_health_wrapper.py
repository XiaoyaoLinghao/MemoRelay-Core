#!/usr/bin/env python3
"""Run one health report and publish only that invocation's verified result."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import uuid

SCRIPT_DIR = os.environ.get("KW_HEALTH_SCRIPT_DIR", "/home/openclaw/.openclaw/knowledge/scripts")
REPORTS_DIR = os.environ.get("KW_HEALTH_REPORTS_DIR", "/home/openclaw/.openclaw/knowledge/health_reports")
WEEKLY_DIR = os.environ.get("KW_WEEKLY_REPORTS_DIR", "/home/openclaw/.openclaw/knowledge/weekly_reports")
ALERT_FLAG = os.environ.get("KW_ALERT_FLAG", "/home/openclaw/.openclaw/knowledge/.alert_pending")
WEEKLY_ALERT = os.environ.get("KW_WEEKLY_ALERT_FLAG", "/home/openclaw/.openclaw/knowledge/.weekly_alert_pending")
HISTORY_PATH = os.environ.get("KW_HEALTH_HISTORY_PATH", "/home/openclaw/.openclaw/knowledge/health_history.jsonl")
TIMEOUT_SECONDS = 120
REQUIRED_CHECKS = {"entity_total", "dma_health", "new_entity_inflow", "embedding_coverage",
                   "orphan_ratio", "cross_day_aggregation", "type_distribution",
                   "name_conflicts", "relation_integrity", "pending_merges", "coverage_days"}


class ReportError(ValueError):
    """A bounded internal reason code, never arbitrary subprocess output."""


def error_code(error):
    if isinstance(error, ReportError):
        return str(error)
    if isinstance(error, subprocess.TimeoutExpired):
        return "subprocess_timeout"
    if isinstance(error, json.JSONDecodeError):
        return "report_json_invalid"
    if isinstance(error, FileNotFoundError):
        return "report_missing"
    if isinstance(error, OSError):
        return "report_io_error"
    if isinstance(error, subprocess.SubprocessError):
        return "subprocess_error"
    return "report_structure_invalid"


def atomic_write(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n",
                                         dir=path.parent, prefix=".kw-report-", delete=False) as f:
            temporary = Path(f.name)
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_json(path, value):
    atomic_write(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


@contextmanager
def report_lock(flag):
    """OS-owned lock serializes report/alert publication, including recovery."""
    path = Path(flag).with_name(Path(flag).name + ".lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        locked = False
        try:
            if os.name == "nt":
                import msvcrt
                if stream.tell() == 0:
                    stream.write(b"\0")
                    stream.flush()
                stream.seek(0)
                try:
                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                    locked = True
                except OSError:
                    pass
            else:
                import fcntl
                try:
                    fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    locked = True
                except BlockingIOError:
                    pass
            yield locked
        finally:
            if locked:
                if os.name == "nt":
                    stream.seek(0)
                    msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(stream, fcntl.LOCK_UN)


def validate_metadata(report, run_id, started, schema):
    if not isinstance(report, dict) or report.get("schema_version") != schema:
        raise ReportError("invalid_report_schema")
    if report.get("run_id") != run_id or type(report.get("checks_complete")) is not bool:
        raise ReportError("invalid_report_identity")
    generated = datetime.fromisoformat(report["generated_at"])
    if generated.tzinfo is None or not started - timedelta(seconds=5) <= generated <= datetime.now(timezone.utc) + timedelta(seconds=5):
        raise ReportError("stale_or_future_report")
    if report.get("report_date") != generated.date().isoformat():
        raise ReportError("invalid_report_date")
    return generated


def validate_daily(report, run_id, started):
    validate_metadata(report, run_id, started, "kw-health-report-v1")
    status = report.get("overall_status")
    if status not in ("ok", "info", "warning", "alert"):
        raise ReportError("invalid_overall_status")
    checks = report.get("checks")
    if not isinstance(checks, dict) or not REQUIRED_CHECKS.issubset(checks):
        raise ReportError("missing_checks")
    for check in checks.values():
        if not isinstance(check, dict):
            raise ReportError("invalid_check")
        if "status" in check and check["status"] not in ("ok", "info", "unknown", "warning", "alert"):
            raise ReportError("invalid_check_status")
        if status == "ok" and (check.get("status") in ("warning", "alert", "unknown") or "error" in check):
            raise ReportError("inconsistent_healthy_report")
        if status != "alert" and check.get("status") == "alert":
            raise ReportError("inconsistent_alert_report")
        if "error" in check and status != "alert":
            raise ReportError("inconsistent_error_report")
    ranks = {"ok": 0, "info": 0, "unknown": 1, "warning": 1, "alert": 2}
    actual = max(ranks.get(check.get("status", "ok"), 0) for check in checks.values())
    if ranks[status] != actual:
        raise ReportError("inconsistent_overall_severity")
    if report["checks_complete"] and "status" not in checks["dma_health"]:
        raise ReportError("missing_dma_status")


def failure_alert(mode, flag, run_id, code):
    alert = {
        "kind": mode, "date": datetime.now().astimezone().date().isoformat(),
        "run_id": run_id, "status": "alert", "summary": "健康检查自身失败：" + code,
        "alerts": [{"name": "health_check_failure", "status": "alert",
                    "message": "健康检查自身失败：" + code}],
    }
    if mode == "weekly":
        # Preserve the established weekly consumer shape (kind/file/summary).
        filename = "health-check-failure-" + run_id + ".md"
        for directory in (Path(WEEKLY_DIR), Path(flag).parent):
            path = directory / filename
            try:
                atomic_write(path, "# KW 周报检查失败\n\n" + alert["summary"] + "\n")
            except OSError:
                continue
            alert["file"] = str(path)
            break
        else:
            alert["artifact_error"] = "failure_report_unavailable"
    write_json(flag, alert)


def append_history(entry):
    path = Path(HISTORY_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _run(mode):
    flag = ALERT_FLAG if mode == "daily" else WEEKLY_ALERT
    directory = REPORTS_DIR if mode == "daily" else WEEKLY_DIR
    script = "kw_health_check.py" if mode == "daily" else "kw_weekly_trend.py"
    run_id = uuid.uuid4().hex
    try:
        with report_lock(flag) as locked:
            if not locked:
                print("Health report already running; existing report/alert retained.", file=sys.stderr)
                return 0
            try:
                Path(directory).mkdir(parents=True, exist_ok=True)
                started = datetime.now(timezone.utc)
                with tempfile.TemporaryDirectory(prefix=".kw-run-", dir=directory) as tmp:
                    output = Path(tmp) / "report.json"
                    result = subprocess.run(
                        [sys.executable, str(Path(SCRIPT_DIR) / script),
                         "--run-id", run_id, "--output", str(output)],
                        capture_output=True, timeout=TIMEOUT_SECONDS, check=False,
                    )
                    if result.returncode != 0:
                        raise ReportError("subprocess_failed")
                    report = json.loads(output.read_text(encoding="utf-8"))
                    if mode == "daily":
                        validate_daily(report, run_id, started)
                        write_json(Path(directory) / (report["report_date"] + ".json"), report)
                        status = report["overall_status"]
                        if status in ("alert", "warning"):
                            write_json(flag, {
                                "kind": "daily", "date": report["report_date"],
                                "run_id": run_id, "status": status,
                                "alerts": [{"name": k, "status": "warning" if v.get("status") == "unknown" else v.get("status"),
                                            "message": v.get("message", v.get("error"))}
                                           for k, v in report["checks"].items()
                                           if v.get("status") in ("alert", "warning", "unknown")],
                            })
                        elif status == "ok" and report["checks_complete"] and report["checks"]["dma_health"].get("status") == "ok":
                            Path(flag).unlink(missing_ok=True)
                    else:
                        generated = validate_metadata(report, run_id, started, "kw-weekly-report-v1")
                        year, week, _ = generated.date().isocalendar()
                        week_label = f"{year}-W{week:02d}"
                        markdown = report.get("markdown")
                        if not report["checks_complete"] or report.get("week") != week_label or not isinstance(markdown, str) or not markdown.strip():
                            raise ReportError("invalid_weekly_report")
                        history = report.get("history_entry")
                        if not isinstance(history, dict) or history.get("run_id") != run_id or history.get("week") != week_label:
                            raise ReportError("invalid_weekly_history")
                        path = Path(directory) / (week_label + ".md")
                        atomic_write(path, markdown)
                        append_history(dict(history, report_path=str(path)))
                        write_json(flag, {"kind": "weekly", "file": str(path),
                                          "run_id": run_id, "summary": "\n".join(markdown.splitlines()[:15])})
                return 0
            except Exception as error:
                # Do not expose arbitrary subprocess output or reuse previous reports.
                code = error_code(error)
                failure_alert(mode, flag, run_id, code)
                print("Health check failed: " + code, file=sys.stderr)
                return 1
    except OSError as error:
        print("Health check could not publish/lock alert: " + type(error).__name__, file=sys.stderr)
        return 1


def run_daily():
    return _run("daily")


def run_weekly():
    return _run("weekly")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["daily", "weekly"])
    args = parser.parse_args()
    sys.exit(run_daily() if args.mode == "daily" else run_weekly())
