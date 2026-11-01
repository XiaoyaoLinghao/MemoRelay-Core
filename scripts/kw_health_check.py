#!/usr/bin/env python3
"""
Knowledge Weaver Data Quality Health Check (每日 cron).

Performs read-only quality checks against the SQLite knowledge database
and emits a JSON report to health_reports/YYYY-MM-DD.json.

Usage:
    kw_health_check.py              # full run, write JSON report
    kw_health_check.py --dry-run    # print JSON to stdout only, no file write
"""

import argparse
import json
import math
import os
import re
import sqlite3
import sys
import tempfile
import uuid
from datetime import date, datetime, timedelta, timezone

import dma_status

# ── Absolute paths ───────────────────────────────────────────────────────────

DB_PATH = os.environ.get(
    "KNOWLEDGE_WEAVER_DB_PATH", "/home/openclaw/.openclaw/knowledge/knowledge.db"
)
REPORTS_DIR = os.environ.get(
    "KW_HEALTH_REPORTS_DIR", "/home/openclaw/.openclaw/knowledge/health_reports"
)
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))

# DMA diagnostic paths (module-level for testability).  Runtime state is read
# from the versioned status file; logs, checkpoints and Memory files are kept
# as supporting diagnostics only.
DMA_LOG_PATH = os.environ.get(
    "KW_DMA_LOG_PATH", "/home/openclaw/.openclaw/logs/daily-memory-archiver.log"
)
DMA_CHECKPOINT_PATH = os.environ.get(
    "KW_DMA_CHECKPOINT_PATH",
    "/home/openclaw/.openclaw/skills/daily-memory-archiver/config/.archive_merge_checkpoint.json",
)
MEMORY_DIR = os.environ.get(
    "KNOWLEDGE_WEAVER_MEMORY_DIR", "/home/openclaw/.openclaw/workspace/memory/"
)
DMA_STATUS_PATH = dma_status.DEFAULT_STATUS_PATH
DMA_EXPECTED_INTERVAL_SECONDS = dma_status.DEFAULT_EXPECTED_INTERVAL_SECONDS
DMA_STATUS_TIMEOUT_SECONDS = dma_status.DEFAULT_STATUS_TIMEOUT_SECONDS
DMA_BACKLOG_TIMEOUT_SECONDS = dma_status.DEFAULT_BACKLOG_TIMEOUT_SECONDS

# Known entity types
KNOWN_TYPES = ("tech", "decision", "risk", "preference", "task", "idea", "fact", "project")


def ensure_dir(path: str) -> None:
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def get_db_connection(readonly: bool = True) -> sqlite3.Connection:
    """Open a read-only connection to the knowledge DB. Raises on failure."""
    if not os.path.isfile(DB_PATH):
        raise FileNotFoundError(f"Database not found: {DB_PATH}")

    if readonly:
        # URI mode with ?mode=ro avoids accidental writes
        uri = f"file:{DB_PATH}?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
    else:
        conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# ── Check functions (each returns a dict) ────────────────────────────────────

def check_entity_total(cur: sqlite3.Cursor) -> dict:
    """Count total entities."""
    cur.execute("SELECT COUNT(*) AS cnt FROM entities")
    row = cur.fetchone()
    return {"total_entities": row["cnt"]}


def _severity_worse(a: str, b: str) -> str:
    """Return the worse (higher priority) of two severity levels."""
    order = {"ok": 0, "info": 0, "unknown": 1, "warning": 1, "alert": 2}
    return a if order.get(a, 0) >= order.get(b, 0) else b


def _aware_now(value: datetime | None = None) -> datetime:
    """Return an aware local timestamp for report and diagnostic comparisons."""
    current = value or datetime.now().astimezone()
    if current.tzinfo is None or current.utcoffset() is None:
        current = current.astimezone()
    return current


def _configured_dma_status_path(path: str | None = None) -> str:
    """Resolve the testable status path, honoring KW_DMA_STATUS_PATH."""
    if path:
        return path
    return os.environ.get("KW_DMA_STATUS_PATH") or DMA_STATUS_PATH


def _configured_seconds(name: str, default: int, override: int | float | None) -> float:
    """Read a positive finite duration from an explicit value or environment."""
    value: int | float | str = override if override is not None else os.environ.get(name, default)
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive finite number") from exc
    if not math.isfinite(parsed) or parsed <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return parsed


def _scan_dma_log(now: datetime) -> tuple[float | None, int, str | None]:
    """Return log age, recent error count, and an observation error code.

    Log data is retained for context only.  It never decides recovery or DMA
    health once runtime-status-v1 is available.
    """
    if not os.path.isfile(DMA_LOG_PATH):
        return None, 0, None

    try:
        mtime = datetime.fromtimestamp(os.path.getmtime(DMA_LOG_PATH), now.tzinfo)
        log_age_hours = round(max(0.0, (now - mtime).total_seconds()) / 3600, 2)
        cutoff = now - timedelta(days=1)
        error_count = 0
        error_pattern = re.compile(r"(ERROR|FATAL|错误|失败|无法)", re.IGNORECASE)
        in_old_block = False
        with open(DMA_LOG_PATH, "r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if line.startswith("[") and len(line) > 20:
                    try:
                        line_ts = datetime.strptime(
                            line[1:20], "%Y-%m-%d %H:%M:%S"
                        ).replace(tzinfo=now.tzinfo)
                        in_old_block = line_ts < cutoff
                    except ValueError:
                        # An unparsable line is diagnostic noise; retain the
                        # previous timestamp block just as the old scanner did.
                        pass
                if not in_old_block and error_pattern.search(line):
                    error_count += 1
        return log_age_hours, error_count, None
    except (OSError, UnicodeError):
        return None, 0, "dma_log_observation_failed"


def _path_age_hours(path: str, now: datetime) -> tuple[float | None, str | None]:
    if not os.path.isfile(path):
        return None, None
    try:
        mtime = datetime.fromtimestamp(os.path.getmtime(path), now.tzinfo)
        return round(max(0.0, (now - mtime).total_seconds()) / 3600, 2), None
    except OSError:
        return None, "dma_checkpoint_observation_failed"


def check_dma_health(
    *,
    now: datetime | None = None,
    status_path: str | None = None,
    expected_interval_seconds: int | float | None = None,
    status_timeout_seconds: int | float | None = None,
    backlog_timeout_seconds: int | float | None = None,
) -> dict:
    """Check DMA using runtime-status-v1 and read-only diagnostics.

    A missing status file is a rollout ``warning`` with incomplete evidence.
    Malformed, unreadable or stale status is an ``alert``.  Historical log
    errors, checkpoint mtime and today's Memory-file absence are informational
    unless the versioned runtime status proves a current failure or backlog.
    """
    current = _aware_now(now)
    expected_interval = _configured_seconds(
        "KW_DMA_EXPECTED_INTERVAL_SECONDS",
        DMA_EXPECTED_INTERVAL_SECONDS,
        expected_interval_seconds,
    )
    status_timeout = _configured_seconds(
        "KW_DMA_STATUS_TIMEOUT_SECONDS",
        DMA_STATUS_TIMEOUT_SECONDS,
        status_timeout_seconds,
    )
    backlog_timeout = _configured_seconds(
        "KW_DMA_BACKLOG_TIMEOUT_SECONDS",
        DMA_BACKLOG_TIMEOUT_SECONDS,
        backlog_timeout_seconds,
    )
    today_str = current.date().isoformat()
    messages: list[str] = []
    status_errors: list[str] = []

    log_freshness_hours, error_count_24h, log_error = _scan_dma_log(current)
    if log_error:
        status_errors.append(log_error)
    checkpoint_freshness_hours, checkpoint_error = _path_age_hours(
        DMA_CHECKPOINT_PATH, current
    )
    if checkpoint_error:
        status_errors.append(checkpoint_error)

    memory_file_exists = False
    memory_file_size = 0
    memory_path = os.path.join(MEMORY_DIR, f"{today_str}.md")
    try:
        if os.path.isfile(memory_path):
            memory_file_exists = True
            memory_file_size = os.path.getsize(memory_path)
    except OSError:
        status_errors.append("memory_observation_failed")

    runtime_status: dict | None = None
    runtime_public: dict | None = None
    runtime_status_errors: list[str] = []
    runtime_classification: dict
    runtime_error: str | None = None
    resolved_status_path = _configured_dma_status_path(status_path)
    try:
        runtime_status = dma_status.read_dma_status(
            resolved_status_path,
            now=current,
            status_timeout_seconds=status_timeout,
        )
        runtime_classification = dma_status.classify_status(
            runtime_status,
            now=current,
            expected_interval_seconds=expected_interval,
            status_timeout_seconds=status_timeout,
            backlog_timeout_seconds=backlog_timeout,
        )
        runtime_public = dma_status.public_status(runtime_status)
        runtime_status_errors = list(runtime_status.get("status_errors", []))
    except dma_status.DmaStatusMissingError:
        runtime_error = dma_status.DmaStatusMissingError.code
        runtime_classification = {
            "status": "warning",
            "level": "warning",
            "state": "unknown",
            "message": "DMA runtime status 尚未提供，当前无法确认最近一次运行结果",
            "alert_codes": [],
            "warning_codes": [runtime_error],
            "recovered_operations": [],
            "evidence_complete": False,
        }
    except dma_status.DmaStatusStaleError as exc:
        runtime_error = dma_status.DmaStatusStaleError.code
        runtime_classification = {
            "status": "alert",
            "level": "alert",
            "state": "stale",
            "message": str(exc),
            "alert_codes": [runtime_error],
            "warning_codes": [],
            "recovered_operations": [],
            "evidence_complete": False,
        }
        if exc.status is not None:
            runtime_status = exc.status
            runtime_status_errors = list(exc.status.get("status_errors", []))
            runtime_public = dma_status.public_status(exc.status)
            runtime_public["age_seconds"] = exc.age_seconds
            runtime_public["stale"] = True
    except dma_status.DmaStatusError as exc:
        runtime_error = exc.code
        runtime_classification = {
            "status": "alert",
            "level": "alert",
            "state": "invalid",
            "message": str(exc),
            "alert_codes": [runtime_error],
            "warning_codes": [],
            "recovered_operations": [],
            "evidence_complete": False,
        }

    status = runtime_classification["status"]
    if runtime_classification.get("level") == "info" and status == "ok":
        status = "info"
    # Diagnostic failures are evidence failures, even though the runtime file
    # itself may describe a healthy service.
    if status_errors:
        status = _severity_worse(status, "alert")
        messages.append("DMA 健康检查无法读取部分诊断信息")
    messages.append(runtime_classification["message"])

    if error_count_24h:
        messages.append(f"DMA 日志近 24 小时记录 {error_count_24h} 条错误（历史信息）")

    if memory_file_exists and memory_file_size < 100:
        status = _severity_worse(status, "warning")
        messages.append(f"今日 Memory 文件存在但较小（{memory_file_size} bytes），请人工查看")
    elif not memory_file_exists:
        # Absence alone is normal for idle/noise-only runs with no expected
        # output.  Do not infer anything from checkpoint or log mtime.
        if runtime_status and runtime_status.get("pending_count") == 0 and runtime_status.get(
            "outcome"
        ) in {"idle", "noise_only", "deferred"}:
            messages.append("当前快照没有待归档消息，今日无需创建 Memory 文件")
        elif runtime_status and runtime_status.get("pending_count") == 0 and runtime_status.get(
            "outcome"
        ) == "archived":
            messages.append("当前快照没有待归档消息，今日没有新增 Memory 文件")
        elif not runtime_status:
            messages.append("今日 Memory 文件是否应生成无法由缺失的 DMA 状态判断")
        else:
            messages.append("今日 Memory 文件尚未生成；请结合 DMA 运行结果判断")

    if checkpoint_freshness_hours is not None:
        messages.append(f"DMA 检查点最近更新于 {checkpoint_freshness_hours:.1f}h 前（诊断信息）")

    if status == "ok" and runtime_status and not status_errors:
        messages.append("DMA 健康检查通过")
    if not messages:
        messages.append("DMA 健康检查通过")

    # A running/unknown status is intentionally incomplete.  A terminal alert
    # is still a valid, complete service observation and should exit 0.
    evidence_complete = bool(runtime_classification.get("evidence_complete", False))
    if status_errors:
        evidence_complete = False

    return {
        "log_freshness_hours": log_freshness_hours,
        "error_count_24h": error_count_24h,
        "memory_file_exists": memory_file_exists,
        "memory_file_size": memory_file_size,
        "checkpoint_freshness_hours": checkpoint_freshness_hours,
        "status": status,
        "runtime_level": runtime_classification.get("level", status),
        "runtime_state": runtime_classification.get("state", "unknown"),
        "runtime_outcome": runtime_status.get("outcome") if runtime_status else None,
        "runtime_status": runtime_public,
        "runtime_status_path": resolved_status_path,
        "runtime_status_error": runtime_error,
        "status_errors": status_errors + runtime_status_errors,
        "diagnostic_errors": status_errors,
        "runtime_status_errors": runtime_status_errors,
        "alert_codes": runtime_classification.get("alert_codes", []),
        "warning_codes": runtime_classification.get("warning_codes", []),
        "recovered_operations": runtime_classification.get("recovered_operations", []),
        "evidence_complete": evidence_complete,
        "message": "; ".join(messages),
    }


def check_orphan_ratio(cur: sqlite3.Cursor) -> dict:
    """Proportion of entities with no relations at all."""
    cur.execute("""
        SELECT COUNT(*) AS total FROM entities
    """)
    total = cur.fetchone()["total"]
    if total == 0:
        return {"orphan_entities": 0, "orphan_ratio": 0.0, "status": "ok", "message": "No entities"}

    cur.execute("""
        SELECT COUNT(*) AS orphans
        FROM entities e
        WHERE NOT EXISTS (SELECT 1 FROM relations r WHERE r.from_entity = e.id)
          AND NOT EXISTS (SELECT 1 FROM relations r WHERE r.to_entity   = e.id)
    """)
    orphans = cur.fetchone()["orphans"]
    ratio = round(orphans / total, 4)
    status = "alert" if ratio > 0.40 else "ok"
    message = (f"Isolated entities above 40% threshold ({ratio:.1%})"
               if ratio > 0.40 else "Acceptable isolation level")
    return {"orphan_entities": orphans, "orphan_ratio": ratio,
            "status": status, "message": message}


def check_name_conflicts(cur: sqlite3.Cursor) -> dict:
    """Entities sharing the same name but different types."""
    cur.execute("""
        SELECT name, COUNT(DISTINCT type) AS type_count,
               GROUP_CONCAT(DISTINCT type) AS types
        FROM entities
        GROUP BY name
        HAVING COUNT(DISTINCT type) > 1
        ORDER BY type_count DESC
    """)
    conflicts = [
        {"name": r["name"], "type_count": r["type_count"], "types": r["types"]}
        for r in cur.fetchall()
    ]
    count = len(conflicts)
    status = "warning" if count > 5 else "ok"
    message = (f"Name-cross-type conflicts exceed threshold ({count} > 5)"
               if count > 5 else "Name conflicts within acceptable range")
    return {"conflict_count": count, "conflicts": conflicts,
            "status": status, "message": message}


def check_cross_day_aggregation(cur: sqlite3.Cursor) -> dict:
    """Report legacy day_count statistics without treating them as a check.

    ``day_count`` currently mixes occurrence and processing counts, so its
    ratio cannot prove cross-day aggregation quality.  Keep the distribution
    visible for migration analysis, but never let it affect overall health.
    """
    cur.execute("SELECT COUNT(*) AS total FROM entities")
    total = cur.fetchone()["total"]
    if total == 0:
        return {
            "day2_entities": 0,
            "day2_ratio": 0.0,
            "status": "info",
            "state": "unknown",
            "affects_overall": False,
            "message": "No entities; cross-day aggregation is informational",
        }

    cur.execute("SELECT COUNT(*) AS cnt FROM entities WHERE day_count = 2")
    day2 = cur.fetchone()["cnt"]
    ratio = round(day2 / total, 4)
    return {
        "day2_entities": day2,
        "day2_ratio": ratio,
        "status": "info",
        "state": "unknown",
        "affects_overall": False,
        "message": (
            f"跨日聚合仅作信息展示：{ratio:.1%} 实体 day_count=2；"
            "现有 day_count 不具备跨日统计语义"
        ),
    }


def check_type_distribution(cur: sqlite3.Cursor) -> dict:
    """Count per type including an 'other' bucket for unrecognised types."""
    type_counts = {}
    for t in KNOWN_TYPES:
        cur.execute("SELECT COUNT(*) AS cnt FROM entities WHERE type = ?", (t,))
        type_counts[t] = cur.fetchone()["cnt"]

    # Catch any non-standard types
    placeholders = ",".join("?" * len(KNOWN_TYPES))
    cur.execute(
        f"SELECT COUNT(*) AS cnt FROM entities WHERE type NOT IN ({placeholders})",
        KNOWN_TYPES,
    )
    type_counts["other"] = cur.fetchone()["cnt"]

    # Also add day_count distribution for richer analysis
    cur.execute("""
        SELECT day_count, COUNT(*) AS cnt
        FROM entities
        GROUP BY day_count
        ORDER BY day_count
    """)
    day_dist = {str(r["day_count"]): r["cnt"] for r in cur.fetchall()}

    return {"type_distribution": type_counts, "day_count_distribution": day_dist}


def check_embedding_coverage(cur: sqlite3.Cursor) -> dict:
    """Check what proportion of entities have vector embeddings."""
    cur.execute("SELECT COUNT(*) AS total FROM entities")
    total = cur.fetchone()["total"]
    if total == 0:
        return {"total_entities": 0, "vector_count": 0, "coverage_ratio": 0.0,
                "status": "ok", "message": "No entities",
                "embedding_model": None, "dimension": None, "orphan_vectors": 0}

    cur.execute("SELECT COUNT(*) AS cnt FROM entity_vectors")
    vector_count = cur.fetchone()["cnt"]
    ratio = round(vector_count / total, 4)

    if ratio < 0.70:
        status = "alert"
    elif ratio < 0.90:
        status = "warning"
    else:
        status = "ok"

    message = (f"Embedding coverage critically low ({ratio:.1%})"
               if ratio < 0.70 else
               f"Embedding coverage below target ({ratio:.1%})"
               if ratio < 0.90 else
               f"Embedding coverage acceptable ({ratio:.1%})")

    # Read embedding metadata (model, dimension) — carried forward from old vector_dimensions
    cur.execute("SELECT key, value FROM embedding_meta")
    meta = {r["key"]: r["value"] for r in cur.fetchall()}

    # Sample first vector to infer dimension
    dimension = None
    cur.execute("SELECT embedding FROM entity_vectors LIMIT 1")
    row = cur.fetchone()
    if row:
        try:
            parsed = json.loads(row["embedding"])
            if isinstance(parsed, list):
                dimension = len(parsed)
        except (json.JSONDecodeError, TypeError):
            pass

    # Check orphan vectors (no matching entity)
    cur.execute("""
        SELECT COUNT(*) AS cnt FROM entity_vectors ev
        WHERE NOT EXISTS (SELECT 1 FROM entities e WHERE e.id = ev.entity_id)
    """)
    orphan_vectors = cur.fetchone()["cnt"]

    return {
        "total_entities": total,
        "vector_count": vector_count,
        "coverage_ratio": ratio,
        "status": status,
        "message": message,
        "embedding_model": meta.get("model", "unknown"),
        "dimension": dimension,
        "orphan_vectors": orphan_vectors,
    }


def check_new_entity_inflow(cur: sqlite3.Cursor) -> dict:
    """Check whether new entities continue to flow into the knowledge base."""
    cur.execute("""
        SELECT COUNT(*) AS cnt FROM entities
        WHERE first_seen >= date('now', '-7 days')
    """)
    new_count = cur.fetchone()["cnt"]

    if new_count == 0:
        status = "alert"
    elif new_count < 5:
        status = "warning"
    else:
        status = "ok"

    message = ("Ingestion pipeline appears stalled — 0 new entities in past 7 days"
               if new_count == 0 else
               f"Low entity inflow — only {new_count} new entities in past 7 days"
               if new_count < 5 else
               f"{new_count} new entities in past 7 days")

    # Last consolidation time (most recent daily_manifest entry)
    cur.execute("SELECT MAX(date) AS last_date FROM daily_manifest")
    row = cur.fetchone()
    last_consolidation = row["last_date"] if row and row["last_date"] else None

    return {
        "new_entities_7d": new_count,
        "status": status,
        "message": message,
        "last_consolidation": last_consolidation,
    }


def check_relation_integrity(cur: sqlite3.Cursor) -> dict:
    """Check referential integrity — dangling from_entity / to_entity references."""
    cur.execute("""
        SELECT COUNT(*) AS cnt FROM relations r
        WHERE NOT EXISTS (SELECT 1 FROM entities e WHERE e.id = r.from_entity)
           OR NOT EXISTS (SELECT 1 FROM entities e WHERE e.id = r.to_entity)
    """)
    broken = cur.fetchone()["cnt"]

    if broken > 10:
        status = "alert"
    elif broken > 0:
        status = "warning"
    else:
        status = "ok"

    message = (f"Critically broken relations ({broken} dangling references)"
               if broken > 10 else
               f"Some broken relations detected ({broken} dangling references)"
               if broken > 0 else
               "All relations have valid entity references")

    return {
        "broken_relations": broken,
        "status": status,
        "message": message,
    }


def check_pending_merges(cur: sqlite3.Cursor) -> dict:
    """Count pending merge reviews."""
    cur.execute("SELECT COUNT(*) AS cnt FROM merge_review WHERE status = 'pending'")
    pending = cur.fetchone()["cnt"]

    # Also get breakdown by kind
    cur.execute("""
        SELECT kind, COUNT(*) AS cnt
        FROM merge_review
        WHERE status = 'pending'
        GROUP BY kind
    """)
    by_kind = {r["kind"]: r["cnt"] for r in cur.fetchall()}

    return {"pending_merges": pending, "pending_by_kind": by_kind}


def check_coverage_days(cur: sqlite3.Cursor) -> dict:
    """Knowledge coverage from daily_manifest."""
    cur.execute("SELECT COUNT(DISTINCT date) AS cnt FROM daily_manifest")
    days = cur.fetchone()["cnt"]

    # Earliest and latest covered dates
    cur.execute("SELECT MIN(date) AS first, MAX(date) AS last FROM daily_manifest")
    row = cur.fetchone()
    return {
        "coverage_days": days,
        "first_coverage": row["first"],
        "last_coverage": row["last"],
    }


def load_previous_report() -> dict | None:
    """Load yesterday's JSON report for trend comparison."""
    yesterday = (date.today() - timedelta(days=1)).isoformat()
    path = os.path.join(REPORTS_DIR, f"{yesterday}.json")
    if os.path.isfile(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return None


def compare_with_previous(current_total: int, prev: dict | None) -> dict:
    """Calculate entity count trend vs previous report."""
    if prev is None or "checks" not in prev:
        return {"trend": "N/A", "previous_total": None, "delta": 0, "delta_pct": 0.0}

    prev_total = prev.get("checks", {}).get("entity_total", {}).get("total_entities")
    if prev_total is None:
        return {"trend": "N/A", "previous_total": None, "delta": 0, "delta_pct": 0.0}

    delta = current_total - prev_total
    pct = round(delta / prev_total * 100, 2) if prev_total else 0.0
    trend = "stable"
    if pct > 10:
        trend = "growing"
    elif pct < -10:
        trend = "shrinking"

    return {"trend": trend, "previous_total": prev_total, "delta": delta, "delta_pct": pct}


# ── Main orchestrator ────────────────────────────────────────────────────────


class HealthCheckError(RuntimeError):
    """The checker could not produce complete evidence."""


def _atomic_write_text(path: str, text: str) -> None:
    """Publish a report atomically without leaving partial JSON behind."""
    parent = os.path.dirname(os.path.abspath(path)) or "."
    ensure_dir(parent)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=parent,
            prefix=".kw-health-",
            suffix=".json",
            delete=False,
        ) as handle:
            temporary = handle.name
            handle.write(text)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary and os.path.exists(temporary):
            try:
                os.remove(temporary)
            except OSError:
                pass


def _run_one_check(
    checks: dict[str, dict],
    checker_errors: list[str],
    name: str,
    function,
    *args,
) -> None:
    """Run one check and fail closed on every exception type."""
    try:
        checks[name] = function(*args)
    except Exception as exc:  # noqa: BLE001 - checker must fail closed
        checker_errors.append(name)
        checks[name] = {
            "status": "alert",
            "error": f"health_check_failed:{name}",
            "error_type": type(exc).__name__,
            "message": "健康检查自身失败：" + name,
        }


def run_health_check(
    dry_run: bool = False,
    *,
    run_id: str | None = None,
    output_path: str | None = None,
    now: datetime | None = None,
) -> str:
    """Execute all checks and return the versioned JSON report.

    ``--output`` selects an invocation-owned report path.  With no explicit
    path, normal runs retain the historical dated report location.  Dry runs
    never create ``REPORTS_DIR`` or any output file.  The returned report may
    still contain a service alert; callers decide exit status from
    ``checks_complete`` and ``checker_errors``.
    """
    current = _aware_now(now)
    report_run_id = run_id or uuid.uuid4().hex
    if not isinstance(report_run_id, str) or not report_run_id.strip():
        raise HealthCheckError("run_id must be a nonempty string")

    # DB open is an evidence failure and cannot be represented as a healthy
    # completed report.  Keep this exception for the CLI, while preserving the
    # string-returning API for successful runs and service alerts.
    try:
        conn = get_db_connection(readonly=True)
    except Exception as exc:  # noqa: BLE001 - fail closed at the boundary
        raise HealthCheckError("database_open_failed") from exc

    cur = conn.cursor()
    checks: dict[str, dict] = {}
    checker_errors: list[str] = []

    def check_dma_at_report_time() -> dict:
        return check_dma_health(now=current)

    check_specs = [
        ("entity_total", check_entity_total, (cur,)),
        ("dma_health", check_dma_at_report_time, ()),
        ("new_entity_inflow", check_new_entity_inflow, (cur,)),
        ("embedding_coverage", check_embedding_coverage, (cur,)),
        ("orphan_ratio", check_orphan_ratio, (cur,)),
        ("cross_day_aggregation", check_cross_day_aggregation, (cur,)),
        ("type_distribution", check_type_distribution, (cur,)),
        ("name_conflicts", check_name_conflicts, (cur,)),
        ("relation_integrity", check_relation_integrity, (cur,)),
        ("pending_merges", check_pending_merges, (cur,)),
        ("coverage_days", check_coverage_days, (cur,)),
    ]
    try:
        for name, function, args in check_specs:
            _run_one_check(checks, checker_errors, name, function, *args)
    finally:
        try:
            conn.close()
        except Exception:  # noqa: BLE001 - close failure is a checker failure
            checker_errors.append("database_close")

    total = checks.get("entity_total", {}).get("total_entities", 0)
    if not isinstance(total, int):
        total = 0

    # Runtime evidence errors are checker errors only when they are not the
    # documented rollout absence.  A fresh running snapshot is incomplete but
    # still a valid observation and exits 0.
    dma_check = checks.get("dma_health", {})
    runtime_error = dma_check.get("runtime_status_error")
    if runtime_error and runtime_error not in {
        dma_status.DmaStatusMissingError.code,
        # A syntactically valid but old snapshot is a DMA scheduling/service
        # alert.  It is incomplete evidence, but it is not a checker crash;
        # keep the named DMA finding in the report and exit 0.
        dma_status.DmaStatusStaleError.code,
    }:
        checker_errors.append("dma_status_evidence")
    if dma_check.get("status_errors") or dma_check.get("runtime_status_errors"):
        checker_errors.append("dma_status_observation")
    # Preserve order while avoiding duplicate failure labels.
    checker_errors = list(dict.fromkeys(checker_errors))

    try:
        prev = load_previous_report()
        trend = compare_with_previous(total, prev)
    except Exception:  # noqa: BLE001 - trend is optional but must not hide failure
        checker_errors.append("trend")
        trend = {"trend": "N/A", "previous_total": None, "delta": 0, "delta_pct": 0.0}

    # Priority: alert > warning > info/ok.  Cross-day is deliberately info and
    # therefore cannot alter this aggregate.
    overall = "ok"
    severity = {"ok": 0, "info": 0, "unknown": 1, "warning": 1, "alert": 2}
    for check in checks.values():
        check_status = check.get("status")
        if severity.get(check_status, 2 if "error" in check else 0) > severity[overall]:
            overall = check_status if check_status in severity else "alert"
    if checker_errors:
        overall = "alert"

    dma_evidence_complete = bool(dma_check.get("evidence_complete", False))
    checks_complete = not checker_errors and dma_evidence_complete
    report = {
        "schema_version": "kw-health-report-v1",
        "run_id": report_run_id,
        "report_date": current.date().isoformat(),
        "generated_at": current.isoformat(timespec="seconds"),
        "overall_status": overall,
        "checks_complete": checks_complete,
        "checks": checks,
        "trend": trend,
        "checker_errors": checker_errors,
        "thresholds_evaluated": {
            "dma_runtime_status_age > 90m": "dma_status_stale" in dma_check.get("alert_codes", [])
                or dma_check.get("runtime_status_error") == dma_status.DmaStatusStaleError.code,
            "dma_pending_or_reconcile_overdue > 24h": any(
                code in dma_check.get("alert_codes", [])
                for code in ("dma_pending_overdue", "dma_reconcile_overdue")
            ),
            "dma_health_log_stale > 24h": False,
            "dma_health_errors > 10": False,
            "dma_health_memory_missing": False,
            "dma_health_checkpoint_stale > 48h": False,
            "new_entity_inflow == 0": checks.get("new_entity_inflow", {}).get("status") == "alert",
            "embedding_coverage < 70%": checks.get("embedding_coverage", {}).get("status") == "alert",
            "orphan_ratio > 40%": checks.get("orphan_ratio", {}).get("status") == "alert",
            "cross_day_day2 > 50%": False,
            "name_conflicts > 5": checks.get("name_conflicts", {}).get("status") == "warning",
            "broken_relations > 0": checks.get("relation_integrity", {}).get("status") in ("warning", "alert"),
            "entity_delta > ±10%": trend.get("trend", "N/A") in ("growing", "shrinking"),
        },
    }

    json_text = json.dumps(report, ensure_ascii=False, indent=2)

    if not dry_run:
        report_path = output_path or os.path.join(
            REPORTS_DIR, f"{current.date().isoformat()}.json"
        )
        _atomic_write_text(report_path, json_text)
        if output_path is None:
            print(f"Report written to {report_path}")

    return json_text


# ── CLI ──────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description="Knowledge Weaver Health Check")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print JSON to stdout only; do not write a report file (legacy alias for --json).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the structured JSON report to stdout.",
    )
    parser.add_argument(
        "--run-id",
        help="Caller-supplied invocation identifier echoed in the report.",
    )
    parser.add_argument(
        "--output",
        dest="output_path",
        help="Write this invocation's report atomically to the exact path.",
    )
    args = parser.parse_args()
    dry_run = args.dry_run or (args.json and not args.output_path)
    try:
        json_text = run_health_check(
            dry_run=dry_run,
            run_id=args.run_id,
            output_path=args.output_path,
        )
    except HealthCheckError as exc:
        print(f"Health check failed: {exc}", file=sys.stderr)
        return 1

    if args.json or args.dry_run:
        print(json_text)
    try:
        report = json.loads(json_text)
    except json.JSONDecodeError:
        print("Health check failed: invalid generated JSON", file=sys.stderr)
        return 1
    # A complete report can legitimately have overall alert/warning because a
    # service is unhealthy.  Incomplete checker/evidence reports are failures;
    # rollout-missing status and fresh-running status remain valid incomplete
    # observations and intentionally exit 0.
    if report.get("checker_errors"):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
