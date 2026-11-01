#!/usr/bin/env python3
"""Read and classify DMA's versioned runtime status.

This module is intentionally standalone.  The deployment copies the ``scripts``
directory without installing the Knowledge Weaver package, so the health check
must not import ``knowledge_weaver`` or mutate any DMA state.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from typing import Any


SCHEMA_VERSION = "dma-runtime-status-v1"
DEFAULT_STATUS_PATH = (
    "/home/openclaw/.openclaw/skills/daily-memory-archiver/config/"
    ".runtime_status.json"
)
DEFAULT_EXPECTED_INTERVAL_SECONDS = 30 * 60
DEFAULT_STATUS_TIMEOUT_SECONDS = 90 * 60
DEFAULT_BACKLOG_TIMEOUT_SECONDS = 24 * 60 * 60
DEFAULT_CLOCK_SKEW_SECONDS = 5 * 60

OUTCOMES = {
    "running",
    "archived",
    "idle",
    "noise_only",
    "deferred",
    "failed",
    "partial",
}
FAILURE_OPERATIONS = ("storage", "summary", "archive")
CODE_PATTERN = r"^[a-z][a-z0-9_.-]*$"
RUN_ID_PATTERN = r"^\S+$"


class DmaStatusError(RuntimeError):
    """Base class for errors in the status evidence."""

    code = "dma_status_error"


class DmaStatusMissingError(DmaStatusError):
    """The status file is not present or is not a regular file."""

    code = "dma_status_missing"


class DmaStatusUnreadableError(DmaStatusError):
    """The status file could not be read."""

    code = "dma_status_unreadable"


class DmaStatusMalformedError(DmaStatusError):
    """The status file is not valid status-v1 JSON."""

    code = "dma_status_malformed"


class DmaStatusStaleError(DmaStatusError):
    """The status evidence is too old for the configured monitoring window."""

    code = "dma_status_stale"

    def __init__(self, message: str, *, status: dict[str, Any] | None = None,
                 age_seconds: float | None = None) -> None:
        super().__init__(message)
        self.status = status
        self.age_seconds = age_seconds


def get_status_path(path: str | None = None) -> str:
    """Return the explicit path, or KW's environment/configured default."""

    if path:
        return path
    return os.environ.get("KW_DMA_STATUS_PATH", DEFAULT_STATUS_PATH)


def _now(value: datetime | None = None) -> datetime:
    """Return an aware timestamp suitable for comparisons."""

    current = value or datetime.now(timezone.utc)
    if current.tzinfo is None or current.utcoffset() is None:
        # Existing callers used naive local datetimes.  Treat those as local
        # time for compatibility while keeping all returned comparisons aware.
        current = current.astimezone()
    return current


def _parse_timestamp(value: Any, field: str) -> datetime | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise DmaStatusMalformedError(f"{field} must be an ISO8601 string or null")
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise DmaStatusMalformedError(f"{field} is not valid ISO8601") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise DmaStatusMalformedError(f"{field} must include a timezone offset")
    return parsed


def _ensure_nonnegative_int(value: Any, field: str, *, nullable: bool = True) -> None:
    if value is None and nullable:
        return
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        suffix = " or null" if nullable else ""
        raise DmaStatusMalformedError(f"{field} must be a nonnegative integer{suffix}")


def _ensure_string(value: Any, field: str, *, nullable: bool = False) -> None:
    if value is None and nullable:
        return
    if not isinstance(value, str) or not value.strip():
        suffix = " or null" if nullable else ""
        raise DmaStatusMalformedError(f"{field} must be a nonempty string{suffix}")


def _validate_failures(value: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(value, dict) or set(value) != set(FAILURE_OPERATIONS):
        raise DmaStatusMalformedError(
            "failures must contain exactly storage, summary and archive"
        )

    validated: dict[str, dict[str, Any]] = {}
    for operation in FAILURE_OPERATIONS:
        entry = value[operation]
        if not isinstance(entry, dict) or set(entry) != {
            "consecutive",
            "last_failed_at",
            "last_recovered_at",
        }:
            raise DmaStatusMalformedError(
                f"failures.{operation} has an invalid field set"
            )
        _ensure_nonnegative_int(entry["consecutive"], f"failures.{operation}.consecutive", nullable=False)
        failed = _parse_timestamp(
            entry["last_failed_at"], f"failures.{operation}.last_failed_at"
        )
        recovered = _parse_timestamp(
            entry["last_recovered_at"], f"failures.{operation}.last_recovered_at"
        )
        # A recovered failure may be followed by a new unresolved failure.  In
        # that legitimate sequence the old recovery precedes the latest
        # failure; only a zero-consecutive entry must have recovery at or after
        # its latest failure.
        if (
            failed
            and recovered
            and recovered < failed
            and entry["consecutive"] == 0
        ):
            raise DmaStatusMalformedError(
                f"failures.{operation}.last_recovered_at precedes last_failed_at"
            )
        validated[operation] = {
            "consecutive": entry["consecutive"],
            "last_failed_at": entry["last_failed_at"],
            "last_recovered_at": entry["last_recovered_at"],
            "_last_failed": failed,
            "_last_recovered": recovered,
        }
    return validated


def validate_status(
    payload: Any,
    *,
    now: datetime | None = None,
    clock_skew_seconds: int = DEFAULT_CLOCK_SKEW_SECONDS,
) -> dict[str, Any]:
    """Validate a decoded DMA runtime status and return a copy.

    Validation is strict for all contract fields.  The v1 producer publishes
    an exact field set, so unsupported extra fields are rejected rather than
    silently interpreted by a monitoring process.
    """

    if not isinstance(payload, dict):
        raise DmaStatusMalformedError("status root must be a JSON object")
    required = {
        "schema_version",
        "run_id",
        "started_at",
        "observed_at",
        "finished_at",
        "outcome",
        "reason",
        "last_completed_at",
        "last_archived_at",
        "pending_count",
        "oldest_pending_at",
        "pending_reconcile_count",
        "oldest_reconcile_at",
        "checkpoint_progress_at",
        "failures",
        "status_errors",
    }
    missing = sorted(required.difference(payload))
    extra = sorted(set(payload).difference(required))
    if missing or extra:
        details: list[str] = []
        if missing:
            details.append("missing=" + ",".join(missing))
        if extra:
            details.append("extra=" + ",".join(extra))
        raise DmaStatusMalformedError(
            "status fields are not exact: " + " ".join(details)
        )

    if payload["schema_version"] != SCHEMA_VERSION:
        raise DmaStatusMalformedError("unsupported DMA status schema_version")
    _ensure_string(payload["run_id"], "run_id")
    _ensure_string(payload["reason"], "reason")
    # DMA reasons are stable codes, not arbitrary log/private text.
    import re

    if not re.fullmatch(RUN_ID_PATTERN, payload["run_id"]):
        raise DmaStatusMalformedError("run_id must not contain whitespace")
    if not re.fullmatch(CODE_PATTERN, payload["reason"]):
        raise DmaStatusMalformedError("reason must be a stable code")
    if not isinstance(payload["outcome"], str) or payload["outcome"] not in OUTCOMES:
        raise DmaStatusMalformedError("unsupported DMA status outcome")

    started = _parse_timestamp(payload["started_at"], "started_at")
    observed = _parse_timestamp(payload["observed_at"], "observed_at")
    finished = _parse_timestamp(payload["finished_at"], "finished_at")
    last_completed = _parse_timestamp(payload["last_completed_at"], "last_completed_at")
    last_archived = _parse_timestamp(payload["last_archived_at"], "last_archived_at")
    oldest_pending = _parse_timestamp(payload["oldest_pending_at"], "oldest_pending_at")
    oldest_reconcile = _parse_timestamp(payload["oldest_reconcile_at"], "oldest_reconcile_at")
    checkpoint_progress = _parse_timestamp(
        payload["checkpoint_progress_at"], "checkpoint_progress_at"
    )

    if started is None or observed is None:
        # These fields are non-null by the contract; _parse_timestamp accepts
        # null for shared helpers, so enforce that fact here.
        raise DmaStatusMalformedError("started_at and observed_at must not be null")
    if payload["outcome"] == "running" and finished is not None:
        raise DmaStatusMalformedError("running status must have finished_at=null")
    if payload["outcome"] != "running" and finished is None:
        raise DmaStatusMalformedError("terminal status must have finished_at")

    if started > observed:
        raise DmaStatusMalformedError("started_at is after observed_at")
    if finished and (finished < started or finished > observed):
        raise DmaStatusMalformedError("finished_at is outside the run interval")

    _ensure_nonnegative_int(payload["pending_count"], "pending_count")
    _ensure_nonnegative_int(payload["pending_reconcile_count"], "pending_reconcile_count")
    if payload["pending_count"] == 0 and oldest_pending is not None:
        raise DmaStatusMalformedError("oldest_pending_at must be null when pending_count=0")
    if payload["pending_reconcile_count"] == 0 and oldest_reconcile is not None:
        raise DmaStatusMalformedError(
            "oldest_reconcile_at must be null when pending_reconcile_count=0"
        )

    if not isinstance(payload["status_errors"], list) or any(
        not isinstance(item, str) or not re.fullmatch(CODE_PATTERN, item)
        for item in payload["status_errors"]
    ):
        raise DmaStatusMalformedError("status_errors must be a list of nonempty strings")
    if len(set(payload["status_errors"])) != len(payload["status_errors"]):
        raise DmaStatusMalformedError("status_errors must not contain duplicates")
    failures = _validate_failures(payload["failures"])

    # Cumulative producer-controlled timestamps must be within this snapshot.
    # A failure can legitimately recur after a prior recovery; the failure
    # validator preserves that sequence when ``consecutive`` is nonzero.
    for field, timestamp in (
        ("last_completed_at", last_completed),
        ("last_archived_at", last_archived),
        ("checkpoint_progress_at", checkpoint_progress),
    ):
        if timestamp and timestamp > observed:
            raise DmaStatusMalformedError(f"{field} is after observed_at")
    for operation in FAILURE_OPERATIONS:
        for field, timestamp in (
            ("last_failed_at", failures[operation]["_last_failed"]),
            ("last_recovered_at", failures[operation]["_last_recovered"]),
        ):
            if timestamp and timestamp > observed:
                raise DmaStatusMalformedError(
                    f"failures.{operation}.{field} is after observed_at"
                )

    current = _now(now)
    skew = timedelta(seconds=max(0, clock_skew_seconds))
    if observed > current + skew:
        raise DmaStatusMalformedError("observed_at is in the future")

    # Source timestamps are collected as part of the same pending snapshot and
    # must not describe an event after that snapshot.  The clock-skew allowance
    # applies only when comparing the snapshot with the consumer's current
    # clock below; it must not hide an internally inconsistent record.
    for field, timestamp in (
        ("oldest_pending_at", oldest_pending),
        ("oldest_reconcile_at", oldest_reconcile),
    ):
        if timestamp and timestamp > observed:
            raise DmaStatusMalformedError(f"{field} is after observed_at")

    # A producer can retain historical timestamps, but no current snapshot may
    # claim a future event.  Keep this check explicit for all optional times.
    for field, timestamp in (
        ("started_at", started),
        ("finished_at", finished),
        ("last_completed_at", last_completed),
        ("last_archived_at", last_archived),
        ("oldest_pending_at", oldest_pending),
        ("oldest_reconcile_at", oldest_reconcile),
        ("checkpoint_progress_at", checkpoint_progress),
    ):
        if timestamp and timestamp > current + skew:
            raise DmaStatusMalformedError(f"{field} is in the future")
    for operation in FAILURE_OPERATIONS:
        for field, timestamp in (
            ("last_failed_at", failures[operation]["_last_failed"]),
            ("last_recovered_at", failures[operation]["_last_recovered"]),
        ):
            if timestamp and timestamp > current + skew:
                raise DmaStatusMalformedError(
                    f"failures.{operation}.{field} is in the future"
                )

    result = dict(payload)
    result["_started"] = started
    result["_observed"] = observed
    result["_finished"] = finished
    result["_last_completed"] = last_completed
    result["_last_archived"] = last_archived
    result["_oldest_pending"] = oldest_pending
    result["_oldest_reconcile"] = oldest_reconcile
    result["_checkpoint_progress"] = checkpoint_progress
    result["_failures"] = failures
    return result


def read_dma_status(
    path: str | None = None,
    *,
    now: datetime | None = None,
    status_timeout_seconds: int = DEFAULT_STATUS_TIMEOUT_SECONDS,
    clock_skew_seconds: int = DEFAULT_CLOCK_SKEW_SECONDS,
) -> dict[str, Any]:
    """Read, validate and freshness-check the DMA status file.

    ``DmaStatusMissingError`` is intentionally distinct from malformed and
    stale evidence.  KW treats rollout absence as unknown while treating any
    other evidence failure as a checker alert.
    """

    status_path = get_status_path(path)
    try:
        exists = os.path.exists(status_path)
        regular_file = os.path.isfile(status_path)
    except OSError as exc:
        raise DmaStatusUnreadableError(
            f"DMA status path cannot be inspected: {status_path}"
        ) from exc
    if not exists:
        raise DmaStatusMissingError(f"DMA status file not found: {status_path}")
    if not regular_file:
        raise DmaStatusUnreadableError(f"DMA status path is not a regular file: {status_path}")
    try:
        with open(status_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except FileNotFoundError as exc:
        raise DmaStatusMissingError(f"DMA status file not found: {status_path}") from exc
    except (OSError, UnicodeError) as exc:
        raise DmaStatusUnreadableError(f"DMA status file unreadable: {status_path}") from exc
    except json.JSONDecodeError as exc:
        raise DmaStatusMalformedError("DMA status file is not valid JSON") from exc

    validated = validate_status(payload, now=now, clock_skew_seconds=clock_skew_seconds)
    current = _now(now)
    age_seconds = (current - validated["_observed"]).total_seconds()
    if age_seconds < 0:
        age_seconds = 0.0
    if age_seconds > status_timeout_seconds:
        raise DmaStatusStaleError(
            f"DMA status is stale ({age_seconds / 3600:.1f}h)",
            status=validated,
            age_seconds=age_seconds,
        )
    validated["age_seconds"] = age_seconds
    validated["stale"] = False
    return validated


# Short aliases make the consumer API easy to discover without coupling callers
# to the exception class names.
read_status = read_dma_status
validate_dma_status = validate_status


def _age_seconds(timestamp: datetime | None, current: datetime) -> float | None:
    if timestamp is None:
        return None
    return max(0.0, (current - timestamp).total_seconds())


def _recovered_operations(status: dict[str, Any]) -> list[str]:
    recovered: list[str] = []
    for operation in FAILURE_OPERATIONS:
        failure = status["_failures"][operation]
        failed = failure["_last_failed"]
        recovered_at = failure["_last_recovered"]
        if (
            failure["consecutive"] == 0
            and failed is not None
            and recovered_at is not None
            and recovered_at >= failed
        ):
            recovered.append(operation)
    return recovered


def classify_status(
    status: dict[str, Any],
    *,
    now: datetime | None = None,
    expected_interval_seconds: int = DEFAULT_EXPECTED_INTERVAL_SECONDS,
    status_timeout_seconds: int = DEFAULT_STATUS_TIMEOUT_SECONDS,
    backlog_timeout_seconds: int = DEFAULT_BACKLOG_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Classify validated status for the KW health policy.

    ``status`` is expected to come from :func:`read_dma_status` or
    :func:`validate_status`.  The returned ``status`` keeps KW's historical
    ``ok``/``warning``/``alert`` vocabulary.  ``level`` distinguishes an
    informational running/deferred state without turning it into an alert.
    """

    current = _now(now)
    observed = status["_observed"]
    age_seconds = float(status.get("age_seconds", _age_seconds(observed, current) or 0.0))
    outcome = status["outcome"]
    level = "ok"
    state = outcome
    messages: list[str] = []
    alert_codes: list[str] = []
    warning_codes: list[str] = []
    recovered = _recovered_operations(status)

    def mark_warning(code: str, message: str) -> None:
        nonlocal level
        if level != "alert":
            level = "warning"
        warning_codes.append(code)
        messages.append(message)

    def mark_alert(code: str, message: str) -> None:
        nonlocal level
        level = "alert"
        alert_codes.append(code)
        messages.append(message)

    if age_seconds > status_timeout_seconds:
        mark_alert(
            "dma_status_stale",
            f"DMA runtime status超过 {status_timeout_seconds / 3600:.1f}h 未更新",
        )
        state = "stale"
    elif age_seconds > expected_interval_seconds:
        mark_warning(
            "dma_schedule_delayed",
            f"DMA 最近一次状态距今 {age_seconds / 60:.1f} 分钟，超过预期运行间隔",
        )

    if outcome == "running":
        # Running is informational while fresh.  A running snapshot is not a
        # completed observation and therefore cannot clear an old alert.
        if state != "stale":
            state = "running"
        messages.append("DMA 当前正在运行，等待终态结果")
    elif outcome in {"failed", "partial"}:
        mark_alert(
            "dma_run_failed" if outcome == "failed" else "dma_run_partial",
            f"DMA 最近一次运行结果为 {outcome}",
        )
    elif outcome == "deferred":
        messages.append(f"DMA 本轮延后处理（{status['reason']}）")
    elif outcome in {"idle", "noise_only", "archived"}:
        messages.append(f"DMA 最近一次运行结果为 {outcome}")

    for operation in FAILURE_OPERATIONS:
        failure = status["_failures"][operation]
        if failure["consecutive"] > 0:
            mark_alert(
                f"dma_{operation}_failure_unresolved",
                f"DMA {operation} 操作仍有 {failure['consecutive']} 次连续失败",
            )

    if status["status_errors"]:
        mark_alert(
            "dma_status_publication_error",
            "DMA runtime status 包含状态发布错误",
        )

    pending = status["pending_count"]
    pending_oldest = status["_oldest_pending"]
    reconcile = status["pending_reconcile_count"]
    reconcile_oldest = status["_oldest_reconcile"]

    # A terminal snapshot with an unknown count cannot certify a healthy
    # no-backlog state.  Keep this as a visible warning for valid service
    # records; publication/status_errors decide whether it is a checker
    # evidence failure.  Running snapshots are already informational.
    if outcome != "running" and pending is None:
        mark_warning(
            "dma_pending_count_unknown",
            "DMA 待归档消息数量未知，无法确认是否需要创建 Memory 文件",
        )
    if outcome != "running" and reconcile is None:
        mark_warning(
            "dma_reconcile_count_unknown",
            "DMA 待补 sidecar 数量未知，无法确认对账是否完成",
        )

    def check_backlog(
        count: int | None,
        oldest: datetime | None,
        label: str,
        overdue_code: str,
        unknown_code: str,
    ) -> None:
        if count is None or count <= 0:
            return
        age = _age_seconds(oldest, current)
        if age is None:
            mark_alert(unknown_code, f"DMA {label} 数量已知但最早时间未知，无法判断是否超时")
            return
        if age > backlog_timeout_seconds:
            mark_alert(
                overdue_code,
                f"DMA {label} 已积压 {age / 3600:.1f}h，超过 {backlog_timeout_seconds / 3600:.1f}h",
            )

    check_backlog(
        pending,
        pending_oldest,
        "待归档消息",
        "dma_pending_overdue",
        "dma_pending_age_unknown",
    )
    check_backlog(
        reconcile,
        reconcile_oldest,
        "待补 sidecar",
        "dma_reconcile_overdue",
        "dma_reconcile_age_unknown",
    )
    if reconcile is not None and reconcile > 0 and "dma_reconcile_overdue" not in alert_codes:
        mark_warning("dma_reconcile_pending", f"DMA 有 {reconcile} 个待补 sidecar")

    if pending is not None and pending > 0 and pending_oldest is None:
        # ``check_backlog`` already marked an alert.  Keep the explicit state
        # visible for callers that only inspect the result fields.
        state = "backlog_unknown"
    elif pending is not None and pending > 0:
        state = "backlog"

    if recovered and level != "alert":
        messages.append("DMA 相关故障已由对应操作成功恢复：" + ", ".join(recovered))

    if not messages:
        messages.append("DMA runtime status healthy")
    # Running is informational; deferred below the deadline and a clean idle /
    # noise-only run are complete observations only when all required evidence
    # is present and no pending backlog exists.
    evidence_complete = outcome != "running" and not status["status_errors"]
    if pending is None or reconcile is None:
        evidence_complete = False
    if level == "alert" and outcome in {"failed", "partial"}:
        # A terminal failure is a complete, valid observation; callers should
        # exit 0 for it and surface the service alert in the report.
        evidence_complete = evidence_complete and True

    return {
        "status": level,
        "level": "info" if level == "ok" and outcome in {"running", "deferred"} else level,
        "state": state,
        "message": "; ".join(messages),
        "alert_codes": alert_codes,
        "warning_codes": warning_codes,
        "recovered_operations": recovered,
        "age_seconds": age_seconds,
        "expected_interval_seconds": expected_interval_seconds,
        "status_timeout_seconds": status_timeout_seconds,
        "backlog_timeout_seconds": backlog_timeout_seconds,
        "evidence_complete": evidence_complete,
    }


def public_status(status: dict[str, Any]) -> dict[str, Any]:
    """Return safe contract fields for inclusion in a KW report.

    Internal parsed datetime objects are deliberately excluded so the report
    remains JSON serializable and contains only producer-owned evidence.
    """

    fields = (
        "schema_version",
        "run_id",
        "started_at",
        "observed_at",
        "finished_at",
        "outcome",
        "reason",
        "last_completed_at",
        "last_archived_at",
        "pending_count",
        "oldest_pending_at",
        "pending_reconcile_count",
        "oldest_reconcile_at",
        "checkpoint_progress_at",
        "failures",
        "status_errors",
    )
    result = {field: status[field] for field in fields}
    result["age_seconds"] = status.get("age_seconds")
    result["stale"] = bool(status.get("stale", False))
    return result
