"""Persistent job queue with leases, heartbeats, fencing and crash recovery (T06).

Rules (engineering §5, JOB-01/AC-13/14/39/52):

* jobs are keyed by ``(kind, input_key, input_version)`` so re-processing the
  same input/version does not duplicate work or model cost;
* claiming issues a fresh ``lease_token``; only the holder can heartbeat or
  commit; every attempt is journaled in ``job_attempts``;
* an expired lease may be reclaimed, but the old worker's commit is rejected by
  token, input version and an upstream generation fence — late results cannot
  resurrect stale state;
* a heartbeat miss alone never marks business work failed; recovery marks the
  attempt interrupted and schedules a bounded retry;
* retries are bounded (default 3) with exponential backoff; permanent errors stop
  automatic retries and keep a queryable failed state;
* commit mutation runs in the same transaction as the job state change.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import sqlite3
import uuid
from typing import Callable

from ..store.db import Store, utc_now

JOB_KINDS = ("extract", "project", "embed", "cleanup", "export")
JOB_STATES = ("pending", "running", "retry_wait", "succeeded", "failed")
DEFAULT_BACKOFF = (10, 60, 300)


class JobError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def _iso(moment: str, seconds: int) -> str:
    parsed = _dt.datetime.fromisoformat(moment.replace("Z", "+00:00"))
    return (
        (parsed + _dt.timedelta(seconds=seconds))
        .astimezone(_dt.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _job_id(kind: str, input_key: str, input_version: int) -> str:
    digest = hashlib.sha256(f"{kind}|{input_key}|{input_version}".encode("utf-8")).hexdigest()
    return "job:" + digest[:40]


def enqueue_in_transaction(
    conn: sqlite3.Connection,
    kind: str,
    input_key: str,
    *,
    input_version: int = 1,
    run_after: str | None = None,
    model_version: str | None = None,
    prompt_version: str | None = None,
    policy_version: str | None = None,
) -> dict:
    """Enqueue on the caller's connection/transaction (e.g. claim writes)."""
    if kind not in JOB_KINDS:
        raise JobError("INVALID_REQUEST", f"unsupported job kind {kind!r}")
    if not isinstance(input_key, str) or not input_key:
        raise JobError("INVALID_REQUEST", "input_key must be a non-empty string")
    job_id = _job_id(kind, input_key, input_version)
    now = utc_now()
    conn.execute(
        "INSERT OR IGNORE INTO jobs (job_id, kind, input_key, input_version, state, "
        "run_after, model_version, prompt_version, policy_version, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, 'pending', ?, ?, ?, ?, ?, ?)",
        (
            job_id,
            kind,
            input_key,
            int(input_version),
            run_after,
            model_version,
            prompt_version,
            policy_version,
            now,
            now,
        ),
    )
    row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
    return dict(row)


def enqueue(
    store: Store,
    kind: str,
    input_key: str,
    *,
    input_version: int = 1,
    run_after: str | None = None,
    model_version: str | None = None,
    prompt_version: str | None = None,
    policy_version: str | None = None,
) -> dict:
    with store.transaction() as conn:
        return enqueue_in_transaction(
            conn,
            kind,
            input_key,
            input_version=input_version,
            run_after=run_after,
            model_version=model_version,
            prompt_version=prompt_version,
            policy_version=policy_version,
        )


def claim(
    store: Store,
    *,
    worker_id: str,
    kinds: tuple[str, ...] | None = None,
    lease_seconds: int = 60,
    limit: int = 1,
    now: str | None = None,
) -> list[dict]:
    """Claim ready jobs and journal a new attempt for each."""
    moment = now or utc_now()
    claimed: list[dict] = []
    with store.transaction() as conn:
        params: list = [moment]
        kind_clause = ""
        if kinds:
            placeholders = ",".join("?" for _ in kinds)
            kind_clause = f" AND kind IN ({placeholders})"
            params.extend(kinds)
        rows = conn.execute(
            "SELECT * FROM jobs WHERE state IN ('pending', 'retry_wait') "
            "AND (run_after IS NULL OR run_after <= ?)" + kind_clause +
            " ORDER BY created_at LIMIT ?",
            (*params, limit),
        ).fetchall()
        for row in rows:
            token = "lease:" + uuid.uuid4().hex
            attempt = int(row["attempts"]) + 1
            conn.execute(
                "UPDATE jobs SET state = 'running', attempts = ?, lease_token = ?, "
                "lease_owner = ?, heartbeat_at = ?, lease_expires_at = ?, updated_at = ? "
                "WHERE job_id = ?",
                (
                    attempt,
                    token,
                    worker_id,
                    moment,
                    _iso(moment, lease_seconds),
                    moment,
                    row["job_id"],
                ),
            )
            conn.execute(
                "INSERT INTO job_attempts (job_id, attempt, lease_token, started_at) "
                "VALUES (?, ?, ?, ?)",
                (row["job_id"], attempt, token, moment),
            )
            claimed.append(
                {
                    **dict(row),
                    "state": "running",
                    "attempts": attempt,
                    "lease_token": token,
                    "lease_owner": worker_id,
                    "heartbeat_at": moment,
                    "lease_expires_at": _iso(moment, lease_seconds),
                }
            )
    return claimed


def heartbeat(store: Store, job_id: str, lease_token: str, *, lease_seconds: int = 60, now: str | None = None) -> bool:
    moment = now or utc_now()
    with store.transaction() as conn:
        cursor = conn.execute(
            "UPDATE jobs SET heartbeat_at = ?, lease_expires_at = ?, updated_at = ? "
            "WHERE job_id = ? AND state = 'running' AND lease_token = ?",
            (moment, _iso(moment, lease_seconds), moment, job_id, lease_token),
        )
        return cursor.rowcount > 0


def fail(
    store: Store,
    job_id: str,
    lease_token: str,
    error_code: str,
    *,
    retryable: bool = True,
    max_attempts: int = 3,
    backoff: tuple[int, ...] = DEFAULT_BACKOFF,
    now: str | None = None,
    detail: str | None = None,
) -> dict:
    moment = now or utc_now()
    with store.transaction() as conn:
        row = conn.execute(
            "SELECT * FROM jobs WHERE job_id = ? AND state = 'running' AND lease_token = ?",
            (job_id, lease_token),
        ).fetchone()
        if row is None:
            raise JobError("STALE_CURSOR", "job lease lost; result not committed")
        attempts = int(row["attempts"])
        if retryable and attempts < max_attempts:
            delay = backoff[min(attempts - 1, len(backoff) - 1)]
            state = "retry_wait"
            run_after = _iso(moment, delay)
        else:
            state = "failed"
            run_after = None
        conn.execute(
            "UPDATE jobs SET state = ?, run_after = ?, lease_token = NULL, lease_owner = NULL, "
            "last_error_code = ?, updated_at = ? WHERE job_id = ?",
            (state, run_after, error_code, moment, job_id),
        )
        conn.execute(
            "UPDATE job_attempts SET finished_at = ?, outcome = ?, error_code = ?, detail = ? "
            "WHERE job_id = ? AND lease_token = ?",
            (moment, state, error_code, detail, job_id, lease_token),
        )
    return {"job_id": job_id, "state": state, "attempts": attempts}


def commit(
    store: Store,
    job_id: str,
    lease_token: str,
    *,
    expected_input_version: int,
    mutate: Callable[[sqlite3.Connection], None] | None = None,
    fence_check: Callable[[sqlite3.Connection], tuple[bool, str]] | None = None,
    now: str | None = None,
) -> dict:
    """Fenced commit: job state + downstream mutation in one transaction."""
    moment = now or utc_now()
    stale_reason: str | None = None
    with store.transaction() as conn:
        row = conn.execute(
            "SELECT * FROM jobs WHERE job_id = ? AND state = 'running' AND lease_token = ?",
            (job_id, lease_token),
        ).fetchone()
        if row is None:
            raise JobError("STALE_CURSOR", "job lease lost; result not committed")
        if int(row["input_version"]) != int(expected_input_version):
            stale_reason = "job input version changed; stale result rejected"
        elif fence_check is not None:
            ok, reason = fence_check(conn)
            if not ok:
                stale_reason = "generation fence rejected the stale result: " + str(reason)[:400]
        if stale_reason is not None:
            # Persist the fencing decision, then raise after the transaction so the
            # failed state is not rolled back by the exception.
            conn.execute(
                "UPDATE jobs SET state = 'failed', lease_token = NULL, lease_owner = NULL, "
                "last_error_code = 'STALE_CURSOR', updated_at = ? WHERE job_id = ?",
                (moment, job_id),
            )
            conn.execute(
                "UPDATE job_attempts SET finished_at = ?, outcome = 'stale', error_code = 'STALE_CURSOR', "
                "detail = ? WHERE job_id = ? AND lease_token = ?",
                (moment, stale_reason[:500], job_id, lease_token),
            )
        else:
            if mutate is not None:
                mutate(conn)
            conn.execute(
                "UPDATE jobs SET state = 'succeeded', lease_token = NULL, lease_owner = NULL, "
                "last_error_code = NULL, updated_at = ? WHERE job_id = ?",
                (moment, job_id),
            )
            conn.execute(
                "UPDATE job_attempts SET finished_at = ?, outcome = 'succeeded' "
                "WHERE job_id = ? AND lease_token = ?",
                (moment, job_id, lease_token),
            )
    if stale_reason is not None:
        raise JobError("STALE_CURSOR", stale_reason)
    return {"job_id": job_id, "state": "succeeded"}


def recover(store: Store, *, now: str | None = None, max_attempts: int = 3, backoff: tuple[int, ...] = DEFAULT_BACKOFF) -> dict:
    """Reclaim jobs whose lease expired; never treat a heartbeat miss as business failure."""
    moment = now or utc_now()
    counts = {"retry_wait": 0, "failed": 0, "interrupted_attempts": 0}
    with store.transaction() as conn:
        rows = conn.execute(
            "SELECT * FROM jobs WHERE state = 'running' AND lease_expires_at IS NOT NULL "
            "AND lease_expires_at < ?",
            (moment,),
        ).fetchall()
        for row in rows:
            attempts = int(row["attempts"])
            if attempts < max_attempts:
                delay = backoff[min(attempts - 1, len(backoff) - 1)]
                conn.execute(
                    "UPDATE jobs SET state = 'retry_wait', run_after = ?, lease_token = NULL, "
                    "lease_owner = NULL, last_error_code = 'LEASE_EXPIRED', updated_at = ? "
                    "WHERE job_id = ?",
                    (_iso(moment, delay), moment, row["job_id"]),
                )
                counts["retry_wait"] += 1
            else:
                conn.execute(
                    "UPDATE jobs SET state = 'failed', lease_token = NULL, lease_owner = NULL, "
                    "last_error_code = 'LEASE_EXPIRED', updated_at = ? WHERE job_id = ?",
                    (moment, row["job_id"]),
                )
                counts["failed"] += 1
            cursor = conn.execute(
                "UPDATE job_attempts SET finished_at = ?, outcome = 'interrupted' "
                "WHERE job_id = ? AND lease_token = ? AND finished_at IS NULL",
                (moment, row["job_id"], row["lease_token"]),
            )
            counts["interrupted_attempts"] += cursor.rowcount
    return counts


def status_counts(store: Store) -> dict:
    with store.read() as conn:
        rows = conn.execute(
            "SELECT kind, state, COUNT(*) AS c FROM jobs GROUP BY kind, state"
        ).fetchall()
    result: dict[str, dict[str, int]] = {}
    for row in rows:
        result.setdefault(row["kind"], {})[row["state"]] = row["c"]
    return result
