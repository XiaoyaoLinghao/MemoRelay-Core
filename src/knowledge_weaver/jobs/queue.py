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


from .repair_versions import EXTRACTION_UPGRADES, MAX_REPAIR_DEPTH, allowed_upgrade


def enqueue_extraction_repair(store: Store, old_job_id: str, *, prompt_version: str = "prompt:7") -> dict:
    """Explicitly upgrade one failed source from v3--v6 to v7, once.

    Preserve the old failed state, error, attempts and timestamps. The new job
    gets a new input version and a durable parent pointer so a later successful
    commit can prove recovery of this exact unit, including after a restart.
    """
    if prompt_version not in EXTRACTION_UPGRADES:
        raise JobError("INVALID_REQUEST", "only registered extraction upgrades are supported")
    with store.transaction() as conn:
        old = conn.execute("SELECT * FROM jobs WHERE job_id=?", (old_job_id,)).fetchone()
        if old is None or old["kind"] != "extract" or old["state"] != "failed" or old["last_error_code"] is None:
            raise JobError("INVALID_REQUEST", "repair requires a recorded failed extraction")
        if not allowed_upgrade(prompt_version, old["prompt_version"]):
            raise JobError("INVALID_REQUEST", "same-processor failures must not be requeued as upgrades")
        existing = conn.execute("SELECT * FROM jobs WHERE repair_of_job_id=?", (old_job_id,)).fetchone()
        if existing is not None:
            if existing["prompt_version"] != prompt_version:
                raise JobError("INVALID_REQUEST", "select the registered failed child for the next upgrade")
            return dict(existing)
        if old["recovered_by_job_id"] is not None:
            raise JobError("INVALID_REQUEST", "inspect the existing recovery association before repairing")
        if conn.execute("SELECT 1 FROM jobs WHERE kind='extract' AND input_key=? AND state='succeeded'", (old["input_key"],)).fetchone():
            raise JobError("INVALID_REQUEST", "an already successful source must not be reprocessed")
        maximum = conn.execute("SELECT MAX(input_version) FROM jobs WHERE kind='extract' AND input_key=?", (old["input_key"],)).fetchone()[0]
        new = enqueue_in_transaction(conn, "extract", old["input_key"], input_version=int(maximum) + 1,
                                     model_version=old["model_version"], prompt_version=prompt_version,
                                     policy_version=old["policy_version"])
        conn.execute("UPDATE jobs SET repair_of_job_id=? WHERE job_id=?", (old_job_id, new["job_id"]))
        # The failed row demonstrably exists before the new job can finish.
        conn.execute("UPDATE jobs SET completed_seq=COALESCE(completed_seq,0) WHERE job_id=?", (old_job_id,))
        return dict(conn.execute("SELECT * FROM jobs WHERE job_id=?", (new["job_id"],)).fetchone())


#: Consecutive user-origin extraction claims served before a ready background
#: extraction job: a persistent 3-user : 1-background fair selection.
USER_PRIORITY_WINDOW = 3

_READY_WHERE = "state IN ('pending', 'retry_wait') AND (run_after IS NULL OR run_after <= ?)"


def _user_input_job_sql(alias: str) -> str:
    """SQL predicate matching a job's input_key to a stored user_input revision."""
    return (
        "EXISTS (SELECT 1 FROM events e JOIN event_revisions er "
        "ON er.event_id = e.event_id WHERE e.origin_kind = 'user_input' "
        f"AND {alias}.input_key = e.event_id || '@' || CAST(er.revision AS TEXT))"
    )


def _consecutive_user_claims(conn: sqlite3.Connection) -> int:
    """Count recent consecutive user-origin extraction claims.

    The scan is capped at :data:`USER_PRIORITY_WINDOW`: only that threshold
    decides whether to serve the background sub-queue, and the attempt journal
    may be large. Ordering by ``attempt_id`` preserves the choice order of an
    earlier ``limit > 1`` claim call, including attempts written by this call.
    """
    rows = conn.execute(
        "SELECT (j.kind = 'extract' AND " + _user_input_job_sql("j") + ") AS is_user "
        "FROM job_attempts ja JOIN jobs j ON j.job_id = ja.job_id WHERE j.kind = 'extract' "
        "ORDER BY ja.attempt_id DESC LIMIT ?",
        (USER_PRIORITY_WINDOW,),
    ).fetchall()
    streak = 0
    for row in rows:
        if not row["is_user"]:
            break
        streak += 1
    return streak


def _one_extract_job(conn: sqlite3.Connection, moment: str, *, user_input: bool):
    """Return the oldest ready extract job from one priority sub-queue."""
    predicate = _user_input_job_sql("jobs")
    if not user_input:
        predicate = "NOT " + predicate
    return conn.execute(
        "SELECT * FROM jobs WHERE " + _READY_WHERE + " AND kind = 'extract' AND "
        + predicate + " ORDER BY created_at, rowid, job_id LIMIT 1",
        (moment,),
    ).fetchone()


def _next_extract_job(conn: sqlite3.Connection, moment: str):
    """Pick one ready extract job, applying 3-user : 1-background fairness.

    Both sub-queues filter state and ``run_after`` before priority, so a
    future-held job neither gets claimed nor triggers a background turn.
    Unrecognized ``input_key`` values (including non-extraction jobs) are
    treated as background. The preferred queue is probed first, so a long user
    stream does not force a background scan when a user job is available.
    """
    serve_user = _consecutive_user_claims(conn) < USER_PRIORITY_WINDOW
    row = _one_extract_job(conn, moment, user_input=serve_user)
    if row is not None:
        return row
    return _one_extract_job(conn, moment, user_input=not serve_user)


def _next_fifo_job(conn: sqlite3.Connection, moment: str, kinds: tuple[str, ...] | None):
    """Original FIFO selection, used for every non-extract-only kind filter."""
    params: list = [moment]
    kind_clause = ""
    if kinds:
        placeholders = ",".join("?" for _ in kinds)
        kind_clause = f" AND kind IN ({placeholders})"
        params.extend(kinds)
    return conn.execute(
        "SELECT * FROM jobs WHERE " + _READY_WHERE + kind_clause +
        " ORDER BY created_at, rowid, job_id LIMIT 1",
        (*params,),
    ).fetchone()


def _claim_row(
    conn: sqlite3.Connection,
    row: sqlite3.Row,
    worker_id: str,
    moment: str,
    lease_seconds: int,
) -> dict:
    token = "lease:" + uuid.uuid4().hex
    attempt = int(row["attempts"]) + 1
    lease_expires_at = _iso(moment, lease_seconds)
    conn.execute(
        "UPDATE jobs SET state = 'running', attempts = ?, lease_token = ?, "
        "lease_owner = ?, heartbeat_at = ?, lease_expires_at = ?, updated_at = ? "
        "WHERE job_id = ?",
        (
            attempt,
            token,
            worker_id,
            moment,
            lease_expires_at,
            moment,
            row["job_id"],
        ),
    )
    conn.execute(
        "INSERT INTO job_attempts (job_id, attempt, lease_token, started_at) "
        "VALUES (?, ?, ?, ?)",
        (row["job_id"], attempt, token, moment),
    )
    return {
        **dict(row),
        "state": "running",
        "attempts": attempt,
        "lease_token": token,
        "lease_owner": worker_id,
        "heartbeat_at": moment,
        "lease_expires_at": lease_expires_at,
    }


def claim(
    store: Store,
    *,
    worker_id: str,
    kinds: tuple[str, ...] | None = None,
    lease_seconds: int = 60,
    limit: int = 1,
    now: str | None = None,
) -> list[dict]:
    """Claim ready jobs and journal a new attempt for each.

    Only ``kinds == ("extract",)`` enables interactive priority: jobs whose
    ``input_key`` names a stored ``origin_kind=user_input`` revision are served
    ahead of background extract jobs, except that after three consecutive user
    claims (counted from ``job_attempts``) one ready background job is served.
    Mixed or non-extract kind filters keep the original FIFO order. Each
    iteration re-reads the journal, so ``limit > 1`` observes the choices made
    earlier in the same call.
    """
    moment = now or utc_now()
    claimed: list[dict] = []
    with store.transaction() as conn:
        while len(claimed) < limit:
            if kinds == ("extract",):
                row = _next_extract_job(conn, moment)
            else:
                row = _next_fifo_job(conn, moment, kinds)
            if row is None:
                break
            claimed.append(_claim_row(conn, row, worker_id, moment, lease_seconds))
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
    diagnostic_detail: str | None = None,
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
        if row["repair_of_job_id"] is not None:
            # Order each completed upgrade attempt for truthful latest-error
            # reporting; successful completion stamps its own later sequence.
            seq = int(conn.execute("SELECT ledger_seq FROM backend_meta WHERE id=1").fetchone()[0]) + 1
            conn.execute("UPDATE backend_meta SET ledger_seq=? WHERE id=1", (seq,))
            conn.execute("UPDATE jobs SET completed_seq=? WHERE job_id=?", (seq,job_id))
        if diagnostic_detail is not None:
            conn.execute("CREATE TABLE IF NOT EXISTS extraction_failure_diagnostics (job_id TEXT NOT NULL, attempt INTEGER NOT NULL, metadata_json TEXT NOT NULL, PRIMARY KEY(job_id,attempt))")
            conn.execute("INSERT OR REPLACE INTO extraction_failure_diagnostics(job_id,attempt,metadata_json) VALUES(?,?,?)", (job_id,attempts,diagnostic_detail))
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
            ancestors = []
            child = row
            while child["repair_of_job_id"] is not None:
                parent = conn.execute("SELECT * FROM jobs WHERE job_id=?", (child["repair_of_job_id"],)).fetchone()
                if (len(ancestors) >= MAX_REPAIR_DEPTH or parent is None
                        or row["kind"] != "extract" or parent["kind"] != "extract"
                        or parent["state"] != "failed" or parent["last_error_code"] is None
                        or parent["input_key"] != row["input_key"]
                        or type(parent["input_version"]) is not int or type(child["input_version"]) is not int
                        or parent["input_version"] >= child["input_version"]
                        or not allowed_upgrade(child["prompt_version"], parent["prompt_version"])
                        or parent["model_version"] != row["model_version"] or parent["policy_version"] != row["policy_version"]
                        or parent["recovered_by_job_id"] is not None or parent["completed_seq"] is None):
                    raise JobError("INVALID_REQUEST", "extraction upgrade ancestor no longer matches")
                ancestors.append(parent)
                child = parent
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
            if ancestors:
                seq = int(conn.execute("SELECT ledger_seq FROM backend_meta WHERE id=1").fetchone()[0]) + 1
                if any(seq <= int(parent["completed_seq"]) for parent in ancestors):
                    raise JobError("INVALID_REQUEST", "extraction upgrade completion order is unproven")
                conn.execute("UPDATE backend_meta SET ledger_seq=? WHERE id=1", (seq,))
                conn.execute("UPDATE jobs SET completed_seq=? WHERE job_id=?", (seq, job_id))
                for parent in ancestors:
                    conn.execute("UPDATE jobs SET recovered_by_job_id=?,recovered_at=? WHERE job_id=?",
                                 (job_id, moment, parent["job_id"]))
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
