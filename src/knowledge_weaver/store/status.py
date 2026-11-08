"""Health and pipeline status for the KW service (no model calls).

AC-52 also needs the *real* synchronous product operations (export, cleanup,
projection rebuild) to write their own durable ledger rows.  They are not queue
workers, so this module provides :func:`record_operation_start` /
:func:`record_operation_finish` (and the :func:`operation_ledger` context
manager): a plain ``jobs`` row whose state moves ``pending -> succeeded`` or
``pending -> failed`` with a visible ``last_error_code``.  The status view keeps
reading the same durable job ledger, so the frozen ``Status`` schema is
unchanged and an unrecovered failure of one operation still survives another
operation's success (F16).

AC-52 round 5: whether a failure is recovered is an *explicit, durable
association*, not a rowid/timestamp inference.  The transaction that completes a
successful operation run also stamps every same-operation ``failed``
operation-ledger row that is already recorded at that moment with
``jobs.recovered_by_job_id`` / ``jobs.recovered_at``.  Recovery therefore
follows actual *completion* causality: a success that finished before a failure
was recorded can never hide it, a failure that finished first is recovered by a
later success, and the association lives in the ledger row itself, so it
survives a process restart and travels with the row through snapshot
export/import.  Queue-backed failures are never recovered this way.

AC-52 round 7: completion causality is proven by a *durable completion sequence*,
not by timestamp text.  The transaction that finishes an operation row (success or
failure) increments ``backend_meta.ledger_seq`` and stamps the row's
``jobs.completed_seq`` in the same transaction, so equal-second completions and
different UTC-offset spellings can no longer be mistaken for each other.  A
recovery link is honoured only when both rows carry a completion sequence and the
success is strictly later (``t.completed_seq > j.completed_seq``).  A database
written by an older schema has its already-recorded links converted on first open
(:func:`store.db._backfill_legacy_recovery_completion_order`) and anything that
cannot be proven stays ``NULL``, i.e. the failure stays visible.  Snapshot import
normalizes the ``jobs`` timestamp columns to canonical UTC seconds, refuses links
without a provable completion order, and seeds the restored ``ledger_seq`` with
``max(completed_seq)`` so a success after a restore still records a strictly
larger sequence.
"""

from __future__ import annotations

import contextlib
import hashlib
import uuid

from knowledge_weaver import contracts

from .db import Store, utc_now

PIPELINE_OPERATIONS = (
    "capture",
    "deliver",
    "extract",
    "project",
    "embed",
    "cleanup",
    "export",
)
#: operation -> persistent job kind (the job ledger is the source of truth).
JOB_BACKED_OPERATIONS = {
    "extract": "extract",
    "project": "project",
    "embed": "embed",
    "cleanup": "cleanup",
    "export": "export",
}

#: Marker written to ``jobs.model_version`` for rows created by a real
#: synchronous operation run (AC-52 round 4).  A later successful run of the same
#: operation recovers earlier ledger failures of that operation; queue-backed
#: units of work are never recovered this way (another job's success cannot erase
#: an unrecovered failure).
OPERATION_LEDGER_VERSION = "kw:operation-ledger:1"
OPERATION_INPUT_PREFIX = "operation:"

#: AC52-R5/R6/R7: SQL predicate over the ``jobs`` alias ``j`` that proves a
#: ``recovered_by_job_id`` association is a *valid* recovery, not merely a
#: non-NULL column.  A recovery link is trusted only when
#:
#: * both ``recovered_by_job_id`` and ``recovered_at`` are present,
#: * the row itself is a recorded ``failed`` operation-ledger row with a visible
#:   ``last_error_code`` (queue-backed work is never recovered this way),
#: * the target job exists, is a different row, has the same ``kind`` (the same
#:   operation/ledger), is itself an operation-ledger row and actually reached
#:   ``succeeded``,
#: * completion causality holds: both rows carry a durable ``completed_seq`` and
#:   the recovering success completed strictly later
#:   (``t.completed_seq > j.completed_seq``).  Timestamp text is deliberately not
#:   used: equal-second completions and mixed-offset spellings cannot prove an
#:   order, and the round-7 schema persists the order instead.
#:
#: Both :func:`_job_pipeline` (status derivation) and
#: ``portability.importing._validate_referential_integrity`` (snapshot import)
#: use this predicate, so an invalid association cannot hide a failure even if
#: one of the two layers is bypassed.  It has no bind parameters and only uses
#: the fixed aliases ``j`` (the candidate row) and ``t`` (the target row).
_VALID_OPERATION_RECOVERY_SQL = (
    "j.recovered_by_job_id IS NOT NULL AND j.recovered_at IS NOT NULL "
    "AND j.state = 'failed' AND j.last_error_code IS NOT NULL "
    f"AND j.model_version = '{OPERATION_LEDGER_VERSION}' "
    "AND j.completed_seq IS NOT NULL "
    "AND EXISTS ("
    "SELECT 1 FROM jobs t WHERE t.job_id = j.recovered_by_job_id "
    "AND t.job_id <> j.job_id AND t.kind = j.kind "
    "AND t.state = 'succeeded' AND t.model_version = j.model_version "
    "AND t.completed_seq IS NOT NULL AND t.completed_seq > j.completed_seq"
    ")"
)
# The only queue recovery is an explicit v3--v6 -> v7 upgrade of the SAME
# source revision, durably requested by the new job and linked in its commit.
from knowledge_weaver.jobs.repair_versions import upgrade_sql, MAX_REPAIR_DEPTH

_VALID_EXTRACT_RECOVERY_SQL = """
    j.kind='extract' AND j.state='failed' AND j.last_error_code IS NOT NULL
    AND j.recovered_by_job_id IS NOT NULL AND j.recovered_at IS NOT NULL
    AND j.completed_seq IS NOT NULL AND EXISTS (
      SELECT 1 FROM jobs t WHERE t.job_id=j.recovered_by_job_id AND t.job_id<>j.job_id
      AND t.kind='extract' AND t.state='succeeded' AND t.input_key=j.input_key
      AND t.model_version IS j.model_version AND t.policy_version IS j.policy_version
      AND t.completed_seq IS NOT NULL AND t.completed_seq>j.completed_seq
      AND EXISTS (
        WITH RECURSIVE chain(job_id, repair_of_job_id, prompt_version, input_version, depth) AS (
          SELECT t.job_id, t.repair_of_job_id, t.prompt_version, t.input_version, 0
          UNION ALL
          SELECT p.job_id,p.repair_of_job_id,p.prompt_version,p.input_version,a.depth+1
          FROM chain a JOIN jobs p ON p.job_id=a.repair_of_job_id
          WHERE a.depth < DEPTH_LIMIT AND p.kind='extract' AND p.state='failed'
          AND p.last_error_code IS NOT NULL AND p.input_key=t.input_key
          AND typeof(p.input_version)='integer' AND typeof(a.input_version)='integer'
          AND p.input_version<a.input_version AND UPGRADE_EDGE
          AND p.model_version IS t.model_version AND p.policy_version IS t.policy_version
          AND p.recovered_by_job_id=t.job_id AND p.recovered_at IS NOT NULL
          AND p.completed_seq IS NOT NULL AND p.completed_seq<t.completed_seq
        ) SELECT 1 FROM chain WHERE job_id=j.job_id AND depth>0
      )
    )
""".replace('DEPTH_LIMIT',str(MAX_REPAIR_DEPTH)).replace('UPGRADE_EDGE',upgrade_sql('a.prompt_version','p.prompt_version'))

VALID_RECOVERY_LINK_SQL = f"(({_VALID_OPERATION_RECOVERY_SQL}) OR ({_VALID_EXTRACT_RECOVERY_SQL}))"



def _operation_kind(operation: str) -> str:
    kind = JOB_BACKED_OPERATIONS.get(operation)
    if kind is None:
        raise ValueError(f"operation {operation!r} is not job-backed")
    return kind


def _operation_job_id(kind: str, token: str) -> str:
    digest = hashlib.sha256(f"{kind}|{OPERATION_INPUT_PREFIX}{token}|1".encode("utf-8")).hexdigest()
    return "job:" + digest[:40]


def record_operation_start(
    store: Store, operation: str, *, now: str | None = None
) -> dict:
    """Write the ``pending`` ledger row of a real operation run (AC-52).

    The row is committed *before* the operation does its durable work, so a
    concurrent ``status`` read can see the operation as pending.  The
    ``model_version`` marker distinguishes these rows from queue jobs of the
    same kind (see :data:`OPERATION_LEDGER_VERSION`).
    """
    kind = _operation_kind(operation)
    moment = now or utc_now()
    token = uuid.uuid4().hex
    job_id = _operation_job_id(kind, token)
    with store.transaction() as conn:
        conn.execute(
            "INSERT INTO jobs "
            "(job_id, kind, input_key, input_version, state, model_version, created_at, updated_at) "
            "VALUES (?, ?, ?, 1, 'pending', ?, ?, ?)",
            (
                job_id,
                kind,
                f"{OPERATION_INPUT_PREFIX}{kind}:{token}",
                OPERATION_LEDGER_VERSION,
                moment,
                moment,
            ),
        )
    return {
        "job_id": job_id,
        "operation": operation,
        "kind": kind,
        "started_at": moment,
    }


def record_operation_finish(
    store: Store,
    entry: dict,
    *,
    error_code: str | None = None,
    now: str | None = None,
) -> dict:
    """Move a started operation ledger row to ``succeeded`` or ``failed``.

    AC-52 round 5: the same transaction that makes a successful run durable
    records the explicit recovery association for every same-operation
    operation-ledger failure that is *already recorded* at that moment.  Only
    failures that completed before this success can be recovered; a failure that
    completes later is never hidden by an earlier-finishing success, whatever
    order the two runs were started in.  The failed row keeps its ``failed``
    state (history) and its ``last_error_code``; it gains
    ``recovered_by_job_id``/``recovered_at`` so the association is durable,
    restart-safe and part of the exported ledger row.

    AC-52 round 7: this transaction also consumes the next durable completion
    sequence from ``backend_meta.ledger_seq`` and stamps it on the finishing row
    (success *and* failure), so completion order is persisted rather than inferred
    from second-granularity timestamp text.  A failed row recorded by an older
    schema has no ``completed_seq``; it can only be recovered by a later success,
    so the same transaction stamps it ``0`` (it existed before this successful
    transaction committed) before linking it.  Recovery links are only written for
    rows whose completion order is strictly smaller than the new sequence.
    """
    state = "failed" if error_code else "succeeded"
    moment = now or utc_now()
    recovered_job_ids: list[str] = []
    with store.transaction() as conn:
        # AC52-R7: completion order is a durable, monotonic fact consumed in the
        # same transaction that commits the completion (BEGIN IMMEDIATE + the
        # process write lock serialize concurrent finishers).
        meta_row = conn.execute(
            "SELECT ledger_seq FROM backend_meta WHERE id = 1"
        ).fetchone()
        seq = int(meta_row["ledger_seq"] if meta_row is not None else 0) + 1
        conn.execute("UPDATE backend_meta SET ledger_seq = ? WHERE id = 1", (seq,))
        cursor = conn.execute(
            "UPDATE jobs SET state = ?, last_error_code = ?, lease_token = NULL, "
            "lease_owner = NULL, updated_at = ?, completed_seq = ? "
            "WHERE job_id = ? AND state = 'pending'",
            (state, error_code, moment, seq, entry["job_id"]),
        )
        if cursor.rowcount != 1:
            raise RuntimeError(
                f"operation ledger row {entry['job_id']} is no longer pending"
            )
        if state == "succeeded":
            # Only rows that provably completed earlier (strictly smaller
            # sequence) may be recovered; a legacy row without a sequence is
            # stamped 0 because it already existed when this transaction began.
            recoverable = (
                "kind = ? AND state = 'failed' AND model_version = ? "
                "AND recovered_by_job_id IS NULL "
                "AND (completed_seq IS NULL OR completed_seq < ?)"
            )
            rows = conn.execute(
                f"SELECT job_id FROM jobs WHERE {recoverable}",
                (entry["kind"], OPERATION_LEDGER_VERSION, seq),
            ).fetchall()
            if rows:
                recovered_job_ids = [row["job_id"] for row in rows]
                conn.execute(
                    "UPDATE jobs SET recovered_by_job_id = ?, recovered_at = ?, "
                    "completed_seq = COALESCE(completed_seq, 0) "
                    f"WHERE {recoverable}",
                    (
                        entry["job_id"],
                        moment,
                        entry["kind"],
                        OPERATION_LEDGER_VERSION,
                        seq,
                    ),
                )
    return {
        **entry,
        "state": state,
        "last_error_code": error_code,
        "finished_at": moment,
        "completed_seq": seq,
        "recovered_job_ids": recovered_job_ids,
    }


def operation_error_code(operation: str, exc: BaseException) -> str:
    """Stable, non-secret error code recorded for a failed operation run."""
    code = getattr(exc, "code", None)
    if isinstance(code, str) and code:
        return code
    return f"{operation.upper()}_FAILED"


@contextlib.contextmanager
def operation_ledger(store: Store, operation: str):
    """Record a real operation run as ``pending -> succeeded/failed`` (AC-52).

    Usage::

        with operation_ledger(store, "export"):
            ...do the real work...

    The failure is recorded with a stable visible ``last_error_code`` and the
    original exception is re-raised unchanged; a failure while recording the
    error can never mask the operation's own failure.
    """
    entry = record_operation_start(store, operation)
    try:
        yield entry
    except BaseException as exc:
        try:
            record_operation_finish(store, entry, error_code=operation_error_code(operation, exc))
        except Exception:  # pragma: no cover - preserve the original failure
            pass
        raise
    else:
        record_operation_finish(store, entry)


def record_operation_result(
    store: Store, operation: str, *, error_code: str | None = None
) -> dict:
    """Record one already-completed operation run (single trailing call).

    Convenience for short atomic entry points (for example a projection rebuild
    whose whole body is one transaction) where a visible ``pending`` phase is not
    useful; long operations should use :func:`operation_ledger`.
    """
    entry = record_operation_start(store, operation)
    return record_operation_finish(store, entry, error_code=error_code)


def health(store: Store) -> dict:
    meta = store.meta()
    integrity = store.quick_check()
    if meta["restore_in_progress"]:
        state = "restoring"
    elif integrity != "ok":
        state = "degraded"
    else:
        state = "ready"
    return {
        "schema_version": contracts.EVENT_SCHEMA_VERSION,
        "status": state,
        "backend_instance_id": meta["backend_instance_id"],
        "restore_epoch": meta["restore_epoch"],
        "watermark": meta["watermark"],
        "supported_contracts": [contracts.CONTRACT_VERSION],
        "database_integrity": integrity,
    }


def _count(store: Store, sql: str, params: tuple = ()) -> int:
    try:
        return store.count(sql, params)
    except Exception:  # pragma: no cover - schema in transition
        return 0


def _last(store: Store, sql: str, params: tuple = ()):
    try:
        with store.read() as conn:
            return conn.execute(sql, params).fetchone()
    except Exception:  # pragma: no cover - schema in transition
        return None


def _job_pipeline(store: Store, kind: str) -> dict:
    """Compute one pipeline's counters from the persistent job ledger (F16).

    States are never hard-coded: pending/retry_wait are visible work, running
    means a worker holds a lease, failed is an unrecovered error, and a success
    only updates ``last_success_at``. Another operation's success (or an idle
    sweep) therefore cannot erase an unrecovered embedding/extraction failure.

    AC-52 round 5: a failure counts as recovered only when its durable ledger
    row carries the explicit ``recovered_by_job_id`` association written by the
    later successful run's own completion transaction.  There is no rowid or
    timestamp ordering heuristic: a success that finished before the failure was
    recorded leaves ``recovered_by_job_id`` NULL, so the failure stays visible.

    AC-52 round 6: a non-NULL association is not trusted by itself.  The row is
    subtracted from the failure counter only while :data:`VALID_RECOVERY_LINK_SQL`
    holds (existing, different, same-operation operation-ledger target that
    really succeeded after the failure was recorded, complete link columns).  An
    invalid relation — the exact shape a corrupt/legacy database or a hand-edited
    snapshot can contain — keeps the failure visible and keeps its error code as
    the pipeline's ``last_error_code``.

    AC-52 round 7: "after the failure" is decided by the durable
    ``jobs.completed_seq`` (strictly greater), never by timestamp text, so two
    operations that complete in the same second (or store mixed UTC offsets) are
    ordered correctly.  The visible error is the latest completion among the
    unrecovered failures (``completed_seq`` descending, ``NULL`` last) rather than
    the lexicographically largest timestamp.
    """
    from .source_duplicates import VALID_DUPLICATE_RESOLUTION_SQL
    with store.read() as conn:
        rows = conn.execute(
            "SELECT state, COUNT(*) AS c FROM jobs WHERE kind = ? GROUP BY state", (kind,)
        ).fetchall()
        counts = {row["state"]: int(row["c"]) for row in rows}
        pending_row = conn.execute(
            "SELECT MIN(created_at) AS oldest FROM jobs WHERE kind = ? "
            "AND state IN ('pending', 'retry_wait')",
            (kind,),
        ).fetchone()
        success_row = conn.execute(
            "SELECT MAX(updated_at) AS last FROM jobs WHERE kind = ? AND state = 'succeeded'",
            (kind,),
        ).fetchone()
        # AC-52 round 5: a failed *operation-ledger* run (a real synchronous
        # product operation such as export/cleanup/projection) is recovered only
        # if a later successful run of the same operation stamped its ledger row
        # with ``recovered_by_job_id`` at completion time.  The failed row stays
        # in the jobs table as history, but the status view no longer reports it
        # as an unrecovered failure.  Queue-backed failures are never
        # auto-recovered: another job's success cannot erase an unrecovered unit
        # of work.
        #
        # AC-52 round 6: only a *valid* association counts.  An invalid link
        # (missing/self/other-operation/non-succeeded target, incomplete
        # columns, impossible completion order) is treated exactly like an
        # unrecovered failure, so status derivation fails safe on its own even
        # when the row arrived through an old database instead of import
        # validation.
        recovered_failures = int(
            conn.execute(
                "SELECT COUNT(*) AS c FROM jobs j WHERE j.kind = ? AND j.state = 'failed' "
                f"AND ({VALID_RECOVERY_LINK_SQL})",
                (kind,),
            ).fetchone()["c"]
        )
        duplicate_failures = int(conn.execute(
            "SELECT COUNT(*) FROM jobs j WHERE j.kind=? AND j.state='failed' "
            "AND NOT ("+VALID_RECOVERY_LINK_SQL+") AND ("+VALID_DUPLICATE_RESOLUTION_SQL+")",
            (kind,)).fetchone()[0])
        error_row = conn.execute(
            "SELECT j.last_error_code FROM jobs j WHERE j.kind = ? "
            "AND j.state IN ('failed', 'retry_wait') AND j.last_error_code IS NOT NULL "
            "AND NOT (j.state = 'failed' "
            f"AND (({VALID_RECOVERY_LINK_SQL}) OR ({VALID_DUPLICATE_RESOLUTION_SQL}))) "
            "ORDER BY j.completed_seq DESC, j.updated_at DESC, j.rowid DESC LIMIT 1",
            (kind,),
        ).fetchone()
    pending = counts.get("pending", 0) + counts.get("retry_wait", 0)
    running = counts.get("running", 0)
    failed = max(0, counts.get("failed", 0) - recovered_failures - duplicate_failures)
    if running:
        state = "running"
    elif failed or counts.get("retry_wait", 0):
        # An unrecovered failure or a scheduled retry keeps the pipeline visibly
        # degraded; an idle sweep or another operation's success cannot clear it.
        state = "degraded"
    else:
        # Queued-but-not-claimed work keeps the pending counter (queue depth)
        # without claiming a worker is running.
        state = "idle"
    return {
        "state": state,
        "pending": pending,
        "failed": failed,
        "oldest_pending_at": pending_row["oldest"] if pending_row is not None else None,
        "last_success_at": success_row["last"] if success_row is not None else None,
        "last_error_code": error_row["last_error_code"] if error_row is not None else None,
    }


def _receipt_pipeline(store: Store, operation: str) -> dict:
    if operation == "capture":
        total = _count(store, "SELECT COUNT(*) FROM receipts")
        last_success = _last(store, "SELECT MAX(created_at) AS last FROM receipts")
        return {
            "state": "idle",
            "pending": 0,
            "failed": 0,
            "oldest_pending_at": None,
            "last_success_at": last_success["last"] if last_success is not None else None,
            "last_error_code": None,
        }
    pending_receipts = _count(
        store, "SELECT COUNT(*) FROM receipts WHERE processing = 'pending'"
    )
    last_success = _last(store, "SELECT MAX(created_at) AS last FROM receipts")
    error = _last(
        store,
        "SELECT error_code FROM receipts WHERE error_code IS NOT NULL "
        "ORDER BY created_at DESC LIMIT 1",
    )
    return {
        "state": "running" if pending_receipts else "idle",
        "pending": pending_receipts,
        "failed": 0,
        "oldest_pending_at": None,
        "last_success_at": last_success["last"] if last_success is not None else None,
        "last_error_code": error["error_code"] if error is not None else None,
    }


def status(store: Store) -> dict:
    pipelines = []
    for operation in PIPELINE_OPERATIONS:
        if operation in JOB_BACKED_OPERATIONS:
            entry = _job_pipeline(store, JOB_BACKED_OPERATIONS[operation])
        else:
            entry = _receipt_pipeline(store, operation)
        pipelines.append({"operation": operation, **entry})
    return {"health": health(store), "pipelines": pipelines}
