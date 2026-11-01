"""Single semantic extraction worker (T07).

Only here do background model calls happen for memory processing. Raw evidence
receipt never waits for the model; extraction is a persistent job with a lease
and a fenced commit, so a stale or duplicated result cannot overwrite current
facts (JOB-01 / LLM-01).
"""

from __future__ import annotations

import json

from knowledge_weaver import contracts
from knowledge_weaver.jobs import (
    JobError,
    claim as claim_jobs,
    commit as commit_job,
    enqueue,
    fail as fail_job,
)
from knowledge_weaver.store.claims import ClaimError
from knowledge_weaver.store.db import utc_now

from .proposals import (
    EXCLUDED_ORIGINS,
    ExtractionError,
    apply_resolved_claims,
    load_event_group,
    resolve_proposal,
    sanitized_parts,
)

MODEL_VERSION = "model:unspecified"
PROMPT_VERSION = "prompt:5"


def extraction_input_key(event_id: str, revision: int) -> str:
    return f"{event_id}@{int(revision)}"


#: Local, rebuildable scan position for the missing-extraction-job reconcile.
#: It is deliberately not part of the portable snapshot: losing it only restarts
#: the scan at the head, which is idempotent and loses no queued work.
RECONCILE_CURSOR_NAME = "extract_missing_jobs"
RECONCILE_STATE_TABLE = "reconcile_state"

_MISSING_EXTRACTION_SQL = (
    "SELECT r.event_id, r.revision, e.origin_kind "
    "FROM event_revisions r JOIN events e ON e.event_id = r.event_id "
    "WHERE r.group_state = 'complete' AND r.lifecycle = 'active' "
    "AND NOT EXISTS (SELECT 1 FROM jobs j WHERE j.kind = 'extract' "
    "AND j.input_key = r.event_id || '@' || CAST(r.revision AS TEXT))"
)


def _ensure_reconcile_state(conn) -> None:
    """Create the local cursor table on first use (not a portable snapshot table)."""
    conn.execute(
        f"CREATE TABLE IF NOT EXISTS {RECONCILE_STATE_TABLE} ("
        "name TEXT PRIMARY KEY, "
        "cursor_event_id TEXT NULL, "
        "cursor_revision INTEGER NULL, "
        "updated_at TEXT NOT NULL)"
    )


def _load_reconcile_cursor(conn) -> tuple[str, int] | None:
    row = conn.execute(
        f"SELECT cursor_event_id, cursor_revision FROM {RECONCILE_STATE_TABLE} WHERE name = ?",
        (RECONCILE_CURSOR_NAME,),
    ).fetchone()
    if row is None or row["cursor_event_id"] is None or row["cursor_revision"] is None:
        return None
    return str(row["cursor_event_id"]), int(row["cursor_revision"])


def _save_reconcile_cursor(conn, cursor: tuple[str, int] | None) -> None:
    conn.execute(
        f"INSERT INTO {RECONCILE_STATE_TABLE} "
        "(name, cursor_event_id, cursor_revision, updated_at) VALUES (?, ?, ?, ?) "
        "ON CONFLICT (name) DO UPDATE SET cursor_event_id = excluded.cursor_event_id, "
        "cursor_revision = excluded.cursor_revision, updated_at = excluded.updated_at",
        (
            RECONCILE_CURSOR_NAME,
            None if cursor is None else cursor[0],
            None if cursor is None else int(cursor[1]),
            utc_now(),
        ),
    )


def _select_missing_extraction_jobs(conn, *, after: tuple[str, int] | None, limit: int):
    """Page over complete/active revisions that really have no extraction job."""
    if after is None:
        return conn.execute(
            _MISSING_EXTRACTION_SQL + " ORDER BY r.event_id, r.revision LIMIT ?",
            (int(limit),),
        ).fetchall()
    return conn.execute(
        _MISSING_EXTRACTION_SQL
        + " AND (r.event_id > ? OR (r.event_id = ? AND r.revision > ?)) "
        "ORDER BY r.event_id, r.revision LIMIT ?",
        (after[0], after[0], int(after[1]), int(limit)),
    ).fetchall()


def reconcile_extraction_jobs(store, *, limit: int = 500) -> dict:
    """Idempotently queue extraction for complete active revisions that lack a job.

    The group-state transition enqueues in the same transaction (see
    ``store.events``), but databases written by older builds or a crash between
    an external enqueue and the commit can leave an active ``complete`` revision
    with no extraction job. This reconciliation is safe to run repeatedly: job
    ids are deterministic, so an existing job is left untouched.

    F07: the scan pages over the *missing-job set only* (``NOT EXISTS`` against
    the jobs ledger) and keeps a persisted forward cursor, so a later gap can
    never be starved by an earlier page of already-queued rows. Excluded
    maintenance/injection origins never get a job, but scanning past them
    advances the cursor instead of blocking the head of the queue. When the scan
    reaches the end of the gap set it wraps and the next periodic run starts a
    fresh full cycle. Counters are honest: ``scanned == queued + skipped``.
    """
    from knowledge_weaver.jobs import enqueue_in_transaction

    page_size = max(1, int(limit))
    queued = 0
    skipped = 0
    scanned = 0
    wrapped = False
    next_cursor: tuple[str, int] | None = None
    with store.transaction() as conn:
        _ensure_reconcile_state(conn)
        cursor = _load_reconcile_cursor(conn)
        rows = _select_missing_extraction_jobs(conn, after=cursor, limit=page_size)
        if not rows and cursor is not None:
            # The cursor passed the end of the ordered gap set: wrap around and
            # start a new cycle in this same call so earlier gaps are covered.
            wrapped = True
            rows = _select_missing_extraction_jobs(conn, after=None, limit=page_size)
        for row in rows:
            scanned += 1
            if row["origin_kind"] in EXCLUDED_ORIGINS:
                # Excluded origins are never semantically ingested; skipping them
                # still advances the cursor, so they cannot clog the head.
                skipped += 1
                continue
            enqueue_in_transaction(
                conn,
                "extract",
                extraction_input_key(row["event_id"], int(row["revision"])),
                input_version=int(row["revision"]),
                model_version=MODEL_VERSION,
                prompt_version=PROMPT_VERSION,
                policy_version=contracts.redaction_policy_version(),
            )
            queued += 1
        if rows and len(rows) == page_size:
            last = rows[-1]
            next_cursor = (str(last["event_id"]), int(last["revision"]))
        else:
            # A partial (or empty) page reached the end of the gap set; reset the
            # cursor so the next periodic run starts a new full cycle.
            next_cursor = None
            if rows:
                wrapped = True
        _save_reconcile_cursor(conn, next_cursor)
    cursor_label = None if next_cursor is None else f"{next_cursor[0]}@{next_cursor[1]}"
    return {
        "queued": queued,
        "skipped": skipped,
        "scanned": scanned,
        "cursor": cursor_label,
        "wrapped": wrapped,
    }


def enqueue_extraction(
    store,
    event_id: str,
    revision: int,
    *,
    model_version: str | None = None,
    prompt_version: str | None = None,
) -> dict:
    return enqueue(
        store,
        "extract",
        extraction_input_key(event_id, revision),
        input_version=int(revision),
        model_version=model_version or MODEL_VERSION,
        prompt_version=prompt_version or PROMPT_VERSION,
        policy_version=contracts.redaction_policy_version(),
    )


def enqueue_extraction_in_transaction(
    conn,
    event_id: str,
    revision: int,
    *,
    model_version: str | None = None,
    prompt_version: str | None = None,
) -> dict:
    """Queue extraction on the caller's transaction (F07 atomic group transition)."""
    from knowledge_weaver.jobs import enqueue_in_transaction

    return enqueue_in_transaction(
        conn,
        "extract",
        extraction_input_key(event_id, revision),
        input_version=int(revision),
        model_version=model_version or MODEL_VERSION,
        prompt_version=prompt_version or PROMPT_VERSION,
        policy_version=contracts.redaction_policy_version(),
    )


def _fence_check(event_id: str, revision: int):
    def check(conn) -> tuple[bool, str]:
        row = conn.execute(
            "SELECT lifecycle, group_state FROM event_revisions WHERE event_id = ? AND revision = ?",
            (event_id, int(revision)),
        ).fetchone()
        if row is None:
            return False, "event revision removed"
        if row["group_state"] != "complete":
            return False, "event revision group is no longer complete"
        if row["lifecycle"] != "active":
            return False, "event revision is no longer active"
        parts = conn.execute(
            "SELECT COUNT(*) AS c, "
            "SUM(CASE WHEN availability = 'full' THEN 1 ELSE 0 END) AS f "
            "FROM event_parts WHERE event_id = ? AND revision = ?",
            (event_id, int(revision)),
        ).fetchone()
        if not parts["c"] or parts["f"] != parts["c"]:
            return False, "source evidence hidden or missing since enqueue"
        return True, "ok"

    return check


def run_extraction_job(store, job: dict, extractor, *, now: str | None = None) -> dict:
    event_id, raw_revision = job["input_key"].rsplit("@", 1)
    revision = int(raw_revision)
    group = load_event_group(store, event_id, revision)
    if group["origin_kind"] in EXCLUDED_ORIGINS:
        return commit_job(
            store,
            job["job_id"],
            job["lease_token"],
            expected_input_version=job["input_version"],
            now=now,
        )
    parts = sanitized_parts(group)
    if not parts:
        # Withheld bodies are never semantically ingested; the receipt remains.
        # This is a deliberate no-op commit, so no evidence fence is applicable.
        return commit_job(
            store,
            job["job_id"],
            job["lease_token"],
            expected_input_version=job["input_version"],
            now=now,
        )
    previous_validation_error = None
    if int(job.get("attempts") or 1) > 1:
        with store.read() as conn:
            previous = conn.execute("SELECT detail,error_code FROM job_attempts WHERE job_id = ? AND attempt = ?", (job["job_id"], int(job["attempts"]) - 1)).fetchone()
        if previous and str(previous["detail"] or "").startswith("FIELD_IDENTITY_COLLISION:"):
            previous_validation_error = previous["detail"]
        elif previous and previous["error_code"] == "MODEL_REVIEW_REQUIRED":
            previous_validation_error = 'MODEL_REVIEW_REQUIRED: Previous new writes were withheld because the proposal contained warnings. Reserve warnings for unresolved contradictions or ambiguities that make the proposed factual content unsafe to retain. Normal caveats are represented by the proper assertion kind/facet: an assistant report stays reported, an open task stays open, a hypothesis stays inferred. Those valid weak grades do not themselves require a warning. Do not invent a user confirmation, evidence or date just to remove warnings. If the exact historical date of last time is not given, do not invent it; retain the supplied native source event time as statement provenance and leave effective_time_quote null. An unknown source clock must remain unknown. Preserve the actual work background and unfinished task; omit unsupported extra keys, not the well-grounded facts.'
    proposal = extractor.extract(
        parts,
        context={
            **({"previous_validation_error": previous_validation_error} if previous_validation_error else {}),
            "event_id": event_id,
            "event_revision": revision,
            "project_ids": list(group["project_ids"]),
            "model_version": job.get("model_version"),
            "prompt_version": job.get("prompt_version"),
            "known_claims": [
                {key: entry[key] for key in ("claim_id", "project_id", "predicate", "facet", "label", "qualifiers")}
                for entry in group.get("claim_candidates", [])
            ],
        },
    )
    issues = contracts.validate_extraction_output(proposal)
    if issues:
        first = issues[0]
        raise ExtractionError(
            "MODEL_UNAVAILABLE",
            f"extraction proposal failed schema at {first.path}: {first.message}",
            retryable=True,
        )
    resolved, warnings = resolve_proposal(group, proposal, parts)
    committed_counts={"written":0,"reused":0}
    def mutate(conn):
        from knowledge_weaver.lifecycle.bootstrap import protect_sources_in_transaction
        protect_sources_in_transaction(conn, event_id=event_id, revision=revision,
            part_indices={part.part_index for part in parts},
            needs_review=bool(warnings or proposal.get("warnings")), now=now)
        from knowledge_weaver.extraction.echoes import split_assistant_echoes
        fresh,reused=split_assistant_echoes(conn,resolved)
        if fresh and proposal.get("warnings"):
            # Disclaimed NEW writes need repair. A zero-write exclusion/reuse
            # changes no remembered truth and must not pay for repeated retries.
            # Ambiguous raw source dispositions remain protected below.
            raise ExtractionError("MODEL_REVIEW_REQUIRED", "model proposal requires review", retryable=True)
        committed_counts.update(written=len(fresh),reused=len(reused))
        if reused:
            import json
            conn.execute("UPDATE job_attempts SET detail=? WHERE job_id=? AND lease_token=?",(json.dumps({"claims_written":len(fresh),"reused_existing_claims":reused},sort_keys=True),job["job_id"],job["lease_token"]))
        if fresh:
            apply_resolved_claims(
                conn,
                fresh,
                model_version=str(job.get("model_version") or MODEL_VERSION),
                prompt_version=str(job.get("prompt_version") or PROMPT_VERSION),
                policy_version=contracts.redaction_policy_version(),
            )
        if not warnings and not proposal.get("warnings"):
            from knowledge_weaver.lifecycle.bootstrap import finalize_sources_in_transaction
            covered = {ref["part_index"] for claim in resolved for ref in claim.evidence
                       if ref["event_id"] == event_id and ref["event_revision"] == revision}
            if not resolved and event_id in proposal.get("excluded_event_ids", []):
                covered = {part.part_index for part in parts}
            # Facts retain their own lifecycle. Ordinary full source bodies are
            # process detail, never an excuse to retain unrelated logs forever.
            classes = {index: "temporary_detail" for index in covered}
            if not resolved and event_id in proposal.get("excluded_event_ids", []):
                classes = {index: "temporary_detail" for index in covered}
            echo_anchors={}
            for reuse in reused:
                for index in reuse["part_indices"]:
                    classes[index]="temporary_detail"
                    old=echo_anchors.get(index,reuse["source_anchor"])
                    anchor=reuse["source_anchor"]
                    echo_anchors[index]=min(old,anchor) if old is not None and anchor is not None else None
            # A correction/API write can already have durably processed this
            # whole source part. Reuse its exact full-span evidence, not a bare
            # old "job succeeded" flag, instead of leaving it pending forever.
            for part in parts:
                proof = conn.execute(
                    "SELECT 1 FROM evidence_links e JOIN claim_revisions r ON r.revision_id=e.revision_id "
                    "WHERE e.event_id=? AND e.event_revision=? AND e.part_index=? "
                    "AND e.body_sha256=? AND e.start_byte=0 AND e.end_byte=? "
                    "AND e.availability='full' AND r.status NOT IN ('expired','revoked') LIMIT 1",
                    (event_id,revision,part.part_index,part.body_sha256,len(part.text.encode("utf-8")))).fetchone()
                if proof:
                    covered.add(part.part_index)
                    classes.setdefault(part.part_index,"temporary_detail")
            completed_parts = {ref["part_index"] for claim in fresh
                if claim.predicate == "task.status" and isinstance(claim.value, dict)
                and claim.value.get("status") in ("reported_done", "verified", "verified_done")
                for ref in claim.evidence if ref["event_id"] == event_id and ref["event_revision"] == revision}
            finalize_sources_in_transaction(conn, event_id=event_id, revision=revision,
                part_indices=covered, class_by_part={**classes, **{index: "completed_task" for index in completed_parts}},
                job_id=job["job_id"], now=now, anchor_by_part=echo_anchors)
    result = commit_job(
        store,
        job["job_id"],
        job["lease_token"],
        expected_input_version=job["input_version"],
        mutate=mutate,
        fence_check=_fence_check(event_id, revision),
        now=now,
    )
    result["claims_written"] = committed_counts["written"]
    result["claims_reused"] = committed_counts["reused"]
    result["warnings"] = warnings
    return result


def run_worker_once(
    store,
    extractor,
    *,
    worker_id: str = "worker:extract",
    lease_seconds: int = 60,
    now: str | None = None,
) -> dict:
    claimed = claim_jobs(
        store,
        worker_id=worker_id,
        kinds=("extract",),
        lease_seconds=lease_seconds,
        limit=1,
        now=now,
    )
    if not claimed:
        return {"claimed": 0}
    job = claimed[0]
    try:
        result = run_extraction_job(store, job, extractor, now=now)
        return {"claimed": 1, "job_id": job["job_id"], "state": "succeeded", "claims_written": result.get("claims_written", 0), **({"claims_reused":result["claims_reused"]} if result.get("claims_reused") else {})}
    except ExtractionError as exc:
        detail = None
        if exc.message.startswith("FIELD_IDENTITY_COLLISION:"):
            detail = exc.message
        elif exc.code == "MODEL_UNAVAILABLE":
            # The provider may have consumed a request whose response was lost.
            # No prices or secrets: unknown billing is never a claim of zero cost.
            detail = json.dumps({
                "billing_outcome": "unknown",
                "provider_usage": None,
                "reason": "external_request_outcome_unknown",
            }, sort_keys=True)
        outcome = fail_job(store, job["job_id"], job["lease_token"], exc.code,
                           retryable=exc.retryable, now=now, detail=detail)
        return {"claimed": 1, "job_id": job["job_id"], "state": outcome["state"], "error": exc.code}
    except ClaimError as exc:
        outcome = fail_job(store, job["job_id"], job["lease_token"], exc.code, retryable=False, now=now)
        return {"claimed": 1, "job_id": job["job_id"], "state": outcome["state"], "error": exc.code}
    except JobError as exc:
        return {"claimed": 1, "job_id": job["job_id"], "state": "stale", "error": exc.code}
