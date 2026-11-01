"""Finish source-detail lifecycle only inside a successful fenced extraction.

Validated claims retain durable knowledge independently of raw process detail.
Unaccounted parts, warnings, model failures and stale commits never unlock
bootstrap protection. Old success ledgers alone are not proof of coverage.
"""
from .retention import _canonical_anchor, _plus_days, EXPIRED_BY_DAYS
from knowledge_weaver.store.db import utc_now



def protect_sources_in_transaction(conn, *, event_id, revision, part_indices, needs_review=False, now=None):
    """Persist unfinished dispositions without repeatedly paying to re-extract."""
    reason = "extraction-review-required" if needs_review else "extraction-disposition-pending"
    for index in sorted(part_indices):
        subject = f"{event_id}#{revision}@{index}"
        conn.execute("INSERT OR IGNORE INTO lifecycle_records (subject_kind,subject_id,class,business_anchor,expires_at,state,generation,reason,updated_at) VALUES ('event_part',?,'unclassified',NULL,NULL,'bootstrap_pending',0,?,?)", (subject,reason,now or utc_now()))

def finalize_sources_in_transaction(conn, *, event_id, revision, part_indices, class_by_part, job_id, now=None, anchor_by_part=None):
    source = conn.execute("SELECT event_time_value, event_time_precision FROM event_revisions WHERE event_id=? AND revision=?", (event_id,revision)).fetchone()
    if source is None:
        return 0
    completed = 0
    for index in sorted(part_indices):
        subject = f"{event_id}#{revision}@{index}"
        existing = conn.execute("SELECT * FROM lifecycle_records WHERE subject_kind='event_part' AND subject_id=?", (subject,)).fetchone()
        if existing is not None and existing["state"] != "bootstrap_pending":
            continue  # Never renew an already classified or tombstoned detail.
        if conn.execute("SELECT 1 FROM tombstones WHERE subject_kind='event_part' AND subject_id=?", (subject,)).fetchone():
            continue
        inherited = (anchor_by_part or {}).get(index) if index in (anchor_by_part or {}) else (source["event_time_value"] if source["event_time_precision"] in ("instant", "day") else None)
        anchor = _canonical_anchor(existing["business_anchor"] if existing is not None and existing["business_anchor"] else inherited)
        chosen_class = class_by_part.get(index, "temporary_detail")
        if existing is not None and existing["reason"] == "imported-history-bootstrap" and existing["class"] == "temporary_detail" and chosen_class == "current_fact":
            chosen_class = "temporary_detail"  # Historical raw detail, extracted knowledge remains.
        # An explicitly classified imported detail keeps that policy; a generic
        # import placeholder can use the validated task/detail classification.
        if existing is not None and existing["reason"] not in ("imported-history-bootstrap", "extraction-disposition-pending", "extraction-review-required"):
            chosen_class = existing["class"]
        if not anchor:
            chosen_class = "unclassified"
        days = EXPIRED_BY_DAYS.get(chosen_class)
        expiry = _plus_days(anchor,days) if anchor and days else None
        conn.execute("INSERT INTO lifecycle_records (subject_kind,subject_id,class,business_anchor,expires_at,state,generation,reason,updated_at) VALUES ('event_part',?,?,?,?, 'active',0,?,?) ON CONFLICT(subject_kind,subject_id) DO UPDATE SET class=excluded.class,business_anchor=excluded.business_anchor,expires_at=excluded.expires_at,state='active',reason=excluded.reason,updated_at=excluded.updated_at", (subject,chosen_class,anchor,expiry,"extraction-complete:"+job_id,now or utc_now()))
        completed += 1
    return completed
