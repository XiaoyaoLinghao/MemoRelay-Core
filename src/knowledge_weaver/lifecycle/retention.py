"""Automatic retention (T13): 30/180/365 business-anchor rules per fact/detail.

Rules (PRD §5, TTL-01/TTL-02):

* temporary debug/failed-attempt detail: 30 days after the related work has
  ended and has no unresolved dependency;
* ordinary completed task process detail: 180 days after completion; key
  conclusions and still-valid configuration are kept as their own facts;
* superseded/retracted history detail: 365 days after replacement/withdrawal;
* current configuration, unfinished tasks, long-term preferences and key
  conclusions: kept while valid, with no access-based renewal;
* unknown lifecycle state stays ``unclassified`` and is never auto-expired;
* imported raw history stays ``bootstrap_pending`` until extraction/valid-
  knowledge identification has run, so import cannot delete un-processed data;
* sweeping writes a minimal no-body tombstone (identity hash + generation) and
  purges claim detail/index payloads; tombstones stop late jobs/imports from
  resurrecting expired bodies;
* retrieval/summary/import/restore/maintenance never move an anchor (they only
  read or rebuild derived state).
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json

from knowledge_weaver.store.db import Store, utc_now
from knowledge_weaver.store.status import operation_ledger

CLASSES = {
    "current_fact",
    "temporary_detail",
    "completed_task",
    "superseded_history",
    "revoked_history",
    "unclassified",
    "bootstrap_pending",
}
EXPIRED_BY_DAYS = {
    "temporary_detail": 30,
    "completed_task": 180,
    "superseded_history": 365,
    "revoked_history": 365,
}
CURRENT_CLASSES = {"current_fact", "unclassified", "bootstrap_pending"}


class RetentionClockError(ValueError):
    """An operator-supplied retention clock is invalid or ambiguous (F09b).

    The CLI error boundary reports ``code = "INVALID_CLOCK"`` so a bad ``--now``
    is a clear refusal instead of a silently wrong expiry decision.
    """

    code = "INVALID_CLOCK"


def _parse(moment: str | None) -> _dt.datetime | None:
    if not moment:
        return None
    try:
        parsed = _dt.datetime.fromisoformat(str(moment).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=_dt.timezone.utc)
    return parsed.astimezone(_dt.timezone.utc)


def _iso(moment: _dt.datetime) -> str:
    return moment.astimezone(_dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _parse_clock(moment: str) -> _dt.datetime:
    """Parse an operator cleanup clock into an aware UTC instant (F09b).

    A cleanup clock decides which bodies are deleted, so an unparseable value or
    one without a timezone offset is refused instead of being guessed as UTC.
    Stored business anchors keep the historical lenient ``_parse`` behaviour.
    """
    raw = moment.strip() if isinstance(moment, str) else ""
    if not raw:
        raise RetentionClockError(
            "retention clock must be a non-empty RFC 3339 timestamp with a timezone offset"
        )
    text = raw[:-1] + "+00:00" if raw[-1:] in ("Z", "z") else raw
    try:
        parsed = _dt.datetime.fromisoformat(text)
    except ValueError as exc:
        raise RetentionClockError(
            f"invalid retention clock {moment!r}: expected an RFC 3339 timestamp "
            "with a timezone offset such as 2026-01-31T00:00:00Z"
        ) from exc
    if parsed.tzinfo is None:
        raise RetentionClockError(
            f"retention clock {moment!r} has no timezone offset; pass an explicit "
            "UTC offset (e.g. 2026-01-31T00:00:00Z or 2026-01-31T08:00:00+08:00)"
        )
    return parsed.astimezone(_dt.timezone.utc)


def _clock_instant(now: str | None) -> _dt.datetime:
    """Resolve a cleanup clock argument to the exact UTC instant it means."""
    return _parse_clock(now if now is not None else utc_now())


def normalize_clock(now: str | None = None) -> str:
    """Return a cleanup clock in canonical UTC form (``...Z``, second precision).

    ``None`` means "the current UTC instant". Offset forms are converted to the
    same instant, so ``+08:00``/``-08:00``/``+00:00`` compare exactly like ``Z``;
    invalid or timezone-naive clocks raise :class:`RetentionClockError`.
    """
    return _iso(_clock_instant(now))


def _is_due(expires_at: str | None, moment: _dt.datetime) -> bool:
    """Compare a stored expiry with a real instant, never with raw strings."""
    parsed = _parse(expires_at)
    return parsed is not None and parsed <= moment


def _due_lifecycle_records(conn, moment: _dt.datetime) -> list:
    """F09b: select active records due at ``moment`` by instant, not text order."""
    rows = conn.execute(
        "SELECT * FROM lifecycle_records WHERE state = 'active' AND expires_at IS NOT NULL"
    ).fetchall()
    return [row for row in rows if _is_due(row["expires_at"], moment)]


def _canonical_anchor(anchor: str | None) -> str | None:
    """Store a lifecycle anchor in the same canonical UTC format as expiries.

    A malformed anchor is left as-is and then yields no expiry: retention never
    deletes data whose anchor it cannot reason about (the caller clock is
    strict, and anchors are stored/imported data rather than operator input).
    """
    parsed = _parse(anchor)
    return _iso(parsed) if parsed is not None else anchor


def _plus_days(moment: str, days: int) -> str | None:
    parsed = _parse(moment)
    if parsed is None:
        return None
    return _iso(parsed + _dt.timedelta(days=days))


def _identity_hash(subject_kind: str, subject_id: str) -> str:
    return hashlib.sha256(f"{subject_kind}|{subject_id}".encode("utf-8")).hexdigest()


def register_lifecycle(
    store: Store,
    *,
    subject_kind: str,
    subject_id: str,
    class_: str,
    business_anchor: str | None,
    state: str = "active",
    reason: str | None = None,
) -> dict:
    """Register/update a lifecycle record; anchors are never renewed by reads."""
    if class_ not in CLASSES:
        raise ValueError(f"unsupported lifecycle class {class_!r}")
    if state not in ("active", "bootstrap_pending", "expired", "tombstoned"):
        raise ValueError(f"unsupported lifecycle state {state!r}")
    days = EXPIRED_BY_DAYS.get(class_)
    anchor = _canonical_anchor(business_anchor)
    expires_at = _plus_days(anchor, days) if anchor and days else None
    now = utc_now()
    with store.transaction() as conn:
        existing = conn.execute(
            "SELECT * FROM lifecycle_records WHERE subject_kind = ? AND subject_id = ?",
            (subject_kind, subject_id),
        ).fetchone()
        tombstoned = conn.execute(
            "SELECT 1 FROM tombstones WHERE subject_kind = ? AND subject_id = ?",
            (subject_kind, subject_id),
        ).fetchone()
        if tombstoned is not None and existing is not None and existing["state"] == "tombstoned":
            return dict(existing)
        generation = int(existing["generation"]) if existing is not None else 0
        conn.execute(
            "INSERT INTO lifecycle_records "
            "(subject_kind, subject_id, class, business_anchor, expires_at, state, generation, reason, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT (subject_kind, subject_id) DO UPDATE SET "
            "class = excluded.class, business_anchor = excluded.business_anchor, "
            "expires_at = excluded.expires_at, reason = excluded.reason, updated_at = excluded.updated_at",
            (
                subject_kind,
                subject_id,
                class_,
                anchor,
                expires_at,
                state,
                generation,
                reason,
                now,
            ),
        )
        row = conn.execute(
            "SELECT * FROM lifecycle_records WHERE subject_kind = ? AND subject_id = ?",
            (subject_kind, subject_id),
        ).fetchone()
    return dict(row)


def classify_claim_revisions(store: Store, *, now: str | None = None) -> int:
    """Derive per-revision lifecycle classes from the current projection status.

    F09b: the classification clock is parsed into a real instant (canonical UTC
    for the ``updated_at`` writes); offset/``Z`` forms are equivalent and an
    invalid or naive clock is refused before the transaction starts.
    """
    moment = normalize_clock(now)
    classified = 0
    with store.transaction() as conn:
        rows = conn.execute(
            "SELECT r.revision_id, r.claim_id, r.revision_number, r.status, r.created_at, "
            "r.status_changed_at, r.event_time_value, r.event_time_precision FROM claim_revisions r"
        ).fetchall()
        for row in rows:
            status = row["status"]
            if status in ("current", "conflict"):
                if row["event_time_precision"] == "unknown" or not row["event_time_value"]:
                    class_, anchor = "unclassified", None
                else:
                    class_ = "current_fact"
                    anchor = row["event_time_value"]
            elif status == "superseded":
                class_ = "superseded_history"
                # Business anchor: the business time of the replacing revision,
                # never the maintenance/sweep wall clock.
                replacing = conn.execute(
                    "SELECT event_time_value FROM claim_revisions "
                    "WHERE claim_id = ? AND revision_number > ? AND event_time_value IS NOT NULL "
                    "ORDER BY revision_number LIMIT 1",
                    (row["claim_id"], row["revision_number"]),
                ).fetchone()
                anchor = (
                    replacing["event_time_value"]
                    if replacing is not None
                    else (row["status_changed_at"] or row["created_at"])
                )
            elif status == "revoked":
                class_ = "revoked_history"
                from knowledge_weaver.store.db import utc_now as _now

                retract = conn.execute(
                    "SELECT r2.event_time_value FROM evidence_links l "
                    "JOIN event_revisions r2 ON r2.event_id = l.event_id "
                    "AND r2.operation = 'retract' AND r2.parent_revision = l.event_revision "
                    "WHERE l.revision_id = ? AND r2.event_time_value IS NOT NULL LIMIT 1",
                    (row["revision_id"],),
                ).fetchone()
                anchor = (
                    retract["event_time_value"]
                    if retract is not None
                    else (row["status_changed_at"] or row["created_at"])
                )
            elif status == "expired":
                continue
            else:
                class_ = "unclassified"
                anchor = row["event_time_value"] or row["created_at"]
            anchor = _canonical_anchor(anchor)
            days = EXPIRED_BY_DAYS.get(class_)
            expires_at = _plus_days(anchor, days) if days else None
            existing = conn.execute(
                "SELECT * FROM lifecycle_records WHERE subject_kind = 'claim_revision' AND subject_id = ?",
                (row["revision_id"],),
            ).fetchone()
            if existing is not None and existing["state"] in ("expired", "tombstoned"):
                continue
            if (
                existing is not None
                and existing["class"] == class_
                and existing["business_anchor"] == anchor
                and existing["expires_at"] == expires_at
            ):
                continue
            conn.execute(
                "INSERT INTO lifecycle_records "
                "(subject_kind, subject_id, class, business_anchor, expires_at, state, generation, reason, updated_at) "
                "VALUES ('claim_revision', ?, ?, ?, ?, 'active', ?, 'auto-classified', ?) "
                "ON CONFLICT (subject_kind, subject_id) DO UPDATE SET "
                "class = excluded.class, business_anchor = excluded.business_anchor, "
                "expires_at = excluded.expires_at, reason = excluded.reason, updated_at = excluded.updated_at",
                (
                    row["revision_id"],
                    class_,
                    anchor,
                    expires_at,
                    int(existing["generation"]) if existing is not None else 0,
                    moment,
                ),
            )
            classified += 1
    return classified


def _tombstone(conn, record, *, generation: int, now: str) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO tombstones "
        "(tombstone_id, subject_kind, subject_id, identity_hash, generation, class, reason, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "tomb:"
            + hashlib.sha256(
                f"{record['subject_kind']}|{record['subject_id']}|{generation}".encode("utf-8")
            ).hexdigest()[:32],
            record["subject_kind"],
            record["subject_id"],
            _identity_hash(record["subject_kind"], record["subject_id"]),
            generation,
            record["class"],
            "retention-expired",
            now,
        ),
    )
    conn.execute(
        "UPDATE lifecycle_records SET state = 'tombstoned', generation = ?, updated_at = ? "
        "WHERE subject_kind = ? AND subject_id = ?",
        (generation, now, record["subject_kind"], record["subject_id"]),
    )


def _expire_claim_revision(conn, revision_id: str, now: str) -> None:
    row = conn.execute(
        "SELECT claim_id, status FROM claim_revisions WHERE revision_id = ?", (revision_id,)
    ).fetchone()
    if row is None:
        return
    conn.execute(
        "UPDATE claim_revisions SET status = 'expired', status_changed_at = ?, "
        "display_text = '', value_json = 'null' WHERE revision_id = ?",
        (now, revision_id),
    )
    conn.execute("DELETE FROM claim_fts WHERE revision_id = ?", (revision_id,))
    # The vector is derived from the body; expiry must drop it too so a late
    # embed job or a direct cache read cannot resurrect the expired detail.
    conn.execute("DELETE FROM claim_vectors WHERE revision_id = ?", (revision_id,))
    conn.execute(
        "UPDATE evidence_links SET availability = 'expired' WHERE revision_id = ?",
        (revision_id,),
    )


def _expire_event_part(conn, event_id_part: str, now: str) -> list[str]:
    """Purge one expired source part and propagate the loss into claim evidence.

    F09: expiring the source detail is not enough — every evidence link pointing
    at that part must become unavailable and the dependent claims/task views must
    be recomputed in the same transaction, so a retained conclusion stays valid
    but can never claim that a now-deleted source is still fully readable.
    """
    if "#" not in event_id_part:
        return []
    event_id, remainder = event_id_part.rsplit("#", 1)
    if "@" not in remainder:
        return []
    revision_str, part_str = remainder.split("@", 1)
    try:
        revision = int(revision_str)
        part_index = int(part_str)
    except ValueError:
        return []
    conn.execute(
        "UPDATE event_parts SET body_text = NULL, availability = 'expired', withheld = 0 "
        "WHERE event_id = ? AND revision = ? AND part_index = ?",
        (event_id, revision, part_index),
    )
    rows = conn.execute(
        "SELECT revision_id FROM evidence_links WHERE event_id = ? AND event_revision = ? "
        "AND part_index = ? AND availability = 'full'",
        (event_id, revision, part_index),
    ).fetchall()
    conn.execute(
        "UPDATE evidence_links SET availability = 'expired' WHERE event_id = ? "
        "AND event_revision = ? AND part_index = ? AND availability = 'full'",
        (event_id, revision, part_index),
    )
    from knowledge_weaver.store.claims import recompute_claim

    claim_ids: list[str] = []
    for row in rows:
        claim = conn.execute(
            "SELECT claim_id FROM claim_revisions WHERE revision_id = ?", (row["revision_id"],)
        ).fetchone()
        if claim is not None and claim["claim_id"] not in claim_ids:
            claim_ids.append(claim["claim_id"])
    for claim_id in claim_ids:
        recompute_claim(conn, claim_id)
    return claim_ids


def sweep(store: Store, *, now: str | None = None) -> dict:
    """Classify facts, expire due details, tombstone and purge payloads.

    F09b: ``now`` is parsed once as a real instant.  Offset clocks are compared
    by instant (``+00:00`` equals ``Z``), an invalid/naive clock is refused
    before any write, and every timestamp this sweep writes is canonical UTC.

    AC-52 round 4: the real cleanup operation records its own durable ledger row
    (``pending`` while it runs, then ``succeeded`` / ``failed`` with a visible
    error code), so a cleanup failure stays diagnosable instead of only being
    visible as a raised exception.
    """
    moment_dt = _clock_instant(now)
    moment = _iso(moment_dt)
    with operation_ledger(store, "cleanup"):
        classified = classify_claim_revisions(store, now=moment)
        counts = {"expired": 0, "tombstoned": 0, "claim_revisions": 0, "event_parts": 0}
        with store.transaction() as conn:
            due = _due_lifecycle_records(conn, moment_dt)
            for record in due:
                revision_generation = 0
                if record["subject_kind"] == "claim_revision":
                    row = conn.execute(
                        "SELECT claim_id FROM claim_revisions WHERE revision_id = ?",
                        (record["subject_id"],),
                    ).fetchone()
                    if row is not None:
                        current = conn.execute(
                            "SELECT projection_generation FROM current_claims WHERE claim_id = ?",
                            (row["claim_id"],),
                        ).fetchone()
                        revision_generation = int(current["projection_generation"]) if current else 0
                    _expire_claim_revision(conn, record["subject_id"], moment)
                    counts["claim_revisions"] += 1
                    from knowledge_weaver.store.claims import recompute_claim

                    if row is not None:
                        recompute_claim(conn, row["claim_id"])
                elif record["subject_kind"] == "event_part":
                    _expire_event_part(conn, record["subject_id"], moment)
                    counts["event_parts"] += 1
                _tombstone(
                    conn,
                    record,
                    generation=max(int(record["generation"]) + 1, revision_generation),
                    now=moment,
                )
                counts["expired"] += 1
                counts["tombstoned"] += 1
    return {"classified": classified, **counts}


def is_tombstoned(store: Store, subject_kind: str, subject_id: str) -> bool:
    with store.read() as conn:
        row = conn.execute(
            "SELECT 1 FROM tombstones WHERE subject_kind = ? AND subject_id = ?",
            (subject_kind, subject_id),
        ).fetchone()
    return row is not None


def tombstone_generation(store: Store, subject_kind: str, subject_id: str) -> int | None:
    with store.read() as conn:
        row = conn.execute(
            "SELECT MAX(generation) AS g FROM tombstones WHERE subject_kind = ? AND subject_id = ?",
            (subject_kind, subject_id),
        ).fetchone()
    return int(row["g"]) if row is not None and row["g"] is not None else None


def lifecycle_status(store: Store, *, now: str | None = None) -> dict:
    """Report lifecycle state; ``due_now`` uses the same normalized clock (F09b)."""
    moment_dt = _clock_instant(now)
    with store.read() as conn:
        rows = conn.execute(
            "SELECT class, state, COUNT(*) AS c FROM lifecycle_records GROUP BY class, state"
        ).fetchall()
        active = conn.execute(
            "SELECT expires_at FROM lifecycle_records "
            "WHERE state = 'active' AND expires_at IS NOT NULL"
        ).fetchall()
        tombstones = conn.execute("SELECT COUNT(*) AS c FROM tombstones").fetchone()
    due_now = sum(1 for row in active if _is_due(row["expires_at"], moment_dt))
    summary: dict[str, dict[str, int]] = {}
    for row in rows:
        summary.setdefault(row["class"], {})[row["state"]] = row["c"]
    return {
        "by_class": summary,
        "due_now": due_now,
        "tombstones": int(tombstones["c"]),
    }
