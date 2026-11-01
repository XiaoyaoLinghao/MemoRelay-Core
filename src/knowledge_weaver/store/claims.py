"""Fact-level claims, revisions, evidence links and the current projection (T05).

Rules implemented here (engineering §6, AC-08/35/44/47):

* a claim key is ``(workspace, project, subject, predicate, facet,
  qualifiers)`` — independent properties and desired/observed facets never
  overwrite each other;
* revisions are append-only and carry their own evidence links (exact event
  revision/part, UTF-8 byte span, part hash);
* explicit corrections are compare-and-swap on the expected revision and
  projection generation and are validated against stored user evidence
  (CLAIM-03); unsupported corrections remain ``pending_evidence`` and create no
  revision;
* authority grades order competing revisions, business time breaks ties,
  explicit ``supersedes`` wins, and contradictory equal-authority/tied-time
  claims stay ``conflict`` with all candidates visible;
* the current projection is rebuildable and carries a generation;
* retracted/withheld evidence invalidates dependent claims inside the same
  transaction (SRC-04) so late jobs cannot keep a claim current.

All write paths funnel through :func:`record_claim_revision`,
:func:`apply_correction`, :func:`invalidate_retracted_evidence`,
:func:`invalidate_withheld_evidence` and :func:`rebuild_projection`; nothing may
edit ``current_claims`` directly.
"""

from __future__ import annotations

import datetime as _dt
import json
import sqlite3
from typing import Any, Iterable

from knowledge_weaver import contracts

from .db import Store, utc_now

FACETS = frozenset(
    {"desired", "observed", "reported", "preference", "decision", "capability"}
)
AUTHORITY_RANK = {
    "user_confirmed": 5,
    "tool_verified": 4,
    "reported": 3,
    "inferred": 2,
    "legacy_unknown": 1,
}
REVISION_STATES = frozenset(
    {"candidate", "current", "conflict", "uncertain", "superseded", "revoked"}
)
CURRENT_STATES = frozenset({"current", "conflict", "uncertain", "revoked", "expired"})


class ClaimError(Exception):
    """A fact-level operation cannot be applied; ``code`` is contract-shaped."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


# ---------------------------------------------------------------------------
# Identity / normalisation
# ---------------------------------------------------------------------------


def _compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def normalize_qualifiers(qualifiers: Any) -> str:
    if qualifiers is None:
        return "{}"
    if not isinstance(qualifiers, dict):
        raise ClaimError("INVALID_REQUEST", "qualifiers must be an object")
    return _compact(qualifiers)


def claim_id_for(
    workspace_id: str,
    project_id: str | None,
    subject_id: str,
    predicate: str,
    facet: str,
    qualifiers: Any = None,
) -> str:
    payload = _compact(
        [workspace_id, project_id, subject_id, predicate, facet, normalize_qualifiers(qualifiers)]
    )
    return "claim:" + contracts.sha256_hex(payload)[:40]


def revision_id_for(
    claim_id: str, revision_number: int, value: Any, display_text: str
) -> str:
    payload = _compact([claim_id, revision_number, value, display_text])
    return "rev:" + contracts.sha256_hex(payload)[:40]


# ---------------------------------------------------------------------------
# Evidence resolution (CLAIM-03 / RAW-01)
# ---------------------------------------------------------------------------


def _part_text(conn: sqlite3.Connection, event_id: str, revision: int, part: int):
    row = conn.execute(
        "SELECT p.body_text, p.body_sha256, p.availability, r.lifecycle, e.role, e.origin_kind, "
        "e.actor_id FROM event_parts p JOIN event_revisions r "
        "ON p.event_id = r.event_id AND p.revision = r.revision "
        "JOIN events e ON e.event_id = p.event_id "
        "WHERE p.event_id = ? AND p.revision = ? AND p.part_index = ?",
        (event_id, int(revision), int(part)),
    ).fetchone()
    return row


def resolve_evidence(
    conn: sqlite3.Connection,
    references: Iterable[dict],
    *,
    require_user: bool = False,
) -> tuple[list[dict], list[str]]:
    """Resolve and verify evidence refs against stored events.

    Returns (links, actors) where each link has the exact stored part hash, span
    and availability. Raises :class:`ClaimError` when an event/revision/part is
    missing, hidden or explicitly not user evidence when required.
    """
    links: list[dict] = []
    actors: list[str] = []
    for reference in references:
        event_id = reference.get("event_id")
        revision = int(reference.get("event_revision") or 0)
        part = int(reference.get("part_index") or 0)
        start = reference.get("start_byte")
        end = reference.get("end_byte")
        expected_hash = reference.get("body_sha256")
        if not isinstance(start, int) or not isinstance(end, int) or start < 0 or end < start:
            raise ClaimError("INVALID_REQUEST", "evidence byte span is invalid")
        row = _part_text(conn, event_id, revision, part)
        if row is None:
            raise ClaimError("SOURCE_NOT_FOUND", "evidence event revision/part not found")
        if row["lifecycle"] == "retracted" or row["availability"] != "full":
            raise ClaimError("SOURCE_EXPIRED", "evidence source is not available")
        if require_user and not (
            row["role"] == "user" and row["origin_kind"] == "user_input"
        ):
            raise ClaimError(
                "INVALID_REQUEST",
                "correction evidence must be native user input, not assistant/tool content",
            )
        if expected_hash and expected_hash != row["body_sha256"]:
            raise ClaimError("INVALID_REQUEST", "evidence body hash does not match the stored part")
        text = row["body_text"]
        if text is None:
            raise ClaimError("SOURCE_EXPIRED", "evidence body is not available")
        encoded = text.encode("utf-8")
        if end > len(encoded):
            raise ClaimError("INVALID_REQUEST", "evidence span exceeds the stored part")
        try:
            quote = encoded[start:end].decode("utf-8")
        except UnicodeDecodeError:
            raise ClaimError("INVALID_REQUEST", "evidence span splits a UTF-8 character") from None
        links.append(
            {
                "event_id": event_id,
                "event_revision": revision,
                "part_index": part,
                "start_byte": start,
                "end_byte": end,
                "body_sha256": row["body_sha256"],
                "quote_sha256": contracts.body_digest(quote),
                "quote": quote,
                "availability": "full",
            }
        )
        if row["actor_id"] not in actors:
            actors.append(row["actor_id"])
    if not links:
        raise ClaimError("INVALID_REQUEST", "at least one evidence reference is required")
    return links, actors



def _index_revision(conn, revision_id: str, display_text: str, value) -> None:
    """Update keyword index and enqueue the optional embedding job atomically."""
    from ..retrieval.keyword import index_revision_in_transaction

    value_text = value if isinstance(value, str) else _compact(value)
    index_revision_in_transaction(conn, revision_id, f"{display_text} {value_text}")
    try:
        from ..embeddings.index import active_model, enqueue_embed_in_transaction

        if active_model():
            enqueue_embed_in_transaction(conn, revision_id)
    except Exception:  # noqa: BLE001 - semantic capability is optional
        pass


# ---------------------------------------------------------------------------
# Claim / revision writes
# ---------------------------------------------------------------------------


def _ensure_claim(
    conn: sqlite3.Connection,
    *,
    workspace_id: str,
    project_id: str | None,
    subject_id: str,
    predicate: str,
    facet: str,
    qualifiers: Any,
) -> str:
    if facet not in FACETS:
        raise ClaimError("INVALID_REQUEST", f"unsupported facet {facet!r}")
    for field, value in (
        ("workspace_id", workspace_id),
        ("subject_id", subject_id),
        ("predicate", predicate),
    ):
        if not isinstance(value, str) or not contracts.TOKEN_RE.match(value) or len(value) > 180:
            raise ClaimError("INVALID_REQUEST", f"{field} must be a short token")
    if project_id is not None and (
        not isinstance(project_id, str)
        or not contracts.TOKEN_RE.match(project_id)
        or len(project_id) > 180
    ):
        raise ClaimError("INVALID_REQUEST", "project_id must be a short token or null")
    qualifiers_json = normalize_qualifiers(qualifiers)
    claim_id = claim_id_for(
        workspace_id, project_id, subject_id, predicate, facet, qualifiers
    )
    conn.execute(
        "INSERT OR IGNORE INTO claims "
        "(claim_id, workspace_id, project_id, subject_id, predicate, facet, qualifiers_json, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            claim_id,
            workspace_id,
            project_id,
            subject_id,
            predicate,
            facet,
            qualifiers_json,
            utc_now(),
        ),
    )
    return claim_id


def _next_revision_number(conn, claim_id: str) -> int:
    row = conn.execute(
        "SELECT COALESCE(MAX(revision_number), 0) AS m FROM claim_revisions WHERE claim_id = ?",
        (claim_id,),
    ).fetchone()
    return int(row["m"]) + 1


def record_claim_revision(store: Store, **kwargs: Any) -> dict:
    """Append one validated revision (convenience transaction wrapper)."""
    with store.transaction() as conn:
        return record_claim_revision_in_transaction(conn, **kwargs)


def record_claim_revision_in_transaction(
    conn: sqlite3.Connection,
    *,
    workspace_id: str,
    project_id: str | None,
    subject_id: str,
    predicate: str,
    facet: str,
    qualifiers: Any,
    value: Any,
    display_text: str,
    authority: str,
    assertion_kind: str,
    event_time: dict | None = None,
    evidence: Iterable[dict] = (),
    supersedes: Iterable[str] = (),
    correction_of: str | None = None,
) -> dict:
    """Append one validated revision inside the caller's transaction (T07/jobs)."""
    if authority not in AUTHORITY_RANK:
        raise ClaimError("INVALID_REQUEST", f"unsupported authority {authority!r}")
    if not isinstance(display_text, str) or not display_text or len(display_text) > 8000:
        raise ClaimError("INVALID_REQUEST", "display_text must be a non-empty string")
    time_value = (event_time or {}).get("value")
    precision = (event_time or {}).get("precision") or "unknown"
    if precision not in ("instant", "day", "unknown"):
        raise ClaimError("INVALID_REQUEST", "unsupported event time precision")
    claim_id = _ensure_claim(
        conn,
        workspace_id=workspace_id,
        project_id=project_id,
        subject_id=subject_id,
        predicate=predicate,
        facet=facet,
        qualifiers=qualifiers,
    )
    links, actors = resolve_evidence(conn, evidence)
    number = _next_revision_number(conn, claim_id)
    revision_id = revision_id_for(claim_id, number, value, display_text)
    conn.execute(
        "INSERT INTO claim_revisions (revision_id, claim_id, revision_number, value_json, "
        "display_text, authority, assertion_kind, event_time_value, event_time_precision, "
        "event_time_timezone, status, supersedes_json, correction_of, source_actors_json, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'candidate', ?, ?, ?, ?)",
        (
            revision_id,
            claim_id,
            number,
            _compact(value),
            display_text,
            authority,
            assertion_kind,
            time_value,
            precision,
            (event_time or {}).get("timezone") or "UTC",
            _compact(list(supersedes)),
            correction_of,
            _compact(actors),
            utc_now(),
        ),
    )
    for link in links:
        conn.execute(
            "INSERT INTO evidence_links (revision_id, event_id, event_revision, part_index, "
            "start_byte, end_byte, body_sha256, quote_sha256, availability, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'full', ?)",
            (
                revision_id,
                link["event_id"],
                link["event_revision"],
                link["part_index"],
                link["start_byte"],
                link["end_byte"],
                link["body_sha256"],
                link["quote_sha256"],
                utc_now(),
            ),
        )
    _index_revision(conn, revision_id, display_text, value)
    projection = recompute_claim(conn, claim_id)
    return {
        "claim_id": claim_id,
        "revision_id": revision_id,
        "revision_number": number,
        "state": projection["state"],
        "selected_revision_ids": projection["selected_revision_ids"],
        "projection_generation": projection["projection_generation"],
    }


# ---------------------------------------------------------------------------
# Selection / projection
# ---------------------------------------------------------------------------


def _time_key(revision: dict):
    precision = revision["event_time_precision"]
    value = revision["event_time_value"]
    if precision == "unknown" or not value:
        return None
    try:
        if precision == "day":
            moment = _dt.datetime.fromisoformat(value + "T00:00:00+00:00")
        else:
            text = value[:-1] + "+00:00" if value.endswith("Z") else value
            moment = _dt.datetime.fromisoformat(text)
            if moment.tzinfo is None:
                return None
            moment = moment.astimezone(_dt.timezone.utc)
    except ValueError:
        return None
    return moment.timestamp()


#: Business times further in the future than this (relative to the trusted
#: ingestion clock) are anomalous: they are isolated as ``uncertain`` instead of
#: silently dominating ``current``. The window leaves room for clock drift and
#: short scheduled/planned timestamps without trusting arbitrary future dates.
FUTURE_TOLERANCE_DAYS = 366


def _parse_iso_ts(value: str | None) -> float | None:
    if not value:
        return None
    try:
        text = value[:-1] + "+00:00" if value.endswith("Z") else value
        parsed = _dt.datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=_dt.timezone.utc)
    return parsed.astimezone(_dt.timezone.utc).timestamp()


def _time_evidence(revision: dict) -> tuple[float | None, bool]:
    """Return (trusted business-time key, anomalous-future flag).

    The ingestion time (revision ``created_at``) is the trusted business clock:
    an event time far in the future is a data anomaly, not a newer truth. Unknown
    precision has no ordering key at all (it never wins a max() comparison).
    """
    key = _time_key(revision)
    if key is None:
        return None, False
    reference = _parse_iso_ts(revision.get("created_at")) or _dt.datetime.now(
        _dt.timezone.utc
    ).timestamp()
    anomalous = key > reference + FUTURE_TOLERANCE_DAYS * 86400
    return key, anomalous


def _source_business_time(link: dict) -> float | None:
    """Business time of the *source event* a revision cites (AC08-R4)."""
    value = link.get("source_event_time_value")
    precision = link.get("source_event_time_precision")
    if not value or precision in (None, "unknown"):
        return None
    try:
        if precision == "day":
            moment = _dt.datetime.fromisoformat(value + "T00:00:00+00:00")
        else:
            text = value[:-1] + "+00:00" if value.endswith("Z") else value
            moment = _dt.datetime.fromisoformat(text)
            if moment.tzinfo is None:
                return None
            moment = moment.astimezone(_dt.timezone.utc)
    except ValueError:
        return None
    return moment.timestamp()


def _ordering_time(revision: dict) -> tuple[float | None, bool]:
    """Business-time key used for current selection (AC08-R4 / AC-35).

    An explicit user correction inherits the corrected fact's *age* (its
    business time, used for retention/explainability), but its authority takes
    effect at the business time of the user's own correction evidence.  A late
    asynchronous extraction whose business time is earlier than the correction
    (e.g. a March record arriving after a June correction) therefore stays
    stale, while a genuinely later native statement can still change the fact.
    The corrected fact's original age is never rewritten: this key only orders
    the projection, retention still sees ``event_time_value``.
    """
    if _is_explicit_correction(revision) and revision["links"]:
        evidence_times = [
            moment
            for moment in (_source_business_time(link) for link in revision["links"])
            if moment is not None
        ]
        if evidence_times:
            key = max(evidence_times)
            reference = _parse_iso_ts(revision.get("created_at"))
            anomalous = (
                reference is not None
                and key > reference + FUTURE_TOLERANCE_DAYS * 86400
            )
            return key, anomalous
    return _time_evidence(revision)


def _load_revisions(conn, claim_id: str) -> list[dict]:
    rows = conn.execute(
        "SELECT r.*, c.predicate FROM claim_revisions r JOIN claims c ON c.claim_id=r.claim_id "
        "WHERE r.claim_id = ? ORDER BY r.revision_number",
        (claim_id,),
    ).fetchall()
    revisions = []
    for row in rows:
        links = [
            dict(link)
            for link in conn.execute(
                "SELECT l.*, er.event_time_value AS source_event_time_value, "
                "er.event_time_precision AS source_event_time_precision, "
                "er.event_time_timezone AS source_event_time_timezone "
                "FROM evidence_links l LEFT JOIN event_revisions er "
                "ON er.event_id = l.event_id AND er.revision = l.event_revision "
                "WHERE l.revision_id = ?",
                (row["revision_id"],),
            ).fetchall()
        ]
        revisions.append(
            {
                **dict(row),
                "supersedes": json.loads(row["supersedes_json"]),
                "source_actors": json.loads(row["source_actors_json"]),
                "value": json.loads(row["value_json"]),
                "links": links,
            }
        )
    return revisions


def _evidence_class(revision: dict) -> str:
    """Accepted support / uncertain / revoked, separate from raw readability."""
    if not revision["links"]:
        return "uncertain"
    availabilities = {link["availability"] for link in revision["links"]}
    if "retracted" in availabilities:
        return "revoked"
    if availabilities == {"full"}:
        return "full"
    # Retention expiry removes raw detail, not the native assertion's truth.
    # Keep ordinary retained facts current with an explicit expired-source flag.
    # A verified task still needs readable checking evidence (not just a label).
    value = revision.get("value")
    needs_live_verification = revision.get("predicate") == "task.status" and isinstance(value,dict) and value.get("status") in {"verified","verified_done"}
    if availabilities <= {"full","expired"} and not needs_live_verification and str(revision.get("display_text") or "").strip():
        return "full"
    return "uncertain"


def _evidence_signature(link: dict) -> tuple[str, int, int, int, int]:
    """Exact source identity of one evidence span (used to spot re-extractions)."""
    return (
        link["event_id"],
        int(link["event_revision"]),
        int(link["part_index"]),
        int(link["start_byte"]),
        int(link["end_byte"]),
    )


def _is_explicit_correction(revision: dict) -> bool:
    """True for revisions written by :func:`apply_correction`.

    Only the correction write path links a replacement to the revision it
    replaces; extraction proposals can merely carry the ``explicit_correction``
    assertion kind but never create those lineage edges.
    """
    return revision["assertion_kind"] == "explicit_correction" and (
        revision["correction_of"] is not None or bool(revision["supersedes"])
    )


def _correction_covered_evidence(revisions: list[dict]) -> set[tuple[str, int, int, int, int]]:
    """Evidence spans that an explicit user correction has already replaced.

    Walks every correction chain backwards (``correction_of``/``supersedes``,
    transitively inside the claim) and returns the exact evidence signatures of
    the replaced revisions. AC-08 "更正优先于异步提炼": a late asynchronous
    extraction that merely replays one of these spans is knowledge the user has
    already corrected and must not compete with the corrected current again.
    The directly replaced revision keeps its own lineage; only its evidence is
    marked as corrected.
    """
    by_id = {revision["revision_id"]: revision for revision in revisions}
    pending: list[str] = []
    for revision in revisions:
        if not _is_explicit_correction(revision):
            continue
        pending.extend(revision["supersedes"])
        if revision["correction_of"]:
            pending.append(revision["correction_of"])
    covered: set[tuple[str, int, int, int, int]] = set()
    visited: set[str] = set()
    while pending:
        revision_id = pending.pop()
        if revision_id in visited:
            continue
        visited.add(revision_id)
        replaced = by_id.get(revision_id)
        if replaced is None:
            continue
        covered.update(_evidence_signature(link) for link in replaced["links"])
        pending.extend(replaced["supersedes"])
        if replaced["correction_of"]:
            pending.append(replaced["correction_of"])
    return covered


def recompute_claim(conn: sqlite3.Connection, claim_id: str) -> dict:
    revisions = _load_revisions(conn, claim_id)
    now = utc_now()
    if not revisions:
        return _store_current(conn, claim_id, [], "uncertain", now)
    superseded_by: set[str] = set()
    for revision in revisions:
        superseded_by.update(revision["supersedes"])
    active = [
        revision
        for revision in revisions
        if revision["revision_id"] not in superseded_by
        and revision["status"] != "expired"
    ]
    # Evidence class decides support; explicit supersede removes a revision.
    supported = [revision for revision in active if _evidence_class(revision) == "full"]
    # AC-08 "更正优先于异步提炼": evidence spans that an explicit user correction
    # already replaced are corrected lineage, not competing candidates. A late
    # asynchronous extraction that replays one of those spans (same event/revision/
    # part/byte range) is kept for history but never re-enters authority competition.
    covered_evidence = _correction_covered_evidence(revisions)
    corrected_replays = {
        revision["revision_id"]
        for revision in supported
        if not _is_explicit_correction(revision)
        and revision["links"]
        and all(_evidence_signature(link) in covered_evidence for link in revision["links"])
    }
    eligible = [
        revision for revision in supported if revision["revision_id"] not in corrected_replays
    ]
    state: str
    selected: list[dict]
    anomalous_future: set[str] = set()
    if not supported:
        classes = {_evidence_class(revision) for revision in active}
        # An expired old history revision must not hide an independent newer
        # fact whose source detail expired but whose retained value is valid.
        if not active and any(revision["status"] == "expired" for revision in revisions):
            state = "expired"
        elif "revoked" in classes:
            state = "revoked"
        elif "uncertain" in classes:
            state = "uncertain"
        else:
            state = "revoked"
        selected = []
    elif not eligible:
        # Every full-evidence candidate only replays evidence the user already
        # corrected; none may become current and no trustworthy candidate remains.
        state = "uncertain"
        selected = []
    else:
        # F05/TIME-02: isolate abnormal future business times *before* authority
        # ranking. A 2099 anomaly must only be excluded itself; it must not shrink
        # the candidate set that authority ranking sees, otherwise a user_confirmed
        # anomaly would hide a well-supported tool_verified current.
        trusted: list[dict] = []
        trusted_time: dict[str, float | None] = {}
        for revision in eligible:
            key, is_anomalous = _ordering_time(revision)
            if is_anomalous:
                # An abnormal future business time is isolated as an explainable
                # uncertain state; it never becomes current and never wins a
                # max() comparison against trusted times.
                anomalous_future.add(revision["revision_id"])
                continue
            trusted.append(revision)
            trusted_time[revision["revision_id"]] = key
        if not trusted:
            # Every supported candidate is an abnormal future time: none may
            # dominate, no current is fabricated, and each stays visibly uncertain.
            state = "uncertain"
            selected = []
        else:
            max_rank = max(AUTHORITY_RANK[revision["authority"]] for revision in trusted)
            top = [
                revision
                for revision in trusted
                if AUTHORITY_RANK[revision["authority"]] == max_rank
            ]
            trusted_times = {
                revision["revision_id"]: trusted_time[revision["revision_id"]]
                for revision in top
                if trusted_time[revision["revision_id"]] is not None
            }
            if trusted_times:
                latest = max(trusted_times.values())
                tied = [
                    revision
                    for revision in top
                    if trusted_times.get(revision["revision_id"]) == latest
                ]
            else:
                # No revision has an orderable, trustworthy business time.
                tied = list(top)
                latest = None
            # AC-08: an explicit user correction outranks an extraction that is
            # appended *after* it at the same authority and business time. The
            # correction expresses the user's decision for that business time;
            # a late replayed old fact is stale and stays as non-current history.
            correction_numbers = [
                revision["revision_number"]
                for revision in tied
                if _is_explicit_correction(revision)
            ]
            if correction_numbers:
                first_correction = min(correction_numbers)
                tied = [
                    revision
                    for revision in tied
                    if _is_explicit_correction(revision)
                    or revision["revision_number"] < first_correction
                ]
            values = {_compact(revision["value"]) for revision in tied}
            if not tied:
                state = "uncertain"
                selected = []
            elif latest is None:
                # TIME-01: unknown business time never silently becomes current;
                # equal-valued unknown times still stay explainable as uncertain.
                state = "uncertain"
                selected = []
            elif len(values) > 1:
                state = "conflict"
                selected = sorted(tied, key=lambda revision: revision["revision_number"])
            else:
                state = "current"
                selected = [max(tied, key=lambda revision: revision["revision_number"])]
    selected_ids = [revision["revision_id"] for revision in selected]
    # Persist per-revision status so history stays explainable.
    selected_set = set(selected_ids)
    revoked_evidence = {
        revision["revision_id"]
        for revision in revisions
        if _evidence_class(revision) == "revoked"
    }
    uncertain_evidence = {
        revision["revision_id"]
        for revision in revisions
        if _evidence_class(revision) == "uncertain"
    }
    for revision in revisions:
        rid = revision["revision_id"]
        if revision["status"] == "expired":
            # Retention owns expiry; projection must not resurrect the revision.
            continue
        if rid in selected_set:
            new_status = "conflict" if state == "conflict" else "current"
        elif rid in superseded_by:
            new_status = "superseded"
        elif rid in revoked_evidence:
            new_status = "revoked"
        elif rid in uncertain_evidence:
            new_status = "uncertain"
        elif rid in anomalous_future:
            # An abnormal future time stays visible and queryable as uncertain.
            new_status = "uncertain"
        elif rid in corrected_replays:
            # A replayed old fact the user already corrected stays as history.
            new_status = "superseded"
        elif _time_key(revision) is None and state != "conflict":
            # TIME-01: unknown business time never silently loses/becomes current.
            new_status = "uncertain"
        else:
            new_status = "superseded"
        if revision["status"] != new_status:
            conn.execute(
                "UPDATE claim_revisions SET status = ?, status_changed_at = ? WHERE revision_id = ?",
                (new_status, now, rid),
            )
    return _store_current(conn, claim_id, selected_ids, state, now)


def _store_current(
    conn: sqlite3.Connection,
    claim_id: str,
    selected_ids: list[str],
    state: str,
    now: str,
) -> dict:
    row = conn.execute(
        "SELECT * FROM current_claims WHERE claim_id = ?", (claim_id,)
    ).fetchone()
    selected_json = _compact(selected_ids)
    if row is None:
        generation = 1
        conn.execute(
            "INSERT INTO current_claims "
            "(claim_id, selected_revision_ids_json, state, projection_generation, updated_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (claim_id, selected_json, state, generation, now),
        )
    else:
        changed = row["selected_revision_ids_json"] != selected_json or row["state"] != state
        generation = int(row["projection_generation"]) + (1 if changed else 0)
        conn.execute(
            "UPDATE current_claims SET selected_revision_ids_json = ?, state = ?, "
            "projection_generation = ?, updated_at = ? WHERE claim_id = ?",
            (selected_json, state, generation, now, claim_id),
        )
    claim = conn.execute(
        "SELECT workspace_id, project_id, subject_id, predicate FROM claims WHERE claim_id = ?",
        (claim_id,),
    ).fetchone()
    if claim is not None and claim["predicate"] == "task.status":
        from . import tasks as task_views

        task_views.recompute_task(
            conn,
            workspace_id=claim["workspace_id"],
            project_id=claim["project_id"],
            subject_id=claim["subject_id"],
        )
    return {
        "claim_id": claim_id,
        "state": state,
        "selected_revision_ids": selected_ids,
        "projection_generation": generation,
    }


def rebuild_projection(store: Store) -> int:
    """Rebuild every claim's current projection from revisions (idempotent).

    AC-52/R4: this is the real projection operation, so it records its own
    durable ledger entry (pending while running, succeeded/failed with a visible
    error code afterwards) instead of relying on a caller to invent a job.
    """
    from .status import operation_ledger

    with operation_ledger(store, "project"):
        with store.transaction() as conn:
            claim_ids = [
                row["claim_id"] for row in conn.execute("SELECT claim_id FROM claims").fetchall()
            ]
            for claim_id in claim_ids:
                recompute_claim(conn, claim_id)
    return len(claim_ids)


# ---------------------------------------------------------------------------
# Evidence invalidation hooks (called from the event inbox transaction)
# ---------------------------------------------------------------------------


def _affected_claims(conn, rows) -> list[str]:
    claim_ids: list[str] = []
    for row in rows:
        claim_id = conn.execute(
            "SELECT claim_id FROM claim_revisions WHERE revision_id = ?",
            (row["revision_id"],),
        ).fetchone()
        if claim_id is not None and claim_id["claim_id"] not in claim_ids:
            claim_ids.append(claim_id["claim_id"])
    return claim_ids


def invalidate_retracted_evidence(conn: sqlite3.Connection, event_id: str, revision: int) -> None:
    """SRC-04: retracting a source removes its support from dependent claims."""
    rows = conn.execute(
        "SELECT * FROM evidence_links WHERE event_id = ? AND event_revision = ?",
        (event_id, int(revision)),
    ).fetchall()
    conn.execute(
        "UPDATE evidence_links SET availability = 'retracted' "
        "WHERE event_id = ? AND event_revision = ?",
        (event_id, int(revision)),
    )
    # A retracted body may no longer support its derived vectors/cache either.
    for row in rows:
        conn.execute("DELETE FROM claim_vectors WHERE revision_id = ?", (row["revision_id"],))
    for claim_id in _affected_claims(conn, rows):
        recompute_claim(conn, claim_id)


def _scrub_revision_derived_content(conn: sqlite3.Connection, revision_id: str) -> None:
    """SEC-02: remove privacy-derived payloads of a downgraded revision.

    A revision whose evidence was redacted/withheld must no longer carry the old
    derived text, value, full-text tokens or vector. The non-secret lineage
    (claim id, revision number, authority, business time, evidence hashes) stays
    so history/explainability survives; only the potentially secret payload is
    cleared. Returns nothing; callers recompute the claim afterwards.
    """
    conn.execute(
        "UPDATE claim_revisions SET display_text = '', value_json = 'null' WHERE revision_id = ?",
        (revision_id,),
    )
    conn.execute("DELETE FROM claim_fts WHERE revision_id = ?", (revision_id,))
    conn.execute("DELETE FROM claim_vectors WHERE revision_id = ?", (revision_id,))


def invalidate_withheld_evidence(conn: sqlite3.Connection, event_id: str, before_revision: int) -> None:
    """SEC-02: a redaction upgrade removes dependent old body support.

    Every older source lineage of the event is affected, whatever its current
    availability: a detail may already be ``expired`` (retention) or
    ``retracted`` before the redaction arrives, and a fact derived from it is
    just as secret-bearing as one whose link is still ``full``. The derived
    payloads (display text, value, keyword tokens, vectors) are therefore purged
    for *all* such revisions in the same transaction, so nothing keeps leaking
    through recall or export.

    Availability keeps its own truthful meaning: only ``full`` links are
    downgraded to ``withheld``; already-``expired``/``retracted`` links keep that
    state (a redaction does not rewrite retention/retraction history). The
    classified source body itself must not survive either: bodies of parts that
    are still ``full`` are purged by the inbox hook, while an older part that
    expired or was retracted earlier still holds its body, so the remaining
    older bodies are purged here in the same transaction. Deletion is later
    zeroed at the file level by the store's secure-delete/checkpoint maintenance.
    """
    rows = conn.execute(
        "SELECT * FROM evidence_links WHERE event_id = ? AND event_revision < ?",
        (event_id, int(before_revision)),
    ).fetchall()
    conn.execute(
        "UPDATE evidence_links SET availability = 'withheld' "
        "WHERE event_id = ? AND event_revision < ? AND availability = 'full'",
        (event_id, int(before_revision)),
    )
    # A redaction classifies every older revision of the event as secret-bearing.
    # The inbox hook in ``events.py`` clears bodies of parts that are still
    # ``full``; parts that expired or were retracted earlier still hold their
    # body, so purge the remaining older bodies here, in the same transaction.
    # Only the payload is removed: availability keeps its own meaning.
    conn.execute(
        "UPDATE event_parts SET body_text = NULL "
        "WHERE event_id = ? AND revision < ? AND body_text IS NOT NULL",
        (event_id, int(before_revision)),
    )
    scrubbed: list[str] = []
    for row in rows:
        if row["revision_id"] not in scrubbed:
            _scrub_revision_derived_content(conn, row["revision_id"])
            scrubbed.append(row["revision_id"])
    affected = _affected_claims(conn, rows)
    # F02-R4: a claim row also carries source-derived metadata (``qualifiers``,
    # i.e. extracted properties such as a connection note).  A redaction
    # classifies the whole touched lineage as secret-bearing, so that metadata
    # may no longer be served by recall, exported in a snapshot, restored or
    # re-indexed.  Clear it in this same transaction; the claim *identity*
    # (claim_id) and the non-secret lineage stay, so history remains explainable
    # and unrelated claims are untouched.
    for claim_id in affected:
        conn.execute(
            "UPDATE claims SET qualifiers_json = '{}' WHERE claim_id = ?", (claim_id,)
        )
    for claim_id in affected:
        recompute_claim(conn, claim_id)


# ---------------------------------------------------------------------------
# Reads
# ---------------------------------------------------------------------------


def get_current(store: Store, claim_id: str) -> dict | None:
    with store.read() as conn:
        row = conn.execute("SELECT * FROM current_claims WHERE claim_id = ?", (claim_id,)).fetchone()
    if row is None:
        return None
    return {
        "claim_id": row["claim_id"],
        "state": row["state"],
        "selected_revision_ids": json.loads(row["selected_revision_ids_json"]),
        "projection_generation": row["projection_generation"],
    }


def current_items(
    store: Store,
    *,
    workspace_id: str,
    project_ids: Iterable[str] | None = None,
    include_states: Iterable[str] = ("current", "conflict", "uncertain"),
) -> list[dict]:
    """Project current claims into contract ``MemoryItem`` shapes (recall input)."""
    allowed_states = set(include_states)
    projects = list(project_ids) if project_ids is not None else None
    items: list[dict] = []
    with store.read() as conn:
        rows = conn.execute(
            "SELECT c.*, cc.selected_revision_ids_json, cc.state, cc.projection_generation "
            "FROM claims c JOIN current_claims cc ON cc.claim_id = c.claim_id "
            "WHERE c.workspace_id = ?",
            (workspace_id,),
        ).fetchall()
        from .entities import resolve_canonical_subject

        for row in rows:
            if row["state"] not in allowed_states:
                continue
            if projects is not None and row["project_id"] not in projects:
                continue
            selected = json.loads(row["selected_revision_ids_json"])
            if selected:
                primary_revision_id = selected[0]
            else:
                # Degraded states (uncertain/revoked) still expose the most recent
                # revision as explainable evidence, marked as not current.
                latest = conn.execute(
                    "SELECT revision_id FROM claim_revisions WHERE claim_id = ? "
                    "ORDER BY revision_number DESC LIMIT 1",
                    (row["claim_id"],),
                ).fetchone()
                if latest is None:
                    continue
                primary_revision_id = latest["revision_id"]
            revision = conn.execute(
                "SELECT * FROM claim_revisions WHERE revision_id = ?", (primary_revision_id,)
            ).fetchone()
            if revision is None:
                continue
            links = conn.execute(
                "SELECT * FROM evidence_links WHERE revision_id = ? ORDER BY event_id, event_revision, part_index, start_byte",
                (primary_revision_id,),
            ).fetchall()
            sources = [
                {
                    "event_id": link["event_id"],
                    "event_revision": link["event_revision"],
                    "part_index": link["part_index"],
                    "start_byte": link["start_byte"],
                    "end_byte": link["end_byte"],
                    "body_sha256": link["body_sha256"],
                    "availability": link["availability"],
                }
                for link in links
            ]
            items.append(
                {
                    "claim_id": row["claim_id"],
                    "revision_id": revision["revision_id"],
                    "project_id": row["project_id"],
                    "subject_id": resolve_canonical_subject(
                        conn,
                        workspace_id=row["workspace_id"],
                        project_id=row["project_id"],
                        subject_id=row["subject_id"],
                    ),
                    "projection_generation": row["projection_generation"],
                    "value": json.loads(revision["value_json"]),
                    "qualifiers": json.loads(row["qualifiers_json"]),
                    "predicate": row["predicate"],
                    "facet": row["facet"],
                    "text": revision["display_text"],
                    "state": row["state"],
                    "evidence_grade": revision["authority"],
                    "event_time": {
                        "value": revision["event_time_value"],
                        "precision": revision["event_time_precision"],
                        "timezone": revision["event_time_timezone"],
                    },
                    "sources": sources,
                    "conflict_revision_ids": selected[1:] if row["state"] == "conflict" else [],
                    "source_actors": json.loads(revision["source_actors_json"]),
                    "relevance_score": 0.0,
                    "source_detail_expired": bool(sources)
                    and any(link["availability"] != "full" for link in sources),
                }
            )
    items.sort(key=lambda item: (item["project_id"] or "", item["subject_id"], item["predicate"], item["facet"]))
    return items


# ---------------------------------------------------------------------------
# Corrections (CAS + evidence validation, CLAIM-02/CLAIM-03)
# ---------------------------------------------------------------------------


def _store_operation(conn, operation_id: str, claim_id: str, result: dict, now: str) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO correction_operations "
        "(operation_id, claim_id, status, result_json, created_at) VALUES (?, ?, ?, ?, ?)",
        (operation_id, claim_id, result["status"], _compact(result), now),
    )


def _correction_result(
    correction: dict, status: str, claim_id: str, revision_id, generation: int, watermark: int
) -> dict:
    return {
        "operation_id": correction["operation_id"],
        "status": status,
        "claim_id": claim_id,
        "revision_id": revision_id,
        "projection_generation": generation,
        "watermark": watermark,
    }


def apply_correction(store: Store, correction: dict) -> dict:
    """Apply a user correction with compare-and-swap and evidence validation."""
    issues = contracts.validate_instance("Correction", correction)
    if issues:
        raise ClaimError("INVALID_REQUEST", f"invalid correction: {issues[0].path}: {issues[0].message}")
    operation_id = correction["operation_id"]
    claim_id = correction["claim_id"]
    with store.transaction() as conn:
        existing = conn.execute(
            "SELECT * FROM correction_operations WHERE operation_id = ?", (operation_id,)
        ).fetchone()
        if existing is not None:
            result = json.loads(existing["result_json"])
            result["status"] = "duplicate"
            return result
        claim = conn.execute("SELECT * FROM claims WHERE claim_id = ?", (claim_id,)).fetchone()
        now = utc_now()
        watermark = store.watermark(conn)
        current = conn.execute(
            "SELECT * FROM current_claims WHERE claim_id = ?", (claim_id,)
        ).fetchone()
        generation = int(current["projection_generation"]) if current is not None else 0
        if claim is None:
            result = _correction_result(correction, "pending_evidence", claim_id, None, generation, watermark)
            _store_operation(conn, operation_id, claim_id, result, now)
            return result
        selected = json.loads(current["selected_revision_ids_json"]) if current is not None else []
        expected_revision = correction["expected_revision_id"]
        revision_exists = conn.execute(
            "SELECT 1 FROM claim_revisions WHERE revision_id = ? AND claim_id = ?",
            (expected_revision, claim_id),
        ).fetchone()
        if (
            revision_exists is None
            or correction["expected_projection_generation"] != generation
            or (selected and expected_revision not in selected)
        ):
            result = _correction_result(correction, "conflict", claim_id, None, generation, watermark)
            _store_operation(conn, operation_id, claim_id, result, now)
            return result
        try:
            links, actors = resolve_evidence(conn, correction["evidence"], require_user=True)
        except ClaimError:
            result = _correction_result(correction, "pending_evidence", claim_id, None, generation, watermark)
            _store_operation(conn, operation_id, claim_id, result, now)
            return result
        # CLAIM-03: the proposed replacement must actually be supported by the quote.
        primary = links[0]["quote"]
        display_text = correction["display_text"]
        value = correction["new_value"]
        supported = display_text in primary or (
            isinstance(value, str) and value and value in primary
        ) or (_compact(value) in primary)
        if not supported:
            result = _correction_result(correction, "pending_evidence", claim_id, None, generation, watermark)
            _store_operation(conn, operation_id, claim_id, result, now)
            return result
        number = _next_revision_number(conn, claim_id)
        revision_id = revision_id_for(claim_id, number, value, display_text)
        event_time = {
            "value": correction.get("reason") and None,  # business time is inherited below
            "precision": "unknown",
            "timezone": "UTC",
        }
        parent_row = conn.execute(
            "SELECT * FROM claim_revisions WHERE revision_id = ?", (expected_revision,)
        ).fetchone()
        if parent_row is not None:
            event_time = {
                "value": parent_row["event_time_value"],
                "precision": parent_row["event_time_precision"],
                "timezone": parent_row["event_time_timezone"],
            }
        conn.execute(
            "INSERT INTO claim_revisions (revision_id, claim_id, revision_number, value_json, "
            "display_text, authority, assertion_kind, event_time_value, event_time_precision, "
            "event_time_timezone, status, supersedes_json, correction_of, source_actors_json, created_at) "
            "VALUES (?, ?, ?, ?, ?, 'user_confirmed', 'explicit_correction', ?, ?, ?, 'candidate', ?, ?, ?, ?)",
            (
                revision_id,
                claim_id,
                number,
                _compact(value),
                display_text,
                event_time["value"],
                event_time["precision"],
                event_time["timezone"],
                _compact([expected_revision]),
                expected_revision,
                _compact(actors),
                now,
            ),
        )
        for link in links:
            conn.execute(
                "INSERT INTO evidence_links (revision_id, event_id, event_revision, part_index, "
                "start_byte, end_byte, body_sha256, quote_sha256, availability, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'full', ?)",
                (
                    revision_id,
                    link["event_id"],
                    link["event_revision"],
                    link["part_index"],
                    link["start_byte"],
                    link["end_byte"],
                    link["body_sha256"],
                    link["quote_sha256"],
                    now,
                ),
            )
        _index_revision(conn, revision_id, display_text, value)
        projection = recompute_claim(conn, claim_id)
        result = _correction_result(
            correction,
            "applied",
            claim_id,
            revision_id,
            projection["projection_generation"],
            store.watermark(conn),
        )
        _store_operation(conn, operation_id, claim_id, result, now)
        return result
