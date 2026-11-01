"""KW event inbox: durable receipt of sanitized source evidence.

Implements the T02 slice of the contract semantics:

* whole-request shape errors (400 INVALID_REQUEST / UNSUPPORTED_SCHEMA / 413)
  are raised as :class:`StoreRejection`; per-part outcomes are receipts;
* ID-01 is recomputed; body digests, chunk geometry and retract shape are
  re-verified before any write (SRC-03);
* same key + same canonical payload returns the stored receipt as ``duplicate``;
  same key + different payload returns ``EVENT_CONFLICT`` and never overwrites
  (ID-02);
* an edit/retract/redact whose parent revision is unknown is stored durably as
  ``pending_dependency`` and never dropped; it is promoted when the parent
  arrives;
* memory injections / maintenance inputs are recorded as ``excluded`` receipts
  and are not semantically ingested (SRC-02);
* withheld bodies are never stored (SEC-01) — receipt only;
* retract marks the target revision's parts retracted; redact purges older
  secret-bearing bodies transactionally (SEC-02 / SRC-04 data effects; claim
  projection effects arrive with T05);
* no model or network call happens here.
"""

from __future__ import annotations

import json
from typing import Any, Callable

from knowledge_weaver import contracts

from .db import Store, utc_now

EXCLUDED_ORIGINS = frozenset({"memory_injection", "maintenance"})

_FP_SOURCE_KEYS = (
    "instance_id",
    "host",
    "actor_id",
    "session_id",
    "native_event_id",
    "native_schema",
    "role",
    "origin_kind",
    "turn_id",
    "parent_event_id",
    "locator",
    "aliases",
)


class StoreRejection(Exception):
    """Whole-request rejection (HTTP 400/413); nothing was committed."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def receipt_id_for(event_id: str, revision: int, part_index: int) -> str:
    digest = contracts.sha256_hex(f"{event_id}|{revision}|{part_index}")
    return "rcpt:" + digest[:32]


def event_payload_fingerprint(event: dict) -> str:
    """Canonical fingerprint used for ID-02 duplicate-vs-conflict decisions."""
    source = event.get("source") or {}
    normalized = {
        "schema_version": event.get("schema_version"),
        "event_id": event.get("event_id"),
        "event_revision": event.get("event_revision"),
        "workspace_id": event.get("workspace_id"),
        "operation": event.get("operation"),
        "parent_revision": event.get("parent_revision"),
        "source": {key: source.get(key) for key in _FP_SOURCE_KEYS},
        "event_time": event.get("event_time"),
        "body": event.get("body"),
        "redaction": event.get("redaction"),
        "project_ids": event.get("project_ids"),
        "derived_from": event.get("derived_from"),
    }
    return contracts.sha256_hex(contracts.compact_json(normalized))


def _receipt(
    event_id: str,
    revision: int,
    part_index: int,
    *,
    disposition: str,
    durable_remote: bool,
    processing: str,
    searchability: str = "none",
    error_code: str | None = None,
    receipt_id: str | None = None,
) -> dict:
    return {
        "event_id": event_id,
        "event_revision": revision,
        "part_index": part_index,
        "disposition": disposition,
        "durable_remote": durable_remote,
        "processing": processing,
        "searchability": searchability,
        "error_code": error_code,
        "receipt_id": receipt_id or receipt_id_for(event_id, revision, part_index),
    }


def _row_receipt(row) -> dict:
    return _receipt(
        row["event_id"],
        row["revision"],
        row["part_index"],
        disposition=row["disposition"],
        durable_remote=bool(row["durable_remote"]),
        processing=row["processing"],
        searchability=row["searchability"],
        error_code=row["error_code"],
        receipt_id=row["receipt_id"],
    )


def _conflict_receipt(event: dict, existing_fingerprint: str) -> dict:
    body = event.get("body") or {}
    return _receipt(
        event["event_id"],
        int(event["event_revision"]),
        int(body.get("part_index") or 0),
        disposition="rejected",
        durable_remote=False,
        processing="not_applicable",
        error_code="EVENT_CONFLICT",
        receipt_id="rcpt:" + contracts.sha256_hex(
            f"{event.get('event_id')}|{event.get('event_revision')}|"
            f"{event_payload_fingerprint(event)}|{existing_fingerprint}"
        )[:32],
    )


def _ensure_event_row(conn, event: dict) -> tuple[Any | None, str | None]:
    """Insert the logical event envelope once; return (row, error)."""
    source = event["source"]
    row = conn.execute("SELECT * FROM events WHERE event_id = ?", (event["event_id"],)).fetchone()
    aliases = json.dumps(source.get("aliases") or [], ensure_ascii=False)
    if row is None:
        conn.execute(
            "INSERT INTO events (event_id, workspace_id, instance_id, host, actor_id, session_id, "
            "native_event_id, native_schema, role, origin_kind, turn_id, parent_event_id, locator, "
            "aliases_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                event["event_id"],
                event["workspace_id"],
                source["instance_id"],
                source["host"],
                source["actor_id"],
                source["session_id"],
                source["native_event_id"],
                source["native_schema"],
                source["role"],
                source["origin_kind"],
                source.get("turn_id"),
                source.get("parent_event_id"),
                source.get("locator") or "",
                aliases,
                utc_now(),
            ),
        )
        return None, None
    stable = (
        row["workspace_id"] == event["workspace_id"]
        and row["instance_id"] == source["instance_id"]
        and row["native_event_id"] == source["native_event_id"]
        and row["host"] == source["host"]
        and row["session_id"] == source["session_id"]
        and row["actor_id"] == source["actor_id"]
        and row["role"] == source["role"]
        and row["origin_kind"] == source["origin_kind"]
    )
    if not stable:
        return row, "EVENT_CONFLICT"
    return row, None


def _parent_known(conn, event_id: str, parent_revision: int | None) -> bool:
    if parent_revision is None:
        return True
    row = conn.execute(
        "SELECT 1 FROM event_revisions WHERE event_id = ? AND revision = ?",
        (event_id, int(parent_revision)),
    ).fetchone()
    return row is not None


def _stored_group_metadata(conn, event_id: str, revision: int) -> dict | None:
    row = conn.execute(
        "SELECT * FROM event_revisions WHERE event_id = ? AND revision = ?",
        (event_id, int(revision)),
    ).fetchone()
    if row is None:
        return None
    return {
        "operation": row["operation"],
        "parent_revision": row["parent_revision"],
        "event_time": (
            row["event_time_value"],
            row["event_time_precision"],
            row["event_time_timezone"],
        ),
        "project_ids": sorted(json.loads(row["project_ids_json"] or "[]")),
        "content_revision": row["content_revision"],
        "part_count": int(row["part_count"]),
        "derived_from": json.loads(row["derived_from_json"] or "[]"),
        "redaction_policy": row["redaction_policy"],
    }


def _incoming_group_metadata(event: dict) -> dict:
    event_time = event["event_time"]
    return {
        "operation": event["operation"],
        "parent_revision": event.get("parent_revision"),
        "event_time": (
            event_time.get("value"),
            event_time["precision"],
            event_time["timezone"],
        ),
        "project_ids": sorted(event.get("project_ids") or []),
        "content_revision": event["body"]["content_revision"],
        "part_count": int(event["body"]["part_count"]),
        "derived_from": list(event.get("derived_from") or []),
        "redaction_policy": event["redaction"]["policy_version"],
    }


def _group_metadata_mismatch(conn, event: dict) -> str | None:
    """SRC-03 across requests: every part must agree with already-stored parts.

    A revision can be split across several HTTP requests, so the batch-level
    check in ``contracts.validate_group_consistency`` is not enough: the second
    request must match the persisted revision envelope (operation, parent,
    business time, project scope, native content revision, total part count and
    derivation lineage). A mismatch is an EVENT_CONFLICT receipt, never a silent
    merge into the first request's group.
    """
    stored = _stored_group_metadata(conn, event["event_id"], int(event["event_revision"]))
    if stored is None:
        return "revision envelope missing"
    incoming = _incoming_group_metadata(event)
    for field, value in incoming.items():
        if stored[field] != value:
            return field
    return None


def _part_indexes_complete(conn, event_id: str, revision: int, part_count: int) -> bool:
    indexes = {
        int(row["part_index"])
        for row in conn.execute(
            "SELECT part_index FROM event_parts WHERE event_id = ? AND revision = ?",
            (event_id, int(revision)),
        )
    }
    return indexes == set(range(int(part_count)))


def _enqueue_extraction_for_revision(store: Store, conn, event_id: str, revision: int) -> None:
    """Queue semantic extraction in the caller's transaction (JOB-01 / F07).

    The job row is committed with the group-state transition, so a crash can
    never leave a complete revision without its extraction job. Excluded
    maintenance/injection origins are never semantically ingested.
    """
    event_row = conn.execute(
        "SELECT origin_kind FROM events WHERE event_id = ?", (event_id,)
    ).fetchone()
    if event_row is None or event_row["origin_kind"] in EXCLUDED_ORIGINS:
        return
    row = conn.execute(
        "SELECT group_state, lifecycle FROM event_revisions WHERE event_id = ? AND revision = ?",
        (event_id, int(revision)),
    ).fetchone()
    if row is None or row["group_state"] != "complete" or row["lifecycle"] != "active":
        return
    parts = conn.execute(
        "SELECT COUNT(*) AS total, "
        "SUM(CASE WHEN availability = 'full' THEN 1 ELSE 0 END) AS full_parts "
        "FROM event_parts WHERE event_id = ? AND revision = ?",
        (event_id, int(revision)),
    ).fetchone()
    # A fully withheld group has no semantic input; extraction would be a no-op.
    if not parts["total"] or int(parts["full_parts"] or 0) != int(parts["total"]):
        return
    from knowledge_weaver.extraction.pipeline import enqueue_extraction_in_transaction

    enqueue_extraction_in_transaction(conn, event_id, int(revision))


def _apply_operation_effects(store: Store, conn, event: dict) -> None:
    """Transactional data effects of a completed revision (SEC-02 / SRC-04)."""
    event_id = event["event_id"]
    revision = int(event["event_revision"])
    operation = event["operation"]
    parent = event.get("parent_revision")
    if operation == "retract":
        if parent is None:
            return
        conn.execute(
            "UPDATE event_parts SET availability = 'retracted' "
            "WHERE event_id = ? AND revision = ? AND availability != 'retracted'",
            (event_id, int(parent)),
        )
        conn.execute(
            "UPDATE event_revisions SET lifecycle = 'retracted' WHERE event_id = ? AND revision = ?",
            (event_id, int(parent)),
        )
        from . import claims as claim_projection

        claim_projection.invalidate_retracted_evidence(conn, event_id, int(parent))
        store.bump_watermark(conn, "retract", f"{event_id}@{int(parent)}")
    elif operation == "redact":
        if parent is None:
            return
        # SEC-02: purge older secret-bearing bodies, keep non-secret lineage.
        conn.execute(
            "UPDATE event_parts SET body_text = NULL, availability = 'withheld' "
            "WHERE event_id = ? AND revision < ? AND availability = 'full' AND body_text IS NOT NULL",
            (event_id, revision),
        )
        from . import claims as claim_projection

        claim_projection.invalidate_withheld_evidence(conn, event_id, revision)
        store.bump_watermark(conn, "redact", f"{event_id}@{revision}")


def _complete_if_ready(store: Store, conn, event: dict) -> bool:
    """Mark a revision complete and apply its effects when all parts arrived."""
    event_id = event["event_id"]
    revision = int(event["event_revision"])
    row = conn.execute(
        "SELECT * FROM event_revisions WHERE event_id = ? AND revision = ?",
        (event_id, revision),
    ).fetchone()
    if row is None:
        return False
    if row["group_state"] == "complete":
        # A redaction upgrade can make an already-"complete" withheld group
        # processable again; the caller re-queues explicitly in that case.
        return True
    arrived = conn.execute(
        "SELECT COUNT(*) AS c FROM event_parts WHERE event_id = ? AND revision = ?",
        (event_id, revision),
    ).fetchone()["c"]
    if arrived < row["part_count"]:
        return False
    if row["group_state"] == "pending_dependency" and not _parent_known(
        conn, event_id, row["parent_revision"]
    ):
        return False
    # SRC-03: the index set must be exactly 0..part_count-1, not just the right
    # number of rows (e.g. parts 0 and 2 of a 2-part group are not complete).
    if not _part_indexes_complete(conn, event_id, revision, int(row["part_count"])):
        return False
    conn.execute(
        "UPDATE event_revisions SET group_state = 'complete' WHERE event_id = ? AND revision = ?",
        (event_id, revision),
    )
    _apply_operation_effects(store, conn, event)
    _enqueue_extraction_for_revision(store, conn, event_id, revision)
    return True


def _promote_pending(store: Store, conn, event_id: str) -> None:
    """Promote stored pending-dependency revisions once their parent is known."""
    for _ in range(100):  # revisions are ordered; loop is bounded and idempotent
        rows = conn.execute(
            "SELECT * FROM event_revisions WHERE event_id = ? AND group_state = 'pending_dependency'",
            (event_id,),
        ).fetchall()
        promoted = False
        for row in rows:
            if not _parent_known(conn, event_id, row["parent_revision"]):
                continue
            event = _revision_to_event_stub(conn, row)
            if _complete_if_ready(store, conn, event):
                promoted = True
            else:
                conn.execute(
                    "UPDATE event_revisions SET group_state = 'incomplete' "
                    "WHERE event_id = ? AND revision = ?",
                    (event_id, row["revision"]),
                )
                promoted = True
        if not promoted:
            return


def _revision_to_event_stub(conn, row) -> dict:
    event_row = conn.execute("SELECT * FROM events WHERE event_id = ?", (row["event_id"],)).fetchone()
    return {
        "event_id": row["event_id"],
        "event_revision": row["revision"],
        "operation": row["operation"],
        "parent_revision": row["parent_revision"],
        "workspace_id": event_row["workspace_id"] if event_row else "",
        "source": {"instance_id": event_row["instance_id"] if event_row else ""},
    }


def _validate_redaction_upgrade(conn, event: dict) -> str | None:
    """SEC-02: redact keeps native content_revision and business time."""
    if event["operation"] != "redact":
        return None
    parent = event.get("parent_revision")
    if parent is None:
        return None
    row = conn.execute(
        "SELECT * FROM event_revisions WHERE event_id = ? AND revision = ?",
        (event["event_id"], int(parent)),
    ).fetchone()
    if row is None:
        return None
    body = event.get("body") or {}
    if row["content_revision"] != body.get("content_revision"):
        return "redact must keep the native content_revision of the redacted revision"
    event_time = event.get("event_time") or {}
    if (
        row["event_time_value"] != event_time.get("value")
        or row["event_time_precision"] != event_time.get("precision")
        or row["event_time_timezone"] != event_time.get("timezone")
    ):
        return "redact must keep the business time of the redacted revision"
    return None


def _try_redaction_upgrade(
    store: Store,
    conn,
    event: dict,
    *,
    existing,
    fingerprint: str,
    text: str,
    withheld_body: bool,
) -> dict | None:
    """F03: accept a sanitized re-submission of a REDACTION_REQUIRED part.

    KW's entry re-check refuses to store a body that still matches the redact:1
    policy and replies ``withheld`` + ``REDACTION_REQUIRED``. The adapter then
    purges its suspect payload, re-reads the native record under the updated
    policy and re-delivers the *same* event/revision/part with the sanitized
    body. Treating that as an EVENT_CONFLICT would permanently lose the event.

    A legitimate upgrade is narrow and auditable:

    * the stored part was withheld with a ``REDACTION_REQUIRED`` receipt;
    * the new payload is marked ``redaction.status=passed`` and contains no
      secret (KW re-checks it);
    * the group envelope (native content revision, business time, project
      scope, part count, parent/operation, derived lineage) is unchanged, so
      this is the same logical event, not a different payload smuggled under
      the same key;
    * the original receipt id is preserved; the transition is watermarked.
    """
    if withheld_body or not text:
        return None
    if not (existing["withheld"] and existing["body_text"] is None):
        return None
    if event["redaction"].get("status") not in ("passed", "redacted"):
        return None
    event_id = event["event_id"]
    revision = int(event["event_revision"])
    part_index = int(event["body"]["part_index"])
    receipt_row = conn.execute(
        "SELECT * FROM receipts WHERE event_id = ? AND revision = ? AND part_index = ?",
        (event_id, revision, part_index),
    ).fetchone()
    if (
        receipt_row is None
        or receipt_row["disposition"] != "withheld"
        or receipt_row["error_code"] != "REDACTION_REQUIRED"
    ):
        return None
    if _group_metadata_mismatch(conn, event) is not None:
        return None
    conn.execute(
        "UPDATE event_parts SET body_text = ?, body_sha256 = ?, body_bytes = ?, withheld = 0, "
        "availability = 'full', payload_sha256 = ?, received_at = ? "
        "WHERE event_id = ? AND revision = ? AND part_index = ?",
        (
            text,
            event["body"]["sha256"],
            len(text.encode("utf-8")),
            fingerprint,
            utc_now(),
            event_id,
            revision,
            part_index,
        ),
    )
    redaction = event["redaction"]
    conn.execute(
        "UPDATE event_revisions SET redaction_policy = ?, redaction_status = ?, "
        "redaction_replacements = ?, credential_refs_json = ? "
        "WHERE event_id = ? AND revision = ?",
        (
            redaction["policy_version"],
            redaction["status"],
            int(redaction.get("replacements") or 0),
            json.dumps(redaction.get("credential_refs") or [], ensure_ascii=False),
            event_id,
            revision,
        ),
    )
    receipt = _receipt(
        event_id,
        revision,
        part_index,
        disposition="accepted",
        durable_remote=True,
        processing="pending",
        receipt_id=receipt_row["receipt_id"],
    )
    conn.execute(
        "UPDATE receipts SET disposition = 'accepted', durable_remote = 1, processing = 'pending', "
        "error_code = NULL WHERE event_id = ? AND revision = ? AND part_index = ?",
        (event_id, revision, part_index),
    )
    store.bump_watermark(conn, "redaction.upgrade", f"{event_id}@{revision}#{part_index}")
    _complete_if_ready(store, conn, event)
    _enqueue_extraction_for_revision(store, conn, event_id, revision)
    _promote_pending(store, conn, event_id)
    return receipt


def _receive_part(
    store: Store,
    event: dict,
    *,
    secret_check: Callable[[str], bool] | None = None,
) -> dict:
    body = event["body"]
    source = event["source"]
    redaction = event["redaction"]
    event_id = event["event_id"]
    revision = int(event["event_revision"])
    part_index = int(body["part_index"])
    part_count = int(body["part_count"])
    fingerprint = event_payload_fingerprint(event)
    text = body.get("text") or ""
    withheld_body = redaction.get("status") == "withheld"
    # REDACTION_REQUIRED is reserved for KW's *own* check catching content the
    # adapter failed to sanitize. An adapter-declared withheld receipt is a
    # normal, expected outcome and must not send DMA into the recovery path.
    withheld_reason = None
    if not withheld_body and secret_check is not None and text and secret_check(text):
        withheld_body = True
        withheld_reason = "REDACTION_REQUIRED"
    excluded = source.get("origin_kind") in EXCLUDED_ORIGINS

    with store.transaction() as conn:
        existing = conn.execute(
            "SELECT * FROM event_parts WHERE event_id = ? AND revision = ? AND part_index = ?",
            (event_id, revision, part_index),
        ).fetchone()
        if existing is not None:
            if existing["payload_sha256"] == fingerprint:
                receipt_row = conn.execute(
                    "SELECT * FROM receipts WHERE event_id = ? AND revision = ? AND part_index = ?",
                    (event_id, revision, part_index),
                ).fetchone()
                if receipt_row is not None:
                    duplicate = _row_receipt(receipt_row)
                    duplicate["disposition"] = "duplicate"
                    return duplicate
                return _receipt(
                    event_id,
                    revision,
                    part_index,
                    disposition="duplicate",
                    durable_remote=True,
                    processing="pending",
                )
            # F06: the same part key re-sent with different metadata/content must
            # match the persisted group envelope, not just the payload digest.
            _, event_error = _ensure_event_row(conn, event)
            if event_error is not None:
                return _conflict_receipt(event, existing["payload_sha256"])
            if _group_metadata_mismatch(conn, event) is not None:
                return _conflict_receipt(event, existing["payload_sha256"])
            upgraded = _try_redaction_upgrade(
                store,
                conn,
                event,
                existing=existing,
                fingerprint=fingerprint,
                text=text,
                withheld_body=withheld_body,
            )
            if upgraded is not None:
                return upgraded
            return _conflict_receipt(event, existing["payload_sha256"])

        _, event_error = _ensure_event_row(conn, event)
        if event_error is not None:
            # Envelope identity/role drift is a conflict, never a silent overwrite.
            return _conflict_receipt(event, event_error)

        if not _parent_known(conn, event_id, event.get("parent_revision")):
            # Retain the input durably; the semantic result stays pending dependency.
            pass
        upgrade_error = _validate_redaction_upgrade(conn, event)
        if upgrade_error is not None:
            return _receipt(
                event_id,
                revision,
                part_index,
                disposition="rejected",
                durable_remote=False,
                processing="not_applicable",
                error_code="INVALID_REQUEST",
            )

        revision_row = conn.execute(
            "SELECT * FROM event_revisions WHERE event_id = ? AND revision = ?",
            (event_id, revision),
        ).fetchone()
        if revision_row is not None:
            # F06: a later part of an already-started revision may arrive in a
            # different request; its group metadata must still match the stored
            # envelope. Otherwise inconsistent part groups could complete early
            # and be extracted under the first request's scope.
            if _group_metadata_mismatch(conn, event) is not None:
                return _conflict_receipt(event, revision_row["content_revision"])
        if revision_row is None:
            parent_pending = not _parent_known(conn, event_id, event.get("parent_revision"))
            event_time = event["event_time"]
            conn.execute(
                "INSERT INTO event_revisions (event_id, revision, operation, parent_revision, "
                "event_time_value, event_time_precision, event_time_timezone, project_ids_json, "
                "redaction_policy, redaction_status, redaction_replacements, credential_refs_json, "
                "content_revision, part_count, derived_from_json, group_state, lifecycle, sequence, received_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    event_id,
                    revision,
                    event["operation"],
                    event.get("parent_revision"),
                    event_time.get("value"),
                    event_time["precision"],
                    event_time["timezone"],
                    json.dumps(event.get("project_ids") or [], ensure_ascii=False),
                    redaction["policy_version"],
                    redaction["status"],
                    redaction["replacements"],
                    json.dumps(redaction.get("credential_refs") or [], ensure_ascii=False),
                    body["content_revision"],
                    part_count,
                    json.dumps(event.get("derived_from") or [], ensure_ascii=False),
                    "pending_dependency" if parent_pending else "incomplete",
                    "excluded" if excluded else "active",
                    store.bump_watermark(conn, "event.revision", f"{event_id}@{revision}"),
                    utc_now(),
                ),
            )

        conn.execute(
            "INSERT INTO event_parts (event_id, revision, part_index, body_text, body_sha256, "
            "body_bytes, format, withheld, availability, payload_sha256, received_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                event_id,
                revision,
                part_index,
                None if withheld_body else text,
                body["sha256"],
                len(text.encode("utf-8")),
                body.get("format") or "text/plain",
                1 if withheld_body else 0,
                "withheld" if withheld_body else "full",
                fingerprint,
                utc_now(),
            ),
        )
        store.bump_watermark(conn, "event.part", f"{event_id}@{revision}#{part_index}")

        if withheld_body:
            disposition, processing, error_code = "withheld", "not_applicable", withheld_reason
        elif excluded:
            disposition, processing, error_code = "excluded", "not_applicable", None
        else:
            parent_pending = not _parent_known(conn, event_id, event.get("parent_revision"))
            disposition = "accepted"
            processing = "pending"
            error_code = "PARENT_PENDING" if parent_pending else None

        receipt = _receipt(
            event_id,
            revision,
            part_index,
            disposition=disposition,
            durable_remote=True,
            processing=processing,
            error_code=error_code,
        )
        conn.execute(
            "INSERT INTO receipts (event_id, revision, part_index, disposition, durable_remote, "
            "processing, searchability, error_code, receipt_id, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                receipt["event_id"],
                receipt["event_revision"],
                receipt["part_index"],
                receipt["disposition"],
                1,
                receipt["processing"],
                receipt["searchability"],
                receipt["error_code"],
                receipt["receipt_id"],
                utc_now(),
            ),
        )

        if disposition in ("accepted", "excluded", "withheld"):
            _complete_if_ready(store, conn, event)
            _promote_pending(store, conn, event_id)
        return receipt


def _schedule_extraction(store, events: list[dict], receipts: list[dict]) -> None:
    """Enqueue extraction for newly complete, active, non-excluded revisions.

    Receipts are already durable at this point; scheduling failure must not turn
    a durable receipt into an error, and the job table (plus T17 reconcile) is
    the recovery point for pending semantic work.
    """
    from knowledge_weaver.extraction.pipeline import enqueue_extraction

    seen: set[tuple[str, int]] = set()
    for event, receipt in zip(events, receipts):
        if receipt.get("disposition") != "accepted":
            continue
        if event.get("source", {}).get("origin_kind") in EXCLUDED_ORIGINS:
            continue
        key = (event["event_id"], int(event["event_revision"]))
        if key in seen:
            continue
        seen.add(key)
        try:
            with store.read() as conn:
                row = conn.execute(
                    "SELECT group_state, lifecycle FROM event_revisions "
                    "WHERE event_id = ? AND revision = ?",
                    key,
                ).fetchone()
            if row is None or row["group_state"] != "complete" or row["lifecycle"] != "active":
                continue
            enqueue_extraction(store, event["event_id"], int(event["event_revision"]))
        except Exception:  # noqa: BLE001 - durable receipt already committed
            continue


def receive_batch(
    store: Store,
    batch: dict,
    *,
    secret_check: Callable[[str], bool] | None = None,
) -> dict:
    """Receive one ``/v1/events:batch`` payload; return a BatchResult dict.

    Authorization (client binding + scope) is enforced by the service layer
    before this call. Whole-request problems raise :class:`StoreRejection`.
    """
    if not isinstance(batch, dict):
        raise StoreRejection("INVALID_REQUEST", "request body must be a JSON object")
    if batch.get("schema_version") not in contracts.SUPPORTED_SCHEMA_VERSIONS:
        raise StoreRejection("UNSUPPORTED_SCHEMA", "unsupported batch schema_version")
    events = batch.get("events")
    if not isinstance(events, list):
        raise StoreRejection("INVALID_REQUEST", "events must be an array")
    for index, event in enumerate(events):
        if not isinstance(event, dict):
            raise StoreRejection("INVALID_REQUEST", f"events[{index}] must be an object")
        if event.get("schema_version") not in contracts.SUPPORTED_SCHEMA_VERSIONS:
            raise StoreRejection(
                "UNSUPPORTED_SCHEMA", f"events[{index}].schema_version is not supported"
            )
    structure_issues = contracts.validate_batch_structure(batch)
    if structure_issues:
        first = structure_issues[0]
        raise StoreRejection("INVALID_REQUEST", f"invalid batch at {first.path}: {first.message}")
    for index, event in enumerate(events):
        issues = contracts.validate_event(event)
        if issues:
            first = issues[0]
            raise StoreRejection(
                "INVALID_REQUEST", f"events[{index}] rejected at {first.path}: {first.message}"
            )
    receipts = [
        _receive_part(store, event, secret_check=secret_check) for event in events
    ]
    # SEC-02: a redaction upgrade purged derived payloads; compact the file so the
    # old bytes do not linger in free pages/WAL (best-effort maintenance after the
    # durable receipt has already committed).
    if any(
        event.get("operation") == "redact" and receipt.get("disposition") == "accepted"
        for event, receipt in zip(events, receipts)
    ):
        try:
            store.purge_deleted_bytes()
        except Exception:  # noqa: BLE001 - receipt is already durable
            pass
    _schedule_extraction(store, events, receipts)
    meta = store.meta()
    return {
        "schema_version": contracts.EVENT_SCHEMA_VERSION,
        "backend_instance_id": meta["backend_instance_id"],
        "restore_epoch": meta["restore_epoch"],
        "watermark": store.watermark(),
        "receipts": receipts,
    }
