"""Read sanitized source evidence by opaque event id + explicit revision/part.

This is NOT a file-system read API: unknown ids are 404, expired/withheld/
retracted bodies are never returned, and the caller must name the revision
instead of guessing "latest" (T02/T05).
"""

from __future__ import annotations

import json

from .db import Store

_HIDDEN_AVAILABILITIES = frozenset({"withheld", "expired", "retracted", "unknown"})


def _source_envelope(event) -> dict:
    source = {
        "instance_id": event["instance_id"],
        "host": event["host"],
        "actor_id": event["actor_id"],
        "session_id": event["session_id"],
        "native_event_id": event["native_event_id"],
        "native_schema": event["native_schema"],
        "role": event["role"],
        "origin_kind": event["origin_kind"],
        "locator": event["locator"],
    }
    if event["turn_id"] is not None:
        source["turn_id"] = event["turn_id"]
    if event["parent_event_id"] is not None:
        source["parent_event_id"] = event["parent_event_id"]
    aliases = json.loads(event["aliases_json"])
    if aliases:
        source["aliases"] = aliases
    return source


def fetch_source(
    store: Store,
    event_id: str,
    revision: int,
    part_index: int = 0,
) -> dict | None:
    with store.read() as conn:
        event = conn.execute("SELECT * FROM events WHERE event_id = ?", (event_id,)).fetchone()
        if event is None:
            return None
        revision_row = conn.execute(
            "SELECT * FROM event_revisions WHERE event_id = ? AND revision = ?",
            (event_id, int(revision)),
        ).fetchone()
        if revision_row is None:
            return None
        part = conn.execute(
            "SELECT * FROM event_parts WHERE event_id = ? AND revision = ? AND part_index = ?",
            (event_id, int(revision), int(part_index)),
        ).fetchone()
        if part is None:
            return None
    availability = part["availability"]
    if revision_row["lifecycle"] == "retracted":
        availability = "retracted"
    text = part["body_text"]
    if availability in _HIDDEN_AVAILABILITIES:
        text = None
    return {
        "event_id": event["event_id"],
        "event_revision": revision_row["revision"],
        "availability": availability,
        "part_index": part["part_index"],
        "part_count": revision_row["part_count"],
        "source": _source_envelope(event),
        "event_time": {
            "value": revision_row["event_time_value"],
            "precision": revision_row["event_time_precision"],
            "timezone": revision_row["event_time_timezone"],
        },
        "text": text,
        "redaction": {
            "policy_version": revision_row["redaction_policy"],
            "status": revision_row["redaction_status"],
            "replacements": revision_row["redaction_replacements"],
            "credential_refs": json.loads(revision_row["credential_refs_json"]),
        },
    }
