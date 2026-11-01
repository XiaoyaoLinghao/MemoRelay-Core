"""Reversible entity aliases / manual merge (AC-47).

Merging two subjects is an **alias**, not a rewrite of historical claims:

* claims keep their original subject ids, evidence and revisions;
* active aliases map a source subject to one canonical subject within the same
  workspace+project (cross-project merges are refused instead of guessed);
* queries resolve aliases at read time, so all properties of both subjects are
  visible under the canonical subject;
* undo deactivates the alias and queries immediately revert; historical claims
  were never modified;
* merge cycles are refused (following active aliases must terminate).
"""

from __future__ import annotations

import hashlib

from knowledge_weaver import contracts
from knowledge_weaver.store.db import Store, utc_now

MAX_ALIAS_DEPTH = 16


class EntityError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def _token(value, field: str) -> str:
    if not isinstance(value, str) or not contracts.TOKEN_RE.match(value) or len(value) > 180:
        raise EntityError("INVALID_REQUEST", f"{field} must be a short token")
    return value


def _entity_id(workspace_id: str, project_id: str | None, subject_id: str) -> str:
    digest = hashlib.sha256(
        contracts.compact_json([workspace_id, project_id, subject_id]).encode("utf-8")
    ).hexdigest()
    return "entity:" + digest[:32]


def _subject_exists(conn, workspace_id: str, project_id: str, subject_id: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM claims WHERE workspace_id = ? AND project_id = ? AND subject_id = ? LIMIT 1",
        (workspace_id, project_id, subject_id),
    ).fetchone()
    return row is not None


def resolve_canonical_subject(
    conn,
    *,
    workspace_id: str,
    project_id: str | None,
    subject_id: str,
) -> str:
    current = subject_id
    for _ in range(MAX_ALIAS_DEPTH):
        row = conn.execute(
            "SELECT canonical_subject_id FROM entity_aliases "
            "WHERE workspace_id = ? AND project_id IS ? AND source_subject_id = ? AND state = 'active' "
            "ORDER BY created_at DESC LIMIT 1",
            (workspace_id, project_id, current),
        ).fetchone()
        if row is None:
            return current
        current = row["canonical_subject_id"]
    # A cycle would mean a corrupt alias table; fail closed instead of looping.
    raise EntityError("INVALID_REQUEST", "entity alias cycle detected")


def merge_entities(
    store: Store,
    *,
    workspace_id: str,
    project_id: str,
    source_subject_id: str,
    canonical_subject_id: str,
    reason: str = "manual merge",
) -> dict:
    workspace_id = _token(workspace_id, "workspace_id")
    project_id = _token(project_id, "project_id")
    source_subject_id = _token(source_subject_id, "source_subject_id")
    canonical_subject_id = _token(canonical_subject_id, "canonical_subject_id")
    if source_subject_id == canonical_subject_id:
        raise EntityError("INVALID_REQUEST", "cannot merge a subject into itself")
    if not isinstance(reason, str) or not reason or len(reason) > 500:
        raise EntityError("INVALID_REQUEST", "merge reason must be a short non-empty string")
    now = utc_now()
    with store.transaction() as conn:
        if not _subject_exists(conn, workspace_id, project_id, source_subject_id):
            raise EntityError("SOURCE_NOT_FOUND", "source subject has no claims in this project")
        if not _subject_exists(conn, workspace_id, project_id, canonical_subject_id):
            raise EntityError("SOURCE_NOT_FOUND", "canonical subject has no claims in this project")
        try:
            target_canonical = resolve_canonical_subject(
                conn, workspace_id=workspace_id, project_id=project_id,
                subject_id=canonical_subject_id,
            )
        except EntityError as exc:
            raise EntityError("INVALID_REQUEST", "existing aliases form a cycle") from exc
        if target_canonical == source_subject_id:
            raise EntityError("INVALID_REQUEST", "merge would create an alias cycle")
        # Manual aliases must not silently change a known entity category or
        # leave two competing active destinations for the same source.
        types=[]
        for subject in (source_subject_id,canonical_subject_id,target_canonical):
            row=conn.execute("SELECT entity_type FROM entities WHERE entity_id=? AND workspace_id=? AND project_id=?",(_entity_id(workspace_id,project_id,subject),workspace_id,project_id)).fetchone()
            types.append(row["entity_type"] if row is not None else "subject")
        if len(set(types))>1:
            raise EntityError("INVALID_REQUEST", "entity types are incompatible or unresolved; register matching types before merging")
        other=conn.execute("SELECT 1 FROM entity_aliases WHERE workspace_id=? AND project_id=? AND source_subject_id=? AND canonical_subject_id<>? AND state='active' LIMIT 1",(workspace_id,project_id,source_subject_id,canonical_subject_id)).fetchone()
        if other:
            raise EntityError("INVALID_REQUEST", "source already has a different active alias; undo that merge first")
        existing = conn.execute(
            "SELECT * FROM entity_aliases WHERE workspace_id = ? AND project_id = ? "
            "AND source_subject_id = ? AND canonical_subject_id = ?",
            (workspace_id, project_id, source_subject_id, canonical_subject_id),
        ).fetchone()
        alias_id = (
            existing["alias_id"]
            if existing is not None
            else "alias:" + hashlib.sha256(
                contracts.compact_json([workspace_id, project_id, source_subject_id, canonical_subject_id, now])
                .encode("utf-8")
            ).hexdigest()[:32]
        )
        if existing is not None:
            conn.execute(
                "UPDATE entity_aliases SET state = 'active', removed_at = NULL WHERE alias_id = ?",
                (alias_id,),
            )
        else:
            conn.execute(
                "INSERT INTO entity_aliases (alias_id, workspace_id, project_id, source_subject_id, "
                "canonical_subject_id, reason, state, created_at) VALUES (?, ?, ?, ?, ?, ?, 'active', ?)",
                (alias_id, workspace_id, project_id, source_subject_id, canonical_subject_id, reason, now),
            )
        conn.execute(
            "INSERT OR IGNORE INTO entities (entity_id, workspace_id, project_id, entity_type, canonical_name, created_at) "
            "VALUES (?, ?, ?, 'subject', ?, ?)",
            (_entity_id(workspace_id, project_id, canonical_subject_id), workspace_id, project_id, canonical_subject_id, now),
        )
    return {
        "alias_id": alias_id,
        "workspace_id": workspace_id,
        "project_id": project_id,
        "source_subject_id": source_subject_id,
        "canonical_subject_id": canonical_subject_id,
        "state": "active",
    }


def undo_merge(
    store: Store,
    *,
    workspace_id: str,
    project_id: str,
    source_subject_id: str,
    canonical_subject_id: str,
) -> dict:
    workspace_id = _token(workspace_id, "workspace_id")
    project_id = _token(project_id, "project_id")
    source_subject_id = _token(source_subject_id, "source_subject_id")
    canonical_subject_id = _token(canonical_subject_id, "canonical_subject_id")
    now = utc_now()
    with store.transaction() as conn:
        cursor = conn.execute(
            "UPDATE entity_aliases SET state = 'removed', removed_at = ? "
            "WHERE workspace_id = ? AND project_id = ? AND source_subject_id = ? "
            "AND canonical_subject_id = ? AND state = 'active'",
            (now, workspace_id, project_id, source_subject_id, canonical_subject_id),
        )
        if cursor.rowcount == 0:
            raise EntityError("SOURCE_NOT_FOUND", "active alias not found")
    return {
        "workspace_id": workspace_id,
        "project_id": project_id,
        "source_subject_id": source_subject_id,
        "canonical_subject_id": canonical_subject_id,
        "state": "removed",
        "removed_at": now,
    }


def list_aliases(store: Store, *, workspace_id: str, project_id: str | None = None, active_only: bool = True) -> list[dict]:
    clauses = ["workspace_id = ?"]
    params: list = [workspace_id]
    if project_id is not None:
        clauses.append("project_id = ?")
        params.append(project_id)
    if active_only:
        clauses.append("state = 'active'")
    with store.read() as conn:
        rows = conn.execute(
            f"SELECT * FROM entity_aliases WHERE {' AND '.join(clauses)} ORDER BY created_at",
            params,
        ).fetchall()
    return [dict(row) for row in rows]
