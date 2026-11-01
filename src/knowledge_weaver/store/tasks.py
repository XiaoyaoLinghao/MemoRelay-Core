"""Task views: reported completion vs verified evidence vs failure (AC-09/36).

A task is a claim family with predicate ``task.status`` under one subject. The
view never collapses the sources:

* an assistant "done" report alone yields ``reported_done`` (never verified);
* a tool observation status ``failed`` keeps the task ``failed`` even when some
  command succeeded (exit 0 only proves the command, not the acceptance);
* ``verified_done`` requires every declared acceptance condition to be listed
  as satisfied AND the backing evidence to be available (full);
* a value the proposal layer marked ``backend_downgraded`` can never verify: the
  view treats it as a report even if its status string claims otherwise;
* retraction/redaction/expiry of evidence recomputes the view through the normal
  claims projection; a revoked verification downgrades the task instead of
  leaving a stale "done".
"""

from __future__ import annotations

import json

from knowledge_weaver.store.db import Store, utc_now

from .claims import AUTHORITY_RANK

TASK_PREDICATE = "task.status"
TASK_STATES = ("open", "reported_done", "verified_done", "failed", "unknown")
#: Task statuses that assert verification; only backend-verified evidence may
#: grant them (LLM-01: the model cannot assign a verified level).
VERIFIED_LEVEL_STATUSES = frozenset({"verified", "verified_done", "passed"})
SELF_REPORT_ASSERTIONS = frozenset({"completion_report", "inference"})


def _value_dict(row) -> dict:
    raw = row["value_json"]
    try:
        value = json.loads(raw)
    except (TypeError, ValueError):
        value = raw
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        return {"status": value}
    return {}


def _current_selected(conn, workspace_id: str, project_id: str | None, subject_id: str) -> list[dict]:
    clauses = ["c.workspace_id = ?", "c.subject_id = ?", "c.predicate = ?"]
    params = [workspace_id, subject_id, TASK_PREDICATE]
    if project_id is None:
        clauses.append("c.project_id IS NULL")
    else:
        clauses.append("c.project_id = ?")
        params.append(project_id)
    sql = (
        "SELECT r.*, c.facet AS claim_facet, c.project_id AS claim_project, "
        "cc.state AS claim_state, cc.selected_revision_ids_json "
        "FROM claims c JOIN claim_revisions r ON r.claim_id = c.claim_id "
        "LEFT JOIN current_claims cc ON cc.claim_id = c.claim_id "
        f"WHERE {' AND '.join(clauses)}"
    )
    rows = conn.execute(sql, params).fetchall()
    selected_ids: set[str] = set()
    for row in rows:
        try:
            selected = json.loads(row["selected_revision_ids_json"] or "[]")
        except (TypeError, ValueError):
            selected = []
        # Only current/conflict/uncertain selected revisions count; expired and
        # revoked evidence must not keep a task verified.
        if selected:
            selected_ids.update(selected)
        elif row["status"] in ("current", "conflict", "uncertain"):
            selected_ids.add(row["revision_id"])
    result = []
    for row in rows:
        if row["revision_id"] not in selected_ids:
            continue
        if row["status"] in ("expired", "revoked", "superseded"):
            continue
        evidence = conn.execute(
            "SELECT availability FROM evidence_links WHERE revision_id = ?",
            (row["revision_id"],),
        ).fetchall()
        availabilities = {link["availability"] for link in evidence}
        if not availabilities or availabilities != {"full"}:
            continue
        result.append(
            {
                "revision_id": row["revision_id"],
                "facet": row["claim_facet"],
                "authority": row["authority"],
                "assertion_kind": row["assertion_kind"],
                "state": row["claim_state"],
                "value": _value_dict(row),
            }
        )
    return result


def derive_status(selected: list[dict]) -> dict:
    reported = None
    verified = None
    failed = None
    conditions: list[str] = []
    satisfied: list[str] = []
    for item in selected:
        value = item["value"]
        status = str(value.get("status") or "").lower()
        item_conditions = value.get("conditions") or value.get("acceptance_conditions") or []
        item_satisfied = value.get("satisfied") or value.get("passed_conditions") or []
        if not isinstance(item_conditions, list):
            item_conditions = [str(item_conditions)]
        if not isinstance(item_satisfied, list):
            item_satisfied = [str(item_satisfied)]
        if status in ("failed", "error", "rejected"):
            failed = failed or item["revision_id"]
            continue
        # Defence in depth (F04): the proposal layer marks a downgraded
        # completion with ``backend_downgraded``. Such a value can never grant a
        # verified level even if a stale/forged status string still says so.
        if status in VERIFIED_LEVEL_STATUSES and value.get("backend_downgraded"):
            reported = reported or item["revision_id"]
            continue
        if status in VERIFIED_LEVEL_STATUSES and item["facet"] == "observed":
            # Backend gate: a verified level requires independently verified
            # evidence (native tool/user), never a self-report or completion
            # report, regardless of what the proposal/claim value says.
            independently_verified = (
                AUTHORITY_RANK.get(item["authority"], 0) >= AUTHORITY_RANK["tool_verified"]
                and item["assertion_kind"] not in SELF_REPORT_ASSERTIONS
            )
            if not independently_verified:
                reported = reported or item["revision_id"]
                continue
            if item["revision_id"] not in [item for item in (verified,)]:
                verified = verified or item["revision_id"]
            conditions.extend(str(condition) for condition in item_conditions)
            satisfied.extend(str(condition) for condition in item_satisfied)
            continue
        if status in ("reported_done", "done", "complete", "completed"):
            reported = reported or item["revision_id"]
    unique_conditions = []
    for condition in conditions:
        if condition not in unique_conditions:
            unique_conditions.append(condition)
    unique_satisfied = []
    for condition in satisfied:
        if condition not in unique_satisfied:
            unique_satisfied.append(condition)
    if failed is not None:
        status = "failed"
    elif (
        verified is not None
        and unique_conditions
        and set(unique_conditions) <= set(unique_satisfied)
    ):
        status = "verified_done"
    elif reported is not None:
        status = "reported_done"
    elif verified is not None:
        status = "open"
    else:
        status = "unknown"
    return {
        "status": status,
        "reported_revision_id": reported,
        "verified_revision_id": verified if status == "verified_done" else None,
        "failed_revision_id": failed,
        "conditions": unique_conditions,
        "satisfied": unique_satisfied,
    }


def recompute_task(
    conn,
    *,
    workspace_id: str,
    project_id: str | None,
    subject_id: str,
) -> dict:
    selected = _current_selected(conn, workspace_id, project_id, subject_id)
    if not selected:
        conn.execute(
            "DELETE FROM task_views WHERE task_id = ? AND workspace_id = ? "
            "AND (project_id IS ? OR project_id = ?)",
            (subject_id, workspace_id, project_id, project_id),
        )
        return {"task_id": subject_id, "status": "unknown", "projection_generation": None}
    derived = derive_status(selected)
    now = utc_now()
    existing = conn.execute(
        "SELECT * FROM task_views WHERE task_id = ?", (subject_id,)
    ).fetchone()
    generation = int(existing["projection_generation"]) if existing is not None else 0
    changed = existing is None or existing["status"] != derived["status"] or (
        existing["reported_revision_id"] != derived["reported_revision_id"]
        or existing["verified_revision_id"] != derived["verified_revision_id"]
        or existing["failed_revision_id"] != derived["failed_revision_id"]
    )
    if changed:
        generation += 1
    conn.execute(
        "INSERT INTO task_views (task_id, workspace_id, project_id, status, reported_revision_id, "
        "verified_revision_id, failed_revision_id, verification_conditions_json, "
        "satisfied_conditions_json, updated_at, projection_generation) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
        "ON CONFLICT (task_id) DO UPDATE SET workspace_id=excluded.workspace_id, "
        "project_id=excluded.project_id, status=excluded.status, "
        "reported_revision_id=excluded.reported_revision_id, verified_revision_id=excluded.verified_revision_id, "
        "failed_revision_id=excluded.failed_revision_id, "
        "verification_conditions_json=excluded.verification_conditions_json, "
        "satisfied_conditions_json=excluded.satisfied_conditions_json, "
        "updated_at=excluded.updated_at, projection_generation=excluded.projection_generation",
        (
            subject_id,
            workspace_id,
            project_id,
            derived["status"],
            derived["reported_revision_id"],
            derived["verified_revision_id"],
            derived["failed_revision_id"],
            json.dumps(derived["conditions"], ensure_ascii=False),
            json.dumps(derived["satisfied"], ensure_ascii=False),
            now,
            generation,
        ),
    )
    return {"task_id": subject_id, "status": derived["status"], "projection_generation": generation}


def list_task_views(
    store: Store,
    *,
    workspace_id: str,
    project_ids: list[str] | None = None,
) -> list[dict]:
    with store.read() as conn:
        rows = conn.execute(
            "SELECT * FROM task_views WHERE workspace_id = ? ORDER BY project_id, task_id",
            (workspace_id,),
        ).fetchall()
    views = []
    for row in rows:
        if project_ids is not None and row["project_id"] not in project_ids:
            continue
        views.append(
            {
                "task_id": row["task_id"],
                "workspace_id": row["workspace_id"],
                "project_id": row["project_id"],
                "status": row["status"],
                "reported_revision_id": row["reported_revision_id"],
                "verified_revision_id": row["verified_revision_id"],
                "failed_revision_id": row["failed_revision_id"],
                "verification_conditions": json.loads(row["verification_conditions_json"] or "[]"),
                "satisfied_conditions": json.loads(row["satisfied_conditions_json"] or "[]"),
                "projection_generation": row["projection_generation"],
                "updated_at": row["updated_at"],
            }
        )
    return views
