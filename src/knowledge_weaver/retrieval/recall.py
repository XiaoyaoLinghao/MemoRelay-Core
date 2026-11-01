"""Hybrid recall: scope filter -> keyword (FTS5) + optional vectors -> RRF -> budget.

Contract behaviour implemented here (AC-26/27/40/48):

* authorization scope is filtered **before** any ranking; unresolved scope returns
  ``degraded`` + explainable scope candidates instead of guessing across projects;
* keyword and semantic rankings are fused with reciprocal-rank fusion; importance
  never overrides relevance (there is no importance input);
* the response separates ``empty`` from ``unavailable`` and reports explicit
  ``incomplete_reasons`` (embedding unavailable, deadline, processing pending,
  scope unresolved, budget);
* pagination cursors are bound to query/scope/watermark; a changed world yields a
  stale-cursor error rather than silently skipping/duplicating rows;
* budgets and limits are enforced with an explicit token estimate;
* no answer-generation model is called on this path.
"""

from __future__ import annotations

import hashlib
import time

from knowledge_weaver import contracts
from knowledge_weaver.store.db import Store

from .keyword import keyword_ranking
from .tokenizer import estimate_tokens

RRF_K = 60
EMBED_INLINE_LIMIT = 200
DEFAULT_INCOMPLETE = []
STATE_ORDER = (
    "current",
    "conflict",
    "uncertain",
    "superseded",
    "revoked",
    "expired",
)


class RecallValidationError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.code = "INVALID_REQUEST"
        self.message = message


class RecallScopeDenied(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.code = "SCOPE_DENIED"
        self.message = message


class RecallCursorError(Exception):
    def __init__(self, message: str = "cursor no longer matches query/scope/watermark") -> None:
        super().__init__(message)
        self.code = "STALE_CURSOR"
        self.message = message


def _cursor_prefix(query: str, project_ids: list[str], intent: str, watermark: int, epoch: str) -> str:
    digest = hashlib.sha256(
        contracts.compact_json(
            [query, sorted(project_ids), intent, int(watermark), epoch]
        ).encode("utf-8")
    ).hexdigest()
    return digest[:24]


def _encode_cursor(prefix: str, offset: int) -> str:
    return f"cur:{prefix}:{offset}"


def _decode_cursor(cursor: str, prefix: str) -> int:
    if not isinstance(cursor, str) or not cursor.startswith("cur:"):
        raise RecallCursorError("malformed cursor")
    parts = cursor.split(":")
    if len(parts) != 3:
        raise RecallCursorError("malformed cursor")
    if parts[1] != prefix:
        raise RecallCursorError()
    try:
        offset = int(parts[2])
    except ValueError:
        raise RecallCursorError("malformed cursor") from None
    if offset < 0:
        raise RecallCursorError("malformed cursor")
    return offset


def _scope_resolution(store: Store, request: dict, binding: dict):
    workspace_id = binding["workspace_id"]
    binding_projects = list(binding.get("project_ids") or [])
    requested = list(request.get("project_ids") or [])
    intent = request["intent"]
    if requested:
        if binding_projects and not set(requested) <= set(binding_projects):
            raise RecallScopeDenied("requested project scope is not authorized")
        return workspace_id, requested, ""
    if intent == "all_projects":
        return workspace_id, binding_projects, ""
    if binding_projects:
        return workspace_id, binding_projects, ""
    # No project scope is resolvable: explain candidates, never search all.
    return workspace_id, [], "scope_unresolved"


def _scope_candidates(store: Store, workspace_id: str) -> list[dict]:
    with store.read() as conn:
        rows = conn.execute(
            "SELECT DISTINCT c.project_id FROM claims c "
            "JOIN current_claims cc ON cc.claim_id = c.claim_id "
            "WHERE c.workspace_id = ? AND c.project_id IS NOT NULL "
            "AND cc.state IN ('current', 'conflict', 'uncertain') "
            "ORDER BY c.project_id LIMIT 10",
            (workspace_id,),
        ).fetchall()
    return [
        {
            "project_id": row["project_id"],
            "label": row["project_id"],
            "match_kind": "semantic_candidate",
        }
        for row in rows
    ]


def _candidate_revisions(store: Store, workspace_id: str, project_ids: list[str], intent: str) -> tuple[set[str], dict[str, dict]]:
    """Return candidate revision ids plus claim metadata for item construction."""
    clauses = ["c.workspace_id = ?"]
    params: list = [workspace_id]
    if project_ids:
        placeholders = ",".join("?" for _ in project_ids)
        clauses.append(f"c.project_id IN ({placeholders})")
        params.extend(project_ids)
    if intent == "tasks":
        clauses.append("c.predicate = 'task.status'")
    if intent == "history":
        clauses.append("1 = 1")
        sql = (
            "SELECT r.*, c.workspace_id, c.project_id, c.subject_id, c.predicate, c.facet, "
            "c.qualifiers_json, cc.state AS claim_state, cc.selected_revision_ids_json "
            "FROM claim_revisions r JOIN claims c ON c.claim_id = r.claim_id "
            "LEFT JOIN current_claims cc ON cc.claim_id = c.claim_id "
            f"WHERE {' AND '.join(clauses)}"
        )
        with store.read() as conn:
            rows = conn.execute(sql, params).fetchall()
        candidates = {row["revision_id"]: dict(row) for row in rows}
        return set(candidates), {}
    clauses.append("cc.state IN ('current', 'conflict', 'uncertain')")
    sql = (
        "SELECT cc.selected_revision_ids_json, cc.state AS claim_state, c.* "
        "FROM claims c JOIN current_claims cc ON cc.claim_id = c.claim_id "
        f"WHERE {' AND '.join(clauses)}"
    )
    with store.read() as conn:
        rows = conn.execute(sql, params).fetchall()
        candidate_ids: set[str] = set()
        claim_info: dict[str, dict] = {}
        for row in rows:
            import json

            selected = json.loads(row["selected_revision_ids_json"])
            if not selected:
                latest = conn.execute(
                    "SELECT revision_id FROM claim_revisions WHERE claim_id = ? "
                    "ORDER BY revision_number DESC LIMIT 1",
                    (row["claim_id"],),
                ).fetchone()
                if latest is None:
                    continue
                selected = [latest["revision_id"]]
            candidate_ids.update(selected)
            claim_info[row["claim_id"]] = {
                "claim_id": row["claim_id"],
                "claim_state": row["claim_state"],
                "selected": selected,
                "workspace_id": row["workspace_id"],
                "project_id": row["project_id"],
            }
    return candidate_ids, claim_info


def _load_items(
    store: Store, revision_ids: list[str], scores: dict[str, float], *, use_revision_status: bool = False
) -> list[dict]:
    if not revision_ids:
        return []
    placeholders = ",".join("?" for _ in revision_ids)
    import json

    with store.read() as conn:
        rows = conn.execute(
            "SELECT r.*, c.workspace_id, c.project_id, c.subject_id, c.predicate, c.facet, "
            "c.qualifiers_json, cc.state AS claim_state, cc.selected_revision_ids_json, "
            "cc.projection_generation "
            "FROM claim_revisions r JOIN claims c ON c.claim_id = r.claim_id "
            "LEFT JOIN current_claims cc ON cc.claim_id = c.claim_id "
            f"WHERE r.revision_id IN ({placeholders})",
            revision_ids,
        ).fetchall()
        links = conn.execute(
            "SELECT * FROM evidence_links WHERE revision_id IN (" + placeholders + ") "
            "ORDER BY event_id, event_revision, part_index, start_byte",
            revision_ids,
        ).fetchall()
        from knowledge_weaver.store.entities import resolve_canonical_subject

        canonical_subjects: dict[str, str] = {
            row["revision_id"]: resolve_canonical_subject(
                conn,
                workspace_id=row["workspace_id"],
                project_id=row["project_id"],
                subject_id=row["subject_id"],
            )
            for row in rows
        }
    links_by_revision: dict[str, list[dict]] = {}
    for link in links:
        links_by_revision.setdefault(link["revision_id"], []).append(
            {
                "event_id": link["event_id"],
                "event_revision": link["event_revision"],
                "part_index": link["part_index"],
                "start_byte": link["start_byte"],
                "end_byte": link["end_byte"],
                "body_sha256": link["body_sha256"],
                "availability": link["availability"],
            }
        )
    items = []
    for row in rows:
        selected = json.loads(row["selected_revision_ids_json"] or "[]")
        if use_revision_status:
            state = row["status"] if row["status"] in STATE_ORDER else "uncertain"
        else:
            claim_state = row["claim_state"] or (
                "current" if row["status"] == "current" else row["status"]
            )
            state = claim_state if claim_state in STATE_ORDER else row["status"]
            if state not in STATE_ORDER:
                state = "uncertain"
        sources = links_by_revision.get(row["revision_id"], [])
        items.append(
            {
                "claim_id": row["claim_id"],
                "revision_id": row["revision_id"],
                "project_id": row["project_id"],
                "subject_id": canonical_subjects[row["revision_id"]],
                "projection_generation": int(row["projection_generation"] or 0),
                "value": json.loads(row["value_json"]),
                "qualifiers": json.loads(row["qualifiers_json"]),
                "predicate": row["predicate"],
                "facet": row["facet"],
                "text": row["display_text"],
                "state": state,
                "evidence_grade": row["authority"],
                "event_time": {
                    "value": row["event_time_value"],
                    "precision": row["event_time_precision"],
                    "timezone": row["event_time_timezone"],
                },
                "sources": sources,
                "conflict_revision_ids": [
                    revision_id for revision_id in selected if revision_id != row["revision_id"]
                ]
                if state == "conflict"
                else [],
                "source_actors": json.loads(row["source_actors_json"]),
                "relevance_score": float(scores.get(row["revision_id"], 0.0)),
                "source_detail_expired": bool(sources)
                and any(source["availability"] != "full" for source in sources),
            }
        )
    items.sort(key=lambda item: (-item["relevance_score"], item["revision_id"]))
    return items


def execute_recall(
    store: Store,
    request: dict,
    *,
    binding: dict,
    embedder=None,
) -> dict:
    started = time.monotonic()
    issues = contracts.validate_instance("RecallRequest", request)
    if issues:
        first = issues[0]
        raise RecallValidationError(f"invalid recall request at {first.path}: {first.message}")
    workspace_id, project_ids, unresolved_reason = _scope_resolution(store, request, binding)
    meta = store.meta()
    watermark = int(meta["watermark"])
    epoch = meta["restore_epoch"]
    base = {
        "schema_version": contracts.EVENT_SCHEMA_VERSION,
        "request_id": request["request_id"],
        "status": "ok",
        "scope_candidates": [],
        "watermark": watermark,
        "restore_epoch": epoch,
        "items": [],
        "next_cursor": None,
        "total_hits": 0,
        "incomplete_reasons": [],
        "used_tokens": 0,
        "truncated": False,
    }
    if unresolved_reason == "scope_unresolved":
        base["status"] = "degraded"
        base["incomplete_reasons"] = ["scope_unresolved"]
        base["scope_candidates"] = _scope_candidates(store, workspace_id)
        return base

    intent = request["intent"]
    cursor_prefix = _cursor_prefix(request["query"], project_ids, intent, watermark, epoch)
    offset = 0
    if request.get("cursor"):
        offset = _decode_cursor(request["cursor"], cursor_prefix)

    page_cap = min(int(request["limit"]), 50)
    candidate_ids, claim_info = _candidate_revisions(store, workspace_id, project_ids, intent)
    reasons: list[str] = []
    scores: dict[str, float] = {}
    if candidate_ids:
        keyword_ranked = keyword_ranking(store, candidate_revision_ids=candidate_ids, query=request["query"])
        for rank, revision_id in enumerate(keyword_ranked):
            scores[revision_id] = scores.get(revision_id, 0.0) + 1.0 / (RRF_K + rank + 1)
        if embedder is None:
            reasons.append("embedding_unavailable")
        else:
            # SEC-01 egress rule: the current user query is outbound data too.
            # Protect it with the shared redact:1 policy before any embedding or
            # model call; a withhold-kind match or a still-matching masked copy
            # keeps the semantic part local (keyword-only for this request).
            outbound_query, query_secret_kinds = contracts.redact_text(request["query"])
            if query_secret_kinds and (
                contracts.would_withhold(request["query"])
                or contracts.contains_secret(outbound_query)
            ):
                reasons.append("embedding_unavailable")
            else:
                try:
                    from .embeddings_index import collect_vectors

                    vectors = collect_vectors(store, candidate_ids, embedder)
                    if vectors:
                        query_vector = embedder.embed([outbound_query])[0]
                        from knowledge_weaver.embeddings.index import vector_ranking

                        for rank, revision_id in enumerate(
                            vector_ranking(query_vector=query_vector, vectors=vectors)
                        ):
                            scores[revision_id] = scores.get(revision_id, 0.0) + 1.0 / (
                                RRF_K + rank + 1
                            )
                    else:
                        reasons.append("embedding_unavailable")
                except Exception:  # noqa: BLE001 - degrade to keyword-only
                    reasons.append("embedding_unavailable")

    if intent == "history":
        ordered = sorted(scores, key=lambda revision_id: (-scores[revision_id], revision_id))
        # Build items only for the requested page window; counts stay complete.
        page_candidate_window = ordered[offset : offset + max(page_cap * 3, 24)]
        items = _load_items(store, page_candidate_window, scores, use_revision_status=True)
        total_hits = len(ordered)
    else:
        claim_map = _claims_for_revisions(store, list(scores))
        best_per_claim: dict[str, tuple[float, str]] = {}
        for revision_id, score in scores.items():
            claim = claim_map.get(revision_id)
            if claim is None:
                continue
            current_best = best_per_claim.get(claim)
            if current_best is None or score > current_best[0]:
                best_per_claim[claim] = (score, revision_id)
        ordered_claims = [
            revision_id
            for _claim, (_score, revision_id) in sorted(
                best_per_claim.items(), key=lambda pair: (-pair[1][0], pair[1][1])
            )
        ]
        window = ordered_claims[offset : offset + max(page_cap * 3, 24)]
        items = _load_items(store, window, scores)
        total_hits = len(ordered_claims)

    budget = int(request["budget_tokens"])
    selected_items = []
    used_tokens = 0
    truncated = False
    consumed = 0
    for item in items:
        if len(selected_items) >= page_cap:
            truncated = True
            break
        cost = estimate_tokens(item["text"])
        if used_tokens + cost > budget:
            truncated = True
            if selected_items:
                break
            # F08: the first candidate must obey the budget too. An oversized
            # single item is skipped (it can never fit any page) while the cursor
            # still advances past it, so pagination cannot stall on one item.
            consumed += 1
            continue
        selected_items.append(item)
        used_tokens += cost
        consumed += 1
    next_cursor = None
    # The cursor advances by every candidate actually consumed (selected *and*
    # budget-skipped); an item that fits never regresses the position. Because a
    # truncated page consumes at least one candidate, a returned cursor always
    # makes progress.
    if consumed and (truncated or offset + consumed < total_hits):
        next_cursor = _encode_cursor(cursor_prefix, offset + consumed)
        reasons.append("budget")
    elapsed_ms = (time.monotonic() - started) * 1000.0
    if elapsed_ms > float(request["deadline_ms"]):
        reasons.append("deadline")
    # A completed job that quarantined/left unaccounted history is not proof
    # that no history exists. Include protected source dispositions in status.
    from knowledge_weaver.jobs import status_counts
    pending_extract = 0
    if not selected_items and total_hits == 0:
        try:
            extract = status_counts(store).get("extract", {})
            pending_extract = extract.get("pending", 0) + extract.get("running", 0) + extract.get("retry_wait", 0)
        except Exception:
            pending_extract = 0
    scope_sql = ""
    scope_params = [workspace_id]
    if project_ids:
        marks = ",".join("?" for _ in project_ids)
        scope_sql = f" AND EXISTS (SELECT 1 FROM json_each(r.project_ids_json) p WHERE p.value IN ({marks}))"
        scope_params.extend(project_ids)
    with store.read() as conn:
        protected = conn.execute(
            "SELECT 1 FROM lifecycle_records l JOIN event_parts p "
            "ON l.subject_id=p.event_id||'#'||p.revision||'@'||p.part_index "
            "JOIN event_revisions r ON r.event_id=p.event_id AND r.revision=p.revision "
            "JOIN events e ON e.event_id=p.event_id "
            "WHERE l.subject_kind='event_part' AND l.state='bootstrap_pending' "
            "AND p.availability='full' AND e.workspace_id=?"+scope_sql+" LIMIT 1",scope_params).fetchone()
    if pending_extract or protected:
        reasons.append("processing_pending")
    unique_reasons = []
    for reason in reasons:
        if reason not in unique_reasons:
            unique_reasons.append(reason)
    items_out = selected_items
    if not items_out and total_hits == 0:
        status = "degraded" if protected else "empty"
    elif unique_reasons and unique_reasons != ["budget"]:
        status = "degraded"
    elif unique_reasons == ["budget"] and truncated:
        status = "degraded"
    else:
        status = "ok"
    base.update(
        {
            "status": status,
            "items": items_out,
            "next_cursor": next_cursor,
            "total_hits": total_hits,
            "incomplete_reasons": unique_reasons,
            "used_tokens": used_tokens,
            "truncated": truncated,
        }
    )
    issues = contracts.validate_instance("RecallResult", base)
    if issues:  # pragma: no cover - would indicate an internal contract bug
        raise RuntimeError(f"recall result violates contract: {issues[0]}")
    return base


def _claims_for_revisions(store: Store, revision_ids: list[str]) -> dict[str, str]:
    """One query for the claim of every scored revision (avoid per-row connections)."""
    if not revision_ids:
        return {}
    mapping: dict[str, str] = {}
    with store.read() as conn:
        for start in range(0, len(revision_ids), 500):
            chunk = revision_ids[start : start + 500]
            placeholders = ",".join("?" for _ in chunk)
            rows = conn.execute(
                f"SELECT revision_id, claim_id FROM claim_revisions WHERE revision_id IN ({placeholders})",
                chunk,
            ).fetchall()
            for row in rows:
                mapping[row["revision_id"]] = row["claim_id"]
    return mapping
