"""Second-pass semantic dedup over EXISTING entities.

Ingest-time resolution (resolver.py) only compares a *freshly extracted* entity
against the DB. It never re-compares two entities that were both created in the
same batch — e.g. a historical rebuild where the LLM phrased the same concept
slightly differently per file (``聚合逻辑集成到HomeBrain`` vs ``…自身不依赖外部脚本``).
Those become distinct ids that never merge, so cross-file accumulation
(day_count / files / importance) silently breaks (real-data finding 2026-06-04:
only 1/262 entities were multi-file).

This pass walks the whole DB, finds same-type vector-similar pairs, and:
  - cosine >= HIGH  -> auto-merge (survivor = higher day_count, accumulating it)
  - MID <= cosine < HIGH -> queue to merge_review (human decides)
  - identifier-like names -> review only, never auto (bge-m3 over-similarity)

Reuses the resolver's bands + identifier guard so behaviour matches ingest-time.
Merges accumulate day_count, which is exactly the signal the rebuild flattened.
"""
from __future__ import annotations

from knowledge_weaver.db import (
    _cosine,
    get_entity_vector,
    insert_review,
    merge_existing_entities,
    search_entity_vectors,
)
from knowledge_weaver.resolver import (
    CANDIDATE_K,
    RESOLVE_HIGH,
    RESOLVE_MID,
    _digit_variant_pair,
    _merge_unreliable,
)


def find_dedupe_actions(conn, *, high: float = RESOLVE_HIGH,
                        mid: float = RESOLVE_MID) -> dict:
    """Compute (without writing) the merge/review actions for the current DB.

    Survivors are processed by descending (day_count, importance) so the entity
    carrying the most accumulated signal stays canonical and absorbs the rest.
    Returns {"merges": [(from_id, into_id, score)], "reviews": [...]}.
    """
    rows = [dict(r) for r in conn.execute(
        "SELECT id, type, name, day_count, importance FROM entities").fetchall()]
    rows.sort(key=lambda r: (r["day_count"] or 0, r["importance"] or 0.0), reverse=True)

    # Pairs already reviewed (pending OR dismissed) — don't re-surface them every
    # cycle. Without this the cron loop re-queues and re-judges (LLM cost) the same
    # distinct pairs forever.
    reviewed: set[frozenset] = {
        frozenset((row[0], row[1]))
        for row in conn.execute(
            "SELECT new_entity_id, candidate_id FROM merge_review "
            "WHERE kind='merge' AND candidate_id IS NOT NULL "
            "AND status IN ('pending','dismissed','rejected')").fetchall()
    }

    merged_away: set[str] = set()
    merges: list[tuple[str, str, float]] = []
    reviews: list[tuple[str, str, float]] = []
    seen_pairs: set[frozenset] = set()

    # Cache decoded embeddings: a candidate that neighbors K survivors would
    # otherwise be SELECT+json.loads'd K times on the cron hot path.
    vec_cache: dict = {}

    def _vec(eid):
        if eid not in vec_cache:
            vec_cache[eid] = get_entity_vector(conn, eid)
        return vec_cache[eid]

    for r in rows:
        if r["id"] in merged_away:
            continue
        vec = _vec(r["id"])
        if not vec:
            continue
        for nb in search_entity_vectors(conn, vec, limit=CANDIDATE_K + 1):
            cid = nb["id"]
            if cid == r["id"] or cid in merged_away or nb["type"] != r["type"]:
                continue
            pair = frozenset((r["id"], cid))
            if pair in seen_pairs or pair in reviewed:
                continue
            seen_pairs.add(pair)
            cvec = _vec(cid)
            if not cvec:
                continue
            cos = _cosine(vec, cvec)
            unreliable = (_merge_unreliable(r["name"]) or _merge_unreliable(nb["name"])
                          or _digit_variant_pair(r["name"], nb["name"]))
            if cos >= high and not unreliable:
                merges.append((cid, r["id"], round(cos, 4)))
                merged_away.add(cid)
            elif cos >= mid:
                reviews.append((cid, r["id"], round(cos, 4)))

    return {"merges": merges, "reviews": reviews}


def _queue_review(conn, cand_id: str, into_id: str, score: float) -> None:
    et = conn.execute("SELECT type FROM entities WHERE id=?", (into_id,)).fetchone()
    insert_review(conn, kind="merge", new_entity_id=cand_id, candidate_id=into_id,
                  entity_type=et["type"] if et else "", score=score,
                  reason="semantic-dedupe", auto_commit=False)


def semantic_dedupe(conn, *, dry_run: bool = False, auto_merge: bool = True,
                    high: float = RESOLVE_HIGH, mid: float = RESOLVE_MID) -> dict:
    """Run the second-pass dedup.

    auto_merge=True  : HIGH band auto-merged, MID band queued (one-off cleanup).
    auto_merge=False : BOTH bands queued to review, nothing merged (safe default
                       for an unattended cron — a human approves every merge).

    Merges are rollback-logged (merge_log) and FTS keeps the merged-away name as
    a searchable alias, so recall via the variant name is preserved.
    """
    actions = find_dedupe_actions(conn, high=high, mid=mid)
    merged = queued = 0
    if not dry_run:
        for from_id, into_id, score in actions["merges"]:
            if auto_merge:
                if merge_existing_entities(conn, from_id=from_id, into_id=into_id,
                                           reason="semantic-dedupe", score=score,
                                           auto_commit=False):
                    merged += 1
            else:
                _queue_review(conn, from_id, into_id, score)
                queued += 1
        for cand_id, into_id, score in actions["reviews"]:
            _queue_review(conn, cand_id, into_id, score)
            queued += 1
        conn.commit()
    return {
        "merged": merged if not dry_run else 0,
        "queued": queued if not dry_run else 0,
        "candidates_high": len(actions["merges"]),
        "candidates_mid": len(actions["reviews"]),
    }
