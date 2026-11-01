"""T1 (part B) — LLM tie-breaker for the merge-review queue.

The resolver / semantic_dedupe deliberately demote uncertain pairs (high signal
but identifier-like / digit-variant) to ``merge_review`` instead of hand-rolling
more regex to decide them. This batch tool asks an LLM the single question that
settles such a pair — "the same thing, or two different things?" — then
auto-merges the 'same', dismisses the 'different', and leaves only genuine
ambiguity for a human.

Generalizes to any identifier format with zero new rules, and clears the review
backlog the cheap heuristics intentionally leave behind. The LLM call
(``llm_judge_pairs``) is isolated so the DB surgery (``apply_tiebreak``) is
deterministically testable with a mock.
"""
from __future__ import annotations


from knowledge_weaver.db import (
    get_entity,
    list_pending_reviews,
    merge_existing_entities,
    set_review_status,
)


def tiebreak_pairs(conn) -> list[dict]:
    """Pending merge-review items joined with both entities' type/name/summary."""
    out = []
    for r in list_pending_reviews(conn, limit=100000):
        if r["kind"] != "merge" or not r["candidate_id"]:
            continue
        a = get_entity(conn, r["new_entity_id"])   # the entity to merge away
        b = get_entity(conn, r["candidate_id"])    # the surviving / canonical one
        if a is None or b is None:
            continue
        out.append({
            "review_id": r["id"], "from_id": a["id"], "into_id": b["id"],
            "a": f"{a['type']}: {a['name']} — {(a['summary'] or '')[:120]}",
            "b": f"{b['type']}: {b['name']} — {(b['summary'] or '')[:120]}",
        })
    return out


def apply_tiebreak(conn, decisions: dict, pairs: list[dict], *,
                   dry_run: bool = False) -> dict:
    """Act on ``{review_id: 'same'|'different'|...}``: merge 'same', dismiss
    'different', keep anything else pending. Merges are rollback-logged."""
    merged = dismissed = kept = 0
    for p in pairs:
        d = (decisions.get(p["review_id"]) or "").strip().lower()
        if d == "same":
            ok = True
            if not dry_run:
                ok = merge_existing_entities(conn, from_id=p["from_id"],
                                             into_id=p["into_id"],
                                             reason="llm-tiebreak", auto_commit=False)
                set_review_status(conn, p["review_id"],
                                  "merged" if ok else "dismissed", auto_commit=False)
            # a no-op merge (target already removed by a prune / earlier merge) is
            # recorded as dismissed, so count it there — not as a phantom merge.
            if ok:
                merged += 1
            else:
                dismissed += 1
        elif d == "different":
            if not dry_run:
                set_review_status(conn, p["review_id"], "dismissed", auto_commit=False)
            dismissed += 1
        else:
            kept += 1
    if not dry_run:
        conn.commit()
    return {"merged": merged, "dismissed": dismissed, "kept": kept,
            "candidates": len(pairs)}


def tiebreak_reviews(conn, judge_fn, *, dry_run: bool = False) -> dict:
    """Resolve pending merge-review pairs via ``judge_fn(pairs) -> {review_id: verdict}``."""
    pairs = tiebreak_pairs(conn)
    decisions = judge_fn(pairs) if pairs else {}
    return apply_tiebreak(conn, decisions, pairs, dry_run=dry_run)


_JUDGE_PROMPT = (
    "你在判断知识库里两个实体是不是同一个东西。给你若干对实体(A / B,各含类型+名称+摘要)。"
    "对每一对,判断 A 与 B 是【同一事物的不同写法】回 same,还是【两个不同的事物】回 different。\n"
    "要点:① 版本号(v0.2.0 vs v2.9.0)、编号(COMP7940 vs COMP7240)、不同端口/批次/序号"
    "= different(同模板的不同实例,不是重复);② 跨语言或缩写别名(HomeBrain vs 家庭大脑、"
    "KW vs Knowledge Weaver)= same;③ 拿不准偏 different(不乱合并)。\n"
    '只输出一个 JSON 对象,形如 {"1":"same","2":"different",...},不要任何其它文字。'
)


def llm_judge_pairs(pairs: list[dict], *, api_url: str, api_key: str, model: str,
                    chunk: int = 20, timeout: float = 90.0) -> dict:
    """Ask an OpenAI-compatible model 'same or different?' per pair (chunked).

    Returns ``{review_id: 'same'|'different'}``; a failed or index-mismatched
    chunk is left out (those reviews stay pending). See _llm.classify_items.
    """
    from knowledge_weaver._llm import classify_items
    return classify_items(
        pairs, key=lambda c: c["review_id"],
        render=lambda c: f'A({c["a"]})  ||  B({c["b"]})',
        system_prompt=_JUDGE_PROMPT, user_prefix="实体对:\n",
        api_url=api_url, api_key=api_key, model=model, chunk=chunk, timeout=timeout)
