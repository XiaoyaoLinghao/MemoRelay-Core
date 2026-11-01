"""W3.4 core: re-type generic RELATES_TO edges into the typed knowledge-graph
vocabulary (relations_schema.REL_TYPES) via an injected typer, or prune them.

Lives in the package (not just the script) so both the one-off rebuild
(scripts/build_typed_relations.py) AND online consolidation (server CLI) can
type new edges. The LLM call (``llm_type_pairs``) is isolated so the DB surgery
(``type_relations``) is deterministically testable with a mock.
"""
from __future__ import annotations

from collections import Counter

from knowledge_weaver.db import get_entity, init_db, insert_relation
from knowledge_weaver.linker import generate_relation_id
from knowledge_weaver.relations_schema import (
    REL_TYPES,
    schema_prompt,
    triplet_hints_prompt,
)


def candidates(conn) -> list[dict]:
    """Generic RELATES_TO edges joined with both endpoints' type/name/summary."""
    rows = conn.execute(
        "SELECT id, from_entity, to_entity, weight, evidence FROM relations "
        "WHERE rel_type='RELATES_TO'"
    ).fetchall()
    out = []
    for r in rows:
        a, b = get_entity(conn, r["from_entity"]), get_entity(conn, r["to_entity"])
        if a is None or b is None:
            continue
        out.append({
            "rel_id": r["id"], "from_id": r["from_entity"], "to_id": r["to_entity"],
            "weight": r["weight"], "evidence": r["evidence"],
            "from": f"{a['type']}: {a['name']} — {(a['summary'] or '')[:150]}",
            "to": f"{b['type']}: {b['name']} — {(b['summary'] or '')[:150]}",
        })
    return out


def type_relations(db_path: str, type_fn, *, dry_run: bool = False) -> dict:
    """Re-type RELATES_TO edges via ``type_fn(cands) -> {rel_id: type|none}``.

    Replaces each edge with a typed edge (closed vocab) or prunes it (none)."""
    conn = init_db(db_path)
    cands = candidates(conn)
    decisions = type_fn(cands) if cands else {}

    typed = pruned = 0
    type_counts: Counter = Counter()
    for c in cands:
        t = decisions.get(c["rel_id"])
        if t not in REL_TYPES:  # none / invalid -> prune spurious edge
            pruned += 1
            if not dry_run:
                conn.execute("DELETE FROM relations WHERE id=?", (c["rel_id"],))
            continue
        typed += 1
        type_counts[t] += 1
        if not dry_run:
            conn.execute("DELETE FROM relations WHERE id=?", (c["rel_id"],))
            insert_relation(conn, {
                "id": generate_relation_id(c["from_id"], c["to_id"], t),
                "from_entity": c["from_id"], "to_entity": c["to_id"],
                "rel_type": t, "weight": c["weight"], "evidence": "llm-typed",
            }, auto_commit=False)
    if not dry_run:
        conn.commit()
    remaining = conn.execute(
        "SELECT rel_type, COUNT(*) c FROM relations GROUP BY rel_type").fetchall()
    conn.close()
    return {
        "candidates": len(cands), "typed": typed, "pruned": pruned,
        "by_type": dict(type_counts),
        "relations_after": {r["rel_type"]: r["c"] for r in remaining},
    }


def build_typing_system_prompt() -> str:
    """Tightened typing prompt: schema + triplet hints + rules (prefer specific
    types; 导致 only for true causation; prune coincidental; 相关 last resort) +
    few-shot. Pulled out so it is testable without an LLM call."""
    return (
        "你是知识图谱关系判定助手。给你若干有方向的实体对（A→B），"
        "请判断每一对中 A 与 B 之间最贴切的关系类型。\n\n"
        + schema_prompt()
        + "\n\n常见组合参考（优先据此往具体类型判，非硬性约束）：\n"
        + triplet_hints_prompt()
        + "\n\n判定规则：\n"
        "1. 优先选具体类型（依赖/使用/包含/导致/取代/关于）；只有确实找不到更具体关系、"
        "且两者确有实质关联时，才用「相关」。\n"
        "2. 「导致」只用于**真正的因果**，判定前先做反事实自问：「若 A 不发生，B 还会发生吗？」"
        "——答『还会/不一定不会』就【不是导致】。下列【一律不算因果】，应判「相关」「关于」或 none，"
        "不要判「导致」：① 先做 A 再做 B 的时间先后；② 排查/处理同一问题的连续步骤；③ A 与 B 只是"
        "伴随出现/同属一件事。「导致」要求 A 是 B 的**直接成因**（决策引发任务、故障引发风险这类）。\n"
        "3. 若两者只是碰巧在同一段对话里出现、彼此没有实质关系 → 输出 none（剪掉），"
        "不要塞进「相关」。\n"
        "4. 注意方向 A→B，按内容判最贴切的类型。\n\n"
        "示例：\n"
        "  A(project: HomeBrain — 智能家庭中枢) → B(tech: FastAPI — 后端框架) ⇒ 使用\n"
        "  A(decision: 改用规则引擎 — 弃用LLM聚合) → B(task: 重构聚合层) ⇒ 导致\n"
        "  A(task: 排查接口限流) → B(task: 检查是否切到DeepSeek) ⇒ 相关  （同一排查的先后步骤，非因果）\n"
        "  A(project: KW — 知识抽取) → B(tech: sqlite-vec — 向量库) ⇒ 依赖\n"
        "  A(fact: 今天天气好) → B(decision: 后端用FastAPI) ⇒ none\n\n"
        "对每一对，按其序号输出关系类型或 none；只输出一个 JSON 对象，"
        '形如 {"1": "使用", "2": "none", ...}，不要任何其它文字。'
    )


def reset_llm_typed_edges(conn, auto_commit: bool = True) -> int:
    """Convert previously LLM-typed edges (evidence='llm-typed') back to
    RELATES_TO so they can be re-typed with an updated prompt. Preserves the
    original DEPENDS_ON / co-occurrence edges (different evidence)."""
    rows = [dict(r) for r in conn.execute(
        "SELECT id, from_entity, to_entity, weight FROM relations "
        "WHERE evidence='llm-typed'").fetchall()]
    for r in rows:
        conn.execute("DELETE FROM relations WHERE id=?", (r["id"],))
        insert_relation(conn, {
            "id": generate_relation_id(r["from_entity"], r["to_entity"], "RELATES_TO"),
            "from_entity": r["from_entity"], "to_entity": r["to_entity"],
            "rel_type": "RELATES_TO", "weight": r["weight"], "evidence": "co_occurrence",
        }, auto_commit=False)
    if auto_commit:
        conn.commit()
    return len(rows)


def llm_type_pairs(cands: list[dict], *, api_url: str, api_key: str, model: str,
                   chunk: int = 20, timeout: float = 90.0) -> dict:
    """Type candidate edges via an OpenAI-compatible chat model (chunked).

    Returns {rel_id: type_string}. A failed/invalid chunk is left out (the edges
    stay RELATES_TO) — see knowledge_weaver._llm.classify_items.
    """
    from knowledge_weaver._llm import classify_items
    return classify_items(
        cands, key=lambda c: c["rel_id"],
        render=lambda c: f'A({c["from"]})  →  B({c["to"]})',
        system_prompt=build_typing_system_prompt(), user_prefix="实体对：\n",
        api_url=api_url, api_key=api_key, model=model, chunk=chunk, timeout=timeout)
