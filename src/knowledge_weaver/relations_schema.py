"""W3.4: schema for typed knowledge-graph relations.

KW's relations are currently generic ``RELATES_TO`` (co-occurrence within a
file). This module defines a small CLOSED relation vocabulary so an LLM can
re-type each candidate edge into a meaningful, directed relation — or abstain
(``none``) to prune spurious co-occurrence edges. Closed vocab is essential:
unconstrained extraction yields thousands of relation types (Neo4j finding).

Entity types (KW): project / preference / decision / fact / risk / task / tech / idea.
"""
from __future__ import annotations

# Closed relation vocabulary (directed A -> B). "相关" is the weak fallback.
REL_TYPES: list[str] = [
    "依赖",   # A depends on B to work / hold
    "使用",   # A uses B (a tech / tool)
    "包含",   # A contains / has sub-part B
    "导致",   # A causes / triggers B
    "取代",   # A supersedes / overturns B (new replaces old)
    "关于",   # A is a fact / decision / risk ABOUT B
    "矛盾",   # A contradicts / conflicts with B
    "相关",   # related, no more specific relation (weak fallback)
]

REL_DESC: dict[str, str] = {
    "依赖": "A 依赖 B 才能工作或成立",
    "使用": "A 使用了 B（某技术/工具）",
    "包含": "A 包含或下属 B",
    "导致": "A 导致或引发了 B",
    "取代": "A 取代/推翻 B（新替旧）",
    "关于": "A 是关于 B 的事实/决策/风险/任务",
    "矛盾": "A 与 B 互相矛盾/冲突",
    "相关": "A 与 B 有关联，但没有上面更具体的关系",
}

# Soft guidance: which directed (from_type -> to_type) pairs each relation
# typically connects. Used to hint the LLM; not enforced as a hard filter.
TRIPLET_HINTS: dict[str, list[tuple[str, str]]] = {
    "依赖": [("project", "project"), ("project", "tech"), ("task", "task"), ("task", "tech")],
    "使用": [("project", "tech"), ("task", "tech"), ("decision", "tech")],
    "包含": [("project", "task"), ("project", "tech"), ("project", "project")],
    "导致": [("decision", "task"), ("decision", "risk"), ("fact", "risk"), ("risk", "risk")],
    "取代": [("decision", "decision"), ("fact", "fact"), ("tech", "tech")],
    "关于": [("fact", "project"), ("fact", "tech"), ("decision", "project"),
             ("risk", "project"), ("task", "project"), ("idea", "project"),
             ("preference", "tech")],
    "矛盾": [("decision", "decision"), ("fact", "fact")],
}


def schema_prompt() -> str:
    """Render the closed relation vocabulary for an LLM typing prompt."""
    lines = ["关系类型（只能从下面这 8 种里选一个，或回答 none 表示两者间没有实质关系）："]
    for t in REL_TYPES:
        lines.append(f"  {t}：{REL_DESC[t]}")
    return "\n".join(lines)


def triplet_hints_prompt() -> str:
    """Render typical (from_type → to_type) combos per relation, to bias the LLM
    toward specific structural types instead of the weak 相关 fallback."""
    lines = []
    for rel, pairs in TRIPLET_HINTS.items():
        combos = "、".join(f"{a}→{b}" for a, b in pairs[:4])
        lines.append(f"  {rel}：{combos}")
    return "\n".join(lines)
