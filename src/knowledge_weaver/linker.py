"""Entity linker — creates relations between extracted entities."""

from __future__ import annotations

import re
from dataclasses import dataclass

from knowledge_weaver.db import (
    get_entity,
    insert_entity,
    insert_relation,
    list_entities_by_type,
)


# Minimum project name length for substring matching (shorter = too ambiguous)
_MIN_PROJECT_NAME_LEN = 3
# Generic project names that should not participate in substring matching
_PROJECT_NAME_BLOCKLIST = {"项目", "系统", "服务", "Project", "System"}


def _is_substring_safe_project_name(name: str) -> bool:
    """Return True if the project name is safe for substring-based matching."""
    return len(name) >= _MIN_PROJECT_NAME_LEN and name not in _PROJECT_NAME_BLOCKLIST


@dataclass
class ExtractedEntity:
    id: str
    type: str
    name: str
    summary: str
    first_seen: str
    last_seen: str
    section_title: str = ""


@dataclass
class ParsedSection:
    title: str
    line_range: tuple[int, int]


@dataclass
class ParsedFile:
    date: str
    path: str
    sections: list[ParsedSection]


@dataclass
class LinkedRelation:
    id: str
    from_entity: str
    to_entity: str
    rel_type: str
    weight: float
    evidence: str


def generate_relation_id(from_id: str, to_id: str, rel_type: str) -> str:
    """Generate consistent deterministic relation ID."""
    return f"rel:{from_id}->{to_id}:{rel_type}"


# Type pairs eligible for co_occurrence linking (avoid O(n²) noise).
# Only include pairs where co-occurrence implies semantic connection.
# tech/fact are too numerous and produce noise when cross-linked.
_CO_OCCURRENCE_PAIRS: frozenset[frozenset[str]] = frozenset({
    frozenset({"decision", "risk"}),
    frozenset({"decision", "project"}),
    frozenset({"decision", "preference"}),
    frozenset({"risk", "project"}),
    frozenset({"risk", "preference"}),
    frozenset({"project", "preference"}),
    frozenset({"project", "task"}),
    frozenset({"decision", "task"}),
    frozenset({"risk", "task"}),
    frozenset({"idea", "project"}),
    frozenset({"idea", "decision"}),
})


MAX_PER_FILE_RELATIONS = 500

_STOPWORDS: set[str] = {
    # 中文停用词
    "的", "了", "是", "在", "和", "与", "为", "以", "被", "把",
    "从", "到", "对", "有", "用", "要", "将", "可", "能", "会",
    "也", "就", "都", "而", "或", "但", "这", "那", "其", "之",
    "不", "很", "上", "下", "中", "大", "小", "多", "少", "更",
    "已", "还", "又", "于", "如", "及", "等", "各", "每", "个",
    "做", "后", "前", "内", "外", "自", "向", "让", "使", "该",
    "吧", "吗", "呢", "啊", "呀", "嘛", "哦", "哈", "哎",
    # 英文停用词
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "at",
    "for", "by", "with", "from", "as", "is", "are", "was", "were",
    "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "will", "would", "could", "should", "may", "might", "shall",
    "it", "its", "this", "that", "these", "those", "we", "our",
    "you", "your", "they", "them", "he", "she", "his", "her",
    "not", "no", "nor", "but", "if", "so", "what", "which", "who",
    "how", "when", "where", "why", "about", "into", "over", "after",
    "before", "between", "under", "above",
}


def _tokenize(text: str) -> list[str]:
    """Tokenize text into Chinese characters and English words."""
    tokens: list[str] = []
    for chunk in re.split(r"\s+", text):
        english_words = re.findall(r"[a-zA-Z]{2,}", chunk)
        tokens.extend(w.lower() for w in english_words)
        chinese_chars = re.findall(r"[\u4e00-\u9fff]", chunk)
        tokens.extend(chinese_chars)
    return tokens


def _has_name_mention(e1: ExtractedEntity, e2: ExtractedEntity) -> bool:
    """Check if either entity's name appears in the other's summary."""
    return e2.name in e1.summary or e1.name in e2.summary


def _shared_non_stopwords(
    e1: ExtractedEntity,
    e2: ExtractedEntity,
    min_shared: int = 2,
) -> list[str]:
    """Return shared non-stopword tokens if count >= min_shared, else [].

    Filters out single-character tokens as they rarely carry meaningful
    semantic signal for relation quality.
    """
    tokens1 = set(_tokenize(e1.name + " " + e1.summary)) - _STOPWORDS
    tokens2 = set(_tokenize(e2.name + " " + e2.summary)) - _STOPWORDS
    # Only keep tokens with 2+ characters (filters single CJK chars like 技,术,关,键)
    tokens1 = {t for t in tokens1 if len(t) >= 2}
    tokens2 = {t for t in tokens2 if len(t) >= 2}
    shared = tokens1 & tokens2
    return list(shared) if len(shared) >= min_shared else []


def link_entities_in_file(
    conn,
    entities: list[ExtractedEntity],
    parsed_file: ParsedFile,
    source_path: str,
) -> list[LinkedRelation]:
    """Create relations between entities within the same file.

    Rules:
    - Same section -> RELATES_TO, weight 0.5
    - Different sections same day -> RELATES_TO, weight 0.3
    - Capped at MAX_PER_FILE_RELATIONS to prevent O(n²) explosion
    """
    relations: list[LinkedRelation] = []

    # Group entities by section for targeted linking
    by_section: dict[str, list[ExtractedEntity]] = {}
    for e in entities:
        by_section.setdefault(e.section_title, []).append(e)


    # Same-section linking — semantic condition (name mention or shared tokens)
    for section_entities in by_section.values():
        if len(section_entities) <= 1:
            continue
        for i in range(len(section_entities)):
            for j in range(i + 1, len(section_entities)):
                if len(relations) >= MAX_PER_FILE_RELATIONS:
                    break
                e1, e2 = section_entities[i], section_entities[j]
                evidence = ""
                weight = 0.5
                if _has_name_mention(e1, e2):
                    evidence = "name_mention"
                else:
                    shared = _shared_non_stopwords(e1, e2)
                    if shared:
                        evidence = "shared_tokens: " + ",".join(sorted(shared))
                    else:
                        # Co-occurrence fallback: only for high-value type pairs
                        type_pair = frozenset({e1.type, e2.type})
                        if type_pair in _CO_OCCURRENCE_PAIRS:
                            evidence = "co_occurrence"
                            weight = 0.3
                if not evidence:
                    continue
                rel = LinkedRelation(
                    id=generate_relation_id(e1.id, e2.id, "RELATES_TO"),
                    from_entity=e1.id, to_entity=e2.id,
                    rel_type="RELATES_TO", weight=weight,
                    evidence=evidence,
                )
                insert_relation(conn, {
                    "id": rel.id, "from_entity": rel.from_entity,
                    "to_entity": rel.to_entity, "rel_type": rel.rel_type,
                    "weight": rel.weight, "evidence": rel.evidence,
                }, auto_commit=False)
                relations.append(rel)
            if len(relations) >= MAX_PER_FILE_RELATIONS:
                break

    # CONTRADICTS detection between opposing decisions
    contradict_relations = link_contradicts_in_file(conn, entities)
    relations.extend(contradict_relations)

    return relations


def link_cross_day(
    conn,
    new_entities: list[ExtractedEntity],
    existing_entity_ids: set[str],
) -> list[LinkedRelation]:
    """Handle cross-day entity continuity — update day_count for known entities.

    Self-loop CONTINUES/REFINES relations are no longer inserted because
    they pollute the relationship graph without adding information (the
    entity ID already encodes continuity).
    """
    for entity in new_entities:
        db_entity = get_entity(conn, entity.id)

        if db_entity is not None:
            new_day_count = db_entity["day_count"] + 1
            conn.execute(
                "UPDATE entities SET day_count=?, last_seen=?, updated_at=datetime('now') WHERE id=?",
                (new_day_count, entity.last_seen, entity.id),
            )
        else:
            insert_entity(conn, {
                "id": entity.id,
                "type": entity.type,
                "name": entity.name,
                "summary": entity.summary,
                "importance": 0.0,
                "first_seen": entity.first_seen,
                "last_seen": entity.last_seen,
                "day_count": 1,
            })

        existing_entity_ids.add(entity.id)

    return []


def _name_mentioned(name: str, text: str) -> bool:
    """True if ``name`` appears in ``text`` as a whole token (K7).

    Word-boundary match so a project 'Home' does NOT match inside 'HomeBrain',
    while CJK names still match (CJK chars are not [A-Za-z0-9] so the boundaries
    are satisfied around them).
    """
    if not name or not text:
        return False
    pat = re.compile(r"(?<![A-Za-z0-9])" + re.escape(name) + r"(?![A-Za-z0-9])",
                     re.IGNORECASE)
    return bool(pat.search(text))


def link_project_dependencies(
    conn,
    entities: list[ExtractedEntity],
) -> list[LinkedRelation]:
    """Detect DEPENDS_ON relations between projects and their sub-entities.

    Rules:
    - If a task/decision's name contains a project entity's name -> DEPENDS_ON (weight 0.7)
    - Also check DB for project entities
    """
    relations: list[LinkedRelation] = []

    projects: list[ExtractedEntity] = [e for e in entities if e.type == "project"]

    seen_names = {p.name for p in projects}
    db_projects = list_entities_by_type(conn, "project")
    for row in db_projects:
        if row["name"] not in seen_names:
            seen_names.add(row["name"])
            projects.append(ExtractedEntity(
                id=row["id"],
                type=row["type"],
                name=row["name"],
                summary=row["summary"],
                first_seen=row["first_seen"],
                last_seen=row["last_seen"],
            ))

    for entity in entities:
        if entity.type == "project":
            continue

        for project in projects:
            if not _is_substring_safe_project_name(project.name):
                continue
            if _name_mentioned(project.name, entity.name) or \
               _name_mentioned(project.name, entity.summary):
                rel = LinkedRelation(
                    id=generate_relation_id(entity.id, project.id, "DEPENDS_ON"),
                    from_entity=entity.id,
                    to_entity=project.id,
                    rel_type="DEPENDS_ON",
                    weight=0.8,
                    evidence="name_mention",
                )
                insert_relation(conn, {
                    "id": rel.id,
                    "from_entity": rel.from_entity,
                    "to_entity": rel.to_entity,
                    "rel_type": rel.rel_type,
                    "weight": rel.weight,
                    "evidence": rel.evidence,
                }, auto_commit=False)
                relations.append(rel)

    return relations


# Opposing stance keywords for CONTRADICTS detection
_CONTRADICT_PAIRS: list[tuple[set[str], set[str]]] = [
    ({"采用", "选择", "决定使用", "改用"}, {"放弃", "排除", "不采用", "弃用"}),
    ({"增加", "新增", "添加", "引入"}, {"移除", "删除", "去掉", "废弃"}),
    ({"升级", "更新到", "迁移到"}, {"回退", "降级", "回滚到"}),
]


def link_contradicts_in_file(
    conn,
    entities: list[ExtractedEntity],
) -> list[LinkedRelation]:
    """Detect CONTRADICTS relations between opposing decision entities.

    Two decisions are flagged as conflicting if their summaries contain
    opposing stance keywords (e.g. "采用X" vs "放弃X") and they share
    at least 1 non-stopword.
    """
    relations: list[LinkedRelation] = []
    decisions = [e for e in entities if e.type == "decision"]
    if len(decisions) < 2:
        return relations

    for i in range(len(decisions)):
        for j in range(i + 1, len(decisions)):
            d1, d2 = decisions[i], decisions[j]
            s1, s2 = d1.summary.lower(), d2.summary.lower()
            for pos_set, neg_set in _CONTRADICT_PAIRS:
                pos_in_s1 = any(kw in s1 for kw in pos_set)
                neg_in_s1 = any(kw in s1 for kw in neg_set)
                pos_in_s2 = any(kw in s2 for kw in pos_set)
                neg_in_s2 = any(kw in s2 for kw in neg_set)
                if (pos_in_s1 and neg_in_s2) or (neg_in_s1 and pos_in_s2):
                    shared = _shared_non_stopwords(d1, d2, min_shared=1)
                    if shared:
                        rel = LinkedRelation(
                            id=generate_relation_id(d1.id, d2.id, "CONTRADICTS"),
                            from_entity=d1.id, to_entity=d2.id,
                            rel_type="CONTRADICTS", weight=0.7,
                            evidence=f"opposing_stance: {','.join(shared[:3])}",
                        )
                        insert_relation(conn, {
                            "id": rel.id, "from_entity": rel.from_entity,
                            "to_entity": rel.to_entity, "rel_type": rel.rel_type,
                            "weight": rel.weight, "evidence": rel.evidence,
                        }, auto_commit=False)
                        relations.append(rel)
                        break
    return relations
