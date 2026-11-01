"""W2: structured-fact ingestion — read LLM-emitted JSON facts from a daily
memory file instead of regex-scraping the prose.

DMA's cloud summarizer emits, alongside the ``### 摘要`` markdown, a
``### 结构化事实`` section containing a fenced ```json array of atomic facts.
KW ingests those structurally, which removes the whole class of Chinese
regex-extraction noise at the source and gives consistent entity typing.

This module is ADDITIVE: the markdown ``### 摘要`` is preserved and the regex
path stays the default. Use ``shadow_compare_file`` to compare the two
extractors on real files before switching the live path (KNOWLEDGE_WEAVER_EXTRACT_MODE=json).
"""
from __future__ import annotations

import json
import os
import re
from typing import Optional

from knowledge_weaver.extractor import ExtractedEntity, generate_entity_id

# KW's closed entity-type set (must match extractor.TYPE_PREFIX).
VALID_TYPES = {
    "project", "preference", "decision", "fact", "risk", "task", "tech", "idea",
}

_FACTS_HEADING = "### 结构化事实"
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json_facts_block(text: str) -> Optional[list[dict]]:
    """Return the parsed list of fact dicts from the ``### 结构化事实`` block,
    or None if the block is absent / empty / not valid JSON array."""
    idx = text.find(_FACTS_HEADING)
    if idx == -1:
        return None
    rest = text[idx + len(_FACTS_HEADING):]
    nxt = rest.find("\n### ")
    if nxt != -1:
        rest = rest[:nxt]
    m = _FENCE_RE.search(rest)
    raw = (m.group(1) if m else rest).strip()
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, list) else None


def facts_to_entities(facts: Optional[list[dict]], relative_path: str) -> list[ExtractedEntity]:
    """Map validated JSON facts to ExtractedEntity objects (dedup by id).

    Invalid facts (unknown type, empty name, non-dict) are skipped. ``aliases``
    is carried into metadata so the W1 resolver / registry can use them.
    """
    out: list[ExtractedEntity] = []
    seen: set[str] = set()
    for f in facts or []:
        if not isinstance(f, dict):
            continue
        etype = f.get("type")
        name = (f.get("name") or "").strip()
        if etype not in VALID_TYPES or not name:
            continue
        eid = generate_entity_id(etype, name)
        # K2: a name that slugifies to empty (emoji / punctuation-only) yields just
        # the type prefix ("fact:"), so all such facts would UPSERT-collapse into one
        # garbage entity. Drop them (the regex path already does via _is_garbage_name).
        if eid.endswith(":"):
            continue
        if eid in seen:
            continue
        seen.add(eid)
        summary = (f.get("summary") or f.get("content") or "").strip()
        meta: dict = {}
        aliases = f.get("aliases")
        if isinstance(aliases, list):
            clean = [a.strip() for a in aliases if isinstance(a, str) and a.strip()]
            if clean:
                meta["aliases"] = clean
        out.append(ExtractedEntity(
            id=eid, type=etype, name=name, summary=summary,
            source_lines=json.dumps([f"{relative_path}:0:0"], ensure_ascii=False),
            metadata=meta,
        ))
    return out


def regex_entities_for_file(path: str) -> list[ExtractedEntity]:
    """Run the legacy regex extraction over a file (for shadow comparison)."""
    from knowledge_weaver.extractor import extract_entities_from_section
    from knowledge_weaver.parser import parse_dma_file

    relative = f"memory/{os.path.basename(path)}"
    parsed = parse_dma_file(path)
    out: list[ExtractedEntity] = []
    seen: set[str] = set()
    for section in parsed.sections:
        if "结构化事实" in (section.title or ""):
            continue  # never regex-scrape the JSON facts block
        for e in extract_entities_from_section(section, relative, dma_category=section.title):
            if e.id not in seen:
                seen.add(e.id)
                out.append(e)
    return out


def shadow_compare_file(path: str) -> dict:
    """Compare regex vs JSON-fact extraction on a file WITHOUT writing anything.

    Returns the entity-id sets that are regex-only, json-only, or in both, so
    you can judge structured-handoff quality before cutting over.
    """
    relative = f"memory/{os.path.basename(path)}"
    regex_ents = regex_entities_for_file(path)
    with open(path, encoding="utf-8") as f:
        text = f.read()
    facts = extract_json_facts_block(text)
    json_ents = facts_to_entities(facts, relative) if facts is not None else []

    regex_map = {e.id: e.name for e in regex_ents}
    json_map = {e.id: e.name for e in json_ents}
    regex_ids = set(regex_map)
    json_ids = set(json_map)

    def fmt(ids, src):
        return sorted(f"{i} ({src[i]})" for i in ids)

    return {
        "file": os.path.basename(path),
        "has_json_block": facts is not None,
        "regex_count": len(regex_ids),
        "json_count": len(json_ids),
        "regex_only": fmt(regex_ids - json_ids, regex_map),
        "json_only": fmt(json_ids - regex_ids, json_map),
        "both": sorted(regex_ids & json_ids),
    }
