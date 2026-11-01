"""Recover real entities lost in a historical rebuild by porting them from the
pre-swap DB.

Real-data finding (2026-06-04): the backfill rebuild dropped 51 decisions
(100% real knowledge, 0 noise) and ~42 meaningful tech terms. Those entities
still exist — cleanly, WITH their accumulated day_count/first_seen — in the
pre-swap DB. Re-extracting them from compressed summaries is strictly lossier
than porting the existing rows. Porting also carries the stored bge-m3 vector,
so no re-embedding is needed (same model/dim).

An entity is "missing" iff its (deterministic) id is absent from the new DB.
Decisions are 0% noise -> port all. Tech is ~80% noise -> gate behind a keep_fn
(rule or LLM keep/drop). After porting, run semantic_dedupe (catch near-dups vs
existing new entities) then build_typed_relations + rebuild_fts.
"""
from __future__ import annotations

import re

from knowledge_weaver.db import get_entity_vector, insert_entity, upsert_entity_vector

# Columns carried over verbatim (incl. accumulated day_count + alias metadata).
PORTABLE_FIELDS = (
    "id", "type", "name", "summary", "importance",
    "first_seen", "last_seen", "day_count", "source_lines", "metadata",
)


def missing_entities(old_conn, new_conn, types: list[str]) -> list[dict]:
    """Old entities of the given types whose id is absent from the new DB."""
    new_ids = {r[0] for r in new_conn.execute("SELECT id FROM entities").fetchall()}
    ph = ",".join("?" * len(types))
    rows = old_conn.execute(
        f"SELECT * FROM entities WHERE type IN ({ph})", tuple(types)
    ).fetchall()
    return [dict(r) for r in rows if r["id"] not in new_ids]


def port_entities(old_conn, new_conn, rows: list[dict], *, keep_fn=None,
                  dry_run: bool = False) -> dict:
    """Insert each (kept) old row into the new DB, carrying its stored vector."""
    ported = skipped = with_vector = 0
    for e in rows:
        if keep_fn is not None and not keep_fn(e):
            skipped += 1
            continue
        vec = get_entity_vector(old_conn, e["id"])
        if vec:
            with_vector += 1
        if not dry_run:
            insert_entity(new_conn, {k: e.get(k) for k in PORTABLE_FIELDS},
                          auto_commit=False)
            if vec:
                upsert_entity_vector(new_conn, e["id"], vec, auto_commit=False)
        ported += 1
    if not dry_run:
        new_conn.commit()
    return {"candidates": len(rows), "ported": ported,
            "skipped": skipped, "with_vector": with_vector}


# Bare filename (alphabetic extension .json/.py) = always noise; a numeric
# extension (.1, glm-5.1) is a version and KEPT — hence the extension is [A-Za-z].
_FILENAME_RE = re.compile(r"^[\w-]+\.[A-Za-z]{1,5}$")
# Short all-caps (RTDL/SOUL/NEON noise, but also real abbrevs AWS/GPT/SQL) —
# can't be told apart by name, so do NOT hard-drop: let the signal gate decide.
_SHORT_ALLCAPS_RE = re.compile(r"^[A-Z]{1,4}$")


def rule_keep_tech(entity: dict) -> bool:
    """Cheap name-only noise check: drop ≤2-char names and bare filenames.

    Short all-caps is NOT dropped here (a real ``AWS``/``GPT`` is indistinguishable
    by name from noise ``SOUL``) — ``signal_keep_tech`` only keeps those with
    accumulated signal, so the name rule no longer silently discards real abbrevs.
    """
    name = (entity.get("name") or "").strip()
    if len(name) <= 2:
        return False
    return not _FILENAME_RE.match(name)


def _has_relation(conn, entity_id: str) -> bool:
    """True if the entity is connected by ≥1 edge in the (old) DB."""
    return conn.execute(
        "SELECT 1 FROM relations WHERE from_entity=? OR to_entity=? LIMIT 1",
        (entity_id, entity_id),
    ).fetchone() is not None


def signal_keep_tech(entity: dict, old_conn, *, min_day_count: int = 2) -> bool:
    """Signal-gate for tech recovery (PRIMARY filter).

    A tech entity is worth keeping only if it left real accumulated signal in the
    old (incrementally-built) DB: it recurred (``day_count >= min_day_count``) OR
    it is connected to something (≥1 relation). A one-off, isolated mention
    (day_count=1, no edges) is dropped as low-value noise — the decision/context
    that referenced it is what carries meaning, and that is recovered separately.

    Filenames / ≤2-char names are excluded first via ``rule_keep_tech``. Short
    all-caps abbrevs (AWS vs SOUL) flow through the same signal test, so a real
    recurring abbrev survives while one-off noise does not.
    """
    if not rule_keep_tech(entity):
        return False
    if (entity.get("day_count") or 0) >= min_day_count:
        return True
    return _has_relation(old_conn, entity["id"])


_KEEP_PROMPT = (
    "你在清洗一个技术知识库。给你若干候选技术实体（name + 摘要）。"
    "判断每一个是否是【有意义的具体技术实体】（如芯片型号 PCA9685、模型名 glm-5.1、"
    "commit hash、具体库/框架/协议名），还是【噪声】（无意义短缩写、泛化词、裸文件名、"
    "拼写碎片）。只对有意义的回 keep，噪声回 drop。"
    '只输出一个 JSON 对象 {"1":"keep","2":"drop",...}，不要任何其它文字。'
)


def llm_keep_filter(rows: list[dict], *, api_url: str, api_key: str, model: str,
                    chunk: int = 25, timeout: float = 90.0) -> set[str]:
    """Entity ids an LLM judges MEANINGFUL (verdict 'keep').

    A failed / index-mismatched chunk is DROPPED (its items left out of the keep
    set), not kept — for a noise filter the safe default is drop, so an API blip
    never re-admits a batch of noise. Re-run when the model is healthy to recover
    any dropped real ones.
    """
    from knowledge_weaver._llm import classify_items
    verdicts = classify_items(
        rows, key=lambda e: e["id"],
        render=lambda e: f'{e.get("name")} — {(e.get("summary") or "")[:80]}',
        system_prompt=_KEEP_PROMPT, user_prefix="候选：\n",
        api_url=api_url, api_key=api_key, model=model, chunk=chunk, timeout=timeout)
    return {eid for eid, v in verdicts.items() if v.lower() == "keep"}
