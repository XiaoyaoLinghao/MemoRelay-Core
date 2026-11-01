"""Entity resolution — vector-first semantic dedup with three-band confidence.

Upgrades the legacy name-first ``_find_similar_entity`` (pipeline.py): that
matched on ``SequenceMatcher`` name ratio first and only consulted embeddings
for the 0.70-0.85 "borderline" band — which *structurally misses* cross-language
/ divergent aliases (e.g. "HomeBrain" vs "家庭大脑", whose name ratio ≈ 0, never
reaching the embedding pass).

This resolver:

1. **Authoritative** — registered aliases (from the MEMORY.md registry)
   deterministically normalize to the canonical entity id, whether or not the
   canonical entity already exists.
2. **Vector-first recall** — fts + vector candidates of the *same type*, scored
   by cosine against the candidate's stored embedding (consistent metric).
3. **Three bands**
   - merge   : name ratio >= NAME_HIGH  OR  cosine >= RESOLVE_HIGH  -> auto-merge
   - review  : RESOLVE_MID <= cosine < RESOLVE_HIGH                 -> review queue
   - distinct: otherwise                                            -> keep separate

Thresholds are env-configurable; calibrate with
``scripts/calibrate_resolution.py`` against a real DB.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from difflib import SequenceMatcher
import re
from typing import Optional

from knowledge_weaver.db import (
    _cosine,
    get_entity_vector,
    search_entities_fts,
    search_entity_vectors,
)


def _env_float(name: str, default: float) -> float:
    try:
        raw = os.environ.get(name)
        return float(raw) if raw else default
    except (TypeError, ValueError):
        return default


# Cosine >= HIGH -> auto-merge; [MID, HIGH) -> review; name ratio >= NAME_HIGH -> auto-merge.
RESOLVE_HIGH = _env_float("KNOWLEDGE_WEAVER_RESOLVE_HIGH", 0.90)
RESOLVE_MID = _env_float("KNOWLEDGE_WEAVER_RESOLVE_MID", 0.82)
NAME_HIGH = _env_float("KNOWLEDGE_WEAVER_RESOLVE_NAME_HIGH", 0.85)

# Structured identifiers (version numbers, course/product codes, paths, filenames)
# get spuriously high bge-m3 cosine even when they denote *different* things
# (real-data finding: v0.2.0 vs v2.9.0 cosine 0.90; COMP7940 vs COMP7240 = 1.0).
# For these, a vector-only match is unreliable -> demote to review, never auto-merge.
# (True duplicates of an identifier share the same name -> same id -> exact-id path,
# so two *different* identifier names reaching the resolver are almost always distinct.)
_IDENTIFIER_RE = re.compile(
    r"^v?\d+(?:[._]\d+)+$"      # version: v0.2.0 / 1.2 / 0.2.0
    r"|^[A-Za-z]{2,}\d{2,}$"    # code: COMP7940 / ABC123
    r"|[/\\]"                   # path separator
    r"|\.\w{1,5}$"              # file extension: .py / .md / .json
)


def _merge_unreliable(name: str) -> bool:
    """True if a name is too short / identifier-like for either cosine OR name
    similarity to be trusted for auto-merge.

    Both signals fail on structured identifiers: cosine (bge-m3 over-similarity)
    AND SequenceMatcher (COMP7940 vs COMP7240 ratio 0.875 >= NAME_HIGH, yet they
    are different course codes). Such cases are demoted to human review.

    ``len <= 3`` is a generalizing guard (kept). ``_IDENTIFIER_RE`` is the
    LLM-unavailable *fallback* for single-name identifiers; the primary,
    enumeration-free guard for identifier *pairs* is ``_digit_variant_pair`` —
    so a brand-new identifier format needs no new regex branch (T1).
    """
    n = (name or "").strip()
    return len(n) <= 3 or bool(_IDENTIFIER_RE.search(n))


def _digit_variant_pair(a: str, b: str) -> bool:
    """True if two names are identical except in their DIGIT components — i.e.
    different instances of the same template, not duplicates.

    Generalizes the ``_IDENTIFIER_RE`` enumeration: mask every digit run to ``#``
    and compare skeletons. ``v0.2.0``/``v2.9.0`` → ``v#.#.#`` (match → distinct);
    ``COMP7940``/``COMP7240`` → ``COMP#`` (match → distinct); ``8080``/``8081``
    → ``#`` (match → distinct). Also catches a short alphanumeric TAIL difference
    around the digits — ``GPT-4``/``GPT-4o`` (digit-stripped ``GPT-`` vs ``GPT-o``,
    differ by ≤2 trailing chars) → distinct. No format is enumerated, so a new
    numbered-identifier shape is covered with zero new rules. Real aliases
    (``HomeBrain``/``家庭大脑``) have differing skeletons → not flagged. Borderline
    cases only demote to review — the LLM tie-breaker makes the final call.
    """
    sa, sb = (a or "").strip(), (b or "").strip()
    if not sa or not sb or sa == sb:
        return False
    if not any(c.isdigit() for c in sa + sb):
        return False
    if re.sub(r"\d+", "#", sa) == re.sub(r"\d+", "#", sb):
        return True  # same template, differ only in digit runs
    # letter-suffixed variant: strip digits entirely; if the remainders differ only
    # by a short trailing add (GPT-4 vs GPT-4o), they are distinct instances.
    da, db = re.sub(r"\d+", "", sa), re.sub(r"\d+", "", sb)
    if da and db and da != db:
        lo, hi = sorted((da, db), key=len)
        if hi.startswith(lo) and len(hi) - len(lo) <= 2:
            return True
    return False


CANDIDATE_K = 10


@dataclass
class Resolution:
    """Outcome of resolving a freshly-extracted entity against existing ones."""

    action: str  # "normalize" | "merge" | "review" | "distinct"
    into_id: Optional[str] = None
    score: float = 0.0
    reason: str = ""


def load_alias_map() -> dict[str, str]:
    """Map each registered alias's entity-id -> canonical entity-id.

    Honors KNOWLEDGE_WEAVER_REGISTRY_PATH; never raises (empty on any failure).
    """
    try:
        from knowledge_weaver.registry import alias_to_canonical_map, load_registry

        path = os.environ.get(
            "KNOWLEDGE_WEAVER_REGISTRY_PATH", "/home/openclaw/.openclaw/workspace/MEMORY.md"
        )
        return alias_to_canonical_map(load_registry(path))
    except Exception:
        return {}


def resolve_entity(
    conn,
    *,
    new_id: str,
    entity_type: str,
    name: str,
    summary: str,
    embedder=None,
    alias_to_canonical: Optional[dict[str, str]] = None,
) -> Resolution:
    """Decide how a freshly-extracted entity relates to existing entities.

    ``new_id`` is the slug id the extractor assigned (only called when no exact
    id match exists). Returns a :class:`Resolution`; the caller acts on it.
    """
    # 1. Authoritative: registered alias -> normalize to canonical id.
    if alias_to_canonical:
        canonical = alias_to_canonical.get(new_id)
        if canonical and canonical != new_id:
            return Resolution("normalize", into_id=canonical, score=1.0,
                              reason="registered-alias")

    # 2. Candidate recall (same type only): fts + vector.
    candidates: dict[str, object] = {}
    try:
        for row in search_entities_fts(conn, name, limit=CANDIDATE_K):
            if row["id"] != new_id and row["type"] == entity_type:
                candidates[row["id"]] = row
    except Exception:
        pass

    query_vec = None
    if embedder is not None and summary:
        try:
            query_vec = embedder.embed(summary[:500]) or None
        except Exception:
            query_vec = None
    if query_vec:
        try:
            for row in search_entity_vectors(conn, query_vec, limit=CANDIDATE_K):
                if row["id"] != new_id and row["type"] == entity_type:
                    candidates.setdefault(row["id"], row)
        except Exception:
            pass

    if not candidates:
        return Resolution("distinct")

    # 3. Score each candidate: name ratio + embedding cosine (consistent metric).
    merge_id: Optional[str] = None
    merge_score, merge_reason = 0.0, ""
    review_id: Optional[str] = None
    review_score = 0.0
    for cid, row in candidates.items():
        cand_name = (row["name"] or "")  # type: ignore[index]
        name_ratio = SequenceMatcher(None, name.lower(), cand_name.lower()).ratio()
        cos = 0.0
        if query_vec:
            cvec = get_entity_vector(conn, cid)
            if cvec:
                cos = _cosine(query_vec, cvec)

        strong_name = name_ratio >= NAME_HIGH
        strong_vec = cos >= RESOLVE_HIGH
        unreliable = (_merge_unreliable(name) or _merge_unreliable(cand_name)
                      or _digit_variant_pair(name, cand_name))

        if (strong_name or strong_vec) and not unreliable:
            s = max(name_ratio, cos)
            if s > merge_score:
                merge_id, merge_score = cid, s
                merge_reason = "name" if strong_name else "vector"
        elif strong_name or strong_vec:
            # Strong signal but identifier-like / too short -> human review,
            # never auto-merge (both cosine and name ratio are unreliable here).
            s = max(name_ratio, cos)
            if s > review_score:
                review_id, review_score = cid, s
        elif cos >= RESOLVE_MID:
            if cos > review_score:
                review_id, review_score = cid, cos

    if merge_id is not None:
        return Resolution("merge", into_id=merge_id, score=round(merge_score, 4),
                          reason=f"auto:{merge_reason}")
    if review_id is not None:
        return Resolution("review", into_id=review_id, score=round(review_score, 4),
                          reason="review:vector")
    return Resolution("distinct")
