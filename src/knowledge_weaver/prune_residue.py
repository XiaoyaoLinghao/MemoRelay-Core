"""T2.1 — one-time prune of legacy-regex tech residue.

The old regex extractor mis-extracted bare version numbers (v2.9.0, v0.2.0,
2.1.140) and project-internal codes (P1-1, WI2, Track2, Phase1-3) as ``tech``
entities. They recur (high day_count), so the signal-gate keeps them; they are
not real standalone tech knowledge. This is a ONE-OFF data cleanup over an
existing DB (NOT a hot-path rule) — a targeted name match is acceptable here.

Conservative by design: ``residue_candidates`` only lists; pruning is explicit
and the caller backs up first (deletes are not auto-rolled-back).
"""
from __future__ import annotations

import re
from collections import Counter

from knowledge_weaver.db import clean_entity_indexes, delete_entity

# A bare version number: v0.2.0 / 0.2.2 / 2.1.140 (requires >=1 dot, all-numeric).
_VERSION_RE = re.compile(r"^v?\d+(?:\.\d+)+$")
# A project-internal code: P1-1 / P2-5 / WI2 / Track2 / Phase1-3. Deliberately
# NOT `T\d+` (collides with real models: T5, T6 transformers) and NOT bare `P\d+`
# (could be a real part) — only the dash/prefixed forms that no real tech uses.
_TAG_RE = re.compile(r"^(?:P\d+-\d+|WI\d+|Track\d+|Phase\d+(?:-\d+)?)$")


def residue_kind(name: str) -> str | None:
    n = (name or "").strip()
    if _VERSION_RE.match(n):
        return "version"
    if _TAG_RE.match(n):
        return "internal-tag"
    return None


def residue_candidates(conn) -> list[dict]:
    """tech entities whose NAME is a bare version number or internal code."""
    rows = conn.execute("SELECT id, name FROM entities WHERE type='tech'").fetchall()
    out = []
    for r in rows:
        kind = residue_kind(r["name"])
        if kind:
            out.append({"id": r["id"], "name": (r["name"] or "").strip(), "kind": kind})
    return out


def prune_residue(conn, *, dry_run: bool = False) -> dict:
    """Delete residue tech entities (and their indexes/edges). Back up first."""
    cands = residue_candidates(conn)
    if not dry_run:
        for c in cands:
            delete_entity(conn, c["id"], auto_commit=False)
            clean_entity_indexes(conn, c["id"])
        conn.commit()
    return {
        "candidates": len(cands),
        "by_kind": dict(Counter(c["kind"] for c in cands)),
        "names": [c["name"] for c in cands],
    }
