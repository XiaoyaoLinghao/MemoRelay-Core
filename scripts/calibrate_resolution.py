#!/usr/bin/env python3
"""Read-only: calibrate W1 entity-resolution thresholds against a real KW DB.

For each entity type, computes every entity's highest cosine similarity to any
OTHER same-type entity, then prints the distribution + the closest pairs. Real
duplicates cluster near the top; the gap below them is a good MID floor. Use
this to pick KNOWLEDGE_WEAVER_RESOLVE_HIGH / _MID instead of guessing.

Read-only: issues only SELECTs; never writes.

Usage:
    python scripts/calibrate_resolution.py [DB_PATH]
    # defaults to $KNOWLEDGE_WEAVER_DB_PATH or ~/.openclaw/knowledge/knowledge.db
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from knowledge_weaver.db import _cosine, init_db  # noqa: E402


def calibrate(db_path: str, top_pairs: int = 20) -> dict:
    """Compute nearest-neighbour cosine stats per type. Returns a summary dict."""
    conn = init_db(db_path)
    try:
        rows = conn.execute(
            "SELECT e.id, e.type, e.name, v.embedding FROM entities e "
            "JOIN entity_vectors v ON v.entity_id = e.id"
        ).fetchall()
    finally:
        conn.close()

    by_type: dict[str, list] = defaultdict(list)
    for r in rows:
        try:
            by_type[r["type"]].append((r["id"], r["name"], json.loads(r["embedding"])))
        except Exception:
            pass

    nn_scores: list[float] = []
    pairs: list[tuple] = []
    for etype, items in by_type.items():
        for i, (_ida, na, va) in enumerate(items):
            best, bestname = 0.0, None
            for j, (_idb, nb, vb) in enumerate(items):
                if i == j:
                    continue
                c = _cosine(va, vb)
                if c > best:
                    best, bestname = c, nb
            if bestname is not None:
                nn_scores.append(best)
                pairs.append((best, etype, na, bestname))

    nn_scores.sort()
    pairs.sort(reverse=True)

    def pct(p: int) -> float:
        if not nn_scores:
            return 0.0
        return nn_scores[min(len(nn_scores) - 1, int(p / 100 * len(nn_scores)))]

    # dedup symmetric pairs for display
    seen: set = set()
    top: list[tuple] = []
    for cos, etype, a, b in pairs:
        key = frozenset((a, b))
        if key in seen:
            continue
        seen.add(key)
        top.append((round(cos, 4), etype, a, b))
        if len(top) >= top_pairs:
            break

    return {
        "db_path": db_path,
        "vectors": len(rows),
        "by_type": {t: len(v) for t, v in by_type.items()},
        "percentiles": {f"p{p}": round(pct(p), 4) for p in (50, 75, 90, 95, 99)},
        "max": round(nn_scores[-1], 4) if nn_scores else 0.0,
        "top_pairs": top,
    }


def main() -> int:
    db_path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else os.environ.get(
            "KNOWLEDGE_WEAVER_DB_PATH",
            os.path.expanduser("~/.openclaw/knowledge/knowledge.db"),
        )
    )
    if not os.path.exists(db_path):
        print(f"DB not found: {db_path}")
        return 1

    s = calibrate(db_path)
    print(f"DB: {s['db_path']}")
    print(f"Entities with vectors: {s['vectors']}  | by type: {s['by_type']}\n")
    if not s["top_pairs"]:
        print("Not enough same-type entities with vectors to calibrate.")
        return 0
    print("Nearest-neighbour cosine distribution (same type):")
    for k, v in s["percentiles"].items():
        print(f"  {k}: {v:.3f}")
    print(f"  max: {s['max']:.3f}\n")
    print("Closest same-type pairs (eyeball real dups vs coincidences):")
    for cos, etype, a, b in s["top_pairs"]:
        print(f"  {cos:.3f}  [{etype}]  {a!r}  ~  {b!r}")
    print(
        "\nSuggestion: set RESOLVE_HIGH just below where real dups stop and "
        "RESOLVE_MID at the coincidence floor.\nCurrent defaults: HIGH=0.90, MID=0.82."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
