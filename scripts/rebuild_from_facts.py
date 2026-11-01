#!/usr/bin/env python3
"""W3: rebuild a CANDIDATE KW DB from daily memory files, then diff it against
the current production DB — WITHOUT touching production.

Reprocesses every memory file (date order) through the same pipeline the live
consolidation uses, but into a SEPARATE candidate DB, with the structured-fact
(JSON) path enabled. Files that carry a ``### 结构化事实`` block use the clean
JSON facts; files without one fall back to regex (so historical days still
contribute until they are backfilled with facts).

Review the diff (what the rebuild gains vs the current DB, and — importantly —
what it would LOSE), then swap the candidate DB in manually only if it looks
right. This script never writes to the production DB.

Usage:
    python scripts/rebuild_from_facts.py <memory_dir> \
        [--current ~/.openclaw/knowledge/knowledge.db] \
        [--candidate /tmp/kw_candidate.db] \
        [--regex]            # rebuild with the OLD regex path (for A/B baseline)
        [--sample N]         # show N sample ids per diff bucket (default 25)
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from knowledge_weaver.db import init_db, list_all_entities  # noqa: E402
from knowledge_weaver.pipeline import run_consolidation  # noqa: E402


def _entities_by_id(db_path: str) -> dict:
    if not db_path or not os.path.exists(db_path):
        return {}
    conn = init_db(db_path)
    try:
        return {r["id"]: r for r in list_all_entities(conn)}
    finally:
        conn.close()


def rebuild_and_diff(memory_dir: str, candidate_db: str, current_db: str | None = None,
                     json_mode: bool = True, embedder=None) -> dict:
    """Build a fresh candidate DB from memory_dir and diff it against current_db.

    Returns a structured report. Never touches current_db (read-only)."""
    if os.path.exists(candidate_db):
        os.remove(candidate_db)

    prev_mode = os.environ.get("KNOWLEDGE_WEAVER_EXTRACT_MODE")
    os.environ["KNOWLEDGE_WEAVER_EXTRACT_MODE"] = "json" if json_mode else "regex"
    try:
        result = run_consolidation(candidate_db, memory_dir=memory_dir, embedder=embedder)
    finally:
        if prev_mode is None:
            os.environ.pop("KNOWLEDGE_WEAVER_EXTRACT_MODE", None)
        else:
            os.environ["KNOWLEDGE_WEAVER_EXTRACT_MODE"] = prev_mode

    cand = _entities_by_id(candidate_db)
    cur = _entities_by_id(current_db) if current_db else {}
    cand_ids, cur_ids = set(cand), set(cur)

    def by_type(d: dict) -> dict:
        return dict(Counter(r["type"] for r in d.values()))

    def label(ids: set, src: dict) -> list:
        return sorted(f"{i}  ({src[i]['name']})" for i in ids)

    return {
        "files_processed": result.files_processed,
        "status": result.status,
        "candidate_total": len(cand),
        "current_total": len(cur),
        "candidate_by_type": by_type(cand),
        "current_by_type": by_type(cur),
        "both": len(cand_ids & cur_ids),
        "only_in_current": label(cur_ids - cand_ids, cur),     # what rebuild LOSES
        "only_in_candidate": label(cand_ids - cur_ids, cand),  # what rebuild GAINS
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="W3 candidate rebuild + diff (read-only on prod)")
    ap.add_argument("memory_dir")
    ap.add_argument("--current", default=os.path.expanduser("~/.openclaw/knowledge/knowledge.db"))
    ap.add_argument("--candidate", default="/tmp/kw_candidate.db")
    ap.add_argument("--regex", action="store_true", help="rebuild via old regex path")
    ap.add_argument("--sample", type=int, default=25)
    args = ap.parse_args()

    embedder = None
    try:
        from knowledge_weaver.embedder import get_embedder
        embedder = get_embedder()
    except Exception:
        pass
    if embedder is None:
        print("WARNING: no embedder configured — resolver vector dedup is limited; "
              "set EMBEDDING_* for a faithful rebuild.\n")

    r = rebuild_and_diff(args.memory_dir, args.candidate, current_db=args.current,
                         json_mode=not args.regex, embedder=embedder)

    print(f"mode={'regex' if args.regex else 'json'}  files={r['files_processed']}  "
          f"status={r['status']}")
    print(f"candidate DB: {args.candidate}  (NOT swapped in)")
    print(f"current:   {r['current_total']:5d}  {r['current_by_type']}")
    print(f"candidate: {r['candidate_total']:5d}  {r['candidate_by_type']}")
    print(f"in both:   {r['both']}\n")

    lost, gained = r["only_in_current"], r["only_in_candidate"]
    print(f"--- only in CURRENT (rebuild would LOSE {len(lost)}) — eyeball: noise or real? ---")
    for x in lost[:args.sample]:
        print(f"  - {x}")
    if len(lost) > args.sample:
        print(f"  ... +{len(lost) - args.sample} more")
    print(f"\n--- only in CANDIDATE (rebuild GAINS {len(gained)}) — should be clean ---")
    for x in gained[:args.sample]:
        print(f"  + {x}")
    if len(gained) > args.sample:
        print(f"  ... +{len(gained) - args.sample} more")

    print("\nTo adopt: back up current, then `cp` the candidate DB over it and restart. "
          "Do NOT swap unless the LOSES are noise and the GAINS look right.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
