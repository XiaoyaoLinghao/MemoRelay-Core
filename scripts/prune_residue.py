#!/usr/bin/env python3
"""T2.1 — prune legacy-regex tech residue (bare version numbers / internal codes
mis-extracted as tech). One-off data cleanup. Core: knowledge_weaver.prune_residue.

Usage (run on a COPY first; deletes are NOT auto-rolled-back — back up the DB):
    python scripts/prune_residue.py <db_path> --dry-run     # list candidates
    python scripts/prune_residue.py <db_path>               # prune them
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from knowledge_weaver.db import init_db  # noqa: E402
from knowledge_weaver.prune_residue import prune_residue, residue_candidates  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Prune version-number / internal-tag tech residue")
    ap.add_argument("db_path")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    conn = init_db(args.db_path)
    cands = residue_candidates(conn)
    print(f"residue candidates: {len(cands)}")
    for c in cands:
        print(f"  [{c['kind']:>12}] {c['name']}")
    if args.dry_run:
        conn.close()
        print("\ndry-run: nothing deleted.")
        return 0
    r = prune_residue(conn)
    conn.close()
    print(f"\npruned {r['candidates']} ({r['by_kind']}). Back up before running for real.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
