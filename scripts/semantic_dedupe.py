#!/usr/bin/env python3
"""Second-pass semantic dedup over an existing KW DB (one-off cleanup). Merges
same-type vector-similar entities the rebuild left as distinct ids, restoring
cross-file day_count accumulation. Core logic: knowledge_weaver.semantic_dedupe.

Usage (run on a COPY; back up first — merges are rollback-logged but destructive):
    python scripts/semantic_dedupe.py <db_path> [--dry-run] [--review-only]
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from knowledge_weaver.db import init_db  # noqa: E402
from knowledge_weaver.semantic_dedupe import semantic_dedupe  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Second-pass semantic dedup")
    ap.add_argument("db_path")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--review-only", action="store_true",
                    help="queue all candidates for review instead of auto-merging HIGH")
    args = ap.parse_args()

    conn = init_db(args.db_path)
    r = semantic_dedupe(conn, dry_run=args.dry_run, auto_merge=not args.review_only)
    conn.close()
    print(f"candidates: high={r['candidates_high']} mid={r['candidates_mid']}")
    print(f"merged={r['merged']} queued={r['queued']}")
    if args.dry_run:
        print("dry-run: nothing written.")
    else:
        print("Resolve queued pairs via kw_review_pending / kw_resolve. "
              "Then run build_typed_relations + rebuild_fts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
