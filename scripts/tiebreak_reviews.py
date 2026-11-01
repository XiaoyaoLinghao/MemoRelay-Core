#!/usr/bin/env python3
"""T1 — clear the merge-review queue with an LLM tie-breaker. Asks 'same thing
or two different things?' per pending pair, auto-merges 'same', dismisses
'different', leaves genuine ambiguity pending. Core: knowledge_weaver.tiebreak.

Config (env): KW_REL_API_URL, KW_REL_API_KEY, KW_REL_MODEL
Usage (run on a COPY first; merges are rollback-logged but destructive):
    KW_REL_API_URL=... KW_REL_API_KEY=... KW_REL_MODEL=deepseek-v4-pro \
      python scripts/tiebreak_reviews.py <db_path> [--dry-run]
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from knowledge_weaver.db import count_pending_reviews, init_db  # noqa: E402
from knowledge_weaver.tiebreak import llm_judge_pairs, tiebreak_reviews  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="LLM tie-breaker over the merge-review queue")
    ap.add_argument("db_path")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--chunk", type=int, default=20)
    args = ap.parse_args()

    api_url = os.environ.get("KW_REL_API_URL")
    api_key = os.environ.get("KW_REL_API_KEY")
    model = os.environ.get("KW_REL_MODEL")
    if not args.dry_run and not (api_url and api_key and model):
        print("Set KW_REL_API_URL / KW_REL_API_KEY / KW_REL_MODEL (or --dry-run).")
        return 1

    conn = init_db(args.db_path)
    print(f"pending reviews before: {count_pending_reviews(conn)}")

    def judge(pairs):
        if args.dry_run:
            return {}
        return llm_judge_pairs(pairs, api_url=api_url, api_key=api_key,
                               model=model, chunk=args.chunk)

    r = tiebreak_reviews(conn, judge, dry_run=args.dry_run)
    print(f"candidates: {r['candidates']}")
    if args.dry_run:
        print("dry-run: not calling LLM / not writing.")
    else:
        print(f"merged: {r['merged']}  dismissed: {r['dismissed']}  kept pending: {r['kept']}")
        print(f"pending reviews after: {count_pending_reviews(conn)}")
        print("\nBack up the DB before running for real; merges are destructive (rollback via merge_log).")
    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
