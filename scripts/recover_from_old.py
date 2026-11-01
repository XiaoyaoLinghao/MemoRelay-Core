#!/usr/bin/env python3
"""Recover entities lost in the historical rebuild by porting them from the
pre-swap DB (carrying their accumulated day_count + stored vector). Core logic
lives in knowledge_weaver.recover.

Decisions are 0% noise -> ported wholesale. Tech is ~80% noise -> filtered by a
coarse rule, or (better) an LLM keep/drop classifier when KW_FACTS_API_* is set.

Usage:
    # decisions only (default), on a COPY of the live DB:
    python scripts/recover_from_old.py <old_db> <new_db> [--dry-run]
    # include tech with rule filter:
    python scripts/recover_from_old.py <old> <new> --types decision,tech
    # include tech with LLM keep/drop:
    KW_FACTS_API_URL=... KW_FACTS_API_KEY=... KW_FACTS_MODEL=deepseek-v4-pro \
      python scripts/recover_from_old.py <old> <new> --types decision,tech --classify-tech

After this, run: semantic_dedupe -> build_typed_relations -> rebuild_fts.
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from knowledge_weaver.db import init_db  # noqa: E402
from knowledge_weaver.recover import (  # noqa: E402
    llm_keep_filter,
    missing_entities,
    port_entities,
    rule_keep_tech,
    signal_keep_tech,
)


def main() -> int:
    ap = argparse.ArgumentParser(description="Recover lost entities from the pre-swap DB")
    ap.add_argument("old_db", help="pre-swap DB (read-only source)")
    ap.add_argument("new_db", help="current DB (a COPY; will be written to)")
    ap.add_argument("--types", default="decision",
                    help="comma list of entity types to recover (default: decision)")
    ap.add_argument("--no-signal-gate", action="store_true",
                    help="disable the tech signal-gate (day_count>=2 OR connected); "
                         "fall back to the lenient name-only rule")
    ap.add_argument("--classify-tech", action="store_true",
                    help="LLM keep/drop to RESCUE meaningful tech that fails the signal "
                         "gate (low-signal one-offs); needs KW_FACTS_API_*")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    types = [t.strip() for t in args.types.split(",") if t.strip()]
    old = sqlite3.connect(args.old_db)
    old.row_factory = sqlite3.Row
    new = init_db(args.new_db)

    total = {"ported": 0, "skipped": 0, "with_vector": 0}
    for t in types:
        rows = missing_entities(old, new, [t])
        keep_fn = None
        if t == "tech":
            if args.no_signal_gate:
                # lenient: name-only rule (kept for comparison / fallback)
                base_keep = rule_keep_tech
            else:
                # PRIMARY: signal-gate — recurred (day_count>=2) OR connected.
                # One-off isolated mentions are dropped as low-value noise.
                base_keep = lambda e: signal_keep_tech(e, old)  # noqa: E731
            if args.classify_tech:
                url = os.environ.get("KW_FACTS_API_URL")
                key = os.environ.get("KW_FACTS_API_KEY")
                model = os.environ.get("KW_FACTS_MODEL")
                if not (url and key and model):
                    print("--classify-tech needs KW_FACTS_API_URL/_KEY/_MODEL"); return 1
                # rescue: LLM judges the name-valid candidates the gate dropped
                remainder = [e for e in rows if rule_keep_tech(e) and not base_keep(e)]
                rescued = llm_keep_filter(remainder, api_url=url, api_key=key, model=model)
                keep_fn = lambda e: base_keep(e) or e["id"] in rescued  # noqa: E731
            else:
                keep_fn = base_keep
        r = port_entities(old, new, rows, keep_fn=keep_fn, dry_run=args.dry_run)
        print(f"[{t}] missing={r['candidates']} ported={r['ported']} "
              f"skipped={r['skipped']} with_vector={r['with_vector']}")
        for k in total:
            total[k] += r[k]

    old.close()
    new.close()
    print(f"\nTOTAL ported={total['ported']} skipped={total['skipped']} "
          f"with_vector={total['with_vector']}")
    if args.dry_run:
        print("dry-run: nothing written.")
    else:
        print("Next: semantic_dedupe -> build_typed_relations -> rebuild_fts. "
              "Back up new_db before running for real.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
