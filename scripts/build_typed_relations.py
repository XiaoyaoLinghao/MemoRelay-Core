#!/usr/bin/env python3
"""W3.4 (one-off): re-type KW's generic RELATES_TO edges into a typed knowledge
graph via the LLM, or prune spurious edges. Core logic lives in
knowledge_weaver.typed_relations (also used for online typing in the CLI).

Config (env): KW_REL_API_URL, KW_REL_API_KEY, KW_REL_MODEL
Usage:
    KW_REL_API_URL=... KW_REL_API_KEY=... KW_REL_MODEL=deepseek-v4-pro \
      python scripts/build_typed_relations.py <db_path> [--dry-run] [--chunk 20]
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from knowledge_weaver.db import init_db  # noqa: E402
from knowledge_weaver.typed_relations import (  # noqa: E402
    llm_type_pairs,
    reset_llm_typed_edges,
    type_relations,
)


def main() -> int:
    ap = argparse.ArgumentParser(description="W3.4 type RELATES_TO edges into a knowledge graph")
    ap.add_argument("db_path")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--reset", action="store_true",
                    help="first revert previously LLM-typed edges back to RELATES_TO "
                         "(to re-type with an updated prompt)")
    ap.add_argument("--chunk", type=int, default=20)
    args = ap.parse_args()

    if args.reset and not args.dry_run:
        _c = init_db(args.db_path)
        n = reset_llm_typed_edges(_c)
        _c.close()
        print(f"reset {n} previously-typed edge(s) back to RELATES_TO")

    api_url = os.environ.get("KW_REL_API_URL")
    api_key = os.environ.get("KW_REL_API_KEY")
    model = os.environ.get("KW_REL_MODEL")
    if not args.dry_run and not (api_url and api_key and model):
        print("Set KW_REL_API_URL / KW_REL_API_KEY / KW_REL_MODEL (or --dry-run).")
        return 1

    def type_fn(cands):
        if args.dry_run:
            return {}
        return llm_type_pairs(cands, api_url=api_url, api_key=api_key,
                              model=model, chunk=args.chunk)

    r = type_relations(args.db_path, type_fn, dry_run=args.dry_run)
    print(f"candidates (RELATES_TO): {r['candidates']}")
    if args.dry_run:
        print("dry-run: not calling LLM / not writing.")
        return 0
    print(f"typed: {r['typed']}  pruned(none): {r['pruned']}")
    print(f"by type: {r['by_type']}")
    print(f"relations after: {r['relations_after']}")
    print("\nBack up the DB before running for real; pruning is destructive.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
