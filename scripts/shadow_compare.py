#!/usr/bin/env python3
"""Read-only: compare regex vs JSON-fact extraction over daily memory files.

Run this BEFORE switching KNOWLEDGE_WEAVER_EXTRACT_MODE=json, to judge whether
the structured handoff yields at-least-as-good entities as the regex path.
Writes nothing to the DB.

Usage:
    python scripts/shadow_compare.py <memory_dir_or_file> [more...]
"""
from __future__ import annotations

import glob
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from knowledge_weaver.json_facts import shadow_compare_file  # noqa: E402


def _iter_files(args):
    for a in args:
        if os.path.isdir(a):
            yield from sorted(glob.glob(os.path.join(a, "*.md")))
        elif os.path.isfile(a):
            yield a


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    files = list(_iter_files(sys.argv[1:]))
    if not files:
        print("No .md files found.")
        return 1

    with_block = tot_regex = tot_json = tot_both = 0
    for path in files:
        r = shadow_compare_file(path)
        if not r["has_json_block"]:
            continue
        with_block += 1
        tot_regex += r["regex_count"]
        tot_json += r["json_count"]
        tot_both += len(r["both"])
        print(f"\n=== {r['file']}  (regex={r['regex_count']} "
              f"json={r['json_count']} both={len(r['both'])}) ===")
        if r["regex_only"]:
            print("  regex-only (JSON missed):")
            for x in r["regex_only"]:
                print(f"    - {x}")
        if r["json_only"]:
            print("  json-only (regex missed / cleaner):")
            for x in r["json_only"]:
                print(f"    + {x}")

    print(f"\n--- {with_block} files with a JSON block | "
          f"regex {tot_regex}, json {tot_json}, both {tot_both} ---")
    if with_block == 0:
        print("No files had a `### 结构化事实` block yet "
              "(DMA side not emitting structured facts).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
