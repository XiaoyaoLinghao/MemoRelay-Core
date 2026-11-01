#!/usr/bin/env python3
"""One-off: re-index entity_fts with jieba Chinese segmentation.

Run once after deploying the jieba-FTS change so existing entities become
searchable by Chinese keywords (the old index used unicode61, which does not
segment Chinese).

Usage:
    python scripts/rebuild_fts.py [db_path]
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from knowledge_weaver.db import init_db, rebuild_fts  # noqa: E402


def main() -> int:
    db = (sys.argv[1] if len(sys.argv) > 1
          else os.environ.get("KNOWLEDGE_WEAVER_DB_PATH",
                              os.path.expanduser("~/.openclaw/knowledge/knowledge.db")))
    if not os.path.exists(db):
        print(f"DB not found: {db}")
        return 1
    conn = init_db(db)
    n = rebuild_fts(conn)
    conn.close()
    print(f"Re-indexed entity_fts with jieba for {n} entities in {db}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
