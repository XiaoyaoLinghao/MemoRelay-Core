#!/usr/bin/env python3
"""One-off: create a project entity node for each registered project that lacks
one (so active_projects / knowledge_trace cover all registered projects).

Usage:
    python scripts/ensure_project_nodes.py [db_path]
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from knowledge_weaver.db import init_db, rebuild_fts  # noqa: E402
from knowledge_weaver.tools import ensure_registered_project_entities  # noqa: E402


def main() -> int:
    db = (sys.argv[1] if len(sys.argv) > 1
          else os.environ.get("KNOWLEDGE_WEAVER_DB_PATH",
                              os.path.expanduser("~/.openclaw/knowledge/knowledge.db")))
    if not os.path.exists(db):
        print(f"DB not found: {db}")
        return 1
    conn = init_db(db)
    created = ensure_registered_project_entities(conn)
    if created:
        rebuild_fts(conn)  # keep FTS in sync for the new nodes
    conn.close()
    print(f"Created {len(created)} project node(s): {created}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
