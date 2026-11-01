#!/usr/bin/env python3
"""Resolve the 7 cross-type name conflicts in KW entities.

For each conflict group, picks the entity with the most relations as canonical,
re-points all relations, deletes shadow entities, and cleans up indices.

Run as: python3 resolve_name_conflicts.py [--dry-run]
"""

import argparse
import sqlite3
import sys
from datetime import datetime

DB_PATH = "/home/openclaw/.openclaw/knowledge/knowledge.db"

# Each entry: (entity_name, canonical_id_excerpt, reason)
# canonical_id_excerpt is a unique substring to identify the canonical entity
MERGES = [
    # (canonical_id_substring, entity_name, reason)
    ("proj:knowledge_weaver", "Knowledge Weaver",
     "project type with 58 relations (vs decision 10, fact 1, risk 3) — project is the natural primary facet"),
    ("task:shanchurongyuknowledge_weaverj", "删除冗余Knowledge Weaver进程",
     "task type with 6 relations (vs decision 5) — task is more actionable for a todo item"),
    ("tech:ruview", "RuView",
     "tech type — both entities have 0 relations; tech is more specific for a software project"),
    ("proj:homebrain", "HomeBrain",
     "project type with 9 relations (vs tech 0) — project is the natural primary facet"),
    ("fact:codeforge", "CodeForge",
     "fact type with 8 relations (vs tech 0) — retains most knowledge connections"),
    ("proj:comp7940", "COMP7940",
     "project type with 4 relations (vs tech 0) — project is the natural primary facet"),
    ("proj:comp7240", "COMP7240",
     "project type with 1 relation (vs tech 0) — project is the natural primary facet"),
]


def run(dry_run: bool = False):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    cur = conn.cursor()

    now = datetime.now().isoformat(timespec="seconds")
    total_relations_moved = 0
    total_entities_removed = 0

    for canonical_id_start, name, reason in MERGES:
        # Find all entities with this name
        cur.execute("SELECT id, type, name FROM entities WHERE name = ?", (name,))
        all_entities = {r["id"]: r["type"] for r in cur.fetchall()}

        canonical_id = None
        for eid in all_entities:
            if eid.startswith(canonical_id_start):
                canonical_id = eid
                break

        if canonical_id is None:
            print(f"SKIP {name}: canonical id starting with '{canonical_id_start}' not found")
            continue

        delete_ids = [eid for eid in all_entities if eid != canonical_id]
        if not delete_ids:
            print(f"SKIP {name}: no shadow entities to merge")
            continue

        print(f"\n{'[DRY-RUN] ' if dry_run else ''}Merging '{name}':")
        print(f"  Keep:   [{all_entities[canonical_id]:10}] {canonical_id}")
        for did in delete_ids:
            print(f"  Delete: [{all_entities[did]:10}] {did}")
        print(f"  Reason: {reason}")

        if dry_run:
            # Count what would move
            for did in delete_ids:
                cur.execute("SELECT COUNT(*) as cnt FROM relations WHERE from_entity = ?", (did,))
                total_relations_moved += cur.fetchone()["cnt"]
                cur.execute("SELECT COUNT(*) as cnt FROM relations WHERE to_entity = ?", (did,))
                total_relations_moved += cur.fetchone()["cnt"]
            total_entities_removed += len(delete_ids)
            continue

        # ── Execute merge ──
        for did in delete_ids:
            # 1. Move relations: from_entity → canonical
            cur.execute("UPDATE relations SET from_entity = ? WHERE from_entity = ?",
                        (canonical_id, did))
            moved = cur.rowcount
            if moved:
                print(f"  Moved {moved} relations (from_entity) from {did[:30]}...")
                total_relations_moved += moved

            # 2. Move relations: to_entity → canonical
            cur.execute("UPDATE relations SET to_entity = ? WHERE to_entity = ?",
                        (canonical_id, did))
            moved = cur.rowcount
            if moved:
                print(f"  Moved {moved} relations (to_entity) from {did[:30]}...")
                total_relations_moved += moved

            # 3. Move access_log references to canonical
            cur.execute("UPDATE access_log SET entity_id = ? WHERE entity_id = ?",
                        (canonical_id, did))

            # 4. Clean FTS index
            cur.execute("DELETE FROM entity_fts WHERE entity_id = ?", (did,))

            # 5. Clean old embedding table
            cur.execute("DELETE FROM entity_vectors WHERE entity_id = ?", (did,))

            # 6. Clean vec0 embedding table (if loaded)
            try:
                cur.execute("DELETE FROM entity_vec WHERE entity_id = ?", (did,))
            except sqlite3.Error:
                pass

            # 7. Delete the shadow entity
            cur.execute("DELETE FROM entities WHERE id = ?", (did,))

            # 8. Log the merge
            cur.execute(
                "INSERT INTO merge_log (merged_from_id, merged_into_id, reason) VALUES (?, ?, ?)",
                (did, canonical_id, reason),
            )

            total_entities_removed += 1

        # 9. Update canonical entity importance
        cur.execute(
            "UPDATE entities SET importance = MAX(importance, 0.5) WHERE id = ?",
            (canonical_id,),
        )

        # 10. Remove stale merge_review entries for this group
        for eid in all_entities:
            cur.execute(
                "DELETE FROM merge_review WHERE new_entity_id = ?",
                (eid,),
            )

    if dry_run:
        print(f"\n[dry-run] Would move {total_relations_moved} relations, remove {total_entities_removed} shadow entities")
    else:
        conn.commit()
        print(f"\nDone. Moved {total_relations_moved} relations, removed {total_entities_removed} shadow entities.")
        print("Don't forget to re-run consolidate and clean_and_rescore to rebuild FTS/embeddings.")

    conn.close()


def main():
    p = argparse.ArgumentParser(description="Resolve KW cross-type name conflicts")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    run(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
