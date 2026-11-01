#!/usr/bin/env python3
"""Clean noisy entities and rescore importance for Knowledge Weaver DB."""

from __future__ import annotations

import argparse
import os
import re
import sqlite3
import sys
from collections import Counter
from datetime import date, timedelta

# Allow imports from project src
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from knowledge_weaver.db import PROJECT_GRACE_DAYS, PROJECT_MIN_DAYS
from knowledge_weaver.registry import load_registered_slugs
from knowledge_weaver.scorer import score_entity
from knowledge_weaver.extractor import (
    _TIMESTAMP_LOG_RE,
    _BRACKET_TS_RE,
    _OPS_LOG_KEYWORDS_RE,
    _TECH_COMMON_WORDS,
    _STRUCTURAL_TECH_RE,
)

# Weak RELATES_TO evidences — old-style section-title-based evidence
# that the semantic linker no longer produces
_WEAK_RELATES_TO_EVIDENCES = (
    # DMA 8 standard categories
    "核心要点", "决策与结论", "已完成事项", "待办与计划",
    "用户偏好与习惯", "技术/项目要点", "风险与注意事项", "创意与想法",
    # Non-standard section titles discovered in historical data
    "技能管理",
    "Agent 连通性测试与 Auto-Announce Bug 排查",
)


def _get_stats(conn: sqlite3.Connection) -> dict:
    """Collect current database statistics."""
    stats: dict = {}

    # Total entity count
    stats["total_entities"] = conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]

    # Entity count by type
    rows = conn.execute("SELECT type, COUNT(*) FROM entities GROUP BY type").fetchall()
    stats["entities_by_type"] = dict(rows)

    # Total relations
    stats["total_relations"] = conn.execute("SELECT COUNT(*) FROM relations").fetchone()[0]

    # Total vectors
    try:
        stats["total_vectors"] = conn.execute("SELECT COUNT(*) FROM entity_vectors").fetchone()[0]
    except sqlite3.OperationalError:
        stats["total_vectors"] = 0

    return stats


def _print_stats(label: str, stats: dict) -> None:
    """Print database statistics."""
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")
    print(f"  Total entities:  {stats['total_entities']}")
    print(f"  By type:")
    for etype, count in sorted(stats["entities_by_type"].items()):
        print(f"    {etype}: {count}")
    print(f"  Total relations: {stats['total_relations']}")
    print(f"  Total vectors:   {stats['total_vectors']}")


def _find_noisy_entity_ids(conn: sqlite3.Connection) -> dict[str, list[str]]:
    """Find IDs of noisy entities grouped by reason."""
    noisy: dict[str, list[str]] = {
        "timestamp_fact": [],
        "common_tech": [],
        "structural_tech": [],
        "provisional_stale": [],
    }

    # Step A0: Provisional stale projects — low day_count + beyond grace period
    # Exclude registered projects (bypass the recurrence gate)
    reg_slugs = load_registered_slugs()
    grace_cutoff = (date.today() - timedelta(days=PROJECT_GRACE_DAYS)).isoformat()
    if reg_slugs:
        slug_placeholders = ",".join("?" * len(reg_slugs))
        rows = conn.execute(
            f"SELECT id FROM entities WHERE type='project' AND day_count < ? AND last_seen < ? AND id NOT IN ({slug_placeholders})",
            (PROJECT_MIN_DAYS, grace_cutoff) + tuple(sorted(reg_slugs)),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT id FROM entities WHERE type='project' AND day_count < ? AND last_seen < ?",
            (PROJECT_MIN_DAYS, grace_cutoff),
        ).fetchall()
    noisy["provisional_stale"] = [r["id"] for r in rows]

    # Step A1: Timestamp fact entities (multiple patterns)
    rows = conn.execute("SELECT id, name FROM entities WHERE type = 'fact'").fetchall()
    for row in rows:
        name = row["name"]
        if _TIMESTAMP_LOG_RE.match(name) or _BRACKET_TS_RE.match(name):
            noisy["timestamp_fact"].append(row["id"])
        elif _OPS_LOG_KEYWORDS_RE.search(name) and len(name) < 120:
            # Short operational log items: cron configs, backup status, daemon notifications
            noisy["timestamp_fact"].append(row["id"])

    # Step A2: Common tech word entities
    common_words_upper = [w.upper() for w in _TECH_COMMON_WORDS]
    if common_words_upper:
        placeholders = ",".join("?" * len(common_words_upper))
        rows = conn.execute(
            f"SELECT id FROM entities WHERE type = 'tech' AND UPPER(name) IN ({placeholders})",
            common_words_upper,
        ).fetchall()
        noisy["common_tech"] = [row["id"] for row in rows]

    # Step A3: Structural tech entities
    rows = conn.execute("SELECT id, name FROM entities WHERE type = 'tech'").fetchall()
    for row in rows:
        if _STRUCTURAL_TECH_RE.match(row["name"]):
            noisy["structural_tech"].append(row["id"])

    return noisy


def _delete_entities_cascade(conn: sqlite3.Connection, entity_ids: list[str]) -> int:
    """Delete entities and their related data in correct order. Returns deleted relation count."""
    if not entity_ids:
        return 0

    placeholders = ",".join("?" * len(entity_ids))

    # Delete relations referencing these entities
    rel_count = conn.execute(
        f"SELECT COUNT(*) FROM relations WHERE from_entity IN ({placeholders}) OR to_entity IN ({placeholders})",
        entity_ids + entity_ids,
    ).fetchone()[0]
    conn.execute(
        f"DELETE FROM relations WHERE from_entity IN ({placeholders}) OR to_entity IN ({placeholders})",
        entity_ids + entity_ids,
    )

    # Delete entity vectors (plain table)
    try:
        conn.execute(
            f"DELETE FROM entity_vectors WHERE entity_id IN ({placeholders})",
            entity_ids,
        )
    except sqlite3.OperationalError:
        pass

    # Sync FTS index
    try:
        conn.execute(
            f"DELETE FROM entity_fts WHERE entity_id IN ({placeholders})",
            entity_ids,
        )
    except sqlite3.OperationalError:
        pass

    # Sync sqlite-vec virtual table (best-effort: extension may not be loaded)
    try:
        conn.enable_load_extension(True)
        import sqlite_vec  # noqa: F811
        sqlite_vec.load(conn)
        conn.execute(
            f"DELETE FROM entity_vec WHERE entity_id IN ({placeholders})",
            entity_ids,
        )
    except Exception:
        pass

    # access_log also FK-references entities(id) (schema). With PRAGMA
    # foreign_keys=ON, NOT clearing it makes `DELETE FROM entities` fail for any
    # entity that has been accessed — the cause of the "FK skip" that left
    # accessed-but-noisy entities un-prunable. Clear it before the entity delete.
    try:
        conn.execute(
            f"DELETE FROM access_log WHERE entity_id IN ({placeholders})",
            entity_ids,
        )
    except sqlite3.OperationalError:
        pass

    # Delete entities
    conn.execute(
        f"DELETE FROM entities WHERE id IN ({placeholders})",
        entity_ids,
    )

    return rel_count


def _delete_weak_relates_to(conn: sqlite3.Connection) -> int:
    """Delete weak RELATES_TO relations. Returns deleted count."""
    placeholders = ",".join("?" * len(_WEAK_RELATES_TO_EVIDENCES))
    count = conn.execute(
        f"SELECT COUNT(*) FROM relations WHERE rel_type = 'RELATES_TO' AND evidence IN ({placeholders})",
        list(_WEAK_RELATES_TO_EVIDENCES),
    ).fetchone()[0]
    conn.execute(
        f"DELETE FROM relations WHERE rel_type = 'RELATES_TO' AND evidence IN ({placeholders})",
        list(_WEAK_RELATES_TO_EVIDENCES),
    )
    return count


def _rescore_entities(conn: sqlite3.Connection) -> int:
    """Rescore all remaining entities. Returns count of rescored entities."""
    # Build access count map
    access_counts: dict[str, int] = {}
    try:
        rows = conn.execute(
            "SELECT entity_id, COUNT(*) as cnt FROM access_log GROUP BY entity_id"
        ).fetchall()
        access_counts = {row["entity_id"]: row["cnt"] for row in rows}
    except sqlite3.OperationalError:
        pass

    entities = conn.execute("SELECT * FROM entities").fetchall()
    today = date.today()
    rescored = 0

    for row in entities:
        entity_dict = {
            "last_seen": row["last_seen"],
            "day_count": row["day_count"],
            "distinct_categories": 0,
            "tags": [],
            "type": row["type"],
        }
        # Try to extract tags from metadata JSON
        try:
            import json
            meta = json.loads(row["metadata"])
            entity_dict["tags"] = meta.get("tags", [])
            entity_dict["distinct_categories"] = meta.get("distinct_categories", 0)
        except (json.JSONDecodeError, TypeError):
            pass

        new_score = score_entity(
            entity_dict,
            access_count=access_counts.get(row["id"], 0),
            today=today,
        )
        conn.execute(
            "UPDATE entities SET importance = ? WHERE id = ?",
            (new_score, row["id"]),
        )
        rescored += 1

    return rescored


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clean noisy entities and rescore importance in Knowledge Weaver DB"
    )
    parser.add_argument(
        "--db-path",
        default="/home/openclaw/.openclaw/knowledge/knowledge.db",
        help="Path to the Knowledge Weaver SQLite database",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print impact summary without making any changes",
    )
    parser.add_argument(
        "--prune-only",
        action="store_true",
        help="Only prune noisy/stale entities; skip rescoring (for periodic cron usage)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.db_path):
        print(f"Error: Database not found at {args.db_path}")
        sys.exit(1)

    conn = sqlite3.connect(args.db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")

    # Pre-cleanup stats
    stats_before = _get_stats(conn)
    _print_stats("BEFORE CLEANUP", stats_before)

    # Find noisy entities
    noisy = _find_noisy_entity_ids(conn)
    all_noisy_ids = []
    for ids in noisy.values():
        all_noisy_ids.extend(ids)

    print(f"\n{'=' * 60}")
    print(f"  NOISY ENTITY ANALYSIS")
    print(f"{'=' * 60}")
    for reason, ids in noisy.items():
        print(f"  {reason}: {len(ids)} entities")
    print(f"  Total noisy entities: {len(all_noisy_ids)}")

    # Count weak RELATES_TO
    placeholders = ",".join("?" * len(_WEAK_RELATES_TO_EVIDENCES))
    weak_rel_count = conn.execute(
        f"SELECT COUNT(*) FROM relations WHERE rel_type = 'RELATES_TO' AND evidence IN ({placeholders})",
        list(_WEAK_RELATES_TO_EVIDENCES),
    ).fetchone()[0]
    print(f"  Weak RELATES_TO relations: {weak_rel_count}")

    if args.dry_run:
        print(f"\n{'=' * 60}")
        print(f"  DRY RUN — no changes made")
        print(f"{'=' * 60}")
        print(f"  Would delete {len(all_noisy_ids)} noisy entities (and their relations/vectors)")
        print(f"  Would delete {weak_rel_count} weak RELATES_TO relations")
        if not args.prune_only:
            print(f"  Would rescore {stats_before['total_entities'] - len(all_noisy_ids)} remaining entities")
        conn.close()
        return

    # Step A: Delete noisy entities (cascade)
    del_rel_count = _delete_entities_cascade(conn, all_noisy_ids)
    print(f"\n  Step A: Deleted {len(all_noisy_ids)} noisy entities, {del_rel_count} related relations")

    # Step B: Delete weak RELATES_TO relations
    weak_deleted = _delete_weak_relates_to(conn)
    print(f"  Step B: Deleted {weak_deleted} weak RELATES_TO relations")

    # Step C: Rescore remaining entities (skip if prune-only)
    if args.prune_only:
        print(f"  Step C: Skipped rescore (--prune-only mode)")
    else:
        rescored = _rescore_entities(conn)
        print(f"  Step C: Rescored {rescored} entities")

    conn.commit()

    # fix: VACUUM to reclaim space from DELETE operations
    conn.execute("VACUUM")

    # Post-cleanup stats
    stats_after = _get_stats(conn)
    _print_stats("AFTER CLEANUP", stats_after)

    # Summary
    print(f"\n{'=' * 60}")
    print(f"  SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Entities deleted: {stats_before['total_entities'] - stats_after['total_entities']}")
    for etype in sorted(set(list(stats_before['entities_by_type'].keys()) + list(stats_after['entities_by_type'].keys()))):
        before = stats_before['entities_by_type'].get(etype, 0)
        after = stats_after['entities_by_type'].get(etype, 0)
        delta = before - after
        if delta > 0:
            print(f"    {etype}: -{delta}")
    print(f"  Relations deleted: {stats_before['total_relations'] - stats_after['total_relations']}")
    print(f"  Vectors deleted: {stats_before['total_vectors'] - stats_after['total_vectors']}")

    conn.close()


if __name__ == "__main__":
    main()
