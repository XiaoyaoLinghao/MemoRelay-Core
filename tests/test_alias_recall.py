"""Merging must preserve lexical recall: a query using a merged-away (variant/
'wrong') name should still find the surviving entity via its alias."""
from knowledge_weaver.db import (
    get_entity,
    init_db,
    insert_entity,
    merge_existing_entities,
    search_entities_fts,
)


def _ent(conn, eid, name):
    insert_entity(conn, {
        "id": eid, "type": "decision", "name": name, "summary": name,
        "importance": 0.6, "first_seen": "2026-01-01", "last_seen": "2026-01-01",
    })


def test_alias_searchable_after_merge(temp_db_path):
    conn = init_db(temp_db_path)
    _ent(conn, "dec:agg_a", "聚合逻辑集成到HomeBrain")
    _ent(conn, "dec:agg_b", "聚合逻辑集成到HomeBrain自身不依赖外部脚本")

    assert merge_existing_entities(conn, from_id="dec:agg_b", into_id="dec:agg_a")
    assert get_entity(conn, "dec:agg_b") is None                 # merged away
    aliases = get_entity(conn, "dec:agg_a")["metadata"]
    assert "不依赖外部脚本" in aliases                            # variant kept as alias

    # querying with the merged-away ('wrong') phrasing still hits the survivor
    ids = [r["id"] for r in search_entities_fts(conn, "自身不依赖外部脚本")]
    assert "dec:agg_a" in ids
    conn.close()
