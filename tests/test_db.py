import sqlite3
from knowledge_weaver.db import (
    init_db, insert_entity, get_entity, list_entities_by_type,
    list_all_entities, search_entities_fts, insert_relation,
    get_relations_for_entity, delete_entity,
    get_manifest, upsert_manifest, list_all_manifest,
    log_access, get_access_count,
)


def test_init_db_creates_tables(temp_db_path):
    conn = init_db(temp_db_path)
    tables = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall()
    table_names = [r[0] for r in tables]
    assert "entities" in table_names
    assert "relations" in table_names
    assert "daily_manifest" in table_names
    assert "access_log" in table_names
    conn.close()


def test_insert_and_get_entity(temp_db_path):
    conn = init_db(temp_db_path)

    insert_entity(conn, {
        "id": "decision:test_rule",
        "type": "decision",
        "name": "测试决策",
        "summary": "这是一个测试",
        "importance": 0.75,
        "first_seen": "2026-05-24",
        "last_seen": "2026-05-24",
        "source_lines": '["memory/2026-05-24.md:10:12"]',
        "metadata": '{"tags": ["test"]}',
    })

    row = get_entity(conn, "decision:test_rule")
    assert row is not None
    assert row["name"] == "测试决策"
    assert row["importance"] == 0.75

    rows = list_entities_by_type(conn, "decision")
    assert len(rows) == 1
    assert rows[0]["id"] == "decision:test_rule"
    conn.close()


def test_upsert_entity_updates_existing(temp_db_path):
    conn = init_db(temp_db_path)

    insert_entity(conn, {
        "id": "task:test_task",
        "type": "task",
        "name": "测试任务",
        "summary": "first version",
        "importance": 0.5,
        "first_seen": "2026-05-22",
        "last_seen": "2026-05-22",
        "day_count": 1,
        "source_lines": '["2026-05-22.md:5:6"]',
    })

    # Upsert with same id
    insert_entity(conn, {
        "id": "task:test_task",
        "type": "task",
        "name": "测试任务 v2",
        "summary": "second version",
        "importance": 0.8,
        "first_seen": "2026-05-22",
        "last_seen": "2026-05-24",
        "day_count": 2,
        "source_lines": '["2026-05-22.md:5:6","2026-05-24.md:3:4"]',
    })

    row = get_entity(conn, "task:test_task")
    assert row["summary"] == "second version"
    assert row["importance"] == 0.8
    assert row["day_count"] == 2
    conn.close()


def test_insert_and_get_relation(temp_db_path):
    conn = init_db(temp_db_path)

    insert_entity(conn, _entity("proj:exampleproject", "project", "ExampleProject"))
    insert_entity(conn, _entity("decision:rule_engine", "decision", "Rule Engine"))

    insert_relation(conn, {
        "id": "rel:exampleproject->rule_engine",
        "from_entity": "proj:exampleproject",
        "to_entity": "decision:rule_engine",
        "rel_type": "RELATES_TO",
        "weight": 0.7,
        "evidence": "co-occurred on 2026-05-24",
    })

    # Get relations from entity
    rels = get_relations_for_entity(conn, "proj:exampleproject")
    assert len(rels) >= 1
    assert rels[0]["rel_type"] == "RELATES_TO"

    # Get relations to entity
    rels2 = get_relations_for_entity(conn, "decision:rule_engine")
    assert len(rels2) >= 1
    conn.close()


def test_search_entities_fts(temp_db_path):
    conn = init_db(temp_db_path)

    insert_entity(conn, _entity("proj:exampleproject", "project", "ExampleProject智能家居"))
    insert_entity(conn, _entity("fact:ha_rule", "fact", "HA命名规则"))
    insert_entity(conn, _entity("risk:cache", "risk", "Docker缓存风险"))

    # Search by name
    results = search_entities_fts(conn, "ExampleProject")
    assert len(results) >= 1
    assert any(r["id"] == "proj:exampleproject" for r in results)

    # Search by summary (summary same as name in _entity helper)
    results2 = search_entities_fts(conn, "命名")
    assert len(results2) >= 1
    conn.close()


def test_list_all_entities(temp_db_path):
    conn = init_db(temp_db_path)

    insert_entity(conn, _entity("proj:a", "project", "A", importance=0.3))
    insert_entity(conn, _entity("proj:b", "project", "B", importance=0.9))
    insert_entity(conn, _entity("proj:c", "project", "C", importance=0.6))

    rows = list_all_entities(conn)
    assert len(rows) == 3
    # Ordered by importance DESC
    assert rows[0]["id"] == "proj:b"
    assert rows[1]["id"] == "proj:c"
    assert rows[2]["id"] == "proj:a"
    conn.close()


def test_delete_entity(temp_db_path):
    conn = init_db(temp_db_path)

    insert_entity(conn, _entity("proj:del_me", "project", "ToDelete"))
    insert_entity(conn, _entity("proj:other", "project", "Other"))
    insert_relation(conn, {
        "id": "rel:del_me->other",
        "from_entity": "proj:del_me",
        "to_entity": "proj:other",
        "rel_type": "RELATES_TO",
        "weight": 0.5,
        "evidence": "test",
    })

    delete_entity(conn, "proj:del_me")

    assert get_entity(conn, "proj:del_me") is None
    # Relations should also be gone
    rels = get_relations_for_entity(conn, "proj:del_me")
    assert len(rels) == 0
    conn.close()


def test_manifest_operations(temp_db_path):
    conn = init_db(temp_db_path)

    upsert_manifest(conn, {
        "date": "2026-05-24",
        "file_path": "/memory/2026-05-24.md",
        "file_hash": "abc123",
        "entity_count": 5,
        "status": "ok",
    })

    row = get_manifest(conn, "2026-05-24")
    assert row is not None
    assert row["file_hash"] == "abc123"
    assert row["entity_count"] == 5

    # Upsert updates existing
    upsert_manifest(conn, {
        "date": "2026-05-24",
        "file_path": "/memory/2026-05-24.md",
        "file_hash": "def456",
        "entity_count": 7,
        "status": "ok",
    })

    row2 = get_manifest(conn, "2026-05-24")
    assert row2["file_hash"] == "def456"
    assert row2["entity_count"] == 7

    # List all
    upsert_manifest(conn, {
        "date": "2026-05-23",
        "file_path": "/memory/2026-05-23.md",
        "file_hash": "ghi789",
        "entity_count": 3,
        "status": "ok",
    })
    all_manifest = list_all_manifest(conn)
    assert len(all_manifest) == 2
    conn.close()


def test_access_log_operations(temp_db_path):
    conn = init_db(temp_db_path)

    insert_entity(conn, _entity("proj:test", "project", "Test"))

    log_access(conn, "proj:test", "knowledge_search", "智能家居")
    log_access(conn, "proj:test", "knowledge_trace", "ExampleProject")

    count = get_access_count(conn, "proj:test")
    assert count == 2

    count_zero = get_access_count(conn, "proj:nonexistent")
    assert count_zero == 0
    conn.close()


def _entity(eid, etype, name, importance=0.5, date="2026-05-24"):
    return {
        "id": eid,
        "type": etype,
        "name": name,
        "summary": name,
        "importance": importance,
        "first_seen": date,
        "last_seen": date,
        "day_count": 1,
        "source_lines": f'["{date}.md:1:1"]',
        "metadata": "{}",
    }

def test_insert_relation_rejects_self_loop(temp_db_path):
    from knowledge_weaver.db import init_db, insert_entity, insert_relation
    conn = init_db(temp_db_path)
    insert_entity(conn, {
        "id": "proj:x", "type": "project", "name": "X", "summary": "X",
        "importance": 0.5, "first_seen": "2026-05-24", "last_seen": "2026-05-24",
        "day_count": 1, "source_lines": "[]", "metadata": "{}",
    })
    insert_relation(conn, {
        "id": "rel:self", "from_entity": "proj:x", "to_entity": "proj:x",
        "rel_type": "CONTINUES", "weight": 1.0, "evidence": "test",
    })
    n = conn.execute("SELECT COUNT(*) FROM relations").fetchone()[0]
    conn.close()
    assert n == 0

def test_vec_available_per_connection(tmp_path):
    """sqlite-vec capability is tracked per connection, not globally."""
    from knowledge_weaver.db import init_db, _can_use_vec
    db1 = init_db(str(tmp_path / "a.db"))
    db2 = init_db(str(tmp_path / "b.db"))
    r1 = _can_use_vec(db1)
    r2 = _can_use_vec(db2)
    assert r1 == r2
    db1.close(); db2.close()


def test_init_db_creates_parent_directory(tmp_path):
    """init_db must create the DB file's parent directory if missing."""
    from knowledge_weaver.db import init_db
    import os
    db_path = str(tmp_path / "nested" / "deeper" / "k.db")
    conn = init_db(db_path)
    conn.close()
    assert os.path.exists(db_path)
