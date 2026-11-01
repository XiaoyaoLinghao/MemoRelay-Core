"""Max-review fixes K1-K8 regression tests."""
from knowledge_weaver.db import (
    count_pending_reviews, get_entity, get_relations_for_entity, init_db,
    insert_entity, insert_relation, insert_review, merge_existing_entities,
    rollback_merge, set_review_status,
)


def _e(conn, eid, etype="tech", name=None, day_count=1):
    insert_entity(conn, {"id": eid, "type": etype, "name": name or eid, "summary": name or eid,
                         "importance": 0.5, "first_seen": "2026-01-01", "last_seen": "2026-01-01",
                         "day_count": day_count})


def test_k2_json_facts_skips_empty_slug():
    from knowledge_weaver.json_facts import facts_to_entities
    ents = facts_to_entities([
        {"type": "fact", "name": "🔥", "summary": "a"},
        {"type": "fact", "name": "✅", "summary": "b"},
        {"type": "fact", "name": "真实决策", "summary": "c"},
    ], "x.md")
    assert all(not e.id.endswith(":") for e in ents)   # no 'fact:' garbage id
    assert len(ents) == 1                              # only the real fact survives


def test_k7_name_mentioned_word_boundary():
    from knowledge_weaver.linker import _name_mentioned
    assert _name_mentioned("HomeBrain", "HomeBrain 部署完成")
    assert not _name_mentioned("Home", "HomeBrain 部署完成")   # NOT substring-inside
    assert _name_mentioned("家庭大脑", "今天讨论家庭大脑架构")   # CJK still matches
    assert not _name_mentioned("KW", "KWX 是别的东西")          # boundary respected


def test_k3_merge_keeps_dst_edge_on_collision(temp_db_path):
    conn = init_db(temp_db_path)
    for eid in ("tech:a", "tech:b", "tech:x"):
        _e(conn, eid)
    from knowledge_weaver.linker import generate_relation_id
    insert_relation(conn, {"id": generate_relation_id("tech:b", "tech:x", "使用"),
                           "from_entity": "tech:b", "to_entity": "tech:x", "rel_type": "使用",
                           "weight": 0.9, "evidence": "rich"})
    insert_relation(conn, {"id": generate_relation_id("tech:a", "tech:x", "使用"),
                           "from_entity": "tech:a", "to_entity": "tech:x", "rel_type": "使用",
                           "weight": 0.3, "evidence": "poor"})
    merge_existing_entities(conn, from_id="tech:a", into_id="tech:b")
    # the B->X edge must keep B's original weight/evidence, not be clobbered by A's
    edges = [r for r in get_relations_for_entity(conn, "tech:b") if r["to_entity"] == "tech:x"]
    assert len(edges) == 1 and edges[0]["weight"] == 0.9 and edges[0]["evidence"] == "rich"
    conn.close()


def test_k4_rollback_restores_dst_and_unstales_review(temp_db_path):
    conn = init_db(temp_db_path)
    _e(conn, "tech:a", day_count=3)
    _e(conn, "tech:b", day_count=5)
    _e(conn, "tech:c")
    # a sibling pending review referencing A (not the one driving the merge)
    insert_review(conn, kind="merge", new_entity_id="tech:a", candidate_id="tech:c",
                  entity_type="tech", score=0.8, reason="t")
    merge_existing_entities(conn, from_id="tech:a", into_id="tech:b")
    assert get_entity(conn, "tech:b")["day_count"] == 8        # accumulated
    assert count_pending_reviews(conn) == 0                    # sibling -> stale
    log_id = conn.execute("SELECT MAX(id) FROM merge_log").fetchone()[0]
    assert rollback_merge(conn, log_id)
    assert get_entity(conn, "tech:a") is not None              # recreated
    assert get_entity(conn, "tech:b")["day_count"] == 5        # K4: restored
    assert count_pending_reviews(conn) == 1                    # K4: sibling un-staled
    conn.close()


def test_k6_trace_depth2_real_rel_type(temp_db_path):
    conn = init_db(temp_db_path)
    _e(conn, "proj:a", "project", "A")
    _e(conn, "task:b", "task", "B")
    _e(conn, "risk:c", "risk", "C")
    from knowledge_weaver.linker import generate_relation_id
    insert_relation(conn, {"id": generate_relation_id("proj:a", "task:b", "依赖"),
                           "from_entity": "proj:a", "to_entity": "task:b", "rel_type": "依赖",
                           "weight": 1.0, "evidence": "x"})
    insert_relation(conn, {"id": generate_relation_id("task:b", "risk:c", "导致"),
                           "from_entity": "task:b", "to_entity": "risk:c", "rel_type": "导致",
                           "weight": 1.0, "evidence": "x"})
    from knowledge_weaver.tools import knowledge_trace
    res = knowledge_trace(conn, topic="A", max_depth=2)
    by_id = {r["entity_id"]: r for r in res["related"]}
    assert by_id["risk:c"]["rel_type"] == "导致"   # K6: not the RELATES_TO fallback
    conn.close()


def test_clean_and_rescore_prunes_accessed_entity(temp_db_path):
    """FK fix: an entity with an access_log row (FK-referencing entities) must be
    deletable by clean_and_rescore's cascade under PRAGMA foreign_keys=ON."""
    import importlib.util
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "car_mod", root / "scripts" / "clean_and_rescore.py")
    car = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(car)

    conn = init_db(temp_db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    _e(conn, "tech:a")
    _e(conn, "tech:b")
    from knowledge_weaver.linker import generate_relation_id
    insert_relation(conn, {"id": generate_relation_id("tech:a", "tech:b", "使用"),
                           "from_entity": "tech:a", "to_entity": "tech:b", "rel_type": "使用",
                           "weight": 1.0, "evidence": "x"})
    conn.execute("INSERT INTO access_log(entity_id, tool, query) VALUES('tech:a','search','q')")
    conn.commit()
    # pre-fix: raised sqlite3.IntegrityError (FK); now deletes cleanly
    car._delete_entities_cascade(conn, ["tech:a"])
    assert get_entity(conn, "tech:a") is None
    assert conn.execute("SELECT COUNT(*) FROM access_log WHERE entity_id='tech:a'").fetchone()[0] == 0
    conn.close()
