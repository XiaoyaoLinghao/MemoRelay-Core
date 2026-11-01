"""W3.4 tests: typed-relation rebuild (LLM mocked; deterministic surgery)."""
import os
import sys

from knowledge_weaver.db import init_db, insert_entity, insert_relation
from knowledge_weaver.linker import generate_relation_id
from knowledge_weaver.relations_schema import REL_TYPES, schema_prompt

from knowledge_weaver.typed_relations import type_relations  # noqa: E402


def _ent(conn, eid, etype, name):
    insert_entity(conn, {
        "id": eid, "type": etype, "name": name, "summary": name + " summary",
        "importance": 0.5, "first_seen": "2026-01-01", "last_seen": "2026-01-01",
    })


def _rel(conn, a, b):
    rid = generate_relation_id(a, b, "RELATES_TO")
    insert_relation(conn, {"id": rid, "from_entity": a, "to_entity": b,
                           "rel_type": "RELATES_TO", "weight": 1.0,
                           "evidence": "co_occurrence"})
    return rid


def test_retypes_and_prunes(temp_db_path):
    conn = init_db(temp_db_path)
    _ent(conn, "proj:a", "project", "A")
    _ent(conn, "tech:b", "tech", "B")
    _ent(conn, "fact:c", "fact", "C")
    rid1 = _rel(conn, "proj:a", "tech:b")
    rid2 = _rel(conn, "proj:a", "fact:c")
    conn.close()

    def mock_type(cands):
        return {c["rel_id"]: ("使用" if c["rel_id"] == rid1 else "none") for c in cands}

    r = type_relations(temp_db_path, mock_type)
    assert r["candidates"] == 2 and r["typed"] == 1 and r["pruned"] == 1

    conn = init_db(temp_db_path)
    rels = {(x["from_entity"], x["to_entity"], x["rel_type"])
            for x in conn.execute("SELECT * FROM relations").fetchall()}
    conn.close()
    assert ("proj:a", "tech:b", "使用") in rels      # re-typed
    assert all(rt != "RELATES_TO" for *_, rt in rels)  # no generic left
    assert not any(to == "fact:c" for _, to, _ in rels)  # pruned


def test_dry_run_no_write(temp_db_path):
    conn = init_db(temp_db_path)
    _ent(conn, "proj:a", "project", "A")
    _ent(conn, "tech:b", "tech", "B")
    rid = _rel(conn, "proj:a", "tech:b")
    conn.close()

    r = type_relations(temp_db_path, lambda c: {rid: "使用"}, dry_run=True)
    assert r["candidates"] == 1
    conn = init_db(temp_db_path)
    rt = {x["rel_type"] for x in conn.execute("SELECT rel_type FROM relations").fetchall()}
    conn.close()
    assert rt == {"RELATES_TO"}  # unchanged in dry-run


def test_invalid_type_pruned(temp_db_path):
    conn = init_db(temp_db_path)
    _ent(conn, "proj:a", "project", "A")
    _ent(conn, "tech:b", "tech", "B")
    rid = _rel(conn, "proj:a", "tech:b")
    conn.close()
    r = type_relations(temp_db_path, lambda c: {rid: "瞎编类型"})
    assert r["typed"] == 0 and r["pruned"] == 1


def test_schema_prompt_lists_all_types():
    p = schema_prompt()
    for t in REL_TYPES:
        assert t in p


def test_typing_prompt_tightened():
    from knowledge_weaver.typed_relations import build_typing_system_prompt
    p = build_typing_system_prompt()
    for t in REL_TYPES:
        assert t in p                      # all 8 types listed
    assert "真正的因果" in p               # 导致 tightened (causation-only)
    assert "优先" in p                     # prefer specific types
    assert "none" in p                     # prune coincidental
    assert "⇒ 使用" in p                   # few-shot examples present
    assert "project→tech" in p             # triplet hints injected


def test_reset_llm_typed_edges(temp_db_path):
    from knowledge_weaver.db import init_db, insert_entity, insert_relation
    from knowledge_weaver.typed_relations import reset_llm_typed_edges
    conn = init_db(temp_db_path)
    for eid, t, n in [("proj:a", "project", "A"), ("tech:b", "tech", "B"), ("proj:c", "project", "C")]:
        insert_entity(conn, {"id": eid, "type": t, "name": n, "summary": n,
                             "importance": 0.5, "first_seen": "2026-01-01", "last_seen": "2026-01-01"})
    insert_relation(conn, {"id": generate_relation_id("proj:a", "tech:b", "使用"),
                           "from_entity": "proj:a", "to_entity": "tech:b", "rel_type": "使用",
                           "weight": 1.0, "evidence": "llm-typed"})
    insert_relation(conn, {"id": generate_relation_id("proj:a", "proj:c", "DEPENDS_ON"),
                           "from_entity": "proj:a", "to_entity": "proj:c", "rel_type": "DEPENDS_ON",
                           "weight": 1.0, "evidence": "co_occurrence"})
    assert reset_llm_typed_edges(conn) == 1
    rels = {(x["from_entity"], x["to_entity"], x["rel_type"])
            for x in conn.execute("SELECT * FROM relations").fetchall()}
    assert ("proj:a", "tech:b", "RELATES_TO") in rels   # llm-typed reverted
    assert ("proj:a", "proj:c", "DEPENDS_ON") in rels   # original preserved
    conn.close()
