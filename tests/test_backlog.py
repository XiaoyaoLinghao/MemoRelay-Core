"""Tests for backlog finalization: W6 embedding-drift, W3.3 supersede."""
from knowledge_weaver.db import (
    check_embedding_drift,
    init_db,
    insert_entity,
    insert_relation,
)
from knowledge_weaver.linker import generate_relation_id


# --------------------------------------------------------------------------- #
# W6: embedding drift detection                                               #
# --------------------------------------------------------------------------- #
def test_embedding_drift_records_then_detects(temp_db_path):
    conn = init_db(temp_db_path)
    d1 = check_embedding_drift(conn, model="bge-m3", dimension=1024)
    assert d1["drift"] is False and d1.get("first_record")  # baseline recorded

    assert check_embedding_drift(conn, model="bge-m3", dimension=1024)["drift"] is False

    d_model = check_embedding_drift(conn, model="other-model", dimension=1024)
    assert d_model["drift"] is True and "model" in d_model["changed"]

    d_dim = check_embedding_drift(conn, model="bge-m3", dimension=768)
    assert d_dim["drift"] is True and "dimension" in d_dim["changed"]
    conn.close()


# --------------------------------------------------------------------------- #
# W3.3: superseded entities de-prioritized + flagged                          #
# --------------------------------------------------------------------------- #
def _dec(conn, eid, name, summary, seen):
    insert_entity(conn, {
        "id": eid, "type": "decision", "name": name, "summary": summary,
        "importance": 0.6, "first_seen": seen, "last_seen": seen, "day_count": 2,
    })


def test_superseded_deprioritized_and_flagged(temp_db_path):
    from knowledge_weaver.tools import knowledge_search, superseded_entity_ids

    conn = init_db(temp_db_path)
    _dec(conn, "decision:new", "新部署方案", "deploy path Y", "2026-02-01")
    _dec(conn, "decision:old", "旧部署方案", "deploy path X", "2026-01-01")
    # A 取代 B  ->  decision:new supersedes decision:old
    insert_relation(conn, {
        "id": generate_relation_id("decision:new", "decision:old", "取代"),
        "from_entity": "decision:new", "to_entity": "decision:old",
        "rel_type": "取代", "weight": 1.0, "evidence": "llm-typed",
    })

    assert superseded_entity_ids(conn) == {"decision:old"}

    r = knowledge_search(conn, query="deploy path", embedder=None, max_results=10)
    flags = {x["entity_id"]: x["superseded"] for x in r["results"]}
    assert flags.get("decision:old") is True
    assert flags.get("decision:new") is False
    ids = [x["entity_id"] for x in r["results"]]
    if "decision:old" in ids and "decision:new" in ids:
        assert ids.index("decision:new") < ids.index("decision:old")  # current first
    conn.close()
