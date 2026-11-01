"""T1 tie-breaker: LLM verdict 'same'->merge, 'different'->dismiss, deterministic
DB surgery with a mock judge (no network)."""
from knowledge_weaver.db import (
    count_pending_reviews,
    get_entity,
    init_db,
    insert_entity,
    insert_review,
)
from knowledge_weaver.tiebreak import tiebreak_reviews


def _ent(conn, eid, name):
    insert_entity(conn, {
        "id": eid, "type": "tech", "name": name, "summary": name,
        "importance": 0.5, "first_seen": "2026-01-01", "last_seen": "2026-01-01",
    })


def test_tiebreak_merges_same_dismisses_different(temp_db_path):
    conn = init_db(temp_db_path)
    _ent(conn, "tech:kw_long", "Knowledge Weaver")
    _ent(conn, "tech:kw_cn", "知识编织器")
    _ent(conn, "tech:v020", "v0.2.0")
    _ent(conn, "tech:v290", "v2.9.0")
    insert_review(conn, kind="merge", new_entity_id="tech:kw_long",
                  candidate_id="tech:kw_cn", entity_type="tech", score=0.85, reason="t")
    insert_review(conn, kind="merge", new_entity_id="tech:v020",
                  candidate_id="tech:v290", entity_type="tech", score=0.92, reason="t")
    assert count_pending_reviews(conn) == 2

    def judge(pairs):  # same for the alias pair, different for the version pair
        return {p["review_id"]: ("same" if "Weaver" in p["a"] else "different")
                for p in pairs}

    r = tiebreak_reviews(conn, judge)
    assert r["merged"] == 1 and r["dismissed"] == 1
    assert get_entity(conn, "tech:kw_long") is None          # merged away
    assert get_entity(conn, "tech:kw_cn") is not None         # survivor
    assert get_entity(conn, "tech:v020") is not None          # version pair kept distinct
    assert get_entity(conn, "tech:v290") is not None
    assert count_pending_reviews(conn) == 0                   # queue cleared
    conn.close()


def test_tiebreak_dry_run_writes_nothing(temp_db_path):
    conn = init_db(temp_db_path)
    _ent(conn, "tech:a_long", "Alpha Service")
    _ent(conn, "tech:b_long", "阿尔法服务")
    insert_review(conn, kind="merge", new_entity_id="tech:a_long",
                  candidate_id="tech:b_long", entity_type="tech", score=0.85, reason="t")
    r = tiebreak_reviews(conn, lambda pairs: {p["review_id"]: "same" for p in pairs},
                         dry_run=True)
    assert r["merged"] == 1 and r["candidates"] == 1
    assert get_entity(conn, "tech:a_long") is not None        # nothing written
    assert count_pending_reviews(conn) == 1
    conn.close()


def test_noop_merge_counts_as_dismissed(temp_db_path):
    """A 'same' verdict whose target was already removed (by a prune or an
    earlier merge in the same batch) is a no-op merge -> counted as dismissed,
    not as a phantom merge."""
    from knowledge_weaver.tiebreak import apply_tiebreak
    conn = init_db(temp_db_path)
    _ent(conn, "tech:c_long", "Cluster Core")
    _ent(conn, "tech:a_long", "Alpha Node")
    _ent(conn, "tech:b_long", "Beta Node")
    r1 = insert_review(conn, kind="merge", new_entity_id="tech:a_long",
                       candidate_id="tech:c_long", entity_type="tech", score=0.9, reason="t")
    r2 = insert_review(conn, kind="merge", new_entity_id="tech:b_long",
                       candidate_id="tech:a_long", entity_type="tech", score=0.8, reason="t")
    # process A->C first (real merge, A removed), then B->A (target A gone -> no-op)
    pairs = [
        {"review_id": r1, "from_id": "tech:a_long", "into_id": "tech:c_long"},
        {"review_id": r2, "from_id": "tech:b_long", "into_id": "tech:a_long"},
    ]
    res = apply_tiebreak(conn, {r1: "same", r2: "same"}, pairs)
    assert res["merged"] == 1 and res["dismissed"] == 1     # not merged==2
    conn.close()


def test_merge_marks_orphan_pending_stale(temp_db_path):
    """#8: merging an entity away marks any OTHER pending review referencing it as
    'stale' so count_pending_reviews isn't permanently inflated by dangling rows."""
    from knowledge_weaver.db import (
        count_pending_reviews, init_db, insert_review, merge_existing_entities,
    )
    conn = init_db(temp_db_path)
    for eid in ["tech:a_node", "tech:b_node", "tech:c_node"]:
        insert_entity(conn, {"id": eid, "type": "tech", "name": eid, "summary": eid,
                             "importance": 0.5, "first_seen": "2026-01-01",
                             "last_seen": "2026-01-01"})
    insert_review(conn, kind="merge", new_entity_id="tech:a_node",
                  candidate_id="tech:c_node", entity_type="tech", score=0.8, reason="t")
    assert count_pending_reviews(conn) == 1
    merge_existing_entities(conn, from_id="tech:a_node", into_id="tech:b_node")
    assert count_pending_reviews(conn) == 0          # orphan review marked stale
    conn.close()
