"""Second-pass semantic dedup: same-type vector-similar pairs auto-merge (HIGH),
queue (MID), and identifier-like names are demoted to review. Controlled vectors
make cosine exact and deterministic."""
import math

from knowledge_weaver.db import (
    count_pending_reviews,
    get_entity,
    init_db,
    insert_entity,
    upsert_entity_vector,
)
from knowledge_weaver.embedder import DEFAULT_DIMENSION
from knowledge_weaver.semantic_dedupe import semantic_dedupe

DIM = DEFAULT_DIMENSION


def _vec(c: float, base_dim: int = 0) -> list[float]:
    v = [0.0] * DIM
    v[base_dim] = c
    v[base_dim + 1] = math.sqrt(max(0.0, 1.0 - c * c))
    return v


def _add(conn, eid, etype, name, vec, day_count=1):
    insert_entity(conn, {
        "id": eid, "type": etype, "name": name, "summary": name,
        "importance": 0.5, "first_seen": "2026-01-01", "last_seen": "2026-01-01",
        "day_count": day_count,
    })
    upsert_entity_vector(conn, eid, vec)


def test_semantic_dedupe_bands(temp_db_path):
    conn = init_db(temp_db_path)
    # decision cluster (dims 0/1): canonical + a HIGH dup + a MID near-miss
    _add(conn, "dec:main", "decision", "聚合逻辑集成到HomeBrain", _vec(1.0), day_count=5)
    _add(conn, "dec:dup", "decision", "聚合逻辑集成到HomeBrain自身不依赖外部脚本", _vec(0.95))
    _add(conn, "dec:mid", "decision", "HomeBrain改用本地聚合", _vec(0.85))
    # identifier-like pair (dims 2/3): high cosine but version names -> review only
    _add(conn, "tech:va", "tech", "v0.2.0", _vec(1.0, base_dim=2))
    _add(conn, "tech:vb", "tech", "v2.9.0", _vec(0.95, base_dim=2))

    r = semantic_dedupe(conn)

    assert r["merged"] == 1                                  # only the HIGH dup auto-merged
    assert get_entity(conn, "dec:dup") is None               # merged away
    main = get_entity(conn, "dec:main")
    assert main["day_count"] == 6                            # 5 + 1: accumulation restored
    assert "不依赖外部脚本" in main["metadata"]               # variant kept as alias
    # MID near-miss + identifier pair both queued, not auto-merged
    assert get_entity(conn, "tech:va") and get_entity(conn, "tech:vb")
    assert count_pending_reviews(conn) >= 2
    conn.close()


def test_semantic_dedupe_dry_run(temp_db_path):
    conn = init_db(temp_db_path)
    _add(conn, "dec:main", "decision", "采用本地规则引擎做聚合", _vec(1.0), day_count=5)
    _add(conn, "dec:dup", "decision", "采用本地规则引擎做聚合不依赖LLM", _vec(0.95))
    r = semantic_dedupe(conn, dry_run=True)
    assert r["candidates_high"] == 1 and r["merged"] == 0
    assert get_entity(conn, "dec:dup") is not None            # nothing written
    conn.close()


def test_semantic_dedupe_review_only(temp_db_path):
    conn = init_db(temp_db_path)
    _add(conn, "dec:main", "decision", "采用本地规则引擎做聚合", _vec(1.0), day_count=5)
    _add(conn, "dec:dup", "decision", "采用本地规则引擎做聚合不依赖LLM", _vec(0.95))
    r = semantic_dedupe(conn, auto_merge=False)
    assert r["merged"] == 0                                   # nothing auto-merged
    assert r["queued"] == 1                                   # HIGH pair queued instead
    assert get_entity(conn, "dec:dup") is not None            # still present, awaiting review
    assert count_pending_reviews(conn) == 1
    conn.close()


def test_skips_already_reviewed_pairs(temp_db_path):
    """A pair with an existing (pending/dismissed) review is not re-surfaced —
    stops the cron loop from re-queuing + re-judging the same distinct pairs."""
    from knowledge_weaver.db import insert_review, set_review_status
    conn = init_db(temp_db_path)
    _add(conn, "dec:main", "decision", "采用本地规则引擎做聚合", _vec(1.0), day_count=5)
    _add(conn, "dec:dup", "decision", "采用本地规则引擎做聚合不依赖LLM", _vec(0.95))
    rid = insert_review(conn, kind="merge", new_entity_id="dec:dup",
                        candidate_id="dec:main", entity_type="decision", score=0.95, reason="t")
    set_review_status(conn, rid, "dismissed")
    r = semantic_dedupe(conn)
    assert r["candidates_high"] == 0 and r["merged"] == 0   # reviewed pair skipped
    conn.close()
