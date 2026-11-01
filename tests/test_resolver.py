"""Deterministic tests for W1 entity resolution (resolver + review queue + log).

Uses a mock embedder producing 1024-dim vectors with *exact* controlled cosine
against a fixed base, so band classification is fully deterministic (no real
embedding model / network needed).
"""
import math

from knowledge_weaver.db import (
    count_pending_reviews,
    get_entity_vector,
    init_db,
    insert_entity,
    insert_review,
    list_pending_reviews,
    record_merge,
    set_review_status,
    upsert_entity_vector,
)
from knowledge_weaver.embedder import DEFAULT_DIMENSION
from knowledge_weaver.resolver import resolve_entity

DIM = DEFAULT_DIMENSION


def _vec_at_cos(c: float) -> list[float]:
    """Unit vector whose cosine with BASE = [1,0,0,...] equals exactly ``c``."""
    v = [0.0] * DIM
    v[0] = c
    v[1] = math.sqrt(max(0.0, 1.0 - c * c))
    return v


BASE = _vec_at_cos(1.0)  # [1, 0, 0, ...]


class MockEmbedder:
    def __init__(self, mapping: dict[str, list[float]]):
        self.mapping = mapping

    def embed(self, text: str):
        for k, v in self.mapping.items():
            if text.startswith(k) or k.startswith(text):
                return v
        return None

    def embed_batch(self, texts):
        return [self.embed(t) for t in texts]


def _add_entity(conn, eid, etype, name, summary, vec=None):
    insert_entity(conn, {
        "id": eid, "type": etype, "name": name, "summary": summary,
        "importance": 0.5, "first_seen": "2026-01-01", "last_seen": "2026-01-01",
    })
    if vec is not None:
        upsert_entity_vector(conn, eid, vec)


# --------------------------------------------------------------------------- #
# Band classification                                                         #
# --------------------------------------------------------------------------- #
def test_high_band_cross_language_merge(temp_db_path):
    """The core fix: HomeBrain / 家庭大脑 (name ratio ~0) merge via vector."""
    conn = init_db(temp_db_path)
    _add_entity(conn, "proj:homebrain", "project", "HomeBrain", "home brain hub", BASE)
    emb = MockEmbedder({"jiating danao": _vec_at_cos(0.95)})

    res = resolve_entity(
        conn, new_id="proj:jiatingdanao", entity_type="project",
        name="家庭大脑", summary="jiating danao", embedder=emb,
    )
    assert res.action == "merge"
    assert res.into_id == "proj:homebrain"
    assert res.reason == "auto:vector"
    conn.close()


def test_mid_band_goes_to_review(temp_db_path):
    conn = init_db(temp_db_path)
    _add_entity(conn, "proj:homebrain", "project", "HomeBrain", "home brain hub", BASE)
    emb = MockEmbedder({"jiating danao": _vec_at_cos(0.85)})  # in [0.82, 0.90)

    res = resolve_entity(
        conn, new_id="proj:jiatingdanao", entity_type="project",
        name="家庭大脑", summary="jiating danao", embedder=emb,
    )
    assert res.action == "review"
    assert res.into_id == "proj:homebrain"
    conn.close()


def test_low_band_distinct(temp_db_path):
    conn = init_db(temp_db_path)
    _add_entity(conn, "proj:homebrain", "project", "HomeBrain", "home brain hub", BASE)
    emb = MockEmbedder({"unrelated": _vec_at_cos(0.50)})

    res = resolve_entity(
        conn, new_id="proj:other", entity_type="project",
        name="完全无关", summary="unrelated", embedder=emb,
    )
    assert res.action == "distinct"
    conn.close()


def test_name_high_same_script_merge(temp_db_path):
    """Same-script near-identical name auto-merges even when cosine is low."""
    conn = init_db(temp_db_path)
    _add_entity(conn, "proj:exampleproject", "project", "ExampleProject",
                "an example", BASE)
    emb = MockEmbedder({"example proj": _vec_at_cos(0.50)})  # low cos on purpose

    res = resolve_entity(
        conn, new_id="proj:exampleprojektt", entity_type="project",
        name="ExampleProjektt", summary="example proj", embedder=emb,
    )
    assert res.action == "merge"
    assert res.into_id == "proj:exampleproject"
    assert res.reason == "auto:name"
    conn.close()


def test_type_isolation(temp_db_path):
    """A high-cosine candidate of a DIFFERENT type is not a merge target."""
    conn = init_db(temp_db_path)
    _add_entity(conn, "person:homebrain", "person", "HomeBrain", "a person", BASE)
    emb = MockEmbedder({"jiating danao": _vec_at_cos(0.95)})

    res = resolve_entity(
        conn, new_id="proj:jiatingdanao", entity_type="project",
        name="家庭大脑", summary="jiating danao", embedder=emb,
    )
    assert res.action == "distinct"
    conn.close()


def test_no_candidates_distinct(temp_db_path):
    conn = init_db(temp_db_path)
    res = resolve_entity(
        conn, new_id="proj:fresh", entity_type="project",
        name="全新项目", summary="brand new", embedder=MockEmbedder({}),
    )
    assert res.action == "distinct"
    conn.close()


def test_authoritative_normalize(temp_db_path):
    """Registered alias normalizes to canonical id, even with no embedder/DB hit."""
    conn = init_db(temp_db_path)
    res = resolve_entity(
        conn, new_id="proj:jiatingdanao", entity_type="project",
        name="家庭大脑", summary="", embedder=None,
        alias_to_canonical={"proj:jiatingdanao": "proj:homebrain"},
    )
    assert res.action == "normalize"
    assert res.into_id == "proj:homebrain"
    assert res.reason == "registered-alias"
    conn.close()


# --------------------------------------------------------------------------- #
# Review queue + merge log helpers                                            #
# --------------------------------------------------------------------------- #
def test_review_queue_idempotent_and_count(temp_db_path):
    conn = init_db(temp_db_path)
    rid1 = insert_review(conn, kind="merge", new_entity_id="proj:a",
                         candidate_id="proj:b", entity_type="project",
                         score=0.85, reason="review:vector")
    rid2 = insert_review(conn, kind="merge", new_entity_id="proj:a",
                         candidate_id="proj:b", entity_type="project",
                         score=0.85, reason="review:vector")
    assert rid1 == rid2  # idempotent per (kind,new,candidate) while pending
    assert count_pending_reviews(conn) == 1

    rows = list_pending_reviews(conn)
    assert len(rows) == 1 and rows[0]["new_entity_id"] == "proj:a"

    set_review_status(conn, rid1, "rejected")
    assert count_pending_reviews(conn) == 0
    conn.close()


def test_record_merge_log(temp_db_path):
    conn = init_db(temp_db_path)
    record_merge(conn, merged_from_id="proj:jiatingdanao",
                 merged_into_id="proj:homebrain", reason="auto:vector",
                 score=0.95, from_snapshot='{"id":"proj:jiatingdanao"}')
    row = conn.execute(
        "SELECT merged_from_id, merged_into_id, score FROM merge_log"
    ).fetchone()
    assert row[0] == "proj:jiatingdanao"
    assert row[1] == "proj:homebrain"
    assert abs(row[2] - 0.95) < 1e-9
    conn.close()


def test_get_entity_vector_roundtrip(temp_db_path):
    conn = init_db(temp_db_path)
    _add_entity(conn, "proj:x", "project", "X", "summary", _vec_at_cos(0.9))
    got = get_entity_vector(conn, "proj:x")
    assert got is not None and len(got) == DIM
    assert abs(got[0] - 0.9) < 1e-6
    assert get_entity_vector(conn, "proj:missing") is None
    conn.close()


# --------------------------------------------------------------------------- #
# Review-time merge of existing entities + rollback                            #
# --------------------------------------------------------------------------- #
def test_merge_existing_entities(temp_db_path):
    import json as _j

    from knowledge_weaver.db import (
        get_entity,
        get_relations_for_entity,
        insert_relation,
        merge_existing_entities,
    )
    from knowledge_weaver.linker import generate_relation_id

    conn = init_db(temp_db_path)
    _add_entity(conn, "proj:b", "project", "BravoCanonical", "canonical", BASE)
    _add_entity(conn, "proj:a", "project", "Bravo别名", "alias dup", _vec_at_cos(0.95))
    _add_entity(conn, "proj:c", "project", "Charlie", "other", _vec_at_cos(0.3))
    insert_relation(conn, {
        "id": generate_relation_id("proj:a", "proj:c", "RELATES_TO"),
        "from_entity": "proj:a", "to_entity": "proj:c", "rel_type": "RELATES_TO",
        "weight": 1.0, "evidence": "co_occurrence",
    })

    assert merge_existing_entities(conn, "proj:a", "proj:b", reason="test")
    assert get_entity(conn, "proj:a") is None  # merged away
    b = get_entity(conn, "proj:b")
    assert "Bravo别名" in _j.loads(b["metadata"] or "{}").get("aliases", [])
    rels = get_relations_for_entity(conn, "proj:b")
    assert any(r["to_entity"] == "proj:c" for r in rels)  # relation repointed
    assert conn.execute("SELECT COUNT(*) FROM merge_log").fetchone()[0] == 1
    conn.close()


def test_rollback_merge(temp_db_path):
    import json as _j

    from knowledge_weaver.db import get_entity, merge_existing_entities, rollback_merge

    conn = init_db(temp_db_path)
    _add_entity(conn, "proj:b", "project", "Canonical", "c", BASE)
    _add_entity(conn, "proj:a", "project", "Dup", "d", _vec_at_cos(0.95))
    merge_existing_entities(conn, "proj:a", "proj:b", reason="test")
    assert get_entity(conn, "proj:a") is None

    log_id = conn.execute("SELECT id FROM merge_log ORDER BY id DESC LIMIT 1").fetchone()[0]
    assert rollback_merge(conn, log_id)
    assert get_entity(conn, "proj:a") is not None
    b = get_entity(conn, "proj:b")
    assert "Dup" not in _j.loads(b["metadata"] or "{}").get("aliases", [])
    conn.close()


def test_resolve_review_merge(temp_db_path):
    from knowledge_weaver.db import count_pending_reviews, get_entity
    from knowledge_weaver.tools import resolve_review

    conn = init_db(temp_db_path)
    _add_entity(conn, "proj:b", "project", "Canonical", "c", BASE)
    _add_entity(conn, "proj:a", "project", "Dup", "d", _vec_at_cos(0.95))
    rid = insert_review(conn, kind="merge", new_entity_id="proj:a",
                        candidate_id="proj:b", entity_type="project",
                        score=0.85, reason="review:vector")
    out = resolve_review(conn, review_id=rid, action="merge")
    assert out["ok"] and out["action"] == "merged"
    assert get_entity(conn, "proj:a") is None
    assert count_pending_reviews(conn) == 0
    conn.close()


def test_resolve_review_reject_keeps_both(temp_db_path):
    from knowledge_weaver.db import count_pending_reviews, get_entity
    from knowledge_weaver.tools import resolve_review

    conn = init_db(temp_db_path)
    _add_entity(conn, "proj:b", "project", "Canonical", "c", BASE)
    _add_entity(conn, "proj:a", "project", "MaybeDup", "d", _vec_at_cos(0.85))
    rid = insert_review(conn, kind="merge", new_entity_id="proj:a",
                        candidate_id="proj:b", entity_type="project",
                        score=0.85, reason="review:vector")
    out = resolve_review(conn, review_id=rid, action="reject")
    assert out["action"] == "kept_separate"
    assert get_entity(conn, "proj:a") is not None
    assert get_entity(conn, "proj:b") is not None
    assert count_pending_reviews(conn) == 0
    conn.close()


def test_reconcile_registry_detects_deletion(temp_db_path, tmp_path):
    from knowledge_weaver.db import count_pending_reviews, insert_entity as _ie, \
        snapshot_registered_slugs
    from knowledge_weaver.extractor import generate_entity_id
    from knowledge_weaver.tools import reconcile_registry_deletions

    conn = init_db(temp_db_path)
    alpha = generate_entity_id("project", "Alpha")
    beta = generate_entity_id("project", "Beta")
    _ie(conn, {"id": beta, "type": "project", "name": "Beta", "summary": "s",
               "importance": 0.5, "first_seen": "2026-01-01",
               "last_seen": "2026-01-01", "day_count": 5})
    snapshot_registered_slugs(conn, {alpha, beta})  # baseline has both

    reg = tmp_path / "MEMORY.md"
    reg.write_text("# M\n\n## 项目标准\n\n- **Alpha** `slug: alpha`\n", encoding="utf-8")
    out = reconcile_registry_deletions(conn, registry_path=str(reg), min_day_count=3)
    assert beta in out["queued_for_review"]
    assert count_pending_reviews(conn) == 1
    conn.close()


def test_calibrate_resolution(temp_db_path):
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
    from calibrate_resolution import calibrate

    conn = init_db(temp_db_path)
    _add_entity(conn, "proj:a", "project", "Alpha", "a", _vec_at_cos(1.0))
    _add_entity(conn, "proj:b", "project", "Alpha2", "b", _vec_at_cos(0.97))
    conn.close()
    s = calibrate(temp_db_path)
    assert s["vectors"] == 2
    assert s["top_pairs"] and s["top_pairs"][0][0] >= 0.95


# --------------------------------------------------------------------------- #
# Identifier guard (real-data finding: short structured ids over-merge)        #
# --------------------------------------------------------------------------- #
def test_version_identifier_demoted_to_review(temp_db_path):
    conn = init_db(temp_db_path)
    _add_entity(conn, "tech:v2_9_0", "tech", "v2.9.0", "tool A version", BASE)
    emb = MockEmbedder({"kw version": _vec_at_cos(0.95)})
    res = resolve_entity(conn, new_id="tech:v0_2_0", entity_type="tech",
                         name="v0.2.0", summary="kw version", embedder=emb)
    assert res.action == "review"  # identifier guard blocks auto-merge
    conn.close()


def test_course_code_demoted_to_review(temp_db_path):
    conn = init_db(temp_db_path)
    _add_entity(conn, "tech:comp7240", "tech", "COMP7240", "a course", BASE)
    emb = MockEmbedder({"another course": _vec_at_cos(0.99)})
    res = resolve_entity(conn, new_id="tech:comp7940", entity_type="tech",
                         name="COMP7940", summary="another course", embedder=emb)
    assert res.action == "review"
    conn.close()


def test_filename_demoted_to_review(temp_db_path):
    conn = init_db(temp_db_path)
    _add_entity(conn, "tech:a", "tech", "test_registry.py", "a test file", BASE)
    emb = MockEmbedder({"another file": _vec_at_cos(0.96)})
    res = resolve_entity(conn, new_id="tech:b", entity_type="tech",
                         name="utils.py", summary="another file", embedder=emb)
    assert res.action == "review"
    conn.close()


def test_short_name_demoted_to_review(temp_db_path):
    conn = init_db(temp_db_path)
    _add_entity(conn, "tech:x1", "tech", "AI", "a field", BASE)
    emb = MockEmbedder({"ml field": _vec_at_cos(0.95)})
    res = resolve_entity(conn, new_id="tech:x2", entity_type="tech",
                         name="ML", summary="ml field", embedder=emb)
    assert res.action == "review"  # len("ML") <= 3
    conn.close()


def test_real_word_vector_still_merges(temp_db_path):
    """Non-identifier words with high cosine still auto-merge (the cross-language case)."""
    conn = init_db(temp_db_path)
    _add_entity(conn, "tech:homebrain", "tech", "HomeBrain", "smart home hub", BASE)
    emb = MockEmbedder({"jiating danao": _vec_at_cos(0.95)})
    res = resolve_entity(conn, new_id="tech:jiatingdanao", entity_type="tech",
                         name="家庭大脑", summary="jiating danao", embedder=emb)
    assert res.action == "merge" and res.reason == "auto:vector"
    conn.close()


def test_digit_variant_pair():
    from knowledge_weaver.resolver import _digit_variant_pair
    assert _digit_variant_pair("v0.2.0", "v2.9.0")        # versions -> distinct
    assert _digit_variant_pair("COMP7940", "COMP7240")    # course codes -> distinct
    assert _digit_variant_pair("8080", "8081")            # ports -> distinct
    assert _digit_variant_pair("glm-5.1", "glm-5.2")      # model versions -> distinct
    assert not _digit_variant_pair("HomeBrain", "家庭大脑")  # real alias -> mergeable
    assert not _digit_variant_pair("sqlite-vec", "sqlite-vec")  # identical
    assert not _digit_variant_pair("HomeBrain", "CodeForge")    # unrelated


def test_insert_review_not_requeued_after_dismiss(temp_db_path):
    from knowledge_weaver.db import (
        count_pending_reviews,
        init_db,
        insert_review,
        set_review_status,
    )
    conn = init_db(temp_db_path)
    rid = insert_review(conn, kind="merge", new_entity_id="ent:a",
                        candidate_id="ent:b", entity_type="tech", score=0.9, reason="t")
    assert count_pending_reviews(conn) == 1
    set_review_status(conn, rid, "dismissed")
    rid2 = insert_review(conn, kind="merge", new_entity_id="ent:a",
                         candidate_id="ent:b", entity_type="tech", score=0.9, reason="t")
    assert rid2 == rid                        # returns the dismissed row
    assert count_pending_reviews(conn) == 0   # NOT re-queued
    conn.close()


def test_digit_variant_letter_suffix():
    from knowledge_weaver.resolver import _digit_variant_pair
    assert _digit_variant_pair("GPT-4", "GPT-4o")      # #3: letter-suffixed -> distinct
    assert _digit_variant_pair("ESP32", "ESP32S")       # 1-char tail add -> distinct
    assert not _digit_variant_pair("HomeBrain", "家庭大脑")   # real alias unaffected
    assert not _digit_variant_pair("FastAPI", "Flask")        # unrelated, no digits
    assert not _digit_variant_pair("v1", "v1beta")            # 4-char tail: too different to flag


def test_insert_review_not_requeued_after_human_reject(temp_db_path):
    from knowledge_weaver.db import (
        count_pending_reviews, init_db, insert_review, set_review_status,
    )
    conn = init_db(temp_db_path)
    rid = insert_review(conn, kind="merge", new_entity_id="ent:a",
                        candidate_id="ent:b", entity_type="tech", score=0.9, reason="t")
    set_review_status(conn, rid, "rejected")              # human kw_resolve reject
    # same pair, AND the reversed direction, must both be recognized as decided
    assert insert_review(conn, kind="merge", new_entity_id="ent:a",
                         candidate_id="ent:b", entity_type="tech", score=0.9, reason="t") == rid
    assert insert_review(conn, kind="merge", new_entity_id="ent:b",
                         candidate_id="ent:a", entity_type="tech", score=0.9, reason="t") == rid
    assert count_pending_reviews(conn) == 0               # not re-queued
    conn.close()
