"""Tests for the retrieval+project-node refinement: jieba FTS, RRF fusion,
registered-project node guarantee. Deterministic."""
from knowledge_weaver.db import (
    build_fts_match,
    init_db,
    insert_entity,
    jieba_tokenize,
    search_entities_fts,
    upsert_entity_vector,
)
from knowledge_weaver.extractor import generate_entity_id

DIM = 1024


def _ent(conn, eid, etype, name, summary):
    insert_entity(conn, {
        "id": eid, "type": etype, "name": name, "summary": summary,
        "importance": 0.5, "first_seen": "2026-01-01", "last_seen": "2026-01-01",
    })


# --------------------------------------------------------------------------- #
# jieba Chinese FTS                                                            #
# --------------------------------------------------------------------------- #
def test_jieba_tokenize_segments_chinese():
    out = jieba_tokenize("家庭大脑中枢")
    assert " " in out and "家庭" in out.split()


def test_build_fts_match_or_joined():
    m = build_fts_match("家庭大脑")
    assert " OR " in m and '"家庭"' in m


def test_chinese_fts_search_finds_entity(temp_db_path):
    conn = init_db(temp_db_path)
    _ent(conn, "proj:hb", "project", "家庭大脑中枢", "智能家庭自动化平台")
    # Chinese keyword that is a *segment* of the name — unicode61 alone would miss it.
    rows = search_entities_fts(conn, "家庭", limit=5)
    assert any(r["id"] == "proj:hb" for r in rows)
    conn.close()


# --------------------------------------------------------------------------- #
# RRF hybrid in knowledge_search                                              #
# --------------------------------------------------------------------------- #
class _Emb:
    def embed(self, q):
        return [1.0] + [0.0] * (DIM - 1)

    def embed_batch(self, ts):
        return [self.embed(t) for t in ts]


def test_knowledge_search_rrf_hybrid(temp_db_path):
    # tech type (not subject to the project recurrence gate)
    from knowledge_weaver.tools import knowledge_search

    conn = init_db(temp_db_path)
    _ent(conn, "tech:a", "tech", "Alpha", "alpha tech summary")
    upsert_entity_vector(conn, "tech:a", [1.0] + [0.0] * (DIM - 1))  # matches query vec
    r = knowledge_search(conn, query="Alpha", embedder=_Emb(), max_results=5)
    assert r["total_hits"] >= 1
    assert any(x["entity_id"] == "tech:a" for x in r["results"])
    conn.close()


def test_knowledge_search_fts_only_no_embedder(temp_db_path):
    from knowledge_weaver.tools import knowledge_search

    conn = init_db(temp_db_path)
    _ent(conn, "tech:a", "tech", "Alpha", "alpha tech")
    r = knowledge_search(conn, query="Alpha", embedder=None, max_results=5)
    assert any(x["entity_id"] == "tech:a" for x in r["results"])
    conn.close()


# --------------------------------------------------------------------------- #
# Registered-project node guarantee                                           #
# --------------------------------------------------------------------------- #
def test_ensure_registered_project_entities(temp_db_path, tmp_path):
    from knowledge_weaver.db import get_entity
    from knowledge_weaver.tools import ensure_registered_project_entities

    reg = tmp_path / "MEMORY.md"
    reg.write_text("# M\n\n## 项目标准\n\n- **HomeBrain** `slug: homebrain`\n",
                   encoding="utf-8")
    conn = init_db(temp_db_path)
    created = ensure_registered_project_entities(conn, registry_path=str(reg))
    hb = generate_entity_id("project", "HomeBrain")
    assert hb in created
    e = get_entity(conn, hb)
    assert e is not None and e["type"] == "project" and e["day_count"] == 2
    # idempotent
    assert ensure_registered_project_entities(conn, registry_path=str(reg)) == []
    conn.close()
