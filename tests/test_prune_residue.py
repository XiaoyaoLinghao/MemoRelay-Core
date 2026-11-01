"""T2.1: prune bare version numbers / internal codes mis-extracted as tech,
without touching real tech."""
from knowledge_weaver.db import get_entity, init_db, insert_entity
from knowledge_weaver.prune_residue import prune_residue, residue_kind


def _t(conn, eid, name):
    insert_entity(conn, {
        "id": eid, "type": "tech", "name": name, "summary": name,
        "importance": 0.5, "first_seen": "2026-01-01", "last_seen": "2026-01-01",
    })


def test_residue_kind():
    assert residue_kind("v2.9.0") == "version"
    assert residue_kind("2.1.140") == "version"
    assert residue_kind("v0.2") == "version"
    assert residue_kind("P1-1") == "internal-tag"
    assert residue_kind("WI2") == "internal-tag"
    assert residue_kind("Track2") == "internal-tag"
    assert residue_kind("Phase1-3") == "internal-tag"
    # real tech must NOT be flagged
    assert residue_kind("sqlite-vec") is None
    assert residue_kind("PCA9685") is None       # chip model (letters+digits, no dots)
    assert residue_kind("glm-5.1") is None        # model name (has letters)
    assert residue_kind("FastAPI") is None
    # #2 regression: 'T\\d+' / bare 'P\\d+' removed — real models must survive
    assert residue_kind("T5") is None             # Google T5 transformer
    assert residue_kind("T6") is None
    assert residue_kind("P1") is None             # bare P-code no longer auto-deleted


def test_prune_residue_keeps_real_tech(temp_db_path):
    conn = init_db(temp_db_path)
    _t(conn, "tech:v290", "v2.9.0")
    _t(conn, "tech:p11", "P1-1")
    _t(conn, "tech:sqlitevec", "sqlite-vec")
    _t(conn, "tech:pca", "PCA9685")

    r = prune_residue(conn)
    assert r["candidates"] == 2 and r["by_kind"] == {"version": 1, "internal-tag": 1}
    assert get_entity(conn, "tech:v290") is None          # version pruned
    assert get_entity(conn, "tech:p11") is None            # tag pruned
    assert get_entity(conn, "tech:sqlitevec") is not None  # real tech kept
    assert get_entity(conn, "tech:pca") is not None        # chip model kept
    conn.close()
