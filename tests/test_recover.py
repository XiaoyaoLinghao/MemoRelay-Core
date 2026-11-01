"""Recovery of rebuild-lost entities from the pre-swap DB: missing detection,
porting with accumulated day_count + vector, and tech noise filtering."""
from knowledge_weaver.db import (
    get_entity,
    get_entity_vector,
    init_db,
    insert_entity,
    insert_relation,
    upsert_entity_vector,
)
from knowledge_weaver.embedder import DEFAULT_DIMENSION
from knowledge_weaver.linker import generate_relation_id
from knowledge_weaver.recover import (
    missing_entities,
    port_entities,
    rule_keep_tech,
    signal_keep_tech,
)

DIM = DEFAULT_DIMENSION


def _add(conn, eid, etype, name, day_count=3, vec=None):
    insert_entity(conn, {
        "id": eid, "type": etype, "name": name, "summary": name + " 摘要",
        "importance": 0.6, "first_seen": "2026-01-01", "last_seen": "2026-02-01",
        "day_count": day_count,
    })
    if vec is not None:
        upsert_entity_vector(conn, eid, vec)


def test_recover_missing_decisions(tmp_path):
    old = init_db(str(tmp_path / "old.db"))
    new = init_db(str(tmp_path / "new.db"))
    _add(old, "dec:wang_interview", "decision", "建议WANG进入下一轮面试",
         day_count=7, vec=[0.1] * DIM)
    _add(old, "dec:shared", "decision", "共同决策", day_count=4)
    _add(new, "dec:shared", "decision", "共同决策")          # already present in new

    rows = missing_entities(old, new, ["decision"])
    assert {r["id"] for r in rows} == {"dec:wang_interview"}   # only the truly-missing one

    res = port_entities(old, new, rows)
    assert res["ported"] == 1 and res["with_vector"] == 1
    ported = get_entity(new, "dec:wang_interview")
    assert ported is not None
    assert ported["day_count"] == 7                            # accumulated signal carried
    assert get_entity_vector(new, "dec:wang_interview") is not None  # vector carried
    old.close()
    new.close()


def test_recover_dry_run_writes_nothing(tmp_path):
    old = init_db(str(tmp_path / "old.db"))
    new = init_db(str(tmp_path / "new.db"))
    _add(old, "dec:x", "decision", "某历史决策", day_count=5)
    rows = missing_entities(old, new, ["decision"])
    res = port_entities(old, new, rows, dry_run=True)
    assert res["candidates"] == 1 and res["ported"] == 1
    assert get_entity(new, "dec:x") is None                   # dry-run: nothing written
    old.close()
    new.close()


def test_rule_keep_tech_filters_noise():
    assert rule_keep_tech({"name": "PCA9685"})        # chip model: keep
    assert rule_keep_tech({"name": "glm-5.1"})        # version-like model name: keep
    assert rule_keep_tech({"name": "sqlite-vec"})     # real lib: keep
    assert rule_keep_tech({"name": "AWS"})            # real abbrev: name rule no longer drops it
    assert rule_keep_tech({"name": "SOUL"})           # short all-caps now deferred to signal gate
    assert not rule_keep_tech({"name": "T6"})         # ≤2 chars: drop
    assert not rule_keep_tech({"name": "config.json"})  # bare filename: drop


def test_signal_keep_tech_gate(tmp_path):
    old = init_db(str(tmp_path / "old.db"))
    # recurring tech (day_count>=2) -> keep
    _add(old, "tech:sqlite_vec", "tech", "sqlite-vec", day_count=4)
    # one-off but connected by an edge -> keep
    _add(old, "tech:pca9685", "tech", "PCA9685", day_count=1)
    _add(old, "proj:home", "project", "HomeBrain", day_count=3)
    insert_relation(old, {"id": generate_relation_id("proj:home", "tech:pca9685", "使用"),
                          "from_entity": "proj:home", "to_entity": "tech:pca9685",
                          "rel_type": "使用", "weight": 1.0, "evidence": "co_occurrence"})
    # real recurring abbrev (#5): short all-caps WITH signal -> keep, not silently dropped
    _add(old, "tech:aws", "tech", "AWS", day_count=6)
    # one-off AND isolated (day_count=1, no edge) -> drop as low-value noise
    _add(old, "tech:oneoff", "tech", "SomeOneOffLib", day_count=1)
    # one-off short all-caps noise (day_count=1, isolated) -> drop
    _add(old, "tech:soul", "tech", "SOUL", day_count=1)
    # bare filename -> drop regardless of signal
    _add(old, "tech:cfg", "tech", "config.json", day_count=8)

    assert signal_keep_tech(dict(get_entity(old, "tech:sqlite_vec")), old)   # recurred
    assert signal_keep_tech(dict(get_entity(old, "tech:pca9685")), old)      # connected
    assert signal_keep_tech(dict(get_entity(old, "tech:aws")), old)          # recurring abbrev kept
    assert not signal_keep_tech(dict(get_entity(old, "tech:oneoff")), old)   # one-off isolated
    assert not signal_keep_tech(dict(get_entity(old, "tech:soul")), old)     # one-off all-caps noise
    assert not signal_keep_tech(dict(get_entity(old, "tech:cfg")), old)      # filename, even w/ signal
    old.close()
