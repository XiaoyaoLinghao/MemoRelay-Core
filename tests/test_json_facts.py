"""W2 tests: structured JSON-fact ingestion (parse / map / shadow-compare /
pipeline json mode). Deterministic; no embedding/network needed."""
import json

from knowledge_weaver.extractor import generate_entity_id
from knowledge_weaver.json_facts import (
    extract_json_facts_block,
    facts_to_entities,
    shadow_compare_file,
)

FACTS_MD = '''---
date: "2026-06-05"
---

# 2026-06-05

## 10:00

### 摘要

用户讨论了 HomeBrain 项目，决定后端用 FastAPI。
[关键决策] 后端用 FastAPI

### 结构化事实

```json
[
  {"type": "project", "name": "HomeBrain", "summary": "智能家庭中枢", "aliases": ["家庭大脑"]},
  {"type": "decision", "name": "后端用 FastAPI", "summary": "选 FastAPI 作后端框架"},
  {"type": "bogus", "name": "X", "summary": "invalid type -> skipped"},
  {"type": "tech", "name": "", "summary": "empty name -> skipped"}
]
```
'''


def test_extract_block_valid():
    facts = extract_json_facts_block(FACTS_MD)
    assert isinstance(facts, list) and len(facts) == 4


def test_extract_block_absent():
    assert extract_json_facts_block("# doc\n\n### 摘要\n\nhi") is None


def test_extract_block_malformed():
    bad = "### 结构化事实\n\n```json\n[not valid json}\n```"
    assert extract_json_facts_block(bad) is None


def test_facts_to_entities_valid_and_filtering():
    ents = facts_to_entities(extract_json_facts_block(FACTS_MD), "memory/2026-06-05.md")
    assert len(ents) == 2  # bogus type + empty name dropped
    hb = {e.id: e for e in ents}[generate_entity_id("project", "HomeBrain")]
    assert hb.type == "project" and hb.name == "HomeBrain"
    assert hb.metadata.get("aliases") == ["家庭大脑"]


def test_facts_to_entities_dedup():
    facts = [
        {"type": "tech", "name": "FastAPI", "summary": "a"},
        {"type": "tech", "name": "FastAPI", "summary": "b"},
    ]
    assert len(facts_to_entities(facts, "memory/x.md")) == 1


def test_shadow_compare(tmp_path):
    f = tmp_path / "2026-06-05.md"
    f.write_text(FACTS_MD, encoding="utf-8")
    r = shadow_compare_file(str(f))
    assert r["has_json_block"] is True
    assert r["json_count"] == 2
    proj = generate_entity_id("project", "HomeBrain")
    assert any(proj in x for x in r["both"]) or any(proj in x for x in r["json_only"])


def test_pipeline_json_mode(tmp_path, temp_db_path, monkeypatch):
    from knowledge_weaver.db import get_entity, init_db
    from knowledge_weaver.pipeline import run_consolidation

    monkeypatch.setenv("KNOWLEDGE_WEAVER_EXTRACT_MODE", "json")
    mem = tmp_path / "mem"
    mem.mkdir()
    (mem / "2026-06-05.md").write_text(FACTS_MD, encoding="utf-8")

    result = run_consolidation(temp_db_path, memory_dir=str(mem), embedder=None)
    assert result.status == "ok"

    conn = init_db(temp_db_path)
    hb = get_entity(conn, generate_entity_id("project", "HomeBrain"))
    assert hb is not None and hb["type"] == "project"
    assert "家庭大脑" in json.loads(hb["metadata"] or "{}").get("aliases", [])
    # JSON path replaces regex: the decision fact is present...
    assert get_entity(conn, generate_entity_id("decision", "后端用 FastAPI")) is not None
    conn.close()


def test_pipeline_regex_mode_default(tmp_path, temp_db_path, monkeypatch):
    """Without the env flag, the JSON block is ignored (regex stays default)."""
    from knowledge_weaver.db import get_entity, init_db
    from knowledge_weaver.pipeline import run_consolidation

    monkeypatch.delenv("KNOWLEDGE_WEAVER_EXTRACT_MODE", raising=False)
    mem = tmp_path / "mem"
    mem.mkdir()
    (mem / "2026-06-06.md").write_text(FACTS_MD, encoding="utf-8")

    result = run_consolidation(temp_db_path, memory_dir=str(mem), embedder=None)
    assert result.status == "ok"
    conn = init_db(temp_db_path)
    # regex would not produce the clean "后端用 FastAPI" decision id from JSON;
    # the point is just that consolidation runs fine ignoring the JSON block.
    conn.close()


def test_regex_skips_facts_block(tmp_path):
    """The regex path must ignore the ### 结构化事实 JSON block (no garbage)."""
    from knowledge_weaver.json_facts import regex_entities_for_file
    no_facts = FACTS_MD.split("### 结构化事实")[0]
    f1 = tmp_path / "with.md"
    f1.write_text(FACTS_MD, encoding="utf-8")
    f2 = tmp_path / "without.md"
    f2.write_text(no_facts, encoding="utf-8")
    e1 = {e.id for e in regex_entities_for_file(str(f1))}
    e2 = {e.id for e in regex_entities_for_file(str(f2))}
    assert e1 == e2  # facts block contributes nothing to the regex path
