"""W3 tests: candidate rebuild + diff (read-only on prod). Deterministic."""
import os
import sys

from knowledge_weaver.db import get_entity, init_db
from knowledge_weaver.extractor import generate_entity_id

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from rebuild_from_facts import rebuild_and_diff  # noqa: E402

FACTS_MD = '''---
date: "2026-06-01"
---

# 2026-06-01

## 10:00

### 摘要

讨论了 HomeBrain 项目与后端选型。
[关键决策] 后端用 FastAPI

### 结构化事实

```json
[
  {"type": "project", "name": "HomeBrain", "summary": "智能家庭中枢", "aliases": ["家庭大脑"]},
  {"type": "decision", "name": "后端用 FastAPI", "summary": "选 FastAPI 作后端"}
]
```
'''

PLAIN_MD = '''---
date: "2026-06-02"
---

# 2026-06-02

## 11:00

### 摘要

用户决定采用 PostgreSQL 数据库。
[关键决策] 采用 PostgreSQL
'''


def test_rebuild_and_diff(tmp_path, monkeypatch):
    from knowledge_weaver.pipeline import run_consolidation

    mem = tmp_path / "mem"
    mem.mkdir()
    (mem / "2026-06-01.md").write_text(FACTS_MD, encoding="utf-8")
    (mem / "2026-06-02.md").write_text(PLAIN_MD, encoding="utf-8")

    # baseline "current" DB built via regex
    monkeypatch.delenv("KNOWLEDGE_WEAVER_EXTRACT_MODE", raising=False)
    current = str(tmp_path / "cur.db")
    run_consolidation(current, memory_dir=str(mem), embedder=None)

    candidate = str(tmp_path / "cand.db")
    r = rebuild_and_diff(str(mem), candidate, current_db=current,
                         json_mode=True, embedder=None)

    assert r["status"] == "ok"
    assert r["files_processed"] == 2
    # candidate (json mode) carries the clean HomeBrain project from the facts block
    conn = init_db(candidate)
    assert get_entity(conn, generate_entity_id("project", "HomeBrain")) is not None
    conn.close()
    # diff is well-formed; json path produces entities the regex baseline didn't
    assert isinstance(r["only_in_candidate"], list)
    assert isinstance(r["only_in_current"], list)
    assert r["candidate_total"] > 0


def test_rebuild_does_not_touch_current(tmp_path, monkeypatch):
    from knowledge_weaver.pipeline import run_consolidation

    mem = tmp_path / "mem"
    mem.mkdir()
    (mem / "2026-06-01.md").write_text(FACTS_MD, encoding="utf-8")
    monkeypatch.delenv("KNOWLEDGE_WEAVER_EXTRACT_MODE", raising=False)
    current = str(tmp_path / "cur.db")
    run_consolidation(current, memory_dir=str(mem), embedder=None)
    before = os.path.getmtime(current)

    rebuild_and_diff(str(mem), str(tmp_path / "cand.db"), current_db=current,
                     json_mode=True, embedder=None)
    assert os.path.getmtime(current) == before  # current DB untouched
