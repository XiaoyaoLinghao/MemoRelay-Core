"""W3.2 tests: facts backfill markdown surgery (deterministic; LLM mocked)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from backfill_facts import _validate, backfill_file, backfill_text  # noqa: E402

DAY_MD = """---
date: "2026-05-20"
---

# 2026-05-20

## 10:00

### 摘要

用户讨论了 HomeBrain 项目，决定后端用 FastAPI。
[关键决策] 后端用 FastAPI

## 11:00

### 摘要

本时段无实质内容（仅系统心跳轮询与每日状态检查，无用户实质输入）。
"""

FACTS = [
    {"type": "project", "name": "HomeBrain", "summary": "智能家庭中枢"},
    {"type": "decision", "name": "后端用 FastAPI", "summary": "选 FastAPI"},
]


def _mock_extract(summary):
    return FACTS


def test_backfill_inserts_block_for_substantive_only():
    new_text, n = backfill_text(DAY_MD, _mock_extract)
    assert n == 1  # 10:00 backfilled; 11:00 (empty) skipped
    assert "### 结构化事实" in new_text
    assert '"name": "HomeBrain"' in new_text
    # the empty 11:00 block did NOT get a facts block
    block_11 = new_text.split("## 11:00", 1)[1]
    assert "### 结构化事实" not in block_11


def test_backfill_idempotent():
    once, n1 = backfill_text(DAY_MD, _mock_extract)
    twice, n2 = backfill_text(once, _mock_extract)
    assert n1 == 1 and n2 == 0  # second pass inserts nothing
    assert once == twice


def test_backfill_file_writes_and_backs_up(tmp_path):
    f = tmp_path / "2026-05-20.md"
    f.write_text(DAY_MD, encoding="utf-8")
    n = backfill_file(str(f), _mock_extract)
    assert n == 1
    assert (tmp_path / "2026-05-20.md.bak").exists()  # backup made
    assert "### 结构化事实" in f.read_text(encoding="utf-8")


def test_backfill_dry_run_no_write(tmp_path):
    f = tmp_path / "d.md"
    f.write_text(DAY_MD, encoding="utf-8")
    backfill_file(str(f), lambda s: FACTS, dry_run=True)
    assert not (tmp_path / "d.md.bak").exists()
    assert "### 结构化事实" not in f.read_text(encoding="utf-8")


def test_validate_filters():
    bad = [
        {"type": "project", "name": "A"},
        {"type": "bogus", "name": "B"},        # bad type
        {"type": "tech", "name": ""},          # empty name
        {"type": "project", "name": "A"},      # dup
        {"type": "tech", "name": "C", "aliases": ["c1", "", 3]},
    ]
    out = _validate(bad)
    names = [(f["type"], f["name"]) for f in out]
    assert names == [("project", "A"), ("tech", "C")]
    assert out[1]["aliases"] == ["c1"]


def test_backfill_skips_when_facts_already_present():
    once, _ = backfill_text(DAY_MD, _mock_extract)
    # craft a different extractor; should not be called because block has facts
    called = {"n": 0}

    def counting(summary):
        called["n"] += 1
        return FACTS

    _, n = backfill_text(once, counting)
    assert n == 0
