"""Tests for the consolidation pipeline module."""

import os
import tempfile

import pytest

from knowledge_weaver.db import get_entity, get_manifest, init_db
from knowledge_weaver.pipeline import ConsolidationResult, compute_file_hash, run_consolidation

FIXTURES = os.path.join(os.path.dirname(__file__), "..", "fixtures")


class MockEmbedder:
    """Mock embedder that returns fixed vectors for testing."""

    def embed(self, text: str) -> list[float]:
        return [0.1] * 1024

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * 1024 for _ in texts]

    def close(self):
        pass


# ---------------------------------------------------------------------------
# compute_file_hash
# ---------------------------------------------------------------------------


class TestComputeFileHash:
    def test_hash_consistency(self, tmp_path):
        f = tmp_path / "test.md"
        f.write_text("hello world", encoding="utf-8")
        h1 = compute_file_hash(str(f))
        h2 = compute_file_hash(str(f))
        assert h1 == h2
        assert len(h1) == 64  # SHA256 hex

    def test_hash_changes_with_content(self, tmp_path):
        f = tmp_path / "test.md"
        f.write_text("content A", encoding="utf-8")
        h1 = compute_file_hash(str(f))
        f.write_text("content B", encoding="utf-8")
        h2 = compute_file_hash(str(f))
        assert h1 != h2

    def test_hash_nonexistent_file(self):
        h = compute_file_hash("/nonexistent/path.md")
        assert h == ""


# ---------------------------------------------------------------------------
# run_consolidation — basic
# ---------------------------------------------------------------------------


class TestRunConsolidation:
    def test_run_consolidation(self, temp_db_path):
        """Run on sample fixtures, verify entities created."""
        result = run_consolidation(
            db_path=temp_db_path,
            memory_dir=FIXTURES,
            embedder=MockEmbedder(),
        )
        assert result.files_processed >= 1
        assert result.entities_created > 0
        assert result.status == "ok"
        assert result.errors == []

        # Verify entities exist in DB
        conn = init_db(temp_db_path)
        rows = conn.execute("SELECT COUNT(*) as cnt FROM entities").fetchone()
        assert rows["cnt"] > 0
        conn.close()

    def test_consolidation_skips_unchanged(self, temp_db_path):
        """Second run should skip processed files (idempotent)."""
        embedder = MockEmbedder()

        # First run
        result1 = run_consolidation(temp_db_path, FIXTURES, embedder)
        assert result1.files_processed >= 1
        first_created = result1.entities_created

        # Second run — same files, should skip
        result2 = run_consolidation(temp_db_path, FIXTURES, embedder)
        assert result2.files_skipped >= result1.files_processed
        assert result2.files_processed == 0
        assert result2.entities_created == 0

    def test_consolidation_detects_changed(self, temp_db_path, tmp_path):
        """Modified file should be re-processed."""
        # Create a single test file
        test_file = tmp_path / "2026-05-20.md"
        test_file.write_text(
            "---\ntitle: test\ndate: 2026-05-20\n---\n\n## 核心要点\n- 测试实体A\n",
            encoding="utf-8",
        )

        # First run
        result1 = run_consolidation(temp_db_path, str(tmp_path), MockEmbedder())
        assert result1.files_processed == 1

        # Modify file
        test_file.write_text(
            "---\ntitle: test\ndate: 2026-05-20\n---\n\n## 核心要点\n- 测试实体A修改版\n",
            encoding="utf-8",
        )

        # Second run — should detect change
        result2 = run_consolidation(temp_db_path, str(tmp_path), MockEmbedder())
        assert result2.files_processed == 1
        assert result2.files_skipped == 0

    def test_consolidation_empty_dir(self, temp_db_path):
        """No md files → empty result."""
        with tempfile.TemporaryDirectory() as empty_dir:
            result = run_consolidation(temp_db_path, empty_dir, MockEmbedder())
            assert result.files_processed == 0
            assert result.files_skipped == 0
            assert result.entities_created == 0
            assert result.status == "ok"

    def test_consolidation_multiple_files(self, temp_db_path):
        """Process 3 sample files, verify cross-day linking."""
        result = run_consolidation(temp_db_path, FIXTURES, MockEmbedder())

        assert result.files_processed == 3
        assert result.entities_created > 0
        assert result.relations_created > 0
        assert result.status == "ok"

        conn = init_db(temp_db_path)

        # ExampleProject project should exist (appears in all 3 files)
        hb = conn.execute(
            "SELECT * FROM entities WHERE id LIKE '%exampleproject%'"
        ).fetchone()
        assert hb is not None, "ExampleProject project entity should exist"

        # ExampleProject should have day_count > 1 (seen across multiple days)
        assert hb["day_count"] >= 2, f"Expected day_count >= 2, got {hb['day_count']}"

        # Cross-day relations should exist
        relations = conn.execute(
            "SELECT COUNT(*) as cnt FROM relations"
        ).fetchone()
        assert relations["cnt"] > 0

        # All 3 manifest entries should exist
        manifest_count = conn.execute(
            "SELECT COUNT(*) as cnt FROM daily_manifest"
        ).fetchone()
        assert manifest_count["cnt"] == 3

        conn.close()


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestConsolidationEdgeCases:
    def test_no_embedder(self, temp_db_path):
        """Pipeline works without an embedder (embeddings skipped)."""
        result = run_consolidation(
            db_path=temp_db_path,
            memory_dir=FIXTURES,
            embedder=None,
        )
        assert result.files_processed >= 1
        assert result.entities_created > 0
        assert result.status == "ok"

    def test_manifest_updated_after_run(self, temp_db_path):
        """Daily manifest is properly recorded."""
        run_consolidation(temp_db_path, FIXTURES, MockEmbedder())

        conn = init_db(temp_db_path)
        manifest = get_manifest(conn, "2026-05-24")
        assert manifest is not None
        assert manifest["file_hash"] != ""
        assert manifest["entity_count"] > 0
        assert manifest["status"] == "ok"
        conn.close()

    def test_importance_scored(self, temp_db_path):
        """Entities should have non-zero importance scores."""
        run_consolidation(temp_db_path, FIXTURES, MockEmbedder())

        conn = init_db(temp_db_path)
        rows = conn.execute(
            "SELECT id, importance FROM entities WHERE importance > 0 LIMIT 5"
        ).fetchall()
        assert len(rows) > 0, "At least some entities should have non-zero importance"
        for row in rows:
            assert 0.0 <= row["importance"] <= 1.0
        conn.close()

    def test_non_date_files_ignored(self, temp_db_path, tmp_path):
        """Files not matching YYYY-MM-DD.md should be ignored."""
        # Create a valid file and an invalid file
        (tmp_path / "2026-05-20.md").write_text(
            "---\ntitle: test\ndate: 2026-05-20\n---\n\n## 核心要点\n- 测试\n",
            encoding="utf-8",
        )
        (tmp_path / "notes.md").write_text("random notes", encoding="utf-8")
        (tmp_path / ".hidden.md").write_text("hidden", encoding="utf-8")

        result = run_consolidation(temp_db_path, str(tmp_path), MockEmbedder())
        assert result.files_processed == 1

    def test_consolidation_result_dataclass(self):
        """ConsolidationResult defaults are correct."""
        r = ConsolidationResult()
        assert r.status == "ok"
        assert r.files_processed == 0
        assert r.files_skipped == 0
        assert r.files_failed == 0
        assert r.entities_created == 0
        assert r.entities_updated == 0
        assert r.relations_created == 0
        assert r.errors == []

def test_no_self_loop_relations_after_consolidation(temp_db_path):
    """After consolidation no relation should have from_entity == to_entity."""
    import os, sqlite3
    from knowledge_weaver.pipeline import run_consolidation
    FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "..", "fixtures")
    run_consolidation(temp_db_path, FIXTURES_DIR, embedder=None)
    c = sqlite3.connect(temp_db_path)
    c.row_factory = sqlite3.Row
    n = c.execute("SELECT COUNT(*) FROM relations WHERE from_entity=to_entity").fetchone()[0]
    c.close()
    assert n == 0, f"Found {n} self-loop relations"

def test_exampleproject_exists_as_clean_project(temp_db_path):
    """ExampleProject must exist as an independent project entity."""
    import os, sqlite3
    from knowledge_weaver.pipeline import run_consolidation
    FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "..", "fixtures")
    run_consolidation(temp_db_path, FIXTURES_DIR, embedder=None)
    c = sqlite3.connect(temp_db_path); c.row_factory = sqlite3.Row
    rows = c.execute("SELECT id, name FROM entities WHERE type='project'").fetchall()
    names = [r["name"] for r in rows]
    c.close()
    assert "ExampleProject" in names, f"Got project names: {names}"


def test_tech_entities_have_no_cjk_noise(temp_db_path):
    """Tech entity names must not contain CJK-number noise from version regex."""
    import os, sqlite3
    from knowledge_weaver.pipeline import run_consolidation
    FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "..", "fixtures")
    run_consolidation(temp_db_path, FIXTURES_DIR, embedder=None)
    c = sqlite3.connect(temp_db_path); c.row_factory = sqlite3.Row
    rows = c.execute("SELECT name FROM entities WHERE type='tech'").fetchall()
    names = [r["name"] for r in rows]
    # P1-5 fixed _TECH_VERSION_RE to exclude CJK — these noise patterns must be gone
    noise = ["但95", "实体压缩87"]
    for n in names:
        for pattern in noise:
            assert pattern not in n, f"Tech name {n!r} contains noise pattern {pattern!r}"
    c.close()

def test_empty_parse_does_not_lock_manifest(temp_db_path, tmp_path):
    """When a file parses to zero sections, its manifest entry must not block re-processing."""
    from knowledge_weaver.pipeline import run_consolidation
    f = tmp_path / "2026-05-20.md"
    f.write_text("not a valid dma file", encoding="utf-8")
    r1 = run_consolidation(temp_db_path, str(tmp_path), embedder=None)
    assert r1.files_processed == 1
    # fix the file, re-run, expect it to be processed (not skipped)
    f.write_text(
        "---\ntitle: t\ndate: 2026-05-20\n---\n\n## 核心要点\n- 启动 NewProj 项目\n",
        encoding="utf-8",
    )
    r2 = run_consolidation(temp_db_path, str(tmp_path), embedder=None)
    assert r2.files_skipped == 0
    assert r2.files_processed == 1


def test_v11_end_to_end_h3_subsection_extraction(temp_db_path, tmp_path):
    """端到端：模拟 DMA v1.1 输出，跑 KW consolidation，验证实体抽取正确。"""
    from knowledge_weaver.pipeline import run_consolidation
    import sqlite3

    src = tmp_path / "agent_x"
    src.mkdir()
    (src / "2026-05-29.md").write_text("""---
title: "2026-05-29 test"
date: "2026-05-29"
---

## 10:00

### 原始细节

**核心要点**
- 09:30 - 用户：我使用 macOS

**决策与结论**
- 09:45 - 决定采用 Python

### 摘要

今天用户决定采用 Python 作为后端语言。

[关键决策] 后端框架：Python FastAPI（用户原话："决定用 FastAPI"）
[关键偏好] 编辑器：vim
[关键风险] 数据库连接池可能耗尽
[已完成] 完成登录模块单元测试
[待办] 接入 OAuth2
""", encoding="utf-8")

    result = run_consolidation(
        db_path=temp_db_path,
        memory_dirs=[("agent_x", str(src))],
        embedder=None,
    )
    assert result.status == "ok"
    assert result.files_processed == 1

    c = sqlite3.connect(temp_db_path); c.row_factory = sqlite3.Row
    types = {r["type"]: r["cnt"] for r in c.execute(
        "SELECT type, COUNT(*) cnt FROM entities GROUP BY type")}
    c.close()

    assert types.get("decision", 0) >= 1, f"missing decision: {types}"
    assert types.get("preference", 0) >= 1, f"missing preference: {types}"
    assert types.get("risk", 0) >= 1, f"missing risk: {types}"
    assert types.get("task", 0) >= 2, f"expected >= 2 tasks: {types}"
    assert types.get("decision", 0) <= 2, \
        f"too many decisions, raw subsection may not be skipped: {types}"
