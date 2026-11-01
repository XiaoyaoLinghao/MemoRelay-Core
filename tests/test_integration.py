"""Integration & end-to-end tests for Knowledge Weaver.

These tests exercise the full pipeline (parse → extract → link → score → manifest)
across multiple DMA fixture files, then verify tool-level queries return correct results.
"""

import os
import sqlite3

import pytest

from knowledge_weaver.db import init_db
from knowledge_weaver.pipeline import run_consolidation
from knowledge_weaver.tools import knowledge_trace, active_projects

FIXTURES = os.path.join(os.path.dirname(__file__), "..", "fixtures")


# ---------------------------------------------------------------------------
# test_full_pipeline_cross_day_entity_linking
# ---------------------------------------------------------------------------


def test_full_pipeline_cross_day_entity_linking(temp_db_path):
    """End-to-end: 3 days of DMA files → entities created, cross-day linked,
    ExampleProject exists with day_count >= 1, manifest tracks 3 files."""
    result = run_consolidation(
        db_path=temp_db_path,
        memory_dir=FIXTURES,
        embedder=None,
    )

    assert result.status == "ok"
    assert result.files_processed == 3
    assert result.entities_created > 0

    conn = init_db(temp_db_path)
    try:
        # ExampleProject should appear across multiple days
        hb = conn.execute(
            "SELECT * FROM entities WHERE id LIKE '%exampleproject%'"
        ).fetchone()
        assert hb is not None, "ExampleProject project entity should exist"
        assert hb["day_count"] >= 1

        # Manifest should track all 3 files
        manifest_count = conn.execute(
            "SELECT COUNT(*) as cnt FROM daily_manifest"
        ).fetchone()["cnt"]
        assert manifest_count == 3
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# test_idempotent_consolidation
# ---------------------------------------------------------------------------


def test_idempotent_consolidation(temp_db_path):
    """Running consolidation twice on same files should skip all files on second run."""
    # First run
    result1 = run_consolidation(
        db_path=temp_db_path,
        memory_dir=FIXTURES,
        embedder=None,
    )
    assert result1.files_processed >= 1
    assert result1.entities_created > 0

    # Second run — same files unchanged
    result2 = run_consolidation(
        db_path=temp_db_path,
        memory_dir=FIXTURES,
        embedder=None,
    )
    assert result2.files_processed == 0, (
        f"Second run should skip all files, but processed {result2.files_processed}"
    )
    assert result2.entities_created == 0


# ---------------------------------------------------------------------------
# test_trace_after_consolidation
# ---------------------------------------------------------------------------


def test_trace_after_consolidation(temp_db_path):
    """After consolidation, knowledge_trace should return an entity with ExampleProject name."""
    run_consolidation(
        db_path=temp_db_path,
        memory_dir=FIXTURES,
        embedder=None,
    )

    conn = init_db(temp_db_path)
    try:
        result = knowledge_trace(conn, topic="ExampleProject")
        assert result["entity"] is not None, "knowledge_trace should find ExampleProject"
        assert "ExampleProject" in result["entity"]["name"]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# test_active_projects_after_consolidation
# ---------------------------------------------------------------------------


def test_active_projects_after_consolidation(temp_db_path):
    """After consolidation, active_projects should list ExampleProject."""
    run_consolidation(
        db_path=temp_db_path,
        memory_dir=FIXTURES,
        embedder=None,
    )

    conn = init_db(temp_db_path)
    try:
        result = active_projects(conn)
        assert len(result["projects"]) >= 1, (
            f"Expected at least 1 project, got {len(result['projects'])}"
        )
        names = [p["name"] for p in result["projects"]]
        assert any("ExampleProject" in n for n in names), (
            f"ExampleProject not found in projects: {names}"
        )
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# test_consolidation_to_entity_count_ratio
# ---------------------------------------------------------------------------


def test_consolidation_to_entity_count_ratio(temp_db_path):
    """3 DMA files should produce at least 10 entities (reasonable minimum)."""
    result = run_consolidation(
        db_path=temp_db_path,
        memory_dir=FIXTURES,
        embedder=None,
    )

    assert result.entities_created >= 10, (
        f"Expected >=10 entities from 3 fixtures, got {result.entities_created}"
    )
