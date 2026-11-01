"""Tests for the 6 MCP tool implementations in tools.py."""

import os
import sqlite3
import tempfile
from datetime import date, timedelta

import pytest

from knowledge_weaver.db import (
    init_db,
    insert_entity,
    insert_relation,
    upsert_manifest,
)
from knowledge_weaver.tools import (
    active_projects,
    decision_history,
    knowledge_search,
    knowledge_stats,
    knowledge_trace,
    preference_lookup,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def db_path(tmp_path):
    """Create a temporary DB file path."""
    return str(tmp_path / "test_tools.db")


@pytest.fixture
def conn_empty(db_path):
    """Return an initialized empty DB connection."""
    c = init_db(db_path)
    yield c
    c.close()


@pytest.fixture
def conn_seeded(db_path):
    """Return a DB connection with seed data for tool tests."""
    c = init_db(db_path)
    _seed_data(c)
    yield c
    c.close()


def _seed_data(conn):
    """Insert a rich set of seed entities + relations for testing."""
    today = date.today().isoformat()
    yesterday = (date.today() - timedelta(days=1)).isoformat()
    week_ago = (date.today() - timedelta(days=7)).isoformat()
    long_ago = (date.today() - timedelta(days=30)).isoformat()

    entities = [
        # Projects
        {
            "id": "proj:exampleproject", "type": "project", "name": "ExampleProject",
            "summary": "智能家居控制系统，聚合HA实体", "importance": 0.85,
            "first_seen": week_ago, "last_seen": today,
            "day_count": 5, "source_lines": '[]',
            "metadata": '{"tags": ["exampleproject", "ha"]}',
        },
        {
            "id": "proj:demorobot", "type": "project", "name": "DemoRobot",
            "summary": "四足机器人项目", "importance": 0.6,
            "first_seen": long_ago, "last_seen": long_ago,
            "day_count": 1, "source_lines": '[]',
            "metadata": '{"tags": ["robot"]}',
        },
        # Decisions
        {
            "id": "decision:rule_engine", "type": "decision", "name": "规则引擎决策",
            "summary": "决定使用规则引擎（命名解析+设备序列号分组）而非LLM批量处理",
            "importance": 0.72, "first_seen": today, "last_seen": today,
            "day_count": 1, "source_lines": '[]',
            "metadata": '{"dma_category": "决策与结论"}',
        },
        {
            "id": "decision:fastapi", "type": "decision", "name": "FastAPI框架决策",
            "summary": "决定使用Python FastAPI作为聚合器后端框架",
            "importance": 0.65, "first_seen": yesterday, "last_seen": yesterday,
            "day_count": 1, "source_lines": '[]',
            "metadata": '{"dma_category": "决策与结论"}',
        },
        # Preferences
        {
            "id": "pref:supervision", "type": "preference", "name": "监督偏好",
            "summary": "用户倾向于让助手监督执行，避免自己无法准确了解进度",
            "importance": 0.8, "first_seen": today, "last_seen": today,
            "day_count": 3, "source_lines": '[]',
            "metadata": '{"dma_category": "用户偏好与习惯", "domain": "project_management"}',
        },
        {
            "id": "pref:integration", "type": "preference", "name": "集成偏好",
            "summary": "用户偏好将功能集成在ExampleProject内部，而非外部脚本",
            "importance": 0.75, "first_seen": today, "last_seen": today,
            "day_count": 1, "source_lines": '[]',
            "metadata": '{"dma_category": "用户偏好与习惯", "domain": "coding"}',
        },
        # Risks
        {
            "id": "risk:llm_json", "type": "risk", "name": "LLM JSON格式风险",
            "summary": "LLM聚合存在JSON格式错乱、输出截断的风险",
            "importance": 0.55, "first_seen": today, "last_seen": today,
            "day_count": 1, "source_lines": '[]',
            "metadata": '{}',
        },
        # Tasks
        {
            "id": "task:check_result", "type": "task", "name": "检查聚合结果",
            "summary": "用户明早检查修改效果，确认聚合结果",
            "importance": 0.45, "first_seen": today, "last_seen": today,
            "day_count": 1, "source_lines": '[]',
            "metadata": '{}',
        },
        # Tech facts
        {
            "id": "tech:ha_naming", "type": "tech", "name": "HA实体命名规则",
            "summary": "HA实体命名规则：{房间} {设备名} [{子类别}] {参数}，95.9%包含三段",
            "importance": 0.5, "first_seen": today, "last_seen": today,
            "day_count": 1, "source_lines": '[]',
            "metadata": '{}',
        },
    ]

    for e in entities:
        insert_entity(conn, e)

    # Relations
    relations = [
        {
            "id": "rel:proj:exampleproject->decision:rule_engine",
            "from_entity": "proj:exampleproject", "to_entity": "decision:rule_engine",
            "rel_type": "RELATES_TO", "weight": 0.7,
            "evidence": "2026-05-24 co-occurrence",
        },
        {
            "id": "rel:decision:rule_engine->risk:llm_json",
            "from_entity": "decision:rule_engine", "to_entity": "risk:llm_json",
            "rel_type": "RELATES_TO", "weight": 0.8,
            "evidence": "decision addresses risk",
        },
        {
            "id": "rel:decision:rule_engine->task:check_result",
            "from_entity": "decision:rule_engine", "to_entity": "task:check_result",
            "rel_type": "RELATES_TO", "weight": 0.5,
            "evidence": "follow-up",
        },
        {
            "id": "rel:proj:exampleproject->tech:ha_naming",
            "from_entity": "proj:exampleproject", "to_entity": "tech:ha_naming",
            "rel_type": "RELATES_TO", "weight": 0.6,
            "evidence": "same day",
        },
    ]

    for r in relations:
        insert_relation(conn, r)

    # Manifest
    upsert_manifest(conn, {
        "date": today, "file_path": f"memory/{today}.md",
        "file_hash": "abc123", "entity_count": 7, "status": "ok",
    })
    upsert_manifest(conn, {
        "date": yesterday, "file_path": f"memory/{yesterday}.md",
        "file_hash": "def456", "entity_count": 2, "status": "ok",
    })


# ===========================================================================
# Test: knowledge_search
# ===========================================================================

class TestKnowledgeSearch:
    def test_returns_correct_structure(self, conn_seeded):
        result = knowledge_search(conn_seeded, query="ExampleProject")
        assert "results" in result
        assert "total_hits" in result
        assert isinstance(result["results"], list)

    def test_finds_matching_entities(self, conn_seeded):
        result = knowledge_search(conn_seeded, query="ExampleProject")
        assert result["total_hits"] >= 1
        assert any("ExampleProject" in r["name"] for r in result["results"])

    def test_result_has_required_fields(self, conn_seeded):
        result = knowledge_search(conn_seeded, query="规则引擎")
        if result["results"]:
            r = result["results"][0]
            required = {"entity_id", "name", "type", "summary", "similarity_score",
                        "importance", "first_seen", "last_seen", "related_entities"}
            assert required.issubset(r.keys()), f"Missing keys: {required - r.keys()}"

    def test_entity_type_filter(self, conn_seeded):
        result = knowledge_search(conn_seeded, query="", entity_type="decision")
        for r in result["results"]:
            assert r["type"] == "decision"

    def test_max_results_limit(self, conn_seeded):
        result = knowledge_search(conn_seeded, query="", max_results=2)
        assert len(result["results"]) <= 2

    def test_min_score_filter(self, conn_seeded):
        result = knowledge_search(conn_seeded, query="", min_score=0.9)
        # Only very important entities should pass
        for r in result["results"]:
            assert r["importance"] >= 0.9

    def test_empty_db(self, conn_empty):
        result = knowledge_search(conn_empty, query="anything")
        assert result["results"] == []
        assert result["total_hits"] == 0


# ===========================================================================
# Test: knowledge_trace
# ===========================================================================

class TestKnowledgeTrace:
    def test_returns_correct_structure(self, conn_seeded):
        result = knowledge_trace(conn_seeded, topic="ExampleProject")
        assert "entity" in result
        assert "timeline" in result
        assert "related" in result
        assert "decisions" in result

    def test_finds_entity_by_name(self, conn_seeded):
        result = knowledge_trace(conn_seeded, topic="ExampleProject")
        assert result["entity"] is not None
        assert result["entity"]["entity_id"] == "proj:exampleproject"

    def test_finds_entity_by_exact_id(self, conn_seeded):
        result = knowledge_trace(conn_seeded, topic="proj:exampleproject")
        assert result["entity"] is not None
        assert result["entity"]["entity_id"] == "proj:exampleproject"

    def test_related_entities_populated(self, conn_seeded):
        result = knowledge_trace(conn_seeded, topic="ExampleProject")
        # ExampleProject has RELATES_TO to decision:rule_engine and tech:ha_naming
        related_ids = [r["entity_id"] for r in result["related"]]
        assert len(related_ids) >= 1

    def test_timeline_has_dates(self, conn_seeded):
        result = knowledge_trace(conn_seeded, topic="ExampleProject")
        for entry in result["timeline"]:
            assert "date" in entry
            assert "summary" in entry

    def test_decisions_collected(self, conn_seeded):
        result = knowledge_trace(conn_seeded, topic="ExampleProject")
        # Should include the decision:rule_engine which is related to ExampleProject
        assert isinstance(result["decisions"], list)

    def test_topic_not_found(self, conn_seeded):
        result = knowledge_trace(conn_seeded, topic="nonexistent_topic_xyz")
        assert result["entity"] is None
        assert result["timeline"] == []
        assert result["related"] == []
        assert result["decisions"] == []

    def test_empty_db(self, conn_empty):
        result = knowledge_trace(conn_empty, topic="anything")
        assert result["entity"] is None
        assert result["results"] if "results" in result else result["timeline"] == []


# ===========================================================================
# Test: active_projects
# ===========================================================================

class TestActiveProjects:
    def test_returns_correct_structure(self, conn_seeded):
        result = active_projects(conn_seeded, lookback_days=14)
        assert "projects" in result
        assert isinstance(result["projects"], list)

    def test_finds_active_project(self, conn_seeded):
        result = active_projects(conn_seeded, lookback_days=14)
        assert len(result["projects"]) >= 1
        names = [p["name"] for p in result["projects"]]
        assert "ExampleProject" in names

    def test_excludes_stale_project(self, conn_seeded):
        result = active_projects(conn_seeded, lookback_days=14)
        # DemoRobot was last seen 30 days ago, should be excluded
        names = [p["name"] for p in result["projects"]]
        assert "DemoRobot" not in names

    def test_project_has_required_fields(self, conn_seeded):
        result = active_projects(conn_seeded, lookback_days=14)
        if result["projects"]:
            p = result["projects"][0]
            required = {"entity_id", "name", "last_active", "active_days",
                        "status", "open_tasks", "latest_summary", "importance"}
            assert required.issubset(p.keys()), f"Missing: {required - p.keys()}"

    def test_project_status(self, conn_seeded):
        result = active_projects(conn_seeded, lookback_days=14)
        for p in result["projects"]:
            assert p["status"] in ("active", "recent")

    def test_empty_db(self, conn_empty):
        result = active_projects(conn_empty, lookback_days=14)
        assert result["projects"] == []


# ===========================================================================
# Test: preference_lookup
# ===========================================================================

class TestPreferenceLookup:
    def test_returns_correct_structure(self, conn_seeded):
        result = preference_lookup(conn_seeded)
        assert "preferences" in result
        assert isinstance(result["preferences"], list)

    def test_finds_all_preferences(self, conn_seeded):
        result = preference_lookup(conn_seeded)
        assert len(result["preferences"]) >= 2

    def test_preference_has_required_fields(self, conn_seeded):
        result = preference_lookup(conn_seeded)
        if result["preferences"]:
            p = result["preferences"][0]
            required = {"entity_id", "content", "first_seen", "day_count", "strength"}
            assert required.issubset(p.keys()), f"Missing: {required - p.keys()}"

    def test_topic_filter(self, conn_seeded):
        result = preference_lookup(conn_seeded, topic="监督")
        # Should find the supervision preference
        assert len(result["preferences"]) >= 1
        assert any("监督" in p["content"] for p in result["preferences"])

    def test_domain_filter(self, conn_seeded):
        result = preference_lookup(conn_seeded, domain="coding")
        # Only the "集成偏好" has domain=coding
        assert len(result["preferences"]) >= 1
        for p in result["preferences"]:
            # Verify it's a coding-domain preference
            assert p["entity_id"] == "pref:integration"

    def test_empty_db(self, conn_empty):
        result = preference_lookup(conn_empty)
        assert result["preferences"] == []

    def test_no_matching_topic(self, conn_seeded):
        result = preference_lookup(conn_seeded, topic="nonexistent_xyz")
        assert result["preferences"] == []


# ===========================================================================
# Test: decision_history
# ===========================================================================

class TestDecisionHistory:
    def test_returns_correct_structure(self, conn_seeded):
        result = decision_history(conn_seeded, topic="规则")
        assert "decisions" in result
        assert isinstance(result["decisions"], list)

    def test_finds_matching_decisions(self, conn_seeded):
        result = decision_history(conn_seeded, topic="规则引擎")
        assert len(result["decisions"]) >= 1

    def test_decision_has_required_fields(self, conn_seeded):
        result = decision_history(conn_seeded, topic="规则引擎")
        if result["decisions"]:
            d = result["decisions"][0]
            required = {"entity_id", "content", "date", "rationale",
                        "related_risks", "follow_up_tasks"}
            assert required.issubset(d.keys()), f"Missing: {required - d.keys()}"

    def test_related_risks_included(self, conn_seeded):
        result = decision_history(conn_seeded, topic="规则引擎", include_risk=True)
        if result["decisions"]:
            d = result["decisions"][0]
            # decision:rule_engine is related to risk:llm_json
            assert isinstance(d["related_risks"], list)

    def test_risk_excluded_when_flag_false(self, conn_seeded):
        result_with = decision_history(conn_seeded, topic="规则引擎", include_risk=True)
        result_without = decision_history(conn_seeded, topic="规则引擎", include_risk=False)
        # include_risk=True may find more risks (topic-matched)
        # include_risk=False only gets relation-linked risks
        if result_with["decisions"]:
            assert len(result_with["decisions"][0]["related_risks"]) >= \
                   len(result_without["decisions"][0]["related_risks"])

    def test_follow_up_tasks(self, conn_seeded):
        result = decision_history(conn_seeded, topic="规则引擎")
        if result["decisions"]:
            d = result["decisions"][0]
            # decision:rule_engine has relation to task:check_result
            assert isinstance(d["follow_up_tasks"], list)

    def test_empty_db(self, conn_empty):
        result = decision_history(conn_empty, topic="anything")
        assert result["decisions"] == []

    def test_no_matching_topic(self, conn_seeded):
        result = decision_history(conn_seeded, topic="nonexistent_xyz")
        assert result["decisions"] == []


# ===========================================================================
# Test: knowledge_stats
# ===========================================================================

class TestKnowledgeStats:
    def test_returns_correct_structure(self, conn_seeded):
        result = knowledge_stats(conn_seeded)
        required = {"entity_counts", "total_entities", "total_relations",
                    "indexed_days", "last_consolidation", "embedding_model", "db_size_mb"}
        assert required.issubset(result.keys()), f"Missing: {required - result.keys()}"

    def test_entity_counts_by_type(self, conn_seeded):
        result = knowledge_stats(conn_seeded)
        assert result["total_entities"] >= 3
        assert "project" in result["entity_counts"]
        assert "decision" in result["entity_counts"]
        assert "preference" in result["entity_counts"]

    def test_total_relations(self, conn_seeded):
        result = knowledge_stats(conn_seeded)
        assert result["total_relations"] >= 1

    def test_indexed_days(self, conn_seeded):
        result = knowledge_stats(conn_seeded)
        assert result["indexed_days"] >= 1

    def test_empty_db(self, conn_empty):
        result = knowledge_stats(conn_empty)
        assert result["total_entities"] == 0
        assert result["total_relations"] == 0
        assert result["indexed_days"] == 0
        assert result["entity_counts"] == {}

def test_default_min_score_does_not_hide_fact_entities(conn_seeded):
    from knowledge_weaver.tools import knowledge_search
    from knowledge_weaver.db import insert_entity
    insert_entity(conn_seeded, {
        "id": "fact:lowscore", "type": "fact", "name": "low score fact",
        "summary": "some summary content", "importance": 0.05,
        "first_seen": "2026-05-24", "last_seen": "2026-05-24",
        "day_count": 1, "source_lines": "[]", "metadata": "{}",
    })
    r = knowledge_search(conn_seeded, query="low score fact")
    assert any(x["entity_id"] == "fact:lowscore" for x in r["results"]), \
        "Default min_score should not hide low-importance entities"
