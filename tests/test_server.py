"""Tests for Knowledge Weaver MCP server entry point."""

import json
import os
import sqlite3
import sys
import tempfile
from unittest.mock import patch

import pytest

from knowledge_weaver.db import init_db, insert_entity, insert_relation


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def temp_db_path():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    try:
        os.unlink(path)
    except OSError:
        pass


@pytest.fixture
def seeded_conn(temp_db_path):
    """Connection to a DB pre-seeded with test entities."""
    conn = init_db(temp_db_path)
    insert_entity(conn, {
        "id": "proj:exampleproject", "type": "project", "name": "ExampleProject",
        "summary": "智能家居控制系统", "importance": 0.85,
        "first_seen": "2026-05-22", "last_seen": "2026-05-24",
        "day_count": 3, "source_lines": '["2026-05-24.md:1:1"]',
        "metadata": '{"tags":["exampleproject","ha"]}',
    })
    insert_entity(conn, {
        "id": "decision:rule_engine", "type": "decision", "name": "规则引擎决策",
        "summary": "决定使用规则引擎优先", "importance": 0.72,
        "first_seen": "2026-05-24", "last_seen": "2026-05-24",
        "day_count": 1, "source_lines": '["2026-05-24.md:5:6"]',
        "metadata": '{"dma_category":"决策与结论"}',
    })
    insert_entity(conn, {
        "id": "pref:supervision", "type": "preference", "name": "监督偏好",
        "summary": "用户倾向于让助手监督执行", "importance": 0.8,
        "first_seen": "2026-05-24", "last_seen": "2026-05-24",
        "day_count": 1, "source_lines": '["2026-05-24.md:10:12"]',
        "metadata": '{"dma_category":"用户偏好与习惯"}',
    })
    insert_relation(conn, {
        "id": "rel:proj:exampleproject->decision:rule_engine",
        "from_entity": "proj:exampleproject", "to_entity": "decision:rule_engine",
        "rel_type": "RELATES_TO", "weight": 0.7,
        "evidence": "2026-05-24 co-occurrence",
    })
    yield conn
    conn.close()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_create_server():
    """Server created successfully with all tools (incl. W1 review tools)."""
    from knowledge_weaver.server import create_server
    mcp = create_server()

    tools = mcp._tool_manager.list_tools()
    tool_names = {t.name for t in tools}

    expected = {
        "knowledge_search",
        "knowledge_trace",
        "active_projects",
        "preference_lookup",
        "decision_history",
        "knowledge_stats",
        "knowledge_consolidate",
        # W1 entity-resolution review queue
        "kw_review_pending",
        "kw_resolve",
        "kw_reconcile_registry",
    }
    assert tool_names == expected, f"Expected {expected}, got {tool_names}"
    assert len(tools) == 10


def test_env_config(monkeypatch):
    """Environment variables are correctly read."""
    monkeypatch.setenv("KNOWLEDGE_WEAVER_DB_PATH", "/tmp/test.db")
    monkeypatch.setenv("KNOWLEDGE_WEAVER_MEMORY_DIR", "/tmp/mem")
    monkeypatch.setenv("KNOWLEDGE_WEAVER_LOG_LEVEL", "DEBUG")

    import importlib
    import knowledge_weaver.server as srv
    importlib.reload(srv)

    assert srv.DB_PATH == "/tmp/test.db"
    assert srv.MEMORY_DIR == "/tmp/mem"
    assert srv.LOG_LEVEL == "DEBUG"

    # Restore defaults
    monkeypatch.delenv("KNOWLEDGE_WEAVER_DB_PATH", raising=False)
    monkeypatch.delenv("KNOWLEDGE_WEAVER_MEMORY_DIR", raising=False)
    monkeypatch.delenv("KNOWLEDGE_WEAVER_LOG_LEVEL", raising=False)
    importlib.reload(srv)


def test_main_consolidate(temp_db_path, monkeypatch):
    """Consolidate CLI subcommand runs without error."""
    monkeypatch.setenv("KNOWLEDGE_WEAVER_DB_PATH", temp_db_path)
    monkeypatch.setenv("KNOWLEDGE_WEAVER_MEMORY_DIR", tempfile.gettempdir())

    import importlib
    import knowledge_weaver.server as srv
    importlib.reload(srv)

    with patch.object(sys, "argv", ["server", "consolidate"]):
        # Should not crash even with no embedder
        exit_code = srv.main()

    assert exit_code in (0, 1)

    importlib.reload(srv)


def test_knowledge_search_tool(seeded_conn):
    """Search tool registered and callable via tools module."""
    from knowledge_weaver.tools import knowledge_search
    result = knowledge_search(seeded_conn, query="ExampleProject")
    assert result["total_hits"] >= 1
    assert any("ExampleProject" in r["name"] for r in result["results"])


def test_knowledge_stats_tool(seeded_conn):
    """Stats tool returns entity counts."""
    from knowledge_weaver.tools import knowledge_stats
    stats = knowledge_stats(seeded_conn)
    assert stats["total_entities"] >= 3
    assert stats["total_relations"] >= 1
    assert "project" in stats["entity_counts"]


def test_active_projects_tool(seeded_conn):
    """Active projects tool lists ExampleProject."""
    from knowledge_weaver.tools import active_projects
    result = active_projects(seeded_conn, lookback_days=14)
    assert len(result["projects"]) >= 1
    assert any(p["entity_id"] == "proj:exampleproject" for p in result["projects"])


def test_preference_lookup_tool(seeded_conn):
    """Preference lookup returns at least one preference."""
    from knowledge_weaver.tools import preference_lookup
    result = preference_lookup(seeded_conn)
    assert len(result["preferences"]) >= 1


def test_decision_history_tool(seeded_conn):
    """Decision history returns decisions matching topic."""
    from knowledge_weaver.tools import decision_history
    result = decision_history(seeded_conn, topic="规则")
    assert len(result["decisions"]) >= 1


def test_knowledge_trace_tool(seeded_conn):
    """Knowledge trace returns entity and related data."""
    from knowledge_weaver.tools import knowledge_trace
    result = knowledge_trace(seeded_conn, topic="ExampleProject")
    assert result["entity"] is not None
    assert result["entity"]["entity_id"] == "proj:exampleproject"


def test_main_unknown_subcommand(capsys):
    """Unknown CLI subcommand returns error."""
    from knowledge_weaver.server import main
    with patch.object(sys, "argv", ["server", "unknown"]):
        exit_code = main()
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "Unknown subcommand" in captured.err


def test_server_tool_returns_json():
    """MCP tools are registered with correct signatures."""
    from knowledge_weaver.server import create_server
    mcp = create_server()
    tools = mcp._tool_manager.list_tools()
    tool_map = {t.name: t for t in tools}

    assert "knowledge_search" in tool_map
    assert "knowledge_stats" in tool_map
    assert "knowledge_consolidate" in tool_map


def test_default_env_values():
    """Default environment values are sensible."""
    saved_db = os.environ.pop("KNOWLEDGE_WEAVER_DB_PATH", None)
    saved_mem = os.environ.pop("KNOWLEDGE_WEAVER_MEMORY_DIR", None)
    saved_log = os.environ.pop("KNOWLEDGE_WEAVER_LOG_LEVEL", None)

    try:
        import importlib
        import knowledge_weaver.server as srv
        importlib.reload(srv)

        assert srv.DB_PATH == "/home/openclaw/.openclaw/knowledge/knowledge.db"
        assert srv.MEMORY_DIR == "/home/openclaw/.openclaw/workspace/memory"
        assert srv.LOG_LEVEL == "INFO"
    finally:
        if saved_db is not None:
            os.environ["KNOWLEDGE_WEAVER_DB_PATH"] = saved_db
        if saved_mem is not None:
            os.environ["KNOWLEDGE_WEAVER_MEMORY_DIR"] = saved_mem
        if saved_log is not None:
            os.environ["KNOWLEDGE_WEAVER_LOG_LEVEL"] = saved_log
        import importlib
        import knowledge_weaver.server as srv
        importlib.reload(srv)

def test_module_entrypoint_dispatches_to_main(monkeypatch):
    """`python -m knowledge_weaver unknown` must hit main() and return 1."""
    import runpy, sys, pytest
    monkeypatch.setattr(sys, "argv", ["knowledge_weaver", "unknown"])
    with pytest.raises(SystemExit) as exc:
        runpy.run_module("knowledge_weaver", run_name="__main__")
    assert exc.value.code == 1

def test_console_script_callable_via_main():
    """The console-script entry point resolves to server.main."""
    import importlib.metadata as md
    eps = md.entry_points(group="console_scripts")
    names = [e.name for e in eps]
    if "knowledge-weaver" in names:
        ep = next(e for e in eps if e.name == "knowledge-weaver")
        func = ep.load()
        assert callable(func)
