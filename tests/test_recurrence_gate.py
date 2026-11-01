"""Tests for the recurrence gate — A-soft project entity promotion."""

import sqlite3
from datetime import date, timedelta

import pytest

from knowledge_weaver.db import (
    PROJECT_GRACE_DAYS,
    PROJECT_MIN_DAYS,
    init_db,
    insert_entity,
    is_provisional_project,
)
from knowledge_weaver.tools import active_projects, knowledge_search

from scripts.clean_and_rescore import _find_noisy_entity_ids


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "test_recurrence.db")


@pytest.fixture
def conn_empty(db_path):
    c = init_db(db_path)
    yield c
    c.close()


@pytest.fixture
def seeded_conn(db_path):
    """Connection with both provisional and established projects."""
    c = init_db(db_path)
    today = date.today().isoformat()
    yesterday = (date.today() - timedelta(days=1)).isoformat()
    week_ago = (date.today() - timedelta(days=7)).isoformat()
    month_ago = (date.today() - timedelta(days=30)).isoformat()

    entities = [
        # Provisional project: day_count=1, seen yesterday (within grace)
        {
            "id": "proj:provisional_new",
            "type": "project",
            "name": "ProvisionalNew",
            "summary": "A new project appearing on only 1 day",
            "importance": 0.4,
            "first_seen": yesterday,
            "last_seen": yesterday,
            "day_count": 1,
            "source_lines": "[]",
            "metadata": "{}",
        },
        # Provisional stale project: day_count=1, seen 30 days ago (beyond grace)
        {
            "id": "proj:provisional_stale",
            "type": "project",
            "name": "ProvisionalStale",
            "summary": "A stale provisional project beyond grace",
            "importance": 0.3,
            "first_seen": month_ago,
            "last_seen": month_ago,
            "day_count": 1,
            "source_lines": "[]",
            "metadata": "{}",
        },
        # Established project: day_count=3, seen today
        {
            "id": "proj:established",
            "type": "project",
            "name": "EstablishedProject",
            "summary": "An established project appearing on 3+ days",
            "importance": 0.85,
            "first_seen": week_ago,
            "last_seen": today,
            "day_count": 3,
            "source_lines": "[]",
            "metadata": '{"tags": ["active"]}',
        },
        # Tech entity — never affected by recurrence gate
        {
            "id": "tech:something",
            "type": "tech",
            "name": "Some Tech",
            "summary": "A tech entity with day_count=1",
            "importance": 0.5,
            "first_seen": yesterday,
            "last_seen": yesterday,
            "day_count": 1,
            "source_lines": "[]",
            "metadata": "{}",
        },
        # Decision entity — never affected by recurrence gate
        {
            "id": "decision:d1",
            "type": "decision",
            "name": "A Decision",
            "summary": "A decision entity",
            "importance": 0.6,
            "first_seen": yesterday,
            "last_seen": yesterday,
            "day_count": 1,
            "source_lines": "[]",
            "metadata": "{}",
        },
        # Task entity — never affected
        {
            "id": "task:t1",
            "type": "task",
            "name": "A Task",
            "summary": "A task entity",
            "importance": 0.3,
            "first_seen": yesterday,
            "last_seen": yesterday,
            "day_count": 1,
            "source_lines": "[]",
            "metadata": "{}",
        },
        # Fact entity — never affected
        {
            "id": "fact:f1",
            "type": "fact",
            "name": "A Fact",
            "summary": "A fact entity",
            "importance": 0.1,
            "first_seen": yesterday,
            "last_seen": yesterday,
            "day_count": 1,
            "source_lines": "[]",
            "metadata": "{}",
        },
        # Risk entity — never affected
        {
            "id": "risk:r1",
            "type": "risk",
            "name": "A Risk",
            "summary": "A risk entity",
            "importance": 0.55,
            "first_seen": yesterday,
            "last_seen": yesterday,
            "day_count": 1,
            "source_lines": "[]",
            "metadata": "{}",
        },
    ]
    for e in entities:
        insert_entity(c, e)

    yield c
    c.close()


# ---------------------------------------------------------------------------
# Tests: is_provisional_project
# ---------------------------------------------------------------------------


class TestIsProvisionalProject:
    def test_project_below_threshold_is_provisional(self):
        base = {"type": "project", "id": "proj:x"}
        assert is_provisional_project({**base, "day_count": 1})

    def test_project_at_threshold_is_not_provisional(self):
        base = {"type": "project", "id": "proj:x"}
        assert not is_provisional_project({**base, "day_count": 2})

    def test_project_above_threshold_is_not_provisional(self):
        base = {"type": "project", "id": "proj:x"}
        assert not is_provisional_project({**base, "day_count": 5})

    def test_non_project_type_is_never_provisional(self):
        assert not is_provisional_project({"type": "tech", "id": "t", "day_count": 1})
        assert not is_provisional_project({"type": "decision", "id": "d", "day_count": 1})
        assert not is_provisional_project({"type": "task", "id": "t", "day_count": 1})
        assert not is_provisional_project({"type": "fact", "id": "f", "day_count": 1})
        assert not is_provisional_project({"type": "risk", "id": "r", "day_count": 1})
        assert not is_provisional_project({"type": "preference", "id": "p", "day_count": 1})

    def test_registered_slug_bypasses_gate(self):
        base = {"type": "project", "id": "proj:x"}
        assert not is_provisional_project({**base, "day_count": 1}, {"proj:x"})

    def test_different_registered_slug_does_not_bypass(self):
        base = {"type": "project", "id": "proj:x"}
        assert is_provisional_project({**base, "day_count": 1}, {"proj:y"})

    def test_missing_day_count_defaults_to_zero(self):
        base = {"type": "project", "id": "proj:x"}
        assert is_provisional_project(base)


# ---------------------------------------------------------------------------
# Tests: active_projects hides provisional
# ---------------------------------------------------------------------------


class TestActiveProjectsHidesProvisional:
    def test_excludes_provisional_project(self, seeded_conn):
        result = active_projects(seeded_conn, lookback_days=14)
        names = [p["name"] for p in result["projects"]]
        assert "EstablishedProject" in names
        assert "ProvisionalNew" not in names
        assert "ProvisionalStale" not in names

    def test_includes_established_project(self, seeded_conn):
        result = active_projects(seeded_conn, lookback_days=14)
        names = [p["name"] for p in result["projects"]]
        assert "EstablishedProject" in names

    def test_status_labeling_unchanged(self, seeded_conn):
        """Existing day_count >= 3 active/recent labeling must remain."""
        result = active_projects(seeded_conn, lookback_days=14)
        for p in result["projects"]:
            assert p["status"] in ("active", "recent")

    def test_empty_db(self, conn_empty):
        result = active_projects(conn_empty, lookback_days=14)
        assert result["projects"] == []


# ---------------------------------------------------------------------------
# Tests: knowledge_search excludes provisional project
# ---------------------------------------------------------------------------


class TestKnowledgeSearchExcludesProvisional:
    def test_search_excludes_provisional_project(self, seeded_conn):
        r = knowledge_search(seeded_conn, query="project", entity_type="project")
        names = [x["name"] for x in r["results"]]
        assert "EstablishedProject" in names
        assert "ProvisionalNew" not in names
        assert "ProvisionalStale" not in names

    def test_search_includes_tech_unaffected(self, seeded_conn):
        r = knowledge_search(seeded_conn, query="tech")
        names = [x["name"] for x in r["results"]]
        assert "Some Tech" in names

    def test_search_includes_decision_unaffected(self, seeded_conn):
        r = knowledge_search(seeded_conn, query="decision")
        names = [x["name"] for x in r["results"]]
        assert "A Decision" in names

    def test_search_includes_task_unaffected(self, seeded_conn):
        r = knowledge_search(seeded_conn, query="task")
        names = [x["name"] for x in r["results"]]
        assert "A Task" in names

    def test_search_includes_fact_unaffected(self, seeded_conn):
        r = knowledge_search(seeded_conn, query="fact")
        names = [x["name"] for x in r["results"]]
        assert "A Fact" in names

    def test_search_includes_risk_unaffected(self, seeded_conn):
        r = knowledge_search(seeded_conn, query="risk")
        names = [x["name"] for x in r["results"]]
        assert "A Risk" in names

    def test_search_fill_rate_max_results_when_provisional_in_top_n(self, seeded_conn):
        """先前切后滤 bug: top N 里混入 provisional 被踢后,合法候选未顶上来→少返.
        注入 >max_results 候选,让前 10 含 2 个 provisional,确认仍返回 max_results 条."""
        conn = seeded_conn

        # Inject 12 project candidates (2 provisional + 10 established) to exceed default max_results=10
        today_s = date.today().isoformat()
        fill_entities = [
            {
                "id": f"proj:fill_{i:02d}",
                "type": "project",
                "name": f"FillProject{i:02d}",
                "summary": f"Fill entity {i}",
                "importance": 0.9 - i * 0.01,
                "first_seen": today_s,
                "last_seen": today_s,
                "day_count": 3,
                "source_lines": "[]",
                "metadata": "{}",
            }
            for i in range(1, 11)
        ]
        for ent in fill_entities:
            conn.execute(
                "INSERT INTO entities (id, type, name, summary, importance, first_seen, last_seen, day_count, source_lines, metadata) "
                "VALUES (:id,:type,:name,:summary,:importance,:first_seen,:last_seen,:day_count,:source_lines,:metadata) "
                "ON CONFLICT(id) DO NOTHING",
                ent,
            )
            # Also register in FTS (OR IGNORE for virtual table compat)
            conn.execute(
                "INSERT OR IGNORE INTO entity_fts(entity_id, name, summary, type) VALUES (?, ?, ?, ?)",
                (ent["id"], ent["name"], ent["summary"], ent["type"]),
            )
        conn.commit()

        r = knowledge_search(conn, query="Fill", max_results=10)
        names = [x["name"] for x in r["results"]]
        assert len(r["results"]) == 10, (
            f"先切后滤 bug: 预期 10 条,实际 {len(r['results'])} 条. "
            f"结果: {names}"
        )
        # Confirm EstablishedProject and at least 9 fill projects present
        fill_count = sum(1 for n in names if n.startswith("FillProject"))
        assert fill_count >= 9, f"至少 9 个 FillProject 应在结果中,实际 {fill_count}"
        # Confirm provisional items not present
        assert "ProvisionalNew" not in names
        assert "ProvisionalStale" not in names


# ---------------------------------------------------------------------------
# Tests: prune removes stale provisional
# ---------------------------------------------------------------------------


class TestPruneRemovesStaleProvisional:
    def test_stale_provisional_marked(self, seeded_conn):
        noisy = _find_noisy_entity_ids(seeded_conn)
        assert "proj:provisional_stale" in noisy["provisional_stale"]

    def test_fresh_provisional_preserved(self, seeded_conn):
        noisy = _find_noisy_entity_ids(seeded_conn)
        assert "proj:provisional_new" not in noisy["provisional_stale"]

    def test_established_project_preserved(self, seeded_conn):
        noisy = _find_noisy_entity_ids(seeded_conn)
        all_noisy = []
        for ids in noisy.values():
            all_noisy.extend(ids)
        assert "proj:established" not in all_noisy

    def test_non_project_types_never_in_provisional_stale(self, seeded_conn):
        noisy = _find_noisy_entity_ids(seeded_conn)
        non_project_ids = [
            "tech:something", "decision:d1", "task:t1",
            "fact:f1", "risk:r1",
        ]
        for eid in non_project_ids:
            assert eid not in noisy["provisional_stale"]

    def test_provisional_stale_category_exists(self, conn_empty):
        noisy = _find_noisy_entity_ids(conn_empty)
        assert "provisional_stale" in noisy
        assert noisy["provisional_stale"] == []
