"""Tests for project registry loader — parse_registry, registered_slugs,
build_project_lexicon, load_registry, and integration hooks."""

import json
import os
import re
import tempfile
from datetime import date, timedelta

import pytest

from knowledge_weaver.registry import (
    ProjectEntry,
    build_project_lexicon,
    load_registered_slugs,
    load_registry,
    parse_registry,
    registered_slugs,
)
from knowledge_weaver.db import (
    init_db,
    insert_entity,
    is_provisional_project,
)
from knowledge_weaver.tools import active_projects, knowledge_search
from scripts.clean_and_rescore import _find_noisy_entity_ids

# ---------------------------------------------------------------------------
# Sample MEMORY.md fragments for testing
# ---------------------------------------------------------------------------

REAL_MEMORY_FRAGMENT = """\
## 项目标准 (Project Registry / 项目登记表)
<!-- 项目登记表：进行中/已完成项目 + 别名 + 状态 + 存储 + 技术栈。进度归 daily memory，不写这里。 -->
<!-- KW loader 读本节产出：① registered_slugs（豁免复现闸门）② 云端词表（仅 规范名+别名+状态）。 -->
<!-- 🔒 标记字段（存储/技术栈/简述/起止）仅本地，绝不外发云端。本节不被 KW 索引为实体（SPEC §11）。 -->
<!-- 完整解析契约见 KW loader plan。 -->

### 进行中 (active)
- **HomeBrain** `slug: homebrain`
  - 别名: 家庭大脑, 家居大脑
  - 状态: active
  - 存储: 192.168.66.68（5 容器，P0/P1 已部署）   🔒
  - 技术栈: 场景引擎(15场景+仲裁+API CRUD) + Web前端(暗色主题,WebSocket) + mi-gpt 语音桥接；agent-coding/deepseek 执行   🔒
  - 起: 2026-05-22
  - 简述: 智能家居中枢   🔒

### 已完成 (done)
（暂无）

### 已归档 (archived)
（暂无）

<!-- 模板（新增项目时复制一块）：
- **<规范名>** `slug: <kebab-slug>`
  - 别名: <别名1>, <别名2>
  - 状态: active | done | archived
  - 存储: <路径 / IP / git repo>   🔒
  - 技术栈: <技术栈，如 Python FastAPI + PostgreSQL 17 / 部署 region / 语言版本>   🔒
  - 起: YYYY-MM-DD
  - 止: YYYY-MM-DD（done / archived 才填）
  - 简述: <一句话>   🔒
-->
"""

MINIMAL_REGISTRY = """\
## 项目标准
### 进行中 (active)
- **TestProject** `slug: test-project`
  - 别名: 测试项目
  - 状态: active
"""

MULTI_PROJECT_REGISTRY = """\
## 项目标准
### 进行中 (active)
- **Alpha** `slug: alpha`
  - 别名: 阿尔法
  - 状态: active

- **Beta** `slug: beta`
  - 别名: 贝塔, β
  - 状态: active

### 已完成 (done)
- **Gamma** `slug: gamma`
  - 状态: done
"""

NO_ALIASES_REGISTRY = """\
## 项目标准
### 进行中 (active)
- **Solo** `slug: solo`
  - 状态: active
"""

SUBSECTION_DEFAULT_STATUS = """\
## 项目标准
### 已完成 (done)
- **ArchivedProject** `slug: ap`
  - 别名: 归档项目
"""

EXPLICIT_STATUS_OVERRIDES_SUBSECTION = """\
## 项目标准
### 进行中 (active)
- **ActuallyDone** `slug: ad`
  - 状态: done
"""

TEMPLATE_ONLY = """\
## 项目标准
### 进行中 (active)
<!-- 模板（新增项目时复制一块）：
- **<规范名>** `slug: <kebab-slug>`
  - 别名: <别名1>, <别名2>
  - 状态: active | done | archived
-->
"""

NO_REGISTRY_SECTION = """\
## 核心偏好
编辑器：vim
---

## 项目标准
<!-- intentionally empty, no project entries -->
"""

# ---------------------------------------------------------------------------
# parse_registry tests
# ---------------------------------------------------------------------------


class TestParseRegistry:
    """Unit tests for parse_registry — pure function, no file I/O."""

    def test_parse_basic(self):
        """Correctly parse HomeBrain from real MEMORY.md fragment."""
        entries = parse_registry(REAL_MEMORY_FRAGMENT)
        assert len(entries) == 1, f"Expected 1 entry, got {len(entries)}: {entries}"

        hb = entries[0]
        assert hb.canonical_name == "HomeBrain"
        assert hb.slug == "homebrain"
        assert "家庭大脑" in hb.aliases
        assert "家居大脑" in hb.aliases
        assert hb.status == "active"

    def test_skip_template_comment(self):
        """<!-- 模板... --> block must NOT produce a project entry."""
        entries = parse_registry(TEMPLATE_ONLY)
        assert len(entries) == 0, f"Template comment should yield no entries, got {entries}"

    def test_skip_html_comments_in_header_area(self):
        """The <!-- --> comments in the section header area should be stripped."""
        # The REAL_MEMORY_FRAGMENT has several <!-- --> comments before the ### heading.
        # They should be stripped, not interfere with parsing.
        entries = parse_registry(REAL_MEMORY_FRAGMENT)
        assert len(entries) == 1

    def test_multiple_projects_across_subsections(self):
        """Parse multiple projects across active/done/archived subsections."""
        entries = parse_registry(MULTI_PROJECT_REGISTRY)
        assert len(entries) == 3

        names = [e.canonical_name for e in entries]
        assert "Alpha" in names
        assert "Beta" in names
        assert "Gamma" in names

        # Gamma is in ### 已完成 (done) subsection
        gamma = [e for e in entries if e.canonical_name == "Gamma"][0]
        assert gamma.status == "done"

    def test_no_aliases(self):
        """Project with no 别名 field should have empty aliases list."""
        entries = parse_registry(NO_ALIASES_REGISTRY)
        assert len(entries) == 1
        assert entries[0].aliases == []

    def test_status_from_subsection_when_field_absent(self):
        """### 已完成 (done) subsection yields default status 'done'."""
        entries = parse_registry(SUBSECTION_DEFAULT_STATUS)
        assert len(entries) == 1
        assert entries[0].status == "done"

    def test_explicit_status_overrides_subsection(self):
        """Explicit 状态: done overrides ### 进行中 (active) default."""
        entries = parse_registry(EXPLICIT_STATUS_OVERRIDES_SUBSECTION)
        assert len(entries) == 1
        assert entries[0].status == "done"

    def test_no_registry_section(self):
        """Text without ## 项目标准 section returns []."""
        entries = parse_registry(NO_REGISTRY_SECTION)
        assert entries == []

    def test_empty_text(self):
        """Empty text returns []."""
        assert parse_registry("") == []
        assert parse_registry("   ") == []

    def test_section_must_start_with_double_hash(self):
        """Only ## 项目标准 (level-2 heading) triggers section extraction."""
        text = "# 项目标准\n### 进行中\n- **X** `slug: x`"
        entries = parse_registry(text)
        assert entries == []

    def test_parse_stops_at_next_double_hash_heading(self):
        """Parsing should stop at the next ## heading."""
        text = """\
## 项目标准
### 进行中 (active)
- **ProjectA** `slug: pa`
  - 状态: active

## 时效规则
- some other content
"""
        entries = parse_registry(text)
        assert len(entries) == 1
        assert entries[0].canonical_name == "ProjectA"

    def test_parse_handles_chinese_paren_subsection(self):
        """### 进行中 (active) where subsection name contains Chinese."""
        entries = parse_registry(REAL_MEMORY_FRAGMENT)
        assert len(entries) >= 1
        assert entries[0].status == "active"

    def test_parse_multiline_alias_with_spaces(self):
        """Aliases with spaces around commas should be trimmed."""
        text = """\
## 项目标准
### 进行中 (active)
- **MultiAlias** `slug: ma`
  - 别名:  foo , bar , baz 
  - 状态: active
"""
        entries = parse_registry(text)
        assert len(entries) == 1
        assert entries[0].aliases == ["foo", "bar", "baz"]

    def test_empty_subsections_handled(self):
        """Subsections with （暂无） should not produce entries."""
        text = """\
## 项目标准
### 进行中 (active)
（暂无）

### 已完成 (done)
- **RealProject** `slug: rp`
  - 状态: done
"""
        entries = parse_registry(text)
        assert len(entries) == 1
        assert entries[0].canonical_name == "RealProject"


# ---------------------------------------------------------------------------
# registered_slugs tests
# ---------------------------------------------------------------------------


class TestRegisteredSlugs:
    """Tests for generating entity IDs from registry entries."""

    def test_registered_slugs_covers_aliases(self):
        """Canonical name + all aliases each generate a slug."""
        entries = [
            ProjectEntry(
                canonical_name="HomeBrain",
                slug="homebrain",
                aliases=["家庭大脑", "家居大脑"],
                status="active",
            )
        ]
        slugs = registered_slugs(entries)

        # Canonical name → entity ID
        assert "proj:homebrain" in slugs
        # Alias 1 → entity ID (pinyin: jiatingdanao)
        assert "proj:jiatingdanao" in slugs
        # Alias 2 → entity ID (pinyin: jiajudanao)
        assert "proj:jiajudanao" in slugs

    def test_registered_slugs_hand_audited_slug_not_in_slugs(self):
        """The hand-written slug is NOT directly used — only slugify(names)."""
        entries = [
            ProjectEntry(
                canonical_name="MyProj",
                slug="my-custom-slug",  # different from pinyin
                aliases=[],
                status="active",
            )
        ]
        slugs = registered_slugs(entries)
        # Only the slugify(canonical_name) result is included
        assert "proj:my_custom_slug" not in slugs
        assert "proj:myproj" in slugs

    def test_empty_entries_returns_empty_set(self):
        """No entries → empty set."""
        assert registered_slugs([]) == set()

    def test_duplicate_aliases_deduplicated(self):
        """Duplicate names across entries are naturally deduplicated by set."""
        entries = [
            ProjectEntry("A", "a", aliases=["x"], status="active"),
            ProjectEntry("B", "b", aliases=["x"], status="active"),
        ]
        slugs = registered_slugs(entries)
        # "x" appears twice but set deduplicates
        expected_x_id = slugs  # just checking it exists once
        # Each alias produces one ID; duplicates handled by set
        assert len([s for s in slugs]) == 3  # A, B, x


# ---------------------------------------------------------------------------
# build_project_lexicon tests
# ---------------------------------------------------------------------------


class TestBuildProjectLexicon:
    """Tests for the cloud-safe lexicon output."""

    def test_lexicon_privacy(self):
        """Lexicon must only contain canonical/aliases/status — no 🔒 field values."""
        entries = [
            ProjectEntry(
                canonical_name="HomeBrain",
                slug="homebrain",
                aliases=["家庭大脑", "家居大脑"],
                status="active",
            )
        ]
        lexicon = build_project_lexicon(entries)

        assert len(lexicon) == 1
        item = lexicon[0]

        # Only three keys allowed
        assert set(item.keys()) == {"canonical", "aliases", "status"}

        # 🔒-marked values must NOT appear
        item_json = json.dumps(item, ensure_ascii=False)
        assert "192.168.66.68" not in item_json
        assert "mi-gpt" not in item_json
        assert "场景引擎" not in item_json
        assert "智能家居中枢" not in item_json
        assert "WebSocket" not in item_json
        assert "deepseek" not in item_json

    def test_empty_lexicon(self):
        """Empty entries → empty list."""
        assert build_project_lexicon([]) == []

    def test_lexicon_multiple_entries(self):
        """Multiple projects produce multiple lexicon items."""
        entries = [
            ProjectEntry("A", "a", aliases=["a1"], status="active"),
            ProjectEntry("B", "b", aliases=[], status="done"),
        ]
        lexicon = build_project_lexicon(entries)
        assert len(lexicon) == 2
        assert lexicon[1]["status"] == "done"
        assert lexicon[1]["aliases"] == []


# ---------------------------------------------------------------------------
# load_registry / load_registered_slugs tests
# ---------------------------------------------------------------------------


class TestLoadRegistry:
    """Tests for file-based loading with caching."""

    def test_graceful_missing_file(self):
        """Missing file returns [] — never raises."""
        entries = load_registry("/nonexistent/path/MEMORY.md")
        assert entries == []

    def test_load_registry_from_temp_file(self, tmp_path):
        """Load a registry from a real temp file."""
        f = tmp_path / "MEMORY.md"
        f.write_text(REAL_MEMORY_FRAGMENT, encoding="utf-8")
        entries = load_registry(str(f))
        assert len(entries) == 1
        assert entries[0].canonical_name == "HomeBrain"

    def test_load_registered_slugs_from_temp_file(self, tmp_path):
        """Load registered slugs from a real temp file."""
        f = tmp_path / "MEMORY.md"
        f.write_text(MINIMAL_REGISTRY, encoding="utf-8")
        slugs = load_registered_slugs(str(f))
        assert isinstance(slugs, set)
        assert len(slugs) >= 1
        # TestProject → proj:testproject
        assert "proj:testproject" in slugs

    def test_load_registry_empty_file(self, tmp_path):
        """Empty file returns []."""
        f = tmp_path / "MEMORY.md"
        f.write_text("", encoding="utf-8")
        entries = load_registry(str(f))
        assert entries == []

    def test_load_registry_cache_invalidation(self, tmp_path):
        """Cache should invalidate on mtime change."""
        f = tmp_path / "MEMORY.md"
        f.write_text(MINIMAL_REGISTRY, encoding="utf-8")

        # Force an older mtime to create a gap
        old_mtime = os.path.getmtime(str(f)) - 10
        os.utime(str(f), (old_mtime, old_mtime))

        entries1 = load_registry(str(f))
        assert len(entries1) == 1
        assert entries1[0].canonical_name == "TestProject"

        # Modify the file (this will set mtime to now)
        f.write_text(REAL_MEMORY_FRAGMENT, encoding="utf-8")

        entries2 = load_registry(str(f))
        assert len(entries2) == 1
        assert entries2[0].canonical_name == "HomeBrain"

    def test_load_registry_no_section(self, tmp_path):
        """File without ## 项目标准 section returns []."""
        f = tmp_path / "MEMORY.md"
        f.write_text("# Just a comment\nNo registry here.", encoding="utf-8")
        entries = load_registry(str(f))
        assert entries == []


# ---------------------------------------------------------------------------
# Integration tests — registered slugs bypass recurrence gate
# ---------------------------------------------------------------------------


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "test_registry_integration.db")


@pytest.fixture
def conn(db_path):
    """Seeded DB with a registered-single-day project and other entities."""
    c = init_db(db_path)
    today = date.today().isoformat()
    month_ago = (date.today() - timedelta(days=30)).isoformat()

    entities = [
        # Registered project: day_count=1, but explicitly registered
        {
            "id": "proj:homebrain",
            "type": "project",
            "name": "HomeBrain",
            "summary": "Registered project with only 1 day",
            "importance": 0.6,
            "first_seen": month_ago,
            "last_seen": month_ago,
            "day_count": 1,
            "source_lines": "[]",
            "metadata": "{}",
        },
        # Unregistered provisional: day_count=1, beyond grace
        {
            "id": "proj:unregistered_stale",
            "type": "project",
            "name": "UnregisteredStale",
            "summary": "Unregistered stale project",
            "importance": 0.3,
            "first_seen": month_ago,
            "last_seen": month_ago,
            "day_count": 1,
            "source_lines": "[]",
            "metadata": "{}",
        },
        # Established project: day_count=3
        {
            "id": "proj:established",
            "type": "project",
            "name": "EstablishedProject",
            "summary": "Established project",
            "importance": 0.85,
            "first_seen": month_ago,
            "last_seen": today,
            "day_count": 3,
            "source_lines": "[]",
            "metadata": "{}",
        },
    ]
    for e in entities:
        insert_entity(c, e)

    yield c
    c.close()


class TestRegisteredSingleDayProjectSurfaces:
    """Integration: registered project (day_count=1) should appear in active_projects
    and knowledge_search despite being below the recurrence threshold."""

    def test_registered_single_day_project_surfaces_in_active_projects(
        self, conn, monkeypatch
    ):
        """day_count=1 + registered → active_projects still returns it."""
        monkeypatch.setattr(
            "knowledge_weaver.tools.load_registered_slugs",
            lambda: {"proj:homebrain"},
        )
        result = active_projects(conn, lookback_days=90)
        names = [p["name"] for p in result["projects"]]
        assert "HomeBrain" in names, (
            f"Registered project should appear, got {names}"
        )

    def test_registered_single_day_project_surfaces_in_knowledge_search(
        self, conn, monkeypatch
    ):
        """day_count=1 + registered → knowledge_search still returns it."""
        monkeypatch.setattr(
            "knowledge_weaver.tools.load_registered_slugs",
            lambda: {"proj:homebrain"},
        )
        result = knowledge_search(
            conn, query="HomeBrain", entity_type="project", max_results=10
        )
        names = [r["name"] for r in result["results"]]
        assert "HomeBrain" in names, (
            f"Registered project should appear in search, got {names}"
        )

    def test_unregistered_stale_is_hidden(self, conn, monkeypatch):
        """Unregistered project with day_count=1 should be hidden."""
        monkeypatch.setattr(
            "knowledge_weaver.tools.load_registered_slugs",
            lambda: {"proj:homebrain"},
        )
        result = knowledge_search(
            conn, query="UnregisteredStale", entity_type="project", max_results=10
        )
        names = [r["name"] for r in result["results"]]
        assert "UnregisteredStale" not in names


class TestPruneSkipsRegistered:
    """Integration: registered project should not appear in prune list."""

    def test_prune_skips_registered(self, conn, monkeypatch):
        """day_count=1 beyond grace + registered → NOT in prune list."""
        monkeypatch.setattr(
            "scripts.clean_and_rescore.load_registered_slugs",
            lambda: {"proj:homebrain"},
        )
        noisy = _find_noisy_entity_ids(conn)
        stale_ids = noisy["provisional_stale"]
        assert "proj:homebrain" not in stale_ids, (
            f"Registered project should not be pruned, but found in {stale_ids}"
        )

    def test_prune_includes_unregistered_stale(self, conn, monkeypatch):
        """Unregistered stale project SHOULD be in prune list."""
        monkeypatch.setattr(
            "scripts.clean_and_rescore.load_registered_slugs",
            lambda: {"proj:homebrain"},
        )
        noisy = _find_noisy_entity_ids(conn)
        stale_ids = noisy["provisional_stale"]
        assert "proj:unregistered_stale" in stale_ids, (
            f"Unregistered stale should be pruneable, got {stale_ids}"
        )

    def test_prune_no_registered_slugs_no_crash(self, conn, monkeypatch):
        """When no registered slugs, prune should work as before."""
        monkeypatch.setattr(
            "scripts.clean_and_rescore.load_registered_slugs",
            lambda: set(),
        )
        noisy = _find_noisy_entity_ids(conn)
        stale_ids = noisy["provisional_stale"]
        # Both are stale since day_count=1 and last_seen=month_ago
        assert "proj:homebrain" in stale_ids
        assert "proj:unregistered_stale" in stale_ids


# ---------------------------------------------------------------------------
# is_provisional_project registered_slugs parameter (A 复现闸门)
# ---------------------------------------------------------------------------


class TestIsProvisionalWithRegistry:
    """Verify is_provisional_project correctly accepts registered_slugs parameter."""

    def test_registered_slug_bypasses_gate(self):
        base = {"type": "project", "id": "proj:x", "day_count": 1}
        assert not is_provisional_project(base, {"proj:x"})

    def test_unregistered_project_is_provisional(self):
        base = {"type": "project", "id": "proj:x", "day_count": 1}
        assert is_provisional_project(base, {"proj:y"})
        assert is_provisional_project(base, set())
        assert is_provisional_project(base, None)

    def test_non_project_types_always_false_even_with_registry(self):
        for etype in ["tech", "decision", "task", "fact", "risk", "preference"]:
            base = {"type": etype, "id": f"{etype}:x", "day_count": 1}
            assert not is_provisional_project(base, {"proj:x"})
