"""Tests for the DMA file parser module."""

import os
import pytest
from knowledge_weaver.parser import (
    ParsedItem,
    ParsedSection,
    ParsedFile,
    parse_dma_file,
    parse_dma_content,
    DMA_CATEGORY_MAP,
)

FIXTURES = os.path.join(os.path.dirname(__file__), "..", "fixtures")


# ---------------------------------------------------------------------------
# test_parse_empty_file
# ---------------------------------------------------------------------------

def test_parse_empty_file():
    result = parse_dma_content("")
    assert isinstance(result, ParsedFile)
    assert result.sections == []


def test_parse_empty_file_with_frontmatter_only():
    content = "---\ntitle: test\ndate: '2026-05-24'\n---\n"
    result = parse_dma_content(content)
    assert result.sections == []
    assert result.date == "2026-05-24"


# ---------------------------------------------------------------------------
# test_parse_frontmatter
# ---------------------------------------------------------------------------

def test_parse_frontmatter():
    content = "---\ntitle: '2026年05月24日 - 会话记忆'\ndate: '2026-05-24'\n---\n\n## 核心要点\n- 09:30 - test item\n"
    result = parse_dma_content(content)
    assert result.date == "2026-05-24"
    assert result.title == "2026年05月24日 - 会话记忆"


def test_parse_frontmatter_missing_date():
    content = "---\ntitle: 'some title'\n---\n\n## 核心要点\n- test\n"
    result = parse_dma_content(content, default_date="2026-01-01")
    assert result.date == "2026-01-01"


def test_parse_frontmatter_uses_default_date_when_absent():
    content = "## 核心要点\n- test\n"
    result = parse_dma_content(content, default_date="2026-05-20")
    assert result.date == "2026-05-20"


# ---------------------------------------------------------------------------
# test_parse_sections
# ---------------------------------------------------------------------------

def test_parse_sections():
    content = (
        "---\ntitle: test\ndate: '2026-05-24'\n---\n"
        "## 核心要点\n- 09:00 - item1\n\n"
        "## 决策与结论\n- 10:00 - item2\n"
    )
    result = parse_dma_content(content)
    assert len(result.sections) == 2
    assert result.sections[0].title == "核心要点"
    assert result.sections[1].title == "决策与结论"


def test_parse_sections_preserve_unknown_heading():
    content = (
        "---\ntitle: test\ndate: '2026-05-24'\n---\n"
        "## 自定义章节\n- some item\n"
    )
    result = parse_dma_content(content)
    assert len(result.sections) == 1
    assert result.sections[0].title == "自定义章节"
    # Unknown section → category should default to "fact"
    assert result.sections[0].category == "fact"


# ---------------------------------------------------------------------------
# test_parse_items
# ---------------------------------------------------------------------------

def test_parse_items_with_timestamp():
    content = (
        "---\ntitle: test\ndate: '2026-05-24'\n---\n"
        "## 核心要点\n"
        "- 09:30 - 第一条要点内容\n"
        "- 10:15 - 第二条要点内容\n"
    )
    result = parse_dma_content(content)
    section = result.sections[0]
    assert len(section.items) == 2

    item1 = section.items[0]
    assert item1.time == "09:30"
    assert "第一条要点内容" in item1.text

    item2 = section.items[1]
    assert item2.time == "10:15"
    assert "第二条要点内容" in item2.text


def test_parse_items_without_timestamp():
    content = (
        "---\ntitle: test\ndate: '2026-05-24'\n---\n"
        "## 核心要点\n"
        "- 没有时间戳的条目\n"
    )
    result = parse_dma_content(content)
    item = result.sections[0].items[0]
    assert item.time is None
    assert "没有时间戳的条目" in item.text


def test_parse_items_with_seconds_timestamp():
    content = (
        "---\ntitle: test\ndate: '2026-05-24'\n---\n"
        "## 核心要点\n"
        "- 09:30:45 - 精确到秒的条目\n"
    )
    result = parse_dma_content(content)
    item = result.sections[0].items[0]
    assert item.time == "09:30:45"


def test_parse_items_bold_markers():
    """Items may contain **bold** text markers that should be preserved in text."""
    content = (
        "---\ntitle: test\ndate: '2026-05-24'\n---\n"
        "## 核心要点\n"
        "- 09:00 - **重要**的要点内容\n"
    )
    result = parse_dma_content(content)
    item = result.sections[0].items[0]
    assert "**重要**" in item.text


# ---------------------------------------------------------------------------
# test_parse_sample_file
# ---------------------------------------------------------------------------

def test_parse_sample_file():
    filepath = os.path.join(FIXTURES, "sample_2026-05-24.md")
    result = parse_dma_file(filepath)

    assert isinstance(result, ParsedFile)
    assert result.date == "2026-05-24"
    assert "会话记忆" in result.title
    assert len(result.sections) == 8  # 8 DMA categories

    # Verify each section has items
    for section in result.sections:
        assert len(section.items) > 0, f"Section '{section.title}' has no items"


# ---------------------------------------------------------------------------
# test_parse_dma_structure
# ---------------------------------------------------------------------------

def test_parse_dma_structure():
    """Verify the DMA categories are recognized and mapped correctly."""
    content = (
        "---\ntitle: test\ndate: '2026-05-24'\n---\n"
        "## 核心要点\n- 09:00 - a\n\n"
        "## 决策与结论\n- 10:00 - b\n\n"
        "## 已完成事项\n- 11:00 - c\n\n"
        "## 待办与计划\n- 12:00 - d\n\n"
        "## 用户偏好与习惯\n- 13:00 - e\n\n"
        "## 技术/项目要点\n- 14:00 - f\n\n"
        "## 风险与注意事项\n- 15:00 - g\n\n"
        "## 创意与想法\n- 16:00 - h\n\n"
        "## 关键讨论\n- 17:00 - i\n"
    )
    result = parse_dma_content(content)

    section_map = {s.title: s for s in result.sections}
    assert len(result.sections) == 9

    for title, expected_cat in DMA_CATEGORY_MAP.items():
        assert title in section_map, f"Missing section: {title}"
        assert section_map[title].category == expected_cat, (
            f"Section '{title}' mapped to '{section_map[title].category}', expected '{expected_cat}'"
        )


# ---------------------------------------------------------------------------
# test_parse_line_range_tracking
# ---------------------------------------------------------------------------

def test_parse_line_range_tracking():
    content = (
        "---\n"              # line 1
        "title: test\n"      # line 2
        "date: '2026-05-24'\n"  # line 3
        "---\n"              # line 4
        "\n"                 # line 5
        "## 核心要点\n"       # line 6
        "- 09:00 - item one\n"  # line 7
        "- 10:00 - item two\n"  # line 8
    )
    result = parse_dma_content(content)
    items = result.sections[0].items

    assert len(items) == 2
    # First item starts at line 7
    assert items[0].line_start == 7
    assert items[0].line_end == 7
    # Second item starts at line 8
    assert items[1].line_start == 8
    assert items[1].line_end == 8


def test_parse_line_range_multiline_item():
    """Multi-line items (continuation lines) should have correct line ranges."""
    content = (
        "---\n"
        "title: test\n"
        "date: '2026-05-24'\n"
        "---\n"
        "\n"
        "## 核心要点\n"
        "- 09:00 - 这是一个\n"
        "  跨越多行的条目\n"
        "- 10:00 - 短条目\n"
    )
    result = parse_dma_content(content)
    items = result.sections[0].items

    assert len(items) == 2
    # First item spans lines 7-8
    assert items[0].line_start == 7
    assert items[0].line_end == 8
    # Second item on line 9
    assert items[1].line_start == 9
    assert items[1].line_end == 9


# ---------------------------------------------------------------------------
# test_parse_file_not_found
# ---------------------------------------------------------------------------

def test_parse_file_not_found():
    result = parse_dma_file("/nonexistent/path/to/file.md")
    assert isinstance(result, ParsedFile)
    assert result.sections == []


# ---------------------------------------------------------------------------
# test_parse_all_three_fixtures
# ---------------------------------------------------------------------------

def test_parse_all_three_fixtures():
    """All three fixture files should parse successfully with 8 sections each."""
    for filename in ["sample_2026-05-24.md", "sample_2026-05-23.md", "sample_2026-05-22.md"]:
        filepath = os.path.join(FIXTURES, filename)
        result = parse_dma_file(filepath)
        assert len(result.sections) == 8, f"{filename}: expected 8 sections, got {len(result.sections)}"
        for section in result.sections:
            assert len(section.items) > 0, f"{filename}: section '{section.title}' has no items"


def test_parser_maps_key_discussion_to_fact():
    """KW SPEC v1.0 §4.2: 关键讨论 must map to fact category."""
    from knowledge_weaver.parser import parse_dma_content
    content = """---
title: t
date: 2026-05-28
---

## 10:00

**关键讨论**
- 讨论了 A 和 B 的差异
"""
    p = parse_dma_content(content)
    cats_with_items = [(s.title, s.category, len(s.items)) for s in p.sections if s.items]
    assert any(t == "关键讨论" and c == "fact" and n >= 1 for t, c, n in cats_with_items), \
        f"关键讨论 not recognized as fact: {cats_with_items}"


# ---------------------------------------------------------------------------
# v1.1 H3 subsection + tag-based extraction tests
# ---------------------------------------------------------------------------

def test_v11_raw_subsection_marks_skip_extraction():
    """### 原始细节 子分区内的 bullet 应该 skip_extraction=True。"""
    from knowledge_weaver.parser import parse_dma_content
    content = """---
title: t
date: 2026-05-29
---

## 10:00

### 原始细节

**核心要点**
- 09:30 - 用户：决定采用 Python

### 摘要

[关键决策] 后端框架：Python FastAPI
"""
    p = parse_dma_content(content)

    raw_items = [it for s in p.sections for it in s.items if it.skip_extraction]
    tag_items = [it for s in p.sections for it in s.items if it.tag]

    assert len(raw_items) == 1, f"expected 1 raw item, got {len(raw_items)}"
    assert "决定采用 Python" in raw_items[0].text
    assert len(tag_items) == 1, f"expected 1 tag item, got {len(tag_items)}"
    assert tag_items[0].tag == "关键决策"
    assert "Python FastAPI" in tag_items[0].text


def test_v11_tag_to_type_table_complete():
    """TAG_TO_TYPE 表必须有完整的 9 个 tag 映射。"""
    from knowledge_weaver.parser import TAG_TO_TYPE

    expected = {
        "关键决策": "decision",
        "关键偏好": "preference",
        "关键事实": "fact",
        "关键风险": "risk",
        "关键技术": "tech",
        "已完成": "task",
        "待办": "task",
        "创意": "idea",
        "关键讨论": "fact",
    }
    assert TAG_TO_TYPE == expected, f"TAG_TO_TYPE mismatch: {TAG_TO_TYPE}"


def test_v11_tag_continuation_lines_merged():
    """带缩进续行的 tag 行应合并到一个 ParsedItem。"""
    from knowledge_weaver.parser import parse_dma_content
    content = """## 10:00

### 摘要

[关键决策] 后端框架：Python FastAPI
  理由：原生 async 支持
  其它候选：Django 已排除
"""
    p = parse_dma_content(content)
    tags = [it for s in p.sections for it in s.items if it.tag]
    assert len(tags) == 1
    text = tags[0].text
    assert "Python FastAPI" in text
    assert "原生 async" in text
    assert "Django" in text


def test_v11_backward_compatible_with_v10():
    """v1.0 扁平格式（无 H3 子分区）必须仍能正确解析。"""
    from knowledge_weaver.parser import parse_dma_content
    content = """---
title: t
date: 2026-05-28
---

## 10:00

**核心要点**
- 09:30 - 用户使用 macOS

**决策与结论**
- 决定采用 Python
"""
    p = parse_dma_content(content)
    fact_sections = [s for s in p.sections if s.category == "fact"]
    decision_sections = [s for s in p.sections if s.category == "decision"]
    assert any(len(s.items) >= 1 for s in fact_sections)
    assert any(len(s.items) >= 1 for s in decision_sections)

    # 所有 items 应该都没有 v1.1 标记
    for s in p.sections:
        for it in s.items:
            assert it.skip_extraction is False
            assert it.tag is None


def test_v11_unknown_h3_falls_back_to_default():
    """未识别的 ### 标题应回退到 v1.0 default 模式，仍解析 **xxx** 和 bullet。"""
    from knowledge_weaver.parser import parse_dma_content
    content = """## 10:00

### 自定义未知段名

**核心要点**
- bullet 1
"""
    p = parse_dma_content(content)
    fact = [s for s in p.sections if s.category == "fact"]
    assert any(len(s.items) >= 1 for s in fact)
    # bullet 不应被标记 skip_extraction
    for s in p.sections:
        for it in s.items:
            assert it.skip_extraction is False


def test_v11_time_slot_resets_subsection_mode():
    """新的 ## HH:MM 时间槽必须重置 subsection_mode 到 default。"""
    from knowledge_weaver.parser import parse_dma_content
    content = """## 10:00

### 原始细节

**核心要点**
- raw 1

## 14:00

**核心要点**
- 14 点之后的 bullet 不应被 skip_extraction
"""
    p = parse_dma_content(content)
    items = []
    for s in p.sections:
        for it in s.items:
            items.append((it.text, it.skip_extraction))

    assert ("raw 1", True) in items
    assert ("14 点之后的 bullet 不应被 skip_extraction", False) in items
