"""DMA daily memory file parser.

Parses DMA-produced `memory/YYYY-MM-DD.md` files into structured data.
These files have a YAML frontmatter block, then sections marked by `##` headings.
Each section has bullet items, optionally prefixed with timestamps like `HH:MM` or `HH:MM:SS`.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path

# Category mapping: DMA section headings → canonical entity types
DMA_CATEGORY_MAP: dict[str, str] = {
    "核心要点": "fact",
    "决策与结论": "decision",
    "已完成事项": "task",
    "待办与计划": "task",
    "用户偏好与习惯": "preference",
    "技术/项目要点": "tech",
    "风险与注意事项": "risk",
    "创意与想法": "idea",
    "关键讨论": "fact",  # KW SPEC v1.0 §4.2 extension category
}

# Regex for timestamp extraction: HH:MM or HH:MM:SS at the start of an item
_TIMESTAMP_RE = re.compile(r"^(\d{1,2}:\d{2}(?::\d{2})?)\s*[-–—]\s*")
# Regex for YAML frontmatter
_FRONTMATTER_RE = re.compile(r"^---\s*$")

# ---------------------------------------------------------------------------
# v1.1: H3 subsection markers
# ---------------------------------------------------------------------------

# H3 标题用于区分"原始细节"与"摘要"两类子分区
_RAW_SUBSECTION_TITLES = frozenset({
    "原始细节", "original", "raw",
})
_SUMMARY_SUBSECTION_TITLES = frozenset({
    "摘要", "summary", "digest",
})

# H3 标题正则：### <title>
_H3_RE = re.compile(r"^###\s+(.+)$")

# Tag 行正则：[<tag>] <content>
_TAG_RE = re.compile(r"^\[([^\]]+)\]\s+(.+)$")

# Tag 字符串 → KW 实体类型映射（**必须与 extractor.py 中 TAG_TO_TYPE 保持同步**）
# 修改本表时同步修改 docs/KW_MEMORY_FILE_SPEC.md §4.4
TAG_TO_TYPE: dict[str, str] = {
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


@dataclass
class ParsedItem:
    """A single bullet item or tag item within a section."""
    text: str
    time: str | None  # extracted timestamp like "09:30" or "09:30:45"
    line_start: int   # 1-indexed
    line_end: int     # 1-indexed
    # v1.1 fields below (defaults preserve v1.0 behavior)
    skip_extraction: bool = False   # True if inside `### 原始细节` subsection
    tag: str | None = None          # Tag name (e.g. "关键决策") if from `[Tag]` line


@dataclass
class ParsedSection:
    """A section marked by a ## heading."""
    title: str           # e.g., "核心要点", "决策与结论"
    category: str        # mapped: 核心要点→fact, 决策与结论→decision, etc.
    items: list[ParsedItem] = field(default_factory=list)


@dataclass
class ParsedFile:
    """A fully parsed DMA daily memory file."""
    date: str            # from frontmatter or filename
    title: str           # from frontmatter
    sections: list[ParsedSection] = field(default_factory=list)


def parse_dma_file(filepath: str) -> ParsedFile:
    """Parse a DMA daily memory file into structured sections and items."""
    path = Path(filepath)
    if not path.exists():
        return ParsedFile(date="", title="", sections=[])

    try:
        content = path.read_text(encoding="utf-8")
    except OSError:
        return ParsedFile(date="", title="", sections=[])

    # Try to extract date from filename as fallback
    filename_date = _extract_date_from_filename(path.name)
    return parse_dma_content(content, default_date=filename_date)


def parse_dma_content(content: str, default_date: str | None = None) -> ParsedFile:
    """Parse DMA content string (for testing without file I/O)."""
    lines = content.splitlines()

    # Phase 1: Extract frontmatter
    title, date, body_start = _parse_frontmatter(lines)
    if not date:
        date = default_date or ""

    # Phase 2: Parse sections and items from remaining lines
    sections = _parse_body(lines, body_start)

    return ParsedFile(date=date, title=title, sections=sections)


def _parse_frontmatter(lines: list[str]) -> tuple[str, str, int]:
    """Extract YAML frontmatter. Returns (title, date, body_start_line_index)."""
    title = ""
    date = ""
    body_start = 0

    if not lines or lines[0].strip() != "---":
        return title, date, body_start

    # Find closing ---
    end_idx = -1
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end_idx = i
            break

    if end_idx == -1:
        return title, date, body_start

    # Parse YAML-like key: value pairs
    for line in lines[1:end_idx]:
        stripped = line.strip()
        if ":" in stripped:
            key, _, value = stripped.partition(":")
            key = key.strip()
            value = value.strip().strip("'\"")
            if key == "title":
                title = value
            elif key == "date":
                date = value

    body_start = end_idx + 1
    return title, date, body_start


_CATEGORY_MARKER_RE = re.compile(r"^\*\*(.+?)\*\*$")


def _parse_body(lines: list[str], start: int) -> list[ParsedSection]:
    """Parse sections and items from the body of the document.

    Supports v1.0 Format A/B and v1.1 H3-subsection format.

    State machine:
      subsection_mode = "default" / "raw" / "summary"

    - "default": v1.0 behavior — `**xxx**` is category marker
    - "raw":     items still parsed but flagged skip_extraction=True
                 (entered on `### 原始细节` / `### original` / `### raw`)
    - "summary": `**xxx**` IGNORED; only `[Tag]` lines become items with tag
                 (entered on `### 摘要` / `### summary` / `### digest`)

    `## HH:MM` time slot resets subsection_mode to "default".
    Other `## heading` and unknown `### heading` also reset to "default".
    """
    sections: list[ParsedSection] = []
    current_category: str = "fact"
    subsection_mode: str = "default"
    i = start

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Skip blank lines
        if not stripped:
            i += 1
            continue

        # ---- v1.1: H3 subsection marker (### foo) ----
        h3_match = _H3_RE.match(stripped)
        if h3_match:
            sub_title = h3_match.group(1).strip()
            if sub_title in _RAW_SUBSECTION_TITLES:
                subsection_mode = "raw"
            elif sub_title in _SUMMARY_SUBSECTION_TITLES:
                subsection_mode = "summary"
            else:
                subsection_mode = "default"
            i += 1
            continue

        # ---- H2 heading: time slot or category ----
        heading_match = re.match(r"^##\s+(.+)$", stripped)
        if heading_match:
            heading = heading_match.group(1).strip()
            mapped = DMA_CATEGORY_MAP.get(heading)
            if mapped is not None:
                # Format A: ## 核心要点
                current_category = mapped
                sections.append(ParsedSection(title=heading, category=mapped))
            elif re.match(r"^\d{1,2}:\d{2}", heading):
                # Format B: ## 10:00 — time slot, reset everything
                current_category = "fact"
                subsection_mode = "default"
            else:
                # Unknown H2: treat as custom section
                current_category = "fact"
                sections.append(ParsedSection(title=heading, category="fact"))
            i += 1
            continue

        # ---- v1.1: in summary subsection, look for [Tag] lines ----
        if subsection_mode == "summary":
            tag_match = _TAG_RE.match(stripped)
            if tag_match:
                tag_name = tag_match.group(1).strip()
                tag_text = tag_match.group(2).strip()
                mapped_type = TAG_TO_TYPE.get(tag_name)
                if mapped_type is not None:
                    # Find or create section for this category
                    target = None
                    for s in reversed(sections):
                        if s.category == mapped_type:
                            target = s
                            break
                    if target is None:
                        title_list = [k for k, v in DMA_CATEGORY_MAP.items() if v == mapped_type]
                        target = ParsedSection(
                            title=title_list[0] if title_list else mapped_type,
                            category=mapped_type,
                        )
                        sections.append(target)
                    item, end_i = _parse_tag_item(lines, i, tag_text, tag_name)
                    target.items.append(item)
                    i = end_i + 1
                    continue
            # Non-tag content in summary subsection: skip (narrative paragraphs)
            i += 1
            continue

        # ---- Category marker (**xxx**) — only in default/raw modes ----
        cat_match = _CATEGORY_MARKER_RE.match(stripped)
        if cat_match:
            candidate = cat_match.group(1).strip()
            mapped = DMA_CATEGORY_MAP.get(candidate)
            if mapped is not None:
                current_category = mapped
                sections.append(ParsedSection(title=candidate, category=mapped))
            i += 1
            continue

        # ---- Bullet item ----
        if stripped.startswith("- "):
            target = None
            for s in reversed(sections):
                if s.category == current_category:
                    target = s
                    break
            if target is None:
                title_list = [k for k, v in DMA_CATEGORY_MAP.items() if v == current_category]
                target = ParsedSection(
                    title=title_list[0] if title_list else current_category,
                    category=current_category,
                )
                sections.append(target)

            item, end_i = _parse_item(lines, i)
            # v1.1: items in raw subsection are not extracted as entities
            if subsection_mode == "raw":
                item.skip_extraction = True
            target.items.append(item)
            i = end_i + 1
            continue

        i += 1

    return sections


def _parse_item(lines: list[str], start: int) -> tuple[ParsedItem, int]:
    """Parse a single bullet item, possibly spanning multiple lines.

    Returns (ParsedItem, last_line_index_inclusive).
    """
    # Extract the item text from the first line
    first_text = lines[start].strip()
    assert first_text.startswith("- "), f"Expected bullet item at line {start + 1}"
    raw_text = first_text[2:].strip()

    # Extract timestamp
    time_match = _TIMESTAMP_RE.match(raw_text)
    time_str: str | None = None
    if time_match:
        time_str = time_match.group(1)
        raw_text = raw_text[time_match.end():].strip()

    # Collect continuation lines (indented lines that follow)
    end_i = start
    j = start + 1
    while j < len(lines):
        next_line = lines[j]
        # Continuation: indented line (starts with spaces) and not empty, not a heading, not a bullet
        if next_line and (next_line[0] in (" ", "\t")) and next_line.strip():
            stripped_next = next_line.strip()
            # But if it looks like a new bullet or heading, stop
            if stripped_next.startswith("- ") or re.match(r"^##\s+", stripped_next):
                break
            raw_text += " " + stripped_next
            end_i = j
            j += 1
        else:
            break

    return ParsedItem(
        text=raw_text,
        time=time_str,
        line_start=start + 1,  # 1-indexed
        line_end=end_i + 1,    # 1-indexed
    ), end_i


def _parse_tag_item(
    lines: list[str], start: int, initial_text: str, tag_name: str,
) -> tuple[ParsedItem, int]:
    """Parse a tag-line item (v1.1 `### 摘要` subsection format).

    Input line (already stripped) has form:
        [<tag_name>] <initial_text>

    Continuation lines (indented) are joined into text.

    Returns (ParsedItem with tag=tag_name, last_line_index_inclusive).
    """
    raw_text = initial_text
    end_i = start
    j = start + 1
    while j < len(lines):
        next_line = lines[j]
        # Continuation: indented non-blank line
        if next_line and (next_line[0] in (" ", "\t")) and next_line.strip():
            stripped_next = next_line.strip()
            # Stop if it looks like a new tag, bullet, or heading
            if (stripped_next.startswith("- ")
                    or _TAG_RE.match(stripped_next)
                    or re.match(r"^##+\s+", stripped_next)):
                break
            raw_text += " " + stripped_next
            end_i = j
            j += 1
        else:
            break

    return ParsedItem(
        text=raw_text,
        time=None,
        line_start=start + 1,
        line_end=end_i + 1,
        skip_extraction=False,
        tag=tag_name,
    ), end_i


def _extract_date_from_filename(filename: str) -> str | None:
    """Try to extract a YYYY-MM-DD date from a filename like '2026-05-24.md'."""
    m = re.match(r"(\d{4}-\d{2}-\d{2})\.md$", filename)
    return m.group(1) if m else None
