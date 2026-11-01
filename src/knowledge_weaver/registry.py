"""Project registry loader for Knowledge Weaver.

Reads the ## 项目标准 section from MEMORY.md, parses registered project
entries, and provides:
  - registered_slugs: entity IDs for use with is_provisional_project bypass
  - build_project_lexicon: minimal cloud-ready wordlist (canonical + aliases + status only)

Design:
  - ProjectEntry.dataclass stores ONLY canonical/aliases/status — no 🔒 fields
  - registered_slugs = generate_entity_id("project", name) for canonical + all aliases
  - Hand-written slug in registry is for validation only; matching uses slugify(names)
  - Graceful degradation: missing file / empty section → [] or set()
  - Cache: path + mtime keyed, invalidated on file change
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Optional

from knowledge_weaver.extractor import generate_entity_id

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class ProjectEntry:
    """A single project entry from the registry.

    Intentionally narrow — never stores storage/tech/dates/description (🔒 fields).
    """

    canonical_name: str
    slug: str  # hand-written slug, for validation only; matching uses slugify(names)
    aliases: list[str] = field(default_factory=list)
    status: str = "active"


# ---------------------------------------------------------------------------
# Cache — path + mtime keyed, invalidated on file change
# ---------------------------------------------------------------------------

_registry_cache: dict[str, tuple[int, list[ProjectEntry]]] = {}


def _cache_key(path: str) -> str:
    return os.path.abspath(path)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def parse_registry(text: str) -> list[ProjectEntry]:
    """Parse the ## 项目标准 section from MEMORY.md text.

    Rules:
      - Extract only the ## 项目标准 section (stop at next ## heading)
      - Strip all HTML comment blocks (<!-- ... -->)
      - Track ### subsection for default status
      - Parse project entries: - **Name** `slug: xxx` + indented fields
      - 🔒 fields (存储/技术栈/起/止/简述) are parsed but immediately discarded

    Returns [] if no section found or parsing fails.
    """
    section = _extract_registry_section(text)
    if not section:
        return []

    entries: list[ProjectEntry] = []
    current_status = "active"

    # Strip HTML comments before parsing
    section = _strip_comments(section)

    lines = section.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Track ### subsection for default status
        if line.startswith("###"):
            current_status = _subsection_status(line)
            i += 1
            continue

        # Detect project entry: - **Name** `slug: xxx`
        match = re.match(r'^-\s+\*\*(.+?)\*\*\s+`slug:\s*(\S+)`', line)
        if match:
            name = match.group(1).strip()
            slug = match.group(2)

            aliases: list[str] = []
            status = current_status

            # Collect indented field lines until next project/heading/break
            i += 1
            while i < len(lines):
                next_line = lines[i]
                stripped = next_line.strip()

                # Stop at next project entry
                if re.match(r'^-\s+\*\*', next_line):
                    break
                # Stop at next subsection heading
                if stripped.startswith("###"):
                    break
                # Stop at blank line followed by non-indented content
                if stripped == "":
                    if i + 1 < len(lines) and not lines[i + 1].startswith("  "):
                        break

                field = _parse_field(next_line)
                if field:
                    field_name, field_value = field
                    if field_name == "别名":
                        aliases = [a.strip() for a in re.split(r"[,、，]", field_value) if a.strip()]
                    elif field_name == "状态":
                        status = field_value
                    # 存储, 技术栈, 起, 止, 简述 → intentionally discarded

                i += 1

            entries.append(
                ProjectEntry(
                    canonical_name=name,
                    slug=slug,
                    aliases=aliases,
                    status=status,
                )
            )
        else:
            i += 1

    return entries


def _extract_registry_section(text: str) -> str:
    """Extract the ## 项目标准 section from markdown text.

    Captures everything from the ## 项目标准 heading line (exclusive)
    to the next ## heading at the same level (exclusive), or EOF.
    Returns "" if heading not found.
    """
    match = re.search(r'^##\s+项目标准.*$', text, re.MULTILINE)
    if not match:
        return ""

    start = match.end() + 1  # skip the heading line itself
    remaining = text[start:]

    next_heading = re.search(r'^##\s+', remaining, re.MULTILINE)
    if next_heading:
        return remaining[: next_heading.start()]

    return remaining


def _strip_comments(text: str) -> str:
    """Remove HTML comment blocks (<!-- ... -->) from text.

    Handles multi-line comments via DOTALL flag.
    """
    return re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)


def _subsection_status(line: str) -> str:
    """Extract default project status from a ### subsection heading.

    Examples:
      ### 进行中 (active)    → "active"
      ### 已完成 (done)       → "done"
      ### 已归档 (archived)   → "archived"
      ### 进行中              → "active"
    """
    # Try parenthesized English status first
    match = re.search(r'\((\w+)\)', line)
    if match:
        return match.group(1).lower()

    # Fallback: map Chinese status words
    line_lower = line.lower()
    if "进行中" in line_lower:
        return "active"
    if "已完成" in line_lower:
        return "done"
    if "已归档" in line_lower:
        return "archived"

    return "active"


def _parse_field(line: str) -> Optional[tuple[str, str]]:
    """Parse an indented field line like '  - 字段名: value 🔒'.

    Returns (field_name, field_value) where field_value has 🔒 markers stripped,
    or None if not a valid field line.
    """
    # Must start with '  - ' (two-space indent + bullet)
    if not re.match(r'^  - ', line):
        return None

    content = line.strip()[2:].strip()  # remove '- ' prefix

    # Must have ':' separator
    if ":" not in content:
        return None

    # Split on first ':'
    idx = content.index(":")
    key = content[:idx].strip()
    value = content[idx + 1 :].strip()

    # Strip trailing 🔒 markers (privacy indicators)
    value = re.sub(r'\s*🔒\s*$', '', value).strip()

    return (key, value)


# ---------------------------------------------------------------------------
# Loader with caching
# ---------------------------------------------------------------------------


def load_registry(
    path: str = "/home/openclaw/.openclaw/workspace/MEMORY.md",
) -> list[ProjectEntry]:
    """Load project registry from MEMORY.md with mtime-based cache.

    Returns [] if file is missing or parsing fails — never raises.
    """
    abs_path = os.path.abspath(path)

    try:
        mtime = int(os.path.getmtime(abs_path))
    except OSError:
        logger.debug("MEMORY.md not found at %s, returning empty registry", abs_path)
        return []

    key = _cache_key(abs_path)
    if key in _registry_cache:
        cached_mtime, cached_entries = _registry_cache[key]
        if cached_mtime == mtime:
            return cached_entries

    try:
        with open(abs_path, "r", encoding="utf-8") as f:
            text = f.read()
    except OSError:
        logger.warning("Failed to read MEMORY.md at %s", abs_path)
        return []

    entries = parse_registry(text)
    _registry_cache[key] = (mtime, entries)
    return entries


def registered_slugs(entries: list[ProjectEntry]) -> set[str]:
    """Generate the complete set of registered entity IDs.

    Each canonical name and each alias produces one entity ID via
    generate_entity_id("project", name).  The hand-written slug is NOT
    included — matching always uses slugify(names) which derives the
    same ID as the parser/extractor pipeline.
    """
    slugs: set[str] = set()
    for entry in entries:
        slugs.add(generate_entity_id("project", entry.canonical_name))
        for alias in entry.aliases:
            slugs.add(generate_entity_id("project", alias))
    return slugs


def load_registered_slugs(
    path: str = "/home/openclaw/.openclaw/workspace/MEMORY.md",
) -> set[str]:
    """Convenience: load registry and return registered entity IDs.

    Returns set() if file missing or no entries.
    """
    entries = load_registry(path)
    return registered_slugs(entries)


def alias_to_canonical_map(entries: list[ProjectEntry]) -> dict[str, str]:
    """Map each registered alias's entity-id -> the canonical entity-id.

    Canonical entity-id = generate_entity_id("project", canonical_name).
    Used by the W1 resolver to deterministically normalize a registered alias
    (e.g. "家庭大脑") onto its canonical entity (e.g. proj:homebrain), whether or
    not the canonical entity already exists in the DB.
    """
    mapping: dict[str, str] = {}
    for entry in entries:
        canonical_id = generate_entity_id("project", entry.canonical_name)
        for alias in entry.aliases:
            alias_id = generate_entity_id("project", alias)
            if alias_id != canonical_id:
                mapping[alias_id] = canonical_id
    return mapping


def build_project_lexicon(entries: list[ProjectEntry]) -> list[dict]:
    """Build minimal cloud-safe lexicon.

    Output fields are ONLY: canonical, aliases, status.
    🔒 fields (storage, tech stack, dates, description) are never present.
    """
    lexicon = []
    for entry in entries:
        lexicon.append(
            {
                "canonical": entry.canonical_name,
                "aliases": entry.aliases,
                "status": entry.status,
            }
        )
    return lexicon


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _main() -> None:
    import argparse
    import json
    import sys

    parser = argparse.ArgumentParser(description="KW Project Registry Loader")
    parser.add_argument(
        "--path",
        default="/home/openclaw/.openclaw/workspace/MEMORY.md",
        help="Path to MEMORY.md",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--lexicon", action="store_true", help="Output JSON lexicon")
    group.add_argument("--slugs", action="store_true", help="Output registered slugs")

    args = parser.parse_args()

    entries = load_registry(args.path)

    if args.lexicon:
        lexicon = build_project_lexicon(entries)
        json.dump(lexicon, sys.stdout, ensure_ascii=False, indent=2)
    elif args.slugs:
        slugs = registered_slugs(entries)
        for s in sorted(slugs):
            print(s)


if __name__ == "__main__":
    _main()
