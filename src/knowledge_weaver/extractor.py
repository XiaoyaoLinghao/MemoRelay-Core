"""Rule-based entity extractor for Knowledge Weaver.

Extracts structured entities from DMA daily memory sections using
regex patterns (Level 1: deterministic, Level 2: heuristic).
No LLM dependency — pure rule-based extraction.
"""

import json
import re
from dataclasses import dataclass, field

from pypinyin import lazy_pinyin

from knowledge_weaver.parser import ParsedSection, ParsedItem

# DMA 8 categories → entity type mapping
CATEGORY_TO_TYPE: dict[str, str] = {
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

# v1.1: Tag → entity type (must stay in sync with parser.py TAG_TO_TYPE)
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

# v1.1: Task status mapping for task-type tags
TAG_TO_TASK_STATUS: dict[str, str] = {
    "已完成": "completed",
    "待办": "todo",
}

# Type prefix mapping for entity IDs
TYPE_PREFIX: dict[str, str] = {
    "project": "proj",
    "preference": "pref",
    "decision": "decision",
    "fact": "fact",
    "risk": "risk",
    "task": "task",
    "tech": "tech",
    "idea": "idea",
}

# Default importance by entity type
DEFAULT_IMPORTANCE: dict[str, float] = {
    "decision": 0.6,
    "preference": 0.6,
    "risk": 0.5,
    "project": 0.5,
    "task": 0.4,
    "tech": 0.4,
    "fact": 0.35,
    "idea": 0.3,
}

# --- Level 1: Deterministic patterns (no hardcoded names — auto-discovered from data) ---

# Project name detection patterns (language-agnostic, no user-specific names)
# Chinese: {name}项目 e.g. "心连心项目" — limit to 2-12 CJK chars immediately before 项目
_PROJECT_CN_RE = re.compile(r"([一-鿿]{2,12})项目")
# Latin project names before 项目: marker-driven with optional whitespace tolerance
# (replaces the old tight-coupling _PROJECT_LATIN_BEFORE_CN_RE which missed "Esp32 项目")
_PROJECT_LATIN_MARKER_RE = re.compile(r"([A-Z][A-Za-z0-9]+)\s*项目")
# English: "X Project" / "X project"
_PROJECT_EN_RE = re.compile(r"\b([A-Z][\w]+)\s+[Pp]roject\b")

# Tech term detection patterns (all auto-discovered, no hardcoded keyword lists)
# ALL_CAPS acronyms (2-8 chars): ESP32, STM32, MQTT, HA, SQL, API
_TECH_ACRONYM_RE = re.compile(r"\b([A-Z][A-Z0-9]{1,7})\b")
# CamelCase tool/framework names (2+ humps): HomeAssistant, TypeScript, NodeJs
_TECH_CAMEL_RE = re.compile(r"\b([A-Z][a-z]+(?:[A-Z][a-z]+)+)\b")
# Capitalized English words in Chinese context: 使用Python开发 → Python
# Uses lookahead/lookbehind to avoid consuming surrounding CN chars for subsequent matches
_TECH_CN_SURROUND_RE = re.compile(r"(?<=[一-鿿，。；：、！？　])([A-Z][a-zA-Z0-9]{1,20})(?=[一-鿿，。；：、！？　])")
# Terms with version suffixes: Python3, ESP32-S3 — ASCII-only, no CJK neighbors
_TECH_VERSION_RE = re.compile(
    r"(?:(?<=^)|(?<=[\s\(\[，。；:、]))"
    r"([A-Za-z][A-Za-z0-9]*[._-]?\d+(?:\.\d+)*[a-z]*)"
    r"(?=[\s\)\]，。；:、]|$)"
)
# Terms in backticks (code references): `nginx.conf`, `device_aggregator.py`
_TECH_BACKTICK_RE = re.compile(r"`([a-zA-Z][\w./_-]+)`")
# Common tech file extensions detected in context
_TECH_EXT_RE = re.compile(r"\b([\w-]+\.(?:py|js|ts|rs|go|java|rb|sh|yaml|yml|json|toml|cfg|conf|sql|md))\b")

# Items that look like DMA processing artifacts, not real knowledge
_GARBAGE_PATTERNS = [
    re.compile(r".*摘要失败.*"),
    re.compile(r".*见日志.*"),
    # KW SPEC v1.0 §6: agent failure placeholders match *<AGENT>-ERR: <reason>*
    re.compile(r".*-ERR:.*"),
    # Wave 9: 心跳/元描述/运维快照噪声（防止心跳时段元描述被抽成实体）
    re.compile(r".*心跳.*"),                       # 覆盖 心跳轮询/心跳检测/心跳中/首次心跳 等所有心跳元描述
    re.compile(r".*heartbeat.*", re.IGNORECASE),   # 英文心跳标记
    re.compile(r".*本次对话(仅|无|只).*"),       # "本次对话仅包含..."、"本次对话无任何实质内容"
    re.compile(r".*无(任何)?实质(性)?(内容|进展|输入).*"),
    re.compile(r".*(仅|只)包含.*(系统|心跳|轮询|检查).*"),
    re.compile(r".*本时段无(提取|实质)内容.*"),
    re.compile(r".*本地提取无输出.*"),
    re.compile(r".*(磁盘|内存)(使用|空间).*\d+(%|G|GB|MB).*"),  # 运维快照 "磁盘使用 37%"
    re.compile(r".*pending 子会话.*"),
    re.compile(r".*DAILY 检查.*"),
]

# Time-stamp log patterns — DMA session metadata, not knowledge
_TIMESTAMP_LOG_RE = re.compile(r"^\d{1,2}:\d{2}\s+UTC")  # "15:20 UTC - ..."
_BRACKET_TS_RE = re.compile(r"^\[\d{1,2}:\d{2}\]")        # "[01:03] - ..."
# Operational log keywords: system config, cron, backup status notifications
_OPS_LOG_KEYWORDS_RE = re.compile(
    r"(?:cron|backup|Dreaming|dreaming|daily.memory|crontab|UTC\d{1,2}:\d{2})",
    re.IGNORECASE,
)

# Tech terms that are too generic to be meaningful knowledge entities
_TECH_COMMON_WORDS: set[str] = {
    "AI", "API", "LLM", "CLI", "UTC", "JSON", "YAML", "HTML", "HTTP",
    "SQL", "URL", "CSS", "XML", "SSH", "SSL", "TLS", "DNS",
    "CPU", "GPU", "RAM", "SSD", "OS", "IP", "TCP", "UDP",
}

# Pure structural markers that are NOT real tech concepts
_STRUCTURAL_TECH_RE = re.compile(
    r"^(?:\d{4}[_-]\d{2}(?:[_-]\d{2})?|P[0-3])$"
)

# Hand-curated list of strings that appear as DMA artifacts (not real entity names).
# Add a string here if it ever appears as a noisy entity name in production data.
_GARBAGE_NAMES = {
    "", "无", "是", "否", "沟通直接", "确认", "-",
    "潜在", "暂缓", "制定", "事项", "恢复", "行", "任务",
    '"', "）", "（", "、", "。", "，",
}

_GARBAGE_NAME_MIN_LENGTH = 2

# Verb prefixes that should be stripped from project names extracted via _PROJECT_CN_RE
_PROJECT_VERB_PREFIXES = ("启动", "完成", "决定", "开始", "上线", "推进", "继续", "完成了")

# Structural words that indicate a phrase rather than a project name
_PROJECT_STRUCTURAL_WORDS = frozenset({
    "基于", "作为", "采用", "使用", "通过", "包含", "利用", "按照", "针对", "关于",
})

# Characters that almost never appear in proper-noun project names;
# their presence signals a sentence/phrase fragment rather than a real project name.
# (Built from 32 known noise samples + common CJK function/verb characters.)
_CN_NON_NAME_CHARS = frozenset(
    # Function words, auxiliaries, conjunctions — almost never in proper project names
    "的了为而非是这那个全部所必须按把被与和或让使其此即则等着过得"
    # High-frequency verbs, predicate fragments, and noise-indicating modifiers
    "无有归匹配声称完倾向强调修改注立成类该项偏"
)

# A real Chinese project name rarely exceeds 8 characters;
# names longer than this are almost certainly sentence fragments.
_PROJECT_CN_MAX_LEN = 8

# Characters that indicate a tech default-fallback name is actually a phrase
_CJK_PUNCTUATION = "、，。：；！？「」『』《》（）()…—·"


def _looks_like_tech_term(name: str) -> bool:
    """Reject fallback tech names that look like a sentence fragment."""
    if any(ch in name for ch in _CJK_PUNCTUATION):
        return False
    # Either starts with ASCII letter (Python, ESP32) OR is short pure-CJK term
    if name and name[0].isascii() and name[0].isalpha():
        return True
    cjk_only = all('一' <= c <= '鿿' for c in name if c.strip())
    return cjk_only and 2 <= len(name) <= 8


def _strip_verb_prefix(name: str) -> str:
    """Remove a known verb prefix from a potential project name.

    If stripping leaves fewer than 2 characters, the result is considered
    invalid, and the caller should skip the match.
    """
    for v in _PROJECT_VERB_PREFIXES:
        if name.startswith(v) and len(name) > len(v):
            return name[len(v):]
    return name


def _is_valid_project_name(name: str) -> bool:
    """True only when name looks like a proper noun (not a sentence fragment).

    Rejects names that contain CJK function words, verb fragments,
    structural markers, or exceed reasonable project name length.
    """
    if not (2 <= len(name) <= _PROJECT_CN_MAX_LEN):
        return False
    for w in _PROJECT_STRUCTURAL_WORDS:
        if w in name:
            return False
    if any(ch in _CN_NON_NAME_CHARS for ch in name):
        return False
    return True

# Time/duration descriptions that are not meaningful entities
_DURATION_RE = re.compile(r'^[\d\-~至到]+[hHdDwW月年天周）)]?$|^[\d.]+\s*[hHdDwW月年天周]$')

# File path pattern
_FILE_PATH_PATTERN = re.compile(
    r"`?([a-zA-Z0-9_./\-]+\.[a-zA-Z]{1,10})`?"
)

# --- Level 2: Heuristic patterns ---

# Decision patterns
_DECISION_PATTERNS = [
    re.compile(r"决定(.+?)(?:[，,。；;]|$)"),
    re.compile(r"采用(.+?)方案"),
    re.compile(r"放弃(.+?)(?:[，,。；;]|$)"),
    re.compile(r"选择(.+?)(?:[，,。；;]|$)"),
]

# Preference patterns
_PREFERENCE_PATTERNS = [
    re.compile(r"用户(?:偏好|倾向|喜欢|习惯)(.+?)(?:[，,。；;]|$)"),
    re.compile(r"用户要求(.+?)(?:[，,。；;]|$)"),
]

# Risk patterns
_RISK_PATTERNS = [
    re.compile(r"(.+?)风险"),
    re.compile(r"(.+?)注意"),
    re.compile(r"(.+?)避免"),
    re.compile(r"(.+?)可能导致(.+)"),
]

# Completed task patterns
_TASK_COMPLETED_PATTERNS = [
    re.compile(r"完成了(.+?)(?:[，,。；;]|$)"),
    re.compile(r"实现了(.+?)(?:[，,。；;]|$)"),
    re.compile(r"部署了(.+?)(?:[，,。；;]|$)"),
    re.compile(r"已(?:完成|实现|部署)(.+?)(?:[，,。；;]|$)"),
]

# Todo task patterns
_TASK_TODO_PATTERNS = [
    re.compile(r"后续(.+?)(?:[，,。；;]|$)"),
    re.compile(r"计划(.+?)(?:[，,。；;]|$)"),
    re.compile(r"预计(.+?)(?:[，,。；;]|$)"),
    re.compile(r"待(?:办|完成|处理)(.+?)(?:[，,。；;]|$)"),
]


@dataclass
class ExtractedEntity:
    id: str
    type: str          # project / decision / preference / fact / risk / task / tech / idea
    name: str          # human-readable name
    summary: str       # extracted summary text
    source_lines: str  # JSON array of "file:start:end"
    metadata: dict = field(default_factory=dict)  # tags, status, etc.


def slugify(name: str) -> str:
    """Convert a name to lowercase_slug format for entity IDs.

    - Chinese characters are converted to pinyin (no tone marks)
    - Non-alphanumeric chars become underscores
    - Consecutive underscores collapsed
    - Truncated to 30 chars for stability
    """
    # Convert Chinese characters to pinyin
    parts = lazy_pinyin(name)
    s = "".join(parts).lower().strip()
    # Replace non-alphanumeric with underscore
    s = re.sub(r"[^a-z0-9]", "_", s)
    # Collapse consecutive underscores
    s = re.sub(r"_+", "_", s)
    # Strip leading/trailing underscores
    s = s.strip("_")
    # Truncate
    return s[:30]


def generate_entity_id(entity_type: str, name: str) -> str:
    """Generate entity_id from type prefix + slug."""
    prefix = TYPE_PREFIX.get(entity_type, entity_type)
    slug = slugify(name)
    return f"{prefix}:{slug}"


# --- Individual extractors (return list[dict]) ---


def extract_projects(text: str) -> list[dict]:
    """Extract project name entities from text using language-agnostic patterns.

    Detects:
    - Chinese: {name}项目 (e.g. "心连心项目" → 心连心)
    - English: CamelCase words (e.g. HomeBrain, OpenClaw)
    - English: "X Project" (e.g. "HomeBrain project")
    - Latin names immediately before 项目 (e.g. "ExampleProject项目")
    """
    results: list[dict] = []
    seen: set[str] = set()
    latin_spans: set[tuple[int, int]] = set()

    # Pattern 1a: Latin project name + optional whitespace + 项目 marker
    # (e.g. "ExampleProject项目", "HomeBrain 项目", "Esp32 项目")
    for m in _PROJECT_LATIN_MARKER_RE.finditer(text):
        name = m.group(1).strip()
        if name and name not in seen and not _is_garbage_name(name) and len(name) >= 3:
            seen.add(name)
            latin_spans.add(m.span(1))
            results.append(_make_entity("project", name, text))

    # Pattern 1b: Chinese project references: {name}项目
    for m in _PROJECT_CN_RE.finditer(text):
        # Skip if this span was already captured by the Latin-before-CN pattern
        if m.span(1) in latin_spans:
            continue
        raw_name = m.group(1).strip()
        name = _strip_verb_prefix(raw_name)
        if name and _is_valid_project_name(name) and name not in seen and not _is_garbage_name(name):
            seen.add(name)
            results.append(_make_entity("project", name, text))

    # Pattern 3: "X Project" or "X project"
    for m in _PROJECT_EN_RE.finditer(text):
        name = m.group(1).strip()
        if name and name not in seen and not _is_garbage_name(name):
            seen.add(name)
            results.append(_make_entity("project", name, text))

    return results


def extract_tech_keywords(text: str) -> list[dict]:
    """Extract technology keyword entities from text using pattern-based detection.

    Detects:
    - ALL_CAPS acronyms (2-8 chars): ESP32, STM32, MQTT, HA, API, SQL
    - CamelCase tool names: HomeAssistant, TypeScript, NodeJs
    - Versioned terms: Python3, ESP32-S3, v2.0
    - Backtick-quoted code references: `nginx.conf`, `device_aggregator.py`
    - File extension references: .py, .js, .yaml, .json
    No hardcoded keyword list — terms are auto-discovered from the content.
    """
    results: list[dict] = []
    seen: set[str] = set()

    # Pattern 1: Backtick-quoted references (most reliable — explicit code mention)
    for m in _TECH_BACKTICK_RE.finditer(text):
        kw = m.group(1).strip()
        _add_tech(kw, results, seen)

    # Pattern 2: File extensions in context
    for m in _TECH_EXT_RE.finditer(text):
        kw = m.group(1).strip()
        _add_tech(kw, results, seen)

    # Pattern 3: Capitalized English words embedded in Chinese text (e.g. 使用Python开发 → Python)
    for m in _TECH_CN_SURROUND_RE.finditer(text):
        kw = m.group(1)
        if len(kw) >= 2:
            _add_tech(kw, results, seen)

    # Pattern 4: ALL_CAPS acronyms
    for m in _TECH_ACRONYM_RE.finditer(text):
        kw = m.group(1)
        if len(kw) >= 2:
            _add_tech(kw, results, seen)

    # Pattern 5: CamelCase tool/framework names (4+ chars to avoid noise)
    for m in _TECH_CAMEL_RE.finditer(text):
        kw = m.group(1)
        if len(kw) >= 4:
            _add_tech(kw, results, seen)

    # Pattern 6: Versioned terms (at least 3 chars)
    for m in _TECH_VERSION_RE.finditer(text):
        kw = m.group(1)
        if len(kw) >= 3 and any(c.isdigit() for c in kw):
            _add_tech(kw, results, seen)

    return results


def _add_tech(kw: str, results: list[dict], seen: set[str]) -> None:
    """Add a tech keyword entity if valid and not a duplicate."""
    kw_lower = kw.lower().rstrip(".,;:!?)")
    if kw_lower in seen:
        return
    if not kw_lower or len(kw_lower) < 2:
        return
    if _is_garbage_name(kw):
        return
    # Skip generic tech terms — they add noise without signal
    if kw.strip().upper() in _TECH_COMMON_WORDS:
        return
    # Skip structural markers posing as tech
    if _STRUCTURAL_TECH_RE.match(kw.strip()):
        return
    # Exclude common English words that happen to match patterns
    if kw_lower in {"the", "and", "for", "are", "was", "all", "can", "has", "had",
                     "not", "but", "our", "you", "his", "her", "its", "who", "how",
                     "new", "now", "one", "two", "ten", "get", "set", "put", "use"}:
        return
    seen.add(kw_lower)
    results.append(_make_entity("tech", kw, f"技术栈关键词: {kw}"))


def _make_entity(entity_type: str, name: str, summary: str) -> dict:
    """Build a minimal entity dict for extractor output."""
    return {
        "id": generate_entity_id(entity_type, name),
        "type": entity_type,
        "name": name,
        "summary": summary[:200],
    }


def extract_decisions(text: str) -> list[dict]:
    """Extract decision entities from text using heuristic patterns."""
    results = []
    for pat in _DECISION_PATTERNS:
        for m in pat.finditer(text):
            content = m.group(1).strip()
            if not content:
                continue
            name = _compact_name(content, 40)
            eid = generate_entity_id("decision", name)
            results.append({
                "id": eid,
                "type": "decision",
                "name": name,
                "summary": text[:200],
            })
    return results


def extract_preferences(text: str) -> list[dict]:
    """Extract preference entities from text."""
    results = []
    for pat in _PREFERENCE_PATTERNS:
        for m in pat.finditer(text):
            content = m.group(1).strip()
            if not content:
                continue
            name = _compact_name(content, 40)
            eid = generate_entity_id("preference", name)
            results.append({
                "id": eid,
                "type": "preference",
                "name": name,
                "summary": text[:200],
            })
    return results


def extract_risks(text: str) -> list[dict]:
    """Extract risk entities from text."""
    results = []
    for pat in _RISK_PATTERNS:
        for m in pat.finditer(text):
            # Build name from matched groups
            groups = [g for g in m.groups() if g]
            name = "：".join(g.strip() for g in groups if g.strip())[:40]
            if not name:
                name = text[:30]
            name = _compact_name(name, 40)
            eid = generate_entity_id("risk", name)
            results.append({
                "id": eid,
                "type": "risk",
                "name": name,
                "summary": text[:200],
            })
    return results


def extract_tasks(text: str) -> list[dict]:
    """Extract task entities from text (both completed and todo)."""
    results = []

    # Completed tasks
    for pat in _TASK_COMPLETED_PATTERNS:
        for m in pat.finditer(text):
            content = m.group(1).strip()
            if not content:
                continue
            name = _compact_name(content, 40)
            eid = generate_entity_id("task", f"完成_{name}")
            results.append({
                "id": eid,
                "type": "task",
                "name": name,
                "summary": text[:200],
                "status": "completed",
            })

    # Todo tasks
    for pat in _TASK_TODO_PATTERNS:
        for m in pat.finditer(text):
            content = m.group(1).strip()
            if not content:
                continue
            name = _compact_name(content, 40)
            eid = generate_entity_id("task", f"计划_{name}")
            results.append({
                "id": eid,
                "type": "task",
                "name": name,
                "summary": text[:200],
                "status": "todo",
            })

    return results


# --- Section-level extraction ---


def _is_garbage(text: str) -> bool:
    """Return True if the text looks like a DMA processing artifact, not real knowledge."""
    stripped = text.strip()
    if stripped in _GARBAGE_NAMES:
        return True
    if len(stripped) <= _GARBAGE_NAME_MIN_LENGTH:
        return True
    for pat in _GARBAGE_PATTERNS:
        if pat.match(stripped):
            return True
    # Operational log items: cron config, backup status, system daemon notifications
    if _OPS_LOG_KEYWORDS_RE.search(stripped) and len(stripped) < 200:
        return True
    return False


def _is_garbage_name(name: str) -> bool:
    """Return True if the extracted entity name is not meaningful."""
    stripped = name.strip()
    if not stripped:
        return True
    if stripped in _GARBAGE_NAMES:
        return True
    # Filter timestamp log entries (e.g. "15:20 UTC - 先生上线")
    if _TIMESTAMP_LOG_RE.match(stripped):
        return True
    # Single-character names are never meaningful
    if len(stripped) <= 1:
        return True
    # Bracket-timestamp format: "[01:03] - ..."
    if _BRACKET_TS_RE.match(stripped):
        return True
    # Check if the name is purely punctuation/symbols
    if all(c in '（）()""''""【】[]{}，。！？、；：…·*' for c in stripped):
        return True
    # Pure numbers (e.g. "275", "3.14")
    if stripped.replace('.', '').replace('-', '').isdigit():
        return True
    # Names starting with markdown formatting artifacts
    if stripped.startswith('**') or stripped.startswith('：') or stripped.startswith('：'):
        return True
    # Duration descriptions like "1-2h）", "4-6周", "2h"
    if _DURATION_RE.match(stripped):
        return True
    return False


def extract_entities_from_item(
    item: ParsedItem,
    category: str,
    source_file: str,
) -> list[ExtractedEntity]:
    """Extract entities from a single parsed item using pattern matching.

    Uses the DMA category to determine primary entity type,
    then runs all applicable extractors.
    """
    text = item.text
    if _is_garbage(text):
        return []
    source_ref = f"{source_file}:{item.line_start}:{item.line_end}"

    # v1.1: tag-driven type takes priority over category
    item_tag = getattr(item, "tag", None)
    if item_tag is not None and item_tag in TAG_TO_TYPE:
        entity_type = TAG_TO_TYPE[item_tag]
    else:
        entity_type = CATEGORY_TO_TYPE.get(category, "fact")

    # v1.1: task status from tag
    task_status_override: str | None = TAG_TO_TASK_STATUS.get(item_tag) if item_tag else None

    seen_ids: set[str] = set()
    entities: list[ExtractedEntity] = []

    def _add(eid: str, etype: str, name: str, summary: str, **meta):
        if eid in seen_ids:
            return
        # Skip entities with garbage names (too short or known junk)
        if _is_garbage_name(name):
            return
        seen_ids.add(eid)
        entities.append(ExtractedEntity(
            id=eid,
            type=etype,
            name=name[:100],
            summary=summary[:500],
            source_lines=json.dumps([source_ref]),
            metadata=meta,
        ))

    # Level 1: Category-specific heuristic extraction (takes priority)
    if entity_type == "decision":
        for d in extract_decisions(text):
            _add(d["id"], "decision", d["name"], d["summary"])
    elif entity_type == "preference":
        for p in extract_preferences(text):
            _add(p["id"], "preference", p["name"], p["summary"])
    elif entity_type == "risk":
        for r in extract_risks(text):
            _add(r["id"], "risk", r["name"], r["summary"])
    elif entity_type == "task":
        for t in extract_tasks(text):
            status = t.get("status", "todo")
            _add(t["id"], "task", t["name"], t["summary"], status=status)

    # Level 2: Project name detection (from any category)
    for proj in extract_projects(text):
        _add(proj["id"], "project", proj["name"], proj["summary"])

    # Level 3: Tech keyword detection (generic, runs after category-specific)
    for tech in extract_tech_keywords(text):
        _add(tech["id"], "tech", tech["name"], tech["summary"])

    # If no entity found, create a default entity for the item
    name = _extract_entity_name(text, entity_type)
    # Skip default entities that add no structure: if the name is a long
    # prefix of the text (name covers >80% of text), it's a redundant copy.
    text_stripped = text.strip()
    if item_tag is None and len(name) >= 4 and name == text_stripped[:len(name)] and len(name) > len(text_stripped) * 0.8:
        return entities
    # For tech type, reject fallback names that look like sentence fragments
    if entity_type == "tech" and not _looks_like_tech_term(name):
        return entities
    eid = generate_entity_id(entity_type, name)
    if task_status_override is not None and entity_type == "task":
        _add(eid, entity_type, name, text, status=task_status_override)
    else:
        _add(eid, entity_type, name, text)

    return entities


def extract_entities_from_section(
    section: ParsedSection,
    source_file: str,
    dma_category: str | None = None,
) -> list[ExtractedEntity]:
    """Extract entities from a parsed section using rules.

    v1.1: items with skip_extraction=True are bypassed (raw detail subsection
    content; cloud summary will provide higher-quality entities).
    """
    all_entities: list[ExtractedEntity] = []
    seen_ids: set[str] = set()
    cat = dma_category if dma_category else section.category

    for item in section.items:
        # v1.1: skip items flagged as raw-detail (### 原始细节)
        if getattr(item, "skip_extraction", False):
            continue
        item_entities = extract_entities_from_item(item, cat, source_file)
        for entity in item_entities:
            if entity.id in seen_ids:
                continue
            seen_ids.add(entity.id)
            all_entities.append(entity)

    return all_entities


def _text_overlap_ratio(a: str, b: str) -> float:
    """Return Jaccard-like character overlap ratio between two strings."""
    set_a = set(a)
    set_b = set(b)
    if not set_a:
        return 0.0
    return len(set_a & set_b) / len(set_a)


def _compact_name(raw: str, max_len: int = 40) -> str:
    """Generate a concise entity name from raw text.

    Truncates at the first clause boundary (。；，) within max_len.
    If still too long, hard-truncates and adds ellipsis.
    """
    s = raw.strip()
    # Try breaking at clause boundaries
    for sep in ("。", "；", "，", ", ", "; "):
        idx = s.find(sep)
        if 4 < idx <= max_len:
            s = s[:idx]
            break
    if len(s) > max_len:
        s = s[:max_len - 1] + "…"
    return s


def _extract_entity_name(text: str, entity_type: str) -> str:
    """Extract a short meaningful name from text based on entity type."""
    # Try sentence patterns for decisions
    if entity_type == "decision":
        for pat in _DECISION_PATTERNS:
            m = pat.search(text)
            if m:
                return _compact_name(m.group(1).strip(), 30)

    # Preference patterns
    if entity_type == "preference":
        for pat in _PREFERENCE_PATTERNS:
            m = pat.search(text)
            if m:
                return _compact_name(m.group(1).strip(), 30)

    # Risk patterns
    if entity_type == "risk":
        for pat in _RISK_PATTERNS:
            m = pat.search(text)
            if m:
                groups = [g for g in m.groups() if g and g.strip()]
                return _compact_name("：".join(g.strip() for g in groups), 30)

    # Task patterns
    if entity_type == "task":
        for pat in _TASK_COMPLETED_PATTERNS + _TASK_TODO_PATTERNS:
            m = pat.search(text)
            if m:
                return _compact_name(m.group(1).strip(), 30)

    # Default: first clause or first 30 chars
    first = text.split("。")[0].split("；")[0].split("，")[0]
    return _compact_name(first, 30)
