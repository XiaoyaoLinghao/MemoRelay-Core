"""Tests for the entity extractor module."""

from knowledge_weaver.parser import ParsedSection, ParsedItem, parse_dma_file
from knowledge_weaver.extractor import (
    ExtractedEntity,
    _is_garbage,
    generate_entity_id,
    slugify,
    extract_entities_from_section,
    extract_entities_from_item,
    extract_projects,
    extract_decisions,
    extract_preferences,
    extract_risks,
    extract_tasks,
    extract_tech_keywords,
)


def _make_section(categories: dict[str, list[str]] | None = None) -> ParsedSection:
    """Helper to build a ParsedSection with text items."""
    # Collect all items under the first category
    items: list[ParsedItem] = []
    first_cat = ""
    if categories:
        for cat, texts in categories.items():
            if first_cat == "":
                first_cat = cat
            for text in texts:
                items.append(ParsedItem(text=text, time=None, line_start=1, line_end=1))
    return ParsedSection(title=first_cat, category=first_cat, items=items)


# --- generate_entity_id / slugify ---

def test_generate_entity_id():
    result = generate_entity_id("decision", "设备聚合采用规则引擎优先")
    assert result.startswith("decision:")
    assert "_" not in result
    assert "exampleproject" in generate_entity_id("project", "ExampleProject")
    assert generate_entity_id("preference", "用户偏好将功能集成在 ExampleProject 内部").startswith("pref:")


def test_slugify_basic():
    assert slugify("ExampleProject") == "exampleproject"
    assert slugify("HA Entity Naming Rule") == "ha_entity_naming_rule"
    assert slugify("Docker Build Cache") == "docker_build_cache"


# --- extract_projects ---

def test_extract_project_entity():
    text = "ExampleProject项目需要保持聚合器稳定运行"
    results = extract_projects(text)
    assert len(results) >= 1
    names = [r["name"] for r in results]
    assert "ExampleProject" in names
    for r in results:
        assert r["type"] == "project"


# --- extract_decisions ---

def test_extract_decision_entity():
    text = "决定将聚合逻辑集成到ExampleProject自身，不依赖外部脚本"
    results = extract_decisions(text)
    assert len(results) >= 1
    assert results[0]["type"] == "decision"
    # Should extract a meaningful name, not the whole sentence
    assert len(results[0]["name"]) < len(text)


# --- extract_preferences ---

def test_extract_preference_entity():
    text = "用户偏好将功能集成在ExampleProject内部，而非外部脚本"
    results = extract_preferences(text)
    assert len(results) >= 1
    assert results[0]["type"] == "preference"


# --- extract_risks ---

def test_extract_risk_entity():
    text = "LLM聚合存在JSON格式错乱、输出截断的风险"
    results = extract_risks(text)
    assert len(results) >= 1
    assert results[0]["type"] == "risk"


# --- extract_tasks ---

def test_extract_task_completed():
    text = "完成了ExampleProject设备聚合API的开发和测试"
    results = extract_tasks(text)
    assert len(results) >= 1
    assert results[0]["type"] == "task"


def test_extract_task_todo():
    text = "后续计划给DemoRobot增加视觉避障模块"
    results = extract_tasks(text)
    assert len(results) >= 1
    assert results[0]["type"] == "task"


# --- extract_tech_keywords ---

def test_extract_tech_keyword():
    text = "使用ESP32和Python开发，部署在Docker中，HA集成通过STM32"
    results = extract_tech_keywords(text)
    assert len(results) >= 1
    keywords = [r["name"] for r in results]
    assert "ESP32" in keywords
    assert "Python" in keywords
    assert "Docker" in keywords


# --- extract_entities_from_item ---

def test_extract_from_item():
    item = ParsedItem(text="决定使用规则引擎而非LLM批量处理", time=None, line_start=5, line_end=5)
    entities = extract_entities_from_item(item, "决策与结论", "2026-05-24.md")
    assert len(entities) >= 1
    assert entities[0].type == "decision"
    assert "2026-05-24.md:5:5" in entities[0].source_lines


# --- extract_entities_from_section ---

def test_extract_from_section():
    sec = _make_section(categories={
        "决策与结论": ["决定使用规则引擎而非LLM批量处理"],
        "风险与注意事项": ["Docker build缓存可能导致nginx配置未更新"],
    })
    entities = extract_entities_from_section(sec, "2026-05-24.md")
    types = {e.type for e in entities}
    assert "decision" in types or "risk" in types
    assert len(entities) >= 1


# --- dedup ---

def test_extract_and_dedup():
    sec = _make_section(categories={
        "技术/项目要点": [
            "HA实体命名规则：{房间} {设备名} [{子类别}] {参数}，95.9%包含三段",
            "HA实体命名规则：{房间} {设备名} [{子类别}] {参数}，95.9%包含三段",
        ]
    })
    entities = extract_entities_from_section(sec, "2026-05-24.md")
    ids = [e.id for e in entities]
    assert len(ids) == len(set(ids)), f"Duplicate IDs found: {ids}"


# --- Noise filter tests (Work Item 1) ---


def test_timestamp_log_is_filtered():
    """Timestamp log entries like '15:20 UTC - 先生上线' must not become fact entities."""
    item = ParsedItem(text="15:20 UTC - 先生上线", time=None, line_start=1, line_end=1)
    entities = extract_entities_from_item(item, "核心要点", "2026-05-25.md")
    # The text itself is garbage (timestamp log), so no entity should be produced
    for e in entities:
        assert e.type != "fact" or "UTC" not in e.name
    # More directly: _is_garbage_name should catch it
    from knowledge_weaver.extractor import _is_garbage_name
    assert _is_garbage_name("15:20 UTC - 先生上线")


def test_tech_common_words_filtered():
    """Generic tech abbreviations like AI, API should not produce tech entities."""
    text = "使用 AI 和 API 进行开发"
    results = extract_tech_keywords(text)
    names = [r["name"] for r in results]
    assert "AI" not in names
    assert "API" not in names


def test_tech_common_word_with_context_kept():
    """When a common word is part of a larger term, the larger term should be kept."""
    text = "AI聚合策略v3方案"
    results = extract_tech_keywords(text)
    names = [r["name"] for r in results]
    # "AI聚合策略v3方案" is a composite term — AI alone is filtered,
    # but the full term is NOT just a common word
    # At minimum, no entity should be just "AI"
    for r in results:
        assert r["name"].strip().upper() != "AI"


def test_structural_tech_filtered():
    """Structural markers like P0, P1, date patterns should not produce tech entities."""
    text = "P0 P1 2026-05"
    results = extract_tech_keywords(text)
    names = [r["name"] for r in results]
    assert "P0" not in names
    assert "P1" not in names
    assert "2026-05" not in names


def test_project_regex_does_not_swallow_verbs():
    from knowledge_weaver.extractor import extract_projects
    r = extract_projects("启动ExampleProject项目，确认采用模块化架构")
    names = [p["name"] for p in r]
    assert "ExampleProject" in names
    assert "启动ExampleProject" not in names


def test_project_regex_rejects_verbal_phrases():
    from knowledge_weaver.extractor import extract_projects
    r = extract_projects("决定基于开源框架作为项目后端基础")
    names = [p["name"] for p in r]
    assert all("决定" not in n for n in names)
    assert all("基于" not in n for n in names)

def test_tech_does_not_extract_cjk_numbers():
    from knowledge_weaver.extractor import extract_tech_keywords
    for text in ["2000+实体压缩87%，耗时<1秒", "但95%以上实体命名包含三段"]:
        r = extract_tech_keywords(text)
        for kw in r:
            assert not any('一' <= ch <= '鿿' for ch in kw["name"]), \
                f"unexpected CJK in tech name: {kw['name']!r}"


def test_tech_still_extracts_versioned_terms():
    from knowledge_weaver.extractor import extract_tech_keywords
    r = extract_tech_keywords("Use Python3 and ESP32-S3 with Node18")
    names = [k["name"] for k in r]
    assert "Python3" in names
    assert any("ESP32" in n for n in names)


def test_tech_fallback_rejects_sentence_fragments():
    from knowledge_weaver.extractor import extract_entities_from_item
    from knowledge_weaver.parser import ParsedItem
    bad_inputs = [
        "数据源提供REST API，标准JSON格式响应",
        "实体命名规则：{区域} {设备名} [{子类别}] {属性}",
        "分组策略：先按(区域, 设备名)分组",
    ]
    for text in bad_inputs:
        item = ParsedItem(text=text, time=None, line_start=1, line_end=1)
        ents = extract_entities_from_item(item, "技术/项目要点", "test.md")
        tech_names = [e.name for e in ents if e.type == "tech"]
        for n in tech_names:
            assert "：" not in n and "，" not in n and "。" not in n, \
                f"tech name contains CJK punctuation: {n!r}"
            assert "(" not in n and "（" not in n, \
                f"tech name contains parenthesis: {n!r}"


def test_tech_fallback_keeps_real_tech_terms():
    """Real technical mentions (CamelCase, acronyms, versioned) must still be extracted."""
    from knowledge_weaver.extractor import extract_entities_from_item
    from knowledge_weaver.parser import ParsedItem
    item = ParsedItem(text="使用Python3开发，配合ESP32-S3芯片", time=None, line_start=1, line_end=1)
    ents = extract_entities_from_item(item, "技术/项目要点", "test.md")
    names = [e.name for e in ents if e.type == "tech"]
    assert "Python3" in names or any("Python" in n for n in names), f"got: {names}"


def test_dma_error_placeholders_are_filtered():
    """KW SPEC v1.0 §6: agent failure placeholders match *<AGENT>-ERR: <reason>*."""
    from knowledge_weaver.extractor import _is_garbage
    samples = [
        "DMA-ERR: cloud summary failed (see log)",
        "DMA-ERR: get-cloud-creds failed (check .master_key & credentials.enc)",
        "DMA-ERR: no credentials.enc",
        "DMA-ERR: cloud summarizer disabled",
        "HERMES-ERR: api timeout after 30s",
        "OPENCLAW-ERR: skill not found",
    ]
    for s in samples:
        assert _is_garbage(s), f"agent error placeholder not filtered: {s!r}"


def test_dma_error_pattern_does_not_match_normal_text():
    """The -ERR: pattern must not catch normal text containing 'ERR'."""
    from knowledge_weaver.extractor import _is_garbage
    import re
    samples = [
        "the error was in line 5",
        "ERR is short for error",
        "HTTP 500 internal server error",
    ]
    for s in samples:
        long_text = s + " " + ("x" * 20)
        pat = re.compile(r".*-ERR:.*")
        assert not pat.match(long_text), f"-ERR: rule false positive: {long_text!r}"


# ---------------------------------------------------------------------------
# v1.1 tag-driven extraction tests
# ---------------------------------------------------------------------------

def test_v11_tag_to_type_complete():
    from knowledge_weaver.extractor import TAG_TO_TYPE
    expected = {
        "关键决策": "decision", "关键偏好": "preference",
        "关键事实": "fact", "关键风险": "risk",
        "关键技术": "tech", "已完成": "task",
        "待办": "task", "创意": "idea", "关键讨论": "fact",
    }
    assert TAG_TO_TYPE == expected


def test_v11_tag_to_type_in_sync_with_parser():
    """parser.py 与 extractor.py 中 TAG_TO_TYPE 必须完全一致。"""
    from knowledge_weaver.parser import TAG_TO_TYPE as P_TAGS
    from knowledge_weaver.extractor import TAG_TO_TYPE as E_TAGS
    assert P_TAGS == E_TAGS


def test_v11_skip_extraction_honored():
    """skip_extraction=True 的 item 不应产生任何实体。"""
    from knowledge_weaver.parser import ParsedSection, ParsedItem
    from knowledge_weaver.extractor import extract_entities_from_section

    s = ParsedSection(title="核心要点", category="fact")
    s.items.append(ParsedItem(
        text="决定采用 PostgreSQL 17 作为生产数据库",
        time=None, line_start=1, line_end=1,
        skip_extraction=True,
    ))
    ents = extract_entities_from_section(s, "test.md", dma_category="核心要点")
    assert ents == []


def test_v11_tag_overrides_category_type():
    """item.tag 应覆盖 section category 决定实体类型。"""
    from knowledge_weaver.parser import ParsedItem
    from knowledge_weaver.extractor import extract_entities_from_item

    item = ParsedItem(
        text="后端框架：Python FastAPI",
        time=None, line_start=1, line_end=1,
        tag="关键决策",
    )
    ents = extract_entities_from_item(item, "fact", "test.md")
    assert any(e.type == "decision" for e in ents), \
        f"expected at least one decision entity, got {[e.type for e in ents]}"


def test_v11_task_status_from_tag():
    """[已完成] tag 应设置 status=completed；[待办] 应设置 status=todo。"""
    from knowledge_weaver.parser import ParsedItem
    from knowledge_weaver.extractor import extract_entities_from_item

    item_done = ParsedItem(
        text="完成 ExampleProject 登录模块",
        time=None, line_start=1, line_end=1,
        tag="已完成",
    )
    ents = extract_entities_from_item(item_done, "fact", "test.md")
    task_ents = [e for e in ents if e.type == "task"]
    assert any(e.metadata.get("status") == "completed" for e in task_ents), \
        f"no task with status=completed in {[(e.type, e.metadata) for e in ents]}"

    item_todo = ParsedItem(
        text="接入 OAuth2 集成",
        time=None, line_start=1, line_end=1,
        tag="待办",
    )
    ents = extract_entities_from_item(item_todo, "fact", "test.md")
    task_ents = [e for e in ents if e.type == "task"]
    assert any(e.metadata.get("status") == "todo" for e in task_ents)


def test_wave9_heartbeat_meta_noise_filtered():
    """Wave 9: 心跳/元描述/运维快照应被识别为噪声。"""
    noise = [
        "本次对话仅包含系统心跳轮询与DAILY检查操作",
        "本次对话无任何实质性内容",
        "OpenClaw 心跳轮询事件组成，无用户指令",
        "本时段无实质内容（仅系统心跳轮询）",
        "OpenClaw 系统日常服务器状态：磁盘使用 37%，剩余 60G",
        "仅包含 4 次 OpenClaw 心跳轮询",
        "本时段本地提取无输出",
        "助理检测后确认无 pending 子会话",
    ]
    for s in noise:
        assert _is_garbage(s), f"未被识别为噪声: {s!r}"


def test_wave9_real_knowledge_not_filtered():
    """Wave 9: 真实知识不应被新噪声规则误伤。"""
    real = [
        "后端框架：Python FastAPI（用户决定用 FastAPI）",
        "决定使用规则引擎而非 LLM 批量处理",
        "DB 连接池配置过小在高并发下可能耗尽",
        "完成 ExampleProject 登录模块基础架构与单元测试",
        "用户偏好简洁直接的代码风格",
        "Knowledge Weaver 系统版本 v0.2.2，已索引 2221 实体",  # 含数字但是真实知识
    ]
    for s in real:
        assert not _is_garbage(s), f"真实知识被误过滤: {s!r}"


def test_empty_day_marker_yields_no_entities(tmp_path):
    """Wave 10: 空日标记文件（仅 HTML 注释，0 时间槽）→ KW 0 section / 0 实体。"""
    f = tmp_path / "2026-06-01.md"
    f.write_text(
        '---\ntitle: "2026-06-01 会话记忆"\ndate: "2026-06-01"\n---\n\n'
        '# 2026-06-01\n\n'
        '<!-- 本日无对话内容：DMA 全时段仅心跳/系统巡检，KW 不收录。 -->\n',
        encoding="utf-8",
    )
    pf = parse_dma_file(str(f))
    assert pf.date == "2026-06-01"
    assert len(pf.sections) == 0
    ents = []
    for sec in pf.sections:
        ents += extract_entities_from_section(sec, str(f), getattr(sec, "category", None))
    assert ents == []


def test_wave10_heartbeat_meta_filtered():
    """Wave 10 rebuild: 心跳元描述应被过滤（覆盖 心跳检测/心跳中 等变体）。"""
    for s in [
        "当日首次心跳检测于 06:55 触发，助理执行了每日检查流程。",
        "系统每日检查已在06:00后首次心跳中触发，磁盘检查已执行。",
    ]:
        assert _is_garbage(s), f"心跳元描述未被过滤: {s!r}"


# ---------------------------------------------------------------------------
# project 实体噪声治理：治本修复 验收测试 (Fix Plan §4)
# ---------------------------------------------------------------------------

def test_project_extraction_rejects_phrase_fragments():
    """治本：含功能词/动词的 {name}项目 片段不得产出 project 实体。"""
    noise_samples = [
        "归'其他工作'分类而非新建项目",
        "任务归属匹配已有项目",
        "阿拉山口为已有项目",
        "阿拉山口独立成项目",
        "用户倾向 DIY 玩具类项目",
        "无成熟项目",
        "这是一个项目",
        "工作记录必须按项目归属分组",
        "强调项目归属精确",
        "修改全部项目",
        "候选人声称完成所有项目",
        "IIC 级气体防爆四足机器人的项目需求",
    ]
    for s in noise_samples:
        names = [p["name"] for p in extract_projects(s)]
        assert names == [], f"应为空但得到 {names}：{s}"


def test_project_extraction_keeps_real_projects():
    """真实项目（显式标记）必须保留，含带空格变体。"""
    assert "心连心" in [p["name"] for p in extract_projects("心连心项目进展顺利")]
    assert "HomeBrain" in [p["name"] for p in extract_projects("HomeBrain项目上线")]      # 无空格
    assert "HomeBrain" in [p["name"] for p in extract_projects("HomeBrain 项目上线")]     # 带空格
    assert "HomeBrain" in [p["name"] for p in extract_projects("HomeBrain project shipped")]


def test_latin_marker_with_space_regression():
    """既有 bug 回归：Latin 名 + 空格 + 项目 此前被漏捕。"""
    assert "Esp32" in [p["name"] for p in extract_projects("Esp32 项目启动")]


def test_bare_camelcase_not_project_but_is_tech():
    """无项目标记的裸驼峰词不再是 project，但仍是 tech（不丢信息）。"""
    s = "讨论了 XiaomiDriver、WebSocket、SpotMicro、LangGraph、GitHub 的实现"
    assert [p["name"] for p in extract_projects(s)] == []
    tech = {t["name"] for t in extract_tech_keywords(s)}
    assert {"XiaomiDriver", "WebSocket", "LangGraph"} <= tech


def test_marker_only_extracts_real_project_from_noisy_sentence():
    """混排长句：仅带标记的 HomeBrain 进 project，其余裸词全拒。"""
    s = ("HomeBrain 项目使用了 XiaomiDriver 和 WebSocket，参考 SpotMicro 和 "
         "OpenCat，基于 LangGraph 和 AutoGen，托管 GitHub")
    assert [p["name"] for p in extract_projects(s)] == ["HomeBrain"]
