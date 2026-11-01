# Knowledge Weaver Memory File Specification (v1.1)

> **本文档是规范性参考（Normative Reference）**。任何为 Knowledge Weaver (KW) 提供输入数据的 agent / skill / 工具必须严格遵守本文规定。
> KW 端解析器以本规范为唯一权威；不在本规范内的格式 KW 不保证识别。

**版本**：1.1
**生效**：2026-05-29
**前版本**：v1.0（2026-05-28）
**适用范围**：任何写入 KW `memory_dir` 的 markdown 文件

## 0. 与上一版的差异（v1.0 → v1.1）

| 变更 | 类型 | 说明 |
|---|---|---|
| §3.4 新增 H3 子分区规则 | 新增 | 区分"原始细节"与"摘要"两种内容形态 |
| §4.4 新增 tag-based 实体标记 | 新增 | 在摘要子分区内用 `[关键X]` 标签代替 `**xxx**` 分类 |
| §5.3 续行规则明确化 | 澄清 | tag 行的多行续写规则 |
| §11 新增**不变量条款** | 新增 | KW MUST NOT 索引 MEMORY.md 等非派生文件 |

**向后兼容**：v1.1 KW 解析器仍能正确解析 v1.0 文件——只是不会触发新增的 H3/tag 路径。生产者可以渐进切换，无需一次性迁移。

---

## 1. 文件命名与位置

### 1.1 文件名格式

```
<YYYY-MM-DD>.md
```

- **YYYY-MM-DD**：ISO 8601 日期，零填充（如 `2026-05-28`，而非 `2026-5-28`）
- **后缀**：`.md`（小写）
- **特殊前缀**：仅 `sample_` 允许（用于测试 fixtures），如 `sample_2026-05-28.md`

**对应正则**（KW 端使用）：
```python
^(?:sample_)?(\d{4}-\d{2}-\d{2})\.md$
```

### 1.2 不符合规范的反例

```
2026-5-28.md           ❌ 月日未零填充
26-05-28.md            ❌ 年份非 4 位
memory-2026-05-28.md   ❌ 自定义前缀（非 sample_）
2026_05_28.md          ❌ 分隔符必须为 -
2026-05-28.markdown    ❌ 扩展名必须为 .md
.2026-05-28.md         ❌ 隐藏文件，KW 跳过
```

### 1.3 目录约定

- 一个 agent 的所有 memory 文件放在一个目录下
- 文件名在该目录内唯一（一天一个文件）
- KW 端通过 `KNOWLEDGE_WEAVER_MEMORY_DIRS` 配置可读取多个目录

---

## 2. 文件编码

- **UTF-8**，无 BOM
- **换行符**：LF（`\n`）；CRLF 会被 KW 容忍但不推荐
- **末尾换行**：建议保留

---

## 3. 文件结构

文件由以下部分组成（按出现顺序）：

```
[Frontmatter] (可选但推荐)
[Body Header] (可选)
[Body Sections]
  [Time Slots] (Format B 推荐)
    [Subsections H3] (v1.1 推荐)
      [Category Markers / Tag Markers / Bullets]
```

### 3.1 YAML Frontmatter（可选但推荐）

```yaml
---
title: "2026-05-28 会话记忆"
date: "2026-05-28"
---
```

**字段**：

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `title` | string | 否 | 人类可读的标题 |
| `date` | string | 否 | YYYY-MM-DD 格式，必须等于文件名中的日期 |

### 3.2 Body Header（可选，纯装饰）

```markdown
# 2026-05-28
```

- KW 不读取此行的内容
- 仅用于人类浏览友好

### 3.3 Body Sections（必须）

KW 支持两种 body 结构。新接入的 agent **必须**使用 §3.4 的 Format B + H3 子分区结构（v1.1 新增）。

#### Format A：直接分类（v1.0 兼容保留）

```markdown
## 核心要点
- bullet 1

## 决策与结论
- bullet 2
```

仍可解析，但不推荐用于新生产者。

#### Format B：时间槽嵌分类（v1.0 兼容保留）

```markdown
## 10:00
**核心要点**
- bullet 1

**决策与结论**
- bullet 2
```

仍可解析，但建议升级到 §3.4 的子分区结构。

### 3.4 Format B + H3 子分区结构（v1.1 推荐）

```markdown
## 10:00                    ← 时间槽（H2）

### 原始细节                ← 子分区（H3），KW 仅读取不抽取实体
**核心要点**
- 09:30 - 用户原话引用

**决策与结论**
- 用户原话引用
（其余分类按需）

### 摘要                    ← 子分区（H3），KW 从 [关键X] tag 抽取实体
今天 ExampleProject 推进 Python FastAPI 选型。
（自由叙事段落，KW 忽略）

[关键决策] 后端框架：Python FastAPI
[关键偏好] 编辑器：vim
[关键风险] DB 连接池可能耗尽
[待办] 接入 OAuth2
```

**子分区识别规则**：

| H3 标题（精确匹配） | KW 行为 |
|---|---|
| `### 原始细节` / `### original` / `### raw` | 整个子分区内：**只解析 bullet 文本** 用于 OpenClaw memory_search，**不抽取实体到 KW DB** |
| `### 摘要` / `### summary` / `### digest` | 整个子分区内：解析 §4.4 的 `[关键X]` tag 行作为实体；忽略其他段落 |
| 其他 H3 标题 | 当作未知子分区，回退到 v1.0 行为：`**xxx**` 仍按分类识别，bullet 仍抽取 |

**子分区的范围**：H3 标题之后到下一个 H3 或 H2 或文件结尾为止。

---

## 4. 分类标题与标签（规范性）

### 4.1 核心 8 分类（v1.0 起，必须支持）

下列 8 个分类用于 v1.0 Format A/B 与 v1.1 `### 原始细节` 子分区内：

| 标题字符串 | KW 实体类型 | 语义 |
|---|---|---|
| `核心要点` | `fact` | 重要事实、状态、当前情况 |
| `决策与结论` | `decision` | 明确的决定、选择、结论 |
| `已完成事项` | `task` (status=completed) | 已完成的工作、实现、修复 |
| `待办与计划` | `task` (status=todo) | 后续要做的事、计划 |
| `用户偏好与习惯` | `preference` | 用户明确表达的偏好、工作习惯 |
| `技术/项目要点` | `tech` | 项目进展、技术方案、工具选择 |
| `风险与注意事项` | `risk` | 潜在问题、需留意的风险 |
| `创意与想法` | `idea` | 新想法、灵感、可能性 |

### 4.2 扩展分类

| 标题字符串 | KW 实体类型 |
|---|---|
| `关键讨论` | `fact` |

### 4.3 字符串匹配要求

- **逐字符匹配**：不允许大小写差异、不允许多余/缺失字符、不允许标点替换
- KW 端用 Python `dict[str, str]` 查找；任何偏差都会落入"静默忽略"路径

### 4.4 摘要子分区的 Tag-based 实体标记（v1.1 新增）

仅在 `### 摘要` 子分区内有效。tag 用方括号包围，置于行首，**整行作为一个实体**。

**Tag → 实体类型映射表**：

| Tag 字符串 | 实体类型 | 备注 |
|---|---|---|
| `[关键决策]` | `decision` | 必须 |
| `[关键偏好]` | `preference` | 必须 |
| `[关键事实]` | `fact` | 必须 |
| `[关键风险]` | `risk` | 必须 |
| `[关键技术]` | `tech` | 必须 |
| `[已完成]` | `task` (status=completed) | 必须 |
| `[待办]` | `task` (status=todo) | 必须 |
| `[创意]` | `idea` | 可选 |
| `[关键讨论]` | `fact` | 可选（语义同 §4.2） |

**Tag 行格式**：
```
[Tag] 内容文本
```

- Tag 必须**置于行首**（前面无空格）
- Tag 与内容之间用**单个空格**分隔
- 内容文本作为实体的 `summary`，KW 自动从中提取 `name`（首条短语 / 前 30 字符）
- **不要在 tag 之前加 `- `**（与 v1.0 bullet 区分）

**多 tag 行示例**：
```markdown
[关键决策] 后端框架：Python FastAPI（理由：原生 async 支持）
[关键偏好] 编辑器：vim，Python 用 jedi 插件
[关键风险] 数据库连接池配置过小可能在高并发下耗尽
[已完成] 完成登录模块基础架构与单元测试
[待办] 接入第三方 OAuth2 集成（预计本周）
[关键事实] ExampleProject 运行环境：macOS 14.5 + Python 3.11 + PostgreSQL 17
```

### 4.5 反例

```
- [关键决策] xxx        ❌ tag 行不能以 - 开头
 [关键决策] xxx         ❌ 行首有空格
[决策] xxx              ❌ tag 名称不在映射表中
[关键决策]xxx           ❌ 缺少 tag 和内容间的空格
[关键决策] [关键风险]   ❌ 一行不能有两个 tag
```

---

## 5. Bullet 格式

### 5.1 基本格式

```
- <content>
```

- 必须以 `- ` 开头（连字符 + 空格），不能用 `*` 或 `+`

### 5.2 时间戳前缀（可选）

```
- 09:30 - <content>
- 09:30:45 - <content>
```

- 时间戳放在内容前，用 ` - ` 分隔
- 支持 `HH:MM` 和 `HH:MM:SS`

### 5.3 续行（多行 bullet 或 tag 行）

```
- 这是一个较长的事项
  续行内容必须缩进
  可以多行

[关键决策] 这是一个较长的决策
  续行内容也必须缩进
```

- 续行必须以空格或 tab 开头
- 续行内容会被合并到上一个 bullet / tag 的 `text`，用单个空格连接

### 5.4 内容建议

- **长度**：建议 ≤ 280 字符；过长会被 KW 截断到 200 字符存入 `entity.summary`
- 避免表格、嵌套列表、Markdown 代码块
- 允许反引号代码片段、加粗、斜体、中英文混合

---

## 6. 失败 / 占位符约定

当 agent 因外部失败需要在 memory 文件中留痕时，**必须使用以下错误前缀**：

```
- *<AGENT-ID>-ERR: <reason>*
```

例如：
```
- *DMA-ERR: cloud summary failed (see log)*
- *HERMES-ERR: api timeout after 30s*
```

KW 端的 `_GARBAGE_PATTERNS` 已包含通用 `.*-ERR:.*` 规则，覆盖所有遵循此约定的 agent。

---

## 7. 一份合规示例（v1.1 推荐结构）

```markdown
---
title: "2026-05-29 会话记忆"
date: "2026-05-29"
---

# 2026-05-29

## 10:00

### 原始细节

**核心要点**
- 09:30 - 用户：我在 macOS 14.5 上用 Python 3.11 开发 ExampleProject

**决策与结论**
- 10:00 - 用户：决定用 FastAPI，因为 vim 配 jedi 很顺

**已完成事项**
- 10:15 - 完成了 ExampleProject 登录模块的单元测试

**待办与计划**
- 10:30 - 后续接入 OAuth2 集成

**用户偏好与习惯**
- 09:30 - 用户：偏好简洁的代码风格

**技术/项目要点**
- 10:00 - ExampleProject 使用 PostgreSQL 17

**风险与注意事项**
- 11:00 - 数据库连接池高并发耗尽风险

**关键讨论**
- 11:30 - 讨论缓存策略选型，未定论

### 摘要

今天 ExampleProject 推进了 Python FastAPI 后端选型，完成了登录模块基础架构与单元测试。
用户表达了对简洁代码风格与 vim 编辑器的偏好。需注意 PostgreSQL 17 连接池在高并发下的耗尽风险。
讨论了缓存策略，未达成结论。

[关键决策] 后端框架：Python FastAPI（用户原话："决定用 FastAPI，因为 vim 配 jedi 很顺"）
[关键偏好] 编辑器：vim + jedi（用户："vim 配 jedi 很顺"）
[关键偏好] 代码风格：简洁，避免过度抽象
[关键事实] ExampleProject 运行环境：macOS 14.5 + Python 3.11 + PostgreSQL 17
[关键风险] DB 连接池配置过小在高并发下可能耗尽
[已完成] 完成登录模块基础架构与单元测试
[待办] 接入第三方 OAuth2 集成

## 14:30

### 原始细节

**已完成事项**
- 14:00 - 修复了 ExampleProject 项目的 SQL 注入风险

### 摘要

修复了 SQL 注入风险，建议加入自动化测试。

[已完成] 修复 ExampleProject SQL 注入风险（位置：登录模块）
[待办] 为 SQL 注入风险加入回归测试

- *DMA-ERR: cloud summary failed (see log)*
```

---

## 8. 兼容性与版本

- 本规范遵循语义化版本（SemVer）
- v1.x 内任何新增分类/tag 必须不破坏 v1.0 的 8 个核心分类与文件名约定
- 破坏性变更（删除分类、改名、改文件名格式）需升级到 v2.0
- KW 端 README 必须声明所支持的 SPEC 版本

---

## 9. 实施检查清单

新 agent 接入 KW 前请逐项确认：

- [ ] 文件名严格 `YYYY-MM-DD.md`
- [ ] UTF-8 编码无 BOM
- [ ] frontmatter `date` 与文件名一致（若使用 frontmatter）
- [ ] 时间槽用 `## HH:MM` 格式
- [ ] **v1.1 推荐**：使用 `### 原始细节` + `### 摘要` 双子分区结构
- [ ] `### 原始细节` 内 `**xxx**` 分类标题逐字符匹配 §4.1/§4.2
- [ ] `### 摘要` 内 `[关键X]` tag 名称在 §4.4 映射表中
- [ ] bullet 以 `- ` 开头
- [ ] tag 行**不**以 `- ` 开头
- [ ] 失败占位符使用 `*<AGENT>-ERR: ...*` 格式
- [ ] 用 KW 的 `parse_dma_content()` 跑过 dry-run 验证

---

## 10. 验证工具

```bash
.venv/bin/python -c "
from knowledge_weaver.parser import parse_dma_content
import pathlib, sys

filepath = sys.argv[1] if len(sys.argv) > 1 else None
if not filepath:
    print('Usage: spec-check.py <memory.md>')
    sys.exit(1)

content = pathlib.Path(filepath).read_text(encoding='utf-8')
p = parse_dma_content(content)
print(f'title: {p.title!r}')
print(f'date:  {p.date!r}')
print(f'sections: {len(p.sections)}')

KNOWN = {
    '核心要点': 'fact', '决策与结论': 'decision',
    '已完成事项': 'task', '待办与计划': 'task',
    '用户偏好与习惯': 'preference', '技术/项目要点': 'tech',
    '风险与注意事项': 'risk', '创意与想法': 'idea',
    '关键讨论': 'fact',
}

for s in p.sections:
    items = len(s.items)
    if s.title not in KNOWN:
        print(f'  ⚠️  UNKNOWN [{s.category}] {s.title!r} ({items} items)')
    else:
        expected = KNOWN[s.title]
        ok = '✅' if s.category == expected else '❌'
        print(f'  {ok} [{s.category}] {s.title} ({items} items)')
" path/to/your/memory.md
```

---

## 11. 不变量（Invariants，规范性）

**KW 端必须遵守的边界**：

> **I-1**：KW **只能**索引符合 §1.1 文件名格式的文件。MEMORY.md / DREAMS.md / TASKS.md / 任何其它人工维护的文件**永远不被 KW 索引**。
>
> 理由：人工维护文件由 OpenClaw 原生机制（bootstrap auto-load / Read tool）覆盖；KW 索引会引入不一致窗口与重建抖动。

> **I-2**：KW 数据库**必须可从 memory 目录完全重建**（除 access_log 等用户行为表外）。任何让 KW DB 拥有 markdown 中不存在的信息的功能均被禁止。

> **I-3**：KW 端的 `DMA_CATEGORY_MAP` / `CATEGORY_TO_TYPE` / tag mapping 三表必须保持同步；任何对其中一个的修改必须同步另外两个并升级 SPEC 版本。

> **I-4**：KW SPEC 版本号必须能在 KW 仓库的 `README.md` 中查到当前支持版本。

---

## 12. 变更历史

| 版本 | 日期 | 变更 |
|---|---|---|
| 1.0 | 2026-05-28 | 初版规范 |
| 1.1 | 2026-05-29 | 新增 H3 子分区与 tag-based 标记；明确不变量 §11 |
