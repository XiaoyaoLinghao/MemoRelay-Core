# Knowledge Weaver

MCP Server for structured knowledge retrieval on top of DMA daily memory files.

## Overview

Knowledge Weaver provides an MCP (Model Context Protocol) server that enables structured retrieval of knowledge from DMA daily memory files. It extracts entities (decisions, projects, risks, preferences, etc.) and their relationships, stores them in a SQLite database with FTS5 and optional sqlite-vec vector search, and exposes 7 query tools + 3 resources.

## Architecture

```
DMA daily .md files → parser → extractor → linker → scorer → embed → DB
                                                                ↓
                                              MCP tools (7) + resources (3)
```

## Setup

```bash
pip install -e ".[dev]"
```

## Usage

### As an MCP server (stdio)

```bash
python -m knowledge_weaver           # default: starts MCP server
python -m knowledge_weaver serve     # explicit
```

### Run consolidation once (offline)

```bash
python -m knowledge_weaver consolidate
```

### Environment variables

| Variable | Default | Description |
|---|---|---|
| `KNOWLEDGE_WEAVER_DB_PATH` | `$HOME/.openclaw/knowledge/knowledge.db` | SQLite db path |
| `KNOWLEDGE_WEAVER_MEMORY_DIR` | `$HOME/.openclaw/workspace/memory` | DMA daily files dir |
| `KNOWLEDGE_WEAVER_LOG_LEVEL` | `INFO` | logging level |
| `EMBEDDING_BASE_URL` | — | OpenAI-compatible base URL |
| `EMBEDDING_API_KEY` | — | API key |
| `EMBEDDING_MODEL` | — | model name |
| `EMBEDDING_DIMENSION` | `1024` | vector dimension |

## DMA daily memory file format

DMA daily memory files use YAML frontmatter + markdown sections:

```markdown
---
title: Daily Memory
date: 2026-05-24
---

## 核心要点
- key point 1
- key point 2

## 决策与结论
- 决策: desc, 背景: context
- 结论1: desc, 背景: context
```

The 8 recognized section headings map to entity types:
`核心要点`→fact, `决策与结论`→decision, `已完成事项`→task, `待办与计划`→task,
`用户偏好与习惯`→preference, `技术/项目要点`→tech, `风险与注意事项`→risk, `创意与想法`→idea

## Memory File Specification

This server reads memory files following [Memory File Specification v1.1](docs/KW_MEMORY_FILE_SPEC.md).
Any agent producing input for this server (DMA, Hermes, custom skills) must conform to that spec.

Key points:
- File name: `YYYY-MM-DD.md`
- YAML frontmatter with `title` / `date` (recommended)
- 8 core categories + 1 extension category (used in `### 原始细节` subsection)
- 9 entity tags in `### 摘要` subsection (v1.1)
- Failure placeholders use `*<AGENT>-ERR: <reason>*` format
- **KW MUST NOT index MEMORY.md or other manually-maintained files** (v1.1 §11)

See the spec for full details.

## MCP tools

| Tool | Description | Key Parameters |
|---|---|---|
| `knowledge_search` | Search entities by keyword or semantic similarity | query, entity_type?, max_results?, min_score? |
| `knowledge_trace` | Trace a topic's full relationship graph | topic, max_depth? |
| `active_projects` | List currently active projects | lookback_days? |
| `preference_lookup` | Look up user preferences and habits | topic?, domain? |
| `decision_history` | Query historical decisions and rationale | topic, include_risk? |
| `knowledge_stats` | Knowledge base health metrics | — |
| `knowledge_consolidate` | Manually trigger consolidation | — |

## MCP resources

| URI | Description |
|---|---|
| `knowledge://stats` | System status overview |
| `knowledge://entity/{entity_id}` | Individual entity details + relations |
| `knowledge://days/{date}` | Daily manifest entry (YYYY-MM-DD) |

## Optional: sqlite-vec & FTS5

- **sqlite-vec**: Enables vector similarity search. Install with `pip install sqlite-vec`. Falls back gracefully to FTS5 + LIKE if unavailable.
- **FTS5**: Full-text search via `entity_fts` virtual table. Always enabled.

## Development

### DMA runtime health reports

Health monitoring uses the additive [DMA runtime status v1 contract](docs/DMA_RUNTIME_STATUS_V1.md).
DMA owns the status file; KW never runs an archive or writes DMA state. Memory SPEC v1.1,
entity counts and scoring remain unchanged. Legacy `day_count` distributions are informational.

Set `KW_DMA_STATUS_PATH` to DMA's `.runtime_status.json` (default:
`/home/openclaw/.openclaw/skills/daily-memory-archiver/config/.runtime_status.json`).
Missing status during rollout is unknown/warning, not confirmed recovery. A fresh running
record does not clear a previous alert. Existing log errors are historical information;
healthy idle does not require today's Memory file. Unknown pending counts are never zero.

Runtime thresholds in seconds: `KW_DMA_EXPECTED_INTERVAL_SECONDS` (1800),
`KW_DMA_STATUS_TIMEOUT_SECONDS` (5400), `KW_DMA_BACKLOG_TIMEOUT_SECONDS` (86400).
Optional evidence paths: `KW_DMA_LOG_PATH`, `KW_DMA_CHECKPOINT_PATH`,
`KNOWLEDGE_WEAVER_MEMORY_DIR`.

Deploy **all three** `scripts/kw_health_check.py`, `scripts/dma_status.py` and
`scripts/kw_weekly_trend.py` together to the configured script directory. Deploy
`deploy/kw_health_wrapper.py` to the actual scheduler entry point. The wrapper accepts
`daily` or `weekly`, validates a unique subprocess report, and atomically publishes it.
Do not deploy only the wrapper against an older checker CLI.

Wrapper path overrides: `KW_HEALTH_SCRIPT_DIR`, `KW_HEALTH_REPORTS_DIR`,
`KW_WEEKLY_REPORTS_DIR`, `KW_ALERT_FLAG`, `KW_WEEKLY_ALERT_FLAG`.
Weekly history can be redirected with `KW_HEALTH_HISTORY_PATH`; the database with
`KNOWLEDGE_WEAVER_DB_PATH`. Defaults retain the existing deployment paths.

The checker/weekly CLI accept `--run-id ID --output PATH` for invocation-owned JSON
reports; plain weekly invocation still writes Markdown. Daily subprocess exit 0
means a valid report, not necessarily a healthy service. Checker execution failures
exit nonzero; the wrapper emits a checker-failure alert instead of reading old output.
Pending alert consumers must support this failure payload for both daily and weekly.
Weekly failures preserve the existing `kind`/`file`/`summary` shape by publishing a
fresh checker-failure Markdown file, falling back to the alert directory if needed.
If both artifact locations fail, a minimal flag carries `summary` and
`artifact_error` without `file`; consumers must still display that failure.
Successful weekly history is appended only after
the final Markdown exists; invocation envelopes do not append history on their own.

For a 30-minute DMA schedule, place the daily KW check at 08:15 and retain runtime
freshness checks: changing the cron minute alone is not a concurrency guarantee.
Deployment/scheduler changes must be applied separately by the deployment owner.
No database migration, historical replay or deployment action is part of this slice.

Targeted local tests (no live database or network required):

```bash
python -m pytest scripts/test_health_wrapper.py scripts/test_weekly_status.py scripts/test_kw_scripts.py scripts/test_health_status.py scripts/test_report_integration.py
```

Set `KW_TEST_DMA_ROOT` to a separate DMA checkout to also exercise the real producer
CLI through the KW wrapper. Otherwise that optional cross-repository test is skipped.

```bash
pytest                                              # run all tests
python -m knowledge_weaver consolidate              # manual consolidation
python scripts/clean_and_rescore.py --db-path <db>   # cleanup noise entities
python scripts/re_embed.py                           # rebuild vector embeddings
```

## License

Private — internal use only.
