#!/usr/bin/env python3
"""W3.2: backfill ``### 结构化事实`` blocks into historical daily memory files.

Historical days predate the W2 summarizer change, so they have a ``### 摘要``
but no structured facts. This script re-extracts atomic facts FROM the preserved
summary (the only thing left once the raw conversation is compacted) via the
LLM, and inserts a ``### 结构化事实`` JSON block into each time-block that lacks
one. The rebuild (rebuild_from_facts.py) can then ingest ALL days cleanly.

Safe: backs up each file before writing; idempotent (blocks that already have a
facts block are skipped); additive (never alters existing content); empty /
heartbeat summaries are skipped.

The LLM call is isolated in ``extract_facts_via_llm`` so the file surgery is
deterministically testable with a mock.

Config (env): KW_FACTS_API_URL, KW_FACTS_API_KEY, KW_FACTS_MODEL
  optional: KW_FACTS_LEXICON (JSON string from `registry --lexicon`)

Usage:
    KW_FACTS_API_URL=... KW_FACTS_API_KEY=... KW_FACTS_MODEL=deepseek-v4 \
      python scripts/backfill_facts.py <memory_dir_or_file> [--dry-run]
"""
from __future__ import annotations

import json
import os
import re
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

VALID_TYPES = {
    "project", "preference", "decision", "fact", "risk", "task", "tech", "idea",
}

_FACTS_HEADING = "### 结构化事实"
_SUMMARY_HEADING = "### 摘要"

_SYS_PROMPT = """你是知识抽取助手。下面给你一段已经过压缩的“会话摘要”，请把其中【每一个】\
独立的知识点穷尽式地抽成结构化原子事实，仅输出一个 JSON 数组，不要任何额外文字、解释或代码围栏。

每条事实：{"type": "...", "name": "...", "summary": "...", "aliases": [...]}
- type 只能取这 8 种之一：project / decision / preference / fact / risk / task / tech / idea
  （project=项目，decision=决策/结论，preference=用户偏好，fact=事实/状态，risk=风险，task=待办或已完成的事项，tech=技术/工具/方案，idea=创意）
- name 用规范名；同一事物始终用同一个名字。若下方提供【已知项目规范名】，必须优先采用其规范名，其它写法放入该条 aliases。
- 【密度优先，不要压缩】：摘要里已经是被浓缩过的要点，请尽量保留细粒度——
  * 每一个不同的决策、事实、风险、任务、技术点都【单独成条】，不要把多个并列要点合并成一条；
  * 同一个项目下若有多个不同的决策/方案/选型（如硬件选型、架构方向、集成方式），必须【分别成条】，不要概括成一条；
  * 宁可多列，也不要漏掉摘要中已写明的任何独立知识点。
- type 要贴切；summary 一句话讲清该条（可含关键细节，不必强行压到很短）；aliases 可省略或为空数组。
- 仅当摘要确实无任何实质知识内容（纯心跳/运维/无意义）时，才输出 []。"""


# --------------------------------------------------------------------------- #
# Markdown surgery (deterministic, testable)                                   #
# --------------------------------------------------------------------------- #
def _extract_summary(block: str) -> str:
    """Return the ### 摘要 subsection text of an H2 block (empty if none)."""
    idx = block.find(_SUMMARY_HEADING)
    if idx == -1:
        return ""
    rest = block[idx + len(_SUMMARY_HEADING):]
    m = re.search(r"\n###?\s|\Z", rest)  # next H2/H3 or end
    return rest[: m.start()].strip() if m else rest.strip()


def _is_empty_summary(summary: str) -> bool:
    s = summary.strip()
    return len(s) < 8 or "无实质内容" in s or "仅系统" in s or "仅心跳" in s


def _format_facts_block(facts: list[dict]) -> str:
    payload = json.dumps(facts, ensure_ascii=False, indent=2)
    return f"\n{_FACTS_HEADING}\n\n```json\n{payload}\n```\n"


def backfill_text(text: str, extract_fn, *, dry_run: bool = False) -> tuple[str, int]:
    """Insert a facts block into each H2 time-block that has a 摘要 but no facts
    block. ``extract_fn(summary_text) -> list[dict]``. Returns (new_text, n).

    In dry_run, counts blocks that WOULD be backfilled without calling the LLM."""
    parts = re.split(r"(?m)^(?=## )", text)
    out, n = [], 0
    for blk in parts:
        if not blk.startswith("## ") or _FACTS_HEADING in blk:
            out.append(blk)
            continue
        summary = _extract_summary(blk)
        if not summary or _is_empty_summary(summary):
            out.append(blk)
            continue
        if dry_run:
            out.append(blk)
            n += 1  # eligible: would be backfilled
            continue
        facts = _validate(extract_fn(summary))
        if not facts:
            out.append(blk)
            continue
        blk = blk.rstrip("\n") + "\n" + _format_facts_block(facts)
        out.append(blk)
        n += 1
    return "".join(out), n


def _validate(facts) -> list[dict]:
    if not isinstance(facts, list):
        return []
    clean = []
    seen = set()
    for f in facts:
        if not isinstance(f, dict):
            continue
        t, name = f.get("type"), (f.get("name") or "").strip()
        if t not in VALID_TYPES or not name or (t, name) in seen:
            continue
        seen.add((t, name))
        item = {"type": t, "name": name, "summary": (f.get("summary") or "").strip()}
        al = f.get("aliases")
        if isinstance(al, list):
            al = [a.strip() for a in al if isinstance(a, str) and a.strip()]
            if al:
                item["aliases"] = al
        clean.append(item)
    return clean


def backfill_file(path: str, extract_fn, *, dry_run: bool = False) -> int:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    new_text, n = backfill_text(text, extract_fn, dry_run=dry_run)
    if n and not dry_run:
        shutil.copy2(path, path + ".bak")
        with open(path, "w", encoding="utf-8") as f:
            f.write(new_text)
    return n


# --------------------------------------------------------------------------- #
# LLM call (isolated; mocked in tests)                                          #
# --------------------------------------------------------------------------- #
def extract_facts_via_llm(summary: str, *, api_url: str, api_key: str, model: str,
                          lexicon: str | None = None, timeout: float = 60.0) -> list[dict]:
    import httpx

    sys_prompt = _SYS_PROMPT
    if lexicon:
        sys_prompt += f"\n\n【已知项目规范名（name 优先采用，其它写法放 aliases）】：\n{lexicon}"
    url = api_url.rstrip("/")
    if not url.endswith("/chat/completions"):
        url += "/chat/completions"
    resp = httpx.post(
        url,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": model, "temperature": 0.2,
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": f"会话摘要：\n\n{summary}"},
            ],
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"].strip()
    content = re.sub(r"^```(?:json)?|```$", "", content, flags=re.MULTILINE).strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return []


def main() -> int:
    import argparse
    import glob

    ap = argparse.ArgumentParser(description="W3.2 backfill ### 结构化事实 into history")
    ap.add_argument("target", help="memory dir or single .md file")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    api_url = os.environ.get("KW_FACTS_API_URL")
    api_key = os.environ.get("KW_FACTS_API_KEY")
    model = os.environ.get("KW_FACTS_MODEL")
    if not args.dry_run and not (api_url and api_key and model):
        print("Set KW_FACTS_API_URL / KW_FACTS_API_KEY / KW_FACTS_MODEL (or use --dry-run).")
        return 1
    lexicon = os.environ.get("KW_FACTS_LEXICON")

    def extract_fn(summary):
        if args.dry_run:
            return []  # dry-run: only report which blocks WOULD be backfilled
        return extract_facts_via_llm(summary, api_url=api_url, api_key=api_key,
                                     model=model, lexicon=lexicon)

    files = ([args.target] if os.path.isfile(args.target)
             else sorted(glob.glob(os.path.join(args.target, "*.md"))))
    total = 0
    for path in files:
        try:
            n = backfill_file(path, extract_fn, dry_run=args.dry_run)
        except Exception as exc:
            print(f"  ! {os.path.basename(path)}: {exc}")
            continue
        if n:
            total += n
            print(f"  {'[dry] ' if args.dry_run else ''}{os.path.basename(path)}: "
                  f"+{n} facts block(s)")
    print(f"\n{'Would backfill' if args.dry_run else 'Backfilled'} {total} block(s) "
          f"across {len(files)} file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
