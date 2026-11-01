#!/usr/bin/env python3
"""
Knowledge Weaver Weekly Trend Report (每周 cron).

Reads the SQLite knowledge database and historical daily JSON health reports
to produce a Markdown weekly summary and append to the trend history file.

Usage:
    kw_weekly_trend.py              # full run, write Markdown + history
    kw_weekly_trend.py --dry-run    # print Markdown to stdout only
"""

import argparse
import json
import math
import os
import sqlite3
import sys
import tempfile
import uuid
from collections import Counter
from datetime import date, datetime, timedelta

# ── Absolute paths ───────────────────────────────────────────────────────────

DB_PATH = os.environ.get("KNOWLEDGE_WEAVER_DB_PATH", "/home/openclaw/.openclaw/knowledge/knowledge.db")
REPORTS_DIR = os.environ.get("KW_HEALTH_REPORTS_DIR", "/home/openclaw/.openclaw/knowledge/health_reports")
WEEKLY_DIR = os.environ.get("KW_WEEKLY_REPORTS_DIR", "/home/openclaw/.openclaw/knowledge/weekly_reports")
HISTORY_PATH = os.environ.get("KW_HEALTH_HISTORY_PATH", "/home/openclaw/.openclaw/knowledge/health_history.jsonl")
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))

KNOWN_TYPES = ("tech", "decision", "risk", "preference", "task", "idea", "fact", "project")


def ensure_dir(path: str) -> None:
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def get_db_connection(readonly: bool = True) -> sqlite3.Connection:
    """Open a read-only connection to the knowledge DB."""
    if not os.path.isfile(DB_PATH):
        raise FileNotFoundError(f"Database not found: {DB_PATH}")
    if readonly:
        uri = f"file:{DB_PATH}?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
    else:
        conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# ── Data collectors ──────────────────────────────────────────────────────────

def collect_current_metrics() -> dict:
    """Gather key metrics from the live database."""
    conn = get_db_connection(readonly=True)
    cur = conn.cursor()

    # Entity total
    cur.execute("SELECT COUNT(*) AS cnt FROM entities")
    entity_total = cur.fetchone()["cnt"]

    # Relations total
    cur.execute("SELECT COUNT(*) AS cnt FROM relations")
    relations_total = cur.fetchone()["cnt"]

    # Type distribution
    type_counts: dict[str, int] = {}
    for t in KNOWN_TYPES:
        cur.execute("SELECT COUNT(*) AS cnt FROM entities WHERE type = ?", (t,))
        type_counts[t] = cur.fetchone()["cnt"]
    cur.execute(
        f"SELECT COUNT(*) AS cnt FROM entities WHERE type NOT IN ({','.join('?' * len(KNOWN_TYPES))})",
        KNOWN_TYPES,
    )
    type_counts["other"] = cur.fetchone()["cnt"]

    # New entities in the last 7 days (by first_seen)
    seven_days_ago = (date.today() - timedelta(days=7)).isoformat()
    cur.execute(
        "SELECT COUNT(*) AS cnt FROM entities WHERE first_seen >= ?",
        (seven_days_ago,),
    )
    new_entities_7d = cur.fetchone()["cnt"]

    # Updated entities in the last 7 days (updated_at != created_at)
    cur.execute(
        "SELECT COUNT(*) AS cnt FROM entities WHERE updated_at >= ? AND updated_at != created_at",
        (seven_days_ago,),
    )
    updated_entities_7d = cur.fetchone()["cnt"]

    # Knowledge coverage days
    cur.execute("SELECT COUNT(DISTINCT date) AS cnt FROM daily_manifest")
    coverage_days = cur.fetchone()["cnt"]

    # Orphan entities
    cur.execute("""
        SELECT COUNT(*) AS cnt FROM entities e
        WHERE NOT EXISTS (SELECT 1 FROM relations r WHERE r.from_entity = e.id)
          AND NOT EXISTS (SELECT 1 FROM relations r WHERE r.to_entity   = e.id)
    """)
    orphan_count = cur.fetchone()["cnt"]

    # Pending merges
    cur.execute("SELECT COUNT(*) AS cnt FROM merge_review WHERE status = 'pending'")
    pending_merges = cur.fetchone()["cnt"]

    # day_count distribution
    cur.execute("""
        SELECT day_count, COUNT(*) AS cnt FROM entities GROUP BY day_count ORDER BY day_count
    """)
    day_count_dist = {str(r["day_count"]): r["cnt"] for r in cur.fetchall()}

    # Embedding coverage
    cur.execute("SELECT COUNT(*) AS cnt FROM entity_vectors")
    vector_count = cur.fetchone()["cnt"]
    embedding_coverage = round(vector_count / entity_total, 4) if entity_total else 0.0

    # New entity inflow rate (entities/day over last 7 days)
    new_entity_inflow_rate = round(new_entities_7d / 7, 2)

    # Previous week inflow rate (14-7 days ago, for trend comparison)
    fourteen_days_ago = (date.today() - timedelta(days=14)).isoformat()
    cur.execute(
        "SELECT COUNT(*) AS cnt FROM entities WHERE first_seen >= ? AND first_seen < ?",
        (fourteen_days_ago, seven_days_ago),
    )
    prev_week_new = cur.fetchone()["cnt"]
    prev_week_inflow_rate = round(prev_week_new / 7, 2)

    conn.close()

    return {
        "entity_total": entity_total,
        "relations_total": relations_total,
        "type_distribution": type_counts,
        "new_entities_7d": new_entities_7d,
        "updated_entities_7d": updated_entities_7d,
        "coverage_days": coverage_days,
        "orphan_count": orphan_count,
        "pending_merges": pending_merges,
        "day_count_distribution": day_count_dist,
        "embedding_coverage": embedding_coverage,
        "new_entity_inflow_rate": new_entity_inflow_rate,
        "prev_week_inflow_rate": prev_week_inflow_rate,
    }


def load_historical_reports(back_days: int = 14) -> list[dict]:
    """Load daily JSON reports from the last N days."""
    reports: list[dict] = []
    today = date.today()
    for i in range(back_days):
        d = today - timedelta(days=i)
        path = os.path.join(REPORTS_DIR, f"{d.isoformat()}.json")
        if os.path.isfile(path):
            try:
                with open(path, "r") as f:
                    reports.append(json.load(f))
            except (json.JSONDecodeError, OSError):
                pass
    # Sort by date ascending
    reports.sort(key=lambda r: r.get("report_date", ""))
    return reports


# ── Analysis helpers ─────────────────────────────────────────────────────────

def calc_entropy(counts: dict[str, int]) -> float:
    """Shannon entropy of a discrete distribution (in bits)."""
    total = sum(counts.values())
    if total == 0:
        return 0.0
    entropy = 0.0
    for v in counts.values():
        if v > 0:
            p = v / total
            entropy -= p * math.log2(p)
    return round(entropy, 4)


def type_drift(reports: list[dict]) -> dict:
    """Compare type distribution entropy between oldest and newest report."""
    if len(reports) < 2:
        return {"entropy_delta": None, "note": "Insufficient historical data (< 2 reports)"}

    oldest = reports[0]
    newest = reports[-1]

    def extract_types(r: dict) -> dict[str, int]:
        td = r.get("checks", {}).get("type_distribution", {}).get("type_distribution", {})
        if isinstance(td, dict):
            return td
        return {}

    old_types = extract_types(oldest)
    new_types = extract_types(newest)

    old_entropy = calc_entropy(old_types)
    new_entropy = calc_entropy(new_types)

    return {
        "oldest_date": oldest.get("report_date"),
        "newest_date": newest.get("report_date"),
        "old_entropy": old_entropy,
        "new_entropy": new_entropy,
        "entropy_delta": round(new_entropy - old_entropy, 4),
    }


def weekly_delta(reports: list[dict]) -> dict | None:
    """Compare entity total between a week ago and now."""
    if len(reports) < 2:
        return None

    today = date.today().isoformat()
    week_ago = (date.today() - timedelta(days=7)).isoformat()

    newest_report = reports[-1]
    oldest_report = None
    for r in reports:
        if r.get("report_date", "") <= week_ago:
            oldest_report = r

    if oldest_report is None:
        oldest_report = reports[0]

    def extract_total(r: dict) -> int | None:
        return r.get("checks", {}).get("entity_total", {}).get("total_entities")

    new_total = extract_total(newest_report)
    old_total = extract_total(oldest_report)

    if new_total is None or old_total is None:
        return None

    delta = new_total - old_total
    pct = round(delta / old_total * 100, 2) if old_total else 0.0

    return {
        "from_date": oldest_report.get("report_date", "unknown"),
        "to_date": newest_report.get("report_date", today),
        "from_total": old_total,
        "to_total": new_total,
        "delta": delta,
        "delta_pct": pct,
    }


# ── Report generation ────────────────────────────────────────────────────────

def atomic_write(path: str, text: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    ensure_dir(parent)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n",
                                         dir=parent, prefix=".kw-weekly-", delete=False) as f:
            temporary = f.name
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


def generate_weekly_report(dry_run: bool = False, *, run_id: str | None = None,
                           output_path: str | None = None) -> str:
    """Generate Markdown weekly report.  Returns the Markdown string."""
    generated = datetime.now().astimezone()
    run_id = run_id or uuid.uuid4().hex

    # Determine ISO week
    today = generated.date()
    iso_year, iso_week, _ = today.isocalendar()
    week_label = f"{iso_year}-W{iso_week:02d}"

    # Collect live metrics
    try:
        metrics = collect_current_metrics()
    except FileNotFoundError as e:
        print(f"FATAL: {e}", file=sys.stderr)
        sys.exit(1)
    except sqlite3.Error as e:
        print(f"FATAL: Cannot read database: {e}", file=sys.stderr)
        sys.exit(1)

    # Load historical reports
    reports = load_historical_reports(back_days=14)

    # Compute derived metrics
    rel_density = (
        round(metrics["relations_total"] / metrics["entity_total"], 4)
        if metrics["entity_total"]
        else 0.0
    )
    orphan_ratio = (
        round(metrics["orphan_count"] / metrics["entity_total"] * 100, 2)
        if metrics["entity_total"]
        else 0.0
    )
    type_entropy = calc_entropy(metrics["type_distribution"])
    drift = type_drift(reports)
    wow = weekly_delta(reports)

    # ── Build Markdown ──
    md_lines: list[str] = []
    md_lines.append(f"# Knowledge Weaver 周报 {week_label}")
    md_lines.append("")
    md_lines.append(f"**生成时间**: {generated.replace(tzinfo=None).isoformat(timespec='seconds')}")
    md_lines.append(f"**报告周期**: {today.isoformat()}")
    md_lines.append("")

    # Section: Overview
    md_lines.append("## 1. 总览")
    md_lines.append("")
    md_lines.append("| 指标 | 当前值 | 环比变化 |")
    md_lines.append("|------|--------|----------|")
    md_lines.append(
        f"| 实体总数 | {metrics['entity_total']} | "
        f"{wow['delta_pct']:+.1f}% ({wow['from_date']} → {wow['to_date']})"
        if wow else
        f"| 实体总数 | {metrics['entity_total']} | N/A |"
    )
    
    md_lines.append(f"| 关系总数 | {metrics['relations_total']} | — |")
    md_lines.append(f"| 关系密度 | {rel_density} | — |")
    md_lines.append(f"| 孤立实体 | {metrics['orphan_count']} ({orphan_ratio:.1f}%) | — |")
    md_lines.append(f"| 知识覆盖天数 | {metrics['coverage_days']} | — |")
    md_lines.append(f"| 待合并队列 | {metrics['pending_merges']} | — |")
    md_lines.append("")

    # Section: 类型分布
    md_lines.append("## 2. 类型分布")
    md_lines.append("")
    md_lines.append(f"**类型熵值**: {type_entropy} bits")
    if drift.get("entropy_delta") is not None:
        sign = "📈 上升" if drift["entropy_delta"] > 0 else "📉 下降" if drift["entropy_delta"] < 0 else "➡️ 持平"
        md_lines.append(f"**分布漂移**: {sign} ({drift['old_entropy']} → {drift['new_entropy']}, Δ={drift['entropy_delta']:+.4f})")
    else:
        md_lines.append(f"**分布漂移**: {drift.get('note', 'N/A')}")
    md_lines.append("")
    md_lines.append("| 类型 | 数量 | 占比 |")
    md_lines.append("|------|------|------|")
    total_types = sum(metrics["type_distribution"].values())
    for t in sorted(metrics["type_distribution"].keys()):
        cnt = metrics["type_distribution"][t]
        pct = round(cnt / total_types * 100, 1) if total_types else 0.0
        md_lines.append(f"| {t} | {cnt} | {pct:.1f}% |")
    md_lines.append("")

    # Section: 活动摘要
    md_lines.append("## 3. 活动摘要（过去 7 天）")
    md_lines.append("")
    md_lines.append(f"- 新增实体: **{metrics['new_entities_7d']}**")
    md_lines.append(f"- 更新实体: **{metrics['updated_entities_7d']}**")
    md_lines.append("")

    # Section: 聚合健康
    md_lines.append("## 4. 聚合健康")
    md_lines.append("")
    md_lines.append("### day_count 分布")
    md_lines.append("")
    md_lines.append("统计信息：现有 day_count 不具备跨日统计语义；此分布不用于判定聚合故障。")
    md_lines.append("")
    md_lines.append("| day_count | 实体数 |")
    md_lines.append("|-----------|--------|")
    for dc, cnt in sorted(metrics["day_count_distribution"].items(), key=lambda x: int(x[0]) if x[0].isdigit() else 0):
        md_lines.append(f"| {dc} | {cnt} |")
    md_lines.append("")

    # Embedding coverage delta (week-over-week from historical reports)
    embedding_coverage_delta = None
    if len(reports) >= 2:
        prev_coverage = reports[-2].get("checks", {}).get("embedding_coverage", {}).get("coverage_ratio")
        cur_coverage = metrics.get("embedding_coverage")
        if prev_coverage is not None and cur_coverage is not None:
            embedding_coverage_delta = round(cur_coverage - prev_coverage, 4)

    # Section: Embedding 健康
    md_lines.append("## 5. 向量嵌入健康")
    md_lines.append("")
    md_lines.append(f"- 嵌入覆盖率: **{metrics['embedding_coverage']:.1%}**")
    if embedding_coverage_delta is not None:
        md_lines.append(f"- 覆盖率周环比: **{embedding_coverage_delta:+.4f}**")
    else:
        md_lines.append(f"- 覆盖率周环比: N/A（历史数据不足）")
    md_lines.append(f"- 新实体流入速率（近7日）: **{metrics['new_entity_inflow_rate']} entities/day**")
    md_lines.append(f"- 新实体流入速率（前7日）: **{metrics['prev_week_inflow_rate']} entities/day**")
    inflow_delta = round(metrics['new_entity_inflow_rate'] - metrics['prev_week_inflow_rate'], 2)
    if metrics['prev_week_inflow_rate'] > 0:
        inflow_pct = round(inflow_delta / metrics['prev_week_inflow_rate'] * 100, 1)
        md_lines.append(f"- 流入速率变化: **{inflow_delta:+.2f} entities/day ({inflow_pct:+.1f}%)**")
    else:
        md_lines.append(f"- 流入速率变化: **{inflow_delta:+.2f} entities/day**")
    md_lines.append("")

    # Section: 趋势建议
    md_lines.append("## 6. 趋势与建议")
    md_lines.append("")

    alerts: list[str] = []
    if orphan_ratio > 40:
        alerts.append(f"- ⚠️ 孤立实体比例达 {orphan_ratio:.1f}%，建议审查是否需要合并或链接")
    if metrics["pending_merges"] > 20:
        alerts.append(f"- ⚠️ 待合并队列积压 {metrics['pending_merges']} 条，建议加速处理")
    if wow and abs(wow["delta_pct"]) > 10:
        direction = "增长" if wow["delta_pct"] > 0 else "减少"
        alerts.append(f"- 📊 实体数量周环比{direction} {wow['delta_pct']:+.1f}%，需关注数据质量")
    if drift.get("entropy_delta") is not None and abs(drift["entropy_delta"]) > 0.3:
        alerts.append(f"- 📊 类型分布漂移显著 (Δ={drift['entropy_delta']:+.4f})，建议检查知识摄入策略变化")

    # Add embedding/inflow alerts
    if metrics["embedding_coverage"] < 0.70:
        alerts.append(f"- ⚠️ 嵌入覆盖率仅 {metrics['embedding_coverage']:.1%}，低于 70% 警戒线")
    elif metrics["embedding_coverage"] < 0.90:
        alerts.append(f"- ⚠️ 嵌入覆盖率 {metrics['embedding_coverage']:.1%}，低于 90% 目标")
    if embedding_coverage_delta is not None and embedding_coverage_delta < -0.05:
        alerts.append(f"- 📉 嵌入覆盖率周环比下降 {abs(embedding_coverage_delta):.1%}，可能嵌入管线异常")
    if metrics["new_entity_inflow_rate"] < 0.5:
        alerts.append(f"- ⚠️ 新实体流入速率极低 ({metrics['new_entity_inflow_rate']} entities/day)，ingestion 可能停摆")
    if inflow_delta < -2:
        alerts.append(f"- 📉 新实体流入速率周环比大幅下降 ({inflow_delta:+.2f} entities/day)")

    if alerts:
        for a in alerts:
            md_lines.append(a)
    else:
        md_lines.append("✅ 本周各项指标正常，无需特别关注。")
    md_lines.append("")

    md_text = "\n".join(md_lines)

    # ── Persist ──
    if not dry_run:
        report_path = output_path or os.path.join(WEEKLY_DIR, f"{week_label}.md")

        # Append to history file (JSONL)
        history_entry = {
            "week": week_label,
            "generated_at": generated.replace(tzinfo=None).isoformat(timespec="seconds"),
            "run_id": run_id,
            "metrics": metrics,
            "derived": {
                "relation_density": rel_density,
                "orphan_ratio": orphan_ratio,
                "type_entropy": type_entropy,
            },
            "drift": drift,
            "wow_delta": wow,
        }
        if output_path:
            envelope = {"schema_version": "kw-weekly-report-v1", "run_id": run_id,
                        "generated_at": generated.isoformat(timespec="seconds"),
                        "report_date": today.isoformat(), "week": week_label,
                        "checks_complete": True, "markdown": md_text,
                        "history_entry": history_entry}
            atomic_write(report_path, json.dumps(envelope, ensure_ascii=False) + "\n")
        else:
            atomic_write(report_path, md_text)
            # Never record success before the corresponding report exists.
            ensure_dir(os.path.dirname(os.path.abspath(HISTORY_PATH)))
            with open(HISTORY_PATH, "a", encoding="utf-8") as hf:
                hf.write(json.dumps(dict(history_entry, report_path=report_path), ensure_ascii=False) + "\n")
                hf.flush()
                os.fsync(hf.fileno())

        print(f"Weekly report written to {report_path}")
        if not output_path:
            print(f"History appended to {HISTORY_PATH}")

    if dry_run:
        print(md_text)

    return md_text


# ── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Knowledge Weaver Weekly Trend Report")
    parser.add_argument("--run-id", help="Caller invocation ID")
    parser.add_argument("--output", help="Write a versioned JSON envelope to this invocation-owned path")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print Markdown to stdout only; do not write files.",
    )
    args = parser.parse_args()
    generate_weekly_report(dry_run=args.dry_run, run_id=args.run_id, output_path=args.output)


if __name__ == "__main__":
    main()
