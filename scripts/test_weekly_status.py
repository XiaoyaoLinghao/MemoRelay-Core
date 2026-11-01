"""Weekly reporting changes, independent of entity ingestion/scoring."""
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("weekly_status_test_target", Path(__file__).with_name("kw_weekly_trend.py"))
weekly = importlib.util.module_from_spec(spec)
spec.loader.exec_module(weekly)


@pytest.fixture
def target(tmp_path, monkeypatch):
    monkeypatch.setattr(weekly, "WEEKLY_DIR", str(tmp_path / "weekly"))
    monkeypatch.setattr(weekly, "HISTORY_PATH", str(tmp_path / "history" / "history.jsonl"))
    monkeypatch.setattr(weekly, "load_historical_reports", lambda **kwargs: [])
    monkeypatch.setattr(weekly, "collect_current_metrics", lambda: {
        "entity_total": 100, "relations_total": 100, "orphan_count": 0,
        "coverage_days": 10, "pending_merges": 0, "type_distribution": {"tech": 100},
        "new_entities_7d": 10, "updated_entities_7d": 10,
        "day_count_distribution": {"2": 100}, "embedding_coverage": 1.0,
        "new_entity_inflow_rate": 2.0, "prev_week_inflow_rate": 2.0,
    })
    return tmp_path


def test_all_day_count_two_is_statistics_only(target):
    report = weekly.generate_weekly_report(dry_run=True)
    assert "现有 day_count 不具备跨日统计语义" in report
    assert "| 2 | 100 |" in report
    assert "聚合引擎可能失效" not in report
    assert "⚠️" not in report
    assert not Path(weekly.WEEKLY_DIR).exists()
    assert not Path(weekly.HISTORY_PATH).exists()


def test_invocation_envelope_does_not_publish_default_weekly(target):
    path = target / "invocation" / "report.json"
    text = weekly.generate_weekly_report(run_id="weekly-id", output_path=str(path))
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["run_id"] == "weekly-id"
    assert report["schema_version"] == "kw-weekly-report-v1"
    assert report["checks_complete"] is True
    assert report["markdown"] == text
    assert not Path(weekly.WEEKLY_DIR).exists()
    assert not Path(weekly.HISTORY_PATH).exists()
    history = report["history_entry"]
    assert history["run_id"] == "weekly-id"
    assert not list(target.rglob(".kw-weekly-*"))


def test_direct_weekly_retains_markdown_output(target):
    text = weekly.generate_weekly_report()
    paths = list(Path(weekly.WEEKLY_DIR).glob("*.md"))
    assert len(paths) == 1
    assert paths[0].read_text(encoding="utf-8") == text


def test_metrics_failure_never_publishes_report(target, monkeypatch):
    def fail():
        raise FileNotFoundError("fixture-db")
    monkeypatch.setattr(weekly, "collect_current_metrics", fail)
    path = target / "invocation.json"
    with pytest.raises(SystemExit):
        weekly.generate_weekly_report(output_path=str(path))
    assert not path.exists()
    assert not Path(weekly.HISTORY_PATH).exists()


def test_output_failure_does_not_append_success_history(target, monkeypatch):
    def fail(*args):
        raise OSError("fixture publication failure")
    monkeypatch.setattr(weekly, "atomic_write", fail)
    with pytest.raises(OSError):
        weekly.generate_weekly_report()
    assert not Path(weekly.HISTORY_PATH).exists()
