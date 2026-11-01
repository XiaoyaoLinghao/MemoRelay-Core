"""Report lifecycle regressions; subprocess fixtures never access live services."""
import importlib.util
import json
from pathlib import Path
import subprocess
from datetime import datetime, timedelta, timezone

import pytest


spec = importlib.util.spec_from_file_location("kw_health_wrapper", Path(__file__).parents[1] / "deploy" / "kw_health_wrapper.py")
wrapper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wrapper)


@pytest.fixture
def setup(tmp_path, monkeypatch):
    for attr, path in {"SCRIPT_DIR": tmp_path / "scripts", "REPORTS_DIR": tmp_path / "daily",
                       "WEEKLY_DIR": tmp_path / "weekly", "ALERT_FLAG": tmp_path / "daily.flag",
                       "WEEKLY_ALERT": tmp_path / "weekly.flag", "HISTORY_PATH": tmp_path / "history.jsonl"}.items():
        monkeypatch.setattr(wrapper, attr, str(path))
    Path(wrapper.SCRIPT_DIR).mkdir()
    return tmp_path


def child(mode="daily", alteration=""):
    """Create an actual subprocess that follows the invocation envelope protocol."""
    code = '''import argparse, json
from datetime import datetime, timezone, timedelta
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument('--run-id'); p.add_argument('--output'); a=p.parse_args()
now=datetime.now(timezone.utc)
year,week,_=now.date().isocalendar()
r={'schema_version':'kw-health-report-v1','run_id':a.run_id,'generated_at':now.isoformat(),
   'report_date':now.date().isoformat(),'overall_status':'ok','checks_complete':True,
   'checks':{'dma_health':{'status':'ok','message':'idle'}}}
for name in ['entity_total','new_entity_inflow','embedding_coverage','orphan_ratio',
             'cross_day_aggregation','type_distribution','name_conflicts','relation_integrity',
             'pending_merges','coverage_days']:
 r['checks'][name]={'status':'ok'}
'''
    if mode == "weekly":
        code += "r.update(schema_version='kw-weekly-report-v1',week=f'{year}-W{week:02d}',markdown='# New weekly report\\nFresh evidence')\n"
        code += "r['history_entry']={'run_id':a.run_id,'week':r['week']}\n"
    code += alteration + "\nPath(a.output).write_text(json.dumps(r),encoding='utf-8')\n"
    name = "kw_health_check.py" if mode == "daily" else "kw_weekly_trend.py"
    (Path(wrapper.SCRIPT_DIR) / name).write_text(code, encoding="utf-8")


def test_healthy_current_report_clears_old_flag(setup):
    child()
    Path(wrapper.ALERT_FLAG).write_text("old", encoding="utf-8")
    assert wrapper.run_daily() == 0
    assert not Path(wrapper.ALERT_FLAG).exists()
    files = list(Path(wrapper.REPORTS_DIR).glob("*.json"))
    assert len(files) == 1
    assert json.loads(files[0].read_text())["checks_complete"] is True
    assert not list(Path(wrapper.REPORTS_DIR).glob(".kw-run-*"))


def test_alert_current_report_replaces_flag(setup):
    child(alteration="r['overall_status']='alert'; r['checks']['dma_health']['status']='alert'")
    assert wrapper.run_daily() == 0
    flag = json.loads(Path(wrapper.ALERT_FLAG).read_text(encoding="utf-8"))
    assert flag["status"] == "alert"
    assert flag["alerts"][0]["name"] == "dma_health"


@pytest.mark.parametrize("alteration", [
    "raise SystemExit(2)",
    "raise SystemExit(0)",
    "r['run_id']='old-run'",
    "r['generated_at']=(now-timedelta(days=1)).isoformat()",
    "r['generated_at']=(now+timedelta(hours=1)).isoformat()",
    "r['generated_at']=now.replace(tzinfo=None).isoformat()",
    "r['report_date']='2000-01-01'",
    "r['checks_complete']='true'",
    "r.pop('overall_status')",
    "r['checks']={}",
    "r['checks'].pop('relation_integrity')",
    "r['checks']['dma_health']['error']='broken'",
    "r['checks']['dma_health']['status']='alert'",
    "r['overall_status']='info'; r['checks']['dma_health']['status']='warning'",
    "r['overall_status']='warning'",
    "r=[]",
    "Path(a.output).write_text('{broken'); raise SystemExit(0)",
    "Path(a.output).write_text('['*3000+']'*3000); raise SystemExit(0)",
])
def test_invalid_invocation_never_reuses_old_healthy_report(setup, alteration):
    child(alteration=alteration)
    Path(wrapper.REPORTS_DIR).mkdir()
    old = Path(wrapper.REPORTS_DIR) / "2000-01-01.json"
    old.write_text('{"overall_status":"ok"}', encoding="utf-8")
    assert wrapper.run_daily() == 1
    flag = json.loads(Path(wrapper.ALERT_FLAG).read_text(encoding="utf-8"))
    assert flag["alerts"][0]["name"] == "health_check_failure"
    assert old.read_text() == '{"overall_status":"ok"}'
    assert not list(Path(wrapper.REPORTS_DIR).glob(".kw-run-*"))


def test_running_incomplete_report_does_not_clear_pending_alert(setup):
    child(alteration="r['checks_complete']=False; r['checks']['dma_health']['status']='info'")
    Path(wrapper.ALERT_FLAG).write_text("previous alert", encoding="utf-8")
    assert wrapper.run_daily() == 0
    assert Path(wrapper.ALERT_FLAG).read_text() == "previous alert"


def test_ok_but_incomplete_never_clears_pending_alert(setup):
    child(alteration="r['checks_complete']=False")
    Path(wrapper.ALERT_FLAG).write_text("previous alert", encoding="utf-8")
    assert wrapper.run_daily() == 0
    assert Path(wrapper.ALERT_FLAG).read_text() == "previous alert"


def test_rollout_unknown_publishes_warning(setup):
    child(alteration="r['checks_complete']=False; r['overall_status']='warning'; r['checks']['dma_health']['status']='warning'")
    assert wrapper.run_daily() == 0
    assert json.loads(Path(wrapper.ALERT_FLAG).read_text(encoding="utf-8"))["status"] == "warning"


def test_subprocess_timeout_is_checker_failure(setup, monkeypatch):
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired("fixture", 0.01)
    monkeypatch.setattr(wrapper.subprocess, "run", timeout)
    assert wrapper.run_daily() == 1
    assert "subprocess_timeout" in Path(wrapper.ALERT_FLAG).read_text(encoding="utf-8")


def test_concurrent_wrapper_does_not_touch_owner_alert(setup, monkeypatch):
    Path(wrapper.ALERT_FLAG).write_text("owner alert", encoding="utf-8")
    def forbidden(*args, **kwargs):
        pytest.fail("competing wrapper started child")
    monkeypatch.setattr(wrapper.subprocess, "run", forbidden)
    with wrapper.report_lock(wrapper.ALERT_FLAG) as acquired:
        assert acquired
        assert wrapper.run_daily() == 0
    assert Path(wrapper.ALERT_FLAG).read_text() == "owner alert"


def test_weekly_uses_current_invocation_instead_of_2026_glob(setup):
    child("weekly")
    Path(wrapper.WEEKLY_DIR).mkdir()
    old = Path(wrapper.WEEKLY_DIR) / "2026-W01.md"
    old.write_text("OLD", encoding="utf-8")
    assert wrapper.run_weekly() == 0
    flag = json.loads(Path(wrapper.WEEKLY_ALERT).read_text(encoding="utf-8"))
    assert flag["summary"].startswith("# New weekly report")
    assert Path(flag["file"]).read_text(encoding="utf-8").endswith("Fresh evidence")


def test_weekly_failure_does_not_send_previous_report(setup):
    child("weekly", "raise SystemExit(1)")
    Path(wrapper.WEEKLY_DIR).mkdir()
    (Path(wrapper.WEEKLY_DIR) / "2026-W01.md").write_text("OLD", encoding="utf-8")
    assert wrapper.run_weekly() == 1
    flag = json.loads(Path(wrapper.WEEKLY_ALERT).read_text(encoding="utf-8"))
    assert Path(flag["file"]).read_text(encoding="utf-8").startswith("# KW 周报检查失败")
    assert "OLD" not in Path(flag["file"]).read_text(encoding="utf-8")
    assert flag["alerts"][0]["name"] == "health_check_failure"
    assert "subprocess_failed" in flag["summary"]


def test_weekly_publication_failure_does_not_append_success_history(setup, monkeypatch):
    child("weekly")
    original = wrapper.atomic_write
    def fail_report(path, text):
        if Path(path).suffix == ".md" and not Path(path).name.startswith("health-check-failure-"):
            raise OSError("fixture-publication-failure")
        return original(path, text)
    monkeypatch.setattr(wrapper, "atomic_write", fail_report)
    assert wrapper.run_weekly() == 1
    assert not Path(wrapper.HISTORY_PATH).exists()


def test_weekly_directory_unavailable_still_publishes_failure(setup):
    Path(wrapper.WEEKLY_DIR).write_text("directory is unavailable", encoding="utf-8")
    assert wrapper.run_weekly() == 1
    flag = json.loads(Path(wrapper.WEEKLY_ALERT).read_text(encoding="utf-8"))
    assert flag["alerts"][0]["name"] == "health_check_failure"
    assert Path(flag["file"]).parent == Path(wrapper.WEEKLY_ALERT).parent
    assert "健康检查自身失败" in Path(flag["file"]).read_text(encoding="utf-8")


def test_weekly_failure_artifact_unavailable_still_attempts_flag(setup, monkeypatch):
    child("weekly", "raise SystemExit(1)")
    original = wrapper.atomic_write
    def fail_markdown(path, text):
        if Path(path).suffix == ".md":
            raise OSError("fixture-artifact-unavailable")
        return original(path, text)
    monkeypatch.setattr(wrapper, "atomic_write", fail_markdown)
    assert wrapper.run_weekly() == 1
    flag = json.loads(Path(wrapper.WEEKLY_ALERT).read_text(encoding="utf-8"))
    assert flag["artifact_error"] == "failure_report_unavailable"
    assert "file" not in flag
