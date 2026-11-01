"""T17: operational CLI (install/doctor/status/export/import/cleanup/workers)."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from knowledge_weaver.cli import main

from ..state.conftest import create_user_event, evidence_ref, record


def _run(capsys, argv):
    code = main(argv)
    captured = capsys.readouterr()
    lines = [line for line in captured.out.splitlines() if line.strip()]
    return code, json.loads("\n".join(lines)) if lines else {}


def _init(db, tmp_path, capsys):
    code, payload = _run(
        capsys,
        [
            "init", "--db", str(db), "--workspace", "owner:demo", "--host", "hermes",
            "--instance", "instance:hermes-test", "--client", "client:test",
            "--project", "project:songta", "--actor", "jun-test",
        ],
    )
    assert code == 0 and payload["ok"]
    return payload


def test_versions_doctor_and_status(tmp_path, capsys):
    db = tmp_path / "kw" / "knowledge.db"
    _init(db, tmp_path, capsys)
    code, payload = _run(capsys, ["versions"])
    assert code == 0 and payload["contract_version"] == "1.0"
    code, payload = _run(capsys, ["doctor", "--db", str(db)])
    assert code == 0 and payload["ok"] is True
    code, payload = _run(capsys, ["status", "--db", str(db)])
    assert code == 0 and payload["status"]["health"]["status"] == "ready"


def test_doctor_fails_without_database(tmp_path, capsys):
    code, payload = _run(capsys, ["doctor", "--db", str(tmp_path / "missing.db")])
    assert code == 1 and payload["ok"] is False


def test_export_import_via_cli(tmp_path, capsys):
    db = tmp_path / "kw" / "knowledge.db"
    from knowledge_weaver.store import Store

    _init(db, tmp_path, capsys)
    store = Store(db)
    record(store, display_text="端口4200", value=4200)
    snapshot = tmp_path / "snapshot"
    code, payload = _run(capsys, ["export", "--db", str(db), "--out", str(snapshot)])
    assert code == 0 and payload["ok"]
    code, payload = _run(capsys, ["validate-snapshot", "--snapshot", str(snapshot)])
    assert code == 0 and payload["ok"]
    target = tmp_path / "restored"
    code, payload = _run(capsys, ["import", "--snapshot", str(snapshot), "--target", str(target)])
    assert code == 0 and payload["ok"]
    code, payload = _run(capsys, ["status", "--db", payload["target_db"]])
    assert code == 0 and payload["status"]["health"]["status"] == "ready"


def test_cleanup_dry_run_then_sweep(tmp_path, capsys):
    from knowledge_weaver.lifecycle import is_tombstoned, register_lifecycle
    from knowledge_weaver.store import Store

    db = tmp_path / "kw" / "knowledge.db"
    _init(db, tmp_path, capsys)
    store = Store(db)
    register_lifecycle(
        store,
        subject_kind="event_part",
        subject_id="evt:x#1@0",
        class_="temporary_detail",
        business_anchor="2020-01-01T00:00:00Z",
    )
    code, payload = _run(capsys, ["cleanup", "--db", str(db), "--dry-run"])
    assert code == 0 and payload["due_now"] >= 1
    assert not is_tombstoned(store, "event_part", "evt:x#1@0")
    code, payload = _run(capsys, ["cleanup", "--db", str(db)])
    assert code == 0 and payload["expired"] >= 1
    assert is_tombstoned(store, "event_part", "evt:x#1@0")


def test_extract_worker_with_external_command(tmp_path, capsys):
    from knowledge_weaver.store import Store, current_items, receive_batch

    db = tmp_path / "kw" / "knowledge.db"
    _init(db, tmp_path, capsys)
    store = Store(db)
    event = create_user_event(store, "端口改为4200")
    proposal_script = tmp_path / "extractor.py"
    proposal_script.write_text(
        """
import json, sys
payload = json.load(sys.stdin)
part = payload["parts"][0]
proposal = {
    "schema_version": "1.0",
    "claims": [{
        "subject_hint": "松塔项目服务器",
        "project_hint": "project:songta",
        "qualifiers": {},
        "predicate": "server.port",
        "facet": "desired",
        "value": 4200,
        "display_text": "端口改为4200",
        "assertion_kind": "explicit_statement",
        "evidence": [{
            "event_id": part["event_id"],
            "event_revision": part["event_revision"],
            "part_index": part["part_index"],
            "support_quote": part["text"],
        }],
        "target_claim_hint": None,
        "effective_time_quote": None,
    }],
    "excluded_event_ids": [],
    "warnings": [],
}
print(json.dumps(proposal, ensure_ascii=False))
""",
        encoding="utf-8",
    )
    code, payload = _run(
        capsys,
        [
            "extract-worker", "--db", str(db), "--once",
            "--command", sys.executable, str(proposal_script),
        ],
    )
    assert code == 0 and payload["processed"]["succeeded"] == 1
    items = current_items(store, workspace_id="owner:demo")
    assert items and items[0]["value"] == 4200


def test_embed_worker_cli(tmp_path, capsys):
    from knowledge_weaver.embeddings.index import configure_model
    from knowledge_weaver.store import Store

    db = tmp_path / "kw" / "knowledge.db"
    _init(db, tmp_path, capsys)
    store = Store(db)
    configure_model("kw:hash-embedding", model_revision="1", dimension=16)
    try:
        claim = record(store, display_text="端口4200", value=4200)
    finally:
        configure_model(None)
    code, payload = _run(capsys, ["embed-worker", "--db", str(db), "--dimension", "16", "--once"])
    assert code == 0 and payload["processed"]["succeeded"] == 1
    with store.read() as conn:
        row = conn.execute("SELECT COUNT(*) AS c FROM claim_vectors").fetchone()
    assert row["c"] == 1


def test_cli_error_is_json_and_nonzero(tmp_path, capsys):
    code, payload = _run(capsys, ["export", "--db", str(tmp_path / "missing.db"), "--out", str(tmp_path / "x")])
    assert code == 2 and payload["ok"] is False


def test_task_and_entity_cli(tmp_path, capsys):
    from knowledge_weaver.store import Store, current_items, list_aliases

    db = tmp_path / "kw" / "knowledge.db"
    _init(db, tmp_path, capsys)
    store = Store(db)
    record(store, display_text="设备A端口4200", value=4200, subject_id="subject:device:a")
    record(store, display_text="设备B备份每周", value="weekly", predicate="backup_interval", subject_id="subject:device:b")

    code, payload = _run(capsys, ["tasks", "--db", str(db), "--workspace", "owner:demo"])
    assert code == 0 and payload["ok"] is True

    code, payload = _run(
        capsys,
        [
            "merge-entity", "--db", str(db), "--workspace", "owner:demo",
            "--entity-project", "project:songta", "--source", "subject:device:b",
            "--canonical", "subject:device:a", "--reason", "same device",
        ],
    )
    assert code == 0 and payload["alias"]["state"] == "active"
    items = current_items(store, workspace_id="owner:demo")
    assert {item["subject_id"] for item in items} == {"subject:device:a"}
    code, payload = _run(
        capsys,
        [
            "undo-merge", "--db", str(db), "--workspace", "owner:demo",
            "--entity-project", "project:songta", "--source", "subject:device:b",
            "--canonical", "subject:device:a",
        ],
    )
    assert code == 0 and payload["undo"]["state"] == "removed"
    assert list_aliases(store, workspace_id="owner:demo") == []
