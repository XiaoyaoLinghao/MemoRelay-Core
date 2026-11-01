"""T14: logical export, hardened import, restore reconciliation and retention."""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from knowledge_weaver import contracts
from knowledge_weaver.lifecycle import classify_claim_revisions, is_tombstoned, sweep
from knowledge_weaver.portability import (
    SnapshotError,
    export_snapshot,
    import_snapshot,
    validate_snapshot,
)
from knowledge_weaver.portability.export import TABLES
from knowledge_weaver.retrieval.keyword import keyword_ranking
from knowledge_weaver.store import Store, current_items, receive_batch
from knowledge_weaver.store.claims import record_claim_revision

from ..state.conftest import create_user_event, evidence_ref, record


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _regenerate_manifest(root: Path) -> None:
    """Re-sign a (possibly tampered) snapshot for negative-path tests."""
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    counts = {"backend": 1}
    for table, filename in TABLES.items():
        lines = [line for line in (root / "records" / filename).read_text(encoding="utf-8").splitlines() if line.strip()]
        counts[table] = len(lines)
    manifest["counts"] = counts
    files = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name in ("manifest.json", "SHA256SUMS"):
            continue
        relative = path.relative_to(root).as_posix()
        files[relative] = {"sha256": _sha256(path), "size": path.stat().st_size}
    manifest["files"] = files
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (root / "SHA256SUMS").open("w", encoding="utf-8") as handle:
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.name != "SHA256SUMS":
                handle.write(f"{_sha256(path)}  {path.relative_to(root).as_posix()}\n")


def _seed(store):
    current = record(
        store,
        display_text="松塔项目端口改为4200",
        value=4200,
        event_time={"value": "2020-02-01T00:00:00Z", "precision": "instant", "timezone": "UTC"},
    )
    old = record(
        store,
        display_text="松塔项目端口为4100",
        value=4100,
        event_time={"value": "2020-01-01T00:00:00Z", "precision": "instant", "timezone": "UTC"},
    )
    record(
        store,
        display_text="端口改为4200（最终）",
        value=4200,
        supersedes=[old["revision_id"]],
        event_time={"value": "2020-02-01T00:00:00Z", "precision": "instant", "timezone": "UTC"},
    )
    withheld = {
        "schema_version": "1.0",
        "event_id": contracts.canonical_event_id("owner:demo", "instance:test", "message:secret"),
        "event_revision": 1,
        "workspace_id": "owner:demo",
        "operation": "create",
        "source": {
            "instance_id": "instance:test",
            "host": "hermes",
            "actor_id": "jun-test",
            "session_id": "session:test",
            "native_event_id": "message:secret",
            "native_schema": "test-1",
            "role": "user",
            "origin_kind": "user_input",
            "locator": "hermes://session:test/message:secret",
        },
        "event_time": {"value": "2020-01-01T00:00:00Z", "precision": "instant", "timezone": "UTC"},
        "body": {"text": "", "sha256": contracts.EMPTY_BODY_SHA256, "part_index": 0, "part_count": 1,
                 "content_revision": "native:1", "format": "text/plain"},
        "redaction": {"policy_version": "redact:1", "status": "withheld", "replacements": 1, "credential_refs": ["credref:api_key:1"]},
        "project_ids": ["project:songta"],
        "parent_revision": None,
        "derived_from": [],
    }
    receive_batch(store, {"schema_version": "1.0", "batch_id": "batch:secret", "events": [withheld]})
    classify_claim_revisions(store, now="2020-03-01T00:00:00Z")
    sweep(store, now="2026-01-01T00:00:00Z")
    return current, old


def test_export_import_roundtrip_preserves_evidence_and_lifecycle(store, tmp_path):
    current, old = _seed(store)
    snapshot = tmp_path / "snapshot"
    exported = export_snapshot(store, snapshot)
    manifest = json.loads((snapshot / "manifest.json").read_text(encoding="utf-8"))
    assert exported["format_version"] == "1.0"
    assert manifest["raw_sqlite_copy"] is False
    assert manifest["vectors"]["included"] is False
    assert not list(snapshot.rglob("*.db"))
    # No secret body was exported, and expired detail was tombstoned.
    assert b"CANARY" not in b"".join(path.read_bytes() for path in snapshot.rglob("*") if path.is_file())
    assert is_tombstoned(store, "claim_revision", old["revision_id"])
    validated = validate_snapshot(snapshot)
    assert validated["manifest"]["counts"]["event_parts"] >= 1

    target = tmp_path / "restored"
    report = import_snapshot(snapshot, target)
    assert report["source_backend_instance_id"] == store.meta()["backend_instance_id"]
    assert report["backend_instance_id"] != report["source_backend_instance_id"]
    assert report["restore_epoch"] != report["previous_restore_epoch"]
    assert report["watermark"] == exported["watermark"]
    assert report["limitations"]
    restored = Store(report["target_db"])
    assert restored.meta()["watermark"] == exported["watermark"]
    assert restored.meta()["restore_epoch"] == report["restore_epoch"]
    items = current_items(restored, workspace_id="owner:demo")
    assert items and items[0]["value"] == 4200
    assert is_tombstoned(restored, "claim_revision", old["revision_id"])
    with restored.read() as conn:
        expired = conn.execute(
            "SELECT status, display_text FROM claim_revisions WHERE revision_id = ?",
            (old["revision_id"],),
        ).fetchone()
        lifecycle = conn.execute(
            "SELECT expires_at FROM lifecycle_records WHERE subject_id = ?", (old["revision_id"],)
        ).fetchone()
    assert expired["status"] == "expired"
    assert expired["display_text"] == ""
    # Restore preserves the original anchor instead of renewing it.
    assert lifecycle["expires_at"].startswith("2021-01-31")
    # Keyword/FTS index was rebuilt.
    with restored.read() as conn:
        revision_ids = {
            row["revision_id"] for row in conn.execute("SELECT revision_id FROM claim_revisions")
        }
    ranked = keyword_ranking(restored, candidate_revision_ids=revision_ids, query="端口")
    assert ranked
    with restored.read() as conn:
        withheld = conn.execute(
            "SELECT body_text, availability FROM event_parts WHERE event_id = ?",
            (contracts.canonical_event_id("owner:demo", "instance:test", "message:secret"),),
        ).fetchone()
    assert withheld is not None
    assert withheld["body_text"] is None


def test_import_refuses_non_empty_target(store, tmp_path):
    _seed(store)
    snapshot = tmp_path / "snapshot"
    export_snapshot(store, snapshot)
    target = tmp_path / "restored"
    target.mkdir()
    (target / "occupied.txt").write_text("x")
    with pytest.raises(SnapshotError):
        import_snapshot(snapshot, target)


def test_checksum_tampering_is_rejected(store, tmp_path):
    _seed(store)
    snapshot = tmp_path / "snapshot"
    export_snapshot(store, snapshot)
    claims = snapshot / "records" / "claims.jsonl"
    claims.write_text(claims.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(SnapshotError):
        validate_snapshot(snapshot)


def test_path_traversal_is_rejected(store, tmp_path):
    _seed(store)
    snapshot = tmp_path / "snapshot"
    export_snapshot(store, snapshot)
    manifest = json.loads((snapshot / "manifest.json").read_text(encoding="utf-8"))
    manifest["files"]["../evil.json"] = {"sha256": "0" * 64, "size": 1}
    (snapshot / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(SnapshotError):
        validate_snapshot(snapshot)


def test_symlink_in_snapshot_is_rejected(store, tmp_path):
    _seed(store)
    snapshot = tmp_path / "snapshot"
    export_snapshot(store, snapshot)
    target = snapshot / "records" / "evil-link"
    target.symlink_to("/etc/passwd")
    with pytest.raises(SnapshotError):
        validate_snapshot(snapshot)


def test_unknown_format_version_is_rejected(store, tmp_path):
    _seed(store)
    snapshot = tmp_path / "snapshot"
    export_snapshot(store, snapshot)
    manifest = json.loads((snapshot / "manifest.json").read_text(encoding="utf-8"))
    manifest["format_version"] = "9.0"
    (snapshot / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    _regenerate_manifest(snapshot)
    with pytest.raises(SnapshotError):
        validate_snapshot(snapshot)


def test_unknown_columns_are_rejected_on_import(store, tmp_path):
    _seed(store)
    snapshot = tmp_path / "snapshot"
    export_snapshot(store, snapshot)
    clients_path = snapshot / "records" / "clients.jsonl"
    rows = [json.loads(line) for line in clients_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if rows:
        rows[0]["unexpected_secret"] = "x"
        clients_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    else:  # pragma: no cover - seed always registers none, add a synthetic row
        clients_path.write_text(json.dumps({"client_id": "c", "unexpected_secret": "x"}) + "\n", encoding="utf-8")
    _regenerate_manifest(snapshot)
    with pytest.raises(SnapshotError):
        import_snapshot(snapshot, tmp_path / "restored")


def test_bootstrap_import_marks_unprocessed_history(store, tmp_path):
    _seed(store)
    snapshot = tmp_path / "snapshot"
    export_snapshot(store, snapshot)
    report = import_snapshot(snapshot, tmp_path / "restored", bootstrap_import=True)
    assert report["bootstrap_pending"] is True
    restored = Store(report["target_db"])
    with restored.read() as conn:
        rows = conn.execute(
            "SELECT state, reason FROM lifecycle_records WHERE subject_kind = 'event_part'"
        ).fetchall()
    assert rows and all(row["state"] == "bootstrap_pending" for row in rows)
    # Retention must not delete un-extracted imported history.
    assert sweep(restored, now="2099-01-01T00:00:00Z")["event_parts"] == 0


def test_task_views_and_entity_aliases_roundtrip(store, tmp_path):
    from knowledge_weaver.store import list_task_views, merge_entities

    from ..state.conftest import record as record_claim

    event = create_user_event(store, "任务状态 reported_done 说明")
    claim = record_claim_revision(
        store,
        workspace_id="owner:demo",
        project_id="project:songta",
        subject_id="subject:task:deploy",
        predicate="task.status",
        facet="reported",
        qualifiers={},
        value="reported_done",
        display_text="任务状态 reported_done",
        authority="reported",
        assertion_kind="completion_report",
        event_time=event["event_time"],
        evidence=[evidence_ref(event)],
    )
    first = record_claim(store, display_text="设备A端口4200", value=4200, subject_id="subject:device:a")
    second = record_claim(store, display_text="设备B备份每周", value="weekly", predicate="backup_interval", subject_id="subject:device:b")
    merge_entities(
        store,
        workspace_id="owner:demo",
        project_id="project:songta",
        source_subject_id="subject:device:b",
        canonical_subject_id="subject:device:a",
    )
    snapshot = tmp_path / "snapshot"
    export_snapshot(store, snapshot)
    report = import_snapshot(snapshot, tmp_path / "restored")
    restored = Store(report["target_db"])
    views = list_task_views(restored, workspace_id="owner:demo")
    assert views and views[0]["status"] == "reported_done"
    from knowledge_weaver.store import list_aliases

    aliases = list_aliases(restored, workspace_id="owner:demo")
    assert len(aliases) == 1
    assert aliases[0]["source_subject_id"] == "subject:device:b"


# ---------------------------------------------------------------------------
# F13: restore applies knowable expiry before the restored DB is readable
# ---------------------------------------------------------------------------


def test_restore_applies_known_expiry_before_source_read(store, tmp_path):
    from knowledge_weaver.lifecycle import register_lifecycle
    from knowledge_weaver.store import fetch_source

    event = create_user_event(store, "应到期的旧调试原文")
    register_lifecycle(
        store,
        subject_kind="event_part",
        subject_id=f"{event['event_id']}#1@0",
        class_="temporary_detail",
        business_anchor="2020-01-01T00:00:00Z",
    )
    snapshot = tmp_path / "snapshot"
    export_snapshot(store, snapshot)
    report = import_snapshot(snapshot, tmp_path / "restored")
    restored = Store(report["target_db"])
    result = fetch_source(restored, event["event_id"], 1, 0)
    assert result["availability"] == "expired"
    assert result["text"] is None
    assert report["restore_expiry"]["event_parts"] >= 1
    with restored.read() as conn:
        due = conn.execute(
            "SELECT COUNT(*) FROM lifecycle_records WHERE state = 'active' "
            "AND expires_at IS NOT NULL AND expires_at <= ?",
            ("2099-01-01T00:00:00Z",),
        ).fetchone()[0]
    assert due == 0, "no knowable-due record may remain active after activation"


def test_restore_keeps_still_valid_conclusion_with_expired_source(store, tmp_path):
    from knowledge_weaver.lifecycle import register_lifecycle
    from knowledge_weaver.store import current_items as restored_items

    event = create_user_event(store, "旧过程证据对应有效结论")
    record(store, display_text="有效结论", value="active", evidence_event=event)
    register_lifecycle(
        store,
        subject_kind="event_part",
        subject_id=f"{event['event_id']}#1@0",
        class_="completed_task",
        business_anchor="2020-01-01T00:00:00Z",
    )
    snapshot = tmp_path / "snapshot"
    export_snapshot(store, snapshot)
    report = import_snapshot(snapshot, tmp_path / "restored")
    restored = Store(report["target_db"])
    items = restored_items(restored, workspace_id="owner:demo")
    assert items
    assert any(item["text"] == "有效结论" for item in items)
    assert all(item["source_detail_expired"] for item in items)


# ---------------------------------------------------------------------------
# F14: capability/resource registry must survive export -> import completely
# ---------------------------------------------------------------------------


def test_registry_survives_export_import(store, tmp_path):
    from knowledge_weaver.resource_registry import (
        list_capabilities,
        list_resources,
        register_capability,
        register_resource,
    )

    recipe = register_capability(
        store,
        workspace_id="owner:demo",
        actor_id="actor:test",
        name="synthetic-skill",
        source_uri="https://example.invalid/skill",
        version="1",
        content_hash="a" * 64,
        install_locator="/synthetic/skill",
    )
    register_resource(
        store,
        workspace_id="owner:demo",
        project_id="project:songta",
        name="nas",
        endpoint="smb://nas.example.invalid/share",
        account_label="readonly",
        credential_ref="credref:test",
        recipe_id=recipe["capability_id"],
    )
    snapshot = tmp_path / "snapshot"
    exported = export_snapshot(store, snapshot)
    manifest = json.loads((snapshot / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["snapshot_schema_version"] == 3
    assert manifest["counts"]["capabilities"] == 1
    assert manifest["counts"]["resources"] == 1
    report = import_snapshot(snapshot, tmp_path / "restored")
    assert report["counts_by_table"]["capabilities"] == 1
    assert report["counts_by_table"]["resources"] == 1
    assert report["referential_integrity"] == "ok"
    restored = Store(report["target_db"])
    assert len(list_capabilities(restored, workspace_id="owner:demo")) == 1
    assert len(list_resources(restored, workspace_id="owner:demo")) == 1


def test_import_rejects_dangling_registry_reference(store, tmp_path):
    from knowledge_weaver.resource_registry import register_capability, register_resource

    recipe = register_capability(
        store,
        workspace_id="owner:demo",
        actor_id="actor:test",
        name="synthetic-skill",
        source_uri="https://example.invalid/skill",
        version="1",
        content_hash="a" * 64,
        install_locator="/synthetic/skill",
    )
    register_resource(
        store,
        workspace_id="owner:demo",
        project_id="project:songta",
        name="nas",
        endpoint="smb://nas.example.invalid/share",
        account_label="readonly",
        credential_ref="credref:test",
        recipe_id=recipe["capability_id"],
    )
    snapshot = tmp_path / "snapshot"
    export_snapshot(store, snapshot)
    # Drop the capability rows but keep the dependent resource: a counts-only
    # check would happily restore a dangling reference.
    (snapshot / "records" / "capabilities.jsonl").write_text("", encoding="utf-8")
    _regenerate_manifest(snapshot)
    with pytest.raises(SnapshotError) as excinfo:
        import_snapshot(snapshot, tmp_path / "restored")
    assert excinfo.value.code == "INVALID_REQUEST"
    assert not (tmp_path / "restored" / "knowledge.db").exists()


# ---------------------------------------------------------------------------
# F17: import target must not be a symlink or write through one
# ---------------------------------------------------------------------------


def test_import_rejects_symlink_target(store, tmp_path):
    snapshot = tmp_path / "snapshot"
    export_snapshot(store, snapshot)
    outside = tmp_path / "outside"
    outside.mkdir()
    link = tmp_path / "target-link"
    link.symlink_to(outside, target_is_directory=True)
    with pytest.raises(SnapshotError) as excinfo:
        import_snapshot(snapshot, link)
    assert excinfo.value.code == "INVALID_REQUEST"
    assert not (outside / "knowledge.db").exists()
    assert list(outside.iterdir()) == []


def test_import_rejects_symlinked_ancestor(store, tmp_path):
    snapshot = tmp_path / "snapshot"
    export_snapshot(store, snapshot)
    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    link = tmp_path / "parent-link"
    link.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(SnapshotError) as excinfo:
        import_snapshot(snapshot, link / "child")
    assert excinfo.value.code == "INVALID_REQUEST"
    assert list(real_parent.iterdir()) == []


def test_failed_import_only_cleans_up_its_own_new_directory(store, tmp_path):
    snapshot = tmp_path / "snapshot"
    export_snapshot(store, snapshot)
    # Corrupt the snapshot after export so the import fails mid-way.
    (snapshot / "records" / "claims.jsonl").write_text("{not json}\n", encoding="utf-8")
    _regenerate_manifest(snapshot)
    existing_empty = tmp_path / "existing-empty"
    existing_empty.mkdir()
    with pytest.raises(SnapshotError):
        import_snapshot(snapshot, existing_empty)
    assert existing_empty.exists(), "a pre-existing user directory must not be deleted"
    created = tmp_path / "created-by-import"
    with pytest.raises(SnapshotError):
        import_snapshot(snapshot, created)
    assert not created.exists(), "the directory created by this import must be cleaned"
