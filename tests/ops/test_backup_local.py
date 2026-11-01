"""Operator local-only profile; all data is isolated in pytest temporary paths."""
import fcntl
import hashlib
import json
import subprocess
import sys

import pytest

from knowledge_weaver.portability import import_snapshot
from knowledge_weaver.store import Store
from .test_backup_script import SCRIPT, _load_module, _seed, _generations


def _tree(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob("*") if p.is_file()}


@pytest.mark.parametrize("keep", [4, 2])
def test_local_only_rotation_and_restore(tmp_path, keep):
    module = _load_module()
    db, local = tmp_path / "kw" / "knowledge.db", tmp_path / "backups"
    _seed(db)
    for day in range(1, 7):
        report = module.run_backup(db_path=db, local_dir=local, keep=keep,
                                   now=f"2026-11-{day:02d}T00:00:00Z")
        assert report["verified"] == {"local": True}
        assert report["mode"] == "local_only" and report["nas_dir"] is None
    assert _generations(local) == [f"gen-202611{day:02d}T000000Z" for day in range(7 - keep, 7)]
    last = local / "generations" / _generations(local)[-1]
    snapshot_before = _tree(last)
    again = module.run_backup(db_path=db, local_dir=local, keep=keep, now="2026-11-06T00:00:00Z")
    assert again["existing"] is True
    assert _tree(last) == snapshot_before
    import_snapshot(last / "snapshot", tmp_path / "restored")
    with Store(db).read() as source, Store(tmp_path / "restored" / "knowledge.db").read() as restored:
        for table in ["claims", "claim_revisions", "evidence_links", "lifecycle_records", "tombstones"]:
            original = sorted(tuple(row) for row in source.execute(f"SELECT * FROM {table}"))
            actual = sorted(tuple(row) for row in restored.execute(f"SELECT * FROM {table}"))
            assert actual == original, table
        assert restored.execute("PRAGMA integrity_check").fetchone()[0] == "ok"


@pytest.mark.parametrize("seam", ["export_snapshot", "validate_snapshot", "_publish"])
def test_local_failure_preserves_successful_copies(tmp_path, monkeypatch, seam):
    module = _load_module()
    db, local = tmp_path / "kw" / "knowledge.db", tmp_path / "backups"
    _seed(db)
    for day in (1, 2):
        module.run_backup(db_path=db, local_dir=local, now=f"2026-11-{day:02d}T00:00:00Z")
    before = _tree(local / "generations")
    def fail(*args, **kwargs):
        raise OSError("synthetic export/verify/publish failure")
    monkeypatch.setattr(module, seam, fail)
    with pytest.raises(OSError, match="synthetic"):
        module.run_backup(db_path=db, local_dir=local, keep=1, now="2026-11-03T00:00:00Z")
    assert _tree(local / "generations") == before
    assert not list(local.glob(".staging-*"))


def test_corruption_is_not_counted_as_a_usable_recovery_point(tmp_path):
    module = _load_module()
    db, local = tmp_path / "kw" / "knowledge.db", tmp_path / "backups"
    _seed(db)
    for day in (1, 2):
        module.run_backup(db_path=db, local_dir=local, now=f"2026-11-{day:02d}T00:00:00Z")
    corrupt = local / "generations" / "gen-20261102T000000Z" / "snapshot" / "records" / "claims.jsonl"
    corrupt.write_text("broken")
    before = _tree(local / "generations")
    with pytest.raises(Exception):
        module.run_backup(db_path=db, local_dir=local, keep=1, now="2026-11-03T00:00:00Z")
    after = _tree(local / "generations")
    assert all(after.get(p) == sha for p, sha in before.items())  # no old copy deleted/rewritten


def test_overlap_refused_without_touching_active_staging(tmp_path):
    db, local = tmp_path / "kw" / "knowledge.db", tmp_path / "backups"
    _seed(db)
    local.mkdir()
    stage = local / ".staging-gen-20261101T000000Z"
    stage.mkdir()
    (stage / "active").write_text("owner run")
    before = _tree(local)
    with (local / ".backup.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = subprocess.run([sys.executable, str(SCRIPT), "--db", str(db),
                                 "--local-dir", str(local), "--now", "2026-11-01T00:00:00Z"],
                                capture_output=True, text=True, timeout=20)
        assert result.returncode == 1
        assert "another backup" in json.loads(result.stdout)["error"]
    assert all(_tree(local).get(p) == sha for p, sha in before.items())
    # The same lock file can be acquired after its owning process releases it.
    result = subprocess.run([sys.executable, str(SCRIPT), "--db", str(db), "--local-dir", str(local)],
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout)["verified"] == {"local": True}


def test_local_generation_can_later_be_copied_to_nas_without_reexport(tmp_path, monkeypatch):
    module = _load_module()
    db, local, nas = tmp_path / "kw" / "knowledge.db", tmp_path / "backups", tmp_path / "nas-simulation"
    _seed(db)
    module.run_backup(db_path=db, local_dir=local, now="2026-11-01T00:00:00Z")
    before = _tree(local / "generations")
    def unexpected(*args, **kwargs):
        raise AssertionError("a successful local generation must be reused")
    monkeypatch.setattr(module, "export_snapshot", unexpected)
    report = module.run_backup(db_path=db, local_dir=local, nas_dir=nas, now="2026-11-01T00:00:00Z")
    assert report["verified"] == {"local": True, "nas": True}
    assert _tree(local / "generations") == before
    first = local / "generations" / report["generation"] / "snapshot"
    second = nas / "generations" / report["generation"] / "snapshot"
    assert _tree(first) == _tree(second)


def test_missing_source_does_not_create_empty_backup_database(tmp_path):
    module = _load_module()
    db, local = tmp_path / "missing.db", tmp_path / "backups"
    with pytest.raises(ValueError, match="must already exist"):
        module.run_backup(db_path=db, local_dir=local)
    assert not db.exists() and not local.exists()


def test_dry_run_is_inspection_not_a_verified_backup(tmp_path):
    module = _load_module()
    db, local = tmp_path / "kw" / "knowledge.db", tmp_path / "backups"
    _seed(db)
    report = module.run_backup(db_path=db, local_dir=local, dry_run=True)
    assert report["dry_run"] is True and report["verified"] == {"local": False}
    assert not local.exists()
