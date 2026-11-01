"""T20 F17 (round 3): hold the import target directory identity via a dir fd.

Counterexamples migrated from the round-2 Codex review (review/T20-CODEX-ROUND2.md
F17): ``_create_target_dir`` checked the new directory identity but closed the
dir fd before writing, so a post-check rename + symlink swap redirected the
database into another directory.  All paths here are synthetic pytest tmp_paths.
"""
from __future__ import annotations

from pathlib import Path

import pytest

import knowledge_weaver.portability.importing as importing
from knowledge_weaver.portability import SnapshotError, export_snapshot, import_snapshot
from knowledge_weaver.store import Store, current_items

from ..state.conftest import record


def _snapshot(store, tmp_path):
    record(store, display_text="端口4200", value=4200)
    snapshot = tmp_path / "snapshot"
    export_snapshot(store, snapshot)
    return snapshot


def _swap_once(monkeypatch, target: Path, *, hook):
    """Install a fault that performs ``hook()`` exactly once during the import."""

    fired = {"swapped": False}

    def wrapped(*args, **kwargs):
        if not fired["swapped"]:
            fired["swapped"] = True
            hook()
        return None

    monkeypatch.setattr(importing, "_insert_rows", wrapped)
    return fired


# ---------------------------------------------------------------------------
# (a) the Codex counterexample: swap between identity check and first write
# ---------------------------------------------------------------------------


def test_F17_directory_swap_after_identity_check_refused(store, tmp_path, monkeypatch):
    snapshot = _snapshot(store, tmp_path)
    target = tmp_path / "dest"
    other = tmp_path / "other"
    other.mkdir()
    original = importing._create_target_dir

    def swapped(target_dir):
        handle = original(target_dir)
        target_dir.rename(tmp_path / "moved")
        target_dir.symlink_to(other, target_is_directory=True)
        return handle

    monkeypatch.setattr(importing, "_create_target_dir", swapped)
    with pytest.raises(SnapshotError) as excinfo:
        import_snapshot(snapshot, target)
    assert excinfo.value.code == "INVALID_REQUEST"
    assert list(other.iterdir()) == [], "database written through swapped symlink after identity check"
    assert not (other / "knowledge.db").exists()


# ---------------------------------------------------------------------------
# (b) static symlink target / symlinked ancestor keep being refused
# ---------------------------------------------------------------------------


def test_F17_static_symlink_target_is_refused(store, tmp_path):
    snapshot = _snapshot(store, tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    link = tmp_path / "target-link"
    link.symlink_to(outside, target_is_directory=True)
    with pytest.raises(SnapshotError) as excinfo:
        import_snapshot(snapshot, link)
    assert excinfo.value.code == "INVALID_REQUEST"
    assert list(outside.iterdir()) == []


def test_F17_symlinked_ancestor_is_refused(store, tmp_path):
    snapshot = _snapshot(store, tmp_path)
    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    link = tmp_path / "parent-link"
    link.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(SnapshotError) as excinfo:
        import_snapshot(snapshot, link / "child")
    assert excinfo.value.code == "INVALID_REQUEST"
    assert list(real_parent.iterdir()) == []


# ---------------------------------------------------------------------------
# (c) normal import still succeeds and restores into a new database
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("preexisting", [False, True])
def test_F17_normal_import_restores_into_new_database(store, tmp_path, preexisting):
    snapshot = _snapshot(store, tmp_path)
    target = tmp_path / "restored"
    if preexisting:
        target.mkdir()
    report = import_snapshot(snapshot, target)
    assert Path(report["target_db"]) == target / "knowledge.db"
    assert (target / "knowledge.db").is_file()
    assert target.is_dir()
    restored = Store(report["target_db"])
    items = current_items(restored, workspace_id="owner:demo")
    assert [item["text"] for item in items] == ["端口4200"]
    with restored.read() as conn:
        assert conn.execute("SELECT COUNT(*) AS c FROM events").fetchone()["c"] >= 1
        assert conn.execute("SELECT COUNT(*) AS c FROM claim_revisions").fetchone()["c"] >= 1


def test_F17_relative_snapshot_and_target_paths_still_restore(store, tmp_path, monkeypatch):
    """The write phase anchors the cwd, so relative source/target paths must be
    resolved before anchoring instead of being re-read inside the target."""
    _snapshot(store, tmp_path)
    monkeypatch.chdir(tmp_path)
    report = import_snapshot("snapshot", "restored")
    assert Path(report["target_db"]) == tmp_path / "restored" / "knowledge.db"
    restored = Store(report["target_db"])
    assert [item["text"] for item in current_items(restored, workspace_id="owner:demo")] == [
        "端口4200"
    ]


# ---------------------------------------------------------------------------
# (d) failure cleanup removes only what this import created
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("preexisting", [False, True])
def test_F17_failed_import_cleanup_keeps_pre_existing_paths(store, tmp_path, monkeypatch, preexisting):
    snapshot = _snapshot(store, tmp_path)

    def boom(*_args, **_kwargs):
        raise SnapshotError("INVALID_REQUEST", "synthetic mid-import failure")

    monkeypatch.setattr(importing, "_insert_rows", boom)
    target = tmp_path / ("existing-empty" if preexisting else "created-by-import")
    if preexisting:
        target.mkdir()
    with pytest.raises(SnapshotError):
        import_snapshot(snapshot, target)
    if preexisting:
        assert target.is_dir(), "a pre-existing directory must never be removed"
        assert list(target.iterdir()) == [], "only files created by this import are cleaned"
    else:
        assert not target.exists(), "the directory created by this import must be cleaned"


# ---------------------------------------------------------------------------
# (e) swap in the middle of the write phase: fd-anchored writes cannot be redirected
# ---------------------------------------------------------------------------


def test_F17_directory_swap_mid_write_does_not_write_through(store, tmp_path, monkeypatch):
    snapshot = _snapshot(store, tmp_path)
    target = tmp_path / "dest"
    moved = tmp_path / "moved"
    other = tmp_path / "other"
    other.mkdir()

    def swap():
        target.rename(moved)
        target.symlink_to(other, target_is_directory=True)

    fired = _swap_once(monkeypatch, target, hook=swap)
    with pytest.raises(SnapshotError) as excinfo:
        import_snapshot(snapshot, target)
    assert excinfo.value.code == "INVALID_REQUEST"
    assert fired["swapped"] is True, "the fault injection must run during the write phase"
    assert list(other.iterdir()) == [], "no write may reach the swapped-in directory"
    assert not (other / "knowledge.db").exists()
    # The import created ``dest`` (now renamed to ``moved``): cleanup must remove
    # the database it created through the held fd, leaving the directory empty.
    assert moved.is_dir()
    assert list(moved.iterdir()) == []
