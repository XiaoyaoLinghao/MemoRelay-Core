"""F17 / AC-53: the import database is built in a private fd-held staging
directory and only the exact created inode is published to the target.

Migrated from the Codex round-3 counterexample
``test_F17_database_file_swap_before_sqlite_open_does_not_modify_foreign``:
after the real ``_create_database_file`` created and closed ``knowledge.db`` the
fault hook replaced that entry with a symlink to a synthetic foreign SQLite
database, and SQLite followed the link on its path-based re-open, writing into
the foreign database.  The fix keeps the created file descriptor, verifies the
directory entry identity immediately before SQLite opens it, and atomically
renames the finished, checkpointed database into the target through directory
fds — so a replaced ``knowledge.db`` name is never handed to SQLite and journal
files are never published.

Only synthetic pytest ``tmp_path`` data is used; no network, no real host data.
Threat-model boundary: this is deterministic same-process fault injection /
path-replacement resistance, not same-OS-user isolation.  ``fchdir`` is
process-global; the module lock only serializes imports in this process.

Round 6 / F17-R5 behaviour-contract upgrade (Lead-authorized minimal edit):
the two staging-replacement nodes below originally ended with
``assert not target.exists()``.  That assertion encoded the pre-round-6
cleanup-by-name bug: it could only pass while failed-import cleanup *deleted*
the already-detected replacement entry (because the replacement lives inside the
import's private staging directory, which lives inside the target directory).
Round-6 F17-R5 requires a detected different inode / symlink / non-regular
staging entry to be preserved and reported instead.  The replacement therefore
stays at its original path and the target directory that holds the private
staging directory must remain.  The new assertions are strictly stronger and
auditable: the original exception interface is unchanged, the preserved entry is
verified byte-for-byte/link-for-link at its original path, and the
``cleanup_report`` / ``cleanup_warnings`` evidence is checked; the unrelated
existing assertions (``INVALID_REQUEST`` code, foreign database bytes unchanged)
are untouched.  Normal cleanup with no replacement is still covered by
``test_F17_failure_after_wal_writes_cleans_staging_and_journals`` (this file)
and by ``tests/portability/test_import_cleanup_fix5.py``.
"""

from __future__ import annotations

import hashlib
import shutil
import sqlite3
from pathlib import Path

import pytest

import knowledge_weaver.portability.importing as importing
from knowledge_weaver.portability import SnapshotError, export_snapshot, import_snapshot
from knowledge_weaver.store import Store, current_items

from ..state.conftest import record


def _snapshot(store, tmp_path) -> Path:
    record(store, display_text="端口4200", value=4200)
    snapshot = tmp_path / "snapshot"
    export_snapshot(store, snapshot)
    return snapshot


def _foreign_database(tmp_path) -> tuple[Path, str]:
    foreign = tmp_path / "foreign.db"
    with sqlite3.connect(foreign) as conn:
        conn.execute("CREATE TABLE sentinel(value TEXT)")
        conn.execute("INSERT INTO sentinel VALUES('untouched')")
    return foreign, hashlib.sha256(foreign.read_bytes()).hexdigest()


def _raise_once(monkeypatch, name: str, action) -> dict:
    """Run ``action()`` once on the first call to ``importing.<name>``."""
    fired = {"done": False}
    original = getattr(importing, name)

    def wrapped(*args, **kwargs):
        if not fired["done"]:
            fired["done"] = True
            action()
        return original(*args, **kwargs)

    monkeypatch.setattr(importing, name, wrapped)
    return fired


# ---------------------------------------------------------------------------
# The Codex counterexample: replaced database file is never handed to SQLite
# ---------------------------------------------------------------------------


def test_F17_database_file_swap_before_sqlite_open_does_not_modify_foreign(
    store, tmp_path, monkeypatch
):
    snapshot = _snapshot(store, tmp_path)
    foreign, before = _foreign_database(tmp_path)
    target = tmp_path / "restored"
    original = importing._create_database_file

    def swap(staging):
        original(staging)
        path = staging.path / "knowledge.db"
        path.unlink()
        path.symlink_to(foreign)

    monkeypatch.setattr(importing, "_create_database_file", swap)
    with pytest.raises(SnapshotError) as excinfo:
        import_snapshot(snapshot, target)
    assert excinfo.value.code == "INVALID_REQUEST"
    assert hashlib.sha256(foreign.read_bytes()).hexdigest() == before, (
        "SQLite modified the synthetic foreign DB through the replaced "
        "knowledge.db symlink"
    )
    # Round 6 / F17-R5: the detected replacement is preserved and reported, not
    # deleted by its allowed name.  It lives inside the private staging
    # directory, so the target directory that holds it must remain (the old
    # ``assert not target.exists()`` could only pass if cleanup deleted the
    # foreign symlink — exactly the bug this round fixes).
    staging_db = next(
        path
        for path in target.rglob("knowledge.db")
        if any(".kw-import-staging-" in part for part in path.relative_to(target).parts)
    )
    assert staging_db.is_symlink(), "the detected replacement symlink was deleted or replaced"
    assert staging_db.readlink() == foreign, "the preserved symlink no longer points at the foreign DB"
    report = excinfo.value.cleanup_report
    assert isinstance(report, dict), "the failed import must expose its cleanup report"
    preserved = [
        entry
        for entry in report["preserved"]
        if entry["location"] == "staging" and entry["name"] == "knowledge.db"
    ]
    assert preserved, report
    assert "symlink" in preserved[0]["reason"], preserved
    assert report["warnings"], "a preserved foreign staging entry must be reported"
    assert excinfo.value.cleanup_warnings == report["warnings"]
    assert {"location": "staging", "name": "knowledge.db"} not in report["removed"], report
    assert target.is_dir(), "the target directory holding the preserved staging entry was removed"


def test_F17_database_file_replaced_by_regular_copy_is_refused(
    store, tmp_path, monkeypatch
):
    """Identity, not just symlink-following, protects the SQLite open."""
    snapshot = _snapshot(store, tmp_path)
    foreign, before = _foreign_database(tmp_path)
    target = tmp_path / "restored"
    original = importing._create_database_file

    def replace_with_copy(staging):
        original(staging)
        path = staging.path / "knowledge.db"
        path.unlink()
        shutil.copyfile(foreign, path)

    monkeypatch.setattr(importing, "_create_database_file", replace_with_copy)
    with pytest.raises(SnapshotError) as excinfo:
        import_snapshot(snapshot, target)
    assert excinfo.value.code == "INVALID_REQUEST"
    assert hashlib.sha256(foreign.read_bytes()).hexdigest() == before
    # Round 6 / F17-R5: the detected replacement copy is preserved and reported,
    # not deleted by its allowed name; the target directory holding the private
    # staging directory must remain (see the module docstring).
    staging_db = next(
        path
        for path in target.rglob("knowledge.db")
        if any(".kw-import-staging-" in part for part in path.relative_to(target).parts)
    )
    assert staging_db.is_file() and not staging_db.is_symlink(), (
        "the detected replacement regular file was deleted or replaced"
    )
    assert staging_db.read_bytes() == foreign.read_bytes(), (
        "the preserved replacement file content changed"
    )
    report = excinfo.value.cleanup_report
    assert isinstance(report, dict), "the failed import must expose its cleanup report"
    preserved = [
        entry
        for entry in report["preserved"]
        if entry["location"] == "staging" and entry["name"] == "knowledge.db"
    ]
    assert preserved, report
    assert "identity changed" in preserved[0]["reason"], preserved
    assert report["warnings"], "a preserved foreign staging entry must be reported"
    assert excinfo.value.cleanup_warnings == report["warnings"]
    assert {"location": "staging", "name": "knowledge.db"} not in report["removed"], report
    assert target.is_dir(), "the target directory holding the preserved staging entry was removed"


# ---------------------------------------------------------------------------
# A database name planted in the public target before activation is refused
# ---------------------------------------------------------------------------


def test_F17_planted_target_database_entry_is_refused_without_following(
    store, tmp_path, monkeypatch
):
    snapshot = _snapshot(store, tmp_path)
    foreign, before = _foreign_database(tmp_path)
    target = tmp_path / "restored"

    def plant():
        target.mkdir(exist_ok=True)
        (target / "knowledge.db").symlink_to(foreign)

    fired = _raise_once(monkeypatch, "_insert_rows", plant)
    with pytest.raises(SnapshotError) as excinfo:
        import_snapshot(snapshot, target)
    assert fired["done"] is True, "the fault injection must run during the build"
    assert excinfo.value.code == "INVALID_REQUEST"
    assert hashlib.sha256(foreign.read_bytes()).hexdigest() == before
    # The planted entry is not followed, overwritten or deleted as "ours".
    assert (target / "knowledge.db").is_symlink()


# ---------------------------------------------------------------------------
# Public path replacement around activation cannot redirect the publish step
# ---------------------------------------------------------------------------


def test_F17_target_swap_before_activation_does_not_receive_the_database(
    store, tmp_path, monkeypatch
):
    snapshot = _snapshot(store, tmp_path)
    target = tmp_path / "restored"
    moved = tmp_path / "moved"
    other = tmp_path / "other"
    other.mkdir()
    original = importing._finalize_staged_database

    def swap_then_finalize(handle, staging):
        original(handle, staging)
        target.rename(moved)
        target.symlink_to(other, target_is_directory=True)

    monkeypatch.setattr(importing, "_finalize_staged_database", swap_then_finalize)
    with pytest.raises(SnapshotError) as excinfo:
        import_snapshot(snapshot, target)
    assert excinfo.value.code == "INVALID_REQUEST"
    assert list(other.iterdir()) == [], "no write may reach the swapped-in directory"
    # The staging database was removed through the fd: the moved real directory
    # the import created contains no published database and no journal leftovers.
    assert moved.is_dir()
    assert list(moved.iterdir()) == []


# ---------------------------------------------------------------------------
# Successful import publishes exactly one checkpointed database file
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("preexisting", [False, True])
def test_F17_success_publishes_only_the_database_file(store, tmp_path, preexisting):
    snapshot = _snapshot(store, tmp_path)
    target = tmp_path / "restored"
    if preexisting:
        target.mkdir()
    report = import_snapshot(snapshot, target)
    assert Path(report["target_db"]) == target / "knowledge.db"
    # No WAL/shm/journal sibling and no private staging directory is published.
    assert sorted(path.name for path in target.iterdir()) == ["knowledge.db"]
    assert not [path for path in target.iterdir() if path.name.startswith(".kw-import")]
    # The published database is usable and has no leftover journal payload.
    restored = Store(report["target_db"])
    assert [item["text"] for item in current_items(restored, workspace_id="owner:demo")] == [
        "端口4200"
    ]
    with restored.read() as conn:
        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"


# ---------------------------------------------------------------------------
# Failure after the database/journal writes leaves no staging residue
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("preexisting", [False, True])
def test_F17_failure_after_wal_writes_cleans_staging_and_journals(
    store, tmp_path, monkeypatch, preexisting
):
    snapshot = _snapshot(store, tmp_path)
    target = tmp_path / "restored"
    if preexisting:
        target.mkdir()
    original = importing._validate_referential_integrity

    def boom(_conn):
        original(_conn)  # validate first, then fail like a real late failure
        raise SnapshotError("INVALID_REQUEST", "synthetic late import failure")

    monkeypatch.setattr(importing, "_validate_referential_integrity", boom)
    with pytest.raises(SnapshotError):
        import_snapshot(snapshot, target)
    if preexisting:
        assert target.is_dir()
        assert list(target.iterdir()) == [], (
            "a failed import must remove its staging database and journal files"
        )
    else:
        assert not target.exists(), "the directory created by this import must be cleaned"
