"""F17-R5 / AC-53: failed-import cleanup deletes only files this import owns.

Migrated from the Codex round-4 counterexamples
(``artifacts/T20-codex-round4/.../repros/test_new_variants.py``):

* ``test_F17_failed_post_activation_cleanup_preserves_replacement_entry``: the
  activation identity check correctly refused a replaced ``target/knowledge.db``
  but cleanup then unlinked the fixed file name and deleted the replacement.
* ``test_F17_failure_does_not_delete_unrelated_staging_file``: staging cleanup
  enumerated the whole private directory and deleted an entry this import never
  created.

The fix verifies the held ``db_file_fd`` ``(st_dev, st_ino)`` against the target
directory entry before any unlink; replacement/symlink/missing/unknown entries
are preserved and reported on the original exception (``cleanup_report`` /
``cleanup_warnings``).  Inside staging only ``knowledge.db`` and the SQLite
journal siblings are deleted by name; anything else is preserved.  Target
``-wal``/``-shm``/``-journal`` siblings are removed only after the database
identity was verified as this import's inode.  A non-empty target directory is
never removed.

Only synthetic pytest ``tmp_path`` data is used; no network, no real host data.
Threat-model boundary: deterministic same-process fault injection and
path-replacement resistance, not same-OS-user isolation (``fchdir`` is
process-global, disclosed in the module docstring).
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
from pathlib import Path

import pytest

import knowledge_weaver.portability.importing as importing
from knowledge_weaver.portability import SnapshotError, export_snapshot, import_snapshot
from knowledge_weaver.store import Store, current_items

from ..state.conftest import record

REPLACEMENT = b"C4_UNRELATED_USER_FILE"


def _snapshot(store, tmp_path) -> Path:
    record(store, display_text="端口4200", value=4200)
    snapshot = tmp_path / "snapshot"
    export_snapshot(store, snapshot)
    return snapshot


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _foreign_database(tmp_path) -> tuple[Path, str]:
    foreign = tmp_path / "foreign.db"
    with sqlite3.connect(foreign) as conn:
        conn.execute("CREATE TABLE sentinel(value TEXT)")
        conn.execute("INSERT INTO sentinel VALUES('untouched')")
    return foreign, _sha256(foreign)


def _cleanup_report(exc: BaseException) -> dict:
    report = getattr(exc, "cleanup_report", None)
    assert isinstance(report, dict), (
        "a failed import must expose its cleanup ownership report on the raised exception"
    )
    assert set(report) == {"removed", "preserved", "warnings"}
    return report


def _preserved(report: dict, location: str, name: str) -> list[dict]:
    return [
        entry
        for entry in report["preserved"]
        if entry["location"] == location and entry["name"] == name
    ]


def _staging_files(target: Path) -> list[Path]:
    return [
        path
        for path in target.rglob("*")
        if any(".kw-import-staging-" in part for part in path.relative_to(target).parts)
    ]


# ---------------------------------------------------------------------------
# Codex round-4 counterexample 1: a replacement target entry is never deleted
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "preexisting", [False, True], ids=["new-target", "existing-target"]
)
def test_F17_post_activation_replacement_is_preserved(
    store, tmp_path, monkeypatch, preexisting
):
    snapshot = _snapshot(store, tmp_path)
    target = tmp_path / "restored"
    if preexisting:
        target.mkdir()
    original = importing._verify_activated_database

    def replace_then_verify(handle, staging):
        path = target / "knowledge.db"
        path.unlink()
        path.write_bytes(REPLACEMENT)
        original(handle, staging)

    monkeypatch.setattr(importing, "_verify_activated_database", replace_then_verify)
    with pytest.raises(SnapshotError) as excinfo:
        import_snapshot(snapshot, target)
    # Existing SnapshotError interface is unchanged ...
    assert excinfo.value.code == "INVALID_REQUEST"
    assert excinfo.value.message == str(excinfo.value)
    # ... and the replacement entry created by someone else is still there.
    entry = target / "knowledge.db"
    assert entry.is_file(), "failure cleanup deleted an entry it did not create"
    assert entry.read_bytes() == REPLACEMENT
    assert target.is_dir(), "a target directory still holding a preserved entry was removed"
    report = _cleanup_report(excinfo.value)
    preserved = _preserved(report, "target", "knowledge.db")
    assert preserved, report
    assert "identity changed" in preserved[0]["reason"]
    assert excinfo.value.cleanup_warnings == report["warnings"]
    assert any("knowledge.db" in warning for warning in excinfo.value.cleanup_warnings)
    assert report["removed"] == []


# ---------------------------------------------------------------------------
# Target journal siblings are kept when the database identity is not ours
# ---------------------------------------------------------------------------


def test_F17_post_activation_replacement_keeps_target_journal_entries(
    store, tmp_path, monkeypatch
):
    snapshot = _snapshot(store, tmp_path)
    target = tmp_path / "restored"
    planted = {
        "knowledge.db-wal": b"FOREIGN_WAL",
        "knowledge.db-shm": b"FOREIGN_SHM",
        "knowledge.db-journal": b"FOREIGN_JOURNAL",
    }
    original = importing._verify_activated_database

    def replace_then_verify(handle, staging):
        path = target / "knowledge.db"
        path.unlink()
        path.write_bytes(REPLACEMENT)
        for name, payload in planted.items():
            (target / name).write_bytes(payload)
        original(handle, staging)

    monkeypatch.setattr(importing, "_verify_activated_database", replace_then_verify)
    with pytest.raises(SnapshotError) as excinfo:
        import_snapshot(snapshot, target)
    assert excinfo.value.code == "INVALID_REQUEST"
    assert (target / "knowledge.db").read_bytes() == REPLACEMENT
    for name, payload in planted.items():
        assert (target / name).read_bytes() == payload, (
            f"cleanup deleted {name} although the database entry was not this import's inode"
        )
    report = _cleanup_report(excinfo.value)
    assert _preserved(report, "target", "knowledge.db")
    assert all(
        entry["location"] != "target" or entry["name"] == "knowledge.db"
        for entry in report["preserved"]
    )


# ---------------------------------------------------------------------------
# A symlink replacement (to a synthetic foreign database) is kept, not followed
# ---------------------------------------------------------------------------


def test_F17_post_activation_symlink_replacement_is_preserved(
    store, tmp_path, monkeypatch
):
    snapshot = _snapshot(store, tmp_path)
    foreign, before = _foreign_database(tmp_path)
    target = tmp_path / "restored"
    original = importing._verify_activated_database

    def symlink_then_verify(handle, staging):
        path = target / "knowledge.db"
        path.unlink()
        path.symlink_to(foreign)
        original(handle, staging)

    monkeypatch.setattr(importing, "_verify_activated_database", symlink_then_verify)
    with pytest.raises(SnapshotError) as excinfo:
        import_snapshot(snapshot, target)
    assert excinfo.value.code == "INVALID_REQUEST"
    assert (target / "knowledge.db").is_symlink(), (
        "failure cleanup replaced/deleted a symlink entry it did not create"
    )
    assert (target / "knowledge.db").resolve() == foreign.resolve()
    assert _sha256(foreign) == before, "cleanup followed the symlink into the foreign DB"
    report = _cleanup_report(excinfo.value)
    preserved = _preserved(report, "target", "knowledge.db")
    assert preserved and "symlink" in preserved[0]["reason"]


# ---------------------------------------------------------------------------
# Deleted database: no other staging/target entry is mistaken for ours
# ---------------------------------------------------------------------------


def test_F17_post_activation_deleted_database_preserves_unrelated_entries(
    store, tmp_path, monkeypatch
):
    snapshot = _snapshot(store, tmp_path)
    target = tmp_path / "restored"
    unrelated_target = target / "unrelated-target-note.txt"
    placed: dict[str, Path] = {}
    original = importing._verify_activated_database

    def delete_then_verify(handle, staging):
        (target / "knowledge.db").unlink()
        unrelated_target.write_text("C4_TARGET_KEEP")
        note = staging.path / "user-note.txt"
        note.write_text("C4_STAGING_KEEP")
        placed["staging"] = note
        original(handle, staging)

    monkeypatch.setattr(importing, "_verify_activated_database", delete_then_verify)
    with pytest.raises(SnapshotError) as excinfo:
        import_snapshot(snapshot, target)
    assert excinfo.value.code == "INVALID_REQUEST"
    assert unrelated_target.read_text() == "C4_TARGET_KEEP", (
        "cleanup deleted an unrelated target entry"
    )
    assert placed["staging"].read_text() == "C4_STAGING_KEEP", (
        "cleanup deleted an unrelated staging entry when the database was missing"
    )
    assert not (target / "knowledge.db").exists()
    assert target.is_dir(), "non-empty target directory must not be removed"
    report = _cleanup_report(excinfo.value)
    target_db = _preserved(report, "target", "knowledge.db")
    assert target_db and "missing" in target_db[0]["reason"]
    staging_note = _preserved(report, "staging", "user-note.txt")
    assert staging_note and "not created by this import" in staging_note[0]["reason"]


# ---------------------------------------------------------------------------
# Codex round-4 counterexample 2: foreign staging entry + original exception
# ---------------------------------------------------------------------------


def test_F17_failure_does_not_delete_unrelated_staging_file(
    store, tmp_path, monkeypatch
):
    snapshot = _snapshot(store, tmp_path)
    target = tmp_path / "restored"
    placed: list[Path] = []
    original = importing._create_database_file

    def add_foreign_then_fail(staging):
        original(staging)
        note = staging.path / "user-note.txt"
        note.write_text("C4_KEEP_UNRELATED")
        placed.append(note)
        raise RuntimeError("injected import failure")

    monkeypatch.setattr(importing, "_create_database_file", add_foreign_then_fail)
    with pytest.raises(RuntimeError) as excinfo:
        import_snapshot(snapshot, target)
    # The original exception is re-raised unchanged, never swallowed/rewrapped.
    assert type(excinfo.value) is RuntimeError
    assert str(excinfo.value) == "injected import failure"
    assert placed and placed[0].is_file(), (
        "staging cleanup enumerated and deleted a file this import never created"
    )
    assert placed[0].read_text() == "C4_KEEP_UNRELATED"
    assert target.is_dir(), "target directory holding a preserved entry was removed"
    assert any(
        path.name == "user-note.txt" for path in _staging_files(target)
    ), "the preserved staging foreign entry disappeared"
    report = _cleanup_report(excinfo.value)
    preserved = _preserved(report, "staging", "user-note.txt")
    assert preserved and "not created by this import" in preserved[0]["reason"]
    assert {"location": "staging", "name": "knowledge.db"} in report["removed"], (
        "the database this import created must still be cleaned up"
    )
    assert excinfo.value.cleanup_warnings == report["warnings"]


# ---------------------------------------------------------------------------
# Positive controls: success publishes one DB; a normal late failure cleans up
# ---------------------------------------------------------------------------


def test_F17_post_activation_failure_removes_only_owned_database(
    store, tmp_path, monkeypatch
):
    """When the entry *is* our inode, cleanup still removes it via the held fd."""
    snapshot = _snapshot(store, tmp_path)
    target = tmp_path / "restored"
    moved = tmp_path / "moved"
    other = tmp_path / "other"
    other.mkdir()
    original = importing._verify_activated_database

    def verify_then_swap_dir(handle, staging):
        original(handle, staging)
        target.rename(moved)
        target.symlink_to(other, target_is_directory=True)

    monkeypatch.setattr(importing, "_verify_activated_database", verify_then_swap_dir)
    with pytest.raises(SnapshotError) as excinfo:
        import_snapshot(snapshot, target)
    assert excinfo.value.code == "INVALID_REQUEST"
    report = _cleanup_report(excinfo.value)
    assert {"location": "target", "name": "knowledge.db"} in report["removed"], (
        "a verified owned database must still be cleaned up"
    )
    assert report["preserved"] == []
    # The database and the staging directory were removed through the held fds;
    # the swapped-in directory and the renamed original never receive a write.
    assert moved.is_dir() and list(moved.iterdir()) == []
    assert list(other.iterdir()) == []
    assert target.is_symlink()


def test_F17_successful_import_publishes_only_database(store, tmp_path):
    snapshot = _snapshot(store, tmp_path)
    target = tmp_path / "restored"
    report = import_snapshot(snapshot, target)
    assert Path(report["target_db"]) == target / "knowledge.db"
    assert sorted(path.name for path in target.iterdir()) == ["knowledge.db"]
    assert not _staging_files(target)
    assert report["cleanup_warnings"] == []
    restored = Store(report["target_db"])
    assert [item["text"] for item in current_items(restored, workspace_id="owner:demo")] == [
        "端口4200"
    ]
    with restored.read() as conn:
        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"


@pytest.mark.parametrize(
    "preexisting", [False, True], ids=["new-target", "existing-target"]
)
def test_F17_failure_after_database_writes_removes_created_files(
    store, tmp_path, monkeypatch, preexisting
):
    snapshot = _snapshot(store, tmp_path)
    target = tmp_path / "restored"
    if preexisting:
        target.mkdir()
    staging_ref: dict = {}
    seen: dict = {}
    original_staging = importing._create_staging_dir
    original_integrity = importing._validate_referential_integrity

    def capture_staging(handle):
        staging = original_staging(handle)
        staging_ref["staging"] = staging
        return staging

    def boom(conn):
        original_integrity(conn)  # fail like a real late import failure
        seen["entries"] = sorted(os.listdir(staging_ref["staging"].fd))
        raise SnapshotError("INVALID_REQUEST", "synthetic late import failure")

    monkeypatch.setattr(importing, "_create_staging_dir", capture_staging)
    monkeypatch.setattr(importing, "_validate_referential_integrity", boom)
    with pytest.raises(SnapshotError) as excinfo:
        import_snapshot(snapshot, target)
    assert type(excinfo.value) is SnapshotError
    assert str(excinfo.value) == "synthetic late import failure"
    assert "knowledge.db" in seen["entries"], "the write phase must have created the database"
    report = _cleanup_report(excinfo.value)
    assert {"location": "staging", "name": "knowledge.db"} in report["removed"]
    assert report["preserved"] == [], report
    if preexisting:
        assert target.is_dir(), "a pre-existing user directory must not be deleted"
        assert list(target.iterdir()) == [], (
            "failed import left database/journal residue behind"
        )
    else:
        assert not target.exists(), "the directory created by this import must be cleaned"
