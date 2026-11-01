"""F17-R5 / AC-53: staging cleanup owns files by inode identity, not by name.

Migrated from the Codex round-5 counterexample
(``artifacts/T20-codex-round5/run-20261031-independent/repros/test_round5_variants.py``):

* ``test_F17_staging_replacement_inode_is_preserved_when_identity_failure_detected``:
  ``_create_database_file`` created and held ``staging/knowledge.db``, the entry
  was then replaced by a different regular file.  The import correctly detected
  the inode change and raised ``SnapshotError``, but failed-import cleanup then
  unlinked the replacement because ``knowledge.db`` is in the allowed-name set
  (``cleanup_report`` was ``removed=[staging knowledge.db]`` / ``preserved=[]`` /
  ``warnings=[]``).

The fix applies the already-held ``db_file_fd`` ``(st_dev, st_ino)`` identity to
the staging entry exactly like the public target:

* a replaced regular file, a symlink or a non-regular entry is preserved and
  reported in ``cleanup_report.preserved`` / ``cleanup_warnings`` (and logged);
* staging ``-wal``/``-shm``/``-journal`` siblings are removed only together with
  an owned database (a replacement's journals are never deleted by name);
* an unrelated staging entry is still preserved and reported, while the files
  this import really created are still cleaned up on a normal late failure;
* the original exception (type, message, ``code``) is re-raised unchanged with
  the cleanup report attached, and a successful import leaves only
  ``target/knowledge.db`` with empty cleanup warnings.

Only synthetic pytest ``tmp_path`` data is used; no network, no real host data.
Threat-model boundary: deterministic same-process fault injection and
path-replacement resistance, not same-OS-user isolation (``fchdir`` is
process-global, disclosed in the importing module docstring).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

import knowledge_weaver.portability.importing as importing
from knowledge_weaver.portability import SnapshotError, export_snapshot, import_snapshot
from knowledge_weaver.store import Store

from ..state.conftest import record

REPLACEMENT = b"C5_FOREIGN_DIFFERENT_INODE"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _snapshot(store, tmp_path) -> Path:
    record(store, display_text="端口4200", value=4200)
    snapshot = tmp_path / "snapshot"
    export_snapshot(store, snapshot)
    return snapshot


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


def _foreign_database(tmp_path) -> Path:
    """A synthetic foreign SQLite file a symlink replacement may point at."""
    import sqlite3

    foreign = tmp_path / "foreign.db"
    with sqlite3.connect(foreign) as conn:
        conn.execute("CREATE TABLE sentinel(value TEXT)")
        conn.execute("INSERT INTO sentinel VALUES('untouched')")
    return foreign


# ---------------------------------------------------------------------------
# Codex round-5 counterexample: a different inode is preserved, never unlinked
# by its allowed name
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("replacement", ["regular", "symlink", "directory"])
def test_F17_staging_replacement_inode_is_preserved(
    store, tmp_path, monkeypatch, replacement
):
    """The exact round-5 counterexample plus symlink/non-regular neighbours.

    The import fails because it detected that ``staging/knowledge.db`` is no
    longer the inode it created; cleanup must keep that entry and report it
    instead of deleting it because the name is ``knowledge.db``.
    """
    snapshot = _snapshot(store, tmp_path)
    target = tmp_path / "restored"
    foreign = _foreign_database(tmp_path)
    placed: list[Path] = []
    original = importing._create_database_file

    def swap(staging):
        original(staging)
        entry = staging.path / "knowledge.db"
        entry.unlink()
        if replacement == "regular":
            entry.write_bytes(REPLACEMENT)
        elif replacement == "symlink":
            entry.symlink_to(foreign)
        else:
            entry.mkdir()
        placed.append(entry)

    monkeypatch.setattr(importing, "_create_database_file", swap)
    with pytest.raises(SnapshotError) as excinfo:
        import_snapshot(snapshot, target)

    # The original exception interface is unchanged (type/message/code).
    assert type(excinfo.value) is SnapshotError
    assert excinfo.value.code == "INVALID_REQUEST"
    assert excinfo.value.message == str(excinfo.value)
    assert "identity changed" in excinfo.value.message
    assert "never handed to SQLite" in excinfo.value.message

    # The replacement created by someone else is still there, untouched.
    assert placed and placed[0].exists(), "failed-import cleanup deleted a foreign staging entry"
    entry = placed[0]
    if replacement == "regular":
        assert entry.is_file() and not entry.is_symlink()
        assert entry.read_bytes() == REPLACEMENT
    elif replacement == "symlink":
        assert entry.is_symlink()
        assert entry.resolve() == foreign.resolve()
    else:
        assert entry.is_dir() and not entry.is_symlink()

    report = _cleanup_report(excinfo.value)
    preserved = _preserved(report, "staging", "knowledge.db")
    assert preserved, report
    reason = preserved[0]["reason"]
    expected_reason = {
        "regular": "identity changed",
        "symlink": "symlink",
        "directory": "not a regular file",
    }[replacement]
    assert expected_reason in reason, (replacement, preserved)
    assert {"location": "staging", "name": "knowledge.db"} not in report["removed"], report
    assert report["warnings"], "a preserved foreign staging entry must be reported as a warning"
    assert excinfo.value.cleanup_warnings == report["warnings"]
    assert any("knowledge.db" in warning for warning in excinfo.value.cleanup_warnings)
    # The target directory still holds the preserved staging entry and is kept.
    assert target.is_dir()
    assert any(path.name == "knowledge.db" for path in _staging_files(target))


@pytest.mark.parametrize("database_state", ["replaced", "deleted"])
def test_F17_staging_journal_siblings_kept_when_database_not_owned(
    store, tmp_path, monkeypatch, database_state
):
    """``-wal``/``-shm``/``-journal`` are removed only with an owned database.

    If the staging database entry was replaced (or deleted) the journal names may
    belong to someone else's database; deleting them by name would destroy
    foreign data, so they are preserved and reported instead.
    """
    snapshot = _snapshot(store, tmp_path)
    target = tmp_path / "restored"
    planted = {
        "knowledge.db-wal": b"C5_FOREIGN_WAL",
        "knowledge.db-shm": b"C5_FOREIGN_SHM",
        "knowledge.db-journal": b"C5_FOREIGN_JOURNAL",
    }
    placed: dict[str, Path] = {}
    original = importing._create_database_file

    def swap(staging):
        original(staging)
        entry = staging.path / "knowledge.db"
        if database_state == "replaced":
            entry.unlink()
            entry.write_bytes(REPLACEMENT)
        else:
            entry.unlink()
        for name, payload in planted.items():
            path = staging.path / name
            path.write_bytes(payload)
            placed[name] = path

    monkeypatch.setattr(importing, "_create_database_file", swap)
    with pytest.raises(SnapshotError) as excinfo:
        import_snapshot(snapshot, target)
    assert excinfo.value.code == "INVALID_REQUEST"

    report = _cleanup_report(excinfo.value)
    for name, payload in planted.items():
        assert placed[name].read_bytes() == payload, (
            f"cleanup deleted {name} although the staging database was not this import's inode"
        )
        preserved = _preserved(report, "staging", name)
        assert preserved, (name, report)
        assert "ownership was not verified" in preserved[0]["reason"], (name, preserved)
    if database_state == "replaced":
        assert _preserved(report, "staging", "knowledge.db"), report
    else:
        # A missing database has nothing to preserve under that name.
        assert not _preserved(report, "staging", "knowledge.db"), report
    assert report["removed"] == [], report
    assert report["warnings"]
    assert excinfo.value.cleanup_warnings == report["warnings"]


# ---------------------------------------------------------------------------
# Unrelated staging entry + owned database: both behaviours in one failure
# ---------------------------------------------------------------------------


def test_F17_unrelated_staging_entry_preserved_while_owned_database_is_cleaned(
    store, tmp_path, monkeypatch
):
    """An unrelated entry is preserved/reported; the database this import really
    created is still removed on a normal injected failure."""
    snapshot = _snapshot(store, tmp_path)
    target = tmp_path / "restored"
    placed: list[Path] = []
    original = importing._create_database_file

    def add_foreign_then_fail(staging):
        original(staging)
        note = staging.path / "user-note.txt"
        note.write_text("C5_KEEP_UNRELATED")
        placed.append(note)
        raise RuntimeError("injected import failure")

    monkeypatch.setattr(importing, "_create_database_file", add_foreign_then_fail)
    with pytest.raises(RuntimeError) as excinfo:
        import_snapshot(snapshot, target)
    # The original exception is re-raised unchanged, never swallowed/rewrapped.
    assert type(excinfo.value) is RuntimeError
    assert str(excinfo.value) == "injected import failure"

    assert placed and placed[0].is_file()
    assert placed[0].read_text() == "C5_KEEP_UNRELATED"
    report = _cleanup_report(excinfo.value)
    preserved = _preserved(report, "staging", "user-note.txt")
    assert preserved and "not created by this import" in preserved[0]["reason"], report
    assert {"location": "staging", "name": "knowledge.db"} in report["removed"], report
    assert excinfo.value.cleanup_warnings == report["warnings"]
    assert target.is_dir(), "target directory holding a preserved entry was removed"
    assert any(path.name == "user-note.txt" for path in _staging_files(target))


# ---------------------------------------------------------------------------
# Positive controls: normal late failure cleans own files; success publishes one
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("preexisting", [False, True], ids=["new-target", "existing-target"])
def test_F17_normal_late_failure_still_cleans_created_files(
    store, tmp_path, monkeypatch, preexisting
):
    """A failure after the database writes still removes this import's own
    staging database (and journals) — the identity rule does not stop normal
    cleanup."""
    snapshot = _snapshot(store, tmp_path)
    target = tmp_path / "restored"
    if preexisting:
        target.mkdir()
    original = importing._validate_referential_integrity

    def boom(conn):
        original(conn)  # fail like a real late import failure
        raise SnapshotError("INVALID_REQUEST", "synthetic late import failure")

    monkeypatch.setattr(importing, "_validate_referential_integrity", boom)
    with pytest.raises(SnapshotError) as excinfo:
        import_snapshot(snapshot, target)
    assert type(excinfo.value) is SnapshotError
    assert str(excinfo.value) == "synthetic late import failure"
    assert excinfo.value.code == "INVALID_REQUEST"
    report = _cleanup_report(excinfo.value)
    assert {"location": "staging", "name": "knowledge.db"} in report["removed"], report
    assert report["preserved"] == [], report
    assert report["warnings"] == [], report
    if preexisting:
        assert target.is_dir()
        assert list(target.iterdir()) == [], "failed import left database/journal residue behind"
    else:
        assert not target.exists(), "the directory created by this import must be cleaned"


def test_F17_successful_import_publishes_only_the_owned_database(store, tmp_path):
    """Success path: only the activated database is published, staging is fully
    removed and cleanup reports nothing preserved or warned about."""
    snapshot = _snapshot(store, tmp_path)
    target = tmp_path / "restored"
    report = import_snapshot(snapshot, target)
    assert Path(report["target_db"]) == target / "knowledge.db"
    assert sorted(path.name for path in target.iterdir()) == ["knowledge.db"]
    assert not _staging_files(target)
    assert report["cleanup_warnings"] == []
    restored = Store(report["target_db"])
    with restored.read() as conn:
        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
