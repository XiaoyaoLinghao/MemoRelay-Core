"""Validated, isolated snapshot import (T14).

Safety rules (AC-31/32/46/53):

* only a new/empty target directory is accepted; the imported database never
  overwrites a live one;
* the target directory is held open by fd for the whole import, and the database
  is built in a private fd-held *staging directory*, checkpointed, identity
  verified against the file descriptor created by this import, and only then
  atomically renamed to ``target/knowledge.db`` through the target's dir_fd — so
  a replaced target path or ``knowledge.db`` entry is never handed to SQLite
  (F17).  ``fchdir`` is process-global; the module lock only serializes imports
  in this process and does not sandbox other threads (same-OS-user tampering is
  outside this threat model, disclosed as a limitation);
* every manifest field is validated: supported format/schema/contract versions,
  declared file set, sizes and SHA-256 hashes, no absolute paths, no ``..``
  traversal, no symlinks in the package;
* unknown record columns/types are rejected instead of silently dropped;
* lifecycle anchors/tombstones are imported as-is (restore does not renew
  retention), expired bodies stay empty;
* a new ``restore_epoch`` is assigned (semantic source backend id is reported,
  not reused as the local backend identity) and the snapshot watermark is
  preserved so adapters can reconcile receipts above it;
* failed/publishing cleanup only deletes files this import demonstrably created
  (F17-R5): ``target/knowledge.db`` is unlinked through the target fd only while
  the held ``db_file_fd`` ``(st_dev, st_ino)`` still matches the entry, and
  target journal siblings only together with such an owned database; a replaced,
  symlinked or missing entry is preserved and reported on the raised exception
  (``cleanup_report`` / ``cleanup_warnings``).  The same identity rule applies
  to the private staging directory: ``staging/knowledge.db`` is unlinked only
  while it is still the inode held by ``db_file_fd``, its SQLite journal
  siblings are removed only together with such an owned database, and every
  other staging entry is preserved instead of being enumerated away — the fixed
  file name is never treated as ownership;
* the durable ``jobs.recovered_by_job_id`` recovery association restored from a
  snapshot is validated as a reference (AC52-R5): it must name an existing,
  different, same-operation operation-ledger job that actually succeeded after
  the failure was recorded.  A non-NULL id is not proof of recovery, so a
  logically corrupt snapshot is refused instead of restoring hidden failures;
* completion order (AC52-R7) is carried by the durable ``jobs.completed_seq``
  column, not by timestamp text: the ``jobs`` timestamp columns
  (``created_at``/``updated_at``/``recovered_at``) are parsed and normalized to
  canonical UTC seconds on import, and a present recovery association whose
  completion order is missing or not strictly later is refused as logically
  corrupt (``INVALID_REQUEST``).  A pre-round-7 snapshot that contains recovery
  links therefore fails closed instead of silently hiding a failure, and the
  restored ``backend_meta.ledger_seq`` is seeded with ``max(completed_seq)`` so a
  success after the restore still completes strictly later than every restored
  row;
* keyword/FTS projections are rebuilt; vectors are rebuilt, not restored;
* optional bootstrap mode marks imported raw parts as ``bootstrap_pending`` so
  retention cannot delete un-extracted history.
"""

from __future__ import annotations

import contextlib
import datetime as _dt
import hashlib
import json
import logging
import os
import sqlite3
import stat
import threading
import uuid
from pathlib import Path, PurePosixPath

from knowledge_weaver import contracts
from knowledge_weaver.store.db import Store, utc_now
from knowledge_weaver.store.status import VALID_RECOVERY_LINK_SQL
from knowledge_weaver.jobs.repair_versions import upgrade_sql

from .export import (
    BACKEND_FILE,
    FORMAT_VERSION,
    RECORDS_DIR,
    SNAPSHOT_SCHEMA_VERSION,
    TABLES,
    V3_TABLES,
    V4_TABLES,
    V5_TABLES,
)

logger = logging.getLogger(__name__)

IMPORT_ORDER = (
    "clients",
    "events",
    "event_revisions",
    "event_parts",
    "receipts",
    "claims",
    "claim_revisions",
    "evidence_links",
    "source_semantic_exclusions",
    "current_claims",
    "task_views",
    "entities",
    "entity_aliases",
    "correction_operations",
    "capabilities",
    "resources",
    "lifecycle_records",
    "tombstones",
    "jobs",
    "job_attempts",
    "source_duplicate_resolutions",
)

SUPPORTED_FORMAT_VERSIONS = {FORMAT_VERSION}
SUPPORTED_SNAPSHOT_SCHEMA_VERSIONS = {1, 2, 3, 4, SNAPSHOT_SCHEMA_VERSION}


def _table_required(table: str, schema_version: int) -> bool:
    """Which record tables a snapshot schema version must contain (F14).

    v1 snapshots predate mandatory record files; v2 added task/entity tables but
    not the registry; v3 requires every exported table, including capabilities
    and resources. A missing required file/count is a validation error, never a
    silent zero-row restore.
    """
    if table in V5_TABLES:
        return schema_version >= 5
    if table in V4_TABLES:
        return schema_version >= 4
    if schema_version >= 3:
        return True
    if schema_version >= 2:
        return table not in V3_TABLES
    return False


class SnapshotError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        # F17-R5: failed-import cleanup ownership evidence.  ``import_snapshot``
        # fills these with what cleanup removed vs. deliberately preserved; the
        # ``(code, message)`` interface is unchanged and validation-only errors
        # simply keep the empty defaults.
        self.cleanup_report: dict | None = None
        self.cleanup_warnings: list[str] = []


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_relative(name: str) -> PurePosixPath:
    if not isinstance(name, str) or not name:
        raise SnapshotError("INVALID_REQUEST", "manifest contains an empty file name")
    if "\\" in name:
        raise SnapshotError("INVALID_REQUEST", f"backslash path separator refused: {name!r}")
    path = PurePosixPath(name)
    if path.is_absolute():
        raise SnapshotError("INVALID_REQUEST", f"absolute snapshot path refused: {name!r}")
    if any(part in ("..", "") for part in path.parts):
        raise SnapshotError("INVALID_REQUEST", f"path traversal refused: {name!r}")
    return path


def _ensure_no_symlinks(root: Path) -> None:
    for path in root.rglob("*"):
        if path.is_symlink():
            raise SnapshotError("INVALID_REQUEST", f"symlink in snapshot refused: {path.name}")


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                raise SnapshotError(
                    "INVALID_REQUEST", f"invalid JSONL in {path.name} line {line_number}"
                ) from None
            if not isinstance(row, dict):
                raise SnapshotError("INVALID_REQUEST", f"non-object record in {path.name}")
            rows.append(row)
    return rows


#: AC52-R7: ``jobs`` timestamp columns normalized to canonical UTC seconds on
#: import.  They are display/diagnostic values only — recovery causality uses the
#: durable ``completed_seq`` — but a restored value must be a comparable instant
#: rather than a string whose lexicographic order might be trusted.
JOBS_TIMESTAMP_FIELDS = ("created_at", "updated_at", "recovered_at")


def _canonical_utc_second(value: str) -> str:
    """Parse an RFC 3339 value and normalize it to ``YYYY-MM-DDTHH:MM:SSZ``."""
    text = value.strip()
    if text[-1:] in ("Z", "z"):
        text = text[:-1] + "+00:00"
    parsed = _dt.datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=_dt.timezone.utc)
    return (
        parsed.astimezone(_dt.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _normalize_jobs_records(rows: list[dict]) -> None:
    """Validate/normalize restored ``jobs`` records (AC52-R7).

    Completion order must be provable, so a present ``completed_seq`` must be a
    non-negative integer (absent/NULL means an older row that never claimed a
    completion order) and the timestamp columns are canonicalized to UTC
    seconds.  An unparseable timestamp on a row that takes part in a recovery
    association (the failed row itself or the target it names) is a logically
    corrupt snapshot and is refused; unrelated rows keep the historical value for
    display because they never prove recovery.
    """
    referenced = {
        row.get("recovered_by_job_id")
        for row in rows
        if row.get("recovered_by_job_id") is not None
    }
    for index, row in enumerate(rows):
        sequence = row.get("completed_seq")
        if sequence is not None and (
            isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0
        ):
            raise SnapshotError(
                "INVALID_REQUEST",
                f"jobs record {index} has an invalid completed_seq {sequence!r}",
            )
        participates = (
            row.get("recovered_by_job_id") is not None
            or row.get("recovered_at") is not None
            or row.get("job_id") in referenced
        )
        for field in JOBS_TIMESTAMP_FIELDS:
            value = row.get(field)
            if value is None:
                continue
            normalized = None
            if isinstance(value, str):
                try:
                    normalized = _canonical_utc_second(value)
                except ValueError:
                    normalized = None
            if normalized is None:
                if participates:
                    raise SnapshotError(
                        "INVALID_REQUEST",
                        f"jobs record {index} has an unparseable {field} involved "
                        "in a recovery association",
                    )
                continue
            row[field] = normalized


def validate_snapshot(source: str | Path) -> dict:
    root = Path(source)
    if root.is_symlink() or not root.is_dir():
        raise SnapshotError("INVALID_REQUEST", "snapshot path must be a real directory")
    _ensure_no_symlinks(root)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise SnapshotError("INVALID_REQUEST", "manifest.json missing")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        raise SnapshotError("INVALID_REQUEST", "manifest.json is not valid JSON") from None
    if manifest.get("format_version") not in SUPPORTED_FORMAT_VERSIONS:
        raise SnapshotError("UNSUPPORTED_SCHEMA", "unsupported snapshot format_version")
    if manifest.get("snapshot_schema_version") not in SUPPORTED_SNAPSHOT_SCHEMA_VERSIONS:
        raise SnapshotError("UNSUPPORTED_SCHEMA", "unsupported snapshot schema version")
    if manifest.get("contract_version") != contracts.CONTRACT_VERSION:
        raise SnapshotError("UNSUPPORTED_SCHEMA", "snapshot contract version mismatch")
    declared_files = manifest.get("files")
    if not isinstance(declared_files, dict) or not declared_files:
        raise SnapshotError("INVALID_REQUEST", "manifest declares no files")
    expected_payload = {
        f"{RECORDS_DIR}/{BACKEND_FILE}",
        f"{RECORDS_DIR}/versions.json".replace("records/versions.json", "config/versions.json"),
        *(f"{RECORDS_DIR}/{filename}" for filename in TABLES.values()),
    }
    for name, expected in declared_files.items():
        relative = _safe_relative(name)
        if not isinstance(expected, dict):
            raise SnapshotError("INVALID_REQUEST", f"bad manifest entry for {name}")
        path = root / relative
        if not path.is_file():
            raise SnapshotError("INVALID_REQUEST", f"declared file missing: {name}")
        if path.stat().st_size != expected.get("size"):
            raise SnapshotError("INVALID_REQUEST", f"size mismatch: {name}")
        if _sha256(path) != expected.get("sha256"):
            raise SnapshotError("INVALID_REQUEST", f"checksum mismatch: {name}")
    actual_payload = {
        path.relative_to(root).as_posix()
        for path in (root / RECORDS_DIR).rglob("*.jsonl")
    } | {
        (root / RECORDS_DIR / BACKEND_FILE).relative_to(root).as_posix(),
        (root / "config" / "versions.json").relative_to(root).as_posix(),
    }
    if set(declared_files) != actual_payload:
        raise SnapshotError("INVALID_REQUEST", "file set does not match the manifest")
    sums_path = root / "SHA256SUMS"
    if not sums_path.is_file():
        raise SnapshotError("INVALID_REQUEST", "SHA256SUMS missing")
    sums: dict[str, str] = {}
    for line in sums_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 2 or len(parts[0]) != 64:
            raise SnapshotError("INVALID_REQUEST", "malformed SHA256SUMS")
        sums[parts[1]] = parts[0]
    expected_sums = set(declared_files) | {"manifest.json"}
    if set(sums) != expected_sums:
        raise SnapshotError("INVALID_REQUEST", "SHA256SUMS does not cover the manifest file set")
    for name, digest in sums.items():
        if _sha256(root / _safe_relative(name)) != digest:
            raise SnapshotError("INVALID_REQUEST", f"SHA256SUMS mismatch: {name}")
    counts = manifest.get("counts") or {}
    snapshot_schema_version = int(manifest.get("snapshot_schema_version") or 1)
    for table, filename in TABLES.items():
        path = root / RECORDS_DIR / filename
        required = _table_required(table, snapshot_schema_version)
        if not path.is_file():
            if required:
                raise SnapshotError("INVALID_REQUEST", f"record file missing: {filename}")
            continue
        rows = _read_jsonl(path)
        if table not in counts:
            if required:
                raise SnapshotError("INVALID_REQUEST", f"record count missing for {table}")
            continue
        if int(counts.get(table, -1)) != len(rows):
            raise SnapshotError("INVALID_REQUEST", f"record count mismatch for {table}")
    backend = json.loads((root / RECORDS_DIR / BACKEND_FILE).read_text(encoding="utf-8"))
    for field in ("backend_instance_id", "restore_epoch", "watermark"):
        if field not in backend:
            raise SnapshotError("INVALID_REQUEST", f"backend record missing {field}")
    return {"manifest": manifest, "root": root, "backend": backend}


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}


def _insert_rows(conn: sqlite3.Connection, table: str, rows: list[dict]) -> int:
    if not rows:
        return 0
    known = _columns(conn, table)
    for index, row in enumerate(rows):
        unknown = set(row) - known
        if unknown:
            raise SnapshotError(
                "UNSUPPORTED_SCHEMA",
                f"{table} record {index} has unsupported columns {sorted(unknown)}",
            )
    columns = list(rows[0])
    placeholders = ",".join("?" for _ in columns)
    sql = f"INSERT INTO {table} ({','.join(columns)}) VALUES ({placeholders})"
    for row in rows:
        conn.execute(sql, [row.get(column) for column in columns])
    return len(rows)


def _rebuild_keyword_index(conn) -> int:
    from knowledge_weaver.retrieval.keyword import index_revision_in_transaction

    conn.execute("DELETE FROM claim_fts")
    rows = conn.execute(
        "SELECT revision_id, display_text, value_json FROM claim_revisions "
        "WHERE display_text IS NOT NULL AND display_text != ''"
    ).fetchall()
    count = 0
    for row in rows:
        value_text = row["value_json"] if isinstance(row["value_json"], str) else ""
        index_revision_in_transaction(
            conn, row["revision_id"], f"{row['display_text']} {value_text}"
        )
        count += 1
    return count


_IMPORTED_DB_FILES = (
    "knowledge.db",
    "knowledge.db-wal",
    "knowledge.db-shm",
    "knowledge.db-journal",
)

# F17-R5: the only staging entries this import may ever delete.  ``knowledge.db``
# is created by ``_create_database_file``; the journal siblings are the files
# SQLite itself creates next to it while the write phase runs.  Anything else in
# the staging directory was not created by this import and must be preserved.
_IMPORTED_DB_FILE_NAMES = frozenset(_IMPORTED_DB_FILES)

# F17: an import keeps the target directory open as a file descriptor for the
# whole build/activate/cleanup sequence, and builds the database in a *private
# staging directory* held open by a second fd.  SQLite can only open a path and
# macOS has no /proc/self/fd, so the write phase temporarily anchors the process
# cwd to that staging fd: every relative database/journal write resolves inside
# the staging inode.  The completed database is checkpointed, its exact inode is
# verified against the file descriptor this import created, and only then is it
# atomically renamed to ``target/knowledge.db`` through the target's held dir_fd.
# A replacement of the public target path — or of any ``knowledge.db`` entry on
# it — can therefore never redirect SQLite writes into another database.
#
# Disclosure: ``fchdir`` is process-global state.  ``_IMPORT_CWD_LOCK`` only
# serializes imports inside this process; it does NOT sandbox unrelated threads
# that use relative paths or change the cwd concurrently, and tampering by the
# same OS user is outside this threat model.
_IMPORT_CWD_LOCK = threading.Lock()


class _HeldTargetDir:
    """An import target directory held open by fd from creation to cleanup."""

    __slots__ = ("path", "fd", "created")

    def __init__(self, path: Path, fd: int, created: bool) -> None:
        self.path = path
        self.fd = fd
        self.created = created

    def close(self) -> None:
        try:
            os.close(self.fd)
        except OSError:  # pragma: no cover - defensive
            pass


class _StagingDir:
    """The private directory where the import database is built (F17).

    SQLite only ever opens ``knowledge.db`` inside this fd-held directory, never
    the public target path.  The created database file is kept open as
    ``db_file_fd`` so its exact ``(st_dev, st_ino)`` identity can be checked
    right before SQLite opens it and right before it is renamed into the target.
    """

    __slots__ = ("name", "path", "fd", "db_file_fd", "activated")

    def __init__(self, name: str, path: Path, fd: int) -> None:
        self.name = name
        self.path = path
        self.fd = fd
        self.db_file_fd: int | None = None
        self.activated = False

    def close(self) -> None:
        if self.db_file_fd is not None:
            try:
                os.close(self.db_file_fd)
            except OSError:  # pragma: no cover - defensive
                pass
            self.db_file_fd = None
        try:
            os.close(self.fd)
        except OSError:  # pragma: no cover - defensive
            pass


def _assert_safe_target(target: Path) -> None:
    """F17: refuse a target (or ancestor) that resolves through a symlink."""
    absolute = Path(os.path.abspath(str(target)))
    if target.is_symlink():
        raise SnapshotError("INVALID_REQUEST", "import target must not be a symlink")
    resolved = Path(os.path.realpath(str(target)))
    if resolved != absolute:
        raise SnapshotError(
            "INVALID_REQUEST",
            "import target resolves through a symlink or non-canonical ancestor; "
            "pass the real path so writes cannot be redirected",
        )
    probe = absolute
    while True:
        if probe.exists() and probe.is_symlink():
            raise SnapshotError("INVALID_REQUEST", f"symlinked path component refused: {probe}")
        if probe.parent == probe:
            break
        probe = probe.parent


def _verify_target_identity(handle: _HeldTargetDir, *, when: str) -> None:
    """F17: fail closed if the held directory and its path ever diverge.

    Checked before/after writing, before activation and around cleanup: the fd's
    ``(st_dev, st_ino)`` must still be the directory the target path resolves to,
    and no ancestor may have become a symlink.  A mismatch raises before the next
    path-based step; the fd-anchored writes themselves never reach another dir.
    """
    try:
        opened = os.fstat(handle.fd)
    except OSError as exc:  # pragma: no cover - defensive
        raise SnapshotError(
            "INVALID_REQUEST", f"import target directory handle is unusable ({when})"
        ) from exc
    if not stat.S_ISDIR(opened.st_mode):
        raise SnapshotError("INVALID_REQUEST", f"import target is not a real directory ({when})")
    try:
        current = os.lstat(str(handle.path))
    except OSError as exc:
        raise SnapshotError(
            "INVALID_REQUEST", f"import target disappeared during import ({when})"
        ) from exc
    if not stat.S_ISDIR(current.st_mode) or (opened.st_dev, opened.st_ino) != (
        current.st_dev,
        current.st_ino,
    ):
        raise SnapshotError(
            "INVALID_REQUEST",
            "import target identity changed during import "
            f"({when}); refusing to write through a replaced path",
        )
    _assert_safe_target(handle.path)


def _verify_staging_identity(
    handle: _HeldTargetDir, staging: _StagingDir, *, when: str
) -> None:
    """F17: the fd-held staging directory must still be the target's entry."""
    try:
        opened = os.fstat(staging.fd)
    except OSError as exc:  # pragma: no cover - defensive
        raise SnapshotError(
            "INVALID_REQUEST", f"import staging handle is unusable ({when})"
        ) from exc
    if not stat.S_ISDIR(opened.st_mode):
        raise SnapshotError(
            "INVALID_REQUEST", f"import staging is not a real directory ({when})"
        )
    try:
        current = os.stat(staging.name, dir_fd=handle.fd, follow_symlinks=False)
    except OSError as exc:
        raise SnapshotError(
            "INVALID_REQUEST", f"import staging disappeared during import ({when})"
        ) from exc
    if not stat.S_ISDIR(current.st_mode) or (opened.st_dev, opened.st_ino) != (
        current.st_dev,
        current.st_ino,
    ):
        raise SnapshotError(
            "INVALID_REQUEST",
            "import staging directory identity changed during import "
            f"({when}); refusing to let SQLite open a replaced path",
        )


def _verify_staging_database_file(staging: _StagingDir, *, when: str) -> None:
    """F17: the staging ``knowledge.db`` entry must still be the inode we created.

    ``db_file_fd`` is the file descriptor of the empty file created with
    ``O_CREAT|O_EXCL|O_NOFOLLOW``; if the directory entry was replaced (for
    example by a symlink to another database) the identity check fails closed
    *before* SQLite gets a chance to open the path.
    """
    if staging.db_file_fd is None:
        raise SnapshotError(
            "INVALID_REQUEST", f"import database file was not created ({when})"
        )
    try:
        held = os.fstat(staging.db_file_fd)
    except OSError as exc:  # pragma: no cover - defensive
        raise SnapshotError(
            "INVALID_REQUEST", f"import database handle is unusable ({when})"
        ) from exc
    try:
        entry = os.stat("knowledge.db", dir_fd=staging.fd, follow_symlinks=False)
    except OSError as exc:
        raise SnapshotError(
            "INVALID_REQUEST", f"import database file disappeared during import ({when})"
        ) from exc
    if (
        stat.S_ISLNK(entry.st_mode)
        or not stat.S_ISREG(entry.st_mode)
        or (entry.st_dev, entry.st_ino) != (held.st_dev, held.st_ino)
    ):
        raise SnapshotError(
            "INVALID_REQUEST",
            "import database file identity changed during import "
            f"({when}); a replaced database path is never handed to SQLite",
        )


def _create_target_dir(target: Path) -> _HeldTargetDir:
    """Create/open the target and keep its identity via an open dir fd (F17).

    The directory is opened with ``O_NOFOLLOW|O_DIRECTORY`` and its
    ``(st_dev, st_ino)`` identity is checked against the created path.  The fd is
    returned to the caller instead of being closed, so later writes are anchored
    to the directory inode we created rather than to re-resolved text.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    created = False
    if not target.exists():
        try:
            os.mkdir(target, 0o700)
        except FileExistsError:
            # The path appeared between the check and mkdir (possibly a symlink);
            # the O_NOFOLLOW open below decides instead of following it.
            pass
        else:
            created = True
    try:
        fd = os.open(
            str(target),
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_DIRECTORY", 0),
        )
    except OSError as exc:
        if created and not target.is_symlink():
            try:
                os.rmdir(str(target))
            except OSError:  # pragma: no cover - moved/replaced concurrently
                pass
        raise SnapshotError("INVALID_REQUEST", "import target is not a real directory") from exc
    handle = _HeldTargetDir(target, fd, created)
    try:
        _verify_target_identity(handle, when="creation")
        os.fchmod(fd, 0o700)
    except BaseException:
        handle.close()
        if created and not target.is_symlink():
            try:
                os.rmdir(str(target))
            except OSError:  # pragma: no cover - moved/replaced concurrently
                pass
        raise
    return handle


def _create_staging_dir(handle: _HeldTargetDir) -> _StagingDir:
    """Create the private staging directory inside the held target (F17).

    The random name is created with ``mkdir(..., dir_fd=target_fd)`` and opened
    ``O_NOFOLLOW|O_DIRECTORY`` relative to that fd; no path re-resolution outside
    the held target is involved.  Mode ``0700`` plus the random name keeps the
    staging out of ordinary same-user cleanup/path confusion (this is not an
    OS-user isolation boundary — see the module disclosure).
    """
    for _attempt in range(16):
        name = f".kw-import-staging-{uuid.uuid4().hex}"
        try:
            os.mkdir(name, 0o700, dir_fd=handle.fd)
        except FileExistsError:  # pragma: no cover - 128-bit name collision
            continue
        break
    else:  # pragma: no cover - defensive
        raise SnapshotError("INVALID_REQUEST", "cannot allocate a private import staging directory")
    try:
        fd = os.open(
            name,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_DIRECTORY", 0),
            dir_fd=handle.fd,
        )
    except OSError as exc:
        try:
            os.rmdir(name, dir_fd=handle.fd)
        except OSError:  # pragma: no cover - moved/replaced concurrently
            pass
        raise SnapshotError(
            "INVALID_REQUEST", "cannot open the private import staging directory"
        ) from exc
    staging = _StagingDir(name, handle.path / name, fd)
    try:
        _verify_staging_identity(handle, staging, when="staging creation")
        os.fchmod(fd, 0o700)
    except BaseException:
        staging.close()
        try:
            os.rmdir(name, dir_fd=handle.fd)
        except OSError:  # pragma: no cover - moved/replaced concurrently
            pass
        raise
    return staging


def _create_database_file(staging: _StagingDir) -> None:
    """Create ``knowledge.db`` inside the fd-held staging dir (F17).

    ``O_CREAT|O_EXCL|O_NOFOLLOW`` relative to the staging dir fd creates a fresh
    regular file (never a followed symlink).  Its fd is kept open on the staging
    handle so every later step can compare the directory entry's identity with
    the inode this import created.
    """
    try:
        file_fd = os.open(
            "knowledge.db",
            os.O_CREAT
            | os.O_EXCL
            | os.O_RDWR
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=staging.fd,
        )
    except FileExistsError as exc:
        raise SnapshotError(
            "INVALID_REQUEST", "import target already contains a database"
        ) from exc
    except OSError as exc:
        raise SnapshotError(
            "INVALID_REQUEST",
            f"cannot create the database file in the import target: {exc.strerror or exc}",
        ) from exc
    staging.db_file_fd = file_fd
    try:
        os.fchmod(file_fd, 0o600)
    except OSError:  # pragma: no cover - platform dependent
        pass
    _verify_staging_database_file(staging, when="database creation")


@contextlib.contextmanager
def _cwd_anchored_to(handle: _HeldTargetDir, staging: _StagingDir):
    """Run the SQLite write phase inside the held staging directory inode (F17).

    SQLite and its ``-wal``/``-shm`` siblings are opened through a relative path,
    so anchoring the cwd to the staging fd keeps every write in the private
    directory we created even if the public target path is later renamed or
    replaced by a symlink.  The database file entry itself was identity-checked
    immediately before this block; SQLite never opens the public target path.
    """
    _verify_staging_identity(handle, staging, when="before anchored writes")
    _verify_staging_database_file(staging, when="before anchored writes")
    with _IMPORT_CWD_LOCK:
        saved = os.open(".", os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fchdir(staging.fd)
            yield
        finally:
            try:
                os.fchdir(saved)
            finally:
                os.close(saved)


def _finalize_staged_database(handle: _HeldTargetDir, staging: _StagingDir) -> None:
    """Checkpoint the staging database and prove only our file remains (F17).

    SQLite runs in WAL mode, so a completed transaction may still have frames in
    ``knowledge.db-wal``.  The staging database is checkpointed and its journal
    siblings are removed relative to the staging fd *before* the single database
    file is renamed into the target; no journal can therefore be silently dropped
    or mistaken for the imported database.  Must run with the cwd anchored to the
    staging directory.
    """
    _verify_staging_identity(handle, staging, when="before checkpoint")
    _verify_staging_database_file(staging, when="before checkpoint")
    try:
        with sqlite3.connect("knowledge.db", timeout=10.0, isolation_level=None) as conn:
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    except sqlite3.Error as exc:
        raise SnapshotError(
            "INVALID_REQUEST", f"cannot checkpoint the imported database: {exc}"
        ) from exc
    for name in _IMPORTED_DB_FILES[1:]:
        try:
            os.unlink(name, dir_fd=staging.fd)
        except FileNotFoundError:
            continue
        except OSError as exc:  # pragma: no cover - defensive
            raise SnapshotError(
                "INVALID_REQUEST", f"cannot remove the staged database journal {name}: {exc}"
            ) from exc
    _verify_staging_database_file(staging, when="after checkpoint")
    try:
        remaining = sorted(os.listdir(staging.fd))
    except OSError as exc:  # pragma: no cover - defensive
        raise SnapshotError(
            "INVALID_REQUEST", f"cannot inspect the import staging directory: {exc}"
        ) from exc
    if remaining != ["knowledge.db"]:  # pragma: no cover - defensive
        raise SnapshotError(
            "INVALID_REQUEST",
            f"import staging directory is not a single finished database: {remaining}",
        )


def _verify_activated_database(handle: _HeldTargetDir, staging: _StagingDir) -> None:
    """F17: the target's ``knowledge.db`` must be exactly the inode we built."""
    if staging.db_file_fd is None:  # pragma: no cover - defensive
        raise SnapshotError("INVALID_REQUEST", "import database file was not created")
    try:
        held = os.fstat(staging.db_file_fd)
    except OSError as exc:  # pragma: no cover - defensive
        raise SnapshotError("INVALID_REQUEST", "import database handle is unusable") from exc
    try:
        entry = os.stat("knowledge.db", dir_fd=handle.fd, follow_symlinks=False)
    except OSError as exc:
        raise SnapshotError(
            "INVALID_REQUEST", "imported database is missing from the target"
        ) from exc
    if (
        stat.S_ISLNK(entry.st_mode)
        or not stat.S_ISREG(entry.st_mode)
        or (entry.st_dev, entry.st_ino) != (held.st_dev, held.st_ino)
    ):
        raise SnapshotError(
            "INVALID_REQUEST",
            "imported database identity changed during activation; "
            "refusing to publish a replaced database path",
        )


def _activate_staged_database(handle: _HeldTargetDir, staging: _StagingDir) -> None:
    """Atomically move the finished database to ``target/knowledge.db`` (F17).

    Both sides are directory fds, so neither a swapped public target path nor a
    replaced ``knowledge.db`` name can redirect the rename.  A name that appeared
    in the target after the empty-target check fails closed instead of being
    overwritten; a symlink planted there is never followed.
    """
    _verify_target_identity(handle, when="before database activation")
    _verify_staging_identity(handle, staging, when="before database activation")
    _verify_staging_database_file(staging, when="before database activation")
    try:
        os.stat("knowledge.db", dir_fd=handle.fd, follow_symlinks=False)
    except FileNotFoundError:
        pass
    else:
        raise SnapshotError(
            "INVALID_REQUEST",
            "import target gained a database file during import; "
            "refusing to overwrite or follow it",
        )
    try:
        os.rename("knowledge.db", "knowledge.db", src_dir_fd=staging.fd, dst_dir_fd=handle.fd)
    except OSError as exc:
        raise SnapshotError(
            "INVALID_REQUEST", f"cannot activate the imported database: {exc}"
        ) from exc
    staging.activated = True
    _verify_activated_database(handle, staging)


def _new_cleanup_report() -> dict:
    """F17-R5: what a failed/publishing import removed vs. deliberately kept."""
    return {"removed": [], "preserved": [], "warnings": []}


def _preserve_cleanup_entry(report: dict, *, location: str, name: str, reason: str) -> None:
    """Record that cleanup left an entry alone instead of deleting it (F17-R5)."""
    report["preserved"].append({"location": location, "name": name, "reason": reason})
    report["warnings"].append(
        f"import cleanup preserved {location} entry {name!r}: {reason}"
    )


def _database_entry_identity(
    *, dir_fd: int, location: str, held_fd: int | None
) -> tuple[bool, str | None]:
    """F17-R5: is the ``knowledge.db`` entry still the exact inode this import made?

    Ownership evidence is the ``(st_dev, st_ino)`` of the file descriptor this
    import created (``_StagingDir.db_file_fd``) compared with an ``lstat`` of the
    directory entry — never the fixed file name and never the directory fd alone.
    A missing entry, a symlink, a non-regular file or any other inode answers
    ``False`` with a reason cleanup can report.
    """
    if held_fd is None:  # pragma: no cover - defensive
        return False, f"{location} database identity is not held"
    try:
        held = os.fstat(held_fd)
    except OSError:  # pragma: no cover - defensive
        return False, f"{location} database handle is unusable"
    try:
        entry = os.stat("knowledge.db", dir_fd=dir_fd, follow_symlinks=False)
    except FileNotFoundError:
        return False, f"{location} database entry is missing"
    except OSError:  # pragma: no cover - defensive
        return False, f"{location} database entry is unreadable"
    if stat.S_ISLNK(entry.st_mode):
        return False, f"{location} database entry is a symlink"
    if not stat.S_ISREG(entry.st_mode):
        return False, f"{location} database entry is not a regular file"
    if (entry.st_dev, entry.st_ino) != (held.st_dev, held.st_ino):
        return False, f"{location} database entry identity changed"
    return True, None


def _remove_staging_dir(
    handle: _HeldTargetDir, staging: _StagingDir, report: dict
) -> None:
    """Remove only this import's private staging entries, via held fds (F17-R5).

    The staging directory is private, but an unrelated entry can still appear in
    it (fault injection, same-user interference).  Ownership of
    ``staging/knowledge.db`` is proven by comparing the held ``db_file_fd``
    ``(st_dev, st_ino)`` with an ``lstat`` of the directory entry — never by the
    fixed file name.  A replaced regular file, a symlink, a non-regular entry or
    a missing database is preserved and reported; the SQLite
    ``-wal``/``-shm``/``-journal`` siblings are removed only together with an
    owned database (they may belong to the replacement).  Every other entry is
    preserved and reported instead of being enumerated away; the directory
    itself is removed only while the held fd identity still matches its target
    entry and it is empty.
    """
    try:
        entries = sorted(os.listdir(staging.fd))
    except OSError as exc:  # pragma: no cover - defensive
        report["warnings"].append(
            f"import cleanup cannot inspect the staging directory: {exc.strerror or exc}"
        )
        return
    # F17-R5: the fixed name is not ownership.  Ask the held fd whether the
    # staging database entry is still exactly the inode this import created.
    db_owned, db_reason = _database_entry_identity(
        dir_fd=staging.fd, location="staging", held_fd=staging.db_file_fd
    )
    if "knowledge.db" in entries:
        if not db_owned:
            _preserve_cleanup_entry(
                report,
                location="staging",
                name="knowledge.db",
                reason=db_reason or "ownership could not be verified",
            )
        else:
            try:
                os.unlink("knowledge.db", dir_fd=staging.fd)
            except FileNotFoundError:  # pragma: no cover - moved/replaced concurrently
                pass
            except OSError as exc:
                # The database stays, so its journal siblings may still belong
                # to it; do not remove them either.
                db_owned = False
                _preserve_cleanup_entry(
                    report,
                    location="staging",
                    name="knowledge.db",
                    reason=f"entry could not be removed ({exc.strerror or exc})",
                )
            else:
                report["removed"].append({"location": "staging", "name": "knowledge.db"})
    for name in entries:
        if name == "knowledge.db":
            continue
        if name not in _IMPORTED_DB_FILE_NAMES:
            _preserve_cleanup_entry(
                report,
                location="staging",
                name=name,
                reason="entry was not created by this import",
            )
            continue
        if not db_owned:
            _preserve_cleanup_entry(
                report,
                location="staging",
                name=name,
                reason="staging database ownership was not verified",
            )
            continue
        try:
            os.unlink(name, dir_fd=staging.fd)
        except FileNotFoundError:  # pragma: no cover - moved/replaced concurrently
            continue
        except OSError as exc:
            _preserve_cleanup_entry(
                report,
                location="staging",
                name=name,
                reason=f"entry could not be removed ({exc.strerror or exc})",
            )
            continue
        report["removed"].append({"location": "staging", "name": name})
    try:
        opened = os.fstat(staging.fd)
        current = os.stat(staging.name, dir_fd=handle.fd, follow_symlinks=False)
        empty = not os.listdir(staging.fd)
        if (
            empty
            and stat.S_ISDIR(current.st_mode)
            and (opened.st_dev, opened.st_ino) == (current.st_dev, current.st_ino)
        ):
            os.rmdir(staging.name, dir_fd=handle.fd)
    except OSError:  # pragma: no cover - moved/replaced concurrently
        pass


def _remove_owned_target_database(
    handle: _HeldTargetDir, staging: _StagingDir, report: dict
) -> None:
    """Unlink ``target/knowledge.db`` only while it is the inode we activated.

    After activation the public entry must still be the exact inode held by
    ``db_file_fd``.  A replacement (regular file, symlink, another inode) or a
    missing entry is not ours: keep it and report it instead of deleting it by
    fixed name.  The target ``-wal``/``-shm``/``-journal`` siblings are removed
    only together with a verified owned database — if ownership cannot be proven
    they are preserved too, since they may belong to the replacement.
    """
    owned, reason = _database_entry_identity(
        dir_fd=handle.fd, location="target", held_fd=staging.db_file_fd
    )
    if not owned:
        _preserve_cleanup_entry(
            report,
            location="target",
            name="knowledge.db",
            reason=reason or "ownership could not be verified",
        )
        return
    for name in _IMPORTED_DB_FILES:
        try:
            os.unlink(name, dir_fd=handle.fd)
        except FileNotFoundError:
            continue
        except OSError as exc:
            report["warnings"].append(
                f"import cleanup could not remove target entry {name!r}: "
                f"{exc.strerror or exc}"
            )
            continue
        report["removed"].append({"location": "target", "name": name})


def _cleanup_created_target(
    handle: _HeldTargetDir, staging: _StagingDir | None
) -> dict:
    """Remove only this import's files, relative to the held fds (F17-R5).

    The private staging directory and the entries this import created in it are
    removed through the fd, so a swapped public path can never make cleanup
    delete another directory's contents.  An activated database is unlinked
    through the target fd only while the held ``db_file_fd`` identity proves the
    entry is still ours; replacements and unrelated entries are preserved and
    reported.  The public target directory itself is removed only while its path
    still resolves to the inode held by the fd, it is empty, and this import
    created it.
    """
    report = _new_cleanup_report()
    try:
        _verify_target_identity(handle, when="before cleanup")
    except SnapshotError:
        # The identity change that aborted the import is exactly what cleanup
        # must survive; the fd-relative cleanups below stay safe regardless.
        pass
    if staging is not None:
        _remove_staging_dir(handle, staging, report)
        if staging.activated:
            _remove_owned_target_database(handle, staging, report)
    if not handle.created:
        return report
    try:
        opened = os.fstat(handle.fd)
        current = os.lstat(str(handle.path))
        empty = not os.listdir(handle.fd)
    except OSError:
        return report
    if not empty or not stat.S_ISDIR(current.st_mode) or (opened.st_dev, opened.st_ino) != (
        current.st_dev,
        current.st_ino,
    ):
        return report
    try:
        os.rmdir(str(handle.path))
    except OSError:  # pragma: no cover - non-empty/moved concurrently
        pass
    return report


def _attach_cleanup_report(exc: BaseException, report: dict) -> None:
    """Expose failed-import cleanup ownership evidence on the original exception.

    The original exception is re-raised unchanged (type, message, ``code``); the
    report is attached as plain attributes and preserved-entry warnings are also
    logged.  That makes the "kept the replacement / foreign entry" semantic
    testable without swallowing or rewrapping the import error.
    """
    warnings = list(report["warnings"])
    for warning in warnings:
        logger.warning("%s", warning)
    try:
        exc.cleanup_report = report  # type: ignore[attr-defined]
        exc.cleanup_warnings = warnings  # type: ignore[attr-defined]
    except (AttributeError, TypeError):  # pragma: no cover - exotic exception types
        pass


def _apply_restore_expiry(conn, *, now: str) -> dict:
    """F13: apply every knowable expiry/tombstone before the restore is usable.

    A snapshot keeps its lifecycle anchors, so an old snapshot can contain details
    whose 30/180/365 window has already closed. Import must apply the current
    retention clock while still in the isolated staging transaction (before
    ``restore_in_progress`` is cleared), not open the restored database first.
    """
    from knowledge_weaver.lifecycle.retention import (
        _due_lifecycle_records,
        _expire_claim_revision,
        _expire_event_part,
        _parse_clock,
        _tombstone,
    )

    # F09b: compare expiry instants, not timestamp text, so imported expiry
    # spellings (offset or Z) cannot expire early/late during restore either.
    due = _due_lifecycle_records(conn, _parse_clock(now))
    counts = {"claim_revisions": 0, "event_parts": 0, "tombstoned": 0}
    for record in due:
        generation = 0
        if record["subject_kind"] == "claim_revision":
            row = conn.execute(
                "SELECT claim_id FROM claim_revisions WHERE revision_id = ?",
                (record["subject_id"],),
            ).fetchone()
            if row is not None:
                current = conn.execute(
                    "SELECT projection_generation FROM current_claims WHERE claim_id = ?",
                    (row["claim_id"],),
                ).fetchone()
                generation = int(current["projection_generation"]) if current is not None else 0
                _expire_claim_revision(conn, record["subject_id"], now)
            counts["claim_revisions"] += 1
        elif record["subject_kind"] == "event_part":
            _expire_event_part(conn, record["subject_id"], now)
            counts["event_parts"] += 1
        _tombstone(
            conn,
            record,
            generation=max(int(record["generation"]) + 1, generation),
            now=now,
        )
        counts["tombstoned"] += 1
    counts["unclassified"] = int(
        conn.execute(
            "SELECT COUNT(*) FROM lifecycle_records WHERE state = 'active' AND expires_at IS NULL"
        ).fetchone()[0]
    )
    return counts


def _validate_referential_integrity(conn) -> dict:
    """F14: full-table reference reconciliation after restore.

    Counts alone are not enough — a snapshot could restore rows that point at
    missing logical objects. Import fails closed on dangling references.

    AC52-R5: the durable ``jobs.recovered_by_job_id`` recovery association is a
    reference too.  A non-NULL id is not proof that a failure was recovered: it
    must name an existing, different, same-operation operation-ledger job that
    actually succeeded after the failure was recorded (see
    ``store.status.VALID_RECOVERY_LINK_SQL``).  A snapshot whose recovery links
    violate that contract is logically corrupt — restoring it would silently
    turn an unrecovered failure green — so it is refused with the module's usual
    ``SnapshotError('INVALID_REQUEST', ...)`` integrity failure.

    AC52-R7: "after the failure was recorded" is proven by the durable
    ``completed_seq`` (strictly greater), so a snapshot whose recovery link lacks
    a completion order (a pre-round-7 snapshot) or whose order does not prove the
    success finished later is refused here for the same reason.  Timestamp text
    cannot substitute for it.
    """
    checks = {
        "event_revisions_without_event": (
            "SELECT COUNT(*) FROM event_revisions r "
            "LEFT JOIN events e ON e.event_id = r.event_id WHERE e.event_id IS NULL"
        ),
        "event_parts_without_revision": (
            "SELECT COUNT(*) FROM event_parts p "
            "LEFT JOIN event_revisions r ON r.event_id = p.event_id AND r.revision = p.revision "
            "WHERE r.event_id IS NULL"
        ),
        "claim_revisions_without_claim": (
            "SELECT COUNT(*) FROM claim_revisions r "
            "LEFT JOIN claims c ON c.claim_id = r.claim_id WHERE c.claim_id IS NULL"
        ),
        "evidence_links_without_revision": (
            "SELECT COUNT(*) FROM evidence_links l "
            "LEFT JOIN claim_revisions r ON r.revision_id = l.revision_id "
            "WHERE r.revision_id IS NULL"
        ),
        "current_claims_without_claim": (
            "SELECT COUNT(*) FROM current_claims c "
            "LEFT JOIN claims l ON l.claim_id = c.claim_id WHERE l.claim_id IS NULL"
        ),
        "resources_without_capability": (
            "SELECT COUNT(*) FROM resources r "
            "LEFT JOIN capabilities c ON c.capability_id = r.recipe_id "
            "WHERE c.capability_id IS NULL"
        ),
        # AC52-R5: every present recovery association (either column) must be a
        # complete, valid, completion-causal recovery of a recorded operation
        # failure.  A dangling/self/other-operation/non-succeeded/incomplete
        # association is a logically corrupt snapshot, not a recovery proof.
        "invalid_recovery_links": (
            "SELECT COUNT(*) FROM jobs j "
            "WHERE (j.recovered_by_job_id IS NOT NULL OR j.recovered_at IS NOT NULL) "
            f"AND NOT ({VALID_RECOVERY_LINK_SQL})"
        ),
    }
    checks["invalid_extraction_upgrade_request"] = f"""SELECT COUNT(*) FROM jobs n LEFT JOIN jobs p ON p.job_id=n.repair_of_job_id
        WHERE n.repair_of_job_id IS NOT NULL AND (p.job_id IS NULL OR n.kind<>'extract' OR p.kind<>'extract'
        OR p.state<>'failed' OR p.last_error_code IS NULL OR n.input_key<>p.input_key OR n.input_version<=p.input_version
        OR NOT {upgrade_sql('n.prompt_version','p.prompt_version')}
        OR n.model_version IS NOT p.model_version OR n.policy_version IS NOT p.policy_version
        OR (n.state='succeeded' AND p.recovered_by_job_id IS NOT n.job_id))"""
    checks["invalid_source_semantic_exclusion"] = """SELECT COUNT(*) FROM source_semantic_exclusions x
        LEFT JOIN event_parts p ON p.event_id=x.event_id AND p.revision=x.event_revision AND p.part_index=x.part_index
        LEFT JOIN events e ON e.event_id=x.event_id
        WHERE p.event_id IS NULL OR p.body_sha256<>x.body_sha256 OR e.role<>'tool' OR e.origin_kind<>'tool_execution'"""
    from knowledge_weaver.store.source_duplicates import VALID_DUPLICATE_RESOLUTION_SQL
    checks["invalid_native_duplicate_resolution"] = (
        "SELECT COUNT(*) FROM source_duplicate_resolutions d LEFT JOIN jobs j ON j.job_id=d.failed_job_id "
        "WHERE j.job_id IS NULL OR NOT ("+VALID_DUPLICATE_RESOLUTION_SQL+")")
    failures = {
        name: int(conn.execute(sql).fetchone()[0])
        for name, sql in checks.items()
        if int(conn.execute(sql).fetchone()[0])
    }
    if failures:
        raise SnapshotError(
            "INVALID_REQUEST", f"snapshot referential integrity failed: {failures}"
        )
    return {
        "tables": {
            table: int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
            for table in TABLES
        },
    }


def import_snapshot(
    source: str | Path,
    target_dir: str | Path,
    *,
    bootstrap_import: bool = False,
) -> dict:
    validated = validate_snapshot(source)
    manifest = validated["manifest"]
    # The cwd is temporarily anchored to the private staging dir fd below, so
    # every path used afterwards must be absolute (a relative snapshot path would
    # otherwise resolve inside the import staging directory).
    root = Path(os.path.abspath(str(validated["root"])))
    backend = validated["backend"]
    # F17: identity checks and fd-relative writes need one canonical absolute
    # path; the target dir fd and (below) the private staging dir fd anchor the
    # whole build/activate/cleanup sequence.
    target = Path(os.path.abspath(str(target_dir)))
    if target.exists() and (not target.is_dir() or any(target.iterdir())):
        raise SnapshotError("INVALID_REQUEST", "import target must be a new or empty directory")
    _assert_safe_target(target)
    handle = _create_target_dir(target)
    staging: _StagingDir | None = None
    db_path = target / "knowledge.db"
    new_backend_id = "kw:" + uuid.uuid4().hex
    new_epoch = "epoch:" + uuid.uuid4().hex[:24]
    counts: dict[str, int] = {}
    try:
        _verify_target_identity(handle, when="before database creation")
        staging = _create_staging_dir(handle)
        _create_database_file(staging)
        with _cwd_anchored_to(handle, staging):
            _verify_target_identity(handle, when="after database creation")
            # F17: SQLite may only open the exact inode this import created;
            # the identity check runs immediately before the path-based open.
            _verify_staging_database_file(staging, when="before SQLite open")
            store = Store(
                "knowledge.db", backend_instance_id=new_backend_id, restore_epoch=new_epoch
            )
            with store.transaction() as conn:
                conn.execute(
                    "UPDATE backend_meta SET restore_in_progress = 1, backend_instance_id = ?, "
                    "restore_epoch = ?, schema_version = ? WHERE id = 1",
                    (new_backend_id, new_epoch, int(manifest.get("snapshot_schema_version") or 1)),
                )
                schema_version = int(manifest.get("snapshot_schema_version") or 1)
                for table in IMPORT_ORDER:
                    filename = TABLES[table]
                    path = root / RECORDS_DIR / filename
                    if not path.is_file():
                        if _table_required(table, schema_version):
                            raise SnapshotError(
                                "INVALID_REQUEST", f"record file missing: {filename}"
                            )
                        continue
                    rows = _read_jsonl(path)
                    if table == "jobs":
                        for row in rows:
                            if row.get("state") == "running":
                                row["state"] = "retry_wait"
                                row["lease_token"] = None
                                row["lease_owner"] = None
                        # AC52-R7: validate the completion-order columns and
                        # canonicalize the timestamp columns before insert; a
                        # link without a strictly later completion order is then
                        # refused by the structural integrity check below (the
                        # staging transaction rolls back, nothing is published).
                        _normalize_jobs_records(rows)
                    counts[table] = _insert_rows(conn, table, rows)
                # AC52-R7: the ledger sequence is local, not portable.  Seed it
                # above every restored completion order so an operation success
                # after the restore still completes strictly later and can recover
                # a failure the snapshot proved earlier.  Leaving it at 0 would
                # make the next completion tie with restored rows.
                conn.execute(
                    "UPDATE backend_meta SET ledger_seq = "
                    "(SELECT COALESCE(MAX(completed_seq), 0) FROM jobs) WHERE id = 1"
                )
                # F13: enforce the current retention clock while still isolated and
                # before the restored database can serve reads.
                restore_expiry = _apply_restore_expiry(conn, now=utc_now())
                integrity = _validate_referential_integrity(conn)
                keyword_indexed = _rebuild_keyword_index(conn)
                watermark = int(manifest.get("watermark") or 0)
                if watermark > 0:
                    conn.execute(
                        "INSERT OR REPLACE INTO watermark_log (seq, kind, ref, created_at) "
                        "VALUES (?, 'restore', 'snapshot-watermark', ?)",
                        (watermark, utc_now()),
                    )
                _verify_target_identity(handle, when="before activation")
                _verify_staging_database_file(staging, when="before activation")
                conn.execute("UPDATE backend_meta SET restore_in_progress = 0 WHERE id = 1")
                if bootstrap_import:
                    now = utc_now()
                    event_ids = [
                        row["event_id"]
                        for row in conn.execute("SELECT DISTINCT event_id FROM event_parts").fetchall()
                    ]
                    for event_id in event_ids:
                        parts = conn.execute(
                            "SELECT revision, part_index FROM event_parts WHERE event_id = ?",
                            (event_id,),
                        ).fetchall()
                        for part in parts:
                            subject = f"{event_id}#{part['revision']}@{part['part_index']}"
                            conn.execute(
                                "INSERT OR IGNORE INTO lifecycle_records "
                                "(subject_kind, subject_id, class, business_anchor, expires_at, state, generation, reason, updated_at) "
                                "VALUES ('event_part', ?, 'temporary_detail', NULL, NULL, 'bootstrap_pending', 0, "
                                "'imported-history-bootstrap', ?)",
                                (subject, now),
                            )
            _verify_target_identity(handle, when="after activation")
            _verify_staging_database_file(staging, when="after activation")
            _finalize_staged_database(handle, staging)
        # F17: publish the finished, checkpointed database atomically through
        # the target dir fd; SQLite never re-opened the public target path.
        _activate_staged_database(handle, staging)
        _verify_target_identity(handle, when="after database activation")
        success_cleanup = _new_cleanup_report()
        _remove_staging_dir(handle, staging, success_cleanup)
        for warning in success_cleanup["warnings"]:
            logger.warning("%s", warning)
        report = {
            "target_db": str(db_path),
            "backend_instance_id": new_backend_id,
            "source_backend_instance_id": backend["backend_instance_id"],
            "restore_epoch": new_epoch,
            "previous_restore_epoch": backend["restore_epoch"],
            "watermark": int(manifest.get("watermark") or 0),
            "contract_version": manifest.get("contract_version"),
            "snapshot_schema_version": manifest.get("snapshot_schema_version"),
            "counts": counts,
            "counts_by_table": integrity["tables"],
            "referential_integrity": "ok",
            "restore_expiry": restore_expiry,
            "keyword_indexed": keyword_indexed,
            "bootstrap_pending": bool(bootstrap_import),
            "rebuild_required": {"vectors": True, "fts": False},
            "limitations": list(manifest.get("limitations") or []),
            "validated_files": len(manifest.get("files") or {}) + 2,
            # F17-R5: empty for a clean publish; non-empty only if a foreign
            # staging entry had to be preserved during cleanup.
            "cleanup_warnings": list(success_cleanup["warnings"]),
        }
        return report
    except BaseException as exc:
        cleanup_report = _cleanup_created_target(handle, staging)
        _attach_cleanup_report(exc, cleanup_report)
        raise
    finally:
        if staging is not None:
            staging.close()
        handle.close()
