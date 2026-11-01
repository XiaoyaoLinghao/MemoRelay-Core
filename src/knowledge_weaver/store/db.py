"""SQLite store for the KW 2.0 unified memory core.

Design constraints (engineering §4-5):

* one local SQLite database in WAL mode on local disk, ``synchronous=FULL`` for
  durable receipts; a single writer is serialized by an in-process lock;
* source evidence is append-only: events, revisions and parts; receipts are
  idempotent per ``(event_id, revision, part_index)``; watermarks increase
  monotonically with every durable write;
* withheld bodies are never stored; the store never talks to the network.

The module is standard-library only.
"""

from __future__ import annotations

import contextlib
import datetime as _dt
import os
import sqlite3
import threading
import uuid
from pathlib import Path
from typing import Any, Iterator

from importlib import resources

SCHEMA_VERSION = 1
DEFAULT_RESTORE_EPOCH = "epoch:1"

_WRITE_LOCK = threading.RLock()


def utc_now() -> str:
    """RFC 3339 UTC timestamp with a fixed precision."""
    return (
        _dt.datetime.now(_dt.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _migration_files() -> list[str]:
    root = resources.files("knowledge_weaver.migrations")
    names = [entry.name for entry in root.iterdir() if entry.name.endswith(".sql")]
    return sorted(names)


def _read_migration(name: str) -> str:
    return resources.files("knowledge_weaver.migrations").joinpath(name).read_text(encoding="utf-8")


#: Idempotent compatibility migrations for databases created by an earlier
#: schema.  SQLite has no ``ADD COLUMN IF NOT EXISTS``, so a plain ``ALTER TABLE``
#: script would break the invariant that a re-run unrecorded migration (crash
#: between the DDL and the ``schema_migrations`` insert) is harmless.  Each
#: column below is guarded by ``PRAGMA table_info`` instead: the first open of an
#: old database adds it, every later open is a no-op, and a concurrent opener
#: that lost the race is tolerated as a duplicate-column error.
COMPATIBILITY_COLUMNS: tuple[tuple[str, str, str], ...] = (
    # AC-52 round 5: explicit completion-causal failure recovery association on
    # the durable operation-ledger row (see ``store.status.record_operation_finish``).
    ("jobs", "recovered_by_job_id", "TEXT NULL"),
    ("jobs", "recovered_at", "TEXT NULL"),
    # AC-52 round 7: durable completion order.  A monotonically increasing ledger
    # sequence is stamped on the ledger row inside the same transaction that
    # completes the operation, so "did this success finish after that failure was
    # recorded?" is a persisted fact instead of a second-granularity string
    # comparison (which cannot order equal timestamps or mixed UTC offsets).
    ("jobs", "completed_seq", "INTEGER NULL"),
    ("backend_meta", "ledger_seq", "INTEGER NOT NULL DEFAULT 0"),
)


def _ensure_column(
    connection: sqlite3.Connection, table: str, column: str, declaration: str
) -> bool:
    """Add one nullable compatibility column if the table does not have it yet."""
    existing = {row[1] for row in connection.execute(f"PRAGMA table_info({table})")}
    if column in existing:
        return False
    try:
        connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {declaration}")
    except sqlite3.OperationalError as exc:  # pragma: no cover - concurrent opener
        if "duplicate column name" not in str(exc).lower():
            raise
        return False
    return True


def _backfill_legacy_recovery_completion_order(connection: sqlite3.Connection) -> None:
    """One-time AC52-R7 upgrade for recovery links written before completion order.

    A pre-round-7 database can already contain the explicit
    ``recovered_by_job_id`` association (written by the round-5/6 code) while
    neither side has a ``completed_seq``.  Dropping those links would resurrect
    failures that a real operation success already recovered, so the upgrade
    records the order the old code had already proven instead of guessing:

    * the failed row existed when the recovering success committed — that is
      exactly why the old transaction linked it — so it completed first and is
      stamped ``0``;
    * the recorded target success gets the next ledger sequence values (taken
      from / written back to ``backend_meta.ledger_seq``), so the strict
      ``completed_seq > completed_seq`` causality stays true.

    The backfill is deliberately conservative (AC52-R6 fail-safe): a link whose
    target is missing, a self link, an incomplete link, a non-succeeded target or
    a target that is not the same operation-ledger job is left with
    ``completed_seq = NULL`` on both sides.  :data:`store.status.VALID_RECOVERY_LINK_SQL`
    then keeps that failure visible; a repair never hides a failure by guesswork.

    It runs only on the open that first adds the ``completed_seq`` column, so it
    can never reinterpret rows created after the upgrade.  A crash between the
    ``ALTER TABLE`` and this backfill leaves the old links unproven (fail-safe
    visible) instead of wedging the next open — the column already exists then,
    and the upgrade is never guessed a second time.
    """
    row = connection.execute("SELECT ledger_seq FROM backend_meta WHERE id = 1").fetchone()
    seq = int(row["ledger_seq"]) if row is not None and row["ledger_seq"] is not None else 0
    start = seq
    links = connection.execute(
        "SELECT job_id, kind, state, model_version, recovered_by_job_id, recovered_at "
        "FROM jobs WHERE recovered_by_job_id IS NOT NULL AND completed_seq IS NULL"
    ).fetchall()
    assigned: dict[str, int] = {}
    for link in links:
        if link["state"] != "failed" or link["recovered_at"] is None:
            continue
        target = connection.execute(
            "SELECT job_id, kind, state, model_version, completed_seq FROM jobs WHERE job_id = ?",
            (link["recovered_by_job_id"],),
        ).fetchone()
        if target is None or target["job_id"] == link["job_id"]:
            continue
        if target["state"] != "succeeded" or target["kind"] != link["kind"]:
            continue
        if (
            link["model_version"] is None
            or target["model_version"] != link["model_version"]
        ):
            continue
        target_id = target["job_id"]
        if target["completed_seq"] is None:
            seq += 1
            assigned[target_id] = seq
            connection.execute(
                "UPDATE jobs SET completed_seq = ? WHERE job_id = ?", (seq, target_id)
            )
        elif target_id not in assigned:
            # A pre-existing completion order that this backfill did not write
            # cannot be used to prove the legacy link; leave it visible.
            continue
        connection.execute(
            "UPDATE jobs SET completed_seq = 0 WHERE job_id = ?", (link["job_id"],)
        )
    if seq != start:
        connection.execute("UPDATE backend_meta SET ledger_seq = ? WHERE id = 1", (seq,))


def _apply_compatibility_migrations(connection: sqlite3.Connection) -> None:
    """Idempotently bring a database created by an older schema up to date."""
    added: set[tuple[str, str]] = set()
    for table, column, declaration in COMPATIBILITY_COLUMNS:
        if _ensure_column(connection, table, column, declaration):
            added.add((table, column))
    if ("jobs", "completed_seq") in added:
        # AC52-R7: only the open that adds the new column can repair the old
        # association; later opens are strictly no-ops.
        _backfill_legacy_recovery_completion_order(connection)


class Store:
    """A local KW database handle.

    Connections are short-lived (opened per read/transaction) so web-handler
    threads never share a connection. Writes take a process-wide lock and use
    ``BEGIN IMMEDIATE`` so the durability rules ("no cursor/outbox write without
    a committed receipt") hold even with several request handlers.
    """

    def __init__(
        self,
        db_path: str | Path,
        *,
        backend_instance_id: str | None = None,
        restore_epoch: str = DEFAULT_RESTORE_EPOCH,
        initialize: bool = True,
    ) -> None:
        self.db_path = Path(db_path)
        self.backend_instance_id = backend_instance_id
        self.restore_epoch = restore_epoch
        if initialize:
            self.initialize()

    # -- connection ---------------------------------------------------------
    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            str(self.db_path), timeout=10.0, isolation_level=None
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=10000")
        # Zero freed row content so purged secret-derived payloads cannot linger
        # in free pages (SEC-02). Paired with purge_deleted_bytes() for WAL pages.
        connection.execute("PRAGMA secure_delete=ON")
        return connection

    @contextlib.contextmanager
    def read(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            yield connection
        finally:
            connection.close()

    @contextlib.contextmanager
    def transaction(self, write: bool = True) -> Iterator[sqlite3.Connection]:
        with _WRITE_LOCK:
            connection = self._connect()
            try:
                connection.execute("BEGIN IMMEDIATE" if write else "BEGIN")
                try:
                    yield connection
                except BaseException:
                    connection.execute("ROLLBACK")
                    raise
                connection.execute("COMMIT")
            finally:
                connection.close()

    # -- lifecycle ----------------------------------------------------------
    def initialize(self) -> None:
        parent = self.db_path.parent
        parent.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(parent, 0o700)
        except OSError:  # pragma: no cover - platform dependent
            pass
        with _WRITE_LOCK:
            connection = self._connect()
            try:
                connection.execute(
                    "CREATE TABLE IF NOT EXISTS schema_migrations ("
                    "version INTEGER PRIMARY KEY, name TEXT NOT NULL, applied_at TEXT NOT NULL)"
                )
                applied = {
                    row["version"]
                    for row in connection.execute("SELECT version FROM schema_migrations")
                }
                for name in _migration_files():
                    version = int(name.split("_", 1)[0])
                    if version in applied:
                        continue
                    # Migration SQL is idempotent (IF NOT EXISTS) and executescript
                    # manages its own transactions; a crash mid-way is repaired by
                    # re-running the unrecorded migration.
                    connection.executescript(_read_migration(name))
                    connection.execute(
                        "INSERT OR REPLACE INTO schema_migrations (version, name, applied_at) "
                        "VALUES (?, ?, ?)",
                        (version, name, utc_now()),
                    )
                # Compatibility migrations are idempotent and applied on every
                # open, so an old database gains the columns exactly once and a
                # crash between DDL and bookkeeping cannot wedge the next open.
                _apply_compatibility_migrations(connection)
                row = connection.execute("SELECT * FROM backend_meta WHERE id = 1").fetchone()
                if row is None:
                    instance_id = self.backend_instance_id or ("kw:" + uuid.uuid4().hex)
                    connection.execute(
                        "INSERT INTO backend_meta "
                        "(id, backend_instance_id, restore_epoch, schema_version, restore_in_progress, created_at) "
                        "VALUES (1, ?, ?, ?, 0, ?)",
                        (instance_id, self.restore_epoch, SCHEMA_VERSION, utc_now()),
                    )
                    self.backend_instance_id = instance_id
                else:
                    self.backend_instance_id = row["backend_instance_id"]
                    self.restore_epoch = row["restore_epoch"]
            finally:
                connection.close()
        try:
            os.chmod(self.db_path, 0o600)
        except OSError:  # pragma: no cover - platform dependent
            pass

    # -- meta / watermark ---------------------------------------------------
    def meta(self, connection: sqlite3.Connection | None = None) -> dict:
        def _read(conn: sqlite3.Connection) -> dict:
            row = conn.execute("SELECT * FROM backend_meta WHERE id = 1").fetchone()
            watermark = conn.execute("SELECT COALESCE(MAX(seq), 0) AS w FROM watermark_log").fetchone()["w"]
            return {
                "backend_instance_id": row["backend_instance_id"],
                "restore_epoch": row["restore_epoch"],
                "schema_version": row["schema_version"],
                "restore_in_progress": bool(row["restore_in_progress"]),
                "watermark": int(watermark),
            }

        if connection is not None:
            return _read(connection)
        with self.read() as conn:
            return _read(conn)

    def watermark(self, connection: sqlite3.Connection | None = None) -> int:
        if connection is not None:
            return int(
                connection.execute("SELECT COALESCE(MAX(seq), 0) AS w FROM watermark_log").fetchone()["w"]
            )
        with self.read() as conn:
            return int(
                conn.execute("SELECT COALESCE(MAX(seq), 0) AS w FROM watermark_log").fetchone()["w"]
            )

    def bump_watermark(self, connection: sqlite3.Connection, kind: str, ref: str) -> int:
        cursor = connection.execute(
            "INSERT INTO watermark_log (kind, ref, created_at) VALUES (?, ?, ?)",
            (kind, ref, utc_now()),
        )
        return int(cursor.lastrowid)

    def set_restore_in_progress(self, value: bool) -> None:
        with self.transaction() as conn:
            conn.execute(
                "UPDATE backend_meta SET restore_in_progress = ? WHERE id = 1",
                (1 if value else 0,),
            )

    def quick_check(self) -> str:
        try:
            with self.read() as conn:
                row = conn.execute("PRAGMA quick_check").fetchone()
                return "ok" if row is not None and row[0] == "ok" else "error"
        except sqlite3.Error:
            return "error"

    def purge_deleted_bytes(self) -> dict:
        """Zero+compact deleted page content after a redaction/retention purge.

        SQLite keeps old row payloads in free pages and WAL frames until they are
        overwritten or checkpointed, so a logical update to ``NULL``/``''`` is not
        enough to guarantee a secret is gone from disk. ``secure_delete`` zeroes
        deleted content, a TRUNCATE checkpoint drops old WAL frames and VACUUM
        rebuilds the file without free pages. This is best-effort maintenance: a
        busy database returns early instead of failing the already-durable
        receipt, and the next maintenance pass retries.
        """
        result = {"checkpoint": False, "vacuum": False}
        with _WRITE_LOCK:
            connection = self._connect()
            try:
                connection.execute("PRAGMA secure_delete=ON")
                try:
                    connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                    result["checkpoint"] = True
                except sqlite3.OperationalError:  # busy readers; retried later
                    return result
                try:
                    connection.execute("VACUUM")
                    result["vacuum"] = True
                except sqlite3.OperationalError:
                    return result
                try:
                    connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                except sqlite3.OperationalError:
                    pass
            finally:
                connection.close()
        return result

    # -- small helpers ------------------------------------------------------
    def count(self, sql: str, params: tuple[Any, ...] = ()) -> int:
        with self.read() as conn:
            row = conn.execute(sql, params).fetchone()
            return int(row[0]) if row is not None else 0


def open_store(db_path: str | Path, **kwargs: Any) -> Store:
    return Store(db_path, **kwargs)
