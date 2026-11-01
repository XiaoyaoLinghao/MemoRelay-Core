"""Portable logical snapshot export (T14).

The snapshot is **logical records**, never a raw SQLite copy, so SQLite free
pages / deleted secret bodies cannot leak. SQLite is only read via a single
consistent read transaction; files are written to a temporary sibling
directory, fully hashed and atomically published.

Contents:

* ``manifest.json`` – format/schema/contract versions, backend identity,
  restore epoch, watermark, file hashes/sizes, counts, clock policy, limitations;
* ``records/*.jsonl`` – backend, clients, events, revisions, parts, receipts,
  claims, claim-revisions, evidence links, current projection, correction
  operations, lifecycle records, tombstones, jobs/attempts;
* ``config/versions.json`` – non-secret version/config identifiers;
* ``SHA256SUMS`` – every file except itself.

Vectors are not exported by default (model-specific, rebuildable); the manifest
says so. Bodies are already redacted; withheld/expired bodies are exported as
null and therefore never restored as readable evidence.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

from knowledge_weaver import contracts
from knowledge_weaver.store.db import Store, utc_now
from knowledge_weaver.store.status import operation_ledger

FORMAT_VERSION = "1.0"
SNAPSHOT_SCHEMA_VERSION = 3
RECORDS_DIR = "records"
CONFIG_DIR = "config"

#: table -> export file name (deterministic import order is in importing.py)
TABLES = {
    "clients": "clients.jsonl",
    "events": "events.jsonl",
    "event_revisions": "event_revisions.jsonl",
    "event_parts": "event_parts.jsonl",
    "receipts": "receipts.jsonl",
    "claims": "claims.jsonl",
    "claim_revisions": "claim_revisions.jsonl",
    "evidence_links": "evidence_links.jsonl",
    "current_claims": "current_claims.jsonl",
    "task_views": "task_views.jsonl",
    "entities": "entities.jsonl",
    "entity_aliases": "entity_aliases.jsonl",
    "correction_operations": "correction_operations.jsonl",
    "capabilities": "capabilities.jsonl",
    "resources": "resources.jsonl",
    "lifecycle_records": "lifecycle_records.jsonl",
    "tombstones": "tombstones.jsonl",
    "jobs": "jobs.jsonl",
    "job_attempts": "job_attempts.jsonl",
}
BACKEND_FILE = "backend.json"
REQUIRED_FILES = ("manifest.json", "SHA256SUMS", f"{CONFIG_DIR}/versions.json", f"{RECORDS_DIR}/{BACKEND_FILE}")

#: Tables added in snapshot schema v3 (capability/resource registry). Older v2
#: snapshots legitimately lack these files; v3 requires them for completeness.
V3_TABLES = frozenset({"capabilities", "resources"})

LIMITATIONS = [
    "deletions_after_snapshot_unknown: the snapshot cannot see deletions that happened after it was taken",
    "registry_references_restored: capability/resource registry rows are exported with the snapshot and their recipe references are reconciled on import",
    "vectors_rebuilt_not_restored: semantic embeddings are rebuilt after import, not restored",
    "dma_receipts_above_snapshot_watermark_require_reconciliation: adapters must re-check receipts above the snapshot watermark after a restore epoch change",
    "old_snapshot_retention_limits: lifecycle anchors are preserved as-is, so very old snapshots may restore already-expired detail only as expired/tombstoned markers",
]


class ExportRefused(ValueError):
    """The snapshot was refused because a logical record still looks secret-bearing."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def _secret_field(row: dict) -> str | None:
    """Return the first column whose value still matches the redact:1 policy."""
    for column, value in row.items():
        if isinstance(value, str):
            text = value
        elif isinstance(value, (list, dict)):
            text = json.dumps(value, ensure_ascii=False)
        else:
            continue
        if contracts.secret_kinds(text):
            return str(column)
    return None


def _assert_rows_secret_free(table: str, rows: list[dict]) -> None:
    """F02 defence in depth: never publish a logical record that still leaks.

    The scrub on the redaction path clears derived payloads, but a bug or an
    older database could still hold a secret-bearing row. Export fails closed
    instead of writing it into a portable archive.
    """
    for index, row in enumerate(rows):
        column = _secret_field(row)
        if column is not None:
            raise ExportRefused(
                "REDACTION_REQUIRED",
                f"refusing to export {table} record {index}: column {column!r} still "
                "matches the redaction policy; scrub it first",
            )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_jsonl(path: Path, rows) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":"), default=str) + "\n")
            count += 1
    return count


def _files_in(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file())


def export_snapshot(store: Store, target: str | Path, *, created_at: str | None = None) -> dict:
    """Export one validated logical snapshot and record it in the export ledger.

    AC-52 round 4: a real export is a product operation and must show up in
    ``status(store)`` as pending while it runs and succeeded (or failed with a
    visible error code) afterwards.  The export's own pending ledger row is not
    written into its own snapshot (it would come back as an orphan job on
    restore); earlier operation history is exported normally.
    """
    target = Path(target)
    if target.exists() and (not target.is_dir() or any(target.iterdir())):
        raise ValueError(f"export target must be absent or an empty directory: {target}")
    parent = target.parent
    parent.mkdir(parents=True, exist_ok=True)
    staging = parent / f".{target.name}.staging-{uuid.uuid4().hex[:8]}"
    staging.mkdir()
    try:
        with operation_ledger(store, "export") as ledger:
            records = staging / RECORDS_DIR
            records.mkdir()
            (staging / CONFIG_DIR).mkdir()
            counts: dict[str, int] = {}
            with store.read() as conn:
                conn.execute("BEGIN")
                try:
                    meta = store.meta(conn)
                    (records / BACKEND_FILE).write_text(
                        json.dumps(
                            {
                                "backend_instance_id": meta["backend_instance_id"],
                                "restore_epoch": meta["restore_epoch"],
                                "schema_version": meta["schema_version"],
                                "watermark": meta["watermark"],
                                "restore_in_progress": meta["restore_in_progress"],
                            },
                            ensure_ascii=False,
                            indent=2,
                        )
                        + "\n",
                        encoding="utf-8",
                    )
                    for table, filename in TABLES.items():
                        rows = [
                            dict(row) for row in conn.execute(f"SELECT * FROM {table}").fetchall()
                        ]
                        if table == "jobs":
                            # Do not export the export's own in-flight ledger row.
                            rows = [row for row in rows if row["job_id"] != ledger["job_id"]]
                        _assert_rows_secret_free(table, rows)
                        counts[table] = _write_jsonl(records / filename, rows)
                finally:
                    conn.execute("COMMIT")
            counts["backend"] = 1
            (staging / CONFIG_DIR / "versions.json").write_text(
                json.dumps(
                    {
                        "contract_version": contracts.CONTRACT_VERSION,
                        "event_schema_version": contracts.EVENT_SCHEMA_VERSION,
                        "extraction_schema_version": contracts.EXTRACTION_SCHEMA_VERSION,
                        "redaction_policy_version": contracts.redaction_policy_version(),
                        "snapshot_schema_version": SNAPSHOT_SCHEMA_VERSION,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            manifest = {
                "format_version": FORMAT_VERSION,
                "snapshot_schema_version": SNAPSHOT_SCHEMA_VERSION,
                "created_at": created_at or utc_now(),
                "contract_version": contracts.CONTRACT_VERSION,
                "event_schema_version": contracts.EVENT_SCHEMA_VERSION,
                "redaction_policy_version": contracts.redaction_policy_version(),
                "backend_instance_id": meta["backend_instance_id"],
                "restore_epoch": meta["restore_epoch"],
                "watermark": meta["watermark"],
                "business_clock_policy": "source business time preserved; import time never becomes business time",
                "counts": counts,
                "files": {},
                "vectors": {"included": False, "reason": "model-specific; rebuilt after import"},
                "record_files_are_redacted": True,
                "raw_sqlite_copy": False,
                "limitations": LIMITATIONS,
            }
            # Hash every payload file before the manifest is written.
            payload_files = [
                path for path in _files_in(staging) if path.name != "manifest.json"
            ]
            for path in payload_files:
                relative = path.relative_to(staging).as_posix()
                manifest["files"][relative] = {
                    "sha256": _sha256(path),
                    "size": path.stat().st_size,
                }
            manifest_path = staging / "manifest.json"
            manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            with (staging / "SHA256SUMS").open("w", encoding="utf-8") as handle:
                for path in _files_in(staging):
                    if path.name == "SHA256SUMS":
                        continue
                    handle.write(f"{_sha256(path)}  {path.relative_to(staging).as_posix()}\n")
            if target.exists():
                target.rmdir()
            os.replace(staging, target)
            result = {
                "target": str(target),
                "format_version": FORMAT_VERSION,
                "watermark": manifest["watermark"],
                "backend_instance_id": manifest["backend_instance_id"],
                "counts": counts,
                "files": len(manifest["files"]) + 2,
                "snapshot_sha256": _sha256(target / "manifest.json"),
            }
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return result
