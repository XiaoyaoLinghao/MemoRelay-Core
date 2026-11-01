# External logical backups (T18)

`examples/backup/weekly_backup.py` is a deterministic external operations script.
It makes no model calls and installs no scheduler. Default retention is four
successful generations; frequency belongs to the operator's external scheduler.

## Local-only mode (current trial policy)

Omit `--nas-dir`. Only the local destination is created and verified. Reports
explicitly say `mode: local_only`, `nas_dir: null`, and `verified: {local: true}`;
this does not claim that an off-machine copy exists.

```bash
python3 examples/backup/weekly_backup.py \
  --db /absolute/source/knowledge.db \
  --local-dir /absolute/backups/unified-memory-local \
  --keep 4
```

The source must already exist and be outside the backup roots. Export reads a
consistent transaction and writes an operation-ledger entry in the source KW
DB; it does not modify fact values, source business times or lifecycle anchors.
This is a logical KW snapshot, not a whole-machine backup: it excludes raw
host history, adapter outboxes/configuration, credentials and model caches.
Vectors are rebuilt after restore. Keep adapter state separately when preparing
a production recovery plan and reconcile receipts above the restored watermark.

## Optional second destination

Add `--nas-dir /absolute/mounted/backup/unified-memory` when a real destination
is available. Local/NAS roots must be separate and non-nested. Both requested
copies must verify before rotation. Local and NAS success markers may record
the original mode in which each generation was created; the current run report
states which copies were actually verified. A previously successful local
generation can be copied to NAS without exporting it again.

## Completion, retention and failure behavior

1. Export into local staging, validate payload hashes and manifest, then publish.
2. If requested, copy to second staging, validate and compare manifest hashes.
3. Publish SUCCESS.json only after the requested copies verify.
4. Validate existing marked generations too; corrupt copies must not count as
   usable recovery points. Only then remove oldest successful generations above
   `--keep` (default 4). Incomplete generations are not counted as successes.
5. Export/copy/validation failures never rotate older successful generations.
   A new incomplete generation may remain after a failure; its staging is cleaned
   and the next same-generation run can retry. A rotation filesystem error can
   leave destinations with different older-generation counts; the verified newest
   copies remain and the command returns non-zero. Retry after fixing storage.
6. A local-root file lock rejects overlaps before touching another run's staging.
   The OS releases it on process exit; do not remove the .backup.lock file.
   Different local roots are separate backup jobs and must not share a NAS root.

`--now` chooses a deterministic generation ID. Repeating a successful generation
validates it without re-exporting or overwriting it. Use a fresh timestamp to
capture changes. `--dry-run` only inspects rotation candidates, creates no backup,
and leaves verification false. JSON output plus exit status is the run result;
a SUCCESS marker alone does not mean a scheduler has been installed.

## Restore drill

Validate the selected `generations/<id>/snapshot` with `kw-memory
validate-snapshot`, then `kw-memory import` it into a **new empty directory**.
Inspect the restore result and reconcile adapter receipts after the restore epoch
change. Business timestamps, source IDs and lifecycle/tombstone state must be
preserved; restore/import/recall must not renew retention. Never replace a live
production DB with an unreviewed trial restore.

The current trial uses only a local copy by explicit user decision. Physical NAS
mount/transfer testing is deferred, not simulated or counted as completed. Local
copies on the same disk do not cover loss of that disk. No scheduler has been
installed by this example or the local-backup trial.
