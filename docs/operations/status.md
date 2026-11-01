# Operational status and recovery

## Read-only diagnostics

```bash
kw-memory versions                       # contract/schema/redaction versions
kw-memory doctor --db DB [--socket S]    # contracts, database integrity, socket permissions
kw-memory status --db DB                 # health + capture/deliver/extract/project/embed/cleanup/export
```

`status` reports each pipeline separately: `pending/failed/last_success_at/
last_error_code`. A pipeline is not reported healthy just because another one is
idle. `restore_in_progress` shows as `restoring`; database integrity failures show
`degraded`. No model call is made by these commands.

## Retention

```bash
kw-memory cleanup --db DB --dry-run      # classify + count due details; no purge
kw-memory cleanup --db DB                # tombstone + purge due details, rebuild projection
```

Anchors are business-time based (source/replacement/retraction time). Access,
summaries, import, restore and maintenance never renew them. Unknown lifecycle
state is never auto-expired; imported raw history stays `bootstrap_pending`
until extraction/valid-knowledge identification has run.

## Jobs and recovery

* extract/embed jobs use leases + fencing; expired leases are reclaimed as
  bounded retries after restart; stale results are rejected (`STALE_CURSOR`).
* unclaimed jobs stay `pending`; failed jobs keep `last_error_code` visible.
* only `extract-worker`/`embed-worker` process jobs; the service itself never
  runs model work inside request handling.

## Export/restore reconciliation

After importing a snapshot, adapters must reconcile receipts above the snapshot
watermark against the new `restore_epoch` (DMA outbox keeps events until a
durable receipt arrives). Snapshot limitations (deletions after snapshot,
vector rebuild, old retention anchors) are listed in the import report and must
be surfaced to the operator, not hidden.

## Rollback boundary

The product never deletes host session files or keychain entries. Rolling back a
pilot means stopping the new adapters/service and restoring the previous
host-side injection path; existing databases and outboxes are kept for audit.
