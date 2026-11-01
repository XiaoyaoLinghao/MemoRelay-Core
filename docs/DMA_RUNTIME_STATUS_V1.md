# DMA runtime status contract v1

This additive contract supplements KW_MEMORY_FILE_SPEC v1.1. It does not change Memory content, session consumption, summarization, compaction, entity counts, or scoring. DMA owns production of this file; KW is a read-only consumer. No consumer may infer recovery from log mtime, process exit 0, or Memory-file absence.

## Scenarios and non-goals
1. A successful archive followed by empty/noise-only runs remains healthy without creating today's Memory file.
2. Storage or summary failure remains visible until the corresponding operation succeeds; unrelated idle/lock-skipped runs cannot clear it.
3. Delayed work, stale scheduling, incomplete runs and observation failures are distinguishable from healthy idle.
4. A failed KW subprocess cannot reuse an old report; a completed healthy report clears only its corresponding pending alert.
No entity migration, historical replay, automatic deployment, or changes to archive decision policy.

## Transport and ownership
DMA publishes UTF-8 JSON atomically to `$DAILY_MEMORY_STATUS_PATH`, default `$DAILY_MEMORY_CONFIG_DIR/.runtime_status.json` (config dir defaults to the installed skill's config). File contains no session text, credentials or session keys.
The archive lock owner is the only producer. A skipped competing invocation does not overwrite the owner's running result. Begin publishes running; completion publishes a terminal outcome, including handled failures with exit 0. A crash leaves running, which expires. Status publication failure must be observable in stderr/log without changing archive policy.
The runtime status feature requires the existing flock-based archive lock. If flock is absent, preserve the original archive behavior but disable status publication and log that monitoring is unavailable; do not introduce a new archive lock/fallback policy in this slice.
KW configures its read path using `KW_DMA_STATUS_PATH`; defaults to the discovered deployment convention documented in its README. It never launches archive or writes DMA state.

## Required JSON fields
- `schema_version`: exactly `dma-runtime-status-v1`.
- `run_id`: nonempty unique string.
- `started_at`, `observed_at`: UTC ISO8601 timestamps with offset.
- `finished_at`: UTC timestamp, null only while running.
- `outcome`: running | archived | idle | noise_only | deferred | failed | partial.
- `reason`: stable code without private text; deferred reasons include threshold, min_messages, cooldown; failures include storage, summary, archive.
- `last_completed_at`: last terminal run timestamp or null (not proof of archival).
- `last_archived_at`: last successfully committed substantive archive timestamp or null. Noise-only, idle, failed/partial and deferred runs do not advance this value.
- `pending_count`: nonnegative integer or null; raw user/assistant messages remaining from the validated run snapshot, not a live count and not a claim that all messages are substantive.
- `oldest_pending_at`: UTC timestamp or null; null if no messages or timestamp is unavailable.
- `pending_reconcile_count`: nonnegative integer or null; sidecars still requiring reconciliation.
- `oldest_reconcile_at`: UTC timestamp or null (sidecar mtime baseline).
- `checkpoint_progress_at`: UTC timestamp or null; advances only when cursor content changes, not on touching a file.
- `failures`: exactly storage, summary, archive entries; each is `{consecutive: nonnegative integer, last_failed_at: timestamp|null, last_recovered_at: timestamp|null}`. Count is consecutive failed attempts for that operation; only successful execution of that operation resets its count. Idle, noise-only, delayed, skipped-lock and a successful storage read do not clear summary failures. Degraded/partial output is not summary recovery.
- `status_errors`: list of stable diagnostic codes; nonempty means monitoring evidence is incomplete.
All fields must be present. Unknown is null, never fabricated zero. Failure history and timestamps survive subsequent runs. Corrupt previous status must be surfaced, not silently reset to a healthy record.

On first publication, existing legacy cloud retry/alert evidence must not be lost: seed unresolved summary failures read-only, or report incomplete summary evidence until an actual summary attempt resolves it. Existing legacy success timestamps must not be relabeled as verified substantive archival. Historical last_archived_at may remain null until the first observed substantive success.

Pending values and timestamps belong to the same run snapshot and may be stale by the time KW reads them. Consumers validate schema/types/time ordering, reject future or stale records with a small clock-skew allowance, and do not replace unknown with zero. A terminal snapshot may retain unknown fields if a source read failed; its explicit failure/status_errors prevents healthy classification.

## KW health policy
Default configured expected DMA interval is 30 minutes; status/running timeout is 90 minutes (configurable). Pending/reconcile overdue threshold defaults to 24 hours (configurable). These are operational defaults, not Memory format requirements.
- Missing status during rollout: warning/unknown, never report DMA as confirmed healthy.
- Malformed, unsupported, unreadable, incomplete evidence: health-check failure alert.
- Fresh running: info; no recovery/old-alert clearing until complete evidence exists.
- Latest failed/partial outcome, unresolved per-operation failures, stale/running timeout, or overdue backlog: alert.
- Non-overdue reconcile backlog: warning. Deferred messages below timeout: info.
- Checkpoint nonmovement alone is not failure; require overdue input backlog.
- Fresh terminal idle/noise/archived with known zero backlog and no unresolved failures: ok, optionally recovered if operation history proves recovery.
- Historical 24h log error counts and cross-day entity ratios are information only.
- Today's Memory absence is normal only when there is no expected output; use wording "current snapshot has no pending messages" unless a same-day no-content result supports a stronger statement.
Keep other existing KW DB checks, but internal exceptions in any check affect the overall result. Unknown/running must not clear previously pending alerts.

## KW report lifecycle
Health subprocess reports use `schema_version=kw-health-report-v1`, a caller-supplied `run_id`, timezone-aware `generated_at`, `report_date`, `overall_status`, `checks`, and boolean `checks_complete`.
Exit 0 means a valid completed report was produced, not necessarily healthy; nonzero means checker execution/evidence failure. Wrapper validates exit code, run_id, schema, fields and freshness and reads only its unique invocation output. Atomic publication; malformed/missing reports create checker-failure alerts. Clear daily alert only with complete overall ok and settled DMA evidence. Weekly uses invocation-owned output too and never searches by a hardcoded year.

A well-formed DMA failed record, including null backlog after a failed storage read, is a service failure rather than a checker crash. Likewise a valid expired running record is a service timeout. They remain named DMA alerts in a valid exit-0 report; malformed evidence or nonempty status_errors is a checker evidence failure. Daily reports must contain all eleven baseline check entries (entity_total, dma_health, new_entity_inflow, embedding_coverage, orphan_ratio, cross_day_aggregation, type_distribution, name_conflicts, relation_integrity, pending_merges, coverage_days); omissions cannot certify recovery.

checks_complete means enough settled evidence to evaluate every check and consider clearing a previous alert. It may be false for a valid service-failure report (for example storage failure leaves backlog unknown); this does not itself require nonzero exit. checker_errors, rather than checks_complete alone, distinguishes checker execution/evidence failure from a valid service alert.

Weekly --output emits a JSON envelope with schema_version=kw-weekly-report-v1, run_id, timezone-aware generated_at, report_date, checks_complete=true, ISO week (YYYY-Www) and markdown. The wrapper validates it and publishes the Markdown filename itself. For both report types, report_date uses generated_at's date and timezone.
The weekly envelope also carries history_entry with matching run_id and week. In invocation mode only the wrapper appends this entry after final Markdown publication; a failed staged or final publication cannot append a success history entry. Standalone weekly generation likewise appends history only after publishing its Markdown. Existing Markdown/history generated_at strings retain their legacy local-time representation; the new envelope uses an aware timestamp. Weekly failure flags retain kind/file/summary by referencing a newly generated checker-failure Markdown, never an old successful report.

## Acceptance
Tests use isolated temporary files/DBs and no real sessions/network calls. Cover success, no-input/noise, delayed input, storage and summary failures including exit-0 paths, partial/reconcile, recovery scope, crash/running expiry, unchanged cursor, missing/corrupt status, malformed reports, subprocess timeout/nonzero, previous report isolation and alert clearing. Retain existing format and archive-policy regression tests. Independent review covers both repository diffs. Linux deployment/runtime evidence remains separate from local fixtures.

If the weekly report directory is unavailable, publish the fresh failure Markdown beside the alert flag. If neither artifact location is writable, still attempt a minimal failure flag with artifact_error=failure_report_unavailable and no file field; consumers must display its summary. Never substitute an old report.

Unknown failure history (corrupt/interrupted status or unreadable/invalid legacy failure markers) persists across idle runs. Automatic clearance requires one complete archived run proving storage, summary and archive success, zero pending and reconciliation backlog, and no new evidence errors.
