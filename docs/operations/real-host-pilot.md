# Real-host pilot closeout (2026-11-01)

The core remains a local standard-library SQLite service. This patch does not switch production hosts, import the user's complete archive, install launchd/cron jobs, add a database service, or change credentials.

## Model configuration

HttpExtractor sends the exact extraction schema and source role/origin/business-time metadata. Provider responses remain untrusted and go through the existing strict evidence/authority validators. Prompt version was prompt:2 in the first pilot; the preproduction source-disposition protocol is prompt:3.

For the tested DeepSeek gateway route, use `kw-memory extract-worker --db <absolute-test-db> --url <gateway>/v1/chat/completions --model deepseek-v4.1-flash --api-key-env <ENVIRONMENT_VARIABLE_NAME> --thinking off`. The actual credential is supplied through the process environment, never an argument/config value or prompt. `provider-default` remains the generic HTTP default; the explicit thinking flag is vendor-specific. The CLI job lease covers its model timeout plus margin. Recommend one semantic extraction worker for this small local deployment; concurrent independent entity creation has not been fully accepted.

## Stable fields and tasks

The model receives up to 48 existing scoped field anchors, prioritized by keyword relevance then recent fallback. Non-secret qualifiers accompany an anchor because they form part of its identity. The raw value is not separately included; ordinary labels can contain fact values. Every outbound anchor field is rechecked against the secret policy. Other projects/workspaces and secret-bearing anchors are omitted.

A target id is accepted only when it names a backend-provided field in the correct project with the same predicate and qualifiers. Different facets retain separate claim keys while referring to the same entity. When a proposal declares a new field with null target but would silently replace a normal, later, equal-or-higher-authority value in an existing field, the write is rejected and its native source remains durable. The bounded job retry receives a recorded FIELD_IDENTITY_COLLISION explanation. Explicit native corrections and normal unknown/future/tied-time conflict handling retain their previous meaning; the guard uses the projection's correction-ordering time, not a rewritten business age. This is a mechanical collision guard, not a proof that a model can never confuse two real-world concepts.

A completion_report using a free-form status predicate is normalized to task.status so task views and verification downgrade rules cannot be bypassed accidentally. The original task-specific predicate contributes to a new task subject when no existing target was given.

## Validated local embedding option

The pilot used BAAI/bge-m3 revision 5617a9f61b028005a4858fdac845db406aefb181, original official ONNX files, 1024 dense dimensions, CLS pooling and L2 normalization. Optional onnxruntime/tokenizers/numpy live in a separate environment, not core site-packages. The test used CPU, two intra-op threads, one inter-op thread and a 512-token input limit. That input limit is a pilot server setting, not the model's maximum context or a general production long-document guarantee. The private test server is stopped after testing; the model cache is retained.

## Results and limits

Read the workspace review/PILOT-CODEX-20261101.md and artifacts/PILOT-20261101/run-a for exact commands, failures, repaired counterexamples, cost uncertainty and scope. In particular, do not claim the full 54-item acceptance or production cutover from unit counts or a six-direction single-fact host exchange. NAS, actual login-Keychain/background/lock/reboot, full native fork/compaction and complete personal-history import remain separate gates. The six real milestone excerpts are a measured small sample, not a completed 956-file migration. Do not use synthetic pilot databases as production memory.

## Preproduction follow-up

The keyword query window is now bounded at 128 terms rather than 32. A normal
multi-question Chinese query previously lost its later subjects before FTS
ranking; this change keeps those terms within the bounded search window. It
does not remove result/token limits: complex summaries may still need a focused
follow-up query. Local-only backup is the current user-selected policy; real
NAS testing is deferred. Production deployment and trial still require user
confirmation.

## Source disposition and historical bootstrap

A warning-free fenced extraction can finish only source parts it actually
accounts for: validated claims, explicit whole-group no-durable-knowledge
disposition, or an already stored full-span matching evidence proof (such as a
user correction already applied by the API). Claim writes, lifecycle disposition
and job success share one transaction. Failed/stale/ambiguous/unaccounted parts
remain protected; a bare historical succeeded-job flag is not coverage proof.

Current native configuration/fact evidence is retained while valid; no blanket
30-day policy is imposed on it. Explicitly imported historical process detail
keeps its original 30-day business-age rule once valid knowledge is extracted;
completed-task parts can use 180 days. Explicit existing classifications win.
Unknown source age stays unclassified and is not replaced with processing time.
Retained facts remain queryable with honest expired-source markers after a
classified detail expires. Maintenance/import/recall does not renew an anchor.

New prompt:3 jobs ask the model to explicitly exclude a whole group only when
it contains no durable facts. Completed old jobs are not automatically rerun or
blindly unlocked. Unresolved protected history is a durable incomplete-retrieval
condition, rather than a false claim that no history exists. Inspect pending
lifecycle records before production migration and do not force-clear them.

## Remembered-answer repetitions

Actual detailed-source recall produced two assistant-only reported revisions of an unchanged user-confirmed port. KW now reuses an exact current field/value for assistant-only repetitions rather than stamping it with the reply time or improving an old hypothesis/legacy record's confidence. This is decided inside the same fenced transaction as job success and source disposition. The original source event is retained and a `job_attempts.detail.reused_existing_claims` audit links it to the existing revision. The event's actual message time is not rewritten; a pure repeated-detail part inherits the remembered fact's retention age, or stays unclassified when that age is uncertain.

Fresh user confirmations and tool observations remain independent evidence. Changed values, entities or qualifiers remain new information; an actual completion report after only a task hypothesis is also preserved as unverified reported progress. A mixed part with new facts follows the normal new-fact retention rules. This is exact typed field/value reuse, not fuzzy text deduplication or a guarantee that a model never proposes an unrelated extra field. Conflict resolution, task verification and current-state selection remain backend rules.

The observed two cases were replayed through the real receive/API/fenced-worker path using normalized claims and exact stored evidence spans. The original provider response was not logged, so this is explicitly a normalized replay of genuinely observed claims rather than a claim of raw-response replay. Old succeeded jobs are not automatically reprocessed or historical revisions deleted; a production migration still starts from the reviewed source/state plan.

### Important execution evidence and uncertain proposals
Prompt4 explicitly preserves executed tool artifact paths, digests and narrow check scopes independently of user plans or completion reports. Model-declared warnings withhold any new fact writes and use the existing persistent three-attempt retry policy with fixed repair feedback. Zero-write exclusions or exact existing-fact reuse can finish without repeating a model call; ambiguous raw-source dispositions remain protected. Free-form warning text is not logged; persistent uncertainty stays failed/pending with source intact. A successful job is never reset automatically. Prompt3 is retained byte-identically for pending/restored v3 jobs; unsupported older prompt versions fail explicitly instead of silently recording a v4 execution as an old processor. Restore matching releases or plan an explicit migration before reprocessing such jobs.

### Full source detail is not the retained fact
Successful ordinary source bodies enter the existing30-day temporary-detail policy at their original business time; completed-task detail keeps180 days, and explicitly classified source policies remain authoritative. Long-lived current facts retain their structured value/text and provenance separately; raw-source expiry is visible through source_detail_expired. Keeping a port value must not make every diagnostic line in the same message permanent.365-day superseded/revoked fact history remains a separate per-revision lifecycle. A source part combining new facts and old assistant echoes uses the oldest inherited echo anchor for full-body retention, while new facts keep their own real event time. Since a part is the source deletion unit, a mixed part can lose its original body earlier than a new fact; the new structured fact remains and its source is explicitly expired, rather than silently extending old text. This changes no native source files and never clears protected unresolved/import batches.

Raw-source TTL expiry does not revoke an ordinary native assertion: current facts stay current with an expired-source marker. Actual withdrawal/withholding still invalidates evidence. Verified task completion keeps the stricter readable-proof requirement and downgrades when that checking evidence expires. An expired old history revision must not hide a newer retained configuration.

### Manual entity aliases
Alias merge is explicit and project-scoped; it never rewrites historical claims/evidence. Known entity categories must match (an untyped generic subject is not silently cast to a known device/person type). Type metadata uses the store's canonical entity identity. A source with a different active destination must be undone before remerging; this prevents two active destinations and equal-time ordering surprises. Existing same-pair merge remains idempotent; compatible untyped subject aliases remain supported.
