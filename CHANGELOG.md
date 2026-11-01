# Changelog

All notable changes to knowledge-weaver. Versioning follows [VERSIONING.md](VERSIONING.md).

## v1.1.7 — runtime health checks and report recovery (2026-09-09)

### Fixed
- Use DMA's versioned runtime status to distinguish healthy idle, unresolved failures,
  stale runs and pending work. Historical errors and an absent daily Memory file no
  longer independently trigger an alert.
- Treat legacy `day_count` distributions as information; the field does not provide
  distinct-day aggregation semantics.
- Validate invocation-owned daily and weekly reports, publish checker failures,
  clear alerts only after complete recovery, and append weekly history after publication.

### Compatibility
- Pair with DMA v1.7.1 for the `dma-runtime-status-v1` producer. Memory SPEC v1.1 and
  the knowledge database schema are unchanged; no historical data migration is required.
- Linux validation: 98 targeted tests passed, including producer-to-wrapper integration.

## v1.1.5 — clean_and_rescore FK fix (2026-06-04)

### Fixed
- `scripts/clean_and_rescore.py` `_delete_entities_cascade` cleared
  relations/vectors/fts/vec before deleting an entity but NOT `access_log` (which
  also FK-references `entities`). Under `PRAGMA foreign_keys=ON` this made
  `DELETE FROM entities` fail for any entity that had been accessed — the source
  of the recurring "FK skip" that left accessed-but-noisy entities un-prunable.
  access_log is now cleared in the cascade. +1 test.

## v1.1.4 — max code-review fixes (2026-06-04)

Full-codebase MAX review (not a diff) — found bugs in older, never-reviewed core.

### Fixed
- **K1** vector search lost distance ordering: `_search_entity_vectors_vec`'s
  `IN (...)` re-fetch returned arbitrary order (only with sqlite-vec, i.e. prod),
  mis-ranking RRF. Results are now re-ordered nearest-first.
- **K2** JSON-facts with an emoji/punctuation-only name slugified to empty → all
  collapsed into one `fact:` entity. Such names are now dropped (like the regex path).
- **K3** merge repointed edges with `INSERT OR REPLACE`, clobbering an edge dst
  already had. Now `INSERT OR IGNORE` keeps dst's weight/evidence.
- **K4** `rollback_merge` didn't reverse dst's accumulators (day_count/source_lines/
  first_seen/importance) and left orphaned reviews 'stale'. The snapshot now stores
  dst's pre-state; rollback restores it and un-stales the reviews.
- **K5** `knowledge_search` ranked/filtered by score re-derived with access_count=0,
  disagreeing with the stored+displayed importance. Now uses stored importance.
- **K6** `knowledge_trace` labelled every depth≥2 edge `RELATES_TO`; now reports
  each entity's real relation type.
- **K7** project→sub-entity dependency used substring match ('Home' ⊂ 'HomeBrain');
  now word-boundary (`_name_mentioned`), still matching CJK.
- **K8** `parse_indexed_verdicts` rejected a chunk on any extra key; now tolerates
  harmless extras (keys ⊇ {1..N}), still rejects missing/shifted.

## v1.1.3 — code-review fixes (2026-06-04)

Ten findings from a high-effort code review of the v1.0–v1.2 work.

### Fixed (correctness)
- **Cost-leak hole**: the v1.1.2 idempotency guard skipped only `dismissed` (LLM),
  not `rejected` (human via kw_resolve), so human-rejected pairs were re-queued and
  re-judged every cycle. insert_review + semantic_dedupe now skip
  `pending|dismissed|rejected`, and insert_review is direction-agnostic (a→b == b→a).
- **Irreversible over-delete**: prune_residue `_TAG_RE` matched real models (T5/T6
  via `T\d+`) and bare `P\d+`; both removed — only dash/prefixed internal codes
  (P1-1/WI2/Track2/Phase1-3) are deleted now.
- **Wrong auto-merge (letter suffix)**: `_digit_variant_pair` now also flags a short
  (≤2 char) alphanumeric tail difference, so GPT-4/GPT-4o are demoted to review
  instead of auto-merged.
- **Wrong auto-merge (index drift)**: LLM verdicts are accepted only when the returned
  JSON keys are exactly `1..N` (parse_indexed_verdicts), preventing a shifted/extra
  index from assigning a verdict to the wrong pair.
- **Recovery dropped real abbrevs**: short all-caps (AWS/GPT/SQL) are no longer
  hard-dropped by name; they pass the signal gate (kept iff recurring/connected).
- **Noise filter unsafe default**: llm_keep_filter now DROPS a failed/invalid chunk
  (was keep-all), so an API blip can't re-admit a batch of noise.
- **Connection leak/lock**: the consolidation semantic-dedupe / tie-break blocks now
  close their DB connection in `finally`.
- **Orphan reviews**: merging an entity away marks other pending reviews referencing
  it `stale`, so the pending count isn't permanently inflated.

### Changed (internal)
- Extracted `knowledge_weaver._llm.classify_items` — one chunked chat caller shared by
  llm_type_pairs / llm_judge_pairs / llm_keep_filter (was ~90 duplicated lines).
- semantic_dedupe caches decoded embeddings per run (no repeated json.loads on the
  cron hot path).

## v1.1.2 — stop re-judging dismissed pairs (2026-06-04)

### Fixed
- Self-maintenance cost leak: with `KW_SEMANTIC_DEDUPE=review` +
  `KW_TIEBREAK_REVIEWS=1`, every consolidation re-queued and re-judged (LLM cost)
  the same vector-similar-but-distinct pairs, because a dismissed pair was not
  treated as "already decided". Now `insert_review` is idempotent against
  pending AND dismissed, and `semantic_dedupe` skips any pair that already has a
  review — so a distinct pair is judged once, not every cron cycle.

## v1.1.1 — tie-break count fix (2026-06-04)

### Fixed
- `tiebreak.apply_tiebreak`: a no-op merge (target already removed by a prune or
  an earlier merge in the same batch) is now counted as `dismissed` — matching
  the review status it sets — instead of being reported as a phantom `merged`.
  Cosmetic count only; data was always correct.

## v1.1.0 — de-ruling + cleanup (2026-06-04)

Implements `KW_MEMORY_FILE_SPEC` v1.1 (unchanged).

### Added
- **T1 — de-enumerated identifier guard**: `_digit_variant_pair` (resolver) flags
  same-template-differ-in-digits pairs (v0.2.0/v2.9.0, COMP7940/COMP7240,
  ports) without enumerating formats — a new identifier shape needs no new regex.
  The old `_IDENTIFIER_RE` stays only as the LLM-unavailable fallback. Used by
  both ingest resolution and `semantic_dedupe`.
- **T1 — LLM tie-breaker** (`tiebreak.py` + `scripts/tiebreak_reviews.py`):
  asks "same thing or two different things?" over the merge-review queue,
  auto-merging 'same' / dismissing 'different' / keeping genuine ambiguity.
  Clears the review backlog; env-gated in consolidation (`KW_TIEBREAK_REVIEWS=1`).

### Changed
- **T4 — `导致` typing prompt tightened**: counterfactual test ("if A didn't
  happen, would B?") + explicit "temporal order / investigation steps /
  co-occurrence are NOT causation" + a negative few-shot. Reduces over-applied
  causal edges (spot-check was 67%).

### Tooling
- **T2.1 — residue prune** (`prune_residue.py` + `scripts/prune_residue.py`):
  one-off cleanup of bare version numbers / internal codes (P1-1, WI2, Track2)
  the legacy regex extractor mis-typed as `tech` (signal-gate can't catch them).

## v1.0.0 — first stable release (2026-06-04)

First production-validated release. Bundles the W1→W3.4 optimization program plus
the post-swap data recovery. Implements `KW_MEMORY_FILE_SPEC` **v1.1**.

### Added
- **W1 — entity resolution**: vector-first three-band resolver (merge / review /
  distinct) with an identifier guard, a `merge_review` queue, `merge_log`
  rollback, and registry alias normalization.
- **W2 — structured handoff**: ingest DMA's `### 结构化事实` JSON facts
  (`KNOWLEDGE_WEAVER_EXTRACT_MODE=json`); registry lexicon injection.
- **W3.1/3.2 — historical rebuild**: `backfill_facts.py`, `rebuild_from_facts.py`.
- **W3.4 — typed knowledge graph**: closed relation vocab
  (依赖/使用/包含/导致/取代/关于/矛盾/相关) with schema-constrained LLM typing,
  triplet hints, and online edge typing in consolidation.
- **Retrieval**: jieba Chinese FTS, RRF fusion (k=60), registered-project nodes.
- **Recovery + dedup**: `recover_from_old.py` ports rebuild-lost entities from the
  pre-swap DB (carrying day_count + vector; tech gated by accumulated signal);
  `semantic_dedupe` second-pass merge (auto / review-only); merging keeps the
  merged-away name as a searchable FTS alias (recall-preserving).
- **Backlog**: W3.3 supersede de-ranking, W6 embedding-drift check.
- `__version__` exposed from package metadata; `VERSIONING.md`.

### Notes
- Production DB rebuilt from 656 noisy → clean structured entities; generic
  RELATES_TO edges typed into the closed vocabulary; 51 real decisions + signal-
  gated tech recovered from the pre-swap DB.
