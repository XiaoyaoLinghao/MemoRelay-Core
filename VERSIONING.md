# VERSIONING — knowledge-weaver (KW)

Rules for bumping KW's version. **Written for an AI maintainer: follow exactly,
do not invent numbers.** This file exists because the version once froze at
`0.1.0` through a large refactor — nobody knew the rule. The fix is a single
source of truth + these rules + a release check.

## Single source of truth
- KW's version is defined in **exactly one place**: `pyproject.toml` → `[project].version`.
- Code exposes it via `importlib.metadata.version("knowledge-weaver")`
  (as `knowledge_weaver.__version__`). **Never hardcode the version anywhere
  else** — README, docstrings, MCP banner must read from the source or be
  matched by the release check. A second hardcoded copy is how versions drift.

## Scheme: SemVer (MAJOR.MINOR.PATCH)
Bump by the **largest** applicable change in the release:
- **MAJOR** — a breaking change: an incompatible MCP tool contract (renamed or
  removed tool, a newly-required arg), a DB / data-format change that requires
  migrating an existing `knowledge.db`, or anything that breaks a caller relying
  on prior behavior.
- **MINOR** — a backward-compatible addition: a new MCP tool, a new optional
  field/param, a new capability that defaults off and does not change existing
  output.
- **PATCH** — a bug fix or internal change with no caller-visible API/behavior change.

> First stable release is `1.0.0` (the W1→W3.4 + recovery milestone). There is
> no `1.1.0` without a `1.0.0` having shipped first — magnitude of work does not
> skip the first-stable cut.

## On every release
1. Bump `[project].version` in `pyproject.toml` (the only place you edit the number).
2. Add a `## vX.Y.Z` entry to `CHANGELOG.md`.
3. Run the consistency check below.
4. `git tag vX.Y.Z` on the release commit and push the tag.

## Spec coupling — KW ↔ DMA  (keep this section byte-identical in both repos)
KW reads the memory files DMA writes. The contract between the two is the
**`KW_MEMORY_FILE_SPEC` version** plus the W2 `### 结构化事实` JSON-facts schema.
The package version numbers do **not** indicate compatibility — the spec does.
- When the memory-file format or the structured-facts schema changes, bump the
  **`KW_MEMORY_FILE_SPEC`** version.
- In the **same coordinated change**, update the "implements spec vX" declaration
  in **both** repos (KW's parser side and DMA's writer side).
- A KW release and a DMA release are compatible **iff they declare the same
  `KW_MEMORY_FILE_SPEC` major**. Never assume the package numbers imply it.

## Release consistency check (gives the rules teeth)
Before tagging, assert all of:
- `pyproject.toml` version **==** the git tag being created.
- No other tracked file hardcodes a different version string (`grep`; README/docs
  must read from source or match exactly).
- The `KW_MEMORY_FILE_SPEC` version KW declares **==** the spec file's own header.
