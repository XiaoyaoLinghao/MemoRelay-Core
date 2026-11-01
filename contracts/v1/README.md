# DMA / KW shared memory contract v1 (frozen)

Canonical source of truth for the DMA 2.0 ↔ KW 2.0 interface. DMA vendors a
checksummed copy under `contracts/v1/` in its own repository.

## Files

| File | Purpose |
|---|---|
| `openapi.json` | OpenAPI 3.1 document, 8 endpoints, 25 component schemas |
| `extraction-output.schema.json` | JSON Schema (2020-12) for the untrusted LLM proposal |
| `semantic-invariants.json` | 22 semantic invariants (ID/SRC/RAW/SEC/CLAIM/TIME/JOB/VEC/TTL/RESTORE/RECALL/EXPORT/LLM) that JSON Schema cannot express |
| `error-catalog.json` | Machine-readable error codes with HTTP status and retryability |
| `valid-examples.json` | Shared positive examples, one per schema where the design supplied one |
| `invalid-examples.json` | Shared negative examples (patch applied to the valid example) |
| `redaction-policy.json` | Shared `redact:1` sanitizer rules (DMA redacts, KW detects only) |
| `redaction-examples.json` | Synthetic canary cases both repositories must pass |
| `VERSION.json` | Contract/schema/redaction versions and compatibility window |
| `SHA256SUMS` | SHA-256 of every other file in this directory |

## Rules

1. **Strict**: every object schema rejects unknown fields (`additionalProperties: false`);
   unknown or unsupported `schema_version` is rejected, never guessed.
2. **Frozen**: files in this directory are not edited in place. A change requires:
   a new `contract_version`, a coordinated DMA vendor update, contract tests in
   both repositories, a `state/DECISIONS.md` entry and Codex review.
3. **Shared samples**: both DMA and KW validate `valid-examples.json` and
   `invalid-examples.json`; passing only in one repository is a contract failure.
4. **Error codes** are stable; messages are not. Responses and logs must never
   contain credentials, secret values, request bodies or raw backend errors.
5. **Transport**: HTTP/1.1 over an owner-only Unix domain socket
   (dir 0700, socket 0600). No TCP listener in the default deployment.

## Validate

From the repository root:

```bash
python scripts/contract_check.py
```

The check verifies checksums, schema structure, strictness, all shared examples,
the error catalog, and the pure-Python `event_id`/digest helpers. It uses only
the Python standard library.

## Versioning

| Version | Meaning |
|---|---|
| `contract_version` | Interface compatibility; DMA and KW must agree on the major |
| `event_schema_version` | Envelope/body schema (`schema_version` field) |
| `extraction_schema_version` | LLM proposal schema |
| `redaction_policy` (`redact:1`) | Sanitizer policy version; a body revision is not accepted under a policy it did not pass |
