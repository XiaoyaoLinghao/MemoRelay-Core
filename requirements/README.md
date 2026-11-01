# Dependency locks and the macOS 13 / Intel clean-install gate (T01 / AC-41)

The baseline test run (2026-10-30) could **not** install KW 1.x in a clean venv:
`pip install` pulled the MCP dependency chain, which needs `cryptography`, which
has no macOS 13 / x86_64 wheel and cannot be built without Rust + OpenSSL
(baseline evidence: `docs/baseline-20261030/evidence/dependency-install.log`).
The baseline then borrowed the production venv read-only; that is not an
acceptable release install.

**Root cause found (T01/M0):** the failure is not the MCP package itself. The
baseline constraints pinned `cryptography==50.0.1/50.0.2`, and cryptography
50.x publishes **no macOS x86_64 wheel**, so pip falls back to an sdist that
needs Rust and OpenSSL headers. MCP 1.30.0 itself is a pure wheel and accepts
`cryptography>=3.4`; cryptography 48.0.1 ships a `universal2` wheel that works
on this Intel Mac. `sqlite-vec` 0.1.9 also has a macOS x86_64 wheel; only
`jieba` is a pure-Python sdist.

## KW 2.0 decision (D004)

* **Core runtime dependencies: none.** The KW 2.0 store, service, contract,
  extraction, retrieval and portability modules use the Python 3.11 standard
  library only (`sqlite3` with FTS5, `http.server` over a Unix socket).
* Every optional capability is an explicit extra: `mcp`, `llm`, `vectors`,
  `tokenizer`, and the legacy `v1` chain. Optional extras must never be able to
  break the core install (AC-41).
* `mcp>=1.0.0,<2` is an **optional, separately installed environment**. The
  `v1` extra constrains `cryptography>=3.4,<50` so pip selects the cryptography
  48.0.1 universal2 wheel instead of the unbootable 50.x sdist. The core still
  installs and runs with zero third-party packages; the optional environment is
  verified in its own venv by `scripts/clean_install_check.py` and never leaks
  into the core venv. The 2.0 core itself does not use MCP.

## Lock files

| File | Contents |
|---|---|
| `build-lock-macos13-x86_64-py311.txt` | `setuptools`, `wheel` for `--no-build-isolation` wheel builds |
| `test-lock-macos13-x86_64-py311.txt` | `pytest` and its pure-Python dependencies |
| `v1-lock-macos13-x86_64-py311.txt` | Optional legacy KW 1.x / MCP environment (32 packages, cryptography 48.0.1 wheel) |

Regenerate with `python scripts/generate_locks.py` (network needed). Verify a
clean install with:

```bash
python scripts/clean_install_check.py \
  --root <workspace>/runtime/m0/<run-id> \
  --artifact-dir <workspace>/artifacts/M0/<run-id>
```

The check creates a brand-new venv (no `--system-site-packages`), installs the
locks with `--require-hashes`, builds and installs a wheel, runs the contract
suites with the installed package, and records machine facts, wheel hashes,
command lines and exit codes as JSON evidence. It also records that the
optional MCP extra is unavailable on this target while the core stays green.
