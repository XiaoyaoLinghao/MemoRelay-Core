# Install and start (developer preview, no automatic deployment)

This project provides a **manual, reviewable** install path. Nothing here edits
launchd/cron, the system keychain, global model config, or an existing
Hermes/DSH installation.

## Clean install

```bash
# Python 3.11 on macOS 13 / Intel; core has zero third-party runtime deps
python3.11 -m venv ~/Library/Application\ Support/UnifiedMemory/venv
~/Library/Application\ Support/UnifiedMemory/venv/bin/pip install knowledge_weaver-<version>.whl
```

The wheel embeds the frozen `contracts/v1` files (the repository copy remains
the source of truth), so `kw-memory doctor` works without a checkout or
`KW_CONTRACTS_DIR`. Optional extras are separate (`mcp`, `llm`, `vectors`,
`tokenizer`); the core never imports them. `jieba`/`sqlite-vec`/`mcp` are not
required for keyword recall or the service.

## Initialize

```bash
kw-memory init --db "$STATE/kw/knowledge.db" \
  --workspace owner:local --host hermes --instance instance:hermes-main \
  --client client:hermes-main --actor jun --project project:songta
kw-memory doctor --db "$STATE/kw/knowledge.db"
```

## Run (foreground; operator chooses service supervision)

```bash
kw-memory serve --db "$STATE/kw/knowledge.db" --socket "$STATE/run/kw.sock"
```

The service listens on an owner-only Unix socket (dir 0700, socket 0600), never
on TCP. Example user-level service files are provided for review only; nothing
installs them automatically.

## Workers

```bash
kw-memory extract-worker --db "$STATE/kw/knowledge.db" --once \
  --command /path/to/external-model-client   # JSON on stdin/stdout; credentials stay in its own env
kw-memory embed-worker --db "$STATE/kw/knowledge.db" --once   # offline hash provider for smoke; real BGE later
kw-memory cleanup --db "$STATE/kw/knowledge.db" --dry-run      # retention report
```

## Export / import / validation

```bash
kw-memory export --db "$STATE/kw/knowledge.db" --out /path/to/snapshot
kw-memory validate-snapshot --snapshot /path/to/snapshot
kw-memory import --snapshot /path/to/snapshot --target /path/to/new-state
```

Import writes only into a new/empty directory, assigns a new `restore_epoch`,
and preserves the snapshot watermark for adapter reconciliation. See
`docs/operations/status.md` for status and recovery.

## Do not

* point an installation at production Hermes/DSH data or the system keychain;
* place `knowledge.db` on a network volume;
* pass credentials on CLI arguments or store them in config/logs.
