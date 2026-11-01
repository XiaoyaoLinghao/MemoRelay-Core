#!/bin/bash
set -e

echo "=== Knowledge Weaver Installation ==="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Create directories
mkdir -p /home/openclaw/.openclaw/knowledge /home/openclaw/.openclaw/logs

# Install package
cd "$SCRIPT_DIR/.."
pip install -e ".[dev]"

# Run first consolidation
python3 -c "
from knowledge_weaver.server import DB_PATH, MEMORY_DIR, get_embedder
from knowledge_weaver.pipeline import run_consolidation
e = get_embedder()
if e:
    result = run_consolidation(DB_PATH, MEMORY_DIR, e)
    print(f'Initial consolidation: {result.status}')
    print(f'  Files processed: {result.files_processed}')
    print(f'  Entities created: {result.entities_created}')
    print(f'  Entities updated: {result.entities_updated}')
else:
    print('WARNING: Embedder not configured. Set EMBEDDING_BASE_URL and EMBEDDING_API_KEY.')
    print('Consolidation skipped. Tools requiring embeddings will not work.')
"

echo ""
echo "=== Next Steps ==="
echo "1. Add MCP config from deploy/openclaw-mcp-config.json to your openclaw.json"
echo "2. Add crontab entry from deploy/crontab-entry"
echo "3. Disable Dreaming: openclaw config set plugins.entries.memory-core.config.dreaming.enabled false"
echo "4. Restart OpenClaw gateway"
echo ""
echo "Installation complete."
