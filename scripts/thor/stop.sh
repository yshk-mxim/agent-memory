#!/usr/bin/env bash
# Stop agent-memory and any managed llama-server processes.
#
# Usage:
#   ./scripts/thor/stop.sh

set -euo pipefail

echo "Stopping agent-memory..."

# Kill uvicorn (agent-memory server)
pkill -f "uvicorn agent_memory.entrypoints.api_server" 2>/dev/null && \
    echo "  agent-memory stopped" || echo "  agent-memory not running"

# Kill any llama-server started by agent-memory
pkill -f "llama-server.*--port 8001" 2>/dev/null && \
    echo "  llama-server stopped" || echo "  llama-server not running"

echo "Done."
