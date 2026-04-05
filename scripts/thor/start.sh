#!/usr/bin/env bash
# Start agent-memory on Jetson AGX Thor with llama.cpp backend.
#
# Supports 3 models via swap (one loaded at a time, max context):
#   - gemma-4-26b-a4b  (MoE, fast: 51 t/s gen)
#   - gemma-4-31b      (Dense, deep: 10 t/s gen)
#   - qwen3-coder-next (Coding specialist)
#
# Usage:
#   ./scripts/thor/start.sh                          # Default: MoE
#   ./scripts/thor/start.sh gemma-4-31b              # Start with dense model
#   ./scripts/thor/start.sh qwen3-coder-next         # Start with coder model
#
# Swap models at runtime:
#   curl -X POST http://localhost:8000/admin/models/swap \
#     -H "X-Admin-Key: $SEMANTIC_ADMIN_KEY" \
#     -H "Content-Type: application/json" \
#     -d '{"model_id": "gemma-4-31b"}'
#
# Or let Claude Code auto-swap by setting model in the request.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Default model (override with first argument)
DEFAULT_MODEL="${1:-gemma-4-26b-a4b}"

# llama-server binary (built from ~/llama.cpp-build)
LLAMA_SERVER="${LLAMA_SERVER:-$HOME/llama.cpp-build/build/bin/llama-server}"

# Admin key for /admin/* endpoints
export SEMANTIC_ADMIN_KEY="${SEMANTIC_ADMIN_KEY:-$(openssl rand -hex 16)}"

# Core settings
export SEMANTIC_BACKEND=llamacpp
export SEMANTIC_LLAMACPP_BASE_URL=http://127.0.0.1:8001
export SEMANTIC_LLAMACPP_SERVER_BINARY="$LLAMA_SERVER"
export SEMANTIC_LLAMACPP_DEFAULT_MODEL="$DEFAULT_MODEL"
export SEMANTIC_LLAMACPP_CACHE_TYPE_K=q8_0
export SEMANTIC_LLAMACPP_CACHE_TYPE_V=q8_0
export SEMANTIC_LLAMACPP_TIMEOUT_S=600
export SEMANTIC_LLAMACPP_AUTO_SWAP=false

# Agent cache settings
export SEMANTIC_AGENT_CACHE_DIR="$HOME/.agent_memory/caches"
export SEMANTIC_AGENT_MAX_AGENTS_IN_MEMORY=8
export SEMANTIC_AGENT_EVICT_TO_DISK=true

# Server settings
export SEMANTIC_SERVER_HOST=0.0.0.0
export SEMANTIC_SERVER_PORT=8000
export SEMANTIC_SERVER_LOG_LEVEL=INFO
export SEMANTIC_SERVER_SEARXNG_URL=http://localhost:8080
export SEMANTIC_SERVER_JINA_READER_URL=http://localhost:3000

# Ensure bandwidth is healthy (Thor memory controller bug)
echo "=== Thor agent-memory launcher ==="
echo "Model: $DEFAULT_MODEL"
echo "Server: $LLAMA_SERVER"
echo "Admin key: $SEMANTIC_ADMIN_KEY"
echo ""

# Check llama-server exists
if [ ! -x "$LLAMA_SERVER" ]; then
    echo "ERROR: llama-server not found at $LLAMA_SERVER"
    echo "Build it: cd ~/llama.cpp-build && git checkout b8665 && export PATH=\$HOME/.local/bin:\$PATH && cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=110 -DCMAKE_BUILD_TYPE=Release && cmake --build build -j\$(nproc) --target llama-server"
    exit 1
fi

# Check Python venv
if [ ! -d "$PROJECT_DIR/.venv" ]; then
    echo "ERROR: Python venv not found at $PROJECT_DIR/.venv"
    echo "Create it: cd $PROJECT_DIR && uv venv && uv pip install -e '.[dev]'"
    exit 1
fi

# Activate venv
source "$PROJECT_DIR/.venv/bin/activate"

echo "Starting agent-memory (llama-server will be started automatically)..."
echo "Swap models: curl -X POST http://localhost:8000/admin/models/swap -H 'X-Admin-Key: $SEMANTIC_ADMIN_KEY' -H 'Content-Type: application/json' -d '{\"model_id\": \"gemma-4-31b\"}'"
echo ""

exec python -m uvicorn agent_memory.entrypoints.api_server:create_app \
    --factory \
    --host "$SEMANTIC_SERVER_HOST" \
    --port "$SEMANTIC_SERVER_PORT" \
    --workers 1
