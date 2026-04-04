#!/usr/bin/env bash
# Swap the currently loaded model.
#
# Usage:
#   ./scripts/thor/swap_model.sh gemma-4-31b      # Switch to dense
#   ./scripts/thor/swap_model.sh gemma-4-26b-a4b   # Switch to MoE
#   ./scripts/thor/swap_model.sh qwen3-coder-next  # Switch to coder

set -euo pipefail

MODEL_ID="${1:-}"
ADMIN_KEY="${SEMANTIC_ADMIN_KEY:-}"
BASE_URL="${SEMANTIC_SERVER_URL:-http://localhost:8000}"

if [ -z "$MODEL_ID" ]; then
    echo "Usage: $0 <model-id>"
    echo ""
    echo "Available models:"
    echo "  gemma-4-26b-a4b   - MoE (fast: 51 t/s gen, 1681 t/s pp)"
    echo "  gemma-4-31b       - Dense (deep: 10 t/s gen, 361 t/s pp)"
    echo "  qwen3-coder-next  - Coding specialist"
    echo ""
    echo "Current model:"
    curl -s "$BASE_URL/admin/models/current" -H "X-Admin-Key: $ADMIN_KEY" 2>/dev/null | python3 -m json.tool || echo "  (server not running)"
    exit 1
fi

if [ -z "$ADMIN_KEY" ]; then
    echo "ERROR: SEMANTIC_ADMIN_KEY not set"
    exit 1
fi

echo "Swapping to $MODEL_ID..."
curl -s -X POST "$BASE_URL/admin/models/swap" \
    -H "X-Admin-Key: $ADMIN_KEY" \
    -H "Content-Type: application/json" \
    -d "{\"model_id\": \"$MODEL_ID\", \"timeout_seconds\": 60}" | python3 -m json.tool
