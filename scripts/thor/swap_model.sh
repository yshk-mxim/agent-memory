#!/usr/bin/env bash
# Swap the currently loaded model on agent-memory.
#
# Usage:
#   ./scripts/thor/swap_model.sh                    # Interactive menu
#   ./scripts/thor/swap_model.sh gemma-4-31b        # Direct swap
#   ./scripts/thor/swap_model.sh 2                  # Pick by number

set -euo pipefail

BASE_URL="${SEMANTIC_SERVER_URL:-http://localhost:8000}"

MODELS=("gemma-4-26b-a4b" "gemma-4-31b" "qwen3-coder-next")
LABELS=(
    "gemma-4-26b-a4b   MoE   51 t/s   262K ctx   Fast interactive, research"
    "gemma-4-31b        Dense 10 t/s   131K ctx   Deep reasoning, architecture"
    "qwen3-coder-next   Hybrid         128K ctx   Coding specialist (SWE 70.6%)"
)

# --- Admin key ---
if [ -z "${SEMANTIC_ADMIN_KEY:-}" ]; then
    read -rsp "Admin key: " SEMANTIC_ADMIN_KEY
    echo ""
fi

# --- Show current model ---
CURRENT=$(curl -s "$BASE_URL/admin/models/current" \
    -H "X-Admin-Key: $SEMANTIC_ADMIN_KEY" 2>/dev/null \
    | python3 -c "import sys,json;print(json.load(sys.stdin).get('model_id','unknown'))" 2>/dev/null || echo "unknown")
echo "Currently loaded: $CURRENT"
echo ""

# --- Pick model ---
if [ -n "${1:-}" ]; then
    # Argument given: number or model name
    if [[ "$1" =~ ^[0-9]+$ ]] && [ "$1" -ge 1 ] && [ "$1" -le ${#MODELS[@]} ]; then
        MODEL="${MODELS[$(($1-1))]}"
    else
        MODEL="$1"
    fi
else
    echo "Available models:"
    for i in "${!MODELS[@]}"; do
        marker="  "
        [[ "${MODELS[$i]}" == "$CURRENT" ]] && marker="▸ "
        printf "  %s%d) %s\n" "$marker" $((i+1)) "${LABELS[$i]}"
    done
    echo ""
    read -rp "Select [1-${#MODELS[@]}]: " choice
    if [[ ! "$choice" =~ ^[0-9]+$ ]] || [ "$choice" -lt 1 ] || [ "$choice" -gt ${#MODELS[@]} ]; then
        echo "Invalid choice" >&2
        exit 1
    fi
    MODEL="${MODELS[$((choice-1))]}"
fi

if [ "$MODEL" = "$CURRENT" ]; then
    echo "$MODEL is already loaded."
    exit 0
fi

echo "Swapping to $MODEL..."
RESP=$(curl -s -w "\n%{http_code}" -X POST "$BASE_URL/admin/models/swap" \
    -H "X-Admin-Key: $SEMANTIC_ADMIN_KEY" \
    -H "Content-Type: application/json" \
    -d "{\"model_id\": \"$MODEL\", \"timeout_seconds\": 60}")

HTTP_CODE=$(echo "$RESP" | tail -1)
BODY=$(echo "$RESP" | sed '$d')

if [ "$HTTP_CODE" = "200" ]; then
    echo "✓ $MODEL is now active."
else
    echo "Failed (HTTP $HTTP_CODE):" >&2
    echo "$BODY" | python3 -m json.tool 2>/dev/null || echo "$BODY" >&2
    exit 1
fi
