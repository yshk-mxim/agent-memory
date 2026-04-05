#!/usr/bin/env bash
# Swap the currently loaded model on agent-memory.
#
# Usage:
#   ./scripts/thor/swap_model.sh                    # Interactive menu
#   ./scripts/thor/swap_model.sh gemma-4-31b        # Direct swap
#   ./scripts/thor/swap_model.sh 2                  # Pick by number
#
# Uses python3+urllib (no curl dependency).

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

# --- HTTP helper (python3, no curl needed) ---
_http() {
    # Usage: _http GET /path  or  _http POST /path '{"json":"body"}'
    python3 -c "
import urllib.request, urllib.error, json, sys
method, path = sys.argv[1], sys.argv[2]
body = sys.argv[3].encode() if len(sys.argv) > 3 else None
req = urllib.request.Request(
    '${BASE_URL}' + path,
    data=body,
    headers={'X-Admin-Key': '${SEMANTIC_ADMIN_KEY}', 'Content-Type': 'application/json'},
    method=method,
)
try:
    with urllib.request.urlopen(req, timeout=90) as resp:
        data = json.loads(resp.read())
        print(json.dumps(data))
except urllib.error.HTTPError as e:
    print(json.dumps({'error': e.read().decode(), 'status': e.code}), file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(json.dumps({'error': str(e)}), file=sys.stderr)
    sys.exit(1)
" "$@"
}

# --- Show current model ---
CURRENT=$(_http GET /admin/models/current 2>/dev/null \
    | python3 -c "import sys,json;print(json.load(sys.stdin).get('model_id','unknown'))" 2>/dev/null \
    || echo "unknown")
echo "Currently loaded: $CURRENT"
echo ""

# --- Pick model ---
if [ -n "${1:-}" ]; then
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
if _http POST /admin/models/swap "{\"model_id\": \"$MODEL\", \"timeout_seconds\": 60}" > /dev/null; then
    echo "Done — $MODEL is now active."
else
    echo "Swap failed." >&2
    exit 1
fi
