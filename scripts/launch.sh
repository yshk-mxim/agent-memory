#!/usr/bin/env bash
# agent-memory — launch server + demo in one command
#
# Usage:
#   scripts/launch.sh                    # Gemma 3 12B (default) + demo UI
#   scripts/launch.sh --server-only      # server only, no Streamlit
#   scripts/launch.sh --stop             # stop running server + demo
#
#   # DeepSeek model:
#   SEMANTIC_MLX_MODEL_ID="mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx" \
#     SEMANTIC_MLX_CACHE_BUDGET_MB=4096 scripts/launch.sh
#
# Environment:
#   SEMANTIC_SERVER_PORT          Server port (default: 8000)
#   STREAMLIT_PORT                Demo UI port (default: 8501)
#   SEMANTIC_MLX_MODEL_ID         HuggingFace model ID (default: Gemma 3 12B Q4)
#   SEMANTIC_MLX_CACHE_BUDGET_MB  GPU memory budget for KV caches (default: 8192)
#   SEMANTIC_MLX_MAX_BATCH_SIZE   Max concurrent sequences (default: 2)
set -euo pipefail

PORT="${SEMANTIC_SERVER_PORT:-8000}"
STREAMLIT_PORT="${STREAMLIT_PORT:-8501}"
SERVER_URL="http://127.0.0.1:${PORT}"
SERVER_ONLY=false
STOP_ONLY=false
SERVER_PID=""
DEMO_PID=""

for arg in "$@"; do
    case "$arg" in
        --server-only) SERVER_ONLY=true ;;
        --stop) STOP_ONLY=true ;;
        --help|-h)
            sed -n '2,/^set /{ /^#/s/^# \{0,1\}//p; }' "$0"
            exit 0 ;;
        *) echo "Unknown option: $arg (try --help)"; exit 1 ;;
    esac
done

info()  { printf "\033[1;34m[INFO]\033[0m  %s\n" "$*"; }
warn()  { printf "\033[1;33m[WARN]\033[0m  %s\n" "$*"; }
error() { printf "\033[1;31m[ERROR]\033[0m %s\n" "$*"; }
ok()    { printf "\033[1;32m[OK]\033[0m    %s\n" "$*"; }

cleanup() {
    echo ""
    info "Shutting down..."
    # SIGTERM for graceful shutdown
    for pid in $SERVER_PID $DEMO_PID; do
        [ -n "$pid" ] && kill -TERM "$pid" 2>/dev/null || true
    done
    # Poll up to 5s for processes to exit
    local i=0
    while [ $i -lt 10 ]; do
        local alive=false
        for pid in $SERVER_PID $DEMO_PID; do
            if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
                alive=true
            fi
        done
        "$alive" || break
        sleep 0.5
        i=$((i + 1))
    done
    # Force kill stragglers
    for pid in $SERVER_PID $DEMO_PID; do
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            warn "Force killing PID $pid"
            kill -9 "$pid" 2>/dev/null || true
        fi
    done
    ok "Shutdown complete"
}
trap cleanup EXIT

# ── Resolve project root (works from any directory) ───────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# ── Helper: graceful stop of a process on a port ────────────────
stop_port() {
    local port=$1 label=$2
    local pid
    pid=$(lsof -ti:"${port}" 2>/dev/null) || return 0
    info "Stopping ${label} (PID ${pid}) on port ${port}..."
    kill -TERM "$pid" 2>/dev/null || true
    local i=0
    while [ $i -lt 10 ]; do
        if ! kill -0 "$pid" 2>/dev/null; then
            ok "${label} stopped gracefully"
            return 0
        fi
        sleep 0.5
        i=$((i + 1))
    done
    warn "${label} did not stop in 5s, force killing..."
    kill -9 "$pid" 2>/dev/null || true
    sleep 1
    ok "${label} stopped"
}

# ── Stop mode ────────────────────────────────────────────────────
if [ "$STOP_ONLY" = true ]; then
    FOUND=false
    if lsof -ti:"${PORT}" &>/dev/null; then
        FOUND=true
        stop_port "${PORT}" "Server"
    fi
    if lsof -ti:"${STREAMLIT_PORT}" &>/dev/null; then
        FOUND=true
        stop_port "${STREAMLIT_PORT}" "Streamlit demo"
    fi
    if [ "$FOUND" = false ]; then
        info "Nothing running on ports ${PORT} or ${STREAMLIT_PORT}."
    fi
    exit 0
fi

# ── Preflight: detect running instances ─────────────────────────
SERVER_RUNNING=false
DEMO_RUNNING=false
if lsof -ti:"${PORT}" &>/dev/null; then SERVER_RUNNING=true; fi
if lsof -ti:"${STREAMLIT_PORT}" &>/dev/null; then DEMO_RUNNING=true; fi

if [ "$SERVER_RUNNING" = true ] || [ "$DEMO_RUNNING" = true ]; then
    echo ""
    warn "Existing instances detected:"
    if [ "$SERVER_RUNNING" = true ]; then
        warn "  Server on port ${PORT} (PID $(lsof -ti:"${PORT}"))"
    fi
    if [ "$DEMO_RUNNING" = true ]; then
        warn "  Streamlit on port ${STREAMLIT_PORT} (PID $(lsof -ti:"${STREAMLIT_PORT}"))"
    fi
    echo ""
    printf "  Stop them and restart? [Y/n] "
    read -r answer </dev/tty
    case "${answer:-Y}" in
        [Yy]|[Yy]es|"")
            if [ "$SERVER_RUNNING" = true ]; then stop_port "${PORT}" "Server"; fi
            if [ "$DEMO_RUNNING" = true ]; then stop_port "${STREAMLIT_PORT}" "Streamlit demo"; fi
            echo ""
            ;;
        *)
            info "Aborted. Use --stop to shut down existing instances."
            exit 0
            ;;
    esac
fi

# Activate .venv if present and not already in a virtualenv
if [ -z "${VIRTUAL_ENV:-}" ] && [ -d ".venv" ]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

python3 -c "import agent_memory" 2>/dev/null || {
    error "agent-memory not installed. Run: pip install -e . (or scripts/setup.sh)"
    exit 1
}

if [ "$SERVER_ONLY" = false ]; then
    python3 -c "import streamlit" 2>/dev/null || {
        info "Installing demo dependencies..."
        pip3 install -r demo/requirements.txt --quiet
    }
fi

# ── Start server ──────────────────────────────────────────────────
MODEL_DISPLAY="${SEMANTIC_MLX_MODEL_ID:-Gemma 3 12B IT Q4 (default)}"
BATCH="${SEMANTIC_MLX_MAX_BATCH_SIZE:-2}"
BUDGET="${SEMANTIC_MLX_CACHE_BUDGET_MB:-8192}"

info "Starting server on port ${PORT}..."
info "Model: ${MODEL_DISPLAY}"
info "Scheduler: on | Batch: ${BATCH} | Cache: ${BUDGET} MB | T=0.3"

SEMANTIC_MLX_SCHEDULER_ENABLED=true \
SEMANTIC_MLX_MAX_BATCH_SIZE="${BATCH}" \
python3 -m agent_memory.entrypoints.cli serve --port "${PORT}" &
SERVER_PID=$!

# ── Wait for readiness ────────────────────────────────────────────
info "Loading model (typically 15-30s)..."
WAITED=0
while [ $WAITED -lt 120 ]; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        error "Server crashed during startup. Check logs above."
        exit 1
    fi
    if curl -sf "${SERVER_URL}/health/ready" &>/dev/null; then
        break
    fi
    sleep 2
    WAITED=$((WAITED + 2))
    [ $((WAITED % 10)) -eq 0 ] && info "Still loading... (${WAITED}s)"
done

if [ $WAITED -ge 120 ]; then
    error "Server did not become ready within 120s."
    exit 1
fi
ok "Server ready on ${SERVER_URL}"

# Print loaded model info
curl -sf "${SERVER_URL}/v1/models" 2>/dev/null | python3 -c "
import sys, json
try:
    m = json.load(sys.stdin).get('data', [{}])[0]
    mid, spec = m.get('id', ''), m.get('spec', {})
    if mid:
        q = f'Q{spec[\"kv_bits\"]}' if spec.get('kv_bits') else 'FP16'
        print(f'  Loaded: {mid} ({spec.get(\"n_layers\", \"?\")} layers, {q})')
except Exception:
    pass
" 2>/dev/null || true

# ── Server-only mode ──────────────────────────────────────────────
if [ "$SERVER_ONLY" = true ]; then
    echo ""
    ok "Server running (--server-only, no demo UI)"
    echo "  Health:  curl ${SERVER_URL}/health/ready"
    echo "  Ctrl+C to stop"
    echo ""
    wait "$SERVER_PID" 2>/dev/null || true
    exit 0
fi

# ── Launch Streamlit demo ─────────────────────────────────────────
info "Starting demo UI on port ${STREAMLIT_PORT}..."
streamlit run demo/app.py \
    --server.port "${STREAMLIT_PORT}" \
    --server.headless true \
    --server.runOnSave false &
DEMO_PID=$!
sleep 2

if ! kill -0 "$DEMO_PID" 2>/dev/null; then
    error "Streamlit failed to start. Check logs above."
    exit 1
fi

ok "Demo UI running at http://localhost:${STREAMLIT_PORT}"
echo ""
echo "============================================"
echo "  Server:  ${SERVER_URL}"
echo "  Demo:    http://localhost:${STREAMLIT_PORT}"
echo "  Ctrl+C to stop both"
echo "============================================"
echo ""

# Wait for either child to exit — cleanup runs via EXIT trap
wait "$SERVER_PID" "$DEMO_PID" 2>/dev/null || true
