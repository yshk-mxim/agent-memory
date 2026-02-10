#!/usr/bin/env bash
# agent-memory — launch server + demo in one command
# Starts Gemma 3 12B with recommended settings, waits for readiness,
# then opens the Streamlit demo UI.
set -euo pipefail

PORT="${SEMANTIC_SERVER_PORT:-8000}"
STREAMLIT_PORT="${STREAMLIT_PORT:-8501}"
SERVER_URL="http://127.0.0.1:${PORT}"
SERVER_PID=""
DEMO_PID=""

info()  { printf "\033[1;34m[INFO]\033[0m  %s\n" "$*"; }
warn()  { printf "\033[1;33m[WARN]\033[0m  %s\n" "$*"; }
error() { printf "\033[1;31m[ERROR]\033[0m %s\n" "$*"; }
ok()    { printf "\033[1;32m[OK]\033[0m    %s\n" "$*"; }

cleanup() {
    info "Shutting down..."
    for pid in $SERVER_PID $DEMO_PID; do
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            kill -TERM "$pid" 2>/dev/null || true
        fi
    done
    sleep 3
    for pid in $SERVER_PID $DEMO_PID; do
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            kill -9 "$pid" 2>/dev/null || true
        fi
    done
    ok "Shutdown complete"
}
trap cleanup EXIT INT TERM

# ── Check for existing server on port ──────────────────────────────
if lsof -ti:"${PORT}" &>/dev/null; then
    error "Port ${PORT} is already in use. Kill the existing process or set SEMANTIC_SERVER_PORT."
    exit 1
fi

# ── Check Python environment ──────────────────────────────────────
if [ -z "${VIRTUAL_ENV:-}" ] && [ -d ".venv" ]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

python3 -c "import agent_memory" 2>/dev/null || python -c "import agent_memory" 2>/dev/null || {
    error "agent-memory not installed. Run: pip install -e . (or scripts/setup.sh)"
    exit 1
}
python3 -c "import streamlit" 2>/dev/null || {
    info "Installing streamlit..."
    pip3 install -r demo/requirements.txt --quiet
}

# ── Start server ───────────────────────────────────────────────────
info "Starting agent-memory server on port ${PORT}..."
info "Model: Gemma 3 12B IT Q4 (default)"
info "Settings: scheduler=on, batch=2, cache_budget=8192 MB, T=0.3 (hardcoded)"

SEMANTIC_MLX_SCHEDULER_ENABLED=true \
SEMANTIC_MLX_MAX_BATCH_SIZE=2 \
python3 -m agent_memory.entrypoints.cli serve --port "${PORT}" &
SERVER_PID=$!

# ── Wait for readiness ─────────────────────────────────────────────
info "Waiting for server to load model and become ready..."
MAX_WAIT=120
WAITED=0
while [ $WAITED -lt $MAX_WAIT ]; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        error "Server process died during startup. Check logs above."
        exit 1
    fi
    if curl -sf "${SERVER_URL}/health/ready" &>/dev/null; then
        break
    fi
    sleep 2
    WAITED=$((WAITED + 2))
    if [ $((WAITED % 10)) -eq 0 ]; then
        info "Still loading... (${WAITED}s)"
    fi
done

if [ $WAITED -ge $MAX_WAIT ]; then
    error "Server did not become ready within ${MAX_WAIT}s."
    exit 1
fi

ok "Server ready on ${SERVER_URL}"

# ── Quick health info ──────────────────────────────────────────────
MODEL_INFO=$(curl -sf "${SERVER_URL}/v1/models" 2>/dev/null | python3 -c "
import sys, json
data = json.load(sys.stdin)
models = data.get('data', [])
if models:
    m = models[0]
    print(f\"  Model: {m.get('id', 'unknown')}\")
    spec = m.get('spec', {})
    if spec:
        print(f\"  Layers: {spec.get('n_layers', '?')}, KV heads: {spec.get('n_kv_heads', '?')}, Head dim: {spec.get('head_dim', '?')}\")
" 2>/dev/null) || true
if [ -n "${MODEL_INFO:-}" ]; then
    echo "$MODEL_INFO"
fi

# ── Launch Streamlit demo ──────────────────────────────────────────
info "Starting Streamlit demo on port ${STREAMLIT_PORT}..."
streamlit run demo/app.py --server.port "${STREAMLIT_PORT}" --server.headless true &
DEMO_PID=$!

sleep 2
ok "Demo running at http://localhost:${STREAMLIT_PORT}"
echo ""
echo "============================================"
echo "  Server:  ${SERVER_URL}"
echo "  Demo UI: http://localhost:${STREAMLIT_PORT}"
echo "  Press Ctrl+C to stop both"
echo "============================================"
echo ""

# ── Wait for either process to exit ────────────────────────────────
wait -n "${SERVER_PID}" "${DEMO_PID}" 2>/dev/null || true
