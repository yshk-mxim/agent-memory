#!/usr/bin/env bash
# agent-memory — guided setup script
# Handles Python check, venv, pip install, HF login, model download, and smoke test.
set -euo pipefail

PYTHON_MIN="3.11"
DEFAULT_MODEL="mlx-community/gemma-3-12b-it-4bit"
HF_MODEL_PAGE="https://huggingface.co/google/gemma-3-12b-it"

info()  { printf "\033[1;34m[INFO]\033[0m  %s\n" "$*"; }
warn()  { printf "\033[1;33m[WARN]\033[0m  %s\n" "$*"; }
error() { printf "\033[1;31m[ERROR]\033[0m %s\n" "$*"; }
ok()    { printf "\033[1;32m[OK]\033[0m    %s\n" "$*"; }

# ── 1. Check Python version ──────────────────────────────────────────
info "Checking Python version..."
if ! command -v python3 &>/dev/null; then
    error "python3 not found. Install Python $PYTHON_MIN+ from python.org"
    exit 1
fi

PY_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_MAJOR=$(python3 -c "import sys; print(sys.version_info.major)")
PY_MINOR=$(python3 -c "import sys; print(sys.version_info.minor)")

if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 11 ]; }; then
    error "Python $PY_VERSION found, but $PYTHON_MIN+ is required."
    exit 1
fi
ok "Python $PY_VERSION"

# ── 2. Check/create virtualenv ───────────────────────────────────────
if [ -z "${VIRTUAL_ENV:-}" ]; then
    if [ -d ".venv" ]; then
        info "Activating existing .venv..."
        # shellcheck disable=SC1091
        source .venv/bin/activate
    else
        info "Creating virtual environment in .venv..."
        python3 -m venv .venv
        # shellcheck disable=SC1091
        source .venv/bin/activate
    fi
    ok "Virtual environment active: $VIRTUAL_ENV"
else
    ok "Already in virtualenv: $VIRTUAL_ENV"
fi

# ── 3. Install package ───────────────────────────────────────────────
info "Installing agent-memory with dev dependencies..."
pip install -e ".[dev]" --quiet
ok "Package installed"

# ── 4. Check HuggingFace token (Gemma 3 is gated) ────────────────────
info "Checking HuggingFace authentication..."
if command -v huggingface-cli &>/dev/null; then
    if huggingface-cli whoami &>/dev/null; then
        HF_USER=$(huggingface-cli whoami 2>/dev/null | head -1)
        ok "Logged in as: $HF_USER"
    else
        warn "Not logged in to HuggingFace."
        echo ""
        echo "  Gemma 3 12B is a gated model. To use it:"
        echo "  1. Accept the license at: $HF_MODEL_PAGE"
        echo "  2. Run: huggingface-cli login"
        echo "  3. Paste your token when prompted"
        echo ""
        read -rp "  Run huggingface-cli login now? [Y/n] " answer
        if [ "${answer:-Y}" != "n" ] && [ "${answer:-Y}" != "N" ]; then
            huggingface-cli login
        else
            warn "Skipping HF login. Model download may fail for gated models."
        fi
    fi
else
    warn "huggingface-cli not found. Install with: pip install huggingface_hub[cli]"
fi

# ── 5. Download default model ─────────────────────────────────────────
info "Downloading $DEFAULT_MODEL (~7 GB)..."
if command -v huggingface-cli &>/dev/null; then
    huggingface-cli download "$DEFAULT_MODEL" --quiet 2>/dev/null && \
        ok "Model downloaded" || \
        warn "Model download failed. You may need to accept the Gemma license or check your token."
else
    warn "huggingface-cli not available, skipping model download."
fi

# ── 6. Run smoke test ────────────────────────────────────────────────
info "Running unit tests..."
if python -m pytest tests/unit -x -q --timeout=30 2>/dev/null; then
    ok "All unit tests passed"
else
    warn "Some tests failed. Check output above."
fi

# ── 7. Done ──────────────────────────────────────────────────────────
echo ""
echo "============================================"
ok "Setup complete!"
echo "============================================"
echo ""
echo "  Start the server:"
echo "    python -m agent_memory.entrypoints.cli serve --port 8000"
echo ""
echo "  Health check:"
echo "    curl -sf http://localhost:8000/health/ready"
echo ""
echo "  Quick test:"
echo '    curl http://localhost:8000/v1/chat/completions \'
echo '      -H "Content-Type: application/json" \'
echo '      -d '\''{"model":"default","messages":[{"role":"user","content":"Hello"}],"max_tokens":50}'\'''
echo ""
