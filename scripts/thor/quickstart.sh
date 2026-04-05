#!/usr/bin/env bash
# Quickstart: set up agent-memory on NVIDIA Jetson AGX Thor from scratch.
#
# Run this ON the Thor device. It will:
#   1. Build llama.cpp from source (sm_110)
#   2. Download model GGUFs (Q4_K_M quantization)
#   3. Set up agent-memory Python environment
#   4. Install SearXNG (Docker) for web search
#   5. Install local URL reader (replaces Jina Reader)
#   6. Create systemd user services for reader
#   7. Print Claude Code client settings
#
# Prerequisites:
#   - NVIDIA Jetson AGX Thor with JetPack 7.1+
#   - CUDA 13.0+ (comes with JetPack)
#   - Docker installed and running
#   - cmake >= 3.20, git, python3 (3.10+), uv (pip install uv)
#   - ~100 GB free disk for models
#   - huggingface-cli installed (pip install huggingface-hub[cli])
#
# Usage:
#   ./scripts/thor/quickstart.sh              # Full setup
#   ./scripts/thor/quickstart.sh --skip-models # Skip model downloads
#   ./scripts/thor/quickstart.sh --skip-docker # Skip SearXNG Docker setup
#   ./scripts/thor/quickstart.sh --only-models # Only download models

set -euo pipefail

# ============================================================================
# Configuration — edit these if your paths differ
# ============================================================================

LLAMA_CPP_DIR="$HOME/llama.cpp-build"
LLAMA_CPP_TAG="b8665"                    # Minimum for Gemma 4 tool calling
MODELS_DIR="$HOME/models"
AGENT_MEMORY_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
READER_SCRIPT="$HOME/reader_server.py"
SEARXNG_DIR="$HOME/searxng"
CACHE_DIR="$HOME/.agent_memory"

# Models to download (HuggingFace repo → local dir → filename)
declare -A MODEL_REPOS=(
    ["gemma4-26b-a4b"]="bartowski/google_gemma-4-26B-A4B-it-GGUF"
    ["gemma4-31b"]="bartowski/google_gemma-4-31B-it-GGUF"
    ["qwen3-coder-next"]="bartowski/Qwen_Qwen3-Coder-480B-A35B-Instruct-GGUF"
    ["qwen35-opus-distilled"]="bartowski/Qwen_Qwen3.5-27B-GGUF"
)
declare -A MODEL_FILES=(
    ["gemma4-26b-a4b"]="gemma-4-26B-A4B-it-Q4_K_M.gguf"
    ["gemma4-31b"]="gemma-4-31B-it-Q4_K_M.gguf"
    ["qwen3-coder-next"]="Qwen3-Coder-Next-Q4_K_M.gguf"
    ["qwen35-opus-distilled"]="Qwen3.5-27B.Q4_K_M.gguf"
)

# ============================================================================
# Parse arguments
# ============================================================================

SKIP_MODELS=false
SKIP_DOCKER=false
ONLY_MODELS=false

for arg in "$@"; do
    case $arg in
        --skip-models) SKIP_MODELS=true ;;
        --skip-docker) SKIP_DOCKER=true ;;
        --only-models) ONLY_MODELS=true ;;
        --help|-h)
            head -28 "$0" | tail -24
            exit 0
            ;;
        *) echo "Unknown argument: $arg"; exit 1 ;;
    esac
done

# ============================================================================
# Helpers
# ============================================================================

info()  { echo -e "\033[1;34m==>\033[0m $*"; }
ok()    { echo -e "\033[1;32m ✓\033[0m $*"; }
warn()  { echo -e "\033[1;33m !\033[0m $*"; }
fail()  { echo -e "\033[1;31m ✗\033[0m $*"; exit 1; }

check_command() {
    command -v "$1" &>/dev/null || fail "$1 not found. Install it first."
}

# ============================================================================
# Preflight checks
# ============================================================================

info "Checking prerequisites..."

check_command git
check_command cmake
check_command python3
check_command nvcc

# Verify we're on aarch64 (Jetson)
ARCH=$(uname -m)
if [ "$ARCH" != "aarch64" ]; then
    warn "Expected aarch64 (Jetson), got $ARCH. Continuing anyway..."
fi

# Check CUDA compute capability
CUDA_VERSION=$(nvcc --version | grep -oP 'release \K[\d.]+')
info "CUDA version: $CUDA_VERSION"

# Check available memory
TOTAL_MEM_GB=$(awk '/MemTotal/ {printf "%.0f", $2/1024/1024}' /proc/meminfo)
info "Total memory: ${TOTAL_MEM_GB} GB"
if [ "$TOTAL_MEM_GB" -lt 64 ]; then
    warn "Less than 64 GB RAM — some models may not fit with full context"
fi

ok "Prerequisites OK"

if [ "$ONLY_MODELS" = true ]; then
    # Jump straight to model downloads
    info "Downloading models only..."
    # (fall through to model download section below)
fi

# ============================================================================
# Step 1: Build llama.cpp
# ============================================================================

if [ "$ONLY_MODELS" = false ]; then

info "Step 1/6: Building llama.cpp ($LLAMA_CPP_TAG) for sm_110..."

if [ -d "$LLAMA_CPP_DIR" ]; then
    cd "$LLAMA_CPP_DIR"
    CURRENT_TAG=$(git describe --tags --exact-match 2>/dev/null || echo "unknown")
    if [ "$CURRENT_TAG" = "$LLAMA_CPP_TAG" ]; then
        ok "llama.cpp already at $LLAMA_CPP_TAG"
    else
        info "Updating from $CURRENT_TAG to $LLAMA_CPP_TAG..."
        git fetch --tags
        git checkout "$LLAMA_CPP_TAG"
    fi
else
    git clone https://github.com/ggml-org/llama.cpp.git "$LLAMA_CPP_DIR"
    cd "$LLAMA_CPP_DIR"
    git checkout "$LLAMA_CPP_TAG"
fi

cmake -B build \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="110" \
    -DCMAKE_BUILD_TYPE=Release

cmake --build build --config Release -j"$(nproc)" --target llama-server

# Verify
"$LLAMA_CPP_DIR/build/bin/llama-server" --version || fail "llama-server build failed"
ok "llama-server built at $LLAMA_CPP_DIR/build/bin/llama-server"

fi  # ONLY_MODELS

# ============================================================================
# Step 2: Download models
# ============================================================================

if [ "$SKIP_MODELS" = false ]; then

info "Step 2/6: Downloading model GGUFs to $MODELS_DIR..."

check_command huggingface-cli

mkdir -p "$MODELS_DIR"

for model_key in "${!MODEL_REPOS[@]}"; do
    repo="${MODEL_REPOS[$model_key]}"
    file="${MODEL_FILES[$model_key]}"
    dest_dir="$MODELS_DIR/$model_key"
    dest_file="$dest_dir/$file"

    if [ -f "$dest_file" ]; then
        ok "$model_key already downloaded ($(du -h "$dest_file" | cut -f1))"
        continue
    fi

    info "Downloading $model_key from $repo..."
    mkdir -p "$dest_dir"
    huggingface-cli download "$repo" "$file" \
        --local-dir "$dest_dir" \
        --local-dir-use-symlinks False
    ok "$model_key downloaded ($(du -h "$dest_file" | cut -f1))"
done

fi  # SKIP_MODELS

if [ "$ONLY_MODELS" = true ]; then
    ok "Model downloads complete."
    exit 0
fi

# ============================================================================
# Step 3: Set up agent-memory Python environment
# ============================================================================

info "Step 3/6: Setting up agent-memory Python environment..."

cd "$AGENT_MEMORY_DIR"

if [ -d ".venv" ]; then
    ok "Python venv already exists"
else
    if command -v uv &>/dev/null; then
        uv venv
        uv pip install -e '.[dev]'
    else
        python3 -m venv .venv
        .venv/bin/pip install -e '.[dev]'
    fi
    ok "Python venv created and packages installed"
fi

# Create cache directories
mkdir -p "$CACHE_DIR/caches" "$CACHE_DIR/llamacpp_slots"
ok "Cache directories created at $CACHE_DIR"

# ============================================================================
# Step 4: Set up SearXNG (Docker)
# ============================================================================

if [ "$SKIP_DOCKER" = false ]; then

info "Step 4/6: Setting up SearXNG web search (Docker)..."

check_command docker

if docker ps --format '{{.Names}}' | grep -q '^searxng$'; then
    ok "SearXNG container already running"
else
    mkdir -p "$SEARXNG_DIR"

    if [ ! -f "$SEARXNG_DIR/settings.yml" ]; then
        SECRET_KEY=$(openssl rand -hex 32)
        cat > "$SEARXNG_DIR/settings.yml" << SXEOF
use_default_settings: true

server:
  bind_address: "0.0.0.0"
  port: 8080
  secret_key: "$SECRET_KEY"
  limiter: false
  image_proxy: false

search:
  safe_search: 0
  formats:
    - html
    - json

engines:
  - name: google
    engine: google
    shortcut: g
    weight: 2
  - name: duckduckgo
    engine: duckduckgo
    shortcut: d
    weight: 1
  - name: bing
    engine: bing
    shortcut: b
    weight: 1
SXEOF
        ok "SearXNG config written"
    fi

    docker run -d --restart always \
        --name searxng \
        -p 8080:8080 \
        -v "$SEARXNG_DIR:/etc/searxng" \
        searxng/searxng

    ok "SearXNG started on port 8080"
fi

else
    warn "Skipping SearXNG Docker setup (--skip-docker)"
fi  # SKIP_DOCKER

# ============================================================================
# Step 5: Set up local URL reader (replaces Jina Reader)
# ============================================================================

info "Step 5/6: Setting up local URL-to-markdown reader..."

# Install html2text if missing
python3 -c "import html2text" 2>/dev/null || pip3 install html2text --break-system-packages

# Write reader script
cat > "$READER_SCRIPT" << 'PYEOF'
#!/usr/bin/env python3
"""Lightweight local URL-to-markdown server. Mimics r.jina.ai interface.

Usage: GET http://localhost:3000/https://example.com/page
"""
import sys
import html2text
import urllib.request
import urllib.error
from http.server import BaseHTTPRequestHandler, HTTPServer

TIMEOUT = 30
PORT = 3000

h2t = html2text.HTML2Text()
h2t.ignore_links = False
h2t.ignore_images = True
h2t.body_width = 0  # no line wrapping


class ReaderHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        print(f"[reader] {fmt % args}", file=sys.stderr)

    def do_GET(self):
        target = self.path.lstrip("/")
        if not target.startswith("http"):
            self.send_error(400, "URL must start with http(s)://")
            return

        try:
            req = urllib.request.Request(
                target,
                headers={
                    "User-Agent": "Mozilla/5.0 (compatible; local-reader/1.0)",
                    "Accept": "text/html,application/xhtml+xml,*/*",
                },
            )
            with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
                content_type = resp.headers.get("Content-Type", "")
                body = resp.read()
        except urllib.error.HTTPError as e:
            self.send_error(e.code, str(e))
            return
        except Exception as e:
            self.send_error(502, str(e))
            return

        if "text/html" not in content_type and "xhtml" not in content_type:
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.end_headers()
            self.wfile.write(body)
            return

        markdown = h2t.handle(body.decode(errors="replace"))
        encoded = markdown.encode()

        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


if __name__ == "__main__":
    server = HTTPServer(("0.0.0.0", PORT), ReaderHandler)
    print(f"[reader] listening on :{PORT}", file=sys.stderr)
    server.serve_forever()
PYEOF
chmod +x "$READER_SCRIPT"
ok "Reader script written to $READER_SCRIPT"

# ============================================================================
# Step 6: Create systemd user services
# ============================================================================

info "Step 6/6: Creating systemd user services..."

SYSTEMD_DIR="$HOME/.config/systemd/user"
mkdir -p "$SYSTEMD_DIR"

# Reader service
cat > "$SYSTEMD_DIR/reader.service" << SVCEOF
[Unit]
Description=Local URL-to-markdown reader server (port 3000)
After=network.target

[Service]
ExecStart=/usr/bin/python3 $READER_SCRIPT
Restart=always
RestartSec=5
StandardOutput=append:$HOME/reader_server.log
StandardError=append:$HOME/reader_server.log

[Install]
WantedBy=default.target
SVCEOF

systemctl --user daemon-reload
systemctl --user enable reader
systemctl --user start reader 2>/dev/null || true
ok "Reader systemd service enabled (port 3000)"

# ============================================================================
# Summary
# ============================================================================

THOR_IP=$(hostname -I | awk '{print $1}')

echo ""
echo "============================================================"
echo "  Setup complete!"
echo "============================================================"
echo ""
echo "Start agent-memory:"
echo "  $AGENT_MEMORY_DIR/scripts/thor/start.sh                # Default: gemma-4-26b-a4b"
echo "  $AGENT_MEMORY_DIR/scripts/thor/start.sh gemma-4-31b    # Dense model"
echo ""
echo "Services running:"
echo "  SearXNG:  http://localhost:8080 (Docker, auto-restart)"
echo "  Reader:   http://localhost:3000 (systemd user service)"
echo ""
echo "Claude Code client settings (.claude/settings.json):"
echo ""
cat << JSONEOF
{
    "env": {
        "ANTHROPIC_BASE_URL": "http://${THOR_IP}:8000",
        "ANTHROPIC_AUTH_TOKEN": "local",
        "ANTHROPIC_MODEL": "gemma-4-26b-a4b",
        "CLAUDE_CODE_ATTRIBUTION_HEADER": "0",
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        "MAX_THINKING_TOKENS": "0",
        "CLAUDE_CODE_MAX_CONTEXT_TOKENS": "100000"
    }
}
JSONEOF
echo ""
echo "First-time Claude Code bypass (run on client machine):"
echo '  echo '\''{"hasCompletedOnboarding": true, "primaryApiKey": "sk-local"}'\'' > ~/.claude.json'
echo ""
echo "Swap models at runtime:"
echo "  $AGENT_MEMORY_DIR/scripts/thor/swap_model.sh"
echo ""
echo "Stop everything:"
echo "  $AGENT_MEMORY_DIR/scripts/thor/stop.sh"
echo "============================================================"
