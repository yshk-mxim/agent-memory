# Thor Setup Guide

Complete guide to setting up agent-memory on NVIDIA Jetson AGX Thor (128 GB unified memory, sm_110).

**Quickstart script**: `scripts/thor/quickstart.sh` automates steps 1–6 below.

## Prerequisites

| Requirement | Version | Check |
|-------------|---------|-------|
| JetPack | 7.1+ | `cat /etc/nv_tegra_release` |
| CUDA | 13.0+ | `nvcc --version` |
| Python | 3.10+ | `python3 --version` |
| cmake | 3.20+ | `cmake --version` |
| Docker | 20+ | `docker --version` |
| git | any | `git --version` |
| uv (recommended) | any | `pip install uv` |
| huggingface-cli | any | `pip install huggingface-hub[cli]` |
| Disk space | ~100 GB | For model GGUFs |

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│  Thor (Jetson AGX, aarch64, sm_110, 128 GB)                  │
│                                                              │
│  ┌──────────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │  llama-server     │  │  SearXNG      │  │  URL Reader   │  │
│  │  :8001            │  │  :8080        │  │  :3000        │  │
│  │  (GGUF inference) │  │  (Docker)     │  │  (systemd)    │  │
│  └────────┬─────────┘  └──────┬───────┘  └──────┬───────┘   │
│           │                   │                  │            │
│  ┌────────┴───────────────────┴──────────────────┴───────┐   │
│  │  agent-memory  :8000                                   │   │
│  │  (Anthropic API ← → OpenAI API translation)           │   │
│  │  + server-side tool execution (WebSearch, WebFetch)     │   │
│  │  + KV cache management (slot save/restore)              │   │
│  └────────────────────────┬──────────────────────────────┘   │
└───────────────────────────┼──────────────────────────────────┘
                            │ HTTP (LAN)
┌───────────────────────────┼──────────────────────────────────┐
│  Client machine           │                                   │
│  ┌────────────────────────┴──────────────────────────────┐   │
│  │  Claude Code CLI                                       │   │
│  │  ANTHROPIC_BASE_URL=http://<thor-ip>:8000             │   │
│  └───────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────┘
```

## Step 1: Build llama.cpp (sm_110)

llama.cpp is the inference backend. Build **b8665+** for Gemma 4 native tool calling support.

```bash
# Clone
git clone https://github.com/ggml-org/llama.cpp.git ~/llama.cpp-build
cd ~/llama.cpp-build
git checkout b8665  # minimum for Gemma 4 tool calling

# Build for sm_110 only (Thor's compute capability)
cmake -B build \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="110" \
    -DCMAKE_BUILD_TYPE=Release

cmake --build build --config Release -j$(nproc) --target llama-server

# Verify
./build/bin/llama-server --version
```

**Why sm_110 only?** Building for all architectures wastes compile time and disk. Thor is sm_110; no other arch will run on this device.

**Why b8665?** Builds b8661–b8665 contain critical Gemma 4 fixes: dedicated tool call parser, `final_logit_softcapping` from GGUF, tokenizer newline fix. See `docs/llamacpp_b8665_upgrade.md` for details.

## Step 2: Download Model GGUFs

All models use Q4_K_M quantization (~4.7 bits/weight). At 128 GB unified memory, one model loads at a time with maximum context.

```bash
mkdir -p ~/models

# Gemma 4 26B-A4B (MoE, ~16 GB) — fast: 51 t/s generation
huggingface-cli download bartowski/google_gemma-4-26B-A4B-it-GGUF \
    gemma-4-26B-A4B-it-Q4_K_M.gguf \
    --local-dir ~/models/gemma4-26b-a4b

# Gemma 4 31B (Dense, ~18 GB) — deep reasoning: 10 t/s generation
huggingface-cli download bartowski/google_gemma-4-31B-it-GGUF \
    gemma-4-31B-it-Q4_K_M.gguf \
    --local-dir ~/models/gemma4-31b

# Qwen3-Coder-Next (Hybrid MoE, ~46 GB) — coding specialist
huggingface-cli download bartowski/Qwen_Qwen3-Coder-480B-A35B-Instruct-GGUF \
    Qwen3-Coder-Next-Q4_K_M.gguf \
    --local-dir ~/models/qwen3-coder-next

# Qwen 3.5 27B Opus-Distilled (Dense, ~16 GB) — distilled reasoning
huggingface-cli download bartowski/Qwen_Qwen3.5-27B-GGUF \
    Qwen3.5-27B.Q4_K_M.gguf \
    --local-dir ~/models/qwen35-opus-distilled
```

### Memory budget (Q4_K_M weights, Q8 KV cache)

| Model | Weights | KV Cache | Slots | Context/Slot | Gen Speed |
|-------|:-------:|:--------:|:-----:|:------------:|:---------:|
| gemma-4-26b-a4b (MoE) | 16 GiB | 102 GiB | 4 | 262K | 51 t/s |
| gemma-4-31b (Dense) | 17 GiB | 101 GiB | 2 | 131K | 10 t/s |
| qwen35-opus-distilled | ~15 GiB | ~103 GiB | 2 | 131K | TBD |
| qwen3-coder-next | ~46 GiB | ~72 GiB | TBD | TBD | TBD |

## Step 3: Set up agent-memory

```bash
cd ~/agent-memory  # or wherever you cloned the repo

# Create venv (uv is faster, pip works too)
uv venv
uv pip install -e '.[dev]'

# Create cache directories
mkdir -p ~/.agent_memory/caches ~/.agent_memory/llamacpp_slots
```

## Step 4: Set up SearXNG (web search)

SearXNG is a self-hosted meta-search engine. The model uses it via the `WebSearch` tool — no external API keys needed.

```bash
mkdir -p ~/searxng

cat > ~/searxng/settings.yml << 'EOF'
use_default_settings: true

server:
  bind_address: "0.0.0.0"
  port: 8080
  secret_key: "REPLACE_ME"  # openssl rand -hex 32
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
EOF

# Replace secret key
sed -i "s/REPLACE_ME/$(openssl rand -hex 32)/" ~/searxng/settings.yml

# Start container (auto-restarts)
docker run -d --restart always \
    --name searxng \
    -p 8080:8080 \
    -v ~/searxng:/etc/searxng \
    searxng/searxng

# Verify
sleep 5
wget -qO- "http://localhost:8080/search?q=test&format=json" | python3 -m json.tool | head -5
```

## Step 5: Set up local URL reader

A lightweight Python server that fetches URLs and converts HTML to markdown. Replaces Jina Reader with zero external dependencies.

```bash
# Install html2text
pip3 install html2text --break-system-packages

# The reader script is included in this repo — copy it to home
cp scripts/thor/reader_server.py ~/reader_server.py
# Or the quickstart.sh script writes it automatically

# Create systemd user service for auto-start
mkdir -p ~/.config/systemd/user

cat > ~/.config/systemd/user/reader.service << EOF
[Unit]
Description=Local URL-to-markdown reader server (port 3000)
After=network.target

[Service]
ExecStart=/usr/bin/python3 $HOME/reader_server.py
Restart=always
RestartSec=5
StandardOutput=append:$HOME/reader_server.log
StandardError=append:$HOME/reader_server.log

[Install]
WantedBy=default.target
EOF

systemctl --user daemon-reload
systemctl --user enable reader
systemctl --user start reader

# Verify
wget -qO- "http://localhost:3000/https://example.com" | head -5
```

## Step 6: Start agent-memory

```bash
# Default model (Gemma 4 26B-A4B MoE, fastest)
scripts/thor/start.sh

# Or start with a specific model
scripts/thor/start.sh gemma-4-31b
scripts/thor/start.sh qwen3-coder-next
scripts/thor/start.sh qwen35-opus-distilled
```

The start script:
- Generates a random admin key for model swapping
- Starts llama-server with the selected model config
- Starts agent-memory (uvicorn) on port 8000
- Connects to SearXNG (port 8080) and URL reader (port 3000)

### Verify

```bash
# Health check
wget -qO- http://localhost:8000/health
# {"status":"ok"}

# List models
wget -qO- http://localhost:8000/v1/models
# Shows all available models with active flag

# Test inference
wget -qO- --post-data='{"model":"gemma-4-26b-a4b","max_tokens":50,"messages":[{"role":"user","content":"Hello"}]}' \
    --header='Content-Type: application/json' \
    http://localhost:8000/v1/messages
```

## Step 7: Configure Claude Code (client machine)

On the machine where you run Claude Code (Mac, Linux, etc.):

### First-time bypass

Claude Code normally requires an Anthropic API key. For local backends:

```bash
echo '{"hasCompletedOnboarding": true, "primaryApiKey": "sk-local"}' > ~/.claude.json
```

### Project settings

Create `.claude/settings.json` in your project directory:

```json
{
    "env": {
        "ANTHROPIC_BASE_URL": "http://<thor-ip>:8000",
        "ANTHROPIC_AUTH_TOKEN": "local",
        "ANTHROPIC_MODEL": "gemma-4-26b-a4b",
        "CLAUDE_CODE_ATTRIBUTION_HEADER": "0",
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        "MAX_THINKING_TOKENS": "0",
        "CLAUDE_CODE_MAX_CONTEXT_TOKENS": "100000"
    }
}
```

Replace `<thor-ip>` with your Thor's IP address (e.g., `192.168.1.100`). Find it with `hostname -I` on Thor.

| Variable | Purpose |
|----------|---------|
| `ANTHROPIC_BASE_URL` | Points Claude Code at agent-memory instead of Anthropic |
| `ANTHROPIC_AUTH_TOKEN` | Any non-empty string (no real auth needed) |
| `ANTHROPIC_MODEL` | Must match a model ID in `config/models/` |
| `CLAUDE_CODE_ATTRIBUTION_HEADER` | `0` disables per-request headers that break prompt caching |
| `CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC` | `1` prevents telemetry to Anthropic servers |
| `MAX_THINKING_TOKENS` | `0` for models without extended thinking |
| `CLAUDE_CODE_MAX_CONTEXT_TOKENS` | Should be ≤ per-slot context (262K for MoE, 131K for dense) |

## Operations

### Swap models at runtime

No restart needed — agent-memory stops the old llama-server and starts a new one (~10-15 seconds):

```bash
# Interactive
scripts/thor/swap_model.sh

# Direct
scripts/thor/swap_model.sh gemma-4-31b
scripts/thor/swap_model.sh 2  # by number
```

Remember to update `ANTHROPIC_MODEL` in Claude Code settings to match.

### Clear caches

```bash
# Clear KV caches (RAM + disk)
scripts/thor/clear_cache.sh

# Manual: clear everything
rm -rf ~/.agent_memory/caches/* ~/.agent_memory/llamacpp_slots/*
```

### Stop everything

```bash
scripts/thor/stop.sh
```

### Check service status

```bash
# agent-memory + llama-server
wget -qO- http://localhost:8000/health

# SearXNG
wget -qO- http://localhost:8080/healthz

# URL reader
wget -qO- http://localhost:3000/https://example.com | head -1

# All processes
ps aux | grep -E 'uvicorn|llama-server|searxng' | grep -v grep
```

## Troubleshooting

### Memory bandwidth degradation

Thor's memory bandwidth can drop from 228 GB/s to ~31 GB/s without warning. Symptoms: generation speed drops to ~7 t/s (MoE) or ~1.5 t/s (dense).

**Fix**: Reboot the device, then run `sudo jetson_clocks`:

```bash
sudo reboot
# After reboot:
sudo jetson_clocks
```

### WebSearch returns "0 searches" or 422 errors

- Check SearXNG is running: `docker ps | grep searxng`
- Check agent-memory was started with `scripts/thor/start.sh` (sets `SEMANTIC_SERVER_SEARXNG_URL`)
- If started manually, ensure `SEMANTIC_SERVER_SEARXNG_URL=http://localhost:8080` is exported

### llama-server crash / zombie process

```bash
# Kill everything and restart clean
scripts/thor/stop.sh
pkill -9 -f llama-server 2>/dev/null
scripts/thor/start.sh
```

### Model swap hangs

Check if the old llama-server is still running:
```bash
ps aux | grep llama-server | grep -v grep
# If stuck, kill it manually:
pkill -9 -f llama-server
```

### CUDA out of memory

Only one model loads at a time. If you see OOM:
1. Stop agent-memory: `scripts/thor/stop.sh`
2. Clear slot caches: `rm -rf ~/.agent_memory/llamacpp_slots/*`
3. Restart: `scripts/thor/start.sh`

### cuBLAS errors (vLLM / PyTorch)

cuBLAS is broken on sm_110 as of CUDA 13.0 — every `torch.mm()` fails with `CUBLAS_STATUS_INVALID_VALUE`. This affects all PyTorch-based inference (vLLM, TGI). **Use llama.cpp** — it has its own CUTLASS-based GEMM kernels and does not depend on cuBLAS. See `docs/vllm_coder_next_thor.md` for details.

## Environment variables reference

All variables used by `scripts/thor/start.sh`:

| Variable | Default | Description |
|----------|---------|-------------|
| `SEMANTIC_BACKEND` | `llamacpp` | Inference backend |
| `SEMANTIC_LLAMACPP_BASE_URL` | `http://127.0.0.1:8001` | llama-server address |
| `SEMANTIC_LLAMACPP_SERVER_BINARY` | `$HOME/llama.cpp-build/build/bin/llama-server` | Path to llama-server binary |
| `SEMANTIC_LLAMACPP_DEFAULT_MODEL` | `gemma-4-26b-a4b` | Model to load on startup |
| `SEMANTIC_LLAMACPP_CACHE_TYPE_K` | `q8_0` | KV cache key quantization |
| `SEMANTIC_LLAMACPP_CACHE_TYPE_V` | `q8_0` | KV cache value quantization |
| `SEMANTIC_LLAMACPP_TIMEOUT_S` | `600` | Request timeout (seconds) |
| `SEMANTIC_LLAMACPP_AUTO_SWAP` | `false` | Auto-swap models (disabled — causes ping-pong) |
| `SEMANTIC_AGENT_CACHE_DIR` | `$HOME/.agent_memory/caches` | Disk cache for agent KV states |
| `SEMANTIC_AGENT_MAX_AGENTS_IN_MEMORY` | `8` | Max concurrent agent contexts |
| `SEMANTIC_AGENT_EVICT_TO_DISK` | `true` | Evict least-recently-used to disk |
| `SEMANTIC_SERVER_HOST` | `0.0.0.0` | Listen address |
| `SEMANTIC_SERVER_PORT` | `8000` | Listen port |
| `SEMANTIC_SERVER_LOG_LEVEL` | `INFO` | Log level |
| `SEMANTIC_SERVER_SEARXNG_URL` | `http://localhost:8080` | SearXNG for WebSearch tool |
| `SEMANTIC_SERVER_JINA_READER_URL` | `http://localhost:3000` | URL reader for WebFetch tool |
| `SEMANTIC_ADMIN_KEY` | (random) | Admin API auth key |

## Port map

| Port | Service | Protocol |
|------|---------|----------|
| 8000 | agent-memory (Anthropic API) | HTTP |
| 8001 | llama-server (OpenAI API) | HTTP |
| 8080 | SearXNG (search) | HTTP |
| 3000 | URL reader (fetch) | HTTP |
