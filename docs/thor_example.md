# Thor Quick Start — Nemotron 3 Super 120B on Jetson AGX Thor

> **v1.1.0-alpha** — functional but not production-hardened.

Run agent-memory with Nemotron 3 Super (120B total, 12B active) on NVIDIA Jetson
AGX Thor, then connect NemoClaw/OpenClaw or Claude Code CLI.

## Architecture

```
Claude Code / NemoClaw
        │
        ▼
  agent-memory (Python)        ← session management, cache persistence,
  port 8000                       Anthropic + OpenAI API
        │
        ▼ HTTP
  vLLM server (Docker)         ← GPU inference, PagedAttention KV cache
  port 5000                       Nemotron 3 Super 120B-A12B NVFP4
        │
        ▼
  NVIDIA Jetson AGX Thor       ← 128GB unified memory, sm_110
```

agent-memory handles session IDs, conversation persistence to disk, prompt prefix
caching, and the Anthropic/OpenAI API translation. vLLM handles GPU inference with
its own PagedAttention KV cache.

## Requirements

- NVIDIA Jetson AGX Thor (128GB unified memory)
- JetPack 7.1+
- Docker with NVIDIA runtime
- ~30GB disk for model (NVFP4 quantized)
- Python 3.12+ (for agent-memory, via `uv`)

## 1. Download the model

SSH into Thor and download to shared HuggingFace cache:

```bash
ssh yshkolni@main4.local

# Install uv if not present
which uv || (wget -qO- https://astral.sh/uv/install.sh | sh)

# Create venv and download model
cd ~/agent-memory
source .venv/bin/activate
python3 -c "
from huggingface_hub import snapshot_download
path = snapshot_download(
    'nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4',
    cache_dir='/home/yshkolni/.cache/huggingface',
)
print(f'Downloaded to: {path}')
"
```

~30GB download. Model is cached at `~/.cache/huggingface/` and shared between
vLLM and any future Edge-LLM deployment.

## 2. Start vLLM server

### Option A: Use existing PyTorch container (faster, no new download)

If you already have a PyTorch container on Thor (e.g., `repnet/pytorch-triton`):

```bash
# Restart container with model cache mounted
docker stop triton_build 2>/dev/null; docker rm triton_build 2>/dev/null
docker run -d --name triton_build \
    --runtime nvidia --gpus all \
    -v /home/yshkolni/.cache/huggingface:/root/.cache/huggingface \
    -v /home/yshkolni/agent-memory:/workspace/agent-memory \
    -p 5000:5000 --network host --ipc=host \
    repnet/pytorch-triton:latest sleep infinity

# Install vLLM into the container
docker exec triton_build pip install vllm

# Start vLLM server
docker exec -d triton_build vllm serve \
    nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 \
    --port 5000 \
    --async-scheduling \
    --dtype auto \
    --kv-cache-dtype fp8 \
    --tensor-parallel-size 1 \
    --attention-backend TRITON_ATTN \
    --gpu-memory-utilization 0.8 \
    --max-num-seqs 32 \
    --enable-chunked-prefill \
    --host 0.0.0.0
```

### Option B: Pre-built vLLM container (~20-30GB download)

```bash
docker run -d \
    --name vllm-nemotron \
    --runtime nvidia --gpus all \
    -v /home/yshkolni/.cache/huggingface:/root/.cache/huggingface \
    -p 5000:5000 --ipc=host \
    nvcr.io/nvidia/vllm:26.02-py3 \
    vllm serve nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 \
        --port 5000 --host 0.0.0.0 \
        --async-scheduling --dtype auto --kv-cache-dtype fp8 \
        --gpu-memory-utilization 0.8 --enable-chunked-prefill
```

Wait for vLLM to load (~2-3 minutes). Verify:

```bash
curl http://localhost:5000/v1/models
# Should show: nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4
```

Expected generation speed: ~10.5 tokens/second.

**If you see `CUBLAS_STATUS_INVALID_VALUE` errors**: use the pre-built container
(not pip install). JetPack 7.1 includes CUDA/cuDNN — don't install them manually.

## 3. Start agent-memory

```bash
cd ~/agent-memory
source .venv/bin/activate

SEMANTIC_BACKEND=vllm \
SEMANTIC_VLLM_BASE_URL=http://localhost:5000 \
SEMANTIC_VLLM_MODEL_ID=nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 \
SEMANTIC_SERVER_PORT=8000 \
python -m uvicorn agent_memory.entrypoints.api_server:create_app \
    --factory --host 0.0.0.0 --port 8000
```

## 4. Verify it works

```bash
# Health check
curl http://localhost:8000/health/live
# Expected: {"status":"alive"}

# Generate (Anthropic Messages API)
curl -s http://localhost:8000/v1/messages \
    -H "Content-Type: application/json" \
    -d '{
        "model": "nemotron-3-super",
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "What is 2+2?"}]
    }' | python3 -m json.tool

# Generate with system prompt + tool definitions
curl -s http://localhost:8000/v1/messages \
    -H "Content-Type: application/json" \
    -d '{
        "model": "nemotron-3-super",
        "max_tokens": 128,
        "system": "You are a coding assistant. Use the provided tools.",
        "tools": [{"name": "read_file", "description": "Read a file", "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}}}],
        "messages": [{"role": "user", "content": "Read the file README.md"}]
    }' | python3 -m json.tool

# Generate (OpenAI API)
curl -s http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "nemotron-3-super",
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "Hello!"}]
    }' | python3 -m json.tool
```

## 5. Connect NemoClaw / OpenClaw

Add to `~/.openclaw/openclaw.json` on Thor:

```json
{
    "models": {
        "providers": {
            "agent-memory": {
                "baseUrl": "http://localhost:8000/v1",
                "apiKey": "local",
                "api": "anthropic-messages",
                "models": [
                    {
                        "id": "nemotron-3-super",
                        "name": "Nemotron 3 Super 120B (local)"
                    }
                ]
            }
        }
    },
    "agents": {
        "defaults": {
            "model": {"primary": "agent-memory/nemotron-3-super"}
        }
    }
}
```

Then:

```bash
openclaw agent --local -m "hello"
```

For NemoClaw with sandbox:

```bash
nemoclaw onboard  # One-time setup: creates sandbox, configures inference
```

## 6. Connect Claude Code CLI (remote from Mac)

Claude Code on your Mac → agent-memory on Thor → vLLM on Thor.

On Mac, add to `~/.claude.json`:

```json
{
    "hasCompletedOnboarding": true,
    "primaryApiKey": "sk-local"
}
```

Add to `~/.claude/settings.json`:

```json
{
    "env": {
        "ANTHROPIC_BASE_URL": "http://main4.local:8000",
        "ANTHROPIC_AUTH_TOKEN": "local",
        "ANTHROPIC_MODEL": "nemotron-3-super",
        "CLAUDE_CODE_ATTRIBUTION_HEADER": "0",
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        "MAX_THINKING_TOKENS": "0"
    },
    "permissions": {
        "allow": [
            "Bash(npm*)", "Bash(node*)", "Bash(python*)", "Bash(pip*)",
            "Bash(git*)", "Bash(ls*)", "Bash(cat*)", "Bash(find*)",
            "Bash(grep*)", "Bash(rg*)", "Bash(mkdir*)", "Bash(touch*)",
            "Read", "Write", "Edit", "Glob", "Grep"
        ]
    }
}
```

Then:

```bash
# Headless test
claude --bare -p "What is 2+2?" --output-format json --max-turns 3

# Interactive
claude
```

## 7. Multi-turn with session persistence

```bash
# Turn 1 (agent-memory caches conversation state)
curl -s http://localhost:8000/v1/messages \
    -H "Content-Type: application/json" \
    -H "X-Session-ID: my-session" \
    -d '{"model":"nemotron-3-super","max_tokens":64,"messages":[{"role":"user","content":"My name is Alice."}]}'

# Turn 2 (same session — conversation context preserved)
curl -s http://localhost:8000/v1/messages \
    -H "Content-Type: application/json" \
    -H "X-Session-ID: my-session" \
    -d '{"model":"nemotron-3-super","max_tokens":64,"messages":[{"role":"user","content":"What is my name?"}]}'
```

Note: With vLLM backend, session persistence saves conversation history to disk
(not raw KV tensors). If vLLM restarts, the next request does a full prefill
from the saved conversation. Within a running vLLM session, PagedAttention
provides native KV cache reuse.

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `SEMANTIC_BACKEND` | `mlx` | Set to `vllm` for Thor |
| `SEMANTIC_VLLM_BASE_URL` | `http://localhost:5000` | vLLM server URL |
| `SEMANTIC_VLLM_MODEL_ID` | `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4` | Model name |
| `SEMANTIC_VLLM_TIMEOUT_S` | `120` | HTTP timeout for vLLM requests |
| `SEMANTIC_SERVER_PORT` | `8000` | agent-memory server port |
| `SEMANTIC_AGENT_CACHE_DIR` | `~/.agent_memory/caches` | Disk cache directory |
| `SEMANTIC_AGENT_EVICTION_POLICY` | `lru-lfu` | Cache eviction policy |
| `SEMANTIC_ADMIN_KEY` | (none) | Admin API key |

## Cache Management

```bash
# View session cache files
ls -la ~/.agent_memory/caches/

# Clear all caches
curl -X DELETE http://localhost:8000/admin/caches \
    -H "X-Admin-Key: your-key"

# View memory/disk stats
curl http://localhost:8000/debug/memory
```

## Model Selection Guide

| Model | Total | Active | VRAM | Tool Use | NemoClaw | Speed |
|-------|-------|--------|------|----------|----------|-------|
| **Nemotron 3 Super 120B** | 120B | **12B** | ~30GB | **Native** | **Official** | ~10 tok/s |
| Nemotron 3 Nano 4B | 4B | 4B | ~8GB | Native | Official | ~40 tok/s |
| Qwen3.5-35B-A3B | 35B | 3B | ~3GB | Native | Supported | ~30 tok/s |
| SmolLM2-135M (test only) | 135M | 135M | ~270MB | No | No | ~130 tok/s |

## Backend Comparison

| Feature | vLLM (current) | Edge-LLM (future) |
|---------|---------------|-------------------|
| Nemotron 3 Super | ✅ Proven | ❌ Not yet supported |
| KV cache persist across restart | ❌ Reprefill needed | ✅ ~2ms restore |
| KV cache persist across eviction | ❌ Lost | ✅ Disk tier |
| Cross-session KV sharing | ❌ | ✅ System prompt cache |
| Setup complexity | Low (Docker) | High (C++ build) |
| Generation speed | ~10.5 tok/s | Similar |

When Edge-LLM adds Nemotron 3 Super support, switch to the TRT backend for
full KV cache persistence. The model files are shared — no re-download needed.

## Troubleshooting

**vLLM won't start**: Use pre-built container `nvcr.io/nvidia/vllm:26.02-py3`,
not pip install. JetPack 7.1 includes CUDA.

**`CUBLAS_STATUS_INVALID_VALUE`**: Don't install CUDA manually alongside JetPack.

**Slow first request**: vLLM prefills the full prompt on first request per session.
Subsequent turns in the same session reuse PagedAttention KV cache.

**Model not found**: Ensure `~/.cache/huggingface` is mounted in the Docker container
via `-v /home/yshkolni/.cache/huggingface:/root/.cache/huggingface`.

**agent-memory can't reach vLLM**: Check `SEMANTIC_VLLM_BASE_URL` matches vLLM's port.
Default vLLM port in our setup is 5000, agent-memory is 8000.
