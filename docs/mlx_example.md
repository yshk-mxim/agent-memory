# MLX Quick Start — Qwen3.5-9B on Apple Silicon

> **v1.1.0-alpha** — functional but not production-hardened.

Run agent-memory with Qwen3.5-9B on your Mac, then connect Claude Code CLI.

## Requirements

- Apple Silicon Mac (M1/M2/M3/M4)
- 16 GB RAM minimum (24 GB recommended)
- Python 3.11+
- mlx-lm >= 0.31.0 (installed automatically)
- ~5 GB disk for model download (automatic on first run)

## 1. Install

```bash
cd agent-memory
git checkout feat/trt-backend
pip install -e ".[dev]"
```

## 2. Start the server

```bash
SEMANTIC_MLX_MODEL_ID=mlx-community/Qwen3.5-9B-MLX-4bit \
SEMANTIC_MLX_KV_BITS=none \
python -m uvicorn agent_memory.entrypoints.api_server:create_app \
    --factory --host 0.0.0.0 --port 8000
```

The model downloads automatically from HuggingFace on first start (~5 GB).
Subsequent starts load from cache (`~/.cache/huggingface/`).

**Note:** `KV_BITS=none` uses FP16 KV cache. Q4 KV cache on mlx-lm 0.31
requires further testing. FP16 uses more memory but is fully verified.

**Note on Qwen3.5 thinking mode:** Qwen3.5 models enter thinking loops by
default, generating `<think>` tags endlessly. agent-memory strips thinking
tags from output, but tokens are still wasted. For Claude Code agentic use,
**Qwen3.5-35B-A3B** (MoE, 3B active) is recommended with thinking disabled.
Alternatively, use `Qwen2.5-14B-Instruct` for stable instruction following
(no tool calling) or `gemma-3-12b-it` (default, no thinking).

**For full agentic Claude Code (tool use, file read/write):** the recommended
setup is `llama.cpp` + `Qwen3.5-35B-A3B-GGUF:Q4_K_M` with
`--chat-template-kwargs '{"enable_thinking": false}'`. See the
[Unsloth guide](https://unsloth.ai/docs/basics/claude-code) for details.
agent-memory can sit in front of llama.cpp as a caching proxy.

## 3. Verify it works

```bash
# Health check
curl http://localhost:8000/health/live
# Expected: {"status":"alive"}

# Generate text (Anthropic Messages API)
curl -s http://localhost:8000/v1/messages \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen3.5-9B",
        "max_tokens": 32,
        "messages": [{"role": "user", "content": "What is 2+2? Answer briefly."}]
    }' | python3 -m json.tool

# Generate text (OpenAI API)
curl -s http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen3.5-9B",
        "max_tokens": 32,
        "messages": [{"role": "user", "content": "Hello!"}]
    }' | python3 -m json.tool

# With system prompt
curl -s http://localhost:8000/v1/messages \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen3.5-9B",
        "max_tokens": 32,
        "system": "You are a helpful assistant. Answer concisely.",
        "messages": [{"role": "user", "content": "What is a KV cache?"}]
    }' | python3 -m json.tool
```

## 4. Connect Claude Code CLI

### One-time setup (skip login screen)

Add to `~/.claude.json` (create if it doesn't exist):

```json
{
    "hasCompletedOnboarding": true,
    "primaryApiKey": "sk-local"
}
```

Add to `~/.claude/settings.json` (create if it doesn't exist):

```json
{
    "env": {
        "ANTHROPIC_BASE_URL": "http://localhost:8000",
        "ANTHROPIC_AUTH_TOKEN": "local",
        "ANTHROPIC_MODEL": "local-model",
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

**`CLAUDE_CODE_ATTRIBUTION_HEADER=0`** must be in `settings.json`, not as an env
var — it prevents a header that invalidates the KV cache with local models.

### Headless mode (scripted)

```bash
ANTHROPIC_BASE_URL=http://localhost:8000 \
ANTHROPIC_AUTH_TOKEN=local \
MAX_THINKING_TOKENS=0 \
claude --bare -p "What files are in this directory?" \
    --output-format json \
    --max-turns 3
```

### Interactive mode

```bash
claude
```

(Uses settings from `~/.claude/settings.json` — no env vars needed after setup.)

### What the server must support (all implemented)

- Streaming SSE (`stream: true`) with Anthropic event format
- Tool use (`tool_use` content blocks in responses)
- `X-Claude-Code-Session-Id` header → KV cache persistence across turns
- `/v1/messages` endpoint (Anthropic Messages API)
- `anthropic-version` header forwarded (accepted, no version gating)

## 5. Connect NemoClaw / OpenClaw

Add to `~/.openclaw/openclaw.json`:

```json
{
    "models": {
        "providers": {
            "agent-memory": {
                "baseUrl": "http://localhost:8000/v1",
                "apiKey": "local",
                "api": "anthropic-messages",
                "models": [
                    {"id": "Qwen3.5-9B", "name": "Qwen3.5-9B (local)"}
                ]
            }
        }
    },
    "agents": {
        "defaults": {
            "model": {"primary": "agent-memory/Qwen3.5-9B"}
        }
    }
}
```

Then: `openclaw agent --local -m "hello"`

## 6. Multi-turn with KV cache persistence

Use the `X-Session-ID` header to reuse KV cache across turns:

```bash
# Turn 1 (cold — creates cache)
curl -s http://localhost:8000/v1/messages \
    -H "Content-Type: application/json" \
    -H "X-Session-ID: my-session" \
    -d '{"model":"Qwen3.5-9B","max_tokens":32,"messages":[{"role":"user","content":"My name is Alice."}]}'

# Turn 2 (warm — reuses cached KV state, ~40% faster)
curl -s http://localhost:8000/v1/messages \
    -H "Content-Type: application/json" \
    -H "X-Session-ID: my-session" \
    -d '{"model":"Qwen3.5-9B","max_tokens":32,"messages":[{"role":"user","content":"What is my name?"}]}'
```

Cache files are saved to `SEMANTIC_AGENT_CACHE_DIR` (default `~/.agent_memory/caches/`).
They survive server restarts.

## Cache Benchmark Results

Tested on Apple Silicon with Qwen3.5-9B-MLX-4bit, FP16 KV cache:

| Test | Latency | Cache |
|------|---------|-------|
| Cold start (first request) | 1.185s | cache_create=12 |
| Hot cache (turn 2, same session) | 0.667s (44% faster) | cache_read=12 |
| Hot cache (turn 3) | 0.643s (46% faster) | cache_read=15 |
| New session (cold) | 0.650s | Independent cache |
| Long prompt (171 tokens) | 1.095s | Scales with input |
| Warm reuse after restart | 0.738s | cache_read=18 |
| Post-clear cold | 0.690s | Fresh cache |

## Cache Management

```bash
# View cache files
ls -la ~/.agent_memory/caches/

# Clear all caches (requires SEMANTIC_ADMIN_KEY)
curl -X DELETE http://localhost:8000/admin/caches \
    -H "X-Admin-Key: your-key"

# View memory stats
curl http://localhost:8000/debug/memory
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `SEMANTIC_MLX_MODEL_ID` | `mlx-community/gemma-3-12b-it-4bit` | HuggingFace model ID |
| `SEMANTIC_MLX_KV_BITS` | `4` | KV cache quantization (`none` for FP16) |
| `SEMANTIC_SERVER_PORT` | `8000` | Server port |
| `SEMANTIC_MLX_CACHE_BUDGET_MB` | `8192` | Max memory for KV caches |
| `SEMANTIC_AGENT_CACHE_DIR` | `~/.agent_memory/caches` | Disk cache directory |
| `SEMANTIC_AGENT_EVICTION_POLICY` | `lru-lfu` | Cache eviction: lru, lfu, lru-lfu |
| `SEMANTIC_AGENT_PIN_SYSTEM_PROMPT_CACHES` | `true` | Pin system prompt caches |
| `SEMANTIC_ADMIN_KEY` | (none) | Admin API key for cache management |

See `docs/configuration.md` for all settings.

## Supported MLX Models

Requires mlx-lm >= 0.31.0. Any `mlx-community` model with MLX weights works.

| Model | RAM | Instruction Following | Thinking |
|-------|-----|----------------------|----------|
| `mlx-community/Qwen3.5-2B-MLX-4bit` | 8GB | Basic | Yes (loops) |
| `mlx-community/Qwen3.5-4B-MLX-4bit` | 12GB | Good | Yes |
| **`mlx-community/Qwen3.5-9B-MLX-4bit`** | **16GB** | **Good** | **Yes** |
| `mlx-community/Qwen3.5-27B-Claude-4.6-Opus-Distilled-MLX-4bit` | 24GB | Best | Yes |
| `mlx-community/Qwen3.5-35B-A3B-4bit` | 16GB | Good (MoE, 3B active) | Yes |
| `mlx-community/gemma-3-12b-it-4bit` | 16GB | Good | No |

## Notes

- **Q4 KV cache**: Verified working on mlx-lm 0.31 (set `SEMANTIC_MLX_KV_BITS=4`).
  FP16 also works (`KV_BITS=none`). Q4 uses ~72% less memory.
- **Thinking mode**: Qwen3.5 models output `<think>...</think>` reasoning by default.
  agent-memory strips thinking tags automatically — clients receive only the final
  answer. Use `MAX_THINKING_TOKENS=0` in Claude Code to prevent the model from
  entering thinking mode.
- **Streaming**: Word-level chunked SSE (TRT) or per-token (MLX scheduler).
  True per-token streaming on TRT requires C++ engine modification.
- **Model swap**: Both backends offload all caches to SSD before swap. MLX reloads
  in-process; TRT requires server restart with new engine path. Old caches preserved
  on disk for rollback.
- **Hybrid cache**: Qwen3.5 uses KVCache (attention) + ArraysCache (Mamba/SSM).
  Both are saved/restored correctly. SSM state is per-layer (not block-split)
  which is architecturally correct.
