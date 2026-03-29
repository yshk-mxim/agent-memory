# MLX Quick Start — Qwen3.5-9B on Apple Silicon

Run agent-memory with Qwen3.5-9B on your Mac, then connect Claude Code CLI.

## Requirements

- Apple Silicon Mac (M1/M2/M3/M4)
- 16 GB RAM minimum
- Python 3.11+
- ~5 GB disk for model download (automatic on first run)

## 1. Install

```bash
cd agent-memory
pip install -e ".[dev]"
```

## 2. Start the server

```bash
SEMANTIC_MLX_MODEL_ID=mlx-community/Qwen3.5-9B-MLX-4bit \
SEMANTIC_SERVER_PORT=8000 \
python -m uvicorn agent_memory.entrypoints.api_server:create_app \
    --factory --host 0.0.0.0 --port 8000
```

The model downloads automatically from HuggingFace on first start (~5 GB).
Subsequent starts load from cache (`~/.cache/huggingface/`).

## 3. Verify it works

```bash
# Health check
curl http://localhost:8000/health/live

# Generate text (Anthropic Messages API)
curl http://localhost:8000/v1/messages \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen3.5-9B",
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "What is 2+2? Answer in one word."}]
    }'

# Generate text (OpenAI API)
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen3.5-9B",
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "Hello!"}]
    }'
```

## 4. Connect Claude Code CLI

```bash
ANTHROPIC_BASE_URL=http://localhost:8000 \
ANTHROPIC_API_KEY=local \
DISABLE_TELEMETRY=1 \
CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1 \
MAX_THINKING_TOKENS=0 \
claude --bare -p "What files are in this directory?" \
    --output-format json \
    --max-turns 3
```

For interactive mode:

```bash
ANTHROPIC_BASE_URL=http://localhost:8000 \
ANTHROPIC_API_KEY=local \
MAX_THINKING_TOKENS=0 \
claude
```

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
# Turn 1
curl http://localhost:8000/v1/messages \
    -H "Content-Type: application/json" \
    -H "X-Session-ID: my-session" \
    -d '{"model":"Qwen3.5-9B","max_tokens":32,"messages":[{"role":"user","content":"My name is Alice."}]}'

# Turn 2 (reuses cached KV state from turn 1)
curl http://localhost:8000/v1/messages \
    -H "Content-Type: application/json" \
    -H "X-Session-ID: my-session" \
    -d '{"model":"Qwen3.5-9B","max_tokens":32,"messages":[{"role":"user","content":"What is my name?"}]}'
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `SEMANTIC_MLX_MODEL_ID` | `mlx-community/gemma-3-12b-it-4bit` | HuggingFace model ID |
| `SEMANTIC_SERVER_PORT` | `8000` | Server port |
| `SEMANTIC_MLX_KV_BITS` | `4` | KV cache quantization (4 or 8) |
| `SEMANTIC_MLX_CACHE_BUDGET_MB` | `8192` | Max memory for KV caches |
| `SEMANTIC_AGENT_CACHE_DIR` | `~/.agent_memory/caches` | Disk cache directory |
| `SEMANTIC_AGENT_EVICTION_POLICY` | `lru-lfu` | Cache eviction: lru, lfu, lru-lfu |

See `docs/configuration.md` for all settings.

## Supported MLX Models

Any model in `mlx-community` with MLX-4bit weights works. Tested:

| Model | RAM | Quality |
|-------|-----|---------|
| `mlx-community/Qwen3.5-2B-MLX-4bit` | 8GB | Basic |
| `mlx-community/Qwen3.5-4B-MLX-4bit` | 12GB | Good |
| **`mlx-community/Qwen3.5-9B-MLX-4bit`** | **16GB** | **Recommended** |
| `mlx-community/Qwen3.5-27B-Claude-4.6-Opus-Distilled-MLX-4bit` | 24GB | Best |
| `mlx-community/gemma-3-12b-it-4bit` | 16GB | Good (default) |
