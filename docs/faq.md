# Frequently Asked Questions

Common questions about agent-memory.

## General Questions

### What is agent-memory?

agent-memory is a production-ready HTTP server for running MLX language models on Apple Silicon with persistent KV cache across sessions. It enables true multi-agent workflows with intelligent cache management.

### Why use agent-memory instead of LM Studio or Ollama?

Key differences:
- **Persistent KV Cache**: Conversations resume instantly without re-computing context
- **Multi-Agent Native**: Manage multiple independent caches simultaneously
- **Tool Calling**: Full Anthropic and OpenAI tool calling support
- **Block Pool Memory**: Budget-limited allocation with LRU eviction

### What makes the cache "semantic"?

The cache stores KV (key-value) tensors from transformer attention layers, capturing the semantic understanding of the conversation. This allows the model to resume with full context awareness.

## Requirements

### What hardware do I need?

- **Minimum**: Apple Silicon M1 with 16GB RAM
- **Recommended**: M2 Max/Ultra or M3 with 32GB+ RAM
- **Storage**: 10GB+ free space for models and caches

### Can I run this on Intel Mac?

No. The MLX framework requires Apple Silicon for Metal GPU acceleration.

### Can I run this on Linux or Windows?

No. MLX is exclusive to Apple Silicon. Consider using llama.cpp or vLLM for other platforms.

### Can I use Docker?

No. Docker doesn't support Metal GPU passthrough on macOS, which MLX requires.

## Installation & Setup

### How do I install agent-memory?

```bash
python3 -m venv venv
source venv/bin/activate
pip install -e .
agent-memory version
```

See [Installation Guide](installation.md) for details.

### Which Python version should I use?

Python 3.11+. Python 3.10 and earlier are not supported.

### Where are caches stored?

Default: `~/.agent_memory/caches/`

Configure via:
```bash
SEMANTIC_AGENT_CACHE_DIR=/custom/path
```

### How much disk space do caches use?

- **DeepSeek-Coder-V2-Lite**: ~100-200MB per agent (1000-token cache)
- **SmolLM2**: ~20-50MB per agent

Caches grow as conversations get longer.

## Models

### Which models are supported?

Currently tested:
- **Gemma 3 12B** (mlx-community/gemma-3-12b-it-4bit) - Default
- **DeepSeek-Coder-V2-Lite** (mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx)
- **SmolLM2** (mlx-community/SmolLM2-135M-Instruct) - Testing

Any MLX-compatible model can be added. See [Adding Models](developer/adding-models.md).

### Can I use Llama 3?

Yes! Any MLX model from Hugging Face Hub works. Example:

```bash
SEMANTIC_MLX_MODEL_ID=mlx-community/Meta-Llama-3-8B-Instruct
```

See [Adding Models](developer/adding-models.md) for adding new models.

### How do I switch models?

**Option 1**: Update `.env` and restart server
```bash
SEMANTIC_MLX_MODEL_ID=mlx-community/SmolLM2-135M-Instruct
agent-memory serve
```

**Option 2**: Use model hot-swap (zero downtime)
```bash
curl -X POST http://localhost:8000/admin/swap \
  -H "X-Admin-Key: your-admin-key" \
  -d '{"model_id": "mlx-community/SmolLM2-135M-Instruct"}'
```

### Why is Gemma 3 12B the default?

Gemma 3 12B offers the best balance of:
- Quality (12B parameters)
- Tool calling support
- Memory efficiency (4-bit quantization)
- Apple Silicon optimization

## Performance

### How fast is generation?

**DeepSeek-Coder-V2-Lite (M2 Max)**:
- First token: ~100-200ms
- Subsequent tokens: ~50-100ms each
- Throughput: 10-15 tokens/second

**SmolLM2 (M1)**:
- First token: ~50ms
- Subsequent tokens: ~20-40ms each
- Throughput: 25-30 tokens/second

### Why is my first request slow?

The first request loads the model (~5-10s for DeepSeek-Coder-V2-Lite). Subsequent requests are much faster.

### How much faster is cache reuse?

**40-60% faster** on session resume by avoiding re-prefill of system prompts and conversation history.

### Why is my server using lots of memory?

This is expected:
- **Model weights**: ~6-8GB (DeepSeek-Coder-V2-Lite)
- **KV cache**: ~4GB (default budget)
- **System overhead**: ~2GB

Total: ~12-14GB for production use.

Reduce with:
```bash
SEMANTIC_MLX_CACHE_BUDGET_MB=2048  # Reduce to 2GB
```

## Tool Calling

### Does this support tool calling?

Yes! Both Anthropic tool_use and OpenAI function calling formats.

### How does tool calling work?

Since MLX models don't natively support tool formats, we use prompt engineering:
1. Tool schemas are added to the system prompt
2. Model is instructed to output specific JSON format
3. We parse the JSON and convert to proper API format

### Which models work best for tools?

**DeepSeek-Coder-V2-Lite** works well with tool calling. **SmolLM2** has basic support but may be less reliable.

### Can I use parallel tool calls?

Yes, with OpenAI format:
```bash
# Model can invoke multiple functions simultaneously
"tool_calls": [
  {"function": {"name": "get_weather", ...}},
  {"function": {"name": "get_time", ...}}
]
```

## Multi-Agent Support

### How many agents can I run?

**In memory (hot tier)**: 5 agents (configurable)
**On disk (warm tier)**: Unlimited (storage-limited)

Configure:
```bash
SEMANTIC_AGENT_MAX_AGENTS_IN_MEMORY=10
```

### What happens when memory is full?

Least-recently-used (LRU) agents are evicted to disk automatically. They reload from disk on next use.

### Can agents share context?

No. Each agent has independent cache. For shared context, use same session_id.

### How do I create separate agents?

Use different session IDs:
```bash
# Agent 1
curl -H "X-Session-ID: agent-alice" http://localhost:8000/v1/messages ...

# Agent 2
curl -H "X-Session-ID: agent-bob" http://localhost:8000/v1/messages ...
```

## API Usage

### Which API formats are supported?

1. **Anthropic Messages API** (`/v1/messages`)
2. **OpenAI Chat Completions** (`/v1/chat/completions`)
3. **Direct Agent API** (`/v1/agents`)

All three share the same underlying engine.

### Are these APIs fully compatible?

**Mostly**, with some limitations:
- Tool calling uses prompt engineering (not native)
- Some advanced features may not be supported

See [API Reference](api-reference.md) for details.

### Can I use the OpenAI Python SDK?

Yes, with base_url override:
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"  # If SEMANTIC_API_KEY not set
)

response = client.chat.completions.create(
    model="gemma-3-12b-it-4bit",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

### Can I use the Anthropic Python SDK?

Not directly (they don't support base_url override). Use requests:
```python
import requests

response = requests.post(
    "http://localhost:8000/v1/messages",
    json={
        "model": "gemma-3-12b-it-4bit",
        "max_tokens": 200,
        "messages": [{"role": "user", "content": "Hello!"}]
    }
)
```

## Configuration

### Where should I put configuration?

Create `.env` file in project root:
```bash
SEMANTIC_MLX_MODEL_ID=mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx
SEMANTIC_MLX_CACHE_BUDGET_MB=4096
...
```

See [Configuration Guide](configuration.md) for all options.

### How do I enable authentication?

Generate API key:
```bash
python3 -c "import secrets; print(secrets.token_urlsafe(32))"
```

Add to `.env`:
```bash
SEMANTIC_API_KEY=your-generated-key-here
```

Use in requests:
```bash
curl -H "X-API-Key: your-generated-key-here" http://localhost:8000/v1/messages ...
```

### How do I run server in background?

**macOS (launchd)**:
See [Deployment Guide](deployment.md#background-process-management)

**Simple background**:
```bash
nohup agent-memory serve > agent-memory.log 2>&1 &
echo $! > agent-memory.pid
```

Stop:
```bash
kill $(cat agent-memory.pid)
```

## Troubleshooting

### Server fails to start

**Check**:
1. Model ID is correct: `pip show mlx-lm`
2. Port is available: `lsof -i :8000`
3. Python version: `python3 --version` (3.11+)
4. MLX installed: `python3 -c "import mlx"`

### Out of memory errors

**Reduce memory usage**:
```bash
SEMANTIC_MLX_CACHE_BUDGET_MB=2048  # Reduce cache
SEMANTIC_MLX_MAX_BATCH_SIZE=2      # Reduce batch size
# Or use smaller model
SEMANTIC_MLX_MODEL_ID=mlx-community/SmolLM2-135M-Instruct
```

### Cache not working

**Check**:
1. Same session_id across requests
2. Message history matches exactly
3. Cache directory is writable: `ls -la ~/.agent_memory/caches/`
4. Model tag hasn't changed

### Slow generation

**Check**:
1. Metal GPU available: `python3 -c "import mlx.core as mx; print(mx.metal.is_available())"`
2. System resources (Activity Monitor)
3. Not using swap memory

**Optimize**:
```bash
SEMANTIC_MLX_PREFILL_STEP_SIZE=1024  # Faster prefill
SEMANTIC_MLX_MAX_BATCH_SIZE=2         # Lower latency
```

### Tool calling not working

**Check**:
1. Tool schema is valid JSON Schema
2. Model supports instruction following (DeepSeek-Coder-V2-Lite works best)
3. Clear tool descriptions

**Try**:
- Add explicit instructions in user message
- Use DeepSeek-Coder-V2-Lite instead of SmolLM2
- Simplify tool schema

## Development

### How do I run tests?

```bash
# Unit tests only
pytest tests/unit/ -v

# Integration tests (no model loading)
pytest tests/integration/ -k "not WithModel" -v

# All tests (including model loading)
pytest tests/ -v
```

See [Testing Guide](testing.md).

### How do I add a new model?

1. Find MLX model on Hugging Face Hub
2. Update configuration
3. Start server and verify
4. Add tests

See [Adding Models](developer/adding-models.md) for detailed steps.

### Can I contribute?

Yes! Contributions welcome. Please:
1. Read [Architecture docs](architecture.md)
2. Follow code quality standards (ruff, mypy)
3. Add tests for new features
4. Update documentation

## Production Deployment

### Is this production-ready?

Yes! Version 1.0.0 includes:
- Comprehensive test coverage (370+ tests)
- Zero known critical bugs
- Complete documentation
- Prometheus metrics
- Health checks

### How do I monitor the server?

**Prometheus metrics**:
```bash
curl http://localhost:8000/metrics
```

**Health checks**:
```bash
curl http://localhost:8000/health
```

See [Monitoring section](user-guide.md#monitoring) in User Guide.

### What about security?

**Best practices**:
1. Enable API key authentication
2. Use reverse proxy (nginx) for TLS
3. Restrict CORS origins
4. Run behind firewall
5. Regular security updates

See [Deployment Guide](deployment.md#security).

### Can I run multiple instances?

Yes! Run on different ports:
```bash
# Instance 1
agent-memory serve --port 8000

# Instance 2
agent-memory serve --port 8001
```

Use load balancer to distribute requests.

## Comparison

### vs LM Studio?

**agent-memory** advantages:
- Persistent KV cache (not just text)
- Multi-agent native support
- Full tool calling support
- Programmatic API

**LM Studio** advantages:
- User-friendly GUI
- Multi-platform
- Broader model support

### vs Ollama?

**agent-memory** advantages:
- Persistent cache across restarts
- Block pool memory management
- Tool calling support
- Multiple API formats

**Ollama** advantages:
- Simpler setup
- Multi-platform
- Larger community

### vs llama.cpp?

**agent-memory** advantages:
- Apple Silicon optimized (MLX)
- Multi-agent orchestration
- Full API compatibility

**llama.cpp** advantages:
- Multi-platform
- More model formats
- Lower memory usage (gguf)

## Getting Help

### Where can I find documentation?

All documentation in `docs/`:
- [Quick Start](quick-start.md)
- [User Guide](user-guide.md)
- [Configuration](configuration.md)
- [Architecture](architecture.md)

### Where do I report bugs?

GitHub Issues (if repository is public)

### How do I get support?

1. Check this FAQ
2. Read relevant documentation
3. Search GitHub Issues
4. Create new issue with details

---

**Still have questions?** See [User Guide](user-guide.md) or check [GitHub Issues](https://github.com/yourusername/semantic-caching-api/issues).
