```markdown
# Configuration Guide

Semantic Caching API uses environment variables for configuration. All settings can be specified via:

- Environment variables
- `.env` file in the project root
- Command-line arguments (for `semantic serve`)

## Configuration Files

### .env File

Create a `.env` file in your project root:

```bash
# MLX Model Configuration
SEMANTIC_MLX_MODEL_ID=mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx
SEMANTIC_MLX_CACHE_BUDGET_MB=4096
SEMANTIC_MLX_MAX_BATCH_SIZE=5

# Agent Cache Configuration
SEMANTIC_AGENT_MAX_AGENTS_IN_MEMORY=5
SEMANTIC_AGENT_CACHE_DIR=~/.semantic/caches

# Server Configuration
SEMANTIC_SERVER_HOST=0.0.0.0
SEMANTIC_SERVER_PORT=8000
SEMANTIC_SERVER_LOG_LEVEL=INFO

# Security (Optional)
SEMANTIC_API_KEY=your-api-key-here
SEMANTIC_ADMIN_KEY=your-admin-key-here
```

## MLX Settings

Control MLX inference engine behavior.

### Environment Variables

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `SEMANTIC_MLX_MODEL_ID` | string | `mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx` | HuggingFace model ID or local path |
| `SEMANTIC_MLX_MAX_BATCH_SIZE` | int | `5` | Maximum concurrent sequences (1-20) |
| `SEMANTIC_MLX_PREFILL_STEP_SIZE` | int | `512` | Tokens per prefill step (128-2048) |
| `SEMANTIC_MLX_KV_BITS` | int\|null | `null` | KV cache quantization (4, 8, or null for FP16) |
| `SEMANTIC_MLX_BLOCK_TOKENS` | int | `256` | Tokens per cache block (64-512) |
| `SEMANTIC_MLX_CACHE_BUDGET_MB` | int | `4096` | Maximum cache memory budget in MB (512-16384) |
| `SEMANTIC_MLX_DEFAULT_MAX_TOKENS` | int | `256` | Default max tokens for generation (1-8192) |
| `SEMANTIC_MLX_DEFAULT_TEMPERATURE` | float | `0.7` | Default sampling temperature (0.0-2.0) |

### Supported Models

**Production-Ready**:
- `mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx` (default, 16B model, 163K context)
- `mlx-community/SmolLM2-135M-Instruct` (lightweight, testing)

**Memory Requirements**:
- DeepSeek-Coder-V2-Lite: 20GB+ RAM recommended (163K context support)
- SmolLM2: 4GB+ RAM sufficient

### Cache Budget Calculation

```python
# Example: DeepSeek-Coder-V2-Lite with 4GB cache budget
bytes_per_block = model_spec.bytes_per_block_per_layer()
total_blocks = (4096 * 1024 * 1024) // bytes_per_block
# ~1400 blocks for DeepSeek-Coder-V2-Lite (each block = 256 tokens)
```

## Agent Settings

Control agent cache lifecycle and persistence.

### Environment Variables

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `SEMANTIC_AGENT_MAX_AGENTS_IN_MEMORY` | int | `5` | Maximum agents with hot caches (1-50) |
| `SEMANTIC_AGENT_CACHE_DIR` | string | `~/.semantic/caches` | Directory for persistent cache storage |
| `SEMANTIC_AGENT_BATCH_WINDOW_MS` | int | `10` | Batch collection window in milliseconds (1-1000) |
| `SEMANTIC_AGENT_LRU_EVICTION_ENABLED` | bool | `true` | Enable LRU eviction when max_agents exceeded |
| `SEMANTIC_AGENT_EVICT_TO_DISK` | bool | `true` | Persist evicted caches to disk (warm tier) |
| `SEMANTIC_AGENT_VALIDATE_MODEL_TAG` | bool | `true` | Validate cache compatibility with current model |

### Cache Tiers

**Hot Tier** (in-memory):
- Fastest access
- Limited by `max_agents_in_memory`
- LRU eviction when full

**Warm Tier** (disk):
- Stored in `cache_dir`
- Automatic persistence when evicted
- Reloaded on next access

## Server Settings

HTTP server configuration.

### Environment Variables

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `SEMANTIC_SERVER_HOST` | string | `0.0.0.0` | Server bind address |
| `SEMANTIC_SERVER_PORT` | int | `8000` | Server port (1024-65535) |
| `SEMANTIC_SERVER_WORKERS` | int | `1` | Worker processes (MLX limits concurrency) |
| `SEMANTIC_SERVER_RATE_LIMIT_PER_AGENT` | int | `60` | Max requests per agent per minute (1-1000) |
| `SEMANTIC_SERVER_RATE_LIMIT_GLOBAL` | int | `1000` | Max global requests per minute (1-10000) |
| `SEMANTIC_SERVER_LOG_LEVEL` | string | `INFO` | Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL) |
| `SEMANTIC_SERVER_CORS_ORIGINS` | string | `http://localhost:3000` | Comma-separated CORS origins (* for all) |

### Workers

**Note**: MLX inference is single-threaded. Multiple workers share the same GPU.

```bash
# Development
SEMANTIC_SERVER_WORKERS=1

# Production (careful with memory)
SEMANTIC_SERVER_WORKERS=2
```

## Security Settings

API keys and sensitive configuration.

### Environment Variables

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `SEMANTIC_API_KEY` | string | `null` | Optional API key for authentication |
| `SEMANTIC_ADMIN_KEY` | string | `null` | Admin API key for model hot-swap |

### Authentication

**API Key** (optional):
```bash
curl -H "X-API-Key: your-api-key" http://localhost:8000/v1/messages
```

**Admin Key** (for hot-swap):
```bash
curl -H "X-Admin-Key: your-admin-key" http://localhost:8000/admin/swap
```

## Tool Calling Configuration

Tool calling is enabled by default when tools are provided in requests.

### Anthropic Tool Use

```bash
curl -X POST http://localhost:8000/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-coder-v2-lite",
    "max_tokens": 200,
    "messages": [
      {"role": "user", "content": "What's the weather in Paris?"}
    ],
    "tools": [
      {
        "name": "get_weather",
        "description": "Get weather for a location",
        "input_schema": {
          "type": "object",
          "properties": {
            "location": {"type": "string"}
          },
          "required": ["location"]
        }
      }
    ]
  }'
```

### OpenAI Function Calling

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-coder-v2-lite",
    "messages": [
      {"role": "user", "content": "What's the weather in Paris?"}
    ],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "get_weather",
          "description": "Get weather",
          "parameters": {
            "type": "object",
            "properties": {
              "location": {"type": "string"}
            }
          }
        }
      }
    ],
    "tool_choice": "auto"
  }'
```

## Example Configurations

### Development (Local Testing)

```bash
# .env.development
SEMANTIC_MLX_MODEL_ID=mlx-community/SmolLM2-135M-Instruct
SEMANTIC_MLX_CACHE_BUDGET_MB=1024
SEMANTIC_MLX_MAX_BATCH_SIZE=2
SEMANTIC_AGENT_MAX_AGENTS_IN_MEMORY=3
SEMANTIC_SERVER_LOG_LEVEL=DEBUG
SEMANTIC_SERVER_CORS_ORIGINS=*
```

### Production (Apple Silicon)

```bash
# .env.production
SEMANTIC_MLX_MODEL_ID=mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx
SEMANTIC_MLX_CACHE_BUDGET_MB=4096
SEMANTIC_MLX_MAX_BATCH_SIZE=5
SEMANTIC_AGENT_MAX_AGENTS_IN_MEMORY=10
SEMANTIC_AGENT_CACHE_DIR=/var/lib/semantic/caches
SEMANTIC_SERVER_HOST=0.0.0.0
SEMANTIC_SERVER_PORT=8000
SEMANTIC_SERVER_LOG_LEVEL=INFO
SEMANTIC_SERVER_RATE_LIMIT_GLOBAL=1000
SEMANTIC_API_KEY=your-production-api-key
SEMANTIC_ADMIN_KEY=your-admin-key
```

### Multi-Tenant

```bash
# .env.multitenant
SEMANTIC_MLX_MODEL_ID=mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx
SEMANTIC_MLX_CACHE_BUDGET_MB=8192
SEMANTIC_MLX_MAX_BATCH_SIZE=10
SEMANTIC_AGENT_MAX_AGENTS_IN_MEMORY=50
SEMANTIC_AGENT_LRU_EVICTION_ENABLED=true
SEMANTIC_AGENT_EVICT_TO_DISK=true
SEMANTIC_SERVER_RATE_LIMIT_PER_AGENT=30
SEMANTIC_SERVER_RATE_LIMIT_GLOBAL=2000
SEMANTIC_API_KEY=required
```

## Validation

Check your configuration:

```bash
semantic config --show
```

## Environment Variable Precedence

1. Command-line arguments
2. Environment variables
3. `.env` file
4. Default values

## See Also

- [User Guide](user-guide.md) - Usage examples
- [Model Onboarding](model-onboarding.md) - Adding new models
- [Deployment Guide](deployment.md) - Production deployment
```
