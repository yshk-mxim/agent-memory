# User Guide

Comprehensive guide for using agent-memory in production and development.

## Table of Contents

- [Server Startup](#server-startup)
- [API Protocols](#api-protocols)
- [Tool Calling](#tool-calling)
- [Streaming](#streaming)
- [Multi-Agent Workflows](#multi-agent-workflows)
- [Cache Management](#cache-management)
- [Monitoring](#monitoring)
- [Troubleshooting](#troubleshooting)

## Server Startup

### Quick Start

Start the server with default settings:

```bash
agent-memory serve
```

This starts the server on `http://0.0.0.0:8000` with the default Gemma 3 model.

### Custom Configuration

Start with custom settings:

```bash
# Specify port
agent-memory serve --port 8080

# Specify host
agent-memory serve --host 127.0.0.1

# Specify model
agent-memory serve --model mlx-community/SmolLM2-135M-Instruct

# Specify log level
agent-memory serve --log-level DEBUG

# Combine options
agent-memory serve --port 8080 --log-level INFO --model mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx
```

### Using Environment Variables

Create a `.env` file in your project root:

```bash
# Server settings
SEMANTIC_SERVER_HOST=0.0.0.0
SEMANTIC_SERVER_PORT=8000
SEMANTIC_SERVER_LOG_LEVEL=INFO

# MLX model settings
SEMANTIC_MLX_MODEL_ID=mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx
SEMANTIC_MLX_CACHE_BUDGET_MB=4096
SEMANTIC_MLX_MAX_BATCH_SIZE=5

# Agent cache settings
SEMANTIC_AGENT_MAX_AGENTS_IN_MEMORY=5
SEMANTIC_AGENT_CACHE_DIR=~/.agent_memory/caches

# Security (optional)
SEMANTIC_API_KEY=your-api-key-here
```

Then start the server:

```bash
agent-memory serve
```

The server will automatically load configuration from `.env`.

### Verify Server is Running

```bash
# Check health
curl http://localhost:8000/health

# Expected response:
# {"status":"healthy"}

# Check metrics
curl http://localhost:8000/metrics

# Get version
agent-memory version
```

## API Protocols

agent-memory supports three API protocols:

### 1. Anthropic Messages API

**Endpoint**: `POST /v1/messages`

**Format**: Compatible with Anthropic Claude API

```bash
curl -X POST http://localhost:8000/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma-3-12b-it-4bit",
    "max_tokens": 200,
    "messages": [
      {"role": "user", "content": "What is semantic caching?"}
    ]
  }'
```

**Response**:
```json
{
  "id": "msg_01ABC...",
  "type": "message",
  "role": "assistant",
  "content": [
    {
      "type": "text",
      "text": "Semantic caching is a technique..."
    }
  ],
  "model": "gemma-3-12b-it-4bit",
  "stop_reason": "end_turn",
  "usage": {
    "input_tokens": 15,
    "output_tokens": 50,
    "cache_creation_input_tokens": 15,
    "cache_read_input_tokens": 0
  }
}
```

### 2. OpenAI Chat Completions API

**Endpoint**: `POST /v1/chat/completions`

**Format**: Compatible with OpenAI Chat Completions API

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma-3-12b-it-4bit",
    "messages": [
      {"role": "user", "content": "What is semantic caching?"}
    ],
    "max_tokens": 200
  }'
```

**Response**:
```json
{
  "id": "chatcmpl-abc123...",
  "object": "chat.completion",
  "created": 1706000000,
  "model": "gemma-3-12b-it-4bit",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Semantic caching is a technique..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 15,
    "completion_tokens": 50,
    "total_tokens": 65
  }
}
```

### 3. Direct Agent API

**Endpoint**: `POST /v1/agents` and `POST /v1/agents/{agent_id}/generate`

**Format**: Direct agent management with explicit cache control

```bash
# Create agent
curl -X POST http://localhost:8000/v1/agents \
  -H "Content-Type: application/json" \
  -d '{"agent_id": "my-agent"}'

# Generate with agent
curl -X POST http://localhost:8000/v1/agents/my-agent/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What is semantic caching?",
    "max_tokens": 200
  }'

# Response:
{
  "text": "Semantic caching is a technique...",
  "tokens_generated": 50,
  "cache_size_tokens": 65
}

# Delete agent
curl -X DELETE http://localhost:8000/v1/agents/my-agent
```

## Tool Calling

agent-memory supports tool calling for both Anthropic and OpenAI formats.

### Anthropic Tool Use

Define tools in the `tools` array with `name`, `description`, and `input_schema`:

```bash
curl -X POST http://localhost:8000/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma-3-12b-it-4bit",
    "max_tokens": 200,
    "messages": [
      {"role": "user", "content": "What is the weather in Paris?"}
    ],
    "tools": [
      {
        "name": "get_weather",
        "description": "Get the current weather for a location",
        "input_schema": {
          "type": "object",
          "properties": {
            "location": {
              "type": "string",
              "description": "The city name"
            },
            "unit": {
              "type": "string",
              "enum": ["celsius", "fahrenheit"],
              "description": "Temperature unit"
            }
          },
          "required": ["location"]
        }
      }
    ]
  }'
```

**Response with tool_use**:
```json
{
  "id": "msg_01ABC...",
  "type": "message",
  "role": "assistant",
  "content": [
    {
      "type": "text",
      "text": "I'll check the weather for you."
    },
    {
      "type": "tool_use",
      "id": "toolu_01ABC...",
      "name": "get_weather",
      "input": {
        "location": "Paris",
        "unit": "celsius"
      }
    }
  ],
  "model": "gemma-3-12b-it-4bit",
  "stop_reason": "tool_use",
  "usage": {...}
}
```

**Continue with tool result**:
```bash
curl -X POST http://localhost:8000/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma-3-12b-it-4bit",
    "max_tokens": 200,
    "messages": [
      {"role": "user", "content": "What is the weather in Paris?"},
      {
        "role": "assistant",
        "content": [
          {"type": "text", "text": "I'\''ll check the weather for you."},
          {
            "type": "tool_use",
            "id": "toolu_01ABC...",
            "name": "get_weather",
            "input": {"location": "Paris", "unit": "celsius"}
          }
        ]
      },
      {
        "role": "user",
        "content": [
          {
            "type": "tool_result",
            "tool_use_id": "toolu_01ABC...",
            "content": "The weather in Paris is 22°C and sunny."
          }
        ]
      }
    ],
    "tools": [...]
  }'
```

### OpenAI Function Calling

Define tools using the OpenAI format with `type: "function"`:

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma-3-12b-it-4bit",
    "messages": [
      {"role": "user", "content": "What is the weather in Paris?"}
    ],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "get_weather",
          "description": "Get the current weather for a location",
          "parameters": {
            "type": "object",
            "properties": {
              "location": {
                "type": "string",
                "description": "The city name"
              },
              "unit": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"]
              }
            },
            "required": ["location"]
          }
        }
      }
    ],
    "tool_choice": "auto",
    "max_tokens": 200
  }'
```

**Response with tool_calls**:
```json
{
  "id": "chatcmpl-abc123...",
  "object": "chat.completion",
  "created": 1706000000,
  "model": "gemma-3-12b-it-4bit",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": null,
        "tool_calls": [
          {
            "id": "call_abc123...",
            "type": "function",
            "function": {
              "name": "get_weather",
              "arguments": "{\"location\": \"Paris\", \"unit\": \"celsius\"}"
            }
          }
        ]
      },
      "finish_reason": "tool_calls"
    }
  ],
  "usage": {...}
}
```

**Continue with tool result**:
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma-3-12b-it-4bit",
    "messages": [
      {"role": "user", "content": "What is the weather in Paris?"},
      {
        "role": "assistant",
        "content": null,
        "tool_calls": [
          {
            "id": "call_abc123...",
            "type": "function",
            "function": {
              "name": "get_weather",
              "arguments": "{\"location\": \"Paris\", \"unit\": \"celsius\"}"
            }
          }
        ]
      },
      {
        "role": "tool",
        "tool_call_id": "call_abc123...",
        "content": "The weather in Paris is 22°C and sunny."
      }
    ],
    "tools": [...],
    "max_tokens": 200
  }'
```

### Tool Choice Parameter (OpenAI)

Control when tools are used:

```bash
# Auto (default) - model decides
"tool_choice": "auto"

# Required - model must use a tool
"tool_choice": "required"

# Specific function - force specific tool
"tool_choice": {
  "type": "function",
  "function": {"name": "get_weather"}
}

# None - disable tools
"tool_choice": "none"
```

## Streaming

Both APIs support Server-Sent Events (SSE) streaming.

### Anthropic Streaming

Set `"stream": true` in the request:

```bash
curl -X POST http://localhost:8000/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma-3-12b-it-4bit",
    "max_tokens": 200,
    "messages": [
      {"role": "user", "content": "Tell me a story"}
    ],
    "stream": true
  }'
```

**Event stream**:
```
data: {"type":"message_start","message":{"id":"msg_01ABC...","type":"message","role":"assistant","content":[],"model":"gemma-3-12b-it-4bit"}}

data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Once"}}

data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" upon"}}

data: {"type":"content_block_stop","index":0}

data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":50}}

data: {"type":"message_stop"}
```

### OpenAI Streaming

Set `"stream": true` in the request:

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma-3-12b-it-4bit",
    "messages": [
      {"role": "user", "content": "Tell me a story"}
    ],
    "stream": true,
    "max_tokens": 200
  }'
```

**Event stream**:
```
data: {"id":"chatcmpl-abc...","object":"chat.completion.chunk","created":1706000000,"model":"gemma-3-12b-it-4bit","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}

data: {"id":"chatcmpl-abc...","object":"chat.completion.chunk","created":1706000000,"model":"gemma-3-12b-it-4bit","choices":[{"index":0,"delta":{"content":"Once"},"finish_reason":null}]}

data: {"id":"chatcmpl-abc...","object":"chat.completion.chunk","created":1706000000,"model":"gemma-3-12b-it-4bit","choices":[{"index":0,"delta":{"content":" upon"},"finish_reason":null}]}

data: {"id":"chatcmpl-abc...","object":"chat.completion.chunk","created":1706000000,"model":"gemma-3-12b-it-4bit","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

### Streaming Tool Calls

Tool calls are streamed incrementally in both formats.

## Multi-Agent Workflows

Use persistent agent IDs to maintain separate conversation caches.

### Anthropic Multi-Agent

Use `X-Session-ID` header or `session_id` in request body:

```bash
# Agent 1 conversation
curl -X POST http://localhost:8000/v1/messages \
  -H "Content-Type: application/json" \
  -H "X-Session-ID: agent-alice" \
  -d '{
    "model": "gemma-3-12b-it-4bit",
    "max_tokens": 200,
    "messages": [
      {"role": "user", "content": "My name is Alice"}
    ]
  }'

# Agent 1 followup (cache reused)
curl -X POST http://localhost:8000/v1/messages \
  -H "X-Session-ID: agent-alice" \
  -d '{
    "model": "gemma-3-12b-it-4bit",
    "max_tokens": 200,
    "messages": [
      {"role": "user", "content": "My name is Alice"},
      {"role": "assistant", "content": "Hello Alice!"},
      {"role": "user", "content": "What is my name?"}
    ]
  }'

# Agent 2 (separate cache)
curl -X POST http://localhost:8000/v1/messages \
  -H "X-Session-ID: agent-bob" \
  -d '{
    "model": "gemma-3-12b-it-4bit",
    "max_tokens": 200,
    "messages": [
      {"role": "user", "content": "My name is Bob"}
    ]
  }'
```

### OpenAI Multi-Agent

Use `X-Session-ID` header or `session_id` in request body:

```bash
# Agent 1
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "X-Session-ID: session-123" \
  -d '{
    "model": "gemma-3-12b-it-4bit",
    "messages": [
      {"role": "user", "content": "Remember: I like Python"}
    ],
    "max_tokens": 200
  }'

# Agent 1 followup
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "X-Session-ID: session-123" \
  -d '{
    "model": "gemma-3-12b-it-4bit",
    "messages": [
      {"role": "user", "content": "Remember: I like Python"},
      {"role": "assistant", "content": "Got it!"},
      {"role": "user", "content": "What language do I like?"}
    ],
    "max_tokens": 200
  }'
```

### Direct Agent API

Explicit agent creation and management:

```bash
# Create multiple agents
curl -X POST http://localhost:8000/v1/agents \
  -d '{"agent_id": "customer-support-1"}'

curl -X POST http://localhost:8000/v1/agents \
  -d '{"agent_id": "customer-support-2"}'

# Use agents independently
curl -X POST http://localhost:8000/v1/agents/customer-support-1/generate \
  -d '{"prompt": "Hello", "max_tokens": 50}'

curl -X POST http://localhost:8000/v1/agents/customer-support-2/generate \
  -d '{"prompt": "Hi there", "max_tokens": 50}'

# List all agents
curl http://localhost:8000/v1/agents

# Delete agents
curl -X DELETE http://localhost:8000/v1/agents/customer-support-1
curl -X DELETE http://localhost:8000/v1/agents/customer-support-2
```

## Cache Management

### Understanding Cache Metrics

**Anthropic format**:
```json
"usage": {
  "input_tokens": 100,
  "output_tokens": 50,
  "cache_creation_input_tokens": 100,  // First request
  "cache_read_input_tokens": 90         // Subsequent requests (cache hit)
}
```

**OpenAI format**:
```json
"usage": {
  "prompt_tokens": 100,
  "completion_tokens": 50,
  "total_tokens": 150
}
```

### Cache Persistence

Caches are automatically persisted to disk:

```bash
# Default location
~/.agent_memory/caches/

# Custom location (via env var)
SEMANTIC_AGENT_CACHE_DIR=/var/lib/agent_memory/caches
```

Cache files are named by agent ID and include metadata about the model.

### Cache Eviction

Configure cache eviction policy:

```bash
# Max agents in memory (hot tier)
SEMANTIC_AGENT_MAX_AGENTS_IN_MEMORY=5

# Enable LRU eviction
SEMANTIC_AGENT_LRU_EVICTION_ENABLED=true

# Evict to disk (warm tier)
SEMANTIC_AGENT_EVICT_TO_DISK=true
```

When `max_agents_in_memory` is reached, least-recently-used agents are evicted to disk.

## Monitoring

### Prometheus Metrics

Metrics are exposed at `/metrics`:

```bash
curl http://localhost:8000/metrics
```

**Available metrics**:
- `semantic_requests_total` - Total requests by endpoint
- `semantic_request_duration_seconds` - Request latency histogram
- `semantic_active_agents` - Current agents in memory
- `semantic_cache_hits_total` - Cache hit count
- `semantic_cache_misses_total` - Cache miss count
- `semantic_tokens_generated_total` - Total tokens generated
- `semantic_generation_duration_seconds` - Generation latency

### Health Checks

```bash
# Basic health check
curl http://localhost:8000/health
# {"status": "healthy"}

# Detailed status (if implemented)
curl http://localhost:8000/status
```

### Logging

Configure log level:

```bash
# Via CLI
agent-memory serve --log-level DEBUG

# Via environment variable
SEMANTIC_SERVER_LOG_LEVEL=DEBUG agent-memory serve
```

**Log levels**:
- `DEBUG` - Verbose logging (token counts, cache operations)
- `INFO` - Standard logging (requests, responses, errors)
- `WARNING` - Warnings only
- `ERROR` - Errors only
- `CRITICAL` - Critical errors only

## Troubleshooting

### Issue: Model Loading Fails

**Symptom**: Server fails to start with model loading error

**Solutions**:
1. Verify model ID is correct (case-sensitive)
2. Check internet connection (models download from Hugging Face)
3. Verify sufficient disk space (~8GB for Gemma 3)
4. Check MLX is installed: `pip show mlx-lm`

### Issue: Out of Memory

**Symptom**: Server crashes or slows down significantly

**Solutions**:
1. Reduce cache budget: `SEMANTIC_MLX_CACHE_BUDGET_MB=2048`
2. Reduce max batch size: `SEMANTIC_MLX_MAX_BATCH_SIZE=2`
3. Reduce agents in memory: `SEMANTIC_AGENT_MAX_AGENTS_IN_MEMORY=3`
4. Use smaller model: `mlx-community/SmolLM2-135M-Instruct`

### Issue: Slow Generation

**Symptom**: Requests take a long time to complete

**Solutions**:
1. Check system resources (CPU, memory, GPU)
2. Reduce max_tokens in requests
3. Use smaller model for faster inference
4. Enable batching if disabled

### Issue: Cache Not Working

**Symptom**: `cache_read_input_tokens` is always 0

**Solutions**:
1. Verify session_id or X-Session-ID is consistent
2. Check message history matches exactly
3. Verify cache directory is writable
4. Check logs for cache errors

### Issue: Tool Calling Not Working

**Symptom**: Model doesn't invoke tools

**Solutions**:
1. Verify tool schema is correct (valid JSON Schema)
2. Check model supports instruction following (Gemma 3 works best)
3. Try explicit instructions in user message
4. Review tool descriptions for clarity

### Issue: Rate Limiting

**Symptom**: 429 Too Many Requests errors

**Solutions**:
1. Reduce request rate
2. Increase rate limits:
   ```bash
   SEMANTIC_SERVER_RATE_LIMIT_PER_AGENT=100
   SEMANTIC_SERVER_RATE_LIMIT_GLOBAL=2000
   ```
3. Use multiple session IDs to distribute load

### Issue: Connection Errors

**Symptom**: Cannot connect to server

**Solutions**:
1. Verify server is running: `curl http://localhost:8000/health`
2. Check host/port configuration
3. Check firewall rules
4. Verify no other process is using the port

### Getting Help

1. **Check logs**: `agent-memory serve --log-level DEBUG`
2. **Review documentation**: See [Configuration Guide](configuration.md)
3. **Check GitHub issues**: Report bugs and feature requests
4. **Test with minimal example**: Isolate the problem

## Best Practices

### Performance

1. **Reuse session IDs**: Maintain consistent session IDs for cache hits
2. **Batch requests**: Use appropriate batch window for high-throughput
3. **Tune cache budget**: Balance memory vs cache capacity
4. **Monitor metrics**: Track cache hit rates and latency

### Reliability

1. **Set timeouts**: Configure appropriate timeouts in client code
2. **Handle errors**: Implement retry logic with exponential backoff
3. **Validate inputs**: Check message formats before sending
4. **Monitor health**: Regular health check polling

### Security

1. **Use API keys**: Enable `SEMANTIC_API_KEY` in production
2. **Restrict CORS**: Configure `SEMANTIC_SERVER_CORS_ORIGINS` appropriately
3. **Network security**: Use reverse proxy (nginx) for TLS
4. **Rotate keys**: Periodically rotate API keys

### Cost Optimization

1. **Use smaller models**: SmolLM2 for development/testing
2. **Limit max_tokens**: Set appropriate limits per use case
3. **Enable caching**: Maximize cache reuse with consistent session IDs
4. **Monitor usage**: Track token consumption via metrics

## See Also

- [Configuration Guide](configuration.md) - Complete configuration reference
- [API Reference](api-reference.md) - Detailed API documentation
- [Adding Models](developer/adding-models.md) - Adding new models
- [Deployment Guide](deployment.md) - Production deployment
