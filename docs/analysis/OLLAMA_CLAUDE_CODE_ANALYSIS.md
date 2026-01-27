# Ollama Claude Code CLI Support: In-Depth Analysis

**Date**: 2026-01-26
**Research Focus**: How Ollama implements Claude Code CLI support, what endpoints are required, and what format/functionality is expected

---

## Executive Summary

**YES**, Ollama fully supports Claude Code CLI as of **v0.14.0** (January 2026) through native Anthropic Messages API compatibility. This analysis covers:

1. ✅ What endpoints Ollama provides
2. ✅ Request/response format and streaming behavior
3. ✅ What Claude Code CLI actually requires
4. ✅ Comparison with our implementation

---

## 1. Ollama's Claude Code Support Timeline

### Release History

- **Before v0.14.0**: Required third-party shims/proxies like `ollama-anthropic-shim` to translate between Anthropic and Ollama formats
- **v0.14.0 (January 2026)**: Native Anthropic Messages API support added
- **Current**: Full compatibility with Claude Code CLI out of the box

**Source**: [Ollama v0.14.0 Release](https://github.com/ollama/ollama/releases/tag/v0.14.0), [Ollama Blog: Claude Code with Anthropic API](https://ollama.com/blog/claude)

---

## 2. Required Endpoints

### Primary Endpoint: `/v1/messages`

**Location**: `http://localhost:11434/v1/messages`

This is the **ONLY endpoint required** for Claude Code CLI to function.

**Implementation**: Routes through Anthropic middleware in `server/routes.go`:
```go
r.POST("/v1/messages", ChatHandler) // with Anthropic middleware
```

**Source**: [Ollama routes.go](https://github.com/ollama/ollama/blob/main/server/routes.go), [Anthropic Compatibility Docs](https://docs.ollama.com/api/anthropic-compatibility)

### Optional Health Endpoint: `/` or `/api/version`

Ollama provides:
- `GET /` - Basic health check (returns 200)
- `HEAD /` - Header-only health check
- `/api/version` - Version information

**Note**: These are NOT required for Claude Code CLI to function.

### Event Logging Endpoint: NOT REQUIRED

The `/api/event_logging/batch` endpoint that Claude Code CLI hits is **NOT required** for functionality:

**What it is**: Claude Code CLI's internal telemetry/analytics endpoint for usage tracking
**What Ollama does**: Does NOT implement this endpoint (returns 404)
**Impact**: ZERO - It's optional telemetry, not functional

**Alternative**: Claude Code supports OpenTelemetry for logging via:
```bash
export CLAUDE_CODE_ENABLE_TELEMETRY=1
export OTEL_LOGS_EXPORTER=otlp
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
```

**Source**: [Claude Code Monitoring Docs](https://code.claude.com/docs/en/monitoring-usage), [Claude Code + OpenTelemetry Guide](https://quesma.com/blog/track-claude-code-usage-and-limits-with-grafana-cloud/)

---

## 3. Request/Response Format

### Request Format (POST /v1/messages)

Ollama accepts standard Anthropic Messages API format:

```json
{
  "model": "qwen3-coder",
  "max_tokens": 1024,
  "messages": [
    {
      "role": "user",
      "content": "Hello!"
    }
  ],
  "system": "You are a helpful assistant",
  "stream": true,
  "tools": [...],
  "temperature": 0.7,
  "top_p": 0.9,
  "top_k": 40,
  "stop_sequences": ["\n\n"]
}
```

**Supported Fields**:
- `model` ✅ (required)
- `max_tokens` ✅ (required)
- `messages` ✅ (required)
- `system` ✅ (string or array)
- `stream` ✅ (true/false)
- `tools` ✅ (tool definitions)
- `temperature`, `top_p`, `top_k` ✅
- `stop_sequences` ✅

**Unsupported Fields**:
- `tool_choice` ❌ (ignored)
- `metadata` ❌
- Prompt caching parameters ❌

**Source**: [Ollama Anthropic Compatibility Docs](https://docs.ollama.com/api/anthropic-compatibility)

### Response Format (Non-Streaming)

```json
{
  "id": "msg_abc123",
  "type": "message",
  "role": "assistant",
  "model": "qwen3-coder",
  "content": [
    {
      "type": "text",
      "text": "Hello! How can I help you?"
    }
  ],
  "stop_reason": "end_turn",
  "usage": {
    "input_tokens": 10,
    "output_tokens": 20
  }
}
```

**Content Block Types**:
- `text` ✅ - Regular text response
- `tool_use` ✅ - Tool invocation with name and input
- `thinking` ✅ - Extended thinking blocks (basic support)

**Stop Reasons**:
- `end_turn` - Natural completion
- `max_tokens` - Token limit reached
- `tool_use` - Model wants to call a tool

---

## 4. Streaming Behavior (CRITICAL)

### Streaming Format: Server-Sent Events (SSE)

When `"stream": true`, Ollama returns SSE events in Anthropic format:

```
event: message_start
data: {"type":"message_start","message":{"id":"msg_123","role":"assistant","model":"qwen3-coder",...}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" there"}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":20}}

event: message_stop
data: {"type":"message_stop"}
```

**SSE Event Types** (in order):
1. `message_start` - Start of response with metadata
2. `content_block_start` - Start of a content block (text, tool_use, thinking)
3. `content_block_delta` - Incremental content (text_delta, input_json_delta, thinking_delta)
4. `content_block_stop` - End of content block
5. `message_delta` - Final metadata (stop_reason, usage)
6. `message_stop` - End of stream
7. `ping` (optional) - Keepalive
8. `error` - Error event

**Delta Types**:
- `text_delta` - Incremental text chunks
- `input_json_delta` - Tool input JSON chunks
- `thinking_delta` - Extended thinking text chunks

**Source**: [Ollama Anthropic Docs - Streaming](https://docs.ollama.com/api/anthropic-compatibility)

### Streaming vs Non-Streaming: Both Supported

**Key Finding**: Ollama supports BOTH modes equally:

```python
# Non-streaming
response = requests.post(
    "http://localhost:11434/v1/messages",
    json={"model": "qwen3-coder", "max_tokens": 100, "stream": false, ...}
)

# Streaming
response = requests.post(
    "http://localhost:11434/v1/messages",
    json={"model": "qwen3-coder", "max_tokens": 100, "stream": true, ...},
    stream=True
)
```

**No preference indicated** - Ollama documentation shows examples for both modes without recommending one over the other.

**Source**: [Ollama Anthropic Compatibility Examples](https://docs.ollama.com/api/anthropic-compatibility), [Ollama Streaming Docs](https://docs.ollama.com/capabilities/streaming)

---

## 5. Tool Calling Support

Ollama fully supports tool calling in Anthropic format:

### Tool Definition

```json
{
  "model": "qwen3-coder",
  "tools": [
    {
      "name": "get_weather",
      "description": "Get the weather in a location",
      "input_schema": {
        "type": "object",
        "properties": {
          "location": {
            "type": "string",
            "description": "City name"
          }
        },
        "required": ["location"]
      }
    }
  ],
  "messages": [{"role": "user", "content": "What's the weather in NYC?"}]
}
```

### Tool Use Response

```json
{
  "content": [
    {
      "type": "tool_use",
      "id": "toolu_abc123",
      "name": "get_weather",
      "input": {"location": "NYC"}
    }
  ],
  "stop_reason": "tool_use"
}
```

### Tool Result Follow-up

```json
{
  "messages": [
    {"role": "user", "content": "What's the weather in NYC?"},
    {
      "role": "assistant",
      "content": [
        {
          "type": "tool_use",
          "id": "toolu_abc123",
          "name": "get_weather",
          "input": {"location": "NYC"}
        }
      ]
    },
    {
      "role": "user",
      "content": [
        {
          "type": "tool_result",
          "tool_use_id": "toolu_abc123",
          "content": "Sunny, 72°F",
          "is_error": false
        }
      ]
    }
  ]
}
```

**Limitation**: `tool_choice` parameter is NOT supported (ignored if provided)

**Source**: [Ollama Anthropic Docs - Tools](https://docs.ollama.com/api/anthropic-compatibility), [Ollama Blog - Streaming Tool Calling](https://ollama.com/blog/streaming-tool)

---

## 6. Multi-modal Support

Ollama supports images in Anthropic format with **base64 encoding only**:

```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "image",
          "source": {
            "type": "base64",
            "media_type": "image/png",
            "data": "iVBORw0KGgoAAAANSUhEUg..."
          }
        },
        {
          "type": "text",
          "text": "What's in this image?"
        }
      ]
    }
  ]
}
```

**Limitation**: Image URLs are NOT supported (only base64)

**Source**: [Ollama Anthropic Docs](https://docs.ollama.com/api/anthropic-compatibility)

---

## 7. Configuration for Claude Code CLI

### Environment Variables

To connect Claude Code CLI to Ollama:

```bash
export ANTHROPIC_BASE_URL=http://localhost:11434
export ANTHROPIC_API_KEY=""              # Required but ignored
export ANTHROPIC_AUTH_TOKEN=ollama        # Required but ignored
```

**Why dummy values**: Claude Code CLI validates that these are set, but Ollama ignores them (no authentication).

### Recommended Models

For Claude Code CLI usage:
- **gpt-oss:20b** - 20B parameter model optimized for code
- **qwen3-coder** - Specialized coding model
- **glm-4.7** - GLM-4.7 model
- **deepseek-r1** - DeepSeek reasoning model

**Context Window Requirement**: At least **64K tokens** recommended for Claude Code CLI (handles large codebases and conversation history)

**Source**: [Ollama Claude Code Docs](https://docs.ollama.com/integrations/claude-code), [Medium: Run Claude Code with Local LLMs](https://medium.com/data-science-in-your-pocket/run-claude-code-with-local-llms-using-ollama-a97d2c2f2bd1)

---

## 8. Comparison: Ollama vs Our Implementation

### What Ollama Does

| Feature | Ollama Status | Notes |
|---------|---------------|-------|
| `/v1/messages` endpoint | ✅ Implemented | Routes through Anthropic middleware |
| Non-streaming responses | ✅ Supported | Standard JSON response |
| SSE streaming | ✅ Supported | Full event support (message_start, content_block_delta, etc.) |
| Tool calling | ✅ Supported | Tools defined, tool_use response, tool_result follow-up |
| Multi-turn conversations | ✅ Supported | Message arrays with role alternation |
| Image input (base64) | ✅ Supported | Vision models only |
| Extended thinking | ✅ Partial | Accepts budget_tokens but doesn't enforce |
| `/health` endpoint | ✅ Implemented | `GET /` returns 200 |
| `/api/event_logging/batch` | ❌ NOT implemented | Returns 404 (not needed) |
| Model name flexibility | ✅ Supported | Any Ollama model name works |

### What Our Implementation Has

| Feature | Our Status | Notes |
|---------|------------|-------|
| `/v1/messages` endpoint | ✅ Implemented | `anthropic_adapter.py` |
| Non-streaming responses | ✅ Supported | Returns MessagesResponse |
| SSE streaming | ✅ Supported | EventSourceResponse with full events |
| Tool calling | ✅ Data models | Request/response types defined |
| Multi-turn conversations | ✅ Supported | Message arrays accepted |
| Image input (base64) | ❌ Not tested | Models may not support |
| Extended thinking | ❌ Not implemented | No thinking block parsing |
| `/health` endpoint | ✅ Implemented | `/health`, `/health/live`, `/health/ready` |
| `/api/event_logging/batch` | ✅ NOW FIXED | Returns `{"status": "ok"}` stub |
| Model name | ⚠️ BUG | Returns requested model, not actual model |

---

## 9. Key Findings for Our Implementation

### ✅ What's Working

1. **Core endpoint**: `/v1/messages` fully implemented
2. **Streaming**: SSE events match Anthropic format
3. **Non-streaming**: Standard JSON responses
4. **Health checks**: Multiple health endpoints
5. **Event logging stub**: Now returns 200 instead of 404

### ⚠️ Potential Issues

1. **Model name bug**: We return `request_body.model` (e.g., "claude-haiku-4-5-20251001") instead of actual loaded model ("gpt-oss-20b-MXFP4-Q4")
   - **Impact**: Minor - Claude CLI doesn't care about model name in response
   - **Fix**: Return actual model from settings (already identified in previous session)

2. **Batch processing**: Our BatchEngine may handle concurrent requests differently than Ollama
   - **Ollama**: Unclear how they handle concurrent requests internally
   - **Our implementation**: BatchEngine processes multiple requests in parallel
   - **Need to test**: Whether concurrent requests cause slowdown

3. **Performance**: 17,616 tokens taking 14 minutes suggests something is wrong
   - **Ollama performance**: Not documented in search results
   - **Expected**: Should be 30-60 seconds based on MLX prefill speeds
   - **Hypothesis**: Issue is NOT with streaming vs non-streaming format

### ❌ Not Causing Issues

1. **Missing `/api/event_logging/batch`**: Now fixed with stub endpoint
2. **Streaming format**: Our SSE implementation matches Ollama's
3. **Request/response format**: Matches Anthropic spec

---

## 10. Performance Analysis

### What Ollama Documentation DOESN'T Say

**Critical Gap**: Ollama documentation does NOT discuss:
- Expected inference speeds
- How concurrent requests are handled
- Whether streaming has overhead vs non-streaming
- Batch processing behavior

### What We Know About MLX Performance

From independent benchmarks:
- **M3 Max (64GB)**: ~421 tokens/sec for 32K context
- **Typical MLX**: ~230 tokens/sec
- **Expected for 17,616 tokens**: 30-60 seconds prefill

**Our actual**: 14 minutes = **21 tokens/sec** ❌

### Hypothesis: NOT a Streaming Issue

**Evidence**:
1. Ollama supports both streaming and non-streaming equally
2. No documentation suggests streaming is slower
3. Our direct non-streaming test showed ~566 tok/s (GOOD)
4. Our streaming test showed ~248 tok/s (acceptable, 2.2x slower but NOT 700x)

**Likely cause**: Something specific to how Claude Code CLI interacts with our server, NOT the streaming mechanism itself.

---

## 11. What to Test Next

### Test 1: Model Name Fix Impact

**Question**: Does returning the wrong model name cause Claude CLI to behave differently?

**Test**:
1. Fix `anthropic_adapter.py` to return actual model name
2. Test with Claude Code CLI
3. Measure performance

**Expected**: No performance impact (just cosmetic)

### Test 2: Concurrent Request Handling

**Question**: Does Claude CLI send concurrent requests that interfere with each other?

**Test**:
1. Monitor incoming requests during Claude CLI session
2. Check if multiple requests arrive simultaneously
3. Profile BatchEngine behavior with concurrent requests

**Expected**: This is the likely culprit (see your server logs showing concurrent requests)

### Test 3: Streaming vs Non-Streaming with Claude CLI

**Question**: Does Claude CLI perform differently with streaming disabled?

**Test**:
1. Modify anthropic_adapter to force `stream=false` internally
2. Test with Claude Code CLI
3. Measure performance

**Expected**: Small improvement (2x) but not the full 700x we need

### Test 4: Request Logging Analysis

**Question**: What exactly is Claude CLI sending during slow requests?

**Test**:
1. Add detailed logging to anthropic_adapter:
   - Request arrival timestamps
   - Token counts
   - BatchEngine submit/complete times
   - Cache hit/miss for each request
2. Run Claude CLI with large prompt
3. Analyze timing breakdown

**Expected**: Will reveal where the 14 minutes is spent

---

## 12. Conclusions

### What Ollama Does for Claude Code CLI

1. **Single endpoint**: `/v1/messages` at `http://localhost:11434`
2. **Both modes**: Streaming and non-streaming equally supported
3. **Full Anthropic format**: Request/response match spec
4. **No special handling**: No evidence of request serialization or special concurrent handling
5. **No telemetry endpoint**: Doesn't implement `/api/event_logging/batch`

### What Our Implementation Should Do

1. ✅ **Keep current structure**: Our `/v1/messages` endpoint is correct
2. ✅ **Keep streaming support**: SSE events are properly formatted
3. ✅ **Event logging stub**: Already added (returns 200)
4. ⚠️ **Fix model name**: Return actual model, not requested model
5. 🔍 **Investigate concurrent requests**: This is likely the root cause of slowdown
6. 🔍 **Profile performance**: Add detailed timing logs to find bottleneck

### Key Insight: It's NOT About Streaming

The research shows:
- Ollama supports both streaming and non-streaming
- No indication that one is faster than the other
- Our direct tests show streaming is only 2x slower, not 700x
- The 14-minute issue is NOT caused by SSE overhead

**Root cause is likely**: How BatchEngine handles concurrent requests from Claude CLI, NOT the streaming mechanism itself.

---

## Sources

All information sourced from:

- [Ollama Claude Code Integration Docs](https://docs.ollama.com/integrations/claude-code)
- [Ollama Anthropic API Compatibility](https://docs.ollama.com/api/anthropic-compatibility)
- [Ollama Blog: Claude Code with Anthropic API](https://ollama.com/blog/claude)
- [Ollama GitHub: routes.go](https://github.com/ollama/ollama/blob/main/server/routes.go)
- [Ollama Streaming Capabilities](https://docs.ollama.com/capabilities/streaming)
- [Claude Code Monitoring Documentation](https://code.claude.com/docs/en/monitoring-usage)
- [Medium: Run Claude Code with Local LLMs Using Ollama](https://medium.com/data-science-in-your-pocket/run-claude-code-with-local-llms-using-ollama-a97d2c2f2bd1)
- [Medium: Connecting Claude Code to Local LLMs](https://medium.com/@michael.hannecke/connecting-claude-code-to-local-llms-two-practical-approaches-faa07f474b0f)
- [Quesma: Claude Code + OpenTelemetry + Grafana](https://quesma.com/blog/track-claude-code-usage-and-limits-with-grafana-cloud/)

---

**Analysis Date**: 2026-01-26
**Ollama Version Analyzed**: v0.14.0+
**Conclusion**: Our implementation is structurally sound. The 14-minute issue requires profiling, not architectural changes.
