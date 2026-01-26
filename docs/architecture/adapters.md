# Adapters Layer

Adapters implement external interfaces for the Semantic Caching API (Hexagonal Architecture).

## Overview

Adapters translate between external protocols and internal domain/application logic. They handle HTTP request/response formatting, validation, and error translation.

## Inbound Adapters

Inbound adapters receive requests from external clients and invoke application layer use cases.

### Anthropic Messages API

**Purpose**: Anthropic Claude-compatible Messages API

**Location**: `src/semantic/adapters/inbound/anthropic_adapter.py`

**Endpoint**: `POST /v1/messages`

**Request Format**:
```python
class MessagesRequest(BaseModel):
    model: str
    max_tokens: int
    messages: list[Message]
    system: str | list[dict] = ""
    stream: bool = False
    tools: list[Tool] | None = None  # Tool calling support
```

**Response Format**:
```python
class MessagesResponse(BaseModel):
    id: str
    type: str = "message"
    role: str = "assistant"
    content: list[ContentBlock]  # TextBlock or ToolUseContentBlock
    model: str
    stop_reason: str  # "end_turn", "max_tokens", "tool_use"
    usage: Usage
```

**Features**:
- Multi-turn conversations
- Tool calling (tool_use)
- Streaming (SSE format)
- Session ID extension (X-Session-ID header)
- Cache metrics (cache_creation, cache_read)

**Tool Calling**:
```python
# Request with tools
{
    "tools": [{
        "name": "get_weather",
        "description": "Get weather",
        "input_schema": {...}
    }]
}

# Response with tool_use
{
    "content": [{
        "type": "tool_use",
        "id": "toolu_123",
        "name": "get_weather",
        "input": {"location": "Paris"}
    }],
    "stop_reason": "tool_use"
}
```

### OpenAI Chat Completions API

**Purpose**: OpenAI-compatible Chat Completions API

**Location**: `src/semantic/adapters/inbound/openai_adapter.py`

**Endpoint**: `POST /v1/chat/completions`

**Request Format**:
```python
class ChatCompletionsRequest(BaseModel):
    model: str
    messages: list[OpenAIChatMessage]
    max_tokens: int | None = None
    stream: bool = False
    tools: list[OpenAITool] | None = None  # Function calling support
    tool_choice: str | dict | None = "auto"
```

**Response Format**:
```python
class ChatCompletionsResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[OpenAIChatChoice]
    usage: OpenAIChatCompletionUsage
```

**Features**:
- OpenAI message format
- Function calling (tool_calls)
- Streaming (SSE format: data: {...} / data: [DONE])
- Session ID extension (X-Session-ID header)
- tool_choice parameter (auto, required, specific function)

**Function Calling**:
```python
# Request with tools
{
    "tools": [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {...}
        }
    }],
    "tool_choice": "auto"
}

# Response with tool_calls
{
    "choices": [{
        "message": {
            "tool_calls": [{
                "id": "call_123",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": "{...}"
                }
            }]
        },
        "finish_reason": "tool_calls"
    }]
}
```

### Direct Agent API

**Purpose**: Explicit agent management with cache control

**Location**: `src/semantic/adapters/inbound/agent_adapter.py`

**Endpoints**:
- `POST /v1/agents` - Create agent
- `POST /v1/agents/{agent_id}/generate` - Generate
- `GET /v1/agents` - List agents
- `DELETE /v1/agents/{agent_id}` - Delete agent

**Features**:
- Explicit agent lifecycle
- Direct cache size reporting
- No message history formatting required

### Health & Metrics

**Health Endpoint**:
```python
GET /health

Response:
{
    "status": "healthy"
}
```

**Metrics Endpoint**:
```bash
GET /metrics

# Prometheus format
semantic_requests_total{endpoint="/v1/messages"} 1523
semantic_active_agents 3
semantic_cache_hits_total 892
```

### Admin Endpoints

**Model Hot-Swap**:
```python
POST /admin/swap
X-Admin-Key: your-admin-key

{
    "model_id": "mlx-community/SmolLM2-135M-Instruct"
}

Response:
{
    "status": "success",
    "old_model": "gemma-3-12b-it-4bit",
    "new_model": "SmolLM2-135M-Instruct"
}
```

## Adapter Responsibilities

### 1. Protocol Translation

Convert external formats to internal domain types:

```python
# Anthropic adapter
def messages_to_prompt(messages: list[Message], tools: list[Tool] | None) -> str:
    """Convert Anthropic messages to prompt string."""
    # Include tool definitions in prompt
    # Format multi-turn conversation
    # Add assistant continuation prompt

# OpenAI adapter
def openai_messages_to_prompt(messages: list[OpenAIChatMessage], tools: list[Tool] | None) -> str:
    """Convert OpenAI messages to prompt string."""
    # Convert role names
    # Handle tool_calls and tool role messages
    # Format function definitions
```

### 2. Validation

Validate requests using Pydantic models:

```python
class MessagesRequest(BaseModel):
    model: str = Field(..., description="Model ID")
    max_tokens: int = Field(..., gt=0, le=8192)
    messages: list[Message] = Field(..., min_items=1)

    @field_validator("messages")
    def validate_message_sequence(cls, v):
        # Validate alternating user/assistant
        # Check first message is user
        return v
```

### 3. Error Translation

Map domain errors to HTTP responses:

```python
try:
    result = batch_engine.submit(...)
except PoolExhaustedError as e:
    # 503 Service Unavailable
    raise HTTPException(status_code=503, detail=f"Capacity exceeded: {e}")
except ValidationError as e:
    # 400 Bad Request
    raise HTTPException(status_code=400, detail=f"Invalid request: {e}")
except SemanticError as e:
    # 500 Internal Server Error
    raise HTTPException(status_code=500, detail="Internal error")
```

### 4. Streaming

Implement Server-Sent Events (SSE) streaming:

**Anthropic Format**:
```python
async def stream_generation(...) -> AsyncIterator[dict]:
    yield {"data": json.dumps({"type": "message_start", ...})}
    yield {"data": json.dumps({"type": "content_block_start", ...})}
    yield {"data": json.dumps({"type": "content_block_delta", "delta": {...}})}
    yield {"data": json.dumps({"type": "content_block_stop", ...})}
    yield {"data": json.dumps({"type": "message_stop"})}
```

**OpenAI Format**:
```python
async def stream_chat_completion(...) -> AsyncIterator[dict]:
    yield {"data": json.dumps({"choices": [{"delta": {"role": "assistant"}}]})}
    yield {"data": json.dumps({"choices": [{"delta": {"content": "Hello"}}]})}
    yield {"data": json.dumps({"choices": [{"delta": {}, "finish_reason": "stop"}]})}
    yield {"data": "[DONE]"}
```

### 5. Tool Calling

Parse model output for tool invocations:

```python
# Anthropic: Parse {"tool_use": {...}} in text
def parse_tool_calls(text: str) -> tuple[str, list[dict]]:
    pattern = r'\{"tool_use":\s*\{[^}]+\}\s*\}'
    matches = re.finditer(pattern, text)
    # Extract tool name and input
    # Return remaining text and tool calls

# OpenAI: Parse {"function_call": {...}} in text
def parse_function_calls(text: str) -> tuple[str, list[dict]]:
    pattern = r'\{"function_call":\s*\{[^}]+\}\s*\}'
    matches = re.finditer(pattern, text)
    # Extract function name and arguments
    # Return remaining text and function calls
```

## Outbound Adapters

Outbound adapters connect application layer to external systems.

### File System Adapter

**Purpose**: Persist agent caches to disk

**Location**: Integrated in `AgentCacheStore`

**Operations**:
- Save cache to `~/.semantic/caches/{agent_id}.cache`
- Load cache from disk
- List cached agents
- Delete cache files

### MLX Inference Adapter

**Purpose**: Load models and perform inference

**Location**: Integrated in `BatchEngine`

**Operations**:
- Load model from Hugging Face Hub
- Tokenize prompts
- Execute MLX inference
- Manage KV cache blocks

### Logging Adapter

**Purpose**: Structured logging

**Implementation**: Python logging module

```python
logger.info(
    f"Request: agent={agent_id}, tokens={len(tokens)}, cache_hit={hit}"
)
```

### Metrics Adapter

**Purpose**: Prometheus metrics collection

**Implementation**: `prometheus_client` library

```python
from prometheus_client import Counter, Histogram

requests_total = Counter("semantic_requests_total", "Total requests")
generation_duration = Histogram("semantic_generation_duration_seconds", "Generation latency")
```

## Dependency Injection

Adapters receive dependencies via FastAPI app state:

```python
# In create_app()
app.state.semantic = SemanticState(
    batch_engine=engine,
    cache_store=store,
    settings=settings,
)

# In endpoint
async def create_message(request: Request):
    engine = request.app.state.semantic.batch_engine
    store = request.app.state.semantic.cache_store
    # Use dependencies
```

## Testing Adapters

**Integration tests** validate end-to-end flows:

```python
def test_anthropic_api_with_tools():
    """Test Anthropic API with tool calling."""
    app = create_app()

    with TestClient(app) as client:
        response = client.post(
            "/v1/messages",
            json={
                "model": "test",
                "max_tokens": 100,
                "messages": [{...}],
                "tools": [{...}],
            }
        )

        assert response.status_code == 200
        data = response.json()

        # Verify tool_use in response
        has_tool_use = any(
            block["type"] == "tool_use"
            for block in data["content"]
        )
```

## API Versioning

All endpoints are versioned under `/v1/`:

- `/v1/messages` - Anthropic Messages API
- `/v1/chat/completions` - OpenAI Chat Completions
- `/v1/agents` - Direct Agent API

Future versions (v2) can be added without breaking existing clients.

## CORS Configuration

CORS is configured via settings:

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.server.cors_origins.split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## Rate Limiting

Rate limits are enforced per-agent and globally:

```python
# Per-agent: 60 req/min (default)
# Global: 1000 req/min (default)

if agent_request_count > rate_limit_per_agent:
    raise HTTPException(status_code=429, detail="Rate limit exceeded")
```

## See Also

- [Domain Layer](domain.md) - Core business logic
- [Application Layer](application.md) - Use case orchestration
- [API Reference](../api-reference.md) - Complete API documentation
- [User Guide](../user-guide.md) - Usage examples
