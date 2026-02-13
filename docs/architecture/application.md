# Application Layer

Application layer orchestrates use cases and coordinates between domain and adapters.

## Overview

The application layer implements use cases by coordinating domain services, managing external dependencies (ports), and handling cross-cutting concerns like logging and metrics.

## Key Components

### AgentCacheStore

**Purpose**: Manages agent cache lifecycle with LRU eviction and disk persistence

**Location**: `src/agent_memory/application/agent_cache_store.py`

**Responsibilities**:
1. Load/save agent caches from/to disk
2. Track agents in memory (hot tier)
3. Evict LRU agents when memory limit reached
4. Validate model tags for cache compatibility

**Interface**:
```python
class AgentCacheStore:
    def __init__(
        self,
        max_agents_in_memory: int,
        cache_dir: Path,
        model_tag: str,
    ):
        """Initialize store with LRU policy."""

    def load(self, agent_id: str) -> CachedKVBlocks | None:
        """Load cache from memory or disk."""

    def save(self, agent_id: str, cache: CachedKVBlocks) -> None:
        """Save cache to memory and optionally disk."""

    def evict_lru() -> str | None:
        """Evict least-recently-used agent."""

    def get_stats() -> dict:
        """Get cache statistics (hot, warm, total)."""
```

**Cache Tiers**:
- **Hot Tier** (Memory): Fast access, limited by `max_agents_in_memory`
- **Warm Tier** (Disk): Slower access, persisted between restarts

**Example**:
```python
store = AgentCacheStore(
    max_agents_in_memory=5,
    cache_dir=Path("~/.agent_memory/caches"),
    model_tag="gemma-3-12b-it-4bit",
)

# Load (checks memory first, then disk)
cache = store.load("agent-1")

# Save (stores in memory, optionally to disk)
store.save("agent-1", updated_cache)

# When memory full
evicted_id = store.evict_lru()  # Removes oldest agent from memory
```

### BlockPoolBatchEngine

**Purpose**: Core inference orchestrator with batching and cache management

**Location**: `src/agent_memory/application/batch_engine.py`

**Responsibilities**:
1. Accept generation requests from multiple agents
2. Batch compatible requests for MLX efficiency
3. Manage KV cache allocation via BlockPool
4. Stream token generation incrementally
5. Track per-agent cache state

**Architecture**:
```
submit() → pending_queue
             ↓
          step() → batch requests
             ↓
          MLX inference
             ↓
          yield results → cache updates
```

**Interface**:
```python
class BlockPoolBatchEngine:
    def submit(
        self,
        agent_id: str,
        prompt: str,
        cache: CachedKVBlocks | None,
        max_tokens: int,
    ) -> str:
        """Submit request, returns unique ID."""

    def step() -> Iterator[GenerationResult]:
        """Execute inference step, yields results for all active requests."""
```

**Batching Strategy**:
- Collects requests for `batch_window_ms` milliseconds
- Groups requests with similar prompt lengths
- Executes batch in single MLX forward pass
- Yields results incrementally as tokens are generated

### Settings Management

**Purpose**: Centralized configuration via Pydantic Settings

**Location**: `src/agent_memory/application/settings.py`

**Structure**:
```python
class MLXSettings(BaseSettings):
    model_id: str = "mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx"
    cache_budget_mb: int = 4096
    max_batch_size: int = 5
    ...

class AgentSettings(BaseSettings):
    max_agents_in_memory: int = 5
    cache_dir: Path = Path.home() / ".semantic" / "caches"
    ...

class ServerSettings(BaseSettings):
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"
    ...

class Settings(BaseSettings):
    mlx: MLXSettings
    agent: AgentSettings
    server: ServerSettings
```

**Configuration Sources** (priority order):
1. Environment variables (`SEMANTIC_MLX_MODEL_ID`)
2. `.env` file
3. Default values

## Use Cases

### UC1: Create Message (Anthropic API)

**Flow**:
1. Adapter receives POST /v1/messages
2. Parse and validate request (Pydantic)
3. Convert messages to prompt string
4. Load agent cache from AgentCacheStore
5. Submit to BatchEngine
6. Execute step() until complete
7. Parse tool calls from response
8. Save updated cache
9. Return formatted response

**Participants**:
- `anthropic_adapter.py` (adapter)
- `AgentCacheStore` (application)
- `BatchEngine` (application)

### UC2: Evict Agent on Memory Pressure

**Flow**:
1. AgentCacheStore detects memory limit reached
2. Identify LRU agent
3. Persist cache to disk (warm tier)
4. Free BlockPool blocks
5. Remove from in-memory tracking

**Participants**:
- `AgentCacheStore` (application)
- `BlockPool` (domain)
- File I/O (outbound port)

### UC3: Model Hot-Swap

**Flow**:
1. Admin endpoint receives swap request
2. Validate new model ID
3. Save all active caches to disk
4. Unload current model
5. Load new model
6. Extract ModelCacheSpec
7. Reinitialize BlockPool with new spec
8. Clear in-memory caches (model_tag changed)

**Participants**:
- `admin_adapter.py` (adapter)
- Model loader (application)
- `AgentCacheStore` (application)

## Ports (Interfaces)

### Inbound Ports

**Implemented by adapters**:
- HTTP API endpoints (Anthropic, OpenAI, Direct Agent)
- Admin endpoints (model swap, metrics)
- Health checks

### Outbound Ports

**Implemented by application for domain**:
- **File I/O**: Cache persistence to disk
- **MLX Inference**: Model loading and generation
- **Logging**: Structured logging
- **Metrics**: Prometheus metrics collection

## Dependency Injection

Application layer uses FastAPI's dependency injection:

```python
# Create app state
app.state.semantic = SemanticState(
    batch_engine=engine,
    cache_store=store,
    settings=settings,
)

# Inject in endpoints
async def create_message(request: Request):
    engine = request.app.state.semantic.batch_engine
    store = request.app.state.semantic.cache_store
    ...
```

## Cross-Cutting Concerns

### Logging

Structured logging with context:

```python
logger.info(
    f"Request: agent_id={agent_id}, tokens={len(tokens)}, "
    f"cache_hit={cache is not None}"
)
```

### Metrics

Prometheus metrics:

```python
requests_total.labels(endpoint="/v1/messages").inc()
cache_hits_total.inc()
generation_duration.observe(duration)
```

### Error Handling

Domain errors are translated to HTTP responses:

```python
try:
    result = batch_engine.submit(...)
except PoolExhaustedError as e:
    raise HTTPException(status_code=503, detail=str(e))
except SemanticError as e:
    raise HTTPException(status_code=400, detail=str(e))
```

## Testing Application Layer

**Integration tests** validate use cases:

```python
def test_anthropic_message_creation():
    """Test complete message creation flow."""
    app = create_app()

    with TestClient(app) as client:
        # First request (cache miss)
        response = client.post("/v1/messages", json={...})
        assert response.status_code == 200
        data = response.json()
        assert data["usage"]["cache_creation_input_tokens"] > 0

        # Second request (cache hit)
        response = client.post("/v1/messages", json={...})
        assert data["usage"]["cache_read_input_tokens"] > 0
```

## Performance Considerations

1. **Batching**: Group requests for efficient MLX inference
2. **Caching**: Minimize disk I/O with hot tier
3. **Streaming**: Yield tokens incrementally for low latency
4. **Resource Limits**: Enforce cache budget and rate limits

## See Also

- [Domain Layer](domain.md) - Core business logic
- [Adapters](adapters.md) - External interfaces
- [Architecture Overview](../architecture.md) - System design
