# Domain Core

Domain layer implementation for agent-memory (Hexagonal Architecture).

## Overview

The domain layer contains core business logic and entities, independent of frameworks and infrastructure. It defines the essential concepts and rules for semantic caching with MLX models.

## Key Entities

### ModelCacheSpec

**Purpose**: Specification for model architecture and cache requirements

**Location**: `src/agent_memory/domain/model_cache_spec.py`

**Attributes**:
- `num_layers: int` - Number of transformer layers
- `num_kv_heads: int` - Number of KV cache heads
- `head_dim: int` - Dimension of each attention head
- `block_size: int` - Tokens per cache block (default: 256)
- `dtype_bytes: int` - Bytes per value (FP16=2, quantized=1)

**Methods**:
```python
def bytes_per_block_per_layer() -> int:
    """Calculate memory per block per layer."""
    # 2 caches (K + V) * num_heads * head_dim * block_size * dtype_bytes
    return 2 * self.num_kv_heads * self.head_dim * self.block_size * self.dtype_bytes

@classmethod
def from_model(cls, model, block_size: int = 256) -> ModelCacheSpec:
    """Extract spec from loaded MLX model."""
    config = model.config
    return cls(
        num_layers=config.get("num_hidden_layers"),
        num_kv_heads=config.get("num_key_value_heads"),
        head_dim=config.get("head_dim"),
        block_size=block_size,
        dtype_bytes=2,  # FP16
    )
```

**Example**:
```python
# Gemma 3 spec
spec = ModelCacheSpec(
    num_layers=42,
    num_kv_heads=16,
    head_dim=256,
    block_size=256,
    dtype_bytes=2,
)

bytes_per_block = spec.bytes_per_block_per_layer()
# = 2 * 16 * 256 * 256 * 2 = 4,194,304 bytes per block per layer
```

### CachedKVBlocks

**Purpose**: Container for KV cache blocks with metadata

**Location**: `src/agent_memory/domain/block_pool.py` (as part of BlockPool)

**Attributes**:
- `blocks: list` - List of KV cache blocks (MLX arrays)
- `total_tokens: int` - Total tokens cached
- `model_tag: str` - Model identifier for validation

**Operations**:
- `append_block(block)` - Add new cache block
- `get_blocks() -> list` - Retrieve all blocks
- `serialize() -> bytes` - Serialize for disk persistence
- `deserialize(bytes) -> CachedKVBlocks` - Load from disk

## Core Services

### BlockPool

**Purpose**: Manages allocation and lifecycle of KV cache blocks

**Location**: `src/agent_memory/domain/block_pool.py`

**Responsibilities**:
1. Allocate blocks from budget-limited pool
2. Track block usage per agent
3. Free blocks when agents are evicted
4. Enforce cache budget constraints

**Interface**:
```python
class BlockPool:
    def __init__(self, total_blocks: int, block_size: int):
        """Initialize pool with total capacity."""

    def allocate(self, agent_id: str, num_blocks: int) -> list[int]:
        """Allocate blocks for agent."""

    def free(self, agent_id: str) -> None:
        """Free all blocks for agent."""

    def get_stats() -> dict:
        """Get pool statistics (used, free, total)."""
```

**Design Pattern**: Resource Pool Pattern

**Example**:
```python
# Create pool with 1000 blocks
pool = BlockPool(total_blocks=1000, block_size=256)

# Allocate 10 blocks for agent
block_ids = pool.allocate("agent-1", num_blocks=10)

# Free when agent is evicted
pool.free("agent-1")
```

### BatchEngine

**Purpose**: Orchestrates batched MLX inference with cache management

**Location**: `src/agent_memory/application/batch_engine.py`

**Responsibilities**:
1. Accept generation requests from multiple agents
2. Batch compatible requests for efficient inference
3. Manage KV cache blocks per agent
4. Stream token generation results
5. Handle cache persistence

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
        """Submit generation request, returns unique ID."""

    def step() -> Iterator[GenerationResult]:
        """Execute one inference step, yields results."""

    def get_agent_cache(self, agent_id: str) -> CachedKVBlocks:
        """Retrieve current cache for agent."""
```

**Design Pattern**: Command Pattern + Iterator

**Example**:
```python
engine = BlockPoolBatchEngine(model, tokenizer, spec, pool_size=1000)

# Submit request
uid = engine.submit(
    agent_id="agent-1",
    prompt="Hello, how are you?",
    cache=None,  # First request
    max_tokens=50,
)

# Execute until complete
for result in engine.step():
    if result.uid == uid:
        print(result.text)  # Incremental text
        if result.finish_reason == "stop":
            break
```

## Value Objects

### GenerationResult

**Purpose**: Immutable result from inference step

**Attributes**:
- `uid: str` - Unique request identifier
- `text: str` - Generated text (incremental)
- `token_count: int` - Tokens generated so far
- `finish_reason: str | None` - "stop", "length", or None (ongoing)

### AgentCacheMetadata

**Purpose**: Metadata for persisted cache

**Attributes**:
- `agent_id: str` - Agent identifier
- `model_tag: str` - Model used for cache
- `total_tokens: int` - Tokens in cache
- `created_at: datetime` - Creation timestamp
- `last_used: datetime` - Last access time

## Domain Rules

### Cache Budget Enforcement

**Rule**: Total allocated blocks must not exceed configured budget

**Implementation**: BlockPool checks available blocks before allocation

**Violation**: Raises `PoolExhaustedError`

**Example**:
```python
# Pool has 1000 blocks, 950 used
try:
    pool.allocate("agent-new", num_blocks=100)  # Would exceed budget
except PoolExhaustedError:
    # Evict LRU agent to make space
    cache_store.evict_lru()
    pool.allocate("agent-new", num_blocks=100)  # Retry
```

### Model Tag Validation

**Rule**: Cached blocks must match current model

**Implementation**: AgentCacheStore validates model_tag on load

**Violation**: Cache is discarded, fresh cache created

**Rationale**: Prevents using incompatible cache from different model architecture

**Example**:
```python
# Load cache
cached = cache_store.load("agent-1")

if cached.model_tag != current_model_tag:
    # Discard incompatible cache
    cached = None
```

### Block Alignment

**Rule**: Cache must be block-aligned (tokens % block_size == 0)

**Implementation**: BatchEngine pads to block boundaries

**Rationale**: MLX KV cache operates on fixed-size blocks

**Example**:
```python
# 300 tokens with block_size=256
# Pad to 512 tokens (2 blocks)
padded_tokens = math.ceil(300 / 256) * 256  # = 512
```

## Domain Events

While not explicitly implemented as event objects, the domain has implicit events:

1. **CacheCreated**: When agent gets first cache block
2. **CacheUpdated**: When cache grows during generation
3. **CacheEvicted**: When agent is removed from memory
4. **CachePersisted**: When cache is saved to disk
5. **PoolExhausted**: When no blocks available

These are handled through method calls rather than event bus.

## Invariants

**Maintained by domain layer**:

1. **Block Accounting**:
   - Sum of allocated blocks ≤ total pool size
   - Each agent's blocks are tracked correctly

2. **Cache Consistency**:
   - Cache token count matches actual block count * block_size
   - Model tag is always set and valid

3. **Resource Safety**:
   - Blocks are freed when agents are deleted
   - No memory leaks from orphaned blocks

4. **Generatio Correctness**:
   - Generated tokens are deterministic given same prompt + cache
   - Cache reuse produces consistent results

## Dependencies

**Domain layer dependencies** (minimal):
- `mlx` - For MLX array types in cache
- `mlx_lm` - For model and tokenizer types
- Python standard library

**No dependencies on**:
- FastAPI
- Pydantic
- File I/O libraries
- Networking

This maintains hexagonal architecture principle of domain independence.

## Testing Domain Layer

**Unit tests** focus on business logic:

```python
def test_model_cache_spec_bytes_calculation():
    """Test memory calculation correctness."""
    spec = ModelCacheSpec(
        num_layers=10,
        num_kv_heads=4,
        head_dim=64,
        block_size=256,
        dtype_bytes=2,
    )

    bytes_per_block = spec.bytes_per_block_per_layer()
    expected = 2 * 4 * 64 * 256 * 2  # 262,144 bytes

    assert bytes_per_block == expected

def test_block_pool_allocation_limit():
    """Test pool enforces budget."""
    pool = BlockPool(total_blocks=10, block_size=256)

    # Allocate 10 blocks
    pool.allocate("agent-1", num_blocks=10)

    # Try to allocate more - should fail
    with pytest.raises(PoolExhaustedError):
        pool.allocate("agent-2", num_blocks=1)
```

## See Also

- [Application Layer](application.md) - Orchestration and use cases
- [Adapters](adapters.md) - External interfaces
- [Architecture Overview](../architecture.md) - System architecture
