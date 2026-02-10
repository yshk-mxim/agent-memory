# API Reference

Complete API documentation for Semantic.

## Overview

This reference documents all public APIs in the Semantic codebase. Documentation is auto-generated from source code docstrings using **mkdocstrings**.

## Domain Layer

### Entities

::: agent_memory.domain.entities.KVBlock
    options:
      show_root_heading: true
      show_source: true

::: agent_memory.domain.entities.AgentBlocks
    options:
      show_root_heading: true
      show_source: true

### Value Objects

::: agent_memory.domain.value_objects.ModelCacheSpec
    options:
      show_root_heading: true
      show_source: true

::: agent_memory.domain.value_objects.CacheKey
    options:
      show_root_heading: true
      show_source: true

::: agent_memory.domain.value_objects.GenerationResult
    options:
      show_root_heading: true
      show_source: true

### Services

::: agent_memory.domain.services.BlockPool
    options:
      show_root_heading: true
      show_source: true
      members_order: source

## Quick Reference

### BlockPool

**Core Service**: Block-pool memory management.

```python
from agent_memory.domain.services import BlockPool
from agent_memory.domain.value_objects import ModelCacheSpec

# Create spec
spec = ModelCacheSpec(
    n_layers=48,
    n_kv_heads=8,
    head_dim=240,
    block_tokens=256,
    layer_types=["global"] * 8 + ["sliding_window"] * 40,
    sliding_window_size=1024,
)

# Initialize pool
pool = BlockPool(spec=spec, total_blocks=100)

# Allocate blocks
blocks = pool.allocate(n_blocks=5, layer_id=0, agent_id="agent_1")

# Check memory
budget = pool.budget()
print(f"Used: {budget['used_mb']:.1f} MB")
print(f"Available: {budget['available_mb']:.1f} MB")

# Free blocks
pool.free(blocks, agent_id="agent_1")

# Reconfigure for new model
new_spec = ModelCacheSpec.from_model(new_model)
pool.reconfigure(new_spec)
```

### ModelCacheSpec

**Value Object**: Model-specific cache configuration.

```python
from agent_memory.domain.value_objects import ModelCacheSpec

# Extract from MLX model
spec = ModelCacheSpec.from_model(model)

# Access properties
bytes_per_block = spec.bytes_per_block_per_layer()  # Memory per block per layer
max_blocks = spec.max_blocks_for_layer("sliding_window")  # Block cap for SWA

# Create manually
spec = ModelCacheSpec(
    n_layers=32,
    n_kv_heads=8,
    head_dim=128,
    block_tokens=256,
    layer_types=["global"] * 32,
)
```

### KVBlock

**Entity**: Single memory block (256 tokens).

```python
from agent_memory.domain.entities import KVBlock

block = KVBlock(
    block_id=42,
    layer_id=0,
    token_count=128,
    layer_data={"k": k_tensor, "v": v_tensor},
)

# Query state
if block.is_full():
    print("Block at capacity (256 tokens)")

if block.is_empty():
    print("Block unused")
```

### AgentBlocks

**Entity**: Agent's allocated blocks across layers.

```python
from agent_memory.domain.entities import AgentBlocks, KVBlock

agent = AgentBlocks(
    agent_id="agent_1",
    blocks={},
    total_tokens=0,
)

# Add block
block = KVBlock(block_id=1, layer_id=0, token_count=256, layer_data=None)
agent.add_block(block)

# Query
print(f"Total blocks: {agent.num_blocks()}")
print(f"Layers: {agent.num_layers()}")
print(f"Tokens: {agent.total_tokens}")

# Get blocks for layer
layer_0_blocks = agent.blocks_for_layer(0)

# Remove block
removed = agent.remove_block(block_id=1, layer_id=0)
```

## Method Index

### BlockPool Methods

| Method | Description |
|--------|-------------|
| `__init__(spec, total_blocks)` | Initialize pool with model spec and block count |
| `allocate(n_blocks, layer_id, agent_id)` | Allocate blocks for agent |
| `free(blocks, agent_id)` | Return blocks to free list |
| `free_agent_blocks(agent_id)` | Free all blocks for an agent |
| `allocated_block_count()` | Count allocated blocks |
| `available_blocks()` | Count free blocks |
| `used_memory()` | Memory used by allocated blocks (bytes) |
| `available_memory()` | Memory available for allocation (bytes) |
| `total_memory()` | Total pool memory (bytes) |
| `budget()` | Memory budget summary (dict) |
| `reconfigure(new_spec)` | Reconfigure pool for new model |
| `max_batch_size()` | Calculate max concurrent agents |

### ModelCacheSpec Methods

| Method | Description |
|--------|-------------|
| `from_model(model)` | Extract spec from MLX model (class method) |
| `bytes_per_block_per_layer()` | Memory per block per layer (bytes) |
| `max_blocks_for_layer(layer_type)` | Block cap for sliding window layers |

### KVBlock Methods

| Method | Description |
|--------|-------------|
| `is_full()` | Check if block has 256 tokens |
| `is_empty()` | Check if block has 0 tokens |

### AgentBlocks Methods

| Method | Description |
|--------|-------------|
| `add_block(block)` | Add block to agent |
| `remove_block(block_id, layer_id)` | Remove and return block |
| `blocks_for_layer(layer_id)` | Get all blocks for a layer |
| `num_blocks()` | Count total blocks |
| `num_layers()` | Count layers with blocks |

## Type Signatures

All public APIs have complete type annotations compatible with `mypy --strict`.

```python
# Example: BlockPool.allocate signature
def allocate(
    self,
    n_blocks: int,
    layer_id: int,
    agent_id: str
) -> list[KVBlock]:
    ...
```

## Error Handling

### ValueError

Raised for invalid inputs:

```python
# Negative block count
pool.allocate(n_blocks=-1, layer_id=0, agent_id="test")
# ValueError: n_blocks must be > 0

# Insufficient blocks
pool.allocate(n_blocks=1000, layer_id=0, agent_id="test")
# ValueError: Cannot allocate 1000 blocks, only 100 available

# Invalid layer
pool.allocate(n_blocks=1, layer_id=999, agent_id="test")
# ValueError: layer_id 999 out of range (0-47)

# Empty agent ID
pool.allocate(n_blocks=1, layer_id=0, agent_id="")
# ValueError: agent_id cannot be empty
```

### Ownership Validation

```python
# Free blocks not owned by agent
pool.free(blocks, agent_id="wrong_agent")
# ValueError: Block 42 not owned by agent 'wrong_agent'
```

## Invariants

### BlockPool Invariants

**Memory Accounting** (ALWAYS holds):
```python
assert pool.allocated_block_count() + pool.available_blocks() == pool._total_blocks
```

**Ownership Tracking**:
```python
# Every allocated block is owned by exactly one agent
# Free list contains no duplicates
```

### KVBlock Invariants

**Token Count** (ALWAYS holds):
```python
assert 0 <= block.token_count <= 256
```

### AgentBlocks Invariants

**Total Tokens** (ALWAYS holds):
```python
assert agent.total_tokens == sum(b.token_count for blocks in agent.blocks.values() for b in blocks)
```

## Examples

See [Quick Start](quick-start.md) and [User Guide](user-guide.md) for complete examples.

## Future APIs

Documentation for application services and adapters will be added as they are implemented:

- **Application Services** (planned)
    - `ConcurrentScheduler`
    - `BlockPoolBatchEngine`
    - `AgentCacheStore`
    - `ModelRegistry`

- **Adapters** (planned)
    - `AnthropicAPIAdapter`
    - `OpenAIAPIAdapter`
    - `MLXBackend`
    - `SafetensorsPersistence`

## Contributing

To add API documentation:

1. Write Google-style docstrings in source code
2. Add `::: module.Class` directive in this file
3. Run `mkdocs serve` to preview
4. Check that docstrings render correctly

See [Developer Guide](developer-guide.md) for docstring format.
