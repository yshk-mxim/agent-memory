# Port Design Strategy: Sprint 2

**Date**: 2026-01-24
**Author**: SE (Software Engineer)
**Status**: ✅ APPROVED for Day 2
**Sprint**: 2 - Block-Pool Batch Engine

---

## Context

Sprint 1 delivered 6 ports (3 inbound, 3 outbound) implementing hexagonal architecture. Sprint 2 planning document proposes 3 additional ports (`GenerationEnginePort`, `CacheStorePort`, `ModelProviderPort`). This document analyzes whether to **extend existing ports** or **create new ports**.

---

## Sprint 1 Ports (Existing)

### Inbound Ports

| Port | Methods | Abstraction Level |
|------|---------|-------------------|
| `InferencePort` | `generate(agent_id, prompt, max_tokens, temperature)` | Request-response, synchronous inference |
| `AgentManagementPort` | `create_agent`, `delete_agent`, `list_agents`, `get_agent_info` | Agent lifecycle (CRUD) |
| `ModelManagementPort` | `load_model`, `unload_model`, `get_current_model`, `list_available_models` | Model hot-swap |

### Outbound Ports

| Port | Methods | Abstraction Level |
|------|---------|-------------------|
| `ModelBackendPort` | `generate(prompt_tokens, cache, ...)`, `extract_model_spec()` | Low-level inference backend (MLX, vLLM) |
| `CachePersistencePort` | `save`, `load`, `exists`, `delete`, `list_cached_agents` | Disk I/O for caches (safetensors) |
| `TokenizerPort` | `encode`, `decode`, `eos_token_id`, `vocab_size` | Text ↔ tokens |

---

## Sprint 2 Proposed Ports (from Planning Doc)

### Proposed Port 1: `GenerationEnginePort`

```python
class GenerationEnginePort(Protocol):
    def submit(self, prompt: str, cache: Any | None, max_tokens: int) -> str: ...
    def step(self) -> Iterator[CompletedGeneration]: ...
```

**Analysis**:
- **Abstraction**: Async/batching engine with submit/step pattern
- **vs InferencePort**: DIFFERENT - InferencePort is synchronous request-response, GenerationEnginePort is async with iterator
- **Decision**: **CREATE NEW PORT** (not an extension)
- **Rationale**: BlockPoolBatchEngine needs batching semantics (submit to queue, step through batch). This is fundamentally different from synchronous `generate()`.

---

### Proposed Port 2: `CacheStorePort`

```python
class CacheStorePort(Protocol):
    def get(self, cache_key: CacheKey) -> AgentBlocks | None: ...
    def put(self, cache_key: CacheKey, blocks: AgentBlocks) -> None: ...
```

**Analysis**:
- **Abstraction**: In-memory cache management (trie, LRU, hot/warm/cold tiers)
- **vs CachePersistencePort**: DIFFERENT - CachePersistencePort is disk I/O, CacheStorePort is memory management
- **Decision**: **CREATE NEW PORT** (separate concern)
- **Rationale**:
  - `CachePersistencePort` handles safetensors serialization to disk
  - `CacheStorePort` handles in-memory cache tiers, prefix matching, LRU eviction
  - These are orthogonal concerns (Sprint 3 will have `AgentCacheStore` that USES both ports)

---

### Proposed Port 3: `ModelProviderPort`

```python
class ModelProviderPort(Protocol):
    def load_model(self, model_id: str) -> tuple[Any, Any]: ...  # (model, tokenizer)
    def extract_spec(self, model: Any) -> ModelCacheSpec: ...
```

**Analysis**:
- **Abstraction**: Model loading + spec extraction
- **vs Existing Ports**:
  - `load_model(model_id)` → **OVERLAPS** with `ModelManagementPort.load_model(model_id)`
  - `extract_spec(model)` → **OVERLAPS** with `ModelBackendPort.extract_model_spec()`
- **Decision**: **DO NOT CREATE** (redundant)
- **Rationale**:
  - Use `ModelManagementPort.load_model()` for loading
  - Use `ModelBackendPort.extract_model_spec()` for spec extraction
  - Creating third port would violate YAGNI and create confusion

**Alternative**: If we need combined operation, add method to existing port:
```python
# Add to ModelManagementPort (inbound):
def load_model_with_spec(self, model_id: str) -> ModelCacheSpec: ...
```

---

## Decision Matrix

| Proposed Port | Decision | Action |
|--------------|----------|--------|
| `GenerationEnginePort` | ✅ **CREATE NEW** | Truly new abstraction (batching) |
| `CacheStorePort` | ✅ **CREATE NEW** | Separate concern (memory vs disk) |
| `ModelProviderPort` | ❌ **REJECT** | Use existing ports |

---

## Final Port Inventory (After Sprint 2)

### Inbound Ports (4 total)

1. **InferencePort** (Sprint 1)
   - Synchronous request-response inference
   - Used by: High-level API adapters (Anthropic, OpenAI)

2. **AgentManagementPort** (Sprint 1)
   - Agent lifecycle (CRUD)
   - Used by: Direct Agent API adapter

3. **ModelManagementPort** (Sprint 1)
   - Model hot-swap operations
   - Used by: Admin API, model registry

4. **GenerationEnginePort** (Sprint 2 ✅ NEW)
   - Async batching engine (submit/step)
   - Used by: Application services that wrap BlockPoolBatchEngine

---

### Outbound Ports (4 total)

1. **ModelBackendPort** (Sprint 1)
   - Low-level inference backend
   - Implemented by: MLXBackendAdapter

2. **CachePersistencePort** (Sprint 1)
   - Disk I/O for caches
   - Implemented by: SafetensorsCacheAdapter

3. **TokenizerPort** (Sprint 1)
   - Text ↔ tokens
   - Implemented by: HFTokenizerAdapter

4. **CacheStorePort** (Sprint 2 ✅ NEW)
   - In-memory cache management (trie, LRU)
   - Implemented by: AgentCacheStore (Sprint 3)

---

## Port Hierarchy & Dependencies

```
┌─────────────────────────────────────────────────────────────┐
│ INBOUND PORTS (Driving Side)                                │
│                                                              │
│ InferencePort (synchronous)                                 │
│   └─> Used by: AnthropicAPIAdapter, OpenAIAPIAdapter        │
│                                                              │
│ GenerationEnginePort (async/batching) ✅ NEW                │
│   └─> Used by: ConcurrentScheduler (wraps BlockPoolEngine)  │
│                                                              │
│ AgentManagementPort (CRUD)                                  │
│   └─> Used by: DirectAgentAPIAdapter                        │
│                                                              │
│ ModelManagementPort (hot-swap)                              │
│   └─> Used by: AdminAPIAdapter, ModelRegistry               │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ OUTBOUND PORTS (Driven Side)                                │
│                                                              │
│ ModelBackendPort (inference)                                │
│   └─> Implemented by: MLXBackendAdapter                     │
│                                                              │
│ CachePersistencePort (disk I/O)                             │
│   └─> Implemented by: SafetensorsCacheAdapter               │
│                                                              │
│ CacheStorePort (memory management) ✅ NEW                   │
│   └─> Implemented by: AgentCacheStore                       │
│                                                              │
│ TokenizerPort (tokenization)                                │
│   └─> Implemented by: HFTokenizerAdapter                    │
└─────────────────────────────────────────────────────────────┘
```

**Key Insight**: Ports are organized by **abstraction level** and **concern**, not by "which sprint created them."

---

## Implementation Plan

### 1. Create `GenerationEnginePort` (Sprint 2, Day 2)

**Location**: `/src/semantic/ports/inbound.py`

```python
from typing import Iterator, Protocol

class GenerationEnginePort(Protocol):
    """Port for async batching inference engine.

    This port defines the contract for batch-based text generation
    where requests are submitted to a queue and processed in batches.
    Distinct from InferencePort which is synchronous request-response.

    Used by: Application services that need batching semantics
    (e.g., ConcurrentScheduler wrapping BlockPoolBatchEngine).
    """

    def submit(
        self,
        agent_id: str,
        prompt: str,
        cache: Any | None = None,
        max_tokens: int = 256,
    ) -> str:
        """Submit a generation request to the batch queue.

        Args:
            agent_id: Unique identifier for the agent.
            prompt: Input text to continue.
            cache: Optional pre-built cache (AgentBlocks).
            max_tokens: Maximum tokens to generate.

        Returns:
            Request UID for tracking this generation.

        Raises:
            PoolExhaustedError: If no blocks available.
            InvalidRequestError: If parameters invalid.
        """
        ...

    def step(self) -> Iterator[CompletedGeneration]:
        """Execute one batch decode step and yield completed generations.

        Yields:
            CompletedGeneration for each sequence that finished this step.
            (finish_reason=stop or finish_reason=length)

        Notes:
            - Call repeatedly until all in-flight requests complete
            - Non-blocking: returns empty iterator if no completions
        """
        ...
```

**CompletedGeneration** value object (add to `domain/value_objects.py`):
```python
@dataclass(frozen=True)
class CompletedGeneration:
    """Result of a completed generation request."""

    uid: str
    text: str
    blocks: AgentBlocks
    finish_reason: str  # "stop", "length", "error"
    token_count: int
```

---

### 2. Create `CacheStorePort` (Sprint 2, Day 2)

**Location**: `/src/semantic/ports/outbound.py`

```python
from typing import Protocol

from semantic.domain.entities import AgentBlocks
from semantic.domain.value_objects import CacheKey

class CacheStorePort(Protocol):
    """Port for in-memory cache management.

    This port defines the contract for cache storage with prefix matching,
    LRU eviction, and tier management (hot/warm/cold). Distinct from
    CachePersistencePort which handles disk I/O.

    Implemented by: AgentCacheStore (Sprint 3).
    """

    def get(self, cache_key: CacheKey) -> AgentBlocks | None:
        """Retrieve cache for agent, with prefix matching.

        Args:
            cache_key: Key containing agent_id + token prefix hash.

        Returns:
            AgentBlocks if cache exists (exact or prefix match), None otherwise.

        Notes:
            - Performs trie-based prefix matching
            - Loads from disk if in warm/cold tier
            - Updates LRU on access
        """
        ...

    def put(self, cache_key: CacheKey, blocks: AgentBlocks) -> None:
        """Store cache for agent in memory.

        Args:
            cache_key: Key containing agent_id + token prefix hash.
            blocks: AgentBlocks to store.

        Raises:
            PoolExhaustedError: If no space for new cache entry.

        Notes:
            - Triggers LRU eviction if memory pressure
            - Evicted caches move to warm tier (disk)
        """
        ...

    def evict(self, agent_id: str) -> None:
        """Manually evict agent cache to disk.

        Args:
            agent_id: Unique identifier for the agent.

        Notes:
            - Moves from hot tier (memory) to warm tier (disk)
            - Frees blocks via BlockPool.free()
        """
        ...

    def delete(self, agent_id: str) -> None:
        """Permanently delete agent cache (memory + disk).

        Args:
            agent_id: Unique identifier for the agent.

        Raises:
            AgentNotFoundError: If agent does not exist.
        """
        ...
```

---

### 3. Update Sprint 2 Plan (Remove `ModelProviderPort`)

**File**: `/project/sprints/sprint_2_block_pool_batch_engine.md`

**Change** (lines 108-127):

**Before**:
```markdown
- `ModelProviderPort` Protocol:
  ```python
  class ModelProviderPort(Protocol):
      def load_model(self, model_id: str) -> tuple[Any, Any]: ...
      def extract_spec(self, model: Any) -> ModelCacheSpec: ...
  ```
```

**After**:
```markdown
[REMOVED - Use existing ModelManagementPort + ModelBackendPort]
```

---

## Rationale

### Why Create `GenerationEnginePort` (Not Extend `InferencePort`)?

**Concern**: "Can't we just add `submit()` and `step()` methods to InferencePort?"

**Answer**: No, because they represent fundamentally different **interaction patterns**:

| Aspect | InferencePort | GenerationEnginePort |
|--------|--------------|----------------------|
| **Pattern** | Synchronous request-response | Async submit/poll |
| **Batching** | No (single request) | Yes (batch decode) |
| **Concurrency** | Caller blocks until done | Caller submits, polls later |
| **Use Case** | Direct API call (immediate response) | Background processing (batched) |
| **Return Type** | `GenerationResult` (complete) | Iterator (streaming) |

**Example**:
```python
# InferencePort usage (synchronous)
result = inference_service.generate(agent_id="a1", prompt="Hello", max_tokens=50)
print(result.text)  # Blocking call, returns when done

# GenerationEnginePort usage (async)
uid = engine.submit(agent_id="a1", prompt="Hello", max_tokens=50)
# ... do other work ...
for completion in engine.step():
    if completion.uid == uid:
        print(completion.text)
```

**Conclusion**: Different semantics → separate ports.

---

### Why Create `CacheStorePort` (Not Extend `CachePersistencePort`)?

**Concern**: "Both deal with caches. Why not one port?"

**Answer**: Different **layers of abstraction**:

| Aspect | CachePersistencePort | CacheStorePort |
|--------|---------------------|----------------|
| **Concern** | Disk I/O (serialization) | Memory management (data structures) |
| **Operations** | save, load, delete (files) | get, put, evict (trie, LRU) |
| **Adapter** | SafetensorsCacheAdapter | AgentCacheStore |
| **Layer** | Infrastructure (disk) | Application (memory) |
| **Used By** | AgentCacheStore | ConcurrentScheduler |

**Dependency**: `AgentCacheStore` (implements `CacheStorePort`) USES `CachePersistencePort` internally:
```python
class AgentCacheStore:
    def __init__(self, persistence: CachePersistencePort, pool: BlockPool):
        self._persistence = persistence  # Uses CachePersistencePort
        self._pool = pool
        self._hot_tier = {}  # In-memory trie

    def evict(self, agent_id: str) -> None:
        """Evict to disk (hot → warm)."""
        blocks = self._hot_tier.pop(agent_id)
        cache = self._blocks_to_cache(blocks)
        self._persistence.save(agent_id, cache)  # Uses CachePersistencePort
```

**Conclusion**: Separate concerns → separate ports.

---

### Why Reject `ModelProviderPort`?

**Concern**: "Sprint 2 plan explicitly lists it. Why reject?"

**Answer**: **Redundancy** violates YAGNI:

| ModelProviderPort Method | Existing Alternative |
|-------------------------|---------------------|
| `load_model(model_id)` | `ModelManagementPort.load_model(model_id)` |
| `extract_spec(model)` | `ModelBackendPort.extract_model_spec()` |

**Sprint 2 plan context**: The plan proposed `ModelProviderPort` because it wasn't aware Sprint 1 already created overlapping ports. Now that we've analyzed, we see redundancy.

**If combined operation needed**, extend existing port:
```python
# Add to ModelManagementPort:
def load_model_with_spec(self, model_id: str) -> ModelCacheSpec:
    """Load model and return its cache spec in one call."""
    self.load_model(model_id)
    return self._backend.extract_model_spec()  # Delegates to ModelBackendPort
```

**Conclusion**: Use existing ports, reject new redundant port.

---

## Impact on Sprint 2 Implementation

### Week 2 Implementation (Day 6-10) Changes

**Before** (Sprint 2 plan assumed 3 new ports):
- Implement `GenerationEnginePort` ✅ KEEP
- Implement `CacheStorePort` ✅ KEEP
- Implement `ModelProviderPort` ❌ REMOVE

**After** (this decision):
- Implement `GenerationEnginePort` (BlockPoolBatchEngine)
- Implement `CacheStorePort` (defer to Sprint 3 - AgentCacheStore)
- Use existing `ModelManagementPort` + `ModelBackendPort`

**Net Effect**: **-1 port** to implement, **+0 methods** to existing ports.

---

## Validation

### Hexagonal Architecture Compliance

✅ **Dependency Rule**: All new ports are protocols, no concrete dependencies in domain.

✅ **Separation of Concerns**: Each port has single responsibility:
- `GenerationEnginePort`: Batching semantics
- `CacheStorePort`: Memory management
- Existing ports: Unchanged

✅ **Testability**: Each port can be mocked independently for unit tests.

✅ **Extensibility**: New adapters can implement ports without modifying domain.

---

### ADR-001 Alignment

**ADR-001 states**:
> "Port interfaces are defined using Protocol (PEP 544), enabling implicit implementation without inheritance."

✅ Both new ports use `Protocol`.

**ADR-001 states**:
> "Domain layer has ZERO imports from mlx, fastapi, safetensors."

✅ Both new ports are in `ports/` layer, not domain. They import domain types (`AgentBlocks`, `CacheKey`) but not infrastructure.

**ADR-001 states**:
> "Avoid creating unnecessary abstraction layers."

✅ We rejected `ModelProviderPort` to avoid redundancy.

---

## Next Steps (Day 2)

1. **SE: Implement port interfaces** (Afternoon, 2 hours)
   - Add `GenerationEnginePort` to `/src/semantic/ports/inbound.py`
   - Add `CacheStorePort` to `/src/semantic/ports/outbound.py`
   - Add `CompletedGeneration` to `/src/semantic/domain/value_objects.py`
   - Update port docstrings with cross-references

2. **SE: Update Sprint 2 plan** (Afternoon, 30 minutes)
   - Remove `ModelProviderPort` from Day 1-2 task list (line 121-127)
   - Add note explaining use of existing ports

3. **SE: Update architecture diagrams** (Sprint 7)
   - Defer to documentation sprint (Sprint 7)
   - Add to `/docs/architecture.md` Mermaid diagrams

---

## References

- **ADR-001**: Hexagonal Architecture (Protocol-based ports)
- **Sprint 2 Plan**: `/project/sprints/sprint_2_block_pool_batch_engine.md` (lines 108-127)
- **Sprint 1 Ports**:
  - `/src/semantic/ports/inbound.py` (InferencePort, AgentManagementPort, ModelManagementPort)
  - `/src/semantic/ports/outbound.py` (ModelBackendPort, CachePersistencePort, TokenizerPort)
- **PEP 544**: Protocols (structural subtyping)

---

**Decision**: ✅ APPROVED
**Author**: SE (Software Engineer)
**Reviewed By**: ML, QE, PM (Day 2 standup)
**Implementation**: Day 2 afternoon (2-3 hours)

