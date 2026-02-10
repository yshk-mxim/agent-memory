# Architecture Overview

Semantic implements **Hexagonal Architecture** (Ports & Adapters pattern) with **Domain-Driven Design** principles.

## Design Philosophy

### Core Principles

1. **Dependency Rule**: All dependencies point inward toward the domain
2. **Domain Isolation**: Core business logic has zero external dependencies
3. **Ports & Adapters**: Infrastructure concerns isolated at boundaries
4. **Testability**: Easy to test with fakes/mocks at port boundaries

### Why Hexagonal?

- **Evolvability**: Swap MLX for another backend without touching domain
- **Testability**: Unit test domain logic without MLX, FastAPI, or disk I/O
- **Clarity**: Clear separation of "what" (domain) vs "how" (adapters)
- **Future-proof**: Add new protocols (gRPC, WebSocket) without domain changes

## Architecture Layers

```mermaid
graph TB
    subgraph "Inbound Adapters (Driving)"
        A[Anthropic API Adapter]
        B[OpenAI API Adapter]
        C[Direct Agent API]
    end
    subgraph "Application Services (Orchestration)"
        D[ConcurrentScheduler]
        E[BlockPoolBatchEngine]
        F[AgentCacheStore]
        G[ModelRegistry]
    end
    subgraph "Domain Core (Business Logic)"
        H[BlockPool Service]
        I[ModelCacheSpec]
        J[AgentBlocks Entity]
        K[KVBlock Entity]
    end
    subgraph "Outbound Adapters (Driven)"
        L[MLX Backend]
        M[Safetensors Persistence]
        N[HF Tokenizer]
        O[Metrics Logger]
    end

    A --> D
    B --> D
    C --> D
    D --> E
    D --> F
    E --> H
    F --> H
    G --> I
    E --> L
    F --> M
    E --> N

    style H fill:#e1f5ff
    style I fill:#e1f5ff
    style J fill:#e1f5ff
    style K fill:#e1f5ff
```

**Dependency Direction**: All arrows point toward the domain core (blue).

## Layer Details

### 1. Domain Core (Zero Dependencies)

**Location**: `src/agent_memory/domain/`

**Purpose**: Pure business logic, independent of frameworks.

**Components**:

```python
# Entities (mutable, identity-based)
class AgentBlocks:
    agent_id: str
    blocks: dict[int, list[KVBlock]]  # layer_id -> blocks
    total_tokens: int

    def add_block(self, block: KVBlock) -> None: ...
    def remove_block(self, block_id: int, layer_id: int) -> KVBlock | None: ...

class KVBlock:
    block_id: int
    layer_id: int
    token_count: int  # 0-256
    layer_data: dict[str, Any] | None

    def is_full(self) -> bool: ...
    def is_empty(self) -> bool: ...

# Value Objects (immutable, value-based equality)
@dataclass(frozen=True)
class ModelCacheSpec:
    n_layers: int
    n_kv_heads: int
    head_dim: int
    block_tokens: int = 256
    layer_types: list[str]
    sliding_window_size: int | None = None

    @classmethod
    def from_model(cls, model: Any) -> "ModelCacheSpec": ...
    def bytes_per_block_per_layer(self) -> int: ...

# Domain Services
class BlockPool:
    def __init__(self, spec: ModelCacheSpec, total_blocks: int): ...
    def allocate(self, n_blocks: int, layer_id: int, agent_id: str) -> list[KVBlock]: ...
    def free(self, blocks: list[KVBlock], agent_id: str) -> None: ...
    def reconfigure(self, new_spec: ModelCacheSpec) -> None: ...
```

**Invariants Enforced**:
- `used_blocks + available_blocks = total_blocks` (always)
- `0 <= block.token_count <= 256` (always)
- Block ownership tracked per agent
- Sliding window layers have max blocks cap

**Tests**: 44 unit tests, 3 property-based tests (Hypothesis), 95%+ coverage

### 2. Ports (Interface Definitions)

**Location**: `src/agent_memory/ports/`

**Purpose**: Define contracts between layers.

```python
# Inbound Ports (driving side)
class InferencePort(Protocol):
    def generate(self, agent_id: str, prompt: str, **kwargs) -> GenerationResult: ...

class AgentManagementPort(Protocol):
    def create_agent(self, agent_id: str, system_prompt: str) -> None: ...
    def delete_agent(self, agent_id: str) -> None: ...

# Outbound Ports (driven side)
class ModelBackendPort(Protocol):
    def load_model(self, model_id: str) -> Any: ...
    def prefill(self, tokens: list[int], cache: Any) -> tuple[Any, Any]: ...
    def decode(self, tokens: list[int], cache: Any) -> tuple[Any, Any]: ...

class CachePersistencePort(Protocol):
    def save_cache(self, cache: Any, path: str) -> None: ...
    def load_cache(self, path: str) -> Any: ...
```

### 3. Application Services (Orchestration)

**Location**: `src/agent_memory/application/`

**Purpose**: Coordinate domain objects and ports.

**Key Services**:

**ConcurrentScheduler** (planned):
- Per-agent locks (prevent race conditions)
- 10ms batching window
- Graceful degradation on pool exhaustion

**BlockPoolBatchEngine** (planned):
- Block-based prefill and decode
- Block allocation/extension during generation
- Cache extraction per sequence

**AgentCacheStore** (planned):
- Trie-based prefix matching
- Three-tier eviction (hot/warm/cold)
- LRU policy

**ModelRegistry** (planned):
- Model hot-swap capability
- TTL-based unloading
- ModelCacheSpec extraction

### 4. Adapters

#### Inbound Adapters (API Protocols)

**Location**: `src/agent_memory/adapters/inbound/`

**Purpose**: Translate external requests to domain operations.

**Planned**:
- `anthropic_api.py` — Anthropic Messages API (SSE streaming, tools, thinking)
- `openai_api.py` — OpenAI-compatible + session_id extension
- `direct_api.py` — Direct agent CRUD + stateful generation

#### Outbound Adapters (Infrastructure)

**Location**: `src/agent_memory/adapters/outbound/`

**Purpose**: Implement ports using real infrastructure.

**Planned**:
- `mlx_backend.py` — MLX model loading, prefill, decode, cache extraction
- `safetensors_cache.py` — Disk persistence with atomic writes
- `hf_tokenizer.py` — HuggingFace tokenizer integration
- `metrics.py` — Structured logging and metrics

## Request Flow

```mermaid
sequenceDiagram
    participant Client
    participant API as Anthropic Adapter
    participant Sched as Scheduler
    participant Engine as BatchEngine
    participant Pool as BlockPool
    participant MLX as MLX Backend
    participant Disk as Cache Persistence

    Client->>API: POST /v1/messages
    API->>Sched: generate(agent_id, prompt)
    Sched->>Sched: Acquire per-agent lock
    Sched->>Disk: Load agent cache (if exists)
    Disk-->>Sched: Cached KV blocks
    Sched->>Pool: Allocate blocks for new tokens
    Pool-->>Sched: Block IDs
    Sched->>Engine: submit(agent_id, prompt, cache)
    Note over Engine: Wait 10ms for batch window
    Engine->>MLX: prefill + decode (batched)
    MLX-->>Engine: Tokens + updated cache
    Engine->>Pool: Free old blocks, allocate new
    Engine-->>Sched: GenerationResult
    Sched->>Disk: Persist updated cache
    Sched-->>API: Response text
    API-->>Client: SSE stream
```

## State Machines

### Agent Cache Lifecycle

```mermaid
stateDiagram-v2
    [*] --> Cold: First request
    Cold --> Hot: Cache loaded to memory
    Hot --> Hot: Generate (cache updated)
    Hot --> Warm: LRU pressure (low priority)
    Warm --> Hot: Cache reloaded
    Warm --> Cold: Evicted to disk
    Cold --> [*]: Agent deleted

    note right of Cold
        No blocks allocated
        Cache on disk only
    end note

    note right of Warm
        Blocks freed
        Cache on disk
        Metadata in memory
    end note

    note right of Hot
        Blocks allocated
        Cache in memory
        Active generation
    end note
```

### Block Lifecycle

```mermaid
stateDiagram-v2
    [*] --> Free: Pool initialization
    Free --> Allocated: allocate()
    Allocated --> InUse: Generation started
    InUse --> Allocated: Generation step
    Allocated --> Free: free()

    note right of Free
        In free list (LIFO)
        No agent owner
    end note

    note right of Allocated
        Owned by agent
        Tracked in AgentBlocks
    end note

    note right of InUse
        Referenced by MLX
        Cannot be freed
    end note
```

## Block Pool Memory Model

```mermaid
classDiagram
    class BlockPool {
        -ModelCacheSpec spec
        -int total_blocks
        -dict~str,AgentBlocks~ agents
        -list~int~ free_blocks
        +allocate(n_blocks, layer_id, agent_id) list~KVBlock~
        +free(blocks, agent_id) None
        +reconfigure(new_spec) None
        +max_batch_size() int
    }

    class ModelCacheSpec {
        +int n_layers
        +int n_kv_heads
        +int head_dim
        +int block_tokens
        +list~str~ layer_types
        +int? sliding_window_size
        +bytes_per_block_per_layer() int
        +max_blocks_for_layer(layer_type) int?
    }

    class AgentBlocks {
        +str agent_id
        +dict~int,list~KVBlock~~ blocks
        +int total_tokens
        +add_block(block) None
        +remove_block(block_id, layer_id) KVBlock?
        +blocks_for_layer(layer_id) list~KVBlock~
    }

    class KVBlock {
        +int block_id
        +int layer_id
        +int token_count
        +dict? layer_data
        +bool is_full()
        +bool is_empty()
    }

    BlockPool --> ModelCacheSpec: parameterized by
    BlockPool --> AgentBlocks: manages
    AgentBlocks --> KVBlock: contains
```

## Model Hot-Swap Flow

```mermaid
flowchart LR
    A[Drain Active Requests] --> B[Evict Hot Caches]
    B --> C[mx.metal.clear_cache]
    C --> D[del model]
    D --> E[Load New Model]
    E --> F[Extract ModelCacheSpec]
    F --> G[Reconfigure BlockPool]
    G --> H[Resume Serving]

    style C fill:#ffe6e6
    style D fill:#ffe6e6
    style G fill:#e6ffe6
```

**Critical Steps**:
1. **Drain**: Wait for in-flight requests to complete
2. **Evict**: Save all hot caches to disk, free blocks
3. **Clear**: `mx.metal.clear_cache()` to reclaim GPU memory
4. **Delete**: `del model` to release Python references
5. **Load**: Load new model from HuggingFace/local
6. **Extract**: Create ModelCacheSpec from new model config
7. **Reconfigure**: Update BlockPool with new spec
8. **Resume**: Accept new requests

**Target**: <30s total latency

## Design Decisions (ADRs)

| ADR | Decision | Rationale |
|-----|----------|-----------|
| ADR-001 | Hexagonal Architecture | Evolvability, testability, clarity |
| ADR-002 | Block Size = 256 Tokens | Universal across all model architectures |
| ADR-004 | Block Gather Strategy | Better memory efficiency than padding |
| ADR-005 | Three-Tier Cache Lifecycle | Balance memory pressure and latency |
| ADR-006 | Multi-Protocol Agent ID | Support content-based + explicit IDs |
| ADR-007 | One Model At A Time | 24GB memory constraint on M4 Pro |

Full ADRs in `project/architecture/` directory.

## Quality Guarantees

- **Domain Coverage**: 95.07% (target: 95%+) ✅
- **Property Tests**: 3 core invariants tested with Hypothesis
- **Type Safety**: mypy --strict (100% type coverage)
- **No External Dependencies**: Domain core imports only from stdlib + domain

## Next Steps

- **Domain Layer**: [Domain entities and services](architecture/domain.md)
- **Application Layer**: [Orchestration services](architecture/application.md)
- **Adapters**: [Infrastructure implementations](architecture/adapters.md)
- **API Reference**: [Complete API docs](api-reference.md)
