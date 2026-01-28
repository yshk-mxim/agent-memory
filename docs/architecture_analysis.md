# Semantic Caching Server - Complete Architecture Analysis

**Version**: 2.0.0
**Date**: 2026-01-27
**Status**: Production Debug Analysis

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Core Data Structures](#2-core-data-structures)
3. [Quantization System](#3-quantization-system)
4. [Cache Architecture](#4-cache-architecture)
5. [Storage System](#5-storage-system)
6. [Memory Management](#6-memory-management)
7. [Execution Flows](#7-execution-flows)
8. [API Layer](#8-api-layer)
9. [Configuration](#9-configuration)
10. [Calculations & Formulas](#10-calculations--formulas)
11. [Known Issues & Fixes](#11-known-issues--fixes)

---

## 1. System Overview

### 1.1 Purpose

The Semantic Caching Server provides persistent, multi-agent KV cache management for LLM inference on Apple Silicon. It enables:

- **Session continuity**: Agents resume conversations without re-processing context
- **Memory efficiency**: Q4 quantization saves 75% memory vs FP16
- **Multi-agent support**: Multiple concurrent agents with isolated caches
- **Claude Code CLI compatibility**: Anthropic Messages API implementation

### 1.2 Technology Stack

```
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI + SSE Starlette                   │
│                    (HTTP/SSE Protocol Layer)                 │
├─────────────────────────────────────────────────────────────┤
│                    Application Services                      │
│   BlockPoolBatchEngine │ AgentCacheStore │ ModelRegistry     │
├─────────────────────────────────────────────────────────────┤
│                    Domain Core (Pure Python)                 │
│   KVBlock │ AgentBlocks │ BlockPool │ ModelCacheSpec         │
├─────────────────────────────────────────────────────────────┤
│                    MLX / mlx_lm Framework                    │
│   BatchGenerator │ QuantizedKVCache │ Metal GPU Backend      │
├─────────────────────────────────────────────────────────────┤
│                    Apple Silicon (M1/M2/M3/M4)               │
│              Unified Memory Architecture (UMA)               │
└─────────────────────────────────────────────────────────────┘
```

### 1.3 Model Configuration (DeepSeek-Coder-V2-Lite)

```python
# Extracted from model.args at runtime
ModelCacheSpec:
    n_layers: 27              # Transformer layers
    n_kv_heads: 16            # Key-Value attention heads
    head_dim: 128             # Dimension per head
    block_tokens: 256         # Tokens per block (universal)
    kv_bits: 4                # Quantization bits (Q4)
    kv_group_size: 64         # Quantization group size
    vocab_size: 102400        # Vocabulary size
    max_context: 100000       # Maximum context length
```

---

## 2. Core Data Structures

### 2.1 KVBlock - The Fundamental Unit

**File**: `src/semantic/domain/entities.py:18-78`

```python
@dataclass
class KVBlock:
    """Single 256-token cache block.

    Memory Layout (Q4 format):
    ┌─────────────────────────────────────────────────────────────┐
    │ block_id: int          │ Unique identifier (0 to total_blocks-1) │
    │ layer_id: int          │ Which transformer layer (0 to 26)       │
    │ token_count: int       │ Actual tokens stored (0 to 256)         │
    │ layer_data: dict | None│ {"k": tensor, "v": tensor}              │
    └─────────────────────────────────────────────────────────────┘

    layer_data["k"] and layer_data["v"] are TUPLES when quantized:
        (weights: mx.array, scales: mx.array, biases: mx.array)

    Tensor Shapes (Q4):
        weights: [1, n_kv_heads, token_count, head_dim // 8]  # Packed 4-bit
        scales:  [1, n_kv_heads, token_count, head_dim // group_size]
        biases:  [1, n_kv_heads, token_count, head_dim // group_size]
    """
    block_id: int
    layer_id: int
    token_count: int = 0
    layer_data: dict[str, Any] | None = None

    def is_full(self) -> bool:
        return self.token_count >= 256

    def is_empty(self) -> bool:
        return self.token_count == 0
```

**Memory per Block (Q4)**:
```
Per token: 16 heads × 128 dim × 0.5 bytes (4-bit) × 2 (K+V) = 2,048 bytes
Per block: 256 tokens × 2,048 = 524,288 bytes ≈ 0.5 MB
Per layer (75 blocks for 19K tokens): 75 × 0.5 = 37.5 MB
All 27 layers: 27 × 37.5 = 1,012 MB ≈ 1 GB
```

### 2.2 AgentBlocks - Per-Agent Cache Container

**File**: `src/semantic/domain/entities.py:81-225`

```python
@dataclass
class AgentBlocks:
    """Collection of blocks for a single agent.

    Structure:
    ┌─────────────────────────────────────────────────────────────┐
    │ agent_id: str                                               │
    │   └── Format: "msg_{hash16}" or "sess_{session_id}"         │
    │                                                             │
    │ blocks: dict[int, list[KVBlock]]                            │
    │   └── layer_id → [block0, block1, ..., blockN]              │
    │   └── Layer 0:  [KVBlock(0,0,256), KVBlock(1,0,256), ...]   │
    │   └── Layer 1:  [KVBlock(75,1,256), KVBlock(76,1,256), ...] │
    │   └── ...                                                   │
    │   └── Layer 26: [KVBlock(1950,26,256), ...]                 │
    │                                                             │
    │ total_tokens: int                                           │
    │   └── Actual tokens in cache (e.g., 19,234)                 │
    │   └── NOT buffer size (which rounds to 256 boundaries)      │
    │                                                             │
    │ token_sequence: list[int]                                   │
    │   └── PROMPT tokens only (not generated)                    │
    │   └── Used for prefix matching                              │
    │   └── Example: [1, 234, 567, 890, ...]                      │
    │                                                             │
    │ metadata: dict[str, Any]                                    │
    │   └── model_id, timestamp, etc.                             │
    └─────────────────────────────────────────────────────────────┘
    """
    agent_id: str
    blocks: dict[int, list[KVBlock]]
    total_tokens: int = 0
    token_sequence: list[int] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def common_prefix_length(self, query_tokens: list[int]) -> int:
        """Find common prefix between stored tokens and query.

        CRITICAL for cache reuse decisions:
        - Exact match: common == len(stored) == len(query)
        - Partial match: common == len(stored) < len(query) → EXTEND
        - Divergent: common < len(stored) → CANNOT reuse

        BPE BOUNDARY ISSUE:
        tokenize("Hello world") + tokenize(" how are you")
            ≠ tokenize("Hello world how are you")

        Solution: Only store PROMPT tokens, not generated tokens.
        """
        stored = self.token_sequence
        common = 0
        for i, (a, b) in enumerate(zip(stored, query_tokens)):
            if a != b:
                break
            common = i + 1
        return common

    def blocks_for_layer(self, layer_id: int) -> list[KVBlock]:
        return self.blocks.get(layer_id, [])

    def num_blocks(self) -> int:
        return sum(len(layer_blocks) for layer_blocks in self.blocks.values())
```

### 2.3 BlockPool - Memory Allocator

**File**: `src/semantic/domain/services.py:22-412`

```python
class BlockPool:
    """Thread-safe block allocation service.

    Memory Model:
    ┌─────────────────────────────────────────────────────────────┐
    │                      Total Blocks: 14,563                    │
    │                      (8192 MB / 0.56 MB per block)           │
    ├─────────────────────────────────────────────────────────────┤
    │  free_list: list[int]                                       │
    │    └── Stack of available block IDs (LIFO for locality)     │
    │    └── [14562, 14561, 14560, ..., 100, 99, ...]             │
    │                                                             │
    │  allocated_blocks: dict[int, KVBlock]                       │
    │    └── block_id → KVBlock instance                          │
    │    └── Tracks which blocks are in use                       │
    │                                                             │
    │  agent_allocations: dict[str, set[int]]                     │
    │    └── agent_id → {block_ids}                               │
    │    └── "msg_abc123" → {0, 1, 2, 75, 76, 77, ...}           │
    │                                                             │
    │  _lock: threading.RLock                                     │
    │    └── Reentrant lock for thread safety                     │
    └─────────────────────────────────────────────────────────────┘

    Operations:
    - allocate(n, layer_id, agent_id) → O(n) pop from free_list
    - free(blocks, agent_id) → O(n) push to free_list
    - reconfigure(new_spec) → Clear all, rebuild for new model
    """

    def __init__(self, spec: ModelCacheSpec, total_blocks: int):
        self._spec = spec
        self._total_blocks = total_blocks
        self._free_list = list(range(total_blocks))  # LIFO stack
        self._allocated_blocks: dict[int, KVBlock] = {}
        self._agent_allocations: dict[str, set[int]] = {}
        self._lock = threading.RLock()

    def allocate(self, n_blocks: int, layer_id: int, agent_id: str) -> list[KVBlock]:
        """Allocate n blocks for agent's layer.

        Raises:
            PoolExhaustedError: If not enough blocks available
        """
        with self._lock:
            if len(self._free_list) < n_blocks:
                raise PoolExhaustedError(
                    f"Need {n_blocks}, only {len(self._free_list)} available"
                )

            blocks = []
            for _ in range(n_blocks):
                block_id = self._free_list.pop()  # LIFO
                block = KVBlock(block_id=block_id, layer_id=layer_id)
                self._allocated_blocks[block_id] = block

                if agent_id not in self._agent_allocations:
                    self._agent_allocations[agent_id] = set()
                self._agent_allocations[agent_id].add(block_id)

                blocks.append(block)

            return blocks

    def free(self, blocks: list[KVBlock], agent_id: str) -> None:
        """Return blocks to pool."""
        with self._lock:
            for block in blocks:
                if block.block_id in self._allocated_blocks:
                    del self._allocated_blocks[block.block_id]
                    self._free_list.append(block.block_id)

                    if agent_id in self._agent_allocations:
                        self._agent_allocations[agent_id].discard(block.block_id)
```

### 2.4 ModelCacheSpec - Model Geometry

**File**: `src/semantic/domain/value_objects.py:21-104`

```python
@dataclass(frozen=True)
class ModelCacheSpec:
    """Immutable specification of model's cache geometry.

    Extracted from model.args at load time.
    Used to calculate memory requirements and validate cache compatibility.
    """
    n_layers: int           # 27 for DeepSeek-Coder-V2-Lite
    n_kv_heads: int         # 16
    head_dim: int           # 128
    block_tokens: int       # 256 (universal)
    layer_types: list[str]  # ["full_attention"] * 27
    sliding_window_size: int | None  # None for full attention
    kv_bits: int | None     # 4 for Q4, None for FP16
    kv_group_size: int | None  # 64 for Q4

    def bytes_per_block_per_layer(self) -> int:
        """Calculate memory per block per layer.

        Q4: head_dim × n_kv_heads × block_tokens × 2 (K+V) × 0.5 bytes
        FP16: head_dim × n_kv_heads × block_tokens × 2 (K+V) × 2 bytes
        """
        bytes_per_element = 0.5 if self.kv_bits == 4 else 2.0
        return int(
            self.head_dim * self.n_kv_heads * self.block_tokens * 2 * bytes_per_element
        )

    def bytes_per_token_all_layers(self) -> int:
        """Total bytes per token across all layers."""
        bytes_per_element = 0.5 if self.kv_bits == 4 else 2.0
        return int(
            self.n_kv_heads * self.head_dim * 2 * bytes_per_element * self.n_layers
        )
        # = 16 × 128 × 2 × 0.5 × 27 = 55,296 bytes/token (Q4)
```

### 2.5 CompletedGeneration - Generation Result

**File**: `src/semantic/domain/value_objects.py:116-124`

```python
@dataclass(frozen=True)
class CompletedGeneration:
    """Result of a completed generation.

    Returned by BatchEngine.step() when a sequence finishes.
    """
    uid: str                    # Unique request ID
    text: str                   # Generated text (decoded)
    blocks: AgentBlocks         # Updated cache blocks
    finish_reason: str          # "stop" (EOS) or "length" (max_tokens)
    token_count: int            # Number of generated tokens
```

---

## 3. Quantization System

### 3.1 Q4 Quantization Format

MLX uses 4-bit quantization with per-group scaling:

```
Original FP16 tensor: [1, 16, 19234, 128]  # [batch, heads, seq, dim]
                      = 78.8 MB

Quantized Q4:
  weights: [1, 16, 19234, 16]   # head_dim/8 = 128/8 = 16 (packed)
           = 9.85 MB
  scales:  [1, 16, 19234, 2]    # head_dim/group_size = 128/64 = 2
           = 0.61 MB
  biases:  [1, 16, 19234, 2]    # Same as scales
           = 0.61 MB

Total Q4: 11.07 MB (14% of FP16) per K or V
         22.14 MB for K+V per layer
         598 MB for all 27 layers (vs 2.13 GB FP16)
```

### 3.2 QuantizedKVCache (MLX-LM)

**File**: `mlx_lm/models/cache.py` (external)

```python
class QuantizedKVCache:
    """MLX's built-in quantized cache.

    Attributes:
        keys: tuple[mx.array, mx.array, mx.array] | None
              (weights, scales, biases) in Q4 format
        values: tuple[mx.array, mx.array, mx.array] | None
        offset: int  # Current sequence position
        group_size: int  # 64
        bits: int  # 4

    CRITICAL: offset must equal actual token count, NOT buffer size.
    BatchGenerator uses offset to determine cache hit vs miss.
    """
    def __init__(self, group_size: int = 64, bits: int = 4):
        self.keys = None
        self.values = None
        self.offset = 0
        self.group_size = group_size
        self.bits = bits

    def size(self) -> int:
        """Return actual sequence length."""
        return self.offset  # PATCHED: was returning 0
```

### 3.3 BatchQuantizedKVCache (Custom Extension)

**File**: `src/semantic/adapters/outbound/mlx_quantized_extensions.py:21-481`

```python
class BatchQuantizedKVCache(_BaseCache):
    """Batched quantized KV cache for BatchGenerator.

    Purpose: Enable Q4 caches with MLX's BatchGenerator.

    Key Method: merge() - Combines multiple QuantizedKVCache into batch.

    CRITICAL FIX (Line 222):
    OLD: headroom = max(num_steps * 10, 16384)  # 16K tokens!
    NEW: headroom = min(max(num_steps + 256, 512), 1024)  # Max 1K

    Memory Impact:
    OLD: 16,384 × 55,296 bytes × 27 layers = 24.4 GB → OOM!
    NEW: 1,024 × 55,296 bytes × 27 layers = 1.5 GB → Safe
    """
    step = 256  # Expansion granularity

    def __init__(self, padding=None, group_size=64, bits=4):
        self.keys = None      # (weights, scales, biases) tuples
        self.values = None
        self.offset = 0
        self._left_padding = []
        self._lengths = []
        self.group_size = group_size
        self.bits = bits

    @classmethod
    def merge(cls, caches: list[QuantizedKVCache]) -> "BatchQuantizedKVCache":
        """Merge multiple Q4 caches into batched cache.

        NO DEQUANTIZATION - stays Q4 throughout!

        Process:
        1. Find max sequence length across caches
        2. Calculate left padding for shorter sequences
        3. Allocate batched Q4 tensors
        4. Copy each cache with padding offset
        5. mx.eval() to materialize
        """
        # ... implementation

    def update_and_fetch(self, keys, values):
        """Append new tokens to Q4 cache.

        Called by model's attention layer during generation.

        Steps:
        1. Check if expansion needed
        2. Quantize new FP16 tokens to Q4
        3. Append to cache tensors
        4. Return trimmed cache for attention

        HEADROOM FIX HERE (Line 222):
        headroom = min(max(num_steps + 256, 512), 1024)
        """
        # ... implementation
```

### 3.4 Quantization Flow

```
┌─────────────────────────────────────────────────────────────┐
│                     STORAGE (Disk)                           │
│  safetensors file with Q4 tuples per layer                   │
│  ├── layer_0_k: (weights, scales, biases)                    │
│  ├── layer_0_v: (weights, scales, biases)                    │
│  └── ...                                                     │
└───────────────────────────┬─────────────────────────────────┘
                            │ load
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                   AgentBlocks (Memory)                       │
│  blocks[layer_id][block_idx].layer_data = {                  │
│      "k": (weights, scales, biases),  # Q4 tuple             │
│      "v": (weights, scales, biases)   # Q4 tuple             │
│  }                                                           │
└───────────────────────────┬─────────────────────────────────┘
                            │ _reconstruct_cache()
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              QuantizedKVCache (MLX Object)                   │
│  .keys = (all_weights, all_scales, all_biases)  # Q4        │
│  .values = (all_weights, all_scales, all_biases) # Q4       │
│  .offset = actual_tokens (e.g., 19234)                       │
│                                                              │
│  DIRECT INJECTION: No FP16 conversion!                       │
│  MLX's quantized_matmul operates on Q4 directly.             │
└───────────────────────────┬─────────────────────────────────┘
                            │ BatchGenerator.insert()
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              BatchQuantizedKVCache (Batched)                 │
│  Merged Q4 caches for concurrent sequences                   │
│  Handles padding for variable-length sequences               │
└───────────────────────────┬─────────────────────────────────┘
                            │ model(tokens, cache=cache)
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              MLX Attention Layer                             │
│  quantized_matmul(Q @ K.T) - operates on Q4 directly        │
│  No dequantization overhead!                                 │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. Cache Architecture

### 4.1 Three-Tier Cache System

**File**: `src/semantic/application/agent_cache_store.py:121-549`

```
┌─────────────────────────────────────────────────────────────┐
│                      HOT TIER (Memory)                       │
│  _hot_cache: dict[str, CacheEntry]                           │
│                                                              │
│  CacheEntry:                                                 │
│    agent_id: str                                             │
│    blocks: AgentBlocks (with layer_data populated)           │
│    model_tag: ModelTag (for compatibility check)             │
│    last_accessed: float (timestamp)                          │
│    access_count: int                                         │
│    is_hot: bool = True                                       │
│                                                              │
│  Capacity: max_hot_agents (default: 5)                       │
│  Eviction: LRU (least recently accessed)                     │
├─────────────────────────────────────────────────────────────┤
│                      WARM TIER (Disk)                        │
│  _warm_cache: dict[str, Path]                                │
│                                                              │
│  Location: ~/.semantic/caches/{agent_id}.safetensors         │
│                                                              │
│  Metadata stored with cache:                                 │
│    - model_id                                                │
│    - n_layers, n_kv_heads, head_dim                          │
│    - total_tokens                                            │
│    - token_sequence (for prefix matching)                    │
│    - timestamp                                               │
│                                                              │
│  Promotion: warm → hot on access                             │
├─────────────────────────────────────────────────────────────┤
│                      COLD TIER (Regenerate)                  │
│                                                              │
│  Agent exists but no cache available.                        │
│  Must regenerate from scratch via chunked prefill.           │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 Cache Operations

```python
class AgentCacheStore:
    def save(self, agent_id: str, blocks: AgentBlocks) -> None:
        """Save cache to hot tier + disk.

        Flow:
        1. Validate model compatibility
        2. Add to _hot_cache
        3. Persist to disk (always, for recovery)
        4. Trigger LRU eviction if over max_hot_agents
        """

    def load(self, agent_id: str) -> AgentBlocks | None:
        """Load cache, checking hot → warm → None.

        Flow:
        1. Check _hot_cache
           - If found AND blocks have data: return
           - If found but layer_data cleared: remove, continue
        2. Check _warm_cache (disk)
           - If found: load, validate, promote to hot
        3. Return None (cache miss)
        """

    def invalidate_hot(self, agent_id: str) -> None:
        """Remove stale hot entry after blocks cleared.

        Called after _reconstruct_cache() clears layer_data.
        Prevents returning empty blocks on next load.
        Warm tier (disk) remains valid for future loads.
        """

    def evict_lru(self, target_count: int) -> int:
        """Evict oldest entries to reach target count.

        Sorts by last_accessed, persists to disk, removes from hot.
        """
```

### 4.3 ModelTag - Cache Compatibility

```python
@dataclass(frozen=True)
class ModelTag:
    """Identifies model for cache compatibility.

    Caches are model-specific. Different models have:
    - Different tokenizers (incompatible token IDs)
    - Different architectures (incompatible KV shapes)

    When model changes, existing caches cannot be reused.
    Agent identity persists, but cache starts fresh.
    """
    model_id: str
    n_layers: int
    n_kv_heads: int
    head_dim: int
    block_tokens: int

    def is_compatible(self, spec: ModelCacheSpec) -> bool:
        return (
            self.n_layers == spec.n_layers and
            self.n_kv_heads == spec.n_kv_heads and
            self.head_dim == spec.head_dim and
            self.block_tokens == spec.block_tokens
        )
```

---

## 5. Storage System

### 5.1 Safetensors Format

**File**: `src/semantic/adapters/outbound/safetensors_cache_adapter.py`

```
File: ~/.semantic/caches/msg_abc123def456.safetensors

Structure:
┌─────────────────────────────────────────────────────────────┐
│ METADATA (JSON header):                                      │
│   {                                                          │
│     "model_id": "mlx-community/DeepSeek-Coder-V2-Lite...",   │
│     "n_layers": 27,                                          │
│     "n_kv_heads": 16,                                        │
│     "head_dim": 128,                                         │
│     "total_tokens": 19234,                                   │
│     "token_sequence": "[1, 234, 567, ...]",  # JSON array   │
│     "timestamp": "2026-01-27T16:00:00Z"                      │
│   }                                                          │
├─────────────────────────────────────────────────────────────┤
│ TENSORS:                                                     │
│   layer_0_k_weights: [1, 16, 19456, 16] uint32               │
│   layer_0_k_scales:  [1, 16, 19456, 2] float16               │
│   layer_0_k_biases:  [1, 16, 19456, 2] float16               │
│   layer_0_v_weights: [1, 16, 19456, 16] uint32               │
│   layer_0_v_scales:  [1, 16, 19456, 2] float16               │
│   layer_0_v_biases:  [1, 16, 19456, 2] float16               │
│   layer_1_k_weights: ...                                     │
│   ...                                                        │
│   layer_26_v_biases: ...                                     │
└─────────────────────────────────────────────────────────────┘

Total tensors: 27 layers × 6 tensors (k/v × weights/scales/biases) = 162
File size: ~1 GB for 19K token cache (Q4)
```

### 5.2 Save/Load Operations

```python
class SafetensorsCacheAdapter:
    def save(self, agent_id: str, blocks: AgentBlocks, metadata: dict) -> Path:
        """Persist cache to disk.

        Steps:
        1. Extract tensors from blocks
        2. Build tensor dict with naming convention
        3. Serialize metadata to JSON
        4. Write to temp file
        5. Atomic rename to final path
        """
        tensors = {}
        for layer_id, layer_blocks in blocks.blocks.items():
            # Concatenate all blocks for layer
            k_parts, v_parts = [], []
            for block in layer_blocks:
                k_parts.append(block.layer_data["k"])
                v_parts.append(block.layer_data["v"])

            # Handle Q4 tuples
            k_full = self._concat_q4(k_parts)
            v_full = self._concat_q4(v_parts)

            # Store with naming convention
            tensors[f"layer_{layer_id}_k_weights"] = k_full[0]
            tensors[f"layer_{layer_id}_k_scales"] = k_full[1]
            tensors[f"layer_{layer_id}_k_biases"] = k_full[2]
            # ... same for v

        # Atomic write
        temp_path = path.with_suffix(".tmp")
        save_file(tensors, temp_path, metadata=metadata)
        temp_path.rename(path)

    def load(self, cache_path: Path) -> tuple[dict, dict]:
        """Load cache from disk.

        Returns:
            (tensors_dict, metadata_dict)
        """
        with safe_open(cache_path, framework="mlx") as f:
            metadata = json.loads(f.metadata().get("metadata", "{}"))
            tensors = {name: f.get_tensor(name) for name in f.keys()}
        return tensors, metadata
```

---

## 6. Memory Management

### 6.1 Budget Calculation

```python
# From api_server.py startup

# Total RAM: 24 GB (M4 Pro)
# Metal limit: ~18 GB (75% of RAM)
# Model weights: ~8 GB (DeepSeek Q4)
# OS overhead: ~2 GB
# Available for cache: ~8 GB

cache_budget_mb = 8192  # 8 GB for cache

# Calculate blocks
mb_per_block = spec.bytes_per_block_per_layer() * spec.n_layers / (1024 * 1024)
             = 524288 * 27 / (1024 * 1024)
             = 13.5 MB per block (all layers)

# Wait, that's wrong. Let me recalculate:
# Per block per layer = 256 tokens × 2048 bytes/token = 524,288 bytes = 0.5 MB
# But we allocate blocks PER LAYER, not across all layers
# So mb_per_block ≈ 0.5 MB

total_blocks = cache_budget_mb / mb_per_block
             = 8192 / 0.56
             = 14,563 blocks
```

### 6.2 Memory Tracking

```python
def _log_memory(self, label: str) -> tuple[float, float, float]:
    """Log MLX memory state."""
    import mlx.core as mx
    active = mx.get_active_memory() / (1024**3)   # Currently allocated
    cache = mx.get_cache_memory() / (1024**3)     # MLX internal cache
    peak = mx.get_peak_memory() / (1024**3)       # Session maximum
    logger.info(f"[MEMORY {label}] Active: {active:.2f}GB, Cache: {cache:.2f}GB, Peak: {peak:.2f}GB")
    return active, cache, peak
```

### 6.3 Memory Lifecycle

```
REQUEST START
├── Baseline: ~8 GB (model weights)
│
├── CACHE LOAD (if hit)
│   ├── Load from disk: +0 GB (mmap)
│   └── After load: ~8 GB
│
├── CACHE RECONSTRUCT
│   ├── Concatenate Q4 blocks: +~1 GB
│   └── After reconstruct: ~9 GB
│
├── CLEAR BLOCKS (after reconstruct)
│   ├── Set layer_data = None
│   ├── gc.collect()
│   └── After clear: ~9 GB (Q4 in KVCache)
│
├── GENERATION
│   ├── BatchGenerator.insert()
│   ├── For each token:
│   │   ├── Forward pass: +~0.5 GB temporary
│   │   ├── update_and_fetch(): +headroom (max 1K tokens)
│   │   ├── mx.eval(): materialize
│   │   └── Clear intermediates
│   └── Peak during generation: ~10-11 GB
│
├── CACHE EXTRACT
│   ├── Quantize if needed
│   ├── Split into blocks
│   └── After extract: ~9 GB
│
└── CACHE SAVE
    ├── Save to disk
    └── After save: ~9 GB (hot tier)
```

---

## 7. Execution Flows

### 7.1 Complete Request Flow

```python
# 1. HTTP Request arrives
@router.post("/v1/messages")
async def create_message(request_body: MessagesRequest, request: Request):
    # anthropic_adapter.py:376-545

    # 2. Convert messages to prompt
    prompt = messages_to_prompt(request_body.messages, request_body.system)

    # 3. Tokenize
    tokens = await asyncio.to_thread(tokenizer.encode, prompt)
    # Example: 19,234 tokens for large context

    # 4. Generate agent ID
    session_id = request.headers.get("X-Session-ID")
    if session_id:
        agent_id = f"sess_{session_id}"
    else:
        agent_id = generate_agent_id_from_tokens(tokens)
        # Hash first 100 tokens → "msg_abc123def456"

    # 5. Cache lookup
    cached_blocks = cache_store.load(agent_id)
    # Returns AgentBlocks or None

    # 6. Submit to batch engine
    uid = batch_engine.submit(
        agent_id=agent_id,
        prompt=prompt,
        cache=cached_blocks,
        max_tokens=request_body.max_tokens
    )

    # 7. Invalidate hot cache (blocks cleared after reconstruct)
    if cached_blocks is not None:
        cache_store.invalidate_hot(agent_id)

    # 8. Run generation
    completion = run_step_for_uid(batch_engine, uid)

    # 9. Save updated cache
    updated_blocks = batch_engine.get_agent_blocks(agent_id)
    cache_store.save(agent_id, updated_blocks)

    # 10. Return response
    return MessagesResponse(...)
```

### 7.2 Submit Flow (batch_engine.py:417-940)

```python
def submit(self, agent_id, prompt, cache, max_tokens) -> str:
    # Check draining
    if self._draining:
        raise PoolExhaustedError("Engine is draining")

    # Tokenize
    prompt_tokens = self._tokenizer.encode(prompt)

    # Cache safety check
    if cache is not None:
        bytes_per_token = spec.bytes_per_token_all_layers()  # 55,296
        max_safe_tokens = (7500 * 1024 * 1024) / bytes_per_token  # ~138K

        if cache.total_tokens > max_safe_tokens:
            cache = None  # Force cache miss

    # CASE A: Cache hit → Reconstruct
    if cache is not None:
        kv_cache = self._reconstruct_cache(cache)
        # Clear blocks to free Q4 memory (now in KVCache)
        for layer_blocks in cache.blocks.values():
            for block in layer_blocks:
                block.layer_data = None
        cache.blocks.clear()

    # CASE B: Cache miss, long prompt → Chunked prefill
    elif len(prompt_tokens) >= threshold:  # 2048
        kv_cache = self._chunked_prefill(prompt_tokens, agent_id)
        # Allocate tracking blocks
        n_blocks = (len(prompt_tokens) + 255) // 256
        blocks = self._pool.allocate(n_blocks, layer_id=0, agent_id=agent_id)

    # CASE C: Cache miss, short prompt → Standard path
    else:
        kv_cache = None
        n_blocks = (len(prompt_tokens) + 255) // 256
        blocks = self._pool.allocate(n_blocks, layer_id=0, agent_id=agent_id)

    # Determine tokens to process
    if kv_cache is not None and cache is None:
        # CASE 1: After chunked prefill
        tokens_to_process = []  # All already processed

    elif kv_cache is not None and cache is not None:
        # CASE 2: Cache hit
        common_prefix = cache.common_prefix_length(prompt_tokens)

        if common_prefix == len(cache.token_sequence) == len(prompt_tokens):
            # Exact match → treat as miss (fresh generation)
            del kv_cache
            kv_cache = None
            tokens_to_process = prompt_tokens

        elif common_prefix == len(cache.token_sequence) < len(prompt_tokens):
            # Partial match → extend
            if cache.total_tokens > common_prefix:
                self._slice_cache_to_length(kv_cache, common_prefix)
            tokens_to_process = prompt_tokens[common_prefix:]

        else:
            # Divergent → cannot reuse
            del kv_cache
            kv_cache = None
            tokens_to_process = prompt_tokens

    else:
        tokens_to_process = prompt_tokens

    # Insert into BatchGenerator
    uids = self._batch_gen.insert(
        prompts=[tokens_to_process],
        max_tokens=[max_tokens],
        caches=[kv_cache] if kv_cache else None
    )

    return uids[0]
```

### 7.3 Chunked Prefill Flow (batch_engine.py:222-315)

```python
def _chunked_prefill(self, tokens: list[int], agent_id: str) -> list[KVCache]:
    """Process tokens in adaptive chunks.

    Memory Savings:
    - Monolithic: Materialize N×N attention matrix
    - Chunked: Materialize chunk×cache attention matrix

    Chunk Sizing (adaptive_chunk_size):
    - cache_pos < 2000: max_chunk (4096)
    - cache_pos < 8000: max_chunk // 2 (2048)
    - cache_pos < 20000: max_chunk // 4 (1024)
    - cache_pos >= 20000: min_chunk (512)
    """
    import mlx.core as mx
    from mlx_lm.models.cache import QuantizedKVCache

    # Create Q4 cache for each layer
    kv_caches = [
        QuantizedKVCache(group_size=64, bits=4)
        for _ in range(self._spec.n_layers)
    ]

    pos = 0
    while pos < len(tokens):
        chunk_size = adaptive_chunk_size(pos)  # 512-4096
        end = min(pos + chunk_size, len(tokens))

        chunk_tokens = mx.array([tokens[pos:end]])

        # Forward pass updates kv_caches in-place
        y = self._model(chunk_tokens, cache=kv_caches)

        # CRITICAL: Force evaluation
        mx.eval(y)

        # CRITICAL: Clear intermediates
        mx.clear_cache()

        pos = end

    return kv_caches
```

### 7.4 Reconstruct Flow (batch_engine.py:1143-1312)

```python
def _reconstruct_cache(self, agent_blocks: AgentBlocks) -> list[KVCache]:
    """Convert AgentBlocks → list[QuantizedKVCache].

    Q4 DIRECT INJECTION: Keep Q4 format, no FP16 conversion!
    """
    from mlx_lm.models.cache import QuantizedKVCache

    cache = []

    for layer_id in range(self._spec.n_layers):
        layer_blocks = agent_blocks.blocks_for_layer(layer_id)

        if not layer_blocks:
            cache.append(QuantizedKVCache())
            continue

        # Collect K/V tensors from blocks
        k_tensors, v_tensors = [], []
        for block in layer_blocks:
            k_tensors.append(block.layer_data["k"])  # Q4 tuple
            v_tensors.append(block.layer_data["v"])  # Q4 tuple

        # Concatenate (stays Q4!)
        k_full, v_full = self._cache_adapter.concatenate_cache_blocks(
            k_tensors, v_tensors
        )

        # Q4 DIRECT INJECTION
        if isinstance(k_full, tuple) and len(k_full) == 3:
            kv_cache = QuantizedKVCache(group_size=64, bits=4)
            kv_cache.keys = k_full      # (weights, scales, biases)
            kv_cache.values = v_full    # (weights, scales, biases)

            # CRITICAL: offset = actual tokens, NOT buffer size!
            kv_cache.offset = agent_blocks.total_tokens

            # Force evaluation
            mx.eval(k_full[0], k_full[1], v_full[0], v_full[1])
        else:
            # FP16 fallback
            kv_cache = KVCache()
            kv_cache.state = (k_full, v_full)
            kv_cache.offset = agent_blocks.total_tokens

        cache.append(kv_cache)

    return cache
```

### 7.5 Extract Flow (batch_engine.py:1314-1484)

```python
def _extract_cache(self, uid, cache, token_sequence) -> AgentBlocks:
    """Convert generation cache → AgentBlocks.

    Steps:
    1. Get KV tensors from cache
    2. Quantize to Q4 if FP16
    3. Split into 256-token blocks
    4. Allocate from pool
    """
    agent_id, _, _, _ = self._active_requests[uid]

    # Extract state from KVCache objects
    if hasattr(cache[0], 'state'):
        cache = [kv.state for kv in cache]  # [(k, v), ...]

    # Quantize FP16 → Q4 if needed
    first_k = cache[0][0]
    if not isinstance(first_k, tuple):  # FP16
        quantized_cache = []
        for k, v in cache:
            k_quant = tuple(mx.quantize(k, group_size=64, bits=4))
            v_quant = tuple(mx.quantize(v, group_size=64, bits=4))
            mx.eval(*k_quant, *v_quant)
            quantized_cache.append((k_quant, v_quant))
        cache = quantized_cache

    # Get sequence length
    seq_len = self._cache_adapter.get_sequence_length(cache[0][0])
    n_blocks = (seq_len + 255) // 256

    # Allocate and populate blocks
    blocks_dict = {}
    for layer_id, (k, v) in enumerate(cache):
        allocated = self._pool.allocate(n_blocks, layer_id, agent_id)

        for block_idx, block in enumerate(allocated):
            start = block_idx * 256
            end = min(start + 256, seq_len)

            k_chunk = self._cache_adapter.slice_cache_tensor(k, start, end)
            v_chunk = self._cache_adapter.slice_cache_tensor(v, start, end)

            block.layer_data = {"k": k_chunk, "v": v_chunk}
            block.token_count = end - start

        blocks_dict[layer_id] = allocated

    return AgentBlocks(
        agent_id=agent_id,
        blocks=blocks_dict,
        total_tokens=seq_len,
        token_sequence=token_sequence
    )
```

---

## 8. API Layer

### 8.1 Anthropic Messages API

**Endpoint**: `POST /v1/messages`

```python
class MessagesRequest(BaseModel):
    model: str                    # e.g., "claude-3-5-sonnet-20241022"
    messages: list[Message]       # Conversation history
    max_tokens: int               # Max generation length
    system: str | list = ""       # System prompt
    stream: bool = False          # Enable SSE streaming
    tools: list[Tool] | None      # Tool definitions
    temperature: float = 1.0      # Sampling temperature

class MessagesResponse(BaseModel):
    id: str                       # "msg_xxx"
    content: list[ContentBlock]   # Generated content
    model: str
    stop_reason: str              # "end_turn", "max_tokens", "tool_use"
    usage: Usage

class Usage(BaseModel):
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int  # Tokens used to create cache
    cache_read_input_tokens: int      # Tokens read from cache
```

### 8.2 Agent ID Generation

```python
def generate_agent_id_from_tokens(tokens: list[int]) -> str:
    """Generate deterministic agent ID from token prefix.

    Uses first 100 tokens for stability:
    - Same conversation → same prefix → same ID → cache hit
    - Different conversations → different prefix → different ID

    Hash is truncated to 16 chars for readability.
    """
    prefix = tokens[:100]
    hash_val = hashlib.sha256(str(prefix).encode()).hexdigest()[:16]
    return f"msg_{hash_val}"
```

### 8.3 Streaming Response

```python
async def stream_generation(...) -> AsyncIterator[dict]:
    """Stream tokens as SSE events.

    Event sequence:
    1. message_start (with usage)
    2. content_block_start (text block)
    3. content_block_delta (token by token)
    4. content_block_stop
    5. message_delta (final usage)
    6. message_stop
    """
    # Submit to engine
    uid = batch_engine.submit(...)

    # Stream message_start
    yield {"event": "message_start", "data": json.dumps(MessageStartEvent(...))}

    # Stream content
    yield {"event": "content_block_start", ...}

    for result in batch_engine.step():
        if result.uid == uid:
            yield {"event": "content_block_delta", "data": json.dumps({
                "type": "text_delta",
                "text": new_text
            })}

    yield {"event": "content_block_stop", ...}
    yield {"event": "message_delta", ...}
    yield {"event": "message_stop", ...}
```

---

## 9. Configuration

### 9.1 Environment Variables

```bash
# Model settings
SEMANTIC_MODEL_ID="mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx"

# Cache settings
SEMANTIC_CACHE_BUDGET_MB=8192
SEMANTIC_MAX_HOT_AGENTS=5
SEMANTIC_CACHE_DIR="~/.semantic/caches"

# MLX settings
SEMANTIC_MLX_KV_BITS=4
SEMANTIC_MLX_KV_GROUP_SIZE=64

# Chunked prefill
SEMANTIC_MLX_CHUNKED_PREFILL_ENABLED=true
SEMANTIC_MLX_CHUNKED_PREFILL_THRESHOLD=2048
SEMANTIC_MLX_CHUNKED_PREFILL_MIN_CHUNK=512
SEMANTIC_MLX_CHUNKED_PREFILL_MAX_CHUNK=4096

# Server
SEMANTIC_HOST=0.0.0.0
SEMANTIC_PORT=8001
SEMANTIC_LOG_LEVEL=INFO
```

### 9.2 Settings Classes

```python
class MLXSettings(BaseSettings):
    kv_bits: int = 4
    kv_group_size: int = 64
    chunked_prefill_enabled: bool = True
    chunked_prefill_threshold: int = 2048
    chunked_prefill_min_chunk: int = 512
    chunked_prefill_max_chunk: int = 4096

class CacheSettings(BaseSettings):
    budget_mb: int = 8192
    max_hot_agents: int = 5
    cache_dir: Path = Path.home() / ".semantic" / "caches"

class ServerSettings(BaseSettings):
    host: str = "0.0.0.0"
    port: int = 8001
    workers: int = 1
    log_level: str = "INFO"
```

---

## 10. Calculations & Formulas

### 10.1 Memory Formulas

```python
# Bytes per token (Q4)
bytes_per_token_q4 = n_kv_heads * head_dim * 2 * 0.5 * n_layers
                   = 16 * 128 * 2 * 0.5 * 27
                   = 55,296 bytes

# Bytes per token (FP16)
bytes_per_token_fp16 = n_kv_heads * head_dim * 2 * 2 * n_layers
                     = 16 * 128 * 2 * 2 * 27
                     = 221,184 bytes

# Q4 savings
savings = 1 - (55296 / 221184) = 75%

# Bytes per block per layer (Q4)
bytes_per_block = block_tokens * n_kv_heads * head_dim * 2 * 0.5
                = 256 * 16 * 128 * 2 * 0.5
                = 524,288 bytes = 0.5 MB

# Total blocks from budget
total_blocks = (cache_budget_mb * 1024 * 1024) / bytes_per_block
             = (8192 * 1024 * 1024) / 524288
             = 16,384 blocks

# But wait - we track per layer, so:
# If budget is 8GB total, and each block is 0.5MB,
# and we need 27 copies (one per layer) of each "logical" block:
total_logical_blocks = total_blocks / n_layers
                     = 16384 / 27
                     = 606 logical blocks

# Actually, the pool allocates per-layer, so:
# total_blocks = 14,563 (as per server logs)
# Each block = 0.56 MB (includes overhead)
```

### 10.2 Capacity Calculations

```python
# Tokens per agent (19K example)
blocks_per_agent = ceil(19234 / 256) = 76 blocks per layer
total_agent_blocks = 76 * 27 = 2,052 blocks

# Max concurrent agents (theoretical)
max_agents = total_blocks / blocks_per_agent_per_layer
           = 14563 / 76
           = 191 agents

# With 5 hot agents limit
actual_max_agents = 5  # LRU eviction kicks in

# Safe cache size (7.5 GB budget)
max_safe_tokens = (7.5 * 1024**3) / bytes_per_token_q4
                = (7.5 * 1024**3) / 55296
                = 142,000 tokens
```

### 10.3 Chunked Prefill Chunk Sizes

```python
def adaptive_chunk_size(cache_pos, min_chunk=512, max_chunk=4096):
    """Larger chunks early, smaller chunks late.

    Memory for attention: O(chunk_size * cache_size)

    cache_pos=0:     chunk=4096, memory ∝ 4096 * 0 = 0
    cache_pos=4000:  chunk=2048, memory ∝ 2048 * 4000 = 8M
    cache_pos=10000: chunk=1024, memory ∝ 1024 * 10000 = 10M
    cache_pos=25000: chunk=512,  memory ∝ 512 * 25000 = 12.5M

    vs monolithic: memory ∝ 19234 * 19234 = 370M
    """
    if cache_pos < 2000:
        return max_chunk  # 4096
    elif cache_pos < 8000:
        return max_chunk // 2  # 2048
    elif cache_pos < 20000:
        return max_chunk // 4  # 1024
    else:
        return min_chunk  # 512
```

---

## 11. Known Issues & Fixes

### 11.1 Headroom OOM (FIXED)

**Location**: `mlx_quantized_extensions.py:218-222`

**Root Cause**: BatchQuantizedKVCache.update_and_fetch() allocated 16K token headroom per expansion, causing 27 × 16K × 55KB = 23.9 GB allocation.

**Fix**:
```python
# OLD (OOM):
headroom = max(num_steps * 10, 16384)

# NEW (Fixed):
headroom = min(max(num_steps + 256, 512), 1024)
```

### 11.2 Native Generation Path (DISABLED)

**Location**: `batch_engine.py:561` (`if False:`)

**Issues**:
- EOS token included in output
- Wrong prompt processing after cache divergence
- Cache corruption (19K → 324 tokens)
- Block pool exhaustion

**Status**: Disabled. BatchGenerator path used for all generation.

### 11.3 Chunked Prefill Handoff (FIXED)

**Location**: `batch_engine.py:772-782`

**Root Cause**: After chunked prefill, `cache` is None (cold start), so prefix matching code was skipped and full prompt passed to BatchGenerator again.

**Fix**:
```python
# CASE 1: kv_cache from chunked prefill (cache is None on cold start)
if kv_cache is not None and cache is None:
    tokens_to_process = []  # All tokens already in cache
```

### 11.4 QuantizedKVCache.size() (PATCHED)

**Location**: `mlx_quantized_extensions.py:508-517`

**Root Cause**: MLX's QuantizedKVCache.size() returned 0, causing BatchGenerator to treat cache hits as misses.

**Fix**: Monkey-patch to return `self.offset`:
```python
def size(self) -> int:
    return self.offset  # Actual token count
```

### 11.5 Cache Divergence Memory Leak (FIXED)

**Location**: `batch_engine.py:866-880`

**Root Cause**: When prompt diverged from cached sequence, reconstructed KV cache wasn't freed before processing full prompt.

**Fix**:
```python
# Free reconstructed cache before treating as miss
if kv_cache is not None:
    del kv_cache
    gc.collect()
    mx.clear_cache()
kv_cache = None
cache = None  # Allow chunked prefill
```

---

*Total codebase: ~7,500 lines of production Python*
*Generated: 2026-01-27*
