# Backend Architecture Plan: Block-Pool KV Cache with Persistent Agents

## Executive Summary

A novel inference backend for Apple Silicon that unifies DynamicKVCache and paged attention concepts into a **block-pool memory manager**. Agents draw fixed-size blocks (256 tokens) from a shared pool, enabling variable-length batching without the padding waste of MLX-LM's BatchKVCache. The system is **model-agnostic**, supporting multiple architectures (Gemma 3, GPT-OSS-20B, Qwen 2.5, Llama) through a `ModelCacheSpec` abstraction. Caches persist to disk and reload on demand, with automatic batch size adjustment based on available memory.

**Target**: Single user, multiple concurrent agents (3-10), M4 Pro 24GB.
**Models**: Gemma 3 12B, GPT-OSS-20B (MoE), Qwen 2.5-14B, Llama-class (one model loaded at a time, hot-swappable).

---

## 1. Research Findings

### 1.1 MLX-LM Current Cache Architecture

From direct source analysis of the installed `mlx_lm` package:

| Class | Strategy | Variable-Length | Persistence |
|-------|----------|-----------------|-------------|
| `KVCache` | Step-based (256 chunks) | Single sequence only | Via save_prompt_cache |
| `RotatingKVCache` | Fixed circular buffer | Single sequence only | Via save_prompt_cache |
| `BatchKVCache` | Left-padded rectangular | Pads to max in batch | No persistence |
| `BatchRotatingKVCache` | Left-padded + rotation | Pads to max in batch | No persistence |
| `QuantizedKVCache` | Step-based, quantized storage | Single sequence only | Via save_prompt_cache |

**Critical gaps**:
- NO paged attention, NO block-based allocation
- BatchKVCache wastes `O(batch * max_seq)` memory via padding
- No way to reclaim memory from finished sequences
- No prefix sharing (physical memory not shared)
- Cache is discarded after batch completes (Issue #548)

### 1.2 MLX-LM Server CacheManager (Trie-Based)

The installed `mlx_lm/server.py` contains a `CacheManager` class implementing:
- **Trie-based prefix lookup**: Token sequences stored in nested dicts
- `fetch_nearest_cache(model, tokens)` → returns (cache, remaining_tokens)
  - Exact match: returns full cache
  - Shorter match: returns prefix cache + remaining tokens to process
  - Longer match: trims longer cache to fit (via `trim_prompt_cache`)
- `insert_cache(model, tokens, prompt_cache)` → stores with reference counting
- **LRU eviction**: `max_size=10` entries, oldest evicted first
- **Deep-copy on shared access**: If `count > 1`, deep-copies cache before returning

### 1.3 MLX Unified Memory

- CPU and GPU share the same physical memory pool
- No data transfer needed between CPU and GPU operations
- `mx.array` allocations visible to both CPU and GPU
- Lazy evaluation: computation graphs executed on demand
- `wired_limit`: Pre-wires memory to prevent page faults during inference
- Metal compute kernels for GPU acceleration
- `mx.compile`: Fuses multiple kernel launches

### 1.4 Apple Silicon Performance Characteristics

- **Prefill (TTFT)**: Compute-bound → benefits from batching to amortize overhead
- **Decode (TPS)**: Memory-bandwidth-bound → each batch member adds KV read cost
- **M4 Pro**: 120 GB/s bandwidth, 24GB unified memory
- **M4 Max**: ~500 GB/s bandwidth, up to 128GB
- **Batching sweet spot**: 3-5 concurrent sequences (M4 Pro), diminishing returns beyond

### 1.5 vLLM/SGLang Concepts (Simplified for Single-GPU)

| vLLM Concept | Our Adaptation | Rationale |
|--------------|----------------|-----------|
| Block table (physical/logical) | Block pool with free list | No CPU-GPU separation needed |
| Page fault + swap | Disk persistence + reload | Unified memory = no swapping needed |
| Prefix caching (hash blocks) | Trie-based prefix matching | Already in mlx-lm server.py |
| Copy-on-write | Reference counting + deep-copy | Already in CacheManager |
| Block allocator | Pool manager with budget | Single pool, budget-aware |
| Continuous batching scheduler | Async queue with lock-per-agent | Already implemented |
| Hybrid KV Cache Manager | ModelCacheSpec per architecture | Layer-group-aware allocation |

---

## 2. Model Architecture Support

### 2.1 Supported Model Architectures

| Model | Params | 4-bit Size | Layers | KV Heads | Head Dim | Attention Pattern | Sliding Window |
|-------|--------|-----------|--------|----------|----------|-------------------|----------------|
| **Gemma 3 12B** | 12B | ~6.5 GB | 48 | 4 | 256 | Hybrid (5:1 SWA:global) | 512 tokens |
| **GPT-OSS-20B** | 21B (3.6B active) | ~11 GB | 24 | 8 | 64 | Alternating (1:1) | 128 tokens |
| **Qwen 2.5-14B** | 14.7B | ~9 GB | 48 | 8 | 128 | Uniform full attention | None |
| **Llama 3.1 8B** | 8B | ~5 GB | 32 | 8 | 128 | Uniform full attention | None |

### 2.2 ModelCacheSpec Abstraction

```python
@dataclass
class ModelCacheSpec:
    """
    Model-agnostic cache specification extracted from model config.
    Determines block allocation strategy and memory budget.
    """
    model_id: str                      # e.g., "mlx-community/gemma-3-12b-it-4bit"
    num_layers: int                    # Total layers in model
    num_kv_heads: int                  # KV heads per layer (GQA)
    head_dim: int                      # Dimension per head
    kv_hidden_size: int                # = num_kv_heads * head_dim

    # Attention pattern per layer
    layer_types: List[str]             # ["full", "swa", "swa", "full", ...]
    sliding_window_size: Optional[int] # Tokens for SWA layers (None = all full)

    # Derived properties
    num_full_layers: int               # Layers with unbounded attention
    num_swa_layers: int                # Layers with sliding window

    # Memory calculations
    bytes_per_token_per_layer: int     # = 2 * kv_hidden_size * dtype_bytes
    model_weight_bytes: int            # 4-bit model size on disk

    @classmethod
    def from_model(cls, model, model_id: str) -> "ModelCacheSpec":
        """Extract spec from loaded mlx-lm model."""
        config = model.args  # ModelArgs from config.json

        # Determine layer types
        layer_types = []
        for i in range(config.num_hidden_layers):
            if hasattr(config, 'sliding_window_pattern'):
                # Gemma 3 style: every Nth layer is global
                if (i % config.sliding_window_pattern
                    == config.sliding_window_pattern - 1):
                    layer_types.append("full")
                else:
                    layer_types.append("swa")
            elif hasattr(config, 'sliding_window') and config.sliding_window:
                # GPT-OSS style: check per-layer config or alternating
                if i % 2 == 0:  # alternating pattern
                    layer_types.append("full")
                else:
                    layer_types.append("swa")
            else:
                # Qwen/Llama: all full attention
                layer_types.append("full")

        return cls(
            model_id=model_id,
            num_layers=config.num_hidden_layers,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.hidden_size // config.num_attention_heads,
            kv_hidden_size=config.num_key_value_heads * (config.hidden_size // config.num_attention_heads),
            layer_types=layer_types,
            sliding_window_size=getattr(config, 'sliding_window', None),
            num_full_layers=layer_types.count("full"),
            num_swa_layers=layer_types.count("swa"),
            bytes_per_token_per_layer=2 * config.num_key_value_heads * (config.hidden_size // config.num_attention_heads) * 2,  # K+V, float16
            model_weight_bytes=0,  # Set after loading
        )
```

### 2.3 Per-Model Cache Characteristics

| Model | KV Bytes/Token/Layer | Full Layers | SWA Layers | Bytes/Token (Full Only) | Bytes/Token (All) |
|-------|---------------------|-------------|------------|------------------------|-------------------|
| **Gemma 3 12B** | 4,096 B | 8 | 40 | 32,768 B (~32 KB) | 196,608 B (~192 KB) |
| **GPT-OSS-20B** | 2,048 B | 12 | 12 | 24,576 B (~24 KB) | 49,152 B (~48 KB) |
| **Qwen 2.5-14B** | 4,096 B | 48 | 0 | 196,608 B (~192 KB) | 196,608 B (~192 KB) |
| **Llama 3.1 8B** | 4,096 B | 32 | 0 | 131,072 B (~128 KB) | 131,072 B (~128 KB) |

**Key insight**: GPT-OSS-20B is the most KV-cache-efficient despite being the largest model (48 KB/token vs 192 KB/token for Qwen/Gemma). This is because:
- Only 12 global layers need full-length cache (12 SWA layers cap at 128 tokens)
- Small head_dim=64 (vs 128 or 256 in others)
- MoE architecture means only 3.6B params active per token (fast decode)

---

## 3. Architecture

### 3.1 Layer Diagram

```
┌─────────────────────────────────────────────────────────┐
│                API Layer (Multi-Protocol)                 │
│  - Anthropic adapter (content-based agent ID)            │
│  - OpenAI-compatible (explicit session_id)               │
│  - Direct agent API (agent_id in request)                │
│  (See anthropic_cli_adapter.md)                          │
└────────────────────────┬────────────────────────────────┘
                         │ generate(agent_id, prompt, max_tokens)
                         ▼
┌─────────────────────────────────────────────────────────┐
│              ConcurrentAgentManager                      │
│  - Per-agent asyncio.Lock (sequential per-agent)         │
│  - Cross-agent parallel batching                         │
│  - Batch worker (10ms collection window)                 │
│  - Model-agnostic (delegates to engine)                  │
└────────────────────────┬────────────────────────────────┘
                         │ submit/step/complete
                         ▼
┌─────────────────────────────────────────────────────────┐
│            BlockPoolBatchEngine                           │
│  - Configured via ModelCacheSpec                          │
│  - Manages active batch of sequences                     │
│  - Allocates/frees blocks from pool                      │
│  - Runs prefill + decode steps                           │
│  - Extracts per-agent cache on completion                │
└────────────────────────┬────────────────────────────────┘
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
┌──────────────────────┐  ┌──────────────────────────────┐
│   BlockPool           │  │   AgentCacheStore             │
│  - Parameterized by   │  │  - Trie-based prefix lookup   │
│    ModelCacheSpec      │  │  - Disk persistence           │
│  - Fixed block_tokens │  │  - LRU eviction from memory   │
│  - Free list          │  │  - Reference counting         │
│  - Budget enforcement │  │  - Model-tagged caches        │
│  - Per-layer-group    │  │  - Quantized storage option   │
│    allocation         │  │  - Invalidation on model swap │
└──────────────────────┘  └──────────────────────────────┘
                              │
                              ▼
                    ┌──────────────────────┐
                    │  ModelRegistry         │
                    │  - Hot-swap models     │
                    │  - TTL-based unload    │
                    │  - Weight caching      │
                    │  - Cache invalidation  │
                    └──────────────────────┘
```

### 3.2 Component Responsibilities

#### ModelCacheSpec
- Extracted from model's `config.json` at load time
- Determines block memory layout (kv_hidden_size × block_tokens × layers)
- Classifies layers as "full" or "swa" for allocation strategy
- Calculates memory budget per block, per agent, and total pool capacity

#### BlockPool
- Pre-allocates a fixed memory budget at startup (e.g., 4GB for caches)
- Manages blocks of shape `(n_kv_heads, block_tokens, head_dim)` per layer
- Each block = `block_tokens` of KV cache for one layer
- **Model-aware**: block memory size varies by model's kv_hidden_size
- Free list tracks available blocks
- `allocate(n_blocks, layer_group) → List[block_id]`
- `free(block_ids)` → returns blocks to free list
- `used_memory()` / `available_memory()` for budget decisions
- **Invalidated and reallocated on model swap**

#### BlockPoolBatchEngine
- Replaces current `BatchedGenerationEngine`
- Configured by `ModelCacheSpec` at initialization
- Manages a "jagged" batch: each sequence has its own block list (not padded to max)
- During prefill: allocates blocks as tokens are processed
- During decode: allocates one block every `block_tokens` generated tokens
- On sequence completion: extracts per-agent cache, frees batch slot
- Dynamic batch sizing: if memory tight, reduces max_batch_size
- **Model-agnostic**: same engine works for any architecture via ModelCacheSpec

#### AgentCacheStore
- Enhanced version of mlx-lm's trie-based CacheManager
- Keyed by token sequence (content-addressable) OR explicit agent_id
- Three tiers: hot (in-memory blocks), warm (on-disk safetensors), cold (evicted)
- **Model-tagged**: each cache entry records which model produced it
- **Invalidation**: when model changes, all caches for old model become invalid
- `get(tokens) → (cache_blocks, remaining_tokens)`: prefix-aware lookup
- `put(agent_id, tokens, cache_blocks)`: store after generation
- `evict_to_disk(agent_id)`: saves blocks to safetensors, frees memory
- `load_from_disk(agent_id) → cache_blocks`: reloads, allocating from pool

#### ModelRegistry
- Manages model lifecycle (load, unload, hot-swap)
- TTL-based unload: idle model freed after configurable timeout
- Only ONE model loaded at a time (24GB constraint)
- When swapping models: evict all hot caches to disk, unload weights, load new model, reconfigure BlockPool
- Weight download/cache in `~/.cache/huggingface/`

---

## 4. Block-Pool Memory Management

### 4.1 Block Structure (Model-Parameterized)

```python
@dataclass
class KVBlock:
    """Single block of KV cache for one layer. Shape determined by ModelCacheSpec."""
    block_id: int
    layer_idx: int
    layer_group: str               # "full" or "swa"
    keys: mx.array                 # Shape: (n_kv_heads, block_tokens, head_dim)
    values: mx.array               # Shape: (n_kv_heads, block_tokens, head_dim)
    used_tokens: int               # 0..block_tokens
    model_id: str                  # Which model produced this block

    @property
    def is_full(self) -> bool:
        return self.used_tokens >= self.block_tokens

    @property
    def memory_bytes(self) -> int:
        """Actual memory consumed by this block."""
        return 2 * self.keys.nbytes  # K + V
```

### 4.2 Agent Block Allocation

Each agent's cache is a list of block IDs per layer:

```python
@dataclass
class AgentBlocks:
    """Per-agent block allocation across all layers."""
    agent_id: str
    model_id: str                          # Which model these blocks belong to
    blocks_per_layer: List[List[int]]      # [layer_idx][block_sequence] → block_id
    total_tokens: int
    layer_groups: Dict[str, List[int]]     # "full" → [layer indices], "swa" → [layer indices]

    def context_length(self) -> int:
        """Total tokens cached for this agent."""
        return self.total_tokens

    def is_valid_for_model(self, model_id: str) -> bool:
        """Check if this cache is compatible with the current model."""
        return self.model_id == model_id
```

### 4.3 Pool Budget Management (Architecture-Aware)

```python
class BlockPool:
    def __init__(self, total_budget_mb: int, spec: ModelCacheSpec):
        """
        Pre-allocate blocks within memory budget.
        Block sizes are determined by the ModelCacheSpec.

        GEMMA 3 12B (4 KV heads, head_dim=256, 48 layers):
          Per block per layer: 4 * 256 * 256 * 2 * 2 bytes = 1MB (K+V, float16)
          Per block (full layers only): 8 * 1MB = 8MB
          Per block (swa layers): 40 * 1MB = 40MB (but capped at 2 blocks/layer)
          Effective per 256 tokens: ~48MB

        GPT-OSS-20B (8 KV heads, head_dim=64, 24 layers):
          Per block per layer: 8 * 64 * 256 * 2 * 2 bytes = 0.5MB
          Per block (full layers): 12 * 0.5MB = 6MB
          Per block (swa layers): 12 * 0.5MB (capped at 1 block/layer since sw=128<256)
          Effective per 256 tokens: ~12MB

        QWEN 2.5-14B (8 KV heads, head_dim=128, 48 layers):
          Per block per layer: 8 * 128 * 256 * 2 * 2 bytes = 1MB
          Per block all layers: 48 * 1MB = 48MB
          Effective per 256 tokens: ~48MB

        Budget 4GB:
          Gemma 3:  ~83 blocks → ~21K tokens across all agents
          GPT-OSS:  ~333 blocks → ~85K tokens across all agents
          Qwen 2.5: ~83 blocks → ~21K tokens across all agents
        """
        self.block_tokens = 256  # Fixed across all models
        self.spec = spec
        self.bytes_per_block = self._compute_block_bytes(spec)
        self.total_blocks = (total_budget_mb * 1024 * 1024) // self.bytes_per_block
        self.free_list: List[int] = list(range(self.total_blocks))
        self._blocks: Dict[int, KVBlock] = {}

    def _compute_block_bytes(self, spec: ModelCacheSpec) -> int:
        """Compute memory per block based on model architecture."""
        # For full layers: each block stores block_tokens of KV
        full_bytes = spec.num_full_layers * spec.bytes_per_token_per_layer * self.block_tokens
        # For SWA layers: also block_tokens, but allocation is capped
        swa_bytes = spec.num_swa_layers * spec.bytes_per_token_per_layer * self.block_tokens
        return full_bytes + swa_bytes

    def reconfigure(self, new_spec: ModelCacheSpec):
        """Reconfigure pool for a different model. Frees all existing blocks."""
        self.spec = new_spec
        self.bytes_per_block = self._compute_block_bytes(new_spec)
        self.total_blocks = (self.budget_bytes) // self.bytes_per_block
        self.free_list = list(range(self.total_blocks))
        self._blocks.clear()
```

### 4.4 Dynamic Batch Sizing

The key trade-off: longer context vs. more parallel agents.

```
Available blocks = total_blocks - reserved_for_active_agents

Scenario A (short contexts, GPT-OSS-20B):
  5 agents × 1K tokens each = 5 × 4 blocks = 20 blocks (240MB)
  → Batch size 5, plenty of room for growth

Scenario B (long context, Qwen 2.5-14B):
  1 agent × 8K tokens = 32 blocks (1.5GB)
  2 agents × 2K tokens = 16 blocks (768MB)
  → Total: 48 blocks, only 3 in batch, reduced parallelism

Scenario C (MoE model, GPT-OSS-20B):
  10 agents × 4K tokens = 160 blocks (1.9GB)
  → Batch size 10 feasible due to small KV footprint!
```

Decision algorithm:
```python
def max_batch_size(self) -> int:
    """Calculate current max batch size based on pool state and model."""
    available = len(self.free_list)
    # Reserve at least 4 blocks (1K tokens) per potential batch member
    min_blocks_per_agent = 4
    return min(
        self.configured_max_batch,
        available // min_blocks_per_agent
    )
```

### 4.5 SWA Layer Block Capping

For models with sliding window attention, blocks for SWA layers are capped:

```python
def max_swa_blocks_per_agent(self, spec: ModelCacheSpec) -> int:
    """Maximum blocks needed per SWA layer per agent."""
    if spec.sliding_window_size is None:
        return 0  # No SWA layers
    # Number of blocks to cover the sliding window
    return math.ceil(spec.sliding_window_size / self.block_tokens)
    # Gemma 3: ceil(512/256) = 2 blocks
    # GPT-OSS: ceil(128/256) = 1 block (only 128 tokens needed!)
```

---

## 5. Variable-Length Batching

### 5.1 The Padding Problem

MLX-LM's `BatchKVCache` uses left-padding:
```
Agent A: 500 tokens  → padded to 4000 (3500 wasted)
Agent B: 4000 tokens → no padding
Agent C: 1200 tokens → padded to 4000 (2800 wasted)
Total waste: 6300 tokens × layers × memory per token
```

### 5.2 Our Solution: Block-List with Masked Attention

Each sequence maintains its own block list (no padding needed):
```
Agent A: [block_0, block_1]           (500 tokens in 2 blocks)
Agent B: [block_2, ..., block_17]     (4000 tokens in 16 blocks)
Agent C: [block_18, ..., block_22]    (1200 tokens in 5 blocks)
```

For the attention computation, we construct a per-sequence view:
```python
def gather_kv_for_sequence(agent_blocks: AgentBlocks, layer_idx: int):
    """Gather KV pairs from blocks into contiguous sequence for attention."""
    blocks = [self.pool.get_block(bid) for bid in agent_blocks.blocks_per_layer[layer_idx]]
    keys = mx.concatenate([b.keys[:, :b.used_tokens, :] for b in blocks], axis=1)
    values = mx.concatenate([b.values[:, :b.used_tokens, :] for b in blocks], axis=1)
    return keys, values
```

### 5.3 Attention Mask Construction

For batched generation with different context lengths, use MLX's mask support:

```python
def create_batch_mask(sequence_lengths: List[int], query_length: int = 1):
    """
    Create attention mask for variable-length batch.
    Works identically regardless of model architecture.
    """
    max_kv = max(sequence_lengths)
    batch_size = len(sequence_lengths)
    mask = mx.zeros((batch_size, 1, query_length, max_kv), dtype=mx.bool_)
    for i, seq_len in enumerate(sequence_lengths):
        mask[i, 0, :, :seq_len] = True
    return mask
```

### 5.4 Hybrid Approach: Padded Blocks, Not Padded Sequences

Key insight: we pad within blocks (to `block_tokens` boundary) but NOT between sequences. The waste is bounded:
- Maximum waste per sequence: `block_tokens - 1` tokens (one partial block)
- For 5 agents: max 1275 tokens wasted (vs. potentially thousands with full left-padding)

---

## 6. Multi-Model Serving Strategy

### 6.1 Single-Model-At-A-Time (24GB Constraint)

On M4 Pro 24GB, only one model fits comfortably with KV cache budget:

| Model | Weights | Remaining for KV | Practical Cache Budget |
|-------|---------|-------------------|----------------------|
| Gemma 3 12B 4-bit | ~6.5 GB | ~17.5 GB | ~12 GB (after OS/overhead) |
| GPT-OSS-20B 4-bit | ~11 GB | ~13 GB | ~9 GB |
| Qwen 2.5-14B 4-bit | ~9 GB | ~15 GB | ~11 GB |
| Llama 3.1 8B 4-bit | ~5 GB | ~19 GB | ~14 GB |

### 6.2 Model Hot-Swap Protocol

```python
class ModelRegistry:
    """Manages model lifecycle with TTL-based unloading."""

    async def swap_model(self, new_model_id: str):
        """
        Hot-swap to a different model.
        1. Evict all HOT caches to disk (tagged with current model_id)
        2. Clear BlockPool (all blocks freed)
        3. Unload current model weights (mx.metal.clear_cache())
        4. Load new model weights
        5. Extract new ModelCacheSpec
        6. Reconfigure BlockPool with new spec
        7. Ready for new requests
        """
        if self.current_model_id == new_model_id:
            return  # Already loaded

        # Evict all active caches to disk
        for agent_id in self.cache_store.hot_agents():
            await self.cache_store.evict_to_disk(agent_id)

        # Clear GPU memory
        self.block_pool.clear()
        del self.model
        mx.metal.clear_cache()

        # Load new model
        self.model, self.tokenizer = load(new_model_id)
        self.spec = ModelCacheSpec.from_model(self.model, new_model_id)
        self.block_pool.reconfigure(self.spec)
        self.current_model_id = new_model_id
```

### 6.3 Cache Compatibility Across Models

Caches are **model-specific** and cannot be reused across architectures:
- Different tokenizers produce different token IDs for same text
- Different layer counts, head dims, attention patterns
- Disk caches tagged with `model_id` in metadata

When a request arrives for a model that isn't currently loaded:
1. Check if request's `model` field matches current model
2. If not, trigger hot-swap (all current agents evicted to disk)
3. Load requested model
4. Check disk for any cached agents for this model
5. Proceed with generation

### 6.4 Model Selection Guidance

| Use Case | Recommended Model | Reasoning |
|----------|-------------------|-----------|
| Max concurrent agents (10+) | GPT-OSS-20B | Smallest KV footprint (48 KB/token) |
| Best quality per token | Qwen 2.5-14B | Dense model, strong benchmarks |
| Hybrid attention (efficiency) | Gemma 3 12B | SWA layers reduce memory for long context |
| Fastest decode (TPS) | GPT-OSS-20B | Only 3.6B active params (MoE) |
| Longest context per agent | GPT-OSS-20B | 128K context, 48 KB/token = 6GB for 128K |

---

## 7. Persistence and Reload

### 7.1 Save Format (Model-Tagged)

Per-agent cache saved as safetensors with model metadata:
```
~/.agent_caches/{model_id}/{agent_id}/
  ├── metadata.json          # model_id, agent_id, total_tokens, quantization
  ├── layer_00_keys.safetensors
  ├── layer_00_values.safetensors
  ├── ...
  ├── layer_47_keys.safetensors
  └── layer_47_values.safetensors
```

**metadata.json**:
```json
{
  "model_id": "mlx-community/gemma-3-12b-it-4bit",
  "agent_id": "coding-agent-001",
  "total_tokens": 4096,
  "num_layers": 48,
  "num_kv_heads": 4,
  "head_dim": 256,
  "layer_types": ["swa", "swa", "swa", "swa", "swa", "full", ...],
  "quantization": {"kv_bits": 8, "group_size": 64},
  "created_at": "2026-01-24T10:00:00Z",
  "last_accessed": "2026-01-24T12:30:00Z"
}
```

### 7.2 Load Strategy

When loading a cache from disk:
1. Verify `model_id` matches currently loaded model
2. Check total tokens needed
3. Check available blocks in pool
4. If insufficient blocks:
   a. Evict LRU agents to free blocks
   b. If still insufficient: truncate cache (drop oldest tokens, keep most recent)
   c. If truncation needed: reduce effective context, log warning
5. Allocate blocks and copy data from safetensors
6. **SWA layers**: only load last `sliding_window_size` tokens (rest are irrelevant)

### 7.3 Quantized Persistence

For 50% disk + memory savings:
- Quantize KV pairs to 8-bit before saving
- Store quantization metadata (scales, biases) alongside
- On load: keep quantized in memory (use `QuantizedKVCache` compatible format)
- Dequantize on-the-fly during attention computation

---

## 8. Architecture-Specific Considerations

### 8.1 Gemma 3 12B (Hybrid: 8 Global + 40 Sliding)

- **Global layers (8)**: Standard KVCache → full context, blocks grow with conversation
- **Sliding layers (40)**: RotatingKVCache(max_size=512) → max 2 blocks per layer per agent
- **Fixed SWA cost per agent**: 40 layers × 2 blocks × 1MB = ~80 MB
- **Variable global cost**: 8 layers × N blocks × 1MB per 256 tokens
- **Persistence**: Sliding layers only persist last 512 tokens; global layers persist full history

### 8.2 GPT-OSS-20B (MoE, Alternating 1:1)

- **Alternating pattern**: Full attention on even layers, SWA(128) on odd layers
- **MoE specifics**: 32 experts, 4 active per token → fast decode but large weights
- **SWA window = 128**: Smaller than block_tokens (256), so each SWA layer needs only 1 block
- **Fixed SWA cost per agent**: 12 layers × 1 block × 0.5MB = ~6 MB (tiny!)
- **Variable global cost**: 12 layers × N blocks × 0.5MB per 256 tokens
- **YaRN RoPE scaling**: Positions up to 128K supported (factor=32)
- **Decode advantage**: Only 3.6B active params → higher tokens/sec than dense models

### 8.3 Qwen 2.5-14B (Uniform Full Attention)

- **Simplest cache**: All 48 layers use standard KVCache, no sliding window
- **All blocks grow equally**: Every layer needs the same number of blocks
- **Higher memory per token**: 192 KB/token (all layers contribute)
- **No SWA optimization**: Cannot cap any layers, all grow with context
- **RoPE theta=1M**: Native 128K context window support
- **QKV bias**: Unlike Llama, has bias terms in Q/K/V projections

### 8.4 Llama 3.1 8B (Uniform Full Attention, Smaller)

- **Smallest model**: Only 5GB weights, most room for KV cache
- **32 layers, all full**: Similar to Qwen but fewer layers
- **128 KB/token**: Moderate memory footprint
- **Best for max context**: 14GB cache budget = ~112K tokens

---

## 9. Memory Budget Calculator (Per Model)

### 9.1 Gemma 3 12B 4-bit on M4 Pro 24GB

```
Model weights (4-bit): ~6.5 GB
MLX overhead:          ~1.0 GB
OS + Apps:             ~4.0 GB
─────────────────────────────
Available for KV:      ~12.5 GB

Per agent at 4K context:
  Global (8 layers):  8 × 16 blocks × 1MB/block-layer = ~128 MB
  Sliding (40 layers): 40 × 2 blocks × 1MB = ~80 MB (fixed)
  Total: ~208 MB per agent at 4K context

Max agents in 12.5 GB:
  12,500 MB / 208 MB ≈ 60 agents (theoretical)
  Practical with headroom: 10-15 agents

With 8-bit quantization: 2× capacity → 20-30 agents
```

### 9.2 GPT-OSS-20B 4-bit on M4 Pro 24GB

```
Model weights (4-bit): ~11.0 GB
MLX overhead:          ~1.0 GB
OS + Apps:             ~4.0 GB
─────────────────────────────
Available for KV:      ~8.0 GB

Per agent at 4K context:
  Global (12 layers):  12 × 16 blocks × 0.5MB = ~96 MB
  Sliding (12 layers): 12 × 1 block × 0.5MB = ~6 MB (fixed, tiny!)
  Total: ~102 MB per agent at 4K context

Max agents in 8.0 GB:
  8,000 MB / 102 MB ≈ 78 agents (theoretical)
  Practical with headroom: 15-25 agents

Note: GPT-OSS's tiny KV footprint makes it IDEAL for many concurrent agents.
With 8-bit quantization: 30-50 agents feasible!
```

### 9.3 Qwen 2.5-14B 4-bit on M4 Pro 24GB

```
Model weights (4-bit): ~9.0 GB
MLX overhead:          ~1.0 GB
OS + Apps:             ~4.0 GB
─────────────────────────────
Available for KV:      ~10.0 GB

Per agent at 4K context:
  All layers (48):  48 × 16 blocks × 1MB = ~768 MB
  No sliding optimization!
  Total: ~768 MB per agent at 4K context

Max agents in 10.0 GB:
  10,000 MB / 768 MB ≈ 13 agents (theoretical)
  Practical with headroom: 3-5 agents

Note: Qwen 2.5's uniform full attention is memory-hungry.
With 8-bit quantization: 6-10 agents.
```

### 9.4 Summary Comparison

| Model | Budget | Per Agent (4K) | Max Agents | With 8-bit |
|-------|--------|---------------|------------|------------|
| Gemma 3 12B | 12.5 GB | ~208 MB | 10-15 | 20-30 |
| GPT-OSS-20B | 8.0 GB | ~102 MB | 15-25 | 30-50 |
| Qwen 2.5-14B | 10.0 GB | ~768 MB | 3-5 | 6-10 |
| Llama 3.1 8B | 14.0 GB | ~512 MB | 5-8 | 10-16 |

---

## 10. Integration with Existing System

### 10.1 What Changes

| Current Component | Status | Notes |
|-------------------|--------|-------|
| `PersistentAgentManager` | Modified | Uses BlockPool, model-aware cache management |
| `ConcurrentAgentManager` | Modified | Budget-aware batch sizing, model-agnostic |
| `BatchedGenerationEngine` | Replaced | New `BlockPoolBatchEngine` with ModelCacheSpec |
| `MLXCacheExtractor` | Removed | Block management handles cache lifecycle |
| `CachePersistence` | Enhanced | Block-aware save/load, model-tagged |
| `APIServer` | Enhanced | Multi-protocol (Anthropic + OpenAI-compat + direct) |
| `MLXModelLoader` | Enhanced | `ModelRegistry` with hot-swap and TTL |

### 10.2 What Stays

- Per-agent asyncio.Lock for sequential per-agent semantics
- Cross-agent parallel batching via batch worker
- 10ms batching window for request collection
- Agent creation/eviction with LRU policy
- Safetensors format for disk persistence

### 10.3 New Components

| Component | Purpose |
|-----------|---------|
| `ModelCacheSpec` | Architecture-agnostic cache specification |
| `ModelRegistry` | Model lifecycle (load, unload, hot-swap, TTL) |
| `MultiProtocolAdapter` | Routes Anthropic/OpenAI/direct requests to engine |
| `AgentIdentifier` | Explicit ID (non-Anthropic) or content-hash (Anthropic) |

---

## 11. Novelty Analysis

### 11.0 Prior Art Acknowledgment

Multi-agent KV cache management is actively researched (2025-2026):
- **KVCOMM** (NeurIPS 2025): Cross-context KV-cache reuse, 70%+ reuse rate
- **KVFlow** (July 2025): Workflow-aware eviction with Agent Step Graph, 2.19x speedup
- **Continuum** (Nov 2025): TTL-based cache retention during tool calls
- **EvicPress** (Dec 2025): Joint compression + eviction, shows LRU alone insufficient
- **vllm-mlx** (2025): vLLM-style paged cache on MLX (but no persistence)
- **vLLM Hybrid KV Cache Manager** (2025): Layer-group-aware allocation for mixed attention

**Our unique angle**: Edge-optimized block-pool with persistence on Apple Silicon unified memory, model-agnostic architecture support, and multi-protocol agent identification.

### 11.1 What's Novel (Not in Any Existing System)

1. **Model-agnostic block-pool on unified memory**: One pool serves Gemma 3 (hybrid), GPT-OSS (MoE, alternating), Qwen (uniform), and Llama via ModelCacheSpec abstraction. No existing edge system handles multiple architectures with different attention patterns through a single pool interface.

2. **Block-pool persistent batching on unified memory**: vLLM discards KV on completion; KVCOMM/KVFlow share within a session but don't persist across restarts. We persist per-agent blocks to disk and reload into batches with zero-copy semantics on unified memory.

3. **Dynamic batch sizing from cache load with architecture awareness**: When an agent's historical cache is loaded, batch capacity adjusts automatically based on the *current model's* KV footprint. GPT-OSS allows 25+ agents; Qwen caps at 5.

4. **Unified block pool for heterogeneous cache types**: Same pool serves both KVCache (global layers) and RotatingKVCache (sliding window), with different allocation strategies per layer type. Extends to any attention pattern.

5. **Multi-protocol agent identification**: Content-based hashing for stateless APIs (Anthropic) AND explicit session_id for stateful APIs (OpenAI-compatible, direct) through a unified agent store.

6. **Model hot-swap with cache preservation**: Evict to disk on model change, reload when model returns. No existing system preserves per-agent caches across model switches.

### 11.2 What's Adapted (Simplified from Existing Work)

- Block concept from vLLM PagedAttention → simplified for single-GPU, no page table
- Layer-group allocation from vLLM Hybrid KV Cache Manager → adapted for edge
- Trie prefix matching from mlx-lm server.py CacheManager → enhanced with persistence
- DynamicKVCache growth → replaced with fixed-block pool (simpler, less fragmentation)
- BatchKVCache left-padding → replaced with per-sequence block lists + masks

### 11.3 What's Reused (From mlx-lm)

- `create_attention_mask()` for causal mask generation
- `save_prompt_cache` / `load_prompt_cache` format compatibility
- `KVCache.update_and_fetch()` interface for block-internal updates
- Model's native `__call__` with `[B, L]` input shape support
- `QuantizedKVCache` quantization format
- `model.make_cache()` introspection for layer type discovery

---

## 12. Performance Expectations

### 12.1 Throughput (M4 Pro, Per Model)

| Model | Scenario | Sequential | Block-Pool Batched | Improvement |
|-------|----------|-----------|-------------------|-------------|
| Gemma 3 12B | 5 agents × 50 tokens | ~8s | ~2-3s | 2.5-4x |
| GPT-OSS-20B | 5 agents × 50 tokens | ~5s | ~1.5-2s | 2.5-3x |
| Qwen 2.5-14B | 3 agents × 50 tokens | ~6s | ~3-4s | 1.5-2x |

Note: GPT-OSS-20B decode is faster (3.6B active params) but fewer available context tokens per agent.

### 12.2 Cache Resume Speed

| Operation | Time | Notes |
|-----------|------|-------|
| Block allocation (256 tokens) | <1ms | Pool free-list pop |
| Cache load from disk (2K tokens) | ~50ms | Safetensors read + block copy |
| Cache load from disk (8K tokens) | ~200ms | Larger read |
| Prefix match (trie lookup) | <1ms | In-memory trie traversal |
| Full re-computation (2K tokens) | ~2-3s | Without cache |
| Model hot-swap | ~15-30s | Unload + load weights |

### 12.3 Memory Overhead vs. Current System

| Metric | Current (BatchKVCache) | Block-Pool | Difference |
|--------|----------------------|------------|------------|
| Padding waste (5 agents, varying context) | ~30-60% | <5% | Major improvement |
| Allocation overhead | Per-step concatenation | Free-list O(1) | Faster |
| Persistence support | None in batch mode | Built-in | New capability |
| Max agents (Gemma 3, 24GB, 4K context) | ~5 (padded) | ~10-15 (no padding) | 2-3x more |
| Max agents (GPT-OSS, 24GB, 4K context) | N/A | ~15-25 | New capability |
| Multi-model support | None | Hot-swap | New capability |

---

## 13. Risk Analysis

### 13.1 Technical Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Block gather/scatter slower than contiguous | Medium | High | Benchmark vs. padded; can fall back to padding |
| MLX lazy eval + block pool = memory spikes | Low | Medium | Force eval after block allocation |
| Quantized attention accuracy loss | Low | Low | Only quantize for storage, dequant for compute |
| Sliding window block reuse correctness | Medium | High | Extensive testing of rotation semantics |
| Model hot-swap latency too high | Medium | Medium | Pre-download models, async loading |
| ModelCacheSpec extraction fails for new models | Low | Medium | Fallback: all-full-attention default |

### 13.2 Design Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Over-engineering for single user | Medium | Medium | Start with simpler BlockPool, add features incrementally |
| Incompatibility with future mlx-lm updates | Medium | Medium | Minimal monkey-patching; wrap don't modify |
| Block size 256 too large for GPT-OSS (sw=128) | Low | Low | SWA layers only use partial blocks |
| Multi-model serving rarely used | Low | Low | Single-model path is simpler default |

---

## 14. Implementation Phases

### Phase 1: ModelCacheSpec + BlockPool + Single-Agent Engine
- Implement `ModelCacheSpec.from_model()` for Gemma 3, GPT-OSS, Qwen
- Implement `BlockPool` with model-parameterized allocate/free/budget
- Implement `BlockPoolEngine` for single-sequence generation using blocks
- Validate: same output as current system for single agent with each model
- Test: block allocation/deallocation, persistence to disk with model tags

### Phase 2: Multi-Agent Batching (Architecture-Agnostic)
- Implement variable-length batch attention with masks (works for all models)
- Implement gather/scatter for block-list → contiguous KV
- Integrate with `ConcurrentAgentManager`
- Test: 3-5 agents generating simultaneously on each supported model

### Phase 3: Dynamic Batch Sizing + Cache Reload
- Implement budget-aware batch size calculation using ModelCacheSpec
- Implement disk load → block allocation → batch integration
- Model-tagged cache validation on load
- Test: agent with 8K cached context joins batch, others wait

### Phase 4: Model Registry + Hot-Swap
- Implement `ModelRegistry` with TTL-based unloading
- Implement model hot-swap protocol (evict → unload → load → reconfigure)
- Cache invalidation on model change
- Test: swap between Gemma and GPT-OSS, verify caches preserved on disk

### Phase 5: Trie-Based Prefix Matching + Multi-Protocol
- Port mlx-lm CacheManager trie to work with block references
- Implement shared-prefix optimization (reference counting)
- Support both content-based and explicit agent_id identification
- Test: two agents with same system prompt share prefix blocks

### Phase 6: Quantization + Optimization
- 8-bit quantization for block storage and persistence
- `mx.compile` optimization for block gather kernels
- Benchmark and tune block_tokens parameter per model
- Per-model optimized decode paths

---

## 15. Expert Debate: Key Decisions

### Decision 1: Block Size = 256 Tokens (Universal)

**For** (System Engineer): Same block_tokens across all models simplifies pool management. Matches MLX-LM's `KVCache.step`. Block *memory* varies by model but token count is constant.

**Against** (ML Engineer): GPT-OSS-20B has sw=128, meaning SWA layers waste half a block. Qwen has no SWA at all.

**Resolution**: Accept 256 as universal. SWA layers with window < 256 only fill partial blocks (bounded waste). The alternative (per-model block sizes) would require separate pools and complicate cross-model code.

### Decision 2: No True Paged Attention (No Block Table)

**Same as before**: Apple Silicon unified memory eliminates the CPU-GPU address translation that page tables solve. Block IDs are just indices into a flat array.

### Decision 3: One Model At A Time (No Co-Loading)

**For** (Memory Expert): 24GB isn't enough for two large models. Gemma 3 (6.5GB) + Qwen (9GB) = 15.5GB weights alone, leaving <8GB for KV caches of both.

**Against** (System Engineer): Could load two small models (Llama 8B + Gemma 3 = 11.5GB weights, 12GB for caches).

**Resolution**: Default to one model. Power users on M4 Max (64GB+) could co-load. For 24GB, hot-swap is the right pattern: unload one, load another, caches preserved on disk.

### Decision 4: ModelCacheSpec Extracted Dynamically

**For** (ML Engineer): Reading from model.args at load time is robust. New models auto-supported if they follow mlx-lm patterns.

**Against** (Attention Expert): Some models have non-standard patterns (GPT-OSS alternating, DeepSeek's MoE attention). Need fallback logic.

**Resolution**: `from_model()` inspects model.args with fallback: if no sliding_window_pattern detected, assume all-full-attention (safest default). Custom overrides available via config file for unusual architectures.

### Decision 5: Caches Are Model-Specific (No Cross-Model Reuse)

**For** (LLM Engineer): Different tokenizers, different architectures = incompatible KV states. A Gemma 3 cache is meaningless to Qwen.

**Against** (System Engineer): Could theoretically share system prompt *text* hash across models (same agent concept, different caches).

**Resolution**: Agent identity is model-independent (same `agent_id` across models). But KV cache on disk is tagged with `model_id`. When loading an agent, only load cache if `model_id` matches current model. Otherwise, agent exists but starts fresh on new model.

### Decision 6: Explicit Agent IDs for Non-Anthropic APIs

**For** (System Engineer): Much simpler than content-based hashing. Non-Anthropic clients (OpenAI-compatible, custom) can pass `session_id` or `agent_id` directly. Better for later systems running locally.

**Against** (Anthropic Expert): Creates two code paths for agent identification.

**Resolution**: Two identification strategies behind one `AgentIdentifier` interface:
1. **Content-based** (Anthropic): Token prefix hash → agent lookup (for Claude Code compatibility)
2. **Explicit** (non-Anthropic): `session_id` field in request → direct agent lookup (preferred for custom clients)

The explicit path is simpler, faster, and recommended for all non-Anthropic integrations.

---

## 16. Security Considerations

**Single User, Multiple Agents**:
- All agents belong to the same user → no cross-user isolation needed
- Agent isolation is for correctness (separate conversations), not security
- Block pool has no access control (any agent can theoretically access any block via bug)
- Disk persistence in user's home directory (~/.agent_caches) → standard file permissions
- No network access control beyond what the API server provides
- Model hot-swap: no security implications (same user, same machine)

---

## 17. Future Considerations (M4 Max / 64GB+)

On larger memory configurations:
- **Co-loading**: Two models simultaneously (e.g., fast model + quality model)
- **Shared pool across models**: Raw byte pool, each model carves out blocks of its required size
- **Speculative decoding**: Small model drafts, large model verifies (both loaded)
- **Adapter-based multi-model**: One base model + multiple LoRA adapters (shared KV cache prefix)

---

## References

- [MLX-LM Cache Source](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/cache.py)
- [MLX-LM BatchGenerator](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/generate.py)
- [vLLM Hybrid KV Cache Manager](https://docs.vllm.ai/en/latest/design/hybrid_kv_cache_manager/)
- [vLLM PagedAttention](https://docs.vllm.ai/en/stable/design/paged_attention/)
- [GPT-OSS-20B Config](https://huggingface.co/openai/gpt-oss-20b/blob/main/config.json)
- [Qwen 2.5-14B](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct)
- [mlx-community/gpt-oss-20b-MXFP4-Q4](https://huggingface.co/mlx-community/gpt-oss-20b-MXFP4-Q4)
- [mlx-community/Qwen2.5-14B-Instruct-4bit](https://huggingface.co/mlx-community/Qwen2.5-14B-Instruct-4bit)
- [KVCOMM (NeurIPS 2025)](https://arxiv.org/abs/2510.12872)
- [KVFlow](https://arxiv.org/abs/2507.07400)
- [Continuum](https://arxiv.org/abs/2511.02230)
- [LMDeploy Interactive API](https://lmdeploy.readthedocs.io/en/v0.5.0/serving/api_server.html)
- [llama.cpp Slot Persistence](https://github.com/ggml-org/llama.cpp/discussions/13606)
