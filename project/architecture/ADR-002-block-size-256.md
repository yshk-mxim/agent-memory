# ADR-002: Block Size = 256 Tokens (Universal)

**Date**: 2026-01-24
**Status**: ✅ ACCEPTED (Sprint 2, Day 1)
**Author**: ML (ML Engineer)
**Deciders**: ML, SE, HW, PM

---

## Context

The block-pool memory management system requires choosing a fixed block size for KV cache allocation. This decision affects:

1. **Memory efficiency** — Smaller blocks reduce waste, larger blocks reduce overhead
2. **MLX compatibility** — Must align with MLX's internal cache step size
3. **Model compatibility** — Must work across diverse architectures (Gemma, Llama, Qwen, MoE)
4. **Performance** — Block gather overhead scales with number of blocks

**Models to support**:
- **Gemma 3 12B**: 48 layers (8 global + 40 sliding window @ 1024 tokens)
- **Llama 3.1 8B**: 32 layers (all global, no sliding window)
- **Qwen 2.5 7B**: 48 layers (all global)
- **Qwen1.5-MoE-A2.7B**: 24 layers (MoE with alternating layers)

**Problem**: What block size maximizes memory efficiency while maintaining MLX compatibility?

---

## Decision

Adopt **256 tokens per block** as the UNIVERSAL block size across all models and layers.

```python
@dataclass(frozen=True)
class ModelCacheSpec:
    block_tokens: int = 256  # UNIVERSAL constant
```

**No exceptions**: Every model architecture uses 256-token blocks, regardless of sliding window size or layer type.

---

## Rationale

### 1. MLX KVCache.step Alignment

MLX's `KVCache.step` parameter defaults to 256 tokens:

```python
# mlx_lm/models/cache.py (mlx-lm v0.30.4)
class KVCache:
    def __init__(self, head_dim, n_kv_heads, step=256):
        self.step = step  # Default: 256
```

**Why this matters**:
- MLX allocates cache in 256-token chunks internally
- Using the same size eliminates alignment issues
- Simplifies block-to-cache reconstruction (no padding needed)

**Validation**: Confirmed in mlx_lm v0.30.4 source code (2026-01-24).

---

### 2. Memory Efficiency Analysis

**Trade-off**: Smaller blocks → less waste, but more allocation overhead.

#### Waste Analysis (Gemma 3 12B)

Sliding window layers have **1024-token limit** (hard cap in architecture):

| Block Size | Blocks per Window | Waste | Overhead |
|-----------|-------------------|-------|----------|
| 128 tokens | 8 blocks | 0% (1024 / 128 = 8 exact) | 8× allocations |
| **256 tokens** | **4 blocks** | **0%** (1024 / 256 = 4 exact) | **4× allocations** |
| 512 tokens | 2 blocks | 0% (1024 / 512 = 2 exact) | 2× allocations |
| 1024 tokens | 1 block | 0% (1024 / 1024 = 1 exact) | 1× allocation |

**Observation**: For Gemma 3's 1024-token sliding window, **all power-of-2 sizes have zero waste**.

**Winner**: 256 tokens balances granularity (4 blocks for flexibility) with low overhead.

---

### 3. Universal Applicability

**Challenge**: Different models have different sliding window sizes.

| Model | Sliding Window | Blocks @ 256 | Waste |
|-------|---------------|--------------|-------|
| Gemma 3 | 1024 tokens | 4 blocks | 0% ✅ |
| GPT-OSS-20B | 128 tokens | 0.5 blocks → 1 block | **128 tokens (50%)** ⚠️ |
| Llama 3.1 | N/A (all global) | Unlimited | 0% ✅ |
| Qwen 2.5 | N/A (all global) | Unlimited | 0% ✅ |

**GPT-OSS-20B concern**: 128-token sliding window wastes 50% (128 unused tokens per block).

**Decision**: **Accept the waste** for GPT-OSS-20B because:
1. MoE models are rare in mlx-community (only 1 of 100+ models)
2. Waste is only 128 tokens × 40 layers = 5K tokens = ~600KB per agent (negligible)
3. Benefit of universal block size outweighs minor waste

**Priority**: Optimize for common case (Gemma 3, Llama, Qwen), tolerate edge case (GPT-OSS).

---

### 4. Performance Implications

**Block gather overhead** (mx.concatenate) scales with number of blocks:

| Context Size | Blocks @ 128 | Blocks @ 256 | Blocks @ 512 |
|--------------|--------------|--------------|--------------|
| 2K tokens | 16 | 8 | 4 |
| 8K tokens | 64 | 32 | 16 |
| 32K tokens | 256 | 128 | 64 |

**Hypothesis**: 32 blocks × 48 layers = 1536 concatenations for 8K context.

**EXP-006 (Sprint 2, Day 7-8)** will validate: Target < 5ms for 8K gather.

**If gather > 5ms**: Document in ADR-004 (one-time gather at restore, not per-step).

---

### 5. Memory Layout Visualization

**Per-block memory** (Gemma 3, float16):
```
n_kv_heads (8) × head_dim (240) × 2 (K+V) × 2 bytes × block_tokens (256)
= 8 × 240 × 2 × 2 × 256
= 1,966,080 bytes
≈ 1.875 MB per block per layer
```

**For 8K context**:
- Global layers (8): 32 blocks × 1.875 MB = 60 MB
- Sliding window layers (40): 4 blocks × 1.875 MB = 7.5 MB
- **Total**: 60 MB + (40 × 7.5 MB) = **360 MB per agent**

**With 8-bit quantization**:
- Total: **180 MB per agent** (halved)

**Pool sizing** (M4 Pro 24GB):
```
Model weights (4-bit):     ~6.0 GB
OS + system:               ~3.0 GB
MLX framework:             ~1.0 GB
Available for caches:      ~14.0 GB
────────────────────────────────────
4 GB cache pool budget:    ~22 agents @ 8K (8-bit quantized)
```

---

## Alternatives Considered

### Alternative 1: Variable Block Size per Model
**Approach**: Gemma 3 uses 256, GPT-OSS uses 128, etc.

**Pros**:
- Zero waste for all models
- Optimal memory efficiency

**Cons**:
- Complex implementation (block size in ModelCacheSpec)
- Harder to reason about pool capacity
- Breaks universal pool abstraction
- **Violates YAGNI** (support for 1 edge-case model)

**Decision**: ❌ REJECTED — Complexity not justified for rare edge case.

---

### Alternative 2: Dynamic Block Size (Configurable)
**Approach**: Make `block_tokens` a runtime parameter.

**Pros**:
- Ultimate flexibility
- User can optimize per deployment

**Cons**:
- Configuration complexity (one more knob)
- Cache files not portable (256-block cache can't load into 128-block pool)
- Testing matrix explodes (test 128, 256, 512, 1024)
- **No clear benefit** (256 works for 99% of models)

**Decision**: ❌ REJECTED — YAGNI. Fixed constant is simpler.

---

### Alternative 3: Larger Block Size (512 or 1024)
**Approach**: Reduce allocation overhead with bigger blocks.

**Pros**:
- Fewer allocations (e.g., 16 blocks vs 32 for 8K context)
- Faster block gather (fewer mx.concatenate calls)

**Cons**:
- Higher waste for short contexts (512-token block for 100-token cache)
- Coarser LRU granularity (evict 512 tokens at once, not 256)
- **Misaligned with MLX step=256** (would need padding)

**Decision**: ❌ REJECTED — Misalignment with MLX is deal-breaker.

---

## Consequences

### Positive

✅ **MLX compatibility**: Aligns perfectly with KVCache.step=256 (no padding needed)
✅ **Zero waste for Gemma 3**: 1024 / 256 = 4 blocks (exact)
✅ **Zero waste for Llama/Qwen**: Global layers have no cap (unlimited blocks)
✅ **Simple implementation**: Single constant, no per-model logic
✅ **Portable caches**: All caches use 256-block format (interoperable)
✅ **Predictable pool sizing**: Fixed block size → easy capacity planning

### Negative

⚠️ **50% waste for GPT-OSS-20B**: 128-token window → 128 unused tokens per block
- **Mitigation**: GPT-OSS is rare (1 of 100+ mlx models); waste is only ~600KB per agent

⚠️ **Coarser granularity than optimal for tiny contexts**: 100-token cache uses full 256-token block (156 tokens unused)
- **Mitigation**: Tiny contexts are rare in production (most prompts > 256 tokens)

### Neutral

🔵 **Block gather overhead**: 32 blocks @ 8K → validate with EXP-006 (target < 5ms)
🔵 **LRU eviction granularity**: Evict 256 tokens at once (vs 128 or 512)

---

## Implementation

### ModelCacheSpec Definition

```python
@dataclass(frozen=True)
class ModelCacheSpec:
    """Model-specific cache geometry."""

    n_layers: int
    n_kv_heads: int
    head_dim: int
    block_tokens: int = 256  # UNIVERSAL constant (ADR-002)
    layer_types: list[str]
    sliding_window_size: int | None = None

    def bytes_per_block_per_layer(self) -> int:
        """Memory per block per layer (float16)."""
        return self.n_kv_heads * self.head_dim * 2 * 2 * self.block_tokens

    def max_blocks_for_layer(self, layer_type: str) -> int | None:
        """Max blocks for this layer type (None = unlimited)."""
        if layer_type == "global":
            return None  # Unlimited
        elif layer_type == "sliding_window":
            if self.sliding_window_size is None:
                raise ValueError("sliding_window layer requires sliding_window_size")
            # Ceiling division: e.g., 1024 / 256 = 4
            return (self.sliding_window_size + self.block_tokens - 1) // self.block_tokens
        else:
            raise ValueError(f"Unknown layer_type: {layer_type}")
```

### Block Allocation Example

```python
# Gemma 3 12B: Allocate for 8K context

spec = ModelCacheSpec(
    n_layers=48,
    n_kv_heads=8,
    head_dim=240,
    block_tokens=256,  # Universal constant
    layer_types=["global"] * 8 + ["sliding_window"] * 40,
    sliding_window_size=1024,
)

pool = BlockPool(spec, total_blocks=1000)

# Global layers: 32 blocks (8K / 256 = 32)
# Sliding window layers: 4 blocks (1024 / 256 = 4, hard cap)

for layer_id in range(spec.n_layers):
    layer_type = spec.layer_types[layer_id]
    if layer_type == "global":
        # Allocate 32 blocks for 8K context
        blocks = pool.allocate(n_blocks=32, layer_id=layer_id, agent_id="agent_1")
    elif layer_type == "sliding_window":
        # Allocate 4 blocks (capped at sliding_window_size)
        blocks = pool.allocate(n_blocks=4, layer_id=layer_id, agent_id="agent_1")

# Total allocated:
# - 8 global layers × 32 blocks = 256 blocks
# - 40 sliding window layers × 4 blocks = 160 blocks
# - Total: 416 blocks × 1.875 MB = 780 MB (float16)
```

---

## Validation

### Sprint 1 Validation (Actual)

- ✅ ModelCacheSpec uses `block_tokens = 256` (verified in code)
- ✅ max_blocks_for_layer() implements ceiling division (1024 / 256 = 4)
- ✅ BlockPool allocates with 256-token granularity (44 tests pass)
- ✅ No model-specific block size logic (YAGNI principle followed)

### Sprint 2 Validation (Planned)

- [ ] EXP-006: Measure block gather overhead for 8K context (32 blocks × 48 layers)
  - Target: < 5ms
  - If > 5ms: Document in ADR-004 (one-time gather strategy)

- [ ] Integration test: Load Gemma 3, generate 8K context, verify:
  - Global layers use 32 blocks
  - Sliding window layers use 4 blocks (capped)
  - No alignment errors with MLX

### Sprint 6 Validation (Planned)

- [ ] Benchmark: Block pooling vs dedicated memory (1, 3, 5 agents)
- [ ] Validate memory savings with real multi-agent workload
- [ ] Measure actual vs predicted pool capacity

---

## Edge Cases

### Case 1: Sliding Window Size < Block Size
**Example**: GPT-OSS-20B has 128-token sliding window.

**Behavior**:
```python
spec.sliding_window_size = 128
spec.block_tokens = 256

max_blocks = spec.max_blocks_for_layer("sliding_window")
# (128 + 256 - 1) // 256 = 383 // 256 = 1

# Result: Allocate 1 block, use 128 tokens, waste 128 tokens
```

**Waste**: 50% (acceptable for rare edge case).

---

### Case 2: Context < Block Size
**Example**: 100-token user prompt.

**Behavior**:
```python
tokens_needed = 100
blocks_needed = (100 + 255) // 256 = 355 // 256 = 1

# Result: Allocate 1 block, use 100 tokens, waste 156 tokens
```

**Waste**: 61% (acceptable for short prompts; rare in production).

**Note**: Initial prompts are usually > 256 tokens (system prompt + user message).

---

### Case 3: Context Exactly Divisible by 256
**Example**: 1024-token context.

**Behavior**:
```python
tokens_needed = 1024
blocks_needed = (1024 + 255) // 256 = 1279 // 256 = 4

# Result: Allocate 4 blocks, use 1024 tokens, waste 0 tokens
```

**Waste**: 0% (perfect alignment).

---

## Future Considerations

### If we add GPU-based inference (vLLM/CUDA):
- vLLM uses 16-token blocks internally (PagedAttention)
- **Option 1**: Map 256-token semantic blocks → 16× vLLM pages (clean abstraction)
- **Option 2**: Reconsider block size if vLLM becomes primary backend

**Decision**: Defer until vLLM integration is planned (Sprint 6+).

### If we add model streaming (continuous prefill):
- 256-token blocks work well for streaming (reasonable chunk size)
- No change needed

### If we add quantized caching (8-bit KV):
- Block size remains 256 tokens
- Memory per block halves (1.875 MB → 0.9375 MB)
- No architectural change needed

---

## Related Decisions

- **ADR-001**: Hexagonal Architecture (domain layer has ModelCacheSpec)
- **ADR-003**: Cache Eviction Strategy (eviction granularity is 256 tokens)
- **ADR-004**: Block Gather Strategy (gather overhead depends on blocks per context)
- **ADR-005**: Composition Pivot (engine manages block allocation at 256-token boundaries)

---

## References

- [MLX KVCache source](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/cache.py) — Confirms step=256 default
- [PagedAttention paper](https://arxiv.org/abs/2309.06180) — vLLM uses 16-token pages
- [production_plan.md](../plans/production_plan.md) — Memory budget calculations
- [backend_plan.md](../plans/backend_plan.md) — Block-pool design

---

**Decision**: ✅ ACCEPTED
**Implemented**: Sprint 1 (ModelCacheSpec), Sprint 2 (BlockPoolBatchEngine)
**Review Date**: Sprint 6 (after multi-agent benchmarks validate memory efficiency)
