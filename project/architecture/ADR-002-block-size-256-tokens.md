# ADR-002: Block Size = 256 Tokens (Universal)

**Date**: 2026-01-24
**Status**: Accepted
**Deciders**: ML, HW, SE

## Context

Block-pool memory management requires choosing a fixed block size (in tokens) that applies across all models and all cache layers. This block size determines:

1. **Memory granularity**: Smaller blocks = finer allocation, more overhead
2. **Pooling efficiency**: Block reuse depends on alignment with model attention patterns
3. **Sliding window compatibility**: Must divide evenly into `sliding_window` parameter
4. **Fragmentation**: Smaller blocks = less wasted space, more metadata

### Model Attention Patterns

| Model | Sliding Window | Global Layers | SW Layers |
|-------|----------------|---------------|-----------|
| Gemma 3 12B | 1024 | 8 | 40 |
| Llama 3.1 8B | — | 32 | 0 |
| Qwen 2.5 7B | — | 28 | 0 |
| GPT-OSS-20B | 2048 (some) | 20 (mixed) | 20 (mixed) |

**Constraint**: Sliding window sizes are multiples of 128-256 tokens in practice.

### Memory Impact

For Gemma 3 12B (n_kv_heads=8, head_dim=256, float16):

| Block Size | Blocks per 8K | Memory per Block per Layer | Metadata Overhead (5%) |
|------------|---------------|---------------------------|----------------------|
| 64 tokens  | 125 | 512 KB | 6.4 KB × 48 layers = 307 KB |
| 128 tokens | 62 | 1 MB | 3.1 KB × 48 layers = 149 KB |
| **256 tokens** | **32** | **2 MB** | **1.6 KB × 48 layers = 77 KB** |
| 512 tokens | 16 | 4 MB | 0.8 KB × 48 layers = 38 KB |

**Trade-off**: Smaller blocks → more fragmentation control, more metadata overhead.

## Decision

**Block size = 256 tokens** across all models and all cache layers.

### Rationale

1. **Sliding Window Alignment**
   - Gemma 3: 1024 ÷ 256 = 4 blocks (exact fit)
   - GPT-OSS-20B (if SW=2048): 2048 ÷ 256 = 8 blocks (exact fit)
   - No partial blocks, no wasted space

2. **Memory Efficiency**
   - 2 MB per block per layer (float16) is manageable
   - Metadata overhead: 77 KB total (0.009% of 832 MB full cache)
   - Pool can handle 5-10 agents with 4 GB budget

3. **Allocation Granularity**
   - Fine enough for incremental growth (every 256 tokens)
   - Coarse enough to minimize block management overhead

4. **Universal Constant**
   - Same block size for all models simplifies `BlockPool` logic
   - No model-specific block sizing needed
   - Easier to reason about memory budgets

5. **Proven by vLLM**
   - vLLM uses 16-token blocks for autoregressive decode
   - Our 256-token blocks are 16× larger (prefill-oriented)
   - Appropriate for multi-agent serving vs single-request latency

## Alternatives Considered

| Option | Pros | Cons | Decision |
|--------|------|------|----------|
| **64 tokens** | Fine granularity, minimal fragmentation | High metadata overhead (307 KB), more allocations | ❌ Rejected |
| **128 tokens** | Good granularity, lower overhead | Doesn't divide 1024 cleanly (8 blocks) | ❌ Rejected |
| **256 tokens** | SW alignment, low overhead, universal | Slightly coarser allocation | ✅ **Selected** |
| **512 tokens** | Very low overhead, fewer allocations | Poor granularity (only 2 blocks per SW) | ❌ Rejected |
| **Variable (per model)** | Optimal per model | Complex pool logic, hard to reason about | ❌ Rejected |

## Consequences

### Positive

- **Exact alignment** with sliding window sizes (no partial blocks)
- **Low metadata overhead**: < 0.01% of total cache memory
- **Simple pool logic**: One block size, universal allocation
- **Predictable memory**: 2 MB × n_layers × n_blocks per agent (float16)

### Negative

- **Granularity limit**: Cannot allocate < 256 tokens (but agents rarely have < 256 token context)
- **8-bit quantization impact**: Block size doesn't reduce (still 256 tokens), but memory per block halves to 1 MB

### Neutral

- **No per-model tuning**: Could optimize further by using 128 for Llama, 256 for Gemma, but complexity not worth it

## Implementation Notes

### BlockPool Interface

```python
class BlockPool:
    BLOCK_TOKENS: int = 256  # Universal constant

    def allocate(self, n_blocks: int) -> list[KVBlock]:
        """Allocate n_blocks of size BLOCK_TOKENS each."""

    def free(self, blocks: list[KVBlock]) -> None:
        """Return blocks to free pool."""
```

### ModelCacheSpec

```python
@dataclass(frozen=True)
class ModelCacheSpec:
    n_layers: int
    n_kv_heads: int
    head_dim: int
    block_tokens: int = 256  # Always 256, included for explicitness
```

### Sliding Window Block Capping

For Gemma 3 sliding window layers (1024 tokens):
```python
max_blocks_per_sw_layer = sliding_window // BLOCK_TOKENS  # 1024 // 256 = 4
```

Enforced in `BlockPool.allocate()` — SW layers never allocate > 4 blocks.

### Memory Calculation

```python
def memory_per_block_per_layer(spec: ModelCacheSpec, quantized: bool = False) -> int:
    """
    Returns bytes per block per layer.

    For Gemma 3 12B (n_kv_heads=8, head_dim=256, block_tokens=256):
    - Float16: 8 * 256 * 2 (K+V) * 2 bytes * 256 tokens = 2 MB
    - 8-bit quantized: 1 MB
    """
    bytes_per_element = 1 if quantized else 2
    return spec.n_kv_heads * spec.head_dim * 2 * bytes_per_element * spec.block_tokens
```

## Validation

### Sprint 1 Validation

- [ ] `BlockPool.BLOCK_TOKENS = 256` enforced in code
- [ ] Unit test: Allocate 4 blocks for SW layer, verify 1024 tokens exactly
- [ ] Unit test: Attempt to allocate 5th block for SW layer → raises `PoolExhaustedError`

### Sprint 2 Validation

- [ ] Integration test: Generate 1000-token context, verify 4 blocks allocated (not 3 or 5)
- [ ] Benchmark: Measure allocation overhead < 1ms (EXP-002)

### Sprint 3 Validation

- [ ] Save cache with 32 blocks (8K context), reload, verify exact block count
- [ ] Model swap: Llama 8B (no SW) → Gemma 3 (SW=1024), verify pool reconfigures correctly

## References

- vLLM PagedAttention: https://arxiv.org/abs/2309.06180 (uses 16-token blocks)
- Gemma 3 config: `sliding_window=1024`, `sliding_window_pattern=6`
- Related: ADR-001 (Hexagonal Architecture), ADR-003 (Eviction Strategy)
- Experiment: EXP-002 (Block allocation overhead < 1ms)
