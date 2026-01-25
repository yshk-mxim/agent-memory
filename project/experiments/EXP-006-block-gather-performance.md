# EXP-006: Block Gather Performance Benchmark

**Date**: TBD (Day 8)
**Sprint**: 2 (Block-Pool Batch Engine)
**Owner**: ML (Machine Learning Engineer), HW (Hardware/Memory Expert)
**Status**: ⏳ PENDING

---

## Objective

Measure cache reconstruction overhead when converting blocks → KVCache objects.

**Critical Requirement**: Gather time must be < 5ms (p95) for 8K context (32 blocks × 48 layers).

---

## Hypothesis

The one-time cache reconstruction cost (`_reconstruct_cache()`) will be negligible compared to generation latency:
- **Target**: p95 < 5ms for 8K context
- **Rationale**: One-time cost at cache restore, not per-step overhead
- **Alternative**: Per-step gather would be 10-100x more expensive

**Risk**: If p95 > 10ms, may need to refactor approach (pre-allocated buffer, lazy gather, or padded strategy).

---

## Method

### Test Setup

**Simulated Cache** (8K context on Gemma 3 12B):

```python
# Gemma 3 12B: 48 layers (8 global + 40 sliding window)
# 8K tokens = 32 blocks (8192 / 256 = 32)

spec = ModelCacheSpec.from_model(model)  # Gemma 3 12B
pool = BlockPool(spec, total_blocks=1000)

# Allocate blocks for 8K context
agent_blocks = AgentBlocks(agent_id="test_agent", total_tokens=8192)

for layer_id in range(spec.n_layers):
    layer_type = spec.layer_types[layer_id]

    if layer_type == "global":
        n_blocks = 32  # 8K / 256
    else:  # sliding_window
        n_blocks = 4   # 1024 / 256 (capped)

    blocks = pool.allocate(n_blocks, layer_id, "test_agent")
    for block in blocks:
        agent_blocks.add_block(block)
```

### Benchmark Harness

**Measurement**: Time the `_reconstruct_cache()` call with `time.perf_counter()`:

```python
import time
import numpy as np

def benchmark_block_gather(agent_blocks, n_runs=100):
    """Benchmark cache reconstruction from blocks."""
    times = []

    for run in range(n_runs):
        # Measure reconstruction time
        start = time.perf_counter()
        cache = engine._reconstruct_cache(agent_blocks)
        end = time.perf_counter()

        elapsed_ms = (end - start) * 1000  # Convert to milliseconds
        times.append(elapsed_ms)

    # Compute statistics
    return {
        "p50": np.percentile(times, 50),
        "p95": np.percentile(times, 95),
        "p99": np.percentile(times, 99),
        "mean": np.mean(times),
        "std": np.std(times),
        "min": min(times),
        "max": max(times),
        "samples": n_runs,
    }
```

---

## Validation Criteria

### Primary Criteria (MUST PASS)

1. **✅ p95 < 5ms** (Target)
   ```python
   stats = benchmark_block_gather(agent_blocks, n_runs=100)
   assert stats["p95"] < 5.0  # milliseconds
   ```
   **Rationale**: 5ms overhead is acceptable for one-time cache restoration.

2. **✅ Consistent Performance** (Low Variance)
   ```python
   # Standard deviation should be < 20% of mean
   assert stats["std"] < stats["mean"] * 0.2
   ```
   **Rationale**: High variance suggests unpredictable performance.

### Secondary Criteria (NICE TO HAVE)

3. **p99 < 10ms**
   ```python
   assert stats["p99"] < 10.0  # Even tail latency is acceptable
   ```

4. **Linear Scaling with Block Count**
   ```python
   # Test with 16 blocks (4K context) and 32 blocks (8K context)
   stats_4k = benchmark_block_gather(blocks_4k)
   stats_8k = benchmark_block_gather(blocks_8k)

   # Time should roughly double (2x blocks = 2x time)
   ratio = stats_8k["mean"] / stats_4k["mean"]
   assert 1.5 < ratio < 2.5  # Within reasonable range
   ```

---

## Expected Results

**Predicted Performance**:

| Metric | Predicted | Target | Status |
|--------|-----------|--------|--------|
| p50 (median) | ~2ms | < 5ms | ✅ PASS |
| p95 | ~4ms | < 5ms | ✅ PASS |
| p99 | ~6ms | < 10ms | ✅ PASS |
| Mean | ~2.5ms | - | - |
| Std Dev | ~0.5ms | < 20% of mean | ✅ PASS |

**Rationale for Predictions**:
- mx.concatenate is highly optimized in MLX
- 32 blocks × 48 layers = 1,536 concatenate operations
- Each concat: ~2-3 μs → Total: ~3-5ms
- Force mx.eval() ensures immediate execution (no lazy deferred cost)

---

## Success Criteria

**GO Criteria**:
- ✅ p95 < 5ms (MUST PASS)
- ✅ Standard deviation < 20% of mean (stability check)

**Conditional GO** (if p95 between 5-10ms):
- Document in ADR-004 as acceptable trade-off
- Rationale: One-time cost at restore, not per-step overhead
- Alternative (per-step gather) would be 10-100x more expensive

**NO-GO** (if p95 > 10ms):
- Investigate mx.concatenate performance
- Consider pre-allocated buffer strategy
- May need to refactor block-to-cache approach

---

## Failure Analysis

**If EXP-006 FAILS (p95 > 5ms)**:

### Investigation Steps

1. **Profile mx.concatenate**:
   ```python
   import time

   # Measure concatenation alone
   k_tensors = [block.layer_data["k"] for block in layer_blocks]
   start = time.perf_counter()
   k_full = mx.concatenate(k_tensors, axis=2)
   mx.eval(k_full)  # Force evaluation
   end = time.perf_counter()

   print(f"Concatenate time: {(end - start) * 1000:.2f}ms")
   ```

2. **Check Block Count Scaling**:
   - Test with 8, 16, 32, 64 blocks
   - Plot time vs block count
   - Confirm O(n) complexity (not O(n²))

3. **Validate MLX Lazy Evaluation**:
   ```python
   # Ensure mx.eval() is called
   # Otherwise, concatenation is deferred until cache is used
   ```

4. **Consider Alternative Strategies** (if > 10ms):
   - **Option A**: Pre-allocated buffer (copy blocks into contiguous memory)
   - **Option B**: Lazy gather (concatenate on first use, cache result)
   - **Option C**: Padded approach (avoid gather entirely)

---

## Multi-Context Testing

**Extended Test Cases**:

| Context Size | Blocks (Global) | Blocks (SWA) | Total Ops | Expected p95 |
|--------------|-----------------|--------------|-----------|--------------|
| 2K tokens | 8 blocks | 4 blocks | 384 ops | ~1ms |
| 4K tokens | 16 blocks | 4 blocks | 768 ops | ~2ms |
| 8K tokens | 32 blocks | 4 blocks | 1,536 ops | ~4ms |
| 16K tokens | 64 blocks | 4 blocks | 3,072 ops | ~8ms |

**Purpose**: Validate linear scaling assumption.

---

## Deliverables

1. **Results Document**: This file updated with actual measurements
2. **Timing Data**: `/project/experiments/data/exp_006_timings.json` (raw samples)
3. **Visualization**: `/project/experiments/data/exp_006_histogram.png` (optional)
4. **Report**: Summary for sprint review

---

## Visualization (Optional)

```python
import matplotlib.pyplot as plt

plt.hist(times, bins=30)
plt.axvline(stats["p95"], color='r', linestyle='--', label='p95 (target: 5ms)')
plt.xlabel('Gather Time (ms)')
plt.ylabel('Frequency')
plt.title('EXP-006: Block Gather Performance (100 runs)')
plt.legend()
plt.savefig('exp_006_histogram.png')
```

---

## Dependencies

**Blocked By**:
- BlockPoolBatchEngine._reconstruct_cache() implementation (Day 7-8)
- AgentBlocks.blocks_for_layer() helper method
- Synthetic block allocation for testing

**Blocks**:
- ADR-004: Block Gather Strategy (decision record)
- Performance optimization decisions (if fails)

---

## Notes

- Use Gemma 3 12B for realistic layer count (48 layers)
- Test with 8K context (typical conversation length)
- Force mx.eval() to ensure immediate execution
- Run 100 samples for statistical significance
- Compare against padded approach baseline (if available)

---

**Status**: ⏳ PENDING (Day 8)
**Last Updated**: 2026-01-24 (stub created Day 3)
