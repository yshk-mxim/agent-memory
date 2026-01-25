# EXP-006: Cache Reconstruction Performance

**Experiment**: Benchmark cache reconstruction latency
**Sprint**: 2 (Block-Pool Batch Engine)
**Status**: 📋 READY TO EXECUTE
**Date**: 2026-01-24

---

## Objective

Measure `_reconstruct_cache()` and `_extract_cache()` performance to validate design targets.

**Target**: p95 < 5ms for reconstruction of 32 blocks × 48 layers (8K tokens, Gemma 3)

---

## Hypothesis

Block-based cache reconstruction has minimal overhead:
- mx.concatenate is fast (GPU operation)
- Predicted: ~3-5ms for 32 blocks × 48 layers
- Not a bottleneck in generation pipeline

---

## Method

### Test Setup

```python
import time
import numpy as np
from mlx_lm import load
from semantic.application.batch_engine import BlockPoolBatchEngine
from semantic.domain.services import BlockPool
from semantic.domain.value_objects import ModelCacheSpec

# Load model
model, tokenizer = load("mlx-community/SmolLM2-135M-Instruct")
spec = ModelCacheSpec.from_model(model)
pool = BlockPool(spec=spec, total_blocks=500)
engine = BlockPoolBatchEngine(model, tokenizer, pool, spec)
```

### Benchmark 1: _reconstruct_cache()

**Purpose**: Measure AgentBlocks → KVCache conversion time

```python
# Generate cache first
uid = engine.submit("agent", "Long prompt..." * 50, max_tokens=100)
completions = list(engine.step())
agent_blocks = completions[0].blocks

# Benchmark reconstruction
times_reconstruct = []
for _ in range(100):
    start = time.perf_counter()
    cache = engine._reconstruct_cache(agent_blocks)
    end = time.perf_counter()
    times_reconstruct.append((end - start) * 1000)  # ms

# Calculate percentiles
p50 = np.percentile(times_reconstruct, 50)
p95 = np.percentile(times_reconstruct, 95)
p99 = np.percentile(times_reconstruct, 99)
mean = np.mean(times_reconstruct)

print(f"Reconstruction: mean={mean:.2f}ms, p50={p50:.2f}ms, p95={p95:.2f}ms, p99={p99:.2f}ms")
```

### Benchmark 2: _extract_cache()

**Purpose**: Measure KVCache → AgentBlocks conversion time

```python
# Generate cache
uid = engine.submit("agent", "Prompt", max_tokens=100)
list(engine.step())  # Complete generation

# Benchmark extraction (happens in step, but measure separately)
times_extract = []
for _ in range(100):
    # Re-submit to get UID
    uid = engine.submit("agent", "Test", max_tokens=10)

    start = time.perf_counter()
    blocks = engine._extract_cache(uid)
    end = time.perf_counter()
    times_extract.append((end - start) * 1000)  # ms

p50 = np.percentile(times_extract, 50)
p95 = np.percentile(times_extract, 95)
p99 = np.percentile(times_extract, 99)

print(f"Extraction: p50={p50:.2f}ms, p95={p95:.2f}ms, p99={p99:.2f}ms")
```

### Benchmark 3: Round-Trip

**Purpose**: Measure full cache lifecycle (reconstruct + generate + extract)

```python
times_roundtrip = []
for _ in range(100):
    # Reconstruct
    start = time.perf_counter()
    cache = engine._reconstruct_cache(agent_blocks)

    # Use cache (submit with cache)
    uid = engine.submit("agent", "Continue", cache=agent_blocks, max_tokens=10)

    # Extract (in step)
    completions = list(engine.step())
    end = time.perf_counter()

    times_roundtrip.append((end - start) * 1000)

print(f"Round-trip: p95={np.percentile(times_roundtrip, 95):.2f}ms")
```

---

## Test Cases

### Case 1: Small Cache (1 block × 12 layers)

**Context Size**: 256 tokens
**Model**: SmolLM2-135M (30 layers)
**Expected**: p95 < 0.1ms

### Case 2: Medium Cache (4 blocks × 24 layers)

**Context Size**: 1024 tokens
**Model**: Qwen 2.5 (hypothetical, or use SmolLM)
**Expected**: p95 < 0.5ms

### Case 3: Large Cache (8 blocks × 30 layers)

**Context Size**: 2048 tokens
**Model**: SmolLM2-135M
**Expected**: p95 < 1ms

### Case 4: Target Case (32 blocks × 48 layers)

**Context Size**: 8192 tokens
**Model**: Gemma 3 12B (if available) or extrapolate
**Expected**: p95 < 5ms (target)

---

## Variables

**Independent Variables**:
- Number of blocks: 1, 2, 4, 8, 16, 32
- Number of layers: 12, 24, 30, 48
- Block size: 256 tokens (constant)

**Dependent Variables**:
- Reconstruction latency (ms)
- Extraction latency (ms)
- Round-trip latency (ms)

**Controlled**:
- Hardware: Same machine for all runs
- Model: Same model per test case
- Warmup: 10 iterations before measurement
- Iterations: 100 samples per case

---

## Expected Results

| Blocks | Layers | Total Ops | Expected p95 | Actual p95 |
|--------|--------|-----------|--------------|------------|
| 1 | 12 | 12 | < 0.1ms | TBD |
| 4 | 24 | 96 | < 0.5ms | TBD |
| 8 | 30 | 240 | < 1ms | TBD |
| 16 | 30 | 480 | < 2ms | TBD |
| 32 | 48 | 1,536 | < 5ms | TBD |

**Calculation**:
- Total ops = blocks × layers
- Each op = 1 mx.concatenate(K) + 1 mx.concatenate(V)
- Expected: ~3μs per concatenate (GPU operation)
- 1,536 ops × 3μs ≈ 4.6ms (within target)

---

## Performance Analysis

### Scaling Behavior

**Linear Scaling Expected**:
- Latency ∝ (number of blocks) × (number of layers)
- Doubling blocks → ~2x latency
- Doubling layers → ~2x latency

**Verify**:
```python
# Plot: latency vs (blocks × layers)
import matplotlib.pyplot as plt

plt.scatter(blocks * layers, latencies)
plt.xlabel("Total Operations (blocks × layers)")
plt.ylabel("p95 Latency (ms)")
plt.title("Cache Reconstruction Scaling")
plt.savefig("exp_006_scaling.png")
```

### Bottleneck Identification

**Potential Bottlenecks**:
1. mx.concatenate GPU kernel
2. mx.eval() synchronization
3. Memory bandwidth
4. Python overhead

**Diagnosis**:
- Profile with MLX profiler
- Measure CPU vs GPU time
- Check memory throughput

---

## Comparison

### vs. Padded Approach

**Hypothesis**: Block-based faster than padding (no wasted compute)

**Test**:
- Generate with 612 tokens (padded: 768, blocks: 3×256)
- Compare: padded vs block reconstruction
- Expected: Block-based 20% faster

### vs. Direct Generation

**Baseline**: Direct generation (no cache)
- First token latency
- Total generation time

**Comparison**:
- Cache reconstruction + generation vs direct
- Expected: Cache faster for prompts > 256 tokens

---

## Success Criteria

### Must Have
- ✅ p95 < 5ms for 32 blocks × 48 layers
- ✅ Linear scaling behavior
- ✅ No memory leaks

### Nice to Have
- ⭐ p95 < 3ms (exceed target)
- ⭐ Faster than padded approach
- ⭐ Sub-millisecond for typical cases (< 8 blocks)

---

## Results

**Status**: ⏳ PENDING EXECUTION

### Reconstruction Performance

| Configuration | Iterations | Mean | p50 | p95 | p99 | Status |
|---------------|------------|------|-----|-----|-----|--------|
| 1 block × 12 layers | 100 | TBD | TBD | TBD | TBD | ⏳ |
| 4 blocks × 24 layers | 100 | TBD | TBD | TBD | TBD | ⏳ |
| 8 blocks × 30 layers | 100 | TBD | TBD | TBD | TBD | ⏳ |
| 32 blocks × 48 layers | 100 | TBD | TBD | TBD | TBD | ⏳ |

### Extraction Performance

| Configuration | p95 | Status |
|---------------|-----|--------|
| 1 block × 12 layers | TBD | ⏳ |
| 8 blocks × 30 layers | TBD | ⏳ |

### Round-Trip Performance

| Configuration | p95 | Status |
|---------------|-----|--------|
| Full cycle (8 blocks) | TBD | ⏳ |

---

## Execution Instructions

```bash
# 1. Set up environment
pip install mlx==0.30.3 mlx-lm==0.30.4 numpy matplotlib

# 2. Run benchmark script
python -m experiments.exp_006_benchmark

# 3. Results will be saved to:
# - experiments/results/exp_006_results.json
# - experiments/results/exp_006_scaling.png
```

---

## Interpretation

### If p95 < 5ms ✅
- **Conclusion**: Design target met
- **Action**: Document results, proceed to Sprint 3

### If 5ms < p95 < 10ms ⚠️
- **Conclusion**: Slower than target but acceptable
- **Action**: Document, consider optimizations
- **Optimizations**:
  - Batch concatenate operations
  - Pre-allocate output tensors
  - Reduce mx.eval() calls

### If p95 > 10ms ❌
- **Conclusion**: Performance issue
- **Action**: Profile and optimize
- **Potential Fixes**:
  - Reduce layers per mx.eval()
  - Use mx.compile() for concatenate
  - Investigate memory bandwidth

---

## Follow-Up Experiments

### EXP-006.1: Optimization Strategies
- Test mx.compile() on _reconstruct_cache()
- Test batched concatenate
- Test alternative tensor layouts

### EXP-006.2: Production Profiling
- Real-world workload (5 concurrent agents)
- Cache reconstruction in production
- End-to-end latency impact

---

## Conclusion

**Status**: ⏳ PENDING EXECUTION

**Expected Outcome**: p95 < 5ms ✅

**Risk**: LOW - Design predicts 3-5ms, conservative target

**Next Steps**:
1. Execute experiment with MLX
2. Record results
3. If PASS: Document and continue
4. If FAIL: Optimize and re-test

---

**Prepared By**: ML (Machine Learning Engineer), HW (Hardware Engineer)
**Date**: 2026-01-24 (Sprint 2, Day 9)
**Status**: 📋 READY TO EXECUTE
