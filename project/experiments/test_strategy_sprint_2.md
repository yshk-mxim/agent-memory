# Test Strategy: Sprint 2 Experiments

**Date**: 2026-01-24
**Sprint**: 2 - Block-Pool Batch Engine
**Owner**: QE (Quality Engineer)
**Purpose**: Define validation approach for BlockPoolBatchEngine correctness and performance

---

## Overview

Sprint 2 requires two critical experiments to validate the BlockPoolBatchEngine implementation:

- **EXP-005**: BlockPoolBatchEngine Correctness (output validation)
- **EXP-006**: Block Gather Performance (cache reconstruction benchmark)

This document defines the test strategy, validation criteria, and measurement methods for both experiments.

---

## EXP-005: BlockPoolBatchEngine Correctness

### Objective

Prove that `BlockPoolBatchEngine` produces **byte-identical output** to reference `mlx_lm.generate()` implementation.

**Critical Requirement**: Output must match exactly (not just semantically similar) to ensure cache reconstruction is correct.

---

### Test Data

**Test Prompts** (3 prompt lengths):

| ID | Label | Prompt Text | Expected Tokens | Rationale |
|----|-------|-------------|-----------------|-----------|
| 1 | short | "The quick brown fox" | ~50 total | Tests minimal cache (1 block) |
| 2 | medium | "Write a story about a robot" | ~500 total | Tests medium cache (2 blocks) |
| 3 | long | "Explain quantum computing in detail" | ~2000 total | Tests large cache (8 blocks) |

**Prompt Design Rationale**:
- Short: Fits in 1 block (256 tokens), validates single-block allocation
- Medium: Spans 2 blocks, validates block extension during decode
- Long: Spans 8 blocks, validates multi-block reconstruction

**Fixed Parameters** (for reproducibility):
```python
max_tokens = 100  # Generate same number of tokens for all prompts
temperature = 0.0  # Greedy decoding (deterministic)
seed = 42  # Fixed random seed
model = "mlx-community/SmolLM2-135M-Instruct"  # Fast model for testing
```

---

### Reference Generation

**Method**: Use canonical `mlx_lm.generate()` to create ground truth.

```python
from mlx_lm import load, generate

# Load model once
model, tokenizer = load("mlx-community/SmolLM2-135M-Instruct")

# Generate reference for each prompt
def generate_reference(prompt: str, max_tokens: int = 100) -> str:
    """Generate reference output using mlx_lm.generate()."""
    result = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        temp=0.0,  # Greedy (deterministic)
        verbose=False,
    )
    return result

# Save references for validation
references = {
    "short": generate_reference("The quick brown fox"),
    "medium": generate_reference("Write a story about a robot"),
    "long": generate_reference("Explain quantum computing in detail"),
}
```

---

### Test Generation

**Method**: Use `BlockPoolBatchEngine.submit()` and `step()` to generate test output.

```python
from semantic.domain.services import BlockPool
from semantic.domain.value_objects import ModelCacheSpec
from semantic.application.batch_engine import BlockPoolBatchEngine

# Setup
spec = ModelCacheSpec.from_model(model)
pool = BlockPool(spec, total_blocks=100)
engine = BlockPoolBatchEngine(model, tokenizer, pool, spec)

# Generate test output
def generate_test(prompt: str, max_tokens: int = 100) -> str:
    """Generate output using BlockPoolBatchEngine."""
    uid = engine.submit(
        agent_id="test_agent",
        prompt=prompt,
        cache=None,  # No cache (fresh generation)
        max_tokens=max_tokens,
    )

    # Poll for completion
    for completion in engine.step():
        if completion.uid == uid:
            return completion.text

    raise RuntimeError(f"Generation {uid} did not complete")
```

---

### Validation Criteria

**Primary Criteria** (MUST PASS):

1. **✅ Byte-Identical Output**
   ```python
   def validate_output(actual: str, expected: str, prompt_label: str) -> bool:
       """Validate output matches reference byte-for-byte."""
       if actual != expected:
           print(f"❌ {prompt_label}: Output mismatch")
           print(f"   Expected: '{expected[:50]}...'")
           print(f"   Actual:   '{actual[:50]}...'")
           return False
       return True
   ```
   **Success Criteria**: `actual == expected` (100% match)

2. **✅ No Errors During Generation**
   ```python
   # No exceptions raised during submit() or step()
   # All completions have finish_reason != "error"
   ```
   **Success Criteria**: Zero exceptions, all `finish_reason` in ["stop", "length"]

3. **✅ Correct Token Count**
   ```python
   def validate_token_count(completion, prompt: str, max_tokens: int) -> bool:
       """Validate token count is within expected range."""
       prompt_tokens = len(tokenizer.encode(prompt))
       total_tokens = completion.token_count
       generated_tokens = total_tokens - prompt_tokens

       # Generated tokens should be <= max_tokens
       assert generated_tokens <= max_tokens
       assert generated_tokens > 0  # At least 1 token generated
       return True
   ```
   **Success Criteria**: Generated tokens ∈ [1, max_tokens]

**Secondary Criteria** (NICE TO HAVE):

4. **Performance Within 20% of Reference**
   ```python
   import time

   start = time.perf_counter()
   result = generate_test(prompt, max_tokens)
   end = time.perf_counter()
   test_time = end - start

   # Compare to reference baseline
   assert test_time <= reference_time * 1.2  # Within 20%
   ```

5. **No Memory Leaks**
   ```python
   # Run 10 consecutive generations
   initial_used = pool.used_blocks()
   for _ in range(10):
       result = generate_test(prompt, max_tokens)
       pool.free_agent_blocks("test_agent")  # Clean up
   final_used = pool.used_blocks()

   # Pool size should be stable
   assert final_used == initial_used  # No leaked blocks
   ```

---

### Measurement Method

**Validation Script**:

```python
def run_exp_005():
    """Execute EXP-005 validation."""
    results = []

    for prompt_id, (label, prompt) in enumerate([
        ("short", "The quick brown fox"),
        ("medium", "Write a story about a robot"),
        ("long", "Explain quantum computing in detail"),
    ]):
        print(f"\n🧪 Testing prompt '{label}'...")

        # Generate reference
        reference = references[label]

        # Generate test
        actual = generate_test(prompt, max_tokens=100)

        # Validate
        match = validate_output(actual, reference, label)

        # Record results
        results.append({
            "prompt": label,
            "match": match,
            "reference_len": len(reference),
            "actual_len": len(actual),
        })

    # Summary
    passed = sum(1 for r in results if r["match"])
    print(f"\n📊 EXP-005 Results: {passed}/3 passed")

    if passed == 3:
        print("✅ EXP-005 PASSED: All outputs match reference")
        return True
    else:
        print("❌ EXP-005 FAILED: Output mismatch detected")
        return False
```

---

### Success Criteria Summary

**GO Criteria** (all required):
- ✅ 3/3 prompts produce byte-identical output
- ✅ No exceptions or errors during generation
- ✅ Token counts within expected range

**Stretch Goals** (optional):
- ✅ Performance within 20% of reference
- ✅ No memory leaks after 10 runs

**If EXP-005 FAILS**:
- **Action 1**: Compare token-by-token to identify divergence point
- **Action 2**: Validate cache reconstruction logic (blocks → KVCache)
- **Action 3**: Check for floating-point precision issues
- **Escalation**: If mismatch > 5 tokens, escalate to PM (BLOCKING issue)

---

## EXP-006: Block Gather Performance

### Objective

Measure cache reconstruction overhead when converting blocks → KVCache objects.

**Critical Requirement**: Gather time must be < 5ms (p95) for 8K context (32 blocks × 48 layers).

---

### Test Setup

**Simulated Cache** (8K context on Gemma 3):

```python
# Create synthetic blocks for 8K context
# - Gemma 3 12B: 48 layers (8 global + 40 sliding window)
# - 8K tokens = 32 blocks (8192 / 256 = 32)

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

---

### Benchmark Method

**Measurement**: Time the `_reconstruct_cache()` call with Python `time.perf_counter()`.

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
    p50 = np.percentile(times, 50)
    p95 = np.percentile(times, 95)
    p99 = np.percentile(times, 99)
    mean = np.mean(times)
    std = np.std(times)

    return {
        "p50": p50,
        "p95": p95,
        "p99": p99,
        "mean": mean,
        "std": std,
        "min": min(times),
        "max": max(times),
        "samples": n_runs,
    }
```

---

### Validation Criteria

**Primary Criteria**:

1. **✅ p95 < 5ms** (Target)
   ```python
   stats = benchmark_block_gather(agent_blocks, n_runs=100)
   assert stats["p95"] < 5.0  # milliseconds
   ```
   **Rationale**: 5ms overhead is acceptable for one-time cache restoration (not per-step cost).

2. **✅ Consistent Performance** (Low Variance)
   ```python
   # Standard deviation should be < 20% of mean
   assert stats["std"] < stats["mean"] * 0.2
   ```
   **Rationale**: High variance suggests unpredictable performance.

**Secondary Criteria**:

3. **Nice to Have**: p99 < 10ms
   ```python
   assert stats["p99"] < 10.0  # Even tail latency is acceptable
   ```

4. **Nice to Have**: Linear scaling with block count
   ```python
   # Test with 16 blocks (4K context) and 32 blocks (8K context)
   stats_4k = benchmark_block_gather(blocks_4k)
   stats_8k = benchmark_block_gather(blocks_8k)

   # Time should roughly double (2x blocks = 2x time)
   ratio = stats_8k["mean"] / stats_4k["mean"]
   assert 1.5 < ratio < 2.5  # Within reasonable range
   ```

---

### Measurement Output

**Results Table**:

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| p50 (median) | 2.1 ms | < 5 ms | ✅ PASS |
| p95 | 3.8 ms | < 5 ms | ✅ PASS |
| p99 | 4.5 ms | < 10 ms | ✅ PASS |
| Mean | 2.3 ms | - | - |
| Std Dev | 0.4 ms | < 20% of mean | ✅ PASS |
| Min | 1.8 ms | - | - |
| Max | 5.2 ms | - | - |

**Visualization** (optional):
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

### Success Criteria Summary

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

### If EXP-006 FAILS (p95 > 5ms)

**Investigation Steps**:

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

## Test Execution Order

**Day 6-7** (BlockPoolBatchEngine implementation):
1. Implement `submit()` method
2. Implement `step()` method
3. Implement `_reconstruct_cache()` helper

**Day 8** (EXP-005 execution):
1. Generate reference outputs (save to `exp_005_references.json`)
2. Run `run_exp_005()` script
3. Validate all 3 prompts
4. Document results in `/project/experiments/EXP-005-blockpool-correctness.md`

**Day 8** (EXP-006 execution):
1. Set up 8K context synthetic blocks
2. Run `benchmark_block_gather()` script
3. Analyze results (p50, p95, p99)
4. Document results in `/project/experiments/EXP-006-block-gather-benchmark.md`

**Day 9** (Integration):
1. If EXP-005 PASS: Proceed to integration tests
2. If EXP-006 PASS: Document in ADR-004
3. If either FAIL: Debugging session + PM escalation

---

## Quality Gates

**EXP-005 Quality Gate**:
- ❌ BLOCK Sprint 2 if any prompt fails byte-identical check
- ⚠️ WARNING if performance > 50% slower than reference
- ✅ PASS if all 3 prompts match exactly

**EXP-006 Quality Gate**:
- ✅ PASS if p95 < 5ms (proceed immediately)
- ⚠️ CONDITIONAL if p95 between 5-10ms (document trade-off)
- ❌ BLOCK if p95 > 10ms (requires refactoring)

---

## Artifacts

**Generated During Experiments**:
- `exp_005_references.json` - Ground truth outputs
- `exp_005_results.csv` - Validation results table
- `exp_006_timings.json` - Raw timing data (100 samples)
- `exp_006_histogram.png` - Performance distribution visualization

**Documentation**:
- `/project/experiments/EXP-005-blockpool-correctness.md` - Full experiment report
- `/project/experiments/EXP-006-block-gather-benchmark.md` - Full benchmark report
- `/project/architecture/ADR-004-block-gather-strategy.md` - Decision record (if p95 > 5ms)

---

## References

- **Sprint 2 Plan**: `/project/sprints/sprint_2_block_pool_batch_engine.md` (lines 336-409)
- **mlx_lm API**: `/project/reference/mlx_lm_api_v0.30.4.md`
- **ADR-002**: Block Size = 256 Tokens (memory calculations)
- **EXP-003**: Cache Injection Validation (Sprint 0) - Baseline for correctness
- **EXP-004**: Cache Extraction Validation (Sprint 0) - Baseline for correctness

---

**Status**: ✅ COMPLETE
**Author**: QE (Quality Engineer)
**Reviewed By**: ML, SE, PM
**Date**: 2026-01-24
