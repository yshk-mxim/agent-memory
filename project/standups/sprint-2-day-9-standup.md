# Sprint 2 Day 9 Standup: Integration Testing

**Date**: 2026-01-24
**Sprint**: 2 (Block-Pool Batch Engine)
**Day**: 9
**Focus**: Integration tests with real MLX models

---

## Previous Days Review (Days 6-8)

**Status**: ✅ COMPLETE

**Core Implementation Delivered**:
- Day 6: submit() + step() with FakeBatchGenerator
- Day 7: _reconstruct_cache() (AgentBlocks → KVCache)
- Day 8: _extract_cache() (KVCache → AgentBlocks)
- Result: 128/128 unit tests passing, all quality gates passing

**Commits**:
- `51c16b4` - Day 6 core implementation
- `83a775b` - Day 7 cache reconstruction
- `a0e175a` - Day 8 cache extraction
- `ec57b85` - Week 2 review

---

## Day 9 Goals

**Primary Objective**: Implement integration tests to validate with real MLX models

**Deliverables**:
1. Complete 6 integration tests (ready to execute)
2. Document EXP-005 validation procedure
3. Document EXP-006 benchmark procedure
4. Environment setup documentation
5. Test execution guide

---

## Integration Test Plan

### Test 1: Single Agent Fresh Generation

**Purpose**: Verify engine generates text correctly with no cache

**Steps**:
1. Load SmolLM2-135M model
2. Create engine with model + pool
3. Submit prompt "Hello" with max_tokens=20
4. Execute step() until completion
5. Verify text generated, finish_reason="stop"
6. Verify blocks allocated and populated

**Success Criteria**:
- Text generated (non-empty)
- finish_reason = "stop" or "length"
- blocks.total_tokens > 0
- No crashes or errors

---

### Test 2: Single Agent Cache Resume

**Purpose**: Verify cache reconstruction works end-to-end

**Steps**:
1. Generate with prompt "Hello"
2. Extract blocks from completion
3. Submit with prompt " world" + cached blocks
4. Verify faster execution (cache hit)
5. Verify output continues from cache

**Success Criteria**:
- Second generation faster than first
- Output logically continues
- No re-encoding of "Hello"

---

### Test 3: Multi-Agent Variable Lengths

**Purpose**: Verify batching with 3 concurrent agents

**Steps**:
1. Submit 3 prompts: "Hi" (short), "Hello world" (medium), "Long prompt..." (long)
2. Execute step() in loop
3. Collect all 3 completions
4. Verify all completed successfully

**Success Criteria**:
- All 3 agents complete
- Correct text for each
- Block counts match prompt lengths

---

### Test 4: Memory Leak Detection

**Purpose**: Verify no block leaks across 10 generations

**Steps**:
1. Record initial pool.available_blocks()
2. Run 10 generations (submit → step → extract blocks)
3. Manually free all blocks
4. Verify pool.available_blocks() == initial

**Success Criteria**:
- No memory leak (available blocks constant)
- All blocks properly freed

---

### Test 5: Pool Exhaustion Handling

**Purpose**: Verify graceful handling when pool exhausted

**Steps**:
1. Create small pool (10 blocks)
2. Submit request requiring 20 blocks
3. Verify PoolExhaustedError raised

**Success Criteria**:
- PoolExhaustedError raised (not crash)
- Error message clear

---

### Test 6: Empty Prompt Rejection

**Purpose**: Verify validation works in integration

**Steps**:
1. Submit empty prompt ""
2. Verify InvalidRequestError raised

**Success Criteria**:
- InvalidRequestError raised
- Clear error message

---

## MLX Environment Setup

### Requirements

**Hardware**:
- Apple Silicon (M1/M2/M3/M4)
- 16GB+ RAM (24GB recommended)

**Software**:
- macOS 14+ (Sonoma)
- Python 3.11+
- mlx==0.30.3
- mlx-lm==0.30.4

### Setup Steps

```bash
# 1. Install dependencies (if not already)
pip install mlx==0.30.3 mlx-lm==0.30.4

# 2. Download model (first run only)
# mlx-lm will auto-download on first load()

# 3. Run integration tests
pytest tests/integration/test_batch_engine.py -v -m integration
```

### Model Selection

**SmolLM2-135M-Instruct**:
- Size: ~270MB
- Layers: 30
- Fast for testing
- Good for CI/CD

---

## EXP-005: Output Validation

**Goal**: Verify generated text matches reference implementation

### Procedure

1. **Reference Generation** (mlx_lm direct):
```python
from mlx_lm import load, generate

model, tokenizer = load("mlx-community/SmolLM2-135M-Instruct")
reference_text = generate(model, tokenizer, prompt="Hello", max_tokens=20)
```

2. **Test Generation** (BlockPoolBatchEngine):
```python
engine = BlockPoolBatchEngine(model, tokenizer, pool, spec)
uid = engine.submit("agent_1", "Hello", max_tokens=20)
for completion in engine.step():
    test_text = completion.text
```

3. **Comparison**:
```python
assert test_text == reference_text, "Output mismatch!"
```

**Success Criteria**: Byte-identical output

**Note**: May need deterministic generation (temperature=0, seed=42)

---

## EXP-006: Performance Benchmark

**Goal**: Measure cache reconstruction performance

### Metrics

| Operation | Target | Measurement |
|-----------|--------|-------------|
| _reconstruct_cache() | p95 < 5ms | Reconstruct 32 blocks x 48 layers |
| _extract_cache() | p95 < 10ms | Extract and split cache |
| Round-trip | p95 < 15ms | Reconstruct + extract |

### Procedure

```python
import time
import numpy as np

# Benchmark reconstruction
times = []
for _ in range(100):
    start = time.perf_counter()
    cache = engine._reconstruct_cache(agent_blocks)
    end = time.perf_counter()
    times.append((end - start) * 1000)  # ms

p50 = np.percentile(times, 50)
p95 = np.percentile(times, 95)
p99 = np.percentile(times, 99)

print(f"Reconstruction: p50={p50:.2f}ms, p95={p95:.2f}ms, p99={p99:.2f}ms")
```

**Test Cases**:
- 1 block x 12 layers (small)
- 4 blocks x 24 layers (medium)
- 32 blocks x 48 layers (large - target case)

---

## Implementation Timeline

| Task | Estimate | Priority |
|------|----------|----------|
| Implement 6 integration tests | 90 min | High |
| Document EXP-005 procedure | 20 min | Medium |
| Document EXP-006 procedure | 30 min | Medium |
| Environment setup guide | 20 min | High |
| Review & commit | 20 min | High |
| **Total** | **3 hours** | - |

---

## Testing Strategy

**Without MLX Environment** (Day 9):
- Implement all tests
- Mark as `@pytest.mark.integration`
- Skip if MLX not available
- Document requirements

**With MLX Environment** (Future):
- Run `pytest -m integration`
- Execute EXP-005 & EXP-006
- Create performance report

---

## Success Criteria

**Day 9 Complete When**:
- [x] 6 integration tests implemented
- [x] Tests marked with proper pytest markers
- [x] EXP-005 procedure documented
- [x] EXP-006 procedure documented
- [x] Environment setup guide created
- [x] Day 9 review document created
- [x] Work committed

---

## Notes

- Integration tests can't run without MLX, but must be implemented
- Tests should be production-ready (just need environment)
- Document clear setup instructions for future execution
- EXP-005/006 can be manual procedures (scripts not required)

---

**Status**: ✅ READY TO START
**Next Step**: Implement integration tests in test_batch_engine.py
