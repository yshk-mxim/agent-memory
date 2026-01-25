# Week 2 Implementation Checklist: BlockPoolBatchEngine

**Sprint**: 2 (Block-Pool Batch Engine)
**Timeline**: Days 6-10
**Owner**: ML (Machine Learning Engineer), SE (Software Engineer)
**Status**: ⏳ PENDING (starts Day 6)

---

## Overview

**Goal**: Implement BlockPoolBatchEngine that generates correct text using block-pool allocation with variable-length batching.

**Design Document**: `/project/architecture/blockpool-batch-engine-design.md` (1,091 lines)

**Exit Criteria**:
- ✅ Output matches reference (greedy, temperature=0)
- ✅ No memory leaks (pool size stable across 10+ generations)
- ✅ < 20% throughput regression vs POC baseline
- ✅ Integration tests pass on Apple Silicon

---

## Implementation Phases

### Phase 1: Core Structure (Day 6)

**Duration**: 4 hours
**Owner**: SE

#### Task 1.1: Create BlockPoolBatchEngine Class

**File**: `/src/semantic/application/batch_engine.py` (new)

**Checklist**:
- [ ] Create file with module docstring
- [ ] Import dependencies (mlx_lm, domain types, ports)
- [ ] Define class with type hints
- [ ] Add class docstring (responsibilities, example)
- [ ] Implement __init__() method (11 steps from design doc)
- [ ] Validate inputs in __init__()
- [ ] Initialize internal state (_batch_gen, _active_requests, etc.)

**Code Template**:
```python
"""BlockPoolBatchEngine - Batched inference with block-pool memory."""

from typing import Any, Iterator

import mlx.core as mx
from mlx_lm import BatchGenerator

from semantic.domain.entities import AgentBlocks, KVBlock
from semantic.domain.services import BlockPool
from semantic.domain.value_objects import (
    CompletedGeneration,
    ModelCacheSpec,
)
from semantic.ports.inbound import GenerationEnginePort


class BlockPoolBatchEngine:
    """Batched inference engine with block-pool memory management.

    Implements GenerationEnginePort for async submit/step pattern.
    Wraps mlx_lm BatchGenerator with block-based KV cache allocation.

    Responsibilities:
    - Allocate blocks for new sequences
    - Reconstruct KVCache from blocks (one-time at restore)
    - Submit requests to batch queue
    - Execute decode steps in batches
    - Extract updated cache back to blocks
    - Free blocks when sequences complete

    Example:
        >>> engine = BlockPoolBatchEngine(model, tokenizer, pool, spec)
        >>> uid = engine.submit("agent_1", "Hello", max_tokens=50)
        >>> for completion in engine.step():
        ...     print(f"{completion.uid}: {completion.text}")
    """

    def __init__(
        self,
        model: Any,  # MLX model
        tokenizer: Any,  # MLX tokenizer
        pool: BlockPool,
        spec: ModelCacheSpec,
    ) -> None:
        """Initialize batch engine."""
        # TODO: Implement (11 steps from design doc)
        pass
```

**Verification**:
- [ ] mypy --strict passes
- [ ] ruff passes
- [ ] Imports resolve correctly

---

#### Task 1.2: Implement submit() Method

**Duration**: 2 hours

**Checklist**:
- [ ] Implement submit() signature (from GenerationEnginePort)
- [ ] Add docstring with Args/Returns/Raises
- [ ] Validate inputs (prompt not empty, max_tokens > 0)
- [ ] Tokenize prompt using self._tokenizer
- [ ] Allocate blocks if no cache provided
- [ ] Reconstruct KVCache from blocks if cache provided
- [ ] Create BatchGenerator lazily (if None)
- [ ] Insert sequence into batch
- [ ] Track UID → agent_id mapping
- [ ] Return UID

**Algorithm** (12 steps from design doc):
1. Validate inputs
2. Tokenize prompt
3. Allocate blocks (if no cache)
4. Reconstruct cache (if cache provided)
5. Lazy-init BatchGenerator
6. Insert into batch
7. Track UID → agent_id
8. Return UID

**Edge Cases**:
- [ ] Empty prompt → raise InvalidRequestError
- [ ] max_tokens <= 0 → raise InvalidRequestError
- [ ] Pool exhausted → raise PoolExhaustedError
- [ ] No model loaded → raise ModelNotFoundError

**Verification**:
- [ ] Unit test: submit with no cache
- [ ] Unit test: submit with existing cache
- [ ] Unit test: validate input errors
- [ ] mypy/ruff clean

---

### Phase 2: Cache Reconstruction (Day 7)

**Duration**: 3 hours
**Owner**: ML

#### Task 2.1: Implement _reconstruct_cache() Helper

**File**: `batch_engine.py` (same file)

**Checklist**:
- [ ] Implement _reconstruct_cache() signature
- [ ] Add docstring with performance note (p95 < 5ms target)
- [ ] Loop over all layers
- [ ] Get blocks for each layer
- [ ] Extract K and V tensors from each block
- [ ] Concatenate along sequence length axis (axis=2)
- [ ] Force evaluation (mx.eval())
- [ ] Append (k_full, v_full) to cache list
- [ ] Return cache list

**Algorithm** (7 steps from design doc):
1. Initialize cache list
2. For each layer_id in range(n_layers):
   - Get layer blocks
   - Extract K tensors
   - Extract V tensors
   - Concatenate K (axis=2)
   - Concatenate V (axis=2)
   - Force mx.eval()
   - Append to cache
3. Return cache

**Performance Target**: p95 < 5ms for 8K context (validated in EXP-006)

**Edge Cases**:
- [ ] Empty blocks (no cache) → return None or empty list
- [ ] Single block → no concatenation needed
- [ ] Sliding window layers → subset of blocks

**Verification**:
- [ ] Unit test: reconstruct from 1 block
- [ ] Unit test: reconstruct from 32 blocks
- [ ] Unit test: verify tensor shapes match spec
- [ ] Benchmark: p95 timing (EXP-006)

---

### Phase 3: Decode Step (Day 8 Morning)

**Duration**: 3 hours
**Owner**: ML

#### Task 3.1: Implement step() Method

**File**: `batch_engine.py` (same file)

**Checklist**:
- [ ] Implement step() signature (yields CompletedGeneration)
- [ ] Add docstring with example
- [ ] Check if batch is empty → return immediately
- [ ] Call batch_gen.next() to execute one decode step
- [ ] Check for finished sequences
- [ ] For each finished sequence:
  - Extract cache
  - Extract updated blocks
  - Create CompletedGeneration
  - Yield to caller
  - Remove from batch
  - Clean up tracking

**Algorithm** (10 steps from design doc):
1. Guard: if no active batch, return
2. Call batch_gen.next()
3. Get finished sequences
4. For each finished:
   - Extract cache
   - Extract blocks
   - Create CompletedGeneration
   - Yield
   - Remove from batch
   - Free tracking
5. Return

**Edge Cases**:
- [ ] No batch created yet → return immediately
- [ ] Batch empty → return immediately
- [ ] No completions this step → yield nothing
- [ ] Error during decode → finish_reason="error"

**Verification**:
- [ ] Unit test: step with 1 sequence
- [ ] Unit test: step with 3 sequences (some complete, some not)
- [ ] Unit test: step with empty batch
- [ ] Integration test: full submit → step → completion cycle

---

### Phase 4: Cache Extraction (Day 8 Afternoon)

**Duration**: 2 hours
**Owner**: ML

#### Task 4.1: Implement _extract_cache() Helper

**File**: `batch_engine.py` (same file)

**Checklist**:
- [ ] Implement _extract_cache() signature
- [ ] Add docstring
- [ ] Call batch_gen.extract_cache(uid)
- [ ] Convert cache back to blocks
- [ ] For each layer:
  - Get K and V tensors
  - Split into block-sized chunks
  - Create KVBlock for each chunk
  - Add to AgentBlocks
- [ ] Return AgentBlocks

**Algorithm** (6 steps from design doc):
1. Call batch_gen.extract_cache(uid)
2. Create AgentBlocks container
3. For each layer in cache:
   - Get K and V tensors
   - Split into chunks (block_tokens size)
   - For each chunk:
     - Create KVBlock
     - Add to AgentBlocks
4. Return AgentBlocks

**Edge Cases**:
- [ ] Cache length not multiple of block_tokens → partial block
- [ ] Sliding window layers → truncate to window size
- [ ] Empty cache → empty AgentBlocks

**Verification**:
- [ ] Unit test: extract 1-block cache
- [ ] Unit test: extract 32-block cache
- [ ] Unit test: roundtrip (reconstruct → extract → reconstruct)
- [ ] Property test: cache size preservation

---

### Phase 5: Integration Testing (Day 9)

**Duration**: 5 hours
**Owner**: QE, ML

#### Task 5.1: EXP-005 - Byte-Identical Output Validation

**File**: `/project/experiments/scripts/run_exp_005.py`

**Checklist**:
- [ ] Remove NotImplementedError placeholders
- [ ] Implement setup_test_environment()
- [ ] Implement generate_reference() (mlx_lm.generate)
- [ ] Implement generate_test() (BlockPoolBatchEngine)
- [ ] Implement validate_output() (byte-for-byte comparison)
- [ ] Implement validate_token_count()
- [ ] Run with 3 prompts (short/medium/long)
- [ ] Verify all 3 pass
- [ ] Document results in `/project/experiments/EXP-005-engine-correctness.md`

**Success Criteria**:
- [ ] 3/3 prompts produce byte-identical output
- [ ] No exceptions during generation
- [ ] Token counts within expected range

**Failure Response**:
- If any prompt fails: Debug cache reconstruction logic
- If > 5 token divergence: Escalate to PM (BLOCKING)

---

#### Task 5.2: EXP-006 - Block Gather Performance Benchmark

**File**: `/project/experiments/scripts/run_exp_006.py`

**Checklist**:
- [ ] Remove NotImplementedError placeholders
- [ ] Implement setup_test_environment()
- [ ] Implement create_synthetic_blocks()
- [ ] Implement benchmark_block_gather()
- [ ] Run benchmark with 100 iterations
- [ ] Test 3 contexts (2K, 4K, 8K)
- [ ] Analyze results (p50, p95, p99)
- [ ] Document results in `/project/experiments/EXP-006-block-gather-performance.md`

**Success Criteria**:
- [ ] p95 < 5ms for 8K context (primary)
- [ ] Standard deviation < 20% of mean
- [ ] Linear scaling with block count

**Conditional Success** (5ms < p95 < 10ms):
- Document in ADR-004 as acceptable trade-off
- One-time cost, not per-step

**Failure Response** (p95 > 10ms):
- Profile mx.concatenate
- Check block count scaling
- Validate MLX lazy evaluation
- Consider alternative strategies

---

#### Task 5.3: Integration Tests

**File**: `/tests/integration/test_batch_engine.py` (new)

**Checklist**:
- [ ] Create integration test file
- [ ] Add @pytest.mark.integration decorator
- [ ] Test 1: Single agent generation (fresh, no cache)
- [ ] Test 2: Single agent with cache resume
- [ ] Test 3: Multi-agent (3 agents, variable lengths)
- [ ] Test 4: Block allocation and free (no leaks)
- [ ] Test 5: Error handling (pool exhausted)
- [ ] Test 6: Empty prompt rejection

**Fixtures**:
- [ ] Load SmolLM2-135M (fast model)
- [ ] Create BlockPool (100 blocks)
- [ ] Create ModelCacheSpec
- [ ] Cleanup after tests (free all blocks)

**Success Criteria**:
- [ ] All 6 integration tests pass
- [ ] No memory leaks detected
- [ ] Performance within 20% of reference

---

### Phase 6: Failure Modes & Documentation (Day 10)

**Duration**: 4 hours
**Owner**: SE, QE

#### Task 6.1: Error Handling & Edge Cases

**Checklist**:
- [ ] Test pool exhaustion during submit()
- [ ] Test invalid inputs (empty prompt, negative max_tokens)
- [ ] Test model not loaded error
- [ ] Test cache reconstruction failure
- [ ] Test batch generator errors
- [ ] Add error recovery logic where needed
- [ ] Document all error paths

**Error Paths**:
- [ ] PoolExhaustedError → graceful failure, clear message
- [ ] InvalidRequestError → validation at submit()
- [ ] ModelNotFoundError → check on init
- [ ] CacheReconstructionError → log and raise

---

#### Task 6.2: Update Documentation

**Checklist**:
- [ ] Update `/project/sprints/sprint_2_block_pool_batch_engine.md` with implementation results
- [ ] Document EXP-005 results (PASS/FAIL)
- [ ] Document EXP-006 results (performance metrics)
- [ ] Create ADR-004 if needed (block gather strategy decision)
- [ ] Update architecture documentation with actual implementation
- [ ] Add docstrings to all public methods
- [ ] Add inline comments for complex logic

---

## Dependencies

### External Dependencies (Already Satisfied)

- ✅ mlx==0.30.3
- ✅ mlx-lm==0.30.4
- ✅ Domain layer (BlockPool, ModelCacheSpec, AgentBlocks)
- ✅ Ports (GenerationEnginePort defined)

### Internal Dependencies

**Blocking**:
- ✅ BlockPool implemented (Sprint 1)
- ✅ ModelCacheSpec implemented (Sprint 1)
- ✅ AgentBlocks implemented (Sprint 1)
- ✅ CompletedGeneration value object (Day 2)
- ✅ GenerationEnginePort defined (Day 2)

**Blocked By**:
- None (all dependencies met)

---

## Risk Assessment

### High Risk Items

1. **Cache Reconstruction Performance** (EXP-006)
   - **Risk**: p95 > 5ms could indicate performance issues
   - **Mitigation**: Benchmark early (Day 7), alternative strategies ready
   - **Fallback**: Document as conditional pass if < 10ms

2. **Byte-Identical Output** (EXP-005)
   - **Risk**: Output divergence indicates cache bug
   - **Mitigation**: Step-by-step validation, token-level comparison
   - **Escalation**: PM if > 5 token divergence (BLOCKING)

### Medium Risk Items

3. **MLX API Compatibility**
   - **Risk**: BatchGenerator API changes in mlx-lm 0.30.4
   - **Mitigation**: API reference doc created (Day 2), version pinned
   - **Fallback**: Adapter wrapper if API unstable

4. **Memory Leaks**
   - **Risk**: Blocks not freed after completion
   - **Mitigation**: Track allocations, verify free in tests
   - **Detection**: Pool size monitoring in integration tests

### Low Risk Items

5. **Type Safety**
   - **Risk**: mypy errors with Protocol types
   - **Mitigation**: Already validated in Day 4 review
   - **Confidence**: HIGH

6. **Test Coverage**
   - **Risk**: Coverage drops below 95%
   - **Mitigation**: Write tests alongside implementation
   - **Confidence**: HIGH

---

## Success Metrics

### Quantitative

| Metric | Target | Measured | Status |
|--------|--------|----------|--------|
| Output correctness | 3/3 byte-identical | TBD (EXP-005) | ⏳ |
| Cache reconstruction | p95 < 5ms | TBD (EXP-006) | ⏳ |
| Memory leaks | 0 (pool stable) | TBD (integration) | ⏳ |
| Throughput regression | < 20% | TBD (integration) | ⏳ |
| Test coverage | >95% | TBD (Day 10) | ⏳ |
| Integration tests | 6/6 passing | TBD (Day 9) | ⏳ |

### Qualitative

- [ ] Code readable and maintainable
- [ ] Algorithms match design document
- [ ] Error handling comprehensive
- [ ] Documentation complete
- [ ] Type safety maintained
- [ ] Performance acceptable

---

## Daily Checklist

### Day 6: Core Structure
- [ ] Task 1.1: Create class and __init__()
- [ ] Task 1.2: Implement submit()
- [ ] Unit tests for submit()
- [ ] mypy/ruff clean
- [ ] Daily review and commit

### Day 7: Cache Reconstruction
- [ ] Task 2.1: Implement _reconstruct_cache()
- [ ] Unit tests for reconstruction
- [ ] Run EXP-006 benchmark (preliminary)
- [ ] Daily review and commit

### Day 8: Decode Step & Extraction
- [ ] Task 3.1: Implement step()
- [ ] Task 4.1: Implement _extract_cache()
- [ ] Unit tests for both methods
- [ ] Run EXP-005 (preliminary)
- [ ] Daily review and commit

### Day 9: Integration Testing
- [ ] Task 5.1: EXP-005 validation
- [ ] Task 5.2: EXP-006 benchmark
- [ ] Task 5.3: Integration test suite
- [ ] Analyze results
- [ ] Daily review and commit

### Day 10: Polish & Documentation
- [ ] Task 6.1: Error handling
- [ ] Task 6.2: Documentation
- [ ] Final quality gate check
- [ ] Sprint 2 review
- [ ] Final commit

---

## Helper Resources

### Design Document
- `/project/architecture/blockpool-batch-engine-design.md`
- Contains full algorithms, sequence diagrams, edge cases

### API Reference
- `/project/reference/mlx_lm_api_v0.30.4.md`
- Contains BatchGenerator API, 14 gotchas, 5 examples

### Test Strategy
- `/project/experiments/test_strategy_sprint_2.md`
- Contains EXP-005/006 methodology, validation criteria

### Sprint Plan
- `/project/sprints/sprint_2_block_pool_batch_engine.md`
- Contains daily status, review notes

---

## Final Checklist (Day 10 Exit Criteria)

### Code Quality
- [ ] All methods implemented (no TODOs)
- [ ] Type hints complete
- [ ] Docstrings complete
- [ ] mypy --strict clean
- [ ] ruff clean
- [ ] Test coverage >95%

### Functionality
- [ ] EXP-005: 3/3 prompts pass (byte-identical)
- [ ] EXP-006: p95 < 5ms (or documented if 5-10ms)
- [ ] Integration tests: 6/6 passing
- [ ] No memory leaks detected
- [ ] Error handling complete

### Documentation
- [ ] All experiments documented (results, analysis)
- [ ] Sprint 2 plan updated
- [ ] ADR-004 created (if needed)
- [ ] Code comments added
- [ ] Architecture updated

---

**Status**: ⏳ PENDING (starts Day 6)
**Last Updated**: 2026-01-24 (Day 4)
**Ready for Week 2**: ✅ YES
