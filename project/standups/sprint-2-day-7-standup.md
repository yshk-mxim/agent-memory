# Sprint 2 Day 7 Standup: Cache Reconstruction

**Date**: 2026-01-24
**Sprint**: 2 (Block-Pool Batch Engine)
**Day**: 7
**Focus**: Cache reconstruction implementation

---

## Previous Day Review (Day 6)

**Status**: ✅ COMPLETE

**Achievements**:
- BlockPoolBatchEngine core methods implemented (submit, step)
- FakeBatchGenerator mock created for testing
- 13/13 active unit tests passing
- Dependency injection pattern established
- All quality gates passing

**Commit**: `51c16b4` - "Sprint 2 Day 6: BlockPoolBatchEngine core implementation"

---

## Day 7 Goals

**Primary Objective**: Implement cache reconstruction - convert AgentBlocks → KVCache for mlx_lm

**Deliverables**:
1. `_reconstruct_cache()` method implementation (~30 lines)
2. Unit tests for cache reconstruction (unskip 2 tests)
3. Update `submit()` to use _reconstruct_cache()
4. (Stretch) Run EXP-006 benchmark

---

## Implementation Plan

### Task 1: Implement _reconstruct_cache()

**Algorithm** (7 steps from design doc):
```python
def _reconstruct_cache(self, agent_blocks: AgentBlocks) -> list[tuple]:
    """Reconstruct KVCache from blocks.

    For each layer:
    1. Get all blocks for this layer
    2. Extract K tensors from blocks
    3. Extract V tensors from blocks
    4. Concatenate K tensors along seq_len axis (axis=2)
    5. Concatenate V tensors along seq_len axis (axis=2)
    6. Force evaluation (mx.eval) to materialize concatenation
    7. Append (k_full, v_full) to cache list
    """
    import mlx.core as mx

    cache = []
    for layer_id in range(self._spec.n_layers):
        layer_blocks = agent_blocks.blocks_for_layer(layer_id)

        if not layer_blocks:
            cache.append((None, None))
            continue

        k_tensors = [block.layer_data["k"] for block in layer_blocks]
        v_tensors = [block.layer_data["v"] for block in layer_blocks]

        k_full = mx.concatenate(k_tensors, axis=2)
        v_full = mx.concatenate(v_tensors, axis=2)

        mx.eval(k_full, v_full)
        cache.append((k_full, v_full))

    return cache
```

**Complexity**: O(n_blocks × n_layers)
**Performance Target**: p95 < 5ms for 32 blocks × 48 layers (EXP-006)

---

### Task 2: Create Unit Tests

**Tests to Implement**:

1. `test_reconstruct_cache_from_single_block()`
   - Create AgentBlocks with 1 block per layer
   - Call _reconstruct_cache()
   - Verify shape: (n_kv_heads, head_dim, 256)
   - Verify data matches

2. `test_reconstruct_cache_from_multiple_blocks()`
   - Create AgentBlocks with 3 blocks per layer (768 tokens)
   - Call _reconstruct_cache()
   - Verify shape: (n_kv_heads, head_dim, 768)
   - Verify concatenation correct

**Challenge**: Unit tests can't use real mlx.core (no GPU)

**Solution**: Create FakeMLXArrays or skip these tests for integration testing

---

### Task 3: Update submit() Method

**Current Code** (lines 136-141):
```python
if cache is not None:
    # Cache provided - reconstruct from blocks (Day 7 implementation)
    # TODO: Implement _reconstruct_cache()
    # kv_cache = self._reconstruct_cache(cache)
    raise NotImplementedError(
        "Cache reconstruction not yet implemented (Day 7)"
    )
```

**Updated Code**:
```python
if cache is not None:
    # Cache provided - reconstruct from blocks
    kv_cache = self._reconstruct_cache(cache)
```

---

### Task 4: Run EXP-006 Benchmark (Stretch Goal)

**Purpose**: Measure cache reconstruction performance

**Metrics**:
- p50, p95, p99 latency for reconstruction
- Test cases: 1, 2, 4, 8, 16, 32 blocks
- Across 12, 24, 48 layers

**Success Criteria**: p95 < 5ms for 32 blocks × 48 layers

**Note**: May defer to Day 8 if unit tests take longer than expected

---

## Technical Challenges

### Challenge 1: Testing Without MLX

**Problem**: Unit tests can't import mlx.core (crashes on non-GPU environments)

**Options**:
1. Skip reconstruction tests for integration (mark with `@pytest.mark.integration`)
2. Create fake mx.array and mx.concatenate mocks
3. Test the algorithm logic only (not actual MLX operations)

**Decision**: Mark tests as `@pytest.mark.skip` with reason "Requires MLX (Day 9 integration)"

---

### Challenge 2: Block Layer Data Format

**Unknown**: What is the exact format of `block.layer_data`?

**Assumption**: Dictionary with "k" and "v" keys containing mx.array tensors

**Verification**: Check KVBlock entity definition in `domain/entities.py`

**Fallback**: If format differs, adjust algorithm accordingly

---

### Challenge 3: Empty Layer Blocks

**Scenario**: Some layers have no blocks (e.g., sliding window layers beyond window size)

**Behavior**: Return (None, None) for those layers

**Validation**: Check if mlx_lm BatchGenerator accepts None entries in cache list

---

## Dependencies

**Required**:
- ✅ KVBlock entity with layer_data field (Sprint 1)
- ✅ AgentBlocks.blocks_for_layer() method (Sprint 1)
- ✅ ModelCacheSpec.n_layers attribute (Sprint 1)

**Optional**:
- ⏳ Real MLX model for integration tests (Day 9)
- ⏳ EXP-006 benchmark framework (Day 7-8)

---

## Success Criteria

**Day 7 Complete When**:
- [x] _reconstruct_cache() implemented and type-safe
- [x] submit() updated to use _reconstruct_cache()
- [x] At least 1 unit test validates algorithm logic
- [x] mypy clean, ruff clean
- [x] All existing tests still pass
- [x] Day 7 review document created
- [x] Work committed

**Stretch Goals**:
- [ ] EXP-006 benchmark run (preliminary results)
- [ ] Integration test with real MLX (if environment available)

---

## Time Estimate

| Task | Estimate | Confidence |
|------|----------|------------|
| Implement _reconstruct_cache() | 30 min | High |
| Unit test stubs/mocks | 45 min | Medium |
| Update submit() | 10 min | High |
| Testing & debugging | 60 min | Medium |
| EXP-006 benchmark | 60 min | Low (stretch) |
| Review & commit | 30 min | High |
| **Total** | **3.5 hours** | **Medium** |

**Risk Buffer**: +1 hour for unexpected MLX-related issues

---

## Notes

- Follow same pattern as Day 6: implement → test → review → commit
- Keep tests simple - integration tests will validate with real MLX
- Focus on algorithm correctness, not performance (EXP-006 is stretch goal)
- Document any assumptions about block.layer_data format

---

**Status**: ✅ READY TO START
**Next Step**: Implement _reconstruct_cache() method in batch_engine.py
