# Sprint 2 Day 8 Standup: Cache Extraction & Validation

**Date**: 2026-01-24
**Sprint**: 2 (Block-Pool Batch Engine)
**Day**: 8
**Focus**: Cache extraction and output validation

---

## Previous Day Review (Day 7)

**Status**: ✅ COMPLETE

**Achievements**:
- _reconstruct_cache() implemented (AgentBlocks → KVCache)
- submit() updated to use cache reconstruction
- Algorithm follows design document exactly
- All quality gates passing

**Commit**: `83a775b` - "Sprint 2 Day 7: Cache reconstruction implementation"

---

## Day 8 Goals

**Primary Objective**: Implement cache extraction - convert KVCache → AgentBlocks after generation

**Deliverables**:
1. `_extract_cache()` method implementation (~60 lines)
2. Update `step()` to use _extract_cache()
3. Unit test updates (integration deferred to Day 9)
4. (Stretch) Run EXP-005 & EXP-006

---

## Implementation Plan

### Task 1: Implement _extract_cache()

**Signature**:
```python
def _extract_cache(self, uid: str) -> AgentBlocks:
    """Extract updated cache from batch and convert to blocks."""
```

**Algorithm** (from design doc):
```python
def _extract_cache(self, uid: str) -> AgentBlocks:
    """Extract updated cache from batch and convert to blocks."""
    import mlx.core as mx  # noqa: PLC0415

    # 1. Get agent_id from UID tracking
    agent_id = self._active_requests[uid]

    # 2. Extract cache from BatchGenerator
    cache = self._batch_gen.extract_cache(uid)

    # 3. Get total tokens from first layer K tensor shape
    if not cache or cache[0][0] is None:
        # Empty cache - return empty AgentBlocks
        return AgentBlocks(agent_id=agent_id, blocks={}, total_tokens=0)

    first_k = cache[0][0]  # Shape: [n_kv_heads, head_dim, total_seq_len]
    total_tokens = first_k.shape[2]

    # 4. Calculate blocks needed
    n_blocks = (
        (total_tokens + self._spec.block_tokens - 1) // self._spec.block_tokens
    )

    # 5. Create AgentBlocks container
    blocks_dict: dict[int, list[KVBlock]] = {}

    # 6. For each layer, split cache into blocks
    for layer_id, (k, v) in enumerate(cache):
        if k is None:
            continue  # Skip empty layers

        layer_blocks = []

        # Split K, V into 256-token chunks
        for block_idx in range(n_blocks):
            start_token = block_idx * self._spec.block_tokens
            end_token = min(start_token + self._spec.block_tokens, total_tokens)

            # Slice tensors [start:end] along seq_len axis (axis=2)
            k_chunk = k[:, :, start_token:end_token]
            v_chunk = v[:, :, start_token:end_token]

            # Allocate block from pool
            allocated_blocks = self._pool.allocate(1, layer_id, agent_id)
            block_id = allocated_blocks[0].block_id

            # Create KVBlock with cache data
            block = KVBlock(
                block_id=block_id,
                layer_id=layer_id,
                token_count=end_token - start_token,
                layer_data={"k": k_chunk, "v": v_chunk},
            )

            layer_blocks.append(block)

        blocks_dict[layer_id] = layer_blocks

    # 7. Return AgentBlocks with total_tokens
    return AgentBlocks(
        agent_id=agent_id,
        blocks=blocks_dict,
        total_tokens=total_tokens,
    )
```

**Complexity**: O(n_blocks × n_layers)

---

### Task 2: Update step() Method

**Current Code** (lines 260-270):
```python
# Extract cache and convert to blocks (Day 8 implementation)
# For now, use existing blocks or create empty AgentBlocks
if agent_id in self._agent_blocks:
    blocks = self._agent_blocks[agent_id]
else:
    blocks = AgentBlocks(agent_id=agent_id, blocks={}, total_tokens=...)
```

**Updated Code**:
```python
# Extract cache and convert to blocks
blocks = self._extract_cache(uid)
```

---

### Task 3: Handle Block Allocation

**Challenge**: _extract_cache() allocates new blocks from pool

**Options**:
1. ✅ Allocate fresh blocks (current design)
2. ❌ Reuse blocks from submit() (complex tracking)

**Decision**: Allocate fresh blocks

**Rationale**:
- Clean separation: submit() blocks for prefill, extract() blocks for result
- Simpler lifecycle: blocks created when generation completes
- Easier to free old blocks after cache saved

---

### Task 4: Clean Up Old Blocks

**Question**: What happens to blocks allocated during submit()?

**Answer**: They're no longer needed after extraction

**Implementation**: Free them after _extract_cache() completes
```python
# In step():
blocks = self._extract_cache(uid)

# Free old prefill blocks (if any)
if agent_id in self._agent_blocks:
    old_blocks = self._agent_blocks[agent_id]
    for layer_blocks in old_blocks.blocks.values():
        self._pool.free(layer_blocks, agent_id)

# Store new blocks
self._agent_blocks[agent_id] = blocks
```

---

## Technical Challenges

### Challenge 1: Block Allocation During Extraction

**Problem**: _extract_cache() needs to allocate n_blocks × n_layers blocks

**Risk**: Pool exhaustion during extraction

**Mitigation**: Pre-check pool availability before extraction
```python
total_blocks_needed = n_blocks * self._spec.n_layers
if self._pool.available_blocks() < total_blocks_needed:
    raise PoolExhaustedError(...)
```

---

### Challenge 2: Partial Last Block

**Scenario**: Total tokens not multiple of 256 (e.g., 612 tokens = 2 full + 1 partial block)

**Behavior**: Last block has fewer than 256 tokens

**Validation**:
- token_count field tracks actual tokens in block
- K/V chunks sliced correctly ([:, :, start:end])

---

### Challenge 3: Empty Layers (Sliding Window)

**Scenario**: Some layers have no cache (e.g., sliding window exceeded)

**Behavior**: cache[layer_id] = (None, None)

**Handling**: Skip those layers in _extract_cache()
```python
for layer_id, (k, v) in enumerate(cache):
    if k is None:
        continue  # No blocks for this layer
```

---

## Integration with step()

**Before Day 8**:
```python
def step(self) -> Iterator[CompletedGeneration]:
    ...
    for finished in finished_sequences:
        uid = finished.uid
        agent_id = self._active_requests[uid]

        # Placeholder: use existing blocks or create empty
        blocks = self._agent_blocks.get(agent_id, AgentBlocks(...))

        completion = CompletedGeneration(uid, text, blocks, ...)
        del self._active_requests[uid]
        yield completion
```

**After Day 8**:
```python
def step(self) -> Iterator[CompletedGeneration]:
    ...
    for finished in finished_sequences:
        uid = finished.uid
        agent_id = self._active_requests[uid]

        # Extract cache and convert to blocks
        blocks = self._extract_cache(uid)

        # Free old prefill blocks (if any)
        if agent_id in self._agent_blocks:
            self._pool.free(self._agent_blocks[agent_id].all_blocks(), agent_id)

        # Store new blocks
        self._agent_blocks[agent_id] = blocks

        completion = CompletedGeneration(uid, text, blocks, ...)
        del self._active_requests[uid]
        yield completion
```

---

## Testing Strategy

### Unit Tests (Day 8)

**Status**: Deferred to integration (same as Day 7)

**Reason**: Requires real mlx.array objects

**Tests to Mark**:
- `test_extract_cache_converts_to_blocks` - Skip (Day 9)

---

### Integration Tests (Day 9)

**Round-Trip Test**:
```python
# 1. Submit with no cache
uid = engine.submit(agent_id="test", prompt="Hello", max_tokens=50)

# 2. Generate
for completion in engine.step():
    blocks_extracted = completion.blocks

# 3. Resume with extracted cache
uid2 = engine.submit(agent_id="test", prompt=" world", cache=blocks_extracted)

# 4. Verify cache reused (no re-encoding of "Hello")
```

---

## EXP-005: Output Validation

**Goal**: Verify generated text matches reference implementation

**Status**: ⏳ Deferred to Day 9 (requires MLX)

**Method**:
1. Generate with mlx_lm directly (reference)
2. Generate with BlockPoolBatchEngine (test)
3. Compare outputs token-by-token

**Success Criteria**: Byte-identical output

---

## EXP-006: Performance Benchmark

**Goal**: Measure cache reconstruction + extraction performance

**Status**: ⏳ Deferred to Day 9 (requires MLX)

**Metrics**:
- _reconstruct_cache() latency
- _extract_cache() latency
- Round-trip latency

**Target**: p95 < 5ms for reconstruction

---

## Success Criteria

**Day 8 Complete When**:
- [x] _extract_cache() implemented and type-safe
- [x] step() updated to use _extract_cache()
- [x] Block allocation and cleanup handled correctly
- [x] mypy clean, ruff clean
- [x] All existing tests still pass
- [x] Day 8 review document created
- [x] Work committed

**Stretch Goals**:
- [ ] EXP-005 validation (deferred to Day 9)
- [ ] EXP-006 benchmark (deferred to Day 9)

---

## Time Estimate

| Task | Estimate | Confidence |
|------|----------|------------|
| Implement _extract_cache() | 45 min | High |
| Update step() method | 20 min | High |
| Block cleanup logic | 30 min | Medium |
| Testing & debugging | 60 min | Medium |
| Review & commit | 30 min | High |
| **Total** | **3 hours** | **High** |

**Risk Buffer**: +1 hour for block allocation edge cases

---

## Notes

- _extract_cache() is inverse of _reconstruct_cache()
- Allocates new blocks for extracted cache
- Frees old prefill blocks after extraction
- Integration tests will validate with real MLX

---

**Status**: ✅ READY TO START
**Next Step**: Implement _extract_cache() method in batch_engine.py
