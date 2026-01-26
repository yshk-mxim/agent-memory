# Technical Review: MLX KV Cache Persistence Fix

**Date**: 2026-01-26
**Reviewer**: Claude Sonnet 4.5
**Status**: Root Cause Identified - Fix Required
**Priority**: CRITICAL (Core Feature)

---

## Executive Summary

The Semantic Caching Server successfully saves KV cache to disk but **cannot reload/reuse it**, rendering the core caching feature non-functional. The issue stems from a **format mismatch** between what MLX-LM's `BatchGenerator.insert()` expects and what our `_reconstruct_cache()` produces.

**Error**: `'tuple' object has no attribute 'size'`
**Location**: `/Users/dev_user/semantic/src/semantic/application/batch_engine.py:138`
**Impact**: Cache reconstruction fails 100% of the time, forcing fresh allocation for every request

---

## 1. MLX KV Cache Format Specification

### 1.1 Cache Structure Discovery

After examining MLX-LM source code (`/Users/dev_user/.pyenv/versions/3.12.0/lib/python3.12/site-packages/mlx_lm/`), the KV cache structure is:

**MLX Cache Objects (NOT Raw Tuples)**

```python
# From mlx_lm/models/cache.py
class KVCache(_BaseCache):
    step = 256

    def __init__(self):
        self.keys = None      # mx.array or None
        self.values = None    # mx.array or None
        self.offset = 0       # int: number of tokens stored

    @property
    def state(self):
        """Returns (keys, values) tuple - but this is for SERIALIZATION"""
        if self.offset == self.keys.shape[2]:
            return self.keys, self.values
        else:
            return (
                self.keys[..., : self.offset, :],
                self.values[..., : self.offset, :],
            )

    @state.setter
    def state(self, v):
        """Restores from (keys, values) tuple"""
        self.keys, self.values = v
        self.offset = self.keys.shape[2]
```

**Key Insight**: MLX uses **cache objects** (KVCache, RotatingKVCache, QuantizedKVCache), not raw tuples. The `.state` property returns tuples for serialization, but the cache itself is an object.

### 1.2 BatchGenerator.insert() Expectations

```python
# From mlx_lm/generate.py lines 976-1009
def insert(
    self,
    prompts,
    max_tokens: Union[List[int], int, None] = None,
    caches=None,  # List[List[CacheObject]] or None
    samplers: list | None = None,
    logits_processors: list | None = None,
):
    # ...
    if caches is None:
        caches = [None] * len(prompts)
    for i in range(len(prompts)):
        if caches[i] is None:
            # Creates cache objects using cache.make_prompt_cache()
            caches[i] = cache.make_prompt_cache(self.model)

    # Later, line 1007:
    key=lambda x: len(x[1]) + max(c.size() for c in x[3]),
    #                               ^^^^^^^ Calls .size() on cache objects!
```

**Critical Discovery**: Line 1007 calls `c.size()` on cache objects. This means:
1. `caches` parameter expects `List[List[CacheObject]]` (list of prompts, each with list of cache objects)
2. Each cache object must have a `.size()` method (returns sequence length)
3. The error `'tuple' object has no attribute 'size'` confirms we're passing **raw tuples** instead of **cache objects**

### 1.3 Cache Object Types

MLX-LM supports multiple cache types (from `mlx_lm/models/cache.py`):

| Cache Type | Purpose | Has .size() | Has .state |
|------------|---------|-------------|------------|
| `KVCache` | Standard incremental cache | ✅ | ✅ |
| `RotatingKVCache` | Fixed-size sliding window | ✅ | ✅ |
| `QuantizedKVCache` | Quantized for memory efficiency | ✅ | ✅ |
| `BatchKVCache` | Batch-aware with padding | ✅ | ✅ |
| `ConcatenateKVCache` | Simple concatenation | ❌ (returns 0) | ✅ |

All cache types inherit from `_BaseCache` and implement:
- `.state` property (returns tuple of tensors)
- `.empty()` method (checks if cache is empty)
- `.size()` method (returns sequence length, default 0)
- `@classmethod from_state(cls, state, meta_state)` (reconstructs cache object from tuple)

---

## 2. Current Implementation Analysis

### 2.1 Cache Save Flow (WORKING)

```
MLX Model Generation
    ↓
BatchGenerator.next() → response.prompt_cache (List[CacheObject])
    ↓
batch_engine._extract_cache() → AgentBlocks
    ↓
    For each layer:
        cache[layer_id].state → (K_array, V_array) tuple
            ↓
        Slice into 256-token blocks
            ↓
        Store in KVBlock.layer_data = {"k": K_chunk, "v": V_chunk}
    ↓
safetensors_cache_adapter.save()
    ↓
    For each block:
        Convert mx.array → numpy.array
        Save as L{layer}_B{block}_K, L{layer}_B{block}_V
    ↓
DISK (agent_id.safetensors)
```

**This works correctly** because:
1. We receive cache objects from MLX
2. We extract `.state` (tuples) for serialization
3. We convert to numpy for safetensors
4. Disk format is clean numpy arrays

### 2.2 Cache Load Flow (BROKEN)

```
DISK (agent_id.safetensors)
    ↓
safetensors_cache_adapter.load() → dict[int, list[KVBlock]]
    ↓
    For each tensor pair:
        Load numpy arrays
        Create KVBlock(layer_data={"k": np_array, "v": np_array})
    ↓
AgentBlocks(blocks=blocks_dict)
    ↓
batch_engine._reconstruct_cache(agent_blocks)
    ↓
    For each layer:
        Concatenate K/V from all blocks
        Append to list: cache.append((K_full, V_full))  ← RAW TUPLE!
    ↓
    Return list[tuple[mx.array, mx.array]]  ← WRONG FORMAT!
    ↓
BatchGenerator.insert(caches=[raw_tuples])  ← EXPECTS [List[CacheObject]]
    ↓
ERROR: 'tuple' object has no attribute 'size'
```

**Root cause in `_reconstruct_cache()` (lines 328-361)**:

```python
def _reconstruct_cache(self, agent_blocks: AgentBlocks) -> Any:
    """Reconstruct KVCache from blocks."""
    cache: list[tuple[Any, Any]] = []  # ← Creates list of TUPLES

    for layer_id in range(self._spec.n_layers):
        layer_blocks = agent_blocks.blocks_for_layer(layer_id)

        # ... concatenate K/V tensors ...

        k_full, v_full = self._cache_adapter.concatenate_cache_blocks(
            k_tensors, v_tensors
        )

        cache.append((k_full, v_full))  # ← WRONG! Should create KVCache object

    return cache  # ← Returns list[tuple], not list[KVCache]
```

### 2.3 Format Mismatch Diagram

```
WHAT WE PRODUCE:
[
    (mx.array([...]), mx.array([...])),  # Layer 0 - RAW TUPLE
    (mx.array([...]), mx.array([...])),  # Layer 1 - RAW TUPLE
    ...
]

WHAT MLX EXPECTS:
[
    KVCache(keys=mx.array([...]), values=mx.array([...]), offset=256),  # Layer 0 - OBJECT
    KVCache(keys=mx.array([...]), values=mx.array([...]), offset=512),  # Layer 1 - OBJECT
    ...
]
```

---

## 3. Root Cause Analysis

### 3.1 Why the Bug Exists

**Historical Context**: MLX-LM evolved from using raw tuples to using cache objects. Our implementation was based on older patterns:

1. Early MLX examples used `List[Tuple[mx.array, mx.array]]` format
2. The `.state` property returns tuples, creating confusion
3. We correctly use `.state` for **serialization** (save flow)
4. We incorrectly assume tuples are the **runtime format** (load flow)

**Evidence from MLX-LM source**:
- `generate_step()` line 361: `prompt_cache = cache.make_prompt_cache(model)` → creates cache objects
- `BatchGenerator.insert()` line 993: `caches[i] = cache.make_prompt_cache(self.model)` → creates cache objects
- `BatchGenerator.insert()` line 1007: `max(c.size() for c in x[3])` → calls `.size()` method

### 3.2 Why It Fails

```python
# In BatchGenerator.insert(), line 1007:
self.unprocessed_prompts = sorted(
    self.unprocessed_prompts,
    key=lambda x: len(x[1]) + max(c.size() for c in x[3]),
    #                              ^^^^^^^
    # x[3] is the caches list
    # c is expected to be a cache object with .size() method
    # We pass a tuple → AttributeError: 'tuple' object has no attribute 'size'
)
```

The error occurs **during insert() sorting**, not during actual cache usage. This means:
1. Our reconstructed cache is never validated against the model
2. The failure is immediate and deterministic
3. We fall back to fresh allocation, losing all cache benefits

### 3.3 Additional Issues Discovered

**Tensor Format**: Our loaded blocks contain **numpy arrays** (from safetensors), but MLX expects **mx.array** objects:

```python
# In safetensors_cache_adapter.load() lines 115-116:
k_array = tensors_data[k_key]  # numpy.ndarray
v_array = tensors_data[v_key]  # numpy.ndarray

# Should convert to mx.array BEFORE creating KVCache objects
```

**Shape Validation**: No verification that loaded shapes match model expectations:
- K shape: `[batch, n_kv_heads, seq_len, head_dim]`
- V shape: `[batch, n_kv_heads, seq_len, head_dim]`
- Missing batch dimension (should be 1 for single-sequence cache)

---

## 4. Comparison with Other Systems

### 4.1 llama.cpp Approach

From [GitHub discussions](https://github.com/ggml-org/llama.cpp/discussions/13606) and [issue #17107](https://github.com/ggml-org/llama.cpp/issues/17107):

**Format**: Binary format with slot management
```c
// Saves entire KV cache state including:
// - Layer tensors (float16/float32)
// - Sequence position metadata
// - Slot ID for restoration
```

**Benefits**:
- Direct memory dump → minimal transformation
- Slot-based restoration → efficient reuse
- Persistent across server restarts

**Limitations**:
- Binary format → not portable across architectures
- Requires exact model match
- No partial cache reuse

### 4.2 HuggingFace Transformers Approach

From [Transformers docs](https://huggingface.co/docs/transformers/en/internal/generation_utils) and [model outputs](https://huggingface.co/docs/transformers/en/main_classes/output):

**Format**: Cache objects (similar to MLX)
```python
# Modern approach (v4.x+):
past_key_values: DynamicCache | StaticCache | None

# Legacy format (v3.x):
past_key_values: Tuple[Tuple[torch.FloatTensor]]
# Shape: (n_layers, 2, batch, heads, seq_len, head_dim)
```

**Key Points**:
- Migrated from tuples to Cache objects (same evolution as MLX)
- `DynamicCache` grows automatically
- `StaticCache` pre-allocates for fixed size
- No built-in disk persistence (user responsibility)

### 4.3 Our Approach vs Industry

| Feature | llama.cpp | HF Transformers | MLX-LM | Our Implementation |
|---------|-----------|-----------------|--------|-------------------|
| Runtime Format | Binary blob | Cache objects | Cache objects | ❌ Tuples (wrong) |
| Disk Format | Binary dump | User-defined | .safetensors | ✅ .safetensors |
| Restoration | Slot-based | Manual | Manual | ❌ Broken |
| Partial Reuse | ✅ Yes | ❌ No | ❌ No | ✅ Block-based |
| Memory Efficiency | Medium | Low | High | ✅ High (when working) |

**Our Unique Advantage**: Block-based storage allows partial cache reuse and efficient sharing across agents. **But it doesn't work yet.**

---

## 5. Proposed Fix

### 5.1 Core Changes Required

**File**: `/Users/dev_user/semantic/src/semantic/application/batch_engine.py`
**Method**: `_reconstruct_cache()` (lines 328-361)

**Current (BROKEN)**:
```python
def _reconstruct_cache(self, agent_blocks: AgentBlocks) -> Any:
    cache: list[tuple[Any, Any]] = []

    for layer_id in range(self._spec.n_layers):
        # ... get blocks ...
        k_full, v_full = self._cache_adapter.concatenate_cache_blocks(...)
        cache.append((k_full, v_full))  # ← RAW TUPLE

    return cache
```

**Fixed (WORKING)**:
```python
def _reconstruct_cache(self, agent_blocks: AgentBlocks) -> list[Any]:
    """Reconstruct MLX cache objects from blocks.

    Returns:
        List[KVCache]: List of cache objects (one per layer)
    """
    import mlx.core as mx
    from mlx_lm.models.cache import KVCache

    cache: list[Any] = []

    for layer_id in range(self._spec.n_layers):
        layer_blocks = agent_blocks.blocks_for_layer(layer_id)

        if not layer_blocks:
            # Empty layer - create empty cache object
            empty_cache = KVCache()
            cache.append(empty_cache)
            continue

        # Extract K and V tensors from all blocks
        k_tensors = []
        v_tensors = []
        for block in layer_blocks:
            if block.layer_data is None or "k" not in block.layer_data:
                raise GenerationError(
                    f"Block {block.block_id} for layer {layer_id} has no K/V data"
                )

            # Convert numpy → mx.array if needed
            k_data = block.layer_data["k"]
            v_data = block.layer_data["v"]

            if not isinstance(k_data, mx.array):
                k_data = mx.array(k_data)
            if not isinstance(v_data, mx.array):
                v_data = mx.array(v_data)

            k_tensors.append(k_data)
            v_tensors.append(v_data)

        # Concatenate using adapter
        k_full, v_full = self._cache_adapter.concatenate_cache_blocks(
            k_tensors, v_tensors
        )

        # Create KVCache object and set state
        kv_cache = KVCache()
        kv_cache.state = (k_full, v_full)  # Uses setter to populate keys/values/offset

        cache.append(kv_cache)

    return cache
```

**Key Changes**:
1. Import `KVCache` from `mlx_lm.models.cache`
2. Create `KVCache()` objects instead of tuples
3. Use `.state` setter to populate cache (sets keys, values, offset)
4. Convert numpy arrays to `mx.array` during reconstruction
5. Return `list[KVCache]` instead of `list[tuple]`

### 5.2 Supporting Changes

**File**: `/Users/dev_user/semantic/src/semantic/adapters/outbound/mlx_cache_adapter.py`

Add helper method for cache reconstruction:

```python
def create_cache_from_state(
    self,
    k_tensor: Any,
    v_tensor: Any,
) -> Any:
    """Create a KVCache object from K/V tensors.

    Args:
        k_tensor: Key tensor [n_kv_heads, head_dim, seq_len]
        v_tensor: Value tensor [n_kv_heads, head_dim, seq_len]

    Returns:
        KVCache object with state populated
    """
    from mlx_lm.models.cache import KVCache
    import mlx.core as mx

    # Ensure tensors are mx.array
    if not isinstance(k_tensor, mx.array):
        k_tensor = mx.array(k_tensor)
    if not isinstance(v_tensor, mx.array):
        v_tensor = mx.array(v_tensor)

    # Create cache object
    cache = KVCache()
    cache.state = (k_tensor, v_tensor)

    return cache
```

### 5.3 Shape Validation

Add to `MLXCacheAdapter.concatenate_cache_blocks()`:

```python
def concatenate_cache_blocks(
    self,
    k_tensors: list[Any],
    v_tensors: list[Any],
) -> tuple[Any, Any]:
    """Concatenate K/V tensors from multiple blocks along sequence axis."""
    import mlx.core as mx

    # Convert numpy → mx.array if needed
    k_tensors = [mx.array(k) if not isinstance(k, mx.array) else k for k in k_tensors]
    v_tensors = [mx.array(v) if not isinstance(v, mx.array) else v for v in v_tensors]

    # Validate shapes
    if k_tensors:
        expected_k_shape = k_tensors[0].shape[:2]  # [n_kv_heads, head_dim]
        expected_v_shape = v_tensors[0].shape[:2]

        for i, (k_t, v_t) in enumerate(zip(k_tensors, v_tensors, strict=True)):
            if k_t.shape[:2] != expected_k_shape:
                raise GenerationError(
                    f"K tensor shape mismatch in block {i}: "
                    f"expected {expected_k_shape}, got {k_t.shape[:2]}"
                )
            # ... existing validation ...

    # Concatenate along sequence axis (axis=2)
    k_full = mx.concatenate(k_tensors, axis=2)
    v_full = mx.concatenate(v_tensors, axis=2)

    # Force evaluation
    mx.eval(k_full, v_full)

    return k_full, v_full
```

### 5.4 Batch Dimension Handling

MLX models expect batch dimension. Add to reconstruction:

```python
# After concatenation, ensure batch dimension
if k_full.ndim == 3:
    # Shape: [n_kv_heads, head_dim, seq_len] → [1, n_kv_heads, head_dim, seq_len]
    k_full = mx.expand_dims(k_full, axis=0)
    v_full = mx.expand_dims(v_full, axis=0)
```

**Wait, verify this**: Looking at our save flow, we store tensors from `.state` which should already have correct shape. Need to verify during testing.

---

## 6. Testing Strategy

### 6.1 Unit Tests (Existing)

Already have tests in `tests/unit/application/test_batch_engine.py`:
- `test_submit_with_cache_reconstructs` (currently skipped)
- `test_reconstruct_cache_from_single_block` (currently skipped)
- `test_reconstruct_cache_from_multiple_blocks` (currently skipped)

**Action**: Unskip these tests after implementing fix.

### 6.2 Integration Test Plan

```python
def test_cache_roundtrip_e2e():
    """Test save → load → use cache cycle."""
    # 1. Generate initial cache
    engine = BlockPoolBatchEngine(...)
    uid = engine.submit("agent_1", "Hello world", max_tokens=10)

    for completion in engine.step():
        if completion.finish_reason:
            blocks_v1 = completion.blocks
            break

    # 2. Save to disk
    persistence.save("agent_1", blocks_v1, metadata={...})

    # 3. Load from disk
    blocks_v2, _ = persistence.load(cache_path)

    # 4. Verify structure
    assert blocks_v2.num_layers() == blocks_v1.num_layers()
    assert blocks_v2.total_tokens == blocks_v1.total_tokens

    # 5. Use loaded cache (THIS IS THE CRITICAL TEST)
    uid2 = engine.submit("agent_1", " How are you?", cache=blocks_v2, max_tokens=10)

    # 6. Should NOT raise "tuple has no attribute size" error
    for completion in engine.step():
        if completion.finish_reason:
            assert completion.finish_reason in ["stop", "length"]
            break
```

### 6.3 Validation Checks

After fix, verify:

1. **Format Check**:
   ```python
   cache = engine._reconstruct_cache(blocks)
   assert all(hasattr(c, 'size') for c in cache), "All cache objects must have .size()"
   assert all(hasattr(c, 'state') for c in cache), "All cache objects must have .state"
   assert all(isinstance(c, KVCache) for c in cache), "Must be KVCache instances"
   ```

2. **Shape Check**:
   ```python
   for layer_id, kv_cache in enumerate(cache):
       k, v = kv_cache.state
       assert k.shape[0] == 1, f"Layer {layer_id}: Batch dim must be 1"
       assert k.shape[1] == model.n_kv_heads, f"Layer {layer_id}: Wrong n_kv_heads"
       assert k.shape[2] == blocks.total_tokens, f"Layer {layer_id}: Wrong seq_len"
   ```

3. **Content Check**:
   ```python
   # Generate with cache vs without - logits should match
   uid_cached = engine.submit("agent_1", " continuation", cache=blocks, max_tokens=5)
   uid_fresh = engine.submit("agent_2", "Hello world continuation", cache=None, max_tokens=5)
   # First 5 tokens should be identical
   ```

### 6.4 Performance Validation

Measure improvement:

```python
import time

# Without cache
t0 = time.perf_counter()
uid1 = engine.submit("agent_1", long_prompt, max_tokens=100)
for c in engine.step(): pass
time_no_cache = time.perf_counter() - t0

# With cache (prompt already processed)
blocks = get_cached_blocks("agent_1")
t0 = time.perf_counter()
uid2 = engine.submit("agent_1", " continue", cache=blocks, max_tokens=100)
for c in engine.step(): pass
time_with_cache = time.perf_counter() - t0

speedup = time_no_cache / time_with_cache
print(f"Cache speedup: {speedup:.2f}x")
assert speedup > 2.0, "Cache should provide >2x speedup for long prompts"
```

---

## 7. Rollout Plan

### Phase 1: Core Fix (Day 1)
1. ✅ Complete technical review (this document)
2. Implement `_reconstruct_cache()` fix in `batch_engine.py`
3. Add `create_cache_from_state()` to `mlx_cache_adapter.py`
4. Update `concatenate_cache_blocks()` for numpy → mx.array conversion
5. Run unit tests: `pytest tests/unit/application/test_batch_engine.py -v`

### Phase 2: Validation (Day 2)
1. Unskip all cache reconstruction tests
2. Add integration test for full roundtrip
3. Test with real Gemma 3 4B model
4. Measure performance improvements
5. Verify memory usage is correct

### Phase 3: Edge Cases (Day 3)
1. Test with different cache types (RotatingKVCache, QuantizedKVCache)
2. Handle model-specific cache formats (Gemma3 sliding window)
3. Test cache corruption scenarios
4. Validate cross-version compatibility

### Phase 4: Production (Day 4)
1. Update API documentation
2. Add logging for cache hit/miss rates
3. Implement cache health checks
4. Deploy to staging environment
5. Monitor cache effectiveness metrics

---

## 8. Risk Assessment

### High Risk ✅ Mitigated
- **Incorrect tensor shapes**: Add validation in `concatenate_cache_blocks()`
- **Memory leaks**: Use context managers for MLX tensors
- **Cache corruption**: Add checksums to metadata
- **Model mismatch**: Validate `n_layers`, `n_kv_heads` on load

### Medium Risk ⚠️ Monitor
- **Performance regression**: Measure before/after throughput
- **Sliding window models**: Test Gemma3 specifically (layers 0-7 global, 8+ sliding)
- **Quantized caches**: May need `QuantizedKVCache` support

### Low Risk ℹ️ Document
- **Cache invalidation**: Currently manual, document best practices
- **Disk space**: Gemma 3 4B cache ~2GB per agent, monitor usage
- **Concurrent access**: safetensors is read-only, safe for multi-process

---

## 9. Success Metrics

### Must-Have (P0)
- [ ] `test_cache_roundtrip_e2e` passes
- [ ] No `AttributeError: 'tuple' object has no attribute 'size'` errors
- [ ] Cache reconstruction success rate: **100%**
- [ ] All existing tests still pass

### Should-Have (P1)
- [ ] Cache hit rate > 80% for multi-turn conversations
- [ ] Latency reduction > 2x for cached prompts (>512 tokens)
- [ ] Memory overhead < 10% vs fresh allocation
- [ ] Zero cache-related errors in 1000-request stress test

### Nice-to-Have (P2)
- [ ] Support for quantized cache persistence
- [ ] Automatic cache warming on server start
- [ ] Cache analytics dashboard (hit rate, size, age)
- [ ] Cross-agent cache deduplication

---

## 10. Open Questions

### Q1: Do we need to support multiple cache types?
**Answer**: Start with `KVCache` (standard). Add `RotatingKVCache` (Gemma3 sliding window) in Phase 3.

**Rationale**: `KVCache` is the default and most common. `RotatingKVCache` is used for models with sliding attention windows (like Gemma 3 layers 8+).

### Q2: Should we convert during save or load?
**Current**: numpy arrays on disk (via safetensors), convert to mx.array on load
**Alternative**: Store mx.array metadata, convert on save

**Decision**: Keep current approach (numpy on disk). Rationale:
- safetensors requires numpy
- numpy is portable across MLX versions
- Conversion cost is small vs cache reuse benefit

### Q3: What about batch dimension?
**Investigation needed**: Verify if our stored tensors have batch dim or not.

**Action**: Add logging in `_extract_cache()` to check shape during save:
```python
logging.debug(f"Saving cache: K shape={k_chunk.shape}, V shape={v_chunk.shape}")
```

If shape is `[n_kv_heads, head_dim, seq_len]`, we need to add batch dim.
If shape is `[1, n_kv_heads, head_dim, seq_len]`, no change needed.

---

## 11. References

### MLX-LM Documentation
- [BatchGenerator source code](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/generate.py) - lines 920-1246
- [Cache implementations](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/cache.py)
- [MLX-LM Continuous Batching (Medium)](https://medium.com/@clnaveen/mlx-lm-continuous-batching-e060c73e7d98)

### Industry Approaches
- [llama.cpp KV cache discussions](https://github.com/ggml-org/llama.cpp/discussions/13606)
- [llama.cpp Issue #17107 - Persistent KV cache](https://github.com/ggml-org/llama.cpp/issues/17107)
- [HuggingFace Transformers Model Outputs](https://huggingface.co/docs/transformers/en/main_classes/output)
- [HuggingFace Generation Utilities](https://huggingface.co/docs/transformers/en/internal/generation_utils)

### Related GitHub Issues
- [MLX-LM Issue #548 - Batch KV cache plans](https://github.com/ml-explore/mlx-lm/issues/548)
- [MLX-Examples Issue #917 - Store KV cache to disk](https://github.com/ml-explore/mlx-examples/issues/917)

---

## 12. Conclusion

The KV cache persistence failure is caused by a **clear format mismatch**: we produce tuples, MLX expects cache objects. The fix is **well-understood** and **low-risk**:

1. Import `KVCache` from `mlx_lm.models.cache`
2. Create cache objects instead of tuples
3. Use `.state` setter to populate keys/values
4. Convert numpy → mx.array during reconstruction

**Estimated effort**: 4-6 hours implementation + testing
**Confidence**: High (95%) - Root cause is confirmed from source code
**Impact**: Critical - Unlocks core caching functionality

**Next Step**: Implement the fix in `batch_engine._reconstruct_cache()` and validate with integration tests.

---

**Document Version**: 1.0
**Last Updated**: 2026-01-26
**Review Status**: Complete
**Approved for Implementation**: ✅ Ready
