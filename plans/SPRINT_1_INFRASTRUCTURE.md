# Sprint 1: Core Infrastructure (Week 1)

**Duration**: 5 days
**Goal**: Implement MLX cache extraction and persistence layer
**Status**: Ready to start

---

## Objectives

- [ ] Expose KV cache from MLX's internal generation
- [ ] Implement cache serialization/deserialization
- [ ] Test cache save/load roundtrip
- [ ] Validate cache reuse reduces prefill time

---

## Daily Breakdown

### Monday: MLX Cache Extraction - Part 1

**Morning (3h)**:
- [ ] Study MLX's internal cache structure
  - Review mlx_lm source code
  - Understand cache format: List[(key, value)] per layer
  - Identify where cache is created in generation loop
- [ ] Create `src/mlx_cache_extractor.py` skeleton
  - Class structure
  - Method signatures
  - Type hints

**Afternoon (2h)**:
- [ ] Implement basic cache extraction
  - Wrap mlx_lm.generate() to capture cache
  - Return tuple: (output_text, kv_cache)
  - Test on simple prompt

**Deliverable**: `src/mlx_cache_extractor.py` (basic structure)

---

### Tuesday: MLX Cache Extraction - Part 2

**Morning (3h)**:
- [ ] Implement cache metadata extraction
  ```python
  def get_cache_size(cache) -> int:
      """Return token count in cache"""

  def get_cache_memory(cache) -> int:
      """Return memory usage in bytes"""

  def validate_cache_structure(cache) -> bool:
      """Verify cache has expected structure"""
  ```

**Afternoon (2h)**:
- [ ] Write unit tests for cache extraction
  - Test with short prompt (100 tokens)
  - Test with long prompt (1000 tokens)
  - Verify cache size matches input
- [ ] Test multi-turn cache accumulation

**Deliverable**: `tests/test_cache_extractor.py`, all tests passing

---

### Wednesday: Cache Persistence - Part 1

**Morning (3h)**:
- [ ] Create `src/cache_persistence.py`
- [ ] Implement cache serialization
  ```python
  def save_agent_cache(agent_id: str, cache, metadata: dict):
      """Save KV cache to disk using safetensors"""
      # Path: ~/.agent_caches/tech_specialist.safetensors
      # Metadata: agent_id, cache_size, timestamp, model, hash
  ```
- [ ] Research safetensors format for MLX arrays

**Afternoon (2h)**:
- [ ] Implement cache deserialization
  ```python
  def load_agent_cache(agent_id: str):
      """Load KV cache from disk"""
      # Returns: (cache, metadata)
  ```
- [ ] Test save/load roundtrip on small cache

**Deliverable**: `src/cache_persistence.py` (save/load working)

---

### Thursday: Cache Persistence - Part 2

**Morning (3h)**:
- [ ] Implement cache management utilities
  ```python
  def list_cached_agents() -> List[str]:
      """List all saved agent IDs"""

  def delete_agent_cache(agent_id: str):
      """Remove cache file"""

  def get_cache_disk_usage() -> dict:
      """Report disk space used by caches"""
  ```

**Afternoon (2h)**:
- [ ] Write unit tests for persistence
  - Test save/load with various cache sizes
  - Test metadata preservation
  - Test cache corruption handling
  - Test file not found scenarios

**Deliverable**: `tests/test_cache_persistence.py`, all tests passing

---

### Friday: Integration Testing & Validation

**Morning (3h)**:
- [ ] End-to-end integration test
  ```python
  # Test flow:
  # 1. Generate with MLXCacheExtractor
  # 2. Save cache with CachePersistence
  # 3. Load cache
  # 4. Generate again with loaded cache
  # 5. Verify: No re-prefill, faster generation
  ```

**Afternoon (2h)**:
- [ ] Performance benchmarking
  - Measure: Time to save cache
  - Measure: Time to load cache
  - Measure: Generation time with vs without cache
  - Document speedup achieved

**Evening (1h)**:
- [ ] Code review and cleanup
  - Add docstrings
  - Add type hints
  - Clean up debug code
  - Update README

**Deliverable**: Working cache extraction + persistence, benchmarks documented

---

## Success Criteria

- ✅ MLXCacheExtractor exposes KV cache from mlx_lm.generate()
- ✅ CachePersistence saves/loads cache to/from disk
- ✅ Cache roundtrip works (save → load → reuse)
- ✅ Cache reuse demonstrably faster than re-prefill
- ✅ All unit tests passing
- ✅ Code documented with docstrings

---

## Technical Details

### MLX Cache Structure

```python
# Cache format (from mlx_lm internals)
cache_history: List[Tuple[mx.array, mx.array]]
# Each tuple: (keys, values) for one layer
# Shape: [batch_size, num_heads, seq_len, head_dim]
# seq_len dimension indicates token count

# Example for Gemma 3 12B:
# - 26 layers
# - cache_history length = 26
# - cache_history[0] = (keys_layer0, values_layer0)
```

### Serialization Strategy

```python
# Use safetensors for secure, efficient serialization
from safetensors import safe_save, safe_load
import mlx.core as mx

def serialize_cache(cache_history):
    tensors = {}
    for layer_idx, (keys, values) in enumerate(cache_history):
        tensors[f"layer_{layer_idx}_keys"] = keys
        tensors[f"layer_{layer_idx}_values"] = values
    return tensors

def deserialize_cache(tensors):
    cache_history = []
    layer_idx = 0
    while f"layer_{layer_idx}_keys" in tensors:
        keys = tensors[f"layer_{layer_idx}_keys"]
        values = tensors[f"layer_{layer_idx}_values"]
        cache_history.append((keys, values))
        layer_idx += 1
    return cache_history
```

### File Structure

```
~/.agent_caches/
├── tech_specialist_001.safetensors (cache file)
├── tech_specialist_001.json (metadata)
├── biz_analyst_001.safetensors
├── biz_analyst_001.json
└── coordinator_001.safetensors
```

---

## Risks & Mitigation

**Risk**: MLX cache structure unclear or hard to extract
- **Mitigation**: Study mlx_lm source code first (Monday AM)
- **Fallback**: Contact MLX community for guidance

**Risk**: Safetensors doesn't work well with MLX arrays
- **Mitigation**: Test early (Wednesday)
- **Fallback**: Use pickle (less secure but functional)

**Risk**: Cache corruption on disk
- **Mitigation**: Add validation checks on load
- **Fallback**: Gracefully handle corruption (regenerate cache)

---

## Deliverables

- [ ] `src/mlx_cache_extractor.py` - KV cache extraction (~200 lines)
- [ ] `src/cache_persistence.py` - Save/load to disk (~250 lines)
- [ ] `tests/test_cache_extractor.py` - Unit tests (~150 lines)
- [ ] `tests/test_cache_persistence.py` - Unit tests (~200 lines)
- [ ] Performance benchmarks documented

---

## Dependencies

**External**:
- mlx, mlx-lm (already installed)
- safetensors (`pip install safetensors`)

**Internal**:
- `src/mlx_utils.py` (created during refactoring)

---

## Next Sprint

**Sprint 2**: Agent Manager Implementation (Week 2)
- Multi-agent orchestration
- LRU eviction policy
- Memory management

---

**Created**: January 23, 2026
**Status**: Ready to start
**Estimated Effort**: 25-30 hours over 5 days
