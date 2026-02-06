# Warm Cache Testing Guide

## Overview

Warm cache testing validates that KV caches can be persisted to disk and reloaded, providing significant speedup over cold start. This document explains the correct testing methodology and safeguards to prevent regressions.

## The Bug (Fixed in commit 5d701b7)

### What Was Broken

The streaming benchmark's "warm" test was using the **same session_id** for both prime and measure requests:

```python
# BROKEN CODE (before fix)
async def run_streaming_warm(...):
    sid = f"stream_warm_{context_tokens}_{run_id}"

    # Prime (cold hit, populates cache)
    await prime_client.send_and_measure(body, session_id=sid)
    await asyncio.sleep(0.5)

    # Measure (WRONG: hits HOT cache, not WARM!)
    body["stream"] = True
    r = await measure_client.send_and_measure(body, session_id=sid)
```

**Problem**: Both requests use same `session_id`, so:
1. Prime creates cache in hot tier (memory) with `dirty=True`
2. 0.5s sleep insufficient for write-behind flush
3. Measure uses SAME session_id → hits HOT cache (still in memory)
4. Never tests disk reload!

**Result**: Warm TTFT ≈ Cold TTFT (no speedup), indicating broken test.

### Root Cause Analysis

The cache store uses a 3-tier architecture:

| Tier | Location | Access Speed | Persistence |
|------|----------|--------------|-------------|
| Hot | Memory | O(1), ~1ms | Volatile |
| Warm | Disk | O(1), ~50-100ms | Persistent |
| Cold | N/A | O(n), seconds | Must regenerate |

The benchmark was testing **hot→hot** when it should test **cold→warm→hot**.

## The Fix

### 1. Evict-Only Mode

Added `keep_disk` parameter to `AgentCacheStore.delete()`:

```python
def delete(self, agent_id: str, keep_disk: bool = False) -> bool:
    """Delete agent cache from all tiers.

    Args:
        keep_disk: If True, flush to disk and keep file for reload.
                  If False, fully delete including disk file.
    """
    with self._lock:
        # Flush dirty cache to disk before eviction
        if agent_id in self._hot_cache:
            entry = self._hot_cache[agent_id]
            if entry.dirty and entry.blocks is not None:
                self._save_to_disk(agent_id)

        # Remove from hot tier
        if agent_id in self._hot_cache:
            del self._hot_cache[agent_id]

        # Keep or remove disk file
        if not keep_disk and agent_id in self._warm_cache:
            cache_path = self._warm_cache[agent_id]
            cache_path.unlink()
            del self._warm_cache[agent_id]
```

### 2. API Support

Added `evict_only` query parameter to DELETE endpoint:

```python
@router.delete("/{agent_id}")
async def delete_agent(
    agent_id: str,
    evict_only: bool = False,  # NEW: Keep disk file
):
    """Delete or evict agent.

    - evict_only=false: Full delete (default)
    - evict_only=true: Evict to disk, keep for reload
    """
    cache_store.delete(agent_id, keep_disk=evict_only)
```

### 3. Correct Warm Test Pattern

```python
# CORRECT CODE (after fix)
async def run_streaming_warm(...):
    sid = f"stream_warm_{context_tokens}_{run_id}"

    # 1. Prime: Create cache in hot tier
    await prime_client.send_and_measure(body, session_id=sid)

    # 2. Evict: Flush to disk, remove from hot tier
    await _delete_agent(base_url, f"oai_{sid}", evict_only=True)
    await asyncio.sleep(1.0)  # Allow disk write to complete

    # 3. Measure: Reload from disk (warm tier)
    body["stream"] = True
    r = await measure_client.send_and_measure(body, session_id=sid)

    # 4. Cleanup
    await _delete_agent(base_url, f"oai_{sid}")
```

## Safeguards Against Regression

### 1. Integration Tests

`tests/integration/test_warm_cache_reload.py`:

```python
def test_warm_reload_after_evict(cache_store):
    """Test cache reload from disk after eviction."""
    # 1. Prime: save to hot tier
    store.save("agent_3", blocks)
    assert "agent_3" in store._hot_cache

    # 2. Evict: flush to disk, remove from hot
    store.delete("agent_3", keep_disk=True)
    assert "agent_3" not in store._hot_cache
    assert "agent_3" in store._warm_cache

    # 3. Reload: should come from disk
    reloaded = store.load("agent_3")
    assert reloaded is not None
    assert adapter.load.called  # Verify disk read
```

### 2. Benchmark Validation

`benchmarks/validate_warm_cache.py`:

```python
# Run after benchmarks to ensure warm speedup
python benchmarks/validate_warm_cache.py \
    benchmarks/results/streaming_*.json \
    --min-speedup 1.5

# Output:
# ❌ VALIDATION FAILED:
#   streaming_20260204.json: 1024 tokens - warm speedup 1.03x < 1.5x
#   (cold=3790ms, warm=3694ms)
```

### 3. Cache Metrics

`CacheMetrics` class tracks hits/misses/disk loads:

```python
>>> metrics = cache_store.get_metrics()
>>> print(f"Hot hits: {metrics.hot_hits}")
>>> print(f"Warm hits: {metrics.warm_hits}")  # Should be >0 for warm tests
>>> print(f"Disk loads: {metrics.disk_loads}")  # Should be >0 for warm tests
>>> print(f"Warm hit rate: {metrics.warm_hit_rate():.1%}")
```

**If warm_hits = 0 during warm test, the test is broken!**

### 4. Pre-Commit Hook

`scripts/check_warm_cache_pattern.py`:

Detects anti-patterns:
- Missing `evict_only=True` in warm test DELETE
- Missing sleep after evict
- Same session_id without eviction

```bash
# Runs automatically on commit
$ git commit -m "..."
⚠️  WARNINGS:
  benchmarks/streaming_benchmark.py:133: Warm test missing sleep after evict

❌ ERRORS:
  benchmarks/streaming_benchmark.py:133: Warm test missing evict_only=True
```

### 5. CI Validation

`.github/workflows/warm-cache-validation.yml`:

```yaml
- name: Run warm cache benchmark
  run: python benchmarks/streaming_benchmark.py --contexts 1024

- name: Validate warm cache speedup
  run: python benchmarks/validate_warm_cache.py *.json --min-speedup 1.5
```

**CI fails if warm TTFT ≈ cold TTFT.**

## Testing Checklist

When modifying cache code, verify:

- [ ] Integration test `test_warm_reload_after_evict` passes
- [ ] Benchmark validation shows speedup ≥ 1.5x
- [ ] Cache metrics show `warm_hits > 0` during warm test
- [ ] Pre-commit hook passes (no anti-patterns)
- [ ] CI workflow passes (warm cache validation)

## Common Mistakes

### ❌ Using same session_id without eviction

```python
# WRONG
sid = "test"
prime(session_id=sid)
sleep(0.5)
measure(session_id=sid)  # Hits hot cache!
```

### ❌ Deleting agent fully (not evicting)

```python
# WRONG
prime(session_id=sid)
DELETE /v1/agents/oai_{sid}  # Removes disk file!
measure(session_id=sid)  # Cache miss, cold start
```

### ❌ Not waiting for disk flush

```python
# WRONG
prime(session_id=sid)
DELETE /v1/agents/oai_{sid}?evict_only=true
measure(session_id=sid)  # Race: disk write may not finish
```

### ✅ Correct Pattern

```python
# CORRECT
sid = "test"
prime(session_id=sid)
DELETE /v1/agents/oai_{sid}?evict_only=true
sleep(1.0)  # Wait for disk flush
measure(session_id=sid)  # Reloads from disk
```

## Debugging Warm Cache Issues

If warm TTFT ≈ cold TTFT:

1. **Check cache metrics**:
   ```python
   metrics = cache_store.get_metrics()
   print(f"Warm hits: {metrics.warm_hits}")  # Should be > 0
   print(f"Disk loads: {metrics.disk_loads}")  # Should be > 0
   ```

2. **Check disk files**:
   ```bash
   ls -lh ~/.semantic/caches/*.safetensors
   # Should see files created after prime
   ```

3. **Check logs**:
   ```
   # Look for these log messages:
   "Flushed dirty cache to disk before eviction: agent_3"
   "Loaded cache from disk: agent_3 (/path/to/cache.safetensors)"
   ```

4. **Verify evict_only parameter**:
   ```bash
   # Should see evict_only=true in DELETE request
   grep "DELETE /v1/agents" server.log
   ```

## References

- Commit 5d701b7: "fix: Fix warm cache test by implementing evict-only mode"
- Issue investigation: `novelty/paper/BENCHMARK_ARCHITECTURE_AUDIT.md`
- Integration tests: `tests/integration/test_warm_cache_reload.py`
- Validation tool: `benchmarks/validate_warm_cache.py`
