# Warm Cache Bug Analysis

## Problem Summary

The warm cache is completely broken - TTFT for warm cache equals cold cache (no speedup).

Example from benchmark:
- Cold 32K: TTFT ~53 seconds
- Warm 32K: TTFT ~53 seconds (SAME AS COLD!)
- Hot 32K: TTFT ~0.7 seconds (75x faster)

## Root Cause

The logs show:
```
[CACHE LOAD] agent=oai_test_warm_123, hot=False, warm=True
Cache miss: oai_test_warm_123
```

The agent IS in the `_warm_cache` dict, but `load()` returns None!

This means `_load_from_disk()` is being called but **failing silently**.

## Evidence

1. **Cache files ARE being created**: File exists on disk after eviction
2. **_warm_cache dict IS populated**: Logs show `warm=True` after eviction
3. **load() IS checking warm cache**: Code path executes correctly
4. **But _load_from_disk() returns None**: Silent failure

## Likely Cause

The `_load_from_disk()` method at line 574-647 can fail silently if:
1. The cache file path doesn't exist (but logs show it does)
2. The cache adapter load fails (most likely!)
3. Model tag is incompatible (unlikely in same session)

## The Real Bug

Looking at `_load_from_disk()` line 583:
```python
blocks_dict, metadata = self._cache_adapter.load(cache_path)
```

If `self._cache_adapter.load()` fails OR returns empty/invalid data, the method catches the exception and returns None silently!

The issue is likely that:
1. The cache adapter is not properly initialized, OR
2. The safetensors file being written is corrupt/incomplete, OR
3. The file is being written but not flushed properly before the warm test

## The Fix

We need to ensure the cache file is properly flushed to disk BEFORE returning from `_save_to_disk()`.

Looking at `_save_to_disk()` at line 509-536:
```python
cache_path = self._cache_adapter.save(agent_id, entry.blocks, metadata)
self._warm_cache[agent_id] = cache_path
```

The adapter's `save()` method should be flushing to disk, but it might be using buffered I/O without explicit flush.

**Solution**: Add explicit disk sync after save in `_save_to_disk()`:

```python
cache_path = self._cache_adapter.save(agent_id, entry.blocks, metadata)
# CRITICAL: Ensure data is actually written to disk before we declare success
# Without this, subsequent loads may read incomplete/corrupt data
if cache_path.exists():
    # Force OS to flush buffers to disk
    import os
    fd = os.open(cache_path, os.O_RDONLY)
    os.fsync(fd)
    os.close(fd)
self._warm_cache[agent_id] = cache_path
```

## Alternative: Warmup Scan on Server Start

Another issue: When the server starts, `_warm_cache` dict is empty. It's only populated when:
1. We explicitly call `_save_to_disk()`, OR
2. We load a file from disk (which requires it to already be in `_warm_cache` - catch-22!)

**Solution**: Scan cache directory on startup and populate `_warm_cache`:

```python
def __init__(...):
    # ... existing init ...

    # Scan cache directory and populate warm tier
    self._scan_cache_directory()

def _scan_cache_directory(self) -> None:
    """Scan cache directory and populate warm tier on startup."""
    if not self.cache_dir.exists():
        return

    for cache_file in self.cache_dir.glob("*.safetensors"):
        agent_id = cache_file.stem
        self._warm_cache[agent_id] = cache_file
        logger.info(f"[CACHE INIT] Found warm cache for {agent_id}")
```

This would make the warm cache work across server restarts!

## Recommended Fix Order

1. Add `_scan_cache_directory()` in `__init__` (server restart support)
2. Add explicit fsync in `_save_to_disk()` (data integrity)
3. Add better error logging in `_load_from_disk()` (debugging)
4. Re-run benchmarks to verify fix
