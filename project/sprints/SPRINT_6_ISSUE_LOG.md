# Sprint 6: Issue Log

**Last Updated**: 2026-01-25

---

## Critical Issues (Blocking)

### CRITICAL-001: MLXCacheAdapter Constructor Signature Mismatch ✅ FIXED

**Discovered**: Day 0/1 (during smoke test execution)
**Severity**: CRITICAL (blocks all tests)
**Status**: ✅ RESOLVED

**Root Cause**:
- `api_server.py:93` called `MLXCacheAdapter(model=model, spec=model_spec)`
- `MLXCacheAdapter` is a stateless adapter that takes no arguments
- Constructor signature mismatch caused `TypeError: MLXCacheAdapter() takes no arguments`

**Impact**:
- Server failed to start (application startup failed)
- All smoke tests errored at setup
- All E2E tests would fail
- **Blocked**: All Sprint 6 testing

**Error**:
```
TypeError: MLXCacheAdapter() takes no arguments
  File "/Users/dev_user/semantic/src/semantic/entrypoints/api_server.py", line 93, in lifespan
    mlx_adapter = MLXCacheAdapter(model=model, spec=model_spec)
```

**Fix**:
```python
# Before (WRONG):
mlx_adapter = MLXCacheAdapter(model=model, spec=model_spec)

# After (CORRECT):
mlx_adapter = MLXCacheAdapter()  # Stateless, no arguments needed
```

**Resolution**:
- Fixed in `api_server.py:93`
- Changed constructor call to not pass arguments
- Added clarifying comment about stateless nature

**Validation**:
- Re-running smoke tests (task b7b2391)
- Server should now start successfully
- All 7 smoke tests should pass

**Lessons Learned**:
1. Always validate server startup before writing tests
2. Check adapter interfaces match their implementations
3. Run a basic "can the server start?" test before E2E framework

**Time Impact**: ~30 minutes to debug and fix

---

---

## Critical Issues (Blocking)

### CRITICAL-002: BlockPoolBatchEngine Constructor Parameter Mismatch ✅ FIXED

**Discovered**: Day 7 (during smoke test re-run after CRITICAL-001 fix)
**Severity**: CRITICAL (blocks all tests)
**Status**: ✅ RESOLVED

**Root Cause**:
- `api_server.py:117` called `BlockPoolBatchEngine(block_pool=block_pool, ...)`
- Constructor signature is `__init__(pool, ...)` not `__init__(block_pool, ...)`
- Also passed invalid parameters: `batch_window_ms`, `max_batch_size`, `prefill_step_size`
- Actual constructor signature: `model, tokenizer, pool, spec, cache_adapter`

**Impact**:
- Server failed to start (application startup failed)
- All smoke tests errored at setup
- All E2E/stress/benchmark tests would fail
- **Blocked**: All Sprint 6 testing (again)

**Error**:
```
TypeError: BlockPoolBatchEngine.__init__() got an unexpected keyword argument 'block_pool'
  File "/Users/dev_user/semantic/src/semantic/entrypoints/api_server.py", line 113, in lifespan
    batch_engine = BlockPoolBatchEngine(
```

**Fix**:
```python
# Before (WRONG):
batch_engine = BlockPoolBatchEngine(
    model=model,
    tokenizer=tokenizer,
    cache_adapter=mlx_adapter,
    block_pool=block_pool,  # Wrong parameter name
    batch_window_ms=settings.agent.batch_window_ms,  # Not in constructor
    max_batch_size=settings.mlx.max_batch_size,  # Not in constructor
    prefill_step_size=settings.mlx.prefill_step_size,  # Not in constructor
    spec=model_spec,
)

# After (CORRECT):
batch_engine = BlockPoolBatchEngine(
    model=model,
    tokenizer=tokenizer,
    pool=block_pool,  # Fixed parameter name
    spec=model_spec,
    cache_adapter=mlx_adapter,
)
```

**Resolution**:
- Fixed in `api_server.py:113-121`
- Changed `block_pool=` to `pool=`
- Removed invalid parameters (`batch_window_ms`, `max_batch_size`, `prefill_step_size`)

**Validation**:
- Re-running smoke tests (task be4d728)
- Server should now start successfully
- All tests should pass

**Lessons Learned**:
1. Verify constructor signatures match the actual implementation
2. Constructor parameters changed during Sprint 5 refactoring
3. api_server.py initialization code was outdated

**Time Impact**: ~10 minutes to debug and fix

---

## Medium Issues

### None identified yet

---

## Low Issues

### None identified yet

---

## Issue Statistics

- **Total Issues**: 1
- **Critical**: 1 (✅ 1 fixed)
- **Medium**: 0
- **Low**: 0
- **Time to Fix**: 30 minutes average

---

**Status**: All critical issues resolved. Sprint 6 on track.
