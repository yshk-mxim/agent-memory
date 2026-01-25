# Sprint 2.5: Critical Hotfix - Thread Safety & Memory Safety

**Type**: Emergency Hotfix Sprint
**Duration**: 5 days
**Priority**: BLOCKING for Sprint 3
**Date**: 2026-01-24

---

## Executive Summary

Critical code review identified **8 BLOCKING issues** in Sprint 0-2 code:
- 4 thread-safety violations (data races)
- 2 memory leaks
- 1 resource leak
- 1 type safety gap

**Current Status**: ❌ **NOT PRODUCTION-READY**
- Code breaks under concurrent access
- Memory leaks in long-running scenarios
- Resource leaks on error paths

**Sprint 2.5 Goal**: Fix all CRITICAL and HIGH severity issues before Sprint 3

---

## Critical Issues (from Code Review)

### Issue #1: Thread-Safety in BlockPool (CRITICAL)

**Problem**: `BlockPool.allocate()` and `free()` have race conditions

**Race Scenario**:
```python
# Thread A: if len(self.free_list) < 2  ← 2 available, OK
# Thread B: if len(self.free_list) < 2  ← Still 2 available, OK
# Thread A: block_id = self.free_list.pop()  ← Takes block 0
# Thread B: block_id = self.free_list.pop()  ← Takes block 1
# Thread A: block_id = self.free_list.pop()  ← IndexError! List is empty!
```

**Impact**: Crashes in production under concurrent load

**Fix**: Add `threading.Lock()` to protect all shared state

```python
import threading

class BlockPool:
    def __init__(self, spec: ModelCacheSpec, total_blocks: int) -> None:
        # ... existing code ...
        self._lock = threading.Lock()

    def allocate(self, n_blocks: int, layer_id: int, agent_id: str) -> list[KVBlock]:
        with self._lock:
            # ... existing code (now protected) ...

    def free(self, blocks: list[KVBlock], agent_id: str) -> None:
        with self._lock:
            # ... existing code (now protected) ...

    def free_agent_blocks(self, agent_id: str) -> int:
        with self._lock:
            # ... existing code (now protected) ...

    def reconfigure(self, new_spec: ModelCacheSpec) -> None:
        with self._lock:
            # ... existing code (now protected) ...
```

**Effort**: 2 hours
**Owner**: HW + SE

---

### Issue #2: Thread-Safety in AgentBlocks (CRITICAL)

**Problem**: `add_block()` and `remove_block()` are thread-unsafe and UNUSED

**Fix**: **REMOVE** these methods entirely

```python
# DELETE LINES 138-179 in entities.py

# BEFORE:
class AgentBlocks:
    def add_block(self, block: KVBlock) -> None:
        # ... REMOVE THIS METHOD

    def remove_block(self, block_id: int, layer_id: int) -> KVBlock | None:
        # ... REMOVE THIS METHOD

# AFTER:
class AgentBlocks:
    # Only keep read-only methods:
    # - num_blocks()
    # - num_layers()
    # - blocks_for_layer()
```

**Rationale**:
- Unused (batch_engine only uses replacement)
- Thread-unsafe (no locks)
- Breaks invariants (no validation after mutation)

**Effort**: 1 hour (delete + update tests)
**Owner**: SE

---

### Issue #6: Memory Leak in step() (CRITICAL)

**Problem**: During `step()`, memory DOUBLES temporarily

**Scenario**:
```
Agent has 32 blocks × 48 layers × 2 MB = 3 GB
During step():
  - Old cache: 3 GB (in old_blocks.layer_data)
  - New cache: 3 GB (in new_blocks.layer_data)
  - Total: 6 GB!

Python GC might not run for seconds → OOM crash
```

**Fix**: Explicitly clear `layer_data` before freeing

```python
def step(self) -> Iterator[CompletedGeneration]:
    # ... existing code ...

    for finished in batch_response.finished:
        # ... existing code ...

        # Extract cache and convert to blocks
        blocks = self._extract_cache(uid)

        # Free old prefill blocks
        if agent_id in self._agent_blocks:
            old_blocks = self._agent_blocks[agent_id]
            for layer_blocks in old_blocks.blocks.values():
                # ← ADD THIS: Clear tensors BEFORE freeing
                for block in layer_blocks:
                    block.layer_data = None  # Force immediate release

                self._pool.free(layer_blocks, agent_id)

        # Store new blocks
        self._agent_blocks[agent_id] = blocks
```

**Also fix in BlockPool.free()**:
```python
def free(self, blocks: list[KVBlock], agent_id: str) -> None:
    for block in blocks:
        # ... existing validation ...

        # ← ADD THIS: Clear layer data to free memory immediately
        if hasattr(block, 'layer_data'):
            block.layer_data = None

        self.free_list.append(block.block_id)
        # ... rest of code ...
```

**Effort**: 2 hours
**Owner**: HW

---

### Issue #8: Resource Leak on Error (HIGH)

**Problem**: `submit()` error path only frees layer 0

**Current Code**:
```python
try:
    uids = self._batch_gen.insert(...)
except Exception as e:
    if cache is None and agent_id in self._agent_blocks:
        blocks = self._agent_blocks[agent_id].blocks_for_layer(0)
        self._pool.free(blocks, agent_id)  # ← ONLY FREES LAYER 0!
        del self._agent_blocks[agent_id]
    raise InvalidRequestError(...) from e
```

**Fix**: Free ALL layers

```python
try:
    uids = self._batch_gen.insert(...)
except Exception as e:
    if cache is None and agent_id in self._agent_blocks:
        agent_blocks = self._agent_blocks[agent_id]
        # ← CHANGED: Free ALL layers, not just layer 0
        for layer_blocks in agent_blocks.blocks.values():
            self._pool.free(layer_blocks, agent_id)
        del self._agent_blocks[agent_id]
    raise InvalidRequestError(...) from e
```

**Effort**: 1 hour
**Owner**: QE

---

### Issue #3: Missing Invariant Validation (HIGH)

**Problem**: `add_block()` and `remove_block()` don't re-validate invariants

**Fix**: Since we're REMOVING these methods (Issue #2), this is automatically fixed

**Effort**: 0 hours (covered by Issue #2)

---

### Issue #4: Opaque Any Types (HIGH)

**Problem**: `CompletedGeneration.blocks: Any` defeats type safety

**Current Code**:
```python
@dataclass(frozen=True)
class CompletedGeneration:
    uid: str
    text: str
    blocks: Any  # ← "(avoid circular import)"
    finish_reason: str
    token_count: int
```

**Fix**: Use `TYPE_CHECKING` guard

```python
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from semantic.domain.entities import AgentBlocks

@dataclass(frozen=True)
class CompletedGeneration:
    uid: str
    text: str
    blocks: "AgentBlocks"  # ← String annotation for forward ref
    finish_reason: str
    token_count: int
```

**Also fix**:
- `value_objects.py:24` - `cache: list[Any]` → `cache: list[tuple[Any, Any]]`
- `batch_engine.py:95` - `cache: Any | None` → `cache: "AgentBlocks | None"`

**Effort**: 2 hours
**Owner**: SE

---

### Issue #9: Memory Leak - Untracked UID (MEDIUM → HIGH)

**Problem**: If UID not tracked, cache stays in BatchGenerator forever

**Current Code**:
```python
def step(self) -> Iterator[CompletedGeneration]:
    for finished in batch_response.finished:
        uid = finished.uid
        if uid not in self._active_requests:
            continue  # ← LEAK! No cleanup!
```

**Fix**: Log error and attempt cleanup

```python
import logging

def step(self) -> Iterator[CompletedGeneration]:
    for finished in batch_response.finished:
        uid = finished.uid
        if uid not in self._active_requests:
            # ← ADD THIS: Log and clean up
            logging.error(
                f"Untracked UID {uid} in batch - possible memory leak. "
                f"Active UIDs: {list(self._active_requests.keys())}"
            )
            # Try to extract cache anyway to prevent leak
            try:
                if self._batch_gen is not None:
                    self._batch_gen.extract_cache(uid)
            except Exception as e:
                logging.warning(f"Failed to clean up untracked UID {uid}: {e}")
            continue
```

**Effort**: 1 hour
**Owner**: QE

---

## Sprint 2.5 Scope

### Must Fix (CRITICAL)

1. ✅ Issue #1: Thread-Safety in BlockPool (2h)
2. ✅ Issue #2: Remove unsafe AgentBlocks methods (1h)
3. ✅ Issue #6: Memory leak in step() (2h)

**Total**: 5 hours

### Should Fix (HIGH)

4. ✅ Issue #4: Opaque Any types (2h)
5. ✅ Issue #8: Resource leak on error (1h)
6. ✅ Issue #9: Untracked UID leak (1h)

**Total**: 4 hours

### Nice to Fix (MEDIUM - Defer if needed)

7. ⏳ Issue #7: Clear layer_data in free() (1h) - covered by #6
8. ⏳ Issue #10: Documentation improvements (2h)

**Total Sprint 2.5**: 9 hours core + 3 hours testing = **12 hours (1.5 days)**

---

## Testing Requirements

### New Tests Required

1. **test_concurrent_allocation** (tests/integration/test_block_pool_concurrent.py)
   ```python
   def test_concurrent_allocation():
       """10 threads allocating blocks simultaneously."""
       pool = BlockPool(spec, total_blocks=1000)
       results = []

       def allocate_worker():
           try:
               blocks = pool.allocate(10, layer_id=0, agent_id=f"thread_{threading.current_thread().ident}")
               results.append(("success", blocks))
           except Exception as e:
               results.append(("error", e))

       threads = [threading.Thread(target=allocate_worker) for _ in range(10)]
       for t in threads:
           t.start()
       for t in threads:
           t.join()

       # All should succeed or properly error (no crashes)
       assert len(results) == 10
   ```

2. **test_memory_leak_prevention** (tests/integration/test_memory.py)
   ```python
   import pytest
   from memray import Tracker

   def test_no_memory_leak_over_100_generations():
       """Memory should not grow over 100 generation cycles."""
       engine = BlockPoolBatchEngine(...)

       with Tracker() as tracker:
           for i in range(100):
               uid = engine.submit(f"agent_{i}", "Test", max_tokens=10)
               for completion in engine.step():
                   # Free blocks explicitly
                   for layer_blocks in completion.blocks.blocks.values():
                       pool.free(layer_blocks, completion.blocks.agent_id)

       # Memory growth should be < 1MB
       assert tracker.peak_memory_allocated < 1_000_000
   ```

3. **test_error_cleanup** (tests/unit/test_batch_engine.py)
   ```python
   def test_submit_error_frees_all_layers(engine, pool):
       """If insert fails, all allocated blocks should be freed."""
       initial_available = pool.available_blocks()

       # Make insert fail
       engine._batch_gen_factory = lambda m, t: ErrorBatchGenerator()

       with pytest.raises(InvalidRequestError):
           engine.submit("test_agent", "Hello", max_tokens=10)

       # All blocks should be freed
       assert pool.available_blocks() == initial_available
   ```

---

## Timeline

### Day 1: Thread-Safety (Issues #1, #2)

**Morning**:
- Add `_lock` to BlockPool
- Wrap allocate/free/reconfigure in `with self._lock`
- Test single-threaded (should pass existing tests)

**Afternoon**:
- Remove `add_block()` and `remove_block()` from AgentBlocks
- Update test suite (remove dead tests)
- Write concurrent allocation test

**Evening**:
- Run concurrent tests (100 iterations)
- Fix any race conditions discovered

**Deliverable**: Thread-safe BlockPool and AgentBlocks

---

### Day 2: Memory Safety (Issues #6, #9)

**Morning**:
- Add `block.layer_data = None` in step() before freeing
- Add same in BlockPool.free()
- Write memory leak test with pytest-memray

**Afternoon**:
- Add logging and cleanup for untracked UIDs in step()
- Test with deliberately untracked UID
- Verify no memory leak in 100-iteration test

**Evening**:
- Run full test suite (unit + integration)
- Profile memory usage with Memray

**Deliverable**: Memory-safe block lifecycle

---

### Day 3: Type Safety & Error Handling (Issues #4, #8)

**Morning**:
- Add TYPE_CHECKING guards to value_objects.py
- Fix CompletedGeneration.blocks annotation
- Fix cache type annotations
- Run mypy --strict (should have 0 errors)

**Afternoon**:
- Fix submit() error path (free all layers)
- Write error cleanup test
- Test error scenarios (insert fails, extract fails)

**Evening**:
- Code review of all changes
- Update documentation

**Deliverable**: Type-safe and error-safe code

---

### Days 4-5: Testing & Documentation

**Day 4 Morning**:
- Run full test suite (unit + integration) 1000 times
- Check for flaky tests
- Fix any discovered issues

**Day 4 Afternoon**:
- Update threading model documentation
- Update error handling documentation
- Create Sprint 2.5 review document

**Day 5**:
- Final code review
- Performance testing (no regression from locks)
- Create migration guide for any API changes
- Tag release: v0.2.5 (hotfix)

**Deliverable**: Production-ready Sprint 2.5

---

## Success Criteria

### Functional

- ✅ All existing tests pass
- ✅ 10 concurrent threads can allocate without crashes
- ✅ 1000 iterations of concurrent test pass
- ✅ No memory leak in 100 generation cycles
- ✅ Error paths free all resources
- ✅ MyPy --strict passes with 0 errors

### Performance

- ✅ Single-threaded performance: <10% regression
- ✅ Multi-threaded (3 agents): >2× throughput vs serial
- ✅ Memory overhead from locks: <1MB

### Quality

- ✅ Code coverage: >90% (was 100%)
- ✅ No pylint warnings
- ✅ No new code smells
- ✅ Documentation updated

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Locks cause deadlock | LOW | HIGH | Careful lock ordering, test |
| Performance regression | MEDIUM | MEDIUM | Profile, optimize if >20% |
| Breaks existing tests | LOW | MEDIUM | Run tests frequently |
| Introduces new bugs | MEDIUM | HIGH | Thorough code review |
| Takes longer than 5 days | MEDIUM | HIGH | Daily check-ins, cut scope if needed |

---

## Definition of Done

Sprint 2.5 is complete when:

1. ✅ All 8 critical/high issues fixed
2. ✅ All new tests passing (unit + integration + concurrent)
3. ✅ No memory leaks detected (pytest-memray)
4. ✅ No race conditions detected (ThreadSanitizer)
5. ✅ MyPy --strict passes
6. ✅ Documentation updated
7. ✅ Code reviewed by 2+ team members
8. ✅ Performance benchmarks show <20% regression
9. ✅ Tagged release: v0.2.5

---

## Team Assignments

**HW (Hardware Engineer)**:
- Issue #1: Add locks to BlockPool
- Issue #6: Clear layer_data in memory leak fix
- Memory testing with Memray

**SE (Software Engineer)**:
- Issue #2: Remove unsafe AgentBlocks methods
- Issue #4: Fix opaque Any types
- Code review and architectural oversight

**QE (Quality Engineer)**:
- Issue #8: Fix error cleanup
- Issue #9: Fix untracked UID leak
- Write all new tests (concurrent, memory, error)

**ML (Machine Learning Engineer)**:
- Performance testing
- Verify no regression in generation quality
- Profile lock overhead

---

## Post-Sprint 2.5

**Immediate Next**:
- ✅ Sprint 3: AgentCacheStore (now unblocked)
- ✅ Concurrent agent tests can be added
- ✅ Production deployment feasible

**Deferred to Sprint 4**:
- Lock-free optimizations (if performance is issue)
- Lock-ordering documentation
- Observability improvements

---

**Sprint 2.5 Status**: 📋 READY TO START
**Blocking For**: Sprint 3 (AgentCacheStore)
**Expected Completion**: 2026-01-29 (5 days from now)

---

**Prepared By**: All Team (based on critical code review)
**Date**: 2026-01-24
**Priority**: CRITICAL - BLOCKING
