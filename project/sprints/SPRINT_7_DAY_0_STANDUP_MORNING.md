# Sprint 7 Day 0: Morning Standup
**Date**: 2026-01-25
**Time**: Morning (Start of Day 0)
**Attendees**: Backend Dev, SysE, PM

---

## Yesterday's Completion

**Sprint 6**: COMPLETE ✅
- Score: 88/100
- All tests passing (35/35)
- Technical debt documented

**Sprint 7 Planning**: COMPLETE ✅
- Developer standup: Consensus reached
- Detailed plan: Created
- Technical Fellows review: APPROVED (93/100)
- Todo list: Created (10 days)

---

## Today's Goals (Day 0)

**Primary**: Implement graceful shutdown + 3-tier health endpoints

**Deliverables**:
1. BatchEngine `drain()` method with request tracking
2. Updated api_server.py shutdown to use `drain()`
3. 3-tier health endpoints (/health/live, /health/ready, /health/startup)
4. E2E test: test_graceful_shutdown.py
5. Integration tests: test_health_endpoints.py (4 tests)

**Exit Criteria**:
- [ ] BatchEngine has working `drain()` method
- [ ] api_server.py calls `drain()` on shutdown
- [ ] /health/live, /health/ready, /health/startup implemented
- [ ] 5 tests passing (1 E2E + 4 integration)

**Estimated**: 6-8 hours

---

## Blockers

**None identified**

**Potential Risks**:
- BatchEngine `drain()` complexity (race conditions)
- Health endpoint testing may require server restart

**Mitigation**:
- Implement drain state flag (Fellows recommendation)
- Use subprocess for health endpoint tests

---

## Technical Fellows Recommendations to Implement

**Critical Fix #1**: Add drain state flag to prevent new requests during drain

```python
class BlockPoolBatchEngine:
    def __init__(self, ...):
        self._draining: bool = False
        self._active_requests: dict[str, CompletionResult] = {}

    def submit(self, ...) -> str:
        if self._draining:
            raise PoolExhaustedError("Engine is draining, not accepting new requests")
        # ... proceed ...

    async def drain(self, timeout_seconds: int = 30) -> int:
        self._draining = True  # Stop accepting new requests
        start_time = time.time()
        drained_count = 0

        while self._active_requests and (time.time() - start_time < timeout_seconds):
            for result in self.step():
                if result.uid in self._active_requests:
                    self._active_requests.pop(result.uid)
                    drained_count += 1
            await asyncio.sleep(0.1)

        if self._active_requests:
            logger.warning(f"{len(self._active_requests)} requests timed out during drain")

        return drained_count
```

---

## Implementation Plan

### Morning Session (4-5 hours): BatchEngine drain()

**Step 1**: Add request tracking to BatchEngine (30 min)
- Add `_active_requests` dict
- Add `_draining` flag
- Track requests in `submit()`

**Step 2**: Implement `drain()` method (2 hours)
- Set `_draining = True`
- Loop until requests complete or timeout
- Remove completed requests from tracking
- Log timeout warnings

**Step 3**: Update api_server.py shutdown (30 min)
- Replace `asyncio.sleep(2)` with `batch_engine.drain(30)`
- Log drained count

**Step 4**: Create E2E test (1-2 hours)
- test_graceful_shutdown_with_active_requests
- Start server, submit 3 requests
- Trigger shutdown mid-request
- Verify all 3 complete
- Verify cache saved

### Afternoon Session (2-3 hours): 3-Tier Health Endpoints

**Step 1**: Implement health endpoints (1.5 hours)
- /health/live - Always 200
- /health/ready - 503 if >90% pool or shutting down
- /health/startup - 503 until model loaded

**Step 2**: Add shutdown flag to lifespan (15 min)
- Set `app.state.shutting_down = False` on startup
- Set `app.state.shutting_down = True` on shutdown

**Step 3**: Create integration tests (1 hour)
- test_health_live_always_200
- test_health_ready_503_when_pool_exhausted
- test_health_ready_503_when_shutting_down
- test_health_startup_503_until_model_loaded

---

## Files to Modify

**Production**:
1. `/Users/dev_user/semantic/src/semantic/application/batch_engine.py`
   - Add `_draining` flag
   - Add `_active_requests` dict
   - Implement `drain()` method
   - Update `submit()` to check drain state

2. `/Users/dev_user/semantic/src/semantic/entrypoints/api_server.py`
   - Replace `asyncio.sleep(2)` with `drain()` call
   - Add 3-tier health endpoints
   - Add `shutting_down` flag to state

**Tests**:
3. `/Users/dev_user/semantic/tests/e2e/test_graceful_shutdown.py` (NEW)
   - test_graceful_shutdown_with_active_requests

4. `/Users/dev_user/semantic/tests/integration/test_health_endpoints.py` (NEW)
   - test_health_live_always_200
   - test_health_ready_503_when_pool_exhausted
   - test_health_ready_503_when_shutting_down
   - test_health_startup_503_until_model_loaded

---

## Success Criteria

**Code**:
- [ ] `drain()` method implemented with state flag
- [ ] `drain()` called in api_server shutdown
- [ ] 3 health endpoints implemented
- [ ] Shutdown flag properly managed

**Tests**:
- [ ] 1 E2E test passing
- [ ] 4 integration tests passing
- [ ] Zero resource warnings

**Quality**:
- [ ] ruff check passes
- [ ] mypy passes
- [ ] Architecture compliance maintained

---

## Next Steps After Completion

1. Evening standup to review completion
2. Run all tests (E2E + integration)
3. Fix any issues found
4. Evening standup to verify clean status
5. Plan Day 1 in detail

---

**Standup Complete**: Ready to start Day 0 implementation

**Start Time**: 2026-01-25 (now)
**Target Completion**: 6-8 hours
