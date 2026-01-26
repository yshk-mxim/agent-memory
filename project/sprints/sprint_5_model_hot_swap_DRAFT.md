# Sprint 5: Model Hot-Swap - Detailed Plan

**Duration**: 10 working days (2 weeks)
**Deliverable**: Dynamic model swapping with cache preservation
**Last Updated**: 2026-01-25

## Day-by-Day Plan

### Day 0: EXP-011 (CRITICAL BLOCKER)
**Goal**: Validate memory reclamation works

**Morning Standup**:
- Review Sprint 4 completion (252/252 tests)
- Understand EXP-011 critical importance
- If EXP-011 fails, entire Sprint 5 blocked

**Tasks**:
1. Create `project/experiments/EXP-011-memory-reclamation.md`
2. Write experiment script:
   ```python
   import mlx.core as mx
   from mlx_lm import load
   import gc

   # Baseline
   baseline = mx.metal.get_active_memory()

   # Load model
   model, tokenizer = load("mlx-community/SmolLM2-135M-Instruct")
   after_load = mx.metal.get_active_memory()

   # Unload
   del model, tokenizer
   gc.collect()
   mx.metal.clear_cache()
   after_unload = mx.metal.get_active_memory()

   # Verify reclamation
   reclaimed_pct = (after_load - after_unload) / (after_load - baseline)
   print(f"Reclaimed: {reclaimed_pct:.1%}")
   assert reclaimed_pct > 0.95, f"Only reclaimed {reclaimed_pct:.1%}"
   ```
3. Run experiment (with `dangerouslyDisableSandbox: true`)
4. Document results

**Exit Criteria**:
- ✅ EXP-011 PASSED (>95% memory reclaimed)
- OR ⛔ EXP-011 FAILED → STOP, escalate to user

---

### Day 1: ModelRegistry Foundation
**Goal**: Create model lifecycle manager

**Morning Standup**: Review EXP-011 success, plan ModelRegistry

**Tasks**:
1. Create `src/semantic/application/model_registry.py`:
   - Class: `ModelRegistry`
   - Methods: `load_model()`, `unload_model()`, `get_current()`
   - State: `_current_model_id`, `_model`, `_tokenizer`, `_spec`
2. Implement model loading:
   ```python
   def load_model(self, model_id: str):
       model, tokenizer = load(model_id)
       spec = get_extractor().extract_spec(model)
       self._model = model
       self._tokenizer = tokenizer
       self._spec = spec
       self._current_model_id = model_id
   ```
3. Implement model unloading (using EXP-011 pattern):
   ```python
   def unload_model(self):
       if self._model is None:
           return
       del self._model, self._tokenizer
       gc.collect()
       mx.metal.clear_cache()
       self._model = None
       self._spec = None
       self._current_model_id = None
   ```
4. Unit tests: `tests/unit/application/test_model_registry.py`

**Exit Criteria**:
- ✅ ModelRegistry class created
- ✅ Load/unload methods working
- ✅ Unit tests passing

---

### Day 2: Hot-Swap Protocol - Drain & Unload
**Goal**: Drain batch engine and unload old model

**Morning Standup**: Review Day 1, plan drain protocol

**Tasks**:
1. Add `drain()` method to BatchEngine:
   - Wait for all active requests to complete
   - Prevent new submissions
   - Return when queue empty
2. Add `shutdown()` method to BatchEngine:
   - Calls `drain()` first
   - Clears `_batch_gen` reference
   - Frees all agent blocks
3. Create `swap_model()` orchestrator in ModelRegistry:
   ```python
   async def swap_model(self, new_model_id: str, batch_engine, cache_store, block_pool):
       # 1. Drain active requests
       await batch_engine.drain()

       # 2. Evict all hot caches to disk
       for agent_id in cache_store.hot_agents():
           cache_store.evict_to_disk(agent_id)

       # 3. Shutdown batch engine
       batch_engine.shutdown()

       # 4. Unload old model
       self.unload_model()

       # 5. Load new model
       self.load_model(new_model_id)

       # 6. Reconfigure BlockPool
       block_pool.reconfigure(self._spec)

       # 7. Update cache store model tag
       new_tag = ModelTag.from_spec(new_model_id, self._spec)
       cache_store.update_model_tag(new_tag)

       # 8. Reinitialize batch engine
       batch_engine.reinitialize(self._model, self._tokenizer, self._spec)
   ```
4. Integration tests

**Exit Criteria**:
- ✅ Drain protocol working
- ✅ swap_model() orchestrates full sequence
- ✅ Integration test: swap succeeds

---

### Day 3: Admin API Endpoints
**Goal**: Add HTTP endpoints for model management

**Tasks**:
1. Create `src/semantic/adapters/inbound/admin_api.py`:
   - POST /admin/models/swap
   - GET /admin/models/current
   - GET /admin/models/available
2. Admin authentication:
   - Separate from API key auth
   - ADMIN_API_KEY environment variable
3. Wire into `api_server.py`:
   - Add router
   - Store ModelRegistry in app.state
4. Integration tests

**Exit Criteria**:
- ✅ Admin endpoints working
- ✅ Authentication enforced
- ✅ Swap via HTTP succeeds

---

### Day 4: Cache Invalidation
**Goal**: Prevent loading incompatible caches

**Tasks**:
1. Add methods to AgentCacheStore:
   - `invalidate_all()`: Mark all caches as stale
   - `update_model_tag(new_tag)`: Update compatibility tag
   - `filter_compatible_caches()`: Only load matching caches
2. Update `load()` to check model compatibility
3. Archive vs delete policy:
   - Move incompatible caches to `.archive/` subdirectory
   - Preserve for potential model return
4. Unit tests

**Exit Criteria**:
- ✅ Incompatible caches not loaded
- ✅ Archive mechanism working

---

### Day 5: Error Recovery & Rollback
**Goal**: Handle swap failures gracefully

**Tasks**:
1. Implement rollback in ModelRegistry:
   ```python
   async def swap_model_with_rollback(self, new_model_id, ...):
       # Save state for rollback
       old_model_id = self._current_model_id
       old_spec = self._spec

       try:
           await self.swap_model(new_model_id, ...)
       except Exception as e:
           # Rollback: reload old model
           self.load_model(old_model_id)
           block_pool.reconfigure(old_spec)
           cache_store.update_model_tag(ModelTag.from_spec(old_model_id, old_spec))
           batch_engine.reinitialize(self._model, self._tokenizer, old_spec)
           raise ModelSwapError(f"Swap failed, rolled back to {old_model_id}") from e
   ```
2. Failure-mode tests:
   - Invalid model ID
   - Load failure (network, disk)
   - Memory exhaustion during load
3. Status monitoring:
   - Track swap state (idle, in_progress, success, failed)
   - Return status via GET /admin/models/status

**Exit Criteria**:
- ✅ Rollback works on failure
- ✅ Server remains stable after failed swap

---

### Day 6: EXP-012 & Performance Validation
**Goal**: Measure swap latency and validate performance

**Tasks**:
1. Create `project/experiments/EXP-012-swap-latency.md`
2. Measure swap time for different models:
   - SmolLM2-135M → Gemma-3-12B
   - Gemma-3-12B → Qwen-2.5-14B
   - Qwen back to SmolLM2
3. Target: < 30 seconds per swap
4. Profile bottlenecks:
   - Model download time (if not cached)
   - Memory clearing time
   - Spec extraction time
5. Optimize if needed

**Exit Criteria**:
- ✅ EXP-012 documented
- ✅ Swap latency < 30s (or documented if slower)

---

### Day 7: Integration Testing
**Goal**: Comprehensive end-to-end tests

**Tasks**:
1. Create `tests/integration/test_model_hot_swap.py`:
   - Test: Swap Gemma → Qwen, verify caches preserved
   - Test: Swap back, Gemma caches reload
   - Test: Active request during swap (graceful completion)
   - Test: Multiple agents, all caches evicted and reloaded correctly
2. Memory leak test:
   - Run 10 consecutive swaps
   - Verify no memory growth
   - Check BlockPool.available_blocks() remains constant
3. Concurrent stress test:
   - 5 agents generating during swap
   - All complete successfully

**Exit Criteria**:
- ✅ All integration tests passing
- ✅ No memory leaks
- ✅ Concurrent requests handled correctly

---

### Day 8: Documentation & Architecture Updates
**Goal**: Complete documentation in docs/ and project/

**Tasks**:
1. Complete `project/architecture/ADR-007-one-model-at-a-time.md`:
   - Context: 24GB M4 Pro constraint
   - Decision: Single model loaded at any time
   - Consequences: Hot-swap protocol required
   - Alternatives considered: Multi-model, process restart
2. Update `docs/architecture.md`:
   - Add "Model Hot-Swap" section with Mermaid diagrams
   - Component diagram (from plan file)
   - Sequence diagram (from plan file)
   - Integration with existing architecture docs
3. Create `docs/model-hot-swap.md`:
   - Hot-swap protocol explanation
   - When to use model swapping
   - Admin API usage examples
   - Memory reclamation details (EXP-011 results)
   - Best practices and troubleshooting
   - Limitations and constraints
4. Update `docs/api-reference.md`:
   - Add Admin API endpoints section
   - POST /admin/models/swap documentation
   - GET /admin/models/current documentation
   - GET /admin/models/available documentation
   - Authentication requirements
5. Update `project/sprints/sprint_5_model_hot_swap.md`:
   - All deliverables documented
   - Implementation details
   - Test results
   - EXP-011 and EXP-012 results
6. Update README.md:
   - Add model hot-swap feature to features list
   - Link to docs/model-hot-swap.md

**Exit Criteria**:
- ✅ ADR-007 complete
- ✅ docs/architecture.md updated with Mermaid diagrams
- ✅ docs/model-hot-swap.md created and comprehensive
- ✅ docs/api-reference.md includes admin endpoints
- ✅ Sprint 5 report complete
- ✅ README.md updated

---

### Day 9: Sprint 4 Deferred Items
**Goal**: Complete carried-over tasks from Sprint 4

**Tasks**:
1. Rate limiter cleanup:
   - Add TTL-based cleanup for `_agent_requests` dict
   - Evict entries older than 24 hours
   - Unit tests
2. Manual EXP-010 validation:
   - Run Claude Code CLI with real model
   - Verify 3-turn conversation works
   - Document results
3. Schemathesis contract tests:
   - Generate OpenAPI schema
   - Run schemathesis against /v1/messages
   - Fix any schema violations
4. CORS production hardening:
   - Configure allowed origins
   - Remove wildcard `*`

**Exit Criteria**:
- ✅ Rate limiter cleanup working
- ✅ EXP-010 validated
- ✅ Schemathesis passing
- ✅ CORS configured

---

### Day 10: Technical Fellows Review & Polish
**Goal**: Final review and quality gates

**Morning Standup**: Prepare for final review

**Tasks**:
1. Run full test suite:
   - Unit: `make test-unit`
   - Integration: `make test-integration`
   - Verify: ALL tests ACTUALLY pass (not just green)
2. Code quality checks:
   - `make lint` — 0 errors
   - `make typecheck` — 0 errors
   - Architecture compliance check (no MLX in domain)
3. Performance validation:
   - EXP-011: Memory reclamation (>95%)
   - EXP-012: Swap latency (<30s)
   - No memory leaks (10 consecutive swaps)
4. Technical Fellows Review:
   - **SE Track**: Architecture review, code quality
   - **ML Track**: MLX memory management, cache invalidation correctness
   - **QE Track**: Test coverage (target: >85% unit, >70% integration)
   - **HW Track**: Memory profile, no leaks, performance
   - Final score and approval
5. Create `project/reviews/SPRINT_5_FELLOWS_REVIEW.md`
6. Update `SPRINT_5_SUMMARY.md`

**Exit Criteria**:
- ✅ All tests passing (TARGET: 300+ total tests)
- ✅ EXP-011 & EXP-012 successful
- ✅ Fellows approval obtained
- ✅ Sprint 5 documentation complete

---

## Expected Outcomes

### Code Additions
| Component | Files | LOC | Tests |
|-----------|-------|-----|-------|
| ModelRegistry | 1 | ~300 | 15 |
| Admin API | 1 | ~150 | 10 |
| BatchEngine extensions | 1 | ~100 | 8 |
| Cache invalidation | 1 | ~80 | 7 |
| Integration tests | 2 | ~400 | 20 |
| **Total** | **6** | **~1,030** | **60+** |

### Test Coverage
- Sprint 4: 252 tests
- Sprint 5: +60 tests
- **Total**: 312+ tests

### Quality Metrics
- Memory reclamation: >95% (EXP-011)
- Swap latency: <30s (EXP-012)
- No memory leaks after 10 swaps
- Architecture compliance: 100%
- Test coverage: >85% unit, >70% integration

---

## Critical Success Factors

1. **EXP-011 MUST PASS** - If memory not reclaimed, entire approach fails
2. **No memory leaks** - 10 consecutive swaps must not grow memory
3. **Graceful drain** - Active requests complete before swap
4. **Cache preservation** - Caches saved to disk, reload on model return
5. **Error recovery** - Failed swap rolls back to previous model

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| EXP-011 fails (memory not reclaimed) | Fallback: Process restart instead of hot-swap |
| Swap too slow (>60s) | Async loading, background download |
| Cache incompatibility bugs | Comprehensive validation, archive old caches |
| Memory leak during swap | Thorough testing, leak detection tools |
| Failed swap leaves server broken | Rollback mechanism, health checks |

---

## Verification Checklist

**End of Sprint 5**:
- [ ] EXP-011 passed (>95% memory reclaimed)
- [ ] EXP-012 passed (<30s swap latency)
- [ ] All 312+ tests passing
- [ ] No memory leaks (10 swaps tested)
- [ ] Admin API working
- [ ] Cache invalidation correct
- [ ] Rollback mechanism tested
- [ ] Technical Fellows approval obtained
- [ ] Documentation complete
