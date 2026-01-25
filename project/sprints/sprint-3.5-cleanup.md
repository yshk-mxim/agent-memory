# Sprint 3.5: Complete Cleanup & Quality Gate

**Sprint**: Sprint 3.5 (Cleanup & Validation)
**Start Date**: 2026-01-24
**Duration**: 3 days (focused execution)
**Status**: 🚀 IN PROGRESS
**Goal**: Fix ALL issues identified in Technical Fellow review - NO deferrals

---

## Sprint Overview

**Purpose**: Address all blockers, critical, major, and minor issues before Sprint 4.

**Exit Criteria**:
- ✅ 0 failing tests (currently 1)
- ✅ 0 mypy errors (currently 3)
- ✅ 0 architecture violations (currently 1: MLX in application)
- ✅ All experiments run and documented (currently 9 missing)
- ✅ Safetensors persistence working (currently stubbed)
- ✅ All deferred issues resolved (#9, #10, #13, NEW-5)
- ✅ Integration tests passing
- ✅ Pre-commit hooks installed
- ✅ Coverage report generated (not estimated)

---

## Issues to Fix (By Priority)

### BLOCKER (Must fix first)

**BLOCKER-1**: Fix failing test `test_step_executes_decode_and_yields_completions`
- Estimated: 1-2 hours
- Owner: SE + QE

**BLOCKER-2**: Run EXP-009 (SSE format), EXP-010 (Claude CLI)
- Estimated: 4 hours
- Owner: ML + QE

**BLOCKER-3**: Implement safetensors persistence in AgentCacheStore
- Estimated: 4-6 hours
- Owner: SE + ML

### CRITICAL

**CRITICAL-1**: Create MLXCacheAdapter, remove MLX from application layer
- Estimated: 3-4 hours
- Owner: SE + ML

**CRITICAL-2**: Fix mypy errors in logging.py
- Estimated: 1 hour
- Owner: SE

**CRITICAL-4**: Run integration tests on Apple Silicon
- Estimated: 2 hours
- Owner: QE + ML

### MAJOR

**MAJOR-1**: Address deferred issues #9, #10, #13
- Issue #9: Refactor ModelCacheSpec.from_model() (4 hours)
- Issue #10: Strategy pattern for extractors (6 hours)
- Issue #13: Replace magic number 256 (1 hour)
- Total: 11 hours
- Owner: SE + ML

**MAJOR-2**: Implement NEW-5 (domain errors)
- Replace 21 ValueError + 1 RuntimeError with domain exceptions
- Estimated: 3 hours
- Owner: SE

**MAJOR-3**: Create experiment scripts for EXP-003, EXP-004
- Estimated: 4 hours
- Owner: ML + QE

### MINOR

**MINOR-1**: Update README claims
- Estimated: 30 minutes
- Owner: DE

**MINOR-2**: Resolve TODO comments
- Estimated: 1 hour
- Owner: SE

**MINOR-3**: Install pre-commit hooks
- Estimated: 15 minutes
- Owner: SysE

**MINOR-4**: Generate coverage report
- Estimated: 30 minutes
- Owner: QE

### EXPERIMENTS

**EXP-002**: Block allocation overhead benchmark
**EXP-004**: Response.prompt_cache extraction validation
**EXP-005**: Engine correctness vs mlx_lm.generate
**EXP-006**: Block gather performance
**EXP-007**: Cache extraction end-to-end
**EXP-008**: Prefix matching hit rate
**EXP-009**: SSE format validation
**EXP-010**: Claude CLI compatibility
**EXP-011**: Model swap memory reclamation

Total: 9 experiments to run/document

---

## Day 1 Plan: Blockers + Critical Issues

### Morning (4 hours)

**Task 1**: Fix BLOCKER-1 (failing test) - 1 hour
- Fix BatchGenerator import or mark as integration test
- Verify all unit tests pass

**Task 2**: Fix CRITICAL-2 (mypy errors) - 1 hour
- Add proper type annotations to logging.py
- Run mypy --strict, verify 0 errors

**Task 3**: Start BLOCKER-3 (safetensors) - 2 hours
- Implement actual save_file() calls
- Implement atomic write (tmp + rename)
- Add model tag validation

### Afternoon (4 hours)

**Task 4**: Complete BLOCKER-3 (safetensors) - 2 hours
- Implement load_file() with validation
- Add tests for save/load roundtrip
- Verify 153+ tests passing

**Task 5**: Start CRITICAL-1 (MLXCacheAdapter) - 2 hours
- Create src/semantic/adapters/outbound/mlx_cache.py
- Define MLXCachePort protocol
- Implement concatenate() method

### Evening Verification

- ✅ All unit tests passing
- ✅ Mypy clean
- ✅ Safetensors working
- ✅ MLXCacheAdapter 50% complete

---

## Day 2 Plan: Critical + Major Issues

### Morning (4 hours)

**Task 1**: Complete CRITICAL-1 (MLXCacheAdapter) - 2 hours
- Remove MLX imports from batch_engine.py
- Update batch_engine to use adapter
- Verify tests pass

**Task 2**: MAJOR-2 (NEW-5 domain errors) - 2 hours
- Create domain error hierarchy
- Replace 10 ValueError instances
- Update tests

### Afternoon (4 hours)

**Task 3**: Complete MAJOR-2 (NEW-5) - 1 hour
- Replace remaining 11 ValueError + 1 RuntimeError
- Verify all tests pass

**Task 4**: MAJOR-1 (Issue #13 - magic number 256) - 1 hour
- Replace hardcoded 256 with spec.block_tokens
- Update tests

**Task 5**: Start MAJOR-1 (Issue #9 - refactor from_model) - 2 hours
- Extract Gemma3Extractor, QwenExtractor, LlamaExtractor classes
- Move logic to separate functions

### Evening Verification

- ✅ 0 architecture violations
- ✅ Domain errors implemented
- ✅ Magic numbers replaced
- ✅ Refactoring 50% complete

---

## Day 3 Plan: Experiments + Integration + Cleanup

### Morning (4 hours)

**Task 1**: Complete MAJOR-1 (Issue #9, #10) - 2 hours
- Complete extractor refactoring
- Implement Strategy pattern
- Verify all tests pass

**Task 2**: Run integration tests (CRITICAL-4) - 2 hours
- Fix any MLX import issues
- Run on Apple Silicon
- Document results

### Afternoon (4 hours)

**Task 3**: Run experiments EXP-002, EXP-005, EXP-006 - 2 hours
- Benchmark block allocation
- Validate engine correctness
- Measure block gather performance
- Document results

**Task 4**: Create experiment scripts (MAJOR-3) - 2 hours
- Write experiments/exp_003_cache_injection.py
- Write experiments/exp_004_cache_extraction.py
- Run and verify

### Evening (4 hours)

**Task 5**: Run BLOCKER-2 experiments (EXP-009, EXP-010) - 2 hours
- Validate SSE format
- Test Claude CLI compatibility
- Document results

**Task 6**: Minor issues + final verification - 2 hours
- Update README
- Resolve TODO comments
- Install pre-commit hooks
- Generate coverage report
- Final test run

---

## Success Criteria

### Quality Gates (All Must Pass)

- [ ] `make test` - 0 failures
- [ ] `make typecheck` - 0 errors
- [ ] `make lint` - 0 errors
- [ ] `make security` - 0 high/critical
- [ ] Architecture: 0 MLX imports in domain/application
- [ ] Coverage: >85% measured (not estimated)

### Experiments (All Must Run)

- [ ] EXP-002: Block allocation < 1ms
- [ ] EXP-003: Cache injection works (script created)
- [ ] EXP-004: Cache extraction works (script created)
- [ ] EXP-005: Engine output matches reference
- [ ] EXP-006: Block gather performance within 20%
- [ ] EXP-007: Cache extraction end-to-end
- [ ] EXP-008: Prefix matching >80% hit rate
- [ ] EXP-009: SSE format matches Anthropic
- [ ] EXP-010: Claude CLI connects
- [ ] EXP-011: Model swap reclaims memory

### Issues (All Must Close)

- [ ] BLOCKER-1: Test failure fixed
- [ ] BLOCKER-2: EXP-009, EXP-010 pass
- [ ] BLOCKER-3: Safetensors working
- [ ] CRITICAL-1: MLX adapter created
- [ ] CRITICAL-2: Mypy clean
- [ ] CRITICAL-4: Integration tests pass
- [ ] MAJOR-1: Issues #9, #10, #13 closed
- [ ] MAJOR-2: NEW-5 implemented
- [ ] MAJOR-3: Experiment scripts created
- [ ] MINOR-1: README updated
- [ ] MINOR-2: TODOs resolved
- [ ] MINOR-3: Pre-commit installed
- [ ] MINOR-4: Coverage measured

---

## Risk Mitigation

### Risk 1: MLX not available in environment
- Mitigation: Mark tests as @pytest.mark.integration, skip if MLX unavailable
- Fallback: Use mocks for unit tests

### Risk 2: Experiments fail
- Mitigation: Document failure reasons, adjust expectations
- Fallback: Mark as PENDING with action items

### Risk 3: Refactoring introduces bugs
- Mitigation: Run full test suite after each change
- Fallback: Git revert, smaller incremental changes

---

## Deliverables

### Code Changes

1. `src/semantic/adapters/outbound/mlx_cache.py` - NEW
2. `src/semantic/domain/errors.py` - UPDATED (domain error hierarchy)
3. `src/semantic/application/batch_engine.py` - UPDATED (remove MLX)
4. `src/semantic/application/agent_cache_store.py` - UPDATED (safetensors)
5. `src/semantic/adapters/config/logging.py` - UPDATED (mypy fixes)
6. `src/semantic/domain/value_objects.py` - UPDATED (refactor from_model)
7. `experiments/exp_002_block_allocation.py` - NEW
8. `experiments/exp_003_cache_injection.py` - NEW
9. `experiments/exp_004_cache_extraction.py` - NEW
10. `experiments/exp_005_engine_correctness.py` - NEW
11. `experiments/exp_006_block_gather_perf.py` - NEW
12. Multiple test files - UPDATED

### Documentation

1. `project/experiments/EXP-002-block-allocation.md` - Results
2. `project/experiments/EXP-004-cache-extraction.md` - Results
3. `project/experiments/EXP-005-engine-correctness.md` - Results
4. `project/experiments/EXP-006-block-gather.md` - Results
5. `project/experiments/EXP-007-cache-e2e.md` - Results
6. `project/experiments/EXP-009-sse-format.md` - Results
7. `project/experiments/EXP-010-claude-cli.md` - Results
8. `project/experiments/EXP-011-model-swap.md` - Results
9. `README.md` - UPDATED
10. `project/sprints/sprint-3.5-completion.md` - Final summary

### Quality Reports

1. Coverage report (pytest-cov HTML)
2. Architecture compliance report
3. Final test results
4. Sprint completion summary

---

## Timeline Summary

**Day 1**: Blockers + Safetensors + MLX adapter start (8 hours)
**Day 2**: Architecture cleanup + Domain errors + Refactoring (8 hours)
**Day 3**: Experiments + Integration + Final cleanup (12 hours)

**Total**: ~28 hours (3.5 days intensive work)

**Confidence**: HIGH (all issues well-scoped, no external dependencies)

---

## Sprint Start

🚀 **Sprint 3.5 is GO!** Starting Day 1 NOW.
