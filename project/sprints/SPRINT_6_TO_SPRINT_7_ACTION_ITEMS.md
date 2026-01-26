# Sprint 6 → Sprint 7: Critical Action Items
## Technical Debt & Risk Mitigation

**Generated**: 2026-01-25
**Source**: Technical Fellows Review (Score: 88/100)
**Status**: Sprint 6 APPROVED with mandatory Sprint 7 requirements

---

## 🔴 CRITICAL (Must Fix Before Heavy Production)

### 1. Implement Proper Graceful Shutdown
**Current State**:
```python
# api_server.py:142-144 (INCOMPLETE)
await asyncio.sleep(2)  # Temporary workaround
```

**Required**:
```python
# Implement drain() in BlockPoolBatchEngine
await batch_engine.drain(timeout_seconds=30)
cache_store.save_all_hot_caches()
```

**Impact**: In-flight requests may be terminated abruptly during shutdown
**Risk**: MEDIUM - Data loss, poor UX under load
**Effort**: 4-6 hours
**Priority**: 🔴 CRITICAL

**Implementation Steps**:
1. Add `drain()` method to BlockPoolBatchEngine
2. Track in-flight requests with unique IDs
3. Implement timeout handling
4. Test with E2E scenario (active requests during shutdown)
5. Update api_server.py to use new drain() method

---

### 2. Execute Stress Tests
**Current State**: 0/12 stress tests run (framework complete, async issues)

**Required**:
- Debug aiohttp async HTTP client integration
- Execute at least 1 stress test (test_pool_exhaustion.py)
- Validate graceful 429 response under real load
- Measure latency distribution (p50, p95, p99)

**Impact**: Unknown behavior under 100+ concurrent requests
**Risk**: HIGH for production, LOW for local dev
**Effort**: 8-12 hours
**Priority**: 🟡 MEDIUM (HIGH if deploying to production)

**Implementation Steps**:
1. Debug async/await integration in stress test harness
2. Fix aiohttp ClientSession connection issues
3. Run test_pool_exhaustion.py successfully
4. Validate pool exhaustion returns 429 (not crash)
5. Document load characteristics in benchmark report

---

### 3. Measure Performance Metrics
**Current State**: Missing latency distribution, no sustained load data

**Required**:
- Add latency tracking to all endpoints (p50, p95, p99)
- Run 1-hour sustained load test
- Measure memory growth over time (<5% target)
- Document production performance characteristics

**Impact**: Cannot predict production behavior
**Risk**: MEDIUM
**Effort**: 6-8 hours
**Priority**: 🟡 MEDIUM

**Implementation Steps**:
1. Add middleware for latency tracking
2. Create test_sustained_load_1hour.py
3. Use psutil to track memory every 5 minutes
4. Generate performance report with percentiles
5. Update BENCHMARK_REPORT.md

---

## 🟡 HIGH PRIORITY (Should Fix)

### 4. Fix Code Quality Issues
**Current State**:
- Ruff warnings: E501 (line length), B904 (exception chaining)
- Mypy configuration issue (module import conflict)

**Required**:
```python
# Fix exception chaining (admin_api.py:206)
except Exception as e:
    raise HTTPException(...) from e  # Add 'from e'

# Fix line length (settings.py:180)
description=(
    "Comma-separated list of allowed CORS origins "
    "(* for all, not recommended for production)"
)
```

**Impact**: CI failures, reduced maintainability
**Risk**: LOW (cosmetic)
**Effort**: 2-4 hours
**Priority**: 🟡 HIGH (prevents CI green)

---

### 5. Create Unified E2E Testing Guide
**Current State**: Individual READMEs exist, no comprehensive guide

**Required**: Create `SPRINT_6_E2E_TESTING_GUIDE.md` with:
- How to run all test suites (smoke, e2e, stress, benchmarks)
- Prerequisites (MLX, models, Apple Silicon)
- Interpreting results
- Adding new tests
- Troubleshooting common issues

**Impact**: Developer onboarding efficiency
**Risk**: LOW
**Effort**: 3-4 hours
**Priority**: 🟡 HIGH (documentation completeness)

---

## 🟢 MEDIUM PRIORITY (Nice to Have)

### 6. Add OpenAI Streaming Backpressure
**Current State**: Streaming loop yields immediately without rate limiting

**Required**:
```python
# openai_adapter.py:141 (add backpressure)
if new_text:
    yield {"data": json.dumps(...)}
    await asyncio.sleep(0.01)  # Prevent overwhelming slow clients
```

**Impact**: Could overwhelm slow network connections
**Risk**: LOW
**Effort**: 30 minutes
**Priority**: 🟢 LOW (monitor in production, fix if needed)

---

### 7. Validate Model Hot-Swap E2E Latency
**Current State**: EXP-012 measured 3.1s, but not validated in E2E tests

**Required**:
- Add timing to test_model_hot_swap_e2e.py
- Validate <30s target
- Compare against EXP-012 results
- Document any deviations

**Impact**: Performance validation completeness
**Risk**: LOW
**Effort**: 2-3 hours
**Priority**: 🟢 LOW (experimental data already exists)

---

## Production Deployment Status

### ✅ APPROVED FOR:
- **Local Development** - Full approval, all features working
- **Light Production (<10 users)** - Approved with monitoring
  - E2E tests validate this scenario
  - Health checks provide degradation signals
  - Performance validated for this scale

### ❌ NOT APPROVED FOR:
- **Heavy Production (>50 concurrent users)** - BLOCKED until:
  - [ ] Graceful shutdown drain() implemented
  - [ ] At least 1 stress test passing
  - [ ] Latency distribution measured
  - [ ] Sustained load validated (1 hour)

---

## Sprint 7 Definition of Done

**Minimum requirements to unblock heavy production**:

1. ✅ Graceful shutdown: `drain()` method implemented and tested
2. ✅ Stress tests: At least 1 passing (pool exhaustion validated)
3. ✅ Performance: Latency distribution documented (p50, p95, p99)
4. ✅ Code quality: All ruff/mypy issues resolved
5. ✅ Documentation: E2E testing guide complete

**Success criteria**:
- Technical Fellows score: >90/100
- All production deployment blockers resolved
- Zero critical technical debt remaining

---

## Risk Summary

| Risk | Severity | Likelihood | Mitigation Required | Priority |
|------|----------|------------|---------------------|----------|
| In-flight requests terminated | MEDIUM | HIGH | Implement drain() | 🔴 CRITICAL |
| Unknown load behavior | HIGH | MEDIUM | Run stress tests | 🟡 MEDIUM |
| Poor production performance | MEDIUM | MEDIUM | Measure latency | 🟡 MEDIUM |
| CI failures | LOW | HIGH | Fix ruff/mypy | 🟡 HIGH |
| Slow client overwhelm | LOW | LOW | Add backpressure | 🟢 LOW |

---

## Estimated Sprint 7 Effort

**Total**: 25-37 hours

**Breakdown**:
- Critical items: 18-26 hours (drain + stress tests + metrics)
- High priority: 5-8 hours (code quality + docs)
- Medium priority: 2-3 hours (backpressure + hot-swap validation)

**Recommendation**: Allocate 1 week (40 hours) for Sprint 7 technical debt resolution

---

## Conclusion

Sprint 6 delivered **88/100** - a solid performance with excellent user-focused prioritization. The OpenAI streaming implementation (bonus feature) demonstrates mature product thinking.

However, **critical technical debt remains** that MUST be addressed before heavy production deployment. The good news: all issues have clear solutions and estimated effort.

**Sprint 7 should focus on**: Production readiness, not new features.

---

**Document Owner**: Technical Fellows Committee
**Next Review**: Sprint 7 completion (estimate: 1 week)
**Contact**: See SPRINT_6_TECHNICAL_FELLOWS_REVIEW.md for full analysis
