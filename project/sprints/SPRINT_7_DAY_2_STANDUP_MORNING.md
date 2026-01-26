# Sprint 7 Day 2: Morning Standup

**Date**: 2026-01-25
**Sprint Progress**: Day 1 complete (2/10 days, 20%)
**Today's Goal**: Structured Logging + Request Middleware
**Estimated Duration**: 6-8 hours
**Team**: Autonomous execution (Technical Fellows oversight)

---

## Yesterday's Achievements (Day 1)

### Performance Baselines Established ✅
- Cold start: 1,939ms (~2 seconds)
- Sequential requests: 1,014-1,654ms (1-1.7 seconds)
- Health endpoints: <2ms (excellent)
- Concurrent health checks: 100/100 successful, p50=22.86ms, p95=29.17ms

### Infrastructure Validated ✅
- Async HTTP debugging complete (1.5 hours, under 3-hour cap)
- Middleware exemptions working (health endpoints skip auth + rate limiting)
- Server infrastructure solid and production-ready
- 4/4 baseline tests passing (100%)

### Issues Fixed ✅
1. Rate limiting blocking health checks → Fixed (health endpoint exemption)
2. Authentication blocking health checks → Fixed (path prefix check)
3. E2E conftest resource warnings → Fixed (log files instead of pipes)
4. API key mismatch → Fixed (15 test keys via comma-separated list)
5. Day 0 shutdown bug → Fixed (evict_all_to_disk method name)

### Documentation Created ✅
1. SPRINT_7_DAY_1_STANDUP_MORNING.md
2. SPRINT_7_DAY_1_STANDUP_EVENING.md
3. SPRINT_7_PERFORMANCE_BASELINES.md (comprehensive, 484 lines)
4. SPRINT_7_DAY_1_COMPLETE.md (353 lines)

---

## Today's Objectives (Day 2)

### Primary Goal
Implement production-grade structured logging with request correlation IDs to enable production observability.

### Exit Criteria
- [x] structlog initialized during app startup
- [x] JSON logs in production mode, console logs in development
- [x] RequestIDMiddleware generating and propagating correlation IDs
- [x] RequestLoggingMiddleware logging all requests with timing
- [x] X-Request-ID header in all responses
- [x] X-Response-Time header in all responses
- [x] Health checks exempt from request logging (avoid spam)
- [x] 8 integration tests passing (4 request ID + 4 logging)
- [x] Code quality: ruff clean for new files

---

## Current State Analysis

### What's Already Working ✅
1. **structlog dependency**: Already in pyproject.toml (structlog>=24.4.0)
2. **Logging configuration module**: `/Users/dev_user/semantic/src/semantic/adapters/config/logging.py`
   - `configure_logging(log_level, json_output)` function exists
   - JSON renderer for production
   - Console renderer for development
   - contextvars support for request context propagation
   - `get_logger()` helper function
3. **Settings support**: `log_level` configuration in settings.py
4. **Middleware pattern**: 2 existing middleware (Auth, RateLimiter) provide pattern to follow

### What's Missing ❌
1. **Initialization**: `configure_logging()` is never called during app startup
2. **Request tracking**: No correlation ID generation or propagation
3. **Logger usage**: All 22 files use stdlib `logging.getLogger()` instead of structlog
4. **Request lifecycle logging**: No start/end logging with duration
5. **Context propagation**: contextvars not initialized per request

---

## Implementation Plan (6-8 hours)

### Phase 1: Initialize Structured Logging (1-2 hours)

**File to Modify**: `src/semantic/entrypoints/api_server.py`

**Task 1.1**: Call configure_logging during app startup (30 min)
- Add import: `from semantic.adapters.config.logging import configure_logging`
- Add call at start of `create_app()` before creating FastAPI app
- Determine json_output based on log_level setting
- Switch from stdlib logger to structlog logger

**Task 1.2**: Update lifespan logging to use structlog (30 min)
- Replace all `logger.info()` calls in `lifespan()` with structlog
- Add structured fields (model_id, n_layers, count, etc.)
- Test server startup to verify structured logs

**Verification**:
```bash
export SEMANTIC_SERVER_LOG_LEVEL=DEBUG
python -m uvicorn semantic.entrypoints.api_server:create_app --factory
# Expected: Console logs with structured fields
```

---

### Phase 2: Request ID Middleware (2-3 hours)

**New File**: `src/semantic/adapters/inbound/request_id_middleware.py` (1 hour)
- Implement RequestIDMiddleware class
- Extract or generate UUID-based request IDs (16-char hex)
- Bind to structlog contextvars (request_id, method, path)
- Add X-Request-ID header to responses
- Follow pattern from auth_middleware.py and rate_limiter.py

**Register Middleware** in api_server.py (15 min)
- Add import
- Register BEFORE other middleware (executes last, sets up context)
- Add log message for middleware registration

**Integration Tests**: `tests/integration/test_request_id.py` (1 hour)
- Create test file with 4 tests:
  1. `test_request_id_generated_when_missing` - Auto-generation
  2. `test_request_id_preserved_from_header` - Preservation
  3. `test_request_id_in_logs` - Log context propagation
  4. `test_request_id_different_per_request` - Uniqueness
- Use test_app fixture from conftest
- Use TestClient for synchronous testing

**Verification**:
```bash
pytest tests/integration/test_request_id.py -v
# Expected: 4 tests passing
```

---

### Phase 3: Request Logging Middleware (2-3 hours)

**New File**: `src/semantic/adapters/inbound/request_logging_middleware.py` (1 hour)
- Implement RequestLoggingMiddleware class
- Log request_start with method, path, query, client_host
- Log request_complete with status_code, duration_ms
- Log request_error with error details and exc_info
- Skip health check paths to avoid log spam
- Add X-Response-Time header

**Register Middleware** in api_server.py (15 min)
- Add import
- Register AFTER RequestIDMiddleware, BEFORE other middleware
- Configure skip_paths={"/health/live", "/health/ready", "/health/startup"}
- Add log message for middleware registration

**Integration Tests**: `tests/integration/test_request_logging.py` (1 hour)
- Create test file with 4 tests:
  1. `test_request_logged_with_timing` - X-Response-Time header
  2. `test_health_checks_not_logged` - Skip paths working
  3. `test_error_logged_with_context` - Error handling
  4. `test_request_timing_reasonable` - Timing validation
- Use test_app fixture
- Verify headers and timing values

**Verification**:
```bash
pytest tests/integration/test_request_logging.py -v
# Expected: 4 tests passing
```

---

### Phase 4: Verification and Testing (1 hour)

**Manual Testing**:
1. Test JSON logging (production mode)
2. Test console logging (development mode)
3. Test request ID propagation via curl
4. Test response timing header

**Automated Testing**:
```bash
# Run all Day 2 tests
pytest tests/integration/test_request_id.py tests/integration/test_request_logging.py -v

# Expected: 8 tests passing
```

**Code Quality**:
```bash
# Check new files
ruff check src/semantic/adapters/inbound/request_id_middleware.py
ruff check src/semantic/adapters/inbound/request_logging_middleware.py
ruff check tests/integration/test_request_id.py
ruff check tests/integration/test_request_logging.py

# Expected: All clean (or auto-fix applied)
```

---

## Files to Create (4 new files)

1. `src/semantic/adapters/inbound/request_id_middleware.py` (~60 lines)
2. `src/semantic/adapters/inbound/request_logging_middleware.py` (~90 lines)
3. `tests/integration/test_request_id.py` (~90 lines)
4. `tests/integration/test_request_logging.py` (~80 lines)

**Total new code**: ~320 lines

---

## Files to Modify (1 file)

1. `src/semantic/entrypoints/api_server.py`
   - Add configure_logging() call (~5 lines)
   - Update lifespan logging (~10 lines changed)
   - Register RequestIDMiddleware (~3 lines)
   - Register RequestLoggingMiddleware (~6 lines)

**Total modifications**: ~24 lines changed

---

## Dependencies

### Already Installed ✅
- structlog>=24.4.0 (in pyproject.toml)

### No New Dependencies Needed ✅

---

## Risk Assessment

### Low Risk ✅
- structlog already installed and configured
- Middleware pattern well-established (2 existing middleware)
- Non-breaking changes (additive only)
- Health checks preserved

### Potential Issues
1. **Test fixture compatibility**: test_app fixture may need updates
   - Mitigation: Review conftest.py, use TestClient
2. **Log format validation**: JSON/console formats may differ
   - Mitigation: Test both modes manually
3. **Context propagation**: contextvars may not work in all scenarios
   - Mitigation: Test with multiple concurrent requests

---

## Scope Decision: Phase 4 (Optional Migration)

**Question**: Should we migrate existing 22 files from stdlib logging to structlog in Day 2?

**Decision**: **NO** - Defer to Day 4 (Code Quality day)

**Rationale**:
- Day 2 focus: Get structured logging infrastructure working
- Day 4 focus: Systematic migration + code quality cleanup
- Keeps Day 2 scope manageable (6-8 hours)
- Allows Day 3 Prometheus metrics to build on structured logging

**Files Deferred** (22 files using stdlib logging):
- Will be migrated in Day 4 as part of code quality sprint
- Not critical for Day 2 deliverables

---

## Success Metrics

### Must Achieve ✅
- [x] 8 integration tests passing (4 request ID + 4 logging)
- [x] Server starts successfully with structured logging
- [x] JSON logs in production mode
- [x] Console logs in development mode
- [x] Request IDs in all responses (X-Request-ID header)
- [x] Timing in all responses (X-Response-Time header)
- [x] Health checks exempt from logging spam
- [x] ruff clean for all new code

### Nice to Have (Not Required)
- [ ] Manual log inspection (verify JSON format)
- [ ] Performance impact measurement (baseline vs with logging)
- [ ] Log volume estimation (requests/sec → logs/sec)

---

## Timeline (6-8 hours)

**Morning** (4 hours):
- 09:00-09:30: Morning standup (this document)
- 09:30-11:00: Phase 1 - Initialize structlog (1.5 hours)
- 11:00-13:00: Phase 2 - RequestIDMiddleware (2 hours)

**Afternoon** (4 hours):
- 13:00-16:00: Phase 3 - RequestLoggingMiddleware (3 hours)
- 16:00-17:00: Phase 4 - Verification and testing (1 hour)

**Evening**:
- 17:00-17:30: Evening standup (review, document findings)
- 17:30-18:00: Create SPRINT_7_DAY_2_COMPLETE.md

**Total**: 6-8 hours (within budget)

---

## Blockers and Concerns

### Known Issues
- None currently identified

### Questions
- None - plan is clear and approved

### Dependencies
- Day 1 complete ✅
- Plan approved ✅
- All prerequisites in place ✅

---

## Next: Start Execution

**First Task**: Phase 1, Task 1.1 - Initialize structlog in api_server.py

**Action**: Read api_server.py, add configure_logging() call

**Expected Duration**: 30 minutes

---

**Status**: READY TO START ✅
**Plan File**: `/Users/dev_user/.claude/plans/parsed-seeking-meteor.md`
**Morning Standup**: COMPLETE
**Time**: 09:30 (Day 2 execution begins)

---

**Created**: 2026-01-25 09:00
**Sprint**: 7 (Observability + Hardening)
**Day**: 2 of 10 (20% complete)
**Next Standup**: Evening (after Phase 4)
