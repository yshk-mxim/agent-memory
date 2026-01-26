# Sprint 7: Developer Standup - Kickoff Meeting
**Date**: 2026-01-25
**Meeting Type**: Planning Standup
**Attendees**: Backend Dev, SysE (Systems Engineer), ML Engineer, OSS Engineer, PM

---

## Purpose

Plan Sprint 7 by integrating:
1. Sprint 6 technical debt (from Technical Fellows review)
2. Original Sprint 7 plan (Observability + Hardening)
3. Previous sprint learnings

---

## Sprint 6 Review - What We Delivered

**Score**: 88/100 ✅

**Strengths**:
- 35/35 tests passing (100%)
- OpenAI streaming fully implemented (bonus)
- Performance validated: 1.6x batching, <1ms cache
- Architecture: 100% hexagonal compliance

**Technical Debt Carried Forward**:
1. 🔴 **CRITICAL**: Graceful shutdown incomplete (uses `asyncio.sleep(2)` not `drain()`)
2. 🔴 **CRITICAL**: Stress tests not executed (0/12 run, async issues)
3. 🟡 **HIGH**: Missing performance metrics (latency p50/p95/p99)
4. 🟡 **HIGH**: Code quality (ruff B904, mypy config)
5. 🟢 **MEDIUM**: E2E testing guide incomplete

**Production Status**:
- ✅ Local dev: Approved
- ✅ Light production (<10 users): Approved with monitoring
- ❌ Heavy production (>50 users): BLOCKED by technical debt

---

## Original Sprint 7 Plan Review

**Focus**: Observability + Hardening (2 weeks)

**Core Deliverables**:
1. Health endpoints (3-tier: /health/live, /health/ready, /health/startup)
2. Prometheus metrics (15+ metrics)
3. Structured JSON logging (structlog)
4. Graceful shutdown (multi-phase drain)
5. Runtime-configurable batch window (admin API)
6. Request middleware (active tracking, 503 on shutdown)
7. CLI entrypoint (`python -m semantic serve`)
8. pip-installable package (Hatchling)
9. CHANGELOG, LICENSE, NOTICE, release notes
10. License compliance (liccheck)
11. SBOM generation (syft + CycloneDX)
12. OpenTelemetry tracing
13. Alerting thresholds for metrics
14. Log retention and rotation policy

**Prometheus Metrics** (15 planned):
- semantic_request_total
- semantic_request_duration_seconds
- semantic_time_to_first_token_seconds
- semantic_request_queue_depth
- semantic_batch_size
- semantic_tokens_generated_total
- semantic_tokens_per_second
- semantic_memory_used_bytes
- semantic_cache_hit_total
- semantic_cache_miss_total
- semantic_pool_utilization_ratio
- semantic_eviction_total
- semantic_agents_active
- semantic_model_swap_duration_seconds
- semantic_cache_persist_duration_seconds

---

## Developer Input - What Needs Integration

### Backend Dev
> "Sprint 6 technical debt is critical. We can't deploy to production until graceful shutdown is properly implemented. The `drain()` method needs to be in BatchEngine with proper request tracking."

**Concern**: Graceful shutdown affects multiple components:
- BatchEngine needs `drain()` method
- API server needs to call it during lifespan shutdown
- Admin API model swap needs to use it
- Tests need to validate it works

**Proposal**: Make graceful shutdown Week 1 Priority 1

### SysE (Systems Engineer)
> "Observability is essential but we're building on a shaky foundation if stress tests aren't passing. How can we instrument metrics for pool exhaustion if we've never validated pool exhaustion behavior?"

**Concern**: Can't properly instrument what we haven't tested

**Proposal**: Fix stress tests BEFORE implementing Prometheus metrics. Metrics should measure known behaviors.

### ML Engineer
> "The performance metrics gap is critical. We need latency distribution (p50, p95, p99) before we can set meaningful alerting thresholds. You can't alert on metrics you haven't measured."

**Concern**: Alerting thresholds require baseline data

**Proposal**: Week 1 should include performance measurement. Use that data to configure alerts in Week 2.

### OSS Engineer
> "The packaging work (CLI, pip install, CHANGELOG, LICENSE, SBOM) is important but it's independent of the technical debt. We can work on that in parallel."

**Proposal**: OSS work can proceed independently while Backend/SysE fix technical debt

### PM (Product Manager)
> "User feedback was clear: they want streaming and realistic concurrency, not extreme stress testing. But we do need to validate graceful degradation. Let's be pragmatic: 1-2 stress tests validating pool exhaustion is enough, not all 12."

**Proposal**: Pragmatic approach - validate key scenarios, not exhaustive stress testing

---

## Integration Strategy - Developer Consensus

### Week 1: Foundation Hardening (Technical Debt + Core Observability)

**Goals**:
1. Resolve ALL Sprint 6 technical debt
2. Implement core observability (health, basic metrics, logging)
3. Validate system behavior under realistic load

**Why Week 1 First**:
- Can't instrument (metrics) what we haven't validated (stress tests)
- Can't set alerts without baseline performance data
- Can't claim production-ready without graceful shutdown

### Week 2: Advanced Observability + Production Packaging

**Goals**:
1. Full Prometheus metrics catalog (15+ metrics)
2. OpenTelemetry tracing
3. Alerting thresholds (based on Week 1 data)
4. Production packaging (CLI, pip, SBOM)
5. Documentation and release prep

**Why Week 2 Second**:
- Builds on Week 1 validated foundation
- Uses Week 1 performance data for alert thresholds
- OSS work can proceed in parallel

---

## Detailed Week 1 Plan (Days 0-4)

### Day 0: Graceful Shutdown + Health Endpoints

**Morning**: Implement `drain()` in BatchEngine
- Add request tracking to BatchEngine
- Implement `drain(timeout_seconds: int)` method
- Track in-flight requests with UIDs
- Timeout handling and forced termination

**Afternoon**: Update api_server.py shutdown
- Replace `asyncio.sleep(2)` with `batch_engine.drain(30)`
- Add health endpoints:
  - `/health/live` - Server process alive (200 always)
  - `/health/ready` - Ready to accept requests (503 if >90% pool or shutting down)
  - `/health/startup` - Startup complete (503 during model load)
- Test graceful shutdown with active requests

**Exit Criteria**:
- [ ] BatchEngine has working `drain()` method
- [ ] api_server.py calls `drain()` on shutdown
- [ ] 3-tier health endpoints implemented
- [ ] E2E test validates graceful shutdown

**Estimated**: 6-8 hours

---

### Day 1: Stress Tests + Performance Measurement

**Morning**: Debug and execute stress tests
- Debug aiohttp async integration issues
- Fix test_pool_exhaustion.py
- Execute at least 2 stress tests:
  - test_pool_exhaustion (validate 429 behavior)
  - test_concurrent_agents (measure latency under load)

**Afternoon**: Measure performance metrics
- Add latency tracking middleware to API server
- Measure p50, p95, p99 latency under various loads
- Document baseline performance characteristics
- Create performance baseline report

**Exit Criteria**:
- [ ] At least 2 stress tests passing
- [ ] Pool exhaustion returns 429 (validated)
- [ ] Latency distribution measured and documented
- [ ] Baseline performance report created

**Estimated**: 8-10 hours

---

### Day 2: Structured Logging + Request Middleware

**Morning**: Implement structured logging
- Add structlog dependency
- Create logging configuration with processors:
  - add_log_level
  - add_timestamp
  - JSONRenderer for production
  - ConsoleRenderer for development
- Add logging to all major operations:
  - Request start/end
  - Batch submission/completion
  - Cache operations
  - Model swap events

**Afternoon**: Request tracking middleware
- Create RequestTrackingMiddleware
- Track active requests (for graceful shutdown)
- Add request ID to all log messages
- Return 503 when server is shutting down
- Log request duration and status

**Exit Criteria**:
- [ ] structlog configured and working
- [ ] JSON logging in production mode
- [ ] RequestTrackingMiddleware implemented
- [ ] 503 returned during shutdown
- [ ] All major operations logged with request_id

**Estimated**: 6-8 hours

---

### Day 3: Basic Prometheus Metrics

**Morning**: Prometheus integration
- Add prometheus_client dependency
- Create metrics registry
- Implement core metrics (5 essential):
  - semantic_request_total (counter)
  - semantic_request_duration_seconds (histogram)
  - semantic_pool_utilization_ratio (gauge)
  - semantic_cache_hit_total (counter)
  - semantic_cache_miss_total (counter)
- Add /metrics endpoint

**Afternoon**: Instrument API endpoints
- Add metrics to message creation endpoints
- Add metrics to batch engine
- Add metrics to cache store
- Test metrics export

**Exit Criteria**:
- [ ] prometheus_client integrated
- [ ] /metrics endpoint working
- [ ] 5 core metrics instrumented
- [ ] Metrics validated with curl /metrics

**Estimated**: 6-8 hours

---

### Day 4: Code Quality + Documentation

**Morning**: Fix code quality issues
- Fix ruff B904 (exception chaining): `raise ... from e`
- Fix ruff E501 (line length)
- Configure mypy properly (fix module import conflict)
- Run full quality checks: `ruff check src/ tests/`
- Run type checks: `mypy src/semantic`

**Afternoon**: Documentation
- Create SPRINT_6_E2E_TESTING_GUIDE.md (consolidated)
- Update SPRINT_7 progress documentation
- Document Week 1 performance baselines
- Create Week 2 detailed plan

**Exit Criteria**:
- [ ] Zero ruff errors
- [ ] mypy --strict passing
- [ ] E2E testing guide complete
- [ ] Week 1 complete, Week 2 ready to start

**Estimated**: 6-8 hours

---

## Detailed Week 2 Plan (Days 5-9)

### Day 5: Extended Prometheus Metrics

**Goal**: Complete full 15+ metrics catalog

**Metrics to Add** (10 more):
- semantic_time_to_first_token_seconds (histogram)
- semantic_request_queue_depth (gauge)
- semantic_batch_size (histogram)
- semantic_tokens_generated_total (counter)
- semantic_tokens_per_second (gauge)
- semantic_memory_used_bytes (gauge)
- semantic_eviction_total (counter, label: tier)
- semantic_agents_active (gauge)
- semantic_model_swap_duration_seconds (histogram)
- semantic_cache_persist_duration_seconds (histogram)

**Exit Criteria**:
- [ ] 15+ Prometheus metrics implemented
- [ ] All metrics documented
- [ ] Metrics validation tests passing

**Estimated**: 6-8 hours

---

### Day 6: OpenTelemetry Tracing

**Goal**: Add distributed tracing

**Implementation**:
- Add opentelemetry-api and opentelemetry-sdk
- Configure OTLP exporter (to Jaeger or console)
- Add trace spans:
  - Request span (entire request lifecycle)
  - Batch submission span
  - Inference span
  - Cache operations span
- Add trace context propagation
- Test with Jaeger UI (optional, dev environment)

**Exit Criteria**:
- [ ] OpenTelemetry integrated
- [ ] Trace spans for key operations
- [ ] OTLP exporter configured
- [ ] Traces visible in console or Jaeger

**Estimated**: 6-8 hours

---

### Day 7: Alerting + Log Retention

**Morning**: Define alerting thresholds
- Create alerts.yml (Prometheus alert rules):
  - PoolUtilizationHigh (>90% for 5 min)
  - CacheEvictionRateHigh (>threshold/min)
  - ModelSwapFailure (any failure)
  - ErrorRateHigh (5xx >5% for 5 min)
  - RequestLatencyHigh (p95 >2s for 5 min)
- Document alert runbooks

**Afternoon**: Log retention policy
- Implement log rotation (daily or 100MB, whichever first)
- Configure retention (7 days local, 30 days remote)
- Add log compression for archived logs
- Document log management policy

**Exit Criteria**:
- [ ] alerts.yml created with 5+ rules
- [ ] Alert runbooks documented
- [ ] Log rotation implemented
- [ ] Log retention policy documented

**Estimated**: 6-8 hours

---

### Day 8: CLI + pip Package

**Morning**: CLI entrypoint
- Create `src/semantic/__main__.py`
- Implement `semantic serve` command
- Add flags: --host, --port, --allow-remote, --log-level
- Test: `python -m semantic serve --help`

**Afternoon**: pip-installable package
- Configure Hatchling in pyproject.toml
- Set version, dependencies, entry points
- Test: `pip install -e .`
- Test: `semantic serve` (from PATH)

**Exit Criteria**:
- [ ] `python -m semantic serve` working
- [ ] CLI flags functional
- [ ] Package installable via pip
- [ ] `semantic` command in PATH

**Estimated**: 6-8 hours

---

### Day 9: OSS Compliance + Release Prep

**Morning**: License compliance
- Add LICENSE file (Apache 2.0 or MIT)
- Add NOTICE file (dependencies attribution)
- Add CHANGELOG.md (from sprints)
- Run liccheck for compliance
- Generate SBOM:
  - syft for dependency scanning
  - Export as CycloneDX 1.6 format

**Afternoon**: Release notes + final docs
- Create RELEASE_NOTES.md for Sprint 7
- Update README.md with observability features
- Document Prometheus metrics in docs/
- Document OpenTelemetry setup in docs/
- Final Sprint 7 completion report

**Exit Criteria**:
- [ ] LICENSE, NOTICE, CHANGELOG.md created
- [ ] liccheck passing
- [ ] SBOM generated (CycloneDX 1.6)
- [ ] Release notes complete
- [ ] Documentation updated

**Estimated**: 6-8 hours

---

## Sprint 7 Test Plan

### New Tests Required

**Week 1**:
1. test_graceful_shutdown_with_active_requests.py (E2E)
2. test_health_endpoints.py (integration)
3. test_pool_exhaustion.py (stress) - FIX AND RUN
4. test_concurrent_agents.py (stress) - FIX AND RUN
5. test_structured_logging.py (unit)
6. test_request_tracking_middleware.py (integration)
7. test_prometheus_metrics.py (integration)

**Week 2**:
8. test_opentelemetry_tracing.py (integration)
9. test_cli_entrypoint.py (integration)
10. test_pip_install.py (E2E)

**Total New Tests**: ~15-20

---

## Quality Gates

### Week 1 Exit Criteria
- [ ] All Sprint 6 technical debt resolved
- [ ] At least 2 stress tests passing
- [ ] Latency baseline documented
- [ ] Graceful shutdown working
- [ ] Health endpoints operational
- [ ] Structured logging implemented
- [ ] Basic Prometheus metrics (5+)
- [ ] Zero ruff/mypy errors

### Week 2 Exit Criteria
- [ ] 15+ Prometheus metrics
- [ ] OpenTelemetry tracing working
- [ ] Alerting thresholds defined
- [ ] Log retention policy implemented
- [ ] CLI entrypoint functional
- [ ] pip-installable package
- [ ] LICENSE, CHANGELOG, SBOM complete
- [ ] Release notes written

### Sprint 7 Final Exit Criteria
- [ ] Technical Fellows score >90/100
- [ ] Production deployment approved (heavy load)
- [ ] Zero critical technical debt
- [ ] All observability features working
- [ ] Package ready for PyPI (optional)

---

## Risk Assessment

### High Risks

**Risk 1**: Stress test debugging takes longer than planned
- **Mitigation**: Cap at 2 tests (pool exhaustion, concurrent agents)
- **Fallback**: Document async issues, defer remaining 10 tests post-Sprint 7

**Risk 2**: OpenTelemetry integration complex
- **Mitigation**: Use console exporter (simplest)
- **Fallback**: Implement basic tracing, defer Jaeger integration

**Risk 3**: SBOM generation tooling issues
- **Mitigation**: Use syft (well-maintained)
- **Fallback**: Manual SBOM if tooling fails

### Medium Risks

**Risk 4**: Prometheus metrics count (15+) is ambitious
- **Mitigation**: Prioritize 10 most critical metrics
- **Fallback**: Deliver 10-12 metrics instead of 15+

---

## Resource Allocation

**Estimated Total Effort**: 60-80 hours (1.5-2 weeks)

**Week 1** (Foundation): 32-42 hours
- Day 0: 6-8h (graceful shutdown + health)
- Day 1: 8-10h (stress tests + perf measurement)
- Day 2: 6-8h (logging + middleware)
- Day 3: 6-8h (basic metrics)
- Day 4: 6-8h (code quality + docs)

**Week 2** (Observability): 28-38 hours
- Day 5: 6-8h (extended metrics)
- Day 6: 6-8h (OpenTelemetry)
- Day 7: 6-8h (alerting + log retention)
- Day 8: 6-8h (CLI + pip)
- Day 9: 6-8h (OSS compliance + release)

---

## Dependencies

**External**:
- prometheus_client (metrics)
- structlog (logging)
- opentelemetry-api, opentelemetry-sdk (tracing)
- hatchling (packaging, already in dev dependencies)
- liccheck (license compliance)
- syft (SBOM generation)

**Internal**:
- Sprint 6 codebase (foundation)
- Existing test infrastructure

---

## Success Metrics

**Quantitative**:
- Technical Fellows score: >90/100
- Test coverage: >85% (maintain from Sprint 5)
- Prometheus metrics: 15+ implemented
- Health endpoints: 3 implemented
- Stress tests: 2+ passing
- Code quality: Zero ruff/mypy errors

**Qualitative**:
- Production deployment approved for heavy load
- Observability enables debugging production issues
- Graceful shutdown prevents data loss
- Package ready for external distribution

---

## Developer Commitments

**Backend Dev**:
- Days 0-1: Graceful shutdown + stress tests
- Day 2: Request middleware
- Day 8: CLI entrypoint

**SysE**:
- Day 1: Performance measurement
- Days 2-3: Logging + basic metrics
- Days 5-7: Extended metrics + tracing + alerting

**ML Engineer**:
- Day 1: Performance baselines
- Days 5-6: Inference metrics + tracing
- Day 7: Performance alert thresholds

**OSS Engineer**:
- Day 4: Code quality fixes
- Days 8-9: Packaging + compliance + release

**PM**:
- Day 4: Week 1 review
- Day 9: Sprint 7 completion review

---

## Next Steps

1. ✅ Developer consensus on plan
2. ⏭️ Technical Fellows review of plan
3. ⏭️ Start Sprint 7 execution
4. ⏭️ Daily standup rhythm (morning plan, evening review)

---

**Meeting Outcome**: CONSENSUS REACHED ✅

**Plan Status**: Ready for Technical Fellows review

**Start Date**: Pending Fellows approval

**Estimated Completion**: 2 weeks (10 working days)
