# Sprint 7: Observability + Hardening - Detailed Execution Plan

**Duration**: 10 days (Day 0-9)
**Starting Point**: Sprint 6 complete (88/100, technical debt documented)
**Deliverable**: Production-ready observability, graceful shutdown, performance validation, pip package
**Exit Criteria**: Technical Fellows score >90/100, approved for heavy production load
**Last Updated**: 2026-01-25

---

## Executive Summary

Sprint 7 integrates Sprint 6 technical debt resolution with the original observability and hardening plan. This creates a production-ready system with comprehensive monitoring, graceful degradation, and proper operational tooling.

### Integration Strategy

**Week 1 (Days 0-4)**: Foundation Hardening
- Resolve ALL Sprint 6 technical debt
- Implement core observability (health, logging, basic metrics)
- Validate system under realistic load

**Week 2 (Days 5-9)**: Advanced Observability + Production Packaging
- Full Prometheus metrics catalog (15+)
- OpenTelemetry tracing
- Alerting thresholds
- Production packaging (CLI, pip, SBOM)

### Why This Order?

1. Can't instrument (metrics) what we haven't validated (stress tests)
2. Can't set alerts without baseline performance data
3. Can't claim production-ready without graceful shutdown
4. Week 2 builds on Week 1 validated foundation

---

## Sprint 6 Technical Debt - Integration

### Mandatory Items (From Technical Fellows Review)

1. **Graceful Shutdown** (Day 0)
   - Current: `asyncio.sleep(2)` (temporary)
   - Required: `batch_engine.drain(timeout_seconds=30)`
   - Impact: CRITICAL - in-flight requests may terminate abruptly

2. **Stress Tests** (Day 1)
   - Current: 0/12 executed (framework complete, async issues)
   - Required: At least 2 passing (pool exhaustion, concurrent agents)
   - Impact: HIGH - unknown behavior under load

3. **Performance Metrics** (Day 1)
   - Current: Missing latency distribution (p50, p95, p99)
   - Required: Baseline performance characteristics
   - Impact: HIGH - can't set meaningful alerts without baselines

4. **Code Quality** (Day 4)
   - Current: Ruff B904, E501 warnings; mypy config issue
   - Required: Zero errors
   - Impact: MEDIUM - prevents CI green

5. **Documentation** (Day 4)
   - Current: Individual test READMEs
   - Required: Unified E2E testing guide
   - Impact: LOW - developer onboarding

---

## Original Sprint 7 Plan - Integration

### Core Deliverables (2-Week Plan)

**Observability**:
- 3-tier health endpoints (/health/live, /health/ready, /health/startup)
- 15+ Prometheus metrics
- Structured JSON logging (structlog)
- OpenTelemetry tracing
- Alerting thresholds
- Log retention and rotation policy

**Hardening**:
- Graceful shutdown (multi-phase drain)
- Runtime-configurable batch window (admin API)
- Request middleware (active tracking, 503 on shutdown)

**Production Packaging**:
- CLI entrypoint (`python -m semantic serve`)
- pip-installable package (Hatchling)
- CHANGELOG, LICENSE, NOTICE, release notes
- License compliance check (liccheck)
- SBOM generation (syft + CycloneDX 1.6)

---

## Day-by-Day Execution Plan

### Day 0: Graceful Shutdown + 3-Tier Health Endpoints

**Goal**: Implement proper graceful shutdown and production health checks

**Morning: Implement `drain()` in BatchEngine** (4-5 hours)

Tasks:
1. Add request tracking to BlockPoolBatchEngine:
   ```python
   class BlockPoolBatchEngine:
       def __init__(self, ...):
           self._active_requests: dict[str, CompletionResult] = {}

       def submit(self, ...) -> str:
           uid = uuid.uuid4().hex
           self._active_requests[uid] = None  # Track until complete
           return uid

       async def drain(self, timeout_seconds: int = 30) -> int:
           """Drain pending requests with timeout.

           Returns:
               Number of requests drained successfully
           """
           start_time = time.time()
           drained_count = 0

           while self._active_requests and (time.time() - start_time < timeout_seconds):
               # Continue stepping to complete active requests
               for result in self.step():
                   if result.uid in self._active_requests:
                       self._active_requests.pop(result.uid)
                       drained_count += 1
               await asyncio.sleep(0.1)  # Brief pause between steps

           # Log any requests that timed out
           if self._active_requests:
               logger.warning(f"{len(self._active_requests)} requests timed out during drain")

           return drained_count
   ```

2. Update api_server.py lifespan shutdown:
   ```python
   # Replace asyncio.sleep(2) with proper drain
   logger.info("Draining pending requests...")
   if batch_engine:
       drained = await batch_engine.drain(timeout_seconds=30)
       logger.info(f"Drained {drained} requests")
   ```

3. Create test: `tests/e2e/test_graceful_shutdown.py`:
   - Start server, submit 3 requests
   - Trigger shutdown while requests in-flight
   - Verify all 3 complete before shutdown
   - Verify cache saved

**Afternoon: 3-Tier Health Endpoints** (2-3 hours)

Tasks:
1. Replace simple `/health` with 3 endpoints in api_server.py:

```python
@app.get("/health/live")
async def health_live():
    """Liveness probe - process alive.

    Returns 200 if server process is running.
    Kubernetes liveness probe should use this.
    """
    return {"status": "alive"}

@app.get("/health/ready")
async def health_ready(response: Response):
    """Readiness probe - ready to accept requests.

    Returns:
        200 if ready, 503 if not ready (high pool utilization or shutting down)

    Kubernetes readiness probe should use this.
    """
    # Check pool utilization
    pool = app.state.semantic.block_pool if hasattr(app.state, "semantic") else None

    if not pool:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {"status": "not_ready", "reason": "pool_not_initialized"}

    utilization = 1.0 - (pool.available_blocks() / pool.total_blocks)

    # Check if shutting down
    if getattr(app.state, "shutting_down", False):
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {"status": "not_ready", "reason": "shutting_down"}

    # Check pool exhaustion
    if utilization > 0.9:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {
            "status": "not_ready",
            "reason": "pool_near_exhaustion",
            "pool_utilization": round(utilization * 100, 1),
        }

    return {"status": "ready", "pool_utilization": round(utilization * 100, 1)}

@app.get("/health/startup")
async def health_startup(response: Response):
    """Startup probe - initialization complete.

    Returns:
        200 if startup complete, 503 if still initializing

    Kubernetes startup probe should use this.
    """
    # Check if model loaded
    if not hasattr(app.state, "semantic") or not app.state.semantic.batch_engine:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {"status": "starting", "reason": "model_loading"}

    return {"status": "started"}
```

2. Add shutdown flag to lifespan:
   ```python
   @asynccontextmanager
   async def lifespan(app: FastAPI):
       # Startup...
       app.state.shutting_down = False
       yield
       # Shutdown...
       app.state.shutting_down = True  # Signal to health checks
       # ... drain and save ...
   ```

3. Create test: `tests/integration/test_health_endpoints.py`:
   - test_health_live_always_200
   - test_health_ready_503_when_pool_exhausted
   - test_health_ready_503_when_shutting_down
   - test_health_startup_503_until_model_loaded

**Deliverables**:
- `src/semantic/application/batch_engine.py` - drain() method
- `src/semantic/entrypoints/api_server.py` - 3-tier health endpoints
- `tests/e2e/test_graceful_shutdown.py` (1 E2E test)
- `tests/integration/test_health_endpoints.py` (4 tests)

**Exit Criteria**:
- [ ] BatchEngine has working `drain()` method
- [ ] api_server.py calls `drain()` on shutdown
- [ ] /health/live, /health/ready, /health/startup implemented
- [ ] 5 tests passing (1 E2E + 4 integration)

**Estimated**: 6-8 hours

---

### Day 1: Stress Tests + Performance Baselines

**Goal**: Execute stress tests and establish performance baselines

**Morning: Debug and Execute Stress Tests** (4-5 hours)

Tasks:
1. Debug async HTTP client integration in stress test harness:
   - Review aiohttp ClientSession usage
   - Fix connection pooling issues
   - Ensure proper async/await throughout
   - Test with simple HTTP endpoint first

2. Fix and execute `test_pool_exhaustion.py`:
   - test_100_plus_concurrent_requests
   - test_graceful_429_when_pool_exhausted

3. Fix and execute `test_concurrent_agents.py`:
   - test_10_agents_50_rapid_requests
   - test_latency_remains_acceptable

4. Document async issues if debugging takes >3 hours:
   - Cap debugging effort
   - Document root cause
   - Defer remaining tests if needed

**Afternoon: Performance Measurement** (3-4 hours)

Tasks:
1. Create latency tracking middleware:
```python
# src/semantic/adapters/inbound/latency_middleware.py

import time
from typing import Callable

from fastapi import Request, Response

class LatencyTrackingMiddleware:
    """Middleware to track request latency."""

    def __init__(self, app):
        self.app = app
        self.latencies: list[float] = []  # In-memory for now

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        start_time = time.time()

        await self.app(scope, receive, send)

        duration = time.time() - start_time
        self.latencies.append(duration)

        return
```

2. Run load tests and collect latency data:
   - 1 agent, 100 requests: Measure p50, p95, p99
   - 5 agents, 100 requests each: Measure p50, p95, p99
   - 10 agents, 50 requests each: Measure p50, p95, p99

3. Calculate percentiles:
   ```python
   import numpy as np
   p50 = np.percentile(latencies, 50)
   p95 = np.percentile(latencies, 95)
   p99 = np.percentile(latencies, 99)
   ```

4. Create `project/sprints/SPRINT_7_PERFORMANCE_BASELINES.md`:
   - Document latency distributions
   - Document throughput (requests/sec)
   - Document pool utilization under load
   - Document cache hit rates
   - Use this data for alert thresholds in Day 7

**Deliverables**:
- Fixed stress tests (at least 2 passing)
- LatencyTrackingMiddleware
- SPRINT_7_PERFORMANCE_BASELINES.md

**Exit Criteria**:
- [ ] At least 2 stress tests passing
- [ ] Pool exhaustion returns 429 (validated)
- [ ] Latency baselines documented (p50, p95, p99)
- [ ] Performance baseline report created

**Estimated**: 8-10 hours

---

### Day 2: Structured Logging + Request Middleware

**Goal**: Implement production-grade logging and request tracking

**Morning: Structured JSON Logging** (3-4 hours)

Tasks:
1. Add structlog dependency:
   ```toml
   [project.dependencies]
   structlog = "^24.1.0"
   ```

2. Create logging configuration in `src/semantic/adapters/config/logging_config.py`:
```python
import logging
import sys

import structlog

def configure_logging(log_level: str = "INFO", json_logs: bool = False):
    """Configure structured logging.

    Args:
        log_level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        json_logs: If True, use JSON renderer; else use console renderer
    """
    processors = [
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]

    if json_logs:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Configure standard logging to use structlog
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level.upper()),
    )
```

3. Initialize in api_server.py:
```python
from semantic.adapters.config.logging_config import configure_logging

def create_app() -> FastAPI:
    settings = get_settings()

    # Configure structured logging
    json_logs = settings.server.log_level == "PRODUCTION"
    configure_logging(log_level=settings.server.log_level, json_logs=json_logs)

    logger = structlog.get_logger(__name__)
    logger.info("Creating FastAPI application")
    # ...
```

4. Update key operations to use structlog:
```python
logger = structlog.get_logger(__name__)

# In batch engine submit:
logger.info(
    "batch_submit",
    agent_id=agent_id,
    prompt_length=len(prompt),
    max_tokens=max_tokens,
    cached_blocks=cached_blocks.total_tokens if cached_blocks else 0,
)

# In cache operations:
logger.info(
    "cache_save",
    agent_id=agent_id,
    total_tokens=blocks.total_tokens,
    duration_ms=duration * 1000,
)
```

**Afternoon: Request Tracking Middleware** (3-4 hours)

Tasks:
1. Create `src/semantic/adapters/inbound/request_tracking.py`:
```python
import time
import uuid
from typing import Callable

import structlog
from fastapi import Request, Response, status
from fastapi.responses import JSONResponse

logger = structlog.get_logger(__name__)

class RequestTrackingMiddleware:
    """Middleware to track active requests and reject requests during shutdown."""

    def __init__(self, app):
        self.app = app
        self.active_requests: dict[str, float] = {}  # request_id -> start_time

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        # Generate request ID
        request_id = uuid.uuid4().hex[:16]
        scope["request_id"] = request_id

        # Check if shutting down
        app_state = scope.get("app")
        if app_state and getattr(app_state.state, "shutting_down", False):
            # Return 503 Service Unavailable
            response = JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={"error": "Server is shutting down"},
            )
            await response(scope, receive, send)
            return

        # Track request
        start_time = time.time()
        self.active_requests[request_id] = start_time

        logger.info(
            "request_start",
            request_id=request_id,
            method=scope["method"],
            path=scope["path"],
        )

        try:
            await self.app(scope, receive, send)
        finally:
            # Untrack request
            duration = time.time() - start_time
            self.active_requests.pop(request_id, None)

            logger.info(
                "request_end",
                request_id=request_id,
                duration_ms=round(duration * 1000, 2),
            )

    def get_active_count(self) -> int:
        """Get number of active requests."""
        return len(self.active_requests)
```

2. Register in api_server.py:
```python
from semantic.adapters.inbound.request_tracking import RequestTrackingMiddleware

def create_app() -> FastAPI:
    # ...
    app.add_middleware(RequestTrackingMiddleware)
```

3. Create tests: `tests/integration/test_request_tracking.py`:
   - test_request_id_added_to_logs
   - test_503_during_shutdown
   - test_active_request_count
   - test_request_duration_logged

**Deliverables**:
- structlog configured
- logging_config.py
- RequestTrackingMiddleware
- Updated logging throughout codebase
- test_request_tracking.py (4 tests)

**Exit Criteria**:
- [ ] structlog configured and working
- [ ] JSON logging in production mode
- [ ] RequestTrackingMiddleware implemented
- [ ] 503 returned during shutdown
- [ ] All major operations logged with request_id
- [ ] 4 tests passing

**Estimated**: 6-8 hours

---

### Day 3: Basic Prometheus Metrics

**Goal**: Implement core Prometheus metrics and /metrics endpoint

**Morning: Prometheus Integration** (3-4 hours)

Tasks:
1. Add prometheus_client dependency:
   ```toml
   [project.dependencies]
   prometheus-client = "^0.20.0"
   ```

2. Create metrics registry in `src/semantic/adapters/outbound/prometheus_metrics.py`:
```python
from prometheus_client import Counter, Gauge, Histogram

# Core metrics (5 essential)
request_total = Counter(
    "semantic_request_total",
    "Total number of requests",
    ["model", "status"],
)

request_duration = Histogram(
    "semantic_request_duration_seconds",
    "Request duration in seconds",
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
)

pool_utilization = Gauge(
    "semantic_pool_utilization_ratio",
    "Pool utilization ratio (0-1)",
)

cache_hit_total = Counter(
    "semantic_cache_hit_total",
    "Total number of cache hits",
)

cache_miss_total = Counter(
    "semantic_cache_miss_total",
    "Total number of cache misses",
)
```

3. Add /metrics endpoint in api_server.py:
```python
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
```

**Afternoon: Instrument API Endpoints** (3-4 hours)

Tasks:
1. Instrument message creation endpoints:
```python
# In anthropic_adapter.py and openai_adapter.py

from semantic.adapters.outbound.prometheus_metrics import (
    request_total,
    request_duration,
    cache_hit_total,
    cache_miss_total,
)

@router.post("/v1/messages")
async def create_message(...):
    start_time = time.time()

    try:
        # ... existing logic ...

        # Track cache hit/miss
        if cached_blocks:
            cache_hit_total.inc()
        else:
            cache_miss_total.inc()

        # ... complete request ...

        request_total.labels(model=request_body.model, status="success").inc()

    except Exception as e:
        request_total.labels(model=request_body.model, status="error").inc()
        raise
    finally:
        duration = time.time() - start_time
        request_duration.observe(duration)
```

2. Instrument pool utilization (periodic update in lifespan):
```python
# In api_server.py lifespan

async def update_pool_metrics():
    """Background task to update pool utilization gauge."""
    while not app.state.shutting_down:
        pool = app.state.semantic.block_pool
        utilization = 1.0 - (pool.available_blocks() / pool.total_blocks)
        pool_utilization.set(utilization)
        await asyncio.sleep(5)  # Update every 5 seconds

# In lifespan startup:
asyncio.create_task(update_pool_metrics())
```

3. Create tests: `tests/integration/test_prometheus_metrics.py`:
   - test_metrics_endpoint_accessible
   - test_request_total_increments
   - test_cache_hit_miss_tracked
   - test_pool_utilization_updates

**Deliverables**:
- prometheus_metrics.py (5 metrics)
- /metrics endpoint
- Instrumented API endpoints
- test_prometheus_metrics.py (4 tests)

**Exit Criteria**:
- [ ] prometheus_client integrated
- [ ] /metrics endpoint working
- [ ] 5 core metrics instrumented
- [ ] Metrics validated with curl /metrics
- [ ] 4 tests passing

**Estimated**: 6-8 hours

---

### Day 4: Code Quality + Week 1 Documentation

**Goal**: Fix all code quality issues and document Week 1 completion

**Morning: Fix Code Quality Issues** (3-4 hours)

Tasks:
1. Fix ruff B904 (exception chaining):
```python
# Find all instances of:
except Exception as e:
    raise HTTPException(...) from e  # Add 'from e'
```

2. Fix ruff E501 (line length) in settings.py:
```python
cors_origins: str = Field(
    default="http://localhost:3000",
    description=(
        "Comma-separated list of allowed CORS origins "
        "(* for all, not recommended for production)"
    ),
)
```

3. Fix mypy configuration:
```toml
# In pyproject.toml
[tool.mypy]
python_version = "3.11"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
exclude = ["tests/"]

[[tool.mypy.overrides]]
module = "mlx_lm.*"
ignore_missing_imports = true
```

4. Run quality checks:
```bash
ruff check src/ tests/
mypy src/semantic
```

5. Fix any remaining issues

**Afternoon: Week 1 Documentation** (3-4 hours)

Tasks:
1. Create `project/sprints/SPRINT_6_E2E_TESTING_GUIDE.md`:
   - Prerequisites (Apple Silicon, MLX, models)
   - Running smoke tests
   - Running E2E tests
   - Running stress tests
   - Running benchmarks
   - Interpreting results
   - Adding new tests
   - Troubleshooting common issues

2. Create `project/sprints/SPRINT_7_WEEK_1_COMPLETION.md`:
   - Week 1 deliverables summary
   - All Sprint 6 technical debt resolved
   - Stress test results
   - Performance baseline data
   - Health endpoint documentation
   - Structured logging examples
   - Prometheus metrics catalog (5 metrics)
   - Code quality status (zero errors)

3. Create detailed Week 2 plan in `project/sprints/SPRINT_7_WEEK_2_PLAN.md`

**Deliverables**:
- Zero ruff/mypy errors
- SPRINT_6_E2E_TESTING_GUIDE.md
- SPRINT_7_WEEK_1_COMPLETION.md
- SPRINT_7_WEEK_2_PLAN.md

**Exit Criteria**:
- [ ] Zero ruff errors
- [ ] mypy --strict passing
- [ ] E2E testing guide complete
- [ ] Week 1 completion documented
- [ ] Week 2 plan ready

**Estimated**: 6-8 hours

---

## Week 1 Summary

**Total Estimated Effort**: 32-42 hours (5 working days)

**Deliverables**:
- ✅ Graceful shutdown with drain()
- ✅ 3-tier health endpoints
- ✅ 2+ stress tests passing
- ✅ Performance baselines documented
- ✅ Structured JSON logging
- ✅ Request tracking middleware
- ✅ Basic Prometheus metrics (5)
- ✅ Zero code quality issues
- ✅ Complete documentation

**Tests Added**: ~15 tests
- 1 E2E (graceful shutdown)
- 8 integration (health, tracking, metrics)
- 2+ stress (pool exhaustion, concurrent agents)

**Sprint 6 Technical Debt**: 100% RESOLVED ✅

---

## Week 2 Plan (Days 5-9)

### Day 5: Extended Prometheus Metrics

**Goal**: Complete full 15+ metrics catalog

**Tasks**:
1. Add 10 more Prometheus metrics:
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

2. Instrument throughout codebase:
   - Batch engine: batch_size, tokens_generated, tokens_per_second
   - Cache store: eviction_total, cache_persist_duration
   - Agent tracking: agents_active
   - Model swap: model_swap_duration
   - Queue: request_queue_depth

3. Create tests: test_extended_prometheus_metrics.py

4. Document all 15+ metrics in `docs/prometheus_metrics.md`

**Exit Criteria**:
- [ ] 15+ Prometheus metrics implemented
- [ ] All metrics documented
- [ ] Metrics validation tests passing

**Estimated**: 6-8 hours

---

### Day 6: OpenTelemetry Tracing

**Goal**: Add distributed tracing for request flows

**Tasks**:
1. Add OpenTelemetry dependencies:
   ```toml
   [project.dependencies]
   opentelemetry-api = "^1.22.0"
   opentelemetry-sdk = "^1.22.0"
   opentelemetry-exporter-otlp = "^1.22.0"
   ```

2. Configure OpenTelemetry in `src/semantic/adapters/config/tracing_config.py`:
```python
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

def configure_tracing(service_name: str = "semantic-cache", use_console: bool = True):
    """Configure OpenTelemetry tracing.

    Args:
        service_name: Service name for traces
        use_console: If True, export to console; else export to OTLP endpoint
    """
    resource = Resource(attributes={"service.name": service_name})

    provider = TracerProvider(resource=resource)

    if use_console:
        processor = BatchSpanProcessor(ConsoleSpanExporter())
    else:
        # Export to OTLP endpoint (Jaeger, Tempo, etc.)
        processor = BatchSpanProcessor(OTLPSpanExporter())

    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)
```

3. Add trace spans to key operations:
```python
from opentelemetry import trace

tracer = trace.get_tracer(__name__)

# In message creation:
with tracer.start_as_current_span("create_message") as span:
    span.set_attribute("model", request_body.model)
    span.set_attribute("max_tokens", request_body.max_tokens)

    with tracer.start_as_current_span("batch_submit"):
        uid = batch_engine.submit(...)

    with tracer.start_as_current_span("inference"):
        result = batch_engine.step()

    with tracer.start_as_current_span("cache_save"):
        cache_store.save(...)
```

4. Create tests: test_opentelemetry_tracing.py

**Exit Criteria**:
- [ ] OpenTelemetry integrated
- [ ] Trace spans for key operations
- [ ] OTLP exporter configured
- [ ] Traces visible in console or Jaeger

**Estimated**: 6-8 hours

---

### Day 7: Alerting Thresholds + Log Retention

**Goal**: Define production alerting and log management

**Morning: Alerting Thresholds** (3-4 hours)

Tasks:
1. Create `deployment/prometheus/alerts.yml`:
```yaml
groups:
  - name: semantic_cache_alerts
    interval: 30s
    rules:
      - alert: PoolUtilizationHigh
        expr: semantic_pool_utilization_ratio > 0.9
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Pool utilization >90% for 5 minutes"
          description: "Pool utilization is {{ $value | humanizePercentage }}"

      - alert: CacheEvictionRateHigh
        expr: rate(semantic_eviction_total[5m]) > 10
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Cache eviction rate >10/min for 5 minutes"

      - alert: ModelSwapFailure
        expr: semantic_model_swap_duration_seconds == -1
        labels:
          severity: critical
        annotations:
          summary: "Model swap failed"

      - alert: ErrorRateHigh
        expr: rate(semantic_request_total{status="error"}[5m]) / rate(semantic_request_total[5m]) > 0.05
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Error rate >5% for 5 minutes"

      - alert: RequestLatencyHigh
        expr: histogram_quantile(0.95, semantic_request_duration_seconds) > 2.0
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "p95 latency >2s for 5 minutes"
          description: "p95 latency is {{ $value }}s"
```

2. Create alert runbooks in `docs/runbooks/`:
   - `pool_utilization_high.md`
   - `cache_eviction_rate_high.md`
   - `model_swap_failure.md`
   - `error_rate_high.md`
   - `request_latency_high.md`

**Afternoon: Log Retention Policy** (3-4 hours)

Tasks:
1. Implement log rotation in logging_config.py:
```python
from logging.handlers import RotatingFileHandler

def configure_logging(log_level: str = "INFO", json_logs: bool = False, log_dir: str = "/var/log/semantic"):
    # ... existing config ...

    # Add rotating file handler
    file_handler = RotatingFileHandler(
        filename=f"{log_dir}/semantic.log",
        maxBytes=100 * 1024 * 1024,  # 100 MB
        backupCount=7,  # Keep 7 days
        encoding="utf-8",
    )

    logging.root.addHandler(file_handler)
```

2. Create log archival script in `scripts/archive_logs.sh`:
```bash
#!/bin/bash
# Archive and compress logs older than 7 days

LOG_DIR="/var/log/semantic"
ARCHIVE_DIR="/var/log/semantic/archive"

find "$LOG_DIR" -name "*.log.*" -mtime +7 -exec gzip {} \;
find "$LOG_DIR" -name "*.log.*.gz" -exec mv {} "$ARCHIVE_DIR" \;

# Delete archives older than 30 days
find "$ARCHIVE_DIR" -name "*.log.*.gz" -mtime +30 -delete
```

3. Document log retention policy in `docs/operations/log_management.md`

**Exit Criteria**:
- [ ] alerts.yml created with 5+ rules
- [ ] Alert runbooks documented
- [ ] Log rotation implemented
- [ ] Log retention policy documented

**Estimated**: 6-8 hours

---

### Day 8: CLI Entrypoint + pip Package

**Goal**: Make package installable and usable via CLI

**Morning: CLI Entrypoint** (3-4 hours)

Tasks:
1. Create `src/semantic/__main__.py`:
```python
import sys
import click
import uvicorn

from semantic.adapters.config.settings import get_settings

@click.group()
def cli():
    """Semantic caching server CLI."""
    pass

@cli.command()
@click.option("--host", default="127.0.0.1", help="Host to bind to")
@click.option("--port", default=8000, type=int, help="Port to bind to")
@click.option("--allow-remote", is_flag=True, help="Allow remote connections (binds to 0.0.0.0)")
@click.option("--log-level", default="INFO", help="Log level (DEBUG, INFO, WARNING, ERROR)")
@click.option("--workers", default=1, type=int, help="Number of worker processes")
def serve(host: str, port: int, allow_remote: bool, log_level: str, workers: int):
    """Start the semantic caching server."""
    if allow_remote:
        host = "0.0.0.0"

    click.echo(f"Starting semantic caching server on {host}:{port}")
    click.echo(f"Log level: {log_level}")
    click.echo(f"Workers: {workers}")

    uvicorn.run(
        "semantic.entrypoints.api_server:create_app",
        host=host,
        port=port,
        log_level=log_level.lower(),
        workers=workers,
        factory=True,
    )

if __name__ == "__main__":
    cli()
```

2. Add click dependency:
   ```toml
   [project.dependencies]
   click = "^8.1.0"
   ```

3. Test CLI:
   ```bash
   python -m semantic serve --help
   python -m semantic serve --port 8080
   ```

4. Create tests: test_cli_entrypoint.py

**Afternoon: pip-installable Package** (3-4 hours)

Tasks:
1. Update pyproject.toml for packaging:
```toml
[project.scripts]
semantic = "semantic.__main__:cli"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/semantic"]
```

2. Test installation:
   ```bash
   pip install -e .
   semantic serve --help
   which semantic
   ```

3. Create installation tests: test_pip_install.py

4. Document installation in README.md:
   ```markdown
   ## Installation

   ```bash
   pip install semantic-cache
   ```

   ## Usage

   ```bash
   semantic serve --host 0.0.0.0 --port 8000
   ```
   ```

**Exit Criteria**:
- [ ] `python -m semantic serve` working
- [ ] CLI flags functional
- [ ] Package installable via pip
- [ ] `semantic` command in PATH

**Estimated**: 6-8 hours

---

### Day 9: OSS Compliance + Release Documentation

**Goal**: Complete licensing, SBOM, and release documentation

**Morning: OSS Compliance** (3-4 hours)

Tasks:
1. Add LICENSE file (Apache 2.0):
   - Copy from https://www.apache.org/licenses/LICENSE-2.0.txt
   - Update copyright year and holder

2. Create NOTICE file:
   ```
   Semantic Cache
   Copyright 2026 [Your Name/Organization]

   This product includes software developed by:
   - MLX (Apple Inc.)
   - FastAPI (Sebastián Ramírez)
   - ... (other dependencies)
   ```

3. Generate CHANGELOG.md from sprints:
   ```markdown
   # Changelog

   ## [0.1.0] - 2026-01-25

   ### Added
   - Sprint 6: E2E testing, benchmarks, production hardening
   - Sprint 7: Observability, graceful shutdown, Prometheus metrics
   - 3-tier health endpoints
   - Structured JSON logging
   - OpenTelemetry tracing
   - 15+ Prometheus metrics
   - CLI entrypoint

   ### Fixed
   - Graceful shutdown implementation
   - Stress test execution
   - Code quality issues

   ### Performance
   - 1.6x batching speedup
   - <1ms cache operations
   - Constant memory usage
   ```

4. Add liccheck for compliance:
   ```toml
   [project.optional-dependencies]
   dev = [
       "liccheck>=0.9.0",
   ]
   ```

   ```bash
   pip install -e ".[dev]"
   liccheck --sfile strategy.ini
   ```

5. Generate SBOM with syft:
   ```bash
   # Install syft
   brew install syft  # or curl -sSfL https://raw.githubusercontent.com/anchore/syft/main/install.sh | sh

   # Generate SBOM
   syft packages . -o cyclonedx-json > sbom.cyclonedx.json
   ```

**Afternoon: Release Documentation** (3-4 hours)

Tasks:
1. Create RELEASE_NOTES.md:
   ```markdown
   # Release Notes - Sprint 7 (v0.1.0)

   ## Overview

   Sprint 7 delivers production-grade observability and hardening features.

   ## Key Features

   ### Observability
   - 3-tier health endpoints (/health/live, /health/ready, /health/startup)
   - 15+ Prometheus metrics
   - Structured JSON logging
   - OpenTelemetry distributed tracing
   - Alerting thresholds and runbooks

   ### Hardening
   - Graceful shutdown with configurable timeout
   - Request tracking middleware
   - 503 responses during shutdown
   - Log rotation and retention

   ### Operations
   - CLI entrypoint: `semantic serve`
   - pip-installable package
   - License compliance (Apache 2.0)
   - SBOM generation

   ## Performance

   (Include baselines from Day 1)

   ## Deployment

   (Include deployment instructions)

   ## Breaking Changes

   None

   ## Deprecations

   None

   ## Known Issues

   None
   ```

2. Update README.md:
   - Add observability features
   - Add installation instructions
   - Add usage examples
   - Add Prometheus metrics documentation
   - Add health endpoint documentation

3. Create `docs/prometheus_metrics.md` with full metrics catalog

4. Create `docs/opentelemetry_tracing.md` with tracing setup

5. Create final Sprint 7 completion report: `project/sprints/SPRINT_7_COMPLETION_REPORT.md`

**Exit Criteria**:
- [ ] LICENSE, NOTICE, CHANGELOG.md created
- [ ] liccheck passing
- [ ] SBOM generated (CycloneDX 1.6)
- [ ] RELEASE_NOTES.md complete
- [ ] Documentation updated
- [ ] Sprint 7 completion report

**Estimated**: 6-8 hours

---

## Week 2 Summary

**Total Estimated Effort**: 28-38 hours (5 working days)

**Deliverables**:
- ✅ 15+ Prometheus metrics
- ✅ OpenTelemetry tracing
- ✅ Alerting thresholds (5+ rules)
- ✅ Log retention policy
- ✅ CLI entrypoint
- ✅ pip-installable package
- ✅ LICENSE, CHANGELOG, NOTICE, SBOM
- ✅ Complete documentation

**Tests Added**: ~10 tests
- Extended metrics tests
- Tracing tests
- CLI tests
- Installation tests

---

## Sprint 7 Complete - Exit Criteria

### Technical Requirements

**Code Quality**:
- [ ] ruff check: 0 errors
- [ ] mypy --strict: 0 errors
- [ ] Test coverage: >85%

**Testing**:
- [ ] All Sprint 6 tests passing (35 tests)
- [ ] All Sprint 7 tests passing (~25 new tests)
- [ ] At least 2 stress tests passing
- [ ] Total: ~305+ tests passing

**Features**:
- [ ] Graceful shutdown with drain()
- [ ] 3-tier health endpoints
- [ ] Structured JSON logging
- [ ] Request tracking middleware
- [ ] 15+ Prometheus metrics
- [ ] OpenTelemetry tracing
- [ ] Alerting thresholds defined
- [ ] Log retention policy
- [ ] CLI entrypoint working
- [ ] pip-installable package

**Documentation**:
- [ ] E2E testing guide
- [ ] Performance baselines
- [ ] Prometheus metrics catalog
- [ ] OpenTelemetry setup guide
- [ ] Alert runbooks
- [ ] Log management policy
- [ ] Release notes
- [ ] Updated README

**Compliance**:
- [ ] LICENSE file
- [ ] NOTICE file
- [ ] CHANGELOG.md
- [ ] SBOM generated
- [ ] License compliance check passing

### Production Deployment Approval

**For Heavy Production (>50 concurrent users)**:
- [ ] All Sprint 6 technical debt resolved
- [ ] Graceful shutdown validated
- [ ] Stress tests passing
- [ ] Performance baselines documented
- [ ] Observability complete
- [ ] Technical Fellows score >90/100

---

## Daily Standup Protocol

### Morning Standup (15 minutes)
1. Review yesterday's completion
2. Identify blockers
3. Plan today's work in detail
4. Update todo list

### Evening Standup (15 minutes)
1. Review completion status
2. Run tests for completed work
3. Identify issues that need fixing

### Fix Cycle (if issues found)
1. Fix identified issues
2. Run tests again
3. Evening standup to review fixes
4. Repeat until clean

### End of Day Planning (15 minutes)
1. Standup to plan next day
2. Document progress
3. Update sprint status

---

## Risk Mitigation

**Risk 1: Stress test debugging exceeds time budget**
- Mitigation: Cap at 2 tests minimum (pool exhaustion, concurrent agents)
- Fallback: Document issues, defer remaining tests

**Risk 2: OpenTelemetry complexity**
- Mitigation: Use console exporter (simplest)
- Fallback: Basic tracing without full OTLP integration

**Risk 3: SBOM generation issues**
- Mitigation: Use syft (well-maintained, widely used)
- Fallback: Manual dependency listing if tooling fails

---

## Success Metrics

**Quantitative**:
- Technical Fellows score: >90/100 (target)
- Sprint 6 score: 88/100 (baseline)
- Test count: 305+ tests passing
- Prometheus metrics: 15+ implemented
- Code quality: 0 errors
- Documentation: 10+ new docs

**Qualitative**:
- Production deployment approved for all load levels
- Observability enables production debugging
- Graceful shutdown prevents data loss
- Package ready for PyPI distribution
- Developer onboarding documentation complete

---

**Plan Status**: Ready for Technical Fellows review
**Estimated Duration**: 10 days (60-80 hours)
**Start Date**: Pending approval
**Expected Completion**: 2026-02-08

---

**Created**: 2026-01-25
**Last Updated**: 2026-01-25
**Version**: 1.0.0
