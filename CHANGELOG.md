# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.2.0] - 2026-01-25

**Sprint 7: Observability + Production Hardening**

This release focuses on production-grade observability, monitoring, and operational excellence.

### Added

#### Observability & Monitoring
- **Structured Logging**: JSON logging (production) and console logging (development) via structlog
- **Request Correlation**: UUID-based request IDs with X-Request-ID header propagation
- **Request Logging Middleware**: Automatic logging of all requests with timing and context
- **Prometheus Metrics Endpoint**: `/metrics` endpoint serving Prometheus exposition format
- **Core Metrics** (5 total):
  - `semantic_request_total`: Counter for HTTP requests by method, path, status
  - `semantic_request_duration_seconds`: Histogram for request latency distribution
  - `semantic_pool_utilization_ratio`: Gauge for BlockPool usage (0.0-1.0)
  - `semantic_agents_active`: Gauge for hot agents in memory
  - `semantic_cache_hit_total`: Counter for cache operations by result (hit/miss)
- **X-Response-Time Header**: Automatic timing header added to all responses

#### Production Operations
- **Graceful Shutdown**: Request draining with 30s timeout, cache persistence on shutdown
- **3-Tier Health Endpoints**:
  - `/health/live`: Liveness probe (always 200 if process alive)
  - `/health/ready`: Readiness probe (200 if ready, 503 if pool exhausted)
  - `/health/startup`: Startup probe (200 after initialization)
- **Alerting Rules**: 10 Prometheus alert rules across 3 severity levels (critical, warning, info)
- **Production Runbook**: Comprehensive operations guide (`docs/PRODUCTION_RUNBOOK.md`)
- **Log Retention Policy**: Hot/warm/cold storage tiers with ILM examples (`config/logging/retention.md`)

#### CLI & Packaging
- **CLI Entrypoint**: `semantic` command with typer framework
  - `serve`: Start server with configurable host, port, workers, log-level, reload
  - `version`: Display version and sprint information
  - `config`: Show current configuration (server, MLX, agent, rate limiting)
- **pip Installation**: Fully installable via `pip install .`
- **Package Version**: Updated to v0.2.0

#### Documentation
- **Performance Baselines**: Documented 1-2s inference, <2ms health checks
- **Week 1 Summary**: Comprehensive Sprint 7 Week 1 completion report
- **Alert Response Procedures**: Runbook with troubleshooting guides
- **Monitoring Setup**: Grafana dashboard examples and PromQL queries

### Changed

- **Middleware Order**: RequestIDMiddleware → RequestLoggingMiddleware → MetricsMiddleware → CORS → Auth → RateLimiter
- **Health Check Logging**: Disabled for `/health/*` endpoints to avoid log spam
- **Package Metadata**: Updated to reflect Sprint 7 scope and v0.2.0
- **Version String**: Updated from "0.1.0-alpha" to "0.2.0"

### Technical Details

#### Middleware Stack
```
Request → RequestIDMiddleware (correlation ID)
       → RequestLoggingMiddleware (timing, context)
       → RequestMetricsMiddleware (Prometheus)
       → CORSMiddleware
       → AuthenticationMiddleware
       → RateLimitMiddleware
       → Handler
```

#### Observability Features
- **Request Tracing**: Every request gets unique correlation ID
- **Structured Context**: Request ID, method, path propagated through structlog contextvars
- **Metrics Auto-Collection**: Request counts, latencies, and pool stats collected automatically
- **Health Monitoring**: 3-tier probe system for Kubernetes-compatible health checks

#### Production Readiness
- Zero downtime deployments via graceful shutdown
- Request draining prevents dropped requests
- Cache persistence preserves agent state across restarts
- Alerting covers all critical scenarios (pool exhaustion, errors, latency)

### Performance

- **Health Check Latency**: <2ms (p95)
- **Inference Latency**: 1-2s per request (MLX on Apple Silicon)
- **Metrics Overhead**: <0.5ms per request
- **Logging Overhead**: <0.1ms per request

### Dependencies Added

- `structlog>=24.4.0` - Structured logging
- `prometheus-client>=0.21.0` - Metrics collection
- `typer>=0.9.0` - CLI framework

### Breaking Changes

None. This release is fully backward compatible with v0.1.0.

### Migration Guide

Upgrading from v0.1.0 to v0.2.0:

1. **Update package**:
   ```bash
   pip install --upgrade semantic-server
   ```

2. **Optional: Configure logging level**:
   ```bash
   # JSON logging (production)
   export SEMANTIC_SERVER_LOG_LEVEL=PRODUCTION

   # Console logging (development)
   export SEMANTIC_SERVER_LOG_LEVEL=DEBUG
   ```

3. **Optional: Use new CLI**:
   ```bash
   # Instead of uvicorn
   semantic serve

   # With custom options
   semantic serve --host 0.0.0.0 --port 8080 --log-level INFO
   ```

4. **Optional: Set up monitoring**:
   - Configure Prometheus to scrape `/metrics` endpoint
   - Import alert rules from `config/prometheus/alerts.yml`
   - Set up log aggregation (ELK, Splunk, Datadog)

No code changes required - all new features are opt-in or automatic.

---

## [0.1.0] - 2026-01-22

**Sprint 6: Multi-Agent Cache Management**

Initial release with production-quality multi-agent LLM inference server.

### Added

#### Core Features
- **Multi-Agent Inference**: Manage multiple independent LLM agents with isolated contexts
- **Block-Pool Memory Management**: Continuous batching with PagedAttention-style block allocation
- **Persistent KV Cache**: Save/load agent states to/from disk using safetensors format
- **MLX Integration**: Apple Silicon-optimized inference via MLX framework
- **Anthropic Messages API**: Compatible with Claude Messages API format
- **FastAPI Server**: Production-grade HTTP API with async request handling

#### Architecture
- **Hexagonal Architecture**: Clean separation of domain, ports, and adapters
- **Ports & Adapters Pattern**: Protocol-based interfaces for all external dependencies
- **Domain-Driven Design**: Pure business logic with no framework dependencies

#### Components

**Domain Layer** (`src/semantic/domain/`):
- `AgentID`: Unique agent identifier value object
- `Message`, `MessageBlock`, `TokenSequence`: Core data models
- `BlockPool`: Memory management with allocation, deallocation, defragmentation
- `KVCache`: Key-value cache abstraction for transformer inference

**Application Layer** (`src/semantic/application/`):
- `BatchEngine`: Continuous batching orchestrator for multi-agent inference
- `AgentCacheStore`: Agent lifecycle management (create, save, load, evict)
- `use_agent_inference`: Application use case for agent inference requests

**Adapters** (`src/semantic/adapters/`):
- **Inbound** (FastAPI):
  - `/v1/messages`: Anthropic Messages API endpoint
  - Authentication middleware (API key validation)
  - Rate limiting middleware (per-agent and global limits)
- **Outbound** (MLX):
  - `MLXCacheAdapter`: KV cache operations via MLX
  - `MLXSpecExtractor`: Model metadata extraction
  - `SafetensorsCacheAdapter`: Persistent cache serialization
- **Config**: Settings management with pydantic-settings

#### API Features
- **Multi-Turn Conversations**: Stateful agent conversations with history
- **Streaming Responses**: SSE-based token streaming
- **Authentication**: API key validation via X-API-Key or Authorization header
- **Rate Limiting**: 60 req/min per agent, 1000 req/min global
- **CORS**: Configurable cross-origin resource sharing

#### Performance
- **Continuous Batching**: Process multiple agents concurrently
- **Cache Persistence**: Resume conversations without recomputing KV cache
- **Memory Efficiency**: Block-level allocation minimizes fragmentation
- **Apple Silicon Optimization**: Native MLX acceleration on M1/M2/M3 chips

### Configuration

Environment variables (via `pydantic-settings`):

```bash
# Server
SEMANTIC_SERVER_HOST=0.0.0.0
SEMANTIC_SERVER_PORT=8000
SEMANTIC_SERVER_WORKERS=1
SEMANTIC_SERVER_LOG_LEVEL=INFO
SEMANTIC_SERVER_CORS_ORIGINS=http://localhost:3000

# MLX
SEMANTIC_MLX_MODEL_ID=mlx-community/gemma-2-2b-it-4bit
SEMANTIC_MLX_CACHE_BUDGET_MB=2048
SEMANTIC_MLX_MAX_BATCH_SIZE=4
SEMANTIC_MLX_PREFILL_STEP_SIZE=512

# Agent
SEMANTIC_AGENT_CACHE_DIR=~/.cache/semantic
SEMANTIC_AGENT_MAX_AGENTS_IN_MEMORY=100

# Rate Limiting
SEMANTIC_SERVER_RATE_LIMIT_PER_AGENT=60
SEMANTIC_SERVER_RATE_LIMIT_GLOBAL=1000

# Authentication
ANTHROPIC_API_KEY=sk-ant-xxx  # Comma-separated for multiple keys
```

### Dependencies

**Core**:
- `mlx==0.30.3` - Apple Silicon ML framework (pinned)
- `mlx-lm==0.30.4` - MLX language model utilities (pinned)
- `fastapi>=0.115.0` - Web framework
- `uvicorn[standard]>=0.32.0` - ASGI server
- `pydantic>=2.10.0` - Data validation
- `pydantic-settings>=2.7.0` - Settings management
- `safetensors>=0.4.0` - Cache persistence
- `transformers>=4.47.0` - Model tokenizers
- `sse-starlette>=2.2.0` - Server-sent events

### Architecture Decisions

See `project/architecture/` for detailed ADRs:
- **ADR-001**: Hexagonal Architecture (Ports & Adapters)
- **ADR-002**: Block-Pool Memory Management
- **ADR-003**: Persistent KV Cache with Safetensors
- **ADR-004**: Multi-Agent Batch Engine

### Known Limitations

- **Apple Silicon Only**: MLX framework requires M1/M2/M3 chips
- **Single-Node Only**: No distributed inference support (v0.1.0)
- **Memory-Bound**: Cache budget must fit in unified memory
- **No Quantization Control**: Uses model's native quantization (e.g., 4-bit)

### Testing

**Test Coverage**: 85%+ (unit, integration, e2e, stress, benchmark)

**Test Suites**:
- Unit tests: Fast tests with mocked boundaries
- Integration tests: Real MLX and disk I/O (Apple Silicon required)
- Smoke tests: Basic server lifecycle validation
- E2E tests: Full-stack multi-agent scenarios
- Stress tests: Load and concurrency testing
- Benchmark tests: Performance validation

### Documentation

- `README.md`: Quick start and overview
- `docs/API.md`: API reference and examples
- `project/architecture/`: Architecture decision records
- `project/sprints/`: Sprint planning and completion reports

---

## Release Schedule

- **v0.1.0** (2026-01-22): Sprint 6 - Multi-Agent Cache Management
- **v0.2.0** (2026-01-25): Sprint 7 - Observability + Production Hardening
- **v0.3.0** (TBD): Sprint 8+ - Extended features (distributed inference, advanced metrics, etc.)

---

## Versioning

This project follows [Semantic Versioning](https://semver.org/):
- **MAJOR**: Incompatible API changes
- **MINOR**: Backward-compatible functionality
- **PATCH**: Backward-compatible bug fixes

---

**Maintainer**: Semantic Team
**License**: MIT
**Repository**: https://github.com/yshk-mxim/semantic-server
