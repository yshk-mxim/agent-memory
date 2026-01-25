# Sprint 4: Multi-Protocol API Adapter

**Status**: In Progress - Day 0
**Start Date**: 2026-01-25
**Target Duration**: 8 working days
**Deliverable**: Claude Code CLI connects with persistent caching; OpenAI-compatible API with session_id; all tests passing

---

## Overview

Sprint 4 implements the inbound API adapters that allow external clients to interact with the semantic caching system. This completes the hexagonal architecture by providing multiple protocol adapters for different client types.

**Key Components**:
- FastAPI server with dependency injection
- Anthropic Messages API (/v1/messages) with SSE streaming
- OpenAI Chat Completions API (/v1/chat/completions) with session_id extension
- Direct Agent API (CRUD endpoints)
- Authentication and rate limiting
- Schemathesis contract testing

**Architecture**:
```
┌─────────────────────────────────────────────────────────┐
│                    Inbound Adapters                      │
├─────────────────────────────────────────────────────────┤
│  AnthropicAdapter  │  OpenAIAdapter  │  DirectAdapter   │
│  (/v1/messages)    │ (/v1/chat/...)  │ (/v1/agents/...) │
└────────────┬────────────────┬───────────────┬───────────┘
             │                │               │
             └────────────────┼───────────────┘
                              │
                    ┌─────────▼─────────┐
                    │  Application Core  │
                    │  (BatchEngine,     │
                    │   CacheStore)      │
                    └─────────┬──────────┘
                              │
                    ┌─────────▼─────────┐
                    │  Domain Layer      │
                    │  (BlockPool,       │
                    │   Entities)        │
                    └─────────┬──────────┘
                              │
                    ┌─────────▼─────────┐
                    │ Outbound Adapters  │
                    │ (MLX, Safetensors) │
                    └────────────────────┘
```

---

## Pre-Sprint State (Day 0 - 2026-01-25)

### Infrastructure Validated
- ✅ **Unit tests**: 112 passed (domain, services, value objects)
- ✅ **Integration tests**: 12 passed, 1 skipped (batch engine, concurrency)
- ⚠️ **Test file naming conflict**: `test_batch_engine.py` exists in both unit/ and integration/
  - Workaround: Run integration tests with explicit path
  - TODO: Rename to avoid collision (defer to Sprint 5)

### Completed Pre-Sprint 4 Work
- ✅ **Remediation** (Commit 5d7fb4b): Removed ~5k lines of dead POC code
- ✅ **MLX Adapter**: `mlx_cache_adapter.py` created
- ✅ **Spec Extractor**: `mlx_spec_extractor.py` created
- ✅ **Safetensors**: `safetensors_cache_adapter.py` created
- ✅ **Domain Errors**: Standardized error hierarchy
- ✅ **Settings**: Pydantic configuration management
- ✅ **Experiments**: EXP-001, EXP-003, EXP-004, EXP-005, EXP-006 completed

### Missing Components (Sprint 4 Scope)
- ❌ **Inbound Adapters**: No API handlers exist
- ❌ **Server Entrypoints**: No api_server.py or cli.py
- ❌ **SSE Streaming**: No streaming implementation
- ❌ **Authentication**: No API key validation
- ❌ **Rate Limiting**: No rate limiter
- ❌ **API Tests**: No integration tests for endpoints
- ❌ **Experiments**: EXP-009 (SSE format), EXP-010 (CLI compatibility) pending

### Dependencies Check
```bash
# TODO: Verify these are installed
fastapi>=0.115.0
uvicorn[standard]>=0.32.0
sse-starlette>=2.1.3
pydantic>=2.0.0
pydantic-settings>=2.0.0
```

---

## Day-by-Day Progress

### Day 0: Pre-Sprint Validation ✅ IN PROGRESS

**Morning Standup**:
- Starting Sprint 4 execution
- Goal: Validate infrastructure readiness
- All 9 days added to todo tracking

**Completed**:
- ✅ Verified unit tests: 112 passed
- ✅ Verified integration tests: 12 passed, 1 skipped
- ✅ Identified test naming conflict (defer fix to Sprint 5)
- ✅ Checked experiments: EXP-001-006 exist, EXP-009-010 deferred to implementation
- ✅ Created Sprint 4 documentation: `project/sprints/sprint_4_multi_protocol_api.md`
- ⏳ Create ADR-006 skeleton
- ⏳ Validate dependencies
- ⏳ Test basic FastAPI hello world

**Blockers**: None

**Next Steps**:
- Create ADR-006 skeleton
- Validate FastAPI dependencies
- Evening standup review

---

### Day 1: Foundation - Server Entrypoint & Health Check

**Status**: Pending

**Planned Tasks**:
- Create `src/semantic/entrypoints/api_server.py`
- Create `src/semantic/entrypoints/cli.py`
- Add dependency injection setup
- Basic server lifecycle tests

**Expected Files**:
- `src/semantic/entrypoints/api_server.py` (~200 lines)
- `src/semantic/entrypoints/cli.py` (~100 lines)
- `tests/integration/test_server_lifecycle.py` (~80 lines)

---

### Day 2: Request Models & Session Management

**Status**: Pending

**Planned Tasks**:
- Create `src/semantic/adapters/inbound/request_models.py`
- Implement session management strategy (simplified content-based)
- Unit tests for validation

**Expected Files**:
- `src/semantic/adapters/inbound/request_models.py` (~300 lines)
- `tests/unit/adapters/test_request_models.py` (~150 lines)

---

### Day 3: Anthropic API - Non-Streaming Endpoint

**Status**: Pending

**Planned Tasks**:
- Create `src/semantic/adapters/inbound/anthropic_adapter.py`
- POST /v1/messages endpoint (non-streaming)
- Wire into api_server.py
- Integration tests

**Expected Files**:
- `src/semantic/adapters/inbound/anthropic_adapter.py` (~250 lines)
- Update `api_server.py` (+30 lines)
- `tests/integration/test_anthropic_api.py` (~150 lines)

---

### Day 4: SSE Streaming - Anthropic Format

**Status**: Pending

**Planned Tasks**:
- Implement SSE event generators
- Enable streaming in BatchEngine
- Integration tests for streaming
- **EXP-009**: Validate SSE format against real Anthropic API

**Expected Files**:
- Update `anthropic_adapter.py` (+150 lines)
- `tests/integration/test_sse_format.py` (~100 lines)
- `project/experiments/EXP-009-anthropic-sse-format.md`

---

### Day 5: Anthropic API - Extended Features

**Status**: Pending

**Planned Tasks**:
- Thinking block support
- Prompt caching
- /v1/messages/count_tokens endpoint
- Tool use streaming

**Expected Files**:
- Update `anthropic_adapter.py` (+100 lines)
- Create `token_counter.py` (~50 lines)
- `tests/integration/test_extended_features.py` (~100 lines)

---

### Day 6: OpenAI & Direct APIs

**Status**: Pending

**Planned Tasks**:
- Create `openai_adapter.py`
- Create `direct_agent_adapter.py`
- Integration tests for both
- **EXP-010**: Validate Claude Code CLI compatibility

**Expected Files**:
- `openai_adapter.py` (~200 lines)
- `direct_agent_adapter.py` (~150 lines)
- `tests/integration/test_openai_api.py` (~80 lines)
- `tests/integration/test_direct_api.py` (~80 lines)
- `project/experiments/EXP-010-claude-cli-compatibility.md`

---

### Day 7: Security & Quality

**Status**: Pending

**Planned Tasks**:
- Authentication middleware
- Rate limiting
- Schemathesis API contract tests
- Failure-mode tests

**Expected Files**:
- `auth_middleware.py` (~100 lines)
- `rate_limiter.py` (~150 lines)
- `tests/integration/test_auth.py` (~80 lines)
- `tests/integration/test_rate_limiting.py` (~80 lines)
- `tests/integration/test_failure_modes.py` (~120 lines)
- `tests/integration/test_schemathesis.py` (~50 lines)

---

### Day 8: Polish, Documentation & Fellows Review

**Status**: Pending

**Planned Tasks**:
- Run full test suite (target: 250+ tests)
- Performance validation (5 concurrent Claude Code sessions)
- Update documentation
- Technical fellows review

**Expected Outcomes**:
- ✅ All tests passing (250+ total)
- ✅ Coverage: >85% unit, >70% integration
- ✅ SSE format matches Anthropic spec
- ✅ Authentication functional
- ✅ Rate limiting working
- ✅ Fellows approval obtained

---

## Key Technical Decisions

### Agent Identification Strategy (Simplified)

**Decision**: Use existing trie-based prefix matching in AgentCacheStore, no complex content hashing.

**Rationale**:
- Claude Code CLI sends full conversation history every request (client-side sessions)
- Each request is independent (like real Anthropic API)
- Existing trie-based cache lookup is sufficient
- No server-side session persistence needed

**Implementation**:
- Cache lookup: Use full conversation token sequence for prefix matching
- AgentCacheStore already implements `find_prefix()` method
- No need for complex hash-based agent ID generation

### SSE Format Compliance

**Requirement**: Must match Anthropic streaming protocol exactly.

**Event Types** (per claude_code_anthropic_guide.md):
- `message_start`: Initial message with ID, model, role
- `content_block_start`: Start of content block (text, thinking, tool_use)
- `content_block_delta`: Incremental content updates
- `content_block_stop`: End of content block
- `message_delta`: Stop reason and usage updates
- `message_stop`: Final event

**Validation**: EXP-009 will compare against real Anthropic API responses.

### Security Approach

**Authentication**:
- ANTHROPIC_API_KEY environment variable
- Support both env var and header-based auth
- Return 401 for invalid/missing keys

**Rate Limiting**:
- Per-agent rate limits (configurable)
- Global rate limits
- In-memory sliding window
- Return 429 with Retry-After header

---

## Code Quality Standards

Following `plans/code_quality_patterns.md`:
- ✅ No MLX/numpy in domain/application layers
- ✅ Dependency injection via ports
- ✅ Comprehensive error handling
- ✅ Type hints with mypy validation
- ✅ Property-based testing where applicable
- ✅ No AI slop (no unnecessary docstrings, no over-engineering)

---

## Testing Strategy

### Unit Tests (Target: +25 tests)
- Request model validation
- Token counting
- Rate limiting logic
- Authentication logic

### Integration Tests (Target: +90 tests)
- API endpoint functionality
- SSE streaming format
- Authentication flows
- Rate limiting enforcement
- Failure modes (malformed requests, errors)
- Schemathesis contract testing

### Performance Tests
- 5 concurrent Claude Code CLI sessions
- Memory leak detection (1-hour run)
- Cache hit rate measurement

---

## Issues and Resolutions

### Pre-Sprint Issues

**Issue**: Test file naming conflict
- **File**: `test_batch_engine.py` exists in both `tests/unit/` and `tests/integration/`
- **Impact**: pytest collection error when running `make test-integration`
- **Workaround**: Run with explicit path: `pytest -v -m integration tests/integration/`
- **Resolution**: Defer rename to Sprint 5 (not blocking Sprint 4 work)

### Sprint 4 Issues

*Issues discovered during Sprint 4 will be documented here*

---

## Risk Register

### High Risk
- **SSE format compliance**: Must match Anthropic spec exactly
  - Mitigation: Create EXP-009 golden file comparison
  - Validation: Test with real Claude Code CLI

### Medium Risk
- **Claude Code CLI compatibility**: Must work with `ANTHROPIC_BASE_URL` override
  - Mitigation: Create EXP-010 integration test
  - Validation: Run 3-turn conversation test

### Low Risk
- **Performance under load**: Must handle 5 concurrent sessions
  - Mitigation: Performance tests in Day 8
  - Validation: 1-hour stress test

---

## Definition of Done

**Sprint 4 is complete when**:
- ✅ All 250+ tests passing (unit + integration)
- ✅ Coverage targets met (>85% unit, >70% integration)
- ✅ SSE format validated (EXP-009)
- ✅ Claude Code CLI works end-to-end (EXP-010)
- ✅ Authentication blocks invalid requests
- ✅ Rate limiting returns 429 correctly
- ✅ No memory leaks (1-hour test)
- ✅ Schemathesis contract tests pass
- ✅ Technical fellows approval obtained
- ✅ ADR-006 finalized
- ✅ All documentation updated

---

## Fellows Review Checklist

### Software Engineering (SE)
- [ ] Architecture compliance (hexagonal, no MLX in domain/app)
- [ ] Port interfaces clean and documented
- [ ] Dependency injection implemented correctly
- [ ] Error handling comprehensive

### Machine Learning (ML)
- [ ] MLX integration clean (adapter pattern)
- [ ] Cache operations correct
- [ ] No ML framework leakage to core

### Quality Engineering (QE)
- [ ] Test coverage >85% unit, >70% integration
- [ ] All tests ACTUALLY pass (not just green)
- [ ] Failure modes tested
- [ ] Contract tests pass (Schemathesis)

### Hardware (HW)
- [ ] Memory leaks prevented
- [ ] Performance acceptable (5 concurrent sessions)
- [ ] Resource cleanup on errors

---

## Next Sprint Preview

**Sprint 5: Production Hardening**
- Performance optimization
- Observability (metrics, tracing)
- Deployment automation
- Load testing
- Documentation polish

---

**Last Updated**: 2026-01-25 (Day 0 in progress)
**Document Owner**: Sprint 4 Team
**Review Cadence**: Daily (evening standup)
