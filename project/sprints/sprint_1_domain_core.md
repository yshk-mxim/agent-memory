# Sprint 1: Domain Core Implementation

**Duration**: 2 weeks (10 working days)
**Status**: In Progress
**Started**: 2026-01-24
**Goal**: Implement domain core with ModelCacheSpec for 4 architectures and BlockPool service

---

## Objectives

Implement the domain layer (pure business logic, zero external dependencies) with:

1. **ModelCacheSpec** extraction for 4 model architectures
2. **BlockPool** service (allocate/free/budget/reconfigure)
3. **Domain entities** (KVBlock, AgentBlocks)
4. **95%+ test coverage** for domain layer
5. **Performance validation** (block allocation < 1ms)

---

## Deliverables

### Code Deliverables

| Component | File | Owner | Status |
|-----------|------|-------|--------|
| Domain entities | `src/semantic/domain/entities.py` | SE | Not Started |
| Value objects (extended) | `src/semantic/domain/value_objects.py` | SE | Partial (skeleton exists) |
| BlockPool service | `src/semantic/domain/services.py` | SE | Not Started |
| ModelCacheSpec tests | `tests/unit/test_model_cache_spec.py` | QE | Not Started |
| BlockPool tests | `tests/unit/test_block_pool.py` | QE | Not Started |
| Property tests | `tests/unit/test_block_pool_properties.py` | QE | Not Started |

### Documentation Deliverables

| Document | Path | Owner | Status |
|----------|------|-------|--------|
| ADR-002: Block Size 256 | `project/architecture/ADR-002-block-size-256.md` | SE, DE | Not Started |
| EXP-001: Model Args | `project/experiments/EXP-001-model-args.md` | ML | Not Started |
| EXP-002: Allocation Overhead | `project/experiments/EXP-002-allocation-overhead.md` | ML, HW | Not Started |

---

## Architecture Context

### Hexagonal Architecture (Domain Layer)

```
┌─────────────────────────────────────────────────────────────┐
│  DOMAIN CORE (No external dependencies)                      │
├─────────────────────────────────────────────────────────────┤
│  Entities:                                                   │
│  - KVBlock         : Single 256-token cache block            │
│  - AgentBlocks     : Collection of blocks for one agent      │
│                                                              │
│  Value Objects:                                              │
│  - ModelCacheSpec  : Per-layer cache geometry                │
│  - CacheKey        : Unique identifier for cached content    │
│  - GenerationResult: Output of text generation               │
│                                                              │
│  Services:                                                   │
│  - BlockPool       : Allocate/free/budget/reconfigure        │
│                                                              │
│  Errors:                                                     │
│  - SemanticError (base) + 8 specific error types            │
└─────────────────────────────────────────────────────────────┘
```

**Dependency Rule**: Domain core has ZERO imports from:
- `mlx`, `mlx_lm` (infrastructure)
- `fastapi`, `uvicorn` (web framework)
- `safetensors`, `transformers` (adapters)

Only imports from: `typing`, `dataclasses`, `abc`, `enum`, Python stdlib

---

## Target Model Architectures

| Model | Type | Layers | Attention | KV Heads | Head Dim | Window |
|-------|------|--------|-----------|----------|----------|--------|
| **Gemma 3 12B** | Hybrid SWA+Global | 48 | 8 global + 40 sliding (pattern=6) | 8 | 256 | 512 |
| **GPT-OSS-20B** | MoE Alternating | 24 | 12 global + 12 sliding | 8 | 64 | 128 |
| **Qwen 2.5-14B** | Uniform Full | 48 | All global | 8 | 128 | ∞ |
| **Llama 3.1-8B** | Uniform Full | 32 | All global | 8 | 128 | ∞ |

**Block Size**: 256 tokens (universal across all models)

**Memory per Block per Layer**:
- Gemma 3: 2 MB (8 × 256 × 2 × 2 × 256)
- GPT-OSS: 0.5 MB (8 × 64 × 2 × 2 × 256)
- Qwen: 1 MB (8 × 128 × 2 × 2 × 256)
- Llama: 1 MB (8 × 128 × 2 × 2 × 256)

---

## Critical Path (Blocking Dependencies)

### Week 1

**Days 1-2: Setup & Entities**
1. SE defines domain entities (`KVBlock`, `AgentBlocks`)
2. ML downloads 4 models, inspects `config.json`
3. QE sets up test infrastructure

**Days 3-4: ModelCacheSpec Implementation**
4. ML executes **EXP-001** (validates `model.args` attributes)
5. ML implements `from_model()` for Gemma 3 + Qwen
6. SE implements BlockPool core (allocate/free)

**Days 5-6: BlockPool Core Complete**
7. ML implements `from_model()` for GPT-OSS + Llama
8. SE adds layer-group-aware allocation
9. QE writes ModelCacheSpec + BlockPool tests

### Week 2

**Days 7-8: Testing & Benchmarking**
10. SE implements `reconfigure()` for model hot-swap
11. ML executes **EXP-002** (block allocation benchmark)
12. QE implements Hypothesis property tests
13. SE writes ADR-002 (Block Size 256 rationale)

**Days 9-10: Exit Gate Preparation**
14. QE runs coverage report, fills gaps to 95%+
15. SE/ML fix issues from testing
16. PM validates exit gate criteria

---

## Work Breakdown by Expert

### SE (Software Engineer) - 8 days

**Day 1-2: Domain Entities**
- [ ] Define `KVBlock` dataclass (block_id, layer_data, token_count, metadata)
- [ ] Define `AgentBlocks` dataclass (agent_id, blocks, total_tokens)
- [ ] Extend `ModelCacheSpec` with `from_model()` classmethod skeleton
- [ ] Write type annotations (mypy --strict compliance)

**Day 3-4: BlockPool Core**
- [ ] Implement `BlockPool.__init__()` (free_list, allocated_blocks, spec)
- [ ] Implement `allocate(n_blocks, layer_type)` with free list management
- [ ] Implement `free(block_ids)` returns blocks to pool
- [ ] Implement `used_memory()`, `available_memory()` budget tracking

**Day 5-6: Layer-Group Allocation**
- [ ] Add per-layer-group tracking (global vs SWA)
- [ ] Implement SWA layer block capping (Gemma: 2 blocks, GPT-OSS: 1 block)
- [ ] Add `max_batch_size()` calculation based on pool state

**Day 7-8: Model Hot-Swap**
- [ ] Implement `reconfigure(new_spec)` for model swap
- [ ] Write ADR-002: Block Size 256 Tokens (Universal)
- [ ] Code review, refactor based on QE feedback

---

### ML (Machine Learning Engineer) - 7 days

**Day 1: Model Download**
- [ ] Download Gemma 3 12B 4-bit from mlx-community
- [ ] Download GPT-OSS-20B from mlx-community (if available)
- [ ] Download Qwen 2.5-14B from mlx-community
- [ ] Download Llama 3.1-8B from mlx-community
- [ ] Inspect `config.json` files manually

**Day 2: EXP-001 Execution**
- [ ] Write script to load all 4 models
- [ ] Print all `model.args` attributes for each model
- [ ] Validate required attributes exist: `num_hidden_layers`, `num_key_value_heads`, `head_dim`, `sliding_window_pattern`
- [ ] Document findings in `project/experiments/EXP-001-model-args.md`

**Day 3-4: Simple Architectures**
- [ ] Implement `ModelCacheSpec.from_model()` for Gemma 3 (hybrid SWA+global)
- [ ] Implement `ModelCacheSpec.from_model()` for Qwen 2.5 (uniform full)
- [ ] Handle missing `sliding_window_pattern` attribute (default to None)

**Day 5-6: Complex Architectures**
- [ ] Implement `ModelCacheSpec.from_model()` for GPT-OSS-20B (MoE alternating)
- [ ] Implement `ModelCacheSpec.from_model()` for Llama 3.1 (uniform full)
- [ ] Document edge cases and fallback logic

**Day 7: EXP-002 Execution**
- [ ] Benchmark `BlockPool.allocate()` and `free()` (1000 iterations)
- [ ] Measure overhead: target < 1ms per operation
- [ ] Document results in `project/experiments/EXP-002-allocation-overhead.md`

---

### QE (Quality Engineer) - 8 days

**Day 1-2: Test Infrastructure**
- [ ] Create `tests/unit/test_model_cache_spec.py`
- [ ] Create `tests/unit/test_block_pool.py`
- [ ] Create `tests/unit/test_block_pool_properties.py`
- [ ] Set up Hypothesis framework (install + configure)

**Day 3-4: ModelCacheSpec Tests**
- [ ] `test_from_model_gemma3()` - validates hybrid SWA+global extraction
- [ ] `test_from_model_gpt_oss()` - validates MoE alternating extraction
- [ ] `test_from_model_qwen()` - validates uniform full extraction
- [ ] `test_from_model_llama()` - validates uniform full extraction
- [ ] Use ML's EXP-001 fixtures for test data

**Day 5-6: BlockPool Unit Tests**
- [ ] `test_allocate_success()` - happy path
- [ ] `test_allocate_pool_exhausted()` - raises `PoolExhaustedError`
- [ ] `test_free_returns_blocks()` - free list grows
- [ ] `test_budget_tracking()` - used + available = total
- [ ] `test_reconfigure_clears_pool()` - model hot-swap support
- [ ] `test_concurrent_agent_creation_race()` - failure mode test

**Day 7-8: Property Tests**
- [ ] Property: `used + available = total` (always)
- [ ] Property: `allocate(n) then free(n)` restores pool state
- [ ] Property: no double-free (idempotent or raises error)
- [ ] Run Hypothesis with 1000 examples per test

**Day 9-10: Coverage & Gap Filling**
- [ ] Run `pytest --cov=semantic.domain --cov-report=html`
- [ ] Identify untested branches/edge cases
- [ ] Fill gaps to reach 95%+ coverage
- [ ] Document any exclusions (if < 95%)

---

### HW (Hardware/Performance Engineer) - 3 days

**Day 1-2: Memory Validation**
- [ ] Create spreadsheet model for all 4 architectures
- [ ] Validate memory formulas (bytes_per_token_per_layer)
- [ ] Calculate max agents at 4K context for each model
- [ ] Review SE's BlockPool data structure design

**Day 5-6: EXP-002 Assistance**
- [ ] Assist ML with block allocation benchmarking
- [ ] Analyze results, identify bottlenecks if > 1ms
- [ ] Suggest optimizations if needed

**Day 7-8: Memory Safety Review**
- [ ] Review BlockPool.reconfigure() implementation
- [ ] Confirm no memory leaks during model swap
- [ ] Document memory budget per model

---

### DE (Documentation Engineer) - 5 days

**Day 1-3: ADR-002 Template**
- [ ] Draft ADR-002: Block Size 256 Tokens (Universal)
- [ ] Rationale section: Why 256 despite varying window sizes?
- [ ] Trade-offs: GPT-OSS wastes memory (window=128 < block=256)

**Day 4-6: Code Documentation**
- [ ] Add Google-style docstrings to `KVBlock`, `AgentBlocks`
- [ ] Add docstrings to BlockPool methods (Args/Returns/Raises)
- [ ] Document `ModelCacheSpec.from_model()` logic for all 4 architectures

**Day 7-8: Complex Logic Comments**
- [ ] Add inline comments for SWA capping logic (CC > 5)
- [ ] Document layer-group allocation algorithm
- [ ] Ensure ruff D rules pass (100% docstring coverage)

**Day 9-10: Final Review**
- [ ] Review all domain code for documentation completeness
- [ ] Validate ruff D passes with zero errors
- [ ] Update sprint status document

---

### PM (Project Manager) - Continuous

**Day 1-2: Sprint Kickoff**
- [ ] Align all experts on domain-first approach
- [ ] Confirm Sprint 0 experiments (EXP-003, EXP-004) passed
- [ ] Set up daily check-ins

**Day 3-5: Mid-Sprint Monitoring**
- [ ] Track EXP-001 results (ML)
- [ ] Monitor ModelCacheSpec development (4 architectures)
- [ ] Facilitate SE + ML sync (Day 3)

**Day 6-8: Late Sprint Monitoring**
- [ ] Track BlockPool implementation + testing
- [ ] Monitor EXP-002 execution (ML + HW)
- [ ] Facilitate SE + QE sync (Day 7)

**Day 9-10: Exit Gate Preparation**
- [ ] Review coverage metrics (target: 95%+)
- [ ] Validate all exit criteria met
- [ ] Prepare Sprint 2 kickoff

---

### OSS (Open Source/Legal) - Spot Checks

**Day 1: Initial Check**
- [ ] Review Sprint 1 plan, confirm no new dependencies

**Day 5: Mid-Sprint Check**
- [ ] Scan `domain/` directory for unexpected imports
- [ ] Validate only stdlib imports (typing, dataclasses, abc)

**Day 10: Final Validation**
- [ ] Ensure mypy passes with domain layer having no external deps
- [ ] Confirm license policy still met

---

### SysE (Systems/Infrastructure) - Monitoring

**Day 1: CI Validation**
- [ ] Verify Sprint 0 CI is functional
- [ ] Confirm mypy --strict passes on existing code

**Day 5: Mid-Sprint Check**
- [ ] Ensure new domain code doesn't break CI
- [ ] Monitor test suite performance

**Day 10: Exit Gate Validation**
- [ ] Run `make lint && make typecheck && make test`
- [ ] Confirm all checks pass
- [ ] Prepare for Sprint 2 integration tests (macOS runner)

---

## Experiments

### EXP-001: Model Args Validation

**Hypothesis**: All 4 target models have consistent `model.args` attributes for cache extraction.

**Method**:
1. Load each model via `mlx_lm.load(model_name)`
2. Inspect `model.args` object
3. Validate required attributes exist:
   - `num_hidden_layers` or `n_layers`
   - `num_key_value_heads` or `n_kv_heads`
   - `head_dim` or `hidden_size / n_heads`
   - `sliding_window_pattern` (optional, may be missing)

**Success Criteria**: All 4 models have extractable attributes (with fallback logic if needed)

**Owner**: ML
**Timeline**: Day 2

---

### EXP-002: Block Allocation Overhead

**Hypothesis**: BlockPool can allocate/free blocks in < 1ms per operation.

**Method**:
1. Initialize BlockPool with 1000 blocks
2. Benchmark `allocate(10)` × 100 iterations
3. Benchmark `free(block_ids)` × 100 iterations
4. Measure average time per operation

**Success Criteria**: < 1ms per allocate/free operation (mean)

**Owner**: ML, HW
**Timeline**: Day 7

---

## Exit Gate Criteria

Sprint 1 is considered **COMPLETE** when all criteria are met:

### Code Quality

- [ ] ✅ **95%+ coverage** on domain layer (`domain/entities.py`, `domain/services.py`, `domain/value_objects.py`)
- [ ] ✅ **mypy --strict** passes with zero errors
- [ ] ✅ **ruff lint** passes with zero errors
- [ ] ✅ **ruff D** (docstring rules) passes with zero errors
- [ ] ✅ **Complexity check** passes (CC < 10, domain logic CC < 7)

### Functional Requirements

- [ ] ✅ **ModelCacheSpec** extracts specs from all 4 architectures (Gemma, GPT-OSS, Qwen, Llama)
- [ ] ✅ **BlockPool** allocates/frees blocks correctly
- [ ] ✅ **BlockPool** tracks budget (used + available = total)
- [ ] ✅ **BlockPool** supports reconfiguration for model hot-swap
- [ ] ✅ **SWA layer capping** logic implemented (bounded memory waste)

### Testing

- [ ] ✅ **All unit tests pass** (ModelCacheSpec + BlockPool)
- [ ] ✅ **Hypothesis property tests pass** (BlockPool invariants, 1000 examples)
- [ ] ✅ **No external dependencies** in domain layer (only stdlib imports)

### Documentation

- [ ] ✅ **ADR-002** documented and reviewed (Block Size 256 rationale)
- [ ] ✅ **EXP-001** documented with results
- [ ] ✅ **EXP-002** documented with results
- [ ] ✅ **All domain code** has Google-style docstrings

### Performance

- [ ] ✅ **Block allocation overhead** < 1ms per operation (EXP-002 result)

---

## Risks & Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| GPT-OSS config.json non-standard | Medium | Medium | Use fallback logic, document edge cases |
| Block allocation > 1ms | Low | Medium | Optimize BlockPool data structures (free list) |
| 95% coverage too aggressive | Low | Low | Focus on domain layer only, use fake adapters |
| ModelCacheSpec extraction complex | Medium | High | Start with simple cases (Qwen, Llama), then Gemma, then GPT-OSS |
| Hypothesis tests difficult | Medium | Low | Start with simple properties, add complexity incrementally |

---

## Integration Points

### Day 3: SE + ML Sync
- **Topic**: Ensure `ModelCacheSpec` structure supports all 4 architectures
- **Deliverable**: Agreed-upon field names and types
- **Outcome**: ML can implement `from_model()` without API changes

### Day 6: SE + HW Sync
- **Topic**: Review BlockPool memory safety, allocation efficiency
- **Deliverable**: Confirmed data structure design (free list)
- **Outcome**: HW approves approach, no performance concerns

### Day 7: SE + QE Sync
- **Topic**: Provide test fixtures, review property test specifications
- **Deliverable**: Agreed-upon test invariants
- **Outcome**: QE can write Hypothesis tests confidently

### Day 9: All Hands Review
- **Topic**: Coverage report, gap-filling priorities
- **Deliverable**: List of untested branches/edge cases
- **Outcome**: Team commits to fill gaps to reach 95%+

---

## Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| **Domain coverage** | 95%+ | `pytest --cov=semantic.domain` |
| **Type errors** | 0 | `mypy --strict src/semantic/domain` |
| **Lint errors** | 0 | `ruff check src/semantic/domain` |
| **Docstring coverage** | 100% | `ruff check --select D src/semantic/domain` |
| **Block allocation time** | < 1ms | EXP-002 benchmark result |
| **Complexity** | CC < 10 (domain < 7) | `ruff check --select C90 src/semantic/domain` |

---

## Notes

- **No MLX in domain layer**: All unit tests use fake port implementations
- **Integration tests in Sprint 2**: Real MLX models will be tested in Sprint 2
- **Port interfaces already complete**: Sprint 0 delivered `ports/outbound.py`
- **Sprint 0 experiments passed**: EXP-003 (cache injection) ✅, EXP-004 (per-sequence extraction) ✅

---

**Last Updated**: 2026-01-24
**Next Review**: Day 5 (mid-sprint checkpoint)
