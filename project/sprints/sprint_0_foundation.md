# Sprint 0: Foundation + Critical Feasibility

**Duration**: 1 week
**Status**: In Progress (Infrastructure complete, experiments pending manual execution)
**Date**: 2026-01-24

## Objectives

1. ✅ Create hexagonal directory structure
2. ✅ Configure build system (pyproject.toml with Hatchling)
3. ✅ Set up quality tooling (ruff, mypy, pre-commit)
4. ✅ Create Makefile with all targets
5. ✅ Create project/ artifact directory
6. ✅ Write ADR-001 (Hexagonal Architecture) and ADR-002 (Block Size)
7. ⏳ **EXP-003**: Validate cache injection into BatchGenerator (BLOCKING)
8. ⏳ **EXP-004**: Validate Response.prompt_cache() extraction (BLOCKING)

## Deliverables

### Infrastructure ✅

**Directory Structure:**
```
semantic/
├── src/semantic/          # Production code (hexagonal layers)
│   ├── domain/            # Pure business logic (no external deps)
│   ├── ports/             # Protocol interfaces
│   ├── application/       # Orchestration services
│   ├── adapters/          # Infrastructure bindings
│   │   ├── inbound/       # API adapters
│   │   ├── outbound/      # MLX, persistence, tokenizer
│   │   └── config/        # Settings management
│   └── entrypoints/       # Server, CLI
├── tests/
│   ├── unit/              # Fast tests, mocked boundaries
│   ├── integration/       # Real MLX + disk
│   ├── smoke/             # Server lifecycle
│   └── e2e/               # Full stack
├── project/               # Planning artifacts
│   ├── architecture/      # ADRs
│   ├── sprints/           # Sprint docs
│   ├── experiments/       # Feasibility results
│   └── templates/         # Document templates
└── config/                # TOML configs + .env
```

**Build Configuration (pyproject.toml):**
- Hatchling build backend
- Version from `__init__.py`
- Dependencies: mlx, mlx-lm, fastapi, pydantic, safetensors, etc.
- Dev dependencies: pytest, ruff, mypy, semgrep, schemathesis, hypothesis
- Quality gates: ruff (S rules, C90), mypy --strict, liccheck

**Quality Tooling:**
- `ruff`: Linter + formatter + import sorter + security (S rules)
- `mypy --strict`: Type checking
- `semgrep`: Custom security rules (configured separately)
- `radon` (via ruff C90): Cyclomatic complexity < 10 (domain: < 7)
- `liccheck`: OSS license policy (no GPL/LGPL/AGPL)
- `codespell`: Documentation spelling
- `pre-commit`: Automated git hooks

**Makefile Targets:**
- `make install`, `make dev-install`
- `make lint`, `make format`, `make typecheck`
- `make security`, `make complexity`, `make licenses`
- `make test`, `make test-unit`, `make test-integration`, `make test-smoke`, `make test-e2e`
- `make bench`, `make docs`, `make clean`
- `make all` (full quality pipeline), `make ci` (PR checks)

**Configuration Management:**
- `config/default.toml`: All defaults
- `config/test.toml`: Test overrides
- `config/.env.example`: Secret template
- Precedence: CLI > ENV > .env > TOML > Pydantic defaults

### Architecture Decisions ✅

**ADR-001: Hexagonal Architecture**
- Decision: Strict ports & adapters pattern
- Domain core has ZERO external dependencies (no mlx, fastapi, safetensors)
- Protocol-based ports (PEP 544)
- Dependency inversion: Adapters → Application → Domain
- Validation: CI checks domain imports, 95% unit coverage target

**ADR-002: Block Size = 256 Tokens**
- Decision: Universal 256-token blocks across all models
- Rationale: Aligns with sliding window (1024 ÷ 256 = 4), low overhead (77 KB metadata)
- Memory: 2 MB per block per layer (float16), 1 MB (8-bit quantized)
- Validation: Unit tests verify 4-block SW allocation, integration tests measure < 1ms overhead

### Critical Experiments ⏳

**EXP-003: Cache Injection Validation**
- **File**: `experiments/exp_003_cache_injection.py`
- **Goal**: Prove `caches=[loaded_cache]` works on `BatchGenerator.insert()`
- **Status**: ⏳ Pending manual execution (MLX requires Metal GPU, blocked by sandbox)
- **Success criteria**: Output with cache matches output without cache
- **Failure mode**: If fails → **INVOKE PLAN B** (sequential engine)

**EXP-004: Per-Sequence Cache Extraction**
- **File**: `experiments/exp_004_cache_extraction.py`
- **Goal**: Prove `Response.prompt_cache()` callable works on completion
- **Status**: ⏳ Pending manual execution (MLX requires Metal GPU, blocked by sandbox)
- **Success criteria**: Extract, save, reload, re-inject cycle works
- **Failure mode**: If fails → **INVOKE PLAN B** (sequential engine)

**Manual Execution Required:**
```bash
# Outside Claude Code sandbox
python experiments/exp_003_cache_injection.py
python experiments/exp_004_cache_extraction.py
```

### Domain Exception Hierarchy ✅

Created `src/semantic/domain/errors.py`:
- `SemanticError` (base)
- `AgentNotFoundError`
- `PoolExhaustedError`
- `CacheCorruptionError`
- `ModelSwapError`
- `CachePersistenceError`
- `InvalidRequestError`
- `ModelNotFoundError`
- `IncompatibleCacheError`

Rationale: More Pythonic than `Result[T]` monads, works with pytest/FastAPI.

## Exit Criteria

| Criterion | Status |
|-----------|--------|
| Directory structure created | ✅ Done |
| pyproject.toml configured | ✅ Done |
| Makefile with all targets | ✅ Done |
| Pre-commit hooks configured | ✅ Done |
| ADR-001 (Hexagonal) written | ✅ Done |
| ADR-002 (Block Size) written | ✅ Done |
| Config files created | ✅ Done (default.toml, test.toml, .env.example) |
| LICENSE (MIT) + NOTICE | ✅ Done |
| EXP-003 passes | ⏳ **Pending manual execution** |
| EXP-004 passes | ⏳ **Pending manual execution** |
| `make lint && make typecheck && make test` passes | ⏳ Pending (no code yet, only scaffolding) |

## Next Steps

### Immediate (Blocking for Sprint 1)

1. **Run EXP-003 and EXP-004 manually** (requires Metal GPU access)
   - If both pass → Continue with Sprint 1 (Domain Core)
   - If either fails → Invoke Plan B (sequential engine)

2. **Baseline POC performance metrics**
   - Record current single-agent decode speed
   - Record current batch-3 decode speed
   - Measure cache load time for 2K, 4K, 8K contexts

### Sprint 1 Preparation

Once experiments pass:
1. Implement `ModelCacheSpec` value object
2. Implement `BlockPool` service (allocate/free/budget)
3. Write unit tests for domain layer (target: 95%+ coverage)
4. Create fake implementations of ports for testing

## Risks

| Risk | Status | Mitigation |
|------|--------|----------|
| EXP-003/004 fail → Plan B required | ⏳ Pending | Plan B documented, sequential engine simpler |
| MLX sandbox limitation | ✅ Mitigated | Created scripts for manual execution |
| Over-engineering for 10 agents | Monitored | Simple LRU first, 3-tier only if needed |
| Breaking changes in mlx_lm | Monitored | Pin to v0.30.4, wrapper with types |

## Metrics

- **Files created**: 21
- **Lines of code**: ~1,200 (scaffolding + config + docs)
- **Quality checks configured**: 11 (ruff, mypy, semgrep, etc.)
- **Makefile targets**: 20
- **ADRs written**: 2
- **Experiments created**: 2 (pending execution)

## Notes

- **MLX limitation**: Claude Code sandbox blocks Metal GPU access. Experiments must run manually outside sandbox.
- **No production code yet**: Sprint 0 is infrastructure only. Domain implementation starts Sprint 1.
- **Configuration-driven**: Zero hardcoded values. All defaults in `config/default.toml`.
- **Dependency direction enforced**: CI will check domain/ has no mlx/fastapi imports (ADR-001 validation).

---

**Sprint 0 Summary**: Infrastructure complete, awaiting critical experiment validation to proceed with Sprint 1.
