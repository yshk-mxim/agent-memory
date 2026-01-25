# ADR-001: Hexagonal Architecture (Ports & Adapters)

**Date**: 2026-01-24
**Status**: ✅ ACCEPTED (Sprint 2, Day 1)
**Deciders**: SE, PM, QE, ML
**Author**: SE (Software Engineer)

## Context

The POC (2,719 LOC) has grown organically with direct dependencies between API handlers, MLX inference, and cache persistence. This creates:

1. **Tight coupling**: API layer imports MLX directly, making testing difficult
2. **Mixed concerns**: Domain logic (block allocation) mixed with infrastructure (safetensors I/O)
3. **Low testability**: Cannot test domain logic without loading ML models
4. **Poor extensibility**: Adding OpenAI API requires duplicating inference logic

We need an architecture that:
- Separates domain logic from infrastructure
- Enables testing without MLX dependencies
- Supports multiple API protocols (Anthropic, OpenAI, Direct)
- Allows swapping infrastructure (e.g., different cache backends)

## Decision

Adopt **Hexagonal Architecture** (aka Ports & Adapters pattern) with strict dependency inversion:

```
Inbound Adapters → Ports → Application Services → Domain Core
                                ↓
                        Outbound Ports → Outbound Adapters
```

### Layers

1. **Domain Core** (`src/semantic/domain/`)
   - Pure Python, zero external dependencies
   - Entities: `AgentContext`, `KVBlock`, `AgentBlocks`
   - Value Objects: `ModelCacheSpec`, `CacheKey`, `GenerationResult`
   - Services: `BlockPool`
   - No imports from: `mlx`, `fastapi`, `safetensors`, `transformers`

2. **Ports** (`src/semantic/ports/`)
   - Protocol classes (Python 3.11+) defining interfaces
   - Inbound: `InferencePort`, `AgentManagementPort`
   - Outbound: `ModelBackendPort`, `CachePersistencePort`, `TokenizerPort`

3. **Application Services** (`src/semantic/application/`)
   - Orchestration logic: `ConcurrentScheduler`, `BlockPoolBatchEngine`, `AgentCacheStore`
   - Depends on domain core + ports (NOT adapters)

4. **Adapters**
   - **Inbound** (`src/semantic/adapters/inbound/`): Protocol-specific API handlers
   - **Outbound** (`src/semantic/adapters/outbound/`): MLX, safetensors, tokenizers
   - Depend on ports, implement interfaces

5. **Entrypoints** (`src/semantic/entrypoints/`)
   - Composition root: wires adapters → application → domain
   - FastAPI app factory, CLI

## Rationale

### Why Hexagonal (vs alternatives)?

| Alternative | Pros | Cons | Decision |
|------------|------|------|----------|
| **Layered (N-tier)** | Simple, familiar | Domain depends on infrastructure | ❌ Rejected |
| **Hexagonal** | Testable, clean separation, extensible | More files, indirection | ✅ **Selected** |
| **Clean Architecture** | Very rigorous | Overkill for project size | ❌ Rejected (too heavy) |
| **Flat module structure** | Minimal files | Poor separation, testing hard | ❌ Rejected (POC anti-pattern) |

### Key Advantages for This Project

1. **Testability**
   - Unit test `BlockPool` without MLX (use fake `ModelBackendPort`)
   - Unit test domain logic at 95%+ coverage with fast tests

2. **Multi-Protocol Support**
   - Add OpenAI adapter without touching domain or application layers
   - Anthropic, OpenAI, Direct APIs all use same `InferencePort`

3. **Infrastructure Swapping**
   - Replace safetensors with HDF5/Parquet without changing domain
   - Mock cache persistence in tests

4. **Dependency Direction**
   - Domain → No external deps
   - Application → Domain only
   - Adapters → Application + Domain + Infrastructure
   - Clean imports, no circular dependencies

## Consequences

### Positive

- **95%+ unit test coverage** on domain layer (target in Sprint 1)
- **Fast tests**: Domain tests run in < 1s (no MLX loading)
- **Easy to add protocols**: New API = new inbound adapter
- **Clear responsibility**: Each layer has single concern

### Negative

- **More files**: ~40 files vs POC's ~15
- **Indirection**: Request flows through 4-5 layers
- **Learning curve**: Team must understand ports/adapters pattern

### Neutral

- **Type annotations required**: All port interfaces use `Protocol` (mypy enforces)
- **Dependency injection**: Entrypoint manually wires dependencies

## Implementation Notes

### Dependency Rule (Enforced by CI)

**Imports MUST follow these rules:**
```python
# ✅ ALLOWED
# domain/ imports: typing, dataclasses, abc (stdlib only)
# application/ imports: domain/, ports/, stdlib
# adapters/ imports: domain/, ports/, application/, infrastructure libs

# ❌ FORBIDDEN
# domain/ imports mlx, fastapi, safetensors → FAIL CI
# application/ imports adapters/ → circular dependency
```

CI check (mypy + custom script):
```bash
# Verify domain has no external deps
grep -r "^import mlx" src/semantic/domain/ && exit 1
grep -r "^from mlx" src/semantic/domain/ && exit 1
```

### Port Interface Pattern

Use `Protocol` (PEP 544) for structural typing:

```python
# ports/outbound.py
from typing import Protocol

class ModelBackendPort(Protocol):
    def generate(self, prompt: str, cache: KVCache | None) -> GenerationResult:
        ...
```

Adapters implement implicitly (no inheritance needed):

```python
# adapters/outbound/mlx_backend.py
class MLXModelBackend:
    def generate(self, prompt: str, cache: KVCache | None) -> GenerationResult:
        # mlx_lm implementation
```

### Composition Root (Entrypoint)

```python
# entrypoints/server.py
def create_app() -> FastAPI:
    # Instantiate outbound adapters
    model_backend = MLXModelBackend(model_id="...")
    cache_persistence = SafetensorsCachePersistence(cache_dir="...")

    # Instantiate application services (inject adapters)
    engine = BlockPoolBatchEngine(model_backend)
    store = AgentCacheStore(cache_persistence)
    scheduler = ConcurrentScheduler(engine, store)

    # Instantiate inbound adapters (inject services)
    anthropic_api = AnthropicAPIAdapter(scheduler)

    # Wire to FastAPI
    app = FastAPI()
    app.include_router(anthropic_api.router)
    return app
```

## Validation

### Sprint 0 Validation

- [x] Directory structure created (all layers present)
- [ ] Domain layer has zero MLX/FastAPI imports (CI check)
- [ ] Port interfaces defined with `Protocol`
- [ ] One end-to-end request flows through all layers

### Sprint 1 Validation

- [ ] Domain unit tests achieve 95%+ coverage
- [ ] Domain tests run in < 1s (no MLX loading)
- [ ] mypy --strict passes on domain + ports

### Sprint 4 Validation

- [ ] Add OpenAI adapter without modifying domain/application
- [ ] Same `InferencePort` serves 3 protocols

## References

- [Hexagonal Architecture (Alistair Cockburn)](https://alistair.cockburn.us/hexagonal-architecture/)
- [Ports and Adapters Pattern](https://herbertograca.com/2017/11/16/explicit-architecture-01-ddd-hexagonal-clean-onion-architectures/)
- Python Protocols (PEP 544): https://peps.python.org/pep-0544/
- Related: ADR-002 (Block Size), ADR-003 (Eviction Strategy)
