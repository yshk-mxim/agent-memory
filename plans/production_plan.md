# Production Development Plan: Block-Pool Multi-Agent Inference Server

## Goal

Transform the working POC (2,719 LOC) into a **production-quality** multi-agent LLM inference server implementing the architecture described in `backend_plan.md`, `anthropic_cli_adapter.md`, and `novelty/continuous_batching.md`. The result is a hexagonal architecture with block-pool memory management, multi-protocol API, model hot-swap, and comprehensive testing/quality infrastructure.

**Scope**: Covers `project/` directory creation, sprint-based execution, multi-expert debate, feasibility experiments, quality gates, and deployment packaging.

---

## Architecture: Hexagonal (Ports & Adapters)

```
┌─────────────────────────────────────────────────────────────────────┐
│  INBOUND ADAPTERS (Driving)                                          │
│  ├── AnthropicAPIAdapter (content-based agent ID, SSE streaming)     │
│  ├── OpenAIAPIAdapter (session_id extension, OpenAI SSE format)      │
│  └── DirectAgentAPIAdapter (agent_id in URL, stateful mode)          │
└────────────────────────────┬────────────────────────────────────────┘
                             │ InferencePort / AgentManagementPort
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│  APPLICATION SERVICES (Orchestration)                                │
│  ├── ConcurrentScheduler (per-agent locks, 10ms batch window)        │
│  ├── BlockPoolBatchEngine (prefill/decode with block allocation)      │
│  ├── AgentCacheStore (prefix matching, LRU eviction)                 │
│  └── ModelRegistry (hot-swap, TTL unload, spec extraction)           │
└────────────────────────────┬────────────────────────────────────────┘
                             │ Uses Domain Core
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│  DOMAIN CORE (No external dependencies)                              │
│  ├── Entities: AgentContext, KVBlock, AgentBlocks                    │
│  ├── Value Objects: ModelCacheSpec, CacheKey, GenerationResult       │
│  ├── Services: BlockPool (allocate/free/budget/reconfigure)          │
│  └── Errors: Domain exception hierarchy                              │
└────────────────────────────┬────────────────────────────────────────┘
                             │ Outbound Ports (Protocols)
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│  OUTBOUND ADAPTERS (Driven)                                          │
│  ├── MLXModelBackend (mlx_lm load/prefill/decode/extract_cache)      │
│  ├── SafetensorsCachePersistence (block save/load, model-tagged)      │
│  ├── HFTokenizerAdapter (encode/decode)                              │
│  └── StructuredLogger (structlog + OpenTelemetry) / MetricsAdapter   │
└─────────────────────────────────────────────────────────────────────┘
```

**Dependency Rule**: All arrows point inward. Domain core has ZERO imports from mlx, fastapi, safetensors.

---

## Corrected mlx_lm API (v0.30.5, Sprint 3.5 Validated)

The plan depends on these **production-validated** mlx_lm capabilities (tested with real MLX models in Sprint 3.5):

```python
from mlx_lm.server import BatchGenerator
from mlx_lm.sample_utils import make_sampler

# 1. Create BatchGenerator (NO tokenizer parameter!)
gen = BatchGenerator(
    model=model,
    stop_tokens=set([tokenizer.eos_token_id]),  # Set of ints
)

# 2. Create sampler (parameter is 'temp', not 'temperature')
sampler = make_sampler(temp=0.0)  # Greedy sampling

# 3. Insert with pre-built caches
uids = gen.insert(
    prompts=[tokens_a, tokens_b],      # List of token lists
    max_tokens=[128, 256],             # MUST be list (one per prompt)
    caches=[cache_a, cache_b] if have_caches else None,
    samplers=[sampler, sampler],       # REQUIRED (one per prompt)
)

# 4. Token accumulation loop (1 token per sequence per call!)
tokens_by_uid = {uid: [] for uid in uids}

while responses := gen.next():  # Returns list[Response]
    # Termination: returns [] (empty list), NOT StopIteration
    if not responses:
        break

    for r in responses:
        # Accumulate token (Response has .token singular, NOT .text)
        if r.finish_reason != "stop":
            tokens_by_uid[r.uid].append(r.token)

        # Handle completion
        if r.finish_reason is not None:  # "stop" or "length"
            # Decode accumulated tokens
            text = tokenizer.decode(tokens_by_uid[r.uid])

            # Cache is direct attribute (NOT callable despite type hint)
            cache = r.prompt_cache  # List[KVCache] objects

# 5. Persist cache to disk (safetensors)
save_prompt_cache("agent_a.safetensors", cache)

# 6. Load cache from disk
loaded_cache = load_prompt_cache("agent_a.safetensors")
```

**Sprint 3.5 Validation Status**:
- ✅ Tested with SmolLM2-135M-Instruct model
- ✅ 5/6 integration tests passing (1 deferred to Sprint 4)
- ✅ Token accumulation works correctly
- ✅ Multi-agent batching works (3 concurrent agents)
- ✅ Memory management validated (no leaks)
- ⚠️ Cache reconstruction needs KVCache objects (Sprint 4 TODO)

**Critical API Discoveries** (Sprint 3.5):
1. BatchGenerator init: NO tokenizer parameter (documentation was incorrect)
2. max_tokens must be list (not scalar)
3. samplers parameter is REQUIRED (not optional)
4. make_sampler uses `temp=`, not `temperature=`
5. Response has .token (singular), must accumulate manually
6. Loop terminates on `next() == []` (not StopIteration)
7. prompt_cache is direct attribute (not callable)
8. Cache format: KVCache objects with .size() method (not tuples)

**Key constraints:**
- No mid-generation cache extraction — only when finish_reason is not None
- Individual sequences complete independently (batch continues for remaining)
- Caller must accumulate tokens (1 per next() call) and decode at completion
- Cache reconstruction needs conversion to KVCache objects (deferred to Sprint 4)
- API pinned to v0.30.5

---

## Corrected Memory Math (Gemma 3 12B)

**Actual parameters** (from config.json):
- `num_hidden_layers`: 48 (8 global + 40 sliding window, pattern=6)
- `num_key_value_heads`: **8** (not 4 as previously assumed)
- `head_dim`: 256
- `sliding_window`: 1024 tokens

**Per-block memory** (256 tokens, one layer, float16):
```
8 kv_heads × 256 head_dim × 2 (K+V) × 2 bytes × 256 tokens = 2 MB per layer per block
```

**With 8-bit quantization**: 1 MB per layer per block

**Full KV cache at 8K context per agent:**

| Config | Global (8 layers × 32 blocks) | Local (40 layers × 4 blocks) | Total |
|--------|-------------------------------|-------------------------------|-------|
| Float16 | 512 MB | 320 MB | **832 MB** |
| 8-bit quantized | 256 MB | 160 MB | **416 MB** |

**Memory budget (M4 Pro 24GB):**
```
Model weights (4-bit):     ~6.0 GB
OS + system overhead:      ~3.0 GB
MLX framework overhead:    ~1.0 GB
Available for caches:      ~14.0 GB
4 GB cache pool budget:    ~9-10 agents at 8K (8-bit quantized)
```

**Note**: Sliding window layers are already bounded at 1024 tokens (4 blocks). Block pooling provides value only for the 8 global layers where context grows unbounded.

---

## Directory Structure

```
semantic/
├── pyproject.toml                     # Build config (Hatchling), deps, tool settings
├── Makefile                           # lint, test, typecheck, format, bench, docs
├── LICENSE                            # MIT
├── NOTICE                             # Apache-2.0 dependency attributions
├── .pre-commit-config.yaml
├── .github/workflows/
│   ├── ci.yml                         # PR: lint + type + unit + smoke
│   └── nightly.yml                    # Full suite + license scan + SBOM
├── config/
│   ├── default.toml                   # All defaults (no hardcoded values in code)
│   ├── test.toml                      # Test overrides
│   └── .env.example                   # Secret template
├── src/semantic/
│   ├── __init__.py                    # importlib.metadata.version()
│   ├── py.typed                       # PEP 561
│   ├── domain/
│   │   ├── entities.py                # AgentContext, KVBlock, AgentBlocks
│   │   ├── value_objects.py           # ModelCacheSpec, CacheKey, GenerationResult
│   │   ├── services.py               # BlockPool logic
│   │   └── errors.py                 # Domain exception hierarchy
│   ├── ports/
│   │   ├── inbound.py                # InferencePort, AgentManagementPort, ModelManagementPort
│   │   └── outbound.py               # ModelBackendPort, CachePersistencePort, TokenizerPort
│   ├── application/
│   │   ├── batch_engine.py            # BlockPoolBatchEngine
│   │   ├── scheduler.py              # ConcurrentScheduler
│   │   ├── agent_store.py            # AgentCacheStore (Dict + prefix matching; trie if profiling shows need)
│   │   └── model_registry.py         # ModelRegistry
│   ├── adapters/
│   │   ├── inbound/
│   │   │   ├── anthropic_api.py       # Anthropic Messages API
│   │   │   ├── openai_api.py          # OpenAI-compatible + session_id
│   │   │   └── direct_api.py          # Direct agent API
│   │   ├── outbound/
│   │   │   ├── mlx_backend.py         # MLX model loading + inference (typed wrapper)
│   │   │   ├── safetensors_cache.py   # Disk persistence
│   │   │   ├── hf_tokenizer.py        # HuggingFace tokenizer
│   │   │   └── metrics.py            # structlog + Prometheus metrics
│   │   └── config/
│   │       └── settings.py            # Pydantic Settings (all config)
│   └── entrypoints/
│       ├── server.py                  # FastAPI app factory + DI composition root
│       └── cli.py                     # CLI entrypoint
├── tests/
│   ├── conftest.py                    # Markers, shared fakes (no MLX deps)
│   ├── unit/                          # Pure domain (mocked boundaries)
│   ├── integration/                   # Real MLX + real disk (skip non-Apple)
│   ├── smoke/                         # Server lifecycle + basic requests
│   └── e2e/                           # Full stack multi-agent flows
├── benchmarks/
│   ├── bench_block_pool.py
│   ├── bench_batched_decode.py
│   └── bench_cache_load.py
├── project/                           # Planning artifacts (see below)
└── docs/
    ├── mkdocs.yml                     # MkDocs Material config + mermaid
    ├── quick-start.md                 # 5 minutes to first request
    ├── installation.md                # Prerequisites, platforms, methods
    ├── user-guide.md                  # Configuration, multi-agent, CLI
    ├── developer-guide.md             # Contributing, testing, code style
    ├── architecture.md                # Hexagonal layers + Mermaid diagrams
    ├── model-onboarding.md            # Adding new model architectures
    ├── api-reference.md               # Auto-generated from docstrings
    └── deployment.md                  # Standalone, launchd (macOS native)
```

---

## Configuration Management (No Hardcoded Values)

```python
# src/semantic/adapters/config/settings.py
class MLXSettings(BaseSettings):
    model_id: str = "mlx-community/gemma-3-12b-it-4bit"
    max_batch_size: int = Field(5, ge=1, le=20)
    prefill_step_size: int = 512
    kv_bits: int | None = None
    block_tokens: int = 256
    cache_budget_mb: int = 4096

class AgentSettings(BaseSettings):
    max_agents_in_memory: int = Field(5, ge=1)
    cache_dir: str = "~/.semantic/caches"
    batch_window_ms: int = Field(10, ge=1, le=1000)

class ServerSettings(BaseSettings):
    host: str = "127.0.0.1"            # CORRECTED: localhost-only by default
    port: int = 8000
    allow_remote: bool = False         # Explicit opt-in for 0.0.0.0
    shutdown_timeout_seconds: float = 30.0

class SecretsSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SEMANTIC_", env_file=".env")
    api_key: SecretStr = Field(default=SecretStr(""))
```

**Precedence**: CLI args > ENV vars > `.env` > `config/{environment}.toml` > `config/default.toml` > Pydantic defaults

---

## Error Handling (Domain Exception Hierarchy)

```python
# src/semantic/domain/errors.py
class SemanticError(Exception):
    """Base for all domain errors."""

class AgentNotFoundError(SemanticError): ...
class PoolExhaustedError(SemanticError): ...
class CacheCorruptionError(SemanticError): ...
class ModelSwapError(SemanticError): ...
class CachePersistenceError(SemanticError): ...
class InvalidRequestError(SemanticError): ...
```

Rationale: Custom exception hierarchy is more Pythonic than `Result[T]` monads and works naturally with pytest assertions and FastAPI exception handlers.

---

## Code Documentation Standards (Production Quality)

All production code must follow these documentation requirements enforced by CI.

### Docstring Requirements (Google Style)

**Required for:**
- All public classes
- All public methods/functions
- All port interfaces (Protocol classes)
- All domain services
- Complex private methods (CC > 5)

**Format** (Google-style, enforced by ruff D rules):

```python
def allocate(self, n_blocks: int, layer_type: str = "global") -> list[KVBlock]:
    """Allocate blocks from the free pool.

    Args:
        n_blocks: Number of blocks to allocate.
        layer_type: Type of layer ("global" or "sliding_window").
            Sliding window layers have block caps enforced.

    Returns:
        List of allocated KVBlock instances.

    Raises:
        PoolExhaustedError: If insufficient blocks available.
        ValueError: If n_blocks < 1 or layer_type invalid.

    Example:
        >>> pool = BlockPool(total_blocks=100, spec=spec)
        >>> blocks = pool.allocate(4, layer_type="sliding_window")
        >>> len(blocks)
        4
    """
```

**Rationale sections** (for non-obvious design choices):

```python
class AgentCacheStore:
    """Manages per-agent KV cache blocks with prefix matching.

    Uses Dict + longest_common_prefix() instead of trie data structure.

    Rationale:
        For ≤10 concurrent agents, Dict O(n) prefix scan is faster than
        trie O(m) traversal due to lower constant factors and cache locality.
        Profiling showed Dict: 0.02ms, Trie: 0.15ms for 10 agents.
        Upgrade to trie only if profiling shows >50ms with production load.
    """
```

### Type Annotations (Always Required)

**Enforced by `mypy --strict`:**

```python
# ✅ GOOD: Full type annotations
def process(
    agent_id: str,
    cache: list[RotatingKVCache] | None,
    timeout: float = 30.0
) -> tuple[str, int]:
    ...

# ❌ BAD: Missing types (mypy error)
def process(agent_id, cache=None):  # type: ignore
    ...
```

**Generic types:**

```python
from typing import Protocol, TypeVar

T = TypeVar("T")

class CacheStore(Protocol[T]):
    """Generic cache store interface."""
    def get(self, key: str) -> T | None: ...
    def put(self, key: str, value: T) -> None: ...
```

### When to Comment (Inline)

**✅ REQUIRED Comments:**

1. **Non-obvious algorithms** (CC > 7):
   ```python
   # Use block-aligned hashing to ensure cache key matches block boundaries.
   # SHA-256 on first 256 tokens per block, then XOR-fold to 64 bits.
   block_hash = self._hash_prefix(tokens[:self.BLOCK_TOKENS])
   ```

2. **Performance-critical sections**:
   ```python
   # PERF: Batch mx.concatenate to avoid N individual Metal GPU calls.
   # Reduces overhead from 15ms to 2ms for 8K context (32 blocks).
   all_keys = mx.concatenate([b.keys for b in blocks], axis=0)
   ```

3. **Workarounds for upstream bugs**:
   ```python
   # WORKAROUND: mlx_lm v0.30.4 BatchGenerator doesn't expose tokenizer.
   # Must pre-tokenize before insert(). Remove if API fixed in v0.31+.
   tokens = self.tokenizer.encode(prompt)
   ```

4. **Security-sensitive code**:
   ```python
   # SECURITY: Validate agent_id to prevent path traversal attacks.
   # Only allow alphanumeric + hyphen to ensure safe filesystem paths.
   if not re.match(r'^[a-zA-Z0-9-]+$', agent_id):
       raise InvalidRequestError(f"Invalid agent_id: {agent_id}")
   ```

5. **Complex business rules** from design docs:
   ```python
   # Per anthropic_cli_adapter.md §3.2: System prompt hash determines agent_id.
   # This enables cache reuse across sessions with identical system prompts.
   agent_id = hashlib.sha256(canonical_system.encode()).hexdigest()[:16]
   ```

**❌ AVOID Comments:**

1. **Redundant with code** (let code self-document):
   ```python
   # BAD: Comment just repeats code
   # Increment the counter
   counter += 1

   # GOOD: Self-documenting with better naming
   completed_requests_count += 1
   ```

2. **Obvious operations**:
   ```python
   # BAD
   # Get the agent from the store
   agent = self.store.get(agent_id)

   # GOOD: No comment needed, operation is clear
   agent = self.store.get(agent_id)
   ```

3. **TODO/FIXME in production code** (use issues instead):
   ```python
   # ❌ BAD: TODO in production
   # TODO: optimize this later
   result = slow_operation()

   # ✅ GOOD: Issue filed, reference in commit/PR
   # See issue #123 for optimization opportunity
   ```

### Self-Documenting Code Principles

**Prefer expressive names over comments:**

```python
# ❌ BAD: Comment explains unclear code
# Check if we have enough blocks
if pool.free > req:
    ...

# ✅ GOOD: Code explains itself
if pool.has_available_blocks(requested_count):
    ...
```

**Extract complex expressions:**

```python
# ❌ BAD: Complex boolean needs comment
# Check if cache is valid and not expired and model matches
if cache and cache.ts > now - 3600 and cache.model == current:
    ...

# ✅ GOOD: Extracted method with descriptive name
if self._is_cache_valid_for_model(cache, current_model):
    ...

def _is_cache_valid_for_model(
    self,
    cache: AgentCache | None,
    model_id: str
) -> bool:
    """Check cache is recent, valid, and matches current model."""
    if cache is None:
        return False
    is_recent = cache.timestamp > time.time() - 3600
    model_matches = cache.model_id == model_id
    return is_recent and model_matches
```

### Documentation Coverage Gates

| Layer | Docstring Coverage | Enforced By |
|-------|-------------------|-------------|
| Domain core | 100% public APIs | ruff D + CI check |
| Ports (Protocols) | 100% all methods | ruff D + CI check |
| Application services | 95% public APIs | ruff D + CI check |
| Adapters | 80% public APIs | ruff D + CI check |
| Tests | Examples in docstrings | pytest --doctest-modules |

### Complexity Threshold → Comment Requirement

| Cyclomatic Complexity | Requirement |
|----------------------|-------------|
| CC ≤ 5 | No inline comments required (self-documenting) |
| 5 < CC ≤ 10 | Algorithm explanation comment required |
| CC > 10 | **Refactor required** (exceeds gate, see Quality Pipeline) |

### API Reference Auto-Generation

All port interfaces and public domain APIs must support mkdocs auto-generation:

```python
class InferencePort(Protocol):
    """Port for inference operations.

    This port defines the contract for text generation services.
    Implementations may use different backends (MLX, vLLM, etc.)
    but must provide consistent semantics.

    Thread safety: Implementations must be thread-safe for
    concurrent access from multiple agents.
    """

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7
    ) -> GenerationResult:
        """Generate text continuation from prompt.

        Args:
            prompt: Input text to continue.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature (0.0 = greedy).

        Returns:
            GenerationResult with text, tokens, and cache.

        Raises:
            ModelNotFoundError: If model not loaded.
            PoolExhaustedError: If no blocks available.
        """
        ...
```

**mkdocs config** (automatically extracts above into API reference):

```yaml
# docs/mkdocs.yml
plugins:
  - mkdocstrings:
      handlers:
        python:
          options:
            show_source: true
            show_root_heading: true
            heading_level: 3
```

### Pre-commit Enforcement

```yaml
# .pre-commit-config.yaml
- repo: https://github.com/astral-sh/ruff-pre-commit
  hooks:
    - id: ruff
      args: [--select=D]  # Docstring rules
```

**Ruff D rules enabled** (pyproject.toml):

```toml
[tool.ruff.lint.pydocstyle]
convention = "google"  # Google-style docstrings

[tool.ruff.lint]
select = [
    "D",      # pydocstyle (docstring rules)
]
ignore = [
    "D100",   # Missing module docstring (allow for simple modules)
    "D104",   # Missing package docstring
    "D203",   # Conflicts with D211
    "D213",   # Conflicts with D212
]
```

### Examples Repository

All complex features require working examples in `examples/`:

```
examples/
├── basic_inference.py          # Simplest usage
├── multi_agent_batching.py     # Concurrent agents
├── cache_persistence.py        # Save/load cycle
└── custom_model_onboarding.py  # Add new architecture
```

**Each example includes:**
- Docstring explaining what it demonstrates
- Inline comments for non-obvious steps
- Expected output in comments
- Link to relevant docs section

---

## 2025-2026 Best Practices Update (Sprint 1 Research)

**Research Date**: January 24, 2026
**Sources**: Industry standards, GitHub portfolio analysis, modern Python tooling

### Critical Additions (Implemented in Sprint 1)

#### 1. Hypothesis Property-Based Testing (CRITICAL ⭐)
**Status**: ✅ Implemented (3 property tests for BlockPool)

- **2025 Standard**: Property-based testing is now a **hiring signal** for senior positions
- **Impact**: Found edge cases traditional tests miss (zero blocks, max blocks, concurrent allocation)
- **Library**: Hypothesis >=6.122.0
- **Coverage**: All core invariants must have property tests
  - BlockPool: `used + available = total` (ALWAYS)
  - KVBlock: `0 <= token_count <= 256` (ALWAYS)
  - AgentBlocks: `total_tokens = sum(block.token_count)` (ALWAYS)

**Example**:
```python
@given(
    total_blocks=st.integers(min_value=10, max_value=100),
    n_blocks=st.integers(min_value=1, max_value=50),
)
def test_property_invariant(total_blocks, n_blocks):
    assume(n_blocks <= total_blocks)
    pool = BlockPool(spec, total_blocks)
    # Property: used + available = total (ALWAYS holds)
    assert pool.allocated() + pool.available() == total_blocks
```

**Reference**: https://hypothesis.readthedocs.io/ | https://danielsarney.com/blog/python-testing-2025/

#### 2. MkDocs + Material Theme (THE 2025 Documentation Standard)
**Status**: ✅ Implemented (13 docs, auto-gen API via mkdocstrings)

- **2025 Standard**: MkDocs Material is used by FastAPI, Pydantic, and 90% of modern Python projects
- **Features Enabled**:
  - Auto-generated API docs from docstrings (mkdocstrings-python)
  - Native Mermaid diagram rendering (6 diagrams in architecture.md)
  - Dark/light mode toggle
  - Mobile-responsive
  - Search with highlighting
  - Code copy buttons
- **Build**: `mkdocs build --strict` (zero warnings ✅)
- **Deploy**: GitHub Pages ready

**Documentation Structure**:
```
docs/
├── index.md                 # Home with architecture diagram
├── quick-start.md           # 5-minute guide
├── installation.md          # Complete setup
├── architecture.md          # Hexagonal + 6 Mermaid diagrams
├── developer-guide.md       # Contributing + testing
├── api-reference.md         # Auto-generated from docstrings
└── (7 more guides)
```

**Reference**: https://realpython.com/python-project-documentation-with-mkdocs/

#### 3. GitHub Portfolio Best Practices (Research Credibility)
**Status**: ✅ Implemented (7 badges, professional README)

- **Statistic**: **71% of hiring managers review GitHub** before proceeding with candidates
- **For Research**: Badges signal quality and professionalism
- **Badges Added**:
  - CI status (green = tests passing)
  - Coverage percentage (95.07% domain)
  - Python version (3.11+)
  - License (MIT)
  - Code style (ruff)
  - Type checking (mypy)

**Professional Tone**: Focused on research contributions, not job-seeking ("Block-pool memory management" not "Impress hiring managers")

**Reference**: https://www.finalroundai.com/articles/github-developer-portfolio

#### 4. Python 3.12 Matrix Builds (2025 Standard)
**Status**: ✅ Implemented (.github/workflows/ci.yml)

- **2025 Standard**: Test on BOTH 3.11 and 3.12 (not just one version)
- **CI Jobs**: All 3 jobs (lint, test, security) run on matrix:
  ```yaml
  strategy:
    matrix:
      python-version: ["3.11", "3.12"]
  ```
- **Why**: Python 3.12 has breaking changes (tomllib, typing improvements)

#### 5. Ruff (200x Faster Than Flake8/Pylint)
**Status**: ✅ Already using ruff

- **2025 Standard**: Ruff has replaced flake8, pylint, isort, black in modern projects
- **Speed**: 200x faster than legacy tools
- **Features**: Linting + formatting + import sorting in one tool
- **Config**: All in `pyproject.toml`

**Reference**: https://simone-carolini.medium.com/modern-python-code-quality-setup/

### Sprint 1 Achievements (January 24, 2026)

| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| **Domain Coverage** | 95%+ | 95.07% | ✅ PASS |
| **Total Tests** | N/A | 112 passing | ✅ PASS |
| **Property Tests** | ≥1 | 3 (BlockPool invariants) | ✅ PASS |
| **MkDocs Build** | Strict mode | 0 warnings | ✅ PASS |
| **GitHub Badges** | ≥5 | 7 badges | ✅ PASS |
| **CI Matrix** | 3.11 + 3.12 | Both versions | ✅ PASS |
| **Type Coverage** | mypy --strict | 100% | ✅ PASS |

### Updated Tool Requirements (2025)

**MUST HAVE** (blocking):
- ✅ Hypothesis (property-based testing)
- ✅ MkDocs + Material theme + mkdocstrings
- ✅ Ruff (linting + formatting)
- ✅ MyPy --strict
- ✅ Pre-commit hooks
- ✅ pytest + pytest-cov + pytest-asyncio

**NICE TO HAVE** (consider in future sprints):
- mkdocs-git-revision-date-localized-plugin (show last updated dates)
- mkdocs-minify-plugin (faster page loads)
- pytest-xdist (parallel test execution)

### Integration into Sprint Plan

**Sprint 0** (add to existing tasks):
- Set up MkDocs + Material theme (DE)
- Configure mkdocstrings for API auto-gen (DE)
- Add GitHub badges to README (DE)
- Configure Python 3.11 + 3.12 matrix in CI (SysE)

**Sprint 1** (add to existing tasks):
- Write Hypothesis property tests for BlockPool (QE) ✅ COMPLETED
- Write 5 core documentation guides (DE)
- Update production_plan.md with 2025 findings (DE) ✅ IN PROGRESS

**Sprint 7** (Documentation sprint):
- Complete all 13 documentation files
- Add Mermaid diagrams (10 total across docs)
- Deploy docs to GitHub Pages
- Verify `mkdocs build --strict` passes

---

## Quality Pipeline

| Tool | Purpose | Config | Gate |
|------|---------|--------|------|
| `ruff` | Lint + format + import sort + security (S rules) | pyproject.toml | 0 errors |
| `mypy --strict` | Type checking | pyproject.toml | 0 errors (new code) |
| `semgrep` | Custom security rules | .semgrep/ | 0 high/critical |
| `ruff C90` | Cyclomatic complexity | pyproject.toml | CC < 10 (domain: < 7) |
| `liccheck` | OSS license policy | pyproject.toml | No GPL/LGPL/AGPL |
| `syft` | SBOM generation (CycloneDX 1.6) | CI config | Generated nightly |
| `pytest-cov` | Coverage | pyproject.toml | See differentiated targets below |
| `schemathesis` | API contract testing | OpenAPI spec | 0 schema violations |
| `hypothesis` | Property-based testing (BlockPool) | conftest.py | All properties pass |
| `codespell` | Documentation spelling | pyproject.toml | 0 errors |
| `pre-commit` | Git hooks | .pre-commit-config.yaml | All checks pass |

**Blocked Licenses**: GPL, GPLv2, GPLv3, LGPL, AGPL, SSPL, EUPL
**Approved Licenses**: MIT, BSD, Apache-2.0, ISC, PSF, Zlib

### Coverage Targets (Differentiated)

| Code Area | Unit | Integration |
|-----------|------|-------------|
| Domain core (services, entities) | 95% | — |
| Application layer (scheduler, engine) | 85% | 80% |
| Adapters (API, MLX, persistence) | 70% | 80% |
| Eviction/concurrency critical paths | 100% | 100% |
| Overall project | 85% | 75% |

---

## Pre-commit Configuration

```yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-toml
      - id: check-merge-conflict
      - id: debug-statements
      - id: check-added-large-files
        args: ['--maxkb=500']      # Prevent accidental model weight commits

  - repo: https://github.com/astral-sh/ruff-pre-commit
    hooks:
      - id: ruff-format
      - id: ruff
        args: [--fix]

  - repo: https://github.com/gitleaks/gitleaks
    hooks:
      - id: gitleaks              # Secret detection (API keys, tokens)

  - repo: https://github.com/pre-commit/mirrors-mypy
    hooks:
      - id: mypy
        additional_dependencies: [types-requests, pydantic]
```

---

## Testing Strategy (4 Levels)

| Level | Scope | Runner | Model | Trigger |
|-------|-------|--------|-------|---------|
| **Unit** | Domain + application (mocked ports) | Any | Fakes | Every push |
| **Integration** | Real MLX + real disk | macos-14 | SmolLM2-135M + fake_hybrid_spec | Every push |
| **Smoke** | Server start + health + basic request | macos-14 | SmolLM2-135M | After unit+int |
| **E2E** | Multi-agent, cache persist, model swap | macos-14 | Gemma 3 12B | Nightly/manual |

**Markers**: `@pytest.mark.unit`, `@pytest.mark.integration`, `@pytest.mark.smoke`, `@pytest.mark.e2e`
**Skip Strategy**: Integration/E2E tests skip on non-Apple Silicon via `@pytest.mark.skipif`
**Async config**: `asyncio_mode = "auto"` in pyproject.toml

### Integration Test Fixtures for Hybrid Attention

SmolLM2-135M uses uniform full attention, which cannot test hybrid SWA+global paths. Add:

```python
@pytest.fixture
def fake_hybrid_cache_spec() -> ModelCacheSpec:
    """Simulates Gemma 3's hybrid pattern: 5 global + 10 sliding window layers."""
    return ModelCacheSpec(
        n_layers=15,
        n_kv_heads=8,
        head_dim=256,
        block_tokens=256,
        layer_types=["global"] * 5 + ["sliding_window"] * 10,
        sliding_window_size=1024,
    )
```

### Required Failure-Mode Test Scenarios

| # | Scenario | Level | Sprint |
|---|----------|-------|--------|
| 1 | Concurrent agent creation (same agent_id race) | Unit | S1 |
| 2 | Pool exhaustion mid-decode (block allocation failure) | Unit | S2 |
| 3 | Cache corruption on disk (truncated safetensors) | Integration | S3 |
| 4 | Batch worker unhandled exception (future resolution) | Unit | S2 |
| 5 | Disk full during eviction (save fails) | Integration | S3 |
| 6 | Cache reload exceeds available pool blocks | Unit | S3 |
| 7 | Model hot-swap during active batch | Integration | S5 |
| 8 | Malformed API request (invalid model, empty messages) | Smoke | S4 |
| 9 | Agent lock starvation (long generation blocking) | Unit | S2 |

---

## Sprint Plan (9 Sprints)

### Sprint 0: Foundation + Critical Feasibility (1 week)

**Deliverable**: CI/CD green, hexagonal scaffold, quality tooling configured, **critical experiments validated**.

**✅ UPDATE (Sprint 3.5)**: EXP-003 and EXP-004 are **ALREADY VALIDATED** with real MLX models. See Sprint 3.5 completion report for details.

| Task | Expert | Status |
|------|--------|--------|
| ✅ **EXP-003: Validate cache injection into BatchGenerator** | ML | **DONE (Sprint 3.5)** |
| ✅ **EXP-004: Validate Response.prompt_cache extraction** | ML | **DONE (Sprint 3.5)** |
| Create hexagonal directory structure | SE | Pending |
| Define Protocol-based ports (GenerationEngine, CacheStore, ModelProvider) | SE | Pending |
| Configure pyproject.toml (Hatchling build, deps, tool settings) | OSS | Pending |
| Set up ruff (with S + C90), mypy --strict, semgrep | QE | Pending |
| Configure pytest with markers, coverage, async (auto mode) | QE | Pending |
| Create Makefile (lint, test, typecheck, format, bench, docs) | SysE | Pending |
| GitHub Actions CI (lint + typecheck + unit) | SysE | Pending |
| Create `project/` artifact directory | DE | Pending |
| Write ADR-001: Hexagonal Architecture | SE | Pending |
| Write ADR-002: Block Size = 256 Tokens | SE | Pending |
| Define quality gate thresholds | QE | Pending |
| Triage existing 30 tests (keep/adapt/replace) | QE | Pending |
| Baseline POC performance metrics | ML | Pending |
| Create NOTICE file for Apache-2.0 deps | OSS | Pending |

**✅ Experiments (VALIDATED in Sprint 3.5)**:
- **✅ EXP-003**: Cache injection validated with real SmolLM2-135M model. Insert with `caches=[cache]` works correctly.
- **✅ EXP-004**: Cache extraction validated. `response.prompt_cache` is direct attribute (not callable). Extraction works on completion.

**Sprint 3.5 Bonus Validation**:
- ✅ Token accumulation pattern confirmed (1 token per next() call)
- ✅ Multi-agent batching works (3 concurrent agents tested)
- ✅ Memory management validated (no block leaks)
- ✅ Pool exhaustion handling works
- ⚠️ Cache reconstruction needs KVCache object conversion (Sprint 4 TODO)

**Critical API Corrections Discovered**:
1. BatchGenerator init: NO tokenizer parameter
2. max_tokens must be list (not scalar)
3. samplers parameter is REQUIRED
4. make_sampler uses `temp=`, not `temperature=`
5. Response.prompt_cache is attribute (not callable)
6. Token accumulation is manual (1 per call)

**Exit Gate**: `make lint && make typecheck && make test` all pass; CI green; ~~EXP-003 + EXP-004 pass~~ **Already validated in Sprint 3.5**

---

### Sprint 1: Domain Core (2 weeks)

**Deliverable**: `ModelCacheSpec` extracts specs from 4 architectures; `BlockPool` allocates/frees correctly.

| Task | Expert |
|------|--------|
| Domain types: ModelCacheSpec, KVBlock, AgentBlocks, CacheKey | SE |
| Domain exceptions: SemanticError hierarchy | SE |
| ModelCacheSpec.from_model() for Gemma 3 (hybrid SWA:global, n_kv_heads=8) | ML |
| ModelCacheSpec.from_model() for GPT-OSS-20B (MoE alternating) | ML |
| ModelCacheSpec.from_model() for Qwen 2.5 (uniform full) | ML |
| ModelCacheSpec.from_model() for Llama 3.1 (uniform full) | ML |
| Port interfaces: ModelBackendPort, CachePersistencePort, TokenizerPort | SE |
| BlockPool: allocate, free, budget, reconfigure, pressure detection | SE, HW |
| Note: SWA layers already bounded at 4 blocks (1024/256) — pool mainly for global layers | ML |
| Unit tests for ModelCacheSpec (all 4 architectures) | QE |
| Unit tests for BlockPool (allocate, free, budget, OOM, concurrent creation race) | QE |
| **Hypothesis property tests for BlockPool invariants** | QE |
| ADR-003: Cache Eviction Strategy (simple LRU vs 3-tier) | SE |

**Experiments**:
- EXP-001: Validate `model.args` attributes across all 4 target models
- EXP-002: Measure block allocation overhead (target < 1ms)

**Exit Gate**: 95%+ coverage on domain layer; mypy clean; all 4 architectures tested; Hypothesis passes

---

### Sprint 2: Block-Pool Batch Engine (2 weeks)

**Deliverable**: Engine generates correct text using block-pool allocation with variable-length batching.

| Task | Expert |
|------|--------|
| Typed MLX adapter layer (wraps mlx_lm with full type annotations) | SE, ML |
| Block-to-cache reconstruction (blocks → KVCache via mx.concatenate, one-time at restore) | ML |
| Cache-to-block extraction (via Response.prompt_cache() on completion) | ML |
| BlockPoolBatchEngine.submit() (allocate blocks, insert into BatchGenerator with caches) | SE |
| BlockPoolBatchEngine.step() (wraps gen.next(), yields completed generations) | ML |
| Block extension during decode (allocate every 256 tokens) | HW |
| Prefill with blocks (chunked, mx.eval after each chunk) | ML |
| Integration test: single agent output matches reference | QE |
| Integration test: 3 agents, variable lengths, correct output | QE |
| **Failure-mode tests: pool exhaustion mid-decode, batch worker crash, lock starvation** | QE |
| Benchmark: block gather vs padded approach | ML, HW |
| ADR-004: Block Gather Strategy (one-time at restore, not per-step) | ML |
| ADR-005: Composition Pivot (why custom engine replaces BatchGenerator wrapping) | SE |

**Experiments**:
- EXP-005: Actual decode throughput at 1, 3, 5, 7, 10 agents
- EXP-006: Block gather (mx.concatenate) overhead for 8K context — target < 5ms

**Exit Gate**: Output matches reference (greedy); no memory leaks; < 20% throughput regression vs POC

---

### Sprint 2.5: Critical Hotfix (1 week) ✅ COMPLETED

**Deliverable**: Fix all critical/high severity issues from Sprint 0-2 code review. Portfolio-quality codebase.

**Status**: ✅ **COMPLETE** (2026-01-24)

| Issue | Severity | Fix | Status |
|-------|----------|-----|--------|
| #1-3: Thread-safety violations (7 data races) | CRITICAL | Added threading.Lock to BlockPool, removed unsafe AgentBlocks methods | ✅ FIXED |
| #4: Opaque Any types (8 instances) | HIGH | TYPE_CHECKING guards, proper annotations | ✅ FIXED |
| #5-7: Memory leaks (3 scenarios) | CRITICAL/HIGH | Explicit layer_data cleanup in free() and step() | ✅ FIXED |
| #8: Partial allocation race in _extract_cache() | CRITICAL | Pre-allocate all blocks atomically per layer | ✅ FIXED |

**Quality Enhancements**:
- Enhanced ruff with 18 rule sets (ERA, SIM, PIE, RET, ARG, PTH, etc.)
- Configured complexity limits (max-args=7, max-branches=12, max-statements=50)
- Added quality pipeline: lint + typecheck + security + vulnerabilities + complexity
- Streamlined dependencies (removed overlapping tools, ruff covers all)

**Test Results**: 108/108 unit tests + 7/7 concurrent tests passing, mypy --strict clean, ruff clean

**Documentation**:
- `project/reviews/sprint-0-2-critical-review.md` (comprehensive review, 1475 lines)
- `project/reviews/sprint-2.5-review.md` (sign-off document)
- `project/reviews/sprint-2.5-complete-fixes.md` (detailed fix summary)

**Commits**: b2a2651, 855b93b

**Exit Gate**: ✅ All critical issues resolved, production-ready

---

### Sprint 3: AgentCacheStore (2 weeks)

**Deliverable**: Cache store with prefix matching, LRU eviction, model-tagged persistence.

| Task | Expert |
|------|--------|
| AgentCacheStore with Dict[str, AgentBlocks] + longest_common_prefix() | SE |
| AgentCacheStore.get() with prefix matching | SE |
| AgentCacheStore.put() with reference counting | SE |
| evict_to_disk() (safetensors, model-tagged, atomic write: tmp + rename) | SE |
| load_from_disk() (validate model tag, allocate blocks, async I/O) | SE |
| LRU eviction policy (simple LRU first; 3-tier only if profiling shows need) | SE |
| Integration test: save/load roundtrip | QE |
| Integration test: prefix matching across turns | QE |
| Integration test: model-tag invalidation | QE |
| **Failure-mode tests: cache corruption, disk full, reload exceeds pool** | QE |
| **DEFERRED from Sprint 2.5: Refactor ModelCacheSpec.from_model() (Issue #9)** | SE, ML |
| **DEFERRED from Sprint 2.5: Strategy pattern for model extractors (Issue #10)** | SE, ML |
| **DEFERRED from Sprint 2.5: Replace magic number 256 with block_tokens (Issue #13)** | SE |
| **NEW-1: Fix TOCTOU race in _extract_cache() (Second Technical Fellow Review)** | SE |
| **NEW-2: Add locks to used_memory() and available_memory() (Second Review)** | SE |

**Deferred Issues from Sprint 2.5 Critical Review**:
- **Issue #9** (MEDIUM, 4h): Refactor `ModelCacheSpec.from_model()` (94 lines, CC=8)
  - Extract `_extract_config()`, `_validate_config()`, `_build_spec()` helper methods
  - Reduce complexity, improve readability
  - Integrate during model onboarding work in Sprint 3
- **Issue #10** (MEDIUM, 6h): Remove Gemma 3 hardcoded special case
  - Create Strategy pattern for model config extraction
  - Add GemmaConfigExtractor, StandardConfigExtractor classes
  - Makes adding new models easier (model onboarding)
- **Issue #13** (LOW, 1h): Replace magic number 256
  - Find all hardcoded `256` and replace with `self._spec.block_tokens`
  - Improves maintainability

**NEW Issues from Second Independent Technical Fellow Review (2026-01-24)**:

**Goal**: Achieve A to A+ grade (currently B+ after first review fixes)

- **NEW-1** (HIGH, 3h): Fix TOCTOU race in `_extract_cache()` availability check
  - **Location**: `src/semantic/application/batch_engine.py:450`
  - **Problem**: Checks `available_blocks()` then allocates in loop, creating race:
    ```python
    # Line 450: Check (with lock)
    if self._pool.available_blocks() < total_blocks_needed:
        raise PoolExhaustedError(...)

    # Line 469: Allocate (with lock, but separate call)
    for layer_id, (k, v) in enumerate(cache):
        allocated_blocks = self._pool.allocate(n_blocks, layer_id, agent_id)
        # Another thread could steal blocks between check and this allocation!
    ```
  - **Risk**: Check passes but allocation fails due to race condition
  - **Fix**: Remove availability check and rely on allocation try/except
    ```python
    # Don't check availability - just try to allocate and handle error
    try:
        for layer_id, (k, v) in enumerate(cache):
            allocated_blocks = self._pool.allocate(n_blocks, layer_id, agent_id)
            # ... populate blocks ...
    except PoolExhaustedError:
        # Rollback already allocated blocks (already implemented in HIGH-4)
        for allocated_layer_id, allocated_layer_blocks in blocks_dict.items():
            self._pool.free(allocated_layer_blocks, agent_id)
        raise
    ```
  - **Why First Review Missed**: Focused on Sprint 2.5 fixes, didn't re-examine _extract_cache() logic flow
  - **Impact on Grade**: B+ → A- (removes last TOCTOU race)

- **NEW-2** (MEDIUM, 2h): Add locks to `used_memory()` and `available_memory()`
  - **Location**: `src/semantic/domain/services.py:282, 291`
  - **Problem**: Same issue as `available_blocks()` - read shared state without lock:
    ```python
    def used_memory(self) -> int:
        return len(self.allocated_blocks) * bytes_per_block  # NO LOCK!

    def available_memory(self) -> int:
        return len(self.free_list) * bytes_per_block  # NO LOCK!
    ```
  - **Risk**: Inconsistent memory accounting under concurrent access
  - **Fix**: Wrap with `with self._lock:` like `available_blocks()`
    ```python
    def used_memory(self) -> int:
        with self._lock:
            bytes_per_block = self.spec.bytes_per_block_per_layer()
            return len(self.allocated_blocks) * bytes_per_block

    def available_memory(self) -> int:
        with self._lock:
            bytes_per_block = self.spec.bytes_per_block_per_layer()
            return len(self.free_list) * bytes_per_block
    ```
  - **Why First Review Missed**: Fixed `available_blocks()` but didn't check other read methods
  - **Impact on Grade**: Completes thread-safety (concurrency score 85 → 95)

- **NEW-3** (MEDIUM, 4h): Add timeout mechanisms to prevent indefinite blocking
  - **Location**: `src/semantic/application/batch_engine.py:191` (insert), line 241 (step)
  - **Problem**: No timeouts anywhere - operations can block forever:
    ```python
    uids = self._batch_gen.insert(...)  # No timeout!
    response = self._batch_gen.next()   # No timeout!
    ```
  - **Risk**: If MLX hangs, entire engine blocks indefinitely
  - **Fix**: Add timeout wrapper using signals or asyncio
    ```python
    import signal
    from contextlib import contextmanager

    @contextmanager
    def timeout(seconds: int):
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Operation timed out after {seconds}s")

        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(seconds)
        try:
            yield
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)

    # Usage:
    try:
        with timeout(30):  # 30s timeout
            uids = self._batch_gen.insert(...)
    except TimeoutError:
        raise InvalidRequestError("Model insert timed out")
    ```
  - **Configuration**: Default 30s, configurable via settings
  - **Why First Review Missed**: Focused on correctness, not reliability features
  - **Impact on Grade**: Improves reliability score (prevents hangs)

- **NEW-4** (MEDIUM, 6h): Implement configuration management (Settings classes)
  - **Location**: `src/semantic/adapters/config/settings.py` (doesn't exist yet)
  - **Problem**: All values hardcoded, no config file support:
    - `max_tokens=256` in multiple places
    - No way to change batch size, timeouts, pool size without code changes
    - Empty `adapters/config/` directory
  - **Risk**: Can't tune for different hardware or use cases
  - **Fix**: Create Pydantic Settings classes (per production plan):
    ```python
    # src/semantic/adapters/config/settings.py
    from pydantic import Field
    from pydantic_settings import BaseSettings, SettingsConfigDict

    class MLXSettings(BaseSettings):
        model_id: str = "mlx-community/gemma-3-12b-it-4bit"
        max_batch_size: int = Field(5, ge=1, le=20)
        prefill_step_size: int = 512
        kv_bits: int | None = None
        block_tokens: int = 256
        cache_budget_mb: int = 4096

    class AgentSettings(BaseSettings):
        max_agents_in_memory: int = Field(5, ge=1)
        cache_dir: str = "~/.semantic/caches"
        batch_window_ms: int = Field(10, ge=1, le=1000)

    class OperationSettings(BaseSettings):
        insert_timeout_s: int = Field(30, ge=1, le=300)
        generate_timeout_s: int = Field(60, ge=1, le=600)
        default_max_tokens: int = Field(256, ge=1, le=8192)

    class Settings(BaseSettings):
        model_config = SettingsConfigDict(
            env_prefix="SEMANTIC_",
            env_file=".env",
            env_file_encoding="utf-8"
        )

        mlx: MLXSettings = Field(default_factory=MLXSettings)
        agent: AgentSettings = Field(default_factory=AgentSettings)
        operation: OperationSettings = Field(default_factory=OperationSettings)
    ```
  - **Precedence**: ENV vars > `.env` > `config/{environment}.toml` > defaults
  - **Why First Review Missed**: Focused on current code, not missing infrastructure
  - **Impact on Grade**: Enables production deployment (no hardcoded values)

- **NEW-5** (LOW, 3h): Replace ValueError/RuntimeError with domain errors
  - **Location**: Throughout codebase (21 occurrences of ValueError, 1 RuntimeError)
  - **Problem**: Using generic exceptions instead of domain errors:
    ```python
    # Current:
    raise ValueError(f"UID {uid} not found in active requests")

    # Should be:
    raise AgentNotFoundError(f"UID {uid} not found in active requests")
    ```
  - **Risk**: Violates hexagonal architecture (leaks implementation details)
  - **Fix**: Map generic exceptions to domain errors:
    - `ValueError` for validation → `InvalidRequestError`
    - `RuntimeError` for state issues → appropriate domain error
    - `KeyError` for missing items → `AgentNotFoundError`
  - **Example**:
    ```python
    # Before:
    if uid not in self._active_requests:
        raise ValueError(f"UID {uid} not found")

    # After:
    if uid not in self._active_requests:
        raise AgentNotFoundError(
            f"UID {uid} not found in active requests. "
            f"Active UIDs: {list(self._active_requests.keys())}"
        )
    ```
  - **Why First Review Missed**: Architecture layer was correct, implementation details not scrutinized
  - **Impact on Grade**: Improves architecture adherence (85 → 90)

- **NEW-6** (MEDIUM, 5h): Implement structured logging throughout
  - **Location**: Entire codebase (only 1 logging statement exists)
  - **Problem**: Minimal logging - only one `logging.error()` call:
    ```python
    # Line 261 in batch_engine.py:
    import logging  # noqa: PLC0415
    logging.error(f"Untracked UID {uid} in batch - possible memory leak.")
    ```
  - **Risk**: Can't debug production issues, no observability
  - **Fix**: Add structured logging using structlog:
    ```python
    # src/semantic/adapters/outbound/logger.py
    import structlog

    def configure_logging():
        structlog.configure(
            processors=[
                structlog.stdlib.add_log_level,
                structlog.stdlib.add_logger_name,
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.JSONRenderer()
            ],
            wrapper_class=structlog.stdlib.BoundLogger,
            logger_factory=structlog.stdlib.LoggerFactory(),
        )

    logger = structlog.get_logger()

    # Usage throughout codebase:
    logger.info("block_allocated",
                agent_id=agent_id,
                layer_id=layer_id,
                n_blocks=n_blocks,
                pool_available=self._pool.available_blocks())

    logger.warning("pool_pressure_high",
                   utilization=utilization_ratio,
                   available_blocks=available,
                   threshold=0.9)

    logger.error("allocation_failed",
                 agent_id=agent_id,
                 requested_blocks=n_blocks,
                 available_blocks=available,
                 exc_info=True)
    ```
  - **Key Logging Points**:
    - Block allocation/free operations
    - Pool pressure warnings (>80% utilization)
    - Cache eviction events
    - Model swap lifecycle
    - Request completion metrics
  - **Why First Review Missed**: Focused on correctness, observability deferred to Sprint 7
  - **Impact on Grade**: Essential for production debugging (reliability score +10)

**Additional Issues from Technical Fellow Review (2026-01-24)**:
- **EXP-007a** (CRITICAL): Measure safetensors I/O latency with async I/O threshold determination
  - Test realistic cache sizes: 4K, 8K, 12K, 16K token caches
  - Determine async I/O threshold (expected ~100MB based on 8K = ~400MB)
  - Validate tmp + rename atomic write doesn't add overhead
- **EXP-latency** (HIGH): Measure actual batch window timing with MLX operations
  - Validate 10ms batch window assumption (currently unvalidated)
  - Measure prefill + decode latency at 1, 3, 5, 7 agents
  - Determine realistic batch window target (10ms may be optimistic)
- **Move EXP-009/010 to Sprint 3 End** (BLOCKING): SSE format compliance tests
  - Run EXP-009 (Anthropic SSE format) before Sprint 4 starts
  - Run EXP-010 (Claude Code CLI) before Sprint 4 starts
  - These validate critical assumptions for Sprint 4

**Second Technical Fellow Review - Grade Improvement Path**:

Current Grade: **B+** (after first review fixes)
Target Grade: **A to A+** (portfolio-quality)

| Issue | Priority | Effort | Grade Impact |
|-------|----------|--------|--------------|
| NEW-1 (TOCTOU race) | HIGH | 3h | B+ → A- (removes last race) |
| NEW-2 (Memory locks) | MEDIUM | 2h | Concurrency 85→95 |
| NEW-3 (Timeouts) | MEDIUM | 4h | Reliability +10 |
| NEW-4 (Config mgmt) | MEDIUM | 6h | Production-ready |
| NEW-5 (Domain errors) | LOW | 3h | Architecture 85→90 |
| NEW-6 (Logging) | MEDIUM | 5h | Observability +10 |
| **Total** | — | **23h** | **B+ → A/A+** |

**Sprint 3 Integration Strategy**:
- **Week 1**: Fix NEW-1 (critical race), NEW-2 (complete thread-safety), NEW-4 (config)
- **Week 2**: Implement AgentCacheStore, NEW-3 (timeouts), NEW-6 (logging)
- **Deferred**: NEW-5 (domain errors) to Sprint 4 cleanup

**Experiments**:
- EXP-007: Verify RotatingKVCache serialization works via safetensors
- EXP-007a: Safetensors I/O latency + async threshold (NEW - Technical Fellow)
- EXP-008: Disk I/O time for 2K, 4K, 8K, 16K token caches (expect 60-80ms for 8K → async required)
- EXP-latency: Batch window timing validation (NEW - Technical Fellow)
- EXP-009: SSE format compliance (MOVED from Sprint 4 - blocking gate)
- EXP-010: Claude Code CLI compatibility (MOVED from Sprint 4 - blocking gate)

**Exit Gate**: Roundtrip byte-identical; model-tag validated; async I/O for caches > 100MB; deferred issues resolved; **EXP-007a, EXP-latency, EXP-009, EXP-010 passing**; **NEW-1, NEW-2, NEW-3, NEW-4, NEW-6 implemented (Grade A)**

---

### **ALPHA GATE (after Sprint 3)**

Go/no-go decision before investing in API layer:
- [ ] BlockPool allocator working correctly
- [ ] BatchEngine generates correct text with blocks
- [ ] Cache persistence and reload working
- [ ] Performance within 20% of POC for single-agent
- [ ] Performance improvement measurable for multi-agent (≥2x for 3+ agents)
- [ ] Memory usage within budget (model + pool < 14GB)

If ANY gate fails: evaluate scope reduction, simpler eviction, or Plan B.

---

### Sprint 4: Multi-Protocol API Adapter (2 weeks)

**Deliverable**: Claude Code CLI connects and gets persistent caching; OpenAI-compatible works with session_id.

| Task | Expert |
|------|--------|
| AgentIdentifier (dual strategy: content-based + explicit) | SE |
| Token prefix hashing (block-aligned, SHA-256) | SE, ML |
| AnthropicAdapter (system array, tools, messages, thinking blocks) | SE |
| Canonical serialization order (tools → system → messages) | SE |
| Full SSE streaming (message_start through message_stop) | SE |
| Tool use streaming (input_json_delta) | SE |
| OpenAIAdapter (session_id extension, OpenAI SSE format) | SE |
| DirectAgentHandler (CRUD, stateful generate) | SE |
| /v1/messages/count_tokens endpoint | SE |
| **Schemathesis API contract tests** | QE |
| **Golden-file SSE format tests (recorded from real Anthropic API)** | QE |
| Integration test: Claude Code CLI 3-turn conversation | QE |
| Integration test: OpenAI client with session_id | QE |
| **Failure-mode tests: malformed requests, invalid model field** | QE |
| ADR-006: Multi-Protocol Agent Identification | SE |
| **DEFERRED from Sprint 2.5: MLXCacheAdapter to wrap mx operations (Issue #1)** | SE, ML |
| **DEFERRED from Sprint 2.5: Add comprehensive error specs to ports (Issue #11)** | SE |
| **DEFERRED from Sprint 3: Replace ValueError/RuntimeError with domain errors (NEW-5)** | SE |

**Deferred Issues from Sprint 2.5 Critical Review**:
- **Issue #1** (MEDIUM, 3h): Create MLXCacheAdapter outbound adapter
  - Move `import mlx.core as mx` from batch_engine to dedicated adapter
  - Fixes architecture violation (application importing infrastructure)
  - Wrap mx.concatenate and cache operations behind port interface
  - Integrate when building outbound adapters in Sprint 4
- **Issue #11** (MEDIUM, 2h): Add comprehensive error specifications to all port interfaces
  - Document all possible exceptions for each port method
  - Include: InvalidRequestError, PoolExhaustedError, CacheCorruptionError, etc.
  - Improves API contract clarity for implementers
- **NEW-5** (LOW, 3h): Replace ValueError/RuntimeError with domain errors (Second Technical Fellow Review)
  - **Location**: Throughout codebase (21 occurrences of ValueError, 1 RuntimeError)
  - **Problem**: Using generic exceptions instead of domain errors violates hexagonal architecture
  - **Mapping**: ValueError → InvalidRequestError, RuntimeError → domain-specific, KeyError → AgentNotFoundError
  - **Benefit**: Improves architecture adherence score (85 → 90)
  - Integrate during API adapter error handling implementation

**Additional Issues from Technical Fellow Review (2026-01-24)**:
- **MED-4** (MEDIUM, 2h): Add timeout on BatchGenerator.insert()
  - Prevent indefinite blocking if mlx_lm hangs
  - Add timeout wrapper or async with cancellation
  - Default timeout: 30s for insert, configurable via settings
- **Hash Collision Strategy** (HIGH, 3h): Document and implement collision handling for prefix matching
  - SHA-256 prefix hashing has 1 in 2^96 collision probability
  - Add collision detection and fallback strategy
  - Document in ADR-006
- **Rate Limiting** (MEDIUM, 4h): Implement request rate limiting
  - Mentioned in risk register but not in sprint tasks
  - Add per-agent and global rate limits
  - Return 429 when limits exceeded
- **Request Authentication** (HIGH, 3h): Add ANTHROPIC_API_KEY validation
  - Validate API keys for production security
  - Support both environment variable and header-based auth
  - Return 401 for invalid/missing keys

**Experiments**:
- EXP-009: Record real Anthropic API SSE responses, replay-test against our format (MOVED to Sprint 3 end)
- EXP-010: Claude Code CLI end-to-end with local server (MOVED to Sprint 3 end)

**Exit Gate**: SSE format matches Anthropic spec exactly; session_id persistence works; Schemathesis passes; deferred issues resolved; **authentication working; rate limiting functional**

---

### Sprint 5: Model Hot-Swap (2 weeks)

**Deliverable**: Server dynamically switches models; agent caches preserved on disk across swaps.

| Task | Expert |
|------|--------|
| ModelRegistry (load, unload, current model tracking) | ML |
| Hot-swap protocol: drain → del model → gc.collect() → mx.clear_cache() → load → reconfigure | ML, HW |
| Model alias resolution (short name → HF ID) | SE |
| Cache invalidation on model change (model-tagged) | SE |
| BlockPool.reconfigure() for new ModelCacheSpec | HW |
| Agent-to-model-cache mapping (identity independent of model) | SE |
| Error recovery if swap fails (restore previous model) | ML |
| **Failure-mode test: swap during active batch** | QE |
| Integration test: swap Gemma 3 → Llama 8B, caches preserved on disk | QE |
| Integration test: swap back, Gemma 3 caches reload | QE |
| ADR-007: One Model At A Time (24GB Constraint) | HW |

**Technical Fellow Review Note (2026-01-24)**:
- **EXP-011 is CRITICAL and BLOCKING**: This experiment validates the core assumption of model hot-swap
- **Recommendation**: Run EXP-011 in Sprint 3 or earlier (not Sprint 5)
- **Risk**: If EXP-011 fails (memory not reclaimed), entire hot-swap architecture needs redesign
- **Fallback**: Process isolation (subprocess) or process restart (defeats hot-swap benefit)

**Experiments**:
- **EXP-011** (CRITICAL - SHOULD RUN EARLY): Does `del model` + `gc.collect()` + `mx.clear_cache()` actually reclaim memory? Monitor `mx.get_active_memory()` before/after.
- EXP-012: Measure total swap latency (drain + unload + load + reconfigure). Target < 30s.

**Exit Gate**: Swap < 30s; memory within 5% of cold-start; no orphaned blocks; **EXP-011 passing**

---

### Sprint 6: Integration, E2E, Benchmarks (2 weeks)

**Deliverable**: Full system tested end-to-end with documented performance.

| Task | Expert |
|------|--------|
| E2E: 5 concurrent Claude Code sessions on Gemma 3 | QE |
| E2E: Cache persistence across server restarts | QE |
| E2E: Model swap mid-session | QE |
| Stress test: pool exhaustion (graceful 429) | QE |
| Stress test: 10 agents, 50 rapid requests | QE |
| Benchmark: sequential vs batched (1, 3, 5 agents) per model | ML |
| Benchmark: cache resume speed (2K, 4K, 8K tokens) | ML |
| Benchmark: memory utilization vs padding approach | HW |
| Fix all integration issues | SE, ML |
| Create benchmark report | DE |
| **Load test: realistic conversation patterns (Technical Fellow)** | QE |
| **Memory profiling: sustained load 1-hour methodology (Technical Fellow)** | QE, HW |
| **Graceful degradation: 80%, 90%, 100% pool utilization (Technical Fellow)** | QE |

**Additional Testing from Technical Fellow Review (2026-01-24)**:
- **Load Testing with Realistic Patterns**: Not just rapid requests, but realistic multi-turn conversations
  - Claude Code CLI typical usage: 5-10 turn conversations
  - Variable request timing (not uniform)
  - Mix of short and long responses
- **Memory Profiling Methodology**: Document approach for 1-hour stress test
  - Measure memory at 1min, 5min, 15min, 30min, 60min intervals
  - Detect memory growth (leak detection)
  - Profile peak memory vs steady-state
- **Graceful Degradation Testing**: Validate behavior under high utilization
  - 80% pool utilization: Should still accept requests
  - 90% pool utilization: May queue or throttle
  - 100% pool utilization: Return 429, maintain stability

**Exit Gate**: All E2E pass on 2+ models; no leaks in 1-hour stress; benchmark documented; **graceful degradation validated**

---

### Sprint 7: Observability + Hardening (2 weeks)

**Deliverable**: Production-grade health, metrics, logging, graceful shutdown.

| Task | Expert |
|------|--------|
| **Health endpoints** (3-tier: /health/live, /health/ready, /health/startup) | SysE |
| **Prometheus metrics** (15+ metrics across request/inference/resource/agent) | SysE |
| **Structured JSON logging** (structlog with processor pipelines) | SysE |
| **Graceful shutdown** (multi-phase drain with configurable timeout) | SysE |
| **Runtime-configurable batch window** (via admin API endpoint) | SysE, ML |
| Request middleware (active request tracking, 503 when shutting down) | SE |
| CLI entrypoint (`python -m semantic serve`) with --host, --port, --allow-remote | SE |
| pip-installable package (Hatchling) | OSS |
| CHANGELOG, LICENSE, NOTICE, release notes | OSS, PM |
| License compliance check (liccheck) | OSS |
| SBOM generation (syft + CycloneDX 1.6) | OSS |
| **OpenTelemetry tracing (Technical Fellow)** | SysE |
| **Alerting thresholds for metrics (Technical Fellow)** | SysE |
| **Log retention and rotation policy (Technical Fellow)** | SysE |

**Additional Observability from Technical Fellow Review (2026-01-24)**:
- **OpenTelemetry Tracing**: Mentioned in plan line 45 but not in Sprint 7 tasks
  - Add distributed tracing for request flows
  - Trace spans for: request → batch → inference → response
  - Export to Jaeger or Tempo for visualization
- **Alerting Thresholds**: Define alert rules for Prometheus metrics
  - Pool utilization > 90% for 5 minutes
  - Cache eviction rate > X/minute
  - Model swap failures
  - 5xx error rate > threshold
- **Log Management**: Production log retention and rotation
  - Rotation policy: daily or 100MB, whichever first
  - Retention: 7 days local, 30 days remote
  - Compression for archived logs

**Prometheus Metrics Catalog:**

| Metric | Type | Labels |
|--------|------|--------|
| `semantic_request_total` | Counter | model, agent_id, status |
| `semantic_request_duration_seconds` | Histogram | — |
| `semantic_time_to_first_token_seconds` | Histogram | — |
| `semantic_request_queue_depth` | Gauge | — |
| `semantic_batch_size` | Histogram | — |
| `semantic_tokens_generated_total` | Counter | — |
| `semantic_tokens_per_second` | Gauge | — |
| `semantic_memory_used_bytes` | Gauge | — |
| `semantic_cache_hit_total` | Counter | — |
| `semantic_cache_miss_total` | Counter | — |
| `semantic_pool_utilization_ratio` | Gauge | — |
| `semantic_eviction_total` | Counter | tier |
| `semantic_agents_active` | Gauge | — |
| `semantic_model_swap_duration_seconds` | Histogram | — |
| `semantic_cache_persist_duration_seconds` | Histogram | — |

**Exit Gate**: Health endpoints respond correctly under all states; metrics exported; graceful shutdown within timeout

---

### Sprint 8: Documentation, Packaging, Deployment (2 weeks)

**Deliverable**: Published v0.1.0 with comprehensive docs.

| Task | Expert |
|------|--------|
| **README.md** (project overview, badges, quick links, architecture summary) | DE |
| **Quick Start guide** (`docs/quick-start.md`) — 5 minutes to first request | DE |
| **Installation guide** (`docs/installation.md`) — prerequisites, platforms, pip, dev setup | DE, SysE |
| **User guide** (`docs/user-guide.md`) — configuration, multi-agent usage, CLI, troubleshooting | DE |
| **Developer guide** (`docs/developer-guide.md`) — contributing, testing, code style, PR process | DE, QE |
| **Architecture guide** (`docs/architecture.md`) — hexagonal layers, Mermaid diagrams, dependency rules | DE, SE |
| **Model onboarding guide** (`docs/model-onboarding.md`) — how to add a new model architecture | DE, ML |
| **API reference** (`docs/api-reference.md`) — mkdocs + autodoc from docstrings | DE |
| **Deployment guide** (`docs/deployment.md`) — standalone, launchd plist, monitoring | DE, SysE |
| **Mermaid diagrams** — architecture, data flow, sequence, state machine, class diagrams | DE, SE |
| `make docs` — build and serve docs locally (mkdocs-material) | DE |

**Exit Gate**: `pip install .` works; CLI starts server; docs build with no broken links; all guides reviewed; Mermaid renders correctly; MIT confirmed

---

## Plan B: Sequential Engine with Shared Cache

**Trigger**: EXP-003 or EXP-004 fails in Sprint 0.

If continuous batching with cache injection/extraction proves infeasible:

1. **Retain** Sprint 1 (BlockPool) and Sprint 3 (AgentCacheStore) — these provide value for prefix sharing and memory management
2. **Replace** Sprint 2 with a sequential inference engine:
   - Agents take turns using the model (FIFO queue)
   - Each agent's KV cache is loaded from disk before inference, saved after
   - No batching, but cache persistence still provides the 47x resume speedup
3. **Sprint 4** (API) remains mostly unchanged (agents still identified, caches still managed)
4. **Trade-off**: 1x throughput (no batching benefit) but simpler, proven architecture

**Decision point**: End of Sprint 0, Day 5. If experiments fail, invoke Plan B immediately.

---

## Multi-Expert Debate Format

### Workshop Structure (Per Sprint)

| Sprint | Duration | Rationale |
|--------|----------|-----------|
| S0-S2 | 2 hours | Architectural foundation, critical experiments |
| S3-S5 | 1.5 hours | Implementation details, fewer decisions |
| S6-S8 | 1 hour | Integration, docs, polish |

1. **Sprint Review** (15 min): PM presents previous sprint outcomes
2. **Expert Concerns** (40 min, 5 min each):
   - SE: Architecture, coupling, complexity
   - ML: Performance, MLX compatibility, model behavior
   - QE: Coverage gaps, test quality, CI stability
   - HW: Memory, bandwidth, hardware constraints
   - OSS: License risks, packaging issues
   - DE: Documentation gaps, API clarity
   - SysE: Deployment, monitoring, reliability
   - PM: Schedule risks, scope creep
3. **Cross-Cutting Conflicts** (20 min): e.g., ML performance vs QE code quality
4. **Decision Round** (30 min): Vote on ADRs, assign experiments
5. **Sprint Commitment** (15 min): Final work items

### Resolution Protocol

| Domain | Final Authority | Consulted |
|--------|----------------|-----------|
| Architecture | SE | ML, QE |
| ML Performance | ML | HW, SE |
| Memory Management | HW | ML, SE |
| Code Quality | QE | SE |
| API Design | SE | DE, ML |
| Schedule/Scope | PM | All |
| License/Legal | OSS | PM |

**Conflict Resolution**: Data first → Time-boxed experiment (2 days max) → Domain authority decides → PM can override for scope

### Experts Bring In Others

- ML can invoke HW for memory bandwidth analysis
- SE can invoke ML for MLX API feasibility
- QE can invoke SysE for CI infrastructure needs
- Any expert can request a web search or experiment to validate claims

### Simplicity Checkpoints (Every Sprint)

At each debate, the team asks:
- "Does this complexity earn its keep for the target use case (≤10 agents on M4 Pro)?"
- "Would a simpler approach work for v0.1 with room to grow later?"
- Track CC per module as a project health indicator

---

## Feasibility Experiments (Complete Registry)

| ID | Question | Method | Success Criteria | Sprint | Blocker? |
|----|----------|--------|-----------------|--------|----------|
| EXP-001 | Are model.args consistent across mlx-community variants? | Load 4 models, inspect attrs | All have num_hidden_layers, num_key_value_heads, head_dim (note: n_kv_heads=8 for Gemma 3) | S1 | No |
| EXP-002 | Block allocation overhead | Allocate/free 1000 blocks, measure time | < 1ms per allocation | S1 | No |
| **EXP-003** | **Can cache be injected into BatchGenerator via `caches` param?** | Insert with `caches=[loaded_cache]`, compare output to fresh generation | **✅ VALIDATED Sprint 3.5**: Cache injection works | **S3.5** | **YES** |
| **EXP-004** | **Can per-sequence cache be extracted via Response.prompt_cache?** | Run 3-sequence batch, extract cache on each completion, save/reload/re-inject | **✅ VALIDATED Sprint 3.5**: Cache extraction works (attribute, not callable) | **S3.5** | **YES** |
| EXP-005 | Decode throughput at various batch sizes | Benchmark 1, 3, 5, 7, 10 agents | Results within 80% of theoretical | S2 | No |
| EXP-006 | Block gather (mx.concatenate) overhead for 8K context | Concatenate 32 blocks per layer, 48 layers, measure time | < 5ms total | S2 | No |
| EXP-007 | RotatingKVCache serialization via safetensors | save_prompt_cache with RotatingKVCache, load, verify state | Round-trip preserves cache state | S3 | No |
| EXP-008 | Disk I/O time for large caches | Time save/load for 2K, 4K, 8K, 16K token caches | Determine async I/O threshold | S3 | No |
| EXP-009 | Anthropic SSE format compliance | Record real API responses, diff against our output | Byte-for-byte match on event names/structure | S4 | No |
| EXP-010 | Claude Code CLI compatibility | `ANTHROPIC_BASE_URL=http://localhost:8000 claude` | Multi-turn conversation works | S4 | No |
| EXP-011 | Model weight memory reclamation | `del model; gc.collect(); mx.clear_cache()`, measure `mx.get_active_memory()` | Memory drops to < 1GB | S5 | No |
| EXP-012 | Total hot-swap latency | Time full drain → unload → load → reconfigure cycle | < 30s total | S5 | No |

---

## Risk Register

| Risk | L | I | Sprint | Mitigation |
|------|---|---|--------|------------|
| **Over-engineering for ≤10 agents** | **H** | **H** | All | Simplicity checkpoints; start simple (Dict not trie, LRU not 3-tier); add complexity only when profiling shows need |
| mlx_lm `caches` param doesn't work as expected | M | H | S0 | EXP-003 in Sprint 0; Plan B ready |
| Block gather slower than padded | M | H | S2 | Benchmark early; gather once at restore, not per-step |
| MLX lazy eval memory spikes | L | M | S2 | Force mx.eval() after allocation |
| SSE format mismatch breaks Claude Code | M | H | S4 | Record/replay real API responses (EXP-009) |
| mlx-lm API breaking change | M | H | All | Pin to v0.30.4; typed adapter wrapper |
| Pool exhaustion under load | M | M | S6 | Graceful 429; LRU eviction |
| Model swap doesn't reclaim memory | M | H | S5 | EXP-011 validates; fallback: process restart |
| Cache corruption on crash | L | M | S3 | Atomic write (tmp + rename); checksum validation |
| Memory fragmentation in long-running server | M | M | S6 | Pre-allocate pool at startup; periodic mx.clear_cache() |
| Disk I/O blocks batch worker | M | M | S3 | Async I/O for caches > 100MB |

---

## Assumptions Register

| ID | Assumption | Validation | Sprint |
|----|-----------|------------|--------|
| A-01 | mlx_lm v0.30.4 `insert(caches=...)` works for cache injection | EXP-003 | S0 |
| A-02 | Response.prompt_cache() returns usable per-sequence cache | EXP-004 | S0 |
| A-03 | model.args contains required attributes for all target models | EXP-001 | S1 |
| A-04 | sliding_window_pattern=6 gives 8 global + 40 SW for Gemma 3 | Config inspection | S1 |
| A-05 | n_kv_heads=8 for Gemma 3 12B (corrected from 4) | Config inspection | S1 |
| A-06 | Block allocation overhead < 1ms | EXP-002 | S1 |
| A-07 | mx.concatenate on blocks < 5ms for 8K context | EXP-006 | S2 |
| A-08 | save_prompt_cache works with RotatingKVCache | EXP-007 | S3 |
| A-09 | Claude Code CLI respects ANTHROPIC_BASE_URL | EXP-010 | S4 |
| A-10 | M4 Pro 24GB holds model (6GB) + 4GB pool + MLX overhead | Memory monitoring | S2 |
| A-11 | del model + gc.collect() + mx.clear_cache() frees GPU memory | EXP-011 | S5 |
| A-12 | Individual sequences complete independently in BatchGenerator | EXP-004 | S0 |

---

## `project/` Directory Artifacts

```
project/
├── sprints/
│   ├── sprint_template.md
│   └── sprint_{0-8}_{name}.md        # Sprint definitions + outcomes
├── architecture/
│   ├── adr_template.md
│   └── ADR-{001-007}-{topic}.md       # Architecture Decision Records
├── experiments/
│   └── EXP-{001-012}-{topic}.md       # Feasibility experiment results
├── quality/
│   ├── gates.yaml                     # Quality gate thresholds
│   └── coverage_report.md
├── risks/
│   └── risk_register.md
├── decisions/
│   └── decision_log.md                # Chronological decisions
├── benchmarks/
│   ├── baseline.json                  # POC baseline performance
│   └── comparison_report.md
└── templates/
    ├── sprint_template.md
    ├── adr_template.md
    └── experiment_template.md
```

---

## Definition of Done

**Feature**: Unit tests + integration test + type annotations + docstrings + no new warnings + code review
**Refactoring**: All existing tests pass + coverage unchanged + no new complexity
**Bug Fix**: Regression test added + root cause documented
**Infrastructure**: CI passes + deployment tested + docs updated

---

## Code Patterns (Enforced)

| Pattern | Anti-Pattern | Enforced By |
|---------|-------------|-------------|
| `Protocol` classes for interfaces | ABC with inheritance | mypy + code review |
| Domain exception hierarchy | Raw exceptions or Result monads | Convention + tests |
| Pydantic Settings for config | env.json + global singleton | ruff T201 (no print) |
| Constructor DI (app factory) | Global mutable state | No globals in domain |
| `@dataclass(frozen=True)` for value objects | Dicts | mypy strict |
| No `Any` in public interfaces | Untyped parameters | mypy disallow_any_generics |
| Typed MLX adapter layer (cast internally) | Untyped mlx_lm calls in domain | Adapter boundary |
| Max CC < 10 per function (CC < 7 for domain) | Deeply nested logic | ruff C90 in CI |
| Async I/O for disk operations > 100MB | Sync I/O blocking event loop | Code review |

---

## Key Files to Implement (from design docs)

- `src/semantic/domain/value_objects.py` — ModelCacheSpec (from `backend_plan.md` §2.2)
- `src/semantic/domain/services.py` — BlockPool (from `backend_plan.md` §4)
- `src/semantic/domain/errors.py` — Domain exception hierarchy
- `src/semantic/ports/inbound.py` — InferencePort, AgentManagementPort (Protocol classes)
- `src/semantic/ports/outbound.py` — ModelBackendPort, CachePersistencePort (Protocol classes)
- `src/semantic/application/batch_engine.py` — BlockPoolBatchEngine (from `backend_plan.md` §3)
- `src/semantic/application/agent_store.py` — AgentCacheStore with prefix matching (from `backend_plan.md` §3.2)
- `src/semantic/application/model_registry.py` — ModelRegistry (from `backend_plan.md` §3.2)
- `src/semantic/adapters/inbound/anthropic_api.py` — Full Anthropic adapter (from `anthropic_cli_adapter.md` §1-4)
- `src/semantic/adapters/inbound/openai_api.py` — OpenAI + session_id (from `anthropic_cli_adapter.md` §16)
- `src/semantic/adapters/outbound/mlx_backend.py` — Typed MLX wrapper (replaces current `mlx_utils.py` + `mlx_cache_extractor.py`)
- `src/semantic/adapters/config/settings.py` — All configuration (replaces `config.py` + hardcoded values)
- `src/semantic/entrypoints/server.py` — App factory, DI composition root

---

## Deployment (macOS Native — No Docker)

MLX requires Metal GPU access. Docker on macOS runs Linux VMs without GPU passthrough. Deployment is **macOS-native only**.

### Development
```bash
python -m semantic serve --host 127.0.0.1 --port 8000
```

### Production (launchd)
```xml
<!-- ~/Library/LaunchAgents/com.semantic.server.plist -->
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "...">
<plist version="1.0">
<dict>
    <key>Label</key><string>com.semantic.server</string>
    <key>ProgramArguments</key>
    <array>
        <string>/path/to/python</string>
        <string>-m</string>
        <string>semantic</string>
        <string>serve</string>
    </array>
    <key>RunAtLoad</key><true/>
    <key>KeepAlive</key><true/>
    <key>StandardOutPath</key><string>/tmp/semantic.log</string>
    <key>StandardErrorPath</key><string>/tmp/semantic.err</string>
</dict>
</plist>
```

### Management
```bash
launchctl load ~/Library/LaunchAgents/com.semantic.server.plist
launchctl unload ~/Library/LaunchAgents/com.semantic.server.plist
launchctl kickstart -k gui/$(id -u)/com.semantic.server  # restart
```

---

## Documentation Plan (Comprehensive)

All documentation lives in `docs/` and is built with **mkdocs-material**. Mermaid diagrams render natively.

### Document Inventory

| Document | Path | Audience | Sprint |
|----------|------|----------|--------|
| README | `README.md` (root) | Everyone | S8 |
| Quick Start | `docs/quick-start.md` | New users | S8 |
| Installation | `docs/installation.md` | Users + devs | S8 |
| User Guide | `docs/user-guide.md` | End users | S8 |
| Developer Guide | `docs/developer-guide.md` | Contributors | S8 |
| Architecture Guide | `docs/architecture.md` | Developers + architects | S8 |
| Model Onboarding | `docs/model-onboarding.md` | ML engineers | S8 |
| API Reference | `docs/api-reference.md` | API consumers | S8 |
| Deployment Guide | `docs/deployment.md` | Operators | S8 |

### Mermaid Diagram Inventory

| Diagram | Type | Location | Purpose |
|---------|------|----------|---------|
| Hexagonal Layers | `graph TB` | architecture.md | Layer dependencies |
| Request Flow | `sequenceDiagram` | architecture.md | End-to-end request lifecycle |
| Agent Lifecycle | `stateDiagram-v2` | architecture.md | Cache state transitions (cold→hot→warm) |
| Block Pool Classes | `classDiagram` | architecture.md | Domain model relationships |
| Hot-Swap Flow | `graph LR` | architecture.md | Model swap sequence |
| Model Decision Tree | `graph TD` | model-onboarding.md | Architecture classification |
| SSE Event Flow | `sequenceDiagram` | api-reference.md | Streaming protocol |
| Batch Window | `sequenceDiagram` | architecture.md | 10ms batching behavior |
| CI Pipeline | `graph LR` | developer-guide.md | Quality gate flow |
| Graceful Shutdown | `sequenceDiagram` | deployment.md | Multi-phase drain |

### Documentation Quality Gates

| Check | Tool | Threshold |
|-------|------|-----------|
| Broken links | mkdocs build --strict | 0 warnings |
| Mermaid syntax | mkdocs-material render | All diagrams render |
| Spelling | codespell | 0 errors |
| Code examples | pytest --doctest-modules | All examples valid |

---

## Verification Plan

1. `make lint` — ruff clean (lint + format + security + complexity)
2. `make typecheck` — mypy --strict clean
3. `make test-unit` — differentiated coverage targets met, all pass
4. `make test-integration` — differentiated coverage targets met, all pass (Apple Silicon)
5. `make test-smoke` — Server starts, health OK, basic request works
6. `make test-e2e` — Multi-agent, persistence, model swap all work
7. `make security` — semgrep clean, no high/critical findings
8. `make licenses` — No GPL/LGPL/AGPL in dependency tree
9. `make bench` — Performance within expected bounds, better than POC for multi-agent
10. `make docs` — mkdocs build --strict (no broken links, Mermaid renders, no warnings)
11. All 9 documentation files present and reviewed
12. `ANTHROPIC_BASE_URL=http://localhost:8000 claude` — Claude Code CLI works end-to-end
13. Graceful shutdown completes within timeout under load
14. Health endpoints report correct status under all conditions

---

## ADR Index

| ADR | Topic | Sprint | Status |
|-----|-------|--------|--------|
| ADR-001 | Hexagonal Architecture | S0 | Planned |
| ADR-002 | Block Size = 256 Tokens (Universal) | S0 | Planned |
| ADR-003 | Cache Eviction Strategy (Simple LRU First) | S1 | Planned |
| ADR-004 | Block Gather Strategy (One-Time at Restore) | S2 | Planned |
| ADR-005 | Composition Pivot (Why Custom Engine) | S2 | Planned |
| ADR-006 | Multi-Protocol Agent Identification | S4 | Planned |
| ADR-007 | One Model At A Time (24GB Constraint) | S5 | Planned |

---

## Tooling Stack Summary

| Category | Choice | Rationale |
|----------|--------|-----------|
| Build backend | Hatchling | Extensible, VCS versioning, plugins |
| Linter/formatter | Ruff | Replaces flake8+black+isort+bandit, 100x faster |
| Type checker | mypy --strict (CI) + Pyright (editor) | Mature plugin ecosystem + fast editor feedback |
| SAST | Ruff S rules + Semgrep | Python-specific + custom organizational rules |
| Complexity | Ruff C90 (max-complexity=10) | Integrated, fast |
| License scanning | liccheck | Policy in pyproject.toml, used by Apache Superset |
| SBOM | Syft → CycloneDX 1.6 JSON | Industry standard, OWASP-backed |
| Logging | structlog | Native structured logging, OpenTelemetry, processor pipelines |
| Metrics | prometheus-fastapi-instrumentator + prometheus_client | Standard observability |
| API testing | Schemathesis | Property-based fuzzing from OpenAPI spec |
| Property testing | Hypothesis | High-value for BlockPool invariants |
| Async testing | pytest-asyncio (auto mode) | Less boilerplate for pure-asyncio |
| Secret detection | gitleaks (pre-commit) | Catches API keys before commit |
| Versioning | importlib.metadata + hatch-vcs (git tags) | Single source of truth |

---

## Key Corrections from Expert Review

| Original Plan | Corrected | Reason |
|---------------|-----------|--------|
| `batch.extract_cache(uid)` method | `Response.prompt_cache()` callable on completion | API doesn't have extract_cache |
| `insert(prompt_cache=...)` | `insert(caches=[...])` parameter | Different parameter name |
| n_kv_heads = 4 | **n_kv_heads = 8** | Actual Gemma 3 12B config |
| Per-block: 1 MB/layer | **Per-block: 2 MB/layer** (float16) | Doubled due to kv_heads correction |
| systemd deployment | **launchd** (macOS native) | macOS doesn't have systemd |
| Docker support | **Removed** | MLX requires Metal, no GPU in Docker |
| 0.0.0.0 default bind | **127.0.0.1** default | Security: localhost-only by default |
| EXP-003/004 in Sprint 2 | **Sprint 0 blockers** | Architecture depends on these |
| Result[T] = Ok[T] \| Err | **Domain exception hierarchy** | More Pythonic |
| CC < 15 | **CC < 10** (domain: CC < 7) | Industry standard |
| Trie for prefix matching | **Dict + longest_common_prefix** (trie only if profiling shows need) | Over-engineering for ≤10 agents |
| 3-tier eviction (hot/warm/cold) | **Simple LRU first** (3-tier only if profiling shows need) | YAGNI |
| Over-engineering risk: M/M | **H/H** | Block pool + trie + 3-tier for 10 agents |
| No Plan B | **Plan B defined** | Sequential engine fallback |
| No Alpha gate | **Alpha gate after Sprint 3** | Go/no-go before API investment |
| Missing experiments | **All 12 defined** with criteria and fallbacks | Complete registry |
| Config: ENV > .env > TOML > defaults | **CLI > ENV > .env > TOML > defaults** | CLI args highest priority |
| mx.metal.clear_cache() frees weights | **del model + gc.collect() + mx.clear_cache()** | clear_cache only frees buffer pool |
