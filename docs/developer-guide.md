# Developer Guide

Contributing to Semantic: setup, code style, testing, and quality standards.

## Development Setup

### Prerequisites

- Mac with Apple Silicon (M1/M2/M3/M4)
- Python 3.11 or 3.12
- Git
- 16GB+ RAM

### Initial Setup

```bash
# Clone repository
git clone https://github.com/yshk-mxim/agent-memory.git
cd agent-memory

# Create virtual environment
python3.12 -m venv venv
source venv/bin/activate

# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Verify Setup

```bash
# Run all quality checks
make lint        # Ruff linting
make typecheck   # MyPy type checking
make test-unit   # Unit tests

# All should pass ✅
```

## Code Style

### Tools

| Tool | Purpose | Config |
|------|---------|--------|
| **ruff** | Linting + formatting + import sorting | `pyproject.toml` |
| **mypy** | Static type checking (strict mode) | `pyproject.toml` |
| **pre-commit** | Git hooks for automated checks | `.pre-commit-config.yaml` |

### Ruff Configuration

```bash
# Check code
make lint

# Auto-fix issues
ruff check --fix src tests

# Format code
ruff format src tests
```

**Settings** (from `pyproject.toml`):
- Line length: 88 (Black-compatible)
- Target: Python 3.11+
- Rules: All recommended + type annotations
- Import sorting: Automatic

### Type Annotations

**All code must be mypy --strict compliant**:

```python
# ✅ GOOD: Full type annotations
def allocate_blocks(
    pool: BlockPool,
    count: int,
    layer_id: int,
    agent_id: str
) -> list[KVBlock]:
    """Allocate blocks from pool.

    Args:
        pool: Block pool instance
        count: Number of blocks to allocate
        layer_id: Layer index (0-based)
        agent_id: Agent identifier

    Returns:
        List of allocated KVBlock instances

    Raises:
        ValueError: If count > available blocks
    """
    return pool.allocate(n_blocks=count, layer_id=layer_id, agent_id=agent_id)

# ❌ BAD: Missing types
def allocate_blocks(pool, count, layer_id, agent_id):
    return pool.allocate(n_blocks=count, layer_id=layer_id, agent_id=agent_id)
```

**No `Any` in public interfaces**:

```python
# ❌ BAD
def process(data: Any) -> Any: ...

# ✅ GOOD
def process(data: dict[str, int]) -> GenerationResult: ...
```

### Naming Conventions

| Type | Convention | Example |
|------|-----------|---------|
| **Modules** | snake_case | `block_pool.py` |
| **Classes** | PascalCase | `BlockPool` |
| **Functions** | snake_case | `allocate_blocks()` |
| **Variables** | snake_case | `total_blocks` |
| **Constants** | UPPER_SNAKE | `MAX_BATCH_SIZE` |
| **Private** | _leading_underscore | `_free_list` |

### Docstrings

**Google style**, required for all public APIs:

```python
def allocate(self, n_blocks: int, layer_id: int, agent_id: str) -> list[KVBlock]:
    """Allocate n blocks for an agent at a specific layer.

    Allocates blocks from the free list (LIFO) and tracks ownership.

    Args:
        n_blocks: Number of blocks to allocate (must be > 0)
        layer_id: Layer index (0 to n_layers-1)
        agent_id: Agent identifier (non-empty string)

    Returns:
        List of allocated KVBlock instances

    Raises:
        ValueError: If n_blocks <= 0 or > available blocks
        ValueError: If layer_id < 0 or >= n_layers
        ValueError: If agent_id is empty

    Example:
        >>> pool = BlockPool(spec, total_blocks=10)
        >>> blocks = pool.allocate(n_blocks=3, layer_id=0, agent_id="agent_1")
        >>> len(blocks)
        3
    """
```

## Testing

### Test Organization

```
tests/
├── unit/                  # Pure domain tests (no MLX, no disk)
│   ├── test_entities.py       # AgentBlocks, KVBlock
│   ├── test_value_objects.py  # ModelCacheSpec, CacheKey
│   └── test_services.py       # BlockPool (44 tests, property-based)
├── integration/           # Real MLX + disk (Apple Silicon only)
├── smoke/                 # Server lifecycle + basic requests
└── e2e/                   # Full stack multi-agent flows
```

### Running Tests

```bash
# All unit tests (fast, no hardware requirements)
make test-unit

# Integration tests (requires Apple Silicon)
make test-integration

# Smoke tests (requires server)
make test-smoke

# E2E tests (full stack, slow)
make test-e2e

# With coverage
pytest tests/unit/ --cov=src/agent_memory/domain --cov-report=html
```

### Writing Tests

#### Unit Tests (Domain Layer)

**Pattern**: Mock all ports, test domain logic only.

```python
import pytest
from agent_memory.domain.services import BlockPool
from agent_memory.domain.value_objects import ModelCacheSpec

@pytest.mark.unit
class TestBlockPool:
    """Unit tests for BlockPool service."""

    def test_allocate_returns_correct_number_of_blocks(self) -> None:
        """Should allocate exactly n_blocks."""
        spec = ModelCacheSpec(
            n_layers=4,
            n_kv_heads=4,
            head_dim=64,
            block_tokens=256,
            layer_types=["global"] * 4,
        )
        pool = BlockPool(spec=spec, total_blocks=10)

        blocks = pool.allocate(n_blocks=3, layer_id=0, agent_id="test")

        assert len(blocks) == 3
        assert all(b.layer_id == 0 for b in blocks)
```

#### Property-Based Tests (Hypothesis)

**Required for core invariants**:

```python
from hypothesis import given, assume
from hypothesis import strategies as st

@given(
    total_blocks=st.integers(min_value=10, max_value=100),
    n_blocks=st.integers(min_value=1, max_value=50),
)
def test_property_used_plus_available_equals_total(
    total_blocks: int,
    n_blocks: int,
) -> None:
    """Property: used_blocks + available_blocks = total_blocks (ALWAYS)."""
    assume(n_blocks <= total_blocks)

    spec = ModelCacheSpec(
        n_layers=4,
        n_kv_heads=4,
        head_dim=64,
        block_tokens=256,
        layer_types=["global"] * 4,
    )
    pool = BlockPool(spec=spec, total_blocks=total_blocks)

    # Invariant holds initially
    assert pool.allocated_block_count() + pool.available_blocks() == total_blocks

    # Allocate
    blocks = pool.allocate(n_blocks=n_blocks, layer_id=0, agent_id="test")

    # Invariant STILL holds
    assert pool.allocated_block_count() + pool.available_blocks() == total_blocks

    # Free
    pool.free(blocks, agent_id="test")

    # Invariant STILL holds
    assert pool.allocated_block_count() + pool.available_blocks() == total_blocks
```

**When to use Hypothesis**:
- Core invariants (memory accounting, ownership tracking)
- Edge cases (zero, max, boundary values)
- Fuzz testing (random inputs)

**Note**: Don't use pytest fixtures with `@given` — create data inline.

#### Integration Tests

**Pattern**: Use real MLX, real disk, skip on non-Apple Silicon.

```python
import pytest
import platform

@pytest.mark.integration
@pytest.mark.skipif(
    platform.machine() != "arm64",
    reason="MLX requires Apple Silicon"
)
def test_mlx_cache_roundtrip(tmp_path):
    """Should save and load cache using real MLX."""
    # Real MLX model loading
    # Real safetensors save/load
    # Verify roundtrip byte-identical
```

### Test Fixtures

**Location**: `tests/conftest.py`

```python
import pytest
from agent_memory.domain.value_objects import ModelCacheSpec

@pytest.fixture
def small_spec() -> ModelCacheSpec:
    """Small ModelCacheSpec for fast tests."""
    return ModelCacheSpec(
        n_layers=4,
        n_kv_heads=4,
        head_dim=64,
        block_tokens=256,
        layer_types=["global"] * 4,
    )

@pytest.fixture
def gemma3_spec() -> ModelCacheSpec:
    """Realistic Gemma 3 12B spec."""
    return ModelCacheSpec(
        n_layers=48,
        n_kv_heads=8,
        head_dim=240,
        block_tokens=256,
        layer_types=["global"] * 8 + ["sliding_window"] * 40,
        sliding_window_size=1024,
    )
```

## Quality Gates

All checks must pass before merging:

```bash
# 1. Linting (ruff)
make lint
# Target: 0 errors

# 2. Type checking (mypy)
make typecheck
# Target: 0 errors (strict mode)

# 3. Security scan (bandit)
make security
# Target: 0 high/critical findings

# 4. Complexity check (radon)
make complexity
# Target: All functions CC < 15

# 5. License check
make licenses
# Target: No GPL/LGPL/AGPL dependencies

# 6. Unit tests
make test-unit
# Target: >85% domain coverage, all tests pass

# 7. Integration tests (Apple Silicon only)
make test-integration
# Target: >70% coverage, all tests pass
```

### Coverage Requirements

| Layer | Target | Actual |
|-------|--------|--------|
| **Domain** | 95%+ | 95.07% ✅ |
| **Application** | 85%+ | (pending) |
| **Adapters** | 70%+ | (pending) |

## PR Process

### Branch Naming

```
feature/add-model-hot-swap
bugfix/block-pool-race-condition
docs/update-architecture-guide
refactor/extract-cache-store
```

### Commit Messages

```
feat: add model hot-swap capability

- Implement ModelRegistry.swap_model()
- Add drain logic for in-flight requests
- Add reconfigure() to BlockPool
- 15 new tests, all passing

Closes #42
```

**Format**: `<type>: <summary>` where type is:
- `feat` — New feature
- `fix` — Bug fix
- `docs` — Documentation
- `refactor` — Code restructure (no behavior change)
- `test` — Add/update tests
- `perf` — Performance improvement
- `chore` — Build/tooling changes

### Pre-PR Checklist

- [ ] All quality gates pass (`make lint && make typecheck && make test-unit`)
- [ ] New code has type annotations
- [ ] New code has docstrings (Google style)
- [ ] New code has unit tests (95%+ coverage)
- [ ] Integration tests added if touching adapters
- [ ] Property tests added if changing invariants
- [ ] Documentation updated if API changed
- [ ] Pre-commit hooks pass

### Review Checklist

Reviewers check:

- [ ] **Architecture**: Follows hexagonal pattern, domain isolated
- [ ] **Types**: mypy --strict passes, no `Any` in public APIs
- [ ] **Tests**: Adequate coverage, property tests for invariants
- [ ] **Docs**: Docstrings clear, architecture docs updated
- [ ] **Performance**: No obvious inefficiencies
- [ ] **Security**: No hardcoded secrets, input validated

## Contributing Workflow

### 1. Setup Development Environment

```bash
git clone https://github.com/yshk-mxim/agent-memory.git
cd agent-memory
pip install -e ".[dev]"
pre-commit install
```

### 2. Create Feature Branch

```bash
git checkout -b feature/my-new-feature
```

### 3. Develop with TDD

```bash
# 1. Write failing test
# 2. Implement feature
# 3. Run tests
make test-unit

# 4. Fix issues
make lint
make typecheck

# 5. Iterate
```

### 4. Commit Changes

```bash
git add .
git commit -m "feat: add my feature"
# Pre-commit hooks run automatically
```

### 5. Push and Create PR

```bash
git push origin feature/my-new-feature
# Create PR on GitHub
```

### 6. Address Review Feedback

```bash
git commit -m "fix: address review comments"
git push
```

## Code Patterns

### Prefer Protocols Over ABCs

```python
# ✅ GOOD: Protocol (structural typing)
from typing import Protocol

class ModelBackendPort(Protocol):
    def load_model(self, model_id: str) -> Any: ...

# ❌ BAD: ABC (inheritance coupling)
from abc import ABC, abstractmethod

class ModelBackendPort(ABC):
    @abstractmethod
    def load_model(self, model_id: str) -> Any: ...
```

### Frozen Dataclasses for Value Objects

```python
from dataclasses import dataclass

# ✅ GOOD: Immutable value object
@dataclass(frozen=True)
class CacheKey:
    agent_id: str
    model_id: str
    prefix_hash: str

# ❌ BAD: Mutable (can be changed after creation)
@dataclass
class CacheKey:
    agent_id: str
    model_id: str
    prefix_hash: str
```

### Constructor Dependency Injection

```python
# ✅ GOOD: Dependencies passed to constructor
class AgentCacheStore:
    def __init__(
        self,
        pool: BlockPool,
        persistence: CachePersistencePort,
    ) -> None:
        self._pool = pool
        self._persistence = persistence

# ❌ BAD: Global state or imports
class AgentCacheStore:
    def __init__(self) -> None:
        self._pool = GLOBAL_POOL  # Bad!
        from .persistence import cache  # Bad!
        self._persistence = cache
```

## Debugging

### Run Single Test

```bash
pytest tests/unit/test_services.py::TestBlockPool::test_allocate_basic -v
```

### Debug with Breakpoint

```python
def allocate(self, n_blocks: int, layer_id: int, agent_id: str) -> list[KVBlock]:
    breakpoint()  # Debugger stops here
    blocks = self._free_blocks[-n_blocks:]
    ...
```

### View Coverage Report

```bash
pytest tests/unit/ --cov=src/agent_memory/domain --cov-report=html
open htmlcov/index.html
```

## Resources

- **Hexagonal Architecture**: [Alistair Cockburn](https://alistair.cockburn.us/hexagonal-architecture/)
- **Domain-Driven Design**: [Eric Evans](https://www.domainlanguage.com/ddd/)
- **Hypothesis**: [Property-based testing guide](https://hypothesis.readthedocs.io/)
- **Ruff**: [Linter documentation](https://docs.astral.sh/ruff/)
- **MyPy**: [Type checking guide](https://mypy.readthedocs.io/)

## Getting Help

- **Questions**: [GitHub Discussions](https://github.com/yshk-mxim/agent-memory/discussions)
- **Bugs**: [GitHub Issues](https://github.com/yshk-mxim/agent-memory/issues)
- **Chat**: Project Slack (request invite in Discussions)
