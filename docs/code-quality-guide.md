# Code Quality Patterns: AI Slop Detection & Prevention

**Version**: 1.0.0
**Date**: 2026-01-25
**Status**: Active
**Scope**: All code in `src/agent_memory/` and `tests/`

---

## Executive Summary

This document establishes code quality standards for the Semantic project, informed by comprehensive research on AI-generated code issues (2025-2026) and detailed review of our existing codebase. It defines:

1. **Red Flags** - Indicators of low-quality AI-generated code ("slop")
2. **Good Patterns** - Standards to follow
3. **Automated Enforcement** - Tools and configurations
4. **Detection Heuristics** - How to identify issues during review

### Key Statistics (Industry Research 2025-2026)

| Metric | Value | Source |
|--------|-------|--------|
| AI code issues vs human | 1.7x more | CodeRabbit |
| Security vulnerabilities in AI code | 45% | Veracode 2025 |
| "Comments Everywhere" pattern | 90-100% | OX Security |
| Hallucinated packages (open-source models) | 21.7% | UTSA Study |
| Developer distrust of AI output | 46% | Stack Overflow 2025 |

---

## 1. Red Flags: AI Slop Indicators

### 1.1 Over-Commenting (SEVERITY: MEDIUM)

**What It Looks Like**:
```python
# BAD: Comments restating obvious code
# 1. Validate inputs
if model is None:
    raise ModelNotFoundError("Model must be loaded")
# 2. Store dependencies
self._model = model
# 3. Initialize cache
self._cache = {}
```

**Why It's Bad**:
- AI fills code with redundant comments as navigation markers
- Comments become noise, obscuring genuinely important notes
- Maintenance burden when code changes but comments don't

**Detection**:
- Numbered comments (# 1., # 2., # 3.)
- Comments that mirror the next line of code
- Single-line statements with multi-line comments

**Good Pattern**:
```python
# GOOD: Only comment non-obvious behavior
self._model = model
self._cache = {}  # LRU cache with max 100 entries
```

---

### 1.2 Excessive Docstrings (SEVERITY: MEDIUM)

**What It Looks Like**:
```python
# BAD: 15-line docstring for trivial method
def is_empty(self) -> bool:
    """Check if the block is empty.

    A block is considered empty when it contains no tokens.
    This is useful for determining whether the block can be
    freed or needs to be retained.

    Returns:
        bool: True if the block has no tokens, False otherwise.

    Example:
        >>> block = KVBlock(block_id=1, layer_id=0, tokens=0)
        >>> block.is_empty()
        True
    """
    return self.tokens == 0
```

**Why It's Bad**:
- Method name already says `is_empty`
- Type hint already says `-> bool`
- Docstring adds no new information

**Detection**:
- Docstring longer than the code it documents
- Examples demonstrating obvious behavior
- Returns section restating the return type hint

**Good Pattern**:
```python
# GOOD: No docstring needed for self-explanatory methods
def is_empty(self) -> bool:
    return self.tokens == 0
```

**When Docstrings ARE Needed**:
- Public API methods
- Non-obvious algorithms
- Side effects that aren't apparent from signature
- Complex parameters with constraints

---

### 1.3 Sprint/Ticket References in Code (SEVERITY: LOW)

**What It Looks Like**:
```python
# BAD: Sprint references become meaningless
# Sprint 3.5 fix: CRITICAL-1 - Remove MLX from application layer
# NEW-5: Domain validation errors
# TODO (Day 7): Implement cache reconstruction
```

**Why It's Bad**:
- References become stale and meaningless
- Pollutes code with project management artifacts
- Creates confusion for new developers

**Detection**:
- Comments containing "Sprint", "Day", "Week"
- Issue tracker format (#123, PROJ-123)
- "NEW-X", "CRITICAL-X", "BLOCKER-X" patterns

**Good Pattern**:
```python
# GOOD: Reference ADRs (Architecture Decision Records) only
# See ADR-002: Universal 256-token block size

# GOOD: Keep sprint references in git commit messages only
```

---

### 1.4 Generic Variable Names (SEVERITY: MEDIUM)

**What It Looks Like**:
```python
# BAD: Generic names tell nothing about purpose
data = fetch_cache()
result = process(data)
temp = result.blocks
item = temp[0]
```

**Detection**:
- Variables named: `data`, `result`, `temp`, `item`, `obj`, `value`
- Loop variables: `x`, `y`, `i`, `j` (except for coordinates/indices)
- Plurals without context: `items`, `things`, `stuff`

**Good Pattern**:
```python
# GOOD: Names describe what the variable holds
agent_cache = fetch_cache()
processed_blocks = process(agent_cache)
first_layer_blocks = processed_blocks.blocks
initial_block = first_layer_blocks[0]
```

---

### 1.5 Magic Numbers (SEVERITY: HIGH)

**What It Looks Like**:
```python
# BAD: Unexplained numbers
blocks_per_agent = (tokens + 255) // 256
if layers > 8:
    global_layers = layers - 8
timeout = 300
chunk_size = 1024 * 1024 * 3
```

**Detection**:
- Numeric literals without named constants
- Same number appears in multiple places
- Numbers with no contextual meaning

**Good Pattern**:
```python
# GOOD: Named constants with documentation
BLOCK_SIZE_TOKENS = 256  # ADR-002: Universal block size
GEMMA3_GLOBAL_ATTENTION_LAYERS = 8  # First 8 layers use global attention
MODEL_LOAD_TIMEOUT_SECONDS = 300
CHUNK_SIZE_BYTES = 3 * 1024 * 1024  # 3MB chunks for streaming
```

---

### 1.6 Placeholder Code (SEVERITY: CRITICAL)

**What It Looks Like**:
```python
# BAD: Incomplete implementations
def process_cache(self) -> None:
    pass  # TODO: implement

def validate(self) -> bool:
    ...  # Will be implemented later

raise NotImplementedError("Coming in Sprint 4")
```

**Detection**:
- `pass` in non-abstract methods
- Ellipsis (`...`) outside Protocol/ABC definitions
- `NotImplementedError` with future promises
- `TODO` without associated issue tracker link

**Good Pattern**:
```python
# GOOD: Either implement or raise explicit error
def process_cache(self) -> None:
    raise NotImplementedError(
        "process_cache requires CachePersistencePort adapter"
    )

# Or better: don't commit until implemented
```

---

### 1.7 Defensive Programming Gone Wrong (SEVERITY: MEDIUM)

**What It Looks Like**:
```python
# BAD: Unnecessary checks
if hasattr(block, 'layer_data'):  # KVBlock ALWAYS has layer_data
    block.layer_data = None

# BAD: Validating already-validated data
def internal_method(self, blocks: list[KVBlock]) -> None:
    if blocks is None:  # Type hint says list, can't be None
        raise ValueError("blocks cannot be None")
```

**Detection**:
- `hasattr` checks for attributes that always exist
- Type validation when type hints guarantee the type
- Null checks on non-optional parameters
- Defensive copies of immutable objects

**Good Pattern**:
```python
# GOOD: Trust the type system
def internal_method(self, blocks: list[KVBlock]) -> None:
    # Type hint ensures blocks is list, no null check needed
    for block in blocks:
        block.layer_data = None
```

---

### 1.8 Silent Exception Swallowing (SEVERITY: HIGH)

**What It Looks Like**:
```python
# BAD: Errors disappear silently
try:
    save_to_disk(cache)
except Exception:
    pass  # Hope it works next time

try:
    convert_tensor(data)
except Exception:
    continue  # Skip bad data silently
```

**Detection**:
- Bare `except:` or `except Exception:`
- Empty exception handlers (`pass`, `continue`)
- No logging in exception blocks

**Good Pattern**:
```python
# GOOD: Specific exceptions with logging
try:
    save_to_disk(cache)
except OSError as e:
    logger.error(f"Failed to save cache: {e}")
    raise CachePersistenceError(f"Disk write failed: {e}") from e
except SerializationError as e:
    logger.warning(f"Serialization failed, retrying: {e}")
    # Only continue if recovery is possible
```

---

### 1.9 Runtime Imports Inside Functions (SEVERITY: MEDIUM)

**What It Looks Like**:
```python
# BAD: Import overhead on every call
def process_tensors(self, tensors: list) -> list:
    import numpy as np  # Imported every time function is called
    import time
    return [np.array(t) for t in tensors]
```

**Detection**:
- `import` statements inside function bodies
- `from X import Y` inside methods
- Especially problematic in hot paths

**Acceptable Exceptions**:
- Lazy loading for optional heavy dependencies
- Breaking circular imports (but fix the architecture instead)

**Good Pattern**:
```python
# GOOD: Top-level imports
import numpy as np
from datetime import datetime

# ACCEPTABLE: Documented lazy loading for optional deps
_mlx = None

def _get_mlx():
    global _mlx
    if _mlx is None:
        import mlx.core
        _mlx = mlx.core
    return _mlx
```

---

### 1.10 Test Code in Production Files (SEVERITY: HIGH)

**What It Looks Like**:
```python
# In production file batch_engine.py:
class BlockPoolBatchEngine:
    def _extract_cache(self, cache):
        # ... production code ...

        # BAD: Test detection in production
        if first_tensor.__class__.__name__ == 'FakeTensor':
            # Unit test mode: special handling
            return self._extract_fake_cache(cache)
```

**Detection**:
- Class name checks (`__class__.__name__`)
- `if DEBUG:` or `if TESTING:` blocks
- Mock detection logic
- `if __name__ == "__main__":` with test code

**Good Pattern**:
```python
# GOOD: Use dependency injection for testability
class BlockPoolBatchEngine:
    def __init__(self, cache_extractor: CacheExtractorPort):
        self._extractor = cache_extractor

    def _extract_cache(self, cache):
        return self._extractor.extract(cache)

# In tests: inject FakeCacheExtractor
# In production: inject RealCacheExtractor
```

---

## 2. Over-Engineering Patterns

### 2.1 Strategy Pattern for Single Implementation

**What It Looks Like**:
```python
# BAD: Full strategy pattern for 2 implementations
class LayerTypeDetectionStrategy(Protocol):
    def detect(self, model: Any, n_layers: int) -> list[str] | None: ...

class Gemma3DetectionStrategy:
    def detect(self, model, n_layers):
        return ["global"] * 8 + ["sliding_window"] * (n_layers - 8)

class UniformAttentionDetectionStrategy:
    def detect(self, model, n_layers):
        return ["full_attention"] * n_layers

# ... hundreds of lines of strategy management ...
```

**Why It's Bad**:
- Only 2 implementations exist
- No runtime strategy switching needed
- Simple if/else would suffice

**Good Pattern**:
```python
# GOOD: Simple, direct approach
def detect_layer_types(model: Any, n_layers: int) -> list[str]:
    if _is_gemma3_architecture(model):
        return ["global"] * 8 + ["sliding_window"] * (n_layers - 8)
    return ["full_attention"] * n_layers
```

**Rule of Three**: Don't create abstraction until you have 3+ real use cases.

---

### 2.2 Factory Patterns for Trivial Construction

**What It Looks Like**:
```python
# BAD: Factory for single object type
class CacheEntryFactory:
    @staticmethod
    def create(agent_id: str, blocks: AgentBlocks) -> CacheEntry:
        return CacheEntry(agent_id=agent_id, blocks=blocks)

# Usage
entry = CacheEntryFactory.create("agent_1", blocks)
```

**Good Pattern**:
```python
# GOOD: Direct construction
entry = CacheEntry(agent_id="agent_1", blocks=blocks)
```

---

### 2.3 God Methods (>50 lines)

**What It Looks Like**:
```python
# BAD: Method doing too many things
def _load_from_disk(self, agent_id: str) -> AgentBlocks | None:
    # 123 lines of:
    # - Path construction
    # - File reading
    # - Header parsing
    # - Validation
    # - Tensor reconstruction
    # - Block creation
    # - Error handling
```

**Good Pattern**:
```python
# GOOD: Decomposed into focused methods
def _load_from_disk(self, agent_id: str) -> AgentBlocks | None:
    path = self._get_cache_path(agent_id)
    if not path.exists():
        return None

    header = self._read_header(path)
    if not self._validate_compatibility(header):
        return None

    tensors = self._read_tensors(path)
    return self._reconstruct_blocks(tensors, header)
```

---

### 2.4 Configuration Complexity Exceeding Needs

**What It Looks Like**:
```python
# BAD: Unused configuration options
class MLXSettings(BaseSettings):
    experimental_feature_1: bool = False  # Never used
    experimental_feature_2: bool = False  # Never used
    legacy_mode: bool = False  # Never used
    future_extension_point: str = ""  # Never used
```

**Detection**:
- Settings with no references in code
- Configuration for hypothetical features
- "Reserved for future use" fields

**Good Pattern**:
- Only add configuration when actually needed
- Remove unused options aggressively

---

## 3. Architecture Violations

### 3.1 Infrastructure in Domain Layer (SEVERITY: CRITICAL)

**What It Looks Like**:
```python
# In domain/value_objects.py - BAD
class ModelCacheSpec:
    @classmethod
    def from_model(cls, model: Any) -> "ModelCacheSpec":
        args = model.args  # MLX-specific knowledge
        if hasattr(model.model, "layers"):  # Framework introspection
            for layer in model.model.layers:
                if hasattr(layer, "use_sliding"):  # MLX-specific
```

**Why It's Critical**:
- Domain should have ZERO knowledge of infrastructure
- Makes domain untestable without MLX
- Violates hexagonal architecture

**Good Pattern**:
```python
# Domain: Pure value object
@dataclass(frozen=True)
class ModelCacheSpec:
    n_layers: int
    n_kv_heads: int
    head_dim: int
    # ... just data, no construction from frameworks

# Adapter: Infrastructure knowledge
class MLXModelSpecExtractor:
    def extract_spec(self, model: Any) -> ModelCacheSpec:
        # All MLX-specific code here
        return ModelCacheSpec(
            n_layers=model.args.num_hidden_layers,
            # ...
        )
```

---

### 3.2 Infrastructure in Application Layer (SEVERITY: CRITICAL)

**What It Looks Like**:
```python
# In application/batch_engine.py - BAD
from mlx_lm.server import BatchGenerator
from mlx_lm.sample_utils import make_sampler

# In application/agent_cache_store.py - BAD
import numpy as np
from safetensors.numpy import save_file, load_file
```

**Good Pattern**:
- Application layer uses PORTS only
- Infrastructure code lives in ADAPTERS
- Inject adapters via constructor

---

### 3.3 Shadowing Built-in Names (SEVERITY: HIGH)

**What It Looks Like**:
```python
# BAD: Shadows Python's built-in TimeoutError
class TimeoutError(Exception):
    """Custom timeout error."""
    pass
```

**Why It's Bad**:
- Python 3.3+ has `TimeoutError` as built-in
- Creates confusion and potential bugs
- `except TimeoutError` catches wrong exception

**Good Pattern**:
```python
# GOOD: Unique name
class OperationTimeoutError(Exception):
    """Timeout during semantic operation."""
    pass
```

---

## 4. Automated Enforcement

### 4.1 Required Tools

| Tool | Purpose | Configuration |
|------|---------|---------------|
| **ruff** | Linting, formatting, import sorting | pyproject.toml |
| **mypy --strict** | Type checking | pyproject.toml |
| **bandit** | Security analysis | pyproject.toml |
| **vulture** | Dead code detection | CLI |
| **radon** | Complexity metrics | CLI |
| **sloppylint** | AI slop detection | CLI (new) |
| **jscpd** | Duplicate code detection | .jscpd.json |

### 4.2 Recommended ruff Rules

```toml
[tool.ruff.lint]
select = [
    "E", "W",     # pycodestyle
    "F",          # pyflakes
    "I",          # isort
    "B",          # flake8-bugbear
    "C4",         # flake8-comprehensions
    "UP",         # pyupgrade
    "ARG",        # flake8-unused-arguments
    "SIM",        # flake8-simplify
    "ERA",        # eradicate (commented-out code)
    "PL",         # pylint
    "RUF",        # ruff-specific
    "PERF",       # perflint
    "S",          # flake8-bandit (security)
    "C90",        # mccabe complexity
    "T20",        # flake8-print
    "TRY",        # tryceratops (exception handling)
]
```

### 4.3 Complexity Thresholds

| Metric | Threshold | Action |
|--------|-----------|--------|
| Cyclomatic complexity | ≤10 | Block merge if exceeded |
| Function length | ≤50 lines | Warning at 30, block at 50 |
| Class length | ≤300 lines | Warning |
| Nesting depth | ≤3 levels | Block if exceeded |
| Duplication | ≤5% | Warning |

### 4.4 Pre-commit Configuration

```yaml
# .pre-commit-config.yaml additions
repos:
  # AI slop detection
  - repo: local
    hooks:
      - id: sloppylint
        name: AI slop detector
        entry: sloppylint
        args: [--ci, --severity, high, src/agent_memory/]
        language: python
        types: [python]
        additional_dependencies: [sloppylint]

  # Dead code detection
  - repo: local
    hooks:
      - id: vulture
        name: Dead code check
        entry: vulture
        args: [src/agent_memory/, --min-confidence, "80"]
        language: python
        types: [python]
        additional_dependencies: [vulture]

  # Complexity check
  - repo: local
    hooks:
      - id: radon-cc
        name: Complexity check
        entry: radon
        args: [cc, --min, C, src/agent_memory/]
        language: python
        pass_filenames: false
        additional_dependencies: [radon]
```

---

## 5. Code Review Checklist

### Before Approving Any PR:

**Architecture**:
- [ ] No MLX/safetensors/numpy imports in domain or application layers
- [ ] All infrastructure code is in adapters
- [ ] Dependencies injected via constructor, not imported directly

**AI Slop**:
- [ ] No numbered comments (# 1., # 2.)
- [ ] No excessive docstrings on trivial methods
- [ ] No sprint/ticket references in code
- [ ] No generic variable names (data, result, temp)
- [ ] No magic numbers without constants

**Quality**:
- [ ] No functions over 50 lines
- [ ] No bare exception handlers
- [ ] No silent exception swallowing
- [ ] No test detection code in production
- [ ] All imports at module level (except documented lazy loading)

**Testing**:
- [ ] No old POC imports (`from src.X`)
- [ ] Tests use dependency injection, not mocks of internals
- [ ] Assertions are meaningful, not just "assert True"

---

## 6. Decision Record

| Decision | Date | Rationale |
|----------|------|-----------|
| Max function length: 50 lines | 2026-01-25 | Beyond this, split into focused functions |
| Max complexity: 10 | 2026-01-25 | Industry standard for maintainability |
| No sprint refs in code | 2026-01-25 | Use git history and ADRs instead |
| Docstrings only when needed | 2026-01-25 | Type hints + good names often suffice |
| Strategy pattern: 3+ impls | 2026-01-25 | Avoid premature abstraction |

---

## Appendix A: Research Sources

1. [The Register - AI-authored code contains worse bugs](https://www.theregister.com/2025/12/17/ai_code_bugs/)
2. [CodeRabbit - State of AI vs Human Code Generation Report](https://www.coderabbit.ai/blog/state-of-ai-vs-human-code-generation-report)
3. [OX Security Report - AI Generated Code Best Practices](https://www.prnewswire.com/news-releases/ox-report-ai-generated-code-violates-engineering-best-practices-undermining-software-security-at-scale-302592642.html)
4. [Qodo - State of AI Code Quality 2025](https://www.qodo.ai/reports/state-of-ai-code-quality/)
5. [Sonar - Poor Code Quality in AI-Accelerated Codebases](https://www.sonarsource.com/blog/the-inevitable-rise-of-poor-code-quality-in-ai-accelerated-codebases/)
6. [Martin Fowler - YAGNI](https://martinfowler.com/bliki/Yagni.html)

---

**Document Owner**: Quality Engineering
**Review Cycle**: Quarterly
**Next Review**: 2026-04-25
