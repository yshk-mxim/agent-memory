# Remediation Plan: Code Quality Cleanup

**Version**: 1.0.0
**Date**: 2026-01-25
**Target**: Sprint 4 Start
**Status**: Ready for Execution

---

## Executive Summary

Comprehensive code review identified **~2,700 lines of dead POC code** to remove, **5 critical architecture violations** to fix, and **numerous AI slop indicators** to clean up. This plan provides prioritized, actionable remediation steps.

### Issue Summary

| Severity | Count | Category |
|----------|-------|----------|
| **CRITICAL** | 5 | Architecture violations, dead code |
| **HIGH** | 9 | God methods, infrastructure leakage, test pollution |
| **MEDIUM** | 18 | Over-commenting, magic numbers, docstring bloat |
| **LOW** | 15+ | Sprint refs, minor improvements |

### Estimated Effort

| Phase | Effort | Description |
|-------|--------|-------------|
| Phase 1: Dead Code Removal | 1 hour | Delete 11 POC files + 6 test files |
| Phase 2: Critical Fixes | 4 hours | Architecture violations |
| Phase 3: High Priority | 3 hours | God methods, infrastructure cleanup |
| Phase 4: Medium Priority | 2 hours | Comment cleanup, constants |
| **Total** | **~10 hours** | |

---

## Phase 1: Dead Code Removal (CRITICAL)

### 1.1 Remove Old POC Source Files

**Location**: `/Users/dev_user/semantic/src/` (root level, NOT `src/semantic/`)

| File | Lines | Reason for Removal |
|------|-------|-------------------|
| `mlx_utils.py` | 72 | Superseded by adapter layer |
| `mlx_cache_extractor.py` | 228 | Superseded by `BlockPoolBatchEngine` |
| `a2a_server.py` | 366 | Unused POC, dead code |
| `cache_persistence.py` | 207 | Superseded by `AgentCacheStore` |
| `batched_engine.py` | 261 | Superseded by `BlockPoolBatchEngine` |
| `api_server.py` | 430 | Superseded, uses deprecated patterns |
| `agent_manager.py` | 414 | God class, superseded |
| `concurrent_manager.py` | 355 | Superseded by new async patterns |
| `__init__.py` | 28 | Exports dead code |

**Action**:
```bash
# Create backup branch
git checkout -b backup/poc-code-before-removal

# Return to main and remove files
git checkout main
rm /Users/dev_user/semantic/src/mlx_utils.py
rm /Users/dev_user/semantic/src/mlx_cache_extractor.py
rm /Users/dev_user/semantic/src/a2a_server.py
rm /Users/dev_user/semantic/src/cache_persistence.py
rm /Users/dev_user/semantic/src/batched_engine.py
rm /Users/dev_user/semantic/src/api_server.py
rm /Users/dev_user/semantic/src/agent_manager.py
rm /Users/dev_user/semantic/src/concurrent_manager.py

# Keep but refactor (Phase 1.2)
# src/config.py -> migrate to Pydantic settings
# src/utils.py -> migrate API clients to adapters
```

**Lines Removed**: ~2,361

### 1.2 Migrate Needed Functionality

**Files to Migrate** (not delete):

| File | Migration Target | Notes |
|------|-----------------|-------|
| `config.py` (105 lines) | Use `semantic.adapters.config.settings` | Ensure all callers use Pydantic |
| `utils.py` (264 lines) | Move `APIClients` to new adapter | Fix outdated model IDs |

**Migration Steps**:

1. **config.py**: Verify nothing imports from `src.config`, then delete
2. **utils.py**: Extract `APIClients` class to `semantic/adapters/outbound/api_clients.py`
   - Fix model ID: `"claude-haiku-3-5-20250110"` → `"claude-haiku-4-5-20251001"`
   - Remove print statements, use logging
   - Remove emojis from output

### 1.3 Remove Old POC Test Files

**Location**: `/Users/dev_user/semantic/tests/` (root level)

| File | Reason for Removal |
|------|-------------------|
| `test_concurrent_manager.py` | Tests deleted `concurrent_manager.py` |
| `test_api_server.py` | Tests deleted `api_server.py` |
| `test_agent_manager.py` | Tests deleted `agent_manager.py` |
| `test_cache_persistence.py` | Tests deleted `cache_persistence.py` |
| `test_cache_extractor.py` | Tests deleted `mlx_cache_extractor.py` |
| `test_a2a_server.py` | Tests deleted `a2a_server.py` |

**Action**:
```bash
rm /Users/dev_user/semantic/tests/test_concurrent_manager.py
rm /Users/dev_user/semantic/tests/test_api_server.py
rm /Users/dev_user/semantic/tests/test_agent_manager.py
rm /Users/dev_user/semantic/tests/test_cache_persistence.py
rm /Users/dev_user/semantic/tests/test_cache_extractor.py
rm /Users/dev_user/semantic/tests/test_a2a_server.py
```

### 1.4 Update pyproject.toml

**Remove** these exclusions (no longer needed after deletion):
```toml
# In [tool.ruff.lint]
extend-exclude = [
    # REMOVE THESE LINES after deleting POC files
    "src/*.py",
    "tests/test_*.py",
]
```

---

## Phase 2: Critical Architecture Fixes

### 2.1 Extract MLX Knowledge from Domain Layer

**File**: `src/semantic/domain/value_objects.py`
**Issue**: `from_model()` method contains MLX-specific knowledge (lines 154-210, 291-342)
**Severity**: CRITICAL

**Current (Bad)**:
```python
# In domain/value_objects.py
class ModelCacheSpec:
    @classmethod
    def from_model(cls, model: Any) -> "ModelCacheSpec":
        args = model.args  # MLX-specific
        # ... 100+ lines of MLX introspection
```

**Fix**:

1. Create new adapter file:
```python
# src/semantic/adapters/outbound/mlx_spec_extractor.py
"""MLX model specification extraction adapter."""

from typing import Any
from semantic.domain.value_objects import ModelCacheSpec

class MLXModelSpecExtractor:
    """Extracts ModelCacheSpec from MLX models."""

    def extract(self, model: Any) -> ModelCacheSpec:
        """Extract cache specification from loaded MLX model."""
        args = model.args
        n_layers = args.num_hidden_layers
        n_kv_heads = self._get_kv_heads(args)
        head_dim = self._compute_head_dim(args)
        layer_types = self._detect_layer_types(model, n_layers)

        return ModelCacheSpec(
            n_layers=n_layers,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            layer_types=layer_types,
        )

    # Move all helper methods here:
    # - _get_kv_heads()
    # - _compute_head_dim()
    # - _detect_layer_types()
    # - Gemma3DetectionStrategy
    # - UniformAttentionDetectionStrategy
```

2. Remove from domain layer:
   - Delete `from_model()` method
   - Delete `Gemma3DetectionStrategy` class
   - Delete `UniformAttentionDetectionStrategy` class
   - Delete `_detect_layer_types()` method

3. Update callers to use extractor adapter.

**Lines Changed**: ~150 lines moved, ~100 lines removed

### 2.2 Extract Infrastructure from Application Layer

**File**: `src/semantic/application/batch_engine.py`
**Issues**:
- Line 204: Direct `from mlx_lm.server import BatchGenerator`
- Line 222: Direct `from mlx_lm.sample_utils import make_sampler`
**Severity**: CRITICAL

**Fix**:

1. Create BatchGenerator factory port:
```python
# src/semantic/ports/outbound.py (add to existing file)

class BatchGeneratorFactoryPort(Protocol):
    """Factory for creating batch generators."""

    def create(
        self,
        model: Any,
        stop_tokens: set[int],
    ) -> Any:
        """Create a batch generator for the model."""
        ...

    def make_sampler(self, temp: float) -> Any:
        """Create a sampler with given temperature."""
        ...
```

2. Create MLX adapter:
```python
# src/semantic/adapters/outbound/mlx_batch_generator.py

from mlx_lm.server import BatchGenerator
from mlx_lm.sample_utils import make_sampler
from semantic.ports.outbound import BatchGeneratorFactoryPort

class MLXBatchGeneratorFactory(BatchGeneratorFactoryPort):
    def create(self, model: Any, stop_tokens: set[int]) -> BatchGenerator:
        return BatchGenerator(model=model, stop_tokens=stop_tokens)

    def make_sampler(self, temp: float) -> Any:
        return make_sampler(temp=temp)
```

3. Update `BlockPoolBatchEngine` to accept factory via constructor.

### 2.3 Extract Infrastructure from AgentCacheStore

**File**: `src/semantic/application/agent_cache_store.py`
**Issues**:
- Lines 354-356: `import numpy as np`, `from safetensors.numpy import save_file`
- Lines 443-452: Raw file I/O with `open()`, `struct.unpack()`
**Severity**: CRITICAL

**Fix**:

1. Create cache persistence port (may already exist, enhance it):
```python
# src/semantic/ports/outbound.py

class CachePersistencePort(Protocol):
    """Persistence operations for agent caches."""

    def save(
        self,
        path: Path,
        tensors: dict[str, Any],
        metadata: dict[str, str],
    ) -> None:
        """Save tensors to disk with metadata."""
        ...

    def load(
        self,
        path: Path,
    ) -> tuple[dict[str, Any], dict[str, str]] | None:
        """Load tensors and metadata from disk."""
        ...
```

2. Create safetensors adapter:
```python
# src/semantic/adapters/outbound/safetensors_persistence.py

import numpy as np
from safetensors.numpy import save_file, load_file
from semantic.ports.outbound import CachePersistencePort

class SafetensorsCachePersistence(CachePersistencePort):
    def save(self, path, tensors, metadata):
        # Move all save logic here
        ...

    def load(self, path):
        # Move all load logic here
        ...
```

3. Update `AgentCacheStore` to use injected persistence adapter.

### 2.4 Fix TimeoutError Shadowing

**File**: `src/semantic/adapters/utils/timeouts.py`
**Issue**: Line 13-15 shadows Python's built-in `TimeoutError`
**Severity**: HIGH

**Fix**:
```python
# Before (BAD)
class TimeoutError(Exception):
    """Custom timeout error."""
    pass

# After (GOOD)
class OperationTimeoutError(Exception):
    """Timeout during semantic operation."""
    pass
```

Update all references to use `OperationTimeoutError`.

### 2.5 Remove Test Code from Production

**File**: `src/semantic/application/batch_engine.py`
**Issue**: Lines 479-501 contain FakeTensor detection logic
**Severity**: HIGH

**Fix**: Use dependency injection pattern:

1. The `_extract_cache` method should use the injected `CacheOperationsPort`
2. For tests, inject a `FakeCacheOperations` adapter
3. Remove all `FakeTensor` class name detection

---

## Phase 3: High Priority Fixes

### 3.1 Split God Methods

**File**: `src/semantic/application/agent_cache_store.py`

| Method | Lines | Target |
|--------|-------|--------|
| `_load_from_disk` | 123 | Split into 4 methods |
| `_save_to_disk` | 78 | Split into 3 methods |

**Fix for `_load_from_disk`**:
```python
def _load_from_disk(self, agent_id: str) -> AgentBlocks | None:
    path = self._get_cache_path(agent_id)
    if not path.exists():
        return None

    header = self._read_cache_header(path)
    if not self._validate_cache_compatibility(header):
        return None

    tensors = self._persistence.load(path)
    return self._reconstruct_blocks_from_tensors(tensors, header)
```

**File**: `src/semantic/application/batch_engine.py`

| Method | Lines | Target |
|--------|-------|--------|
| `_extract_cache` | 148 | Split into 3 methods |

### 3.2 Fix Exception Handling

**File**: `src/semantic/application/agent_cache_store.py`

**Lines 411-414** (silent exception):
```python
# Before (BAD)
except Exception:
    continue

# After (GOOD)
except (ValueError, TypeError) as e:
    logger.warning(f"Failed to convert block {block_idx}: {e}")
    continue
```

**Lines 541-543**:
```python
# Before (BAD)
except Exception:
    return None

# After (GOOD)
except (OSError, ValueError) as e:
    logger.error(f"Failed to load cache for {agent_id}: {e}")
    return None
```

### 3.3 Move Runtime Imports to Module Level

**Files with runtime imports**:

| File | Line | Import |
|------|------|--------|
| `batch_engine.py` | 154, 298 | `import logging` |
| `agent_cache_store.py` | 126 | `import time` |
| `agent_cache_store.py` | 448 | `import struct` |
| `mlx_cache_adapter.py` | 46 | `import mlx.core as mx` |

**Fix**: Move all imports to module level (except documented lazy loading).

For MLX (acceptable lazy loading pattern):
```python
# At module level
_mx = None

def _get_mlx():
    global _mx
    if _mx is None:
        import mlx.core as mx
        _mx = mx
    return _mx
```

---

## Phase 4: Medium Priority Cleanup

### 4.1 Remove Sprint/Ticket References

**Search pattern**: `Sprint|NEW-|CRITICAL-|BLOCKER-|Day \d|Week \d`

**Files affected**:
- `batch_engine.py`: Lines 68, 149, 475, 503, 514
- `agent_cache_store.py`: Lines 8-9, 372, 479
- `value_objects.py`: Line 112
- `services.py`: Lines 212, 255
- `entities.py`: Line 15
- `errors.py`: Line 44
- `logging.py`: Line 35
- `outbound.py`: Line 189
- `mlx_cache_adapter.py`: Line 1

**Action**: Remove these comments entirely or replace with ADR references.

### 4.2 Define Magic Number Constants

**File**: `src/semantic/domain/entities.py`
Already has `BLOCK_SIZE_TOKENS = 256` - GOOD

**File**: `src/semantic/domain/value_objects.py`
```python
# Add at top
GEMMA3_GLOBAL_ATTENTION_LAYERS = 8  # First 8 layers use global attention
```

**File**: `src/semantic/application/agent_cache_store.py`
```python
# Add at top
CACHE_HEADER_SIZE_BYTES = 8
BLOCK_ID_MULTIPLIER = 1000  # For synthetic block IDs
```

**File**: `src/semantic/application/batch_engine.py`
```python
# Add at top
FINISH_REASON_STOP = "stop"
FINISH_REASON_LENGTH = "length"
```

### 4.3 Reduce Docstring Verbosity

**Files with excessive docstrings**:
- `entities.py`: Lines 21-46 (25-line docstring for 4-field dataclass)
- `entities.py`: Lines 54-68 (4-line docstring for `is_full()`)
- `value_objects.py`: Lines 273-288 (14-line docstring for one-liner)
- `services.py`: Lines 23-70 (47-line class docstring)

**Rule**: If method name + type hints explain behavior, no docstring needed.

**Example Fix**:
```python
# Before: 25 lines of docstring
def is_empty(self) -> bool:
    """Check if the block is empty.

    A block is considered empty when...
    [20 more lines]
    """
    return self.tokens == 0

# After: No docstring needed
def is_empty(self) -> bool:
    return self.tokens == 0
```

### 4.4 Fix Docstring/Implementation Mismatches

**File**: `src/semantic/domain/services.py`
**Line 124 vs 138**: Docstring says `ValueError`, code raises `BlockOperationError`

```python
# Fix docstring to match implementation:
"""
Raises:
    BlockOperationError: If n_blocks <= 0 or layer_id invalid.
"""
```

### 4.5 Remove Dead Code

**File**: `src/semantic/application/agent_cache_store.py`
**Line 186**: `self._prefix_trie: dict[int | str, Any] = {}` - declared but never used

**Action**: Remove or implement prefix trie functionality.

---

## Phase 5: Configuration Updates

### 5.1 Add New Pre-commit Hooks

Add to `.pre-commit-config.yaml`:
```yaml
# AI slop detection
- repo: local
  hooks:
    - id: sloppylint
      name: AI slop detector
      entry: sloppylint
      args: [--ci, --severity, high, src/semantic/]
      language: python
      types: [python]
      additional_dependencies: [sloppylint]

# Dead code detection
- repo: local
  hooks:
    - id: vulture
      name: Dead code check
      entry: vulture
      args: [src/semantic/, --min-confidence, "80"]
      language: python
      types: [python]
      additional_dependencies: [vulture]

# Complexity check
- repo: local
  hooks:
    - id: radon-complexity
      name: Complexity check
      entry: radon
      args: [cc, --min, C, src/semantic/]
      language: python
      pass_filenames: false
      additional_dependencies: [radon]
```

### 5.2 Update Makefile

Add new targets:
```makefile
# Clean old POC code
clean-poc:
	@echo "Files to remove:"
	@ls -la src/*.py 2>/dev/null || echo "No POC files found"
	@ls -la tests/test_*.py 2>/dev/null || echo "No old tests found"

# AI slop check
slop-check:
	sloppylint src/semantic/ --severity medium

# Dead code check
dead-code:
	vulture src/semantic/ --min-confidence 80

# Complexity check
complexity:
	radon cc src/semantic/ --min C --show-complexity
```

### 5.3 Update Coverage Configuration

In `pyproject.toml`:
```toml
[tool.coverage.run]
omit = [
    "*/tests/*",
    "*/__init__.py",
    "src/*.py",  # Exclude old POC code
    "tests/test_*.py",  # Exclude old POC tests
]
```

---

## Verification Checklist

After completing all phases, verify:

### Phase 1 Verification
- [ ] `ls src/*.py` returns only `__init__.py` (if kept) or empty
- [ ] `ls tests/test_*.py` returns empty
- [ ] `make test-unit` passes
- [ ] `make test-integration` passes

### Phase 2 Verification
- [ ] `grep -r "from mlx" src/semantic/domain/` returns nothing
- [ ] `grep -r "from mlx" src/semantic/application/` returns nothing
- [ ] `grep -r "import numpy" src/semantic/application/` returns nothing
- [ ] `grep -r "safetensors" src/semantic/application/` returns nothing

### Phase 3 Verification
- [ ] No method in application layer > 50 lines
- [ ] `grep -rn "except Exception:" src/semantic/` returns nothing (or documented cases only)
- [ ] All imports at module level (except documented lazy loading)

### Phase 4 Verification
- [ ] `grep -rn "Sprint\|NEW-\|CRITICAL-" src/semantic/` returns nothing
- [ ] All magic numbers have named constants
- [ ] `make typecheck` passes

### Phase 5 Verification
- [ ] `make slop-check` passes
- [ ] `make dead-code` passes
- [ ] `make complexity` shows all functions ≤ C grade

---

## Post-Remediation Metrics

Track these metrics before and after:

| Metric | Before | After | Target |
|--------|--------|-------|--------|
| Total source lines | ~6,000 | ~3,300 | -45% |
| Files with MLX imports in domain | 1 | 0 | 0 |
| Files with MLX imports in application | 2 | 0 | 0 |
| Methods > 50 lines | 5 | 0 | 0 |
| Sprint references in code | 15+ | 0 | 0 |
| Unit test coverage | ~75% | 85%+ | 85% |

---

## Risk Assessment

| Risk | Mitigation |
|------|------------|
| Breaking changes during refactor | Create backup branch, run tests after each phase |
| Missing functionality after POC removal | Review POC code for unique features before deletion |
| New architecture violations | Add CI checks for import violations |

---

**Document Owner**: Engineering Lead
**Approved By**: (pending)
**Execution Start**: Sprint 4, Day 1
