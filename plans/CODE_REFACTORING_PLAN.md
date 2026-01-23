# Code Refactoring Plan: Transition to Persistent Multi-Agent POC

**Date**: January 23, 2026
**Purpose**: Reorganize codebase for new POC direction
**Target**: Clean separation of archived code vs new POC code

---

## Current State

### Existing Code (`src/`)

| File | Lines | Purpose | Status |
|------|-------|---------|--------|
| `semantic_isolation_mlx.py` | 439 | MLX-based semantic isolation tester | ♻️ Extract utilities, archive rest |
| `semantic_isolation.py` | 825 | HuggingFace-based semantic isolation | ❌ Archive (not using HF) |
| `dataset_generator.py` | ~400 | Generate multi-agent examples | ❌ Archive (POC uses manual examples) |
| `evaluator.py` | ~380 | Automated metrics suite | ❌ Archive (not needed for POC) |
| `llm_judge.py` | ~270 | Claude AI judge integration | ❌ Archive (not needed for POC) |
| `rule_checker.py` | ~820 | Rule-based validation | ❌ Archive (not needed for POC) |
| `compression.py` | ~150 | KV cache compression experiments | ❌ Archive |
| `validator.py` | ~260 | Dataset validation | ❌ Archive |
| `config.py` | ~90 | Configuration management | ✅ Keep (may be useful) |
| `utils.py` | ~240 | General utilities | ✅ Keep (may be useful) |

### Existing Tests (`tests/`)

| File | Purpose | Status |
|------|---------|--------|
| `test_kv_cache_isolation.py` | Tests semantic isolation | ❌ Archive |
| `test_evaluators.py` | Tests metrics suite | ❌ Archive |
| `test_gemma_only.py` | Tests Gemma API | ❌ Archive |
| `test_apis.py` | Tests API integrations | ❌ Archive |

### Existing Scripts (`scripts/`)

| File | Purpose | Status |
|------|---------|--------|
| `test_mlx_basic.py` | Basic MLX functionality test | ✅ Keep (useful for validation) |
| `test_gemma3_simple.py` | Simple Gemma 3 test | ✅ Keep (useful) |
| `debug_mlx_context.py` | MLX context debugging | ✅ Keep (may be useful) |
| `archive/` | 8 old scripts | ✅ Already archived |

---

## Refactoring Strategy

### Phase 1: Archive Old Code

**Create archive structure**:
```
archive/
└── semantic_isolation/
    ├── README.md (explanation of archived code)
    ├── src/
    │   ├── semantic_isolation_mlx.py
    │   ├── semantic_isolation.py
    │   ├── dataset_generator.py
    │   ├── evaluator.py
    │   ├── llm_judge.py
    │   ├── rule_checker.py
    │   ├── compression.py
    │   └── validator.py
    └── tests/
        ├── test_kv_cache_isolation.py
        ├── test_evaluators.py
        ├── test_gemma_only.py
        └── test_apis.py
```

**Actions**:
1. Create `archive/semantic_isolation/src/` and `archive/semantic_isolation/tests/`
2. Move old code files to archive
3. Create comprehensive README explaining what was archived and why

### Phase 2: Extract Reusable Utilities

**From `semantic_isolation_mlx.py`, extract**:

```python
# NEW FILE: src/mlx_utils.py

class MLXModelLoader:
    """Utilities for loading and managing MLX models"""

    @staticmethod
    def load_model(model_name="mlx-community/gemma-3-12b-it-4bit"):
        """Load model and tokenizer"""
        from mlx_lm import load
        return load(model_name)

    @staticmethod
    def get_model_memory_usage():
        """Report model memory usage"""
        import mlx.core as mx
        return {
            'active_memory': mx.metal.get_active_memory(),
            'peak_memory': mx.metal.get_peak_memory(),
        }

    @staticmethod
    def clear_cache():
        """Clear Metal cache"""
        import mlx.core as mx
        mx.metal.clear_cache()
```

**Keep from existing**:
- `config.py` - May be useful for model configuration
- `utils.py` - General utilities (logging, timing, etc.)

### Phase 3: Create New POC Structure

**New files to create**:

```
src/
├── __init__.py (existing, keep)
├── config.py (existing, keep)
├── utils.py (existing, keep)
├── mlx_utils.py (NEW - extracted utilities)
├── mlx_cache_extractor.py (NEW - Component 1)
├── cache_persistence.py (NEW - Component 2)
└── agent_manager.py (NEW - Component 3)

tests/
├── __init__.py (NEW)
├── test_cache_extractor.py (NEW)
├── test_cache_persistence.py (NEW)
└── test_agent_manager.py (NEW)

demo_persistent_agents.py (NEW - user-facing demo)
```

---

## Detailed Refactoring Steps

### Step 1: Create Archive Structure

```bash
# Create archive directories
mkdir -p archive/semantic_isolation/src
mkdir -p archive/semantic_isolation/tests

# Create archive README
cat > archive/semantic_isolation/README.md << 'EOF'
# Archive: Semantic Isolation Code (Jan 22-23, 2026)

## What This Code Was

Implementation of semantic KV cache partitioning for virtual multi-agent systems.

### Components Archived

**Source Files** (`src/`):
- `semantic_isolation_mlx.py` - MLX-based semantic isolation tester (4 conditions)
- `semantic_isolation.py` - HuggingFace-based version
- `dataset_generator.py` - Multi-agent example generation
- `evaluator.py` - 16 mechanical metrics + evaluation suite
- `llm_judge.py` - Claude AI judge integration (3 qualitative metrics)
- `rule_checker.py` - Rule-based validation system
- `compression.py` - KV cache compression experiments
- `validator.py` - Dataset quality validation

**Test Files** (`tests/`):
- `test_kv_cache_isolation.py` - Tests for 4-condition isolation
- `test_evaluators.py` - Tests for automated metrics
- `test_gemma_only.py` - Gemma API tests
- `test_apis.py` - API integration tests

### Validation Results

From `results/validation_001_isolation_test_mlx.json`:
- ✅ Semantic isolation achieved (separate caches per cluster)
- ✅ Cache sizes: 419 (technical) + 452 (business) + 828 (synthesis) = 1699 tokens
- ✅ ~20% faster than sequential (36.58s vs 45.06s)

### Why Archived

**Fundamental issue**: Single-turn per agent scenarios don't benefit from KV cache isolation.
- Agents responded once each in validation
- Multi-turn specialists required for approach to have value
- Better opportunity: Persistent multi-agent memory (fills gap in existing tools)

**Decision**: Pivot to capability demonstration POC (2-3 weeks) vs research publication (15 weeks)

### Reusable Components

Extracted to `src/mlx_utils.py`:
- Model loading utilities
- Memory usage reporting
- Cache clearing functions

See: `/Users/dev_user/semantic/plans/POC_PLAN.md`

---

**Archived**: January 23, 2026
**Pivot Date**: January 23, 2026
EOF
```

### Step 2: Move Files to Archive

```bash
# Move source files
cd /Users/dev_user/semantic
mv src/semantic_isolation_mlx.py archive/semantic_isolation/src/
mv src/semantic_isolation.py archive/semantic_isolation/src/
mv src/dataset_generator.py archive/semantic_isolation/src/
mv src/evaluator.py archive/semantic_isolation/src/
mv src/llm_judge.py archive/semantic_isolation/src/
mv src/rule_checker.py archive/semantic_isolation/src/
mv src/compression.py archive/semantic_isolation/src/
mv src/validator.py archive/semantic_isolation/src/

# Move test files
mv tests/test_kv_cache_isolation.py archive/semantic_isolation/tests/
mv tests/test_evaluators.py archive/semantic_isolation/tests/
mv tests/test_gemma_only.py archive/semantic_isolation/tests/
mv tests/test_apis.py archive/semantic_isolation/tests/
```

### Step 3: Extract Reusable Utilities

Create `src/mlx_utils.py`:

```python
"""
MLX Utilities for Persistent Multi-Agent System

Extracted from semantic_isolation_mlx.py (archived)
Provides basic MLX model management utilities.
"""

import mlx.core as mx
from mlx_lm import load, generate
from typing import Tuple, Optional, Dict
import logging

logger = logging.getLogger(__name__)


class MLXModelLoader:
    """Utilities for loading and managing MLX models"""

    @staticmethod
    def load_model(model_name: str = "mlx-community/gemma-3-12b-it-4bit"):
        """
        Load MLX model and tokenizer

        Args:
            model_name: MLX model identifier

        Returns:
            (model, tokenizer) tuple
        """
        logger.info(f"Loading model: {model_name}")
        model, tokenizer = load(model_name)
        logger.info("Model loaded successfully")
        return model, tokenizer

    @staticmethod
    def get_memory_usage() -> Dict[str, int]:
        """
        Get current MLX memory usage

        Returns:
            Dictionary with active_memory and peak_memory in bytes
        """
        return {
            'active_memory': mx.metal.get_active_memory(),
            'peak_memory': mx.metal.get_peak_memory(),
            'active_memory_gb': mx.metal.get_active_memory() / (1024**3),
            'peak_memory_gb': mx.metal.get_peak_memory() / (1024**3),
        }

    @staticmethod
    def clear_cache():
        """Clear MLX Metal cache"""
        logger.info("Clearing MLX cache")
        mx.metal.clear_cache()

    @staticmethod
    def set_wired_limit(size_gb: int):
        """
        Set wired memory limit (macOS 15+ only)

        Args:
            size_gb: Memory size in GB to wire
        """
        if hasattr(mx, 'set_wired_limit'):
            logger.info(f"Setting wired memory limit: {size_gb}GB")
            mx.set_wired_limit(size_gb * 1024**3)
        else:
            logger.warning("mx.set_wired_limit not available (requires macOS 15+)")
```

### Step 4: Update Remaining Files

**Update `src/__init__.py`**:
```python
"""
Persistent Multi-Agent Memory System

A demonstration of persistent agent memory using KV cache
persistence on Mac with unified memory architecture.
"""

__version__ = "0.1.0"
__author__ = "Your Name"

from .mlx_utils import MLXModelLoader

__all__ = ['MLXModelLoader']
```

**Keep `src/config.py` and `src/utils.py`** as-is (may be useful).

### Step 5: Clean Up Results Directory

```bash
# Archive old validation results
mkdir -p archive/semantic_isolation/results
mv results/validation_001_isolation_test_mlx.json archive/semantic_isolation/results/
mv results/exp1_*.json archive/semantic_isolation/results/ 2>/dev/null || true
```

---

## New File Structure (After Refactoring)

```
/Users/dev_user/semantic/
├── README.md (UPDATE with new POC focus)
├── requirements.txt (existing)
├── demo_persistent_agents.py (NEW)
│
├── src/
│   ├── __init__.py (UPDATED)
│   ├── config.py (KEPT)
│   ├── utils.py (KEPT)
│   ├── mlx_utils.py (NEW - extracted)
│   ├── mlx_cache_extractor.py (NEW - Week 1)
│   ├── cache_persistence.py (NEW - Week 1)
│   └── agent_manager.py (NEW - Week 2)
│
├── tests/
│   ├── __init__.py (NEW)
│   ├── test_cache_extractor.py (NEW - Week 1)
│   ├── test_cache_persistence.py (NEW - Week 1)
│   └── test_agent_manager.py (NEW - Week 2)
│
├── scripts/
│   ├── README.md (existing)
│   ├── test_mlx_basic.py (KEPT)
│   ├── test_gemma3_simple.py (KEPT)
│   ├── debug_mlx_context.py (KEPT)
│   └── archive/ (existing, contains 8 old scripts)
│
├── plans/
│   ├── POC_PLAN.md (NEW)
│   ├── CODE_REFACTORING_PLAN.md (THIS FILE)
│   └── archive_semantic_isolation/ (archived plans)
│
├── novelty/
│   ├── EDGE_KV_CACHE_NOVELTY_REVIEW.md (kept)
│   ├── EXISTING_TOOLS_COMPARISON.md (kept)
│   └── archive_semantic_isolation/ (archived novelty files)
│
└── archive/
    └── semantic_isolation/
        ├── README.md (NEW - explanation)
        ├── src/ (8 archived source files)
        ├── tests/ (4 archived test files)
        └── results/ (archived validation results)
```

---

## Execution Checklist

### Pre-Refactoring
- [ ] Commit current state (before refactoring)
- [ ] Create backup if needed

### Refactoring Execution
- [ ] Create archive directories
- [ ] Create archive README.md
- [ ] Move old source files to archive
- [ ] Move old test files to archive
- [ ] Move old results to archive
- [ ] Create `src/mlx_utils.py` (extract utilities)
- [ ] Update `src/__init__.py`
- [ ] Clean up `src/` (only keep config, utils, mlx_utils, __init__)
- [ ] Clean up `tests/` (remove old tests, keep structure)

### Post-Refactoring
- [ ] Verify archive structure is correct
- [ ] Verify src/ has only 4 files (config, utils, mlx_utils, __init__)
- [ ] Verify tests/ is empty (ready for new tests)
- [ ] Update main README.md with new POC focus
- [ ] Commit refactored structure
- [ ] Push to remote

---

## Git Commit Message

```
Refactor: Archive semantic isolation code, prepare for POC

Major restructuring to pivot from semantic isolation research to
persistent multi-agent memory POC.

Archived:
- Semantic isolation implementation (semantic_isolation_mlx.py, 439 lines)
- HuggingFace version (semantic_isolation.py, 825 lines)
- Dataset generator, evaluator, metrics suite
- All associated tests and validation results
- Old research plans (updated_plan.v3.md, sprints 00-03)
- Novelty analysis files (debates, original NOVELTY.md)

Extracted:
- MLX utilities to src/mlx_utils.py (reusable model loading, memory management)

New structure:
- plans/POC_PLAN.md - Focused 2-3 week POC plan
- novelty/EDGE_KV_CACHE_NOVELTY_REVIEW.md - Comprehensive survey
- novelty/EXISTING_TOOLS_COMPARISON.md - LM Studio/Ollama/llama.cpp analysis

Reason for pivot:
- Original approach only valuable for multi-turn specialists
- Validation showed single-turn per agent (minimal benefit)
- Better opportunity: Fill gap in existing tools (persistent agent memory)

Next: Implement 3-component architecture (Week 1-2)
```

---

**Created**: January 23, 2026
**Status**: Ready to execute
**Estimated Time**: 30-45 minutes to complete refactoring
