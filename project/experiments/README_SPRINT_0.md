# Sprint 0 Critical Experiments

## MLX Sandbox Limitation

These experiments require Metal GPU access and cannot run in Claude Code sandbox.

**To run manually:**
```bash
# Outside sandbox
python experiments/exp_003_cache_injection.py
python experiments/exp_004_cache_extraction.py
```

## EXP-003: Cache Injection Validation

**Goal**: Prove that `caches=[loaded_cache]` parameter works on `BatchGenerator.insert()`

**Success criteria**: Output with pre-built cache matches output without cache

**File**: `exp_003_cache_injection.py`

## EXP-004: Per-Sequence Cache Extraction

**Goal**: Prove that per-sequence cache extraction works on completion via `Response.prompt_cache()`

**Success criteria**:
- Extract cache from each sequence in a batch
- Save/reload/re-inject cycle works
- Sequences complete independently (don't wait for full batch)

**File**: `exp_004_cache_extraction.py`

## BLOCKING Status

These experiments are **BLOCKING** for Sprint 1. If either fails, **invoke Plan B** (sequential engine).

## Expected Results

- ✅ Both pass → Continue with continuous batching architecture
- ❌ Either fails → Invoke Plan B (sequential engine with shared cache)

Failure triggers documented in `plans/production_plan.md` § Plan B.
