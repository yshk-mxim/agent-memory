# ADR-007: One Model At A Time (24GB Constraint)

**Status**: Draft
**Date**: 2026-01-25
**Decision Makers**: System Architect, ML Engineer, HW Engineer

## Context

On Apple Silicon M4 Pro with 24GB unified memory, loading multiple large language models simultaneously is not feasible:

- **Gemma 3 12B** (4-bit): ~6.5 GB weights + ~12 GB cache budget = 18.5 GB
- **Qwen 2.5-14B** (4-bit): ~9 GB weights + ~11 GB cache budget = 20 GB
- **Llama 3.1 8B** (4-bit): ~5 GB weights + ~14 GB cache budget = 19 GB

System overhead (OS + apps) requires ~4-5 GB, leaving insufficient room for multiple models.

## Decision

**Load only ONE model at a time**. When users need a different model, implement a **hot-swap protocol**:

1. Drain active requests
2. Evict all caches to disk (tagged with model ID)
3. Unload old model (`del` + `gc.collect()` + `mx.clear_cache()`)
4. Load new model
5. Reconfigure BlockPool for new model dimensions
6. Allow cached agents to resume when their model returns

## Consequences

### Positive

- **Memory efficiency**: Full 24GB available for single model + cache
- **Flexibility**: Support multiple models without hardware upgrade
- **Cache persistence**: Agents don't lose context when model swaps
- **Proven reclamation**: EXP-011 shows 100% memory recovery

### Negative

- **Swap latency**: 15-30s model switch time
- **Request interruption**: Active requests must complete before swap
- **Complexity**: Orchestration logic for drain/unload/load sequence

## Alternatives Considered

### 1. Multi-Model Co-Loading
**Rejected**: Even smallest combination (Llama 8B + SmolLM2 135M) leaves minimal cache budget

### 2. Process-Level Swapping
**Rejected as primary**: Slower (60s+), more complex IPC, but kept as fallback if hot-swap fails

### 3. Model Quantization (8-bit, 4-bit)
**Already Used**: All models use 4-bit quantization; further reduction degrades quality

## Implementation Notes

- ModelRegistry manages lifecycle
- BlockPool.reconfigure() already supports spec changes
- Cache compatibility validated via ModelTag
- Admin API triggers swaps (POST /admin/models/swap)

## References

- EXP-011: Memory reclamation validation (100% reclaimed)
- Sprint 5: Model Hot-Swap implementation
- backend_plan.md: Multi-model serving strategy
