# EXP-011: MLX Memory Reclamation Validation

**Date**: 2026-01-25
**Status**: IN PROGRESS
**Critical**: YES - Sprint 5 BLOCKER

## Objective

Validate that unloading an MLX model reclaims >95% of allocated memory.

**Hypothesis**: `del model` + `gc.collect()` + `mx.metal.clear_cache()` will reclaim memory

**If FAILS**: Entire Sprint 5 hot-swap architecture is blocked. Fallback to process-level swapping.

## Experimental Design

1. Measure baseline memory: `mx.metal.get_active_memory()`
2. Load model: `mlx_lm.load("mlx-community/SmolLM2-135M-Instruct")`
3. Measure post-load memory
4. Unload: `del model, tokenizer; gc.collect(); mx.metal.clear_cache()`
5. Measure post-unload memory
6. Calculate reclamation %: `(post_load - post_unload) / (post_load - baseline)`

**Success Criteria**: Reclamation >95%

## Results

**Date Run**: 2026-01-25
**Model**: mlx-community/SmolLM2-135M-Instruct

| Metric | Value |
|--------|-------|
| Baseline memory | 0.00 MB |
| After load | 257.50 MB |
| After unload | 0.00 MB |
| Model size | 257.50 MB |
| **Reclaimed** | **257.50 MB (100.0%)** |
| Residual | 0.00 MB |

**Reclamation Rate**: 100.0% ✅ (exceeds 95% requirement)

## Conclusion

**PASS** ✅

The experiment conclusively demonstrates that:
1. `del model, tokenizer` + `gc.collect()` + `mx.clear_cache()` reclaims 100% of model memory
2. No memory leaks or residual allocations
3. **Hot-swap architecture is VIABLE for Sprint 5**

### Implementation Notes

- Use `mx.get_active_memory()` instead of deprecated `mx.metal.get_active_memory()`
- Use `mx.clear_cache()` instead of deprecated `mx.metal.clear_cache()`

### Next Steps

Sprint 5 can proceed with confidence. The hot-swap protocol will work as designed:
1. Drain requests
2. Unload old model
3. `gc.collect()` + `mx.clear_cache()`
4. Load new model
5. Reconfigure BlockPool

No fallback to process restart needed.
