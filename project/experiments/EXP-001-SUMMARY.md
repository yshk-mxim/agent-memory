# EXP-001 Summary: Model Args Validation

**Date**: 2026-01-24
**Status**: ✅ COMPLETE
**Document**: `/Users/dev_user/semantic/project/experiments/EXP-001-model-args.md`

## What We Did

Downloaded and inspected 4 MLX models to validate that all required attributes for cache spec extraction are accessible via `model.args`:

1. **Gemma 3 12B** (`mlx-community/gemma-3-12b-it-4bit`) - Hybrid SWA+Global attention
2. **Qwen1.5-MoE-A2.7B** (`mlx-community/Qwen1.5-MoE-A2.7B-4bit`) - Mixture of Experts
3. **Qwen 2.5-14B** (`mlx-community/Qwen2.5-14B-Instruct-4bit`) - Uniform full attention
4. **Llama 3.1-8B** (`mlx-community/Meta-Llama-3.1-8B-Instruct-4bit`) - Uniform full attention

## Key Findings

### ✅ All Required Attributes Are Accessible

| Attribute | Primary Name | Gemma 3 Location | Computation |
|-----------|--------------|------------------|-------------|
| Layer count | `num_hidden_layers` | `text_config['num_hidden_layers']` | Direct |
| KV heads | `num_key_value_heads` | `text_config['num_key_value_heads']` | Direct |
| Head dim | COMPUTED | COMPUTED | `hidden_size // num_attention_heads` |
| Sliding window | `sliding_window` | `text_config['sliding_window']` | Direct |

### ⚠️ Special Cases

1. **Gemma 3 has nested config**: All attributes in `args.text_config` dict, not as direct attributes
2. **No sliding_window_pattern attribute**: Must use model-type heuristics or inspect layer objects
3. **Head dim should always be computed**: Don't rely on `head_dim` attribute existing
4. **Layer pattern detection requires three-tier approach**: Check `layer_types`, inspect layers, fallback to heuristics

## Quick Reference: Attribute Extraction

```python
# Standard models (Qwen, Llama)
num_layers = args.num_hidden_layers
num_kv_heads = args.num_key_value_heads
hidden_size = args.hidden_size
num_heads = args.num_attention_heads
sliding_window = args.sliding_window  # None if not used

# Gemma 3 (nested config)
num_layers = args.text_config['num_hidden_layers']
num_kv_heads = args.text_config['num_key_value_heads']
hidden_size = args.text_config['hidden_size']
num_heads = args.text_config['num_attention_heads']
sliding_window = args.text_config['sliding_window']

# ALWAYS compute head_dim (never rely on attribute)
head_dim = hidden_size // num_heads
```

## Hybrid Attention Detection (Three-Tier Approach)

### Tier 1: Check `layer_types` attribute (most reliable)
```python
if hasattr(args, 'layer_types'):
    # Llama 3.1 has this: ['full_attention', 'full_attention', ...]
    # If multiple types exist, it's hybrid
```

### Tier 2: Inspect layer objects
```python
if hasattr(model.model, 'layers'):
    # Check each layer's use_sliding attribute
    for layer in model.model.layers:
        if hasattr(layer, 'use_sliding'):
            # Track which layers use sliding window
```

### Tier 3: Model-type heuristics (fallback)
```python
HYBRID_PATTERNS = {
    'gemma3': {
        'global_layers': 8,      # First 8 layers
        'sliding_layers': 40,    # Remaining 40 layers
        'sliding_window': 1024
    }
}
```

## Implementation Checklist

For `ModelCacheSpec.from_model()`:

- [ ] Handle standard models (Qwen, Llama)
- [ ] Handle Gemma 3 nested text_config
- [ ] ALWAYS compute head_dim (hidden_size // num_attention_heads)
- [ ] Handle None sliding_window (means full attention)
- [ ] Implement three-tier layer pattern detection
- [ ] Add unit tests for all 4 models
- [ ] Document Gemma 3 special handling

## Test Coverage

All 4 models successfully loaded and inspected:

- ✅ Gemma 3 12B: Nested config, hybrid attention, sliding_window=1024
- ✅ Qwen1.5-MoE-A2.7B: MoE architecture, 60 experts, 4 experts per token
- ✅ Qwen 2.5-14B: Standard config, full attention (sliding_window disabled)
- ✅ Llama 3.1-8B: Standard config, has layer_types attribute, use_sliding=False

## Files Generated

1. `/Users/dev_user/semantic/project/experiments/EXP-001-model-args.md` - Full findings document
2. `/Users/dev_user/semantic/model_inspection_results.json` - Raw inspection data
3. `/Users/dev_user/semantic/inspect_models.py` - Initial inspection script
4. `/Users/dev_user/semantic/inspect_detailed.py` - Detailed inspection script
5. `/Users/dev_user/semantic/inspect_qwen_moe.py` - MoE inspection script

## Next Steps

1. Implement `ModelCacheSpec.from_model()` using recommended extraction logic
2. Add unit tests for all 4 models
3. Document Gemma 3 special handling in architecture docs
4. Update sprint plan with findings

## Answers to Original Questions

| Question | Answer |
|----------|--------|
| Can we extract required attributes from model.args? | ✅ YES (with special handling for Gemma 3) |
| Are attribute names consistent? | ⚠️ MOSTLY (primary names consistent, Gemma nested) |
| Does sliding_window_pattern attribute exist? | ❌ NO (use model-type heuristics) |
| Can we distinguish layer types programmatically? | ⚠️ PARTIALLY (via layer_types or layer inspection) |

---

**Ready for implementation!** See full document for complete extraction logic and code examples.
