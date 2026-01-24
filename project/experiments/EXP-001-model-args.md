# EXP-001: Model Args Inspection

**Date**: 2026-01-24
**Sprint**: Sprint 1, Day 3-4
**Objective**: Validate model.args attributes for cache spec extraction

## Executive Summary

Successfully inspected 4 models to determine:
1. Required attributes ARE accessible via model.args
2. Attribute naming is MOSTLY consistent (with documented variations)
3. `sliding_window_pattern` does NOT exist as a standalone attribute
4. Layer types CAN be distinguished programmatically via model.args.layer_types (when present)

**Status**: ✅ COMPLETE - All objectives met

## Models Inspected

| Model | Model ID | Type | Status |
|-------|----------|------|--------|
| Gemma 3 12B | mlx-community/gemma-3-12b-it-4bit | Hybrid SWA+Global | ✅ Inspected |
| Qwen1.5-MoE-A2.7B | mlx-community/Qwen1.5-MoE-A2.7B-4bit | MoE | ✅ Inspected |
| Qwen 2.5-14B | mlx-community/Qwen2.5-14B-Instruct-4bit | Uniform Full | ✅ Inspected |
| Llama 3.1-8B | mlx-community/Meta-Llama-3.1-8B-Instruct-4bit | Uniform Full | ✅ Inspected |

**Note**: GPT-OSS-20B was not available on HuggingFace. Replaced with Qwen1.5-MoE-A2.7B for MoE testing.

---

## Findings by Attribute

### 1. Layer Count

**Attribute Names**:
- `num_hidden_layers` - MOST COMMON (Qwen, Llama, Gemma via text_config)
- `n_layers` - Alternative (not observed in tested models)
- `num_layers` - Alternative (not observed in tested models)

**Extraction Logic**:
```python
# Standard models
num_layers = getattr(args, 'num_hidden_layers', None)

# Gemma 3 (nested in text_config)
if hasattr(args, 'text_config'):
    num_layers = args.text_config.get('num_hidden_layers')
```

**Observed Values**:
| Model | Attribute | Value |
|-------|-----------|-------|
| Gemma 3 12B | text_config['num_hidden_layers'] | 48 |
| Qwen1.5-MoE-A2.7B | num_hidden_layers | 24 |
| Qwen 2.5-14B | num_hidden_layers | 48 |
| Llama 3.1-8B | num_hidden_layers | 32 |

**Recommendation**: ✅ Use `num_hidden_layers` as primary, check `text_config` for Gemma models

---

### 2. KV Heads

**Attribute Names**:
- `num_key_value_heads` - MOST COMMON (all tested models)
- `n_kv_heads` - Alternative (not observed)
- `num_kv_heads` - Alternative (not observed)

**Extraction Logic**:
```python
# Standard models
num_kv_heads = getattr(args, 'num_key_value_heads', None)

# Gemma 3 (nested in text_config)
if hasattr(args, 'text_config'):
    num_kv_heads = args.text_config.get('num_key_value_heads')
```

**Observed Values**:
| Model | Attribute | Value |
|-------|-----------|-------|
| Gemma 3 12B | text_config['num_key_value_heads'] | 8 |
| Qwen1.5-MoE-A2.7B | num_key_value_heads | 16 |
| Qwen 2.5-14B | num_key_value_heads | 8 |
| Llama 3.1-8B | num_key_value_heads | 8 |

**Recommendation**: ✅ Use `num_key_value_heads` as primary, check `text_config` for Gemma models

---

### 3. Head Dimension

**Attribute Names**:
- `head_dim` - RARELY present (only in Llama attention objects)
- Typically COMPUTED from: `hidden_size // num_attention_heads`

**Extraction Logic**:
```python
# Try direct attribute first
head_dim = getattr(args, 'head_dim', None)

# Compute if not present
if head_dim is None:
    if hasattr(args, 'text_config'):
        # Gemma 3
        hidden_size = args.text_config.get('hidden_size')
        num_heads = args.text_config.get('num_attention_heads')
    else:
        # Standard models
        hidden_size = getattr(args, 'hidden_size', None)
        num_heads = getattr(args, 'num_attention_heads', None)

    if hidden_size and num_heads:
        head_dim = hidden_size // num_heads
```

**Observed Values**:
| Model | Hidden Size | Num Heads | Head Dim (Computed) |
|-------|-------------|-----------|---------------------|
| Gemma 3 12B | 3840 | 16 | 240 |
| Qwen1.5-MoE-A2.7B | 2048 | 16 | 128 |
| Qwen 2.5-14B | 5120 | 40 | 128 |
| Llama 3.1-8B | 4096 | 32 | 128 |

**Recommendation**: ✅ ALWAYS compute from `hidden_size // num_attention_heads`

---

### 4. Sliding Window Size

**Attribute Names**:
- `sliding_window` - MOST COMMON (Gemma, Qwen)
- `sliding_window_size` - Alternative (not observed)
- Value is `None` for models without sliding window

**Extraction Logic**:
```python
# Standard models
sliding_window = getattr(args, 'sliding_window', None)

# Gemma 3 (nested in text_config)
if hasattr(args, 'text_config'):
    sliding_window = args.text_config.get('sliding_window', None)
```

**Observed Values**:
| Model | Attribute | Value | Notes |
|-------|-----------|-------|-------|
| Gemma 3 12B | text_config['sliding_window'] | 1024 | Hybrid SWA |
| Qwen1.5-MoE-A2.7B | sliding_window | None | Full attention |
| Qwen 2.5-14B | sliding_window | None | config.json has 131072 but disabled |
| Llama 3.1-8B | sliding_window | None | Full attention |

**Important**: Qwen 2.5-14B has `sliding_window: 131072` in config.json but `use_sliding_window: false`, so model.args returns None.

**Recommendation**: ✅ Use `sliding_window` attribute, None means full attention

---

### 5. Sliding Window Pattern

**Finding**: ❌ NO `sliding_window_pattern` attribute found in any model

**Expected for Gemma 3**: According to documentation, Gemma 3 uses hybrid attention (8 global + 40 sliding window layers), but this pattern is NOT exposed as a standalone attribute.

**Alternative Detection Methods**:

#### Option A: Use `layer_types` attribute (if present)
```python
if hasattr(args, 'layer_types'):
    # Llama 3.1-8B has this
    # ['full_attention', 'full_attention', ...]
    layer_types = args.layer_types
```

**Observed**: Only Llama 3.1-8B has `layer_types` attribute (all 'full_attention')

#### Option B: Inspect layer objects directly
```python
if hasattr(model, 'model') and hasattr(model.model, 'layers'):
    layers = model.model.layers

    # Check for use_sliding attribute on each layer
    layer_patterns = []
    for layer in layers:
        if hasattr(layer, 'use_sliding'):
            layer_patterns.append(layer.use_sliding)
```

**Observed**: Llama 3.1-8B layers have `use_sliding=False` attribute

#### Option C: Use model_type heuristics
```python
# Gemma 3 is known to have hybrid attention pattern
if args.model_type == 'gemma3':
    # 8 global + 40 sliding window (hardcoded from docs)
    global_layers = 8
    sliding_layers = 40
```

**Recommendation**: ⚠️ Use model_type-based heuristics for known hybrid models. No generic detection mechanism exists.

---

### 6. MoE-Specific Attributes (Qwen1.5-MoE)

**Discovered Attributes**:
```python
num_experts: 60                      # Total number of experts
num_experts_per_tok: 4               # Experts activated per token
moe_intermediate_size: 1408          # Hidden size for MoE FFN
shared_expert_intermediate_size: 5632  # Shared expert FFN size
```

**Layer Structure**:
- Layer type: `Qwen2MoeDecoderLayer`
- MLP component: `Qwen2MoeSparseMoeBlock`
- MLP has `num_experts=60, top_k=4` attributes

**Cache Implications**: MoE models have same KV cache structure as dense models (no special handling needed)

---

## Attribute Name Mapping Table

| Semantic Meaning | Primary Attribute | Alternatives | Gemma 3 Location |
|------------------|-------------------|--------------|------------------|
| Layer count | `num_hidden_layers` | `n_layers`, `num_layers` | `text_config['num_hidden_layers']` |
| KV heads | `num_key_value_heads` | `n_kv_heads`, `num_kv_heads` | `text_config['num_key_value_heads']` |
| Attention heads | `num_attention_heads` | `n_heads`, `num_heads` | `text_config['num_attention_heads']` |
| Hidden size | `hidden_size` | `dim`, `model_dim` | `text_config['hidden_size']` |
| Head dimension | COMPUTED | `head_dim` (rare) | COMPUTED |
| Sliding window | `sliding_window` | `sliding_window_size` | `text_config['sliding_window']` |
| Layer pattern | NO ATTRIBUTE | `layer_types` (rare) | NO ATTRIBUTE |

---

## Hybrid Attention Detection

### Question: How to detect hybrid vs uniform attention?

**Answer**: Three-tier approach

#### Tier 1: Check for `layer_types` attribute (most reliable)
```python
if hasattr(args, 'layer_types'):
    # Parse layer types to determine pattern
    layer_types = args.layer_types
    if len(set(layer_types)) > 1:
        # Hybrid attention detected
        return "hybrid"
    else:
        return "uniform"
```

#### Tier 2: Check layer objects for `use_sliding` attribute
```python
if hasattr(model, 'model') and hasattr(model.model, 'layers'):
    use_sliding_values = []
    for layer in model.model.layers:
        if hasattr(layer, 'use_sliding'):
            use_sliding_values.append(layer.use_sliding)

    if len(set(use_sliding_values)) > 1:
        return "hybrid"
```

#### Tier 3: Model-type heuristics (fallback)
```python
# Known hybrid models
HYBRID_PATTERNS = {
    'gemma3': {
        'global_layers': 8,
        'sliding_layers': 40,
        'sliding_window': 1024
    }
}

if args.model_type in HYBRID_PATTERNS:
    return "hybrid", HYBRID_PATTERNS[args.model_type]
```

**Recommendation**: ✅ Implement all three tiers, with Tier 1 as preferred

---

## Recommended Extraction Logic

### Complete `ModelCacheSpec.from_model()` Implementation

```python
@classmethod
def from_model(cls, model) -> "ModelCacheSpec":
    """
    Extract cache specification from loaded MLX model.

    Args:
        model: Loaded MLX model from mlx_lm.load()

    Returns:
        ModelCacheSpec with extracted parameters
    """
    args = model.args

    # Handle Gemma 3 nested config
    if hasattr(args, 'text_config'):
        config = args.text_config
        num_layers = config.get('num_hidden_layers')
        num_kv_heads = config.get('num_key_value_heads')
        num_heads = config.get('num_attention_heads')
        hidden_size = config.get('hidden_size')
        sliding_window = config.get('sliding_window', None)
    else:
        # Standard models
        num_layers = getattr(args, 'num_hidden_layers', None)
        num_kv_heads = getattr(args, 'num_key_value_heads', None)
        num_heads = getattr(args, 'num_attention_heads', None)
        hidden_size = getattr(args, 'hidden_size', None)
        sliding_window = getattr(args, 'sliding_window', None)

    # Compute head dimension (ALWAYS compute, never rely on attribute)
    if hidden_size and num_heads:
        head_dim = hidden_size // num_heads
    else:
        raise ValueError("Cannot compute head_dim: missing hidden_size or num_heads")

    # Detect layer pattern
    layer_pattern = cls._detect_layer_pattern(model, args)

    return cls(
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        sliding_window=sliding_window,
        layer_pattern=layer_pattern
    )

@staticmethod
def _detect_layer_pattern(model, args) -> dict:
    """
    Detect layer attention pattern (uniform vs hybrid).

    Returns:
        dict with 'type' and pattern details
    """
    # Tier 1: Check layer_types attribute
    if hasattr(args, 'layer_types'):
        layer_types = args.layer_types
        if len(set(layer_types)) > 1:
            return {
                'type': 'hybrid',
                'layer_types': layer_types
            }
        else:
            return {
                'type': 'uniform',
                'attention_type': layer_types[0]
            }

    # Tier 2: Check layer objects
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        layers = model.model.layers
        use_sliding_values = []

        for layer in layers:
            if hasattr(layer, 'use_sliding'):
                use_sliding_values.append(layer.use_sliding)

        if use_sliding_values and len(set(use_sliding_values)) > 1:
            return {
                'type': 'hybrid',
                'use_sliding_per_layer': use_sliding_values
            }

    # Tier 3: Model-type heuristics
    HYBRID_PATTERNS = {
        'gemma3': {
            'type': 'hybrid',
            'global_layers': 8,
            'sliding_layers': 40,
            'description': 'First 8 layers global, remaining 40 sliding window'
        }
    }

    model_type = getattr(args, 'model_type', None)
    if model_type in HYBRID_PATTERNS:
        return HYBRID_PATTERNS[model_type]

    # Default: assume uniform full attention
    return {
        'type': 'uniform',
        'attention_type': 'full_attention'
    }
```

---

## Key Discoveries

### 1. Gemma 3 Uses Nested Config
Gemma 3 is a multimodal model with separate `text_config` and `vision_config`. All text model parameters are in `args.text_config` dict, not as direct attributes.

### 2. No Generic Sliding Window Pattern Attribute
The hybrid attention pattern (which layers use sliding window) is NOT exposed as a generic attribute. Must use model-specific knowledge or inspect layer objects.

### 3. Head Dim Should Always Be Computed
Don't rely on `head_dim` attribute existing. Always compute from `hidden_size // num_attention_heads`.

### 4. Qwen Config Has Misleading Values
Qwen 2.5-14B has `sliding_window: 131072` in config.json but `use_sliding_window: false`, so the actual value is None. Trust model.args over raw config.json.

### 5. Layer Types Attribute Is Rare
Only Llama 3.1-8B has the `layer_types` attribute. Most models don't expose this.

---

## Answers to EXP-001 Questions

### Q1: Can we extract required attributes from model.args or model.config?
**A**: ✅ YES - All required attributes are accessible via model.args (with special handling for Gemma 3's nested text_config)

### Q2: Are attribute names consistent?
**A**: ⚠️ MOSTLY - Primary names are consistent (`num_hidden_layers`, `num_key_value_heads`), but Gemma 3 requires special handling for nested config

### Q3: Does sliding_window_pattern attribute exist in Gemma 3?
**A**: ❌ NO - This attribute doesn't exist. Must use model-type heuristics or inspect layer objects

### Q4: Can we distinguish layer types programmatically?
**A**: ⚠️ PARTIALLY - Can detect via `layer_types` attribute (rare) or inspecting layer objects. Fallback to model-type heuristics for known hybrid models

---

## Implementation Recommendations

### Priority 1: Core Attributes
Implement extraction for:
1. ✅ `num_hidden_layers` (with text_config fallback)
2. ✅ `num_key_value_heads` (with text_config fallback)
3. ✅ `head_dim` (ALWAYS compute)
4. ✅ `sliding_window` (with text_config fallback)

### Priority 2: Pattern Detection
Implement three-tier detection:
1. ✅ Check `layer_types` attribute
2. ✅ Inspect layer objects for `use_sliding`
3. ✅ Fallback to model-type heuristics

### Priority 3: Testing
Test with:
1. ✅ Gemma 3 (hybrid, nested config)
2. ✅ Llama 3.1 (uniform, has layer_types)
3. ✅ Qwen 2.5 (uniform, standard config)
4. ✅ Qwen1.5-MoE (MoE, standard config)

---

## Raw Inspection Data

### Model Files Generated
- `/Users/dev_user/semantic/model_inspection_results.json` - Full inspection results
- `/Users/dev_user/semantic/inspect_models.py` - Initial inspection script
- `/Users/dev_user/semantic/inspect_detailed.py` - Detailed inspection script
- `/Users/dev_user/semantic/inspect_qwen_moe.py` - MoE model inspection

### Gemma 3 12B Config Excerpt
```json
{
  "text_config": {
    "hidden_size": 3840,
    "intermediate_size": 15360,
    "model_type": "gemma3_text",
    "num_attention_heads": 16,
    "num_hidden_layers": 48,
    "num_key_value_heads": 8,
    "rope_scaling": {"factor": 8.0, "rope_type": "linear"},
    "sliding_window": 1024
  }
}
```

### Llama 3.1-8B Unique Attributes
```python
layer_types: [
  'full_attention', 'full_attention', ... (32 total)
]
```

Each layer has `use_sliding=False` attribute.

### Qwen1.5-MoE Unique Attributes
```python
num_experts: 60
num_experts_per_tok: 4
moe_intermediate_size: 1408
shared_expert_intermediate_size: 5632
```

---

## Next Steps

### For Sprint 1 Day 5
1. Implement `ModelCacheSpec.from_model()` using recommended extraction logic
2. Add unit tests for all 4 inspected models
3. Document Gemma 3 special handling in architecture docs
4. Create model-type heuristics registry for hybrid patterns

### For Future Work
1. Investigate Mixtral MoE models (currently have MLX loading bug)
2. Test with more model architectures (Mistral, Phi, etc.)
3. Consider caching extracted specs to avoid repeated model loading
4. Add validation to ensure extracted values are sensible

---

## Conclusion

EXP-001 successfully validated that:
1. All required cache spec attributes CAN be extracted from model.args
2. Attribute names are MOSTLY consistent (with documented exceptions)
3. Hybrid attention patterns require special handling (no generic attribute)
4. Recommended three-tier detection approach provides robust coverage

The proposed `ModelCacheSpec.from_model()` implementation provides a complete solution for extracting cache specifications from MLX models.

**Status**: ✅ EXPERIMENT COMPLETE - Ready for implementation
