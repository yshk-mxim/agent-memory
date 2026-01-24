# Model Comparison Table - EXP-001 Results

Quick reference table comparing all 4 inspected models.

## Cache Spec Attributes

| Model | Layers | KV Heads | Attn Heads | Head Dim | Hidden Size | Sliding Window |
|-------|--------|----------|------------|----------|-------------|----------------|
| **Gemma 3 12B** | 48 | 8 | 16 | 240 | 3840 | 1024 |
| **Qwen1.5-MoE-A2.7B** | 24 | 16 | 16 | 128 | 2048 | None |
| **Qwen 2.5-14B** | 48 | 8 | 40 | 128 | 5120 | None |
| **Llama 3.1-8B** | 32 | 8 | 32 | 128 | 4096 | None |

## Attention Patterns

| Model | Attention Type | Pattern Details | Detection Method |
|-------|----------------|-----------------|------------------|
| **Gemma 3 12B** | Hybrid SWA+Global | 8 global + 40 sliding window (1024 tokens) | Model-type heuristic |
| **Qwen1.5-MoE-A2.7B** | Uniform Full | All layers full attention | sliding_window=None |
| **Qwen 2.5-14B** | Uniform Full | All layers full attention | sliding_window=None |
| **Llama 3.1-8B** | Uniform Full | All layers full attention | layer_types + use_sliding |

## Special Attributes

| Model | Unique Attributes | Notes |
|-------|------------------|-------|
| **Gemma 3 12B** | `text_config` (nested), `vision_config` | Multimodal model, all text params in text_config |
| **Qwen1.5-MoE-A2.7B** | `num_experts=60`, `num_experts_per_tok=4` | MoE architecture, 60 experts, activates 4 per token |
| **Qwen 2.5-14B** | Standard | Config has sliding_window=131072 but disabled |
| **Llama 3.1-8B** | `layer_types` list, `use_sliding` per layer | Only model with explicit layer_types attribute |

## Config Access Patterns

| Model | num_layers | num_kv_heads | head_dim | sliding_window |
|-------|-----------|--------------|----------|----------------|
| **Gemma 3 12B** | `args.text_config['num_hidden_layers']` | `args.text_config['num_key_value_heads']` | COMPUTE | `args.text_config['sliding_window']` |
| **Qwen1.5-MoE-A2.7B** | `args.num_hidden_layers` | `args.num_key_value_heads` | COMPUTE | `args.sliding_window` |
| **Qwen 2.5-14B** | `args.num_hidden_layers` | `args.num_key_value_heads` | COMPUTE | `args.sliding_window` |
| **Llama 3.1-8B** | `args.num_hidden_layers` | `args.num_key_value_heads` | COMPUTE | `args.sliding_window` |

**COMPUTE** = `hidden_size // num_attention_heads`

## Model Types

| Model | model_type | Model Class | Layer Type |
|-------|-----------|-------------|------------|
| **Gemma 3 12B** | `gemma3` | Model | (nested) |
| **Qwen1.5-MoE-A2.7B** | `qwen2_moe` | Model | Qwen2MoeDecoderLayer |
| **Qwen 2.5-14B** | `qwen2` | Model | TransformerBlock |
| **Llama 3.1-8B** | `llama` | Model | TransformerBlock |

## Implementation Priority

Based on complexity and special handling needed:

1. **Easy**: Llama 3.1-8B, Qwen 2.5-14B (standard attributes)
2. **Medium**: Qwen1.5-MoE-A2.7B (MoE but standard cache)
3. **Hard**: Gemma 3 12B (nested config + hybrid attention)

## Cache Size Estimation (per layer, per batch item, per token)

Formula: `num_kv_heads * head_dim * 2 (K+V) * bytes_per_element`

Assuming float16 (2 bytes):

| Model | KV Heads | Head Dim | Cache per Token | Cache for 1024 Tokens |
|-------|----------|----------|-----------------|----------------------|
| **Gemma 3 12B** | 8 | 240 | 7,680 bytes | 7.5 MB |
| **Qwen1.5-MoE-A2.7B** | 16 | 128 | 8,192 bytes | 8 MB |
| **Qwen 2.5-14B** | 8 | 128 | 4,096 bytes | 4 MB |
| **Llama 3.1-8B** | 8 | 128 | 4,096 bytes | 4 MB |

Multiply by `num_layers` for total cache size per sequence.

## Validation Checklist

For each model, verify extraction of:

- [x] **Gemma 3 12B**: Nested text_config, head_dim=240, sliding_window=1024, hybrid pattern
- [x] **Qwen1.5-MoE-A2.7B**: MoE attributes, head_dim=128, no sliding window
- [x] **Qwen 2.5-14B**: Standard config, head_dim=128, no sliding window
- [x] **Llama 3.1-8B**: layer_types attribute, head_dim=128, use_sliding=False

---

**Generated**: 2026-01-24 from EXP-001 inspection results
**Source**: `/Users/dev_user/semantic/project/experiments/EXP-001-model-args.md`
