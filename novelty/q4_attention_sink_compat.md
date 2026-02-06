# Q4 KV Cache with Attention Sinks: Dequantize Fallback

## Executive Summary

Models that use **attention sinks** (e.g., GPT-OSS-20B) are incompatible with MLX's quantized SDPA kernel, which raises `ValueError: Quantized SDPA does not support attention sinks`. We implemented a runtime monkey-patch that preserves Q4 KV cache **storage** while falling back to FP16 SDPA for the **compute** path when sinks are detected.

**Result**: GPT-OSS-20B runs with Q4 KV cache end-to-end. Multi-turn cache reuse achieves **2.2x speedup**, matching our other Q4 models.

## The Problem

### Attention Sinks

Attention sinks are a technique where certain token positions (typically start-of-sequence) receive artificially high attention weights to prevent catastrophic forgetting during long-context inference. GPT-OSS implements this with a learnable per-head sink vector:

```python
# mlx_lm/models/gpt_oss.py
class AttentionBlock(nn.Module):
    def __init__(self, config):
        self.sinks = mx.zeros((config.num_attention_heads,))

    def __call__(self, x, mask, cache):
        # ...
        v_hat = scaled_dot_product_attention(
            q, k, v, cache, self.sm_scale, mask=mask, sinks=self.sinks
        )
```

### The MLX Limitation

MLX provides two SDPA paths:

1. **FP16 path** (`mx.fast.scaled_dot_product_attention`) — supports `sinks` parameter
2. **Quantized path** (`quantized_scaled_dot_product_attention`) — uses `mx.quantized_matmul`, does NOT support `sinks`

The dispatcher in `mlx_lm/models/base.py` guards against the unsupported combination:

```python
def scaled_dot_product_attention(queries, keys, values, cache, scale, mask, sinks=None):
    if hasattr(cache, "bits"):          # Quantized cache detected
        if sinks is not None:           # But model uses sinks
            raise ValueError("Quantized SDPA does not support attention sinks.")
        return quantized_scaled_dot_product_attention(...)
    else:
        return mx.fast.scaled_dot_product_attention(..., sinks=sinks)
```

This is a **Python-level guard**, not a Metal kernel limitation. The quantized kernel (`mx.quantized_matmul`) could compute attention scores just fine — the issue is that sink reweighting logic isn't wired into the quantized path.

## The Solution: Dequantize Fallback

We monkey-patch `scaled_dot_product_attention` to add a third code path:

```
Q4 cache + no sinks  →  quantized SDPA (fast, Q4 compute)
Q4 cache + sinks     →  dequantize Q4 → FP16 SDPA with sinks  ← NEW
FP16 cache           →  standard FP16 SDPA
```

When sinks are present with a quantized cache:
1. Dequantize the Q4 key/value tuples back to FP16 arrays using `mx.dequantize`
2. Call `mx.fast.scaled_dot_product_attention` with FP16 arrays and sinks
3. The dequantized tensors are transient — freed after the attention computation

### Implementation

```python
# src/semantic/adapters/outbound/mlx_sink_compat.py

def _patched_sdpa(queries, keys, values, cache, scale, mask, sinks=None):
    if hasattr(cache, "bits"):
        if sinks is not None:
            # Dequantize Q4 tuples → FP16 for sink-aware attention
            k_fp = mx.dequantize(keys[0], scales=keys[1], biases=keys[2],
                                 group_size=cache.group_size, bits=cache.bits)
            v_fp = mx.dequantize(values[0], scales=values[1], biases=values[2],
                                 group_size=cache.group_size, bits=cache.bits)
            return mx.fast.scaled_dot_product_attention(
                queries, k_fp, v_fp, scale=scale, mask=mask, sinks=sinks)
        return quantized_scaled_dot_product_attention(
            queries, keys, values, scale=scale, mask=mask,
            group_size=cache.group_size, bits=cache.bits)
    return mx.fast.scaled_dot_product_attention(
        queries, keys, values, scale=scale, mask=mask, sinks=sinks)

# Applied on module import
mlx_lm.models.base.scaled_dot_product_attention = _patched_sdpa
```

## Memory Trade-Off

| Aspect | Q4 Only | Q4 + Sinks (Fallback) | Full FP16 |
|--------|---------|----------------------|-----------|
| **KV Storage** | 25% of FP16 | 25% of FP16 | 100% |
| **During Attention** | 25% (quantized matmul) | ~125% transient* | 100% |
| **Net Savings** | 75% | ~70% | 0% |

*Transient: dequantized FP16 tensors exist only during one attention computation, then freed. The Q4 storage remains.

The key insight: **storage dominates**. For a 32K token context, the KV cache is stored the entire time in Q4 (75% savings). The FP16 dequantization only exists during one forward pass computation (~milliseconds), then the temporary is freed.

## Benchmark Results: GPT-OSS-20B (MXFP4-Q4)

**Configuration**: Apple Silicon M4 Pro, 24GB, MLX 0.30.5, Q4 KV (group_size=64)

### Cold Start

| Context | TTFT | E2E |
|---------|------|-----|
| 200 tokens | 1,280ms | 1,281ms |
| 2,000 tokens | 3,102ms | 3,103ms |

### Multi-Turn Cache Reuse

| Turn | E2E | TPS | Speedup |
|------|-----|-----|---------|
| Turn 1 (cold) | 3,117ms | 20.5 | — |
| Turn 2 (warm) | 1,437ms | 44.5 | **2.2x** |
| Turn 3 (warm) | 1,421ms | 45.0 | **2.2x** |

### Thinking vs Non-Thinking

| Mode | Output Tokens | E2E | TPS |
|------|--------------|-----|-----|
| Thinking (CoT) | 256 | 3,635ms | 70.4 |
| Non-thinking | 64 | 1,083ms | 59.1 |

### Memory

- Model: ~10.6 GB
- Block pool: 82,850 blocks at 0.1 MB/block (Q4)
- Active 21B params, ~3.6B active per token (Sparse MoE)

## Why Not Other Approaches?

| Approach | Verdict | Reason |
|----------|---------|--------|
| Remove the check entirely | **Dangerous** | Sinks silently ignored → degraded quality |
| Hybrid FP16 sinks + Q4 rest | Complex | Requires splitting KV per-head, custom merge |
| Rewrite quantized SDPA with sinks | Over-engineered | Duplicates Metal kernel work, maintenance burden |
| Disable sinks for Q4 models | Wrong trade-off | Sinks provide real long-context benefits |
| **Dequantize fallback** | **Chosen** | Safe, simple, correct, ~20 lines |

## Relationship to Q4 Direct Injection

This technique complements the [Q4 Direct Injection](q4_direct_injection.md) pipeline:

1. **Q4 Direct Injection**: Stores and loads KV cache in Q4 format end-to-end, bypassing mlx_lm's default FP16 cache
2. **Q4 Sink Compat** (this doc): Enables Q4 storage for models with attention sinks by dequantizing transiently during attention compute

Together, they enable Q4 KV caching for **all** model architectures — with or without attention sinks.

## Files

- `src/semantic/adapters/outbound/mlx_sink_compat.py` — monkey-patch module
- `src/semantic/entrypoints/api_server.py` — import point (applied before first inference)
- `config/models/gpt-oss-20b-mxfp4.toml` — model profile
