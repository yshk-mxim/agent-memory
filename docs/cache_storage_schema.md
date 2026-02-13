# Cache Storage Schema (Safetensors Format)

## Overview

Agent caches are persisted to disk in [safetensors](https://github.com/huggingface/safetensors) format with model-tagged metadata for compatibility validation.

## File Format

Each agent's cache is stored as a single `.safetensors` file:

```
~/.agent_memory/caches/<agent_id>.safetensors
```

## Metadata Fields

Stored in safetensors metadata header (JSON):

```json
{
  "__metadata__": {
    "agent_id": "agent_123",
    "model_id": "mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx",
    "model_tag": {
      "n_layers": 48,
      "n_kv_heads": 8,
      "head_dim": 256,
      "block_tokens": 256
    },
    "total_tokens": 1024,
    "created_at": "2026-01-29T10:30:00Z",
    "version": "1.0"
  }
}
```

## Tensor Layout

KV cache tensors are stored as flattened arrays per layer:

### For each layer (0..n_layers-1):

```
k_layer_{layer_id}:  shape=[n_kv_heads, head_dim, total_tokens], dtype=float16
v_layer_{layer_id}:  shape=[n_kv_heads, head_dim, total_tokens], dtype=float16
```

### Example for 12-layer model with 1024 tokens:

```
k_layer_0:  [4, 64, 1024] = 262,144 values
v_layer_0:  [4, 64, 1024] = 262,144 values
k_layer_1:  [4, 64, 1024] = 262,144 values
v_layer_1:  [4, 64, 1024] = 262,144 values
...
k_layer_11: [4, 64, 1024] = 262,144 values
v_layer_11: [4, 64, 1024] = 262,144 values
```

Total tensors: 24 (12 layers × 2 tensors per layer)

## Block Reconstruction

When loading from disk:

1. **Validate metadata**: Check model_tag compatibility
2. **Load tensors**: Read all k/v tensors for all layers
3. **Split into blocks**: Divide sequence dimension by block_tokens (256)
4. **Allocate blocks**: Get blocks from BlockPool
5. **Populate blocks**: Copy tensor slices into block.layer_data

## Atomic Write Protocol

To prevent corruption on crash:

1. Write to temporary file: `<agent_id>.safetensors.tmp`
2. Flush and fsync
3. Atomic rename: `mv .tmp .safetensors`

## Compatibility Validation

Before loading cache:

```python
def validate_compatibility(saved_tag: ModelTag, current_spec: ModelCacheSpec) -> bool:
    return (
        saved_tag.n_layers == current_spec.n_layers and
        saved_tag.n_kv_heads == current_spec.n_kv_heads and
        saved_tag.head_dim == current_spec.head_dim and
        saved_tag.block_tokens == current_spec.block_tokens
    )
```

If incompatible → reject cache (treat as cache miss)

## File Size Estimation

For a cache with N tokens:

```
size_bytes = n_layers × 2 (k+v) × n_kv_heads × head_dim × N × sizeof(float16)

Example (12 layers, 4 heads, 64 dim, 1024 tokens):
= 12 × 2 × 4 × 64 × 1024 × 2 bytes
= 12,582,912 bytes
= ~12 MB
```

## Implementation Notes

- Use `safetensors.numpy.save_file()` for writing
- Use `safetensors.numpy.load_file()` for reading
- Metadata is automatically JSON-encoded by safetensors
- All tensors must be contiguous numpy arrays
- FP16 dtype recommended for space efficiency
