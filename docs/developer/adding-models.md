# Adding Support for a New Model

This guide covers the steps to add a new model to the agent-memory server. The system currently supports Gemma 3 12B and DeepSeek-Coder-V2-Lite as primary models.

## 1. Create a TOML Config in config/models/

Each supported model has a TOML profile in `config/models/`. This profile is used for profiling and benchmarking, and provides optimal runtime parameters.

Create a new file following the naming convention: `<model-slug>.toml`. The slug is derived from the last path component of the HuggingFace model ID, lowercased.

### Required Sections

Use the existing configs as reference:
- `config/models/gemma-3-12b-it-4bit.toml` (hybrid attention, GQA)
- `config/models/deepseek-coder-v2-lite-4bit.toml` (MLA, asymmetric K/V)

```toml
[model]
model_id = "mlx-community/your-model-id"
n_layers = 32          # Total transformer layers
n_kv_heads = 8         # Number of key-value attention heads
head_dim = 128         # Dimension per attention head

[inference]
temperature = 0.7
top_p = 0.95
top_k = 50
repetition_penalty = 1.0
extended_thinking = false
supports_thinking = false

[optimal]
max_batch_size = 2
prefill_step_size = 256
kv_bits = 4
kv_group_size = 64
chunked_prefill_enabled = true
chunked_prefill_threshold = 2048
chunked_prefill_min_chunk = 512
chunked_prefill_max_chunk = 2048
batch_window_ms = 10
scheduler_enabled = true
max_agents_in_memory = 5
evict_to_disk = true

[thresholds]
long_context_threshold = 4000
high_batch_threshold = 3
memory_pressure_mb = 14000        # Adjust for model size
min_cache_benefit_ratio = 0.8
```

The `[benchmark]` section is optional and populated after profiling.

### Finding Model Parameters

Pull the values from the model's `config.json` on HuggingFace:

- `n_layers`: `num_hidden_layers`
- `n_kv_heads`: `num_key_value_heads`
- `head_dim`: Usually `hidden_size / num_attention_heads`, but check the model's MLX implementation (e.g., Gemma 3 uses a dataclass default of 256, not the computed ratio)
- `sliding_window`: If the model uses hybrid attention

For models with nested configs (like Gemma 3 with `text_config`), the spec extractor handles this automatically.

## 2. Verify the Spec Extractor Detects the Architecture

The spec extractor (`src/agent_memory/adapters/outbound/mlx_spec_extractor.py`) automatically detects model geometry from a loaded MLX model. It handles:

### Attention Architecture Detection

The extractor uses a three-tier approach for layer type detection:

1. **Tier 1**: Check `model.args.layer_types` attribute (if the MLX model exposes it)
2. **Tier 2**: Inspect layer objects for `use_sliding` attribute
3. **Tier 3**: Strategy-based detection (Gemma 3 hybrid pattern, or uniform global as fallback)

### Head Dimension Extraction

The extractor tries multiple paths to find attention modules:

1. `model.language_model.model.layers` (Gemma 3 nested architecture)
2. `model.model.layers` (standard models)
3. `model.layers` (direct layer access)

For each path, it checks:
- **DeepSeek V2 MLA**: Looks for `qk_nope_head_dim`, `qk_rope_head_dim`, and `v_head_dim` on the attention module. If found, K dim = `qk_nope_head_dim + qk_rope_head_dim`, V dim = `v_head_dim`.
- **Standard models**: Uses `attn.head_dim` for symmetric K=V.
- **Fallback**: Computes `hidden_size // num_attention_heads`.

### When You Need to Modify the Extractor

Add a new `LayerTypeDetectionStrategy` subclass if the model has a non-standard layer pattern (e.g., alternating attention types, partial sliding window). Register it in `MLXModelSpecExtractor.__init__()`:

```python
class YourModelDetectionStrategy(LayerTypeDetectionStrategy):
    def detect_layer_types(self, _model, args, n_layers):
        model_type = getattr(args, "model_type", "unknown")
        if model_type == "your_model_type":
            # Return list of "global" / "sliding_window" per layer
            return ["global"] * n_layers
        return None
```

Add it to `self._strategies` **before** the `UniformAttentionDetectionStrategy` (which is the catch-all fallback).

### ModelCacheSpec Fields

The extractor produces a `ModelCacheSpec` (defined in `src/agent_memory/domain/value_objects.py`):

| Field | Type | Description |
|---|---|---|
| `n_layers` | `int` | Total transformer layers |
| `n_kv_heads` | `int` | KV attention heads per layer |
| `head_dim` | `int` | K head dimension |
| `block_tokens` | `int` | Tokens per cache block (from `BLOCK_SIZE_TOKENS`) |
| `layer_types` | `list[str]` | Per-layer type: `"global"` or `"sliding_window"` |
| `sliding_window_size` | `int \| None` | Window size for sliding window layers |
| `kv_bits` | `int \| None` | Quantization bits (4, 8, or None for FP16) |
| `kv_group_size` | `int` | Quantization group size |
| `v_head_dim` | `int \| None` | V head dim when asymmetric (e.g., DeepSeek MLA: 128 vs K=192). None means K=V. |

## 3. Check Chat Template Compatibility

The chat template adapter (`src/agent_memory/adapters/outbound/chat_template_adapter.py`) and adapter helpers (`src/agent_memory/adapters/inbound/adapter_helpers.py`) handle model-specific template formatting.

Things to verify:
- Does the model's chat template close assistant messages with an EOS token? (DeepSeek does this, requiring the `generation_prefix` workaround)
- Does `add_generation_prompt=True` work correctly with the tokenizer?
- Are there special tokens that need handling?

The coordination service (`src/agent_memory/application/coordination_service.py`) hardcodes temperature at 0.3 (line 617). If your model needs a different temperature, you will need to modify this or make it configurable per-model.

## 4. Test with Unit Tests

Run the full unit test suite to catch any regressions:

```bash
python -m pytest tests/unit -x -q --timeout=30
```

If you added a new detection strategy, add unit tests that mock the model's `args` and layer structure. The spec extractor tests should cover:

- Correct `n_layers`, `n_kv_heads`, `head_dim` extraction
- Correct `layer_types` detection
- Asymmetric K/V detection (if applicable)
- Fallback behavior when attributes are missing

## 5. Test with a Real Inference Run

Start the server with your model and verify end-to-end:

```bash
SEMANTIC_MLX_MODEL_ID="mlx-community/your-model-id" \
SEMANTIC_MLX_CACHE_BUDGET_MB=4096 \
python -m agent_memory.entrypoints.cli serve --port 8000
```

Wait for the readiness check:

```bash
curl -sf http://localhost:8000/health/ready
```

Send a test request:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "your-model-id",
    "messages": [{"role": "user", "content": "Hello, what model are you?"}],
    "max_tokens": 64
  }'
```

Verify:
- Model loads without errors
- Spec extractor logs show correct geometry (check for `ModelCacheSpec` in debug logs)
- First response generates correctly (cold path)
- Second response with the same agent_id uses cache (warm/hot path)
- Graceful shutdown completes all 6 stages (see [debugging.md](debugging.md))

## 6. Memory Budget Considerations

Different architectures have different memory profiles:

- **Standard models**: Default `SEMANTIC_MLX_CACHE_BUDGET_MB=8192` usually works
- **MoE models** (like DeepSeek): Need extra headroom for intermediate activations. Use `SEMANTIC_MLX_CACHE_BUDGET_MB=4096` to leave room.
- **Large models**: Adjust `memory_pressure_mb` in the TOML `[thresholds]` section based on your model's weight size

The Q4 KV cache memory per block per layer is approximately:

```
bytes = n_kv_heads * head_dim * block_tokens * 2 (K+V) * 0.5 (4-bit)
      + scales/biases overhead
```

Set `max_agents_in_memory` in the TOML config based on how many concurrent agent caches fit within the budget.
