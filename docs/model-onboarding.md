# Model Onboarding Guide

Guide for adding and supporting new MLX model architectures in Semantic Caching API.

## Table of Contents

- [Supported Models](#supported-models)
- [Adding New Models](#adding-new-models)
- [ModelCacheSpec Extraction](#modelcachespec-extraction)
- [Testing New Models](#testing-new-models)
- [Performance Tuning](#performance-tuning)
- [Troubleshooting](#troubleshooting)

## Supported Models

Semantic Caching API currently supports the following MLX models:

### Production Models

#### DeepSeek-Coder-V2-Lite (Default)

**Model ID**: `mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx`

**Specifications**:
- Architecture: DeepSeek-Coder-V2-Lite (Google DeepMind)
- Parameters: 12B (4-bit quantized)
- Context: 8K tokens
- Memory: ~6GB quantized
- Use Case: Production inference with tool calling

**Configuration**:
```bash
SEMANTIC_MLX_MODEL_ID=mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx
SEMANTIC_MLX_CACHE_BUDGET_MB=4096
SEMANTIC_MLX_MAX_BATCH_SIZE=5
```

**Architecture Details**:
- Layers: 42
- KV Heads: 16
- Head Dimension: 256
- Block Size: 256 tokens
- Supports: Tool calling, streaming, multi-turn

### Testing Models

#### SmolLM2-135M-Instruct

**Model ID**: `mlx-community/SmolLM2-135M-Instruct`

**Specifications**:
- Architecture: SmolLM2 (Hugging Face)
- Parameters: 135M
- Context: 2K tokens
- Memory: ~1GB
- Use Case: Testing, development, CI/CD

**Configuration**:
```bash
SEMANTIC_MLX_MODEL_ID=mlx-community/SmolLM2-135M-Instruct
SEMANTIC_MLX_CACHE_BUDGET_MB=1024
SEMANTIC_MLX_MAX_BATCH_SIZE=2
```

**Architecture Details**:
- Layers: 30
- KV Heads: 9
- Head Dimension: 64
- Block Size: 256 tokens
- Supports: Basic inference, streaming

## Adding New Models

### Prerequisites

1. **MLX Compatibility**: Model must be available in MLX format
2. **Hugging Face Hub**: Model should be on Hugging Face Hub
3. **Architecture Support**: Model architecture must be compatible with MLX-LM

### Step 1: Identify Model

Find an MLX-compatible model:

```bash
# Search MLX community models
# https://huggingface.co/mlx-community

# Common architectures:
# - Gemma 2/3
# - Llama 2/3
# - Mistral
# - Qwen
# - DeepSeek
```

### Step 2: Test Model Loading

Verify the model loads in MLX:

```python
from mlx_lm import load

model_id = "mlx-community/your-model-name"
model, tokenizer = load(model_id)

print(f"Model loaded: {model}")
print(f"Config: {model.config}")
```

### Step 3: Extract ModelCacheSpec

The system automatically extracts `ModelCacheSpec` from the model config:

```python
from agent_memory.domain.model_cache_spec import ModelCacheSpec

# Automatic extraction
spec = ModelCacheSpec.from_model(model, block_size=256)

print(f"Layers: {spec.num_layers}")
print(f"KV Heads: {spec.num_kv_heads}")
print(f"Head Dim: {spec.head_dim}")
print(f"Block Size: {spec.block_size}")
print(f"Bytes per block: {spec.bytes_per_block_per_layer()}")
```

**Required Config Fields**:
- `num_hidden_layers` or `num_layers` - Number of transformer layers
- `num_key_value_heads` or `num_attention_heads` - KV cache heads
- `head_dim` or `hidden_size / num_attention_heads` - Head dimension

### Step 4: Configure Cache Budget

Calculate appropriate cache budget:

```python
# Example: DeepSeek-Coder-V2-Lite with 256-token blocks
bytes_per_block = spec.bytes_per_block_per_layer()
# ~6MB per block for DeepSeek-Coder-V2-Lite

# 4GB cache budget
cache_budget_mb = 4096
total_blocks = (cache_budget_mb * 1024 * 1024) // bytes_per_block
# ~700 blocks for DeepSeek-Coder-V2-Lite

# Total cacheable tokens
total_tokens = total_blocks * spec.block_size
# ~179,200 tokens
```

### Step 5: Add to Configuration

Update configuration with new model:

```bash
# .env
SEMANTIC_MLX_MODEL_ID=mlx-community/your-model-name
SEMANTIC_MLX_CACHE_BUDGET_MB=4096
SEMANTIC_MLX_MAX_BATCH_SIZE=5
SEMANTIC_MLX_DEFAULT_MAX_TOKENS=256
```

### Step 6: Start Server

Test the server with the new model:

```bash
semantic serve --model mlx-community/your-model-name --log-level DEBUG
```

## ModelCacheSpec Extraction

`ModelCacheSpec` is automatically extracted from model configuration during initialization.

### Architecture Patterns

#### Standard Transformer (Gemma, Llama)

```python
config = {
    "num_hidden_layers": 42,
    "num_key_value_heads": 16,
    "head_dim": 256,
}

spec = ModelCacheSpec(
    num_layers=42,
    num_kv_heads=16,
    head_dim=256,
    block_size=256,  # Configurable
    dtype_bytes=2,    # FP16 (or 1 for quantized)
)
```

#### Grouped-Query Attention (GQA)

Models like Llama 2/3 use fewer KV heads than query heads:

```python
config = {
    "num_hidden_layers": 32,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,  # GQA: 8 KV heads, 32 query heads
    "head_dim": 128,
}
```

#### Multi-Query Attention (MQA)

Some models use single KV head:

```python
config = {
    "num_hidden_layers": 24,
    "num_attention_heads": 16,
    "num_key_value_heads": 1,  # MQA: 1 KV head shared
    "head_dim": 64,
}
```

### Extraction Logic

The system tries multiple config key patterns:

```python
def extract_spec(model_config: dict) -> ModelCacheSpec:
    # Try different key names
    num_layers = (
        model_config.get("num_hidden_layers") or
        model_config.get("num_layers") or
        model_config.get("n_layers")
    )

    num_kv_heads = (
        model_config.get("num_key_value_heads") or
        model_config.get("num_attention_heads")
    )

    head_dim = (
        model_config.get("head_dim") or
        model_config.get("hidden_size") // model_config.get("num_attention_heads")
    )

    return ModelCacheSpec(
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        block_size=256,  # From config
        dtype_bytes=2,    # FP16 default
    )
```

### Validation

The system validates extracted specs:

```python
def validate_spec(spec: ModelCacheSpec) -> None:
    assert spec.num_layers > 0, "num_layers must be positive"
    assert spec.num_kv_heads > 0, "num_kv_heads must be positive"
    assert spec.head_dim > 0, "head_dim must be positive"
    assert spec.block_size > 0, "block_size must be positive"
    assert spec.block_size % 64 == 0, "block_size should be multiple of 64"
```

## Testing New Models

### Unit Tests

Create model-specific unit tests:

```python
"""Unit tests for NewModel support."""

import pytest
from agent_memory.domain.model_cache_spec import ModelCacheSpec


@pytest.mark.unit
def test_newmodel_spec_extraction():
    """Test ModelCacheSpec extraction for NewModel."""
    config = {
        "num_hidden_layers": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
    }

    spec = ModelCacheSpec.from_config(config, block_size=256)

    assert spec.num_layers == 32
    assert spec.num_kv_heads == 8
    assert spec.head_dim == 128
    assert spec.block_size == 256

    # Calculate memory requirements
    bytes_per_block = spec.bytes_per_block_per_layer()
    assert bytes_per_block > 0
```

### Integration Tests

Create integration tests for all API endpoints:

```python
"""Integration tests for NewModel."""

import pytest
from fastapi.testclient import TestClient

from agent_memory.entrypoints.api_server import create_app


@pytest.mark.integration
class TestNewModelAnthropicAPI:
    """Test NewModel with Anthropic Messages API."""

    def test_newmodel_anthropic_api(self):
        """NewModel should work with Anthropic API."""
        app = create_app()

        with TestClient(app) as client:
            response = client.post(
                "/v1/messages",
                json={
                    "model": "your-model-name",
                    "max_tokens": 50,
                    "messages": [{"role": "user", "content": "Hello!"}],
                },
            )

            assert response.status_code == 200
            data = response.json()
            assert "content" in data
            assert len(data["content"]) > 0


@pytest.mark.integration
class TestNewModelOpenAIAPI:
    """Test NewModel with OpenAI Chat Completions API."""

    def test_newmodel_openai_api(self):
        """NewModel should work with OpenAI API."""
        app = create_app()

        with TestClient(app) as client:
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "your-model-name",
                    "messages": [{"role": "user", "content": "Hello!"}],
                    "max_tokens": 50,
                },
            )

            assert response.status_code == 200
            data = response.json()
            assert "choices" in data
            assert len(data["choices"]) > 0


@pytest.mark.integration
class TestNewModelCachePersistence:
    """Test NewModel cache persistence."""

    def test_newmodel_cache_grows(self):
        """Cache should persist across requests."""
        app = create_app()

        with TestClient(app) as client:
            # First request
            response1 = client.post(
                "/v1/messages",
                json={
                    "model": "your-model-name",
                    "max_tokens": 30,
                    "messages": [{"role": "user", "content": "First"}],
                },
            )

            assert response1.status_code == 200
            data1 = response1.json()

            # Second request (should hit cache)
            response2 = client.post(
                "/v1/messages",
                json={
                    "model": "your-model-name",
                    "max_tokens": 30,
                    "messages": [
                        {"role": "user", "content": "First"},
                        {"role": "assistant", "content": data1["content"][0]["text"]},
                        {"role": "user", "content": "Second"},
                    ],
                },
            )

            assert response2.status_code == 200
            data2 = response2.json()

            # Verify cache was used
            assert data2["usage"]["cache_read_input_tokens"] > 0
```

### Running Tests

```bash
# Run unit tests
pytest tests/unit/test_newmodel.py -v

# Run integration tests
pytest tests/integration/test_newmodel.py -v

# Run all model tests
pytest tests/integration/test_*_model.py -v
```

## Performance Tuning

### Cache Budget Optimization

**Small Models** (< 1B params):
```bash
SEMANTIC_MLX_CACHE_BUDGET_MB=1024  # 1GB
SEMANTIC_MLX_MAX_BATCH_SIZE=2
```

**Medium Models** (1-10B params):
```bash
SEMANTIC_MLX_CACHE_BUDGET_MB=4096  # 4GB
SEMANTIC_MLX_MAX_BATCH_SIZE=5
```

**Large Models** (10-70B params):
```bash
SEMANTIC_MLX_CACHE_BUDGET_MB=8192  # 8GB
SEMANTIC_MLX_MAX_BATCH_SIZE=3
```

### Batch Size Tuning

Balance throughput vs latency:

- **High Throughput**: `MAX_BATCH_SIZE=10` (more concurrent requests)
- **Low Latency**: `MAX_BATCH_SIZE=2` (faster individual responses)
- **Balanced**: `MAX_BATCH_SIZE=5` (default)

### Prefill Step Size

Controls prompt processing speed:

```bash
# Faster prefill (more memory)
SEMANTIC_MLX_PREFILL_STEP_SIZE=1024

# Slower prefill (less memory)
SEMANTIC_MLX_PREFILL_STEP_SIZE=256

# Default
SEMANTIC_MLX_PREFILL_STEP_SIZE=512
```

### KV Cache Quantization

Reduce memory usage with quantized KV cache:

```bash
# FP16 (default, highest quality)
SEMANTIC_MLX_KV_BITS=null

# 8-bit quantization (2x memory savings)
SEMANTIC_MLX_KV_BITS=8

# 4-bit quantization (4x memory savings, lower quality)
SEMANTIC_MLX_KV_BITS=4
```

### Memory Budget Calculation

Calculate optimal settings:

```python
# Target: 16GB total system memory, 8GB for cache

# Model loading: ~6GB (DeepSeek-Coder-V2-Lite)
# Cache budget: 4GB
# System overhead: 2GB
# Remaining: 4GB for other operations

cache_budget_mb = 4096

# Calculate blocks
spec = model_cache_spec  # Extracted from model
bytes_per_block = spec.bytes_per_block_per_layer()
num_blocks = (cache_budget_mb * 1024 * 1024) // bytes_per_block

# Calculate token capacity
total_tokens = num_blocks * spec.block_size

print(f"Cache capacity: {total_tokens:,} tokens")
print(f"Blocks: {num_blocks}")
```

## Troubleshooting

### Issue: Model Fails to Load

**Symptom**: `ModelNotFoundError` or download fails

**Solutions**:
1. Verify model ID is correct (case-sensitive)
2. Check Hugging Face Hub availability
3. Verify internet connection
4. Try manual download:
   ```bash
   huggingface-cli download mlx-community/model-name
   ```

### Issue: ModelCacheSpec Extraction Fails

**Symptom**: `KeyError` or `AttributeError` during startup

**Solutions**:
1. Inspect model config:
   ```python
   from mlx_lm import load
   model, _ = load("model-id")
   print(model.config)
   ```
2. Check for required fields: `num_hidden_layers`, `num_key_value_heads`, `head_dim`
3. Add fallback logic for missing fields
4. Report issue with model config structure

### Issue: Out of Memory During Inference

**Symptom**: `RuntimeError: Out of memory` or crash

**Solutions**:
1. Reduce cache budget:
   ```bash
   SEMANTIC_MLX_CACHE_BUDGET_MB=2048
   ```
2. Reduce batch size:
   ```bash
   SEMANTIC_MLX_MAX_BATCH_SIZE=2
   ```
3. Use KV cache quantization:
   ```bash
   SEMANTIC_MLX_KV_BITS=8
   ```
4. Use smaller model for testing

### Issue: Slow Inference

**Symptom**: Requests take very long

**Solutions**:
1. Verify Metal GPU is available:
   ```python
   import mlx.core as mx
   print(mx.metal.is_available())  # Should be True
   ```
2. Check system resources (Activity Monitor)
3. Reduce max_tokens in requests
4. Increase prefill step size:
   ```bash
   SEMANTIC_MLX_PREFILL_STEP_SIZE=1024
   ```

### Issue: Cache Not Persisting

**Symptom**: `cache_read_input_tokens` always 0

**Solutions**:
1. Verify cache directory is writable:
   ```bash
   ls -la ~/.semantic/caches/
   ```
2. Check agent ID consistency
3. Verify model tag matches:
   ```bash
   SEMANTIC_AGENT_VALIDATE_MODEL_TAG=true
   ```
4. Check logs for cache save/load errors

### Issue: Poor Quality Responses

**Symptom**: Model generates nonsense or off-topic responses

**Solutions**:
1. Verify model is instruction-tuned (e.g., `-it` suffix)
2. Adjust temperature:
   ```bash
   SEMANTIC_MLX_DEFAULT_TEMPERATURE=0.7
   ```
3. Check prompt formatting
4. Try different model variant
5. Increase max_tokens for longer responses

## Model Compatibility Matrix

| Model Family | Tested | Cache Support | Tool Calling | Notes |
|--------------|--------|---------------|--------------|-------|
| DeepSeek-Coder-V2-Lite | ✅ | ✅ | ✅ | Production ready |
| SmolLM2 | ✅ | ✅ | ⚠️ | Testing only, basic tool support |
| Llama 3 | 🔄 | ✅ | ✅ | Compatible, not tested |
| Mistral | 🔄 | ✅ | ✅ | Compatible, not tested |
| Qwen | 🔄 | ✅ | ⚠️ | Compatible, tool support varies |
| DeepSeek | 🔄 | ✅ | ⚠️ | Compatible, tool support varies |

**Legend**:
- ✅ Fully supported and tested
- ⚠️ Partial support or requires tuning
- 🔄 Compatible but not verified
- ❌ Not supported

## Best Practices

1. **Start Small**: Test with SmolLM2 before production models
2. **Monitor Memory**: Use Activity Monitor to track memory usage
3. **Validate Cache**: Run multi-turn tests to verify cache persistence
4. **Benchmark Performance**: Measure latency and throughput
5. **Test Tool Calling**: Verify tool calling works with your model
6. **Document Settings**: Save optimal settings for each model
7. **Version Control**: Track model IDs and configurations

## See Also

- [Configuration Guide](configuration.md) - Model configuration reference
- [Testing Guide](testing.md) - Model testing strategies
- [Architecture: ModelCacheSpec](architecture/domain.md#modelcachespec) - Technical details
- [MLX Documentation](https://ml-explore.github.io/mlx/build/html/index.html) - MLX framework
