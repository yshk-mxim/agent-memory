# MLX Migration Plan: Semantic KV Cache Isolation

**Goal:** Port semantic isolation test from HuggingFace Transformers to MLX for better performance on Mac with Gemma 3 12B.

---

## Why MLX?

1. **Performance:** ~230 tok/s vs ~150 tok/s (llama.cpp) on Apple Silicon
2. **Native Metal:** Optimized by Apple for M-series chips
3. **KV Cache Access:** Lower-level API allows cache manipulation
4. **Larger Models:** Can run Gemma 3 12B with 4-bit quantization (~6GB RAM)
5. **Unified Memory:** Seamless CPU/GPU data sharing

---

## API Comparison

### HuggingFace (Current)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b-it")
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")

# Forward pass with cache
outputs = model(
    input_ids=input_ids,
    past_key_values=past_kv,  # Previous cache
    use_cache=True,
    return_dict=True
)
past_kv = outputs.past_key_values  # Updated cache
logits = outputs.logits
```

**Cache Structure:**
```python
past_key_values: Tuple[Tuple[torch.Tensor, torch.Tensor], ...]
# - Length = num_layers (e.g., 28 layers for Gemma 2B)
# - Each element = (keys, values) for that layer
# - keys/values shape: [batch, num_heads, seq_len, head_dim]
```

### MLX (Target)

```python
import mlx.core as mx
from mlx_lm import load

model, tokenizer = load("mlx-community/gemma-2-9b-it-4bit")

# MLX models expose internal structure
# Need to implement custom generation loop
```

**Cache Structure (based on MLX docs):**
```python
cache: List[Tuple[mx.array, mx.array]]
# - Length = num_layers
# - Each element = (keys, values) for that layer
# - keys/values shape: [batch, seq_len, num_heads, head_dim]
```

---

## Implementation Changes Needed

### 1. Model Loading

**Current (HuggingFace):**
```python
def __init__(self, model_name="google/gemma-2-2b-it", device="auto"):
    self.model = AutoModelForCausalLM.from_pretrained(model_name)
    self.tokenizer = AutoTokenizer.from_pretrained(model_name)
```

**New (MLX):**
```python
def __init__(self, model_name="mlx-community/gemma-2-9b-it-4bit"):
    from mlx_lm import load
    self.model, self.tokenizer = load(model_name)
```

**Complexity:** ✅ Easy - Just different import

---

### 2. Tokenization

**Current (HuggingFace):**
```python
inputs = self.tokenizer(text, return_tensors="pt").to(device)
input_ids = inputs.input_ids  # torch.Tensor
```

**New (MLX):**
```python
input_ids = self.tokenizer.encode(text)  # List[int]
input_ids = mx.array([input_ids])  # MLX array [1, seq_len]
```

**Complexity:** ✅ Easy - Similar API

---

### 3. Forward Pass with Cache

**Current (HuggingFace):**
```python
outputs = self.model(
    input_ids=input_ids,
    past_key_values=past_kv,
    use_cache=True,
    return_dict=True
)
next_token_logits = outputs.logits[:, -1, :]
past_kv = outputs.past_key_values
```

**New (MLX) - REQUIRES CUSTOM IMPLEMENTATION:**

MLX doesn't expose a high-level API like this. Need to implement manually:

```python
# Option A: Use mlx_lm's internal generation (simplest)
from mlx_lm.utils import generate_step

# This is what mlx_lm uses internally
logits, cache = generate_step(
    prompt=input_ids,
    model=self.model,
    cache=cache  # Pass previous cache
)
```

**Option B: Implement from scratch (more control):**

```python
# Access model layers directly
def forward_with_cache(self, input_ids, cache=None):
    """
    Manual forward pass through model layers.

    Args:
        input_ids: mx.array of shape [batch, seq_len]
        cache: List of (keys, values) tuples, one per layer

    Returns:
        logits: mx.array of shape [batch, seq_len, vocab_size]
        new_cache: Updated cache
    """
    x = self.model.model.embed_tokens(input_ids)

    if cache is None:
        cache = [None] * len(self.model.model.layers)

    new_cache = []
    for i, layer in enumerate(self.model.model.layers):
        # Each layer returns (output, (keys, values))
        x, layer_cache = layer(x, cache=cache[i])
        new_cache.append(layer_cache)

    x = self.model.model.norm(x)
    logits = self.model.lm_head(x)

    return logits, new_cache
```

**Complexity:** ⚠️ Moderate - Need to understand MLX model internals

---

### 4. Token Sampling

**Current (HuggingFace):**
```python
next_token = torch.argmax(logits, dim=-1, keepdim=True)
```

**New (MLX):**
```python
next_token = mx.argmax(logits, axis=-1, keepdims=True)
```

**Complexity:** ✅ Easy - Nearly identical

---

### 5. Cache Size Measurement

**Current (HuggingFace):**
```python
def get_cache_size(self, cache):
    if cache is None:
        return 0
    # cache[0][0] = keys for first layer
    return cache[0][0].shape[2]  # seq_len dimension
```

**New (MLX):**
```python
def get_cache_size(self, cache):
    if cache is None or len(cache) == 0:
        return 0
    # cache[0][0] = keys for first layer
    return cache[0][0].shape[1]  # seq_len dimension (different axis!)
```

**Complexity:** ✅ Easy - Just axis index change

---

## Complete Implementation Strategy

### Approach 1: Use mlx_lm's Internal API (Recommended)

**Pros:**
- Faster to implement
- Leverages tested code
- Handles edge cases

**Cons:**
- Less explicit control
- May need to read mlx_lm source code

**Implementation:**
```python
from mlx_lm import load
from mlx_lm.utils import generate_step
import mlx.core as mx

class SemanticIsolationTester_MLX:
    def __init__(self, model_name="mlx-community/gemma-2-9b-it-4bit"):
        self.model, self.tokenizer = load(model_name)

    def build_cache_for_cluster(self, turns):
        """Build KV cache for a semantic cluster."""
        cache = None

        for turn in turns:
            # Tokenize
            input_ids = self.tokenizer.encode(turn["text"])
            input_ids = mx.array([input_ids])

            # Forward pass (accumulate cache)
            logits, cache = generate_step(
                prompt=input_ids,
                model=self.model,
                cache=cache
            )

        return cache

    def generate_from_cache(self, cache, prompt, max_tokens=300):
        """Generate output using existing cache."""
        input_ids = self.tokenizer.encode(prompt)
        input_ids = mx.array([input_ids])

        # Start with cached context
        current_cache = cache
        generated = []

        for _ in range(max_tokens):
            logits, current_cache = generate_step(
                prompt=input_ids,
                model=self.model,
                cache=current_cache
            )

            # Sample next token
            next_token = mx.argmax(logits[:, -1, :], axis=-1)

            if next_token.item() == self.tokenizer.eos_token_id:
                break

            generated.append(next_token.item())
            input_ids = next_token[None]  # Reshape for next iteration

        return self.tokenizer.decode(generated)
```

**Estimated Time:** 2-4 hours

---

### Approach 2: Full Custom Implementation

**Pros:**
- Complete control
- Exact equivalent to HuggingFace version
- Educational

**Cons:**
- More complex
- Need deep understanding of transformer architecture
- More debugging

**Estimated Time:** 6-8 hours

---

## Testing Strategy

### Phase 1: Basic Validation
1. Load Gemma 3 12B in MLX ✓
2. Generate simple text (no cache) ✓
3. Verify generation quality ✓

### Phase 2: Cache Implementation
4. Build cache for single turn ✓
5. Generate with pre-built cache ✓
6. Verify cache reuse (measure tokens) ✓

### Phase 3: Semantic Isolation
7. Build separate caches for 3 clusters ✓
8. Generate outputs for each cluster ✓
9. Compare cache sizes ✓

### Phase 4: Full Experiment
10. Run validation_001 example ✓
11. Compare with HuggingFace results ✓
12. Run full 20 examples ✓

---

## Expected Performance Gains

### Current (Gemma 2 2B via HuggingFace on MPS):
- Model size: 2B parameters
- Generation speed: ~13 tokens/sec (based on test results)
- Memory: ~4GB

### Target (Gemma 3 12B via MLX on MPS):
- Model size: 12B parameters (6x larger)
- Generation speed: ~50-70 tokens/sec (estimated based on MLX benchmarks)
- Memory: ~7GB (4-bit quantization)
- **Quality:** Significantly better outputs

---

## File Changes Required

### New Files:
1. `src/semantic_isolation_mlx.py` - MLX implementation
2. `tests/test_mlx_implementation.py` - MLX-specific tests
3. `docs/MLX_API_GUIDE.md` - Documentation

### Modified Files:
1. `src/semantic_isolation.py` - Add MLX backend option
2. `requirements.txt` - Add mlx and mlx-lm
3. `DAY_5_POC_STATUS.md` - Document MLX results

---

## Dependencies

```bash
# Install MLX packages
pip install mlx mlx-lm

# Verify installation
python -c "import mlx.core as mx; print(mx.__version__)"
python -c "from mlx_lm import load; print('MLX-LM installed')"
```

**Size:** ~200MB download

---

## Risk Assessment

### Low Risk:
- ✅ MLX is stable and well-maintained by Apple
- ✅ mlx-lm has active community
- ✅ Gemma models available in mlx-community

### Medium Risk:
- ⚠️ API less documented than HuggingFace
- ⚠️ May need to read source code for details
- ⚠️ Debugging might be harder (less stack overflow posts)

### Mitigation:
- Start with Approach 1 (use mlx_lm internals)
- Test incrementally
- Keep HuggingFace version as reference

---

## Decision Matrix

| Criterion | HuggingFace (Current) | MLX (Proposed) |
|-----------|----------------------|----------------|
| **Model Size** | Gemma 2 2B | Gemma 3 12B (6x larger) |
| **Speed** | ~13 tok/s | ~50-70 tok/s (4-5x faster) |
| **Output Quality** | Poor (2B too small) | Much better (12B capable) |
| **Implementation Time** | Done | 2-4 hours |
| **API Maturity** | Very mature | Maturing |
| **Documentation** | Excellent | Good |
| **Cache Control** | Excellent | Good (requires more code) |
| **Mac Optimization** | Generic | Native (best for Mac) |

**Recommendation:** **Switch to MLX** - The quality and speed gains justify the 2-4 hour implementation effort.

---

## Next Steps

If you approve, I will:

1. **Install MLX** (2 minutes)
2. **Create basic test** - Load Gemma 3 12B and generate text (10 minutes)
3. **Implement cache builder** - Build KV cache for turns (30 minutes)
4. **Implement cache-based generation** - Generate with pre-built cache (30 minutes)
5. **Port full semantic isolation** - All 4 conditions (1 hour)
6. **Run validation test** - Compare results (30 minutes)
7. **Document findings** - Update status files (30 minutes)

**Total Time:** ~3.5 hours

**Output:**
- Working MLX implementation
- Gemma 3 12B results on validation_001
- Comparison with Gemma 2 2B results
- Documented performance metrics

---

**Ready to proceed?** Just say "yes" and I'll start with installing MLX and creating a basic test.
