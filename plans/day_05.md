# Day 5 (Friday): KV Cache Compression Setup & Validation

**Week 1 - Day 5**

---

## CRITICAL UPDATE

**Previous Approach (FLAWED):** Text-level compression - removing sentences from input
**Why Flawed:** Tests "instruction-following with incomplete input", not "instruction-following with degraded memory"

**New Approach (CORRECT):** Real KV cache manipulation via HuggingFace Transformers
**Why Correct:** Model processes all input, then forgets during inference (simulates real-world KV cache eviction)

**Expert Consensus:** Text compression and KV cache compression are fundamentally different phenomena. Only KV cache manipulation tests the research hypothesis.

---

**Objectives:**
- Set up Gemma 3 12B with 4-bit quantization for direct KV cache access
- Implement StreamingLLM KV cache compression
- Implement random eviction baseline (control condition)
- Validate setup works correctly on 2-3 test examples
- Prepare experiment scaffold for Day 6 full run

**Tasks:**

| Task | Time | Details |
|------|------|---------|
| Install dependencies | 1h | transformers, accelerate, bitsandbytes |
| Load Gemma 3 12B with 4-bit quantization | 1h | Verify fits in 24GB RAM, test inference |
| Implement StreamingLLM KV compression | 1.5h | Keep first N + last M tokens in cache |
| Implement random eviction baseline | 1h | Random KV pair deletion (control) |
| Test on 2-3 RDIC examples | 1h | Verify compression works, output is coherent |
| Document KV cache structure | 30m | Tensor shapes, memory usage, compression effects |
| Create experiment runner scaffold | 1h | Setup for Day 6 full experiment |

---

## Setup Code

### 1. Install Dependencies

```bash
pip install transformers accelerate bitsandbytes torch
# transformers: Model loading and inference
# accelerate: Multi-GPU and memory management
# bitsandbytes: 4-bit quantization
```

### 2. Load Gemma 3 12B with 4-bit Quantization

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

# Configure 4-bit quantization to fit in 24GB RAM
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

# Load model (~7GB with 4-bit quantization)
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-3-12b-it",
    quantization_config=quantization_config,
    device_map="auto",
    torch_dtype=torch.float16
)

tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-12b-it")
```

**Memory Breakdown:**
- Model weights (4-bit): ~7GB
- KV cache (fp16): ~2GB per 2K tokens
- Activations: ~2-4GB
- Total: ~11-13GB (fits comfortably in 24GB)

### 3. Implement StreamingLLM KV Cache Compression

```python
def compress_kv_cache_streaming(past_key_values, keep_initial=100, keep_recent=100):
    """
    StreamingLLM compression: Keep initial tokens (system prompt) + recent tokens (sliding window).

    This is TRUE KV cache compression:
    - Model processes all tokens initially
    - We evict middle KV pairs after encoding
    - Model tries to attend but finds gaps in memory

    Args:
        past_key_values: Tuple of (num_layers) tuples of (key, value) tensors
        keep_initial: Number of initial tokens to retain
        keep_recent: Number of recent tokens to retain

    Returns:
        Compressed KV cache tuple
    """
    if past_key_values is None:
        return None

    compressed = []
    for layer_cache in past_key_values:
        key, value = layer_cache
        # key/value shape: [batch_size, num_heads, seq_len, head_dim]
        seq_len = key.shape[2]

        if seq_len <= keep_initial + keep_recent:
            # Cache not long enough to compress yet
            compressed.append(layer_cache)
        else:
            # Compress: Keep initial + recent, EVICT middle
            new_key = torch.cat([
                key[:, :, :keep_initial, :],      # Initial tokens
                key[:, :, -keep_recent:, :]       # Recent tokens
            ], dim=2)
            new_value = torch.cat([
                value[:, :, :keep_initial, :],
                value[:, :, -keep_recent:, :]
            ], dim=2)
            compressed.append((new_key, new_value))

    return tuple(compressed)
```

### 4. Implement Random Eviction Baseline

```python
def compress_kv_cache_random(past_key_values, retention_rate=0.5):
    """
    Random eviction: Randomly discard (1 - retention_rate) of KV pairs.

    This is a CONTROL condition to show that:
    - Principled compression (StreamingLLM) should outperform random
    - If they're equivalent, any memory loss degrades performance

    Args:
        past_key_values: Tuple of (num_layers) tuples of (key, value) tensors
        retention_rate: Fraction of KV pairs to keep (0.5 = keep 50%)

    Returns:
        Compressed KV cache with random eviction
    """
    if past_key_values is None:
        return None

    compressed = []
    for layer_cache in past_key_values:
        key, value = layer_cache
        seq_len = key.shape[2]
        keep_count = max(1, int(seq_len * retention_rate))

        # Randomly select indices to keep
        indices = torch.randperm(seq_len)[:keep_count].sort()[0]

        new_key = key[:, :, indices, :]
        new_value = value[:, :, indices, :]
        compressed.append((new_key, new_value))

    return tuple(compressed)
```

---

## Validation Test

Run 2-3 examples to verify the setup works:

```python
def test_kv_compression():
    # Load test example
    example = load_json("data/test.json")[0]

    # Test 1: Baseline (no compression)
    print("Test 1: Baseline (no compression)")
    past_kv = None
    for turn in example['turns']:
        inputs = tokenizer(turn['content'], return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            past_key_values=past_kv,
            max_new_tokens=100,
            return_dict_in_generate=True
        )
        past_kv = outputs.past_key_values
        print(f"  KV cache size: {past_kv[0][0].shape[2]} tokens")

    # Test 2: StreamingLLM compression
    print("\nTest 2: StreamingLLM compression")
    past_kv = None
    for turn in example['turns']:
        inputs = tokenizer(turn['content'], return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            past_key_values=past_kv,
            max_new_tokens=100,
            return_dict_in_generate=True
        )
        past_kv = outputs.past_key_values
        past_kv = compress_kv_cache_streaming(past_kv, keep_initial=50, keep_recent=50)
        print(f"  KV cache size after compression: {past_kv[0][0].shape[2]} tokens")

    # Test 3: Random eviction
    print("\nTest 3: Random eviction")
    past_kv = None
    for turn in example['turns']:
        inputs = tokenizer(turn['content'], return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            past_key_values=past_kv,
            max_new_tokens=100,
            return_dict_in_generate=True
        )
        past_kv = outputs.past_key_values
        past_kv = compress_kv_cache_random(past_kv, retention_rate=0.5)
        print(f"  KV cache size after compression: {past_kv[0][0].shape[2]} tokens")

if __name__ == "__main__":
    test_kv_compression()
```

**Expected Output:**
- Baseline: KV cache grows with each turn (e.g., 50 → 150 → 300 tokens)
- StreamingLLM: KV cache stays at ~100 tokens after compression kicks in
- Random: KV cache ~50% of baseline size

**Success Criteria:**
- Model loads without OOM errors
- KV cache compression reduces tensor size
- Generated text is still coherent (not gibberish)
- Ready to run full experiment on Day 6

---

## Files to Create

- `/Users/dev_user/semantic/src/kv_cache_compression.py` - Compression functions
- `/Users/dev_user/semantic/src/transformers_inference.py` - Gemma 3 wrapper with KV cache access
- `/Users/dev_user/semantic/experiments/exp1_kv_compression.py` - Main experiment runner (scaffold)
- `/Users/dev_user/semantic/tests/test_kv_compression.py` - Unit tests for compression functions

---

## Success Criteria

- [ ] Gemma 3 12B loads with 4-bit quantization in <24GB RAM
- [ ] Can access and manipulate `past_key_values` directly
- [ ] StreamingLLM compression reduces KV cache size by ~50%
- [ ] Random eviction compression works as control
- [ ] Model generates coherent responses with compressed cache (not degraded to gibberish)
- [ ] 2-3 validation examples run successfully
- [ ] Experiment scaffold ready for Day 6 full run

---

## Key Insights

**Why This Approach is Correct:**

1. **Text Compression (Previous):**
   - Removes information BEFORE model sees it
   - Model never encodes removed sentences
   - Tests: "Can you follow instructions with incomplete input?"
   - NOT what happens in real LLM deployments

2. **KV Cache Compression (New):**
   - Model processes ALL input initially
   - Eviction happens AFTER encoding
   - Model tries to attend but finds gaps in memory
   - Tests: "Can you follow instructions with degraded memory?"
   - EXACTLY what happens in real LLM deployments with cache eviction

**Analogy:**
- Text compression: Never hearing a sentence
- KV cache compression: Hearing it but forgetting parts later

These produce **different behaviors** and must be tested separately.

---

## Next Steps

**Day 6:**
- Run full experiment on 20-30 test examples
- Compare baseline vs. StreamingLLM vs. random eviction
- Evaluate with hybrid evaluator (rule-based + LLM judge)
- Generate Figure 1 and Table 1
- Statistical analysis

**Day 7:**
- Analyze results in depth
- Document methodology
- Prepare for Week 2 R1 clustering

---

**Quick Reference:**
- **Previous Day:** [Day 4](day_04.md) - Evaluation framework
- **Next Day:** [Day 6](day_06.md) - Run full KV compression experiment
- **Complete Plan:** [Complete 3-Week Plan](../complete_plan.md)

---

*Last Updated: 2026-01-22 (Post-expert debate correction)*
