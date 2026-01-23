# Context Length & KV Cache Size Analysis

**Date:** 2026-01-22
**Issue:** Test examples too short for meaningful KV cache compression

---

## Problem

Test examples average **~154 tokens** (range: 126-192 tokens)

With StreamingLLM parameters `keep_initial=100, keep_recent=100`:
- Compression activates when cache > 200 tokens
- **None of our examples reach this threshold**
- Compression would NEVER kick in!

This means the experiment wouldn't test anything - baseline and "compressed" conditions would be identical.

---

## Solution: Adjusted Compression Parameters

### Option 1: Smaller Compression Window (CHOSEN)
Use `keep_initial=30, keep_recent=30`:
- Compression activates when cache > 60 tokens
- **All examples exceed this (min: 126 tokens)**
- Compression ratio: 60/154 = ~39% of cache retained
- Strong compression effect expected

### Option 2: Add Filler Context (Rejected - Changes Dataset)
- Add 200+ tokens of filler text to each example
- Problem: Changes the dataset, may affect instruction-following
- Problem: Artificial, not realistic

### Option 3: Multi-Turn Cumulative Compression (Also Implemented)
- Compress after EACH turn, not just at end
- Turn 1: 50 tokens → no compression
- Turn 2: 100 tokens → compress to 60 tokens (30+30)
- Turn 3: 110 tokens → compress to 60 tokens again
- By final turn, early instructions may be evicted

**Final Approach:** Combine Option 1 + 3
- Use `keep_initial=30, keep_recent=30`
- Compress after each turn
- Ensures meaningful compression even with short examples

---

## Updated Compression Parameters

### StreamingLLM:
```python
def compress_kv_cache_streaming(past_key_values, keep_initial=30, keep_recent=30):
    """
    Adjusted for short examples (~150 tokens).
    Compression activates when cache > 60 tokens.
    Retains ~40% of cache.
    """
    # ... same implementation as before
```

### When to Compress:
```python
for turn_idx, turn in enumerate(example['turns']):
    # Generate response
    outputs = model.generate(..., past_key_values=past_kv)
    past_kv = outputs.past_key_values

    # Compress after EVERY turn (not just at end)
    if compression_policy == 'streaming':
        past_kv = compress_kv_cache_streaming(past_kv)
    elif compression_policy == 'random':
        past_kv = compress_kv_cache_random(past_kv, retention_rate=0.4)  # 40% retained

    # Log cache size for analysis
    cache_sizes.append(past_kv[0][0].shape[2])
```

---

## Expected Behavior

### Example with 3 turns (~150 tokens total):

**Baseline (No Compression):**
- After Turn 1: 50 tokens
- After Turn 2: 100 tokens
- After Turn 3: 150 tokens
- Final cache: 150 tokens (100% retention)

**StreamingLLM (keep_initial=30, keep_recent=30):**
- After Turn 1: 50 tokens (no compression, < 60 threshold)
- After Turn 2: 100 tokens → compress to 60 tokens
- After Turn 3: 110 tokens (60 + new 50) → compress to 60 tokens
- Final cache: 60 tokens (40% retention)

**What Gets Evicted:**
- Middle portion of Turn 1 (early instruction details)
- Most of Turn 2 (conflicting instruction)
- Retains: System prompt + Turn 1 start + Turn 3 recent context

**Hypothesis:** Turn 1 instructions will be partially evicted, causing model to favor Turn 2 instructions, degrading RDIC score.

---

## Validation Test

Before running full experiment, verify compression activates:

```python
def test_compression_activates():
    example = load_json("data/test.json")[0]

    past_kv = None
    cache_sizes = []

    print(f"Testing: {example['id']}")
    print(f"Expected total tokens: ~{estimate_tokens(example)}")
    print()

    for i, turn in enumerate(example['turns'], 1):
        # Generate
        outputs = model.generate(..., past_key_values=past_kv)
        past_kv = outputs.past_key_values

        size_before = past_kv[0][0].shape[2]
        print(f"Turn {i}: Cache size before compression: {size_before} tokens")

        # Compress
        past_kv = compress_kv_cache_streaming(past_kv, keep_initial=30, keep_recent=30)

        size_after = past_kv[0][0].shape[2]
        print(f"         Cache size after compression: {size_after} tokens")
        print(f"         Compression activated: {'YES' if size_after < size_before else 'NO'}")
        print()

        cache_sizes.append(size_after)

    print(f"Final cache: {cache_sizes[-1]} tokens")
    print(f"Compression ratio: {cache_sizes[-1] / estimate_tokens(example) * 100:.1f}%")

    return cache_sizes

# Expected output:
# Turn 1: 50 tokens → 50 tokens (no compression)
# Turn 2: 100 tokens → 60 tokens (YES)
# Turn 3: 110 tokens → 60 tokens (YES)
# Final: 60 tokens (40% retention)
```

---

## Impact on Results

With adjusted parameters:
- **Baseline:** Full 150-token cache retained
- **StreamingLLM:** ~60-token cache (40% retention)
- **Expected degradation:** 10-20% score reduction
- **Statistical power:** Should be sufficient with 20-30 examples

---

## Alternative: Test on Longer Examples

If results are inconclusive with short examples, consider:

1. **Generate longer test examples** (400+ tokens):
   - Add 2-3 more turns between conflicting instructions
   - Use more detailed contexts
   - Use standard parameters (keep_initial=100, keep_recent=100)

2. **Use existing longer examples from train set**:
   - Filter for examples > 400 tokens
   - Run experiment on these
   - Compare to short examples

---

## Key Takeaways

1. **Context length MUST match compression parameters**
   - If examples are ~150 tokens, use keep=30+30
   - If examples are ~400 tokens, use keep=100+100

2. **Compress after EACH turn, not just once**
   - Allows compression to accumulate effects
   - Tests realistic multi-turn scenario

3. **Log cache sizes for validation**
   - Verify compression activates
   - Track how much is retained at each turn
   - Confirm early instructions are actually evicted

4. **Adjust compression rate to example length**
   - Short examples (~150 tokens): 40% retention
   - Long examples (~400 tokens): 50% retention
   - Goal: Meaningful compression without complete information loss

---

**Status:** Parameters adjusted for short examples. Ready to test.

**Last Updated:** 2026-01-22
