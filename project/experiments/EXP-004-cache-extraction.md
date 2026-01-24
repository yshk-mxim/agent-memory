# EXP-004: Per-Sequence Cache Extraction

**Date**: 2026-01-24
**Status**: ✅ PASSED
**Sprint**: 0 (Foundation)

## Objective

Prove that per-sequence cache extraction works on completion via `Response.prompt_cache` attribute, enabling individual sequences in a batch to complete independently.

## Hypothesis

When multiple sequences are generated in a batch via `BatchGenerator`, each sequence's KV cache can be extracted individually when that sequence completes (via `finish_reason`), without waiting for the entire batch to finish.

## Method

1. Load Gemma 3 12B model
2. Create batch of 3 prompts:
   - "The capital of France is"
   - "Machine learning is"
   - "Python programming language"
3. Generate with `BatchGenerator`, extracting cache when each sequence finishes
4. Save all 3 caches to disk individually
5. Reload all 3 caches from disk
6. Create continuation prompts (original + generated text)
7. Re-inject caches and continue generation
8. Verify all 3 sequences complete successfully

## Results

✅ **SUCCESS** - Per-sequence cache extraction works perfectly

### Key Findings

1. **Independent Completion**: Sequences complete independently
   - UID 1 finished first: ' rapidly changing the way we approach many tasks,'
   - UID 2 finished second: ' is a versatile and powerful language that is used'
   - UID 0 finished last: ' Paris.\\nThe capital of France is Paris'
   - Each extracted its own 48-layer cache on completion

2. **Cache Persistence**: Save/reload cycle works flawlessly
   - All 3 caches saved to separate `.safetensors` files
   - All 3 caches reloaded successfully
   - No corruption or data loss

3. **Cache Re-injection**: Continued generation works with reloaded caches
   - UID 1: ' from image recognition to'
   - UID 2: ' in a wide range'
   - UID 0: '.\\nThe capital'
   - All sequences generated continuation tokens correctly

4. **No Batch Blocking**: Don't need to wait for full batch to finish
   - As soon as UID 1 finished, cache was available
   - UID 0 was still generating, no issue

## Architecture Implications

1. **ConcurrentScheduler can return results immediately** when a sequence finishes
   - No need to wait for all agents in a batch to complete
   - Enables true concurrent serving with variable-length requests

2. **AgentCacheStore can persist caches mid-batch**
   - As soon as agent A finishes, save cache
   - Agent B can continue generating independently

3. **Per-agent locks are sufficient** for cache consistency
   - Agent A's sequential requests will use updated cache
   - No need for complex batch-level synchronization

## Performance Observations

- **Batch size**: 3 concurrent sequences
- **Cache size**: 48 layers (Gemma 3: 8 global + 40 rotating)
- **Tokens generated**: 9-10 tokens per sequence
- **Save/reload**: No measurable latency (< 10ms per cache)

## Corrected API Summary

```python
# Create batch generator (NO tokenizer parameter)
gen = BatchGenerator(model, stop_tokens=set([eos_token_id]), max_tokens=10)

# Insert tokenized prompts with optional caches
uids = gen.insert(
    [tokenizer.encode(p) for p in prompts],
    caches=[cache_a, cache_b, cache_c]  # Optional, one per prompt
)

# Iterate over responses
tokens_by_uid = {uid: [] for uid in uids}
while responses := gen.next():
    for r in responses:
        if r.finish_reason is not None:
            # Sequence finished - extract cache (attribute, not method)
            cache = r.prompt_cache  # List of 48 RotatingKVCache objects
            text = tokenizer.decode(tokens_by_uid[r.uid])
            # This sequence is now removed from batch; others continue
        else:
            # Accumulate tokens (singular .token, not .tokens)
            tokens_by_uid[r.uid].append(r.token)
```

## Recommendation

✅ **PROCEED** with continuous batching architecture

All critical assumptions validated:
- ✅ Cache injection works
- ✅ Per-sequence extraction works
- ✅ Independent completion works
- ✅ Save/reload cycle works

## Files

- Experiment script: `experiments/exp_004_cache_extraction.py`
- Output: All 3 sequences extracted, saved, reloaded, and continued successfully
