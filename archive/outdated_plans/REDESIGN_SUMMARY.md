# KV Cache Compression Redesign Summary

**Date:** 2026-01-22
**Status:** CRITICAL CORRECTION IMPLEMENTED

---

## Problem Identified

The initial Day 5 experiment used **text-level compression** (removing sentences from input), which does NOT test KV cache compression effects.

### Why Text Compression is Wrong:
- **What it does:** Removes information BEFORE the model sees it
- **What it tests:** "Can you follow instructions with incomplete input?"
- **Problem:** This is not what happens in real LLM deployments with KV cache eviction

### Why KV Cache Compression is Correct:
- **What it does:** Model processes ALL input, then evicts KV pairs during inference
- **What it tests:** "Can you follow instructions with degraded memory?"
- **Benefit:** This is EXACTLY what happens in real LLM deployments

**Analogy:**
- Text compression = Never hearing a sentence
- KV cache compression = Hearing it but forgetting parts later

These produce fundamentally different behaviors.

---

## Expert Debate Consensus

Conducted internal debate with 3 AI/LLM experts:
- **Dr. Cache** - KV cache optimization expert
- **Dr. Practical** - Implementation feasibility expert
- **Dr. Theory** - Instruction-following research expert

### Key Findings:

1. **Text compression is NOT a valid proxy** for KV cache compression
2. **HuggingFace Transformers** provides direct KV cache access (`past_key_values`)
3. **StreamingLLM** is the best compression method for MVP (simple, established, meaningful)
4. **4-bit quantization** required to fit Gemma 3 12B in 24GB RAM
5. **Random eviction** needed as control to show compression strategy matters

---

## Updated Approach

### Day 5 (Setup & Validation)
- Load Gemma 3 12B with 4-bit quantization via Transformers
- Implement StreamingLLM KV cache compression (keep first N + last M tokens)
- Implement random eviction baseline (control)
- Test on 2-3 examples to validate setup

### Day 6 (Full Experiment)
- Run 20-30 test examples in 3 conditions:
  1. **Baseline:** No KV compression
  2. **StreamingLLM:** Principled compression (50%)
  3. **Random Eviction:** Control condition
- **Incremental execution:** Process one example at a time, save after each, show progress
- Evaluate with hybrid evaluator (rule-based + Claude 4.5 Haiku)
- Statistical analysis and visualizations

### Day 7 (Buffer & Analysis)
- Deep analysis of results
- Document methodology
- Prepare for Week 2 R1 clustering

---

## Technical Implementation

### Model Loading (4-bit Quantization):
```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-3-12b-it",
    quantization_config=quantization_config,
    device_map="auto"
)
```

**Memory:** ~7GB (model) + ~2GB (KV cache) + ~2-4GB (activations) = ~11-13GB total (fits in 24GB)

### StreamingLLM Compression:
```python
def compress_kv_cache_streaming(past_key_values, keep_initial=100, keep_recent=100):
    compressed = []
    for layer_cache in past_key_values:
        key, value = layer_cache
        seq_len = key.shape[2]

        if seq_len <= keep_initial + keep_recent:
            compressed.append(layer_cache)
        else:
            # Keep initial + recent, EVICT middle
            new_key = torch.cat([
                key[:, :, :keep_initial, :],
                key[:, :, -keep_recent:, :]
            ], dim=2)
            new_value = torch.cat([
                value[:, :, :keep_initial, :],
                value[:, :, -keep_recent:, :]
            ], dim=2)
            compressed.append((new_key, new_value))

    return tuple(compressed)
```

### Incremental Experiment Execution:
```python
def run_experiment_incremental(test_examples, condition):
    results = []

    for i, example in enumerate(test_examples, 1):
        print(f"[{i}/{len(test_examples)}] Processing {example['id']}...")

        # Run example
        result = process_example(example, condition)
        results.append(result)

        # Save intermediate results
        save_json(results, f"results/exp1_{condition}_partial.json")

        # Print progress
        current_mean = np.mean([r['mean_score'] for r in results])
        print(f"  Score: {result['mean_score']:.3f}")
        print(f"  Running mean: {current_mean:.3f}")
        print(f"  Progress: {i}/{len(test_examples)} ({i/len(test_examples)*100:.1f}%)\n")

    return results
```

This allows:
- Seeing progress in real-time
- Resuming if interrupted
- Early stopping if results are clear
- Better debugging (can inspect partial results)

---

## Files Updated

### Complete Plan
- `/Users/dev_user/semantic/complete_plan.md`
  - Updated executive summary with Gemma 3 12B, Claude 4.5 models
  - Replaced Day 5 with KV cache setup approach
  - Updated Day 6 to full experiment execution
  - Updated Day 7 to buffer/analysis

### Daily Plans
- `/Users/dev_user/semantic/plans/day_05.md` - KV cache compression setup
- `/Users/dev_user/semantic/plans/day_06.md` - Full experiment with incremental execution
- `/Users/dev_user/semantic/plans/day_07.md` - Buffer and analysis

### Model References Fixed
- All references to Llama 3.1 → Gemma 3 12B
- All references to Claude 3.5 → Claude 4.5 Haiku/Sonnet
- Added model IDs explicitly:
  - `claude-sonnet-4-5-20250929` (dataset generation)
  - `claude-haiku-4-5-20251001` (evaluation judge)
  - `google/gemma-3-12b-it` (test model)

### CLI Usage for Cost Savings
- Evaluation can use Claude CLI via Task tool where applicable
- See `/Users/dev_user/semantic/claude.md` for integration details
- Reduces direct API call costs

---

## Key Benefits of New Approach

1. **Research Validity:** Tests actual KV cache compression, not text compression
2. **Reproducibility:** Clear methodology with direct cache manipulation
3. **Feasibility:** 4-bit quantization fits in 24GB RAM
4. **Incremental Progress:** See results as they come in, not just at the end
5. **Scientific Rigor:** Random eviction control validates compression strategy matters
6. **Real-World Relevance:** Tests exactly what happens in production LLM deployments

---

## Next Steps

1. **Immediate:** Implement Day 5 setup (KV cache compression functions)
2. **Tomorrow:** Run Day 6 full experiment
3. **Sunday:** Analyze and document results
4. **Monday (Week 2):** Begin R1 clustering experiments

---

## Lessons Learned

1. **Always validate approach against research question** - Text vs. KV compression seem similar but test different phenomena
2. **Expert debate is valuable** - Multiple perspectives caught critical flaw
3. **Incremental execution is better** - Allows progress tracking, resumption, early insights
4. **Model references matter** - Must use correct versions throughout (Claude 4.5, not 3.5)
5. **Memory constraints drive implementation** - 4-bit quantization enables Gemma 3 12B on 24GB RAM

---

**Status:** Ready to proceed with corrected approach starting Day 5.

**Last Updated:** 2026-01-22
