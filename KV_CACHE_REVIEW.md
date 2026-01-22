# HYPERCRITICAL REVIEW: KV Cache Isolation Implementation

**Date:** 2026-01-22
**File:** `src/semantic_isolation.py`
**Reviewer:** Claude Sonnet 4.5 (Self-Review with Magnifying Glass)

---

## 🔴 CRITICAL ISSUES (Must Fix)

### Issue 1: Generation Prompts Leak Semantic Information

**Location:** All condition final generation calls

**Current Code:**
```python
output_technical = self.generate_from_cache(
    past_kv,
    "Based on the technical performance analysis, provide your recommendations:",
    max_new_tokens=300
)
```

**Problem:** The prompt "technical performance analysis" PRIMES the model to think about technical topics, even if the cache is mixed!

**Impact:**
- **Sequential condition looks better than it should** - prompts compensate for cache pollution
- **Not testing pure cache isolation** - testing cache + targeted prompting
- **Confounds the experiment** - can't distinguish cache benefits from prompt benefits

**Evidence of Contamination:**
- Sequential cache: 750 tokens (technical + business mixed)
- Prompt: "Based on the **technical performance analysis**..."
- Model sees: [mixed cache] + [technical prompt] → focuses on technical despite pollution
- This MASKS the interference we're trying to measure!

**Fix:** Use semantically NEUTRAL prompts:
```python
# Bad (current)
"Based on the technical performance analysis, provide your recommendations:"

# Good (neutral)
"Generate output A based on the provided context:"

# Better (completely neutral)
"Provide your response:"
```

**Severity:** 🔴 **CRITICAL** - Invalidates the core hypothesis test

**Recommendation:** Create a `use_neutral_prompts=True` parameter and test both ways.

---

### Issue 2: Generated Outputs Injected into Cluster 3 Cache (Message Passing ≠ Isolation)

**Location:** `condition_4_semantic()`, cluster 3 processing

**Current Code:**
```python
synthesis_context = f"""
Technical Analysis Summary:
{outputs['technical']}  # ← GENERATED TEXT FROM CLUSTER 1

Business Strategy Summary:
{outputs['business']}  # ← GENERATED TEXT FROM CLUSTER 2

Now consider the executive synthesis instructions:
"""

# This text gets added to cluster 3's KV cache!
inputs = self.tokenizer(synthesis_context, return_tensors="pt")
outputs_obj = self.model(**inputs, past_key_values=past_kv_c3, use_cache=True)
past_kv_c3 = outputs_obj.past_key_values
```

**Problem:** Cluster 3's cache contains GENERATED TEXT from clusters 1 and 2, not just input turns.

**Is This a Bug?** No, it's **intentional** (message passing from multi-agent pattern), BUT:
1. The generated text from cluster 1 contains technical jargon
2. The generated text from cluster 2 contains business terms
3. Cluster 3's cache now has BOTH semantic spaces
4. This creates **indirect leakage** through the generated outputs

**Measurement:**
- If cluster 1 output says "optimize database queries, add Redis cache, horizontal scaling"
- Cluster 3 cache now contains these technical terms
- This could influence cluster 3's synthesis

**Expected Behavior:**
- Cluster 1 cache: Technical INPUT turns only ✓
- Cluster 2 cache: Business INPUT turns only ✓
- Cluster 3 cache: Synthesis INPUT turns + OUTPUTS from c1 & c2 ⚠️

**Is This OK?**
- For multi-agent simulation: YES (this is message passing)
- For pure isolation testing: NO (cluster 3 is not isolated)

**Severity:** 🟡 **MEDIUM** - Intended design but needs documentation and measurement

**Recommendation:**
1. Document this as "message passing" not "pure isolation"
2. Measure semantic leakage in cluster 3 outputs
3. Consider testing a "pure isolation" variant where cluster 3 only sees cluster 3 turns

---

### Issue 3: Cache Not Freed Between Conditions (Memory Accumulation)

**Location:** `test_all_conditions()`

**Current Code:**
```python
results = {}
results['sequential'] = self.condition_1_sequential(example)
results['prompted'] = self.condition_2_prompted(example)
results['turn_based'] = self.condition_3_turn_based(example)
results['semantic'] = self.condition_4_semantic(example)
```

**Problem:** Each `IsolationResult` contains the `past_key_values` tensors in memory!

**Memory Calculation:**
For Gemma 2 12B (assumed: 24 layers, 16 heads, 128 head_dim):
- Cache size = 24 layers × seq_len tokens × 2 (key+value) × 16 heads × 128 dim × 2 bytes (fp16)
- For 750-token cache: 24 × 750 × 2 × 16 × 128 × 2 = **147 MB**
- After 4 conditions: **~600 MB** in cache objects

**Impact:**
- For 1 example: Manageable (600MB)
- For 20 examples × 4 conditions: **12 GB** just in cached results!
- Risk of OOM when running full experiment

**Fix:**
```python
import gc
import torch

def test_all_conditions(self, example):
    results = {}

    # Condition 1
    results['sequential'] = self.condition_1_sequential(example)
    # Don't store the actual cache tensors
    results['sequential'].past_key_values = None
    torch.cuda.empty_cache()
    gc.collect()

    # Repeat for other conditions...
```

**Severity:** 🟡 **MEDIUM** - Will cause OOM on full 20-example experiment

**Recommendation:** Clear GPU cache between conditions and don't store `past_key_values` in results.

---

### Issue 4: No Determinism (Results Not Reproducible)

**Location:** `generate_from_cache()`

**Current Code:**
```python
outputs = self.model.generate(
    **inputs,
    past_key_values=past_key_values,
    max_new_tokens=max_new_tokens,
    temperature=0.7,  # ← STOCHASTIC
    top_p=0.9,
    do_sample=True  # ← NON-DETERMINISTIC
)
```

**Problem:** Same example will produce DIFFERENT outputs on each run!

**Impact:**
- Can't reproduce exact results
- Can't debug specific failures
- Hard to verify if changes improve performance

**Fix:**
```python
def __init__(self, ..., random_seed: int = 42):
    self.random_seed = random_seed
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)

def generate_from_cache(self, ..., deterministic: bool = False):
    if deterministic:
        temp, top_p, do_sample = 0.0, 1.0, False
    else:
        temp, top_p, do_sample = 0.7, 0.9, True

    outputs = self.model.generate(
        temperature=temp,
        top_p=top_p,
        do_sample=do_sample,
        ...
    )
```

**Severity:** 🟡 **MEDIUM** - Reduces research reproducibility

**Recommendation:** Add `deterministic=True` mode and set random seeds.

---

### Issue 5: No Validation of Cluster Labels

**Location:** `condition_4_semantic()`

**Current Code:**
```python
cluster_1_turns = [t for t in example['turns'] if t['cluster'] == 1]
cluster_2_turns = [t for t in example['turns'] if t['cluster'] == 2]
cluster_3_turns = [t for t in example['turns'] if t['cluster'] == 3]
```

**Problem:** What if:
- Example doesn't have 'cluster' field? → KeyError
- Clusters labeled 0, 1, 2 instead of 1, 2, 3? → Empty lists
- Some turns missing cluster labels? → Silently skipped

**Current Behavior:**
```python
# If cluster field is missing:
cluster_1_turns = []  # Empty! No error raised!
# Then:
for turn in cluster_1_turns:  # Loop never executes
    ...
# Result: Cluster 1 cache is None, generation fails later
```

**Fix:**
```python
def validate_example(self, example):
    """Validate example has required structure."""
    assert 'turns' in example, "Example missing 'turns' field"
    assert len(example['turns']) > 0, "Example has no turns"

    for turn in example['turns']:
        assert 'cluster' in turn, f"Turn {turn.get('turn_id', '?')} missing 'cluster' field"

    clusters = {t['cluster'] for t in example['turns']}
    assert clusters == {1, 2, 3}, f"Expected clusters {{1,2,3}}, got {clusters}"

    # Check distribution
    c1_count = sum(1 for t in example['turns'] if t['cluster'] == 1)
    c2_count = sum(1 for t in example['turns'] if t['cluster'] == 2)
    c3_count = sum(1 for t in example['turns'] if t['cluster'] == 3)

    assert c1_count >= 3, f"Cluster 1 has only {c1_count} turns (need ≥3)"
    assert c2_count >= 3, f"Cluster 2 has only {c2_count} turns (need ≥3)"
    assert c3_count >= 3, f"Cluster 3 has only {c3_count} turns (need ≥3)"
```

**Severity:** 🟡 **MEDIUM** - Silent failures are worse than crashes

**Recommendation:** Add validation before processing.

---

## 🟡 MODERATE ISSUES (Should Fix)

### Issue 6: Cache Size Measurement May Be Inaccurate

**Location:** `get_cache_size()`

**Current Code:**
```python
def get_cache_size(self, past_key_values):
    if past_key_values is None:
        return 0
    return past_key_values[0][0].shape[2]  # seq_len from first layer
```

**Assumptions:**
1. `past_key_values[0]` exists (first layer)
2. `past_key_values[0][0]` exists (key tensor)
3. Shape is `[batch_size, num_heads, seq_len, head_dim]`
4. Batch size is 1

**Potential Failures:**
- Empty cache structure? → IndexError
- Different shape format? → Wrong dimension
- Batch size > 1? → Incorrect count

**Fix:**
```python
def get_cache_size(self, past_key_values):
    """Get KV cache size in tokens with validation."""
    if past_key_values is None:
        return 0

    try:
        # Structure: tuple of (num_layers) tuples of (key, value) tensors
        # key shape: [batch_size, num_heads, seq_len, head_dim]
        key_tensor = past_key_values[0][0]

        assert key_tensor.dim() == 4, f"Expected 4D tensor, got {key_tensor.dim()}D"
        batch_size, num_heads, seq_len, head_dim = key_tensor.shape

        assert batch_size == 1, f"Batch size {batch_size} != 1 (not supported)"

        return seq_len

    except Exception as e:
        print(f"Warning: Could not get cache size: {e}")
        return 0
```

**Severity:** 🟡 **MEDIUM** - Could silently report wrong sizes

---

### Issue 7: Turn Text Extraction Skips Empty Turns Silently

**Location:** All conditions

**Current Code:**
```python
for turn in example['turns']:
    turn_text = turn.get('instruction', '') or turn.get('content', '') or turn.get('query', '')
    if not turn_text:
        continue  # ← SILENT SKIP
```

**Problem:** If a turn has all fields empty/None, it's skipped without logging.

**Impact:**
- Example claims 15 turns, but only 12 are processed
- Cache size expectations are wrong
- User doesn't know why

**Fix:**
```python
skipped_turns = []
for i, turn in enumerate(example['turns']):
    turn_text = turn.get('instruction', '') or turn.get('content', '') or turn.get('query', '')
    if not turn_text:
        skipped_turns.append(turn.get('turn_id', i))
        continue

if skipped_turns:
    print(f"Warning: Skipped {len(skipped_turns)} empty turns: {skipped_turns}")
```

**Severity:** 🟡 **MEDIUM** - Silent data loss

---

### Issue 8: Context Window Overflow Not Checked

**Location:** All generation calls

**Max Context:** Gemma 2 12B has 8,192 token context window

**Current Usage:**
- Sequential: 750 tokens (input) + 50 (prompt) + 300 (generation) = 1,100 tokens ✓
- Semantic cluster 3: 550 (input + c1/c2 outputs) + 50 (prompt) + 400 (generation) = 1,000 tokens ✓

**Safe for POC**, but what about:
- 20-turn examples (1,000+ tokens)
- Very long generated outputs (500+ tokens)
- 50-turn conversations (2,000+ tokens)

**Fix:**
```python
MAX_CONTEXT = 8192  # Gemma 2 limit

def validate_context_window(self, cache_size, prompt, max_new_tokens):
    """Check if generation will exceed context window."""
    prompt_tokens = len(self.tokenizer.encode(prompt))
    total = cache_size + prompt_tokens + max_new_tokens

    if total > MAX_CONTEXT:
        raise ValueError(
            f"Context overflow: {cache_size} (cache) + {prompt_tokens} (prompt) + "
            f"{max_new_tokens} (generation) = {total} > {MAX_CONTEXT} max"
        )
```

**Severity:** 🟡 **MEDIUM** - Will fail on longer examples without clear error

---

## 🟢 MINOR ISSUES (Nice to Fix)

### Issue 9: Gemma 3 vs Gemma 2 Naming Inconsistency

**Problem:** Code says `gemma-2-12b-it` but comments/docs say "Gemma 3"

**Reality:** Gemma 3 doesn't exist yet (as of Jan 2026). Latest is Gemma 2.

**Fix:** Change all references to "Gemma 2 12B"

**Severity:** 🟢 **MINOR** - Cosmetic confusion

---

### Issue 10: No Timing Breakdown

**Current:** Total time per condition

**Better:** Time breakdown:
- Cache building time
- Generation time (per output)
- Total time

**Severity:** 🟢 **MINOR** - Nice for analysis

---

## ✅ THINGS THAT ARE CORRECT

### ✓ Cache Isolation Architecture

**Cluster 1 and 2 caches are truly isolated:**
```python
# Cluster 1
past_kv_c1 = None
for turn in cluster_1_turns:
    outputs = model(..., past_key_values=past_kv_c1)
    past_kv_c1 = outputs.past_key_values

# Cluster 2 - FRESH START
past_kv_c2 = None  # Does NOT see cluster 1
for turn in cluster_2_turns:
    outputs = model(..., past_key_values=past_kv_c2)
    past_kv_c2 = outputs.past_key_values
```

**This is correct!** Model cannot cross-attend because KV pairs don't exist in the other cache.

---

### ✓ No Cache Pollution from Generated Text

```python
def generate_from_cache(self, past_key_values, prompt, ...):
    outputs = self.model.generate(...)
    generated_text = self.tokenizer.decode(...)
    return generated_text  # Returns TEXT, not updated cache
```

**Correct!** We don't capture the updated cache, so generated text doesn't pollute the input caches.

---

### ✓ Model Loading with 4-bit Quantization

```python
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)
```

**Correct setup** for fitting 12B model in 24GB VRAM.

---

## 📊 SEVERITY SUMMARY

| Severity | Count | Issues |
|----------|-------|--------|
| 🔴 Critical | 2 | Generation prompts leak semantics, Message passing in cluster 3 |
| 🟡 Medium | 6 | Memory accumulation, No determinism, No validation, Cache size accuracy, Silent skips, Context overflow |
| 🟢 Minor | 2 | Naming inconsistency, No timing breakdown |
| ✅ Correct | 3 | Cache isolation architecture, No generated text pollution, 4-bit quantization |

---

## 🔧 RECOMMENDED FIXES (Priority Order)

### Priority 1 (Critical - Fix Before Testing)

1. **Use neutral generation prompts** to avoid semantic priming
2. **Document message passing** in cluster 3 (not a bug, but needs clarity)
3. **Add cache cleanup** between conditions to prevent OOM

### Priority 2 (Important - Fix Before Full Experiment)

4. **Set random seed** for reproducibility
5. **Validate example structure** before processing
6. **Check context window** before generation

### Priority 3 (Nice to Have)

7. Fix Gemma 2/3 naming
8. Add timing breakdown
9. Log skipped turns

---

## 🧪 TEST RECOMMENDATIONS

### Test 1: Verify Cache Isolation

```python
# After running semantic condition:
assert len(results['semantic'].cache_sizes) == 3
c1_size = results['semantic'].cache_sizes['cluster_1_technical']
c2_size = results['semantic'].cache_sizes['cluster_2_business']
seq_size = results['sequential'].cache_sizes['unified']

assert c1_size < seq_size, "Cluster 1 should be smaller than sequential"
assert c2_size < seq_size, "Cluster 2 should be smaller than sequential"
```

### Test 2: Verify No Leakage Between Clusters 1 and 2

```python
# Check that technical output doesn't appear in business output
technical_terms = ['database', 'query', 'latency', 'cache', 'throughput']
business_output = results['semantic'].outputs['business'].lower()

leakage_count = sum(1 for term in technical_terms if term in business_output)
print(f"Technical terms in business output: {leakage_count}/{len(technical_terms)}")

# Expected: Low leakage for semantic, high leakage for sequential
```

### Test 3: Measure Memory Usage

```python
import torch

before = torch.cuda.memory_allocated() / 1e9
results = tester.test_all_conditions(example)
after = torch.cuda.memory_allocated() / 1e9

print(f"Memory increase: {after - before:.2f} GB")
# Should be <1GB for one example
```

---

**Status:** Implementation is **MOSTLY CORRECT** but has **2 critical issues** that must be fixed before testing.

**Recommendation:** Fix Priority 1 issues, then proceed with validation testing.
