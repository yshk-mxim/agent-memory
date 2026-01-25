# EXP-005: Output Validation

**Experiment**: Validate BlockPoolBatchEngine output matches reference
**Sprint**: 2 (Block-Pool Batch Engine)
**Status**: 📋 READY TO EXECUTE
**Date**: 2026-01-24

---

## Objective

Verify that BlockPoolBatchEngine generates identical output to reference mlx_lm implementation.

**Success Criteria**: Byte-identical output for same prompt/seed/temperature

---

## Hypothesis

Block-pool memory management does not affect generation correctness:
- Reconstructing cache from blocks → KVCache should preserve all data
- Batch inference with blocks should match sequential inference
- Output should be deterministic with same seed

---

## Method

### Reference Implementation

Use mlx_lm directly (no block pool):

```python
from mlx_lm import load, generate

model, tokenizer = load("mlx-community/SmolLM2-135M-Instruct")

# Deterministic generation
reference_text = generate(
    model,
    tokenizer,
    prompt="Hello, how are you?",
    max_tokens=50,
    temp=0.0,  # Deterministic
    repetition_penalty=1.0,
)

print(f"Reference: {reference_text}")
```

### Test Implementation

Use BlockPoolBatchEngine:

```python
from semantic.application.batch_engine import BlockPoolBatchEngine
from semantic.domain.services import BlockPool
from semantic.domain.value_objects import ModelCacheSpec

# Create engine
spec = ModelCacheSpec.from_model(model)
pool = BlockPool(spec=spec, total_blocks=100)
engine = BlockPoolBatchEngine(model, tokenizer, pool, spec)

# Generate
uid = engine.submit(
    agent_id="test",
    prompt="Hello, how are you?",
    max_tokens=50,
)

# Execute
for completion in engine.step():
    test_text = completion.text
    print(f"Test: {test_text}")
```

### Comparison

```python
# Token-level comparison
reference_tokens = tokenizer.encode(reference_text)
test_tokens = tokenizer.encode(test_text)

assert reference_tokens == test_tokens, "Token mismatch!"
assert reference_text == test_text, "Text mismatch!"

print("✅ Output validation PASSED")
```

---

## Test Cases

### Test 1: Short Prompt (No Cache)

**Prompt**: "Hello"
**Max Tokens**: 20
**Expected**: Identical output

### Test 2: Medium Prompt (Multi-Block)

**Prompt**: "The quick brown fox jumps over the lazy dog. This is a longer prompt to test multiple blocks."
**Max Tokens**: 50
**Expected**: Identical output

### Test 3: With Cache Resume

**Prompt 1**: "Hello, how are you?"
**Prompt 2**: " I'm doing well, thanks!" (with cache from Prompt 1)
**Expected**: Cache resume produces same continuation

---

## Variables

**Controlled**:
- Model: SmolLM2-135M-Instruct
- Temperature: 0.0 (deterministic)
- Seed: None (or set to fixed value)
- Max tokens: Fixed
- Repetition penalty: 1.0

**Measured**:
- Output text (string)
- Output tokens (list[int])
- Token count
- Finish reason

---

## Expected Results

| Test Case | Reference Output | Test Output | Match? |
|-----------|------------------|-------------|--------|
| Short prompt | "world" | "world" | ✅ |
| Medium prompt | "..." | "..." | ✅ |
| Cache resume | "..." | "..." | ✅ |

---

## Potential Issues

### Issue 1: Non-Determinism

**Problem**: Even with temp=0, output might vary due to floating-point precision

**Mitigation**:
- Set explicit seed if mlx_lm supports it
- Run multiple times, verify consistency
- Accept minor variation in final token if logits very close

### Issue 2: Cache Reconstruction Differences

**Problem**: Blocks → KVCache → Blocks might introduce artifacts

**Diagnosis**:
- Compare K/V tensor values directly
- Check for numerical precision loss
- Verify mx.concatenate preserves data exactly

### Issue 3: Batch vs Sequential Differences

**Problem**: Batch inference might use different kernels

**Diagnosis**:
- Test single-agent batch vs direct generation
- Compare intermediate hidden states
- Verify attention mask handling

---

## Execution Environment

**Requirements**:
- Apple Silicon (M1/M2/M3/M4)
- mlx==0.30.3
- mlx-lm==0.30.4
- 16GB+ RAM

**Setup**:
```bash
# Install dependencies
pip install mlx==0.30.3 mlx-lm==0.30.4

# Run experiment
python -m experiments.exp_005_validation
```

---

## Data Collection

Record for each test case:
- Reference output (text + tokens)
- Test output (text + tokens)
- Match status (✅/❌)
- Execution time (both implementations)
- Memory usage (both implementations)

---

## Analysis

### Success Metrics

- ✅ 100% match rate across all test cases
- ✅ Byte-identical output
- ✅ Token-identical output

### Acceptable Deviations

- ⚠️ Final token differs if logit difference < 0.01
- ⚠️ Whitespace differences (strip before compare)

### Failure Modes

- ❌ Different tokens generated
- ❌ Different finish_reason
- ❌ Crash or error

---

## Results

**Status**: ⏳ PENDING EXECUTION

**To Run**:
1. Set up MLX environment
2. Load SmolLM2-135M-Instruct
3. Execute reference + test implementations
4. Compare outputs
5. Record results in this section

**Expected Completion**: When MLX environment available

---

## Conclusion

**Hypothesis Validation**: ⏳ PENDING

**Next Steps**:
- Execute experiment with MLX
- Document results
- If validation fails, debug cache reconstruction
- If validation passes, proceed to EXP-006

---

**Prepared By**: ML (Machine Learning Engineer)
**Date**: 2026-01-24 (Sprint 2, Day 9)
**Status**: 📋 READY TO EXECUTE
