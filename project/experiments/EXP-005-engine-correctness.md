# EXP-005: BlockPoolBatchEngine Correctness Validation

**Date**: TBD (Day 8)
**Sprint**: 2 (Block-Pool Batch Engine)
**Owner**: ML (Machine Learning Engineer)
**Status**: ⏳ PENDING

---

## Objective

Prove that `BlockPoolBatchEngine` produces **byte-identical output** to reference `mlx_lm.generate()` implementation.

**Critical Requirement**: Output must match exactly (not just semantically similar) to ensure cache reconstruction is correct.

---

## Hypothesis

The BlockPoolBatchEngine wrapper around mlx_lm's BatchGenerator will produce identical output to the canonical `mlx_lm.generate()` when using:
- Same prompt
- Same model
- Same generation parameters (temperature=0.0, greedy decoding)
- Same random seed

Any divergence indicates a bug in block-to-cache reconstruction or cache-to-block extraction logic.

---

## Method

### Test Data

**Test Prompts** (3 prompt lengths):

| ID | Label | Prompt Text | Expected Tokens | Rationale |
|----|-------|-------------|-----------------|--------------|
| 1 | short | "The quick brown fox" | ~50 total | Tests minimal cache (1 block) |
| 2 | medium | "Write a story about a robot" | ~500 total | Tests medium cache (2 blocks) |
| 3 | long | "Explain quantum computing in detail" | ~2000 total | Tests large cache (8 blocks) |

**Fixed Parameters** (for reproducibility):
```python
max_tokens = 100  # Generate same number of tokens for all prompts
temperature = 0.0  # Greedy decoding (deterministic)
seed = 42  # Fixed random seed
model = "mlx-community/SmolLM2-135M-Instruct"  # Fast model for testing
```

### Reference Generation

Use canonical `mlx_lm.generate()` to create ground truth:

```python
from mlx_lm import load, generate

model, tokenizer = load("mlx-community/SmolLM2-135M-Instruct")

def generate_reference(prompt: str, max_tokens: int = 100) -> str:
    """Generate reference output using mlx_lm.generate()."""
    result = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        temp=0.0,  # Greedy (deterministic)
        verbose=False,
    )
    return result
```

### Test Generation

Use `BlockPoolBatchEngine.submit()` and `step()`:

```python
from semantic.domain.services import BlockPool
from semantic.domain.value_objects import ModelCacheSpec
from semantic.application.batch_engine import BlockPoolBatchEngine

spec = ModelCacheSpec.from_model(model)
pool = BlockPool(spec, total_blocks=100)
engine = BlockPoolBatchEngine(model, tokenizer, pool, spec)

def generate_test(prompt: str, max_tokens: int = 100) -> str:
    """Generate output using BlockPoolBatchEngine."""
    uid = engine.submit(
        agent_id="test_agent",
        prompt=prompt,
        cache=None,  # No cache (fresh generation)
        max_tokens=max_tokens,
    )

    # Poll for completion
    for completion in engine.step():
        if completion.uid == uid:
            return completion.text

    raise RuntimeError(f"Generation {uid} did not complete")
```

---

## Validation Criteria

### Primary Criteria (MUST PASS)

1. **✅ Byte-Identical Output**
   ```python
   def validate_output(actual: str, expected: str, prompt_label: str) -> bool:
       """Validate output matches reference byte-for-byte."""
       if actual != expected:
           print(f"❌ {prompt_label}: Output mismatch")
           print(f"   Expected: '{expected[:50]}...'")
           print(f"   Actual:   '{actual[:50]}...'")
           return False
       return True
   ```
   **Success Criteria**: `actual == expected` (100% match)

2. **✅ No Errors During Generation**
   - No exceptions raised during submit() or step()
   - All completions have finish_reason in ["stop", "length"]

3. **✅ Correct Token Count**
   ```python
   def validate_token_count(completion, prompt: str, max_tokens: int) -> bool:
       """Validate token count is within expected range."""
       prompt_tokens = len(tokenizer.encode(prompt))
       total_tokens = completion.token_count
       generated_tokens = total_tokens - prompt_tokens

       assert generated_tokens <= max_tokens
       assert generated_tokens > 0
       return True
   ```
   **Success Criteria**: Generated tokens ∈ [1, max_tokens]

### Secondary Criteria (NICE TO HAVE)

4. **Performance Within 20% of Reference**
   - Compare end-to-end latency
   - BlockPoolBatchEngine should be within 20% of mlx_lm.generate()

5. **No Memory Leaks**
   ```python
   # Run 10 consecutive generations
   initial_used = pool.used_blocks()
   for _ in range(10):
       result = generate_test(prompt, max_tokens)
       pool.free_agent_blocks("test_agent")
   final_used = pool.used_blocks()

   assert final_used == initial_used  # No leaked blocks
   ```

---

## Success Criteria

**GO Criteria** (all required):
- ✅ 3/3 prompts produce byte-identical output
- ✅ No exceptions or errors during generation
- ✅ Token counts within expected range

**Stretch Goals** (optional):
- ✅ Performance within 20% of reference
- ✅ No memory leaks after 10 runs

---

## Failure Analysis

**If EXP-005 FAILS**:

1. **Action 1**: Compare token-by-token to identify divergence point
   ```python
   ref_tokens = tokenizer.encode(reference_output)
   test_tokens = tokenizer.encode(test_output)
   for i, (r, t) in enumerate(zip(ref_tokens, test_tokens)):
       if r != t:
           print(f"Divergence at position {i}: ref={r}, test={t}")
           break
   ```

2. **Action 2**: Validate cache reconstruction logic
   - Inspect blocks → KVCache conversion
   - Check tensor shapes at each layer
   - Verify mx.concatenate correctness

3. **Action 3**: Check for floating-point precision issues
   - Compare cache tensors numerically (tolerance: 1e-6)
   - Verify mx.eval() is called (lazy evaluation issue)

4. **Escalation**: If mismatch > 5 tokens, escalate to PM (BLOCKING issue)

---

## Expected Results

**Predicted Outcome**: ✅ PASS

**Rationale**:
- EXP-003 (Sprint 0) validated cache injection works
- EXP-004 (Sprint 0) validated cache extraction works
- BlockPool allocate/free is working (EXP-002 passed)
- mlx_lm BatchGenerator API is well-understood

**Risk**: LOW (validated components)

---

## Deliverables

1. **Results Document**: This file updated with PASS/FAIL status
2. **Reference Outputs**: `/project/experiments/data/exp_005_references.json`
3. **Validation Log**: `/project/experiments/data/exp_005_results.csv`
4. **Report**: Summary for sprint review

---

## Dependencies

**Blocked By**:
- BlockPoolBatchEngine.submit() implementation (Day 6-7)
- BlockPoolBatchEngine.step() implementation (Day 8-9)
- BlockPoolBatchEngine._reconstruct_cache() implementation (Day 7-8)

**Blocks**:
- Sprint 2 completion (exit gate depends on this)
- Integration test suite (needs byte-identical guarantee)

---

## Notes

- Use SmolLM2-135M for speed (full generation in ~2-3 seconds)
- Temperature=0.0 ensures deterministic output (greedy decoding)
- Cache=None for this experiment (testing fresh generation first)
- Follow-up experiment: EXP-007 (cache resume correctness) can test with cache

---

**Status**: ⏳ PENDING (Day 8)
**Last Updated**: 2026-01-24 (stub created Day 3)
