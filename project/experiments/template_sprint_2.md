# EXP-XXX: [Experiment Title]

**Date**: YYYY-MM-DD
**Status**: ⏳ PENDING | 🔄 RUNNING | ✅ PASSED | ❌ FAILED
**Sprint**: 2 - Block-Pool Batch Engine
**Owner**: ML | QE | SE
**Duration**: X hours/days

---

## Objective

[1-2 sentences: What specific question does this experiment answer?]

**Example**: Prove that BlockPoolBatchEngine produces byte-identical output to reference mlx_lm.generate() implementation.

---

## Hypothesis

[What do we expect to happen? Be specific and measurable.]

**Example**:
- Output text will match reference exactly (byte-for-byte)
- Generation time will be within 20% of reference
- No memory leaks (pool size stable across 10 runs)

---

## Prerequisites

**Models Required**:
- [ ] SmolLM2-135M-Instruct (for fast testing)
- [ ] Gemma 3 12B (for production validation)
- [ ] Other: _______________

**Dependencies**:
- [ ] BlockPool implementation (Sprint 1)
- [ ] ModelCacheSpec implementation (Sprint 1)
- [ ] mlx_lm v0.30.4 installed
- [ ] Other: _______________

**Code Components**:
- [ ] Component A exists and tested
- [ ] Component B exists and tested
- [ ] Other: _______________

---

## Method

### Setup

1. **Environment**:
   ```python
   # Model and dependencies
   from mlx_lm import load
   model, tokenizer = load("mlx-community/SmolLM2-135M-Instruct")

   # Test components
   from semantic.domain.services import BlockPool
   from semantic.domain.value_objects import ModelCacheSpec
   ```

2. **Test Data**:
   ```python
   # Define test prompts
   test_prompts = [
       ("short", "The quick brown fox", 50),
       ("medium", "Write a story about a robot", 500),
       ("long", "Explain quantum computing in detail", 2000),
   ]
   ```

3. **Configuration**:
   ```python
   # Fixed parameters for reproducibility
   max_tokens = 100
   temperature = 0.0  # Greedy decoding
   seed = 42
   ```

### Execution Steps

**Step 1**: [First action]
```python
# Code snippet for Step 1
```

**Step 2**: [Second action]
```python
# Code snippet for Step 2
```

**Step 3**: [Third action]
```python
# Code snippet for Step 3
```

### Validation

**Criteria**:
- [ ] Criterion 1: [Specific, measurable condition]
- [ ] Criterion 2: [Specific, measurable condition]
- [ ] Criterion 3: [Specific, measurable condition]

**Measurement Method**:
```python
# How to measure success
def validate_result(actual, expected):
    assert actual == expected  # Byte-identical
    assert len(actual) > 0  # Non-empty output
    # ... additional checks
```

---

## Success Criteria

**Must Pass** (all required):
1. ✅ Criterion 1: [e.g., Output matches reference (100% accuracy)]
2. ✅ Criterion 2: [e.g., No errors or exceptions during execution]
3. ✅ Criterion 3: [e.g., Performance within acceptable range]

**Nice to Have** (optional):
- Criterion 4: [e.g., Execution time < 5 seconds]
- Criterion 5: [e.g., Memory usage < 1GB]

---

## Failure Plan

**If Experiment Fails**:

**Scenario A**: [Specific failure mode, e.g., "Output mismatch"]
- **Action**: [What to do, e.g., "Compare token-by-token to identify divergence point"]
- **Escalation**: [When to escalate, e.g., "If mismatch > 5%, escalate to PM"]

**Scenario B**: [Another failure mode, e.g., "Memory leak detected"]
- **Action**: [What to do]
- **Escalation**: [When to escalate]

**Critical Failure Trigger**: [Condition that stops experiment immediately]
- Example: "If 3+ consecutive runs fail, STOP and escalate to PM"

---

## Results

### Execution Log

**Run 1** (YYYY-MM-DD HH:MM):
```
[Paste console output]
```

**Run 2** (YYYY-MM-DD HH:MM):
```
[Paste console output]
```

### Data

| Run | Prompt | Expected Output | Actual Output | Match? | Time (s) | Memory (MB) |
|-----|--------|-----------------|---------------|--------|----------|-------------|
| 1   | short  | "..."           | "..."         | ✅     | 1.2      | 450         |
| 2   | medium | "..."           | "..."         | ✅     | 3.5      | 480         |
| 3   | long   | "..."           | "..."         | ✅     | 8.1      | 520         |

### Observations

**Positive Findings**:
- Finding 1: [What worked well]
- Finding 2: [Unexpected benefit]

**Issues Encountered**:
- Issue 1: [Problem + workaround]
- Issue 2: [Problem + resolution]

**Anomalies**:
- Anomaly 1: [Unexpected behavior + explanation]

---

## Analysis

### Hypothesis Validation

| Hypothesis Statement | Result | Evidence |
|---------------------|--------|----------|
| Hypothesis 1        | ✅ PASS | [Data supporting/refuting] |
| Hypothesis 2        | ❌ FAIL | [Data supporting/refuting] |
| Hypothesis 3        | ⚠️ PARTIAL | [Data supporting/refuting] |

### Root Cause Analysis (if failed)

**Problem**: [What went wrong]

**Investigation**:
1. Step 1: [What we checked]
2. Step 2: [What we found]
3. Step 3: [Conclusion]

**Root Cause**: [Final determination]

**Fix Applied**: [How we addressed it]

---

## Conclusions

### Summary

[2-3 sentences: High-level outcome of experiment]

### Decision

✅ **GO**: Proceed with implementation
❌ **NO-GO**: Block implementation until issue resolved
⚠️ **CONDITIONAL GO**: Proceed with caveats

**Rationale**: [Why this decision was made]

### Next Steps

1. **Immediate**: [Action to take right away]
2. **Sprint 2**: [Follow-up tasks for this sprint]
3. **Future**: [Defer to later sprint]

---

## Artifacts

**Code**:
- Experiment script: `/project/experiments/scripts/exp_XXX_script.py`
- Test fixtures: `/project/experiments/fixtures/exp_XXX_fixtures.py`
- Data files: `/project/experiments/data/exp_XXX_results.json`

**Documentation**:
- Related ADRs: ADR-XXX, ADR-YYY
- Related experiments: EXP-AAA, EXP-BBB
- External references: [Links to papers, docs, etc.]

---

## Metadata

**Git Commit**: [Commit SHA if code was checked in]
**Environment**:
- Python: 3.12.0
- mlx: [version]
- mlx_lm: 0.30.4
- Platform: macOS 14 (Apple Silicon)

**Execution Time**: [Total time from start to finish]
**Reviewer**: [Who reviewed and approved this experiment]
**Approval Date**: YYYY-MM-DD

---

**Status**: ⏳ PENDING
**Last Updated**: YYYY-MM-DD
