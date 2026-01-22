# How to Review the Day 2 Dataset

**File to Review**: `REVIEW_10_EXAMPLES.txt`
**Checklist to Fill**: `DAY_2_MANUAL_REVIEW_CHECKLIST.md`

---

## What Changed (Important!)

The dataset has been **restructured** from the original design:

### ❌ OLD Design (Unresolvable Merge)
```
Turn 3: "Write using BOTH formal AND casual tone"
Problem: Impossible to satisfy - can't do both at once
```

### ✅ NEW Design (Context Isolation Test)
```
Turn 3: "Provide two versions: formal first, then casual"
Purpose: Tests if RDIC can maintain both contexts separately
```

**Why This Matters**: The old design tested if models could do the impossible. The new design tests if RDIC can prevent instruction degradation under KV cache compression.

---

## How to Review Each Example

For each of the 10 examples in `REVIEW_10_EXAMPLES.txt`, check:

### 1. Genuine Conflict ✓/✗
**Question**: Do Turn 1 and Turn 2 create incompatible semantic contexts?

**Example**:
- Turn 1: "Use formal professional language"
- Turn 2: "Use casual friendly language"
- ✓ These create different semantic spaces

### 2. Realistic Scenario ✓/✗
**Question**: Could this context-switching happen in real conversations?

**Example**:
- User first wants formal letter, then realizes recipient is a friend
- ✓ Realistic change of requirements

### 3. Tests Context Isolation ✓/✗
**Question**: Does Turn 3 require maintaining BOTH contexts separately?

**Example**:
- Turn 3: "Give me both versions: formal first, then casual"
- ✓ Requires system to remember both instruction contexts

### 4. Clear Ground Truth ✓/✗
**Question**: Are the semantic clusters well-separated?

**Example**:
- Clusters: `["formal_professional", "casual_friendly"]`
- ✓ Clear semantic separation

### 5. RDIC Value ✓/✗
**Question**: Would isolating KV contexts prevent instruction degradation?

**Read the "WHAT THIS TESTS" and "RDIC VALUE" fields**

**Example**:
- "Under KV cache compression, the earlier formal instruction often degrades when casual instruction appears"
- ✓ RDIC's context isolation would prevent this degradation

---

## What You're Looking For

### ✅ GOOD Examples

**Example that PASSES**:
```
Turn 1: "Be formal" → Creates formal_context
Turn 2: "Be casual" → Creates casual_context
Turn 3: "Show both versions" → Requires accessing both contexts

Without RDIC: KV compression causes system to forget "be formal"
With RDIC: Both contexts isolated, both accessible
```

### ❌ BAD Examples

**Example that FAILS**:
```
Turn 1: "Be helpful"
Turn 2: "Be thorough"
Turn 3: "Write a helpful, thorough response"

Problem: These aren't conflicting - they can coexist easily
No context isolation needed
```

---

## Review Process

### Step 1: Read the Example
Open `REVIEW_10_EXAMPLES.txt` and read Example 1 completely:
- All 3 turns
- Ground truth clusters
- "WHAT THIS TESTS" field
- "RDIC VALUE" field

### Step 2: Check Each Criterion
For Example 1, check:
- [ ] 1. Genuine Conflict?
- [ ] 2. Realistic Scenario?
- [ ] 3. Tests Context Isolation?
- [ ] 4. Clear Ground Truth?
- [ ] 5. RDIC Value?

### Step 3: Mark PASS or FAIL
- **PASS** = All 5 criteria ✓
- **FAIL** = Any criterion ✗

### Step 4: Fill the Checklist
Open `DAY_2_MANUAL_REVIEW_CHECKLIST.md` and fill in your answers.

### Step 5: Repeat for All 10
Do this for Examples 1-10.

### Step 6: Count Results
- Need ≥7/10 PASS to meet 70% target
- If <7/10 PASS, we need to iterate on prompts

---

## Key Differences from Original Design

| Aspect | OLD (Unresolvable) | NEW (Context Isolation) |
|--------|-------------------|------------------------|
| Turn 3 Query | "Use BOTH formal AND casual" | "Show both: formal first, then casual" |
| Tests | Can model do impossible? | Can RDIC prevent degradation? |
| Expected Result | Failure (impossible task) | Success (if contexts isolated) |
| RDIC Value | None (task is impossible) | High (prevents instruction loss) |
| Baseline Comparison | N/A | Baseline forgets formal, RDIC maintains both |

---

## What Success Looks Like

### If 7-10 Examples PASS:
✅ Dataset is ready for RDIC experiments
✅ Examples test context isolation effectively
✅ Proceed to Day 3 (scale to 300+ examples)

### If <7 Examples PASS:
⚠️ Need to iterate on generation prompts
⚠️ Regenerate batch with improved prompts
⚠️ Review again before proceeding

---

## Quick Reference: The 5 Criteria

1. **Genuine Conflict**: Incompatible semantic contexts?
2. **Realistic Scenario**: Could happen in real conversations?
3. **Tests Context Isolation**: Turn 3 requires both contexts?
4. **Clear Ground Truth**: Well-separated semantic clusters?
5. **RDIC Value**: Would context isolation prevent degradation?

**All 5 must be ✓ for example to PASS**

---

## Questions?

- Check `REVIEW_10_EXAMPLES.txt` for full example details
- Check `debate_plan.md` for RDIC research context
- Check `MODEL_MIGRATION.md` for why we're using Gemma 3 12B

---

**Start reviewing now!** Open `REVIEW_10_EXAMPLES.txt` and `DAY_2_MANUAL_REVIEW_CHECKLIST.md` side by side.
