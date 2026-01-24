# RDIC Instruction Conflict Dataset

**Version:** 2.0 (Context Isolation Test)
**Created:** 2026-01-22
**Total Examples:** 100 (80 train, 20 test)
**Purpose:** Testing context isolation capabilities of RDIC framework

---

## Overview

This dataset contains 100 multi-turn conversations designed to test whether language models can maintain separate semantic contexts under KV cache compression. Each example presents conflicting instructions across turns and tests if the system can preserve both contexts independently.

### Research Goal

**Can RDIC prevent instruction degradation by isolating conflicting semantic contexts in separate KV caches?**

Without RDIC, KV compression causes later instructions to degrade earlier instructions. This dataset tests whether RDIC's context isolation can maintain both instruction contexts when Turn 3 requests both versions.

---

## Dataset Structure

### Files

| File | Examples | Purpose |
|------|----------|---------|
| `conflict_dataset.json` | 100 | Complete dataset (all batches combined) |
| `train.json` | 80 | Training set (80%) |
| `test.json` | 20 | Test set (20%) |
| `batch_001.json` | 30 | First batch (restructured from v1.0) |
| `batch_002.json` | 35 | Second batch (generated with v2.0 design) |
| `batch_003.json` | 35 | Third batch (generated with v2.0 design) |

### Example Structure

Each example contains:

```json
{
  "id": "conflict_001",
  "conflict_type": "tone_professional_vs_friendly",
  "domain": "legal_writing",
  "purpose": "context_isolation_test",
  "turns": [
    {
      "turn_id": 1,
      "role": "user",
      "instruction": "<first semantic context>",
      "content": "<scenario setup>"
    },
    {
      "turn_id": 2,
      "role": "user",
      "instruction": "<conflicting semantic context>",
      "content": "<context switch>"
    },
    {
      "turn_id": 3,
      "role": "user",
      "query": "<request for both versions separately>",
      "expected_behavior": "<what RDIC should maintain>",
      "rdic_value": "<why context isolation prevents degradation>"
    }
  ],
  "ground_truth_clusters": ["<context_1>", "<context_2>"],
  "metadata": {
    "generated_at": "2026-01-22T...",
    "model": "claude-sonnet-4-5-20250929",
    "restructured_at": "2026-01-22T..."
  }
}
```

### Turn Structure

- **Turn 1**: Establishes first semantic context with specific instruction/constraint
- **Turn 2**: Establishes second, conflicting semantic context with different instruction
- **Turn 3**: Requests both contexts separately (e.g., "Show me both versions"), testing if system maintained both contexts under KV compression

---

## Distribution Statistics

### Conflict Type Distribution

Perfect even distribution across all 5 conflict types:

| Conflict Type | Count | Percentage |
|---------------|-------|------------|
| **Tone** | 20 | 20% |
| **Detail** | 20 | 20% |
| **Style** | 20 | 20% |
| **Content** | 20 | 20% |
| **Format** | 20 | 20% |
| **TOTAL** | **100** | **100%** |

### Domain Distribution

Good coverage across 10 different domains:

| Domain | Count |
|--------|-------|
| academic_writing | 10 |
| business_email | 9 |
| code_comments | 12 |
| creative_writing | 13 |
| customer_support | 11 |
| educational_content | 6 |
| legal_writing | 10 |
| medical_records | 7 |
| social_media | 12 |
| technical_documentation | 10 |
| **TOTAL** | **100** |

### Train/Test Split Distribution

Random 80/20 split (seed=42):

| Type | Train | Test | Total |
|------|-------|------|-------|
| content | 17 | 3 | 20 |
| detail | 18 | 2 | 20 |
| format | 14 | 6 | 20 |
| style | 16 | 4 | 20 |
| tone | 15 | 5 | 20 |
| **TOTAL** | **80** | **20** | **100** |

---

## Conflict Types

### 1. Tone Conflicts (20 examples)

Conflicts in formality, professionalism, or emotional tone that create incompatible semantic contexts.

**Subtypes:**
- formal_vs_casual
- professional_vs_friendly
- serious_vs_humorous
- empathetic_vs_objective

**Example:**
- Turn 1: "Use strictly professional legal language"
- Turn 2: "Use warm, friendly conversational tone"
- Turn 3: "Show me both: professional version first, then friendly version"

### 2. Detail Conflicts (20 examples)

Conflicts in level of detail or comprehensiveness that create different information density contexts.

**Subtypes:**
- brief_vs_detailed
- concise_vs_comprehensive
- summary_vs_exhaustive
- minimal_vs_verbose

**Example:**
- Turn 1: "Keep responses brief - maximum 2-3 sentences"
- Turn 2: "Provide exhaustive, comprehensive details covering every scenario"
- Turn 3: "Give me both: brief version first, then comprehensive version"

### 3. Style Conflicts (20 examples)

Conflicts in writing style or audience level that create incompatible presentation contexts.

**Subtypes:**
- technical_vs_layperson
- academic_vs_conversational
- jargon_heavy_vs_simple
- formal_vs_informal

**Example:**
- Turn 1: "Use technical jargon and specialist terminology"
- Turn 2: "Use accessible language for general audience"
- Turn 3: "Show both: technical version first, then accessible version"

### 4. Content Conflicts (20 examples)

Conflicts in what information to include or exclude that create different content structure contexts.

**Subtypes:**
- citations_vs_no_citations
- examples_vs_no_examples
- opinions_vs_facts_only
- background_vs_direct_answer

**Example:**
- Turn 1: "Include no examples - conceptual explanation only"
- Turn 2: "Include multiple code examples and practical demonstrations"
- Turn 3: "Provide both: conceptual version first, then example-rich version"

### 5. Format Conflicts (20 examples)

Conflicts in presentation structure or organization that create incompatible formatting contexts.

**Subtypes:**
- structured_vs_freeform
- bullets_vs_paragraphs
- lists_vs_prose
- sections_vs_continuous

**Example:**
- Turn 1: "Use bullet points and structured lists"
- Turn 2: "Use flowing paragraph prose"
- Turn 3: "Show both: structured version first, then narrative version"

---

## Context Isolation Design (v2.0)

### What Changed from v1.0

**OLD Design (v1.0) - Unresolvable Merge:**
- Turn 3: "Write using BOTH formal AND casual tone"
- Problem: Impossible task - asks model to do the impossible
- Not useful for RDIC research

**NEW Design (v2.0) - Context Isolation Test:**
- Turn 3: "Show me both versions: formal first, then casual"
- Purpose: Test if RDIC can maintain both semantic contexts separately
- Aligned with RDIC research goals

### Why This Matters

**Without RDIC:**
- KV cache compression causes earlier instructions to degrade
- Turn 2 instruction overwrites Turn 1 instruction in compressed cache
- System "forgets" first constraint

**With RDIC:**
- Conflicting contexts isolated in separate KV caches
- Both instruction contexts preserved
- System can access both when Turn 3 requests both versions

---

## Validation

All 100 examples have passed structural validation:

- ✓ Valid structure: 100/100 (100%)
- ✓ Has purpose field: 100/100 (100%)
- ✓ Has expected_behavior: 100/100 (100%)
- ✓ Has RDIC value: 100/100 (100%)
- ✓ Ground truth clusters: 100/100 (100%)
- ✓ Context isolation queries: 100/100 (100%)

Validator: `src/validator.py`

---

## Usage

### Loading the Dataset

```python
import json

# Load complete dataset
with open('data/conflict_dataset.json', 'r') as f:
    dataset = json.load(f)

# Load train/test split
with open('data/train.json', 'r') as f:
    train_set = json.load(f)

with open('data/test.json', 'r') as f:
    test_set = json.load(f)
```

### Validating Examples

```bash
# Validate a batch
python src/validator.py data/batch_001.json

# Validate complete dataset
python src/validator.py data/conflict_dataset.json
```

### Generating New Examples

```python
from src.dataset_generator import ConflictDatasetGenerator

generator = ConflictDatasetGenerator()

example = generator.generate_example(
    conflict_type='tone',
    conflict_subtype='formal_vs_casual',
    domain='business_email'
)
```

**Note:** Generator now includes context isolation design in prompts (v2.0).

---

## Quality Assurance

### Manual Review Process

A subset of 10 examples (10%) were manually reviewed:

**Review Criteria:**
1. ✓ Genuine Conflict - Instructions create incompatible semantic contexts
2. ✓ Realistic Scenario - Could happen in real conversations
3. ✓ Tests Context Isolation - Turn 3 requires maintaining BOTH contexts separately
4. ✓ Clear Ground Truth - Semantic clusters are well-separated
5. ✓ RDIC Value - Context isolation would prevent instruction degradation

**Results:**
- All 10 reviewed examples passed all 5 criteria
- Quality score: 100% (10/10)

### Restructuring History

- **batch_001.json**: Restructured from v1.0 (unresolvable merge) to v2.0 (context isolation)
- **batch_002.json**: Generated with old prompt, restructured to v2.0
- **batch_003.json**: Generated with old prompt, restructured to v2.0

All restructuring performed via `restructure_all_batches.py` script.

---

## Related Files

- **Schema:** `conflict_schema.json` (v2.0)
- **Generator:** `src/dataset_generator.py` (updated for v2.0)
- **Validator:** `src/validator.py`
- **Restructure Tool:** `restructure_all_batches.py`
- **Review Guide:** `HOW_TO_REVIEW.md`
- **Review Checklist:** `DAY_2_MANUAL_REVIEW_CHECKLIST.md`
- **Review Report:** `batch_001_review.md`

---

## Version History

### v2.0 (2026-01-22)
- **Change:** Restructured to context isolation test design
- **Reason:** Original "unresolvable merge" design didn't align with RDIC research
- **Impact:** All 100 examples now test context isolation under KV compression
- **Migration:** All batches restructured via automated script

### v1.0 (2026-01-22)
- **Initial version:** 30 examples with unresolvable merge design
- **Deprecated:** Replaced with v2.0 on same day after realizing design flaw

---

## Citation

If you use this dataset, please cite:

```
RDIC Instruction Conflict Dataset v2.0
Context Isolation Testing for KV Cache Compression
Generated: 2026-01-22
Model: claude-sonnet-4-5-20250929
Total Examples: 100 (80 train, 20 test)
```

---

## Contact

For questions about this dataset:
- See `debate_plan.md` for RDIC research context
- See `MODEL_MIGRATION.md` for model selection rationale
- See `HOW_TO_REVIEW.md` for review methodology

---

**Dataset Status:** ✓ Complete and validated
**Last Updated:** 2026-01-22
**Version:** 2.0 (Context Isolation Test)
