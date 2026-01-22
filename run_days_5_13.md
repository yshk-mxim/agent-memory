# Automated Execution Plan: Days 5-13

## Overview
This document tracks the automated execution of Days 5-13 using parallel agents.

## Execution Strategy

### Day 4 (RUNNING)
- **Agent:** adaa272 (background)
- **Status:** In progress
- **Task:** Build evaluation framework (rule-based + LLM judge)

### Day 5: Experiment 1 - Compression Degradation
- **Dependencies:** Day 4 evaluators
- **Key tasks:**
  - Implement compression.py
  - Run baseline (full context) on 50 test examples with Gemma 3
  - Run compressed (50%) on same 50 examples
  - Evaluate and compare
  - Generate figures/tables

### Days 6-7: Buffer/Rest
- Validate all results
- Fix any issues
- Prepare for Week 2

### Day 8: DeepSeek R1 Clustering (Run 1)
- **API Required:** DeepSeek R1 (requires sandbox bypass)
- **Task:** Run R1 on full dataset to discover instruction clusters
- **Output:** Cluster taxonomy, conflict matrix

### Day 9: R1 Stability Testing (Runs 2-3)
- **API Required:** DeepSeek R1 (2 more runs)
- **Task:** Verify clustering stability
- **Metrics:** ARI, NMI between runs

### Day 10: Cluster Validation
- **Dependencies:** Days 8-9 R1 results
- **Task:** Compare R1 clusters to ground truth
- **Output:** Confusion matrix, alignment metrics, figures

### Day 11: Semantic Isolation Framework
- **Task:** Implement RDIC-style semantic isolation
- **Components:**
  - Router (embedding-based)
  - Turn-based baseline
  - Semantic isolation system

### Day 12: Experiment 3 - Isolation Comparison
- **Dependencies:** Day 11 framework
- **Task:** Compare semantic vs turn-based on 20 test examples
- **Output:** Performance metrics, statistical tests, figures

### Day 13: Buffer/Extended Analysis
- Review all Week 2 results
- Run ablations if time permits
- Begin paper outline

## Parallel Execution Plan

1. **Day 4 → Day 5** (sequential, Day 5 needs evaluators)
2. **Days 8-9** can be batched (3 R1 runs)
3. **Day 11** can start while Day 10 generates figures
4. **Day 12** sequential after Day 11

## Commit Points
- After Day 4 completion
- After Day 5 completion
- After Days 6-7 buffer
- After Days 8-10 (R1 clustering complete)
- After Days 11-12 (isolation framework complete)
- After Day 13 (Week 2 complete)
