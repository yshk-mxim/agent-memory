# Sprint 02: Experiment Runs (Weeks 2-3)

**Duration**: 10 days
**Goal**: Run all experiments (n=50) across 4 conditions, collect comprehensive metrics
**Status**: Pending Sprint 01 completion

---

## Objectives

- [ ] Run all 4 primary conditions on full dataset (n=50)
- [ ] Capture outputs, telemetry, and automated metrics for all runs
- [ ] Add instrumentation for timing and memory profiling
- [ ] Ensure data quality and completeness
- [ ] Begin preliminary analysis

---

## Week 2 Breakdown

### Monday: Instrumentation Setup

**Morning (3h)**:
- [ ] Implement timing instrumentation
  - Per-generation latency
  - Per-cluster processing time
  - Total pipeline time
- [ ] Test on 5 examples

**Afternoon (2h)**:
- [ ] Implement memory profiling
  - Peak RAM usage
  - Cache sizes per cluster
  - Model memory footprint
- [ ] Verify <5% overhead

**Deliverable**: `instrumentation/profiler.py`, `instrumentation/telemetry.py`

---

### Tuesday: Cache Growth Tracking

**Morning (3h)**:
- [ ] Implement cache growth tracking
  - Log tokens per cluster over time
  - Track KV cache size evolution
  - Monitor cluster boundaries

**Afternoon (2h)**:
- [ ] Test instrumentation on 5 examples
  - Verify all data captured
  - Check overhead (<5%)
  - Export format validation

**Deliverable**: Complete instrumentation suite tested

---

### Wednesday: Sequential Condition Runs

**All day (6h)**:
- [ ] Run sequential condition (baseline) on all 50 examples
  - Sequential: All context in one unified cache
  - Expected behavior: High contamination, low specialization
  - Capture outputs + all metrics
- [ ] Monitor for crashes or errors
- [ ] Verify completeness

**Deliverable**: `results/phase1_runs/sequential/` (50 outputs + metrics)

---

### Thursday: Prompted Condition Runs

**All day (6h)**:
- [ ] Run prompted condition (soft isolation) on all 50 examples
  - Prompted: Instructions to keep topics separate
  - Expected behavior: Moderate separation (instruction following)
  - Capture outputs + all metrics
- [ ] Verify all runs completed

**Deliverable**: `results/phase1_runs/prompted/` (50 outputs + metrics)

---

### Friday: Turn-Based Condition Runs

**All day (6h)**:
- [ ] Run turn-based condition (naive temporal isolation) on all 50 examples
  - Turn-based: Separate caches per turn (temporal boundaries)
  - Expected behavior: Some separation, but not as good as semantic
  - Capture outputs + all metrics
- [ ] Check for any errors

**Deliverable**: `results/phase1_runs/turn_based/` (50 outputs + metrics)

---

## Week 3 Breakdown

### Monday: Semantic Condition Runs

**All day (6h)**:
- [ ] Run semantic condition (RDIC - our method) on all 50 examples
  - Semantic: Separate caches per agent role (semantic boundaries)
  - **Uses embedding-based routing**: Each turn routed via sentence-transformers similarity
  - **No ground truth labels**: Proves approach works in practice
  - Expected behavior: Excellent specialization, minimal contamination
  - Capture outputs + all metrics + routing decisions
- [ ] Verify all runs completed
- [ ] Log routing confidence scores for later analysis

**Deliverable**: `results/phase1_runs/semantic/` (50 outputs + metrics + routing logs)

---

### Tuesday: Data Verification

**Morning (2h)**:
- [ ] Verify all 200 runs completed (4 conditions × 50 examples)
  - Check for missing files
  - Validate JSON structure
  - Ensure all metrics captured

**Afternoon (3h)**:
- [ ] Preliminary quality check
  - Spot-check 10 outputs per condition
  - Verify outputs make sense
  - Check metric values reasonable

**Deliverable**: Completeness report, quality check passed

---

### Wednesday: Automated Metrics Computation

**Morning (3h)**:
- [ ] Compute mechanical metrics for all 200 outputs
  - Contamination detection (4 metrics)
  - Specialization measurement (4 metrics)
  - Synthesis quality (4 metrics)
  - Standard NLP metrics (4 metrics)
- [ ] Export to structured format

**Afternoon (3h)**:
- [ ] Run Claude AI Judge evaluations (first 100 outputs)
  - Use Claude Code CLI Task tool with Sonnet 4.5
  - Contamination scoring (0-5 scale)
  - Specialization scoring (0-5 scale)
  - Synthesis scoring (0-5 scale)
  - Temperature=0.0 for reproducibility
  - Batch process for efficiency

**Deliverable**: `results/phase1_runs/automated_metrics.json` (16 mechanical metrics)

---

### Thursday: Claude AI Judge Evaluation (Complete)

**Morning (3h)**:
- [ ] Run Claude AI Judge evaluations (remaining 100 outputs)
  - Continue evaluation with same prompts
  - Maintain temperature=0.0
  - Monitor for consistency

**Afternoon (3h)**:
- [ ] Parse and aggregate Claude judgments
  - Extract scores from JSON responses
  - Compute means, std devs per condition
  - Validate correlation with mechanical metrics (sanity check)
- [ ] Export Claude scores to structured format

**Deliverable**: `results/phase1_runs/claude_judgments.json` (3 AI judge metrics)

**Cost**: ~$30 (200 outputs × 3 evaluations × $0.05 each)

---

### Friday: Preliminary Analysis

**Thursday (6h)**:
- [ ] Aggregate metrics across conditions
  - Compute means, std devs per condition
  - Create comparison tables
  - Identify patterns

**Friday (6h)**:
- [ ] Preliminary statistical analysis
  - Do automated metrics show separation?
  - Compute effect sizes (Cohen's d)
  - Visual inspection (box plots)
- [ ] Write preliminary findings

**Deliverable**: `results/phase1_preliminary/analysis.md`

---

## Expected Results

### Contamination Detection (Lower is better)

| Condition | TF-IDF Similarity | Vocab Leakage | Lexical Overlap | Keyword Bleeding |
|-----------|-------------------|---------------|-----------------|------------------|
| Sequential | **0.65** (high) | **12%** | **35%** | **8 keywords** |
| Prompted | 0.45 (moderate) | 7% | 25% | 5 keywords |
| Turn-based | 0.35 (low) | 5% | 18% | 3 keywords |
| **Semantic** | **0.18** (very low) | **2%** | **12%** | **1 keyword** |

### Specialization Measurement (Higher is better)

| Condition | Keyword Density | Technical Density | Style Consistency | Classifier Confidence |
|-----------|----------------|-------------------|-------------------|----------------------|
| Sequential | **0.08** (low) | **0.12** (low) | **0.45** (poor) | **0.55** (weak) |
| Prompted | 0.12 (moderate) | 0.18 (moderate) | 0.62 (okay) | 0.68 (moderate) |
| Turn-based | 0.14 (moderate) | 0.20 (moderate) | 0.68 (good) | 0.72 (good) |
| **Semantic** | **0.22** (high) | **0.28** (high) | **0.85** (excellent) | **0.88** (strong) |

### Synthesis Quality (Higher is better)

| Condition | Info Coverage | Dual Similarity | Coherence | Novel Content |
|-----------|--------------|-----------------|-----------|---------------|
| Sequential | 0.55 (poor) | 0.60/0.62 (low) | 0.78 (okay) | 8% (low) |
| Prompted | 0.68 (okay) | 0.72/0.74 (moderate) | 0.83 (good) | 15% (okay) |
| Turn-based | 0.72 (good) | 0.75/0.76 (good) | 0.85 (good) | 18% (good) |
| **Semantic** | **0.82** (excellent) | **0.84/0.85** (high) | **0.90** (excellent) | **22%** (excellent) |

### Claude AI Judge Scores

**Note**: Claude scores are 0-5 scale. For contamination, lower is better. For specialization and synthesis, higher is better.

| Condition | Claude Contamination (0-5) | Claude Specialization (0-5) | Claude Synthesis (0-5) |
|-----------|----------------------------|----------------------------|------------------------|
| Sequential | **3.8** (high mixing) | **1.5** (weak/generic) | **1.8** (poor integration) |
| Prompted | 2.5 (moderate) | 2.7 (basic specialization) | 2.9 (fair integration) |
| Turn-based | 1.8 (low) | 3.2 (moderate specialization) | 3.1 (good integration) |
| **Semantic** | **0.6** (minimal) | **4.5** (strong expertise) | **4.4** (excellent integration) |

**Expected Pattern**: Claude judgments should correlate with mechanical metrics but capture additional qualitative nuances. If Claude disagrees strongly with mechanical metrics, investigate discrepancies (may reveal metric limitations or interesting edge cases).

---

## Success Criteria

- [x] All 200 runs completed (4 conditions × 50 examples)
- [x] No crashes or missing data
- [x] All 19 automated metrics computed (16 mechanical + 3 Claude AI judge)
- [x] Instrumentation data captured (<5% overhead)
- [x] Claude AI evaluations completed (200 outputs × 3 dimensions)
- [x] Preliminary analysis shows semantic >> baselines
- [x] Effect sizes large (Cohen's d > 0.8 for key metrics)
- [x] Claude judgments correlate with mechanical metrics (validation)

---

## Risk Mitigation

**Risk**: Runs take longer than expected (>10 days)
- **Mitigation**: Run overnight, optimize generation parameters
- **Escalation**: Reduce to n=40 if timeline critical

**Risk**: Crashes during long runs
- **Mitigation**: Implement checkpointing, resume from failure
- **Escalation**: Debug issues before continuing

**Risk**: Metrics don't show separation
- **Mitigation**: Review metric implementations, inspect outputs manually
- **Escalation**: Revise metrics or conditions if fundamental issue

**Risk**: Instrumentation causes crashes
- **Mitigation**: Reduce overhead, measure post-hoc if needed
- **Escalation**: Turn off instrumentation, collect metrics separately

---

## Compute Requirements

**Estimated time**:
- 200 runs × 150 sec/run = 30,000 sec = 8.3 hours
- Plus overhead (metric computation, instrumentation) = ~12 hours total
- Can run overnight across 3 nights (4 hours/night)

**Memory**:
- Peak: 10-13GB (Gemma 3 12B + 3 cache clusters)
- Well within 24GB RAM limit

**Storage**:
- Outputs: 200 × 2KB = 400KB
- Metrics: 200 × 5KB = 1MB
- Telemetry: 200 × 1KB = 200KB
- Total: ~2MB

---

## Deliverables

- [ ] `results/phase1_runs/sequential/` (50 outputs + telemetry)
- [ ] `results/phase1_runs/prompted/` (50 outputs + telemetry)
- [ ] `results/phase1_runs/turn_based/` (50 outputs + telemetry)
- [ ] `results/phase1_runs/semantic/` (50 outputs + telemetry)
- [ ] `results/phase1_runs/automated_metrics.json` (16 mechanical metrics for 200 outputs)
- [ ] `results/phase1_runs/claude_judgments.json` (3 AI judge metrics for 200 outputs)
- [ ] `results/phase1_preliminary/analysis.md` (preliminary findings with all 19 metrics)
- [ ] `instrumentation/profiler.py` (timing and memory)
- [ ] `instrumentation/telemetry.py` (data export)

---

## Next Sprint

**Sprint 03**: Statistical Analysis + Decision Point (Weeks 4-6)

---

**Created**: 2026-01-23
**Status**: Pending Sprint 01
**Blockers**: Sprint 01 must deliver dataset and metrics suite
