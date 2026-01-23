# Updated Development Plan v1: Virtual Multi-Agent via Semantic KV Cache Partitioning

**Date**: 2026-01-22
**Status**: Initial plan based on debate consensus
**Source**: DEBATE_FINAL_CONSENSUS.md + NOVELTY.md

---

## Executive Summary

**Goal**: Develop and rigorously evaluate a virtual multi-agent system within a single LLM via semantic KV cache partitioning, achieving 3X memory efficiency vs true multi-agent systems.

**Timeline**: 10-12 weeks to publication-ready submission
**Hardware**: Mac with 24GB RAM (MLX framework)
**Model**: Gemma 3 12B (4-bit quantization, ~7-10GB)
**Budget**: $50-75 for evaluation APIs (Claude Haiku as judge)

**Key Innovation**: Enable agent-level specialization and coordination within single model's KV cache, preventing cross-contamination while maintaining synthesis capability through coordinator pattern.

---

## Phase 1: Rigorous Evaluation Foundation (Weeks 1-5)

### Week 1: Dataset Generation and Expansion

**Objectives**:
- Expand from n=1 to n=50 diverse examples
- Cover 5 domains (not just software engineering)
- Ensure genuine multi-agent scenarios

**Tasks**:

| Day | Task | Time | Details |
|-----|------|------|---------|
| Mon | Design domain taxonomy | 3h | 5 domains: coding, research, business, customer support, creative |
| Mon | Create generation prompts | 2h | Claude Sonnet prompts for each domain |
| Mon | Generate coding examples (10) | 2h | Multi-file debugging + docs + review |
| Tue | Generate research examples (10) | 3h | Literature review + experiment + writing |
| Tue | Generate business examples (10) | 3h | Technical + strategy + synthesis |
| Wed | Generate support examples (10) | 3h | Technical support + billing + account |
| Wed | Generate creative examples (10) | 3h | Storytelling + editing + analysis |
| Thu | Validation and refinement | 4h | Manual review, regenerate if needed |
| Thu | Create train/val/test splits | 1h | 30 train, 10 val, 10 test |
| Fri | Document dataset statistics | 2h | Domain distribution, cluster analysis |

**Deliverables**:
- `data/virtual_agents_dataset_v1.json` (50 examples)
- `data/dataset_statistics.md` (comprehensive analysis)
- Train/val/test splits with balanced domains

**Success Criteria**:
- [ ] 50 valid examples across 5 domains
- [ ] Each example has 3 distinct agent roles
- [ ] Clear semantic boundaries between agents
- [ ] Manual validation >80% quality

---

### Week 2: Evaluation Framework Development

**Objectives**:
- Implement blind evaluation protocol
- Recruit independent raters
- Develop automated metrics
- Create evaluation rubric

**Tasks**:

| Day | Task | Time | Details |
|-----|------|------|---------|
| Mon | Design evaluation rubric | 3h | Criteria for specialization, synthesis, contamination |
| Mon | Implement blinding system | 2h | Anonymize conditions, randomize presentation |
| Tue | Recruit 3 independent raters | 3h | Graduate students or researchers |
| Tue | Rater training session | 2h | Explain rubric, practice on golden examples |
| Wed | Implement automated metrics | 4h | ROUGE, BERTScore, semantic similarity |
| Thu | Create evaluation pipeline | 4h | Batch processing, inter-rater reliability |
| Fri | Pilot evaluation on 5 examples | 3h | Test process, measure agreement |
| Fri | Refine rubric based on pilot | 2h | Address ambiguities, improve clarity |

**Evaluation Rubric (0-5 scale)**:

**Agent Specialization** (intra-cluster coherence):
- 0: No coherence, random content
- 3: Generally on-topic with occasional drift
- 5: Perfect specialization, no off-topic content

**Cross-Contamination** (inter-cluster separation):
- 0: Complete mixing, no boundaries
- 3: Some contamination, mostly separated
- 5: Zero contamination, clear boundaries

**Synthesis Quality** (coordinator effectiveness):
- 0: No integration, just concatenation
- 3: Basic integration, identifies some connections
- 5: Deep synthesis, strategic insights

**Deliverables**:
- `evaluation/rubric.md` (detailed scoring criteria)
- `evaluation/rater_training.md` (training materials)
- `evaluation/automated_metrics.py` (ROUGE, BERTScore, etc.)
- `evaluation/pipeline.py` (end-to-end evaluation)

**Success Criteria**:
- [ ] 3 trained independent raters recruited
- [ ] Cohen's kappa >0.6 (substantial agreement)
- [ ] Automated metrics correlate with human scores
- [ ] Evaluation pipeline processes 50 examples in <4 hours

---

### Week 3: Instrumentation and Measurement

**Objectives**:
- Add latency instrumentation
- Measure memory usage accurately
- Track cache sizes dynamically
- Implement logging and telemetry

**Tasks**:

| Day | Task | Time | Details |
|-----|------|------|---------|
| Mon | Design instrumentation architecture | 2h | What to measure, where to inject |
| Mon | Implement timing decorators | 3h | Generation time, switch overhead |
| Tue | Implement memory profiling | 3h | Peak RAM, cache sizes per cluster |
| Tue | Add dynamic cache tracking | 2h | Log cache growth over turns |
| Wed | Implement telemetry system | 4h | Structured logging, metrics export |
| Thu | Test instrumentation overhead | 2h | Ensure <5% performance impact |
| Thu | Create visualization tools | 3h | Plot cache sizes, latency over time |
| Fri | Run instrumented pilot on 5 examples | 2h | Verify data collection |
| Fri | Document instrumentation | 2h | How to interpret metrics |

**Metrics to Collect**:

**Latency**:
- Total generation time per condition
- Per-turn generation latency
- Agent switch overhead (semantic only)
- Coordinator synthesis time

**Memory**:
- Peak RAM usage
- Cache size per cluster (tokens)
- Cache growth rate per turn
- Memory footprint vs true multi-agent

**Quality** (from evaluation):
- Specialization score per agent
- Cross-contamination percentage
- Synthesis quality score

**Deliverables**:
- `instrumentation/profiler.py` (timing and memory)
- `instrumentation/telemetry.py` (logging system)
- `instrumentation/visualize.py` (plotting tools)
- `instrumentation/README.md` (interpretation guide)

**Success Criteria**:
- [ ] All metrics collected automatically
- [ ] Instrumentation overhead <5%
- [ ] Visualization tools generate publication-quality figures
- [ ] Memory measurements accurate to ±50MB

---

### Week 4: Run Rigorous Evaluation

**Objectives**:
- Run all 4 conditions on 50 examples
- Collect all metrics (quality, latency, memory)
- Ensure blind evaluation process
- Generate statistical analyses

**Tasks**:

| Day | Task | Time | Details |
|-----|------|------|---------|
| Mon | Run sequential condition (baseline) | 4h | All 50 examples |
| Tue | Run prompted condition (soft isolation) | 4h | All 50 examples |
| Wed | Run turn-based condition (naive) | 4h | All 50 examples |
| Thu | Run semantic condition (RDIC) | 4h | All 50 examples |
| Fri | Export anonymized outputs for raters | 2h | Blind, randomized |
| Fri | Distribute to 3 raters | 1h | Email with instructions |

**Weekend**: Raters evaluate (200 outputs × 3 raters = 600 ratings)

**Deliverables**:
- `results/exp_full_evaluation/` (all condition outputs)
- `results/exp_full_evaluation/metrics.json` (automated metrics)
- `results/exp_full_evaluation/telemetry/` (logs, cache sizes)
- `results/exp_full_evaluation/blinded_outputs/` (for raters)

**Success Criteria**:
- [ ] All 50 examples run across 4 conditions (200 runs total)
- [ ] No crashes or errors
- [ ] All metrics collected
- [ ] Raters receive blinded outputs

---

### Week 5: Statistical Analysis and Results

**Objectives**:
- Collect and aggregate ratings
- Compute inter-rater reliability
- Run statistical tests (t-tests, effect sizes)
- Generate tables and figures

**Tasks**:

| Day | Task | Time | Details |
|-----|------|------|---------|
| Mon | Collect ratings from 3 raters | 1h | Download, parse |
| Mon | Compute inter-rater reliability | 2h | Cohen's kappa, Fleiss' kappa |
| Mon | Aggregate ratings (mean per example) | 1h | Average across raters |
| Tue | Statistical testing | 4h | Paired t-tests, Wilcoxon, Bonferroni |
| Wed | Effect size calculations | 2h | Cohen's d, confidence intervals |
| Wed | Domain-specific analysis | 3h | Which domains benefit most? |
| Thu | Generate tables | 3h | LaTeX tables with all statistics |
| Thu | Generate figures | 3h | Box plots, bar charts, heatmaps |
| Fri | Write results section draft | 4h | Section 5 of paper |
| Fri | Review and revise | 2h | Check all numbers |

**Statistical Tests**:

**Primary Hypothesis** (Semantic vs Sequential):
- Paired t-test (specialization, contamination, synthesis)
- Cohen's d (effect size)
- 95% confidence intervals

**Secondary Hypothesis** (Semantic vs Prompted):
- Tests whether architectural isolation beats instructional
- Critical for novelty claim

**Tertiary Hypothesis** (Semantic vs Turn-Based):
- Tests whether semantic beats temporal boundaries

**Multiple Comparison Correction**:
- Bonferroni correction (4 conditions → 6 pairwise comparisons)
- Report adjusted p-values

**Deliverables**:
- `results/exp_full_evaluation/statistics.json` (all p-values, effect sizes)
- `results/exp_full_evaluation/tables/` (LaTeX tables)
- `results/exp_full_evaluation/figures/` (publication-quality plots)
- `paper/sections/05_results.md` (results section draft)

**Success Criteria**:
- [ ] Cohen's kappa >0.6 (inter-rater agreement)
- [ ] Semantic significantly beats sequential (p<0.05, d>0.5)
- [ ] Semantic significantly beats prompted (p<0.05, d>0.3)
- [ ] All figures at 300 DPI, publication-ready

---

## Phase 2: Comparative Benchmarking (Weeks 6-9)

### Week 6: True Multi-Agent Baseline

**Objectives**:
- Implement true 3-agent system (separate models)
- Measure memory usage accurately
- Compare quality against virtual multi-agent
- Validate 3X memory efficiency claim

**Tasks**:

| Day | Task | Time | Details |
|-----|------|------|---------|
| Mon | Design true multi-agent architecture | 2h | 3 separate Gemma 3 12B instances |
| Mon | Implement agent communication | 3h | Pass outputs between agents |
| Tue | Test on 5 examples | 3h | Verify works end-to-end |
| Tue | Measure memory footprint | 2h | Peak RAM per agent |
| Wed | Run on 10 test examples | 4h | Full pipeline |
| Thu | Evaluate quality (blind) | 3h | Same rubric as Phase 1 |
| Thu | Compare vs semantic condition | 2h | Quality parity? |
| Fri | Memory efficiency analysis | 3h | 3X claim validation |
| Fri | Write comparison section | 2h | Document findings |

**Memory Comparison**:
- True multi-agent: 3 × model_size = ?GB
- Virtual multi-agent (semantic): 1 × model_size + cache overhead = ?GB
- Ratio: X:1 memory efficiency

**Quality Comparison**:
- Do both achieve similar specialization?
- Is synthesis quality equivalent?
- Any advantage to true multi-agent?

**Deliverables**:
- `src/true_multiagent.py` (3-agent implementation)
- `results/exp_multiagent_baseline/` (outputs and metrics)
- `results/exp_multiagent_baseline/memory_comparison.md`
- `paper/sections/06_discussion.md` (comparison discussion)

**Success Criteria**:
- [ ] True multi-agent runs successfully
- [ ] Memory measurements accurate
- [ ] Quality comparable to semantic isolation
- [ ] 3X memory claim validated (or corrected)

---

### Week 7: Ablation Studies

**Objectives**:
- Test sensitivity to hyperparameters
- Vary cluster count (2, 3, 5, 7)
- Vary routing threshold
- Measure robustness

**Tasks**:

| Day | Task | Time | Details |
|-----|------|------|---------|
| Mon | Design ablation experiments | 2h | What to vary, expected effects |
| Mon | Implement cluster count variation | 3h | 2-cluster, 5-cluster, 7-cluster |
| Tue | Run 2-cluster on 10 examples | 2h | Only technical + synthesis |
| Tue | Run 5-cluster on 10 examples | 3h | More fine-grained separation |
| Wed | Run 7-cluster on 10 examples | 3h | Very fine-grained |
| Wed | Analyze cluster count impact | 2h | Quality vs efficiency trade-off |
| Thu | Routing threshold experiments | 4h | Vary semantic similarity cutoff |
| Fri | Cache size limit experiments | 3h | What if cache budget is limited? |
| Fri | Write ablation section | 2h | Document robustness |

**Ablation Variables**:

**Cluster Count**:
- 2 clusters: One specialist + coordinator
- 3 clusters: Two specialists + coordinator (default)
- 5 clusters: Four specialists + coordinator
- 7 clusters: Six specialists + coordinator

**Expected**: Diminishing returns beyond 3-4 clusters

**Routing Threshold**:
- Strict (0.9 similarity): Forces separation
- Moderate (0.7 similarity): Default
- Loose (0.5 similarity): Allows more overlap

**Expected**: Sweet spot around 0.6-0.8

**Deliverables**:
- `results/exp_ablations/` (all ablation results)
- `results/exp_ablations/cluster_count_analysis.md`
- `results/exp_ablations/threshold_sensitivity.md`
- `paper/sections/05_results.md` (ablation subsection)

**Success Criteria**:
- [ ] Tested 2, 3, 5, 7 cluster configurations
- [ ] Identified optimal configuration
- [ ] Demonstrated robustness to hyperparameters
- [ ] Sweet spot analysis documented

---

### Week 8: Additional Baselines

**Objectives**:
- Implement random clustering baseline
- Implement fixed clustering baseline
- Test attention-based clustering
- Show semantic clustering is necessary

**Tasks**:

| Day | Task | Time | Details |
|-----|------|------|---------|
| Mon | Implement random clustering | 2h | Random assignment to clusters |
| Mon | Run random on 10 examples | 2h | Expect worse than semantic |
| Tue | Implement fixed clustering | 2h | By turn position (1-5, 6-10, 11-15) |
| Tue | Run fixed on 10 examples | 2h | Temporal vs semantic |
| Wed | Implement attention-based | 3h | Cluster by attention patterns |
| Wed | Run attention-based on 10 examples | 3h | Compare to semantic |
| Thu | Statistical comparison | 3h | All baselines vs semantic |
| Fri | Write baseline comparison | 3h | Why semantic wins |

**Baselines**:

**Random Clustering**:
- Randomly assign turns to clusters
- Control: Tests whether any separation helps

**Fixed Clustering**:
- By turn position (temporal boundaries)
- Like turn-based but with clustering

**Attention-Based Clustering**:
- Cluster by attention pattern similarity
- Alternative to semantic (reasoning-based)

**Deliverables**:
- `src/baselines.py` (all baseline implementations)
- `results/exp_baselines/` (outputs and metrics)
- `results/exp_baselines/comparison.md` (why semantic wins)
- `paper/sections/05_results.md` (baseline subsection)

**Success Criteria**:
- [ ] Random < Fixed < Semantic (hypothesis)
- [ ] Attention-based comparison informative
- [ ] Statistical significance for semantic superiority
- [ ] Clear explanation of why semantic works

---

### Week 9: Multi-Model Validation

**Objectives**:
- Test on additional model sizes
- Validate generalization
- Test on Llama, Qwen, etc.
- Document model-specific findings

**Tasks**:

| Day | Task | Time | Details |
|-----|------|------|---------|
| Mon | Set up Gemma 2 9B | 2h | Smaller model |
| Mon | Run semantic on 10 examples (9B) | 3h | Does it still work? |
| Tue | Set up Llama 3.1 8B | 2h | Different architecture |
| Tue | Run semantic on 10 examples (Llama) | 3h | Cross-model validation |
| Wed | Set up Qwen 2.5 14B | 2h | Another architecture |
| Wed | Run semantic on 10 examples (Qwen) | 3h | Robustness check |
| Thu | Compare across models | 3h | Which benefit most? |
| Fri | Write model comparison | 3h | Generalization analysis |

**Models to Test**:
- Gemma 3 12B (primary)
- Gemma 2 9B (smaller)
- Llama 3.1 8B (different architecture)
- Qwen 2.5 14B (larger)

**Expected**: Benefit is architecture-agnostic

**Deliverables**:
- `results/exp_multimodel/` (per-model results)
- `results/exp_multimodel/comparison.md`
- `paper/sections/05_results.md` (generalization subsection)

**Success Criteria**:
- [ ] Works on at least 3 different models
- [ ] Consistent benefit across architectures
- [ ] Any model-specific insights documented

---

## Phase 3: Real-World Validation (Weeks 10-12)

### Week 10: Deployment Case Study

**Objectives**:
- Deploy on local AI assistant (Ollama or LM Studio)
- Collect real usage telemetry
- Measure practical benefits
- User feedback

**Tasks**:

| Day | Task | Time | Details |
|-----|------|------|---------|
| Mon | Set up Ollama with Gemma 3 12B | 2h | Local deployment |
| Mon | Integrate semantic isolation | 3h | Plugin or wrapper |
| Tue | Recruit 10-15 users (developers) | 2h | Beta testers |
| Tue | User training session | 2h | How to use, what to expect |
| Wed-Fri | Users use system (async) | - | Collect telemetry |
| Fri | Collect telemetry data | 2h | Parse logs |
| Fri | User interviews | 2h | Qualitative feedback |

**Telemetry to Collect**:
- Cache sizes over time
- Agent switch frequency
- User satisfaction (survey)
- Task completion rate

**Deliverables**:
- `deployment/ollama_integration.py`
- `results/exp_deployment/telemetry.json`
- `results/exp_deployment/user_feedback.md`
- `paper/sections/06_discussion.md` (deployment subsection)

**Success Criteria**:
- [ ] 10-15 users successfully use system
- [ ] Telemetry collected automatically
- [ ] User satisfaction >3.5/5
- [ ] Real-world use cases documented

---

### Week 11: Paper Writing

**Objectives**:
- Write complete paper draft
- Integrate all results
- Polish figures and tables
- Prepare for submission

**Tasks**:

| Day | Task | Time | Details |
|-----|------|------|---------|
| Mon | Write Introduction (1.5 pages) | 4h | Problem, gap, contribution |
| Mon | Write Abstract (200 words) | 1h | Summary |
| Tue | Write Related Work (1.5 pages) | 4h | FlowKV, EpiCache, multi-agent |
| Tue | Review Methods section | 2h | Ensure complete |
| Wed | Integrate all results | 4h | Combine Phase 1-3 |
| Wed | Write Discussion (1 page) | 2h | Implications, limitations |
| Thu | Write Conclusion (0.5 pages) | 1h | Summary of contributions |
| Thu | Polish all figures | 3h | 300 DPI, consistent style |
| Fri | Polish all tables | 2h | LaTeX formatting |
| Fri | Full read-through | 3h | Coherence, flow |

**Paper Structure** (8-10 pages):

1. **Abstract** (200 words)
2. **Introduction** (1.5 pages)
   - Problem: Multi-agent systems are expensive
   - Gap: No single-model multi-agent simulation
   - Contribution: Virtual agents via KV cache partitioning
3. **Related Work** (1.5 pages)
   - Multi-agent LLM systems
   - KV cache optimization (FlowKV, EpiCache)
   - Multi-user serving (SafeKV, vLLM)
4. **Method** (2 pages)
   - Architecture (3-cluster design)
   - Semantic clustering (DeepSeek R1)
   - Coordinator pattern
5. **Experiments** (2.5 pages)
   - Dataset (50 examples, 5 domains)
   - Evaluation (3 raters, blind)
   - Baselines (4 conditions)
6. **Results** (2 pages)
   - Main results (semantic >> baselines)
   - Ablations (cluster count, thresholds)
   - Multi-model validation
7. **Discussion** (1 page)
   - Memory efficiency (3X validated)
   - Deployment case study
   - Limitations
8. **Conclusion** (0.5 pages)

**Deliverables**:
- `paper/rdic_paper.pdf` (complete draft)
- `paper/sections/*.md` (all sections)
- `paper/figures/` (publication-quality)
- `paper/tables/` (LaTeX formatted)

**Success Criteria**:
- [ ] Complete draft exists
- [ ] All figures at 300 DPI
- [ ] All results integrated
- [ ] Paper reads smoothly end-to-end

---

### Week 12: Revision and Submission

**Objectives**:
- Address any remaining issues
- Final proofread
- Prepare Arxiv submission
- Submit to venue

**Tasks**:

| Day | Task | Time | Details |
|-----|------|------|---------|
| Mon | Full revision | 4h | Content, clarity, flow |
| Mon | Check all references | 1h | Citations correct |
| Tue | Grammar and spell check | 2h | Proofread |
| Tue | Verify all numbers | 2h | Double-check statistics |
| Wed | Prepare Arxiv package | 3h | LaTeX source, figures |
| Wed | Test Arxiv compilation | 1h | No errors |
| Thu | Submit to Arxiv | 1h | Upload |
| Thu | Prepare venue submission | 2h | ICML/NeurIPS format |
| Fri | Submit to venue | 1h | Complete submission |
| Fri | Celebrate | - | 🎉 |

**Submission Checklist**:
- [ ] Paper compiles without errors
- [ ] All figures embedded correctly
- [ ] References complete
- [ ] Code repository public
- [ ] Reproducibility instructions clear

**Target Venues**:
- **ICML 2026** (July, deadline ~Jan 31)
- **NeurIPS 2026** (December, deadline ~May 15)
- **EMNLP 2026** (November, deadline ~June 1)
- **ACL 2026** (August, deadline ~Feb 15)

**Deliverables**:
- `paper/rdic_paper_final.pdf`
- Arxiv submission (public)
- Venue submission (ICML/NeurIPS/EMNLP)
- GitHub repository (public)

**Success Criteria**:
- [ ] Arxiv submission complete
- [ ] Venue submission complete
- [ ] All materials publicly available

---

## Budget Breakdown

| Item | Cost | Notes |
|------|------|-------|
| **Phase 1** | | |
| Claude API (dataset generation) | $15-20 | 50 examples × ~$0.30 each |
| Claude Haiku (evaluation) | $20-30 | 200 outputs × 3 metrics × $0.03 |
| Rater compensation | $0 | Volunteer graduate students |
| **Phase 2** | | |
| Additional API calls (baselines) | $10-15 | Extra generations |
| **Phase 3** | | |
| Deployment testing | $5-10 | Minimal API usage |
| **Total** | **$50-75** | Conservative estimate |

---

## Risk Mitigation

### Risk 1: Can't Recruit 3 Independent Raters

**Mitigation**:
- Offer authorship or acknowledgment
- Reach out to research groups
- Fallback: 2 raters + author (clearly disclosed)

### Risk 2: Statistical Tests Show No Significance

**Mitigation**:
- Increase sample size to n=100 if needed
- Report effect sizes even if p>0.05
- Honest assessment: "Marginal benefit in these conditions"

### Risk 3: True Multi-Agent Quality is Better

**Mitigation**:
- Expected: Quality parity, not superiority
- Reframe: "Achieves 90% of quality at 33% memory cost"
- Still valuable trade-off

### Risk 4: 3X Memory Claim Doesn't Hold

**Mitigation**:
- Measure accurately with instrumentation
- Report actual ratio (may be 2X or 2.5X)
- Worst case: Still significant efficiency gain

### Risk 5: Deployment Study Fails

**Mitigation**:
- Not critical for paper (nice-to-have)
- Can defer to follow-up work
- Phase 1-2 alone are sufficient for publication

---

## Success Criteria (Overall)

### Publication-Ready Paper Requires:

1. ✅ **n≥50 diverse examples** across 5 domains
2. ✅ **Blind evaluation** with 3 independent raters
3. ✅ **Cohen's kappa >0.6** (inter-rater reliability)
4. ✅ **Statistical significance** (p<0.05, d>0.5) for semantic vs baselines
5. ✅ **Memory efficiency** validated (measure actual ratio)
6. ✅ **Ablation studies** (cluster count, thresholds)
7. ✅ **Multi-model validation** (3+ models)
8. ✅ **All figures at 300 DPI**
9. ✅ **Complete related work** (FlowKV, EpiCache clearly distinguished)
10. ✅ **Public code repository** with reproducibility instructions

### Optional But Strengthening:

11. ⚪ True multi-agent baseline (quality comparison)
12. ⚪ Deployment case study (real-world validation)
13. ⚪ Additional baselines (attention-based, random)
14. ⚪ Error analysis (when does it fail?)

---

## Timeline Summary

| Phase | Weeks | Key Deliverables |
|-------|-------|------------------|
| **Phase 1: Rigorous Evaluation** | 1-5 | 50 examples, blind evaluation, statistics |
| **Phase 2: Benchmarking** | 6-9 | Multi-agent baseline, ablations, multi-model |
| **Phase 3: Real-World** | 10-12 | Deployment, paper, submission |
| **Total** | **12 weeks** | Publication-ready submission |

---

## Open Questions for Debate

### Question 1: Router Agent Architecture

**Proposal**: Use a separate model instance (lightweight) as a router agent with its own KV cache to decide where to route requests or create new clusters.

**Advantages**:
- Fully self-consistent (all routing decisions in KV cache)
- Can reason about routing (not just embedding similarity)
- Scales to dynamic cluster discovery

**Disadvantages**:
- Adds overhead (extra model instance)
- Complicates architecture
- May not be necessary if embedding routing works

**Debate Topic**: Is router agent worth the complexity?

### Question 2: Semantic Discovery Method

**Current**: DeepSeek R1 preprocessing (one-time)

**Alternative 1**: Embedding-based clustering (automatic)
**Alternative 2**: Attention pattern clustering (model-driven)
**Alternative 3**: Router agent (dynamic)

**Debate Topic**: Which discovery method is best?

### Question 3: Coordinator Pattern

**Current**: Cluster 3 sees outputs from Clusters 1-2 (message passing)

**Alternative 1**: Cluster 3 also sees their caches (full access)
**Alternative 2**: No coordinator, direct agent-to-agent communication
**Alternative 3**: Hierarchical coordination (multiple levels)

**Debate Topic**: Is current coordinator pattern optimal?

---

## Next Steps

1. **Review this plan** via multi-round debate (skeptics + proponents)
2. **Investigate router agent architecture** (literature review + debate)
3. **Finalize plan** (updated_plan.vN.md when consensus reached)
4. **Begin Phase 1** (Week 1: Dataset generation)

---

**Date**: 2026-01-22
**Status**: Initial plan v1 - Ready for debate review
**Next**: Multi-round debate to refine plan and investigate router agent architecture
