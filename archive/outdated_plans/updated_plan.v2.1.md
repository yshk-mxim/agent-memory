# Updated Development Plan v2: Virtual Multi-Agent via Semantic KV Cache Partitioning

**Date**: 2026-01-23
**Status**: Plan v2.1 incorporating Round 1 & 2 debate feedback
**Changes from v1**: Extended to 15 weeks, fixed infeasible components, added missing baselines
**Changes from v2**: 7 minor clarifications from Round 2 consensus

---

## Executive Summary

**Goal**: Develop and rigorously evaluate virtual multi-agent system within single LLM via semantic KV cache partitioning, achieving 3X memory efficiency vs parallel true multi-agent (infeasible on consumer hardware) and 2-3X latency advantage vs sequential true multi-agent.

**Timeline**: **15 weeks** (extended from 12 to address critical issues)
**Hardware**: Mac with 24GB RAM (MLX framework)
**Model**: Gemma 3 12B (4-bit quantization, ~7-10GB)
**Budget**: $60-90 for evaluation APIs

**Key Changes from v1**:
- ✅ Added Week 0 for pilot testing
- ✅ Fixed Week 6 true multi-agent (sequential execution to avoid OOM)
- ✅ Extended rater recruitment (Weeks 1-2 parallel recruiting)
- ✅ Added missing baselines (random clustering, no-coordinator)
- ✅ Extended paper writing (4 weeks instead of 1)
- ✅ Dropped router agent (defer to future work per debate consensus)
- ✅ Target NeurIPS 2026 (May 15 deadline) instead of ICML

---

## Phase 0: Pilot Testing (Week 0)

### Week 0: End-to-End Pilot with n=5

**Objectives**:
- Validate full pipeline before scaling
- Test all 4 conditions on small sample
- Identify technical issues early
- Refine evaluation rubric

**Tasks**:

| Day | Task | Time | Details |
|-----|------|------|---------|
| Mon | Generate 5 pilot examples | 2h | 1 per domain (coding, research, business, support, creative) |
| Mon | Implement all 4 conditions | 3h | Sequential, prompted, turn-based, semantic |
| Tue | Run pilot experiment | 3h | 5 examples × 4 conditions = 20 runs |
| Tue | Self-evaluate outputs | 2h | Apply rubric, identify issues |
| Wed | Refine rubric based on pilot | 2h | Fix ambiguities, add examples |
| Wed | Instrumentation test | 2h | Verify telemetry collection works |
| Thu | Fix any technical bugs | 3h | MLX issues, cache management, etc. |
| Thu | Document lessons learned | 1h | What works, what needs fixing |
| Fri | Finalize experiment protocol | 2h | Lock down procedure for Phase 1 |

**Deliverables**:
- `data/pilot_examples.json` (5 examples)
- `results/pilot/` (outputs from 20 runs)
- `evaluation/rubric_v2.md` (refined based on pilot)
- `docs/pilot_lessons.md` (issues and fixes)

**Success Criteria**:
- [ ] All 4 conditions run without crashes
- [ ] Rubric identifies clear quality differences
- [ ] Instrumentation captures all metrics
- [ ] No major blockers identified

---

## Phase 1: Rigorous Evaluation (Weeks 1-6)

**Note on Baseline Conditions**:
- **Primary conditions** (n=50, full statistical analysis): Sequential, prompted, turn-based, semantic, random clustering
- **Secondary baselines** (n=10, exploratory comparison): No-coordinator, true multi-agent
- Secondary baselines are reported descriptively without formal hypothesis testing due to smaller sample size

### Week 1: Dataset Generation + Begin Rater Recruitment

**Objectives**:
- Generate 50 diverse examples (up from pilot's 5)
- Begin recruiting 4 independent raters (1 backup)
- Ensure balanced domain coverage

**Tasks**:

| Day | Task | Time | Details |
|-----|------|------|---------|
| Mon | **Parallel**: Recruit raters (Task 1) | 2h | Email to research groups, post on forums |
| Mon | Generate coding examples (10) | 3h | Multi-file debugging + docs + review |
| Tue | **Parallel**: Follow up on raters | 1h | Answer questions, schedule interviews |
| Tue | Generate research examples (10) | 4h | Literature review + experiment + writing |
| Wed | **Parallel**: Rater interviews | 2h | 30 min each × 4 candidates |
| Wed | Generate business examples (10) | 3h | Technical + strategy + synthesis |
| Thu | **Parallel**: Confirm 3 raters (+1 backup) | 1h | Send contracts/agreements |
| Thu | Generate support examples (10) | 3h | Technical support + billing + account |
| Fri | Generate creative examples (10) | 3h | Storytelling + editing + analysis |
| Fri | Validation and quality check | 2h | Manual review, regenerate if <80% quality |

**Deliverables**:
- `data/full_dataset_v1.json` (50 examples)
- `data/domain_statistics.md` (balance analysis)
- `evaluation/raters.md` (3 confirmed + 1 backup)

**Success Criteria**:
- [ ] 50 examples with >80% validation quality
- [ ] 10 examples per domain (balanced)
- [ ] 3 raters confirmed + 1 backup recruited
- [ ] Train/val/test splits created (30/10/10)

---

### Week 2: Evaluation Framework + Rater Training

**Objectives**:
- Build comprehensive evaluation system
- Train raters thoroughly (4-hour session)
- Implement automated metrics
- Create web interface for rating

**Tasks**:

| Day | Task | Time | Details |
|-----|------|------|---------|
| Mon | Design web interface for raters | 3h | Simple Flask app, randomized presentation |
| Mon | Implement blinding system | 2h | Anonymize conditions completely |
| Tue | Implement automated metrics | 4h | ROUGE, BERTScore, semantic similarity |
| Wed | Create rater training materials | 3h | Slides, examples, rubric walkthrough |
| Wed | **CRITICAL: Rater training session** | 4h | Live training with all 3 raters |
| Thu | Post-training calibration | 2h | Raters score 10 golden examples, compare |
| Thu | Create evaluation pipeline | 3h | Batch processing, auto metric calculation |
| Fri | Test full pipeline on pilot data | 2h | Verify works end-to-end |
| Fri | Document evaluation protocol | 2h | Step-by-step instructions |

**Rater Training Agenda** (4 hours):

**Hour 1: Introduction and Rubric**
- Project goals and context
- The 3 dimensions (specialization, contamination, synthesis)
- 0-5 scale with concrete examples

**Hour 2: Guided Practice**
- Score 5 examples together as a group
- Discuss disagreements
- Refine understanding

**Hour 3: Independent Calibration**
- Each rater scores 5 golden examples alone
- Compare ratings, compute agreement
- Discuss discrepancies

**Hour 4: Q&A and Web Interface Tutorial**
- Questions and edge cases
- Walk through rating interface
- Set expectations (10 hours of rating work)

**Deliverables**:
- `evaluation/web_interface/` (Flask app for rating)
- `evaluation/automated_metrics.py` (ROUGE, BERTScore)
- `evaluation/pipeline.py` (end-to-end)
- `evaluation/training_slides.pdf` (for raters)
- `evaluation/golden_examples.json` (calibration set)

**Success Criteria**:
- [ ] Raters complete 4-hour training
- [ ] Cohen's kappa >0.7 minimum (>0.8 ideal) on golden set (after training)
- [ ] Web interface functional and usable
- [ ] Automated metrics implemented and tested

---

### Week 3: Instrumentation + Begin Experiment Runs

**Objectives**:
- Add comprehensive instrumentation
- Begin running evaluation experiments
- Monitor for technical issues

**Tasks**:

| Day | Task | Time | Details |
|-----|------|------|---------|
| Mon | Implement timing instrumentation | 3h | Generation time, per-turn latency |
| Mon | Implement memory profiling | 2h | Peak RAM, cache sizes |
| Tue | Implement cache growth tracking | 3h | Log tokens per cluster over time |
| Tue | Test instrumentation on 5 examples | 2h | Verify <5% overhead |
| Wed | **Run sequential condition (50 examples)** | 4h | Baseline |
| Thu | **Run prompted condition (50 examples)** | 4h | Soft isolation |
| Fri | **Run turn-based condition (50 examples)** | 4h | Naive isolation |
| Fri | Check for any errors/crashes | 1h | Verify all runs completed |

**Deliverables**:
- `instrumentation/profiler.py`
- `instrumentation/telemetry.py`
- `results/phase1_runs/sequential/` (50 outputs + metrics)
- `results/phase1_runs/prompted/` (50 outputs + metrics)
- `results/phase1_runs/turn_based/` (50 outputs + metrics)

**Success Criteria**:
- [ ] All instrumentation working, <5% overhead
- [ ] 150 runs completed (3 conditions × 50 examples)
- [ ] No crashes or missing data
- [ ] Telemetry logs captured

---

### Week 4: Complete Runs + Distribute to Raters

**Objectives**:
- Complete semantic condition
- Anonymize and randomize outputs
- Distribute to raters
- Begin automated metric analysis

**Tasks**:

| Day | Task | Time | Details |
|-----|------|------|---------|
| Mon | **Run semantic condition (50 examples)** | 4h | RDIC - our method |
| Mon | Verify all runs completed | 1h | 200 total outputs |
| Tue | Anonymize all outputs | 2h | Remove condition labels |
| Tue | Randomize presentation order | 1h | Different order per rater |
| Wed | Upload to rating interface | 2h | Populate web app |
| Wed | **Distribute to 3 raters** | 1h | Send emails with links |
| Thu | Compute automated metrics | 4h | ROUGE, BERTScore for all 200 outputs |
| Fri | Preliminary automated analysis | 3h | Do automated metrics show differences? |
| Fri | Monitor rater progress | 1h | Check if anyone has questions |

**Rater Workload**:
- 50 examples × 4 conditions = 200 outputs (primary conditions only)
- Additional: ~30 outputs from secondary baselines (random, no-coordinator, true multi-agent)
- Total: ~230 outputs per rater
- ~3 minutes per output = 11-13 hours total per rater
- **Evaluation window**: 1-2 weeks (flexible based on rater availability)
  - **Option 1**: 1 week intensive (Week 4-5, ~2.5 hours/day)
  - **Option 2**: 2 weeks relaxed (Weeks 4-5, ~1.5 hours/day)
- Fallback if recruitment difficult: Reduce to n=30 examples per rater (7.5 hours total)

**Deliverables**:
- `results/phase1_runs/semantic/` (50 outputs + metrics)
- `results/phase1_runs/blinded/` (anonymized for raters)
- `results/phase1_runs/automated_metrics.json` (ROUGE, BERTScore)

**Success Criteria**:
- [ ] All 200 runs completed (4 conditions × 50 examples)
- [ ] Blinded outputs distributed to 3 raters
- [ ] Automated metrics show expected pattern
- [ ] Raters begin rating work

---

### Week 5: Rating Collection + Statistical Analysis

**Objectives**:
- Collect ratings from all 3 raters
- Compute inter-rater reliability
- Run statistical tests
- Generate results tables and figures

**Tasks**:

| Day | Task | Time | Details |
|-----|------|------|---------|
| Mon | Monitor rater progress | 1h | Send reminders if needed |
| Mon | Begin preliminary analysis on partial data | 3h | Early trends |
| Tue | **Collect all ratings** (by Tue EOD) | 1h | Download from interface |
| Tue | Verify completeness | 1h | Check no missing ratings |
| Wed | Compute inter-rater reliability | 2h | Cohen's kappa, Fleiss' kappa, Krippendorff's α |
| Wed | Aggregate ratings (mean per output) | 1h | Average across 3 raters |
| Thu | **Statistical testing** | 3h | Paired t-tests, effect sizes, corrections |
| Thu | **Rubric validation** | 1h | Convergent validity (human vs automated, r>0.6), discriminant validity (inter-dimension correlation <0.7) |
| Fri | Generate tables (LaTeX) | 2h | Main results, ablations |
| Fri | Generate figures (matplotlib) | 3h | Box plots, bar charts |

**Statistical Analysis Plan**:

**Primary Tests**:
1. **Semantic vs Sequential**: Paired t-test (specialization, contamination, synthesis)
   - H1: Semantic > Sequential on all 3 dimensions
   - α = 0.05, two-tailed

2. **Semantic vs Prompted**: Paired t-test
   - H2: Semantic > Prompted (proves architecture > instructions)
   - Most critical for novelty claim

3. **Semantic vs Turn-Based**: Paired t-test
   - H3: Semantic > Turn-Based (semantic > temporal boundaries)

**Multiple Comparison Correction**:
- Use **False Discovery Rate (FDR)** correction (not Bonferroni)
- FDR is less conservative, appropriate for exploratory analysis

**Effect Sizes**:
- Cohen's d for all pairwise comparisons
- Interpret: |d| > 0.2 small, > 0.5 medium, > 0.8 large

**Power Analysis** (post-hoc):
- Compute achieved power (1 - β)
- Report: Can detect effect sizes ≥ X with 80% power

**Deliverables**:
- `results/phase1_analysis/ratings.json` (all 3 raters)
- `results/phase1_analysis/inter_rater_reliability.md` (kappa values)
- `results/phase1_analysis/statistics.json` (all p-values, effect sizes)
- `results/phase1_analysis/tables/table1_main_results.tex`
- `results/phase1_analysis/figures/fig1_specialization.png`

**Success Criteria**:
- [ ] Cohen's kappa >0.7 minimum (>0.8 ideal, indicating substantial to almost perfect agreement)
- [ ] Semantic significantly > all baselines (p<0.05, FDR-corrected)
- [ ] Effect sizes medium to large (d>0.5)
- [ ] All figures at 300 DPI

---

### Week 6: Error Analysis + Missing Baselines

**Objectives**:
- Analyze where semantic isolation fails
- Add missing baselines (random, no-coordinator)
- Understand failure modes

**Tasks**:

| Day | Task | Time | Details |
|-----|------|------|---------|
| Mon | **Implement random clustering baseline** | 2h | Random assignment to clusters |
| Mon | Run random on 10 examples | 1h | Control condition |
| Tue | **Implement no-coordinator baseline** | 2h | 2 clusters only, no synthesis |
| Tue | Run no-coordinator on 10 examples | 1h | Tests if coordinator necessary |
| Wed | **Error analysis**: Find worst semantic examples | 2h | Where did semantic fail? |
| Wed | Qualitative analysis of failures | 3h | Why? Pattern? |
| Thu | **Error analysis**: Compare to baselines on failures | 2h | Did baselines also fail? |
| Thu | Document failure modes | 2h | When does semantic not help? |
| Fri | Synthesize error analysis | 2h | Write up findings |
| Fri | Add to results section | 2h | Limitations and failure modes |

**Error Analysis Questions**:
1. Which domains does semantic struggle with?
2. Are there examples where prompted beats semantic?
3. Do failures correlate with semantic distance between agents?
4. Is coordinator actually necessary (no-coordinator test)?

**Deliverables**:
- `results/phase1_baselines/random/` (10 outputs)
- `results/phase1_baselines/no_coordinator/` (10 outputs)
- `results/phase1_analysis/error_analysis.md` (comprehensive)
- `paper/sections/06_discussion.md` (limitations subsection)

**Success Criteria**:
- [ ] Random baseline confirmed as floor (worst performance)
- [ ] No-coordinator test shows coordinator adds value
- [ ] Failure modes documented honestly
- [ ] Limitations section drafted

---

## Phase 2: Benchmarking (Weeks 7-9)

### Week 7: True Multi-Agent Baseline (Sequential Execution)

**Objectives**:
- Implement true 3-agent system WITHOUT parallel execution
- Measure memory and quality
- Validate 3X memory efficiency claim

**Rationale for Sequential**:
- Parallel 3×12B = 21-30GB + overhead = OOM crash on 24GB RAM
- Sequential: Run Agent 1, save output → Run Agent 2, save output → Run Agent 3
- Peak memory: max(10GB, 10GB, 10GB) = 10GB (feasible)

**Tasks**:

| Day | Task | Time | Details |
|-----|------|------|---------|
| Mon | Design sequential 3-agent protocol | 2h | Agent 1 → Agent 2 → Agent 3 (coordinator). **Communication**: Each agent receives all previous outputs via concatenation in context (Agent 2 gets Agent 1 output, Agent 3 gets Agent 1+2 outputs), mirroring semantic coordinator structure for fair comparison. |
| Mon | Implement sequential agent execution | 3h | Load/unload model between agents |
| Tue | Test on 5 examples | 3h | Verify correctness |
| Tue | Measure peak memory per agent | 2h | Instrumentation |
| Wed | **Run true multi-agent on 10 test examples** | 4h | Subset for comparison |
| Thu | Evaluate quality (same rubric) | 3h | Does true multi-agent win? |
| Thu | Compare quality: semantic vs true multi-agent | 2h | Quality parity analysis |
| Fri | **Memory efficiency calculation** | 2h | Peak(semantic) vs Peak(true multi-agent) |
| Fri | Write comparison subsection | 2h | Results section |

**Memory Calculation**:

**Virtual Multi-Agent (Semantic)**:
- 1 model loaded = 10GB (Gemma 3 12B 4-bit)
- 3 clusters in cache = +500MB
- **Total: ~10.5GB**

**True Multi-Agent (Sequential)**:
- Max across 3 agents = max(10GB, 10GB, 10GB) = 10GB
- **Total: ~10GB**

**Wait, where's the 3X?**

**True Multi-Agent (Parallel - Ideal)**:
- 3 models simultaneously = 3 × 10GB = 30GB
- This is what you'd need for TRUE parallelism
- But infeasible on 24GB RAM

**Conclusion**: Semantic achieves **3X memory efficiency vs parallel true multi-agent (30GB)**, which is the standard multi-agent architecture but infeasible on consumer hardware (24GB RAM). Compared to sequential execution workaround (10GB), semantic uses comparable memory but provides **2-3X latency advantage** due to cache reuse (no model reloading between agent switches). Report both:
- vs Parallel (standard architecture, infeasible): 10.5GB vs 30GB = **2.9X memory efficiency**
- vs Sequential (feasible workaround): 10.5GB vs 10GB = **comparable memory, 2-3X latency speedup**

**Deliverables**:
- `src/true_multiagent_sequential.py`
- `results/phase2_multiagent/outputs/` (10 examples)
- `results/phase2_multiagent/memory_comparison.md`
- `paper/sections/05_results.md` (multi-agent comparison)

**Success Criteria**:
- [ ] True multi-agent runs without OOM
- [ ] Quality is comparable (not better)
- [ ] Memory efficiency claim nuanced (3X vs parallel, 1X vs sequential)
- [ ] Semantic has latency advantage (cache reuse)

---

### Week 8: Quick Ablations + Multi-Model Test

**Objectives**:
- Test cluster count sensitivity (2, 3, 5)
- Test on one additional model (Llama 3.1 8B)
- Show robustness

**Tasks**:

| Day | Task | Time | Details |
|-----|------|------|---------|
| Mon | **Ablation: 2 clusters** (specialist + coordinator) | 2h | 10 examples |
| Mon | **Ablation: 5 clusters** (4 specialists + coordinator) | 2h | 10 examples |
| Tue | Analyze cluster count impact | 2h | 2 vs 3 vs 5 performance |
| Tue | Document sweet spot | 1h | Is 3 optimal? |
| Wed | Set up Llama 3.1 8B (MLX) | 2h | Alternative model |
| Wed | Run semantic on Llama (10 examples) | 3h | Cross-model validation |
| Thu | Evaluate Llama outputs | 2h | Does it work? |
| Thu | Compare Gemma vs Llama | 2h | Model-specific findings |
| Fri | Write ablation + multi-model sections | 3h | Results subsections |

**Deliverables**:
- `results/phase2_ablations/2clusters/` (10 outputs)
- `results/phase2_ablations/5clusters/` (10 outputs)
- `results/phase2_multimodel/llama/` (10 outputs)
- `paper/sections/05_results.md` (ablation + generalization)

**Success Criteria**:
- [ ] 3 clusters is near-optimal (not much gain from 5)
- [ ] Semantic works on Llama (cross-architecture)
- [ ] Ablations show robustness to hyperparameters

---

### Week 9: Optional Deployment Study

**Objectives**:
- Deploy on local AI assistant (if time permits)
- Collect real-world telemetry
- User feedback

**Tasks**:

| Day | Task | Time | Details |
|-----|------|------|---------|
| Mon | Integrate with Ollama/LM Studio | 3h | Plugin or wrapper |
| Tue | Recruit 5-10 beta users | 2h | Developers only |
| Wed-Thu | Users test system (async) | - | Collect telemetry |
| Fri | Collect feedback and telemetry | 2h | Parse logs |
| Fri | Write deployment subsection | 2h | Discussion section |

**Note**: This is OPTIONAL. If behind schedule, skip and focus on paper writing.

**Deliverables**:
- `deployment/integration.py`
- `results/phase2_deployment/telemetry.json` (if completed)
- `paper/sections/06_discussion.md` (deployment subsection if completed)

---

## Phase 3: Paper Writing (Weeks 10-13)

### Week 10: Introduction + Related Work

**Objectives**:
- Write compelling Introduction
- Write comprehensive Related Work
- Establish clear positioning

**Tasks**:

| Day | Task | Time | Details |
|-----|------|------|---------|
| Mon | Outline Introduction structure | 1h | Hook, problem, gap, contribution |
| Mon | Write Introduction draft | 4h | 1.5 pages |
| Tue | Write Abstract draft | 2h | 200 words |
| Tue | Outline Related Work structure | 2h | 5 subsections (added multi-persona, MoE from Round 2 feedback) |
| Wed | Write Related Work: Multi-Agent Systems | 2h | 0.5 pages |
| Wed | Write Related Work: KV Cache Optimization | 2h | FlowKV, EpiCache clearly distinguished |
| Thu | Write Related Work: Multi-User Serving | 2h | Complementary, not competing |
| Thu | Write positioning paragraph | 1h | Where does our work fit? |
| Fri | Revise Introduction + Related Work | 3h | Coherence, flow |
| Fri | Draft contribution list | 1h | 3-4 clear contributions |

**Related Work Structure**:

**Section 2.1: Multi-Agent LLM Systems**
- True multi-agent (separate models): Memory overhead
- Agent coordination: Orchestration patterns
- **Our work**: Virtual agents via cache partitioning

**Section 2.2: Multi-Persona LLM Systems** (NEW from Round 2 feedback)
- Persona-augmented LLMs: Character consistency via prompting
- Role-conditioned prompting: Instructional methods
- **Distinction**: They use soft isolation (prompts), we use hard isolation (KV cache architecture)
- **Our comparison**: Prompted condition (their approach) vs Semantic (our approach)

**Section 2.3: KV Cache Optimization**
- **FlowKV** (compression for long conversations)
- **EpiCache** (eviction for memory management)
- **Our work**: Isolation for agent quality (complementary)

**Section 2.4: Multi-User Serving**
- SafeKV, vLLM: Per-user isolation for privacy
- **Our work**: Per-agent isolation for quality (orthogonal layer)

**Section 2.5: Mixture-of-Experts (MoE)** (NEW from Round 2 feedback)
- MoE models (GPT-4, Mixtral): Training-time specialization via expert modules
- Router selects experts based on input
- **Distinction**: MoE requires training, our approach is inference-time (works with any pretrained model)
- **Complementary**: Could combine KV partitioning with MoE models

**Deliverables**:
- `paper/sections/01_abstract.md`
- `paper/sections/02_introduction.md`
- `paper/sections/03_related_work.md`

**Success Criteria**:
- [ ] Introduction clearly states problem and contribution
- [ ] Related work distinguishes from FlowKV/EpiCache
- [ ] Positioning as complementary is clear

---

### Week 11: Methods + Experiments

**Objectives**:
- Write Methods section (architecture, clustering, evaluation)
- Write Experiments section (dataset, protocol, baselines)

**Tasks**:

| Day | Task | Time | Details |
|-----|------|------|---------|
| Mon | Write Section 4.1: Architecture | 2h | 3-cluster design, coordinator pattern |
| Mon | Write Section 4.2: Semantic Clustering | 2h | DeepSeek R1 discovery |
| Tue | Write Section 4.3: Evaluation Protocol | 3h | Rubric, raters, metrics |
| Tue | Create architecture diagram (Figure 1) | 2h | Visual of 3-cluster design |
| Wed | Write Section 5.1: Dataset | 2h | 50 examples, 5 domains |
| Wed | Write Section 5.2: Baselines | 2h | 4 conditions + rationale |
| Thu | Write Section 5.3: Implementation | 2h | MLX, Gemma 3 12B, instrumentation |
| Thu | Polish all methods content | 2h | Ensure reproducibility |
| Fri | Review Methods + Experiments | 3h | Check completeness |

**Deliverables**:
- `paper/sections/04_methods.md`
- `paper/sections/05_experiments.md`
- `paper/figures/fig1_architecture.pdf`

**Success Criteria**:
- [ ] Methods are fully reproducible from description
- [ ] All hyperparameters documented
- [ ] Figures integrated properly

---

### Week 12: Results + Discussion

**Objectives**:
- Write Results section (all findings)
- Write Discussion (implications, limitations)
- Integrate all tables and figures

**Tasks**:

| Day | Task | Time | Details |
|-----|------|------|---------|
| Mon | Write Section 6.1: Main Results | 3h | Semantic >> baselines |
| Mon | Integrate Table 1 (main results) | 1h | LaTeX table |
| Tue | Write Section 6.2: Ablations | 2h | Cluster count, robustness |
| Tue | Integrate Figure 2 (box plots) | 2h | Visualization |
| Wed | Write Section 6.3: Multi-Agent Comparison | 2h | Quality + memory |
| Wed | Write Section 6.4: Error Analysis | 2h | Failure modes |
| Thu | Write Section 7: Discussion | 3h | Implications, limitations |
| Thu | Write Section 8: Conclusion | 1h | Summary |
| Fri | Full read-through (Sections 1-8) | 3h | Coherence check |
| Fri | Revise based on read-through | 2h | Fix issues |

**Deliverables**:
- `paper/sections/06_results.md`
- `paper/sections/07_discussion.md`
- `paper/sections/08_conclusion.md`
- `paper/tables/` (all LaTeX tables)
- `paper/figures/` (all 300 DPI figures)

**Success Criteria**:
- [ ] All results reported with statistics
- [ ] Discussion addresses "so what?"
- [ ] Limitations are honest and complete
- [ ] Paper reads smoothly end-to-end

---

### Week 13: Revision + Polishing

**Objectives**:
- Complete paper revision
- Polish figures and tables
- Final proofread

**Tasks**:

| Day | Task | Time | Details |
|-----|------|------|---------|
| Mon | Full paper revision (content) | 4h | Clarity, flow, coherence |
| Tue | Full paper revision (technical accuracy) | 3h | Check all numbers |
| Tue | Polish all figures | 2h | Consistent style, 300 DPI |
| Wed | Polish all tables | 2h | LaTeX formatting |
| Wed | Reference check | 2h | All citations correct |
| Thu | Grammar and spell check | 3h | Proofread |
| Thu | Convert to LaTeX (if needed) | 2h | NeurIPS template |
| Fri | Test compilation | 1h | No errors |
| Fri | Final read-through | 2h | One last check |

**Deliverables**:
- `paper/rdic_paper_draft.pdf` (complete)
- `paper/rdic_paper.tex` (LaTeX source)

**Success Criteria**:
- [ ] Paper compiles without errors
- [ ] All figures at 300 DPI
- [ ] All references correct
- [ ] <10 pages (NeurIPS limit)

---

## Phase 4: Submission Prep (Week 14)

### Week 14: Code Release + Arxiv + Venue Submission

**Objectives**:
- Prepare public GitHub repository
- Submit to Arxiv
- Submit to NeurIPS 2026

**Tasks**:

| Day | Task | Time | Details |
|-----|------|------|---------|
| Mon | Code cleanup | 3h | Remove debug code, add comments |
| Mon | Write comprehensive README | 2h | How to reproduce |
| Tue | Write REPRODUCE.md | 2h | Step-by-step instructions |
| Tue | Test reproducibility | 2h | Fresh checkout, run from scratch |
| Wed | Prepare Arxiv package | 2h | LaTeX + figures |
| Wed | Submit to Arxiv | 1h | Upload |
| Thu | Prepare NeurIPS submission | 2h | Format check |
| Thu | **Submit to NeurIPS 2026** | 1h | Complete submission |
| Fri | Make GitHub repo public | 1h | Release code |
| Fri | Social media announcements | 1h | Twitter/X thread |

**Submission Checklist**:
- [ ] Paper compiles (NeurIPS LaTeX template)
- [ ] Supplementary materials (if any)
- [ ] Code repository public with MIT license
- [ ] Arxiv submission complete
- [ ] NeurIPS submission complete

**Target Venue**: **NeurIPS 2026** (May 15 deadline)
- Start date: Jan 23
- 15 weeks = ~April 30
- 2 weeks buffer before deadline = ✅ Feasible

**Deliverables**:
- `README.md` (comprehensive)
- `REPRODUCE.md` (step-by-step)
- `LICENSE` (MIT)
- Arxiv paper (public)
- NeurIPS submission (complete)
- GitHub repository (public)

**Success Criteria**:
- [ ] Arxiv submission public
- [ ] NeurIPS submission complete
- [ ] Code reproducible from fresh checkout

---

## Phase 5: Buffer Week (Week 15)

### Week 15: Contingency Buffer

**Purpose**: Address any delays or unexpected issues

**Possible Uses**:
- Re-run experiments if technical failures occurred
- Additional revision if reviewers at Arxiv raise issues
- Extend deployment study if it was skipped
- Additional ablations if requested
- Extra time for paper writing if behind

**This buffer provides resilience to timeline.**

---

## Budget Breakdown (Updated)

| Item | Cost | Notes |
|------|------|-------|
| **Phase 0** | | |
| Claude API (pilot) | $2-3 | 5 examples |
| **Phase 1** | | |
| Claude API (dataset) | $20-25 | 50 examples × ~$0.40 |
| Claude Haiku (evaluation) | $25-35 | 200 outputs × 3 metrics |
| Rater compensation | $0 | Volunteer (or $300 if paid at $100 each) |
| **Phase 2** | | |
| Additional generations | $15-20 | Ablations, baselines |
| **Total** | **$60-90** | (Or $360-390 if raters paid) |

---

## Changes from v1

### Critical Fixes

1. **Extended timeline**: 12 weeks → **15 weeks**
2. **Added Week 0**: Pilot testing to catch issues early
3. **Fixed Week 6**: Sequential true multi-agent (not parallel, avoids OOM)
4. **Extended rater recruitment**: Parallel recruiting in Weeks 1-2
5. **Extended paper writing**: 1 week → 4 weeks (Weeks 10-13)
6. **Added buffer**: Week 15 for contingencies
7. **Changed venue**: ICML → NeurIPS 2026 (May 15 deadline)

### Additions

8. **Rater training**: 4-hour live session (Week 2)
9. **Web interface**: For easier rating (Week 2)
10. **Missing baselines**: Random clustering, no-coordinator (Week 6)
11. **Error analysis**: Dedicated time (Week 6)
12. **FDR correction**: Instead of Bonferroni (less conservative)

### Scope Reductions

13. **Dropped router agent**: Defer to future work (debate consensus)
14. **Reduced multi-model**: 2 models (Gemma + Llama), not 4
15. **Optional deployment**: Week 9 can be skipped if behind
16. **Condensed ablations**: 2 days instead of full week

---

## Risk Mitigation

### Risk 1: Raters Drop Out

**Mitigation**:
- Recruit 4 raters (1 backup) in Week 1
- Pay honorarium ($100 each) if needed
- Distribute work over 2 weeks (not rushed)

### Risk 2: True Multi-Agent Memory Issues

**Already Mitigated**:
- Sequential execution instead of parallel
- Test on 5 examples first (Week 7 Tue)
- Can skip if still fails (not critical for novelty)

### Risk 3: Statistical Tests Non-Significant

**Mitigation**:
- Use FDR correction (less conservative than Bonferroni)
- Report effect sizes even if p>0.05
- Honest assessment: "Marginal benefit" still publishable

### Risk 4: Behind Schedule

**Mitigation**:
- Week 15 buffer
- Optional components (deployment, multi-model) can be cut
- Focus on Phase 1 (core evaluation) - that alone is sufficient

---

## Success Criteria (Publication-Ready)

### Must-Have (Blocks Publication)

1. ✅ n≥50 diverse examples across 5 domains
2. ✅ Blind evaluation with 3 independent raters
3. ✅ Cohen's kappa >0.7 minimum (>0.8 ideal, indicating substantial to almost perfect inter-rater reliability)
4. ✅ Statistical significance (p<0.05 FDR-corrected, d>0.5)
5. ✅ Error analysis (failure modes documented)
6. ✅ Complete related work (FlowKV/EpiCache distinction)
7. ✅ All figures at 300 DPI
8. ✅ Public code repository

### Strongly Recommended (Strengthens Paper)

9. ⚪ True multi-agent baseline (quality + memory comparison)
10. ⚪ Ablation studies (cluster count sensitivity)
11. ⚪ Multi-model validation (at least Gemma + Llama)
12. ⚪ Additional baselines (random, no-coordinator)

### Optional (Nice-to-Have)

13. ⚪ Deployment case study (real-world telemetry)
14. ⚪ Attention-based clustering baseline
15. ⚪ Hierarchical clustering exploration

---

## Timeline Summary

| Phase | Weeks | Key Deliverables |
|-------|-------|------------------|
| **Phase 0: Pilot** | 0 | 5 examples, technical validation |
| **Phase 1: Evaluation** | 1-6 | 50 examples, blind evaluation, statistics, error analysis |
| **Phase 2: Benchmarking** | 7-9 | True multi-agent, ablations, multi-model |
| **Phase 3: Writing** | 10-13 | Complete paper draft, revision |
| **Phase 4: Submission** | 14 | Arxiv + NeurIPS submission |
| **Phase 5: Buffer** | 15 | Contingency |
| **Total** | **15 weeks** | **Publication-ready submission** |

**Target**: NeurIPS 2026 (May 15 deadline)
**Start**: Jan 23, 2026
**Completion**: ~April 30, 2026 (2 weeks before deadline)

---

## Deferred to Future Work

Based on debate consensus, the following are explicitly **deferred**:

1. **Router agent architecture** (not novel per literature review, adds complexity)
2. **Hierarchical coordination** (multiple coordinator levels)
3. **Dynamic cluster discovery** (inference-time clustering)
4. **Cross-model cache sharing** (combining with C2C communication)
5. **Production deployment** (beyond POC)

These can be follow-up papers after establishing the core contribution.

---

## Next Steps

1. ✅ **This plan (v2) accepted by debate participants** (achieved consensus)
2. ➡️ **Begin Week 0** (pilot testing with n=5)
3. Iterate on plan if pilot reveals issues → v3 if needed
4. Proceed with Phase 1 upon successful pilot

---

**Date**: 2026-01-23
**Status**: Plan v2 ready for final debate round
**Changes**: Incorporated all Round 1 debate feedback
**Next**: Final debate review to achieve consensus → Begin execution
