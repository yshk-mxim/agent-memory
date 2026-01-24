# Updated Development Plan v3: Virtual Multi-Agent via Semantic KV Cache Partitioning

**Date**: 2026-01-23
**Status**: Plan v3 - Automated-First Evaluation Strategy
**Changes from v2.1**: Automated-first approach to address lack of independent human raters
**Changes from v2**: Extended to 15 weeks, fixed infeasible components, added missing baselines
**Changes from v1**: 7 minor clarifications from Round 2 consensus

---

## Executive Summary

**Goal**: Develop and rigorously evaluate virtual multi-agent system within single LLM via semantic KV cache partitioning, achieving 3X memory efficiency vs parallel true multi-agent (infeasible on consumer hardware) and 2-3X latency advantage vs sequential true multi-agent.

**Timeline**: **13-15 weeks** (flexible based on evaluation results)
**Hardware**: Mac with 24GB RAM (MLX framework)
**Model**: Gemma 3 12B (4-bit quantization, ~7-10GB)
**Budget**: $60-90 for APIs + $30 for Claude AI judge + $0-120 for optional MTurk validation = $90-240 total

**Key Innovation in v3: Automated-First Evaluation**

**Problem Identified**:
- No access to independent human raters (only family members available)
- MTurk risks: AI-generated responses, quality control challenges
- Original plan (v2.1) heavily dependent on human evaluation

**Solution - Three-Tier Evaluation Strategy**:

1. **Primary Evidence (Weeks 1-6)**: Comprehensive automated metrics suite (n=50)
   - Contamination detection metrics (4 mechanical)
   - Specialization measurement metrics (4 mechanical)
   - Synthesis quality metrics (4 mechanical)
   - Standard NLP metrics (4 mechanical: ROUGE, BERTScore, perplexity, embeddings)
   - **Claude AI Judge metrics** (3 qualitative: contamination, specialization, synthesis via Claude Sonnet 4.5)

2. **Decision Point (Week 6)**: Evaluate results and decide on human validation
   - Strong automated results + budget → Proceed to MTurk (Weeks 7-8)
   - Moderate results or no budget → Workshop venues, automated-only

3. **Optional Validation (Weeks 7-8)**: MTurk with anti-AI safeguards (n=20-30 subset)
   - Time tracking, attention checks, consistency checks
   - Justification requirements, qualification tests

**Publication Strategy**:
- **Strong results + human validation** → NeurIPS 2026 main conference
- **Strong results, automated only** → NeurIPS workshops, EMNLP workshops
- **Moderate results** → ArXiv preprint, iterate for next cycle

**Positioning**: "Automated metrics for scalable evaluation, with optional human validation for high-stakes claims"

**Key Changes from v2.1**:
- ✅ Removed rater recruitment dependency (Weeks 1-2)
- ✅ Removed rater training (Week 2)
- ✅ Removed web interface development (Week 2)
- ✅ Added comprehensive automated metrics suite (Weeks 2-3)
- ✅ Added Week 6 decision point (proceed with human eval?)
- ✅ Made human evaluation optional (Weeks 7-8)
- ✅ Reduced timeline flexibility (13-15 weeks vs fixed 15)
- ✅ Reduced budget uncertainty ($60-90 + $0-120 vs $60-390)

---

## Phase 0: Pilot Testing (Week 0)

### Week 0: End-to-End Pilot with n=5

**Objectives**:
- Validate full pipeline before scaling
- Test all 4 conditions on small sample
- Identify technical issues early
- Refine automated metrics

**Tasks**:

| Day | Task | Time | Details |
|-----|------|------|---------|
| Mon | Set up embedding model | 1h | Install sentence-transformers, test `all-MiniLM-L6-v2` (~80MB) |
| Mon | Implement embedding clustering | 2h | Cluster discovery + turn routing via cosine similarity |
| Mon | Generate 5 pilot examples | 2h | 1 per domain (coding, research, business, support, creative) |
| Mon | Implement all 4 conditions | 2h | Sequential, prompted, turn-based, semantic (with embedding routing) |
| Tue | Run pilot experiment | 3h | 5 examples × 4 conditions = 20 runs |
| Tue | Apply automated metrics | 2h | Test metric implementations |
| Wed | Refine metrics based on pilot | 2h | Fix issues, validate calculations |
| Wed | Instrumentation test | 2h | Verify telemetry collection works |
| Thu | Fix any technical bugs | 3h | MLX issues, cache management, embedding routing, etc. |
| Thu | Document lessons learned | 1h | What works, what needs fixing |
| Fri | Finalize experiment protocol | 2h | Lock down procedure for Phase 1 |

**Note on Semantic Clustering**:
- POC uses **embedding-based routing** (sentence-transformers, ~200MB, ~10-50ms per decision)
- **No ground truth labels** - proves approach works in practice
- Same architecture used in POC and production deployment
- See `plans/SEMANTIC_CLUSTERING_APPROACH.md` and `plans/EMBEDDING_CLUSTERING_IN_POC.md` for details

**Deliverables**:
- `src/embedding_clustering.py` (embedding-based routing implementation)
- `data/pilot_examples.json` (5 examples)
- `results/pilot/` (outputs from 20 runs + routing logs)
- `evaluation/metrics_v1.py` (automated metrics tested)
- `docs/pilot_lessons.md` (issues and fixes)

**Success Criteria**:
- [ ] All 4 conditions run without crashes
- [ ] Automated metrics show clear separation between conditions
- [ ] Instrumentation captures all metrics
- [ ] No major blockers identified

---

## Phase 1: Rigorous Evaluation (Weeks 1-6)

**Note on Baseline Conditions**:
- **Primary conditions** (n=50, full statistical analysis): Sequential, prompted, turn-based, semantic, random clustering
- **Secondary baselines** (n=10, exploratory comparison): No-coordinator, true multi-agent
- Secondary baselines are reported descriptively without formal hypothesis testing due to smaller sample size

### Week 1: Dataset Generation

**Objectives**:
- Generate 50 diverse examples (up from pilot's 5)
- Ensure balanced domain coverage
- Create validation splits

**Tasks**:

| Day | Task | Time | Details |
|-----|------|------|---------|
| Mon | Generate coding examples (10) | 3h | Multi-file debugging + docs + review |
| Tue | Generate research examples (10) | 4h | Literature review + experiment + writing |
| Wed | Generate business examples (10) | 3h | Technical + strategy + synthesis |
| Thu | Generate support examples (10) | 3h | Technical support + billing + account |
| Fri | Generate creative examples (10) | 3h | Storytelling + editing + analysis |
| Fri | Validation and quality check | 2h | Manual review, regenerate if <80% quality |

**Deliverables**:
- `data/full_dataset_v1.json` (50 examples)
- `data/domain_statistics.md` (balance analysis)
- `data/splits/` (train/val/test: 30/10/10)

**Success Criteria**:
- [ ] 50 examples with >80% validation quality
- [ ] 10 examples per domain (balanced)
- [ ] Train/val/test splits created
- [ ] Examples require multi-agent collaboration

---

### Week 2: Comprehensive Automated Metrics Suite

**Objectives**:
- Implement contamination detection metrics
- Implement specialization measurement metrics
- Implement synthesis quality metrics
- Implement standard NLP metrics

**Tasks**:

| Day | Task | Time | Details |
|-----|------|------|---------|
| Mon | **Contamination Detection** | 4h | See details below |
| Mon | Setup metric evaluation framework | 1h | Batch processing pipeline |
| Tue | **Specialization Measurement** | 4h | See details below |
| Tue | Test contamination + specialization | 1h | Verify on pilot data |
| Wed | **Synthesis Quality Metrics** | 4h | See details below |
| Wed | Test synthesis metrics | 1h | Verify calculations |
| Thu | **Standard NLP Metrics** | 3h | ROUGE, BERTScore, perplexity |
| Thu | Integration testing | 2h | All metrics on pilot data |
| Fri | Optimize metric computation | 2h | Parallelize, cache embeddings |
| Fri | Document metrics clearly | 2h | Formulas, interpretations, thresholds |

**Contamination Detection Metrics**:

1. **TF-IDF Similarity Between Specialists**:
   - Compute TF-IDF vectors for each specialist's output
   - Calculate cosine similarity between specialist pairs
   - **Target**: <0.3 for good isolation (low contamination)
   - **Interpretation**: High similarity = information leakage

2. **Domain-Specific Vocabulary Leakage**:
   - Define domain-specific lexicons (coding, business, research terms)
   - Count technical terms appearing in wrong specialist's output
   - **Target**: <5% cross-domain vocabulary bleeding
   - **Example**: Code tokens in business specialist output

3. **Lexical Overlap Percentage**:
   - Jaccard similarity of unique tokens between specialists
   - Exclude common stopwords
   - **Target**: <20% overlap for distinct specialists
   - **Formula**: |tokens_A ∩ tokens_B| / |tokens_A ∪ tokens_B|

4. **Cross-Cluster Keyword Bleeding**:
   - Extract top-k keywords from each specialist (k=20)
   - Measure how many appear in other specialists
   - **Target**: <3 keywords shared between specialists
   - **Method**: TF-IDF or KeyBERT extraction

**Specialization Measurement Metrics**:

1. **Cluster-Specific Keyword Density**:
   - Define expected keywords per cluster (from prompts/design)
   - Measure density of these keywords in outputs
   - **Target**: >2X density in correct specialist vs others
   - **Example**: "function", "debug", "error" for coding specialist

2. **Technical Density Scores**:
   - Code token frequency (function names, operators, keywords)
   - Jargon frequency (domain-specific terminology)
   - Formality scores (for business vs creative)
   - **Target**: Clear separation between specialist types

3. **Style Consistency Within Clusters**:
   - Vocabulary diversity (Type-Token Ratio)
   - Sentence length distribution
   - Lexical sophistication (word rarity)
   - **Target**: Low variance within specialist, high variance between

4. **Domain Classifier Confidence**:
   - Train simple classifier (logistic regression on TF-IDF)
   - Classify specialist outputs into domains
   - **Target**: >80% classification accuracy for semantic condition
   - **Baseline**: ~33% for sequential (no specialization)

**Synthesis Quality Metrics**:

1. **Information Coverage**:
   - Extract key facts/entities from each specialist output
   - Measure % of facts appearing in final synthesis
   - **Target**: >70% coverage (synthesis preserves specialist info)
   - **Method**: Named entity recognition + keyword extraction

2. **Semantic Similarity to Both Specialists**:
   - Compute embedding similarity (sentence-transformers)
   - Synthesis should be semantically similar to both specialists
   - **Target**: cosine similarity >0.6 to each specialist
   - **Interpretation**: Synthesis integrates both perspectives

3. **Coherence Scores**:
   - **BERTScore**: Semantic similarity to reference (if available)
   - **Perplexity**: Fluency using pretrained LM
   - **Target**: Perplexity <50 (fluent), BERTScore >0.85
   - **Method**: Use GPT-2 for perplexity, deberta-xlarge for BERTScore

4. **Novel Content Generation**:
   - Measure synthesis content NOT in specialist outputs
   - **Target**: 10-20% novel content (synthesis adds value)
   - **Formula**: tokens_synthesis - (tokens_spec1 ∪ tokens_spec2)
   - **Interpretation**: Synthesis is more than concatenation

**Standard NLP Metrics**:

1. **ROUGE-L**: Longest common subsequence (content preservation)
2. **BERTScore**: Contextual embedding similarity
3. **Perplexity**: Fluency using GPT-2
4. **Embedding Similarity**: Sentence-transformers cosine similarity

**Deliverables**:
- `evaluation/metrics/contamination.py` (4 contamination metrics)
- `evaluation/metrics/specialization.py` (4 specialization metrics)
- `evaluation/metrics/synthesis.py` (4 synthesis metrics)
- `evaluation/metrics/standard.py` (ROUGE, BERTScore, perplexity)
- `evaluation/pipeline.py` (unified batch processing)
- `evaluation/METRICS.md` (comprehensive documentation)

**Success Criteria**:
- [ ] All 12 custom metrics implemented and tested
- [ ] Standard metrics (ROUGE, BERTScore) working
- [ ] Metrics show clear separation on pilot data
- [ ] Computation time <5 minutes for 50 examples
- [ ] Documentation includes interpretation guidelines

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

### Week 4: Complete Runs + Automated Analysis

**Objectives**:
- Complete semantic condition
- Compute all automated metrics
- Begin preliminary statistical analysis

**Tasks**:

| Day | Task | Time | Details |
|-----|------|------|---------|
| Mon | **Run semantic condition (50 examples)** | 4h | RDIC - our method |
| Mon | Verify all runs completed | 1h | 200 total outputs |
| Tue | **Compute contamination metrics** | 3h | All 4 conditions × 50 examples |
| Tue | **Compute specialization metrics** | 3h | All 4 conditions × 50 examples |
| Wed | **Compute synthesis metrics** | 3h | All 4 conditions × 50 examples |
| Wed | **Compute standard metrics** | 2h | ROUGE, BERTScore, perplexity |
| Thu | Aggregate all metrics | 2h | Create master results dataframe |
| Thu | Preliminary statistical tests | 3h | Do metrics show differences? |
| Fri | Generate preliminary visualizations | 3h | Box plots, distribution plots |
| Fri | Document initial findings | 2h | Trends, patterns, surprises |

**Deliverables**:
- `results/phase1_runs/semantic/` (50 outputs + metrics)
- `results/phase1_metrics/contamination.csv` (all conditions)
- `results/phase1_metrics/specialization.csv` (all conditions)
- `results/phase1_metrics/synthesis.csv` (all conditions)
- `results/phase1_metrics/standard.csv` (ROUGE, BERTScore, etc.)
- `results/phase1_metrics/master_results.csv` (aggregated)
- `results/phase1_analysis/preliminary_plots/` (visualizations)

**Success Criteria**:
- [ ] All 200 runs completed (4 conditions × 50 examples)
- [ ] All metrics computed without errors
- [ ] Preliminary analysis shows expected patterns
- [ ] Metrics distinguish between conditions

---

### Week 5: Statistical Analysis + Results Generation

**Objectives**:
- Run comprehensive statistical tests
- Compute effect sizes and power analysis
- Generate publication-ready tables and figures
- Validate metric reliability

**Tasks**:

| Day | Task | Time | Details |
|-----|------|------|---------|
| Mon | **Statistical testing** | 4h | Paired t-tests, effect sizes, FDR correction |
| Mon | Power analysis | 1h | Achieved power calculations |
| Tue | **Metric validation** | 3h | Convergent/discriminant validity |
| Tue | Correlation analysis | 2h | Which metrics correlate? |
| Wed | Generate LaTeX tables | 3h | Main results, ablations, metric comparisons |
| Wed | Generate publication figures | 3h | Box plots, bar charts (300 DPI) |
| Thu | Per-domain analysis | 3h | Do results hold across all domains? |
| Thu | Identify best/worst examples | 2h | Qualitative analysis preparation |
| Fri | Comprehensive results writeup | 4h | Summarize all findings |
| Fri | Prepare decision point materials | 2h | Evidence for Week 6 decision |

**Statistical Analysis Plan**:

**Primary Tests**:
1. **Semantic vs Sequential**: Paired t-test on all metrics
   - H1: Semantic > Sequential on contamination, specialization, synthesis
   - α = 0.05, two-tailed

2. **Semantic vs Prompted**: Paired t-test on all metrics
   - H2: Semantic > Prompted (proves architecture > instructions)
   - Most critical for novelty claim

3. **Semantic vs Turn-Based**: Paired t-test on all metrics
   - H3: Semantic > Turn-Based (semantic > temporal boundaries)

**Multiple Comparison Correction**:
- Use **False Discovery Rate (FDR)** correction (not Bonferroni)
- FDR is less conservative, appropriate for multiple metrics

**Effect Sizes**:
- Cohen's d for all pairwise comparisons
- Interpret: |d| > 0.2 small, > 0.5 medium, > 0.8 large

**Power Analysis** (post-hoc):
- Compute achieved power (1 - β)
- Report: Can detect effect sizes ≥ X with 80% power

**Metric Validation**:
- **Convergent validity**: Do related metrics correlate? (r > 0.6)
- **Discriminant validity**: Do distinct metrics diverge? (r < 0.7)
- **Internal consistency**: Cronbach's α for multi-item constructs

**Deliverables**:
- `results/phase1_analysis/statistics.json` (all p-values, effect sizes)
- `results/phase1_analysis/power_analysis.md` (achieved power)
- `results/phase1_analysis/metric_validation.md` (validity checks)
- `results/phase1_analysis/tables/table1_main_results.tex`
- `results/phase1_analysis/tables/table2_metric_correlations.tex`
- `results/phase1_analysis/figures/fig1_specialization.png` (300 DPI)
- `results/phase1_analysis/figures/fig2_contamination.png` (300 DPI)
- `results/phase1_analysis/figures/fig3_synthesis.png` (300 DPI)
- `results/phase1_analysis/domain_breakdown.md` (per-domain results)

**Success Criteria**:
- [ ] Semantic significantly > all baselines on key metrics (p<0.05, FDR-corrected)
- [ ] Effect sizes medium to large (d>0.5) on primary metrics
- [ ] Metrics show good convergent and discriminant validity
- [ ] Results consistent across domains
- [ ] All figures at 300 DPI

---

### Week 6: Error Analysis + Missing Baselines + DECISION POINT

**Objectives**:
- Analyze where semantic isolation fails
- Add missing baselines (random, no-coordinator)
- **CRITICAL: Decide whether to proceed with human evaluation**
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
| Fri | **DECISION POINT: Human evaluation?** | 3h | Analyze evidence, decide |
| Fri | Prepare next phase plan | 2h | Week 7-8 plan or skip to writing |

**Error Analysis Questions**:
1. Which domains does semantic struggle with?
2. Are there examples where prompted beats semantic?
3. Do failures correlate with semantic distance between agents?
4. Is coordinator actually necessary (no-coordinator test)?
5. Do automated metrics align with qualitative assessment?

**Deliverables**:
- `results/phase1_baselines/random/` (10 outputs)
- `results/phase1_baselines/no_coordinator/` (10 outputs)
- `results/phase1_analysis/error_analysis.md` (comprehensive)
- `results/phase1_analysis/failure_patterns.md` (categorized)
- `results/phase1_analysis/decision_point_report.md` (recommendation)
- `paper/sections/06_discussion_draft.md` (limitations subsection)

**Success Criteria**:
- [ ] Random baseline confirmed as floor (worst performance)
- [ ] No-coordinator test shows coordinator adds value
- [ ] Failure modes documented honestly
- [ ] Limitations section drafted
- [ ] Clear recommendation on human evaluation

---

## DECISION POINT: Proceed with Human Evaluation?

**Week 6 Friday: Evaluate Evidence and Decide**

### Decision Criteria

**Proceed to Human Evaluation (Weeks 7-8) if ALL of the following**:

1. **Strong Automated Results**:
   - Semantic significantly outperforms all baselines (p < 0.01)
   - Effect sizes large (Cohen's d > 0.8) on primary metrics
   - Consistent results across all 5 domains
   - Low failure rate (<20% of examples)

2. **Clear Separation**:
   - Automated metrics show obvious quality differences
   - Qualitative inspection confirms metric validity
   - Failure cases are edge cases, not systematic

3. **Budget Available**:
   - $120-150 available for MTurk subset validation
   - Time available (Weeks 7-8 within timeline)

4. **Publication Ambition**:
   - Targeting top-tier conference (NeurIPS main track)
   - Need human validation for high-stakes claims
   - Results warrant premium venue

**Skip Human Evaluation (Go to Week 9 Paper Writing) if ANY of the following**:

1. **Moderate/Weak Results**:
   - Effect sizes small to medium (d < 0.8)
   - Results not consistent across domains
   - High failure rate (>20%)

2. **Budget Constraints**:
   - Limited budget for MTurk
   - Time pressure (behind schedule)

3. **Venue Flexibility**:
   - Targeting workshop venues (don't require human eval)
   - ArXiv preprint + iterate for next cycle

4. **Automated Evidence Sufficient**:
   - Comprehensive automated metrics tell clear story
   - Qualitative analysis strongly supports metrics
   - Human validation would be confirmatory, not essential

### Decision Outcomes

**OPTION A: Proceed to Human Evaluation**
- Execute Weeks 7-8 (MTurk with anti-AI safeguards)
- Target: NeurIPS 2026 main conference
- Timeline: 15 weeks total
- Budget: $60-90 (APIs) + $120-150 (MTurk) = $180-240

**OPTION B: Skip to Paper Writing**
- Skip to Week 9 (paper writing begins)
- Target: NeurIPS workshops, EMNLP workshops, or ArXiv
- Timeline: 13 weeks total (saves 2 weeks)
- Budget: $60-90 (APIs only)

### Recommendation Process

**Friday Week 6 Activities**:
1. Review all automated metrics and statistical tests
2. Qualitative inspection of best/worst examples
3. Calculate publication readiness score (see below)
4. Make final decision with written justification

**Publication Readiness Score** (out of 10):
- Strong significance (p < 0.01): +2 points
- Large effect sizes (d > 0.8): +2 points
- Consistent across domains: +2 points
- Low failure rate (<20%): +2 points
- Automated metrics validated qualitatively: +2 points

**Decision Rule**:
- Score ≥ 8 + budget available → Proceed to MTurk
- Score < 8 or no budget → Skip to paper writing

---

## Phase 1B (OPTIONAL): Human Validation (Weeks 7-8)

**NOTE**: This phase is OPTIONAL and only executed if Week 6 decision is "PROCEED"

### Week 7: MTurk Setup + Anti-AI Safeguards

**Objectives**:
- Design MTurk HITs with anti-AI protections
- Create qualification test
- Pilot with small batch
- Implement quality control

**Tasks**:

| Day | Task | Time | Details |
|-----|------|------|---------|
| Mon | Design HIT interface | 3h | Simple, clear rating interface |
| Mon | Design qualification test | 2h | 10 examples with obvious answers |
| Tue | Implement anti-AI safeguards | 4h | See details below |
| Tue | Create golden examples | 2h | 5 attention checks with clear answers |
| Wed | Select subset for human eval | 2h | 20-30 examples (stratified sample) |
| Wed | Pilot with 3 workers | 2h | Test HIT, refine instructions |
| Thu | Analyze pilot results | 2h | Check quality, adjust if needed |
| Thu | Launch qualification test | 1h | Workers must pass to access main HITs |
| Fri | Launch main HITs (partial) | 2h | 50% of subset |
| Fri | Monitor for quality issues | 2h | Flag suspicious responses |

**Anti-AI Safeguards**:

1. **Time Tracking**:
   - Log time spent on each rating
   - Flag completions <2 minutes per example (too fast)
   - Flag completions >15 minutes per example (potential automation)
   - **Threshold**: 3-10 minutes expected per example

2. **Attention Checks**:
   - Insert 3-5 golden examples with obvious answers
   - Example: "Rate this clearly excellent synthesis as 5/5"
   - Workers who fail >1 attention check are rejected
   - **Distribution**: 1 attention check per 7-10 real examples

3. **Consistency Checks**:
   - Include 2-3 duplicate examples (same output, different IDs)
   - Flag if ratings differ by >2 points on duplicates
   - Request explanation for large discrepancies
   - **Threshold**: Standard deviation <1.5 across duplicates

4. **Justification Required**:
   - Workers must provide 1-2 sentence explanation per rating
   - Check for generic/copy-pasted explanations
   - Flag workers with >50% similar justifications
   - **Method**: TF-IDF similarity between justifications

5. **Qualification Test**:
   - 10 examples with clear quality differences
   - Must achieve >70% agreement with expert ratings
   - Provides training and filters low-quality workers
   - **Pass rate**: Expected ~60% of applicants

6. **Worker Reputation**:
   - Require >95% approval rate
   - Require >100 approved HITs (experienced workers)
   - Restrict to US-based workers (English fluency)
   - **Pool size**: Expected 200-500 qualified workers

**MTurk Subset Selection**:
- Select 20-30 examples from full dataset (n=50)
- Stratified sampling: 4-6 examples per domain
- Include best and worst examples from automated metrics
- Balanced across conditions (5-8 examples per condition)

**Cost Calculation**:
- 25 examples × 4 conditions = 100 outputs
- 3 workers per output = 300 ratings
- $0.40 per rating (2-4 minutes work)
- Qualification test: $0.20 × 50 workers = $10
- **Total: $120-130 + $10 qual test = $130-140**

**Deliverables**:
- `evaluation/mturk/hit_template.html` (rating interface)
- `evaluation/mturk/qualification_test.json` (10 examples)
- `evaluation/mturk/golden_examples.json` (attention checks)
- `evaluation/mturk/quality_control.py` (automated checks)
- `evaluation/mturk/pilot_results.json` (3 pilot workers)

**Success Criteria**:
- [ ] Qualification test filters low-quality workers
- [ ] Pilot workers pass attention checks
- [ ] Time tracking shows reasonable completion times
- [ ] HIT interface is clear and usable
- [ ] Anti-AI safeguards implemented and working

---

### Week 8: MTurk Collection + Human-Automated Comparison

**Objectives**:
- Collect all human ratings
- Validate quality control
- Compare human ratings to automated metrics
- Compute inter-rater reliability

**Tasks**:

| Day | Task | Time | Details |
|-----|------|------|---------|
| Mon | Launch remaining HITs | 1h | Complete 50% from Week 7 |
| Mon | Monitor for quality issues | 3h | Real-time flagging, worker messaging |
| Tue | **Collect all ratings** (by Tue EOD) | 1h | Download from MTurk |
| Tue | Apply quality control filters | 2h | Remove flagged workers |
| Wed | Compute inter-rater reliability | 2h | Cohen's kappa, Fleiss' kappa, ICC |
| Wed | Aggregate ratings (mean per output) | 1h | Average across 3 workers |
| Thu | **Human-Automated Comparison** | 3h | Correlation analysis |
| Thu | Statistical tests on human ratings | 3h | Same tests as automated metrics |
| Fri | Generate comparison tables | 2h | Human vs automated results |
| Fri | Write human validation section | 3h | Results subsection |

**Quality Control Application**:
- Filter out workers who failed attention checks
- Filter out workers with inconsistent duplicates
- Filter out workers with suspicious timing
- Filter out workers with generic justifications
- **Expected rejection rate**: 10-20% of workers

**Human-Automated Comparison Analysis**:

1. **Correlation Analysis**:
   - Pearson correlation between human ratings and automated metrics
   - **Target**: r > 0.6 (moderate to strong correlation)
   - **Interpretation**: Automated metrics track human judgment

2. **Agreement Analysis**:
   - Do humans and automated metrics agree on ranking?
   - Spearman rank correlation for condition ordering
   - **Target**: ρ > 0.8 (high rank agreement)

3. **Divergence Analysis**:
   - Where do humans and automated metrics disagree?
   - Qualitative analysis of divergent examples
   - **Insight**: What do humans value that metrics miss?

4. **Statistical Validation**:
   - Run same statistical tests on human ratings
   - Compare p-values and effect sizes to automated
   - **Target**: Similar significance and effect sizes

**Inter-Rater Reliability Targets**:
- **Cohen's kappa**: >0.6 (substantial agreement)
- **Fleiss' kappa**: >0.6 (substantial agreement for 3+ raters)
- **Intraclass Correlation Coefficient (ICC)**: >0.7
- **Interpretation**: κ=0.6-0.8 substantial, κ>0.8 almost perfect

**Deliverables**:
- `results/phase1b_human/raw_ratings.json` (all workers)
- `results/phase1b_human/filtered_ratings.json` (after QC)
- `results/phase1b_human/quality_control_report.md` (rejection analysis)
- `results/phase1b_human/inter_rater_reliability.md` (kappa values)
- `results/phase1b_human/human_automated_comparison.md` (correlation)
- `results/phase1b_human/tables/table3_human_results.tex`
- `results/phase1b_human/figures/fig4_human_vs_automated.png`
- `paper/sections/05_results_human_validation.md` (subsection)

**Success Criteria**:
- [ ] Inter-rater reliability >0.6 (substantial agreement)
- [ ] Human ratings confirm automated metric findings
- [ ] Correlation between human and automated >0.6
- [ ] Statistical significance consistent across both
- [ ] Quality control filtered out <20% of workers

---

## Phase 2: Benchmarking (Weeks 7-9 or 9-11)

**NOTE**: Week numbers depend on whether Phase 1B was executed
- If Phase 1B executed: Phase 2 = Weeks 9-11
- If Phase 1B skipped: Phase 2 = Weeks 7-9

### Week 7/9: True Multi-Agent Baseline (Sequential Execution)

**Objectives**:
- Implement true 3-agent system WITHOUT parallel execution
- Measure memory and quality
- Validate memory efficiency claim

**Rationale for Sequential**:
- Parallel 3×12B = 21-30GB + overhead = OOM crash on 24GB RAM
- Sequential: Run Agent 1, save output → Run Agent 2, save output → Run Agent 3
- Peak memory: max(10GB, 10GB, 10GB) = 10GB (feasible)

**Tasks**:

| Day | Task | Time | Details |
|-----|------|------|---------|
| Mon | Design sequential 3-agent protocol | 2h | Agent 1 → Agent 2 → Agent 3 (coordinator) |
| Mon | Implement sequential agent execution | 3h | Load/unload model between agents |
| Tue | Test on 5 examples | 3h | Verify correctness |
| Tue | Measure peak memory per agent | 2h | Instrumentation |
| Wed | **Run true multi-agent on 10 test examples** | 4h | Subset for comparison |
| Thu | Apply automated metrics | 3h | Same metrics as Phase 1 |
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

**True Multi-Agent (Parallel - Ideal)**:
- 3 models simultaneously = 3 × 10GB = 30GB
- This is what you'd need for TRUE parallelism
- But infeasible on 24GB RAM

**Conclusion**: Semantic achieves **3X memory efficiency vs parallel true multi-agent (30GB)**, which is the standard multi-agent architecture but infeasible on consumer hardware (24GB RAM). Compared to sequential execution workaround (10GB), semantic uses comparable memory but provides **2-3X latency advantage** due to cache reuse (no model reloading between agent switches).

**Communication Protocol**:
- Agent 1 generates output
- Agent 2 receives Agent 1 output via concatenation in context
- Agent 3 (coordinator) receives Agent 1 + Agent 2 outputs
- Mirrors semantic coordinator structure for fair comparison

**Deliverables**:
- `src/true_multiagent_sequential.py`
- `results/phase2_multiagent/outputs/` (10 examples)
- `results/phase2_multiagent/memory_comparison.md`
- `results/phase2_multiagent/latency_analysis.md`
- `paper/sections/05_results_multiagent.md` (comparison subsection)

**Success Criteria**:
- [ ] True multi-agent runs without OOM
- [ ] Quality is comparable (not better than semantic)
- [ ] Memory efficiency claim nuanced (3X vs parallel, 1X vs sequential)
- [ ] Semantic has 2-3X latency advantage (cache reuse)

---

### Week 8/10: Quick Ablations + Multi-Model Test

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
| Thu | Apply automated metrics to ablations | 2h | Same metrics as Phase 1 |
| Thu | Compare Gemma vs Llama | 2h | Model-specific findings |
| Fri | Write ablation + multi-model sections | 3h | Results subsections |

**Deliverables**:
- `results/phase2_ablations/2clusters/` (10 outputs)
- `results/phase2_ablations/5clusters/` (10 outputs)
- `results/phase2_multimodel/llama/` (10 outputs)
- `results/phase2_analysis/cluster_count_analysis.md`
- `paper/sections/05_results_ablations.md` (subsection)

**Success Criteria**:
- [ ] 3 clusters is near-optimal (not much gain from 5)
- [ ] Semantic works on Llama (cross-architecture)
- [ ] Ablations show robustness to hyperparameters

---

### Week 9/11: Optional Deployment Study

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
- `paper/sections/06_discussion_deployment.md` (if completed)

---

## Phase 3: Paper Writing (Weeks 10-13 or 12-15)

**NOTE**: Week numbers depend on whether Phase 1B was executed
- If Phase 1B executed: Phase 3 = Weeks 12-15 (uses buffer week)
- If Phase 1B skipped: Phase 3 = Weeks 10-13

### Week 10/12: Introduction + Related Work

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
| Tue | Outline Related Work structure | 2h | 5 subsections |
| Wed | Write Related Work: Multi-Agent Systems | 2h | 0.5 pages |
| Wed | Write Related Work: Multi-Persona Systems | 2h | Distinguish from prompting |
| Thu | Write Related Work: KV Cache Optimization | 2h | FlowKV, EpiCache clearly distinguished |
| Thu | Write Related Work: Multi-User Serving | 2h | Complementary, not competing |
| Fri | Write Related Work: Mixture-of-Experts | 2h | Training-time vs inference-time |
| Fri | Revise Introduction + Related Work | 3h | Coherence, flow |

**Related Work Structure**:

**Section 2.1: Multi-Agent LLM Systems**
- True multi-agent (separate models): Memory overhead
- Agent coordination: Orchestration patterns
- **Our work**: Virtual agents via cache partitioning

**Section 2.2: Multi-Persona LLM Systems**
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

**Section 2.5: Mixture-of-Experts (MoE)**
- MoE models (GPT-4, Mixtral): Training-time specialization
- Router selects experts based on input
- **Distinction**: MoE requires training, our approach is inference-time
- **Complementary**: Could combine KV partitioning with MoE models

**Deliverables**:
- `paper/sections/01_abstract.md`
- `paper/sections/02_introduction.md`
- `paper/sections/03_related_work.md`

**Success Criteria**:
- [ ] Introduction clearly states problem and contribution
- [ ] Related work distinguishes from all prior approaches
- [ ] Positioning as complementary is clear

---

### Week 11/13: Methods + Experiments

**Objectives**:
- Write Methods section (architecture, clustering, evaluation)
- Write Experiments section (dataset, protocol, baselines)

**Tasks**:

| Day | Task | Time | Details |
|-----|------|------|---------|
| Mon | Write Section 4.1: Architecture | 2h | 3-cluster design, coordinator pattern |
| Mon | Write Section 4.2: Semantic Clustering | 2h | Clustering approach |
| Tue | Write Section 4.3: Evaluation Protocol | 3h | Automated metrics, optional human |
| Tue | Create architecture diagram (Figure 1) | 2h | Visual of 3-cluster design |
| Wed | Write Section 5.1: Dataset | 2h | 50 examples, 5 domains |
| Wed | Write Section 5.2: Baselines | 2h | 4 conditions + rationale |
| Thu | Write Section 5.3: Implementation | 2h | MLX, Gemma 3 12B, instrumentation |
| Thu | Write Section 5.4: Automated Metrics | 2h | 12 custom metrics + 4 standard |
| Fri | Polish all methods content | 3h | Ensure reproducibility |

**Deliverables**:
- `paper/sections/04_methods.md`
- `paper/sections/05_experiments.md`
- `paper/figures/fig1_architecture.pdf`

**Success Criteria**:
- [ ] Methods are fully reproducible from description
- [ ] All metrics documented with formulas and interpretations
- [ ] All hyperparameters documented
- [ ] Figures integrated properly

---

### Week 12/14: Results + Discussion

**Objectives**:
- Write Results section (all findings)
- Write Discussion (implications, limitations)
- Integrate all tables and figures

**Tasks**:

| Day | Task | Time | Details |
|-----|------|------|---------|
| Mon | Write Section 6.1: Automated Results | 3h | All metrics, all comparisons |
| Mon | Integrate Table 1 (main results) | 1h | LaTeX table |
| Tue | Write Section 6.2: Human Validation (if done) | 2h | MTurk results |
| Tue | Integrate Figure 2 (box plots) | 2h | Visualization |
| Wed | Write Section 6.3: Multi-Agent Comparison | 2h | Quality + memory |
| Wed | Write Section 6.4: Ablations | 2h | Cluster count, multi-model |
| Thu | Write Section 6.5: Error Analysis | 2h | Failure modes |
| Thu | Write Section 7: Discussion | 3h | Implications, limitations |
| Fri | Write Section 8: Conclusion | 1h | Summary |
| Fri | Full read-through (Sections 1-8) | 3h | Coherence check |

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

### Week 13/15: Revision + Polishing

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
- [ ] <10 pages (NeurIPS limit) or appropriate for venue

---

## Phase 4: Submission Prep (Week 14 or 16)

### Week 14/16: Code Release + Arxiv + Venue Submission

**Objectives**:
- Prepare public GitHub repository
- Submit to Arxiv
- Submit to target venue

**Tasks**:

| Day | Task | Time | Details |
|-----|------|------|---------|
| Mon | Code cleanup | 3h | Remove debug code, add comments |
| Mon | Write comprehensive README | 2h | How to reproduce |
| Tue | Write REPRODUCE.md | 2h | Step-by-step instructions |
| Tue | Test reproducibility | 2h | Fresh checkout, run from scratch |
| Wed | Prepare Arxiv package | 2h | LaTeX + figures |
| Wed | Submit to Arxiv | 1h | Upload |
| Thu | Prepare venue submission | 2h | Format check |
| Thu | **Submit to venue** | 1h | Complete submission |
| Fri | Make GitHub repo public | 1h | Release code |
| Fri | Social media announcements | 1h | Twitter/X thread |

**Target Venue Selection** (based on Week 6 decision):

**Option A (Human Validation Done)**:
- **Primary**: NeurIPS 2026 main conference (May 15 deadline)
- **Fallback**: EMNLP 2026 main conference (if NeurIPS too ambitious)

**Option B (Automated Only)**:
- **Primary**: NeurIPS 2026 workshops (various deadlines)
- **Backup**: EMNLP 2026 workshops
- **Alternative**: ArXiv preprint + iterate for next cycle

**Submission Checklist**:
- [ ] Paper compiles (venue LaTeX template)
- [ ] Supplementary materials (if any)
- [ ] Code repository public with MIT license
- [ ] Arxiv submission complete
- [ ] Venue submission complete

**Deliverables**:
- `README.md` (comprehensive)
- `REPRODUCE.md` (step-by-step)
- `LICENSE` (MIT)
- Arxiv paper (public)
- Venue submission (complete)
- GitHub repository (public)

**Success Criteria**:
- [ ] Arxiv submission public
- [ ] Venue submission complete
- [ ] Code reproducible from fresh checkout

---

## Phase 5: Buffer Week (Week 15 or 17)

### Week 15/17: Contingency Buffer

**Purpose**: Address any delays or unexpected issues

**Possible Uses**:
- Re-run experiments if technical failures occurred
- Additional revision if reviewers at Arxiv raise issues
- Extend deployment study if it was skipped
- Additional ablations if requested
- Extra time for paper writing if behind
- MTurk revision if initial results need refinement

**This buffer provides resilience to timeline.**

---

## Budget Breakdown (Updated for v3)

### Minimum Budget (Automated Only)

| Item | Cost | Notes |
|------|------|-------|
| **Phase 0** | | |
| Claude API (pilot) | $2-3 | 5 examples |
| **Phase 1** | | |
| Claude API (dataset) | $20-25 | 50 examples × ~$0.40 |
| Claude Haiku (metrics) | $25-35 | Automated metrics computation |
| **Phase 2** | | |
| Additional generations | $15-20 | Ablations, baselines |
| **Total (Automated)** | **$60-90** | No human raters |

### Maximum Budget (With Human Validation)

| Item | Cost | Notes |
|------|------|-------|
| **Phase 0-2** | $60-90 | Same as above |
| **Phase 1B (Optional)** | | |
| MTurk qualification test | $10 | 50 workers × $0.20 |
| MTurk ratings (subset) | $120 | 300 ratings × $0.40 |
| **Total (With Human)** | **$190-220** | If Week 6 decision is "proceed" |

**Budget Flexibility**:
- Minimum path: $60-90 (viable for all researchers)
- Maximum path: $190-220 (only if justified by strong results)
- Decision point at Week 6 provides budget control

---

## Timeline Summary

### Two Possible Paths

**Path A: Automated Only** (13 weeks)
| Phase | Weeks | Key Deliverables |
|-------|-------|------------------|
| **Phase 0: Pilot** | 0 | 5 examples, technical validation |
| **Phase 1: Evaluation** | 1-6 | 50 examples, automated metrics, statistics |
| **Phase 2: Benchmarking** | 7-9 | True multi-agent, ablations, multi-model |
| **Phase 3: Writing** | 10-13 | Complete paper draft, revision |
| **Phase 4: Submission** | 14 | Arxiv + workshop submission |
| **Total** | **14 weeks** | Workshop/ArXiv ready |

**Path B: With Human Validation** (15 weeks)
| Phase | Weeks | Key Deliverables |
|-------|-------|------------------|
| **Phase 0: Pilot** | 0 | 5 examples, technical validation |
| **Phase 1: Evaluation** | 1-6 | 50 examples, automated metrics, decision point |
| **Phase 1B: Human** | 7-8 | MTurk validation on subset |
| **Phase 2: Benchmarking** | 9-11 | True multi-agent, ablations, multi-model |
| **Phase 3: Writing** | 12-15 | Complete paper draft, revision (uses buffer) |
| **Phase 4: Submission** | 16 | Arxiv + NeurIPS submission |
| **Total** | **16 weeks** | Conference ready |

**Target Venues**:
- **Path A**: NeurIPS workshops, EMNLP workshops, ArXiv
- **Path B**: NeurIPS 2026 main conference (May 15 deadline)

**Start**: Jan 23, 2026
**Path A Completion**: ~May 1 (14 weeks)
**Path B Completion**: ~May 15 (16 weeks, right at NeurIPS deadline)

---

## Success Criteria (Publication-Ready)

### Must-Have (Blocks Publication)

1. ✅ n≥50 diverse examples across 5 domains
2. ✅ Comprehensive automated metrics suite (16 total metrics)
3. ✅ Statistical significance (p<0.05 FDR-corrected, d>0.5)
4. ✅ Error analysis (failure modes documented)
5. ✅ Complete related work (all distinctions clear)
6. ✅ All figures at 300 DPI
7. ✅ Public code repository
8. ✅ Metric validation (convergent/discriminant validity)

### Strongly Recommended (Strengthens Paper)

9. ⚪ True multi-agent baseline (quality + memory comparison)
10. ⚪ Ablation studies (cluster count sensitivity)
11. ⚪ Multi-model validation (at least Gemma + Llama)
12. ⚪ Additional baselines (random, no-coordinator)

### Optional (For Top-Tier Venues)

13. ⚪ Human validation on subset (MTurk with anti-AI safeguards)
14. ⚪ Inter-rater reliability >0.6 (if human validation done)
15. ⚪ Human-automated correlation >0.6 (validates automated metrics)

### Venue-Specific Requirements

**NeurIPS Main Conference**:
- Strong automated results (d > 0.8)
- Human validation on subset (n=20-30)
- Inter-rater reliability >0.6
- Novel contribution clearly positioned

**Workshop Venues**:
- Strong automated results (d > 0.5)
- Comprehensive metrics (no human required)
- Honest limitations discussion
- Clear positioning vs prior work

**ArXiv Preprint**:
- Complete methodology
- Automated results with statistics
- Reproducible code
- Honest assessment (can report negative results)

---

## Risk Mitigation (Updated for v3)

### Risk 1: Automated Metrics Don't Show Clear Differences

**Mitigation**:
- Pilot testing (Week 0) will catch this early
- 16 diverse metrics increase chance of capturing differences
- Qualitative analysis can supplement weak quantitative results
- Honest reporting: "Marginal benefit" still publishable at workshops

### Risk 2: Week 6 Decision is Difficult

**Mitigation**:
- Clear decision criteria (score ≥ 8 out of 10)
- Written justification required
- Default to conservative choice (skip MTurk if uncertain)
- Workshop venues are respectable fallback

### Risk 3: MTurk Quality Control Fails (if pursued)

**Mitigation**:
- Comprehensive anti-AI safeguards
- Qualification test filters low-quality workers
- Pilot with 3 workers before full launch
- Can reject up to 20% of workers based on QC
- Backup: Use only automated metrics if human data unusable

### Risk 4: Behind Schedule

**Mitigation**:
- Week 15/17 buffer
- Optional components (deployment, multi-model) can be cut
- Phase 1B is entirely optional (saves 2 weeks)
- Focus on Phase 1 (core evaluation) - that alone is sufficient

### Risk 5: Budget Constraints

**Mitigation**:
- Path A (automated only) requires only $60-90
- MTurk is optional and only pursued if justified
- Can reduce MTurk subset size (20 vs 30 examples)
- No rater compensation needed (vs $300 in v2.1)

---

## Changes from v2.1

### Critical Changes

1. **Removed rater dependency**: No need to recruit, train, or pay human raters
2. **Added automated metrics suite**: 16 comprehensive metrics (12 custom + 4 standard)
3. **Added decision point**: Week 6 evaluation determines path forward
4. **Made human validation optional**: Weeks 7-8 only if justified
5. **Reduced minimum budget**: $60-90 (vs $360-390 if raters paid)
6. **Added timeline flexibility**: 13-15 weeks depending on path
7. **Expanded venue options**: Workshops viable with automated-only

### Additions

8. **Contamination metrics**: 4 metrics to detect information leakage
9. **Specialization metrics**: 4 metrics to measure agent distinctiveness
10. **Synthesis metrics**: 4 metrics to evaluate coordinator quality
11. **MTurk anti-AI safeguards**: 6 protections against automated responses
12. **Human-automated comparison**: Validate automated metrics with human subset
13. **Publication readiness score**: Quantitative decision framework
14. **Metric validation**: Convergent/discriminant validity analysis

### Preserved from v2.1

15. **Week 0 pilot**: Still included (critical for validation)
16. **FDR correction**: Still using (less conservative than Bonferroni)
17. **Missing baselines**: Random, no-coordinator still included
18. **Error analysis**: Dedicated time in Week 6
19. **True multi-agent comparison**: Sequential execution (avoids OOM)
20. **Related work positioning**: All 5 subsections preserved
21. **Memory claim framing**: 3X vs parallel, 2-3X latency vs sequential

---

## Deferred to Future Work

Based on debate consensus and automated-first strategy, the following are explicitly **deferred**:

1. **Router agent architecture** (not novel per literature review)
2. **Hierarchical coordination** (multiple coordinator levels)
3. **Dynamic cluster discovery** (inference-time clustering)
4. **Cross-model cache sharing** (combining with C2C communication)
5. **Production deployment** (beyond POC)
6. **Large-scale human evaluation** (n=50 with full human rating)
7. **Real-world user studies** (longitudinal deployment)

These can be follow-up papers after establishing the core contribution with automated metrics.

---

## Evaluation Strategy Rationale

### Why Automated-First?

**Problem with Original Plan (v2.1)**:
- Assumed access to 3 independent human raters
- Reality: Only family members available (bias concerns)
- MTurk risks: AI cheating, quality control challenges
- Expensive: $300+ for comprehensive human evaluation

**Advantages of Automated-First**:
1. **Scalability**: Can evaluate n=50 examples comprehensively
2. **Reproducibility**: Metrics are deterministic and replicable
3. **Cost-effective**: $60-90 vs $360-390
4. **No recruitment delays**: No rater training or coordination
5. **Comprehensive coverage**: 16 metrics > 3 human dimensions
6. **Validation ready**: Can add human validation if results warrant

**Addressing Skepticism**:
- **"Automated metrics miss nuance"**: We include 16 diverse metrics capturing multiple dimensions
- **"Need human validation for top venues"**: We provide optional path (Weeks 7-8) if results strong
- **"Metrics might not correlate with human judgment"**: We validate metrics if we pursue MTurk
- **"Workshops are less prestigious"**: True, but v3 provides flexible path to conferences if justified

### Metric Design Philosophy

**Coverage**: 4 contamination + 4 specialization + 4 synthesis + 4 standard = 16 total metrics

**Validity**:
- Convergent: Related metrics should correlate (r > 0.6)
- Discriminant: Distinct constructs should diverge (r < 0.7)
- Predictive: Metrics should predict human judgment (if validated)

**Interpretability**: Each metric has:
- Clear formula or computation method
- Expected range and target values
- Concrete interpretation guidelines
- Failure mode identification criteria

**Robustness**: Multiple metrics per dimension reduces single-metric dependency

---

## Next Steps

1. ✅ **This plan (v3) addresses evaluation feasibility concerns**
2. ➡️ **Review and approve v3 approach**
3. ➡️ **Begin Week 0** (pilot testing with n=5, validate automated metrics)
4. ➡️ **Iterate if pilot reveals metric issues** → v3.1 if needed
5. ➡️ **Execute Phase 1** (Weeks 1-6) with automated-first strategy
6. ➡️ **Week 6 decision point** (human validation or proceed to writing?)
7. ➡️ **Complete path A or B** based on results and budget

---

**Date**: 2026-01-23
**Status**: Plan v3 ready for review
**Changes**: Automated-first evaluation strategy to address rater availability
**Next**: Review and approval → Begin execution with Week 0 pilot
