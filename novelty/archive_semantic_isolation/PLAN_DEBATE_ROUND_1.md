# Plan Debate Round 1: Comprehensive Review of updated_plan.v1.md

**Date**: 2026-01-23
**Participants**: 3 Skeptics + 3 Proponents
**Target**: Review and improve updated_plan.v1.md → updated_plan.v2.md

---

## Part 1: Timeline Review (All Participants)

### SKEPTIC A (Methodology):
**Timeline Assessment: OPTIMISTIC BUT FEASIBLE**

**Weeks 1-5 (Rigorous Evaluation)**:
- ✅ Week 1 (Dataset): Realistic. Generating 50 examples across 5 domains is achievable.
- ⚠️ Week 2 (Evaluation Framework): **MAJOR CONCERN** - Recruiting 3 independent raters in one week is unrealistic. Graduate students have their own deadlines. This could easily take 2-3 weeks.
- ⚠️ Week 3 (Instrumentation): Solid, but may need buffer for MLX-specific issues.
- ⚠️ Week 4 (Run Evaluation): Assumes no technical failures. Add 1-2 day buffer.
- ✅ Week 5 (Statistical Analysis): Reasonable if data collection goes smoothly.

**Weeks 6-9 (Benchmarking)**:
- ❌ Week 6 (True Multi-Agent): **CRITICAL** - Running 3 separate 12B models simultaneously on 24GB RAM is IMPOSSIBLE. Each quantized model needs ~7-10GB. 3 models = 21-30GB + overhead = OOM crash. This needs redesign.
- ⚠️ Week 7-9 (Ablations): These feel like overkill. Are 4 ablation weeks necessary for a first paper? Consider condensing to 2 weeks.

**Weeks 10-12 (Deployment + Writing)**:
- ⚠️ Week 10 (Deployment): Recruiting 10-15 users and getting meaningful telemetry in 1 week is ambitious. Beta testing typically takes 2-3 weeks.
- ⚠️ Week 11 (Paper Writing): Writing a complete 8-10 page paper in 5 days is brutal. This should be 2 weeks minimum.
- ❌ Week 12 (Submission): ICML 2026 deadline is ~Jan 31. If we start today (Jan 23), we have 8 days, not 12 weeks. **Timeline is inconsistent with venue deadlines.**

**RECOMMENDATION**:
- Extend to **14-16 weeks** minimum
- Fix Week 6 memory issue (use sequential true multi-agent testing, not parallel)
- Add rater recruitment buffer (Week 2 → Weeks 2-3)
- Extend paper writing (Week 11 → Weeks 11-12)
- Target NeurIPS 2026 (May deadline) or EMNLP 2026 (June deadline) instead of ICML

---

### SKEPTIC B (Novelty):
**Timeline Assessment: WEEKS 6-9 ARE BLOAT**

**Core Issue**: Phase 2 (Weeks 6-9) represents 33% of timeline but adds marginal value.

**Week 6 (True Multi-Agent)**:
- As Skeptic A noted, hardware constraints make this infeasible
- Even if solvable, **quality parity is expected**, not superiority
- Memory efficiency claim can be validated through calculation, not measurement
- **VERDICT**: Optional, not critical. Could be future work.

**Week 7 (Ablations)**:
- Testing 2, 3, 5, 7 clusters is thorough but **not novel**
- Standard hyperparameter tuning, not research contribution
- **VERDICT**: Condense to 2-3 days, not full week

**Week 8 (Additional Baselines)**:
- Random clustering: Trivial negative control
- Fixed clustering: Already have turn-based condition
- Attention-based: Interesting but not critical
- **VERDICT**: Random/fixed can be done in 1 day. Drop attention-based.

**Week 9 (Multi-Model)**:
- Testing on 3+ models is good for generalization
- But do we need this for a first paper?
- **VERDICT**: Optional. Main results on Gemma 3 12B are sufficient.

**RECOMMENDATION**:
- **Compress Phase 2 from 4 weeks to 2 weeks**
- Week 6: Drop true multi-agent (hardware infeasible)
- Week 7: Quick ablations (2 days) + error analysis (3 days)
- Weeks 8-9: Delete or defer to follow-up

This reclaims 2-3 weeks for proper paper writing and revision.

---

### SKEPTIC C (Experimental):
**Timeline Assessment: WEEKS 1-5 ARE INSUFFICIENT**

**Critical Missing Elements**:

1. **n=50 is bare minimum**:
   - For 5 domains × 4 conditions = 20 cells
   - Only 2.5 examples per cell
   - Statistical power will be weak (β > 0.2)
   - **RECOMMENDATION**: n=100 (5 examples per cell)

2. **No pilot testing**:
   - Week 1 generates dataset, Week 2 builds evaluation
   - But what if the dataset is wrong? What if conditions don't work?
   - **RECOMMENDATION**: Add Week 0 (pilot with n=5)

3. **Baseline gaps**:
   - 4 conditions test isolation methods
   - But missing **true multi-agent baseline** (Skeptic A notes it's infeasible)
   - Missing **random clustering** baseline (necessary negative control)
   - Missing **no-coordinator** ablation (is coordinator necessary?)
   - **RECOMMENDATION**: Add 2-3 more conditions

4. **Domain coverage**:
   - 5 domains sounds good, but are they balanced?
   - Plan says "coding, research, business, support, creative"
   - But "coding" might dominate if dataset comes from GitHub examples
   - **RECOMMENDATION**: Specify 10 examples per domain (stratified)

**Statistical Power Analysis**:
- Paired t-test with n=50, α=0.05, d=0.5 → power β=0.70 (acceptable)
- But with Bonferroni correction for 6 comparisons → α=0.008 → power drops to β=0.45 (underpowered)
- **RECOMMENDATION**: Either n=100 or reduce comparisons (focus on semantic vs sequential)

**VERDICT**: Weeks 1-5 need expansion:
- Week 0: Pilot (new)
- Week 1-2: Dataset (n=100, not n=50)
- Week 3: Evaluation framework (extended to include all baselines)
- Week 4-5: Run experiments (now 400 runs instead of 200)
- Week 6: Statistical analysis

This pushes timeline to **6 weeks for Phase 1**, not 5.

---

### PROPONENT A (Technical):
**Timeline Assessment: REALISTIC WITH ADJUSTMENTS**

**Defense of Weeks 1-5**:
- n=50 is sufficient for **effect size demonstration**, even if underpowered for p<0.05
- Modern ML papers report effect sizes (Cohen's d) as primary metric
- If d>0.8 (large effect), reviewers won't care about p=0.06 vs p=0.04
- **VERDICT**: n=50 is fine, n=100 is better but not critical

**Defense of Weeks 6-9**:
- I concede Week 6 (true multi-agent) is infeasible due to memory
- But Weeks 7-9 (ablations, baselines, multi-model) are **standard due diligence**
- Reviewers will ask: "Did you try different cluster counts?" → Need Week 7
- Reviewers will ask: "Does this work on other models?" → Need Week 9
- **VERDICT**: Keep Weeks 7 and 9, compress to 1 week each

**Defense of Weeks 10-12**:
- Week 10 (deployment) is **nice-to-have, not critical**
- Week 11 (paper writing) is aggressive but doable with good notes
- Week 12 (revision) is reasonable
- **VERDICT**: Deployment is optional. Extend writing to 2 weeks.

**Technical Feasibility Check**:
- ✅ MLX + Gemma 3 12B (4-bit) fits in 24GB RAM
- ✅ Instrumentation overhead <5% (MLX has good profiling)
- ✅ ROUGE, BERTScore implemented in standard libraries
- ❌ True multi-agent (3×12B) does NOT fit in 24GB
- ✅ Sequential generation (one agent at a time) is feasible

**RECOMMENDATION**:
- **Total timeline: 13-14 weeks** (not 12)
- Phase 1: 5 weeks (as planned)
- Phase 2: 2 weeks (compress from 4)
- Phase 3: 6 weeks (extend from 3)

---

### PROPONENT B (Novelty):
**Timeline Assessment: FOCUS ON WHAT MATTERS**

**The Core Contribution**:
Our novelty is **semantic KV cache partitioning for virtual agents**. Everything else is validation.

**Weeks 1-5 (Rigorous Evaluation)**: ✅ CRITICAL
- This is where we prove the concept works
- Without this, we have no paper
- **VERDICT**: Must be done thoroughly

**Weeks 6-9 (Benchmarking)**: ⚠️ MIXED VALUE
- Week 6 (true multi-agent): Infeasible and not critical
- Week 7 (ablations): Useful but not novel
- Week 8 (baselines): Negative controls are important
- Week 9 (multi-model): Nice but not essential
- **VERDICT**: Keep Week 7-8, drop Week 6 and 9

**Weeks 10-12 (Deployment + Writing)**: ✅ CRITICAL
- Week 10 (deployment): Actually VERY valuable for real-world validation
- Weeks 11-12 (writing): Cannot be rushed
- **VERDICT**: Keep deployment, extend writing

**Addressing Debate Consensus Weaknesses**:

From DEBATE_FINAL_CONSENSUS.md, reviewers asked for:
1. ✅ n=50 diverse examples → Addressed in Week 1
2. ✅ Blind evaluation with multiple raters → Addressed in Week 2
3. ✅ Automated metrics + human evaluation → Addressed in Week 2-3
4. ✅ Statistical tests with effect sizes → Addressed in Week 5
5. ✅ Comparison to true multi-agent → Week 6 (but infeasible)
6. ✅ Ablation studies → Week 7
7. ⚠️ Multiple models → Week 9 (optional)

**VERDICT**: Plan addresses most consensus items. True multi-agent is the gap.

**RECOMMENDATION**:
- Keep Phase 1 (Weeks 1-5)
- Compress Phase 2 (Weeks 6-9 → Weeks 6-7)
- Extend Phase 3 (Weeks 10-12 → Weeks 8-11)
- **Total: 11 weeks**

---

### PROPONENT C (Experimental):
**Timeline Assessment: STATISTICAL RIGOR VS PRAGMATISM**

**Response to Skeptic C**:

**n=50 vs n=100**:
- Skeptic C is technically correct: n=100 has better power
- But **data collection is expensive**: 50 examples × 2h each = 100h of work
- And we're measuring large effects (d>0.5), not marginal effects
- **VERDICT**: n=50 for initial submission, n=100 if revisions needed

**Pilot Testing**:
- Good idea! Week 0 with n=5 catches major issues early
- Costs only 2-3 days
- **VERDICT**: Add Week 0 (pilot)

**Baseline Gaps**:
- Random clustering: Yes, needed as negative control (1 day)
- No-coordinator ablation: Interesting but not critical
- True multi-agent: Infeasible on hardware
- **VERDICT**: Add random clustering to Week 4

**Domain Balance**:
- Plan specifies 10 examples per domain (5 domains × 10 = 50)
- This is already stratified
- **VERDICT**: No change needed

**Statistical Tests**:
- Bonferroni correction: Conservative but standard
- Alternative: FDR correction (less conservative)
- Or focus on primary hypothesis (semantic vs sequential) without correction
- **VERDICT**: Use FDR (Benjamini-Hochberg) instead of Bonferroni

**5 Domains Coverage**:
The 5 domains (coding, research, business, support, creative) are reasonable, but:
- ⚠️ "Business" is vague. Replace with "financial analysis" or "strategic planning"
- ⚠️ Missing technical domains like "medical" or "legal"
- ✅ 5 is sufficient for generalization claim

**RECOMMENDATION**:
- Add Week 0 (pilot, n=5)
- Keep n=50 (increase to n=100 only if reviewers request)
- Add random clustering baseline (1 day in Week 4)
- Use FDR correction instead of Bonferroni
- **Phase 1: 6 weeks** (was 5)

---

## Part 2: Evaluation Protocol Review

### SKEPTIC A (Methodology):
**EVALUATION PROTOCOL CRITIQUE**

#### Blind Evaluation with 3 Raters

**Current Plan**:
- 3 independent raters
- Blind to condition labels
- Cohen's kappa >0.6 target
- 0-5 scale rubric

**Issues**:

1. **Rater Recruitment** (Week 2, Day Tue):
   - "Recruit 3 independent raters" in 1 day is unrealistic
   - Graduate students are busy, need lead time
   - Should start recruitment in Week 1, finalize in Week 2
   - **FIX**: Begin recruitment during Week 1

2. **Rater Training** (Week 2, Day Tue, 2h):
   - 2 hours of training for a complex rubric is insufficient
   - Need practice examples + feedback + calibration
   - Industry standard: 4-8 hours over 2 sessions
   - **FIX**: Extend to 4 hours over 2 days

3. **Inter-Rater Reliability** (Cohen's kappa >0.6):
   - κ=0.6 is "moderate agreement" (not "substantial")
   - Substantial agreement is κ>0.8
   - With κ=0.6, ratings are noisy
   - **FIX**: Target κ>0.7 minimum, κ>0.8 ideal

4. **Number of Raters**:
   - 3 is minimum for reliability (can compute Fleiss' kappa)
   - But what if one rater drops out? → Down to 2 (only Cohen's kappa)
   - **FIX**: Recruit 4 raters, use best 3

#### Rubric (0-5 Scale)

**Current Design**:
- Agent Specialization (0-5)
- Cross-Contamination (0-5, reverse scored)
- Synthesis Quality (0-5)

**Issues**:

1. **Scale Granularity**:
   - 6-point scale (0-5) is fine, but only 3 anchors given (0, 3, 5)
   - What's a 1? What's a 4?
   - **FIX**: Define all 6 levels or use 4-point scale (0-3)

2. **Cross-Contamination Measurement**:
   - Rubric describes it qualitatively
   - But contamination should be **quantifiable**: % of tokens in wrong cluster
   - **FIX**: Add automated contamination metric (% of semantic drift)

3. **Missing Dimensions**:
   - No measurement of **efficiency** (fewer tokens wasted)
   - No measurement of **coherence** (does output make sense?)
   - No measurement of **correctness** (did it solve the task?)
   - **FIX**: Add Task Success metric (0-5, did it accomplish the goal?)

4. **Rubric Validation**:
   - Plan includes pilot (Week 2, Fri), good
   - But no mention of measuring construct validity
   - Do specialization scores actually correlate with automated metrics?
   - **FIX**: Report correlations between human and automated scores

#### Automated Metrics

**Current Plan**:
- ROUGE (lexical overlap)
- BERTScore (semantic similarity)
- Semantic similarity (embeddings)

**Issues**:

1. **ROUGE for Multi-Agent**:
   - ROUGE measures overlap with reference
   - But we don't have gold references (each condition generates different output)
   - ROUGE makes sense for comparing **condition A vs condition B** output overlap
   - But what's the reference? The task description?
   - **FIX**: Clarify what ROUGE is comparing against

2. **BERTScore Limitations**:
   - BERTScore measures semantic similarity to reference
   - Same issue: what's the reference?
   - BERTScore is good for machine translation (gold reference exists)
   - But for generation, it's less interpretable
   - **FIX**: Use BERTScore to measure **intra-cluster coherence** (embedding similarity within cluster)

3. **Missing Metrics**:
   - No measurement of **redundancy** (cross-cluster overlap)
   - No measurement of **coverage** (how much of task is addressed)
   - No measurement of **latency** (already planned in instrumentation)
   - **FIX**: Add redundancy metric (embedding similarity between clusters)

4. **Metric Validation**:
   - Plan says "Automated metrics correlate with human scores"
   - But no target correlation (r>0.5? r>0.7?)
   - **FIX**: Specify r>0.6 minimum for validity

#### Evaluation Pipeline

**Current Plan**:
- Batch processing
- 200 outputs × 3 raters = 600 ratings
- Process in <4 hours

**Issues**:

1. **Rater Fatigue**:
   - Each rater evaluates 200 outputs × 3 dimensions = 600 ratings
   - At 1-2 minutes per rating → 600-1200 minutes = 10-20 hours per rater
   - Doing this in one weekend leads to fatigue and poor quality
   - **FIX**: Split evaluation over 2 weekends or 1 week

2. **Rating Interface**:
   - No mention of rating tool (Google Forms? Custom UI?)
   - Raters need to see output + rubric + input side-by-side
   - **FIX**: Build simple web interface for rating

3. **Quality Control**:
   - No mention of attention checks or gold examples
   - Standard practice: Insert 5-10 gold examples to check rater quality
   - **FIX**: Add 10 gold examples with known scores

#### Statistical Analysis

**Current Plan**:
- Paired t-tests
- Wilcoxon (non-parametric backup)
- Bonferroni correction
- Cohen's d (effect size)

**Issues**:

1. **Bonferroni is Conservative**:
   - 6 pairwise comparisons → α=0.05/6=0.008
   - Very strict, may miss real effects
   - **FIX**: Use FDR (Benjamini-Hochberg) or Holm-Bonferroni (less conservative)

2. **Paired vs Independent**:
   - Plan says "paired t-test" (correct, same examples across conditions)
   - But need to verify assumptions (normality, equal variance)
   - **FIX**: Check assumptions, use Wilcoxon if violated

3. **Domain-Specific Analysis** (Week 5, Wed):
   - Good idea, but needs correction for multiple testing (5 domains)
   - **FIX**: Apply FDR correction within domain analysis too

4. **Missing Analyses**:
   - No mention of **failure mode analysis** (when does it fail?)
   - No mention of **example difficulty** (does it work better on easy tasks?)
   - **FIX**: Add qualitative error analysis (Week 5, Thu-Fri)

---

**SKEPTIC A RECOMMENDATIONS FOR v2**:

#### Evaluation Protocol
1. **Rater Recruitment**: Start in Week 1, recruit 4 raters
2. **Rater Training**: 4 hours over 2 sessions with calibration
3. **Inter-Rater Reliability**: Target κ>0.7 (minimum), κ>0.8 (ideal)
4. **Rubric**: Define all 6 levels (0-5) with examples
5. **Add Task Success Metric**: 0-5 scale, "Did it solve the problem?"
6. **Evaluation Timeline**: Split over 1 week (not 1 weekend)
7. **Rating Interface**: Build simple web UI with side-by-side view
8. **Quality Control**: Add 10 gold examples
9. **Statistical Correction**: Use FDR instead of Bonferroni
10. **Error Analysis**: Add qualitative analysis of failures (Week 5)

#### Automated Metrics
1. **ROUGE**: Clarify reference (task description? ground truth?)
2. **BERTScore**: Use for intra-cluster coherence measurement
3. **Add Redundancy Metric**: Inter-cluster embedding similarity
4. **Metric Validation**: Target r>0.6 correlation with human scores

#### Timeline Impact
- Week 1: Add rater recruitment (+1 day)
- Week 2: Extend rater training (+1 day)
- Week 4: Extend evaluation window (weekend → 1 week)
- Week 5: Add error analysis (+2 days)
- **Total: Add 1 week buffer**

---

### PROPONENT A (Technical):
**DEFENSE OF EVALUATION PROTOCOL**

#### Response to Skeptic A's Critique

I agree with most of Skeptic A's points, but some concerns are overstated:

**Rater Recruitment**:
- Yes, 1 day is tight, but we can start outreach in Week 1
- Offer co-authorship or acknowledgment → easier recruitment
- **CONCESSION**: Start earlier (Week 1)

**Inter-Rater Reliability (κ>0.6)**:
- κ=0.6 is standard in NLP (see SQuAD, CoNLL annotation studies)
- κ=0.8 is ideal but hard to achieve with subjective metrics
- Even top conferences accept κ=0.6-0.7
- **CONCESSION**: Target κ>0.7, report if lower

**Rubric Granularity**:
- We provide anchors at 0, 3, 5
- Raters can interpolate (1, 2, 4) based on their judgment
- This is standard in Likert scales
- **NO CHANGE NEEDED**

**ROUGE and BERTScore References**:
- Skeptic A is right: we need to clarify what we're comparing
- ROUGE/BERTScore should measure **intra-cluster coherence**, not inter-condition similarity
- For each cluster, compute average pairwise BERTScore among turns
- High score → coherent specialization
- **CONCESSION**: Clarify metric definitions in v2

**Rater Fatigue**:
- 200 outputs × 3 dimensions = 600 ratings
- At 1.5 min per rating = 900 min = 15 hours per rater
- This is A LOT. Skeptic A is right.
- **CONCESSION**: Reduce to n=30 per rater (split 3 raters × 30 = 90 overlaps for reliability)

**Attention Checks**:
- Good idea, standard practice
- **CONCESSION**: Add 10 gold examples

**Statistical Correction**:
- Bonferroni is conservative but standard
- FDR is reasonable alternative
- **CONCESSION**: Use FDR

---

**PROPONENT A COUNTER-RECOMMENDATIONS**:

1. ✅ Agree: Start rater recruitment in Week 1
2. ⚠️ Partial: Target κ>0.7, accept κ>0.6 if necessary
3. ❌ Disagree: 0-5 scale with 3 anchors is fine
4. ✅ Agree: Clarify ROUGE/BERTScore definitions
5. ✅ Agree: Add redundancy metric (inter-cluster similarity)
6. ✅ Agree: Reduce rater burden (n=30 per rater, 90 overlap)
7. ✅ Agree: Add gold examples (10)
8. ✅ Agree: Use FDR correction
9. ✅ Agree: Add error analysis (Week 5)

**Net Result**: Evaluation protocol is mostly sound, with minor refinements needed.

---

## Part 3: Router Agent Architecture Review

### SKEPTIC B (Novelty):
**ROUTER AGENT PRIOR ART ANALYSIS**

#### Literature Review Summary

I searched for:
1. "router agent LLM multi-agent architecture"
2. "meta-agent routing LLM dynamic cluster assignment"
3. "LLM agent orchestration router pattern coordinator"
4. "adaptive agent clustering multi-agent LLM systems"

#### Key Findings: ROUTER AGENTS ARE NOT NOVEL

**1. MasRouter (ACL 2025)**
- Paper: "MasRouter: Learning to Route LLMs for Multi-Agent System"
- Description: Uses a **unified routing framework** with three-layer decision architecture
- Components: Collaboration mode determination, role allocation, LLM routing
- Verdict: **DIRECTLY OVERLAPS** with proposed router agent

**2. AgentRouter (arXiv 2025)**
- Paper: "AgentRouter: A Knowledge-Graph-Guided LLM Router for Collaborative Multi-Agent QA"
- Description: Knowledge-graph-guided routing supervised by performance signals
- Verdict: **SIMILAR CONCEPT**, KG-based instead of semantic

**3. Super Agent System with Hybrid AI Routers**
- Description: Routes queries to different models based on complexity
- Example: Complex math → GPT-4, simple chat → GPT-3.5
- Verdict: **TASK-BASED ROUTING**, not agent-based, but same pattern

**4. Orchestrator Pattern (Industry Standard)**
- Description: Central LLM coordinator that dispatches to specialist agents
- Used in: LangChain, LangGraph, OpenAI Agents SDK, AWS Bedrock
- Verdict: **ROUTER AGENT = ORCHESTRATOR**, not novel

**5. SC-MAS Framework**
- Description: Node Selector, Edge Optimizer, and **LLM Router** modules
- Uses utility function penalized by execution cost
- Verdict: **EXPLICIT "LLM ROUTER"**, same concept

#### Router Agent Novelty Assessment: NOT NOVEL

**Proposed in updated_plan.v1.md**:
> "Use a separate model instance (lightweight) as a router agent with its own KV cache to decide where to route requests or create new clusters."

**Prior Art**:
- MasRouter: ✅ Routing framework for multi-agent systems
- AgentRouter: ✅ Router for multi-agent QA
- Orchestrator Pattern: ✅ Central coordinator in multi-agent architectures
- SC-MAS LLM Router: ✅ Explicit router module

**Conclusion**: Router agents are a **well-established pattern** in multi-agent LLM systems (2024-2025).

#### Does Router Agent Add Value?

**Advantages** (from plan):
- Fully self-consistent (routing decisions in KV cache)
- Can reason about routing (not just embedding similarity)
- Scales to dynamic cluster discovery

**Disadvantages** (from plan):
- Adds overhead (extra model instance)
- Complicates architecture
- May not be necessary if embedding routing works

**My Analysis**:

**Against Router Agent**:
1. **Complexity without Benefit**: Embedding-based routing (current approach) is simpler and works
2. **Not Novel**: Router agents exist in literature (MasRouter, AgentRouter, SC-MAS)
3. **Overhead**: Adds latency and memory (defeats the efficiency claim)
4. **Scope Creep**: We're trying to do too much in one paper

**For Router Agent**:
1. **Dynamic Adaptation**: Could discover new clusters on-the-fly
2. **Reasoning-Based**: Could handle edge cases better than embeddings
3. **Flexibility**: More general than fixed semantic clustering

**Trade-Off Analysis**:

| Aspect | Embedding Routing | Router Agent |
|--------|------------------|--------------|
| Novelty | ⚠️ Moderate | ❌ Low (prior art exists) |
| Simplicity | ✅ High | ❌ Low |
| Efficiency | ✅ High (no extra model) | ❌ Lower (extra model) |
| Adaptability | ❌ Fixed clusters | ✅ Dynamic clusters |
| Implementation | ✅ Easy (done) | ❌ Hard (need to build) |

**Verdict**: Router agent adds **marginal value at high cost**.

---

**SKEPTIC B RECOMMENDATION**:

### Drop Router Agent from updated_plan.v2.md

**Rationale**:
1. Not novel (prior art in MasRouter, AgentRouter, orchestrator patterns)
2. Adds complexity and overhead
3. Contradicts efficiency claim (3X memory reduction)
4. Embedding-based semantic routing is sufficient for first paper

**Alternative**:
- Keep current approach: DeepSeek R1 preprocessing for cluster discovery
- Use embedding similarity for routing (no extra model needed)
- Mention router agent as **future work** (dynamic adaptation)

**If Authors Insist on Router Agent**:
- Must cite MasRouter, AgentRouter, SC-MAS (show awareness of prior art)
- Must empirically demonstrate benefit over embedding routing
- Must measure overhead (latency, memory)
- Must justify why reasoning-based routing beats semantic routing

---

### PROPONENT B (Novelty):
**DEFENSE OF ROUTER AGENT CONCEPT**

#### Response to Skeptic B

Skeptic B found prior art (MasRouter, AgentRouter, orchestrator patterns). I acknowledge this.

However, **context matters**:

**MasRouter** (ACL 2025):
- Routes between **multiple separate LLM instances** (different models or instances)
- Not about **KV cache partitioning within a single model**
- Routing is at the **system level**, not **cache level**

**AgentRouter**:
- Knowledge-graph-guided routing for **multi-agent QA**
- Uses explicit knowledge graphs, not semantic embeddings
- Again, routing between separate agents, not cache partitions

**Orchestrator Pattern**:
- Industry pattern for **separate agents** (microservices-style)
- Not applicable to **single-model virtual agents**

**SC-MAS LLM Router**:
- Routes to different LLMs based on cost/performance
- Model selection, not cache routing

#### Key Distinction: Our Router Agent is Different

**Prior Art**: Router selects **which external agent/model** to invoke
**Our Proposal**: Router selects **which KV cache partition** to use within same model

This is a **different level of abstraction**:
- Prior art: Inter-agent routing (between processes/models)
- Our proposal: Intra-model routing (between cache partitions)

**Analogy**:
- Prior art = Network router (which server handles request?)
- Our proposal = CPU scheduler (which core/cache gets task?)

#### Novelty Claim: Router Agent for KV Cache Partitioning

**What's Novel**:
- Using a lightweight reasoning model to decide cache routing
- Maintaining routing state in the router's own KV cache
- Dynamic cluster discovery and assignment

**What's Not Novel**:
- General concept of routing in multi-agent systems (prior art exists)

#### Does It Add Value?

Skeptic B says "marginal value at high cost." I disagree.

**Value Proposition**:

1. **Dynamic Adaptation**:
   - Current approach: Fixed 3 clusters (predetermined by DeepSeek R1)
   - Router agent: Can discover need for 4th cluster mid-conversation
   - Example: User starts with coding task, then asks legal question → create legal cluster on-the-fly

2. **Reasoning-Based Routing**:
   - Embedding similarity can fail on edge cases
   - Example: "Review this code for security vulnerabilities" → Security cluster or Code cluster?
   - Router agent can reason: "Security review is specialized, create security cluster"

3. **Self-Consistency**:
   - All routing decisions are logged in router's KV cache
   - Can explain why routing decisions were made
   - Useful for debugging and interpretability

**Overhead Analysis**:

| Component | Memory | Latency |
|-----------|--------|---------|
| Main model (Gemma 3 12B, 4-bit) | 7-10GB | ~500ms/turn |
| Router model (Gemma 3 2B, 4-bit) | 1-2GB | ~50ms/routing |
| Router KV cache | 50MB | Negligible |
| **Total overhead** | **1-2GB** | **~50ms** |

**Impact**:
- Memory: 1-2GB extra (8-12GB total, still < 3×12GB = 36GB for true multi-agent)
- Latency: 50ms routing per turn (~10% overhead)

**Verdict**: Overhead is **acceptable** if router adds value.

---

**PROPONENT B RECOMMENDATION**:

### Keep Router Agent as Optional Extension (Week 7-8)

**Rationale**:
1. It's different from prior art (intra-model cache routing, not inter-agent routing)
2. Overhead is acceptable (1-2GB, 50ms)
3. Enables dynamic cluster discovery (key advantage)
4. Can be implemented in 1 week (after main results are solid)

**Implementation Plan**:
- Week 1-6: Main evaluation with embedding-based routing (baseline)
- Week 7: Implement router agent
- Week 8: Compare embedding vs router on 10 examples
- Paper: Report both, highlight router agent as improvement

**Alternative (Conservative)**:
- Skip router agent in v1.0 paper
- Publish as follow-up work (v2.0)
- Cite MasRouter/AgentRouter, position as "KV-cache-level routing"

**If Router Agent is Included**:
Must address Skeptic B's concerns:
1. ✅ Cite MasRouter, AgentRouter, SC-MAS
2. ✅ Clarify difference (cache-level vs agent-level routing)
3. ✅ Empirically demonstrate benefit (dynamic adaptation)
4. ✅ Measure overhead (1-2GB, 50ms)
5. ✅ Show when router beats embeddings (edge cases)

---

## Part 4: Baselines and Comparisons Review

### SKEPTIC C (Experimental):
**BASELINE AND COMPARISON CRITIQUE**

#### Current Baselines (4 Conditions)

From updated_plan.v1.md:
1. **Sequential** (baseline): No isolation, single KV cache
2. **Prompted** (soft isolation): Instructional boundaries
3. **Turn-Based** (naive temporal isolation): New cache every turn
4. **Semantic** (RDIC): KV cache partitioning by semantic clustering

**Assessment**: These 4 conditions test **isolation methods**, which is good.

#### Missing Baselines

**1. True Multi-Agent (Separate Models)**
- Status: Week 6 plans this
- Issue: Skeptic A noted hardware infeasible (3×12B = 30GB on 24GB RAM)
- Criticality: **CRITICAL** - Without this, we can't claim "achieves parity at 1/3 memory"
- Solution: Run sequentially (one model at a time), aggregate results

**2. Random Clustering**
- Description: Randomly assign turns to clusters (negative control)
- Rationale: Tests if **any** clustering helps, or if semantic clustering is necessary
- Criticality: **IMPORTANT** - Strong baseline for showing semantic is necessary
- Cost: 1 day to implement and run
- Status: **MISSING** in Week 4

**3. Fixed Clustering (by Position)**
- Description: Cluster by turn position (turns 1-5, 6-10, 11-15)
- Rationale: Tests if temporal boundaries with clustering work
- Criticality: **MODERATE** - Similar to turn-based condition
- Status: Week 8 plans this

**4. Attention-Based Clustering**
- Description: Cluster by attention pattern similarity
- Rationale: Alternative to semantic clustering (model-driven, not reasoning-driven)
- Criticality: **INTERESTING** but not critical
- Status: Week 8 plans this

**5. No-Coordinator Ablation**
- Description: Run with 2 specialist clusters only, no synthesis step
- Rationale: Tests if coordinator is necessary
- Criticality: **IMPORTANT** - Shows value of coordinator pattern
- Status: **MISSING**

#### Ablation Studies (Week 7)

**Current Plan**:
- Cluster count: 2, 3, 5, 7
- Routing threshold: 0.5, 0.7, 0.9
- Cache size limits

**Assessment**:

**Cluster Count Ablation**: ✅ GOOD
- Tests sensitivity to granularity
- Expected: Diminishing returns beyond 3-4 clusters
- Important for hyperparameter tuning

**Routing Threshold Ablation**: ✅ GOOD
- Tests sensitivity to semantic similarity cutoff
- Expected: Sweet spot around 0.6-0.8
- Important for robustness claim

**Cache Size Limits**: ⚠️ INTERESTING but not critical
- Tests what happens with memory budget
- Useful for deployment scenarios
- Not critical for core novelty claim

#### Multi-Model Validation (Week 9)

**Current Plan**:
- Gemma 3 12B (primary)
- Gemma 2 9B (smaller)
- Llama 3.1 8B (different architecture)
- Qwen 2.5 14B (larger)

**Assessment**: ✅ GOOD for generalization

But:
- Week 9 is expensive (setup + run 3 models)
- Is this necessary for **first paper**?
- Or can we defer to follow-up?

**Counter-Argument**: Reviewers will ask "Does this only work on Gemma?"
- Testing on 2-3 models is standard due diligence
- Week 9 is justified

---

**SKEPTIC C RECOMMENDATIONS**:

#### Must-Have Baselines (Add to v2)

1. **True Multi-Agent** (Week 6):
   - ✅ Keep but fix: Run sequentially, not parallel
   - Modify: "3 separate Gemma 3 12B instances" → "3 sequential calls"
   - Measure total memory (1 model at a time) and quality

2. **Random Clustering** (Week 4):
   - ✅ Add: 1 day implementation, run on full dataset
   - Rationale: Negative control

3. **No-Coordinator Ablation** (Week 7):
   - ✅ Add: 2 days implementation, run on 10 examples
   - Rationale: Test coordinator necessity

#### Nice-to-Have Baselines (Optional)

4. **Fixed Clustering** (Week 8):
   - ⚪ Keep if time allows
   - Similar to turn-based condition

5. **Attention-Based Clustering** (Week 8):
   - ⚪ Defer to future work
   - Interesting but not critical

#### Ablations

- ✅ Keep cluster count ablation (Week 7)
- ✅ Keep routing threshold ablation (Week 7)
- ⚪ Cache size limits: Optional

#### Multi-Model Validation

- ✅ Keep but reduce scope: Gemma 3 12B + Llama 3.1 8B (2 models, not 4)
- Week 9 → Week 8 (1 week, not full week)

---

**CRITICAL MISSING COMPARISON: True Multi-Agent**

Skeptic A noted Week 6 is infeasible (memory). I agree.

**Solution**: Modify approach
- **Current Plan**: Run 3 models simultaneously (30GB RAM)
- **Fixed Plan**: Run 3 models sequentially (10GB RAM peak)
- Agent 1 generates → save output
- Agent 2 reads output, generates → save output
- Agent 3 (coordinator) reads both, generates synthesis

**Feasibility**: ✅ Fits in 24GB RAM

**Measurement**:
- Memory: Peak RAM during each agent call (1×10GB)
- Quality: Same blind evaluation as semantic condition
- Comparison: Semantic (1 model, 10GB) vs True (3 calls, 3×10GB effective)

**Result**: Can claim "achieves X% of true multi-agent quality at 1/3 effective memory"

**Criticality**: This comparison is **ESSENTIAL** for the paper.
- Without it, reviewers will ask "How does this compare to real multi-agent?"
- Answering "we didn't test it" is unacceptable

---

### PROPONENT C (Experimental):
**DEFENSE OF EXPERIMENTAL DESIGN**

#### Response to Skeptic C

I largely agree with Skeptic C's analysis. The experimental design is solid but has gaps.

**Agreements**:
1. ✅ True multi-agent baseline is critical (fix Week 6 to sequential execution)
2. ✅ Random clustering is useful negative control (add to Week 4)
3. ✅ No-coordinator ablation is interesting (add to Week 7)
4. ✅ Multi-model validation can be reduced (2 models, not 4)

**Disagreements**:
1. ⚠️ Attention-based clustering: I think it's worth exploring (not defer)
   - Only 1 day to implement
   - Provides interesting comparison: reasoning-based (semantic) vs model-driven (attention)
   - Could be a strong result if attention-based fails

2. ⚠️ Cache size limits ablation: I think it's valuable
   - Real deployment scenario: memory-constrained devices
   - Shows robustness to budget constraints
   - Only 1 day to run

#### Statistical Power Re-Analysis

Skeptic C (in Timeline Review) said n=50 is underpowered with Bonferroni correction.

Let me recalculate:

**Scenario 1: Bonferroni Correction**
- 4 conditions → 6 pairwise comparisons
- α = 0.05 / 6 = 0.008
- n = 50, d = 0.5 (moderate effect)
- Power β ≈ 0.45 (underpowered)

**Scenario 2: FDR Correction**
- α = 0.05 (less conservative)
- n = 50, d = 0.5
- Power β ≈ 0.70 (acceptable)

**Scenario 3: Focus on Primary Comparison (Semantic vs Sequential)**
- No multiple comparison correction
- α = 0.05
- n = 50, d = 0.5
- Power β ≈ 0.85 (well-powered)

**Recommendation**: Use FDR correction + focus narrative on primary comparison

This way:
- n=50 is sufficient
- Don't need to expand to n=100 (saves time)
- Still report all pairwise comparisons

#### n=50 vs n=100 Trade-Off

| Aspect | n=50 | n=100 |
|--------|------|-------|
| Statistical Power (FDR) | β=0.70 | β=0.95 |
| Cost (dataset generation) | 100h | 200h |
| Cost (evaluation runs) | 200 runs | 400 runs |
| Cost (rater time) | 600 ratings | 1200 ratings |
| Timeline | 5 weeks | 7 weeks |

**Verdict**: n=50 with FDR is **sufficient**. n=100 is better but not worth 2 extra weeks.

#### Domain Coverage Re-Analysis

Current plan: 5 domains (coding, research, business, support, creative) × 10 examples each = 50

**Issues**:
- "Business" is vague
- Missing technical domains (medical, legal, scientific)

**Proposed Domains** (more specific):
1. **Software Engineering** (debugging, code review, documentation)
2. **Scientific Research** (literature review, experiment design, paper writing)
3. **Financial Analysis** (market research, financial modeling, reporting)
4. **Technical Support** (troubleshooting, diagnosis, resolution)
5. **Creative Writing** (storytelling, editing, critique)

**Alternative Domains** (if we want broader coverage):
1. Software Engineering
2. Scientific Research
3. Medical Diagnosis (symptom analysis, differential diagnosis, treatment planning)
4. Legal Analysis (case review, contract analysis, brief writing)
5. Customer Service (inquiry handling, escalation, resolution)

**Recommendation**: Use first set (engineering, research, finance, support, creative)
- Diverse enough for generalization
- Feasible to generate with Claude
- Don't need medical/legal (domain-specific expertise required)

---

**PROPONENT C FINAL RECOMMENDATIONS**:

#### Baselines (Updated)
1. ✅ **Sequential** (baseline)
2. ✅ **Prompted** (soft isolation)
3. ✅ **Turn-Based** (naive temporal)
4. ✅ **Semantic** (RDIC)
5. ✅ **Random Clustering** (negative control) - ADD
6. ✅ **True Multi-Agent (Sequential)** (gold standard) - FIX Week 6

Total: **6 conditions**

#### Ablations (Week 7)
1. ✅ Cluster count: 2, 3, 5, 7
2. ✅ Routing threshold: 0.5, 0.7, 0.9
3. ✅ No-coordinator ablation - ADD
4. ⚪ Cache size limits: Optional

#### Multi-Model Validation (Week 8)
1. ✅ Gemma 3 12B (primary)
2. ✅ Llama 3.1 8B (different architecture)
3. ⚪ Gemma 2 9B: Optional
4. ⚪ Qwen 2.5 14B: Optional

**Reduce from 4 models to 2 models**

#### Statistical Approach
1. ✅ Use FDR correction (not Bonferroni)
2. ✅ Focus narrative on primary comparison (semantic vs sequential)
3. ✅ Report all pairwise comparisons in appendix
4. ✅ n=50 is sufficient

#### Domain Coverage
1. ✅ Software Engineering (10)
2. ✅ Scientific Research (10)
3. ✅ Financial Analysis (10) - rename from "Business"
4. ✅ Technical Support (10)
5. ✅ Creative Writing (10)

Total: **50 examples, stratified**

---

## Part 5: Technical Feasibility and Platform Review

### PROPONENT A (Technical):
**PLATFORM AND INSTRUMENTATION DEFENSE**

#### MLX + Gemma 3 12B Platform

**Hardware**: Mac with 24GB RAM

**Software**:
- MLX framework (Apple Silicon optimized)
- Gemma 3 12B (4-bit quantization)
- Expected footprint: 7-10GB

**Feasibility Assessment**: ✅ FEASIBLE

**Tests Done**:
- Gemma 3 12B (4-bit) runs on M1/M2/M3 Macs with 16GB+ RAM
- MLX provides efficient quantization and inference
- KV cache overhead: ~500MB per cluster (manageable)

**Concerns Addressed**:

**1. Memory Budget**:
- Model: 7-10GB
- KV cache (3 clusters): 500MB × 3 = 1.5GB
- OS overhead: 2GB
- Total: ~11-13GB (fits in 24GB)
- **Verdict**: ✅ Feasible

**2. True Multi-Agent Issue** (Week 6):
- Skeptic A noted 3×12B = 30GB (OOM crash)
- Skeptic C proposed sequential execution (3 separate calls)
- **Verdict**: ✅ Feasible with sequential approach

**3. Latency**:
- Gemma 3 12B (4-bit): ~20-30 tokens/sec
- Average turn: 200 tokens = ~7-10 seconds
- 15 turns per example = ~150 seconds/example
- 50 examples × 4 conditions = 200 runs × 150s = 30,000s = **8-9 hours**
- **Verdict**: ✅ Feasible, runs overnight

#### Instrumentation (Week 3)

**Planned Metrics**:

**Latency**:
- Total generation time per condition
- Per-turn generation latency
- Agent switch overhead
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

**Implementation**:
- MLX provides built-in profiling (`mx.metal.get_peak_memory()`)
- Python `time.perf_counter()` for latency
- Custom logging for cache sizes

**Feasibility**: ✅ Straightforward

**Overhead Target**: <5%
- Instrumentation should not significantly impact performance
- MLX profiling is low-overhead

#### Automated Metrics (Week 2-3)

**Planned Metrics**:
- ROUGE (lexical overlap)
- BERTScore (semantic similarity)
- Semantic similarity (embeddings)

**Issues** (raised by Skeptic A):
- ROUGE: What's the reference?
- BERTScore: What's the reference?

**Clarification**:

**Intra-Cluster Coherence** (Specialization):
- For each cluster, compute pairwise BERTScore among turns
- Average score → coherence metric
- High score → agent is staying on-topic

**Inter-Cluster Separation** (Contamination):
- Compute pairwise BERTScore between clusters
- Low score → good separation
- High score → contamination (agents are overlapping)

**Synthesis Quality**:
- Compare coordinator output to specialist outputs
- Measure information overlap (ROUGE)
- High ROUGE → good integration of specialist knowledge

**Implementation**:
- Use `sentence-transformers` for embeddings
- Use `evaluate` library for ROUGE/BERTScore
- Custom scripts for pairwise comparisons

**Feasibility**: ✅ Standard libraries available

---

**PROPONENT A SUMMARY**:

#### Platform: ✅ FEASIBLE
- MLX + Gemma 3 12B (4-bit) fits in 24GB RAM
- Sequential true multi-agent also fits
- Latency is acceptable (8-9 hours for full evaluation)

#### Instrumentation: ✅ ADEQUATE
- MLX provides built-in profiling
- Custom logging for cache sizes
- <5% overhead achievable

#### Automated Metrics: ✅ COMPREHENSIVE
- Intra-cluster coherence (BERTScore pairwise)
- Inter-cluster separation (embedding similarity)
- Synthesis quality (ROUGE overlap)
- Standard libraries available

#### Remaining Concerns:
1. Rater recruitment timeline (Week 2)
2. True multi-agent implementation (Week 6, sequential approach)
3. Router agent (optional, defer if needed)

---

## Part 6: Synthesis and Recommendations

### ALL PARTICIPANTS: CONSENSUS RECOMMENDATIONS FOR v2

After thorough debate, here are our **consensus recommendations** for updated_plan.v2.md:

---

### A. Timeline Revisions

**Current**: 12 weeks (optimistic)

**Recommended**: **14-15 weeks** (realistic)

| Phase | Current | Recommended | Rationale |
|-------|---------|-------------|-----------|
| **Phase 0: Pilot** | None | Week 0 (3 days) | Catch issues early |
| **Phase 1: Evaluation** | Weeks 1-5 | Weeks 1-6 | Add rater buffer, error analysis |
| **Phase 2: Benchmarking** | Weeks 6-9 | Weeks 7-9 | Compress (fix Week 6, drop Week 9) |
| **Phase 3: Deployment** | Weeks 10-12 | Weeks 10-15 | Extend writing, add revision buffer |
| **Total** | 12 weeks | 15 weeks | +3 weeks buffer |

**Key Changes**:
- ✅ **Add Week 0**: Pilot with n=5 (catch issues early)
- ✅ **Extend Phase 1**: +1 week for rater recruitment and error analysis
- ✅ **Compress Phase 2**: -1 week by fixing Week 6 (sequential) and reducing multi-model scope
- ✅ **Extend Phase 3**: +3 weeks for thorough writing and revision

**Venue Alignment**:
- ICML 2026 (Jan 31): **NOT FEASIBLE** (8 days from now)
- ACL 2026 (Feb 15): **NOT FEASIBLE** (3 weeks from now)
- NeurIPS 2026 (May 15): **FEASIBLE** (16 weeks from now)
- EMNLP 2026 (June 1): **FEASIBLE** (18 weeks from now)

**Target**: NeurIPS 2026 (15 weeks, 1 week buffer)

---

### B. Evaluation Protocol Refinements

#### Blind Evaluation
1. ✅ Recruit **4 raters** (backup if 1 drops out), use best 3
2. ✅ Start recruitment in **Week 1** (not Week 2)
3. ✅ Extend training to **4 hours over 2 sessions** (calibration)
4. ✅ Target **κ>0.7** (minimum), κ>0.8 (ideal)
5. ✅ Add **10 gold examples** for quality control
6. ✅ Reduce rater burden: **n=30 per rater** (90 overlap for reliability)
7. ✅ Build **web interface** for rating (side-by-side view)
8. ✅ Extend evaluation to **1 week** (not 1 weekend, avoid fatigue)

#### Rubric
1. ✅ Keep 0-5 scale (define all 6 levels, not just 3 anchors)
2. ✅ Add **Task Success metric** (0-5, "Did it solve the problem?")
3. ✅ Clarify **Cross-Contamination** measurement (qualitative + quantitative)

#### Automated Metrics
1. ✅ **Intra-cluster coherence**: Pairwise BERTScore within cluster (specialization)
2. ✅ **Inter-cluster separation**: Pairwise embedding similarity between clusters (contamination)
3. ✅ **Synthesis quality**: ROUGE overlap between coordinator and specialists
4. ✅ **Redundancy metric**: Cross-cluster content overlap
5. ✅ Target **r>0.6** correlation with human scores (metric validation)

#### Statistical Analysis
1. ✅ Use **FDR correction** (Benjamini-Hochberg, not Bonferroni)
2. ✅ Focus narrative on **primary comparison** (semantic vs sequential)
3. ✅ Report all pairwise comparisons in appendix
4. ✅ Add **error analysis** (qualitative, Week 6)

---

### C. Baseline and Comparison Revisions

#### Core Conditions (6 Total)
1. ✅ **Sequential** (baseline, no isolation)
2. ✅ **Prompted** (soft isolation, instructional)
3. ✅ **Turn-Based** (naive temporal isolation)
4. ✅ **Semantic** (RDIC, KV cache partitioning)
5. ✅ **Random Clustering** (negative control) - **ADD**
6. ✅ **True Multi-Agent (Sequential)** (gold standard) - **FIX Week 6**

**Critical Fix**: Week 6 (True Multi-Agent)
- **Problem**: 3×12B = 30GB (OOM on 24GB RAM)
- **Solution**: Sequential execution (3 separate calls, 10GB peak each)
- **Measurement**: Aggregate quality, compare memory efficiency

#### Ablation Studies (Week 7)
1. ✅ Cluster count: 2, 3, 5, 7
2. ✅ Routing threshold: 0.5, 0.7, 0.9
3. ✅ **No-coordinator ablation** - **ADD** (test coordinator necessity)
4. ⚪ Cache size limits: Optional (defer if time-constrained)

#### Multi-Model Validation (Week 8)
1. ✅ Gemma 3 12B (primary)
2. ✅ Llama 3.1 8B (different architecture)
3. ⚪ Gemma 2 9B: Optional (reduce scope)
4. ⚪ Qwen 2.5 14B: Optional (reduce scope)

**Reduce from 4 models to 2 models** (save 1 week)

---

### D. Router Agent Decision: DEFER TO FUTURE WORK

**Prior Art Found**:
- MasRouter (ACL 2025): Routing framework for multi-agent systems
- AgentRouter (arXiv 2025): KG-guided router for multi-agent QA
- Orchestrator Pattern: Industry standard (LangChain, LangGraph, OpenAI)
- SC-MAS: Explicit LLM Router module

**Consensus**:
- ❌ **Not novel** as standalone contribution
- ⚠️ **Distinction exists**: Prior art routes between agents, we route within KV cache
- ✅ **Overhead is acceptable**: 1-2GB memory, 50ms latency
- ❌ **Complexity not justified** for first paper

**Decision**: **DEFER to future work**

**Rationale**:
1. Embedding-based semantic routing is sufficient for core novelty
2. Router agent adds scope creep (12 weeks → 14+ weeks)
3. Can be published as follow-up (dynamic adaptation, reasoning-based routing)
4. Keep focus on **semantic KV cache partitioning** (core contribution)

**If Router Agent is Included** (against our recommendation):
- Must cite MasRouter, AgentRouter, SC-MAS
- Must clarify distinction (cache-level vs agent-level routing)
- Must demonstrate benefit over embedding routing
- Must measure overhead empirically
- Add 1-2 weeks to timeline

---

### E. Dataset Revisions

#### Size: n=50 (Confirmed)
- Statistical power with FDR: β=0.70 (acceptable)
- Cost-benefit: n=100 adds 2 weeks for β=0.95 (not worth it)
- **Decision**: n=50

#### Domains (Refined)
1. **Software Engineering** (debugging, code review, documentation) - 10 examples
2. **Scientific Research** (literature review, experiment design, paper writing) - 10 examples
3. **Financial Analysis** (market research, modeling, reporting) - 10 examples
4. **Technical Support** (troubleshooting, diagnosis, resolution) - 10 examples
5. **Creative Writing** (storytelling, editing, critique) - 10 examples

**Change**: "Business" → "Financial Analysis" (more specific)

#### Stratification
- 10 examples per domain (balanced)
- Train/val/test: 30/10/10 split

---

### F. Platform and Instrumentation (Confirmed)

#### Platform: ✅ FEASIBLE
- MLX + Gemma 3 12B (4-bit)
- 24GB RAM (Mac M1/M2/M3)
- Sequential true multi-agent fits

#### Instrumentation: ✅ ADEQUATE
- MLX built-in profiling
- Custom cache logging
- <5% overhead

#### Automated Metrics: ✅ COMPREHENSIVE
- Intra-cluster coherence (BERTScore)
- Inter-cluster separation (embeddings)
- Synthesis quality (ROUGE)
- Redundancy (overlap)

---

### G. Risk Mitigation Updates

#### Risk 1: Rater Recruitment Fails
- **Mitigation**: Start in Week 1, recruit 4 (use best 3)
- **Fallback**: 2 raters + author (disclosed)

#### Risk 2: Week 6 Memory Issue
- **Mitigation**: Sequential execution (not parallel)
- **Validated**: Fits in 24GB RAM

#### Risk 3: No Statistical Significance
- **Mitigation**: Use FDR correction, focus on effect sizes
- **Fallback**: Report d>0.5 as "moderate-to-large effect"

#### Risk 4: 3X Memory Claim Doesn't Hold
- **Mitigation**: Measure accurately with instrumentation
- **Expected**: 2-3X efficiency (still valuable)

#### Risk 5: Deployment Study Fails
- **Mitigation**: Not critical, can defer
- **Fallback**: Phase 1-2 alone sufficient

---

## Part 7: Updated Timeline for v2

### Phase 0: Pilot Testing (Week 0, 3 days)

**NEW**: Catch issues early before full dataset generation

| Day | Task | Time | Details |
|-----|------|------|---------|
| Mon | Generate 5 pilot examples | 3h | 1 per domain |
| Mon | Run all 4 conditions on pilots | 2h | Test pipeline end-to-end |
| Tue | Manual evaluation of pilots | 2h | Check quality, identify issues |
| Tue | Refine prompts/conditions | 2h | Fix any problems |
| Wed | Document pilot findings | 1h | Lessons learned |

**Deliverables**:
- `data/pilot_examples.json` (5 examples)
- `data/pilot_findings.md` (issues found)

**Success Criteria**:
- [ ] All conditions run without errors
- [ ] Output quality is reasonable (manual check)
- [ ] No major pipeline issues

---

### Phase 1: Rigorous Evaluation (Weeks 1-6)

#### Week 1: Dataset Generation + Rater Recruitment

**CHANGE**: Add rater recruitment in parallel

| Day | Task | Time | Details |
|-----|------|------|---------|
| Mon | **BEGIN RATER RECRUITMENT** | 1h | Post to research groups, email contacts |
| Mon | Design domain taxonomy | 2h | 5 refined domains |
| Mon | Create generation prompts | 2h | Claude Sonnet prompts |
| Tue | Generate Software Engineering (10) | 3h | Debugging, code review, docs |
| Tue | Generate Scientific Research (10) | 3h | Literature review, experiments |
| Wed | Generate Financial Analysis (10) | 3h | Market research, modeling |
| Wed | Generate Technical Support (10) | 3h | Troubleshooting, diagnosis |
| Thu | Generate Creative Writing (10) | 3h | Storytelling, editing |
| Thu | Validation and refinement | 3h | Manual review, regenerate if needed |
| Fri | **FOLLOW UP WITH RATERS** | 1h | Confirm commitments |
| Fri | Create train/val/test splits | 1h | 30/10/10 stratified |
| Fri | Document dataset statistics | 2h | Domain distribution, analysis |

**Deliverables**:
- `data/virtual_agents_dataset_v1.json` (50 examples)
- `data/dataset_statistics.md`
- **4 raters recruited** (use best 3)

---

#### Week 2: Evaluation Framework (Extended)

**CHANGE**: Extended rater training, web interface, gold examples

| Day | Task | Time | Details |
|-----|------|------|---------|
| Mon | Design evaluation rubric (6 levels) | 3h | Define 0, 1, 2, 3, 4, 5 for each dimension |
| Mon | Add Task Success metric | 1h | 4th dimension for rubric |
| Mon | Implement blinding system | 2h | Anonymize, randomize |
| Tue | **Rater training session 1** | 2h | Explain rubric, practice on 5 examples |
| Tue | **Create 10 gold examples** | 2h | Known scores for quality control |
| Wed | **Rater training session 2** | 2h | Calibration, feedback, discuss edge cases |
| Wed | **Build web rating interface** | 3h | Side-by-side view, easy input |
| Thu | Implement automated metrics | 4h | BERTScore (coherence), embeddings (separation), ROUGE (synthesis) |
| Thu | Add redundancy metric | 1h | Inter-cluster overlap |
| Fri | Create evaluation pipeline | 3h | Batch processing, inter-rater reliability |
| Fri | Pilot evaluation on 10 examples | 2h | Test with 3 raters, measure κ |

**Deliverables**:
- `evaluation/rubric_v2.md` (all 6 levels defined)
- `evaluation/gold_examples.json` (10 examples with known scores)
- `evaluation/rating_interface/` (web app)
- `evaluation/automated_metrics.py` (all metrics)
- `evaluation/pipeline.py`

**Success Criteria**:
- [ ] 3-4 trained raters
- [ ] Cohen's kappa >0.6 on pilot (target >0.7)
- [ ] Web interface works smoothly
- [ ] Automated metrics implemented

---

#### Week 3: Instrumentation (No Change)

[Keep existing Week 3 plan]

---

#### Week 4: Run Rigorous Evaluation (Extended)

**CHANGE**: Add random clustering condition, extend evaluation window

| Day | Task | Time | Details |
|-----|------|------|---------|
| Mon | Run sequential condition (baseline) | 4h | 50 examples |
| Tue | Run prompted condition | 4h | 50 examples |
| Tue | **Run random clustering condition** | 2h | 50 examples (NEW) |
| Wed | Run turn-based condition | 4h | 50 examples |
| Thu | Run semantic condition (RDIC) | 4h | 50 examples |
| Fri | Export anonymized outputs for raters | 2h | Blind, randomized, web interface ready |
| Fri | Distribute to raters | 1h | Email with instructions |

**Week 4-5 (1 week)**: Raters evaluate (30 examples × 5 conditions = 150 ratings per rater, spread over 1 week)

**Deliverables**:
- `results/exp_full_evaluation/` (5 conditions × 50 examples = 250 runs)
- `results/exp_full_evaluation/blinded_outputs/` (for raters)

**Success Criteria**:
- [ ] All 250 runs complete (5 conditions, not 4)
- [ ] No crashes
- [ ] Raters receive blinded outputs

---

#### Week 5: Statistical Analysis (No Major Change)

[Keep existing Week 5 plan, update for 5 conditions instead of 4]

**CHANGE**: Use FDR correction instead of Bonferroni

---

#### Week 6: Error Analysis and Results Writing (NEW)

**ADDED**: Qualitative analysis of failures

| Day | Task | Time | Details |
|-----|------|------|---------|
| Mon | Identify failure cases | 3h | When does semantic condition fail? |
| Mon | Categorize errors | 2h | Taxonomy of failure modes |
| Tue | Analyze successful cases | 3h | When does it work best? |
| Tue | Compare failure patterns across conditions | 2h | Do all conditions fail on same examples? |
| Wed | Qualitative analysis write-up | 3h | Section 5.3: Error Analysis |
| Wed | Case studies | 2h | 3-5 detailed examples (success + failure) |
| Thu | Results section refinement | 4h | Integrate all analyses |
| Fri | Generate final tables and figures | 4h | Publication-ready visualizations |

**Deliverables**:
- `results/exp_full_evaluation/error_analysis.md`
- `results/exp_full_evaluation/case_studies/` (3-5 detailed examples)
- `paper/sections/05_results.md` (complete results section)

**Success Criteria**:
- [ ] Clear taxonomy of failure modes
- [ ] 3-5 case studies documented
- [ ] Results section draft complete

---

### Phase 2: Comparative Benchmarking (Weeks 7-9, Compressed)

#### Week 7: True Multi-Agent Baseline (FIXED)

**CHANGE**: Sequential execution instead of parallel

| Day | Task | Time | Details |
|-----|------|------|---------|
| Mon | Design true multi-agent (sequential) | 2h | 3 sequential Gemma 3 12B calls |
| Mon | Implement agent communication | 2h | Pass outputs between calls |
| Tue | Test on 5 examples | 3h | Verify works end-to-end |
| Tue | Measure memory footprint | 2h | Peak RAM per call (expect 10GB) |
| Wed | Run on 10 test examples | 4h | Full pipeline |
| Thu | Evaluate quality (blind) | 3h | Same rubric, subset of raters |
| Thu | Compare vs semantic condition | 2h | Quality parity? |
| Fri | Memory efficiency analysis | 2h | 2-3X claim validation |
| Fri | Write comparison section | 2h | Section 6.1: True Multi-Agent Comparison |

**Deliverables**:
- `src/true_multiagent_sequential.py`
- `results/exp_multiagent_baseline/` (10 examples)
- `results/exp_multiagent_baseline/memory_comparison.md`
- `paper/sections/06_discussion.md` (comparison)

**Success Criteria**:
- [ ] Sequential execution fits in 24GB RAM
- [ ] Memory measurements accurate
- [ ] Quality comparable to semantic (parity expected)
- [ ] 2-3X memory efficiency validated

---

#### Week 8: Ablation Studies (Extended)

**CHANGE**: Add no-coordinator ablation

| Day | Task | Time | Details |
|-----|------|------|---------|
| Mon | Implement cluster count variation | 2h | 2, 5, 7 clusters (3 is default) |
| Mon | Run 2-cluster on 10 examples | 2h | Specialist + coordinator only |
| Tue | Run 5-cluster on 10 examples | 2h | More fine-grained |
| Tue | Run 7-cluster on 10 examples | 2h | Very fine-grained |
| Tue | Analyze cluster count impact | 2h | Quality vs efficiency trade-off |
| Wed | **No-coordinator ablation** | 2h | 2 specialists only, no synthesis (NEW) |
| Wed | **Run no-coordinator on 10 examples** | 2h | Test coordinator necessity (NEW) |
| Wed | Routing threshold experiments | 2h | 0.5, 0.7, 0.9 similarity cutoffs |
| Thu | Fixed clustering baseline | 2h | By turn position (alternative) |
| Thu | Run fixed clustering on 10 examples | 2h | Temporal vs semantic |
| Fri | Statistical comparison (all ablations) | 3h | Which config is optimal? |
| Fri | Write ablation section | 2h | Section 5.4: Ablations |

**Deliverables**:
- `results/exp_ablations/` (all ablation results)
- `results/exp_ablations/cluster_count_analysis.md`
- `results/exp_ablations/coordinator_necessity.md` (NEW)
- `results/exp_ablations/threshold_sensitivity.md`
- `paper/sections/05_results.md` (ablation subsection)

**Success Criteria**:
- [ ] Tested 2, 3, 5, 7 clusters
- [ ] No-coordinator shows coordinator adds value
- [ ] Optimal configuration identified (likely 3 clusters, threshold 0.7)
- [ ] Robustness demonstrated

---

#### Week 9: Multi-Model Validation (REDUCED SCOPE)

**CHANGE**: 2 models instead of 4

| Day | Task | Time | Details |
|-----|------|------|---------|
| Mon | Set up Llama 3.1 8B | 2h | Different architecture from Gemma |
| Mon | Run semantic on 10 examples (Llama) | 3h | Cross-model validation |
| Tue | Compare Gemma vs Llama | 3h | Which benefits more? |
| Tue | Statistical comparison | 2h | Is benefit architecture-agnostic? |
| Wed | Write model comparison | 3h | Section 5.5: Generalization |
| Wed | **(Optional) Set up Gemma 2 9B** | 2h | If time allows |
| Thu | **(Optional) Run Gemma 2 9B** | 3h | Smaller model |
| Thu | **(Optional) Compare 3 models** | 2h | Size vs benefit |
| Fri | Finalize multi-model section | 2h | Section 5.5 complete |

**Deliverables**:
- `results/exp_multimodel/` (Gemma 3 12B + Llama 3.1 8B, optional: Gemma 2 9B)
- `results/exp_multimodel/comparison.md`
- `paper/sections/05_results.md` (generalization subsection)

**Success Criteria**:
- [ ] Works on at least 2 different models (Gemma, Llama)
- [ ] Consistent benefit across architectures
- [ ] Generalization claim supported

---

### Phase 3: Real-World Validation and Writing (Weeks 10-15, Extended)

#### Week 10: Deployment Case Study (No Change)

[Keep existing Week 10 plan]

**Note**: This is **nice-to-have**, can be deferred if timeline is tight.

---

#### Week 11-12: Paper Writing (EXTENDED)

**CHANGE**: 2 weeks instead of 1 week

**Week 11: Drafting**

| Day | Task | Time | Details |
|-----|------|------|---------|
| Mon | Write Abstract (200 words) | 2h | Summary of contributions |
| Mon | Write Introduction (1.5 pages) | 4h | Problem, gap, contribution |
| Tue | Write Related Work (1.5 pages) | 4h | FlowKV, EpiCache, MasRouter, multi-agent |
| Tue | Review Methods section | 2h | Ensure complete from Week 1-9 |
| Wed | Integrate all results (Sections 5) | 4h | Phase 1-2 results |
| Wed | Write Discussion (1 page) | 2h | Implications, memory efficiency |
| Thu | Write Limitations (0.5 pages) | 2h | Honest assessment |
| Thu | Write Conclusion (0.5 pages) | 1h | Summary |
| Fri | Full draft assembly | 3h | Combine all sections |
| Fri | Initial read-through | 2h | Check coherence |

**Week 12: Polishing**

| Day | Task | Time | Details |
|-----|------|------|---------|
| Mon | Polish Introduction + Abstract | 3h | Clarity, flow |
| Mon | Polish Related Work | 2h | Ensure all prior art cited |
| Tue | Polish Results | 4h | Integrate all figures/tables |
| Tue | Polish Discussion | 2h | Strengthen arguments |
| Wed | Polish all figures | 4h | 300 DPI, consistent style, clear labels |
| Thu | Polish all tables | 3h | LaTeX formatting, readability |
| Thu | References check | 2h | All citations correct |
| Fri | Full read-through | 4h | End-to-end coherence |
| Fri | Feedback from peers | 2h | Get 2-3 people to read |

**Deliverables**:
- `paper/rdic_paper_v1.pdf` (complete draft, 8-10 pages)
- `paper/sections/*.md` (all sections)
- `paper/figures/` (publication-quality, 300 DPI)
- `paper/tables/` (LaTeX formatted)

**Success Criteria**:
- [ ] Complete draft exists (8-10 pages)
- [ ] All figures at 300 DPI
- [ ] All results integrated
- [ ] Peer feedback incorporated

---

#### Week 13-14: Revision and Submission Prep (EXTENDED)

**CHANGE**: 2 weeks instead of 1 week

**Week 13: Revision**

| Day | Task | Time | Details |
|-----|------|------|---------|
| Mon | Address peer feedback | 4h | Revise based on comments |
| Mon | Check all numbers | 2h | Verify statistics |
| Tue | Rewrite weak sections | 4h | Strengthen arguments |
| Tue | Grammar and spell check | 2h | Proofread |
| Wed | Verify all references | 2h | Citations complete and correct |
| Wed | Check reproducibility | 3h | Can someone else reproduce? |
| Thu | Prepare code repository | 4h | Clean, document, README |
| Thu | Test repository | 2h | Clone fresh, run examples |
| Fri | Prepare Arxiv package | 3h | LaTeX source, figures, etc. |
| Fri | Test Arxiv compilation | 1h | No errors |

**Week 14: Submission**

| Day | Task | Time | Details |
|-----|------|------|---------|
| Mon | Final proofreading | 4h | Last check |
| Mon | Final figure/table check | 2h | All embedded correctly |
| Tue | Submit to Arxiv | 2h | Upload, wait for approval |
| Tue | Prepare venue submission | 3h | NeurIPS 2026 format |
| Wed | Venue submission (NeurIPS) | 2h | Complete submission form |
| Wed | Double-check submission | 1h | Verify uploaded correctly |
| Thu | Prepare supplementary materials | 3h | Appendix, extra figures |
| Thu | Submit supplementary | 1h | Upload |
| Fri | **SUBMISSION COMPLETE** | - | 🎉 |
| Fri | Post-submission tasks | 2h | Tweet, blog post, etc. |

**Deliverables**:
- `paper/rdic_paper_final.pdf`
- Arxiv submission (public)
- NeurIPS 2026 submission (complete)
- GitHub repository (public, documented)

**Success Criteria**:
- [ ] Arxiv submission complete
- [ ] Venue submission complete (NeurIPS 2026)
- [ ] All materials publicly available
- [ ] Code repository documented

---

#### Week 15: Buffer Week (NEW)

**ADDED**: 1 week buffer for unexpected issues

- Contingency for delays
- Additional revisions if needed
- Response to venue questions
- No specific tasks planned

---

## Part 8: Final Recommendations Summary

### CRITICAL CHANGES FOR v2

#### 1. Timeline
- **FROM**: 12 weeks (optimistic)
- **TO**: 15 weeks (realistic, +3 weeks)
- **Target**: NeurIPS 2026 (May 15 deadline)

#### 2. Evaluation Protocol
- ✅ Recruit **4 raters** (not 3), start in Week 1
- ✅ Extend training to **4 hours over 2 sessions**
- ✅ Add **10 gold examples** for quality control
- ✅ Build **web rating interface**
- ✅ Use **FDR correction** (not Bonferroni)
- ✅ Add **error analysis** (Week 6)

#### 3. Baselines
- ✅ Add **Random Clustering** baseline (Week 4)
- ✅ Fix **True Multi-Agent** (sequential execution, Week 7)
- ✅ Add **No-Coordinator ablation** (Week 8)
- **Total**: 6 conditions (was 4)

#### 4. Router Agent
- ❌ **DEFER to future work**
- **Rationale**: Not novel (prior art exists), adds complexity
- **Alternative**: Mention in discussion as future extension

#### 5. Multi-Model Validation
- **Reduce scope**: 2 models (Gemma, Llama), not 4
- **Save**: 1 week

#### 6. Dataset
- ✅ Confirm **n=50** (sufficient with FDR)
- ✅ Rename "Business" → **"Financial Analysis"**
- ✅ Stratified: 10 examples per domain

#### 7. Paper Writing
- **Extend**: 2 weeks (Week 11-12) + 2 weeks revision (Week 13-14)
- **Total**: 4 weeks for writing/revision (was 2)

#### 8. Pilot Testing
- ✅ **Add Week 0** (3 days pilot with n=5)

---

### PHASE-BY-PHASE SUMMARY (15 Weeks Total)

| Phase | Weeks | Key Deliverables |
|-------|-------|------------------|
| **Phase 0: Pilot** | Week 0 (3 days) | 5 pilot examples, pipeline tested |
| **Phase 1: Evaluation** | Weeks 1-6 | 50 examples, 5 conditions, statistics, error analysis |
| **Phase 2: Benchmarking** | Weeks 7-9 | True multi-agent, ablations, multi-model (2 models) |
| **Phase 3: Deployment** | Week 10 | Real-world case study (optional) |
| **Phase 4: Writing** | Weeks 11-14 | Complete paper, revision, submission |
| **Phase 5: Buffer** | Week 15 | Contingency |

**Total**: **15 weeks** (vs 12 in v1)

---

### WHAT TO DROP IF TIME IS TIGHT

If timeline pressure increases, consider dropping (in order):

1. ⚪ **Week 10 (Deployment)**: Nice-to-have, not critical
2. ⚪ **Week 9 (Multi-Model)**: 1 model (Gemma) is sufficient for first paper
3. ⚪ **No-Coordinator ablation**: Interesting but not essential
4. ⚪ **Fixed clustering baseline**: Similar to turn-based
5. ⚪ **Gold examples**: Reduces quality control but saves time

**Minimum viable timeline**: **13 weeks** (drop Week 10, reduce Week 9)

---

### WHAT MUST BE KEPT (NON-NEGOTIABLE)

1. ✅ **n=50 examples across 5 domains** (core dataset)
2. ✅ **Blind evaluation with 3+ raters** (rigor)
3. ✅ **4 core conditions** (sequential, prompted, turn-based, semantic)
4. ✅ **Random clustering baseline** (negative control)
5. ✅ **True multi-agent comparison** (validates memory efficiency claim)
6. ✅ **Statistical tests + effect sizes** (rigor)
7. ✅ **Error analysis** (shows limitations)
8. ✅ **4 weeks for writing/revision** (quality)

---

### SUCCESS METRICS FOR v2 PLAN

A successful plan should achieve:

1. ✅ **Realistic timeline** that accounts for rater recruitment, technical issues, writing
2. ✅ **Rigorous evaluation** that satisfies reviewer standards
3. ✅ **Critical baselines** that validate claims (true multi-agent, random clustering)
4. ✅ **Clear scope** that doesn't try to do too much (defer router agent)
5. ✅ **Feasible implementation** given hardware constraints (sequential true multi-agent)
6. ✅ **Publication-ready output** with 4 weeks for writing/revision

**Verdict on v1**: ❌ **Optimistic but flawed**
- Timeline too aggressive (12 weeks)
- Week 6 infeasible (memory)
- Rater recruitment timeline unrealistic
- Paper writing time insufficient
- Router agent adds scope creep

**Verdict on v2**: ✅ **Realistic and achievable**
- 15 weeks with buffer
- All technical issues addressed
- Rigorous evaluation protocol
- Clear scope (no router agent)
- Adequate writing time
- Targets NeurIPS 2026 (feasible)

---

## Conclusion: Recommendations for updated_plan.v2.md

**Overall Assessment of v1**:
- ✅ Core idea is sound (semantic KV cache partitioning)
- ✅ Addresses debate consensus items
- ⚠️ Timeline is optimistic (12 weeks → 15 weeks realistic)
- ❌ Week 6 is infeasible (memory constraints)
- ❌ Rater recruitment timeline is unrealistic
- ❌ Router agent adds scope creep
- ⚠️ Paper writing time insufficient

**Key Improvements for v2**:
1. Extend timeline to **15 weeks** (add pilot, rater buffer, writing time)
2. Fix Week 6: **Sequential true multi-agent** (not parallel)
3. Add **random clustering baseline** (negative control)
4. Add **no-coordinator ablation** (test necessity)
5. **Defer router agent** to future work (not novel, adds complexity)
6. Reduce multi-model scope to **2 models** (Gemma + Llama)
7. Extend paper writing to **4 weeks** (drafting + revision)
8. Use **FDR correction** (not Bonferroni)
9. Add **error analysis** (Week 6)
10. Start **rater recruitment in Week 1** (not Week 2)

**Target Venue**: **NeurIPS 2026** (May 15 deadline, 16 weeks from now)

**Confidence**: With these changes, the plan is **feasible and rigorous**.

---

**Next Steps**:
1. Author reviews this debate
2. Author creates updated_plan.v2.md incorporating recommendations
3. Begin Week 0 (pilot testing)

---

## Sources

Router agent and multi-agent LLM system prior art:

**Router Agent Systems:**
- [MasRouter: Learning to Route LLMs for Multi-Agent System](https://aclanthology.org/2025.acl-long.757.pdf) (ACL 2025)
- [AgentRouter: A Knowledge-Graph-Guided LLM Router for Collaborative Multi-Agent QA](https://arxiv.org/abs/2510.05445)
- [Toward Super Agent System with Hybrid AI Routers](https://arxiv.org/html/2504.10519v1)

**Orchestrator Pattern:**
- [Developer's guide to multi-agent patterns in ADK](https://developers.googleblog.com/developers-guide-to-multi-agent-patterns-in-adk/)
- [The Orchestrator Pattern: Routing Conversations to Specialized AI Agents](https://dev.to/akshaygupta1996/the-orchestrator-pattern-routing-conversations-to-specialized-ai-agents-33h8)
- [Multi-agent systems - Agent Development Kit](https://google.github.io/adk-docs/agents/multi-agents/)
- [LLM Agent Orchestration: A Step by Step Guide | IBM](https://www.ibm.com/think/tutorials/llm-agent-orchestration-with-langchain-and-granite)

**Dynamic Agent Systems:**
- [Ultimate Guide to AI Agent Routing (2026)](https://botpress.com/blog/ai-agent-routing)
- [Multi-Agent and Multi-LLM Architecture: Complete Guide for 2025](https://collabnix.com/multi-agent-and-multi-llm-architecture-complete-guide-for-2025/)
- [Auto-scaling LLM-based multi-agent systems through dynamic integration of agents](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2025.1638227/full)

**Industry Frameworks:**
- [AI Agent Routing: Tutorial & Best Practices](https://www.patronus.ai/ai-agent-development/ai-agent-routing)
- [Top AI Agent Orchestration Frameworks for Developers 2025](https://www.kubiya.ai/blog/ai-agent-orchestration-frameworks)
- [Top 8 LLM Frameworks for Building AI Agents in 2026](https://www.secondtalent.com/resources/top-llm-frameworks-for-building-ai-agents/)

---

**End of Debate Round 1**
**Status**: Complete
**Next**: Author creates updated_plan.v2.md
