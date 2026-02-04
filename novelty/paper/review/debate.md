# Expert Panel Debate: COLM 2026 Paper
## "Agent Memory Below the Prompt"

**Date**: 2026-02-04
**Format**: Structured debate with 6 expert panelists
**Goal**: Rigorous evaluation from multiple perspectives

---

## Panel Composition

| Panelist | Role | Expertise | Stance |
|----------|------|-----------|--------|
| **Dr. A** | Hostile Critic | ML Systems, serving infrastructure | Skeptical (wants to reject) |
| **Dr. B** | Senior Researcher | KV cache optimization, quantization | Balanced technical evaluation |
| **Dr. C** | Edge Computing Practitioner | On-device ML, Apple Silicon | Practical relevance focus |
| **Dr. D** | Statistician | Experimental design, benchmarking | Methodology rigor |
| **Dr. E** | COLM Program Committee | Academic writing, presentation | Venue fit and clarity |
| **Dr. F** | Industry Expert (Apple/NVIDIA) | Hardware architecture, production systems | Real-world deployment |

---

## Round 1: Initial Assessments

### Dr. A (Hostile Critic): REJECT

"This paper is engineering integration masquerading as research novelty. The three claimed contributions are: (1) PagedAttention blocks saved to disk (not novel), (2) batched Q4 inference (absent from MLX is implementation detail, not research), (3) 'working memory' which is just KV cache reuse with marketing spin.

The evaluation is woefully inadequate: only 2 models tested despite claiming 4 architectures, no perplexity evaluation, no baseline comparisons with llama.cpp or LM Studio. The 81.6× 'hot cache' speedup is trivial—of course in-memory is faster than computation.

Most damaging: the '72% memory savings' is only for KV cache, which is <2% of total system memory. This is buried in the text but prominently featured as a headline result. Deceptive presentation.

**Vote**: REJECT. Resubmit as systems paper after adding quality evaluation and honest positioning."

---

### Dr. B (Senior Researcher): WEAK ACCEPT

"I disagree with Dr. A's harsh assessment. While the individual techniques exist, the *composition* is non-trivial. BatchQuantizedKVCache enabling concurrent inference over Q4 caches is genuinely missing from MLX upstream—I checked. The merge()/extract() operations bridging per-agent persistence and batched inference represent real systems work.

The evaluation concerns are valid: 2 models is limited, and the lack of perplexity data is a gap. However, the authors cite prior work (KIVI, KVQuant) showing <1% degradation for 4-bit quantization, which provides some confidence.

The 'working memory' framing is the weakest contribution—it's conceptual rather than algorithmic—but the cross-phase injection mechanism is concretely implemented and demonstrated.

**Strengths**: Clear problem (multi-agent cold-start on edge), clean system design, solid empirical speedups (10.5× warm at 16K is meaningful).

**Weaknesses**: Limited model coverage, no quality evaluation, could use more baselines.

**Vote**: WEAK ACCEPT. The systems contribution is sufficient if authors add clarity on positioning."

---

### Dr. C (Edge Practitioner): STRONG ACCEPT

"From a practitioner standpoint, this solves a real problem I encounter constantly: multi-agent LLM workflows on Mac hardware. The 40-second cold-start scenario (5 agents × 4K context) is painfully familiar. The disk reload achieving 1.95--10.5× speedup is directly useful.

The UMA-specific design is appropriate: zero-copy paths, Metal GPU integration, MLX lazy evaluation discipline. The character-level prefix matching is clever—BPE tokenization inconsistencies do occur in practice, especially with structured prompts.

Dr. A's critique about 'only 2 models' misses the point: Gemma 3 and DeepSeek-Coder-V2-Lite cover the most common edge inference targets. The model-agnostic block pool means adding Llama/Qwen support is straightforward.

**Practical impact**: This enables multi-agent coding assistants, debate systems, and collaborative analysis tools on consumer hardware. The open-source release will benefit the community.

**Vote**: STRONG ACCEPT. This is exactly the kind of practical systems work COLM should publish."

---

### Dr. D (Statistician): WEAK REJECT

"My concerns are methodological. The paper claims 'median over 3 runs' but provides no variance, confidence intervals, or statistical significance testing. With only 3 samples, outliers can skew results. Industry standard is 5-10 runs minimum.

The 2.0--4.3× E2E speedup range is cited in the abstract but only 2.2× is shown at 4K context. Where do 2.0× and 4.3× come from? Without full benchmark data, I cannot verify these claims.

The staggered arrivals experiment (Section 4.4) uses a single scenario: User A at t=0, User B at t=2s. What about different arrival patterns? What about 3+ concurrent users? The generalization from one data point is weak.

No ablation studies: How much does character-level matching contribute vs token-level? How much does the interleaved scheduler improve over naive batching? These design choices are unmotivated empirically.

**Vote**: WEAK REJECT. Need more robust statistics and ablation studies. With revisions, this could be acceptable."

---

### Dr. E (COLM PC Editor): BORDERLINE ACCEPT

"From a presentation standpoint, the paper is well-written and structured. The abstract is concise (178 words), figures are clear (though I'd like to see data points on the TTFT chart, not just lines), and the related work survey is comprehensive.

However, I share Dr. A's concern about overselling novelty. The introduction should explicitly state this is a *systems integration* paper, not claiming algorithmic breakthroughs. The current framing invites rejection from reviewers who expect novelty.

The '72% memory savings' claim needs prominent caveats. It's technically true but misleading without context. A reader might think the total system uses 72% less memory, when it's actually ~1% total reduction.

**Writing issues fixed**:
- ✓ No em dashes (checked)
- ✓ No banned AI slop vocabulary
- ✓ Numerical claims traced to sources

**Remaining issues**:
- Working memory contribution is underdeveloped (qualitative only)
- Limitations section is honest but exposes significant gaps (no perplexity, only 2 models)

**Vote**: BORDERLINE ACCEPT. If authors clarify positioning (systems paper, not algorithmic) and add perplexity evaluation, I'd upgrade to accept."

---

### Dr. F (Industry Expert): STRONG ACCEPT

"As someone who ships production LLM systems, I value practical contributions over theoretical novelty. This paper delivers:

1. **Real hardware convergence**: M4 Pro at 273 GB/s matching DGX Spark is significant. Consumer hardware reaching datacenter bandwidth is a tipping point. The authors correctly note compute still favors datacenter, but memory bandwidth parity opens new deployment models.

2. **Quantification of cold-start penalty**: The 8s vs 400ms comparison (M4 Pro vs A100 for 4K prefill) quantifies the datacenter-edge gap. This justifies the entire system.

3. **Production-ready implementation**: Open-source MLX code, safetensors persistence format, model-agnostic design. This isn't a research prototype—it's deployable.

Dr. D's statistics concerns are valid but not disqualifying. In industry, we often ship on less rigorous validation. The 3-run median is acceptable for systems work. The trend is clear: warm cache is consistently faster across all context lengths.

**Competitive analysis**: llama.cpp has `--prompt-cache` but it's FP16 and token-based. Ollama doesn't persist across restarts. LM Studio 0.4.0 has parallel inference but no explicit cache management API. This paper's contribution is the *combination*: Q4 + disk + per-agent + batched.

**Vote**: STRONG ACCEPT. This will have practical impact and advance edge inference tooling."

---

## Round 2: Cross-Examination

### Dr. A challenges Dr. C:

**Dr. A**: "You claim this 'solves a real problem,' but where's the user study? How do you know 26% per-agent latency reduction in batched mode is acceptable? Real users might reject this tradeoff."

**Dr. C**: "Fair point on user study, but the staggered arrivals result shows the tradeoff is favorable: User B gets 2.6× faster service for User A's 4% penalty. The net system benefit (1.86× total TTFT) speaks for itself. Besides, users can choose sequential mode if they prefer—the system supports both."

**Dr. A**: "That doesn't address my core concern: this is engineering, not research. COLM is a research conference, not an MLSys or OSDI."

**Dr. B**: "COLM explicitly includes systems work in its scope. PagedAttention (SOSP'23) is widely cited in LLM research. Practical systems that enable new capabilities are valuable contributions."

---

### Dr. D challenges Dr. F:

**Dr. D**: "You say '3-run median is acceptable,' but that's industry practice, not academic standards. For a research venue, we need reproducibility and statistical rigor. What if the reported numbers are outliers?"

**Dr. F**: "The consistency across context lengths argues against outliers. Look at Table 1: the warm TTFT progression (901ms → 1,192ms → 1,680ms → 3,307ms → 6,544ms) is smooth. Outliers would show jumps or reversals. The trends are clear even with 3 samples."

**Dr. D**: "Trends yes, but confidence intervals would strengthen the claim. A footnote stating 'representative results' or adding error bars would address my concern without major additional work."

**Dr. B**: "Agreed. The authors should add standard deviation or min/max ranges in a revision. This is fixable."

---

### Dr. E challenges Dr. B:

**Dr. E**: "You call BatchQuantizedKVCache 'non-trivial,' but Dr. A argues it's just batching + quantization, both well-known. What's genuinely novel here?"

**Dr. B**: "The challenge is that MLX's quantized attention (`quantized_scaled_dot_product_attention`) doesn't support batched inputs with variable-length sequences. The paper's merge() operation handles left-padding and stacking, and extract() reverses it. This isn't trivial plumbing—it requires understanding MLX's lazy evaluation model and ensuring quantization isn't undone during batch operations.

I checked MLX's GitHub (issue #2955, PR history): no batched quantized KV cache implementation exists. The authors had to build it. That's a systems contribution."

**Dr. A**: "But that's still MLX-specific engineering. If PyTorch or another framework already supports this, then it's not a research contribution—it's porting."

**Dr. B**: "PyTorch doesn't have native Q4 KV cache attention like MLX does. The quantization format differs (MLX uses group-wise Q4_K_M, PyTorch uses different schemes). The implementation isn't a direct port."

---

### Dr. F challenges Dr. E:

**Dr. F**: "You said 'working memory is underdeveloped.' But the prisoner's dilemma and gossip network examples demonstrate concrete multi-phase scenarios. What more do you want?"

**Dr. E**: "I want *quantitative* evaluation. Does cached context improve task accuracy vs re-prefilling? Do agents make different decisions with stale cache vs fresh context? The paper admits in Limitation #4: 'qualitative working memory evaluation only.' That's an acknowledgment of incompleteness."

**Dr. F**: "Fair, but this is a systems paper, not a cognitive science study. The working memory framing is conceptual positioning, not the main contribution. The core value is the 10.5× speedup and 72% KV cache savings, which are quantified."

**Dr. C**: "I agree with Dr. F. The working memory narrative is useful for framing, but the empirical speedups stand on their own. If reviewers want more, it can be future work."

---

## Round 3: Final Votes and Revisions

### Dr. A (Hostile Critic): REJECT → WEAK REJECT (after debate)

**Final stance**: "The debate has convinced me the BatchQuantizedKVCache implementation is non-trivial, but I still maintain this is systems engineering, not research novelty. The lack of quality evaluation (perplexity) and limited baselines are disqualifying for a top venue.

**Required revisions for acceptance**:
1. Add perplexity evaluation for Q4 vs FP16
2. Benchmark against llama.cpp --prompt-cache and LM Studio
3. Expand to 4 models as claimed (add Llama 3.1, Qwen 2.5)
4. Reposition explicitly as 'systems integration' paper in intro

**Vote**: WEAK REJECT (borderline). With all 4 revisions, I'd upgrade to WEAK ACCEPT."

---

### Dr. B (Senior Researcher): WEAK ACCEPT → ACCEPT (after debate)

**Final stance**: "The cross-examination clarified the novelty of BatchQuantizedKVCache and the non-trivial UMA-specific design. I'm upgrading from WEAK ACCEPT to ACCEPT.

**Minor revisions requested**:
1. Add standard deviation or min/max to benchmark results (addresses Dr. D)
2. Explicitly state 'systems paper' positioning in intro (addresses Dr. A and Dr. E)
3. Add note about total memory context for 72% KV cache savings (addresses Dr. A)

**Vote**: ACCEPT. The contributions are sufficient for a systems-oriented venue."

---

### Dr. C (Edge Practitioner): STRONG ACCEPT (unchanged)

**Final stance**: "No change in my assessment. This is exactly what the community needs: practical, reproducible systems work that enables real applications.

**Suggestions (not required)**:
- Open-source the code ASAP so practitioners can reproduce
- Add a 'getting started' guide for users on Mac hardware
- Consider a demo video showing multi-agent workflow

**Vote**: STRONG ACCEPT."

---

### Dr. D (Statistician): WEAK REJECT → BORDERLINE (after debate)

**Final stance**: "Dr. F's argument about smooth trends across context lengths is persuasive. The results are likely representative, not outliers. However, I still want stronger statistical rigor.

**Required revisions**:
1. Increase to 5 runs minimum (median + quartiles)
2. Add confidence intervals or standard deviation to key metrics
3. Provide full benchmark data (all context lengths for E2E speedup, not just 4K)
4. Add ablation study: character-level vs token-level matching contribution

**Vote**: BORDERLINE (leaning accept if revisions addressed)."

---

### Dr. E (COLM PC Editor): BORDERLINE ACCEPT → WEAK ACCEPT (after debate)

**Final stance**: "The discussion has clarified that COLM values systems work, and this paper fits that scope. The positioning as a systems paper (per Dr. B's revision #2) resolves my main concern.

**Required revisions**:
1. Explicit positioning statement in intro: 'This is a systems paper...'
2. Prominent caveat for 72% memory savings (KV-only, not total)
3. Add data points to Figure 2 (TTFT chart) for transparency

**Vote**: WEAK ACCEPT (conditional on clarifications)."

---

### Dr. F (Industry Expert): STRONG ACCEPT (unchanged)

**Final stance**: "No change. This is production-ready work that advances edge inference. The debate has not revealed any disqualifying flaws.

**One suggestion**: Include a 'deployment guide' appendix showing how to integrate this into existing agent frameworks (AutoGen, LangChain, etc.).

**Vote**: STRONG ACCEPT."

---

## Consensus Summary

### Vote Tally

| Panelist | Final Vote | Strength |
|----------|-----------|----------|
| Dr. A | Weak Reject | 2/10 |
| Dr. B | Accept | 7/10 |
| Dr. C | Strong Accept | 9/10 |
| Dr. D | Borderline | 5/10 |
| Dr. E | Weak Accept | 6/10 |
| Dr. F | Strong Accept | 9/10 |

**Average score**: 6.3/10 (BORDERLINE ACCEPT)

**Consensus**: ACCEPT WITH MAJOR REVISIONS

---

## Ranked Action Items (by consensus)

### Tier 1: MUST FIX (required by 4+ panelists)

1. **Explicit positioning as systems paper** (Dr. A, Dr. B, Dr. E)
   - Add statement in Introduction: "This is a systems paper focused on practical edge inference, not claiming algorithmic novelty."
   - Status: **DONE** (added in response to feedback.md)

2. **Clarify 72% KV cache savings context** (Dr. A, Dr. E, implicit from Dr. F)
   - Add note that this is KV-only, not total system memory
   - Status: **DONE** (added in response to feedback.md)

3. **Add statistical rigor** (Dr. D, Dr. B)
   - Increase to 5 runs or add standard deviation/min/max
   - Status: **NOT DONE** (user should rerun benchmarks)

### Tier 2: STRONGLY RECOMMENDED (2-3 panelists)

4. **Add perplexity evaluation** (Dr. A, Dr. E implied)
   - Measure Q4 vs FP16 generation quality
   - Status: **NOT DONE** (would require new experiments)

5. **Expand model coverage** (Dr. A, Dr. D)
   - Add Llama 3.1 and Qwen 2.5 benchmarks
   - Status: **NOT DONE** (models "supported" but not benchmarked)

6. **Add baseline comparisons** (Dr. A)
   - Benchmark vs llama.cpp --prompt-cache, LM Studio, Ollama
   - Status: **NOT DONE** (no external baseline comparisons)

7. **Provide full benchmark data** (Dr. D)
   - Show E2E speedup at all context lengths, not just 4K
   - Status: **PARTIAL** (need to verify against novelty.md data)

### Tier 3: NICE TO HAVE (1-2 panelists)

8. **Ablation studies** (Dr. D)
   - Character-level vs token-level matching contribution
   - Interleaved scheduler vs naive batching
   - Status: **NOT DONE**

9. **Add data points to figures** (Dr. E)
   - Show individual measurements, not just fitted lines
   - Status: **CAN BE ADDED** (modify pgfplots charts)

10. **Deployment guide** (Dr. F)
    - Appendix showing integration with AutoGen/LangChain
    - Status: **NOT DONE**

---

## Meta-Committee Decision

**Recommendation to Senior Area Chair**: **CONDITIONAL ACCEPT**

**Rationale**: The panel is divided (2 strong accept, 2 weak accept, 1 borderline, 1 weak reject), but the consensus leans toward acceptance if critical revisions are made. The systems contribution is real, the problem is practical, and the evaluation (while limited) demonstrates clear benefits.

**Required revisions before acceptance**:
1. ✅ Explicit positioning as systems paper (DONE)
2. ✅ Clarify 72% savings is KV-only (DONE)
3. ⬜ Add statistical rigor (5 runs + std dev)
4. ⬜ Add perplexity evaluation OR cite extensive prior work validating Q4 accuracy

**Strongly recommended revisions**:
- Expand model coverage to all 4 claimed architectures
- Add external baseline comparisons (llama.cpp, LM Studio)
- Provide ablation studies for design choices

**Timeline**: If authors address required revisions within 2 weeks, recommend ACCEPT. Otherwise, encourage resubmission to systems track (MLSys, EuroSys) with expanded evaluation.

---

**Final Meta-Committee Vote**: **ACCEPT (conditional on revisions 1-4)**
