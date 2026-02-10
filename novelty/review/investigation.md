# Forensic Investigation: Methodology Analysis of semantic_colm2026.tex

**Date**: 2026-02-09
**Scope**: Critical examination of experimental methodology, analytical claims, hardware specifications, and theoretical framing in the COLM 2026 submission "Agent Memory Below the Prompt."
**Verdict**: The paper presents a working system with real benchmark data, but several methodological choices create gaps between what is claimed and what is demonstrated. The most serious issue is the perplexity evaluation, which does not measure what the paper says it measures.

---

## Table of Contents

1. [Area 1: Perplexity Methodology](#area-1-perplexity-methodology)
2. [Area 2: Sliding Window Evaluation](#area-2-sliding-window-evaluation)
3. [Area 3: Ablation Validity](#area-3-ablation-validity)
4. [Area 4: Hardware Table Accuracy](#area-4-hardware-table-accuracy)
5. [Area 5: Nielsen Response Time Thresholds](#area-5-nielsen-response-time-thresholds)
6. [Summary of Findings](#summary-of-findings)

---

## Area 1: Perplexity Methodology

### What the Paper Claims

The paper claims "<0.1 perplexity degradation" from Q4 KV cache quantization (Abstract, Section 5.5 Limitations, Section 6 Conclusion, Appendix C). The abstract states: "Q4 quantization fits 3.6x more agent contexts into fixed device memory than FP16, with <0.1 perplexity degradation." This is the central quality claim that justifies the entire Q4 pipeline.

### What Was Actually Done

The perplexity evaluation code (`/Users/dev_user/semantic/benchmarks/perplexity_benchmark.py`) reveals that the "<0.1 perplexity degradation" claim is **not based on actual Q4 KV cache inference**. The methodology is:

1. Run the model forward pass with standard FP16 KV cache (the default `mlx-lm` path).
2. Obtain the final logits tensor.
3. Quantize the **logits** to Q4 (4-bit, group size 64) using `mx.quantize()`.
4. Dequantize back to the original dtype.
5. Compute perplexity from the round-tripped logits.
6. Compare against the unperturbed FP16 logits.

The code explicitly documents this in comments (lines 92-103 of `perplexity_benchmark.py`):

```python
"""
Strategy: Run the model normally (FP16 KV internally), then quantize
and dequantize the logits to simulate the noise floor of Q4 KV cache.

This is an upper bound on Q4 KV degradation because:
- Q4 noise in KV cache propagates through subsequent attention layers
- Quantizing final logits applies noise only once
- Actual Q4 KV cache error is distributed across layers
"""
```

### Why This Is Problematic

**The simulation does not measure Q4 KV cache quality.** Logit quantization and KV cache quantization are fundamentally different operations:

1. **Error propagation**: In actual Q4 KV cache inference, quantization error is introduced at every attention layer. For a 48-layer model (Gemma 3), Q4 rounding error in the KV cache at layer 1 alters the attention output, which feeds into layer 2's KV computation, compounding the error through the entire network. The final logits reflect the accumulated error from all 48 layers.

2. **What logit quantization measures**: Quantizing the final logits applies a single round of Q4 noise to the output distribution. This measures the sensitivity of the softmax probability distribution to uniform Q4 perturbation at the output level. It does not capture the layer-by-layer error accumulation that characterizes actual KV cache quantization.

3. **Direction of the bound is wrong**: The code comments claim this is "an upper bound" (worst-case estimate). The reasoning given is that Q4 noise in KV cache "propagates through subsequent attention layers" while logit quantization "applies noise only once." This reasoning is backwards. The single-point logit perturbation is almost certainly a **lower bound** on actual degradation, because:
   - Actual KV cache quantization applies error at 48 (Gemma) or 27 (DeepSeek) injection points.
   - Each injection point introduces error that compounds through residual connections.
   - Attention softmax is highly sensitive to small perturbations in key vectors (which determine attention routing), more so than to perturbations in output logits.
   - Research from KVTuner (2025) shows that "attention distribution shifts at the token level in specific sensitive heads can degrade final accuracy," demonstrating that intermediate attention errors do not translate linearly to output logit errors -- they can be amplified.

4. **Prior work comparison is misleading**: The paper cites KIVI, KVQuant, QuantSpec, RotateKV, and XQuant for their "<0.1 PPL degradation" findings. Those studies all measure perplexity with **actual quantized KV caches during inference**, not via logit perturbation. The paper's methodology is not comparable to those studies, yet the citation framing implies equivalence.

### The Perplexity Table Is Empty

Appendix C (Table 7, `\label{tab:perplexity}`) contains placeholder dashes:

```
Gemma 3 12B         --- --- ---
DeepSeek-V2-Lite    --- --- ---
```

The paper states: "[Results to be filled after running benchmarks/perplexity_benchmark.py.]" This means the claimed "<0.1 perplexity degradation" has never been measured, even with the flawed simulation. The claim in the abstract and conclusion is entirely based on extrapolation from prior work on different models with different quantization implementations.

### What Would Strengthen the Claim

1. **Gold standard**: Implement a Q4 KV cache forward pass where each layer's KV tensors are quantized to Q4 after computation and dequantized before the next attention operation. This is what KIVI, KVQuant, and other cited works do. The system already has `QuantizedKVCache` and `BatchQuantizedKVCache` implementations that operate in Q4; these could be used for perplexity evaluation.

2. **Practical alternative**: Run the full inference pipeline (cold prefill with Q4 KV cache, then decode) on a standard benchmark (WikiText-2, C4, or MMLU) and compare outputs against FP16 inference. This would measure end-to-end quality including the actual Q4 quantization path.

3. **Minimum fix**: Fill in the perplexity table with actual measurements from the simulation, but add a clear caveat: "These values represent logit-level Q4 round-trip noise, not actual KV cache quantization degradation. Actual degradation may differ due to error accumulation across layers."

4. **Task-based evaluation**: Measure downstream task accuracy (e.g., MMLU, HumanEval, or even the paper's own keyword-based quality metric) under FP16 vs Q4 KV cache conditions. This would directly demonstrate whether Q4 caching preserves practical output quality.

### Severity

**High.** The "<0.1 perplexity degradation" claim appears in the abstract, is repeated in the conclusion, and is used to justify the entire Q4 pipeline design. It is based on a methodology that does not measure what it purports to measure, and the actual measurements have never been run. A reviewer who examines `perplexity_benchmark.py` or notices the empty table will flag this as a significant credibility issue.

---

## Area 2: Sliding Window Evaluation

### What the Paper Claims

Appendix C describes the perplexity evaluation methodology: "We process the text in 512-token sliding windows and compute per-token log-likelihood." The code uses `WINDOW_SIZE = 512` and `STRIDE = 256`.

### What This Means

Each evaluation window contains 512 tokens. The stride of 256 means consecutive windows overlap by 256 tokens. For each window, the model processes all 512 tokens in a single forward pass (no KV cache carryover between windows). Only the non-overlapping tokens (the last 256 in each window after the first) contribute to the perplexity calculation.

### Limitations

1. **Maximum context is 512 tokens**: Each evaluation window is independent. The model never sees more than 512 tokens of context when computing the log-likelihood of any token. This is a severe limitation for evaluating a system that claims to handle contexts of 1K-32K tokens. The entire paper's value proposition is about persisting and reusing long KV caches. Evaluating quality with 512-token windows says nothing about whether Q4 quantization degrades quality at 4K, 8K, 16K, or 32K context lengths.

2. **No long-range dependency evaluation**: Recent research ("What is Wrong with Perplexity for Long-context Language Modeling?", 2024; "Rethinking Perplexity: Revealing the Impact of Input Length on Perplexity Evaluation in LLMs", 2025) demonstrates that sliding-window perplexity systematically undervalues long-context modeling ability. Window size has a significant effect on reported perplexity: increasing the window from 16 to 1024 tokens reduces perplexity by 57-62% in some models. At 512 tokens, the evaluation captures only local coherence.

3. **No error accumulation measurement**: In a real multi-turn conversation, the KV cache grows incrementally. Q4 quantization error from turn 1 persists in the cache when turn 2 is processed, and so on. Over a 5-phase, 25-turn conversation (the paper's prisoner's dilemma scenario), quantization errors could accumulate. The sliding window evaluation is stateless between windows and cannot detect this.

4. **Gemma 3's sliding window architecture**: Gemma 3 12B uses a 1024-token sliding window for 40 of its 48 attention layers. Evaluating with a 512-token window means the sliding window mechanism is never tested at capacity. At 512 tokens, all layers (both global and sliding window) see the full context, so the hybrid attention architecture is never exercised in its distinctive mode.

5. **Memory-driven sizing**: The code comments explain the choice: "Memory-aware sizing... We evaluate ~8K tokens total to stay well within budget." The window size and total evaluation size are chosen for device memory constraints, not for methodological soundness. This is an engineering constraint, not a scientific design.

### What Would Strengthen the Claim

1. **Evaluate at target context lengths**: Run perplexity evaluation at 4K, 8K, and 16K context windows -- the same lengths used in the TTFT benchmarks. This would show whether Q4 quality holds at the context lengths the paper actually claims to support.

2. **Multi-turn accumulation test**: Process a long document incrementally (simulating multi-turn conversation), quantizing the KV cache to Q4 at each turn, and compare final perplexity against a single-pass FP16 evaluation. This would capture error accumulation.

3. **Use the system's own Q4 cache**: Run the full system pipeline (cold prefill -> Q4 cache save -> warm reload -> generate) and compare output quality against a non-cached baseline. The system exists and works; use it for evaluation.

4. **Downstream task evaluation at scale**: Instead of perplexity on WikiText-2, evaluate on a task that requires long-context understanding (e.g., multi-document QA, long-form summarization) with and without Q4 caching.

### Severity

**Medium.** The 512-token window is a reasonable starting point for a memory-constrained device, and prior work confirms that 4-bit KV cache quantization generally has low impact on quality. But the evaluation does not support claims about long-context quality, and the paper's main contribution is a long-context system. There is a disconnect between what is evaluated and what is claimed.

---

## Area 3: Ablation Validity

### What the Paper Claims

Table 5 (Section 4.4) presents an "ablation analysis" claiming to "isolate each component's contribution." Four components are ablated: persistence (30x TTFT reduction), Q4 vs FP16 (3.6x capacity), batching (2.0x SysTPS), and cross-phase injection (1.9x TTFT reduction).

The paper states: "All numbers come from existing benchmark data (Tables 3-6) or analytical calculations (Table 3)."

### What Was Actually Done

These are not ablations in the standard machine learning sense. A proper ablation removes one component at a time from the system and measures the impact. What the paper presents is:

1. **Persistence**: Compares warm TTFT (513 ms) against cold TTFT (15,502 ms). This is a comparison of two operational modes, not an ablation. It does not answer "what happens if we remove persistence from the system?" because the cold path IS the system without persistence. This is valid as a comparison but should not be labeled an ablation.

2. **Q4 vs FP16**: The "without" value (5 agents at 8K) comes from the analytical formula in Table 3. No FP16 system was built, benchmarked, or run. The number is derived from the calculation: `15.2 GB / (3.0 GB per agent at FP16 8K) = 5 agents`. This is a mathematical calculation, not a measurement. The implicit claim is that the system could not serve more than 5 agents with FP16 caches, but this was never empirically tested.

3. **Batching**: Compares SysTPS (22.4 for batch=2) against per-agent TPS (11.2, which is SysTPS/2). The footnote acknowledges that "Per-agent TPS = SysTPS/2, representing single-agent throughput." This is not an ablation of the batching component. It is a comparison of batch=2 throughput against half of batch=2 throughput. A proper ablation would compare batch=2 SysTPS against batch=1 SysTPS measured on the same workload. The batch=1 data exists in the paper's measurements but is not used here.

4. **Cross-phase**: Compares persistent Phase 5 TTFT (1,705 ms) against cold Phase 5 TTFT (3,292 ms). This is the same operational mode comparison as #1, applied to a later phase. It is a valid comparison but is not an ablation of the cross-phase injection mechanism.

### Why This Matters

Calling these "ablations" implies a level of experimental rigor that is not present. Ablation studies are a specific methodology where you remove one component and re-run the full system to measure its individual contribution. The paper instead presents:
- Two mode comparisons (persistence, cross-phase): valid but not ablations
- One analytical calculation (Q4 capacity): valid but not empirical
- One arithmetic identity (batching): SysTPS vs SysTPS/2 is not informative

### What Would Strengthen the Claim

1. **Real ablation for persistence**: Build a variant of the system that does NOT persist caches (always cold-starts). Measure TTFT across the same configuration matrix. Compare against the full system. This is essentially what the cold/warm comparison does, so relabeling it as "comparison" rather than "ablation" would be sufficient.

2. **Real ablation for Q4 vs FP16**: Implement FP16 KV cache storage (the system architecture supports it -- just skip quantization). Run the same benchmark suite with FP16 caches. Measure actual agent capacity (how many agents can be loaded simultaneously before OOM), actual TTFT (FP16 disk I/O is slower because files are 3.6x larger), and actual throughput (FP16 attention may be faster per token because no dequantization is needed). The tradeoff is more nuanced than "3.6x more agents."

3. **Real ablation for batching**: Compare batch=1 warm SysTPS against batch=2 warm SysTPS for the same workload. The batch=1 data appears to exist in the measurements (the per-agent TPS column). If batch=1 warm SysTPS is close to 11.2 (matching per-agent TPS), that confirms linear scaling. If it differs, that reveals the overhead of batching.

4. **Real ablation for cross-phase**: Run the 5-phase scenario with persistence enabled but cross-phase cache extension disabled (each phase loads the Phase 1 cache only, not the accumulated cache). This would isolate the contribution of cache accumulation from the contribution of basic persistence.

5. **Relabel the table**: If empirical ablations are infeasible, rename the section from "Ablation Analysis" to "Component Contribution Analysis" and clearly state that the numbers are derived from operational comparisons and analytical calculations, not from controlled ablation experiments.

### Severity

**Medium.** The individual numbers are not wrong -- the comparisons are drawn from real measurements (except Q4 capacity, which is analytical). The issue is framing: calling them "ablations" implies a methodology that was not followed. A reviewer familiar with ablation study conventions will notice this.

---

## Area 4: Hardware Table Accuracy

### What the Paper Claims

Table 1 and Appendix F (Table 8) list memory capacity, memory bandwidth, and SSD/PCIe bandwidth for six devices: M4 Pro, M4 Max, DGX Spark, RTX 5090, RTX 4090, and iPhone 17 Pro.

### Verification of Each Specification

#### M4 Pro (Mac Mini)
| Spec | Paper | Verified | Status |
|------|-------|----------|--------|
| Memory | 24 GB | 24 GB (configurable 24-48 GB) | Correct |
| Bandwidth | 273 GB/s | 273 GB/s (Apple spec) | Correct |
| SSD | 7 GB/s | ~7 GB/s (Apple internal SSD) | Correct |
| Type | Unified | Unified LPDDR5X | Correct |

**Verdict**: All specifications are correct for the base M4 Pro configuration (24 GB).

#### M4 Max (MacBook)
| Spec | Paper | Verified | Status |
|------|-------|----------|--------|
| Memory | 128 GB | 36-128 GB (configurable) | Partially correct |
| Bandwidth | 546 GB/s | 546 GB/s (40-core GPU variant) | Correct with caveat |
| SSD | 7 GB/s | ~7 GB/s | Correct |
| Type | Unified | Unified LPDDR5X | Correct |

**Caveat**: The 546 GB/s bandwidth applies only to the 40-core GPU variant of M4 Max. The 32-core GPU variant has 410 GB/s. The paper lists the highest configuration without noting the variant. The 128 GB memory also applies only to the highest configuration. The extended table (Appendix F) shows "36-128" for memory, which is accurate, but the main Table 1 shows only 128. This is misleading but not incorrect.

#### DGX Spark
| Spec | Paper | Verified | Status |
|------|-------|----------|--------|
| Memory | 128 GB | 128 GB LPDDR5X | Correct |
| Bandwidth | 273 GB/s | 273 GB/s | Correct |
| SSD | 11 GB/s | ~11.4 GiB/s (sequential read, 1M block, 16 threads) | Correct |
| Type | Unified | Unified (Grace Blackwell GB10) | Correct |

**Verdict**: All specifications are correct. The SSD bandwidth of 11 GB/s is confirmed by StorageReview benchmarks (11.4 GiB/s peak sequential read).

#### RTX 5090
| Spec | Paper | Verified | Status |
|------|-------|----------|--------|
| Memory | 32 GB | 32 GB GDDR7 | Correct |
| Bandwidth | 1,792 GB/s | 1,792 GB/s (512-bit, 28 Gbps GDDR7) | Correct |
| SSD (PCIe) | 64 GB/s | 64 GB/s (PCIe 5.0 x16 unidirectional) | Correct |
| Type | Discrete | Discrete VRAM | Correct |

**Verdict**: All specifications are correct. The "SSD" column for discrete GPUs represents PCIe host-device bandwidth, which is noted in the table footnote. PCIe 5.0 x16 provides 64 GB/s unidirectional bandwidth.

#### RTX 4090
| Spec | Paper | Verified | Status |
|------|-------|----------|--------|
| Memory | 24 GB | 24 GB GDDR6X | Correct |
| Bandwidth | 1,008 GB/s | 1,008 GB/s (384-bit, 21 Gbps GDDR6X) | Correct |
| SSD (PCIe) | 32 GB/s | 32 GB/s (PCIe 4.0 x16 unidirectional) | Correct |
| Type | Discrete | Discrete VRAM | Correct |

**Verdict**: All specifications are correct. PCIe 4.0 x16 provides approximately 32 GB/s unidirectional bandwidth, matching the table.

#### iPhone 17 Pro
| Spec | Paper | Verified | Status |
|------|-------|----------|--------|
| Memory | 12 GB | 12 GB LPDDR5X | Correct |
| Bandwidth | 77 GB/s | 76.8 GB/s (LPDDR5X 9600 MT/s) | Correct (rounded) |
| SSD | 2 GB/s | ~2 GB/s (NAND flash) | Plausible |
| Type | Unified | Unified | Correct |

**Verdict**: Memory and bandwidth are correct (77 GB/s is a reasonable rounding of 76.8 GB/s). The 2 GB/s SSD bandwidth is plausible for NAND flash on a mobile device, though Apple does not publish exact NVMe sequential read speeds for iPhone. This is an estimate.

### Extended Table (Appendix F) Additional Entries

The extended table adds M4 (MacBook Air), M4 Ultra (Studio), and prices. The M4 bandwidth of 120 GB/s, M4 Ultra bandwidth of 819 GB/s, and price points all match published Apple specifications and retail prices.

### Overall Assessment

**The hardware specifications are accurate.** All verifiable numbers match published specifications from the respective manufacturers. The RTX PCIe bandwidth numbers correctly distinguish between VRAM bandwidth and host-device transfer bandwidth, and the footnote makes this clear. The one minor issue is that Table 1 presents the highest M4 Max configuration (128 GB, 546 GB/s) without noting that lower configurations exist, but the extended table in the appendix corrects this.

### Severity

**Low.** The hardware table is factually accurate. No corrections needed.

---

## Area 5: Nielsen Response Time Thresholds

### What the Paper Claims

Section 2.3 states: "Nielsen's thresholds [nielsen1993] identify 100 ms as instantaneous, 1 s as acceptable, and 10 s as the limit before users disengage. No current local AI system meets the 1 s threshold at long context [nielsen2024speed]."

The paper uses the 1-second threshold as the target for TTFT, framing warm-cache TTFT of 513 ms (Gemma 4K) as crossing "Nielsen's 1 s threshold into acceptable territory."

### The 1993 Reference

The citation is to Nielsen, J. (1993). *Usability Engineering*. Academic Press. Chapter 5 defines three thresholds:
- 0.1 seconds: system reaction feels instantaneous
- 1.0 seconds: user's flow of thought stays uninterrupted
- 10 seconds: limit of user's attention; beyond this, users want to switch tasks

These thresholds are based on earlier human factors research (Miller, 1968; Card et al., 1983) grounded in neuropsychological constants. Nielsen himself has repeatedly stated that these thresholds remain valid because they derive from human cognitive processing speeds, which have not changed. In a 2023 Substack post, he reaffirmed: "When something has held for 55 years in the computer business, it'll probably remain true for many more laps around the sun."

**The 1993 reference is valid.** The thresholds are not arbitrary 1993 technology conventions; they are empirically grounded in human cognitive timing that has not changed.

### The 2024 Reference

The bibliography entry (`nielsen2024speed`) cites "The Need for Speed in the Age of AI" attributed to "Nielsen Norman Group, 2024." The actual article is titled "The Need for Speed in AI" and was published on Jakob Nielsen's Substack (jakobnielsenphd.substack.com) and on uxtigers.com in August 2023, not 2024. The article's core argument is that AI tools need sub-second response times just like all previous UI paradigms, and that current AI systems generally fail to meet this standard.

**Issues with the citation**:
1. The year is wrong: the article was published in August 2023, not 2024. The bibliography lists `year={2024}`.
2. The title is slightly different: the actual title is "The Need for Speed in AI," not "The Need for Speed in the Age of AI."
3. The publication venue is Nielsen's personal Substack/website, not the Nielsen Norman Group's main website (nngroup.com). Nielsen separated from NN/g and now publishes independently.

These are minor bibliographic errors that do not affect the substance of the argument.

### Applicability to AI Agent Interaction

The paper's use of the 1-second threshold for TTFT in AI agent interaction deserves scrutiny:

1. **Different interaction model**: Nielsen's thresholds were developed for direct-manipulation GUIs where the user performs an action and waits for a response. AI chat interaction is different: the user types a prompt, submits, and waits. The "submission to first visible response" is more analogous to a page load than a button click. For page loads, Nielsen used the 10-second threshold, not the 1-second threshold.

2. **Streaming changes perception**: The paper measures TTFT (time to first token), not time to complete response. With streaming, the user sees tokens appearing progressively. Research on streaming AI responses (including Nielsen's own article) suggests that progressive display substantially increases perceived responsiveness. The effective perceived wait time for a 500 ms TTFT with streaming is much lower than a 500 ms wait for a complete response. The paper's TTFT metric is actually more favorable than the threshold implies.

3. **Multi-agent context**: In a multi-agent system, the user is typically observing agent-to-agent interactions, not waiting for a direct response. The tolerance for latency in observed multi-agent dialogue (where the user watches agents converse) may be quite different from the tolerance for latency in direct user-to-agent interaction. The paper does not distinguish between these interaction modes.

4. **The paper's argument is still sound**: Despite these nuances, the core argument holds: reducing TTFT from 15.5 seconds to 513 ms is a qualitative improvement in user experience, regardless of which exact threshold applies. At 15.5 seconds, users disengage. At 513 ms, the response feels responsive. The Nielsen framing provides a convenient vocabulary for this argument, even if the exact threshold boundaries may not directly apply to AI agent interaction.

### What Would Strengthen the Claim

1. **Fix the citation**: Correct the year to 2023 and the title to "The Need for Speed in AI."

2. **Acknowledge the interaction model difference**: Add a sentence noting that Nielsen's thresholds were developed for direct-manipulation interfaces, and that AI agent interaction may have different dynamics (especially with streaming and multi-agent observation).

3. **Cite AI-specific latency research**: If available, cite studies measuring user tolerance for AI chatbot response latency specifically. Nielsen's 2023 article itself notes that no current AI system meets the 1-second threshold, which supports the paper's argument without relying on the 1993 GUI thresholds directly.

4. **Distinguish TTFT from end-to-end latency**: The paper already notes that TTFT is 84-94% of total latency for short outputs (Section 2.3), which is good. Making this distinction more explicit in the Nielsen threshold discussion would strengthen the framing.

### Severity

**Low.** The Nielsen references are substantively valid despite minor bibliographic errors. The 1993 thresholds remain accepted in HCI literature. The argument that sub-second TTFT represents a qualitative user experience improvement is well-supported. The main risks are (a) a reviewer catching the incorrect year and (b) a reviewer arguing that 1993 GUI thresholds do not apply to 2026 AI interaction. Both are addressable with minor revisions.

---

## Summary of Findings

| Area | Severity | Core Issue |
|------|----------|------------|
| Perplexity methodology | **High** | Simulated via logit quantization, not actual Q4 KV cache. Results table is empty. Claim in abstract is unsupported by the paper's own methodology. |
| Sliding window evaluation | **Medium** | 512-token window cannot evaluate quality at the 4K-32K context lengths the paper targets. No multi-turn error accumulation testing. |
| Ablation validity | **Medium** | Labeled as "ablation" but consists of mode comparisons, analytical calculations, and an arithmetic identity. No true component-removal ablation. |
| Hardware table accuracy | **Low** | All specifications verified correct against manufacturer data. Minor M4 Max configuration ambiguity. |
| Nielsen thresholds | **Low** | Substantively valid. Minor bibliographic errors (wrong year, slightly wrong title). Interaction model differences unacknowledged but do not invalidate the argument. |

### Recommendations in Priority Order

1. **Critical**: Either run actual Q4 KV cache perplexity evaluation (using the system's own `QuantizedKVCache` for forward passes) or remove the "<0.1 perplexity degradation" claim from the abstract and conclusion. Do not publish with an empty results table.

2. **Important**: Add perplexity evaluation at longer context lengths (at minimum 4K, ideally 8K and 16K) to match the TTFT evaluation matrix. If memory constraints prevent this, acknowledge the limitation explicitly.

3. **Important**: Relabel Section 4.4 from "Ablation Analysis" to "Component Contribution Analysis" or similar. If true ablations are feasible (especially Q4 vs FP16 and batch=1 vs batch=2), run them.

4. **Minor**: Fix the Nielsen 2024 citation to 2023 and correct the title.

5. **Minor**: Note the M4 Max configuration variant in Table 1 or add a footnote indicating the 546 GB/s applies to the 40-core GPU variant.

### What the Paper Gets Right

The paper has genuine strengths that this investigation does not diminish:

- **Real system with real benchmarks**: 216 individual measurements per model, with thermal-aware cooldown and median reporting. The TTFT and throughput numbers are from actual runs, not simulations.
- **Two architecturally distinct models**: Testing on both GQA (Gemma 3) and MLA/MoE (DeepSeek) demonstrates generality of the approach.
- **Honest limitations section**: The paper acknowledges two models tested, fixed output length, and no working memory quality metric.
- **Hardware table is accurate**: Every specification was verified against manufacturer data.
- **The core contribution is real**: Persistent Q4 KV cache with 30-130x TTFT reduction is a genuine and significant engineering achievement. The methodology issues are about how the quality-safety claim is supported, not about whether the system works.
