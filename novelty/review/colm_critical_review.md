# COLM 2026 Critical Review

**Paper**: "Agent Memory Below the Prompt: Persistent Q4 KV Cache for Multi-Agent LLM Inference on Edge Devices"

**Venue**: COLM 2026 (Conference on Language Modeling, ~29% acceptance rate)

**Review Date**: February 9, 2026

**Reviewer**: Independent critical assessment synthesizing full paper reading, claim audit, methodology investigation, simulated peer review, and literature analysis.

---

## 1. Acceptance Likelihood Assessment

**Overall Score: 4.5 / 10**

**Acceptance Probability: 15--20%**

This is a well-executed systems paper that occupies a genuine gap in the design space (persistent Q4 KV cache with batched inference on unified memory hardware), but it faces an uphill battle at COLM because the contribution is engineering integration, not ML insight. The paper does not teach us something new about how language models work; it teaches us how to build efficient infrastructure around them. That distinction matters at COLM.

### Scoring Against COLM's 5 Dimensions

**1. Empiricism (5/10)**

The paper's measurement methodology is above average for systems papers: 198 measurements per model, thermal-aware cooldown, median-of-3 reporting, explicit quality checks. The TTFT and throughput numbers are drawn from real benchmark runs with verified data (the claim audit found 91/118 claims matching exactly, 13 minor rounding discrepancies).

However, the empiricism has significant holes:
- The perplexity evaluation uses a logit-level proxy, not actual Q4 KV cache inference. The paper is now transparent about this (calling it "a logit-level proxy" and an "estimate"), but the abstract still says "<0.1 PPL" without the "estimated" qualifier present in the body.
- Only two models are tested. Adding Llama 3 (standard GQA, the universal baseline) would take the generalization story from "two models" to "two architectures, three models."
- Evaluation runs on a single hardware device (M4 Pro, 24 GB).
- No baseline comparison against any existing system (vllm-mlx, llama.cpp with `--slot-save-path`, Ollama).
- The ablation table (Table 5) reuses numbers from other tables and includes a tautological batching comparison (SysTPS vs SysTPS/2 is always 2.0x by definition).
- The multi-agent scenarios are relatively modest: 4 agents at 1--3K context (prisoner's dilemma) and 10 experts at ~4K (wiki routing). The system is designed for 10+ agents at 8K+, but the evaluation never reaches that scale.

**2. Technological Impact (6/10)**

The paper addresses a real deployment pain point. Multi-agent cold-start latency on edge devices is a concrete problem that practitioners encounter. The 30x TTFT reduction at 4K context is a qualitative improvement (15.5s to 513ms), not an incremental optimization. The system is open-source, works on commodity hardware, and exposes an OpenAI-compatible API.

Points deducted because:
- The system is MLX-specific. Apple Silicon is a real and growing platform, but the majority of edge inference still runs on llama.cpp (cross-platform) or runs on CUDA (RTX/Jetson). Portability to PyTorch/CUDA is discussed but not demonstrated.
- The "infrastructure layer" claim (AutoGen, CrewAI, LangGraph integration) is untested. No framework integration is demonstrated.
- The system handles only batch=2. Real multi-agent systems with 5--20 agents would need deeper batching or sophisticated scheduling.

**3. Ambition/Vision (5/10)**

The paper frames persistent Q4 KV cache as "agent memory below the prompt" -- a layer between the model and the agentic framework. This is a forward-looking architectural vision. The cross-phase context injection (treating KV cache as working memory that accumulates across conversation phases) is the most conceptually interesting contribution.

However, the vision is constrained by what is demonstrated. The cross-phase injection is validated only on a 5-phase scenario with modest context (1--3K tokens per agent). A truly ambitious demonstration would show 10+ agents, 10+ phases, reaching 16K--32K context, with task-completion metrics showing that persistent cache enables behaviors that cold-start cannot (information retention across phases, strategy consistency, etc.).

**4. Understanding/Depth (3/10)**

This is the weakest dimension. COLM values papers that deepen our understanding of language models. The accepted COLM 2025 KV cache papers (PyramidKV, SQuat, SentenceKV) all revealed something about how attention works -- attention pyramids vary by layer, quantization hurts reasoning in structured ways, attention clusters at sentence boundaries. This paper reveals nothing about attention mechanisms, quantization error propagation, or language model behavior.

The closest candidate for an ML insight is the observation that character-level prefix matching is more robust than token-ID matching because BPE tokenization is context-dependent. This is a genuine and potentially useful observation about tokenization, but it receives exactly one paragraph (Section 3.3) and no empirical evaluation. How often does token-ID matching fail in multi-agent scenarios? What is the false-negative rate? This could be a small but legitimate contribution to understanding, but it is underdeveloped.

The perplexity evaluation, even in its improved form with actual numbers, uses a logit-level proxy rather than measuring actual Q4 KV cache quality. This means the paper does not even fully characterize the quality impact of its own core mechanism. The paper acknowledges this ("end-to-end perplexity with our Q4 KV cache pipeline remains future work"), but it means the paper cannot claim to deepen understanding of Q4 KV cache quality behavior.

**5. Clarity/Honesty/Trust (7/10)**

This is the paper's strongest dimension. The writing is direct, precise, and unusually transparent:
- The limitations section lists five specific weaknesses, including "no working memory quality metric."
- The perplexity methodology note (Appendix E) explicitly states it is a proxy, not a direct measurement.
- The ablation table states that numbers come from existing tables and analytical calculations.
- The MLX engineering notes (Appendix B) honestly catalog failure modes.
- The claim audit found zero incorrect claims (all 10 "mismatches" are rounding, labeling, or framing issues, not fabricated data).

Points deducted for:
- The abstract says "<0.1 perplexity degradation" without the word "estimated." The body and appendix qualify this, but the abstract -- the most-read part -- does not.
- Figure 1 annotates "81.6x TTFT (hot)" which is from older data; current Table 3 yields 81.4x at 16K.
- Figure 2 caption says "40--130x at 32K" but the actual range at 32K is 69--130x (the minimum is DeepSeek warm at 69x, not 40x).
- Figure 2 caption says "sub-second reload regardless of context length" but Gemma warm at 32K is 1621ms.
- The Appendix D per-layer Q4 breakdown has two canceling arithmetic errors (data term missing /2, scales/biases missing x2) that produce a confusing intermediate result of "792 MB, with overhead: ~432 MB."
- Section 4.3 says "DeepSeek is consistently ~2.9x faster" but the actual range is 2.1--2.9x; 2.1x at 16K is not "approximately 3x."

---

## 2. Title and Abstract Assessment

### Title

"Agent Memory Below the Prompt: Persistent Q4 KV Cache for Multi-Agent LLM Inference on Edge Devices"

**Assessment**: The title is well-constructed but oversells scope in one dimension.

- "Agent Memory Below the Prompt" is an evocative subtitle that correctly positions the contribution (infrastructure layer beneath agent logic).
- "Persistent Q4 KV Cache" is precise and accurately describes the mechanism.
- "Multi-Agent LLM Inference" is accurate.
- **"Edge Devices" oversells scope.** All evaluation is on a single Apple Silicon device. The paper does not test any other edge platform (RTX, DGX Spark, iPhone, Android). "Apple Silicon" or "Unified Memory Hardware" would be more accurate.

**Suggested revision**: "Agent Memory Below the Prompt: Persistent Q4 KV Cache for Multi-Agent LLM Inference on Apple Silicon" -- or keep "Edge Devices" if at least one additional device class (e.g., RTX 4090 with CUDA) is benchmarked by submission time.

### Abstract

The abstract is generally effective but has three issues:

1. **"<0.1 perplexity degradation" lacks qualification.** The body calls this an "estimate" based on a "logit-level proxy." The abstract should say "estimated <0.1 PPL degradation (consistent with prior Q4 KV quantization literature)" or similar. As written, a reader assumes this was measured end-to-end.

2. **"4x more agent contexts" vs "3.6x".** The abstract says "4x" but Table 2 shows the ratio varies from 3.6x (at 4K, 8K) to 4.0x (at 16K, 32K). The conclusion says "4x (12 vs 3 agents at 8K)" but 12/3 = 4.0x only because of floor rounding; the memory ratio is 3.56x. Using "4x" is generous. Consider "~4x" or "3.6x" for precision.

3. **The opening hook could be stronger.** The current opening describes the system's capabilities. A more compelling abstract would open with the problem: "Five agents, each holding 4K tokens of context. The server restarts. Total cold-start time: 77 seconds." This vivid framing appears in Section 1 but not the abstract.

---

## 3. Section-by-Section Necessity Review

### Section 1: Introduction (~0.9 pages)

**Verdict: Keep, minor trim.**

The 77-second opening hook is effective and should remain. The "Contributions" paragraph lists 5 items, which is one too many. Contribution (4) ("infrastructure layer for agentic frameworks via OpenAI-compatible API") is not substantiated by any integration test and should be folded into the discussion rather than listed as a contribution. Reduce to 4 contributions.

### Section 2: Background (~1.5 pages)

**Verdict: Trim by ~0.3 pages.**

- **Section 2.1 (Multi-Agent Memory Problem)**: Keep. The position-bias argument (citing "Lost in the Middle") is the strongest ML-grounded motivation and essential for COLM.

- **Section 2.2 (Edge Device Constraints)**: Trim. The GDPR/HIPAA sentence (line 100) is underdeveloped and invites challenge. Either develop it into a proper paragraph with specific scenarios or remove it. The memory budget calculation (24 - 6.8 - 7 = 10.2 GB) is useful and should stay, but note that the paper uses "7 GB OS/system" here but "2 GB OS" in the evidence verification -- these should be reconciled.

- **Section 2.3 (Interactivity and TTFT)**: Keep. The Nielsen threshold framing and the 84%/94% prefill-dominance calculation are effective. The RAG comparison paragraph is important for positioning.

**Table 1 (Hardware)**: Keep but fix. Remove iPhone 17 Pro (unreleased device) or replace with iPhone 16 Pro. Add footnote noting M4 Max bandwidth varies by configuration (410 or 546 GB/s). This table earns its space because it contextualizes the "why edge?" argument concretely.

### Section 3: System Design (~2.3 pages)

**Verdict: Keep, minor restructure.**

- **Section 3.1 (Block Pool)**: Keep. Essential to the contribution.
- **Section 3.2 (Q4 Pipeline)**: Keep. The 0.281 ratio derivation and Table 2 (FP16 vs Q4 capacity) are among the paper's strongest analytical contributions.
- **Section 3.3 (Prefix Matching)**: Keep but add empirical data. The character-level vs token-ID matching observation is the paper's best candidate for an ML insight. Add a paragraph showing how often this matters in the benchmark scenarios. If character-level matching never actually produces a different result than token-ID matching in the tested scenarios, acknowledge that.
- **Section 3.4 (Batched Quantized Inference)**: Restructure. The concurrency model description mixes the contribution (BatchQuantizedKVCache) with the limitation (single-thread due to MLX's lack of thread safety). Separate these clearly. The B=1 split workaround is an MLX constraint, not a design innovation.
- **Section 3.5 (Cross-Phase Context Injection)**: Keep. This is the most conceptually novel component.
- **Section 3.6 (Architectural Coverage)**: Keep. The Gemma/DeepSeek comparison with specific architectural details (GQA vs MLA, symmetric vs asymmetric K/V dims) demonstrates genuine engineering depth.

**Figure 1 (Architecture Diagram)**: Keep but fix annotations. The "81.6x TTFT (hot)" annotation uses stale data; update to match current Table 3 values. The "2.0--4.3x E2E speedup" annotation is not traceable to any table in the paper and should be sourced or removed.

**Table 2 (FP16 vs Q4 Capacity)**: Keep. One of the paper's strongest tables.

### Section 4: Evaluation (~2.5 pages)

**Verdict: Keep, fix issues.**

- **Section 4.1 (Setup)**: Fix the measurement count. The current text says "66 unique configurations... 198 individual measurements passing quality checks." Appendix C says the same. But the earlier draft (seen by simulated reviewers) said 72/216/198, which contradicts the appendix. Ensure all counts are consistent throughout.

- **Section 4.2 (TTFT Scaling)**: Keep. This is the paper's core result. The narrative is clear and the data is verified.

- **Section 4.3 (Batched Throughput)**: Keep but fix "consistently ~2.9x faster" to "approximately 2--3x faster" (actual range: 2.1--2.9x).

- **Section 4.4 (Ablation Analysis)**: Rename to "Component Contribution Analysis." The batching row is tautological (SysTPS vs SysTPS/2 is always 2.0x by definition). Either replace it with a real batch=1 vs batch=2 comparison or add a footnote explaining that the 2.0x is definitional, not a measured throughput gain from batching.

- **Section 4.5 (Multi-Phase)**: Keep. The phase-by-phase table (Table 6) showing growing speedup from 1.1x to 1.9x across 5 phases is the best demonstration of the cross-phase injection's value. DeepSeek's modest gains (1.0x to 1.3x) are honestly reported.

- **Section 4.6 (Multi-Agent Routing)**: Keep but address DeepSeek quality. DeepSeek scores 3/10 quality in Phase 1 and 4/10 in Phase 2. The paper should state whether this is a model issue (DeepSeek is optimized for code, not Wikipedia knowledge retrieval) or a caching issue. Running the same queries on DeepSeek without caching (cold-start) and showing the same low quality scores would exonerate the caching system.

**Table 3 (TTFT)**: Essential. Keep.

**Table 4 (Batch Throughput)**: Keep.

**Table 5 (Ablation)**: Rename and fix as described above.

**Table 6 (Phase Persistence)**: Keep. Well-structured comparison.

**Table 7 (Wiki Routing)**: Keep but address the quality scores.

**Figure 2 (TTFT Scaling)**: Keep but fix caption. Three errors: (1) "40--130x at 32K" should be "69--130x at 32K"; (2) "sub-second reload regardless of context length" is false (Gemma warm at 32K is 1621ms); (3) the log-log visualization is effective and the data points are verified to match Table 3.

### Section 5: Discussion (~1.0 pages)

**Verdict: Trim by ~0.3 pages.**

- **Section 5.1 (Infrastructure Layer)**: Trim. The "latency hiding" argument (1/N of cold-start on critical path) is theoretical and unmeasured. Either measure it or remove the specific claim. The framework-layer framing is fine as a brief positioning paragraph but does not warrant a full subsection without integration evidence.

- **Section 5.2 (Persistent Cache vs RAG vs Message Passing)**: Keep. Table 8 (approaches comparison) is useful for positioning.

- **Section 5.3 (Novelty Comparison)**: Keep. Table 9 (feature comparison with 10 prior systems) is the paper's strongest positioning artifact. The "BQ4" column being unique to this work is a genuine differentiator.

- **Section 5.4 (Portability)**: Keep as-is. The discussion of what is portable (block pool, safetensors format, prefix matching) vs what is MLX-specific (lazy evaluation, Metal buffer management, single-thread scheduler) is informative.

- **Section 5.5 (Limitations)**: Keep and expand. Add model specificity (KV cache is model-specific; model updates invalidate all caches; RAG survives model swaps but KV caches do not). This was raised by simulated reviewer R3 and is a legitimate limitation that deserves explicit acknowledgment.

**Table 8 (Approaches Comparison)**: Keep.

**Table 9 (Novelty Comparison)**: Keep. One of the paper's best tables.

### Section 6: Related Work (~0.8 pages)

**Verdict: Keep as-is.**

The related work is thorough: 34 references across 4 subsections (KV cache management, KV cache compression, agent memory, multi-agent KV systems, edge inference). The differentiation for each cited system is specific and honest. The bib file contains 15 additional uncited entries (including Memory3, DroidSpeak, "When KV Cache Reuse Fails") that could strengthen the narrative if space permits.

### Section 7: Conclusion (~0.4 pages)

**Verdict: Keep, minor fix.**

The conclusion correctly summarizes the headline numbers. Fix "negligible estimated quality loss (<0.1 PPL, Appendix E)" -- the word "estimated" is correctly present here, unlike the abstract. Consider also adding model specificity to the future work paragraph.

### Appendix Assessment

**Appendix A (safetensors Format)**: Keep. The tensor schema is a reproducibility artifact. Other implementations can use it directly.

**Appendix B (MLX Engineering Notes)**: Keep for the camera-ready but consider cutting for page count during review. The failure-mode table is useful for practitioners but does not strengthen the scientific contribution.

**Appendix C (Benchmark Configuration)**: Keep. Essential for reproducibility. Fix the measurement count arithmetic (the "divided by 3 for median = 66" phrasing is confusing; just say "66 unique configurations, each measured 3 times = 198 total measurements").

**Appendix D (FP16 vs Q4 Memory Analysis)**: Keep but fix the intermediate calculation errors. The per-layer Q4 breakdown has two canceling arithmetic errors (data should be 8,388,608 not 16,777,216; scales+biases should be 1,048,576 not 524,288). The final answer via the ratio formula is correct (432 MB), but the intermediate "792 MB. With overhead: ~432 MB" is confusing and wrong. Either fix the intermediate steps or remove them and keep only the ratio-based calculation.

**Appendix E (Perplexity Evaluation)**: Keep. The methodology note ("this measures output-level sensitivity to Q4 noise, not layer-by-layer KV cache quantization") is honest and important. The numbers (Gemma +0.06, DeepSeek +0.07) are now filled in. The comparison with prior work is well-organized. However, this appendix would be significantly stronger with actual Q4 KV cache perplexity (not the logit proxy). The system has `QuantizedKVCache` that operates in Q4 during inference; using it for a perplexity evaluation would convert the estimate into a measurement.

**Appendix F (Staggered Arrivals)**: Keep but fix the 1.36x claim. The claim audit found this uses inconsistent metrics (absolute time for sequential vs relative time for batched). The true speedup from Agent B's perspective is ~1.04x, not 1.36x. This error must be corrected.

**Appendix G (Hardware Landscape) -- Table 10**: This was specifically flagged for assessment. **Recommendation: Cut or drastically condense.**

Table 10 (extended hardware specs) repeats Table 1 data with added price and configuration range columns. It occupies ~0.5 pages of appendix space. Problems:
- The prices add noise without signal for a scientific paper.
- The M4/M4 Pro/M4 Max/M4 Ultra lineage is Apple marketing taxonomy, not scientific content.
- No experiment is run on any device except the M4 Pro, so the extended specs are speculation about where the system could hypothetically run.
- The DGX Spark price ($3,999) and availability claims are unverified at retail.

If any hardware landscape content is retained, it should be a single paragraph noting that the design is applicable to other unified-memory devices, with a brief mention of the DGX Spark as a higher-capacity option. The full table should be cut.

**Appendix H (Detailed Figures)**: The three TikZ figures (architectural comparison, phase timeline, wiki routing) are well-executed and provide visual clarity. Keep.

**Figure 3 (Staggered Arrivals)**: Fix as noted above.

**Figure 4 (Architectural Comparison)**: Keep. The side-by-side Gemma GQA vs DeepSeek MLA diagram is informative.

**Figure 5 (Phase Timeline)**: Keep. Shows cache state transitions across the 5-phase scenario.

**Figure 6 (Wiki Routing)**: Keep. Shows the 3-phase routing protocol.

---

## 4. The Core Problem: Why This Paper Might Not Get Accepted

The fundamental gap is that this paper is a systems engineering paper submitted to a language modeling venue.

COLM 2026 values papers that advance our understanding of language models. The accepted COLM 2025 KV cache papers all had an ML insight at their core:

- **PyramidKV** discovered that attention patterns form pyramids across layers -- an observation about how LMs allocate attention.
- **SQuat** discovered that quantization hurts reasoning in structured, predictable ways -- an observation about how quantization interacts with chain-of-thought.
- **SentenceKV** discovered that attention clusters at sentence boundaries -- an observation about how LMs segment information.

This paper discovers nothing about language models. It discovers that you can persist Q4 KV caches to disk and reload them faster than re-computing from scratch. This is valuable engineering but does not deepen understanding of how LMs process, store, or retrieve information.

The paper's closest approach to an ML insight is the character-level prefix matching observation (BPE tokenization is context-dependent, so token-ID matching fails where character-level matching succeeds). But this is undeveloped -- no empirical measurement of how often it matters, no analysis of which tokenizers are most affected, no connection to broader understanding of tokenization behavior.

A secondary problem is evaluation scope. The paper is designed for 10+ agents at 8K+ context with cross-phase persistence across 10+ phases. But it is evaluated with 4 agents at 1--3K context across 5 phases (prisoner's dilemma) or 10 agents at ~4K context with 3 phases (wiki routing). The evaluation does not reach the regime where the system's full value is demonstrated. At the context lengths tested, even cold-start is tolerably fast (1--4 seconds for DeepSeek), so the speedup from persistence, while technically large (30x at 4K), does not represent a qualitative change in what is possible.

A third problem is the absence of baselines. The paper compares only against itself (cold vs warm/hot). A reader cannot determine whether this system is better than llama.cpp with `--slot-save-path`, vllm-mlx with prefix caching, or even Ollama's `keep_alive`. These are the practical alternatives a deployment engineer would consider, and the paper provides zero data for the comparison.

---

## 5. What Would Get It Much Higher (Top 5 Changes)

Ranked by expected impact on acceptance probability:

### Change 1: Add an ML Insight (Impact: +15--20% acceptance probability)

Invest one page in an empirical study of Q4 KV cache quality degradation across layers. Run the model with Q4 KV cache at each layer individually (layer $l$ uses Q4, all others use FP16) and measure per-layer perplexity impact. This would reveal:
- Which layers are most sensitive to Q4 quantization?
- Do global attention layers (Gemma's 8) degrade differently from sliding window layers (Gemma's 40)?
- Does MLA (DeepSeek) degrade differently from GQA (Gemma)?
- Is there a layer-selective strategy that could push to 2-bit for insensitive layers while keeping 4-bit for sensitive ones?

This would transform the paper from "we built a system" to "we built a system AND discovered that Q4 sensitivity varies by layer type/position, informing future quantization strategies." Even a modest finding (e.g., "the first and last 10% of layers are 3x more sensitive to Q4 quantization than middle layers") would be a genuine ML contribution.

COLM explicitly accepts papers on "efficient LMs" and "inference," but the bar is an insight, not just a system.

### Change 2: Benchmark Against vllm-mlx and llama.cpp (Impact: +10% acceptance probability)

Run vllm-mlx (the closest MLX-native system) and llama.cpp with `--slot-save-path` (the most widely deployed edge inference engine with FP16 slot persistence) on the same hardware with the same models and context lengths. Measure:
- Cold TTFT, warm TTFT for both baselines
- Memory consumption (peak Metal/VRAM usage)
- Agent capacity at 8K context

Expected outcome: llama.cpp FP16 slot persistence achieves similar warm TTFT (both are I/O-bound) but uses 3.6x more disk space and memory, limiting agent count. vllm-mlx prefix caching works within a session but loses all caches on server restart. This would clearly demonstrate the paper's unique value: Q4 persistence enables more agents AND survives restarts.

This is probably the single most impactful change for reviewer persuasion, because reviewers will ask "why not just use X?" and the paper currently has no answer backed by data.

### Change 3: End-to-End Q4 KV Cache Perplexity (Impact: +8% acceptance probability)

Replace the logit-level proxy with actual Q4 KV cache perplexity. The system already has `QuantizedKVCache` that operates in Q4 during inference. Run WikiText-2 evaluation using this cache at 4K and 8K context windows (not the current 512-token windows). Report the measured perplexity delta, which should be close to the prior literature's <0.1 figure.

This converts "estimated <0.1 PPL" into "measured 0.08 PPL" (or whatever the actual number is). The difference in reviewer confidence is enormous. The logit proxy is an appropriate screening method during development, but a submission should measure the actual mechanism.

### Change 4: Scale the Multi-Agent Evaluation (Impact: +7% acceptance probability)

Design a 10-phase scenario with 8+ agents where context grows to 16K+ by the final phases. The system's value proposition is strongest at long context (130x speedup at 32K vs 30x at 4K). The current evaluation never reaches the regime where cold-start is truly painful (>30 seconds). A scenario where Phase 10 would take 90+ seconds cold-start but completes in <2 seconds with warm cache is dramatically more compelling than the current Phase 5 at 1.9x.

Add a task-completion metric: inject a fact in Phase 1, test recall in Phase 10. Show that persistent cache and re-prefill produce identical recall. This would validate the "working memory" claim with evidence rather than assertion.

### Change 5: Add a Third Model (Impact: +5% acceptance probability)

Add Llama 3.1 8B (or Llama 3 8B Instruct) as a third model. This is standard GQA with different head dimensions than Gemma, and is the most widely used baseline model in the field. If the system handles Llama 3 without modification, it demonstrates that ModelCacheSpec truly generalizes. If it requires code changes, the paper should report what changed and why.

Three models across two attention mechanisms (GQA with two different configurations + MLA) is substantially more convincing than two models. The unit test suite (792 tests) reportedly covers Llama 3, so the system already works -- it just needs benchmark numbers.

---

## 6. Comparison with Accepted COLM Papers

### PyramidKV (COLM 2025)

**ML Insight**: Attention patterns form pyramids -- lower layers attend broadly, upper layers attend narrowly. This observation motivated a dynamic KV selection algorithm that allocates more cache budget to lower layers.

**Comparison**: PyramidKV starts with an ML observation (attention pyramids) and builds a system around it. This paper starts with a system need (multi-agent cold-start latency) and builds engineering around it. PyramidKV tells us something about how LMs allocate attention. This paper tells us how to make cache reloading fast. The insight depth gap is significant.

### SQuat (COLM 2025)

**ML Insight**: KV cache quantization hurts reasoning in structured ways -- it disproportionately affects chain-of-thought reasoning steps, not just overall perplexity. The interaction between quantization error and multi-step reasoning is non-obvious.

**Comparison**: SQuat investigates the quality impact of KV quantization deeply, revealing structured failure modes. This paper uses Q4 quantization but measures quality only via a logit proxy. If this paper included a per-layer sensitivity analysis (Change 1 above), it would approach SQuat's depth on the quality dimension while having a stronger systems contribution.

### SentenceKV (COLM 2025)

**ML Insight**: LMs' attention patterns cluster at sentence boundaries, not token boundaries. This observation enables sentence-level KV cache management that is both more efficient and more semantically meaningful.

**Comparison**: SentenceKV's insight is about how LMs structure information internally. This paper's prefix matching operates at the character level, not the attention level. There is no analysis of whether agents' attention patterns change across conversation phases, whether Q4 quantization affects attention distribution, or whether the block-pool granularity (256 tokens) aligns with any meaningful structure in the attention patterns.

### E2-RAG (COLM 2025)

**ML Insight**: Coupling KV cache with retrieval enables end-to-end optimization that neither caching nor retrieval alone can achieve.

**Comparison**: This paper explicitly positions KV cache persistence as complementary to RAG (Table 8, Section 5.2). It does not couple them. An extension that combines persistent agent cache with RAG for external knowledge -- and shows the combined latency profile -- would be a stronger contribution.

### Summary

The accepted COLM KV cache papers follow a pattern: **ML observation leads to algorithmic insight leads to system design.** This paper follows a different pattern: **deployment pain point leads to system design leads to performance measurement.** The latter is valuable but does not match COLM's demonstrated preference for understanding-first contributions. The paper is a better fit for MLSys, EuroSys, MobiSys, or ASPLOS.

---

## 7. Strengths (What Reviewers Will Like)

**S1. The cold-start problem is precisely quantified and well-motivated.** "Five agents, each holding 4,096 tokens. Server restarts. 77 seconds of dead time." This is a concrete, memorable framing that immediately establishes the problem's significance. The connection to position bias (citing "Lost in the Middle") provides ML justification for per-agent isolation.

**S2. BatchQuantizedKVCache is a genuine first.** No prior system (vLLM, SGLang, vllm-mlx, LMCache, or any cited work) provides batched inference over quantized KV caches. Table 9's "BQ4" column is uniquely checked for this work. This is a real systems contribution.

**S3. The dual-architecture evaluation demonstrates non-trivial generalization.** Testing on Gemma 3 12B (dense GQA, symmetric K/V, hybrid sliding window + global attention) AND DeepSeek-Coder-V2-Lite 16B (MoE, MLA, asymmetric K=192/V=128) requires handling fundamentally different cache structures. The `v_head_dim` field added to ModelCacheSpec for MLA detection shows the authors encountered and solved a real edge case.

**S4. The measurement methodology is rigorous.** 198 measurements per model, thermal-aware cooldown (monitoring CPU junction temperature), median-of-3 reporting, explicit quality checks. The claim audit verified 91/118 claims match source data exactly, with 13 minor rounding discrepancies and zero fabrications.

**S5. The paper is unusually transparent about limitations.** Five specific limitations in Section 5.5, honest acknowledgment that the perplexity evaluation is a proxy, explicit statement that the ablation uses existing table numbers. This level of candor is refreshing and builds trust.

**S6. The novelty comparison table (Table 9) is convincing.** Ten prior systems across five capability dimensions, with specific categorization of each. This is the kind of positioning evidence that helps reviewers quickly assess where the contribution sits.

**S7. The safetensors format specification (Appendix A) is a standalone contribution.** The tensor naming convention, bfloat16 handling for Gemma's scales, and the complete schema are immediately usable by other implementations. This is a reproducibility artifact of genuine value.

**S8. The Q4 memory analysis is mathematically rigorous.** The 0.281 ratio derivation is correct, the capacity calculations follow from it, and the paper shows both models' capacity at four context lengths. The FP16 vs Q4 comparison (12 vs 3 agents at 8K) is the clearest quantification of Q4's value.

---

## 8. Line-Level Issues

### Factual Errors

1. **Figure 1 annotation**: "81.6x TTFT (hot)" -- current Table 3 data yields 81.4x for Gemma 16K hot (71132/874 = 81.37). Update to 81.4x or use a different context length's number.

2. **Figure 2 caption**: "40--130x at 32K tokens" -- the actual range at 32K is 69x (DeepSeek warm, 48258/697) to 130x (Gemma hot, 165189/1276). The minimum is not 40x at any 32K configuration.

3. **Figure 2 caption**: "sub-second reload regardless of context length" -- Gemma warm at 32K is 1621ms (1.6 seconds). Change to "sub-second reload up to 16K context" or "near-second reload across context lengths."

4. **Figure 2 caption**: "(27 vs 48)" is correct for layer counts. Earlier reviews flagged "46" but the current text correctly says 48.

5. **Appendix D**: The per-layer Q4 intermediate calculation says "data = 16,777,216 bytes" but the correct value is 8,388,608 bytes (the /2 for 4-bit packing was omitted). The "scales+biases = 524,288 bytes" should be 1,048,576 bytes (both scales AND biases). The two errors cancel, and the final answer (432 MB via the ratio) is correct, but the intermediate "792 MB. With overhead: ~432 MB" makes no sense -- 432 is less than 792, not "with overhead."

6. **Appendix F (Staggered Arrivals)**: The "1.36x" staggered speedup claim for DeepSeek uses inconsistent metrics. It compares sequential Agent B's absolute time from experiment start (8.7s) against batched Agent B's time from processing start (6.4s). Using consistent metrics, the actual speedup is ~1.04x. This must be corrected.

7. **Section 4.3**: "DeepSeek is consistently ~2.9x faster" -- the actual range is 2.1x (16K) to 2.9x (1K). At 16K the ratio is only 2.1x, which is not "approximately 3x."

8. **Abstract**: "4x more agent contexts" -- the ratio is 3.6x at 8K (the specific number cited in the conclusion: 12 vs 3). The "4x" figure only holds at 16K and 32K due to floor rounding.

### Unclear Prose

9. **Section 2.2, line 98**: "24 GB - 6.8 GB weights - 7 GB OS/system ~ 10.2 GB for KV caches." But Appendix D uses a 15.2 GB budget (24 - 6.8 - 2), and the evidence verification document uses 15.2 GB. There are two different memory budgets in the paper. Which is correct? If OS overhead is 7 GB, the budget is 10.2 GB and Table 2's capacity numbers are too high. If OS overhead is 2 GB, the budget is 15.2 GB and Section 2.2 is wrong. These need to be reconciled.

   **Resolution upon closer inspection**: Section 2.2 says "24 GB - 6.8 GB weights - 7 GB OS/system ~ 10.2 GB" while Table 2 shows capacity numbers consistent with 15.2 GB budget (matching the evidence doc's 24 - 6.8 - 2 = 15.2). The discrepancy appears to be that Section 2.2 uses a conservative OS estimate (7 GB) while the capacity analysis uses an aggressive one (2 GB). The paper should use one consistent figure. The aggressive 2 GB is closer to macOS's marginal inference-time overhead (not total OS memory), but should be explicitly stated as such.

10. **Section 3.3**: "An 80% common-prefix threshold determines reuse eligibility." What happens at 79%? The cache is entirely discarded? This seems like a cliff edge. Is there partial reuse for lower overlap percentages? The DIVERGE match type is mentioned but not explained.

11. **Section 3.4**: "Two agents' decode steps execute as one Metal kernel dispatch." This implies the GPU processes both agents simultaneously. But earlier the paper says operations are time-sliced on a single thread and batch-2 is split into two batch-1 calls via mx.compile. These statements appear contradictory. The resolution is that the merge/update/extract operations create a unified batch tensor that the GPU processes as one dispatch, even though the compilation path splits the operations. This should be clearer.

### Redundancies

12. The Q4 ratio (0.281) is derived in Section 3.2 (line 135), repeated in Appendix D, and the resulting capacity numbers appear in Table 2 (Section 3.2), the Appendix D table, and the ablation table (Table 5). The three presentations of the same calculation could be consolidated.

13. The "5 agents, 4K tokens, 77 seconds" framing appears in the abstract, Section 1 (line 46), and can be reconstructed from Section 2.3. Two appearances are fine; three is redundant.

14. The GDPR/HIPAA mention (Section 2.2, line 100) and the "privacy" motivation never appear again. Either develop it into a coherent thread (connect to PROMPTPEEK's finding that shared caches enable prompt reconstruction, motivating per-agent isolation for privacy) or remove it.

### Missing Content

15. **Model specificity limitation**: KV cache is model-specific. A Gemma 3 cache cannot be used by Gemma 4 or by DeepSeek. Model updates invalidate all cached state. This is not mentioned in the limitations section. RAG text chunks survive model swaps; KV caches do not. This tradeoff deserves explicit discussion.

16. **Working memory quality**: The paper claims KV cache serves as "working memory" but provides zero quality evidence for cross-phase information retention. Does an agent in Phase 5 actually recall Phase 1 information? Both persistent-cache and re-prefill produce the same context, so the answer should be "yes, identically." But this should be demonstrated, not assumed. Even a small table showing bit-exact or token-exact output equivalence between warm-cache and cold-start would suffice.

17. **DeepSeek wiki routing quality**: 3/10 Phase 1 quality and 4/10 Phase 2 quality for DeepSeek is alarming. The paper should explicitly state whether this is a model limitation (DeepSeek is a code model, not a knowledge-retrieval model) or a caching artifact. Running the same queries without caching and showing the same low scores would definitively answer this.

---

## Summary

This is a well-engineered systems paper that fills a genuine gap in the multi-agent inference landscape. The BatchQuantizedKVCache, the ModelCacheSpec abstraction, the safetensors persistence format, and the cross-phase context injection are real contributions with no prior equivalent. The 30--130x TTFT speedups are measured, verified, and significant.

The paper's primary weakness for COLM is the absence of an ML insight. It is fundamentally a systems integration paper: it combines known techniques (Q4 quantization, disk persistence, batched inference, prefix matching) for a specific hardware target (Apple Silicon) to solve a specific deployment problem (multi-agent cold-start latency). This is valuable work, but COLM's demonstrated acceptance pattern favors papers that teach us something new about how language models work.

The secondary weaknesses -- no baseline comparisons, logit-proxy perplexity, modest evaluation scale, single hardware device -- are all fixable within a revision cycle. The five changes recommended in Section 5, if executed, would collectively raise the acceptance probability from ~15--20% to ~40--50%. The single most impactful change would be adding a per-layer Q4 sensitivity analysis (Change 1), which would introduce the ML insight dimension currently missing.

If COLM is the priority venue, the authors should invest in Changes 1--3 before the March 31 deadline. If the March 31 deadline is too tight for the ML insight work, the paper is a strong fit for MLSys 2026 (systems contribution emphasis) or MobiSys 2026 (edge/mobile systems emphasis) as-is, with the baseline comparisons and end-to-end perplexity added.
