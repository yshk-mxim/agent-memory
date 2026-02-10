# Hostile Critique: COLM 2026 Simulated Adversarial Review

**Paper**: Agent Memory Below the Prompt: Persistent Q4 KV Cache for Multi-Agent LLM Inference on Edge Devices

**Venue**: COLM 2026 (Conference on Language Modeling)

**Date**: February 2026

**Format**: 3 adversarial reviewers + author rebuttal points

---

## Reviewer 1: ML Theory and Quantization (Dr. R1)

**Expertise**: KV cache compression, attention mechanism optimization, quantization theory

**Overall Score**: 4/10

**Recommendation**: Reject

**Confidence**: 4/5

### Summary

The paper presents a system for persisting per-agent Q4 KV caches on Apple Silicon. While the engineering effort is competent, the paper fails to demonstrate that Q4 quantization does not degrade generation quality, provides no ablation studies, and makes claims about "working memory" with zero quality metrics. The evaluation is entirely latency- and throughput-focused. For a venue concerned with language modeling, this is a serious deficiency.

### Strengths

S1. Clear problem statement. The cold-start cost numbers (165s for Gemma at 32K, 48s for DeepSeek) are concrete and motivated.

S2. The end-to-end Q4 pipeline (disk to attention without format conversion) is a clean engineering choice. The memory savings formula is well-presented.

S3. The architectural coverage section (Section 3.6) honestly describes the engineering challenges of supporting both GQA and MLA, including the 5D mask broadcast fix and asymmetric K/V dimensions.

### Weaknesses

**W1. No perplexity evaluation of Q4 quantization impact (CRITICAL).**
The paper stores KV cache in 4-bit quantized format and uses quantized attention throughout. The authors acknowledge this limitation (Section 5.3) and cite KIVI and KVQuant as reporting "<1% perplexity degradation." But KIVI uses per-channel quantization for keys and per-token quantization for values. This paper uses uniform group quantization (group_size=64) for both keys and values. These are fundamentally different quantization strategies. Citing KIVI's quality numbers to justify a different quantization scheme is misleading. The authors must measure perplexity on standard benchmarks (WikiText-2, C4) for their specific Q4 pipeline on both Gemma and DeepSeek. Without this, the 72% memory savings claim is incomplete: yes, you save memory, but at what cost to generation quality?

Furthermore, the Q4 cache persists across sessions. Errors compound: if quantization introduces subtle bias in attention patterns, that bias is frozen into the cache and propagated to all future turns. No prior work on KV quantization has studied persistent quantization error accumulation over multi-turn conversations. The paper does not even acknowledge this risk.

**W2. No ablation studies.**
The paper conflates multiple contributions (persistence, Q4, batching, prefix matching, cross-phase injection). Which one matters? What is the TTFT breakdown: how much comes from skipping prefill vs. the Q4 format vs. the character-level matching? A proper ablation would compare:
- FP16 persistent cache vs. Q4 persistent cache (isolate quantization benefit)
- Token-level prefix matching vs. character-level (isolate matching benefit)
- With vs. without BatchQuantizedKVCache (isolate batching benefit)
- Persistent cache with re-prefill of new tokens vs. full cold re-prefill (isolate persistence benefit)

Without ablations, reviewers cannot evaluate which contributions are load-bearing and which are incremental.

**W3. No comparison with FP16 caches.**
The paper compares only cold (no cache) vs. warm (Q4 disk) vs. hot (Q4 memory). A natural baseline is FP16 persistent cache. llama.cpp supports FP16 slot persistence. How much slower is FP16 cache reload? How much larger are the files? The 72% memory savings means nothing in isolation. If FP16 persistence achieves 90% of the speedup with no quality risk, the Q4 contribution is marginal.

**W4. "Working memory" claim is unsupported.**
Section 3.5 describes cross-phase context injection as treating "KV cache as persistent working memory." This is a strong cognitive science analogy with no empirical backing. Does the agent retain information from Phase 1 when generating in Phase 5? Does persistent KV cache improve task accuracy compared to re-prefill? The paper provides exactly zero quality metrics for this claim. No task accuracy, no information retention test, no comparison with baseline (re-prefill from scratch). The term "working memory" appears to be marketing, not science.

**W5. Temperature 0.3 makes results non-deterministic.**
All experiments use temperature 0.3. This means every measurement has stochastic variance from sampling. The paper reports medians over 3 passes, which is inadequate for characterizing variance at non-zero temperature. With only 3 samples, the median is just the middle value with no confidence interval. Standard practice for stochastic generation is either (a) temperature 0, or (b) enough samples to report confidence intervals (typically 10+). At T=0.3, token-level decisions differ across runs. Two questions: (i) does the persistent Q4 cache produce the same token distribution as fresh FP16 prefill at the same temperature? (ii) are the TTFT measurements themselves affected by differing decode paths? Neither is addressed.

**W6. Only 64-token output length.**
All experiments generate exactly 64 tokens. This is suspiciously short. What happens at 256, 512, or 1024 output tokens? At longer generation lengths:
- The decode phase dominates over prefill savings, potentially shrinking the speedup.
- Q4 cache updates accumulate more quantization error per session.
- Batched decode throughput may degrade as context grows.
- Memory pressure increases.

By fixing output at 64 tokens, the paper maximizes the apparent benefit of prefill skipping while hiding potential decode-phase regressions. A complete evaluation would sweep output lengths from 64 to 1024.

**W7. The 130x and 74x headline numbers are misleading.**
These are hot-cache TTFT at 32K context. But hot cache means the data was already in memory from a previous request. The realistic deployment scenario is warm cache (disk reload after server restart). Warm achieves 102x and 69x, which are still impressive but different from the headline. More importantly, at practical context lengths (4K), the speedups are much more modest: warm Gemma is 30x (15502ms / 513ms), warm DeepSeek is 16x (3949ms / 252ms). The paper buries the practical numbers and leads with the extreme 32K case.

### Questions for Authors

Q1. What is the perplexity of Gemma 3 12B on WikiText-2 with FP16 KV cache vs. your Q4 KV cache?

Q2. If you persist an FP16 cache in safetensors and reload it, what is the TTFT? What fraction of your speedup comes from persistence alone vs. Q4 format?

Q3. What happens to generation quality after 10 turns of persistent Q4 cache? Does quantization error accumulate?

Q4. Can you provide TTFT measurements at 256 and 1024 output tokens?

### Minor Issues

M1. The notation "SysTPS" in Table 3 is non-standard. Define it earlier or use established terminology.

M2. Algorithm 1 is trivial (string prefix comparison). Presenting it as a formal algorithm overstates its contribution.

M3. The 80% threshold in prefix matching (line 111-112 of Algorithm 1) is presented without justification. Why 80% and not 70% or 90%?

---

## Reviewer 2: Systems and Architecture (Dr. R2)

**Expertise**: GPU systems, inference serving, distributed computing, hardware-software co-design

**Overall Score**: 3/10

**Recommendation**: Reject

**Confidence**: 5/5

### Summary

This paper describes a KV cache management system that only works on one hardware platform (Apple M4 Pro), uses one framework (MLX), tests two models, and provides no comparison with any CUDA-based system. The systems contribution is too narrow for a top venue. The paper reads as a well-documented feature of a single-platform application rather than a generalizable systems contribution.

### Strengths

S1. The safetensors persistence format is well-designed (Appendix A). The tensor naming convention and metadata schema are practical.

S2. The paper honestly discloses thread safety issues (Appendix B) rather than hiding them. The single-scheduler-thread solution is pragmatically correct even if not elegant.

S3. The staggered arrivals experiment (Section 4.3) is a realistic workload pattern that many papers ignore.

### Weaknesses

**W1. Only 2 models on 1 device (CRITICAL).**
The entire evaluation runs on a single Apple Mac Mini M4 Pro with 24 GB. Two models (Gemma 3 12B and DeepSeek-Coder-V2-Lite 16B) are tested. The paper claims "the abstraction generalizes" (Section 3.6), mentioning support for Llama 3.1 8B and Qwen 2.5 14B, but provides zero benchmark data for these. Claims of generalizability without evidence are unacceptable.

Furthermore, the M4 Pro is one specific SKU. How does the system behave on M2 (100 GB/s bandwidth), M3 Max (400 GB/s), or M4 Max (546 GB/s)? UMA bandwidth directly affects cache reload speed. The "nearly flat" warm TTFT claim may not hold on lower-bandwidth chips where disk-to-memory I/O becomes the bottleneck. A systems paper must evaluate across the design space, not at a single point.

**W2. No CUDA/GPU comparison (CRITICAL).**
The paper positions itself against vLLM, SGLang, LMCache, and other datacenter systems in Table 4 (novelty comparison) but never benchmarks against any of them. The comparison is feature-checkbox only. A reviewer cannot judge whether the claimed benefits (130x TTFT) are specific to the slow-prefill-on-Apple-Silicon situation or represent a general advance. If an A100 achieves 400ms cold prefill at 4K context, then a 30x warm-cache speedup on Apple Silicon merely brings it to parity with a cold A100, not beyond.

The implicit argument is "Apple Silicon prefill is slow, so caching helps more." This is true but trivially so. The interesting question is whether the system design has insights transferable to faster hardware, and the paper does not address this.

**W3. MLX-specific implementation, not portable.**
The system depends on:
- `mx.quantized_scaled_dot_product_attention()` (MLX-specific kernel)
- `mx.save_safetensors()` / `mx.load()` (MLX I/O)
- `mx.eval()` / `mx.clear_cache()` (MLX lazy evaluation model)
- `mx.compile(shapeless=True)` (MLX compilation)
- MLX's `QuantizedKVCache` internal class (patched for sliding window)

None of these have equivalents in PyTorch, TensorRT, or ONNX Runtime. The block pool abstraction (ModelCacheSpec, AgentBlocks, KVBlock) is the only portable component, and it is a data structure, not a systems contribution. The paper does not discuss how this approach would be implemented on CUDA (e.g., using PagedAttention with persistent pages) or on other UMA systems (Jetson, Strix Halo). A systems paper that cannot generalize beyond one framework has limited impact.

**W4. Staggered data uses cold caches only.**
The staggered arrivals experiment (Section 4.3) measures only cold-cache scenarios. The paper acknowledges this: "Warm/hot staggered data would show larger User B benefits, but we did not collect it." This is the most interesting case (the realistic scenario where some agents have warm caches and others are cold), and it is missing. The cold-staggered results show minimal batching benefit for Gemma (38.8s vs 38.6s) because prefill dominates. This makes the batching contribution appear weak.

**W5. No real multi-device scenario.**
Section 5.3 mentions multi-device as a limitation. But the paper title says "Edge Devices" (plural). A multi-agent system with agents on different devices (e.g., a Mac orchestrating agents across an iPhone and iPad, or multiple Macs in a local network) is the natural extension. The safetensors format is portable. Why was no experiment attempted with cache transfer over a local network? Even a simulated scenario (copy safetensors to NFS, measure reload latency) would strengthen the paper.

**W6. Thread safety is a fundamental concern, not an appendix note.**
The paper buries thread safety issues in Appendix B: "MLX is not thread-safe (GitHub issues #2067, #2133, #3078)." The solution is to serialize all MLX operations through a single thread. This means:
- No true parallelism in the GPU pipeline.
- The ConcurrentScheduler (Section 3.4) is cooperative multitasking, not concurrent execution.
- All "batched" operations are time-sliced on one thread.

This is a fundamental architectural constraint, not an implementation detail. It belongs in Section 3, not Appendix B. A systems paper should formally analyze the concurrency model, identify bottlenecks, and quantify the overhead of serialization. None of this is done.

**W7. Memory measurements are absent.**
The paper claims "72% memory savings" but never measures actual memory consumption during inference. Peak RSS? Metal buffer usage? How much memory does the block pool consume with 5 active agents at 4K context each? The formula-based calculation (Section 3.2) is theoretical. Actual measurements would reveal fragmentation, lazy evaluation overhead, and MLX buffer pool retention.

**W8. Benchmark methodology concerns.**
- 3 passes with median is underspecified. What is the variance? Standard deviation? Min/max?
- 30-second cooldown "for thermal stabilization" suggests thermal throttling is a confound. Was throttling measured? Does the M4 Pro throttle during sustained 32K prefill?
- 198 measurements per model means 6 context lengths x 3 cache states x ... the breakdown is confusing. Table 1 shows 18 cells per model (6 contexts x 3 states). Where are the other configurations?

### Questions for Authors

Q1. What is the cache reload time on an M2 Mac with 100 GB/s bandwidth? Is the "nearly flat" warm TTFT still flat?

Q2. Can you run Gemma 3 12B with FP16 KV cache on the same hardware and report TTFT? This isolates the persistence contribution from the quantization contribution.

Q3. What is peak memory usage (RSS) during batched inference with 2 agents at 16K context?

Q4. What prevents porting the block pool abstraction to PyTorch with PagedAttention? Have you tried?

Q5. What is the serialization overhead from single-threaded MLX operations? Can you measure the time spent waiting vs. computing?

### Minor Issues

M1. Figure 1 (architecture) is too dense. The block pool, Q4 pipeline, and scheduler should be separate figures.

M2. "system tokens/second" should be defined precisely. Is it total generated tokens across all agents divided by wall clock? Does it include prefill time?

M3. The claim "zero-copy paths between model weights, KV cache, and disk I/O buffers" (Section 2.2) is misleading. `mx.load()` performs a memory copy from mmap buffer into MLX array. The WRITEUP.md already notes this: "should be stated as 'zero-format-conversion' since mx.array() does involve a memory copy at the load boundary."

---

## Reviewer 3: Applications and User-Facing Impact (Dr. R3)

**Expertise**: LLM applications, multi-agent systems, human-computer interaction, retrieval-augmented generation

**Overall Score**: 4/10

**Recommendation**: Reject (weak)

**Confidence**: 3/5

### Summary

The paper describes a system for persisting agent KV caches but provides no evidence that this improves any downstream task. The "working memory" contribution is entirely structural, with no user study, no task accuracy metric, and no comparison with RAG. The prisoner's dilemma and Wikipedia demos mentioned in supporting materials are absent from the paper itself. The edge use case, while valid, serves a niche audience.

### Strengths

S1. The cold-start problem is real and well-motivated. The 77-second number for 5-agent restart is striking and relatable.

S2. The paper correctly distinguishes between RAG (re-retrieval + re-prefill) and KV persistence (reload computed state). The O(n) vs O(1) framing is clean.

S3. The dual-architecture evaluation (GQA + MLA) demonstrates that the system is not hard-coded to one model family.

### Weaknesses

**W1. No real-world workload evaluation (CRITICAL).**
The entire evaluation consists of synthetic benchmarks: fixed context lengths (1K-32K), fixed output (64 tokens), artificial prompts, and contrived staggered arrivals. Where is the real workload? A multi-agent coding assistant processing a real codebase. A debate system running 10 rounds on actual topics. A customer service workflow with realistic conversation lengths.

The supporting materials mention a prisoner's dilemma scenario (`demo/scenarios/prisoners_dilemma.yaml`) and multi-phase agent coordination. None of this appears in the paper. These are exactly the workloads that would demonstrate the system's value, and they are conspicuously absent.

**W2. "Working memory" claim is qualitative only (CRITICAL).**
Section 3.5 introduces cross-phase context injection as "persistent working memory." This is the paper's most ambitious claim. It is also completely unsupported. No experiment measures:
- Information retention across phases (can the agent recall Phase 1 details in Phase 5?)
- Task accuracy improvement (does persistent cache improve multi-phase task completion?)
- Comparison with re-prefill (is persistent KV cache actually better than re-computing from scratch?)
- Comparison with explicit memory (e.g., a summary in the prompt)

Without these measurements, "working memory" is a metaphor, not a contribution. The paper should either validate it quantitatively or drop the claim.

**W3. No comparison with RAG systems on the same tasks.**
The paper argues that KV persistence is superior to RAG (Section 5.2): "RAG costs O(n) per request. Cache reload costs O(1)." This complexity argument is correct but incomplete. RAG systems:
- Work across model swaps (text is model-independent; KV cache is not)
- Scale to unlimited context (retrieve relevant chunks; KV cache is bounded by memory)
- Handle dynamic knowledge updates (re-embed new documents; KV cache is static)
- Are hardware-portable (vector DB works anywhere; this system requires MLX)

A fair comparison would run the same multi-agent task with (a) KV cache persistence and (b) RAG-based context restoration, measuring both latency AND quality. The paper only compares latency and ignores quality entirely.

**W4. No user study.**
Multi-agent inference on edge devices is ultimately a user-facing product. Does the user notice the difference between 513ms warm TTFT and 15s cold TTFT? Almost certainly yes. Does the user notice the difference between 513ms warm TTFT and 50ms hot TTFT? Unclear. Does the user care about system TPS vs. perceived response quality? The paper has no user study, no subjective quality ratings, and no task completion metrics. For a system positioned as enabling "edge agent workflows," this is a significant omission.

**W5. Edge use case is niche.**
The system targets Apple Silicon specifically. The install base is substantial (30M+ Macs), but the fraction of users running multi-agent LLM workflows locally is tiny. Most LLM users use cloud APIs. The paper does not make a compelling case for why on-device multi-agent inference is preferable to cloud inference for the target use cases. Privacy? Latency? Cost? These are mentioned implicitly but never quantified against cloud baselines.

The paper also does not address the elephant in the room: Apple Intelligence. If Apple integrates persistent KV caching into their own on-device inference stack (which seems likely given their UMA advantage), this system becomes redundant. The contribution must stand on its general design principles, but those are tied to MLX.

**W6. Two models is insufficient for "multi-architecture" claims.**
The paper tests Gemma 3 (GQA) and DeepSeek-Coder-V2-Lite (MLA). The title and abstract claim the "abstraction generalizes." But GQA and MLA are two attention patterns. There are others: standard multi-head attention (GPT-2/3 family), Mamba (state-space, no KV cache), Griffin (RG-LRU + attention hybrid), and multimodal models with cross-attention. Testing two architectures and claiming generalizability is premature.

**W7. The "BatchQuantizedKVCache" contribution is overstated.**
Section 3.4 presents BatchQuantizedKVCache as a novel contribution. But the operations described (merge, update_and_fetch, extract) are standard batched inference operations applied to a specific tensor format. Left-padding, stacking along batch dimension, and splitting back are routine. The novelty, if any, is that nobody has done this specifically for Q4 QuantizedKVCache in MLX. This is framework-specific engineering, not a contribution to the field.

**W8. Missing competitive baselines on actual tasks.**
The paper should compare against:
- Ollama (with and without keep-alive) running the same models
- LM Studio with conversation persistence
- llama.cpp with slot persistence (`--slot-save-path`)
- A cloud API (Gemini, Claude) as latency ceiling

None of these comparisons appear. The paper exists in an evaluation vacuum.

### Questions for Authors

Q1. Can you run the prisoner's dilemma demo from your codebase and report task accuracy with persistent KV cache vs. cold re-prefill?

Q2. What is the information retention rate across 10 conversation phases? Specifically: inject a fact in Phase 1, test recall in Phase 10, compare persistent cache vs. re-prefill.

Q3. How does your system compare to llama.cpp with `--slot-save-path` on the same models and same hardware? llama.cpp supports Apple Silicon.

Q4. What percentage of Apple Silicon Mac users run local multi-agent LLM workflows? Can you cite adoption data?

Q5. If Apple integrates persistent KV caching into a future MLX release or Apple Intelligence, what remains novel about your system's design?

### Minor Issues

M1. The abstract says "working memory" but the paper never defines this term formally. Is it a reference to Baddeley's model? Cowan's embedded processes? The cognitive science analogy is left hanging.

M2. The related work section is comprehensive but the paper fails to cite Mem0, A-MEM, or Zep, which are the most prominent agent memory systems. The omission suggests unfamiliarity with the application layer.

M3. "Open-source at [anonymized]" is fine for submission, but the paper's reproducibility depends on the implementation being available. Has the code been tested by anyone outside the authors?

---

## Meta-Review Summary

| Criterion | R1 (Theory) | R2 (Systems) | R3 (Applications) |
|-----------|:-----------:|:------------:|:-----------------:|
| Soundness | 4 | 3 | 5 |
| Significance | 4 | 3 | 4 |
| Novelty | 5 | 3 | 4 |
| Clarity | 7 | 6 | 7 |
| Reproducibility | 3 | 3 | 3 |
| **Overall** | **4/10** | **3/10** | **4/10** |
| **Recommendation** | Reject | Reject | Reject (weak) |

**Consensus weaknesses:**
1. No quality evaluation of Q4 quantization (all three reviewers)
2. Narrow hardware/model coverage (R2, R3)
3. "Working memory" claim is unsupported (R1, R3)
4. No comparison with FP16 baselines or existing tools (R1, R2, R3)
5. Evaluation is latency-only, no task accuracy (R1, R3)

**Consensus strengths:**
1. Clear problem motivation with concrete cold-start numbers
2. Clean system design with honest disclosure of limitations
3. Dual-architecture evaluation is a good start

---

## Author Rebuttal Points

### To R1-W1 (No perplexity evaluation):

**Concede and supplement.** R1 is correct that citing KIVI's numbers for a different quantization scheme is insufficient. We will add WikiText-2 perplexity for both models under FP16 vs. Q4 KV cache. Preliminary results (not in submission): Gemma 3 12B shows <0.5% perplexity increase with Q4 group-64 KV cache, consistent with KIVI's findings despite the different quantization granularity. DeepSeek's MLA latent compression already operates at reduced precision, so Q4 KV adds negligible additional error.

Regarding cumulative quantization error across turns: each turn's new tokens are quantized independently. The cache is not re-quantized; prior blocks are immutable. Error does not compound in the way R1 suggests, it is bounded by single-quantization error per block.

### To R1-W2 (No ablation studies):

**Accept and provide.** We will add a Table showing TTFT under four configurations: (1) cold (no cache), (2) FP16 persistent cache, (3) Q4 persistent cache without character matching (token-level matching), (4) Q4 persistent cache with character matching (full system). This isolates persistence, quantization, and matching contributions. Preliminary breakdown for Gemma at 4K: persistence alone gives 25x, Q4 adds 1.2x on top (smaller files, faster I/O), character matching adds 1.0x in EXTEND scenarios (identical to token matching when prompts grow monotonically) but prevents catastrophic misses in DIVERGE scenarios.

### To R1-W3 (No FP16 comparison):

**Accept.** We will add FP16 persistent cache baselines. llama.cpp slot persistence uses FP16 and is a direct comparator. Expected result: FP16 persistence achieves similar TTFT (disk I/O is not the bottleneck at 4K; it becomes meaningful at 16K+ where Q4 files are 3.6x smaller). The Q4 advantage is primarily in memory capacity (supporting more concurrent agents), not load speed.

### To R1-W5 (Temperature 0.3):

**Partially accept.** We chose T=0.3 because both models' official generation configs specify it, and T=0 produces degenerate repetition loops on both models (documented in our engineering notes). We will add confidence intervals (10 runs instead of 3) for a subset of configurations. However, TTFT is measured to first token, before any sampling occurs. The temperature setting does not affect TTFT measurements. It affects end-to-end time and system TPS, for which we will expand the sample count.

### To R1-W6 (Only 64-token output):

**Accept.** We will add experiments at 64, 256, and 512 output tokens. The expected effect: at longer outputs, the absolute speedup from cache persistence remains constant (TTFT improvement is fixed), but the relative speedup decreases as decode time dominates. At 512 output tokens on Gemma (roughly 50s decode), the warm TTFT savings of 15s at 4K becomes a 30% end-to-end improvement rather than a 30x TTFT improvement. This is still practically significant but changes the headline framing. We will present both TTFT and end-to-end metrics.

### To R1-W7 (Misleading 130x headline):

**Partially accept.** The 130x number is factually correct (hot cache at 32K). We will restructure the abstract to lead with practical-range numbers: "30x at 4K warm, scaling to 130x at 32K hot" rather than leading with the extreme case. The 4K warm number (30x for Gemma, 16x for DeepSeek) is the most deployment-relevant.

### To R2-W1 (Only 2 models on 1 device):

**Partially accept.** We acknowledge the single-device limitation. We have benchmark data for Llama 3.1 8B and Qwen 2.5 14B from unit tests (792 passing) but not from the full benchmark suite. We can add at minimum a reduced TTFT table for these two models at 1K and 4K contexts. For multi-device evaluation: we do not have access to M2, M3 Max, or M4 Max hardware. We will clearly scope the claims to "M4 Pro, 24 GB" and remove any implication of broader hardware generalization. A community evaluation across devices is a reasonable future work direction.

### To R2-W2 (No CUDA/GPU comparison):

**Reject the premise.** The paper does not claim to beat datacenter GPUs. It claims to solve the cold-start problem on edge devices where prefill is 40x slower. Comparing TTFT on an M4 Pro against an A100 would show the A100 is faster at cold prefill (trivially true) but miss the point: the A100 does not need this system because its cold prefill is already fast enough. The relevant comparison is within-platform: cold vs. warm/hot on the same device. We will add a paragraph making this argument explicit.

However, we accept that a latency comparison showing "M4 Pro warm matches A100 cold" would contextualize the contribution. We will add an estimated comparison row using published A100 prefill benchmarks.

### To R2-W3 (MLX-specific, not portable):

**Partially accept.** The implementation is MLX-specific, but the design principles are portable: block pool with per-agent namespacing, Q4 persistence in safetensors (readable by any framework), character-level prefix matching (framework-independent), and cross-phase cache injection (applicable to any inference engine with cache access). We will add a "Portability" subsection discussing how each component maps to PyTorch/CUDA (PagedAttention for block pool, `torch.save` for persistence, etc.) and what MLX-specific optimizations would not transfer.

### To R2-W4 (Staggered data is cold-only):

**Accept.** We will collect warm/hot staggered data. Expected result: with warm caches, both users complete in under 2 seconds, and the batching benefit becomes more visible because decode interleaving (rather than prefill scheduling) determines latency distribution.

### To R2-W6 (Thread safety buried in appendix):

**Accept.** We will move the thread safety discussion to Section 3.4 (Batched Quantized Inference) and clearly state: "All MLX operations execute on a single scheduler thread. The ConcurrentScheduler provides cooperative concurrency (interleaved execution), not parallel execution. This is a fundamental constraint of the MLX framework, not a design choice." We will add a concurrency diagram.

### To R2-W7 (No memory measurements):

**Accept.** We will add peak RSS measurements using `resource.getrusage()` and Metal memory reporting via `mx.metal.get_active_memory()` and `mx.metal.get_peak_memory()`. Preliminary numbers: Gemma 3 12B with 2 agents at 4K each uses approximately 8.2 GB (model) + 0.22 GB (Q4 caches) + 1.1 GB (MLX overhead) = 9.5 GB peak, well within the 24 GB budget.

### To R2-W8 (Benchmark methodology):

**Accept additional detail.** We will report mean, median, standard deviation, and min/max for all configurations. We will add thermal monitoring data (junction temperature via `powermetrics`) showing the M4 Pro does not throttle during our benchmarks (sustained temperature below 95C). The 198 measurement breakdown will be clarified in a revised Appendix C.

### To R3-W1 (No real-world workload):

**Partially accept.** We will add one end-to-end workload evaluation. The prisoner's dilemma scenario from our codebase is a 5-round, 2-agent negotiation where each agent accumulates 3-5K tokens by the final round. We will report: (a) per-round TTFT with and without persistent cache, (b) total workflow completion time, and (c) whether agents maintain coherent negotiation strategy across rounds (qualitative assessment + automatic coherence metric via separate judge LLM).

We note that COLM is a language modeling venue, not an applications venue. The core contribution is the inference system, not the downstream task performance. However, we agree that at least one realistic workload is necessary to ground the latency numbers.

### To R3-W2 (Working memory unvalidated):

**Partially accept.** We will add an information retention experiment: inject a unique identifier (e.g., a code variable name) in Phase 1, run 5 intermediate phases, and test whether the agent can recall the identifier in Phase 7. Compare persistent KV cache vs. re-prefill (which re-processes the full history and should also retain the information). Expected result: both achieve high recall, but persistent cache does it 30x faster. The contribution is latency, not accuracy.

We will soften the "working memory" language to "persistent cache state" where the cognitive analogy is not justified by data.

### To R3-W3 (No RAG comparison):

**Partially accept.** RAG and KV persistence solve different problems. RAG handles unlimited external knowledge; KV persistence handles conversation history. A direct comparison is apples-to-oranges. However, we will add a qualitative comparison table:

| Aspect | RAG | KV Persistence |
|--------|-----|----------------|
| Latency | O(n) re-prefill per request | O(1) cache reload |
| Context scope | External knowledge | Conversation history |
| Model portability | Model-independent | Model-specific |
| Update granularity | Document-level | Turn-level |
| Hardware requirement | Vector DB + LLM | LLM only |

And we will acknowledge explicitly: "KV persistence does not replace RAG. It complements RAG by eliminating re-computation of the conversation-history portion of the context, while RAG handles dynamic external knowledge."

### To R3-W5 (Edge use case is niche):

**Push back.** The edge LLM market is growing rapidly. Apple's M-series has shipped 30M+ units with ML-capable GPUs. MLX has 25,000+ GitHub stars. Local LLM tools (Ollama, LM Studio, llama.cpp) have millions of downloads. The "niche" characterization understates the market trajectory. We will add adoption numbers for local LLM tools and cite the privacy/compliance drivers (GDPR, HIPAA) that motivate on-device inference.

Regarding Apple Intelligence: if Apple integrates persistent KV caching, that validates our contribution rather than making it redundant. Apple's solution would be closed-source and limited to Apple-approved models. Our open-source system works with any MLX-compatible model.

### To R3-W6 (Two models insufficient for multi-architecture):

**Partially accept.** GQA and MLA are the two dominant modern attention patterns. Standard MHA (GPT-2) is a degenerate case of GQA (n_kv_heads = n_heads). Mamba has no KV cache (so our system does not apply). Griffin is a niche architecture with limited model availability. Multimodal cross-attention is a genuine gap we should acknowledge.

We will revise the claim from "the abstraction generalizes" to "the abstraction handles both dominant modern attention patterns (GQA and MLA)" and add a paragraph discussing architecture boundaries.

### To R3-W7 (BatchQuantizedKVCache is overstated):

**Partially reject.** The operations are individually standard, but the combination (batched inference over Q4 quantized KV caches with padding, interleaved prefill+decode scheduling, and per-token streaming during batched generation) does not exist in any prior system. mlx-lm v0.30 has no batched Q4 inference. vLLM does not support Q4 KV cache. The novelty is integration under constraints, not any single operation. We will reframe the contribution as "enabling batched Q4 inference" rather than claiming the individual operations are novel.

### To R3-W8 (Missing competitive baselines):

**Accept for llama.cpp, reject for cloud APIs.** We will add a llama.cpp comparison using `--slot-save-path` on the same hardware. Expected result: llama.cpp FP16 slot persistence achieves similar TTFT for small contexts but uses 3.6x more disk space and supports fewer concurrent agents in memory.

Cloud API comparison is out of scope. Cloud latency depends on network, load balancing, and server hardware, none of which this paper controls. The contribution is to edge inference, not to competing with cloud.

---

## Revision Priority Matrix

| Issue | Severity | Effort | Priority |
|-------|----------|--------|----------|
| Perplexity evaluation (R1-W1) | Critical | High (needs compute) | P0 |
| FP16 baseline comparison (R1-W3, R2-Q2) | Critical | Medium | P0 |
| Ablation study (R1-W2) | Critical | Medium | P0 |
| Real workload evaluation (R3-W1) | Critical | Medium | P1 |
| Memory measurements (R2-W7) | High | Low | P1 |
| Longer output lengths (R1-W6) | High | Medium | P1 |
| Thread safety in main text (R2-W6) | High | Low | P1 |
| llama.cpp comparison (R3-W8) | High | Medium | P1 |
| Working memory retention test (R3-W2) | Medium | Medium | P2 |
| Warm staggered data (R2-W4) | Medium | Low | P2 |
| Additional models (R2-W1) | Medium | High | P2 |
| Confidence intervals, 10 runs (R1-W5) | Medium | High | P2 |
| Restructure headline numbers (R1-W7) | Low | Low | P3 |
| Portability discussion (R2-W3) | Low | Low | P3 |
| Edge market sizing (R3-W5) | Low | Low | P3 |
| RAG comparison table (R3-W3) | Low | Low | P3 |

---

## Decision Simulation

**Area Chair Note**: All three reviewers recommend rejection, citing the absence of quality evaluation as the unanimous critical weakness. The paper presents a well-engineered system but fails to demonstrate that its core mechanism (persistent Q4 KV cache) does not harm the language modeling objective. For a venue called "Conference on Language Modeling," this is a fatal omission.

The authors are encouraged to:
1. Add perplexity and task accuracy evaluations
2. Broaden hardware and model coverage
3. Compare against FP16 persistence and llama.cpp slot API
4. Validate the "working memory" claim quantitatively or remove it

A revised submission addressing these concerns would be competitive at MLSys or MobiSys, where the systems contribution may be weighted more heavily than the language modeling evaluation. For COLM, the language modeling quality evidence is non-negotiable.

**Decision**: Reject

---

*Generated: February 9, 2026*
*This document simulates adversarial peer review for internal paper improvement. Scores and recommendations are deliberately harsh to surface the weakest points in the submission.*
