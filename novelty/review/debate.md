# Expert Panel Debate: "Agent Memory Below the Prompt"

**Venue**: COLM 2026 Program Committee Simulation
**Paper**: "Agent Memory Below the Prompt: Quantized KV Cache Management for Multi-Agent Inference on Unified Memory Hardware"
**Format**: 4 rounds, 4 panelists

---

## Panelists

| Tag | Role | Orientation |
|-----|------|-------------|
| **SR** | Systems Researcher | Practical systems contributions, engineering rigor |
| **MT** | ML Theorist | Formal analysis, generalization guarantees, ablations |
| **PR** | Practitioner | Deployability, pain-point relevance, tomorrow-morning utility |
| **CR** | COLM Reviewer | Venue fit, novelty relative to prior art, evaluation completeness |

---

## Round 1: Initial Assessment

### SR (Systems Researcher)

This is a serious systems paper disguised as an ML paper. The authors built a three-tier KV cache hierarchy (hot/warm/cold) with quantized block management, concurrent scheduling with chunked prefill, and they actually shipped it on real hardware. The 130x TTFT reduction on 32K-token Gemma contexts is not a toy number --- that is the difference between "unusable" and "interactive" for a local agent system.

What I find most compelling is the `BatchQuantizedKVCache` implementation. There is no upstream equivalent in MLX. The authors had to solve GQA mask broadcasting for batched 5D tensors, work around Metal assertion failures from concurrent eval, implement a B=1 split-and-concat strategy because `mx.compile(shapeless=True)` crashes on dynamic batch dimensions, and add `mx.synchronize()` barriers at precisely the right points to prevent cross-stream corruption. This is not "we called an API." This is low-level GPU runtime engineering.

The dual-architecture support is also non-trivial. Dense GQA (Gemma 3, symmetric K/V heads) and MoE MLA (DeepSeek V2, asymmetric K=192 / V=128) go through one abstraction. The spec extractor auto-detects MLA via `qk_nope_head_dim` + `qk_rope_head_dim` attributes. Most papers pick one architecture and call it a day.

My concern is that the evaluation is entirely latency-focused. No throughput saturation curves, no memory fragmentation analysis over long sessions, no comparison against vLLM's PagedAttention or SGLang's RadixAttention even on different hardware. The paper would be stronger with at least a conceptual comparison showing what transfers and what does not.

**Initial score: 6.5/10. Strong engineering, incomplete evaluation.**

---

### MT (ML Theorist)

I will be direct: this paper has zero formal analysis. No perplexity evaluation. No proof or even empirical evidence that the Q4 quantized KV cache preserves output quality relative to FP16. The claim of "72% memory savings" is meaningless without a quality degradation bound. You could get 100% memory savings by deleting the cache entirely.

The 198 measurements per model, 3 passes each, is respectable for a systems benchmark, but the statistical methodology is not described. Are these means? Medians? What are the confidence intervals? What is the variance across passes? With only 2 models (Gemma 3 12B and DeepSeek-Coder-V2-Lite), the generalization story is extremely thin. These are both instruction-tuned chat models under 16B parameters. Would this work on Llama 3 70B? On a model with different attention patterns? We have no idea.

The ablation structure is also missing. The paper bundles multiple techniques --- quantized caching, block pooling, chunked prefill, concurrent scheduling, thread safety machinery --- but never isolates their individual contributions. Which of these matters most? If I only implement the Q4 cache without the scheduler, what do I get? If I use the scheduler without quantization, what changes? Without ablations, the contribution is a monolithic system that either works or does not.

I do acknowledge that the thread safety analysis is technically interesting. The identification of MLX's non-thread-safe `unordered_map` in `StreamContext` and the engineering around it (single-thread inference + `mlx_io_lock` for cross-thread I/O) reflects genuine understanding. But understanding a bug is not the same as a scientific contribution.

**Initial score: 4/10. Interesting engineering report. Not a research paper in its current form.**

---

### PR (Practitioner)

I want to run multi-agent systems on my MacBook Pro. Today. This paper tells me exactly how to do that.

Let me count the pain points this solves. First: KV cache memory. A 12B model at FP16 with 4K context per agent eats your entire 24GB unified memory budget for 2-3 agents. Q4 quantization drops that by 72%, so now I can fit 6-8 agents. That is the difference between a toy demo and a useful system.

Second: time-to-first-token. Without caching, every agent turn re-processes the full conversation history. At 32K tokens, that is multiple seconds on Apple Silicon. With hot cache, it is 40ms. That makes multi-turn agent dialogue feel like a chat app, not a batch job.

Third: concurrent inference. The BatchQuantizedKVCache with the interleaved scheduler means two agents can think simultaneously. The staggered benchmark shows User B's TTFT drops dramatically because prefill chunks are interleaved with User A's decode. This is real concurrent inference on a single GPU, not sequential pretending to be parallel.

Fourth: the architecture handles both Gemma and DeepSeek through one code path. I do not want to maintain two inference backends.

What I wish the paper included: actual agent task completion benchmarks. How does a ReAct agent perform with this cache versus without? Does the Q4 quantization cause any degradation on tool-use accuracy? I suspect the answer is "no measurable difference," but the paper should prove it.

Also missing: a comparison with llama.cpp's server mode, which also does KV cache management on Apple Silicon. Practitioners will ask this immediately.

**Initial score: 7/10. I would use this. The evaluation gaps are real but the system is real too.**

---

### CR (COLM Reviewer)

COLM's scope is "language modeling" interpreted broadly, and the call for papers explicitly welcomes systems contributions that advance how language models are deployed and used. This paper fits. It is not a stretch.

On novelty: the individual techniques (KV cache quantization, block pooling, chunked prefill) exist in the literature. vLLM introduced PagedAttention, SGLang introduced RadixAttention with prefix sharing, and quantized KV caches appear in several concurrent submissions. However, the combination for unified memory hardware is new. Nobody has published a three-tier quantized cache hierarchy specifically for Apple Silicon's shared CPU/GPU memory architecture. The UMA-specific optimizations (no PCIe transfer overhead, direct Metal buffer management, MLX-native safetensors persistence) are genuinely different from the datacenter-focused prior art.

The dual-architecture abstraction (GQA + MLA) is a real contribution. Most systems papers target one attention variant. Supporting both through a single `ModelCacheSpec` with runtime detection is useful and non-obvious, especially given the asymmetric dimensions in MLA.

Evaluation concerns: 2 models is thin for a venue like COLM. The top-tier systems papers at OSDI/SOSP/MLSys typically evaluate on 5-8 models across multiple sizes. The absence of quality evaluation (perplexity, downstream task accuracy) is a gap that reviewers will flag. The 198-measurement methodology is rigorous for latency, but latency alone is insufficient.

I also note that the paper does not compare against any published baseline system. Even an apples-to-oranges comparison (vLLM on A100 vs. this system on M4 Pro, normalized by hardware cost or power) would contextualize the contribution.

**Initial score: 5.5/10. Venue fit is fine. Novelty is incremental-to-moderate. Evaluation is below the acceptance bar.**

---

## Round 2: Strongest Criticism

### SR attacks: No comparison with existing systems

The elephant in the room is that this paper exists in a vacuum. vLLM's PagedAttention, SGLang's RadixAttention, TensorRT-LLM's KV cache reuse, and even llama.cpp's context shifting all solve overlapping problems. The paper never once positions itself against these systems. Not a latency comparison, not an architectural comparison, not even a table saying "Feature X exists in System Y but not in ours."

This matters because the claimed contributions --- block pooling, tiered caching, prefix reuse --- are well-explored in the datacenter setting. The paper's implicit argument is "but those are for CUDA, ours is for Metal/UMA." That is a valid argument! But you have to make it explicitly. Show me why PagedAttention's virtual memory abstraction does not translate to UMA. Explain why RadixAttention's trie-based prefix matching is or is not applicable when you have zero PCIe overhead. Without this analysis, a reviewer cannot assess whether the contribution is "we ported known ideas to new hardware" or "UMA fundamentally changes the design space." I suspect it is the latter, but the paper does not prove it.

---

### MT attacks: No quality evaluation whatsoever

I want to be precise about what is missing. The paper claims 72% memory savings from Q4 KV cache quantization. Quantization is lossy. The paper presents zero evidence that this lossy compression does not degrade output quality. This is not a minor omission --- it is a missing experiment that undermines the central claim.

Consider: if Q4 quantization causes even 0.5 perplexity points of degradation, the memory savings are buying you worse outputs. In a multi-agent system where agents read each other's cached context, quantization errors could compound across turns. Agent A's Q4-corrupted context influences Agent B's generation, which is cached in Q4 and read by Agent C. Nobody has measured this cascading degradation.

The paper needs at minimum: (1) perplexity on a standard benchmark (WikiText-103, C4) comparing FP16 KV cache vs. Q4 KV cache, (2) a downstream task evaluation showing agent task completion is preserved, and (3) an analysis of error propagation in multi-turn multi-agent dialogue. Without these, the "72% savings" headline is marketing, not science.

---

### PR attacks: Only 2 models, both small

I am enthusiastic about this system, but I have to be honest: 2 models is not enough. Both Gemma 3 12B and DeepSeek-Coder-V2-Lite are under 16B parameters. Both are 4-bit quantized. Both fit comfortably on a 24GB M4 Pro.

The interesting question is what happens at the boundary. Can this system run Llama 3.1 8B alongside a 4B model? What about Qwen2.5 32B Q4, which barely fits in 24GB? What happens when the block pool is 90% full and two agents simultaneously request 4K-token prefills? The paper never explores resource pressure scenarios.

More importantly: the practitioner community is moving toward 30B-70B models on 64GB and 128GB Apple Silicon machines (M3/M4 Ultra). The paper's evaluation on a 24GB machine with sub-16B models feels like it is targeting yesterday's hardware. I would want to see at least one experiment on an M-series Ultra with a larger model to demonstrate the architecture scales.

---

### CR attacks: Incremental novelty, no formal contribution

Let me steelman the rejection case. Each individual component has clear prior art:

- **Quantized KV cache**: Introduced in KIVI (ICML 2024), also in QServe, KVQuant. Not new.
- **Block pooling**: vLLM's PagedAttention (SOSP 2023). Adapted, not invented.
- **Tiered caching**: Standard in storage systems (L1/L2/L3, memcached tiers). Applied, not novel.
- **Chunked prefill**: Sarathi (ISCA 2024), also in DeepSpeed-FastGen. Known technique.
- **Concurrent scheduling**: Orca (OSDI 2022) iteration-level scheduling. Adapted for single-GPU.

The paper combines these into a working system for Apple Silicon. Is that combination novel enough for a top venue? At OSDI or MLSys, a "good engineering" paper can get in if the evaluation is comprehensive. At COLM, the bar is different --- reviewers expect insight about language modeling, not just faster inference. What does this paper teach us about how language models work or should be deployed? The thread safety findings are specific to MLX, not generalizable. The UMA memory model is specific to Apple, not transferable.

A workshop paper at the COLM Systems Track, absolutely. A main conference paper? I am not convinced the novelty bar is met.

---

## Round 3: Strongest Defense

### SR defends: The BatchQuantizedKVCache is a genuine systems contribution

Let me push back on the "incremental combination" critique. The `BatchQuantizedKVCache` is not a port of PagedAttention. It solves a fundamentally different problem.

PagedAttention manages FP16 KV blocks in CUDA virtual memory with page tables. It never touches the attention computation itself. `BatchQuantizedKVCache` operates inside the attention kernel. It handles quantized block concatenation, GQA head expansion with 5D mask broadcasting, dynamic batch dimension compilation (the B=1 split exists because `mx.compile` cannot handle changing batch sizes), and sliding window / global attention hybrid masking for architectures like Gemma 3 that use both within the same model.

No published system does this. vLLM's quantized cache (merged Q1 2025) operates at the page level, not the attention level. SGLang's cache is FP16 only. TensorRT-LLM's FP8 KV cache uses hardware-accelerated dequantization on Hopper GPUs that does not exist on Apple Silicon.

The dual-architecture support (GQA + MLA) through a single abstraction is also not "just engineering." MLA's asymmetric key/value dimensions (K=192, V=128) break every assumption in standard KV cache implementations. The spec extractor that auto-detects MLA via attention module attributes and adjusts block allocation accordingly is a design contribution that would benefit any inference system.

The thread safety work is MLX-specific, yes, but the methodology is transferable. Identifying that a framework's stream context uses unprotected data structures, then designing an architecture (single-thread inference + lock-guarded I/O) that works within those constraints --- this is a pattern that applies to any emerging ML framework with incomplete concurrency support.

---

### MT defends: The measurement methodology is actually rigorous

I have been critical, but I should acknowledge what the paper does well on the evaluation front.

198 measurements per model, 3 passes each, covering cold start, warm cache, and hot cache conditions across multiple context lengths (1K, 2K, 4K, 8K, 16K, 32K) and generation lengths (32, 64, 128, 256) --- this is more thorough than most systems papers I review. The factorial design (context length x generation length x cache state x batch size) is proper experimental design.

The 130x TTFT reduction is also not cherry-picked. It is the hot-cache case at 32K tokens, which is the scenario where caching provides maximum benefit. The paper also reports the less impressive numbers (warm cache, shorter contexts) where the speedup is smaller. This is honest reporting.

I stand by my criticism that quality evaluation is missing. But I want to be fair: the Q4 KV cache quantization used here (group size 64, symmetric quantization) is the same scheme validated in KIVI and subsequent work. Those papers showed <0.1 perplexity degradation on standard benchmarks. It would be trivial for the authors to run those experiments and cite the prior validation. The fact that they did not is an oversight, not evidence that Q4 degrades quality.

If the authors added perplexity numbers in a revision and they aligned with KIVI's findings (as I expect they would), my score would move significantly.

---

### PR defends: This solves a real gap that no other system addresses

I want to make the deployment argument concrete. Today, if I want to run multi-agent inference on Apple Silicon, my options are:

1. **llama.cpp server**: KV cache management exists but no multi-agent cache sharing, no block pooling, no tiered persistence. Each agent conversation starts cold.
2. **MLX stock**: `mlx-lm` has `generate()` and `stream_generate()`. Single-sequence only. No batching, no cache persistence, no concurrent inference. The `QuantizedKVCache` exists but has bugs (the sliding window mask issue the paper identifies and fixes).
3. **vLLM/SGLang**: CUDA only. Do not run on Metal.
4. **Ollama**: Wraps llama.cpp. Same limitations plus additional overhead.

There is literally no published system that provides: quantized KV cache + block pooling + tiered persistence + concurrent batched inference + multi-architecture support, on Apple Silicon. The paper fills a gap that every Mac-based ML developer feels daily.

The "only 2 models" criticism is valid but overstated. Gemma 3 (dense GQA, sliding window + global hybrid) and DeepSeek V2 (MoE MLA, asymmetric KV dims) are not two random models --- they are representatives of the two dominant modern attention architectures. If the system handles both, it very likely handles Llama, Mistral, Qwen, and Phi, which all use standard GQA.

---

### CR defends: COLM's scope is broader than pure ML theory

I want to respond to my own earlier skepticism. COLM 2025 accepted several systems-oriented papers: efficient inference papers, serving system papers, and papers about how deployment constraints shape model design. The venue explicitly states it covers "all aspects of language modeling, from algorithms to systems to applications."

The question is not whether this is a systems paper at COLM --- it clearly fits. The question is whether the systems contribution is substantial enough. And I think the answer depends on how you weigh the following:

The paper identifies and solves a real problem (multi-agent KV cache management on UMA hardware) that no prior system addresses. The solution involves genuine technical depth (batched quantized attention, hybrid sliding/global mask handling, thread safety on a non-thread-safe framework). The evaluation, while incomplete on the quality axis, is thorough on the latency axis.

COLM is also a relatively young venue still defining its identity. Accepting systems papers that demonstrate deep understanding of language model internals (attention patterns, KV cache structure, quantization implications) helps establish COLM as the home for this kind of work. Sending it to MLSys means losing the language-modeling-specific insights. Sending it to OSDI means losing the ML audience who would benefit most.

---

## Round 4: Final Verdict

### SR (Systems Researcher)

**Verdict: Weak Accept (6/10, confidence: high)**

The core systems contribution --- `BatchQuantizedKVCache` with dual-architecture support and thread-safe concurrent scheduling on UMA hardware --- is real and non-trivial. No published system provides this functionality. The implementation depth (B=1 split compilation, hybrid mask handling, MLX thread safety engineering) reflects genuine expertise.

The paper falls short on comparative evaluation. Adding a structured comparison with vLLM/SGLang (even architectural, not necessarily empirical) and at least one quality metric would push this to a solid accept. As submitted, it is a borderline paper with a strong engineering core.

I would champion this paper in a discussion if the other reviews are not hostile.

---

### MT (ML Theorist)

**Verdict: Weak Reject (4.5/10, confidence: medium)**

I cannot in good conscience recommend acceptance without quality evaluation. The 72% memory savings claim is hollow without perplexity or downstream task numbers. The absence of ablations means I cannot determine which components contribute what. The 2-model evaluation does not support generalization claims.

However, I am not confident in this rejection. If the authors submitted a revision with: (a) perplexity on WikiText/C4 comparing FP16 vs Q4 KV cache, (b) ablations isolating quantization, block pooling, and scheduling contributions, and (c) one additional model (e.g., Llama 3 8B), I would likely flip to weak accept. The underlying system is clearly functional and the measurement methodology is sound.

My concern is that accepting this paper as-is sets a precedent that systems papers at COLM do not need quality evaluation. That precedent would be harmful.

---

### PR (Practitioner)

**Verdict: Accept (7/10, confidence: high)**

This paper solves a problem I have today with a system that works today. The gap in the ecosystem is real --- there is no multi-agent quantized KV cache system for Apple Silicon. The dual-architecture support is forward-looking and well-designed. The 130x TTFT improvement is not an incremental gain; it is a qualitative change in what is possible.

I acknowledge the evaluation gaps. No perplexity numbers, no agent task benchmarks, only 2 models. These are real shortcomings. But the systems contribution stands on its own merits: nobody else has built this, the engineering is deep and correct (198 measurements, thread safety analysis, hybrid mask handling), and the design decisions are well-motivated by hardware constraints.

The ML theorist's concerns about quality degradation are valid in principle but unlikely in practice. Q4 KV quantization at group-size 64 has been validated in multiple prior works. The authors should cite these and add a confirmation experiment, but the absence of that experiment does not invalidate the system.

I would advocate for acceptance with a request for quality numbers in the camera-ready version.

---

### CR (COLM Reviewer)

**Verdict: Borderline / Weak Reject (5/10, confidence: medium)**

This is a close call, and my recommendation depends on the overall quality of the COLM 2026 submission pool.

**For acceptance**: The paper addresses a genuine gap (UMA-specific multi-agent inference), the technical depth is substantial, the venue fit is appropriate, and the measurement methodology exceeds what most systems papers provide on the latency axis.

**Against acceptance**: The evaluation is incomplete (no quality metrics, no baselines, 2 models), the novelty is a combination of known techniques on new hardware rather than a fundamentally new idea, and the generalizability beyond Apple Silicon and MLX is unclear.

My recommendation is weak reject with encouragement to resubmit. Specifically, the authors should add: (1) perplexity comparison validating Q4 does not degrade quality, (2) at least one additional model architecture (Llama 3 would be natural), (3) a structured comparison table against vLLM and SGLang explaining what transfers to UMA and what does not, and (4) an ablation isolating the contribution of each component.

If even two of these four items were addressed, I would move to weak accept. The underlying system is strong. The paper, as submitted, does not fully convey that strength to a reviewer who cannot run the code.

---

## Score Summary

| Panelist | Score | Verdict | Confidence |
|----------|-------|---------|------------|
| SR (Systems) | 6.0 | Weak Accept | High |
| MT (Theory) | 4.5 | Weak Reject | Medium |
| PR (Practitioner) | 7.0 | Accept | High |
| CR (COLM) | 5.0 | Borderline / Weak Reject | Medium |

**Average: 5.625 / 10**

**Consensus**: The system is a genuine contribution that fills a real gap. The paper does not yet meet the evaluation bar for a top venue. With quality metrics, ablations, and system comparisons, it likely clears that bar. The recommendation is conditional: revise and resubmit with the identified experiments.

---

## Key Actionable Items Identified by the Panel

1. **Add perplexity evaluation** (FP16 vs Q4 KV cache on WikiText-103 and C4) --- addresses MT's core objection and is likely a low-effort, high-impact addition.
2. **Add ablation study** isolating: Q4 quantization alone, block pooling alone, scheduler alone, all combined --- required to understand component contributions.
3. **Add at least one more model** (Llama 3 8B recommended as standard GQA baseline) --- strengthens generalization.
4. **Add structured comparison with vLLM/SGLang** --- even an architectural feature table with qualitative UMA-vs-datacenter analysis would substantially improve positioning.
5. **Add agent task completion benchmark** --- run a ReAct or tool-use evaluation with and without the cache system to demonstrate end-to-end utility.
6. **Report statistical methodology** --- specify whether results are means/medians, include confidence intervals or standard deviations across the 3 passes.
