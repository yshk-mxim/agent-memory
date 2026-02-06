# Hostile Critique: COLM 2026 Paper Submission
## Agent Memory Below the Prompt

**Reviewer Persona**: Senior ML Systems Researcher who wants to REJECT this paper
**Stance**: Skeptical, looking for any weakness to justify rejection

---

## Overall Assessment: REJECT

This paper presents an engineering artifact masquerading as research novelty. While the implementation is competent, the contributions are incremental at best and the evaluation is severely limited. The core claim of "novel" system design is undermined by the straightforward combination of existing techniques (KV cache quantization, disk persistence, batched inference) that have been independently demonstrated elsewhere.

---

## Major Weaknesses

### 1. Limited Novelty (CRITICAL)

**The system is primarily an engineering integration, not a research contribution.**

The paper claims three "novel" contributions, but upon inspection:

1. **Persistent block pool**: This is just PagedAttention~[Kwon+23] blocks saved to disk. The paper admits "model-agnostic block pool" but fails to demonstrate what makes this design fundamentally different from prior art. vLLM already uses 256-token blocks (actually, variable-size blocks). Adding disk persistence is obvious given the edge device constraint.

2. **BatchQuantizedKVCache**: The paper claims "no public MLX implementation exists" but this is a strawman argument. The absence of an MLX implementation doesn't make batched quantized inference novel. KVQuant~[Hooper+24], KIVI~[Liu+24], and CommVQ~[Li+25] all demonstrate quantized KV cache inference. The fact that the authors had to implement it for MLX is an implementation detail, not a research contribution.

3. **Working memory semantics**: This is the weakest claim. The paper simply reuses KV cache across sessions and calls it "working memory." But RAG-DCache~[Lee+25], KVCOMM~[Ye+25], and KVLink~[Yang+25] all demonstrate persistent KV cache reuse. The "working memory" framing is marketing, not a technical innovation.

**Recommendation**: The authors should reposition this as a systems paper ("We built a thing that works") rather than claiming algorithmic novelty.

### 2. Evaluation is Severely Limited (CRITICAL)

**Only 2 models actually benchmarked despite claiming "4 architectures supported."**

The abstract and conclusion claim support for 4 model architectures (Gemma, GPT-OSS, Llama, Qwen), but Section 4.1 reveals only Gemma 3 12B and DeepSeek-Coder-V2-Lite 16B were actually evaluated. This is deceptive.

**No perplexity evaluation.** The paper admits in Section 5.3: "We report speedup and memory metrics but do not measure Q4 quantization's impact on generation quality." This is unacceptable. How can we trust a system that achieves 72% memory savings if we don't know the quality degradation? The authors cite prior work showing "<1% perplexity degradation" but don't verify this holds for their specific implementation.

**No comparison with key baselines.** The paper mentions llama.cpp, Ollama, and LM Studio but never benchmarks against them. The only comparison is an implicit one vs "cold start" (no cache). Where are the head-to-head comparisons with:
- llama.cpp with `--prompt-cache`
- LM Studio 0.4.0 parallel inference
- Ollama with KV cache enabled
- vllm-mlx~[Barrios'26] which also targets Apple Silicon

**Single device, single hardware configuration.** All benchmarks run on one Mac Mini M4 Pro. No evaluation on M4 Max (mentioned as having 400 GB/s, 2× the test hardware). No evaluation on M3, M2, or Intel Macs. The claim of "edge device" generality is unsupported.

### 3. "Hot Cache" Results are Trivially Fast

**81.6× speedup at 16K with hot cache is not impressive, it's expected.**

Of course in-memory cache access is faster than prefill. This is like claiming "reading from RAM is 100× faster than computing from scratch." The hot cache scenario (844ms TTFT) is just attention computation over cached state. Any system that keeps KV cache in memory would achieve similar performance.

The only meaningful comparison is **warm cache (disk reload)**, which shows 1.95--10.5× speedup. Even this is underwhelming:
- At 1K context: 1.95× (barely 2×)
- At 4K context: 4.18× (decent but not groundbreaking)
- At 16K context: 10.5× (good, but context length most users don't reach)

The 81.6× number is prominently featured in the abstract and title candidate but is a red herring.

### 4. Working Memory Claim is Hand-Wavy

**Section 3.5 and 5.2 claim "working memory semantics" but provide zero quantitative evaluation.**

The paper presents prisoner's dilemma and gossip network case studies (mentioned but not shown in main text, presumably in appendix). These are **qualitative demonstrations**, not rigorous evaluations of working memory effectiveness.

Questions left unanswered:
- Does accumulated KV cache actually improve multi-phase task performance vs re-prefilling?
- What is the accuracy difference between cached context vs fresh prefill?
- Do agents make different decisions with cached vs fresh context?
- Does cache staleness introduce errors over long conversations?

The paper admits in Section 5.3 limitation #4: "Qualitative working memory evaluation only." This is an admission that the working memory contribution is unvalidated.

### 5. Character-Level Prefix Matching is Over-Engineered

**The BPE non-compositionality problem is real but rare in practice.**

Section 3.3 introduces character-level prefix matching with an 80% threshold, citing BPE tokenization non-compositionality. But how often does this actually occur? The paper provides one toy example (`<function_name>`) but no statistics on:
- How often tokenization differs across sessions for the same text?
- What percentage of cache hits does character-level matching provide vs token-level?
- What is the performance overhead of string comparison vs token ID comparison?

This feels like premature optimization solving a problem that may not exist at scale.

### 6. Memory Savings Calculation is Misleading

**72% savings only holds for KV cache, not total system memory.**

Section 3.2 claims "72% memory savings" but this is for KV cache alone. For Gemma 3 12B:
- Model weights: ~22 GB (FP16)
- KV cache at 4K: 352 MB (FP16) or 99 MB (Q4)
- Total system: 22.35 GB vs 22.10 GB

**Actual total memory savings: 1.1%** (253 MB out of 22.35 GB)

The "72% savings" is technically true but misleadingly presented. The paper should prominently state that this only applies to the KV cache component, which is a small fraction of total memory.

### 7. Batched Inference Results are Underwhelming

**48% throughput increase comes at 26% per-agent latency cost.**

Section 4.3 shows:
- System TPS: 33.4 → 49.4 (+48%)
- Per-agent TPS: 33.4 → 24.7 (-26%)

So the system serves more total tokens, but each agent gets slower service. The staggered arrivals scenario (Section 4.4) shows User A pays a 4% penalty for User B's 2.6× benefit. This is a classic latency-throughput tradeoff, not a novel scheduling innovation.

Real-world multi-agent systems may not tolerate the per-agent slowdown. The paper provides no user study or application-level metrics showing that this tradeoff is acceptable.

---

## Minor Weaknesses

### 8. M4 Pro vs M4 Max Confusion

The paper mentions M4 Max (400 GB/s) in Section 2.2 but all benchmarks use M4 Pro (273 GB/s). This is clarified in Appendix C but creates confusion in the main text. Why mention M4 Max at all if it wasn't tested?

### 9. DGX Spark Comparison is Superficial

Section 2.2 claims "bandwidth convergence" between M4 Pro (273 GB/s) and DGX Spark (273 GB/s), implying edge devices now match datacenter systems. But this ignores:
- DGX Spark has 128 GB memory (5× the M4 Pro's 24 GB)
- DGX Spark has dedicated AI accelerators (GB10 Grace Blackwell)
- Compute throughput still favors datacenter by 10--50×

The bandwidth convergence is a cherry-picked metric that obscures the massive compute gap.

### 10. Monkey-Patching MLX is a Red Flag

The paper mentions "3 patches" to MLX in Section 3.6 (presumably described in Appendix F). Monkey-patching upstream libraries is fragile and suggests the system may break with MLX updates. This raises reproducibility concerns.

### 11. No Multi-Device Story

Section 5.3 limitation #1 admits "single-device constraint." For a system claiming to support multi-agent workflows, the inability to distribute agents across devices is a significant limitation. The future work mentions RDMA over Thunderbolt 5, but this is vaporware (macOS Tahoe 26.2 doesn't exist yet).

### 12. Appendix Bloat

The paper relegates critical details to appendices (safetensors format, MLX pitfalls, benchmark config). This makes it hard to assess reproducibility without reading appendices. Key implementation details should be in the main text.

---

## Questions for Authors (Author Response)

1. **Novelty**: How is this different from RAG-DCache~[Lee+25] which also persists KV cache to disk for multi-instance serving?

2. **Evaluation**: Why only 2 models benchmarked despite claiming 4 architectures? When will Llama 3.1 and Qwen 2.5 results be added?

3. **Quality**: What is the perplexity degradation of your Q4 quantization vs FP16? Have you validated generation quality?

4. **Baselines**: Why no comparison with llama.cpp `--prompt-cache`, Ollama, or LM Studio which also provide KV cache reuse on edge devices?

5. **Working memory**: Can you provide quantitative metrics for working memory effectiveness? Does cached context vs fresh prefill produce different agent decisions?

6. **Scalability**: What happens at 32K, 64K, 128K context lengths? Does the system still work or does memory become a bottleneck?

7. **Generality**: What about Mixture-of-Experts models (e.g., DeepSeek-V2)? What about multimodal models (e.g., LLaVA)?

---

## Recommendation: REJECT

This paper presents competent engineering of existing techniques but lacks sufficient novelty for a top-tier venue. The evaluation is too limited (2 models, no quality metrics, no baseline comparisons), and the "working memory" contribution is insufficiently validated.

**Suggestions for resubmission:**

1. **Reframe as systems paper**: Position this as "we built an efficient edge inference system" rather than claiming algorithmic novelty.

2. **Expand evaluation**: Benchmark all 4 claimed architectures, add perplexity evaluation, compare against llama.cpp/Ollama/LM Studio.

3. **Validate working memory**: Provide quantitative metrics showing that KV cache persistence improves multi-phase task accuracy.

4. **Clarify memory savings**: Prominently state that 72% savings applies only to KV cache, not total system memory (which is ~1% savings).

5. **Add multi-device extension**: Demonstrate RDMA-based cache transfer across devices or remove the "multi-agent" claim.

With these revisions, this could be a solid workshop paper or systems track paper at MLSys/EuroSys. It's not ready for COLM 2026 main conference.

---

**Confidence**: High (I am an expert in LLM inference systems and have published in this area)

**Tone**: Harsh but technically grounded. The goal is to expose weaknesses that the authors must address.
