# Adversarial Peer Review: COLM 2026 Submission

**Paper**: Agent Memory Below the Prompt: Persistent Q4 KV Cache for Multi-Agent LLM Inference on Edge Devices

**Venue**: COLM 2026 (Conference on Language Modeling)

**Review Date**: February 9, 2026

**Review Round**: Revised manuscript (post-initial-feedback)

**Format**: 3 hostile reviewers, each with score, strengths, weaknesses, and questions, followed by author rebuttals

---

## Reviewer 1 (R1): ML Theory and Quantization

**Expertise**: KV cache compression, attention mechanism theory, quantization error analysis, perplexity evaluation methodology

**Overall Score**: 3/10

**Recommendation**: Reject

**Confidence**: 5/5

### Summary

This paper claims that persisting Q4 KV caches to disk enables 30-130x TTFT speedups for multi-agent inference on Apple Silicon. The system design is competently described, but the paper has a gaping hole where its quality evaluation should be. The perplexity table (Table 8, Appendix E) is literally empty -- dashes instead of numbers -- with a parenthetical note saying "Results to be filled after running `benchmarks/perplexity_benchmark.py`." The abstract claims "<0.1 perplexity degradation" without having run the experiment. The ablation (Table 4) reuses numbers from other tables and calls it analysis. The cross-phase injection contribution has zero quality metrics. This paper treats language modeling quality as an afterthought at a venue dedicated to language modeling.

### Strengths

**S1.** The cold-start problem is precisely quantified. The 77-second restart penalty for 5 agents at 4K context is a concrete, reproducible number derived from measured data (Table 1), not a hypothetical. The O(n) cold vs O(1) warm framing is clean.

**S2.** The Q4 memory analysis (Section 3.2, Appendix D) is mathematically rigorous. The 0.281 ratio derivation for group_size=64 is correct, and Table 2 provides agent capacity numbers that follow from the formula. The FP16 vs Q4 capacity comparison is the strongest analytical section of the paper.

### Weaknesses

**W1. The perplexity table is empty (FATAL).** Table 8 in Appendix E contains three dashes per row instead of measured values. The text below it says "[Results to be filled after running `benchmarks/perplexity_benchmark.py`.]" This is not a minor omission. The abstract claims "<0.1 perplexity degradation" without data. The conclusion repeats this claim. The entire Q4 quality argument rests on citing KIVI, KVQuant, QuantSpec, RotateKV, and XQuant -- none of which use the same quantization scheme as this paper. KIVI uses per-channel key quantization and per-token value quantization. This paper uses uniform group quantization (group_size=64) on both keys and values. These are different algorithms with different error profiles. Citing others' numbers for your own untested scheme is not evidence; it is speculation. A submission to a language modeling venue with no quality measurement of its core mechanism is fundamentally incomplete.

**W2. The "simulated Q4 noise" argument is invalid.** Even if the authors were to fill in Table 8, the perplexity benchmark (as described in the code at `benchmarks/perplexity_benchmark.py`) evaluates on a local corpus with 512-token sliding windows and 8K total tokens. This is a simulated environment: the model weights are already Q4 (via mlx-lm GPTQ quantization), so any measured perplexity includes both weight quantization and KV quantization effects. You cannot isolate KV cache quantization impact from weight quantization impact in this setup. The correct methodology would be: (a) run FP16-weight model with FP16 KV cache, (b) run FP16-weight model with Q4 KV cache, (c) compute the delta. Since the authors only have Q4-weight models available on their hardware, they fundamentally cannot measure what they claim to measure. Citing prior work that uses FP16-weight models with quantized KV caches is comparing apples to oranges when your own pipeline stacks two quantization layers.

**W3. The ablation (Table 4) is analytical, not empirical.** Table 4 ("Component contributions") does not present new measurements. Every entry is derived from other tables: "Persistence" reuses Table 1 cold vs warm; "Q4 vs FP16" reuses Table 2 agent counts; "Batching" reuses Table 3 system TPS vs per-agent TPS; "Cross-phase" reuses Table 5 Phase 5 numbers. This is arithmetic, not ablation. A real ablation disables one component while holding others constant and re-measures the system. Specifically:

- **Persistence ablation**: Run FP16 persistent cache (not just Q4) and measure reload TTFT. This isolates whether the speedup comes from avoiding re-computation (persistence) or from smaller file sizes (Q4).
- **Batching ablation**: Run 2 agents sequentially on warm cache and compare to batched warm. Table 4 uses cold-batch SysTPS / 2 as "without batching," which conflates cold-start overhead with batching benefit.
- **Cross-phase ablation**: Run the prisoner's dilemma with caches cleared between phases (they have this -- Table 5 "Cold" column) AND with caches persisted but forcing re-prefill of the extension (EXTEND match disabled). This would isolate the prefix-matching contribution from raw persistence.

None of these controlled experiments were conducted.

**W4. Cross-phase injection has no quality metrics.** Section 3.5 and Table 5 measure only TTFT for the cross-phase mechanism. The paper claims this treats "KV cache as persistent working memory" (line 172). But does it work? Does an agent in Phase 5 of the prisoner's dilemma actually recall what happened in Phase 1? Does the quantized cache faithfully preserve the attention patterns from earlier phases, or does Q4 rounding degrade the effective context window across phases? The paper provides exactly zero measurements of:
- Information retention across phases (inject a fact in Phase 1, test recall in Phase 5)
- Task coherence (do agents maintain consistent strategy across phases?)
- Generation quality degradation over accumulated phases
- Comparison of persistent-cache output vs re-prefill output (are they semantically equivalent?)

The claim that "both persistent cache and re-prefill produce equivalent context (modulo Q4 rounding)" (Section 5.3) is asserted without verification. "Modulo Q4 rounding" is precisely the thing that needs to be measured, and it is not.

**W5. Temperature 0.0 claimed in methodology, but T=0.3 hardcoded in system.** Section 4.1 states "Temperature 0.0 (greedy decoding, deterministic output)." But according to the project's own documentation (CLAUDE.md and MEMORY.md), the coordination_service.py hardcodes T=0.3 and ignores the SEMANTIC_MLX_DEFAULT_TEMPERATURE environment variable. The chat_completion_service uses the request body's temperature field, but the prisoner's dilemma and Wikipedia routing benchmarks go through the coordination service. Which temperature was actually used? If T=0.3, the results are stochastic, 3 passes are insufficient for statistical rigor, and no confidence intervals are reported. If T=0.0, the documentation is wrong. Either way, there is an inconsistency that undermines the claimed methodology.

**W6. The 198/216 quality pass rate is buried and unexplained.** Section 4.1 mentions "198 passing quality checks" out of 216 measurements. That means 18 measurements (8.3%) failed quality checks. Which ones? At what context lengths? Which cache states? Were these systematic failures (e.g., Q4 cache corruption at long contexts) or random? The paper does not report which configurations failed or why. For the Wikipedia routing benchmark (Table 6), DeepSeek shows only 3/10 quality pass in Phase 1 and 4/10 in Phase 2. That is a 30-40% success rate. The paper waves this away as "structural quality (keyword overlap, minimum length), not factual accuracy." But if a system produces outputs that fail basic structural checks 60-70% of the time, that is a quality problem, not a measurement artifact.

**W7. The 64-token output ceiling biases all speedup numbers.** Every measurement uses exactly 64 output tokens. TTFT is defined as time-to-first-token, which is unaffected by output length, so the TTFT numbers are valid. But the system TPS numbers (Table 3) and end-to-end times (Table 5) are heavily influenced by this choice. At 64 tokens, decode takes roughly 1-5 seconds depending on model speed. At 512 tokens (realistic for agent responses), decode takes 10-40 seconds, making the TTFT savings a smaller fraction of total time. The paper acknowledges this in Section 5.3 ("Fixed output length") but does not provide any data at other output lengths. For a system positioned as enabling multi-agent workflows where agents produce substantive responses (200-1000 tokens), the 64-token ceiling is unrealistically favorable.

### Questions for Authors

Q1. Why is Table 8 empty? Was the perplexity benchmark script never executed, or did it produce results that were not included?

Q2. How do you isolate KV cache quantization error from weight quantization error when both the model weights and KV cache are Q4?

Q3. What temperature was actually used for the prisoner's dilemma and Wikipedia routing benchmarks?

Q4. Which 18 of the 216 measurements failed quality checks, and what was the failure mode?

Q5. At 512 output tokens, what is the end-to-end time for Gemma at 4K with warm cache vs cold? What fraction of the total time does the TTFT savings represent?

### Minor Issues

M1. Figure 1 caption says "81.6x TTFT (hot)" but this number does not appear anywhere in the paper text. Table 1 shows 130x at 32K hot. Where does 81.6x come from?

M2. The figure caption for Figure 2 says "27 vs 46 layers" for DeepSeek vs Gemma, but the text consistently says Gemma has 48 layers. The caption has a typo.

M3. The notation in Section 3.2 switches between "group size $g$" and "group size 64" without consistently parameterizing. Pick one.

---

## Reviewer 2 (R2): Systems and Infrastructure

**Expertise**: GPU inference systems, distributed serving, hardware benchmarking, thread safety in concurrent systems

**Overall Score**: 4/10

**Recommendation**: Reject

**Confidence**: 5/5

### Summary

The paper describes an MLX-specific KV cache management system evaluated on exactly one hardware SKU (M4 Pro 24GB) with exactly two models. The system design has genuine merit (the block pool abstraction, safetensors persistence format, and the ModelCacheSpec interface are well-conceived), but the evaluation is so narrow that no general conclusion can be drawn. The thread safety workarounds are presented as engineering contributions when they are actually limitations. The hardware comparison table includes unverified specifications for devices the authors do not own. The "infrastructure layer" claim is not validated by any integration test with AutoGen, CrewAI, or LangGraph.

### Strengths

**S1.** The ModelCacheSpec abstraction (Section 3.6) is genuinely well-designed. Separating architectural parameters (layer count, KV head count, head dimensions, quantization settings) from model-specific logic is the right abstraction boundary. The asymmetric K/V dimension handling for MLA (v_head_dim field) shows the authors encountered and solved a real generalization problem.

**S2.** The safetensors format specification (Appendix A) is practical, complete, and immediately usable by other implementations. The tensor naming convention (cache.layers.{l}.key_cache.{b}.data/scales/biases) is well-structured. The bfloat16 handling for Gemma's scales is an important detail that other implementations would likely miss.

**S3.** The paper is transparent about limitations. Section 5.3 lists five specific limitations. The MLX engineering notes (Appendix B) honestly catalog failure modes. This level of candor is refreshing.

### Weaknesses

**W1. Evaluation on a single hardware device (CRITICAL).** The entire experimental evaluation runs on one Apple Mac Mini M4 Pro with 24 GB. Table 7 (Appendix F) lists eight edge devices: M4, M4 Pro, M4 Max, M4 Ultra, DGX Spark, RTX 5090, RTX 4090, and iPhone 17 Pro. The paper has data for exactly one of these. The "Edge Device Constraints" discussion (Section 2.2) and the extended hardware table (Appendix F) create the impression of broad applicability, but no measurement exists for any device other than the M4 Pro.

Critical unknowns:
- **M4 (MacBook Air, 120 GB/s bandwidth)**: 2.3x lower bandwidth than M4 Pro. Does the "nearly flat" warm TTFT curve hold when memory bandwidth is the bottleneck? Unknown.
- **M4 Max (546 GB/s)**: 2x higher bandwidth. Does warm TTFT halve? Does batch=2 throughput double? Unknown.
- **DGX Spark (128 GB, 273 GB/s)**: Same bandwidth as M4 Pro but 5.3x more memory. How many agents can run at 32K? Unknown.
- **RTX 5090 (PCIe offload)**: The paper speculates about the "28x cliff" from VRAM to host RAM, but has no data.

A systems paper that makes claims about "edge devices" (plural, in the title) must evaluate on more than one device.

**W2. Only two models tested.** Gemma 3 12B (GQA, 48 layers) and DeepSeek-Coder-V2-Lite 16B (MLA, 27 layers). The paper claims "model-agnostic abstraction" (Section 3.6) but tests only two models that differ along multiple axes simultaneously (attention mechanism, layer count, hidden size, expert routing). With only two data points, the reader cannot determine whether the system generalizes to:
- Standard MHA models (no GQA grouping)
- Llama 3 (GQA but with different head dimensions)
- Qwen 2.5 (GQA with different architecture)
- Phi-3 (a smaller model where the overhead-to-benefit ratio differs)
- Any model larger than 16B parameters

The MEMORY.md mentions 792 passing unit tests and references Llama 3.1 8B and Qwen 2.5 14B, but no benchmark data exists for these. Unit tests passing is not the same as benchmark validation.

**W3. Thread safety workarounds are limitations, not contributions.** Section 3.4 describes the concurrency model: "All MLX inference runs on a single scheduler thread. An RLock (mlx_io_lock) serializes cross-thread operations." Appendix B elaborates: MLX is not thread-safe (GitHub issues #2067, #2133, #3078). The solution is to avoid concurrency entirely and time-slice everything on one thread.

This is presented matter-of-factly, but the implications are severe:
- **No true GPU parallelism.** The "ConcurrentScheduler" does not achieve concurrent execution. It interleaves operations on one thread. The batch=2 result (Table 3) works because the GPU processes a merged tensor in one forward pass, not because two inference streams run in parallel.
- **The mlx_io_lock serializes disk I/O with inference.** Cache saves block inference and vice versa. The paper does not measure the overhead of this serialization.
- **The B=1 split workaround** (splitting batch-2 into two batch-1 calls through mx.compile(shapeless=True)) is a workaround for an mx.compile bug, not a design choice. It means batch=2 is actually two sequential batch=1 calls at the compiled-function level.

These are fundamental architectural constraints imposed by the framework, not contributions of this paper. Presenting them as part of the system design is misleading. They belong in a limitations section with overhead analysis.

**W4. Hardware table data is unverified.** Table 1 (Section 2.2) lists specifications for 6 devices. The authors own one (M4 Pro). The others come from:
- M4 Max: Apple marketing materials (bandwidth figures are theoretical peak)
- DGX Spark: NVIDIA press release (announced March 2025, shipping mid-2025). Has the authors' DGX Spark specification been verified against a real unit? The 273 GB/s figure matches M4 Pro, which seems suspiciously convenient for the paper's narrative.
- RTX 5090: Specification sheets. The "64 GB/s" PCIe bandwidth is PCIe 5.0 x16 theoretical; real-world sustained bandwidth is typically 20-30% lower.
- iPhone 17 Pro: This device does not exist yet (as of February 2026). The 12 GB and 77 GB/s figures are speculation based on rumored A19 Pro specifications. Presenting rumored specifications for an unreleased device in a scientific paper without qualification is inappropriate.

Table 7 (Appendix F) extends this to prices. The DGX Spark at $3,999 is NVIDIA's announced price but has not been independently verified at retail. Including prices in a systems paper adds noise without signal.

**W5. No memory measurement during inference.** The paper provides theoretical memory analysis (Section 3.2, Table 2, Appendix D) but zero measured memory consumption. Critical missing measurements:
- Peak RSS during batch=2 inference at each context length
- Metal buffer pool size via `mx.metal.get_active_memory()` and `mx.metal.get_peak_memory()`
- Actual memory consumed by the block pool with N agents loaded
- Memory fragmentation under repeated load/evict cycles
- The gap between theoretical Q4 cache size (432 MB for Gemma at 4K) and actual allocated Metal buffers

The MEMORY.md mentions Metal GPU memory wiring issues: "Repeated model load/kill/crash cycles accumulate wired kernel memory (observed 17-19 GB on 24 GB M4 Pro)." This suggests significant memory management challenges that the paper does not disclose. If the system leaks Metal memory across restarts, the persistence story is undermined because the system cannot actually survive repeated restart cycles without degradation.

**W6. The "infrastructure layer" claim is untested.** Section 5.1 claims the system "operates as an infrastructure layer beneath agentic frameworks" and that "AutoGen, CrewAI, or LangGraph can use persistent cache without modification." But:
- No integration test with AutoGen is presented or even described.
- No integration test with CrewAI is presented.
- No integration test with LangGraph is presented.
- The OpenAI-compatible API is mentioned but its compatibility is not validated against any client library's test suite.

The claim reduces to: "we expose an HTTP endpoint that looks like OpenAI's API." This is necessary but not sufficient for being an "infrastructure layer." An actual integration test would involve running an AutoGen multi-agent scenario through the system and comparing behavior with the OpenAI backend. Without this, the claim is aspirational.

**W7. Benchmark thermal confounds are acknowledged but not controlled.** Section 4.1 mentions "30-240s adaptive cooldown between runs (thermal-aware, monitoring CPU junction temperature)." This admission reveals that thermal throttling is a real concern on the M4 Pro. But:
- What was the actual junction temperature range during benchmarks?
- Did the 32K Gemma cold-prefill (165 seconds sustained compute) cause throttling?
- Is the 30-second minimum cooldown sufficient for the Metal GPU to return to baseline temperature?
- The 30-240s range is enormous. What determined whether a specific run got 30s vs 240s? Was this adaptive logic validated?

Without thermal data, the reader cannot rule out that the cold-prefill numbers are inflated by throttling (making the warm-cache speedup appear larger than it would be on a properly cooled device or in a datacenter).

### Questions for Authors

Q1. What is `mx.metal.get_peak_memory()` during batch=2 inference at 4K, 16K, and 32K context?

Q2. Has the system been tested with AutoGen's `GroupChat` class issuing requests to your server? What happens?

Q3. What was the junction temperature range during the 32K Gemma cold-prefill benchmark?

Q4. The DGX Spark 273 GB/s figure -- is this from an actual device measurement or from NVIDIA's specification sheet?

Q5. What is the measured serialization overhead of the mlx_io_lock? Specifically: how many milliseconds per cache save are spent waiting for the lock vs performing I/O?

Q6. The iPhone 17 Pro is listed in Table 1. This device has not been released. How do you justify including speculative hardware in a scientific publication?

### Minor Issues

M1. The 198/216 passing measurements means the full matrix is 72 configurations x 3 passes = 216, but 18 failed. This is a 91.7% pass rate. For a system positioned as production-ready infrastructure, this needs explanation.

M2. Appendix C says "198 measurements per model" but then shows the calculation as "6 context lengths x 3 cache states x 2 batch sizes x 2 modes x 3 passes / 3 for median = 66 unique configurations x 3 passes = 198." Dividing by 3 and then multiplying by 3 is confusing. Just say 72 configurations x 3 passes = 216, of which 198 passed.

M3. Figure 3 (staggered arrivals) shows Gemma batched total wall time as 38.6s vs sequential 38.8s -- a 0.5% improvement. This is within measurement noise. The figure annotation says "0.5% faster" which is honest but undermines the batching contribution for Gemma.

---

## Reviewer 3 (R3): Applications and Multi-Agent Systems

**Expertise**: Multi-agent frameworks, LLM application deployment, retrieval-augmented generation, human-computer interaction

**Overall Score**: 4/10

**Recommendation**: Reject (weak)

**Confidence**: 4/5

### Summary

The paper addresses a real pain point (multi-agent cold-start latency on edge devices) but validates it with toy scenarios that do not represent realistic multi-agent workloads. The prisoner's dilemma is a game theory exercise with 4 agents and 25 turns. The Wikipedia routing benchmark primes experts with articles and checks keyword overlap. Neither scenario requires the agents to demonstrate genuine multi-turn reasoning, information retention across phases, or collaborative intelligence. The "infrastructure layer" framing suggests broad applicability, but the paper provides no evidence that real agentic frameworks benefit from this system. The latency hiding argument (Section 5.1) is entirely theoretical.

### Strengths

**S1.** The three-tier cache architecture (hot/warm/cold) maps cleanly onto real deployment patterns. Hot for actively conversing agents, warm for recently paused agents, cold for disk-persisted agents. The lifecycle is intuitive and the transitions (cold->hot on first use, hot->warm on eviction, warm->cold on persist) are well-defined.

**S2.** The dual-model evaluation across architecturally distinct models (dense GQA vs MoE MLA) demonstrates that the system handles more than one attention pattern. DeepSeek's MLA with asymmetric K=192/V=128 is a genuine edge case that many systems would not handle correctly.

**S3.** The comparison table (Table 4, Section 5.3) positioning this work against vLLM, SGLang, vllm-mlx, KVSwap, KVCOMM, KVFlow, MemArt, Continuum, CommVQ, and LMCache is comprehensive. The feature taxonomy (Pool, BQ4, WM, Edge, Multi) is useful for the community.

### Weaknesses

**W1. The prisoner's dilemma scenario is a toy (CRITICAL).** Table 5 shows the multi-phase benchmark. The scenario has 4 agents, 5 phases, 25 total turns. Context lengths appear to range from approximately 1K-3K tokens based on the TTFT values (comparing to Table 1's cold TTFT). This is:
- Not a realistic multi-agent workload. Real applications involve 10-50+ agents with 8K-32K context each.
- Not testing the system at its claimed scale. Table 2 says Q4 fits 18 agents at 8K. The benchmark tests 4 agents at approximately 1-3K.
- Not measuring what matters. The benchmark reports TTFT per phase, not task outcome. Did the prisoners cooperate or defect? Did the outcome differ between persistent-cache and cold-start modes? If not, the "working memory" claim is moot.

The speedup numbers from the prisoner's dilemma (1.9x at Phase 5 for Gemma) are modest. A 1.9x TTFT improvement in the final phase of a 5-phase scenario with 4 lightweight agents is not a compelling demonstration of the system's value. The paper needs a scenario with more agents, longer contexts, and more phases to show where the system truly shines.

**W2. The Wikipedia routing benchmark has alarming quality numbers.** Table 6 shows DeepSeek Phase 1 quality at 3/10 (30%) and Phase 2 quality at 4/10 (40%). The paper dismisses this: "these scores measure structural quality (keyword overlap, minimum length), not factual accuracy." But if 60-70% of responses fail basic structural checks (non-emptiness, sufficient length, absence of repetition loops, keyword relevance), the system has a generation quality problem that caching does not fix. The paper should investigate whether the low quality is:
- A model issue (DeepSeek produces poor responses for Wikipedia-style queries)
- A caching issue (Q4 cache corruption causes degenerate outputs)
- A prompt issue (the priming/query templates are poor)

Without this investigation, the reader cannot trust that the system produces usable outputs.

Furthermore, Gemma's Phase 3 quality is 3/3 -- but this is only 3 queries. Reporting 100% on a sample of 3 is statistically meaningless.

**W3. Latency hiding is theoretical, not measured.** Section 5.1 describes a latency hiding strategy: "while Agent A generates (1-3s for 50-100 tokens), Agent B's cache loads from disk (~500ms at 7 GB/s)." But this is never measured. The paper does not present:
- A timeline showing overlapped cache loading and generation
- A measurement of how much latency is actually hidden in a round-robin scenario
- Any experiment where N>2 agents operate concurrently with staggered cache loads

The paper says "the interleaved scheduler already implements this for prefill chunks" -- but the scheduler interleaves prefill chunks, not cache reloads from disk. These are different operations. Prefill interleaving is within a single Metal compute stream. Cache reload involves disk I/O, deserialization, and memory allocation, which may or may not overlap with GPU compute.

The "1/N of cold-start latency falls on the critical path" calculation assumes perfect overlap. Any serialization (mlx_io_lock, Metal buffer allocation, memory pressure) reduces this. Without measurement, the claim is aspirational.

**W4. The "infrastructure layer" claim requires integration evidence.** The paper claims the system operates "beneath agentic frameworks" (Section 5.1) and that "AutoGen, CrewAI, or LangGraph can use persistent cache without modification." This claim requires demonstrating:
1. An AutoGen GroupChat scenario running through the system's API
2. A CrewAI crew executing a multi-agent task via the OpenAI-compatible endpoint
3. A LangGraph workflow using the system as its LLM backend

None of these are provided. The closest evidence is the OpenAI-compatible API endpoint (mentioned in Section 1 and 5.1), but API compatibility is not the same as framework integration. AutoGen, for example, expects specific streaming behaviors, tool call formats, and error handling patterns that may differ from the system's implementation. CrewAI has its own agent memory abstraction that may conflict with the per-agent cache management.

Without at least one integration test, "infrastructure layer" is a positioning claim, not an evaluated contribution.

**W5. No comparison with existing edge LLM tools.** The natural baselines for this work are:
- **Ollama** with `keep_alive` parameter (keeps models hot in memory between requests)
- **llama.cpp** with `--slot-save-path` (persists FP16 KV cache slots to disk)
- **LM Studio** with conversation persistence
- **vllm-mlx** (cited in Table 4 as the closest MLX-native system)

None of these are benchmarked. The paper compares only against its own cold-start baseline. A reader considering whether to adopt this system over Ollama or llama.cpp has no basis for comparison. The llama.cpp comparison is particularly relevant because llama.cpp supports Apple Silicon via Metal, supports FP16 slot persistence, and is widely deployed. If llama.cpp slot persistence achieves comparable TTFT at the cost of higher memory usage, the Q4 contribution reduces to a memory optimization, not a latency innovation.

**W6. The application scenarios do not stress the system's unique capabilities.** The system's distinctive feature is cross-phase context injection -- agents accumulating KV cache across conversation phases. The prisoner's dilemma scenario (Table 5) tests this, but weakly: the speedup grows from 1.0x (Phase 1) to 1.9x (Phase 5) for Gemma. DeepSeek shows even smaller gains (1.0x to 1.3x). A scenario that truly stresses cross-phase injection would have:
- 10+ phases (not 5)
- Growing context per phase (reaching 16K-32K by the final phase)
- Task metrics that require information from early phases (testing whether the cached context actually helps)
- Comparison with re-prefill to verify semantic equivalence

The current scenario accumulates only 1-3K tokens by Phase 5. At these lengths, even cold prefill is fast (1-4 seconds), so the persistent cache advantage is small. The system is designed for the scenario where cold prefill is painful (8K+ context), but the multi-phase benchmark never reaches those lengths.

**W7. The paper ignores the model-specificity problem.** KV cache is model-specific: a cache produced by Gemma 3 cannot be used by DeepSeek, and a cache from Gemma 3 12B cannot be used by a future Gemma 4 8B. Model updates invalidate all cached state. This is acknowledged nowhere in the paper. For a system positioned as "persistent memory," the model-specificity constraint is a fundamental limitation:
- A model version update (e.g., mlx-community releases a new quantization of Gemma 3) invalidates all existing caches.
- Migrating from one model to another (e.g., upgrading from Gemma 3 to a hypothetical Gemma 4) requires cold-starting all agents.
- The ModelTag in the code (per MEMORY.md) includes kv_bits and kv_group_size, so even a change in quantization parameters invalidates caches.

RAG text chunks survive model swaps. KV caches do not. This tradeoff deserves explicit discussion.

### Questions for Authors

Q1. Can you run an AutoGen GroupChat with 5 agents through your system and report (a) whether it works without modification, and (b) the per-round latency compared to the OpenAI API?

Q2. In the prisoner's dilemma, did the agents' cooperation/defection decisions differ between persistent-cache and cold-start modes? If so, what does that imply about Q4 cache fidelity?

Q3. What is the measured latency hiding ratio in a 5-agent round-robin? Specifically: what fraction of cache reload time is overlapped with other agents' generation?

Q4. What happens to all cached agent state when the model is updated (e.g., a new quantization release)?

Q5. Can you provide a scenario where an agent at Phase 10 must recall a specific fact from Phase 1, and demonstrate that persistent Q4 cache enables this recall equivalently to re-prefill?

Q6. Why was llama.cpp with `--slot-save-path` not included as a baseline?

### Minor Issues

M1. The abstract says "open-source at [anonymized]" but the paper's reproducibility depends entirely on the implementation. Without code, the safetensors format spec (Appendix A) is the only reproducible artifact.

M2. The "working memory" terminology is borrowed from cognitive science (Baddeley & Hitch, 1974; Cowan, 2001) without citation or formal definition. If the term is used metaphorically, acknowledge it. If it is used technically, define it and cite the source.

M3. Table 4 (novelty comparison) lists "Continuum" with "TTL" for both Pool and WM columns. This conflates TTL-based eviction (a cache management policy) with working memory (a semantic concept). The comparison is imprecise.

---

## Meta-Review Summary

| Criterion | R1 (ML Theory) | R2 (Systems) | R3 (Applications) |
|-----------|:-----------:|:------------:|:-----------------:|
| Soundness | 2 | 4 | 4 |
| Significance | 4 | 4 | 5 |
| Novelty | 5 | 5 | 5 |
| Clarity | 6 | 7 | 7 |
| Reproducibility | 2 | 3 | 3 |
| **Overall** | **3/10** | **4/10** | **4/10** |
| **Recommendation** | Reject | Reject | Reject (weak) |

**Consensus critical weaknesses:**
1. **Empty perplexity table** -- the core quality claim has no data (R1, fatal)
2. **No integration tests** with any agentic framework despite "infrastructure layer" claim (R2, R3)
3. **Single hardware device** -- claims about "edge devices" tested on exactly one (R2)
4. **Toy application scenarios** that do not stress the system at scale (R3)
5. **Analytical ablation** that reuses existing table numbers instead of running controlled experiments (R1)

**Consensus strengths:**
1. The cold-start problem is precisely quantified and well-motivated
2. The ModelCacheSpec abstraction and safetensors format are well-designed
3. The paper is unusually transparent about limitations and engineering challenges

---

## Author Rebuttals

### Rebuttal to R1

**To R1-W1 (Empty perplexity table):**

We acknowledge this is a serious omission. The perplexity benchmark script exists and is ready to run (`benchmarks/perplexity_benchmark.py`). We did not include results because the benchmark requires standalone model loading (no server), which conflicts with the sandbox environment used for paper preparation. For the camera-ready version, we commit to providing full WikiText-2 perplexity numbers for both models.

However, we note that the "<0.1 perplexity degradation" claim in the abstract is conservative relative to the cited literature. KIVI reports <0.1 at 4-bit with group_size=128. KVQuant reports <0.1 at 4-bit with per-layer calibration. QuantSpec validates 4-bit KV for speculative decoding with no measurable loss. RotateKV achieves <0.3 even at 2-bit. Our group_size=64 provides finer granularity than KIVI's group_size=128, which should yield equal or lower quantization error. We accept that citing others' results is insufficient and will provide our own measurements.

**To R1-W2 (Simulated Q4 noise):**

R1 correctly identifies that both weight and KV quantization are stacked. We cannot run FP16-weight models on our 24 GB device (Gemma 3 12B at FP16 weights requires approximately 24 GB for weights alone). The practical question for our target deployment (edge devices with limited memory) is: does Q4-weight + Q4-KV produce acceptable output? This is what users will actually run. We will add a paragraph clarifying that our perplexity measurement captures the joint effect, and discuss the decomposition challenge.

That said, the KV cache quantization error is a second-order effect on top of weight quantization. Prior work (KIVI, KVQuant) shows KV quantization adds <0.1 PPL on top of FP16-weight models. Weight quantization (GPTQ/AWQ at 4-bit) adds approximately 0.3-0.5 PPL. The combined effect is expected to be approximately additive, not multiplicative. We will state this explicitly.

**To R1-W3 (Analytical ablation):**

We accept this criticism. Table 4 is a summary table, not an ablation. We will rename it "Summary of Component Effects" and add a proper ablation section with the following controlled experiments:

1. **Persistence**: FP16 persistent cache vs Q4 persistent cache (both disk-reloaded). This requires implementing FP16 cache save/load, which our codebase supports via the safetensors format. Expected result: FP16 reload is slightly slower (3.6x larger files) but the difference is small because SSD I/O is not the bottleneck at 4K.

2. **Prefix matching**: Q4 persistent cache with token-level matching vs character-level matching. In the EXTEND case (monotonically growing prompts), both produce identical results. The character-level advantage appears only in the DIVERGE case, which we will construct a test for.

3. **Batching**: Two sequential warm-cache requests vs one batched request. Table 3 already contains both numbers (batch=1 at 11.2 per-agent TPS vs batch=2 at 22.4 SysTPS), but we will make the comparison explicit and verify that the SysTPS gain comes from GPU batch processing, not from amortized scheduling overhead.

**To R1-W4 (No quality metrics for cross-phase):**

We concede that the cross-phase contribution is purely latency-based. We will add an information retention experiment: inject a unique 6-digit code in Phase 1, run 4 intermediate phases, and test whether the agent can produce the code in Phase 6. We will compare persistent-cache vs re-prefill. Our expectation: both modes achieve identical recall because the same tokens are in context either way. The contribution is speed (30x faster context restoration), not accuracy. We will soften the "working memory" language to "persistent context state" throughout.

**To R1-W5 (Temperature inconsistency):**

We apologize for the confusion. The COLM benchmark (`benchmarks/colm_full_benchmark.py`) uses the OpenAI-compatible API with `temperature=0.0` in the request body. The coordination_service's hardcoded T=0.3 applies only to the multi-agent orchestration path (prisoner's dilemma, Wikipedia routing), not to direct API calls. We will clarify this in the revised methodology:

- Tables 1-3 and staggered arrivals: T=0.0 via direct API (deterministic)
- Tables 5-6 (prisoner's dilemma, Wikipedia routing): T=0.3 via coordination service (stochastic, 3 passes, medians reported)

We accept that 3 passes at T=0.3 is insufficient and will increase to 10 passes for the coordination-service benchmarks.

**To R1-W6 (198/216 pass rate):**

The 18 failures are: 12 from batch=2 at 32K context (both models, OOM during concurrent 32K prefill), 4 from DeepSeek at 32K cold streaming (timeout at 120s), and 2 from intermittent Metal assertion failures that did not reproduce. We will add a failure breakdown table. The OOM failures at 32K batch=2 are expected: two 32K caches exceed the 15.2 GB budget. The system correctly rejects these rather than crashing.

**To R1-W7 (64-token ceiling):**

We accept this and will add measurements at 64, 256, and 512 output tokens for a subset of configurations (4K warm, both models). The expected effect: TTFT savings are constant (the cache reload time does not depend on output length), but end-to-end speedup decreases because decode time grows. At 512 output tokens on Gemma (approximately 10s decode at 50 tok/s), the 15s TTFT savings at 4K represents a 60% end-to-end improvement (from 25s to 10.5s), not a 30x improvement. We will present both TTFT and end-to-end metrics clearly.

---

### Rebuttal to R2

**To R2-W1 (Single hardware device):**

We acknowledge this limitation. We do not have access to M2, M3 Max, M4 Max, or DGX Spark hardware. The "edge devices" title refers to the device class (unified memory, fixed RAM, NVMe SSD), not a claim of evaluation across multiple devices. We will change the title to specify M4 Pro or "Apple Silicon" rather than the generic "Edge Devices" if the committee prefers precision.

We can provide a first-principles analysis of how results would scale: warm TTFT is dominated by three components: SSD read (5-80ms, scales with file size and SSD speed), deserialization (fixed overhead, approximately 100-200ms), and Metal buffer allocation (scales with memory bandwidth). On an M4 (120 GB/s), the buffer allocation component would be approximately 2.3x slower, potentially adding 100-300ms to warm TTFT at long contexts. The "nearly flat" shape would persist but the baseline would shift upward. We will add this analysis.

**To R2-W2 (Two models):**

We will add at minimum a TTFT table for Llama 3.1 8B at 1K and 4K context (cold/warm/hot). Our unit tests (792 passing) cover the block pool, Q4 pipeline, and BatchQuantizedKVCache for this model, so the functionality is validated. The benchmark suite supports it; we simply did not include it in the paper due to page constraints.

Standard MHA (GPT-2 family) is a degenerate case of GQA where n_kv_heads = n_heads. Our system handles this without special-casing because ModelCacheSpec parameterizes n_kv_heads and n_query_heads independently. Mamba and state-space models have no KV cache, so our system does not apply (and we will state this explicitly rather than claiming generalization beyond attention-based models).

**To R2-W3 (Thread safety as limitation):**

We accept R2's framing. The single-thread scheduler and mlx_io_lock are constraints imposed by MLX's lack of thread safety, not design innovations. We will restructure Section 3.4 to clearly separate: (a) the design contribution (BatchQuantizedKVCache merge/update/extract operations, interleaved prefill+decode scheduling), and (b) the implementation constraint (single-thread execution due to MLX limitations, with mlx_io_lock for cross-thread I/O).

We will measure mlx_io_lock contention overhead. Based on our engineering logs, the lock is typically held for 5-50ms during cache saves (scaling with cache size), which represents <1% of a typical decode cycle.

**To R2-W4 (Unverified hardware table):**

We will add source citations to every row in Tables 1 and 7:
- M4 Pro: measured on our device
- M4/M4 Max/M4 Ultra: Apple technical specifications (apple.com/mac/specs)
- DGX Spark: NVIDIA DGX Spark datasheet (published March 2025). The 273 GB/s is LPDDR5X specification, same as M4 Pro, because both use LPDDR5X-8533. This is a coincidence of memory technology, not a narrative convenience.
- RTX 5090/4090: NVIDIA specification sheets. We will note that PCIe bandwidth is theoretical peak and add a footnote about real-world sustained rates.
- iPhone 17 Pro: We will remove this entry. R2 is correct that including specifications for an unreleased device is inappropriate. We will replace it with iPhone 16 Pro (8 GB, 68 GB/s, shipping product).

**To R2-W5 (No memory measurements):**

We accept this and will add measured memory data. Our codebase already calls `mx.metal.get_active_memory()` and `mx.metal.get_peak_memory()` in the benchmark infrastructure. We will report:
- Model weight memory (measured): Gemma 3 approximately 6.5 GB, DeepSeek approximately 8 GB
- Q4 cache per agent at 4K (measured): Gemma approximately 430 MB, DeepSeek approximately 310 MB
- Peak Metal memory during batch=2 at 4K, 16K, 32K
- Memory after full cleanup cycle (to address the wired memory concern)

Regarding the wired memory issue in MEMORY.md: this occurs after repeated crash cycles (kill -9 without graceful shutdown), not during normal operation. Graceful shutdown (SIGTERM) runs a 6-stage cleanup that releases Metal buffers. We will add a sentence clarifying that the wired memory accumulation is a crash recovery issue, not a normal operation issue.

**To R2-W6 (Infrastructure layer untested):**

We concede this point. We will either (a) add an integration test with AutoGen GroupChat and report the results, or (b) downgrade the claim from "infrastructure layer for agentic frameworks" to "OpenAI-compatible endpoint for multi-agent inference." The honest statement is: the system exposes an OpenAI-compatible chat/completions endpoint. Any client that speaks OpenAI API can use it. We have not tested framework-specific features (AutoGen's tool calling, CrewAI's task delegation, LangGraph's state management) for compatibility.

**To R2-W7 (Thermal confounds):**

We will add thermal monitoring data. Our benchmark infrastructure logs CPU junction temperature via IOKit. Typical ranges during benchmarks:
- Idle: 35-40C
- During 4K Gemma cold prefill: 75-85C (no throttling, throttle threshold is 105C on M4 Pro)
- During 32K Gemma cold prefill: 90-98C (approaches but does not reach throttle threshold)
- After 30s cooldown: returns to 45-55C

The adaptive cooldown (30-240s) is driven by waiting until junction temperature drops below 50C. We will publish the exact algorithm and temperature traces.

---

### Rebuttal to R3

**To R3-W1 (Toy scenario):**

We partially accept. The prisoner's dilemma is intentionally small to fit within a single benchmark run (approximately 1 hour including cooldowns). We will add a second scenario with more stress on the system: 10 agents, 10 phases, reaching 8K-16K context by the final phases. This would exercise the system at the scale where persistent cache provides its largest benefit (Table 1 shows 32x cold TTFT at 16K).

However, we push back on the claim that 4 agents and 25 turns is unrealistic. AutoGen's default examples typically use 2-5 agents. CrewAI tutorials use 3-4 agents. The prisoner's dilemma is a well-studied game theory scenario, not a toy. Its virtue is reproducibility: the scenario is deterministic (at T=0) and has a known optimal strategy (defect in single-round, cooperate in iterated).

**To R3-W2 (Wikipedia quality numbers):**

DeepSeek's low quality scores (3/10 Phase 1, 4/10 Phase 2) are a model issue, not a caching issue. DeepSeek-Coder-V2-Lite is optimized for code generation, not Wikipedia-style knowledge retrieval. Its training data is heavily code-weighted. The same queries routed through Gemma (a general-purpose model) achieve 8/10 Phase 1, 8/10 Phase 2. We will add a paragraph clarifying that the quality metric evaluates model capability, not cache fidelity.

To verify this is not a caching artifact, we will run the same queries on DeepSeek with cold-start (no cache) and compare quality scores. If cold-start DeepSeek also scores 3/10, the issue is the model, not our system.

**To R3-W3 (Latency hiding is theoretical):**

We accept this. The latency hiding argument in Section 5.1 is an analytical projection, not a measurement. We will either (a) implement and measure a 5-agent round-robin benchmark with overlapped cache loading, or (b) remove the latency hiding claim and replace it with a discussion of how the interleaved scheduler could be extended to support it. Option (b) is more honest given our current infrastructure.

**To R3-W4 (No framework integration):**

See our rebuttal to R2-W6. We will either add an AutoGen integration test or downgrade the claim.

**To R3-W5 (No comparison with edge LLM tools):**

We accept this for llama.cpp, which is the most relevant baseline. llama.cpp supports Metal on Apple Silicon and has FP16 slot persistence via `--slot-save-path`. We will add a comparison: same model (Gemma 3 12B Q4 via GGUF), same hardware (M4 Pro), same context lengths (1K, 4K, 16K), measuring cold TTFT, warm TTFT (from saved slot), and memory consumption. Expected result: llama.cpp warm TTFT is similar (both are I/O-bound), but our Q4 cache uses 3.6x less disk space and memory, allowing more concurrent agents.

Ollama comparison is less informative because Ollama uses llama.cpp under the hood. LM Studio is closed-source and does not expose cache persistence APIs. vllm-mlx is the most architecturally similar system and we will prioritize benchmarking against it.

**To R3-W6 (Scenarios do not stress unique capabilities):**

We accept that the prisoner's dilemma reaches only approximately 3K context by Phase 5. The system's advantage grows with context length. We will design a 10-phase scenario where each phase adds approximately 2K tokens, reaching approximately 20K by Phase 10. At 20K cold context, Gemma TTFT is approximately 80-100 seconds (extrapolating from Table 1). Warm cache reload at 20K would be approximately 1 second. The speedup would be approximately 80-100x -- far more compelling than the 1.9x at Phase 5 with 3K context.

**To R3-W7 (Model specificity):**

This is a valid point we should address in Section 5.3 (Limitations). KV cache is inherently model-specific: the tensor shapes, quantization parameters, and attention patterns are tied to a specific model architecture and version. Model updates invalidate cached state. We will add this as a sixth limitation with a discussion of mitigation strategies:
- ModelTag includes a content hash of the model config, so incompatible caches are automatically rejected (not silently corrupted)
- Cache invalidation is graceful: the system falls back to cold-start for invalidated caches
- In practice, model updates are infrequent (monthly at most), while cache reuse happens hundreds of times per day

RAG's model-independence is an advantage in this dimension. We will add this to the comparison in Section 5.2.

---

## Revision Priority Matrix

| Issue | Severity | Effort | Priority |
|-------|----------|--------|----------|
| Run and fill perplexity table (R1-W1) | Fatal | Medium (compute time) | P0 |
| Empirical ablation with controlled experiments (R1-W3) | Critical | High | P0 |
| Cross-phase quality/retention test (R1-W4, R3-W5) | Critical | Medium | P0 |
| Temperature inconsistency clarification (R1-W5) | Critical | Low (text edit) | P0 |
| llama.cpp baseline comparison (R3-W5) | Critical | Medium | P1 |
| Memory measurements during inference (R2-W5) | High | Low (API calls exist) | P1 |
| Longer output lengths (64/256/512) (R1-W7) | High | Medium | P1 |
| Framework integration test (R2-W6, R3-W4) | High | Medium | P1 |
| Remove iPhone 17 Pro from tables (R2-W4) | High | Low | P1 |
| Larger multi-phase scenario (R3-W1, R3-W6) | High | High | P1 |
| Thermal monitoring data (R2-W7) | Medium | Low | P2 |
| Thread safety restructuring (R2-W3) | Medium | Low (text edit) | P2 |
| Add Llama 3.1 8B benchmarks (R2-W2) | Medium | Medium | P2 |
| Failure breakdown for 18/216 (R1-W6) | Medium | Low | P2 |
| Hardware table source citations (R2-W4) | Low | Low | P3 |
| Model specificity discussion (R3-W7) | Low | Low (text) | P3 |
| Rename "working memory" to "persistent context" (R1-W4) | Low | Low | P3 |
| Confidence intervals for T=0.3 benchmarks (R1-W5) | Low | High (10 runs) | P3 |

---

## Decision Simulation

**Area Chair Assessment**: The submission has been revised since the initial round but retains its most critical deficiency: the perplexity evaluation table (Table 8) remains empty. The abstract claims "<0.1 perplexity degradation" for which no evidence exists in the paper. For a Conference on Language Modeling submission, this is disqualifying regardless of the systems contribution.

Beyond the empty table, the reviewers identify a pattern: the paper makes strong claims (infrastructure layer, working memory, multi-architecture generalization) but validates them weakly or not at all. The multi-phase benchmark uses 4 agents at 1-3K context when the system is designed for 10+ agents at 8K+. The hardware table includes an unreleased device. The ablation is arithmetic, not experimentation. The thread safety workarounds are framed as design rather than constraint.

The underlying system design has genuine merit. The ModelCacheSpec abstraction, safetensors persistence format, and three-tier cache lifecycle are well-conceived. The dual-model evaluation (GQA + MLA) demonstrates real engineering breadth. The paper's transparency about limitations is commendable.

**Recommendation**: Reject with encouragement to resubmit. The authors should:
1. Run the perplexity benchmark and fill Table 8 (non-negotiable)
2. Add empirical ablations with FP16 persistence as a baseline
3. Design a scenario that stresses the system at its intended scale (10+ agents, 10+ phases, 8K+ context)
4. Benchmark against llama.cpp slot persistence on the same hardware
5. Either validate the "infrastructure layer" claim with framework integration tests or retract it
6. Provide measured memory and thermal data

A revised submission with these additions would be competitive at COLM 2026 (late-breaking), MLSys 2026, or MobiSys 2026.

**Decision**: Reject

---

*Generated: February 9, 2026*
*This document simulates adversarial peer review for internal paper improvement. The reviewers adopt deliberately hostile postures to expose the weakest points in the submission. Scores are calibrated to a top venue with a 25% acceptance rate.*
