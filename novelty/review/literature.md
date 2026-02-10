# Expanded Annotated Bibliography

**Paper:** "Agent Memory Below the Prompt: Persistent Q4 KV Cache for Multi-Agent LLM Inference on Edge Devices"
**Target venue:** COLM 2026
**Last updated:** 2026-02-09

This document covers every reference in the paper's bibliography (`semantic_colm2026.bib`), plus references present in the bib file but not cited in the main text. Each entry includes a 1--2 sentence summary, its relevance to the paper, and how the paper cites it. The final section identifies coverage gaps -- areas the paper does not cite but arguably should.

---

## 1. KV Cache Management (7 references)

### vLLM / PagedAttention
**Cite key:** `kwon2023pagedattention`
**Full title:** Efficient Memory Management for Large Language Model Serving with PagedAttention
**Authors:** Kwon, Li, Zhuang, Sheng, Zheng, Yu, Gonzalez, Zhang, Stoica
**Venue:** SOSP 2023

**Summary:** Introduces paged virtual-memory-style KV cache allocation that eliminates memory fragmentation and enables near-zero-waste sharing during parallel sampling, achieving 2--4x throughput improvement over existing systems.

**Relevance:** The block-based allocation concept directly inspired the paper's `BlockPoolBatchEngine`. However, vLLM discards caches after request completion and targets datacenter GPUs with discrete VRAM, not unified memory. The paper's contribution is making blocks persistent across sessions and per-agent isolated.

**How cited:** Referenced in Introduction (Sec 1) as a system that discards cache after each request; in the novelty comparison table (Table 6, "Paged" pool); and in Related Work (Sec 6) as achieving 2--4x throughput. Also mentioned for token-ID-based prefix matching that breaks under context-dependent BPE.

---

### SGLang / RadixAttention
**Cite key:** `zheng2024sglang`
**Full title:** SGLang: Efficient Execution of Structured Language Model Programs
**Authors:** Zheng, Yin, Xie, Huang, Sun, Yu, Cao, Kozyrakis, Stoica, Gonzalez, Barrett, Sheng
**Venue:** NeurIPS 2024

**Summary:** Uses a radix tree to maintain and automatically reuse KV cache prefixes across structured generation programs, achieving 5x throughput improvement for multi-call LLM programs.

**Relevance:** SGLang's radix-tree prefix sharing is the closest datacenter analog to the paper's character-level prefix matching. The paper argues its approach is more robust because it compares raw text rather than token IDs (avoiding BPE context-dependence). SGLang also discards cache after request completion.

**How cited:** Referenced alongside vLLM in Introduction and Related Work; in Table 6 ("Radix" pool). Mentioned for token-ID prefix caching that the paper's character-level approach improves upon.

---

### vllm-mlx
**Cite key:** `barrios2026vllmmlx`
**Full title:** Native LLM and MLLM Inference at Scale on Apple Silicon
**Authors:** Barrios, Wayner
**Venue:** arXiv:2601.19139, January 2026

**Summary:** Ports vLLM's continuous batching and prefix caching to Apple Silicon via MLX, achieving 21--87% higher throughput than llama.cpp across models up to 30B parameters.

**Relevance:** The closest existing system on the same hardware platform (Apple Silicon + MLX). The paper positions itself as adding what vllm-mlx lacks: per-agent isolation, disk persistence across sessions, Q4 KV pipeline, and multi-agent coordination.

**How cited:** In Table 6 ("Prefix" pool, no persistence, no batched Q4); in Related Work as MLX-native prefix caching without cross-session persistence; in the novelty discussion as one of the two closest systems.

---

### LMCache
**Cite key:** `lmcache2025`
**Full title:** LMCache: Engine-Agnostic Persistent KV Store for LLM Serving
**Authors:** LMCache Team
**Venue:** Tech report, lmcache.ai, 2025

**Summary:** First open-source library for multi-tier KV cache storage (GPU -> CPU -> SSD -> S3) with engine-agnostic integration for vLLM and SGLang backends in cloud deployments.

**Relevance:** Validates the multi-tier storage pattern (the paper's hot/warm/cold tiers mirror LMCache's GPU/CPU/disk/remote tiers). LMCache targets enterprise cloud deployments; the paper targets single-device edge. LMCache does not perform Q4 quantization at the cache level.

**How cited:** In Table 6 ("Chunk" pool); in Related Work as engine-agnostic persistent KV storage with tiered offloading for cloud.

---

### Continuum
**Cite key:** `li2025continuum`
**Full title:** Continuum: Efficient and Robust Multi-Turn LLM Agent Scheduling with KV Cache Time-to-Live
**Authors:** Li, Mang, He, Zhang, Mao, Chen, Zhou, Cheung, Gonzalez, Stoica
**Venue:** arXiv:2511.02230, November 2025

**Summary:** Assigns TTL values to cached KV entries for multi-turn agent scheduling, selectively retaining caches in GPU memory between tool-call turns. Achieves 2.7x TTFT reduction on SWE-Bench with Llama-3.1.

**Relevance:** Addresses the same multi-turn agent scheduling problem but in a datacenter GPU setting. The paper's warm-tier metadata tracking serves a similar purpose to TTL-based pinning, but within a unified memory budget.

**How cited:** In Table 6 ("TTL" pool and working memory); in Related Work for multi-turn scheduling with 2.7x TTFT reduction.

---

### DistServe
**Cite key:** `zhong2024distserve`
**Full title:** DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving
**Authors:** Zhong, Liu, Chen, Hu, Zhu, Liu, Jin, Zhang
**Venue:** OSDI 2024

**Summary:** Disaggregates prefill and decode phases across separate GPU clusters to optimize goodput by removing interference between compute-bound prefill and memory-bandwidth-bound decode.

**Relevance:** The prefill-decode separation pattern influenced the paper's ConcurrentScheduler, which interleaves chunked prefill with decode steps in a single-node setting. DistServe operates at datacenter scale; the paper applies the principle to a single device.

**How cited:** In Related Work alongside Sarathi-Serve as disaggregating prefill/decode for datacenter-scale throughput.

---

### Sarathi-Serve
**Cite key:** `agrawal2024sarathi`
**Full title:** Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve
**Authors:** Agrawal, Kedia, Panwar, Mohan, Kwatra, Gulavani, Tumanov, Ramjee
**Venue:** OSDI 2024

**Summary:** Uses chunked prefill to pipeline prefill and decode operations within a single iteration, reducing decode latency variance while maintaining throughput in datacenter GPU settings.

**Relevance:** The chunked prefill concept (breaking long prefills into 256-token chunks) is directly used in the paper's ConcurrentScheduler for interleaved prefill+decode. Sarathi-Serve targets datacenter GPUs; the paper applies chunked prefill on a single Apple Silicon device.

**How cited:** In Related Work alongside DistServe as disaggregating prefill/decode for datacenter-scale throughput.

---

## 2. KV Cache Compression (8 references)

### KIVI
**Cite key:** `liu2024kivi`
**Full title:** KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache
**Authors:** Liu, Yuan, Jin, Zhong, Xu, Braverman, Chen, Hu
**Venue:** ICML 2024

**Summary:** Demonstrates that keys and values have different optimal quantization strategies (per-channel for keys, per-token for values), achieving 2.6x memory reduction at 2-bit with less than 0.1 perplexity degradation.

**Relevance:** Establishes the baseline quality result that 4-bit KV quantization is nearly lossless. The paper cites KIVI's perplexity numbers to support its own Q4 quality claim. The paper uses uniform group quantization (scales + biases) rather than KIVI's asymmetric channel/token approach, driven by MLX's `mx.quantized_matmul` API.

**How cited:** In Limitations (Sec 5.5) for Q4 quality justification; in Appendix C (perplexity) for <0.1 PPL degradation at 4-bit.

---

### KVQuant
**Cite key:** `hooper2024kvquant`
**Full title:** KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization
**Authors:** Hooper, Kim, Mohammadzadeh, Mahoney, Shao, Keutzer, Gholami
**Venue:** NeurIPS 2024

**Summary:** Introduces per-layer sensitivity-weighted non-uniform datatypes for KV cache quantization, enabling 10M token context on A100-80GB with less than 0.1 perplexity degradation at 3-bit.

**Relevance:** Provides per-layer sensitivity analysis that informs understanding of why not all layers tolerate quantization equally -- relevant to the paper's MLA-aware spec extractor for DeepSeek's asymmetric K=192/V=128. Also cited for quality justification.

**How cited:** In Limitations alongside KIVI for Q4 quality; in Appendix C for <0.1 PPL degradation; in Related Work for per-layer sensitivity analysis enabling 10M context.

---

### CommVQ
**Cite key:** `li2025commvq`
**Full title:** CommVQ: Commutative Vector Quantization for KV Cache Compression
**Authors:** Li, Zhang, Zhao, Zheng, Yang, Ding
**Venue:** ICML 2025

**Summary:** Achieves 87.5% KV cache size reduction at 2-bit using vector quantization codebooks that commute with RoPE (rotary positional embedding), eliminating the need to dequantize before applying position encoding.

**Relevance:** Represents a more aggressive compression approach (2-bit VQ vs the paper's 4-bit group quantization). The paper mentions CommVQ as a future direction for adaptive bit-width in the Conclusion. CommVQ's RoPE-commutative property is architecturally elegant but requires codebook overhead that MLX's quantized matmul does not support natively.

**How cited:** In Table 6 ("2bit" BQ4 entry); in Related Work for 87.5% reduction at 2 bits; in Conclusion as a future direction alongside RotateKV.

---

### QuantSpec
**Cite key:** `li2025quantspec`
**Full title:** QuantSpec: Self-Speculative Decoding with Hierarchical Quantized KV Cache
**Authors:** Li, Tomar, Gholami
**Venue:** ICML 2025

**Summary:** Combines speculative decoding with hierarchical INT4/INT8 KV cache switching, using the quantized cache for draft generation and full-precision for verification. Achieves 2.5x speedup with >90% acceptance rate.

**Relevance:** Validates that 4-bit KV cache produces output of sufficient quality to drive speculative decoding drafts -- a strong indirect validation of the paper's Q4 pipeline quality. The paper mentions QuantSpec in Appendix C for quality justification.

**How cited:** In Related Work for validating 4-bit cache quality in speculative decoding; in Appendix C for no measurable quality loss at 4 bits.

---

### RotateKV
**Cite key:** `rotatekv2025`
**Full title:** RotateKV: Accurate and Robust 2-Bit KV Cache Quantization for LLMs via Outlier-Aware Adaptive Rotations
**Authors:** Tao et al.
**Venue:** IJCAI 2025

**Summary:** Uses outlier-aware rotation transformations to redistribute extreme values across channels before quantization, enabling accurate 2-bit KV cache with less than 0.3 perplexity degradation.

**Relevance:** Demonstrates that even 2-bit is feasible with careful handling of outliers. The paper cites RotateKV in the Conclusion as a future direction for adaptive bit-width and in Appendix C for quality numbers at 2 bits.

**How cited:** In Appendix C for <0.3 PPL degradation at 2 bits; in Conclusion as a 2-bit technique for future work.

---

### MiniKV
**Cite key:** `minikv2025`
**Full title:** MiniKV: Pushing the Limits of LLM Inference via 2-Bit Layer-Discriminative KV Cache
**Authors:** Shi et al.
**Venue:** ACL 2025 Findings

**Summary:** Applies layer-discriminative quantization to KV cache, assigning different bit widths to different layers based on their sensitivity, pushing to 2-bit for less-sensitive layers.

**Relevance:** MiniKV is in the bib file but **not cited in the paper text**. It would strengthen the compression-related discussion by showing that layer-discriminative bit allocation is a viable strategy -- relevant to the paper's model-agnostic abstraction that could theoretically support per-layer bit widths.

**How cited:** Present in bib file only. Not cited in the paper.

---

### CacheGen
**Cite key:** `liu2024cachegen`
**Full title:** CacheGen: KV Cache Compression and Streaming for Fast Large Language Model Serving
**Authors:** Liu, Li, Cheng, Ray, Huang, Zhang, Du, Yao, Lu, Ananthanarayanan, Maire, Hoffmann, Holtzman, Jiang
**Venue:** ACM SIGCOMM 2024

**Summary:** Encodes KV caches into compact bitstreams for network transmission, achieving 3.5--4.3x compression for cache-aware serving over wide-area networks.

**Relevance:** CacheGen is in the bib file but **not cited in the paper text**. Its approach of treating KV cache as a serializable, compressible artifact is conceptually aligned with the paper's safetensors-based cold storage, though CacheGen uses custom bitstream encoding for network transmission rather than on-disk persistence.

**How cited:** Present in bib file only. Not cited in the paper.

---

### XQuant
**Cite key:** `tomar2025xquant`
**Full title:** XQuant: Breaking the Memory Wall for LLM Inference with KV Cache Rematerialization
**Authors:** Tomar, Hooper, Lee, Xi, Tiwari, Kang, Manolache, Mahoney, Keutzer, Gholami
**Venue:** arXiv:2508.10395, August 2025

**Summary:** Instead of caching KV pairs, caches layer input activations X and rematerializes K/V on-the-fly, achieving 10--12.5x memory savings vs FP16 with near-zero accuracy loss by trading compute for memory.

**Relevance:** Represents a fundamentally different approach -- rematerialization rather than compression. The paper cites XQuant in Appendix C for quality justification (<0.05 PPL degradation at 4-bit with rematerialization).

**How cited:** In Appendix C for <0.05 PPL degradation using rematerialization at 4 bits.

---

## 3. Model Weight Quantization (2 references) -- NEW

### GPTQ
**Cite key:** `frantar2023gptq`
**Full title:** GPTQ: Accurate Post-Training Quantization for Generative Pre-Trained Transformers
**Authors:** Frantar, Ashkboos, Hoefler, Alistarh
**Venue:** ICLR 2023

**Summary:** Post-training one-shot weight quantization using approximate second-order (Hessian-based) methods, achieving 4-bit quantization of GPT-scale models with less than 0.5 perplexity degradation.

**Relevance:** Both evaluation models (Gemma 3 12B and DeepSeek-Coder-V2-Lite 16B) use 4-bit quantized weights via mlx-lm's GPTQ-style quantization. The perplexity numbers measured include both weight quantization (GPTQ) and KV cache quantization (Q4 pipeline) effects, making GPTQ essential context for interpreting quality results.

**How cited:** In Appendix C (perplexity) for <0.5 PPL degradation on LLaMA models at 4-bit weights. Used to contextualize total quantization impact.

---

### AWQ
**Cite key:** `lin2024awq`
**Full title:** AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration
**Authors:** Lin, Tang, Tang, Yang, Chen, Wang, Xiao, Dang, Gan, Han
**Venue:** MLSys 2024

**Summary:** Identifies that only 1% of salient weight channels disproportionately affect quality, and protects them during quantization. Achieves 4-bit quantization with quality matching GPTQ at lower calibration cost.

**Relevance:** AWQ's activation-aware approach is particularly relevant for edge deployment where model size directly constrains what fits in device memory. The paper cites AWQ alongside GPTQ to establish that 4-bit weight quantization is a well-validated technique, supporting the paper's use of Q4-weight models.

**How cited:** In Appendix C alongside GPTQ for weight quantization quality at 4 bits.

---

## 4. Agent Memory (4 references)

### EM-LLM
**Cite key:** `jiang2025emllm`
**Full title:** Human-inspired Episodic Memory for Infinite Context LLMs
**Authors:** Jiang et al.
**Venue:** ICLR 2025

**Summary:** Organizes tokens into episodic events using Bayesian surprise for segmentation and similarity-based retrieval, achieving 30.5% improvement over RAG on LongBench for infinite-context tasks.

**Relevance:** Operates at the token-embedding level to organize and retrieve past context, while the paper operates at the KV-cache level. Both aim to give LLMs persistent memory beyond the current context window. EM-LLM's cognitive-science-inspired segmentation could complement the paper's block-based persistence.

**How cited:** In Related Work (agent memory paragraph) for organizing tokens into episodic events with 30.5% improvement over RAG.

---

### A-MEM
**Cite key:** `xu2025amem`
**Full title:** A-MEM: Agentic Memory for LLM Agents
**Authors:** Xu, Zhang et al.
**Venue:** NeurIPS 2025

**Summary:** Implements a Zettelkasten-inspired memory system where agents dynamically create, link, and evolve structured memory notes for long-term knowledge accumulation.

**Relevance:** Operates at the semantic text level (note networks) rather than the KV cache level. The paper's persistent KV cache could serve as the low-level infrastructure beneath A-MEM's high-level memory abstractions -- A-MEM manages what to remember, the paper's system manages how to store and reload it efficiently.

**How cited:** In Related Work for Zettelkasten-style note networks as agent memory.

---

### MemArt
**Cite key:** `memart2026iclr`
**Full title:** KVCache-Centric Memory for LLM Agents
**Authors:** Anonymous (under review)
**Venue:** ICLR 2026 submission

**Summary:** Introduces KV-cache-centric memory with reusable blocks, retrieval via attention scores in latent space, and decoupled position encoding for safe block reuse. Achieves 91--135x prefill reduction on LoCoMo benchmark.

**Relevance:** The single closest prior work. Shares the KV-cache-as-memory philosophy. Key differences: MemArt targets datacenter deployments, lacks Q4 quantization pipeline, has no disk persistence, and does not handle multi-agent isolation. The paper identifies MemArt as one of two closest systems in the novelty discussion.

**How cited:** In Table 6 ("Reuse" pool, working memory, no edge/Q4); in Related Work for KV-cache-centric memory with 91--135x reduction; in novelty comparison discussion.

---

### Memory3
**Cite key:** `yang2025memory3`
**Full title:** Memory3: Language Modeling with Explicit Memory
**Authors:** Yang, Lin et al.
**Venue:** arXiv (referenced via MemOS, 2024)

**Summary:** Proposes explicit memory as a third form of LLM knowledge (alongside context KV cache and model parameters), demonstrating that explicit memory can outperform both larger models and RAG for knowledge-intensive tasks.

**Relevance:** Memory3 is in the bib file but **not cited in the paper text**. Its three-form memory taxonomy (parameters, context, explicit) maps directly to the paper's system: model weights (parameters), hot KV cache (context), cold safetensors (explicit memory). Citing it would strengthen the conceptual framing.

**How cited:** Present in bib file only. Not cited in the paper.

---

## 5. Multi-Agent Systems (7 references)

### KVCOMM
**Cite key:** `ye2025kvcomm`
**Full title:** KVCOMM: Online Cross-context KV-cache Communication for Efficient LLM-based Multi-agent Systems
**Authors:** Ye, Gao, Ma, Wang, Fu, Chung, Lin, Liu, Zhang, Zhuo, Chen
**Venue:** NeurIPS 2025

**Summary:** Enables cross-context KV cache sharing in multi-agent systems by estimating cross-context deviations using an anchor pool, achieving >70% cache reuse with 7.8x speedup in 5-agent settings.

**Relevance:** The most directly comparable multi-agent KV cache work. KVCOMM approximates cross-agent sharing; the paper maintains per-agent isolation with exact computation. This is a fundamental design divergence: KVCOMM trades accuracy for memory, while the paper trades memory for accuracy (using Q4 to fit more isolated caches).

**How cited:** In Table 6 ("Share" working memory); in Related Work for 7.8x speedup and >70% cross-context reuse.

---

### KVFlow
**Cite key:** `pan2025kvflow`
**Full title:** KVFlow: Efficient Prefix Caching for Accelerating LLM-Based Multi-Agent Workflows
**Authors:** Pan, Patel, Hu, Shen, Guan, Li, Qin, Wang, Ding
**Venue:** NeurIPS 2025

**Summary:** Models agent execution as a step graph and uses workflow-aware cache eviction and prefetching, achieving 2.19x concurrent speedup over SGLang's radix cache.

**Relevance:** Addresses the same multi-agent workflow scheduling problem. KVFlow's step-graph model could inform the paper's cross-phase context injection by explicitly modeling phase dependencies. KVFlow targets datacenter; the paper targets single-device edge.

**How cited:** In Table 6 ("Prefix" pool, "Flow" working memory); in Related Work for workflow-aware eviction with 2.19x speedup.

---

### DroidSpeak
**Cite key:** `liu2024droidspeak`
**Full title:** DroidSpeak: KV Cache Sharing for Cross-LLM Communication and Multi-LLM Serving
**Authors:** Liu, Huang, Yao, Feng, Gu, Du, Li, Cheng, Jiang, Lu, Musuvathi, Choukse
**Venue:** arXiv:2411.02820, November 2024

**Summary:** Enables KV cache reuse across different LLM architectures by selectively recomputing only a few layers, achieving 4x throughput with 3.1x faster TTFT in cross-model serving.

**Relevance:** DroidSpeak is in the bib file but **not cited in the paper text**. Its cross-architecture KV sharing is relevant because the paper supports two architecturally distinct models (Gemma/GQA and DeepSeek/MLA). However, the paper does not attempt cross-model cache sharing -- it uses model-tag compatibility checking to prevent incompatible reuse.

**How cited:** Present in bib file only. Not cited in the paper.

---

### PROMPTPEEK
**Cite key:** `promptpeek2025`
**Full title:** I Know What You Asked: Prompt Leakage via KV-Cache Sharing in Multi-Tenant LLM Serving
**Authors:** NDSS 2025 Authors
**Venue:** NDSS 2025

**Summary:** Demonstrates that shared KV caches in multi-tenant serving enable 99% prompt reconstruction attacks, exposing a serious privacy vulnerability in prefix-sharing schemes.

**Relevance:** Directly motivates the paper's per-agent isolation design. If caches are shared across agents, one agent can reconstruct another's prompt. The paper's block pool enforces namespace isolation at the data structure level, preventing cross-agent information leakage.

**How cited:** In Related Work for showing that shared KV caches enable 99% prompt reconstruction attacks, motivating per-agent isolation.

---

### SagaLLM
**Cite key:** `chang2025sagallm`
**Full title:** SagaLLM: Multi-Agent Orchestration with Transaction-Based Context Management
**Authors:** Chang et al.
**Venue:** VLDB 2025

**Summary:** Applies database transaction semantics (ACID properties) to multi-agent context management, identifying "context loss" across agent boundaries as a fundamental limitation of current multi-agent systems.

**Relevance:** SagaLLM's identification of context loss directly supports the paper's motivation: agents lose their KV cache state at boundaries (server restarts, phase transitions). The paper's persistence mechanism is a direct solution to SagaLLM's diagnosed problem.

**How cited:** In Background (Sec 2.1) for identifying "context loss" as a fundamental limitation of multi-agent systems.

---

### AutoGen
**Cite key:** `wu2023autogen`
**Full title:** AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation
**Authors:** Wu, Bansal, Zhang, Wu, Li, Zhu, Jiang, Zhang, Zhang, Liu, Awadallah, White, Burger, Wang
**Venue:** arXiv:2308.08155, 2023

**Summary:** A framework for building multi-agent conversational systems where agents with different roles (assistant, user proxy, critic) collaborate through structured conversations to solve tasks.

**Relevance:** AutoGen is the primary example of an agentic framework that the paper's system sits beneath as infrastructure. The paper explicitly states it exposes an OpenAI-compatible API so AutoGen (and CrewAI, LangGraph) can use persistent cache without modification.

**How cited:** In Introduction (Contributions); in Background for multi-agent scaling; in Discussion (Sec 5.1) as a framework using the system via OpenAI API; in Conclusion.

---

### Multi-Agent Survey
**Cite key:** `guo2024multiagent`
**Full title:** Large Language Model Based Multi-Agents: A Survey of Progress and Challenges
**Authors:** Guo, Chen, Wang, Chang, Pei, Chawla, Wiest, Zhang
**Venue:** IJCAI 2024

**Summary:** Comprehensive survey of LLM-based multi-agent systems covering architecture patterns, communication mechanisms, role assignment, and evaluation, identifying scalability and context management as key open challenges.

**Relevance:** Provides the broader landscape context for why multi-agent cache management matters. The paper cites this survey for the claim that real agentic workflows scale to 5--20+ agents with specialized roles requiring independent conversational state.

**How cited:** In Background (Sec 2.1) for the claim that agentic workflows scale to 5--20+ agents with independent state.

---

## 6. Edge/On-Device Inference (4 references)

### KVSwap
**Cite key:** `zhang2024kvswap`
**Full title:** KVSwap: Disk-aware KV Cache Offloading for Long-Context On-device Inference
**Authors:** Zhang, Xia, Wang
**Venue:** arXiv:2511.11907, November 2025

**Summary:** Designed for mobile/embedded systems with unified memory, uses disk-based KV cache offloading with compact metadata for preloading decisions. Achieves 1.8x (NVMe) to 4.1x (eMMC) throughput improvements for long-context inference.

**Relevance:** The closest prior work for on-device KV cache disk management. Key differences: KVSwap optimizes for bandwidth-constrained mobile storage and single-user long-context inference, while the paper targets Apple Silicon's high-bandwidth NVMe with multi-agent isolation and Q4 quantization.

**How cited:** In Table 6 (no pool, edge support, no Q4/persistence); in Related Work for disk-aware offloading on mobile devices.

---

### Kelle
**Cite key:** `kelle2025micro`
**Full title:** Kelle: Co-design KV Caching and eDRAM for Edge Computing
**Authors:** MICRO 2025 Authors
**Venue:** IEEE/ACM MICRO 2025

**Summary:** Co-designs KV cache management with embedded DRAM for custom edge accelerators, achieving 3.9x speedup through hardware-software co-optimization.

**Relevance:** Represents the hardware-accelerator approach to edge KV cache management, requiring specialized silicon. The paper's software-only approach achieves comparable goals on commodity Apple Silicon without custom hardware. Kelle validates that edge KV cache is an important problem space.

**How cited:** In Related Work for hardware co-design with 3.9x speedup, noting it requires specialized hardware.

---

### Local LLM Benchmark
**Cite key:** `perez2025localllm`
**Full title:** Production-Grade Local LLM Inference on Apple Silicon
**Authors:** Perez et al.
**Venue:** arXiv:2511.05502, 2025

**Summary:** Benchmarks local LLM inference performance on Apple Silicon, providing baseline throughput and latency numbers for various model sizes without addressing multi-agent cache management.

**Relevance:** Provides the baseline performance context for Apple Silicon inference that the paper builds upon. Specifically, it does not address multi-agent scenarios, persistent caching, or KV cache quantization -- the paper's contributions.

**How cited:** In Related Work for benchmarking local inference on Apple Silicon without multi-agent cache management.

---

### Krul
**Cite key:** `wen2025krul`
**Full title:** Krul: Optimizing On-Device Large Language Model Deployment
**Authors:** Wen et al.
**Venue:** arXiv, 2025

**Summary:** Optimizes on-device LLM deployment with a focus on model loading, memory management, and runtime optimization for mobile and edge devices.

**Relevance:** Addresses on-device LLM deployment optimization but does not tackle KV cache persistence or multi-agent scenarios. The paper cites it to show that existing edge-focused work leaves the cache persistence problem unaddressed.

**How cited:** In Related Work for on-device optimization without KV persistence.

---

## 7. Attention & Context (4 references)

### Lost in the Middle
**Cite key:** `liu2024lost`
**Full title:** Lost in the Middle: How Language Models Use Long Contexts
**Authors:** Liu, Lin, Hewitt, Paranjape, Bevilacqua, Petroni, Liang
**Venue:** TACL, Volume 12, 2024

**Summary:** Demonstrates that language models disproportionately attend to information at the beginning and end of long contexts, with information in the middle receiving significantly less attention weight ("position bias").

**Relevance:** A core motivation for the paper's per-agent isolation design. If 5 agents' 4K contexts are concatenated into one 20K prompt, agents in the middle suffer from position bias. Separate KV caches per agent eliminate this. Cited twice: in Introduction and Background.

**How cited:** In Introduction (Sec 1) and Background (Sec 2.1) to motivate why concatenating agent contexts into one prompt is problematic.

---

### MInference
**Cite key:** `jiang2024minference`
**Full title:** MInference 1.0: Accelerating Pre-filling for Long-Context LLMs via Dynamic Sparse Attention
**Authors:** Jiang, Li, Zhang, Wu, Luo, Ahn, Han, Abdi, Li, Lin, Yang, Qiu
**Venue:** NeurIPS 2024 (Spotlight)

**Summary:** Accelerates prefill by identifying and exploiting sparse attention patterns (A-shape, Vertical-Slash, Block-Sparse) that emerge in long-context inference, reducing prefill FLOPS.

**Relevance:** MInference addresses the same prefill bottleneck the paper targets but via compute optimization rather than cache persistence. The paper cites it alongside "Lost in the Middle" for position bias evidence. MInference could complement the paper's approach for cold-start scenarios where persistence is not available.

**How cited:** In Background (Sec 2.1) alongside "Lost in the Middle" for position bias in concatenated contexts.

---

### RAGCache
**Cite key:** `jin2024ragcache`
**Full title:** RAGCache: Efficient Knowledge Caching for Retrieval-Augmented Generation
**Authors:** Jin, Zhang, Jiang, Fan, Zhu, Luo, Jin
**Venue:** ACM TCS, 2024

**Summary:** Caches intermediate KV states from RAG document processing to avoid re-computing attention over frequently retrieved documents, reducing RAG inference latency.

**Relevance:** RAGCache applies KV persistence to RAG workloads (caching per-document KV states), while the paper applies it to agent workloads (caching per-agent conversation states). The paper explicitly distinguishes its approach from RAG-based caching in Table 5 and Discussion.

**How cited:** In Background (Sec 2.3) for caching intermediate KV states across RAG queries; noted as targeting datacenter deployments.

---

### Fusion RAG Cache
**Cite key:** `fusionragcache2025`
**Full title:** Fusion RAG Cache: Optimizing Retrieval-Augmented Generation with Persistent KV States
**Authors:** Li et al.
**Venue:** arXiv, 2025

**Summary:** Reports that prefill accounts for 95.53% of RAG inference time and proposes persistent KV states to eliminate redundant re-computation across RAG queries.

**Relevance:** The 95.53% prefill figure is a key data point in the paper's argument that prefill dominates latency and cache persistence is the right solution. The paper uses this number in Background (Sec 2.3) to argue that RAG does not solve the prefill problem -- it worsens it.

**How cited:** In Background (Sec 2.3) for the claim that prefill accounts for 95.5% of RAG inference time.

---

## 8. Human Factors (2 references)

### Nielsen 1993
**Cite key:** `nielsen1993`
**Full title:** Usability Engineering
**Authors:** Nielsen, Jakob
**Venue:** Academic Press, 1993

**Summary:** Establishes the three fundamental response-time thresholds for interactive systems: 100ms (instantaneous), 1s (acceptable, maintains flow), 10s (limit before user disengagement). Chapter 5 is the canonical reference.

**Relevance:** Provides the human-factors framework for interpreting TTFT measurements. The paper's central claim -- that warm cache crosses the 1s threshold into "acceptable" territory -- is grounded in Nielsen's thresholds. At 4K context, Gemma warm TTFT is 513ms, under the 1s threshold.

**How cited:** In Background (Sec 2.3) for the 100ms/1s/10s thresholds that frame TTFT significance.

---

### Nielsen 2024
**Cite key:** `nielsen2024speed`
**Full title:** The Need for Speed in the Age of AI
**Authors:** Nielsen, Jakob
**Venue:** Nielsen Norman Group, 2024

**Summary:** Updates the 1993 response-time framework for AI systems, arguing that no current AI system (including ChatGPT, Claude, Gemini) meets the 1-second response-time threshold for interactive use, especially at long context.

**Relevance:** Provides the contemporary benchmark against which the paper's TTFT results are measured. If no current AI system meets 1s TTFT, and the paper's warm cache achieves 513ms at 4K context, this positions the system as crossing a significant threshold.

**How cited:** In Background (Sec 2.3) for the claim that no current local AI system meets the 1s threshold.

---

## Uncited Bib Entries

The following entries exist in `semantic_colm2026.bib` but are not `\cite{}`d anywhere in the paper text:

| Cite Key | Title | Category |
|---|---|---|
| `lee2025ragdcache` | Shared Disk KV Cache Management for RAG-Powered LLMs | KV cache management |
| `feng2024evicpress` | EvicPress: Joint KV-Cache Compression and Eviction | KV cache compression |
| `bui2024trimkv` | Cache What Lasts: Token Retention for Memory-Bounded KV Cache | KV cache compression |
| `kim2026fastkvzip` | Fast KVzip: Gated KV Eviction | KV cache compression |
| `liu2024cachegen` | CacheGen: KV Cache Compression and Streaming | KV cache compression |
| `minikv2025` | MiniKV: 2-Bit Layer-Discriminative KV Cache | KV cache compression |
| `yang2025memory3` | Memory3: Language Modeling with Explicit Memory | Agent memory |
| `liu2024droidspeak` | DroidSpeak: Cross-LLM KV Cache Sharing | Multi-agent |
| `jeon2025lragent` | LRAgent: KV Cache Sharing for Multi-LoRA Agents | Multi-agent |
| `liang2026kvfails` | When KV Cache Reuse Fails in Multi-Agent Systems | Multi-agent |
| `yang2025kvlink` | KVLink: Efficient KV Cache Reuse | KV cache reuse |
| `sarthi2024raptor` | RAPTOR: Recursive Abstractive Processing for Retrieval | RAG |
| `sentencekv2025` | SentenceKV: Sentence-Level Semantic KV Caching | KV cache management |
| `kvsplit2025` | KVSplit: Differentiated Precision KV Cache on Apple Silicon | Quantization |
| `apple2024m4pro` | Mac mini M4 Pro Technical Specifications | Hardware |
| `nvidia2025dgxspark` | NVIDIA DGX Spark | Hardware |
| `nvidia2024a100bench` | LLM Inference Benchmark: A100 vs H100 | Hardware |

Several of these (especially `liang2026kvfails`, `jeon2025lragent`, `yang2025memory3`, `sentencekv2025`) would strengthen the paper if cited. The "When KV Cache Reuse Fails" paper in particular provides evidence for per-agent isolation. LRAgent addresses multi-LoRA multi-agent KV sharing, a natural extension. Memory3's three-form memory taxonomy maps to the paper's architecture.

---

## Coverage Gaps

Areas the paper does not cite and arguably should:

### 1. On-Device Inference Runtimes (Llama-Specific)

- **MLC-LLM** (Chen et al., 2023): Universal deployment of LLMs across platforms (iOS, Android, Metal, CUDA, Vulkan) with WebGPU support. Directly competes in the edge inference space. Its universal compilation approach contrasts with the paper's MLX-specific implementation.
- **llamafile** (Mozilla, 2024): Single-file executable LLM deployment using cosmopolitan libc for cross-platform portability. No KV cache persistence, but a relevant edge deployment baseline.
- **llama.cpp** (Gerganov, 2023): The de facto on-device inference engine. The paper mentions llama.cpp indirectly via vllm-mlx's comparison but never cites it directly. Given that llama.cpp is the primary alternative runtime on Apple Silicon, a direct citation and comparison would be expected.
- **Ollama** (2023): Popular user-facing LLM inference wrapper built on llama.cpp. Widely deployed on Apple Silicon but lacks persistent KV cache or multi-agent support.

### 2. Speculative Decoding Systems

- **Medusa** (Cai et al., NeurIPS 2024): Multiple decode heads for speculative token prediction. Relevant because Q4 KV cache could serve as a draft cache in speculative pipelines.
- **SpecInfer** (Miao et al., OSDI 2024): Speculative inference with tree-based verification. The paper cites QuantSpec but not the broader speculative decoding literature.
- **EAGLE / EAGLE-2** (Li et al., ICML 2024): Auto-regressive draft model using feature-level rather than token-level prediction. Relevant for future work combining persistence with speculation.

### 3. Long-Context and Sparse Attention

- **Infini-Attention** (Munkhdalai et al., Google, 2024): Compressive memory integrated into the attention mechanism for infinite context. A different approach to the "agent memory" problem that operates within the model architecture rather than externally.
- **Ring Attention** (Liu et al., 2024): Distributed attention across sequence-parallel devices. Relevant for multi-device extension mentioned in the paper's limitations.
- **StreamingLLM** (Xiao et al., ICLR 2024): Attention sink tokens for infinite streaming generation. Relevant because the paper's system could benefit from sink-token-aware cache eviction.

### 4. Quantization Beyond KV Cache

- **SqueezeLLM** (Kim et al., ICML 2024): Non-uniform quantization with dense-and-sparse decomposition. Relevant for the weight quantization discussion alongside GPTQ/AWQ.
- **QuIP#** (Tseng et al., ICML 2024): Incoherence-based 2-bit weight quantization. Pushes weight quantization further than the paper's 4-bit models.
- **AQLM** (Egiazarian et al., ICML 2024): Additive quantization for sub-2-bit LLMs. Demonstrates that even 2-bit weights are feasible.

### 5. KV Cache Eviction and Token Selection

- **H2O (Heavy Hitter Oracle)** (Zhang et al., NeurIPS 2023): Identifies "heavy hitter" tokens that dominate attention and selectively retains them. Relevant for the paper's hot-tier eviction policy.
- **Scissorhands** (Liu et al., NeurIPS 2024): Importance-aware KV cache compression via token selection. Could improve the paper's LRU-based eviction.
- **PyramidKV** / **PyramidInfer** (2024): Layer-wise KV budget allocation based on attention pattern analysis. Relevant for adaptive per-layer cache management.

### 6. Multi-Agent Communication Protocols

- **A2A (Agent-to-Agent)** by Google (2025): Standardized agent communication protocol. The paper mentions it in passing (Table 5) but does not cite it formally. A formal citation would strengthen the positioning.
- **MCP (Model Context Protocol)** by Anthropic (2024): Protocol for tool use and context injection. Also mentioned in Table 5 without formal citation.

### 7. Apple Silicon and Metal Specific

- **MLX benchmarking paper** (arXiv:2510.18921, 2025): In the bib file's extended references (R37) but not cited. Would provide baseline performance context.
- **Metal Performance Shaders** documentation: The paper discusses Metal buffer management and thread-safety issues but does not cite Apple's Metal documentation or the MLX GitHub issues (#2067, #2133, #3078) formally. Since these are URL references, they could be footnotes.

### 8. Unified Memory Architecture Analysis

- **Jain et al. (2024)**: Studies of unified vs discrete memory architectures for ML workloads. The paper's argument that unified memory advantages persist for KV cache operations would benefit from formal architectural analysis citations.

### 9. Privacy and Security

- The paper cites PROMPTPEEK for cache-sharing privacy risks but does not cite broader differential privacy or federated learning work relevant to on-device inference privacy. For a COLM submission, this may be acceptable given the systems focus.

### 10. Benchmark Methodology

- **MLPerf Inference** (Reddi et al., 2020): The standard ML inference benchmark suite. The paper uses custom benchmarks; citing MLPerf would contextualize the methodology choices.

---

## Summary Statistics

| Category | Cited in Paper | In Bib Only | Coverage Gaps |
|---|---|---|---|
| KV Cache Management | 7 | 2 | 0 |
| KV Cache Compression | 5 | 4 | 3 |
| Model Weight Quantization | 2 | 0 | 3 |
| Agent Memory | 3 | 1 | 1 |
| Multi-Agent Systems | 5 | 2 | 2 |
| Edge/On-Device | 4 | 2 | 4 |
| Attention & Context | 4 | 1 | 3 |
| Human Factors | 2 | 0 | 0 |
| Hardware | 0 | 3 | 1 |
| Speculative Decoding | 0 | 0 | 3 |
| Quantization (general) | 0 | 0 | 3 |
| **Total** | **32** | **15** | **23** |

The paper cites 32 unique references in the text. The bib file contains 47 entries (15 uncited). There are approximately 23 identified gaps across 10 categories. The most significant gaps are in on-device inference runtimes (llama.cpp, MLC-LLM, llamafile), speculative decoding, and KV cache eviction strategies. The most impactful additions would be a direct llama.cpp citation (primary competition), the "When KV Cache Reuse Fails" paper already in the bib (supports per-agent isolation), and Memory3 already in the bib (supports the memory taxonomy framing).
