# Extended Bibliography for COLM 2026 Submission

This document catalogs 40+ references organized by category, supporting the
claims and positioning of the Semantic Caching Server paper. Each entry
includes full title, authors, venue/year, and a brief relevance statement.

---

## 1. KV Cache Management Systems

### [R1] vLLM / PagedAttention

**Title:** Efficient Memory Management for Large Language Model Serving with PagedAttention

**Authors:** Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, Ion Stoica

**Venue:** SOSP 2023 (29th ACM Symposium on Operating Systems Principles)

**Relevance:** Introduced virtual-memory-inspired paged KV cache allocation that
eliminates fragmentation and enables near-zero-waste memory sharing during
parallel sampling. The block-based allocation scheme directly inspired our
`BlockPoolBatchEngine` design, though vLLM targets datacenter GPUs with
discrete VRAM rather than unified memory.

---

### [R2] SGLang / RadixAttention

**Title:** SGLang: Efficient Execution of Structured Language Model Programs

**Authors:** Lianmin Zheng, Liangsheng Yin, Zhiqiang Xie, Chuyue Sun, Jeff Huang, Cody Hao Yu, Shiyi Cao, Christos Kozyrakis, Ion Stoica, Joseph E. Gonzalez, et al.

**Venue:** NeurIPS 2024

**Relevance:** Introduced RadixAttention, which maintains a radix tree of KV
cache entries for automatic prefix sharing and reuse at runtime. SGLang's
approach to structured generation programs with automatic cache management
parallels our multi-agent coordination, but operates at the request scheduling
level in a datacenter context rather than as a persistent agent memory system.

---

### [R3] vllm-mlx

**Title:** Native LLM and MLLM Inference at Scale on Apple Silicon

**Authors:** Wayner Barrios

**Venue:** arXiv:2601.19139, January 2026

**Relevance:** Ports vLLM's continuous batching and prefix caching concepts to
Apple Silicon via the MLX backend. Achieves 21--87% higher throughput than
llama.cpp across models up to 30B parameters. Our system differs by adding
persistent 3-tier cache management (hot/warm/cold) with disk-backed
safetensors persistence and multi-agent coordination, capabilities absent
from vllm-mlx.

---

### [R4] LMCache

**Title:** LMCache: An Efficient KV Cache Layer for Enterprise-Scale LLM Inference

**Authors:** Yuhan Liu, Hanchen Li, Yihua Cheng, Siddhant Ray, Yuyang Huang, Qizheng Zhang, Kuntai Du, Jiayi Yao, Shan Lu, Ganesh Ananthanarayanan, et al.

**Venue:** arXiv:2510.09665, October 2025

**Relevance:** The first open-source library providing multi-tier KV cache
storage across GPU, CPU, disk, and S3 for vLLM and SGLang backends. LMCache
validates the multi-tier storage pattern we independently developed for
Apple Silicon, though LMCache targets datacenter deployments with disaggregated
storage. Our 3-tier design (hot in-memory, warm metadata, cold safetensors on
disk) is purpose-built for unified memory systems.

---

### [R5] Shared RAG-DCache

**Title:** Shared Disk KV Cache Management for Efficient Multi-Instance Inference in RAG-Powered LLMs

**Authors:** Hyungwoo Lee, et al.

**Venue:** IEEE CLOUD 2025 / arXiv:2504.11765, April 2025

**Relevance:** Proposes proactive disk-based KV cache generation and sharing
across multiple LLM instances for RAG workloads, achieving 15--71% throughput
increases. Demonstrates that disk-backed KV cache sharing is viable even with
I/O overhead -- a conclusion our safetensors-based cold storage tier
independently confirms on local NVMe.

---

### [R6] Continuum

**Title:** Continuum: Efficient and Robust Multi-Turn LLM Agent Scheduling with KV Cache Time-to-Live

**Authors:** Hanchen Li, Qiuyang Mang, Runyuan He, Qizheng Zhang, Huanzhi Mao, Xiaokun Chen, Hangrui Zhou, Alvin Cheung, Joseph Gonzalez, Ion Stoica

**Venue:** arXiv:2511.02230, November 2025

**Relevance:** Introduces TTL-based KV cache pinning for multi-turn agent
workloads, selectively retaining caches in GPU memory between tool-call turns.
Evaluated on SWE-Bench and BFCL with Llama-3.1 8B/70B. Our warm-tier metadata
tracking serves a similar purpose -- predicting which agent caches to retain --
but operates within a unified memory budget rather than discrete GPU memory.

---

### [R7] LRAgent

**Title:** LRAgent: Efficient KV Cache Sharing for Multi-LoRA LLM Agents

**Authors:** Hyesung Jeon, et al.

**Venue:** arXiv:2602.01053, February 2026

**Relevance:** Decomposes the KV cache into shared base and adapter-dependent
components for multi-LoRA agent systems. Introduces Flash-LoRA-Attention to
avoid full-dimension materialization of low-rank cache. Addresses a different
multi-agent sharing scenario (heterogeneous LoRA adapters) than our work (same
model, different agent contexts), but validates the importance of efficient
KV cache sharing in agentic workloads.

---

### [R8] Mooncake

**Title:** Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving

**Authors:** Ruoyu Qin, Zheming Li, Weiran He, Mingxing Zhang, Yongwei Wu, Weimin Zheng, Xinran Xu, et al.

**Venue:** USENIX FAST 2025 (Best Paper Award) / arXiv:2407.00079

**Relevance:** The production serving platform for Kimi (Moonshot AI), featuring
KVCache-centric disaggregated prefill/decode clusters using CPU, DRAM, SSD,
and RDMA resources. Achieved 59--498% throughput improvements. Validates the
principle that KV cache should be a first-class scheduling citizen, which our
design applies at the single-node edge scale rather than datacenter scale.

---

## 2. KV Cache Compression

### [R9] KIVI

**Title:** KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache

**Authors:** Zirui Liu, Jiayi Yuan, Hongye Jin, Shaochen Zhong, Zhaozhuo Xu, Vladimir Braverman, Beidi Chen, Xia Hu

**Venue:** ICML 2024

**Relevance:** Demonstrated that keys and values have different optimal
quantization strategies: per-channel for keys, per-token for values.
Achieved 2.6x memory reduction with 2-bit quantization. Our Q4 quantized
KV cache uses group quantization (scales + biases per group) rather than
KIVI's asymmetric approach, trading slightly higher bit-width for
compatibility with MLX's native `mx.quantized_matmul`.

---

### [R10] KVQuant

**Title:** KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization

**Authors:** Coleman Hooper, Sehoon Kim, Hiva Mohammadzadeh, Michael W. Mahoney, Yakun Sophia Shao, Kurt Keutzer, Amir Gholami

**Venue:** NeurIPS 2024

**Relevance:** Introduced per-layer sensitivity-weighted non-uniform datatypes
for KV cache quantization, achieving <0.1 perplexity degradation at 3-bit.
Their per-layer sensitivity analysis informed our understanding that not all
layers tolerate quantization equally -- reflected in our MLA-aware spec
extractor that handles DeepSeek's asymmetric K=192/V=128 dimensions.

---

### [R11] CommVQ

**Title:** CommVQ: Commutative Vector Quantization for KV Cache Compression

**Authors:** Junyan Li, Yang Zhang, Muhammad Yusuf Hassan, Talha Chafekar, Tianle Cai, Zhile Ren, Pengsheng Guo, et al.

**Venue:** ICML 2025

**Relevance:** Apple-affiliated research achieving 87.5% KV cache size reduction
via 2-bit vector quantization with RoPE-commutative codebooks. Demonstrates
that extreme compression (even 1-bit) is feasible with VQ approaches. Our
system uses 4-bit group quantization as a pragmatic middle ground that works
with MLX's existing quantized matmul primitives without codebook overhead.

---

### [R12] CacheGen

**Title:** CacheGen: KV Cache Compression and Streaming for Fast Large Language Model Serving

**Authors:** Yuhan Liu, Hanchen Li, Yihua Cheng, Siddhant Ray, Yuyang Huang, Qizheng Zhang, Kuntai Du, Jiayi Yao, Shan Lu, Ganesh Ananthanarayanan, et al.

**Venue:** ACM SIGCOMM 2024

**Relevance:** Encodes KV caches into compact bitstreams for network
transmission, achieving 3.5--4.3x compression. Demonstrates that KV cache
can be treated as a serializable, compressible artifact -- a principle our
safetensors-based cold storage follows, though we use quantized tensor
serialization rather than custom bitstream encoding.

---

### [R13] QuantSpec

**Title:** QuantSpec: Self-Speculative Decoding with Hierarchical Quantized KV Cache

**Authors:** Rishabh Tiwari, Haocheng Xi, Aditya Tomar, Coleman Hooper, Sehoon Kim, Maxwell Horton, Mahyar Najibi, Michael W. Mahoney, Kurt Keutzer, Amir Gholami

**Venue:** ICML 2025

**Relevance:** Apple-affiliated work combining speculative decoding with
hierarchical INT4/INT8 KV cache switching. Achieves 2.5x speedup with >90%
acceptance rate. Demonstrates that quantized KV cache can serve dual purposes
(draft and target model), suggesting future directions for our quantized
cache to participate in speculative workflows.

---

### [R14] XQuant

**Title:** XQuant: Breaking the Memory Wall for LLM Inference with KV Cache Rematerialization

**Authors:** Aditya Tomar, Coleman Hooper, Minjae Lee, Haocheng Xi, Rishabh Tiwari, Wonjun Kang, Luca Manolache, Michael W. Mahoney, Kurt Keutzer, Amir Gholami

**Venue:** arXiv:2508.10395, August 2025

**Relevance:** Instead of caching KV pairs, caches the layer input activations X
and rematerializes K/V on-the-fly. Achieves 10--12.5x memory savings vs FP16
with near-zero accuracy loss. Represents a fundamentally different approach to
the memory wall -- trading compute for memory -- that could complement our
quantized KV cache on compute-rich Apple Silicon.

---

### [R15] Fast KVzip

**Title:** Fast KVzip: Efficient and Accurate LLM Inference with Gated KV Eviction

**Authors:** Jang-Hyun Kim, Dongyoon Han, Sangdoo Yun

**Venue:** arXiv:2601.17668, January 2026 (NAVER AI Lab)

**Relevance:** Uses lightweight sink-attention gating modules to identify and
evict non-critical KV pairs, achieving 70% eviction with near-lossless
quality across Qwen2.5-1M, Qwen3, and Gemma3 families. Directly applicable
to our Gemma 3 target model, suggesting that learned eviction gates could
improve our LRU-based hot cache eviction policy.

---

### [R16] EvicPress

**Title:** EvicPress: Joint KV-Cache Compression and Eviction for Efficient LLM Serving

**Authors:** Shaoting Feng, Yuhan Liu, Hanchen Li, Xiaokun Chen, Samuel Shen, Kuntai Du, Zhuohan Gu, Rui Zhang, Yuyang Huang, Yihua Cheng, et al.

**Venue:** arXiv:2512.14946, December 2025

**Relevance:** Jointly optimizes eviction and compression decisions across all
KV caches using a unified utility function that balances quality and delay.
Achieves 2.19x TTFT improvement. Our 3-tier system performs a simpler version
of this joint optimization: LRU eviction from hot to warm, with quantized
compression during cold-tier serialization.

---

## 3. Agent Memory Systems

### [R17] EM-LLM

**Title:** Human-inspired Episodic Memory for Infinite Context LLMs

**Authors:** Zafeirios Fountas, Martin Benfeghoul, Adnan Oomerjee, Fenia Christopoulou, Gerasimos Lampouras, Haitham Bou Ammar, Jun Wang

**Venue:** ICLR 2025

**Relevance:** Applies cognitive science episodic memory models to enable
infinite-context processing in LLMs, using surprise-based segmentation and
similarity-based retrieval. Our agent cache store serves a similar architectural
role -- providing persistent, retrievable context for agents -- but operates at
the KV cache level rather than the token embedding level.

---

### [R18] Memory^3

**Title:** Memory^3: Language Modeling with Explicit Memory

**Authors:** Hongbo Yang, Boyan Lin, et al.

**Venue:** arXiv:2407.01178, July 2024

**Relevance:** Proposes explicit memory as the third form of memory in LLMs
(alongside context KV cache and model parameters), achieving better
performance than larger models and RAG. Our 3-tier cache system can be viewed
as an implementation of this explicit memory hierarchy for on-device agents,
where cold-tier safetensors represent durable explicit memories.

---

### [R19] MemOS

**Title:** MemOS: A Memory OS for AI System

**Authors:** MemTensor team (building on Memory^3)

**Venue:** arXiv:2507.03724, July 2025

**Relevance:** Extends Memory^3 into a full memory operating system that unifies
plaintext memory, activation memory, and parameter memory. Our system
implements a narrower but deeper version of this vision -- managing specifically
the KV cache memory layer with hot/warm/cold tiers, persistence, and
multi-agent coordination.

---

### [R20] MemArt

**Title:** KVCache-Centric Memory for LLM Agents

**Authors:** (Under review)

**Venue:** Under review at ICLR 2026

**Relevance:** The closest work to our agent cache design. Stores conversational
turns as reusable KV cache blocks, retrieves via attention scores in latent
space, and uses decoupled position encoding for safe block reuse. Achieves
11%+ accuracy improvement over plaintext memory on LoCoMo. Our system shares
the KV-centric memory philosophy but adds quantized compression, disk
persistence, and multi-agent block pool management not present in MemArt.

---

### [R21] A-MEM

**Title:** A-MEM: Agentic Memory for LLM Agents

**Authors:** Wujiang Xu, Zujie Liang, Kai Mei, Hang Gao, Juntao Tan, Yongfeng Zhang

**Venue:** NeurIPS 2025

**Relevance:** Implements a Zettelkasten-inspired memory system where agents
dynamically create, link, and evolve structured memory notes. Operates at the
semantic text level rather than the KV cache level. Our system could complement
A-MEM by providing the low-level KV cache persistence that A-MEM's high-level
memory abstractions lack.

---

### [R22] RAPTOR

**Title:** RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval

**Authors:** Parth Sarthi, Salman Abdullah, Aditi Tuli, Shubh Khanna, Anna Goldie, Christopher D. Manning

**Venue:** ICLR 2024

**Relevance:** Builds hierarchical summaries via recursive clustering and
abstraction for multi-level retrieval. While not a KV cache system, RAPTOR's
tree-structured memory organization parallels our agent cache hierarchy. The
recursive abstraction principle could inform future warm-tier compression
strategies.

---

### [R23] KVLink

**Title:** KVLink: Accelerating Large Language Models via Efficient KV Cache Reuse

**Authors:** Jingbo Yang, Bairu Hou, Wei Wei, Yujia Bao, Shiyu Chang

**Venue:** NeurIPS 2025

**Relevance:** Enables concatenation of independently precomputed KV caches
from different documents via positional embedding adjustment and trainable
bridge tokens. Achieves 96% TTFT reduction with 4% accuracy improvement.
Relevant to our multi-agent scenario where different agents may need to share
context segments.

---

## 4. Multi-Agent KV Cache

### [R24] KVCOMM

**Title:** KVCOMM: Online Cross-context KV-cache Communication for Efficient LLM-based Multi-agent Systems

**Authors:** Hancheng Ye, Zhengqi Gao, Mingyuan Ma, Qinsi Wang, Yuzhe Fu, Ming-Yu Chung, Yueqian Lin, Zhijian Liu, Jianyi Zhang, Danyang Zhuo, Yiran Chen

**Venue:** NeurIPS 2025

**Relevance:** The most directly comparable multi-agent KV cache work. KVCOMM
estimates cross-context KV cache deviations using an anchor pool, enabling
>70% reuse across agents with 7.8x speedup in 5-agent settings. Our approach
differs fundamentally: rather than approximating cross-context reuse, we
maintain per-agent caches with shared block pool allocation, trading memory
for exact computation.

---

### [R25] KVFlow

**Title:** KVFlow: Efficient Prefix Caching for Accelerating LLM-Based Multi-Agent Workflows

**Authors:** Zaifeng Pan, et al.

**Venue:** NeurIPS 2025

**Relevance:** Models agent execution as a step graph and assigns
steps-to-execution priority for workflow-aware KV cache eviction and
prefetching. Achieves 2.19x speedup over SGLang's radix cache on concurrent
workflows. Our `ConcurrentScheduler` with interleaved prefill/decode addresses
the same staggered execution pattern, but at the single-node level with MLX
stream management.

---

### [R26] DroidSpeak

**Title:** DroidSpeak: KV Cache Sharing for Cross-LLM Communication and Multi-LLM Serving

**Authors:** Yuhan Liu, Yuyang Huang, Jiayi Yao, Shaoting Feng, Zhuohan Gu, Kuntai Du, Hanchen Li, Yihua Cheng, Junchen Jiang, Shan Lu, Madan Musuvathi, Esha Choukse

**Venue:** arXiv:2411.02820, November 2024

**Relevance:** Enables KV cache reuse across different LLM architectures by
selectively recomputing a few layers, achieving 4x throughput with 3.1x
faster TTFT. Relevant because our system supports multiple model architectures
(Gemma 3, DeepSeek) with model-tag compatibility checking -- though we do not
attempt cross-model KV sharing.

---

### [R27] When KV Cache Reuse Fails

**Title:** When KV Cache Reuse Fails in Multi-Agent Systems: Cross-Candidate Interaction is Crucial for LLM Judges

**Authors:** Sichu Liang, Zhenglin Wang, Jiajia Chu, Pengfei Xia, Hui Zang, Deyu Zhou

**Venue:** arXiv:2601.08343, January 2026

**Relevance:** Critical cautionary result showing that KV cache reuse strategies
effective for generation agents can severely perturb judge behavior in
multi-agent systems. Judge consistency degrades even when end-task accuracy
appears stable. Motivates our design choice to maintain per-agent caches
rather than aggressive cross-agent sharing.

---

### [R28] SharedContext (DistServe-related)

**Title:** DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving

**Authors:** Yinmin Zhong, Shengyu Liu, Junda Chen, Jianbo Hu, Yibo Zhu, Xuanzhe Liu, Xin Jin, Hao Zhang

**Venue:** OSDI 2024

**Relevance:** While not specifically a multi-agent system, DistServe's
disaggregated prefill/decode architecture enables shared context processing
across requests. The prefill-decode separation pattern influenced our
`ConcurrentScheduler` design, which interleaves chunked prefill with decode
steps in a single-node setting.

---

## 5. Edge / On-Device Inference

### [R29] KVSwap

**Title:** KVSwap: Disk-aware KV Cache Offloading for Long-Context On-device Inference

**Authors:** Huawei Zhang, Chunwei Xia, Zheng Wang

**Venue:** arXiv:2511.11907, November 2025

**Relevance:** Designed for mobile/embedded systems with unified memory where
CPU/GPU share DRAM. Uses disk-based KV cache with compact metadata for
preloading decisions. Achieves 1.8x (NVMe) to 4.1x (eMMC) throughput
improvements. The closest prior work to our cold-tier design, but optimizes
for bandwidth-constrained mobile storage rather than Apple Silicon's
high-bandwidth NVMe.

---

### [R30] TRIM-KV

**Title:** Cache What Lasts: Token Retention for Memory-Bounded KV Cache in LLMs

**Authors:** Ngoc Bui, Shubham Sharma, Simran Lamba, Saumitra Mishra, Rex Ying

**Venue:** arXiv:2512.03324, December 2024

**Relevance:** Learns per-token importance via lightweight retention gates that
predict long-term utility with temporal decay. Outperforms full-cache models
in some settings, suggesting selective retention regularizes noise. The
retention-gate concept could enhance our hot-tier eviction policy, replacing
LRU with learned importance.

---

### [R31] Asteria / CXL-SpecKV

**Title:** CXL-SpecKV: A Disaggregated FPGA Speculative KV-Cache for Datacenter LLM Serving

**Authors:** (CXL-SpecKV research team)

**Venue:** arXiv:2512.11920, December 2025

**Relevance:** Uses FPGA-based CXL interconnects for speculative KV cache
prefetching in disaggregated architectures. Represents the hardware
acceleration approach to KV cache management. Our software-only approach
achieves similar goals (reduced TTFT, efficient memory use) on commodity
Apple Silicon without specialized hardware.

---

### [R32] llama.cpp

**Title:** llama.cpp: LLM Inference in C/C++

**Authors:** Georgi Gerganov and contributors

**Venue:** Open-source project, 2023--present (github.com/ggml-org/llama.cpp)

**Relevance:** The de facto standard for on-device LLM inference, supporting
x86, ARM, Metal, CUDA, and Vulkan backends with GGML quantization. Our
primary comparison target for Apple Silicon performance. llama.cpp lacks
multi-agent coordination, persistent KV cache, and semantic caching -- the
features that define our contribution. vllm-mlx reports 21--87% higher
throughput than llama.cpp on MLX-native inference.

---

### [R33] KVSplit

**Title:** KVSplit: Differentiated Precision KV Cache Quantization for Apple Silicon

**Authors:** Dipam Paul

**Venue:** Open-source project, github.com/dipampaul17/KVSplit, 2025

**Relevance:** Applies different quantization precision to keys vs values
(e.g., K8V4) on Apple Silicon with Metal support, achieving 59% memory
reduction with <1% quality loss. Validates the asymmetric K/V quantization
approach on Apple Silicon. Our system uses uniform Q4 for both K and V but
stores bfloat16 scales/biases, a design choice driven by MLX's
`mx.quantized_matmul` API constraints.

---

## 6. Apple Silicon / MLX Ecosystem

### [R34] MLX Framework

**Title:** MLX: Efficient and Flexible Machine Learning on Apple Silicon

**Authors:** Awni Hannun, Jagrit Digani, Angelos Katharopoulos, Ronan Collobert

**Venue:** Apple Machine Learning Research, December 2023 (github.com/ml-explore/mlx)

**Relevance:** The foundation framework for our entire system. MLX provides
unified memory (arrays live in shared CPU/GPU memory without transfers),
lazy evaluation with graph-based compilation, and Metal-backed GPU kernels.
Our system extensively uses MLX primitives including `mx.quantized_matmul`,
`mx.fast.scaled_dot_product_attention`, `mx.compile(shapeless=True)`, and
`mx.save_safetensors`/`mx.load` for cache persistence.

---

### [R35] mlx-lm

**Title:** mlx-lm: Run LLMs with MLX

**Authors:** Awni Hannun, Angelos Katharopoulos, et al. (Apple MLX team)

**Venue:** Open-source library, github.com/ml-explore/mlx-lm, 2024--present

**Relevance:** Provides model loading, tokenization, KV cache management, and
generation loops for LLMs on MLX. Our system builds on mlx-lm's
`QuantizedKVCache` and model architecture support, extending it with
`BatchQuantizedKVCache`, fused Q4 attention patches, sliding window mask
fixes, and the concurrent scheduler. We patch several mlx-lm internals
including `scaled_dot_product_attention` and `clip_residual`.

---

### [R36] Apple M4 Pro Technical Specifications

**Title:** Apple M4 Pro Chip

**Authors:** Apple Inc.

**Venue:** Apple Newsroom, October 2024

**Relevance:** Our primary evaluation hardware: 14-core CPU (10P+4E), 20-core
GPU, 24 GB unified memory with 273 GB/s bandwidth. The unified memory
architecture is central to our design -- eliminating PCIe transfer overhead
that datacenter GPU systems face when moving KV cache between CPU and GPU
memory. The 273 GB/s bandwidth enables our hot-tier to serve at memory speed.

---

### [R37] Benchmarking MLX on Apple Silicon

**Title:** Benchmarking On-Device Machine Learning on Apple Silicon with MLX

**Authors:** (Research team)

**Venue:** arXiv:2510.18921, October 2025

**Relevance:** Systematic benchmarking of MLX performance across Apple Silicon
generations for ML workloads. Provides context for our benchmark results and
validates the performance characteristics of the MLX framework on which our
system is built.

---

## 7. Supplementary References

### [R38] GQA / Grouped-Query Attention

**Title:** GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints

**Authors:** Joshua Ainslie, James Lee-Thorp, Michiel de Jong, Yinfei Yang, Cuthan O'Brien, Julian Auli

**Venue:** EMNLP 2023

**Relevance:** Describes the grouped-query attention mechanism used by Gemma 3,
where multiple query heads share a single KV head. Our fused Q4 attention
implementation must handle GQA's 5D reshape and mask broadcast correctly for
batch>1 inference -- a non-trivial engineering challenge documented in our
codebase.

---

### [R39] MLA / Multi-head Latent Attention (DeepSeek-V2)

**Title:** DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model

**Authors:** DeepSeek-AI

**Venue:** arXiv:2405.04434, May 2024

**Relevance:** Introduces Multi-head Latent Attention (MLA) with asymmetric
K/V dimensions (K=192 via nope+rope, V=128). Our spec extractor detects MLA
via `qk_nope_head_dim` and `qk_rope_head_dim` attributes, and our
`ModelCacheSpec` includes a `v_head_dim` field specifically for MLA's
asymmetric dimensions -- a distinction no other edge caching system handles.

---

### [R40] Gemma 3

**Title:** Gemma 3: Open Models Based on Gemini Technology

**Authors:** Google DeepMind (Gemma team)

**Venue:** Google Technical Report, March 2025

**Relevance:** Our primary evaluation model. Gemma 3 12B uses hybrid attention
(sliding window for 41/46 layers, global for 5/46), GQA, and bfloat16
scales/biases in Q4 quantized form. Our system handles all three of these
architecture-specific requirements through the fused attention patches,
sliding window mask fix, and bfloat16-aware safetensors serialization.

---

### [R41] LoRA

**Title:** LoRA: Low-Rank Adaptation of Large Language Models

**Authors:** Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen

**Venue:** ICLR 2022

**Relevance:** Foundation for multi-LoRA agent systems (LRAgent, R7). While our
current system does not use LoRA adapters, the KV cache sharing challenges
introduced by LoRA-based agent differentiation represent a natural extension
of our multi-agent architecture.

---

### [R42] FlashAttention

**Title:** FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness

**Authors:** Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Re

**Venue:** NeurIPS 2022

**Relevance:** The foundational fused attention kernel that eliminates
materialization of the full attention matrix. MLX's
`mx.fast.scaled_dot_product_attention` implements a similar fused kernel for
Metal GPUs. Our system dispatches between this fused FP16 path and a Q4 path
using `mx.quantized_matmul`, depending on cache quantization state.

---

### [R43] Transformers Library

**Title:** HuggingFace Transformers: State-of-the-Art Natural Language Processing

**Authors:** Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, et al.

**Venue:** EMNLP 2020 (Systems Demonstrations)

**Relevance:** Provides model architectures and tokenizers used by mlx-lm. Our
system pins `transformers>=4.47.0,<5.0.0` due to a v5.0.0rc1 SentencePiece
tokenizer regression that strips space markers during decode, a bug that
caused spacing corruption in DeepSeek outputs.

---

### [R44] SafeTensors

**Title:** safetensors: A Simple, Safe Way to Store and Distribute Tensors

**Authors:** HuggingFace team (Nicolas Patry, et al.)

**Venue:** Open-source library, github.com/huggingface/safetensors, 2023

**Relevance:** The serialization format for our cold-tier KV cache persistence.
We use MLX's native `mx.save_safetensors`/`mx.load` for zero-copy
(de)serialization with metadata headers encoding model tags, sequence
positions, and quantization parameters. The format's header-only metadata
reads enable our warm-tier to track cache contents without loading tensors.

---

### [R45] FastAPI / Uvicorn

**Title:** FastAPI: Modern, Fast Web Framework for Building APIs

**Authors:** Sebastian Ramirez

**Venue:** Open-source framework, 2019--present

**Relevance:** Our API server runtime. The FastAPI lifespan manager orchestrates
our 6-stage graceful shutdown sequence (stop scheduler, drain requests, persist
caches, shutdown engine, unload model, release GPU memory), which is critical
for preventing Metal GPU memory leaks on Apple Silicon.

---

*Total references: 45*

*Last updated: 2026-02-09*
