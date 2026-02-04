# Literature Search: COLM 2026 Paper
## 40+ Reference Comprehensive Review

**Date**: 2026-02-04
**Purpose**: Identify all relevant prior work, find supporting/contradicting sources

---

## Current Bibliography Status

**Total references in paper**: 29 entries
**Target for comprehensive coverage**: 40-50 references
**Additional references needed**: 11-21

---

## Categories and Coverage

### 1. KV Cache Management Systems (Current: 5, Target: 7-8)

**Currently cited**:
- ✓ vLLM + PagedAttention (Kwon et al., SOSP 2023)
- ✓ SGLang + RadixAttention (Zheng et al., NeurIPS 2024)
- ✓ vllm-mlx (Barrios, arXiv 2026)
- ✓ RAG-DCache (Lee et al., arXiv 2025)
- ✓ KVSwap (Zhang et al., arXiv 2024)

**Additional to consider**:
- [ ] LMCache (GitHub: LMCache/LMCache) - Disk/CPU/S3 KV cache (mentioned but not formally cited)
- [ ] HiCache - SGLang extension
- [ ] llm-d - Cache salting/session routing
- [ ] Sarathi (OSDI 2024) - Chunked prefill for batched serving
- [ ] Orca (OSDI 2022) - Iteration-level scheduling

**Recommendation**: Add LMCache as formal citation, consider Sarathi for batched inference context.

### 2. KV Cache Compression & Quantization (Current: 8, Target: 10-12)

**Currently cited**:
- ✓ KIVI (Liu et al., ICML 2024)
- ✓ KVQuant (Hooper et al., NeurIPS 2024)
- ✓ CacheGen (Liu et al., SIGCOMM 2024)
- ✓ CommVQ (Li et al., ICML 2025)
- ✓ QuantSpec (Li et al., ICML 2025)
- ✓ XQuant (Tomar et al., arXiv 2025)
- ✓ EvicPress (Feng et al., arXiv 2024)
- ✓ TRIM-KV (Bui et al., arXiv 2024)
- ✓ Fast KVzip (Kim et al., arXiv 2026)

**Additional to consider**:
- [ ] GEAR (MLSys 2024) - Gradient-based KV cache eviction
- [ ] H2O (NeurIPS 2023) - Heavy-hitter oracle for KV cache
- [ ] Scissorhands (Arxiv 2023) - Structured pruning of KV cache
- [ ] StreamingLLM (Arxiv 2023) - Attention sinks for infinite context
- [ ] PyramidKV (COLM 2025) - Dynamic KV cache compression

**Recommendation**: Add H2O and StreamingLLM as they're foundational eviction/streaming work. PyramidKV from COLM 2025 is a good peer venue citation.

### 3. Agent Memory & RAG Systems (Current: 4, Target: 5-6)

**Currently cited**:
- ✓ EM-LLM (ICLR 2025)
- ✓ Memory3 (arXiv 2024)
- ✓ MemArt (ICLR 2026 submission)
- ✓ RAPTOR (Sarthi et al., ICLR 2024)

**Additional to consider**:
- [ ] MemGPT (Arxiv 2023) - Virtual context management
- [ ] Reflexion (NeurIPS 2023) - Verbal reinforcement learning with episodic memory
- [ ] RET-LLM (EMNLP 2023) - Retrieval-enhanced LLMs

**Recommendation**: Add MemGPT as it's highly cited for agent memory work.

### 4. Multi-Agent & Agentic Workloads (Current: 6, Target: 6-7)

**Currently cited**:
- ✓ KVCOMM (Ye et al., NeurIPS 2025)
- ✓ KVFlow (Pan et al., NeurIPS 2025)
- ✓ Continuum (Li et al., arXiv 2025)
- ✓ LRAgent (Jeon et al., arXiv 2025)
- ✓ KVLink (Yang et al., NeurIPS 2025)
- ✓ "When KV Cache Reuse Fails" (Liang et al., arXiv 2026)

**Additional to consider**:
- [ ] AutoGen (Microsoft, 2023) - Multi-agent conversation framework
- [ ] MetaGPT (arXiv 2023) - Multi-agent meta-programming

**Recommendation**: Coverage is good. AutoGen could be cited as motivating multi-agent use case.

### 5. Cross-LLM & Distributed Systems (Current: 2, Target: 3-4)

**Currently cited**:
- ✓ DroidSpeak (Liu et al., arXiv 2024)
- ✓ KVSplit (GitHub project 2025)

**Additional to consider**:
- [ ] FlexGen (ICML 2023) - High-throughput LLM inference with limited resources
- [ ] PetS (OSDI 2023) - Offloading for LLM serving
- [ ] DeepSpeed-Inference - Distributed LLM serving

**Recommendation**: Add FlexGen as it's a foundational edge/resource-constrained inference paper.

### 6. Apple Silicon / Edge ML Systems (Current: 1, Target: 3-4)

**Currently cited**:
- ✓ Apple M4 Pro specs (Apple Inc., 2024)

**Additional to consider**:
- [ ] MLX Documentation (Apple, 2023) - Apple's ML framework for Apple Silicon
- [ ] llama.cpp (Gerganov, GitHub) - CPU/Metal inference engine
- [ ] Ollama (GitHub) - Local LLM inference tool
- [ ] LM Studio - Desktop LLM inference (mentioned but not cited)

**Recommendation**: Add llama.cpp and MLX as foundational Apple Silicon inference tools.

### 7. Quantization Foundations (Current: 0, Target: 2-3)

**Currently cited**: None (specific to KV cache quantization only)

**Additional to consider**:
- [ ] GPTQ (Frantar et al., ICLR 2023) - Post-training quantization
- [ ] AWQ (Lin et al., MLSys 2024) - Activation-aware weight quantization
- [ ] QuaRot (Ashkboos et al., ICML 2024) - Quaternion-based rotation for quantization

**Recommendation**: Add GPTQ and AWQ as foundational quantization work that enables KV cache quantization.

### 8. Attention Mechanisms & Optimizations (Current: 0, Target: 2-3)

**Currently cited**: None

**Additional to consider**:
- [ ] Flash Attention (Dao et al., NeurIPS 2022) - Memory-efficient attention
- [ ] Flash Attention 2 (Dao, 2023) - Improved version
- [ ] Grouped-Query Attention (Ainslie et al., 2023) - Efficiency technique used in Gemma/Llama

**Recommendation**: Add Flash Attention 2 as it underlies MLX attention kernels.

### 9. Hardware Specifications (Current: 3, Target: 3-4)

**Currently cited**:
- ✓ Apple M4 Pro (Apple Inc., 2024)
- ✓ NVIDIA DGX Spark (NVIDIA, 2025)
- ✓ NVIDIA A100 benchmarks (Hyperstack, 2024)

**Additional to consider**:
- [ ] NVIDIA H100 specs - For future work comparison
- [ ] AMD Strix Halo - Competitor UMA architecture

**Recommendation**: Current coverage sufficient. Can add H100 if discussing future work.

### 10. Benchmark Datasets & Evaluation (Current: 0, Target: 1-2)

**Currently cited**: None

**Additional to consider**:
- [ ] LongBench (Bai et al., 2023) - Long-context evaluation
- [ ] RULER (ICLR 2024) - Long-context needle-in-haystack
- [ ] InfiniteBench (Zhang et al., 2024) - Extremely long context evaluation

**Recommendation**: Add LongBench as it's mentioned in EM-LLM comparison but not formally cited.

---

## Proposed Additional Citations (Priority Order)

### Tier 1: Essential Additions (Should Add)

1. **LMCache** (GitHub project, 2024)
   - Reason: Mentioned in Related Work but not formally cited
   - Section: 6.1 KV Cache Management Systems
   - Type: misc (GitHub)

2. **Flash Attention 2** (Dao, 2023)
   - Reason: Foundational for MLX attention kernels
   - Section: 2. Background or 6. Related Work
   - Type: article (arXiv)

3. **GPTQ** (Frantar et al., ICLR 2023)
   - Reason: Foundational quantization work enabling KV quantization
   - Section: 6.2 KV Cache Compression
   - Type: inproceedings

4. **H2O** (Zhang et al., NeurIPS 2023)
   - Reason: Heavy-hitter oracle is seminal KV eviction work
   - Section: 6.2 or 6.4
   - Type: inproceedings

5. **StreamingLLM** (Xiao et al., Arxiv 2023)
   - Reason: Attention sinks for long-context inference
   - Section: 6.2 or 6.4
   - Type: article

6. **FlexGen** (Sheng et al., ICML 2023)
   - Reason: Resource-constrained inference on edge devices
   - Section: 6.4 Edge/On-Device Systems
   - Type: inproceedings

7. **llama.cpp** (Gerganov, GitHub)
   - Reason: Widely used CPU/Metal inference, mentioned as baseline
   - Section: 6.4
   - Type: misc

8. **MemGPT** (Packer et al., Arxiv 2023)
   - Reason: Virtual context management for agents
   - Section: 6.3 Agent Memory
   - Type: article

9. **MLX** (Apple, 2023)
   - Reason: Framework used for entire system, should be cited
   - Section: 2. Background or 4.1 Setup
   - Type: misc

10. **PyramidKV** (COLM 2025)
    - Reason: Peer venue, dynamic KV cache compression
    - Section: 6.2
    - Type: inproceedings

### Tier 2: Nice to Have (Optional)

11. **Sarathi** (Agrawal et al., OSDI 2024)
    - Reason: Chunked prefill for batched serving
    - Section: 6.1

12. **AWQ** (Lin et al., MLSys 2024)
    - Reason: Activation-aware quantization
    - Section: 6.2

13. **AutoGen** (Microsoft, 2023)
    - Reason: Multi-agent conversation framework motivation
    - Section: 1. Introduction

14. **LongBench** (Bai et al., 2023)
    - Reason: Long-context evaluation suite
    - Section: 6.3 (when comparing EM-LLM)

---

## Contradicting Sources (To Acknowledge)

### Sources That Challenge Our Claims

1. **"When KV Cache Reuse Fails"** (Liang et al., 2026)
   - Already cited ✓
   - Finding: Judge consistency drops with cache reuse in multi-agent LLM judges
   - Our position: We acknowledge cache staleness as a limitation (Section 5.3)

2. **Character-Level vs Token-Level Matching**
   - No known paper directly contradicting this approach
   - Most prior work uses token-level matching (vLLM, SGLang)
   - Our approach is novel but may be questioned for overhead

3. **Q4 Quality Impact**
   - CommVQ, KVQuant, KIVI all report <1% perplexity degradation
   - We don't provide perplexity evaluation (acknowledged in limitations)
   - Not contradictory, but a gap in our evaluation

---

## Coverage Analysis

### By Venue

| Venue | Current | Recommended | Notes |
|-------|---------|-------------|-------|
| ICLR | 3 | 4-5 | Add GPTQ, potentially RULER |
| ICML | 4 | 5-6 | Add FlexGen, potentially QuaRot |
| NeurIPS | 5 | 6-7 | Add H2O, Reflexion |
| OSDI/SOSP | 1 | 2 | Add Sarathi or Orca |
| MLSys | 0 | 1 | Add AWQ or GEAR |
| COLM | 0 | 1-2 | Add PyramidKV, potentially others |
| arXiv/Preprints | 13 | 15-17 | Add MemGPT, StreamingLLM, Flash Attention 2 |
| GitHub/Misc | 3 | 5-6 | Add LMCache, llama.cpp, MLX, AutoGen |

### By Research Area

| Area | Current | Recommended | Gap |
|------|---------|-------------|-----|
| KV Cache Systems | 5 | 8 | Need LMCache, Sarathi, Orca |
| Compression/Quant | 9 | 12-14 | Need H2O, StreamingLLM, GPTQ, AWQ, PyramidKV |
| Agent Memory | 4 | 5-6 | Need MemGPT, AutoGen |
| Multi-Agent | 6 | 6-7 | Good coverage |
| Edge/Apple Silicon | 1 | 4-5 | Need llama.cpp, MLX, Ollama |
| Attention Mechanisms | 0 | 2-3 | Need Flash Attention 2, GQA |
| Hardware Specs | 3 | 3-4 | Good coverage |

---

## Recommended Action Plan

### To Reach 40 References (Need +11)

**Add these 11 citations**:

1. LMCache (misc)
2. Flash Attention 2 (article)
3. GPTQ (inproceedings, ICLR 2023)
4. H2O (inproceedings, NeurIPS 2023)
5. StreamingLLM (article)
6. FlexGen (inproceedings, ICML 2023)
7. llama.cpp (misc)
8. MemGPT (article)
9. MLX (misc)
10. PyramidKV (inproceedings, COLM 2025)
11. AWQ (inproceedings, MLSys 2024)

**Total**: 29 + 11 = 40 references

### To Reach 45 References (Need +16)

Add above 11, plus:

12. Sarathi (inproceedings, OSDI 2024)
13. AutoGen (misc)
14. LongBench (article)
15. Ollama (misc)
16. Grouped-Query Attention (article)

**Total**: 29 + 16 = 45 references

---

## Search Queries for Missing Papers

### For User to Run

```
1. "LMCache KV cache disk CPU S3" site:github.com
2. "Flash Attention 2 Dao 2023" site:arxiv.org
3. "GPTQ post-training quantization Frantar" site:arxiv.org
4. "H2O Heavy Hitter Oracle KV cache" site:arxiv.org
5. "StreamingLLM infinite context attention sinks" site:arxiv.org
6. "FlexGen high-throughput LLM limited resources" site:arxiv.org
7. "llama.cpp Georgi Gerganov" site:github.com
8. "MemGPT virtual context management" site:arxiv.org
9. "MLX Apple machine learning framework" site:ml-explore.github.io
10. "PyramidKV dynamic KV cache compression" site:openreview.net
11. "AWQ activation-aware weight quantization" site:arxiv.org
```

---

## Summary

**Current state**: 29 references, good coverage of recent (2024-2026) KV cache work
**Recommended additions**: 11-16 references to reach 40-45 target
**Priority**: Add foundational work (Flash Attention, GPTQ, H2O) and Apple Silicon tools (MLX, llama.cpp)
**Gaps**: Attention mechanisms, quantization foundations, edge tooling

**Paper has solid coverage of cutting-edge work but could benefit from more foundational citations to establish context.**
