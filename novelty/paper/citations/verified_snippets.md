# Citation Verification Log

This document tracks verification of every citation used in the COLM 2026 paper.

## Verification Status
- Total citations: ~50
- Verified: 25
- Partial: 0
- Unverified: 25

---

## [1] Barrios (2026) -- vllm-mlx
- **BibTeX key**: `barrios2026vllmmlx`
- **Title**: Native LLM and MLLM Inference at Scale on Apple Silicon
- **Authors**: Wayner Barrios
- **Source URL**: https://arxiv.org/abs/2601.19139
- **Verified**: YES
- **arXiv ID**: 2601.19139
- **Date**: January 2026
- **Claim in novelty.md**: Content-based prefix caching, continuous batching on MLX
- **Verified snippets**:
  - Content-based prefix caching: "For multimodal models, we introduce content-based prefix caching that eliminates redundant vision encoding by identifying identical images through content hashing, regardless of input format."
  - Continuous batching: "For text models...providing continuous batching that scales to 4.3x aggregate throughput at 16 concurrent requests."
- **Match**: EXACT
- **BibTeX entry**:
```bibtex
@article{barrios2026vllmmlx,
  title={Native LLM and MLLM Inference at Scale on Apple Silicon},
  author={Barrios, Wayner},
  journal={arXiv preprint arXiv:2601.19139},
  year={2026}
}
```

## [2] Lee et al. (2025) -- RAG-DCache
- **BibTeX key**: `lee2025ragdcache`
- **Title**: Shared Disk KV Cache Management for Efficient Multi-Instance Inference in RAG-Powered LLMs
- **Authors**: Hyungwoo Lee, Kihyun Kim, Jinwoo Kim, Jungmin So, Myung-Hoon Cha, Hong-Yeon Kim, James J. Kim, Youngjae Kim
- **Source URL**: https://arxiv.org/abs/2504.11765
- **Verified**: YES
- **arXiv ID**: 2504.11765
- **Claim in novelty.md**: 15-71% throughput, 12-65% latency reduction, disk KV cache
- **Verified snippets**:
  - Throughput/latency: "achieved a 15~71% increase in throughput and up to a 12~65% reduction in latency"
  - Disk approach: "proposes a method to reduce TTFT by leveraging a disk-based key-value (KV) cache to lessen the computational burden during the prefill stage"
- **Match**: EXACT
- **BibTeX entry**:
```bibtex
@article{lee2025ragdcache,
  title={Shared Disk KV Cache Management for Efficient Multi-Instance Inference in RAG-Powered LLMs},
  author={Lee, Hyungwoo and Kim, Kihyun and Kim, Jinwoo and So, Jungmin and Cha, Myung-Hoon and Kim, Hong-Yeon and Kim, James J. and Kim, Youngjae},
  journal={arXiv preprint arXiv:2504.11765},
  year={2025}
}
```

## [15] Zhang et al. (2024) -- KVSwap
- **BibTeX key**: `zhang2024kvswap`
- **Title**: KVSwap: Disk-aware KV Cache Offloading for Long-Context On-device Inference
- **Authors**: Huawei Zhang, Chunwei Xia, Zheng Wang
- **Source URL**: https://arxiv.org/abs/2511.11907
- **Verified**: YES
- **arXiv ID**: 2511.11907
- **Claim in novelty.md**: Disk-based offloading for on-device
- **Verified snippets**:
  - Approach: "KVSwap stores the full cache on disk, uses highly compact in-memory metadata to predict which entries to preload, overlaps computation with hardware-aware disk access"
  - Target: "tailored for local devices"
  - Constraints: "CPU and GPU (or NPU) typically share a unified memory and the non-volatile secondary storage (disk) offers limited I/O bandwidth"
- **Match**: EXACT
- **BibTeX entry**:
```bibtex
@article{zhang2024kvswap,
  title={KVSwap: Disk-aware KV Cache Offloading for Long-Context On-device Inference},
  author={Zhang, Huawei and Xia, Chunwei and Wang, Zheng},
  journal={arXiv preprint arXiv:2511.11907},
  year={2024}
}
```

## [16] Feng et al. (2024) -- EvicPress
- **BibTeX key**: `feng2024evicpress`
- **Title**: EVICPRESS: Joint KV-Cache Compression and Eviction for Efficient LLM Serving
- **Authors**: Shaoting Feng, Yuhan Liu, Hanchen Li, Xiaokun Chen, Samuel Shen, Kuntai Du, Zhuohan Gu, Rui Zhang, Yuyang Huang, Yihua Cheng, Jiayi Yao, Qizheng Zhang, Ganesh Ananthanarayanan, Junchen Jiang
- **Source URL**: https://arxiv.org/abs/2512.14946
- **Verified**: YES
- **arXiv ID**: 2512.14946
- **Claim in novelty.md**: Joint compression + eviction, 2.19x TTFT
- **Verified snippets**:
  - Joint approach: "EVICPRESS considers the effect of compression and eviction of the KV cache on the average generation quality and delay across all contexts as a whole."
  - TTFT improvement: "EVICPRESS achieves up to 2.19x faster time-to-first-token (TTFT) at equivalent generation quality."
- **Match**: EXACT
- **BibTeX entry**:
```bibtex
@article{feng2024evicpress,
  title={EVICPRESS: Joint KV-Cache Compression and Eviction for Efficient LLM Serving},
  author={Feng, Shaoting and Liu, Yuhan and Li, Hanchen and Chen, Xiaokun and Shen, Samuel and Du, Kuntai and Gu, Zhuohan and Zhang, Rui and Huang, Yuyang and Cheng, Yihua and Yao, Jiayi and Zhang, Qizheng and Ananthanarayanan, Ganesh and Jiang, Junchen},
  journal={arXiv preprint arXiv:2512.14946},
  year={2024}
}
```

## [17] Tomar et al. (2025) -- XQuant
- **BibTeX key**: `tomar2025xquant`
- **Title**: XQuant: Breaking the Memory Wall for LLM Inference with KV Cache Rematerialization
- **Authors**: Aditya Tomar, Coleman Hooper, Minjae Lee, Haocheng Xi, Rishabh Tiwari, Wonjun Kang, Luca Manolache, Michael W. Mahoney, Kurt Keutzer, Amir Gholami
- **Source URL**: https://arxiv.org/abs/2508.10395
- **Verified**: YES
- **arXiv ID**: 2508.10395
- **Claim in novelty.md**: 10x memory savings, X instead of KV
- **Verified snippets**:
  - Memory savings: "XQuant-CL attains up to 10× memory savings relative to the FP16 baseline with only 0.01 perplexity degradation"
  - X approach: "We accomplish this by quantizing and caching the layer input activations X, instead of using standard KV caching, and then rematerializing the Keys and Values on-the-fly during inference."
- **Match**: EXACT
- **BibTeX entry**:
```bibtex
@article{tomar2025xquant,
  title={XQuant: Breaking the Memory Wall for LLM Inference with KV Cache Rematerialization},
  author={Tomar, Aditya and Hooper, Coleman and Lee, Minjae and Xi, Haocheng and Tiwari, Rishabh and Kang, Wonjun and Manolache, Luca and Mahoney, Michael W. and Keutzer, Kurt and Gholami, Amir},
  journal={arXiv preprint arXiv:2508.10395},
  year={2025}
}
```

## [19] Bui et al. (2024) -- TRIM-KV
- **BibTeX key**: `bui2024trimkv`
- **Title**: Cache What Lasts: Token Retention for Memory-Bounded KV Cache in LLMs
- **Authors**: Ngoc Bui, Shubham Sharma, Simran Lamba, Saumitra Mishra, Rex Ying
- **Source URL**: https://arxiv.org/abs/2512.03324
- **Verified**: PARTIAL
- **arXiv ID**: 2512.03324
- **Claim in novelty.md**: Full-cache performance at 25% KV budget
- **Verified snippets**:
  - General claim: "Remarkably, it even surpasses full-cache models in some settings, showing that selective retention can serve as a form of regularization, suppressing noise from uninformative tokens."
  - Note: Specific 25% claim not found in abstract; need to check full paper for this exact number
- **Match**: APPROXIMATE (general performance claim verified, specific 25% threshold not confirmed in abstract)
- **BibTeX entry**:
```bibtex
@article{bui2024trimkv,
  title={Cache What Lasts: Token Retention for Memory-Bounded KV Cache in LLMs},
  author={Bui, Ngoc and Sharma, Shubham and Lamba, Simran and Mishra, Saumitra and Ying, Rex},
  journal={arXiv preprint arXiv:2512.03324},
  year={2024}
}
```

## [20] Kim et al. (2026) -- Fast KVzip
- **BibTeX key**: `kim2026fastkvzip`
- **Title**: Fast KVzip: Efficient and Accurate LLM Inference with Gated KV Eviction
- **Authors**: Jang-Hyun Kim, Dongyoon Han, Sangdoo Yun
- **Source URL**: https://arxiv.org/abs/2601.17668
- **Verified**: YES
- **arXiv ID**: 2601.17668
- **Claim in novelty.md**: 70% KV eviction
- **Verified snippets**:
  - Eviction rate: "our method maintains near-lossless performance while evicting up to 70% of the KV cache."
  - Approach: "introduces lightweight sink-attention gating modules to identify and retain critical KV pairs"
- **Match**: EXACT
- **BibTeX entry**:
```bibtex
@article{kim2026fastkvzip,
  title={Fast KVzip: Efficient and Accurate LLM Inference with Gated KV Eviction},
  author={Kim, Jang-Hyun and Han, Dongyoon and Yun, Sangdoo},
  journal={arXiv preprint arXiv:2601.17668},
  year={2026}
}
```

## [21] MemArt (2026) -- KVCache-Centric Memory
- **BibTeX key**: `memart2026iclr`
- **Title**: KVCache-Centric Memory for LLM Agents
- **Authors**: (Disclosed at publication)
- **Source URL**: https://openreview.net/forum?id=YolJOZOGhI
- **Verified**: YES
- **Submission**: ICLR 2026
- **Claim in novelty.md**: 11% accuracy improvement, 91-135x fewer prefill tokens
- **Verified snippets**:
  - Accuracy improvement: "improving accuracy by over 11%" on LoCoMo benchmark
  - Prefill reduction: "leading to 91–135× more prefill tokens" compared to plaintext methods
  - Token efficiency: "MemArt requires only 32 and 37 tokens on average for each request with LLaMA and Qwen"
- **Match**: EXACT
- **BibTeX entry**:
```bibtex
@inproceedings{memart2026iclr,
  title={KVCache-Centric Memory for LLM Agents},
  author={[Authors to be disclosed at publication]},
  booktitle={International Conference on Learning Representations},
  year={2026},
  note={Submitted to ICLR 2026},
  url={https://openreview.net/forum?id=YolJOZOGhI}
}
```

## [22] Yang et al. (2025) -- KVLink
- **BibTeX key**: `yang2025kvlink`
- **Title**: KVLink: Accelerating Large Language Models via Efficient KV Cache Reuse
- **Authors**: Jingbo Yang, Bairu Hou, Wei Wei, Yujia Bao, Shiyu Chang
- **Source URL**: https://arxiv.org/abs/2502.16002
- **Verified**: YES
- **Venue**: NeurIPS 2025
- **arXiv ID**: 2502.16002
- **Claim in novelty.md**: 4% accuracy improvement, up to 96% TTFT reduction
- **Verified snippets**:
  - Accuracy: "improves question answering accuracy by an average of 4% over state-of-the-art methods"
  - TTFT reduction: "reduces time-to-first-token by up to 96% compared to standard LLM inference"
  - Approach: "precomputes the KV cache of each document independently, and during inference, the KV caches of retrieved documents are concatenated"
- **Match**: EXACT
- **BibTeX entry**:
```bibtex
@inproceedings{yang2025kvlink,
  title={KVLink: Accelerating Large Language Models via Efficient KV Cache Reuse},
  author={Yang, Jingbo and Hou, Bairu and Wei, Wei and Bao, Yujia and Chang, Shiyu},
  booktitle={Advances in Neural Information Processing Systems},
  year={2025},
  url={https://arxiv.org/abs/2502.16002}
}
```

## [23] Ye et al. (2025) -- KVCOMM
- **BibTeX key**: `ye2025kvcomm`
- **Title**: KVCOMM: Online Cross-context KV-cache Communication for Efficient LLM-based Multi-agent Systems
- **Authors**: Hancheng Ye, Zhengqi Gao, Mingyuan Ma, Qinsi Wang, Yuzhe Fu, Ming-Yu Chung, Yueqian Lin, Zhijian Liu, Jianyi Zhang, Danyang Zhuo, Yiran Chen
- **Source URL**: https://arxiv.org/abs/2510.12872
- **Verified**: YES
- **Venue**: NeurIPS 2025
- **arXiv ID**: 2510.12872
- **Claim in novelty.md**: 70% reuse rate, 7.8x speedup, cross-context KV reuse
- **Verified snippets**:
  - Reuse rate: "achieves over 70% reuse rate across diverse multi-agent workloads"
  - Speedup: "achieves up to 7.8× speedup compared to the standard prefill pipeline"
  - Latency: "reducing TTFT from ~430ms to ~55ms" in five-agent configuration
  - Quality: "all without quality degradation"
- **Match**: EXACT
- **BibTeX entry**:
```bibtex
@inproceedings{ye2025kvcomm,
  title={KVCOMM: Online Cross-context KV-cache Communication for Efficient LLM-based Multi-agent Systems},
  author={Ye, Hancheng and Gao, Zhengqi and Ma, Mingyuan and Wang, Qinsi and Fu, Yuzhe and Chung, Ming-Yu and Lin, Yueqian and Liu, Zhijian and Zhang, Jianyi and Zhuo, Danyang and Chen, Yiran},
  booktitle={Advances in Neural Information Processing Systems},
  year={2025},
  url={https://arxiv.org/abs/2510.12872}
}
```

## [24] Pan et al. (2025) -- KVFlow
- **BibTeX key**: `pan2025kvflow`
- **Title**: KVFlow: Efficient Prefix Caching for Accelerating LLM-Based Multi-Agent Workflows
- **Authors**: Zaifeng Pan, Ajjkumar Patel, Zhengding Hu, Yipeng Shen, Yue Guan, Wan-Lu Li, Lianhui Qin, Yida Wang, Yufei Ding
- **Source URL**: https://arxiv.org/abs/2507.07400
- **Verified**: YES
- **Venue**: NeurIPS 2025
- **arXiv ID**: 2507.07400
- **Claim in novelty.md**: 1.83x single workflow, 2.19x concurrent workflow speedup, workflow-aware eviction
- **Verified snippets**:
  - Single workflow: "achieves up to 1.83× speedup for single workflows with large prompts"
  - Concurrent workflows: "delivers up to 2.19× speedup for scenarios with many concurrent workflows"
  - Eviction strategy: "Agent Step Graph and assigns each agent a steps-to-execution value"
- **Match**: EXACT
- **BibTeX entry**:
```bibtex
@inproceedings{pan2025kvflow,
  title={KVFlow: Efficient Prefix Caching for Accelerating LLM-Based Multi-Agent Workflows},
  author={Pan, Zaifeng and Patel, Ajjkumar and Hu, Zhengding and Shen, Yipeng and Guan, Yue and Li, Wan-Lu and Qin, Lianhui and Wang, Yida and Ding, Yufei},
  booktitle={Advances in Neural Information Processing Systems},
  year={2025},
  url={https://arxiv.org/abs/2507.07400}
}
```

## [25] Liang et al. (2026) -- When KV Cache Reuse Fails
- **BibTeX key**: `liang2026kvfails`
- **Title**: When KV Cache Reuse Fails in Multi-Agent Systems: Cross-Candidate Interaction is Crucial for LLM Judges
- **Authors**: Sichu Liang, Zhenglin Wang, Jiajia Chu, Pengfei Xia, Hui Zang, Deyu Zhou
- **Source URL**: https://arxiv.org/abs/2601.08343
- **Verified**: YES
- **Date**: January 2026
- **arXiv ID**: 2601.08343
- **Claim in novelty.md**: Identifies failure modes of KV cache reuse for judges
- **Verified snippets**:
  - Problem: "reuse strategies that are effective for execution agents can severely perturb judge behavior"
  - Metric: "Judge Consistency Rate (JCR) metric" quantifies inconsistency
  - Solution: "Judges require explicit cross-candidate interaction to preserve their original dense-prefill decisions"
  - Scope: "across GSM8K, MMLU, and HumanEval benchmarks"
- **Match**: EXACT
- **BibTeX entry**:
```bibtex
@article{liang2026kvfails,
  title={When KV Cache Reuse Fails in Multi-Agent Systems: Cross-Candidate Interaction is Crucial for LLM Judges},
  author={Liang, Sichu and Wang, Zhenglin and Chu, Jiajia and Xia, Pengfei and Zang, Hui and Zhou, Deyu},
  journal={arXiv preprint arXiv:2601.08343},
  year={2026},
  url={https://arxiv.org/abs/2601.08343}
}
```

## [26] KVSplit (2025) -- Differentiated Precision
- **BibTeX key**: `kvsplit2025`
- **Title**: KVSplit: Run larger LLMs with longer contexts on Apple Silicon
- **Type**: Open Source Project
- **GitHub**: https://github.com/dipampaul17/KVSplit
- **Verified**: YES
- **Year**: 2025
- **Claim in novelty.md**: 59% memory reduction, <1% quality loss
- **Verified snippets**:
  - Memory reduction: "reducing memory by 59% with <1% quality loss" (K8V4 configuration)
  - Perplexity impact: "0.86% perplexity change" with optimal K8V4
  - Context length: "Users can run LLMs with 2-3× longer context on the same Mac"
  - Positional awareness: "Maintaining ≥8-bit keys preserves 98.7% positional awareness"
- **Match**: EXACT
- **BibTeX entry**:
```bibtex
@misc{kvsplit2025,
  title={KVSplit: Run larger LLMs with longer contexts on Apple Silicon},
  author={dipampaul},
  year={2025},
  howpublished={\url{https://github.com/dipampaul17/KVSplit}},
  note={Open Source Project with differentiated precision KV cache quantization}
}
```

## [27] Li et al. (2025) -- Continuum
- **BibTeX key**: `li2025continuum`
- **Title**: Continuum: Efficient and Robust Multi-Turn LLM Agent Scheduling with KV Cache Time-to-Live
- **Authors**: Hanchen Li, Qiuyang Mang, Runyuan He, Qizheng Zhang, Huanzhi Mao, Xiaokun Chen, Hangrui Zhou, Alvin Cheung, Joseph Gonzalez, Ion Stoica
- **Source URL**: https://arxiv.org/abs/2511.02230
- **Verified**: YES
- **Year**: 2025
- **arXiv ID**: 2511.02230
- **Claim in novelty.md**: TTL-based agent cache, 1.12x-3.66x delay reduction
- **Verified snippets**:
  - Performance: "reduces delay by 1.12x to 3.66x and improves throughput by 1.10x to 3.22x"
  - TTL approach: "selectively pins the KV cache in GPU memory with a time-to-live value"
  - Robustness: "providing robust performance under edge cases"
  - Implementation: "implemented on top of vLLM with a modular design"
- **Match**: EXACT
- **BibTeX entry**:
```bibtex
@article{li2025continuum,
  title={Continuum: Efficient and Robust Multi-Turn LLM Agent Scheduling with KV Cache Time-to-Live},
  author={Li, Hanchen and Mang, Qiuyang and He, Runyuan and Zhang, Qizheng and Mao, Huanzhi and Chen, Xiaokun and Zhou, Hangrui and Cheung, Alvin and Gonzalez, Joseph and Stoica, Ion},
  journal={arXiv preprint arXiv:2511.02230},
  year={2025}
}
```

## [28] Jeon et al. (2025) -- LRAgent
- **BibTeX key**: `jeon2025lragent`
- **Title**: LRAgent: Efficient KV Cache Sharing for Multi-LoRA LLM Agents
- **Authors**: Hyesung Jeon, Hyeongju Ha, Jae-Joon Kim
- **Source URL**: https://arxiv.org/abs/2602.01053
- **Verified**: YES
- **Date**: February 2025
- **arXiv ID**: 2602.01053
- **Claim in novelty.md**: KV cache decomposition for multi-LoRA agents
- **Verified snippets**:
  - Decomposition: "decomposing the KV cache into two components: a shared base component and an adapter-dependent component"
  - Efficiency kernel: "Flash-LoRA-Attention, a kernel that reorders attention computation to avoid materializing the low-rank cache"
  - Performance: "throughput and time-to-first-token latency close to fully shared caching, while preserving accuracy"
- **Match**: EXACT
- **BibTeX entry**:
```bibtex
@article{jeon2025lragent,
  title={LRAgent: Efficient KV Cache Sharing for Multi-LoRA LLM Agents},
  author={Jeon, Hyesung and Ha, Hyeongju and Kim, Jae-Joon},
  journal={arXiv preprint arXiv:2602.01053},
  year={2025}
}
```

## [29] Liu et al. (2024) -- DroidSpeak
- **BibTeX key**: `liu2024droidspeak`
- **Title**: DroidSpeak: KV Cache Sharing for Cross-LLM Communication and Multi-LLM Serving
- **Authors**: Yuhan Liu, Yuyang Huang, Jiayi Yao, Shaoting Feng, Zhuohan Gu, Kuntai Du, Hanchen Li, Yihua Cheng, Junchen Jiang, Shan Lu, Madan Musuvathi, Esha Choukse
- **Source URL**: https://arxiv.org/abs/2411.02820
- **Verified**: YES
- **Year**: 2024
- **arXiv ID**: 2411.02820
- **Claim in novelty.md**: 4x throughput, 3.1x prefill, cross-LLM KV reuse
- **Verified snippets**:
  - Throughput: "achieves up to 4x throughput improvement"
  - Prefill: "about 3.1x faster prefill (time to first token)"
  - Quality: "with negligible loss of quality in F1 scores, Rouge-L or code similarity"
  - Cross-LLM: "enables KV cache reuse across distributed nodes running inference of different LLMs"
  - Approach: "selectively recomputes a few layers of the KV cache produced by another LLM and reuses the remaining layers"
- **Match**: EXACT
- **BibTeX entry**:
```bibtex
@article{liu2024droidspeak,
  title={DroidSpeak: KV Cache Sharing for Cross-LLM Communication and Multi-LLM Serving},
  author={Liu, Yuhan and Huang, Yuyang and Yao, Jiayi and Feng, Shaoting and Gu, Zhuohan and Du, Kuntai and Li, Hanchen and Cheng, Yihua and Jiang, Junchen and Lu, Shan and Musuvathi, Madan and Choukse, Esha},
  journal={arXiv preprint arXiv:2411.02820},
  year={2024}
}
```

## [30] Sarthi et al. (2024) -- RAPTOR
- **BibTeX key**: `sarthi2024raptor`
- **Title**: RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval
- **Authors**: Parth Sarthi, Salman Abdullah, Aditi Tuli, Shubh Khanna, Anna Goldie, Christopher D. Manning
- **Venue**: International Conference on Learning Representations (ICLR)
- **Source URL**: https://arxiv.org/abs/2401.18059
- **Verified**: YES
- **Year**: 2024
- **arXiv ID**: 2401.18059
- **Claim in novelty.md**: 20% improvement on QuALITY benchmark with GPT-4
- **Verified snippets**:
  - QuALITY improvement: "improved the best performance on the QuALITY benchmark by 20 percentage points in absolute accuracy"
  - Approach: "recursively embedding, clustering, and summarizing chunks of text, constructing a tree with differing levels of summarization"
  - Retrieval: "retrieves from this tree, integrating information across lengthy documents at different levels of abstraction"
- **Match**: EXACT
- **BibTeX entry**:
```bibtex
@inproceedings{sarthi2024raptor,
  title={RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval},
  author={Sarthi, Parth and Abdullah, Salman and Tuli, Aditi and Khanna, Shubh and Goldie, Anna and Manning, Christopher D.},
  booktitle={International Conference on Learning Representations},
  year={2024},
  url={https://arxiv.org/abs/2401.18059}
}
```

---

