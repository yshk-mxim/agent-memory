# Thor Model Benchmarks — Published Scores

Last updated: 2026-04-03

Published benchmark scores for the three models available on Thor via agent-memory.
All numbers are from official model cards and leaderboards.

## Summary Table

| Benchmark | Gemma 4 31B (Dense) | Gemma 4 26B-A4B (MoE) | Qwen3-Coder-Next (80B-A3B) |
|-----------|:-------------------:|:----------------------:|:--------------------------:|
| **SWE-bench Verified** | 76.8%¹ | 63.8%¹ | 70.6% |
| **SWE-bench Pro** | 51%¹ | 44%¹ | 44.3% |
| **SWE-bench Multilingual** | — | — | 62.8% |
| **Terminal-Bench 2.0** | 50.8%² | —³ | 36.2% |
| **LiveCodeBench v6** | 80.0% | 77.1% | — |
| **Codeforces ELO** | 2150 | 1718 | — |
| **MMLU Pro** | 85.2% | 82.6% | — |
| **AIME 2026 (no tools)** | 89.2% | 88.3% | — |
| **GPQA Diamond** | 84.3% | 82.3% | — |
| **Tau2 (avg over 3)** | 76.9% | 68.2% | — |
| **HLE (no tools)** | 19.5% | 8.7% | — |
| **BigBench Extra Hard** | 74.4% | 64.8% | — |
| **MMMU Pro** | 76.9% | 73.8% | — |

¹ Agent-scaffolded (model card reports with agent framework, not raw pass@1).
² From llm-stats.com comparison tables.
³ Google describes 26B-A4B as "excelling at Terminal-Bench 2.0" but no specific number published.

## Model Details

### Gemma 4 31B (Dense)
- **Architecture**: Dense transformer, 30.7B parameters, 60 layers
- **Context**: 256K tokens
- **Modalities**: Text + Image input, Text output
- **Best for**: Deep reasoning, architecture review, security audit
- **Thor performance**: ~10 tok/s gen, ~361 tok/s prefill (Q4_K_M, Q8 KV)
- **Source**: [HuggingFace](https://huggingface.co/google/gemma-4-31b-it)

### Gemma 4 26B-A4B (MoE)
- **Architecture**: Mixture-of-Experts, 26B total / 4B active, 56 layers
- **Context**: 256K tokens
- **Modalities**: Text + Image + Audio input, Text output
- **Best for**: Fast interactive coding, research, triage, agent loops
- **Thor performance**: ~51 tok/s gen, ~1,681 tok/s prefill (Q4_K_M, Q8 KV)
- **Source**: [HuggingFace](https://huggingface.co/google/gemma-4-26B-A4B)

### Qwen3-Coder-Next (80B-A3B)
- **Architecture**: Hybrid MoE with DeltaNet layers, 80B total / 3B active
- **Context**: 128K tokens (GGUF)
- **Best for**: Coding tasks, SWE-bench class problems, agentic coding
- **Thor performance**: TBD (DeltaNet architecture)
- **Source**: [Qwen Blog](https://qwen.ai/blog?id=qwen3-coder-next), [arXiv 2603.00729](https://arxiv.org/abs/2603.00729)

## Comparison with Frontier Models (for reference)

| Model | SWE-bench Verified | Terminal-Bench 2.0 | Notes |
|-------|:------------------:|:------------------:|-------|
| Claude Opus 4.6 | ~72%* | 58.0% (Claude Code) | API model |
| GPT-5.3-Codex | — | 75.1% (Simple Codex) | API model |
| Gemini 3.1 Pro | 83.9% | 63.5%* | API model |
| **Gemma 4 31B** | **76.8%** | **50.8%** | **Runs on Thor** |
| **Qwen3-Coder-Next** | **70.6%** | **36.2%** | **Runs on Thor** |
| **Gemma 4 26B-A4B** | **63.8%** | **—** | **Runs on Thor** |

*Approximate / leaked numbers. Terminal-Bench scores depend heavily on agent scaffolding.

## Key Takeaways

1. **Gemma 4 31B achieves 76.8% SWE-bench Verified** — competitive with frontier API models at a fraction of the cost (runs locally on Thor).
2. **Qwen3-Coder-Next at 70.6% SWE-bench Verified with only 3B active params** — remarkable efficiency, purpose-built for coding agents.
3. **Gemma 4 26B-A4B trades ~13% SWE-bench accuracy for 5x generation speed** — ideal for interactive use and agent loops where latency matters more than peak accuracy.
4. **Terminal-Bench scores are agent-dependent** — the same model scores very differently with different agent scaffolding. Our numbers reflect model capability, not specific agent performance.
