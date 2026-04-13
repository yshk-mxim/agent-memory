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
- **Thor performance (no spec decode)**: ~8.7 tok/s gen, ~361 tok/s prefill (Q4_K_M, Q8 KV)
- **Thor performance (E2B Q3 spec decode)**: ~15.4 tok/s think ON (1.76x), ~18.7 tok/s think OFF (2.10x)
- **Draft model**: Gemma 4 E2B Q3_K_XL (2.8 GB, ~55% acceptance rate)
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

## Speculative Decoding (Default for Gemma 4 31B)

The 31B dense model uses **Gemma 4 E2B Q3_K_XL** as a speculative decoding
draft model by default. The draft generates 16 candidate tokens per cycle;
the main 31B verifies in a single forward pass. Output distribution is
identical — spec decode only affects speed, never quality.

### Configuration

Set in `config/models/gemma-4-31b.toml` via `[llamacpp] extra_args`:

```toml
extra_args = [
    "--reasoning", "auto", "--reasoning-format", "deepseek", "-fa", "on",
    "--model-draft", "~/models/gemma4-e2b/gemma-4-E2B-it-UD-Q3_K_XL.gguf",
    "--gpu-layers-draft", "99",
    "--draft", "16",
]
```

- **Draft model**: `unsloth/gemma-4-E2B-it-GGUF:UD-Q3_K_XL` (2.8 GB)
- **n_slots**: 1 (draft model occupies GPU alongside main)
- **ctx_size**: 131072 (draft KV cache adds ~2 GB at full context)

### Benchmark Results

| Config | tok/s | Acceptance | Speedup |
|--------|------:|:----------:|--------:|
| No spec decode, think ON | 8.7 | — | 1.00x |
| No spec decode, think OFF | 8.9 | — | 1.00x |
| **E2B Q3, think ON** | **15.4** | **57%** | **1.76x** |
| **E2B Q3, think OFF** | **18.7** | **51%** | **2.10x** |
| E4B Q2, think ON | 13.3 | 58% | 1.52x |
| E4B Q2, think OFF | 15.3 | 65% | 1.71x |

E2B Q3 beats E4B Q2 despite slightly lower acceptance rate because the
smaller draft model generates candidates faster, offsetting the difference.

### Per-Request Thinking Control

The server runs `--reasoning auto`, which enables thinking by default.
To disable thinking per-request, callers **must** pass:

```json
{"chat_template_kwargs": {"enable_thinking": false}}
```

Without this flag, the model consumes the token budget on reasoning tokens.
`reasoning_effort: "none"` does NOT work — only `chat_template_kwargs` controls
the Jinja template's thinking behavior. agent-memory's adapter handles this
automatically via the `disable_thinking` parameter.

### Why 26B-A4B Doesn't Use Spec Decode

At 51 tok/s, the MoE model is already too fast — the E2B draft at ~2B
dense params is comparable compute to a single MoE forward pass. Benchmark
showed 0.80x (slower) with spec decode on 26B-A4B.

### Memory Budget (128 GB unified)

| Component | Size |
|-----------|-----:|
| 31B Q4_K_M weights | ~18 GB |
| E2B Q3 draft weights | ~2.8 GB |
| 31B KV cache (131K, 1 slot, Q8) | ~17 GB |
| E2B KV cache (131K, Q8) | ~2-3 GB |
| Agent memory cache (up to 4 agents) | ~32 GB |
| **Total peak** | **~74 GB** |

~54 GB headroom. Agent caches evict to disk under pressure (`evict_to_disk = true`).

### Slot Management

Different models have different slot counts (31B: 1 slot, 26B-A4B: 4 slots).
The swap orchestrator reads `n_slots` from each model's TOML config via
`LlamaCppModelLoader.current_n_slots` and resizes the `SlotTracker` on
each swap. No fixed slot count anywhere in the chain — the TOML is the
single source of truth.

## Key Takeaways

1. **Gemma 4 31B achieves 76.8% SWE-bench Verified** — competitive with frontier API models at a fraction of the cost (runs locally on Thor).
2. **Gemma 4 31B with E2B Q3 spec decode runs at 15-19 tok/s** — 1.8-2.1x faster than baseline with identical output quality.
3. **Qwen3-Coder-Next at 70.6% SWE-bench Verified with only 3B active params** — remarkable efficiency, purpose-built for coding agents.
4. **Gemma 4 26B-A4B trades ~13% SWE-bench accuracy for 5x generation speed** — ideal for interactive use and agent loops where latency matters more than peak accuracy.
5. **Terminal-Bench scores are agent-dependent** — the same model scores very differently with different agent scaffolding. Our numbers reflect model capability, not specific agent performance.
