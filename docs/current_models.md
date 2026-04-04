# Available Models on Thor

Three models available via agent-memory on NVIDIA Jetson AGX Thor (128 GB).
One model loaded at a time for maximum context window. Default: **gemma-4-26b-a4b**.

## Models

| Model | Architecture | Active Params | Gen t/s | Context | Best For |
|-------|-------------|:------------:|--------:|--------:|----------|
| **gemma-4-26b-a4b** (default) | MoE | 4B | 51 | 262K/slot × 4 | Fast interactive, research, agent loops |
| **gemma-4-31b** | Dense | 31B | 10 | 131K/slot × 2 | Deep reasoning, architecture, security |
| **qwen3-coder-next** | Hybrid (DeltaNet) | 3B | TBD | 128K | Coding specialist (SWE-bench 70.6%) |

## Published Benchmarks

| Benchmark | gemma-4-31b | gemma-4-26b-a4b | qwen3-coder-next |
|-----------|:-----------:|:---------------:|:----------------:|
| SWE-bench Verified | 76.8% | 63.8% | 70.6% |
| SWE-bench Pro | 51% | 44% | 44.3% |
| Terminal-Bench 2.0 | 50.8% | — | 36.2% |
| LiveCodeBench v6 | 80.0% | 77.1% | — |
| MMLU Pro | 85.2% | 82.6% | — |
| AIME 2026 (no tools) | 89.2% | 88.3% | — |
| GPQA Diamond | 84.3% | 82.3% | — |

Sources: [Google Gemma 4 model card](https://huggingface.co/google/gemma-4-31b-it), [Qwen3-Coder-Next blog](https://qwen.ai/blog?id=qwen3-coder-next)

## Swapping Models

Auto-swap is disabled (Claude Code sends parallel requests with different model names, causing destructive ping-pong). Swap manually via the interactive script or admin API.

### Interactive script (recommended)

```bash
~/agent-memory/scripts/thor/swap_model.sh
```

Shows current model, lists available models with a `▸` marker, prompts for selection:
```
Currently loaded: gemma-4-26b-a4b

Available models:
  ▸ 1) gemma-4-26b-a4b   MoE   51 t/s   262K ctx   Fast interactive, research
    2) gemma-4-31b        Dense 10 t/s   131K ctx   Deep reasoning, architecture
    3) qwen3-coder-next   Hybrid         128K ctx   Coding specialist (SWE 70.6%)

Select [1-3]:
```

Also accepts direct arguments:
```bash
~/agent-memory/scripts/thor/swap_model.sh gemma-4-31b    # By name
~/agent-memory/scripts/thor/swap_model.sh 2              # By number
```

Requires `SEMANTIC_ADMIN_KEY` env var (or prompts for it).

### Admin API

```bash
curl -X POST http://localhost:8000/admin/models/swap \
  -H "X-Admin-Key: $SEMANTIC_ADMIN_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model_id": "gemma-4-31b"}'
```

### Other scripts

| Script | Purpose |
|--------|---------|
| `scripts/thor/start.sh` | Start agent-memory (default: gemma-4-26b-a4b) |
| `scripts/thor/start.sh gemma-4-31b` | Start with a specific model |
| `scripts/thor/stop.sh` | Stop agent-memory and llama-server |
| `scripts/thor/clear_cache.sh` | Clear all KV caches |
| `scripts/thor/swap_model.sh` | Interactive model swap |

Swap takes ~10-15 seconds (stop old server → start new → load GGUF → health check).

## Memory Budget (Q4_K_M weights, Q8 KV cache)

| Model | Model Size | KV Cache | Slots | Context/Slot |
|-------|:----------:|:--------:|:-----:|:------------:|
| gemma-4-26b-a4b | 16 GiB | 102 GiB | 4 | 262K |
| gemma-4-31b | 17 GiB | 101 GiB | 2 | 131K |
| qwen3-coder-next | ~16 GiB | ~102 GiB | TBD | TBD |
