# Benchmarks

Performance measurement scripts for agent-memory. All benchmarks target Apple Silicon with MLX.

## Key benchmark scripts

| Script | What it measures |
|--------|------------------|
| `full_benchmark.py` | Full paper benchmark: cold/warm TTFT, TPS, concurrent batch=2, staggered arrivals |
| `paper_benchmark.py` | Simplified version of the full benchmark for quick validation |
| `perplexity_benchmark.py` | Q4 KV cache perplexity impact vs FP16 baseline |
| `staggered_benchmark.py` | Interleaved prefill latency hiding with staggered agent arrivals |
| `wikipedia_routing_benchmark.py` | Multi-agent Wikipedia routing with cache reuse |
| `prisoner_dilemma_benchmark.py` | Multi-turn prisoner's dilemma with persistent agent memory |
| `streaming_benchmark.py` | SSE streaming latency and inter-token timing |

## Running benchmarks

All benchmarks require a running server. Start it first:

```bash
# Gemma 3 12B (default)
python -m agent_memory.entrypoints.cli serve --port 8000

# Or DeepSeek-Coder-V2-Lite
SEMANTIC_MLX_MODEL_ID="mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx" \
SEMANTIC_MLX_CACHE_BUDGET_MB=4096 \
python -m agent_memory.entrypoints.cli serve --port 8000
```

Then run the benchmark:

```bash
# Full paper benchmark (both models, ~30 min)
python benchmarks/full_benchmark.py

# Perplexity measurement
python benchmarks/perplexity_benchmark.py

# Quick streaming test
python benchmarks/streaming_benchmark.py
```

## Results

Tracked result files in `results/`:

- `full_gemma_merged.json` — Gemma 3 12B IT Q4 full benchmark
- `full_deepseek_merged.json` — DeepSeek-Coder-V2-Lite Q4 full benchmark

These are the results reported in the paper. Raw per-run files are also included for reproducibility.

## Hardware

All results measured on:
- Apple M4 Pro, 24 GB unified memory
- macOS 15.x (Sequoia)
- Python 3.12, MLX 0.30.3, mlx-lm 0.30.4

## Interpreting results

- **TTFT** (time to first token): measures prefill latency. The key metric for cache effectiveness.
- **TPS** (tokens per second): decode throughput after prefill.
- **E2E** (end-to-end): total request time including prefill and decode.
- **Cold** vs **Warm**: cold = no prior cache (full prefill), warm = cache hit (skip prefill).

The primary claim is 30-130x TTFT speedup on cache hits, depending on context length.
Longer contexts benefit more because prefill cost scales with sequence length while
cache reload cost scales with file size (which is smaller due to Q4 quantization).
