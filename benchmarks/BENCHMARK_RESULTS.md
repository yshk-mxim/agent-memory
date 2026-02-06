# Benchmark Results

Performance measurements for the Semantic Cache API on Apple Silicon.

## Test Environment

| Component | Detail |
|-----------|--------|
| Hardware | Apple M4 Pro, 24 GB unified memory |
| Model | DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx (16B, Q4) |
| Framework | MLX 0.30.3, mlx-lm 0.25.2 |
| OS | macOS 15.x (Sequoia) |
| Python | 3.12 |
| KV Cache | Q4 quantized (4-bit), 256-token blocks |

All benchmarks use the OpenAI `/v1/chat/completions` API with `X-Session-ID` headers for cache affinity. Each measurement is the median of 3 runs.

---

## 1. Cold Start Latency

First-request latency at different context lengths (no prior cache).

| Context (tokens) | TTFT (ms) | E2E (ms) | TPS | Peak RAM (MB) |
|-------------------|-----------|----------|-----|---------------|
| 200 | 388 | 993 | 105.8 | 9,082 |
| 500 | 523 | 1,123 | 93.1 | 9,112 |
| 1,000 | 742 | 1,392 | 83.2 | 9,178 |
| 2,000 | 1,643 | 2,487 | 75.8 | 9,364 |
| 4,000 | 3,390 | 4,600 | 56.1 | 9,819 |
| 8,000 | 7,898 | 9,487 | 40.5 | 11,335 |
| 16,000 | 19,627 | 24,104 | 13.4 | 13,027 |
| 32,000 | 48,058 | 60,836 | 5.0 | 14,855 |

Prefill is compute-bound on Apple Silicon: TTFT scales roughly linearly with context length. Memory usage scales modestly thanks to Q4 KV quantization.

---

## 2. Multi-Turn Speedup

Three-turn conversation with ~2,000 token initial context. Turn 2 and Turn 3 reuse the KV cache from previous turns via the EXTEND path.

| Turn | E2E (ms) | TPS | Speedup vs T1 |
|------|----------|-----|----------------|
| Turn 1 (cold) | 2,409 | 26.6 | -- |
| Turn 2 (warm) | 1,229 | 52.1 | **2.0x** |
| Turn 3 (warm) | 1,246 | 51.4 | **1.9x** |

Cache reuse eliminates re-prefill of all prior context. The speedup is consistent across turns (no degradation at deeper conversation depth).

---

## 3. Prefix Sharing

Two different user queries sharing a 1,000-token system prompt. The second request reuses the cached system prefix.

| Request | E2E (ms) | TPS | Speedup |
|---------|----------|-----|---------|
| A (cold system prefix) | 1,554 | 40.2 | -- |
| B (shared prefix) | 965 | 64.4 | **1.6x** |

The shared prefix cache matches at the character level, so different user queries that share the same system prompt benefit automatically.

---

## 4. Output Scaling

Fixed 2,000-token context, varying output length.

| Output Tokens | E2E (ms) | TPS |
|---------------|----------|-----|
| 16 | 1,222 | 13.1 |
| 64 | 1,640 | 39.0 |
| 128 | 2,196 | 54.6 |
| 256 | 2,213 | 54.2 |
| 512 | 2,217 | 54.1 |

TPS stabilizes at ~54 tok/s once prefill is amortized. Short outputs (16 tokens) show low apparent TPS because prefill dominates.

---

## 5. Concurrent Batching

Two concurrent requests, each generating 64 tokens from a 2,000-token context.

| Metric | Value |
|--------|-------|
| Wall time | 4,338 ms |
| Total tokens | 128 |
| System TPS | 29.5 |
| Sequential estimate | ~4,800 ms |
| Improvement | ~11% |

Concurrent batching with interleaved prefill/decode provides modest throughput gains. The primary benefit is reduced head-of-line blocking.

---

## 6. Comparison with LM Studio

Side-by-side comparison using the same model (DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx) and identical prompts. Each server had exclusive GPU access during its benchmarks.

### Multi-Turn (3 turns, ~2K context, 64 output tokens)

| Server | Turn 1 | Turn 2 | Turn 3 | T1 to T2 |
|--------|--------|--------|--------|----------|
| **Semantic** | 2,379 ms | 1,202 ms | 1,205 ms | **2.0x** |
| LM Studio | 707 ms* | 778 ms | 2,845 ms | 0.9x |

\* LM Studio Turn 1 median includes cross-run prefix caching (runs 2-3 reused run 1's prompt cache: 1,498/1,499 tokens cached). Cold Turn 1 was 2,304 ms.

Key observations:
- LM Studio Turn 2 is faster (778 ms vs 1,202 ms) due to llama.cpp's optimized Metal kernels
- **LM Studio Turn 3 regresses** to 2,845 ms (only 1,565/3,213 tokens cached = 49%)
- **Semantic maintains consistent ~1.2s** for all follow-up turns (cache properly extends)
- Semantic's advantage grows with deeper conversations

### Prefix Sharing (1K system prefix, median of 3 runs)

| Server | Prompt A (cold) | Prompt B (reuse) | Speedup |
|--------|-----------------|------------------|---------|
| **Semantic** | 1,554 ms | 965 ms | **1.6x** |
| LM Studio | 699 ms | 702 ms | 1.0x |

LM Studio shows 1.0x median because the cross-run cache already covers both prompts after the first run. First-run speedup was 2.07x.

### Summary

| Capability | Semantic | LM Studio |
|------------|----------|-----------|
| Cold short TTFT | 388 ms | ~116 ms |
| Multi-turn consistency | Stable at all depths | Degrades at Turn 3+ |
| Max practical context | 50K+ tokens | ~4K tokens |
| Disk persistence | Yes (safetensors) | No |
| Multi-agent isolation | Yes (per-session) | No |
| Q4 KV cache | Yes (75% savings) | No |
| Cache extends across turns | Yes | Partial |

---

## Methodology

- **Output tokens**: 64 per request unless noted
- **Temperature**: 0.0 (deterministic)
- **Runs per measurement**: 3 (median reported)
- **GPU exclusivity**: Each server benchmarked alone to avoid contention
- **Warm-up**: 1 short request discarded before measurements
- **Benchmark scripts**: `benchmarks/openai_benchmark.py`, `benchmarks/comprehensive_sweep.py`

See `benchmarks/BENCHMARK_METHODOLOGY.md` for full methodology details.
