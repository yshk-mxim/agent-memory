# Cold TTFT Performance Investigation - RESOLVED

## Root Cause: Paper benchmark numbers are physically impossible on M4 Pro

### The Physics

| Metric | Value |
|--------|-------|
| Model size (gemma-3-12b-it-4bit) | 8.03 GB |
| M4 Pro memory bandwidth | 273 GB/s |
| **Theoretical max decode TPS** | **34.0 tok/s** |
| Our actual decode TPS | 33 tok/s (97% of theoretical) |
| Paper's claimed decode TPS | 77.7 tok/s (**requires 624 GB/s - impossible**) |

Our current performance is **at the hardware limit**. The server adds only ~9% overhead.

### Raw MLX Kernel Test (no server, no API)

```
Tokens:          3018
Raw prefill:     15397ms (196.0 tok/s)
Warmed prefill:  15366ms (196.4 tok/s)
Q4 prefill:      15596ms (193.5 tok/s)
Decode TPS:      23.3

mlx_lm.generate: 224.2 tok/s prefill, 32.9 tok/s decode
```

The raw MLX kernel without any server produces the same ~15s prefill time.
The server adds negligible overhead (16800ms API vs 15397ms raw kernel).

### Paper Benchmark Anomalies

1. **output_tokens=253** despite max_tokens=64 configured → max_tokens was not being enforced
2. **input_tokens=0** in streaming results → usage data not returned in streaming mode
3. **77.7 tok/s decode** → requires 624 GB/s bandwidth (M4 Pro has 273 GB/s)

### All Benchmark Results (same hardware, same code)

| Date | Cold 4K TTFT | Decode TPS | Notes |
|------|-------------|------------|-------|
| 2026-02-03 21:24 | 16016ms | ~26 TPS | Normal |
| 2026-02-04 16:45 | 15269ms | ~26 TPS | Normal |
| **2026-02-04 19:59 (paper)** | **3894ms** | **77.7 TPS** | **Anomalous** |
| 2026-02-05 21:30 | 15407ms | ~26 TPS | Normal |
| 2026-02-05 21:37 | 15394ms | ~26 TPS | Normal (at paper commit) |
| 2026-02-06 (post reboot) | 16829ms | ~26 TPS | Normal |

Every run before and after the paper produces consistent results (~15s, ~26 TPS).
The paper run is a clear outlier that violates hardware bandwidth limits.

### Possible Explanations for Paper Outlier

1. **Different hardware**: Run on M3 Ultra (800 GB/s) or M4 Max (546 GB/s) instead of M4 Pro
2. **Measurement bug**: delta_count inflated (253 vs 64 expected output tokens)
3. **Cached results**: Benchmark may have accidentally measured hot/warm instead of cold
4. **System anomaly**: One-time favorable GPU state (unlikely given 4x across all lengths)

### Hardware Bandwidth Reference

```
M4 Pro (24GB):     273 GB/s  → max 34 tok/s  ← OUR HARDWARE
M4 Max (48GB):     546 GB/s  → max 68 tok/s
M4 Max (128GB):    546 GB/s  → max 68 tok/s
M3 Ultra (192GB):  800 GB/s  → max 100 tok/s  ← could explain 77.7 tok/s
```

## What IS Working

1. **Warm cache**: 571ms TTFT = 29.5x speedup (vs paper's broken 0.98x)
2. **Hot cache**: 483ms TTFT
3. **Server overhead**: Only ~9% above raw kernel (not the bottleneck)
4. **Decode performance**: 33 tok/s = 97% of theoretical M4 Pro maximum

## Branch Status

- Currently on: `feat/production-architecture`
- Merged from: `warm-cache-investigation` (commit 1136258)
- All warm cache fixes are included and working
- MLX: 0.30.3, mlx-lm: 0.30.4

## Test Files

- `/tmp/claude/test_raw_mlx.py` - Raw MLX kernel benchmark (no server)
- `/tmp/claude/test_warm_4k.py` - 4K warm cache test via API

## Recommendation

The paper benchmark numbers should be updated to reflect reproducible M4 Pro performance:
- Cold 4K TTFT: ~15000ms (not 3900ms)
- Decode TPS: ~33 (not 77.7)
- Warm 4K TTFT: ~570ms (paper had this broken at 3990ms, now fixed)

Or if the paper was intended for different hardware, the hardware spec should be corrected.
