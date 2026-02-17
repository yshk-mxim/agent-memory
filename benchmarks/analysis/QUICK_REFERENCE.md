# Benchmark Analysis Quick Reference

**Source Data**: `/Users/dev_user/agent-memory/benchmarks/results/full_gemma_20260215_232610.json`

**Analysis Files** (in this directory):
- `BENCHMARK_REPORT.md` - Complete consolidated report
- `benchmark_summary_report.md` - Detailed narrative report
- `latex_tables.tex` - 6 publication-ready LaTeX tables
- `generate_paper_stats.py` - Extract paper-ready statistics
- `analyze_benchmark_v2.py` - Full detailed analysis with tables

---

## Key Numbers for Paper (Copy-Paste Ready)

### Single Request Warm Cache Speedup
- 1K tokens: **8.3x** (3,964ms → 475ms)
- 2K tokens: **14.4x** (7,119ms → 495ms)
- 4K tokens: **27.3x** (15,736ms → 577ms)

### Concurrent Request Warm Cache Speedup (Batch=2)
- 1K tokens: **3.4x** (6,491ms → 1,930ms)
- 2K tokens: **5.7x** (11,806ms → 2,068ms)
- 4K tokens: **11.4x** (25,274ms → 2,225ms)

### Decode Throughput (Batch=1, Streaming)
- Cold cache: 23-26 tps
- Warm cache: 23-27 tps
- Hot cache: 23-26 tps
- **Cache has minimal impact on decode** (±1-3 tps variance)

---

## Quality Metrics
- Total measurements: **216**
- Quality pass rate: **100%** (216/216)
- Error rate: **0%** (0/216)
- Variance: **<4% relative to median** across 6 passes

---

## LaTeX Quick Copy

### Inline Values
```latex
\newcommand{\warmSpeedupOneK}{8.3}
\newcommand{\warmSpeedupTwoK}{14.4}
\newcommand{\warmSpeedupFourK}{27.3}
\newcommand{\warmSpeedupConcOneK}{3.4}
\newcommand{\warmSpeedupConcTwoK}{5.7}
\newcommand{\warmSpeedupConcFourK}{11.4}
\newcommand{\ttftColdOneK}{3964}
\newcommand{\ttftWarmOneK}{475}
\newcommand{\ttftColdFourK}{15736}
\newcommand{\ttftWarmFourK}{577}
\newcommand{\tpsCold}{26.1}
\newcommand{\tpsWarm}{27.1}
```

### Usage in Text
```latex
Warm cache provides \warmSpeedupOneK× speedup at 1K context and
\warmSpeedupFourK× speedup at 4K context.
```

---

## Regenerate Analysis

```bash
# Full detailed analysis
python analyze_benchmark_v2.py

# Paper-ready stats only
python generate_paper_stats.py
```

---

## Key Claims (100% Backed by Data)

✅ Cache effectiveness **increases with context size** (8.3x → 27.3x)
✅ Decode throughput **independent of cache state** (±1-3 tps)
✅ Warm cache TTFT **constant regardless of context** (~500-700ms)
✅ **100% quality pass rate** across 216 measurements
✅ Concurrent requests show **expected 2x TPS reduction** (26→11 tps)
✅ Batch=2 cold TTFT **1.6x slower** than batch=1 (concurrent prefill overhead)
✅ Batch=2 warm TTFT **4x slower** than batch=1 (sequential serving)

---

## Table Numbers (Rounded for Paper)

### Table 1: Single Request TTFT
| Context | Cold (ms) | Warm (ms) | Speedup |
|---------|-----------|-----------|---------|
| 1K      | 3,964     | 475       | 8.3×    |
| 2K      | 7,119     | 495       | 14.4×   |
| 4K      | 15,736    | 577       | 27.3×   |

### Table 2: Concurrent TTFT (Batch=2)
| Context | Cold (ms) | Warm (ms) | Speedup |
|---------|-----------|-----------|---------|
| 1K      | 6,491     | 1,930     | 3.4×    |
| 2K      | 11,806    | 2,068     | 5.7×    |
| 4K      | 25,274    | 2,225     | 11.4×   |

### Table 3: Decode TPS (Batch=1)
| Context | Cold | Warm | Hot  |
|---------|------|------|------|
| 1K      | 26.1 | 27.1 | 26.4 |
| 2K      | 24.9 | 25.7 | 25.0 |
| 4K      | 23.2 | 23.4 | 23.1 |

---

## Narrative Snippets

### Cache Benefit
> "Warm KV cache reduces TTFT from 15.7 seconds to 577ms for 4K token contexts, a **27.3× speedup**. This speedup increases with context size: 8.3× at 1K, 14.4× at 2K, and 27.3× at 4K tokens."

### Decode Stability
> "Decode throughput remains stable at 23-27 tokens/second regardless of cache state, demonstrating that cache benefits apply exclusively to the prefill phase."

### Quality Assurance
> "Across 216 measurements spanning 36 configurations with 6 passes each, we observed a **100% quality pass rate** with zero errors, demonstrating robust and consistent performance."

### Concurrent Performance
> "With concurrent requests (batch size 2), per-request decode throughput halves from 26 to 11 tokens/second as expected when sharing GPU resources. Cold-start TTFT increases by 1.6× due to concurrent prefill overhead, while warm-cache TTFT increases by 4× due to sequential serving of cached requests."

---

**Last Updated**: 2026-02-16
**Git SHA**: c0e61b6
**Benchmark Duration**: 2h 21m (04:26 - 06:47 UTC)
