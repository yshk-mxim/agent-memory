# Comprehensive Claim Audit -- COLM 2026 Paper

**Date**: 2026-02-09
**Auditor**: Claude Opus 4.6
**Paper**: `novelty/paper/semantic_colm2026.tex`
**Source data**: `benchmarks/results/colm_full_gemma_merged.json`, `benchmarks/results/colm_full_deepseek_merged.json`, `benchmarks/results/prisoner_dilemma_*.json`, `benchmarks/results/wiki_routing_*.json`
**Method**: Median of 3 passes per configuration unless otherwise noted. All benchmark JSON files loaded and cross-referenced programmatically.

---

## Summary of Findings

- **Total claims audited**: 118
- **MATCH**: 91 (claim matches source data within rounding tolerance)
- **MINOR**: 13 (off by rounding, <2% discrepancy)
- **MISMATCH**: 10 (claim does not match data or calculation)
- **NEEDS-CHECK**: 4 (external claim, cannot verify from benchmark data alone)

### Critical Issues

1. **Staggered arrival 1.36x speedup (Figure 4)**: Compares inconsistent metrics (absolute time for sequential vs relative time for batched). True speedup is ~1.04x.
2. **Section 4.1 measurement count**: Claims 72 configs / 216 measurements / 198 quality-passing. Actual: 66 configs / 198 measurements / 198 quality-passing. Contradicts own Appendix C.
3. **Figure 2 caption**: Says "27 vs 46 layers" -- Gemma has 48 layers, not 46.
4. **Figure 2 caption**: Claims "40-130x at 32K" -- actual range at 32K is 69-130x.
5. **Wiki routing text**: Claims "largest cold TTFT is 31s (Monte Carlo)" -- actual max is 28.3s (central_limit_theorem); Monte Carlo is 20.8s.
6. **Appendix D per-layer breakdown**: Intermediate calculation has two canceling errors (forgot /2 for data, forgot x2 for scales+biases). Final total is correct.

---

## Table 1: Hardware Specifications (Table 1 in paper)

| Claim | Value | Source | Status |
|-------|-------|--------|--------|
| M4 Pro memory | 24 GB | Apple spec sheet | NEEDS-CHECK |
| M4 Pro bandwidth | 273 GB/s | Apple spec sheet | NEEDS-CHECK |
| M4 Pro SSD | 7 GB/s | Apple spec sheet | NEEDS-CHECK |
| M4 Max memory | 128 GB | Apple spec sheet | NEEDS-CHECK |
| M4 Max bandwidth | 546 GB/s | Apple spec sheet | MATCH (published) |
| DGX Spark memory | 128 GB | NVIDIA spec | MATCH (published) |
| DGX Spark bandwidth | 273 GB/s | NVIDIA spec | MATCH (published) |
| RTX 5090 VRAM | 32 GB | NVIDIA spec | MATCH (published) |
| RTX 5090 bandwidth | 1792 GB/s | NVIDIA spec | MATCH (published) |
| RTX 5090 PCIe bandwidth | 64 GB/s | PCIe 5.0 x16 spec | MATCH (published) |
| RTX 4090 VRAM | 24 GB | NVIDIA spec | MATCH (published) |
| RTX 4090 bandwidth | 1008 GB/s | NVIDIA spec | MATCH (published) |
| iPhone 17 Pro memory | 12 GB | Projected spec | NEEDS-CHECK |
| iPhone 17 Pro bandwidth | 77 GB/s | Projected spec | NEEDS-CHECK |
| RTX 5090 bandwidth cliff | 28x | 1792/64 = 28 | **MATCH** |

---

## Table 2: FP16 vs Q4 Agent Capacity (Table 2 in paper, Gemma only)

Budget: 24 GB - 6.8 GB weights - 2 GB OS = 15.2 GB. Gemma 3 12B: 48 layers, 8 KV heads, head dim 256, group size 64.

| Context | Paper FP16/agent | Calc FP16/agent | Paper Q4/agent | Calc Q4/agent | Paper FP16 fits | Calc FP16 fits | Paper Q4 fits | Calc Q4 fits | Status |
|---------|-----------------|-----------------|---------------|---------------|----------------|----------------|--------------|-------------|--------|
| 4K | 1.5 GB | 1.50 GB | 0.42 GB | 0.42 GB | 10 | 10 | 36 | 36 | **MATCH** |
| 8K | 3.0 GB | 3.00 GB | 0.84 GB | 0.84 GB | 5 | 5 | 18 | 18 | **MATCH** |
| 16K | 6.0 GB | 6.00 GB | 1.7 GB | 1.69 GB | 2 | 2 | 8 | 9 | **MINOR** (Q4 fits: floor(15.2/1.69)=9 but paper rounds 1.69->1.7, floor(15.2/1.7)=8) |
| 32K | 12.0 GB | 12.00 GB | 3.4 GB | 3.38 GB | 1 | 1 | 4 | 4 | **MATCH** |

### Q4/FP16 ratio
| Claim | Value | Calculation | Status |
|-------|-------|-------------|--------|
| Q4/FP16 ratio | 0.281 | (0.5 + 4/64) / 2 = 0.28125 | **MATCH** |
| Memory reduction | 72% | 1 - 0.281 = 0.719 | **MATCH** |

---

## Table 3: TTFT by Cache State (Table 3 in paper)

All values: streaming, batch=1, median of 3 passes.

### Gemma 3

| Context | Cache | Paper (ms) | Data (ms) | Status |
|---------|-------|-----------|-----------|--------|
| 1K | Cold | 4007 | 4007 | **MATCH** |
| 1K | Warm | 527 | 527 | **MATCH** |
| 1K | Hot | 668 | 668 | **MATCH** |
| 2K | Cold | 7363 | 7363 | **MATCH** |
| 2K | Warm | 532 | 532 | **MATCH** |
| 2K | Hot | 688 | 688 | **MATCH** |
| 4K | Cold | 15502 | 15502 | **MATCH** |
| 4K | Warm | 513 | 513 | **MATCH** |
| 4K | Hot | 709 | 709 | **MATCH** |
| 8K | Cold | 32944 | 32944 | **MATCH** |
| 8K | Warm | 590 | 590 | **MATCH** |
| 8K | Hot | 762 | 762 | **MATCH** |
| 16K | Cold | 71132 | 71132 | **MATCH** |
| 16K | Warm | 808 | 808 | **MATCH** |
| 16K | Hot | 874 | 874 | **MATCH** |
| 32K | Cold | 165189 | 165189 | **MATCH** |
| 32K | Warm | 1621 | 1621 | **MATCH** |
| 32K | Hot | 1276 | 1276 | **MATCH** |

### DeepSeek

| Context | Cache | Paper (ms) | Data (ms) | Status |
|---------|-------|-----------|-----------|--------|
| 1K | Cold | 1090 | 1090 | **MATCH** |
| 1K | Warm | 217 | 217 | **MATCH** |
| 1K | Hot | 356 | 356 | **MATCH** |
| 2K | Cold | 1884 | 1884 | **MATCH** |
| 2K | Warm | 285 | 285 | **MATCH** |
| 2K | Hot | 376 | 376 | **MATCH** |
| 4K | Cold | 3949 | 3949 | **MATCH** |
| 4K | Warm | 252 | 252 | **MATCH** |
| 4K | Hot | 372 | 372 | **MATCH** |
| 8K | Cold | 8541 | 8541 | **MATCH** |
| 8K | Warm | 307 | 307 | **MATCH** |
| 8K | Hot | 412 | 412 | **MATCH** |
| 16K | Cold | 19193 | 19192 | **MINOR** (data median=19192.5, paper rounds to 19193) |
| 16K | Warm | 430 | 430 | **MATCH** |
| 16K | Hot | 484 | 484 | **MATCH** |
| 32K | Cold | 48258 | 48258 | **MATCH** |
| 32K | Warm | 697 | 697 | **MATCH** |
| 32K | Hot | 652 | 652 | **MATCH** |

---

## Table 4: Batched Throughput (Table 4 in paper)

All values: batch=2, non-streaming, median of 3 passes. SysTPS = system tokens/second. Per = per-agent TPS = SysTPS/2.

### Gemma 3

| Context | Cache | Paper SysTPS | Data SysTPS | Paper Per | Data Per | Status |
|---------|-------|-------------|-------------|-----------|----------|--------|
| 1K | Cold | 10.2 | 10.2 | 5.1 | 5.1 | **MATCH** |
| 1K | Warm | 22.4 | 22.4 | 11.2 | 11.2 | **MATCH** |
| 1K | Hot | 22.0 | 22.0 | 11.0 | 11.0 | **MATCH** |
| 4K | Cold | 3.3 | 3.3 | 1.6 | 1.6 | **MATCH** |
| 4K | Warm | 19.8 | 19.8 | 9.9 | 9.9 | **MATCH** |
| 4K | Hot | 20.0 | 20.0 | 10.0 | 10.0 | **MATCH** |
| 16K | Cold | 0.8 | 0.8 | 0.4 | 0.4 | **MATCH** |
| 16K | Warm | 13.3 | 13.3 | 6.7 | 6.7 | **MATCH** |
| 16K | Hot | 13.6 | 13.6 | 6.8 | 6.8 | **MATCH** |

### DeepSeek

| Context | Cache | Paper SysTPS | Data SysTPS | Paper Per | Data Per | Status |
|---------|-------|-------------|-------------|-----------|----------|--------|
| 1K | Cold | 43.6 | 43.6 | 21.8 | 21.8 | **MATCH** |
| 1K | Warm | 64.8 | 64.8 | 32.4 | 32.4 | **MATCH** |
| 1K | Hot | 65.2 | 65.2 | 32.6 | 32.6 | **MATCH** |
| 4K | Cold | 13.8 | 13.8 | 6.9 | 6.9 | **MATCH** |
| 4K | Warm | 55.1 | 55.1 | 27.6 | 27.6 | **MATCH** |
| 4K | Hot | 55.8 | 55.8 | 27.9 | 27.9 | **MATCH** |
| 16K | Cold | 3.2 | 3.2 | 1.6 | 1.6 | **MATCH** |
| 16K | Warm | 28.2 | 28.2 | 14.1 | 14.1 | **MATCH** |
| 16K | Hot | 35.9 | 35.9 | 18.0 | 18.0 | **MATCH** |

---

## Table 5: Ablation Analysis (Table 5 in paper)

| Component | Metric | Paper "With" | Paper "Without" | Paper Effect | Verified | Status |
|-----------|--------|-------------|----------------|-------------|----------|--------|
| Persistence | TTFT Gemma 4K | 513 | 15502 | 30x | 15502/513=30.2x | **MATCH** (rounds to 30x) |
| Q4 vs FP16 | Agents at 8K | 18 | 5 | 3.6x | 18/5=3.6x | **MATCH** |
| Batching | SysTPS Gemma 1K warm | 22.4 | 11.2* | 2.0x | 22.4/11.2=2.0x | **MATCH** (footnote clarifies 11.2 = SysTPS/2) |
| Cross-phase | TTFT Phase 5 | 1705 | 3292 | 1.9x | 3292/1705=1.93x | **MATCH** |

Note: The "without batching" baseline uses per-agent TPS from batch=2 (SysTPS/2=11.2), not actual batch=1 throughput (which is 22.8 TPS). The footnote correctly discloses this, but the 2.0x improvement is a tautology (SysTPS vs SysTPS/2 is always 2.0x by definition). The meaningful comparison would be batch=2 SysTPS (22.4) vs batch=1 TPS (22.8), showing batching yields ~equivalent total throughput (not 2x improvement).

---

## Table 6: Multi-Phase Prisoner Dilemma (Table 6 in paper)

### Gemma 3

| Phase | Paper Cold | Data Cold | Paper Pers | Data Pers | Paper x | Calc x | Status |
|-------|-----------|-----------|-----------|-----------|---------|--------|--------|
| 1: Interrogation A | 1136 | 1135.6 | 1079 | 1079.3 | 1.1 | 1.05 | **MINOR** (rounds 1.05 to 1.1) |
| 2: Interrogation B | 1119 | 1118.7 | 976 | 975.6 | 1.2 | 1.15 | **MINOR** (rounds 1.15 to 1.2) |
| 3: The Yard | 1648 | 1647.9 | 1019 | 1019.1 | 1.6 | 1.62 | **MATCH** |
| 4: Final Reckoning | 2195 | 2194.7 | 1250 | 1250.1 | 1.8 | 1.76 | **MINOR** (rounds 1.76 to 1.8) |
| 5: Verdict | 3292 | 3291.5 | 1705 | 1704.7 | 1.9 | 1.93 | **MATCH** |
| Total wall (s) | 72.9 | 72.9 | 56.1 | 56.1 | 1.3 | 1.30 | **MATCH** |

### DeepSeek

| Phase | Paper Cold | Data Cold | Paper Pers | Data Pers | Paper x | Calc x | Status |
|-------|-----------|-----------|-----------|-----------|---------|--------|--------|
| 1: Interrogation A | 477 | 476.9 | 460 | 460.1 | 1.0 | 1.04 | **MATCH** (1.04 rounds to 1.0 at 1dp) |
| 2: Interrogation B | 465 | 465.4 | 430 | 429.5 | 1.1 | 1.08 | **MATCH** |
| 3: The Yard | 532 | 532.2 | 474 | 474.2 | 1.1 | 1.12 | **MATCH** |
| 4: Final Reckoning | 664 | 664.3 | 542 | 541.5 | 1.2 | 1.23 | **MATCH** |
| 5: Verdict | 874 | 874.4 | 649 | 648.8 | 1.3 | 1.35 | **MATCH** |
| Total wall (s) | 33.6 | 33.6 | 27.8 | 27.8 | 1.2 | 1.21 | **MATCH** |

---

## Table 7: Wikipedia Routing (Table 7 in paper)

### Gemma 3

| Phase | Paper TTFT | Data TTFT avg | Paper Quality | Data Quality | Status |
|-------|-----------|---------------|---------------|-------------|--------|
| 1: Priming (cold) | 20514 | 20513.5 | 8/10 | 8/10 | **MATCH** |
| 2: Queries (warm) | 847 | 846.6 | 8/10 | 8/10 | **MATCH** |
| 3: Repeated (hot) | 860 | 860.3 | 3/3 | 3/3 | **MATCH** |
| Warm/cold speedup | 24.2x | 24.23x | -- | -- | **MATCH** |
| Hot/cold speedup | 23.8x | 23.84x | -- | -- | **MATCH** |

### DeepSeek

| Phase | Paper TTFT | Data TTFT avg | Paper Quality | Data Quality | Status |
|-------|-----------|---------------|---------------|-------------|--------|
| 1: Priming (cold) | 5140 | 5140.1 | 3/10 | 3/10 | **MATCH** |
| 2: Queries (warm) | 396 | 396.1 | 4/10 | 4/10 | **MATCH** |
| 3: Repeated (hot) | 424 | 424.1 | 2/3 | 2/3 | **MATCH** |
| Warm/cold speedup | 13.0x | 12.98x | -- | -- | **MATCH** |
| Hot/cold speedup | 12.1x | 12.12x | -- | -- | **MATCH** |

---

## Appendix Table: FP16 vs Q4, Both Models (Appendix D)

### DeepSeek (27 layers, 16 KV heads, K=192, V=128)

| Context | Paper FP16 fits | Calc FP16 fits | Paper Q4 fits | Calc Q4 fits | Status |
|---------|----------------|----------------|--------------|-------------|--------|
| 4K | 14 | 14 | 50 | 51 | **MINOR** (paper rounds 303.8 MB to 304 MB, floor(15200/304)=50 vs floor(15200/303.8)=50.04, but exact gives 51) |
| 8K | 7 | 7 | 25 | 25 | **MATCH** |
| 16K | 3 | 3 | 12 | 12 | **MATCH** |
| 32K | 1 | 1 | 6 | 6 | **MATCH** |

---

## Derived Claims: Speedup Ratios

### Abstract and Introduction

| Claim Location | Claim Text | Claimed Value | Calculated Value | Status |
|---------------|------------|--------------|-----------------|--------|
| Abstract | Warm disk reload Gemma 4K | 15.5s to 513ms (30x) | 15502/513=30.2x | **MATCH** |
| Abstract | Warm disk reload DeepSeek 4K | 3.9s to 252ms (16x) | 3949/252=15.7x | **MATCH** (rounds to 16x) |
| Abstract | Hot speedup Gemma 32K | 130x | 165189/1276=129.5x | **MATCH** (rounds to 130x) |
| Abstract | Hot speedup DeepSeek 32K | 74x | 48258/652=74.0x | **MATCH** |
| Abstract | Warm speedup Gemma 32K | not in abstract | 101.9x | -- |
| Abstract | SysTPS Gemma 1K warm batch=2 | 22.4 | 22.4 | **MATCH** |
| Abstract | SysTPS DeepSeek 1K warm batch=2 | 64.8 | 64.8 | **MATCH** |
| Abstract | Q4 fits 3.6x more agents than FP16 | 3.6x | 18/5=3.6x | **MATCH** |
| Abstract | <0.1 perplexity degradation | <0.1 | Appendix table empty ("---") | **MISMATCH** (data not collected; claim sourced from cited literature, not own measurements) |
| Abstract | Reload in under 700ms | <700ms | Gemma 4K warm=513ms, hot=709ms | **MINOR** (hot exceeds 700ms; warm is under. Context-dependent.) |
| Intro | 5 agents * 15.5s = 77s | 77s | 5*15.502=77.5s | **MINOR** (77.5 not 77) |
| Intro | Apple Silicon ~260 tok/s | ~260 | 4096/15.502=264 | **MATCH** (approximate) |
| Intro | Datacenter gap 40x | 40x | 10000/264=37.9x | **MINOR** (38x, not 40x; approximate) |
| Intro | Context restoration 513ms (warm) or 709ms (hot) at 4K | 513/709 | 512.5/708.8 | **MATCH** |

### Section 4.2 (TTFT Scaling)

| Claim | Claimed Value | Calculated Value | Status |
|-------|--------------|-----------------|--------|
| Gemma 32K takes 165 seconds (2.75 min) | 165s / 2.75min | 165.189s / 2.75min | **MATCH** |
| DeepSeek 3.4-3.9x faster cold prefill | 3.4-3.9x | Range: 3.4x (32K) to 3.9x (2K,4K,8K) | **MATCH** |
| Gemma warm 513-1621ms across 1K-32K | 513-1621 | 527-1621 (1K=527, lowest is 4K=513) | **MINOR** (range start 527 at 1K, not 513; 513 is 4K) |
| DeepSeek warm 217-697ms across 1K-32K | 217-697 | 217-697 | **MATCH** |
| Gemma warm 32K is 102x faster | 102x | 165189/1621=101.9x | **MATCH** |
| DeepSeek warm 32K is 69x faster | 69x | 48258/697=69.2x | **MATCH** |
| Gemma hot 32K is 130x faster | 130x | 165189/1276=129.5x | **MATCH** (rounds to 130x) |
| DeepSeek hot 32K is 74x faster | 74x | 48258/652=74.0x | **MATCH** |
| Gemma hot 668-1276ms | 668-1276 | 668-1276 | **MATCH** |
| DeepSeek hot 356-652ms | 356-652 | 356-652 | **MATCH** |
| Hot-warm gap within 2x | <2x | Max ratio: Gemma 1K hot/warm=668/527=1.27x | **MATCH** |

### Section 4.3 (Batched Throughput)

| Claim | Claimed Value | Calculated Value | Status |
|-------|--------------|-----------------|--------|
| DeepSeek consistently 3x faster than Gemma | 3x | Range: 2.1x (16K) to 2.9x (1K) | **MISMATCH** ("3x" overstates; actual range 2.1-2.9x) |
| DeepSeek 4K warm: 55.1 SysTPS (27.6 per) vs Gemma 19.8 (9.9) | 55.1/19.8 | 55.1/19.8 | **MATCH** |

### Section 4.5 (Multi-Phase)

| Claim | Claimed Value | Calculated Value | Status |
|-------|--------------|-----------------|--------|
| Phase 5 persistent 1.9x faster (Gemma) | 1.9x | 3292/1705=1.93x | **MATCH** |
| Phase 5 persistent 1.3x faster (DeepSeek) | 1.3x | 874/649=1.35x | **MATCH** |
| Gemma total wall 23% reduction | 23% | (72906-56059)/72906=23.1% | **MATCH** |
| DeepSeek total wall 17% reduction | 17% | (33580-27833)/33580=17.1% | **MATCH** |

### Section 4.6 (Wiki Routing)

| Claim | Claimed Value | Calculated Value | Status |
|-------|--------------|-----------------|--------|
| Cold priming averages 20.5s (Gemma) | 20.5s | 20513.5ms=20.5s | **MATCH** |
| Cold priming averages 5.1s (DeepSeek) | 5.1s | 5140.1ms=5.1s | **MATCH** |
| Warm 847ms Gemma, 24.2x faster | 847ms / 24.2x | 846.6ms / 24.23x | **MATCH** |
| Warm 396ms DeepSeek, 13.0x | 396ms / 13.0x | 396.1ms / 12.98x | **MATCH** |
| Largest cold TTFT is 31s (Monte Carlo at 3K words) | 31s, Monte Carlo | Max is 28.3s, central_limit_theorem | **MISMATCH** (wrong expert name and wrong value) |
| Reduced to 761ms warm | 761ms | Monte Carlo query phase: 746.8ms | **MISMATCH** (off by 14ms, and Monte Carlo is not the max-TTFT expert) |
| Quality 80% (Gemma Phase 2) | 80% | 8/10=80% | **MATCH** |
| Quality 30% (DeepSeek Phase 1) | 30% | 3/10=30% | **MATCH** |

---

## Conclusion Claims

| Claim | Claimed Value | Calculated Value | Status |
|-------|--------------|-----------------|--------|
| Gemma 32K hot: 165s to 1.3s (130x) | 1.3s / 130x | 1.276s / 129.5x | **MATCH** |
| DeepSeek 32K: 48s to 652ms (74x) | 74x | 48258/652=74.0x | **MATCH** |
| Warm 32K: 102x and 69x | 102x / 69x | 101.9x / 69.2x | **MATCH** |
| Q4 fits 3.6x more (18 vs 5 at 8K) | 18 vs 5 = 3.6x | 18/5=3.6x | **MATCH** |
| Batched: 22 SysTPS (Gemma), 65 (DeepSeek) 1K warm | 22 / 65 | 22.4 / 64.8 | **MATCH** (rounded) |
| 1.9x TTFT reduction, 23% wall time | 1.9x / 23% | 1.93x / 23.1% | **MATCH** |
| 24x TTFT reduction wiki routing | 24x | 24.2x | **MATCH** |

---

## Figure-Specific Claims

### Figure 1 (Architecture Diagram)

| Annotation | Claimed Value | Verified Value | Status |
|-----------|--------------|---------------|--------|
| 2.0-4.3x E2E speedup | 2.0-4.3x | Not directly verifiable from TTFT data; appears to be warm/cold E2E ratio | **NEEDS-CHECK** |
| 72% memory savings | 72% | 1-0.281=71.9% | **MATCH** |
| 81.6x TTFT (hot) | 81.6x | Gemma 16K hot/cold=81.4x (closest match) | **MINOR** (81.4 vs 81.6) |

### Figure 2 (TTFT Scaling)

| Claim | Claimed Value | Verified Value | Status |
|-------|--------------|---------------|--------|
| Caption: "27 vs 46 layers" | 46 (Gemma) | 48 | **MISMATCH** (Gemma has 48 layers, not 46) |
| Caption: "40-130x at 32K" | 40-130x | 69-130x at 32K | **MISMATCH** (minimum at 32K is 69x for DeepSeek warm, not 40x) |
| Caption: "sub-second reload regardless of context" | <1s at all ctx | Gemma warm 16K=808ms, 32K=1621ms | **MISMATCH** (Gemma warm 32K is 1.6s, not sub-second) |
| TikZ data points match Table 3 | -- | All 36 data points match | **MATCH** |
| 130x annotation arrow at 32K | 130x | 129.5x | **MATCH** |

### Figure 4 (Staggered Arrivals)

| Claim | Claimed Value | Data Value | Status |
|-------|--------------|-----------|--------|
| Gemma Seq total wall | 38.8s | median=38.8s | **MATCH** |
| Gemma Bat total wall | 38.6s | median=38.6s | **MATCH** |
| Gemma Seq B wait | 36.5s | median(B_delay+B_ttft)=36.1s | **MINOR** (36.1 not 36.5) |
| Gemma Bat B wait | 33.8s | median(B_ttft)=33.8s | **MATCH** |
| DeepSeek Seq total wall | 9.5s | median=9.5s | **MATCH** |
| DeepSeek Bat total wall | 9.4s | median=9.4s | **MATCH** |
| DeepSeek Seq B wait | 8.7s | median(B_delay+B_ttft)=8.6s | **MINOR** (8.6 not 8.7) |
| DeepSeek Bat B wait | 6.4s | median(B_ttft)=6.4s | **MATCH** |
| DeepSeek batched 1.36x faster | 1.36x | **INCONSISTENT METRIC** | **MISMATCH** (see below) |
| Gemma batched 0.5% faster wall | 0.5% | 38.8/38.6=1.005=0.5% | **MATCH** |

**Critical Issue -- Staggered 1.36x claim**: The figure compares `B_start_delay + B_ttft` (absolute from t=0) for sequential mode against just `B_ttft` (relative to B's processing start) for batched mode. These are different metrics:

- Sequential B perceived wait from submission: `(B_start_delay - 2000) + B_ttft` = 6674ms = 6.7s
- Batched B perceived wait from submission: `(B_start_delay - 2000) + B_ttft` = 6434ms = 6.4s
- True speedup from B's perspective: 6.7/6.4 = **1.04x** (not 1.36x)

Alternatively, measuring consistently as absolute time from experiment start:
- Sequential: `B_start_delay + B_ttft` = 8674ms = 8.7s
- Batched: `B_start_delay + B_ttft` = 8434ms = 8.4s
- Speedup: 8.7/8.4 = **1.03x** (not 1.36x)

The 1.36x (=8.7/6.4) compares an absolute time against a relative time, which is not a valid speedup measurement.

---

## Methodology Claims (Section 4.1)

| Claim | Paper Value | Actual Value | Status |
|-------|-----------|-------------|--------|
| 72 unique configurations per model | 72 | 66 (32K batch=2 missing, all 6 combos) | **MISMATCH** |
| 216 individual measurements per model | 216 | 198 | **MISMATCH** |
| 198 passing quality checks | 198 | 198 (but ALL 198 pass, not 198 of 216) | **MISMATCH** (implies 18 failed; actually 0 failed) |
| Appendix C: 66 configs, 198 measurements | 66/198 | 66/198 | **MATCH** |
| 3 passes per configuration | 3 | 3 (verified for all configs) | **MATCH** |
| Temperature 0.0 (greedy) | T=0.0 | Config file confirms T=0.0 | **MATCH** |
| Output length 64 tokens | 64 | Config file confirms 64 | **MATCH** |
| 30-240s adaptive cooldown | 30-240s | Config: min=10s, max=240s | **MINOR** (config min is 10s, paper says 30s) |
| Median values reported | median | Confirmed via 3-pass median calculation | **MATCH** |

**Note**: Section 4.1 and Appendix C contradict each other. Section 4.1 claims 72 configs / 216 measurements / 198 quality-passing. Appendix C correctly states 66 configs / 198 measurements. The 32K batch=2 configurations were not run (likely due to OOM), reducing 72 to 66.

---

## Background Section Claims

| Claim | Source | Status |
|-------|--------|--------|
| 25x attention cost (5 agents 4K -> 20K) | (20K)^2/(4K)^2 = 25 | **MATCH** |
| 24 GB - 6.8 GB - 2 GB = 15.2 GB budget | Arithmetic | **MATCH** |
| 100ms / 1s / 10s Nielsen thresholds | Nielsen 1993 citation | **MATCH** (cited) |
| Prefill is 84% of latency (4K, 3s decode) | 15502/(15502+3000)=83.8% | **MATCH** |
| Prefill is 94% (50 tokens, 1s decode) | 15502/(15502+1000)=93.9% | **MATCH** |
| RAG prefill 95.5% | fusionragcache2025 citation | **MATCH** (cited) |
| Gemma warm 4K crosses 1s Nielsen threshold | 513ms < 1000ms | **MATCH** |

---

## Architecture Claims (Section 3)

| Claim | Verified Against | Status |
|-------|-----------------|--------|
| Gemma 48 layers, 8 global + 40 sliding window (1024) | MEMORY.md / HF config | **MATCH** |
| Gemma 8 KV heads, 16 query heads, GQA n_rep=2, head_dim=256 | MEMORY.md / HF config | **MATCH** |
| DeepSeek 27 layers, 16 KV heads, K=192 (128+64), V=128, MLA | MEMORY.md / HF config | **MATCH** |
| DeepSeek MoE 2 of 6 active experts | MEMORY.md | **MATCH** |
| Block size 256 tokens | Code + appendix | **MATCH** |
| Group size 64 | Code + appendix | **MATCH** |
| Q4 ratio 0.281 | (0.5+4/64)/2=0.28125 | **MATCH** |
| 4096 MB cache budget for DeepSeek (MoE overhead) | Code / CLAUDE.md | **MATCH** |

---

## Appendix D: FP16 vs Q4 Per-Layer Breakdown

| Calculation Step | Paper Value | Correct Value | Status |
|-----------------|-----------|--------------|--------|
| FP16 per-layer at 4K | 33,554,432 bytes = 32 MB | 33,554,432 = 32 MB | **MATCH** |
| FP16 total (48 layers) | 1,536 MB | 1,536 MB | **MATCH** |
| Q4 data per-layer at 4K | 16,777,216 bytes | 8,388,608 bytes | **MISMATCH** (forgot /2 for 4-bit packing) |
| Q4 scales+biases per-layer | 524,288 bytes | 1,048,576 bytes | **MISMATCH** (forgot x2 for separate scales AND biases) |
| Q4 per-layer total | 17,301,504 = 16.5 MB | 9,437,184 = 9.0 MB | **MISMATCH** (two errors cancel) |
| Q4 per-layer x 48 | 792 MB | 432 MB | **MISMATCH** (intermediate; paper acknowledges with "With overhead: ~432 MB") |
| Q4 final total via ratio | ~432 MB | 432 MB | **MATCH** (the ratio calculation is correct) |
| Ratio 432/1536 | 0.281 | 0.281 | **MATCH** |
| DeepSeek FP16 at 4K: K=25,165,824, V=16,777,216 | Stated | 16*192*4096*2=25,165,824; 16*128*4096*2=16,777,216 | **MATCH** |
| DeepSeek per layer 40 MB | 40 MB | (25165824+16777216)/1024^2=40 MB | **MATCH** |
| DeepSeek x 27 layers = 1,080 MB | 1,080 MB | 40*27=1,080 MB | **MATCH** |
| DeepSeek Q4 total = 304 MB | 304 MB | 1080*0.281=303.5 MB | **MINOR** (303.5 rounded to 304) |

**Note on canceling errors**: The per-layer Q4 breakdown has two arithmetic errors:
1. Data should be `(2*8*256*4096)/2 = 8,388,608` not `16,777,216` (forgot the /2 for 4-bit)
2. Scales+biases should count both scales AND biases: `2 * 8 * 256 * 64 * 2 * 2 = 1,048,576` not `524,288`

The errors conveniently cancel (one halves, one doubles), and the final total via the ratio formula (1536 * 0.281 = 432 MB) is correct. However, the intermediate "792 MB" with "With overhead: ~432 MB" makes no sense -- 432 is less than 792, not "with overhead." This sentence should be rewritten.

---

## Cross-Reference Consistency

### Speedup claims across sections

| Value | Abstract | Intro | Section 4 | Conclusion | Consistent? |
|-------|----------|-------|-----------|-----------|-------------|
| Gemma 4K warm 30x | "30x" | "15.5s to 513ms" | Table 3: 15502/513=30x | -- | **YES** |
| DeepSeek 4K warm 16x | "16x" | -- | Table 3: 3949/252=16x | -- | **YES** |
| Gemma 32K hot 130x | "130x" | -- | "130x" | "130x" | **YES** |
| DeepSeek 32K hot 74x | "74x" | -- | "74x" | "74x" | **YES** |
| SysTPS 22.4/64.8 | "22.4 and 64.8" | -- | Table 4 | "22 and 65" (rounded) | **YES** |
| 3.6x more agents | "3.6x" | Contributions | Table 5 | "3.6x (18 vs 5)" | **YES** |
| <0.1 perplexity | "<0.1" | -- | Sec 5 | "<0.1" | **Unverified** (table empty) |
| Phase 5 1.9x | -- | -- | Table 6/Sec 4.5 | "1.9x" | **YES** |
| 23% wall time | -- | -- | Table 6/Sec 4.5 | "23%" | **YES** |
| 24x wiki routing | -- | -- | Table 7/Sec 4.6 | "24x" | **YES** |

### Model spec consistency

| Spec | Section 3 | Section 4 | Appendix C | Appendix D | Figure 2 caption |
|------|-----------|-----------|-----------|-----------|-------------------|
| Gemma layers | 48 | 48 | 48 | 48 | **46** (ERROR) |
| Gemma KV heads | 8 | 8 | 8 | 8 | -- |
| Gemma head dim | 256 | 256 | -- | 256 | -- |
| DeepSeek layers | 27 | 27 | 27 | 27 | 27 |
| DeepSeek KV heads | 16 | 16 | -- | 16 | -- |
| DeepSeek K/V dim | 192/128 | 192/128 | K=192/V=128 | 192/128 | -- |

---

## Perplexity Claims

| Claim | Source | Status |
|-------|--------|--------|
| <0.1 perplexity degradation Q4 vs FP16 | Cited literature (KIVI, KVQuant) | Own Table 9 is empty ("---") |
| Appendix E references prior work at <0.1 | KIVI, KVQuant, QuantSpec | Correctly cited |
| "consistent with prior Q4 quantization results" | Literature review | Fair characterization |

**Assessment**: The paper claims <0.1 PPL degradation in the abstract and conclusion, but Table 9 (Appendix E) has no data -- entries are "---" with "[Results to be filled]." The claim is supported only by cited literature, not own measurements. This should be clearly flagged as "expected based on prior work" rather than stated as a measured result.

---

## Summary of All MISMATCH Issues

| # | Location | Issue | Severity |
|---|----------|-------|----------|
| 1 | Figure 4 + Appendix F | 1.36x staggered speedup uses inconsistent metrics (absolute vs relative time). True speedup is ~1.04x. | **HIGH** |
| 2 | Section 4.1 | Claims 72 configs / 216 measurements / 198 quality-passing. Actual: 66 / 198 / 198. Contradicts Appendix C. | **MEDIUM** |
| 3 | Figure 2 caption | "27 vs 46 layers" -- Gemma has 48 layers. | **MEDIUM** |
| 4 | Figure 2 caption | "40-130x at 32K" -- actual range at 32K is 69-130x. | **MEDIUM** |
| 5 | Figure 2 caption | "sub-second reload regardless of context" -- Gemma warm at 32K is 1621ms. | **MEDIUM** |
| 6 | Section 4.6 text | "largest cold TTFT is 31s (Monte Carlo)" -- max is 28.3s (central_limit_theorem). | **MEDIUM** |
| 7 | Section 4.6 text | "reduced to 761ms warm" -- Monte Carlo query TTFT is 746.8ms. | **LOW** |
| 8 | Section 4.3 | "DeepSeek consistently 3x faster" -- actual range 2.1-2.9x. | **LOW** |
| 9 | Abstract | "<0.1 perplexity degradation" -- own data table is empty; claim from literature only. | **MEDIUM** |
| 10 | Appendix D | Per-layer Q4 breakdown has two canceling arithmetic errors; "with overhead: ~432 MB" after computing 792 MB is confusing. | **LOW** |

---

## Recommendations

1. **Fix staggered figure**: Use consistent metric for both modes (e.g., B's perceived wait from submission time, or absolute time from t=0). The 1.36x claim must be corrected.
2. **Fix measurement counts in Section 4.1**: Change to "66 unique configurations per model, each measured 3 times = 198 individual measurements" or explain why 32K batch=2 was excluded.
3. **Fix Figure 2 caption**: Change "46" to "48" for Gemma layers. Change "40-130x" to "69-130x" or qualify the context length. Remove "sub-second regardless of context" or limit to specific contexts.
4. **Fix wiki routing text**: Correct the max cold TTFT expert and value.
5. **Fix Appendix D**: Correct the per-layer intermediate calculations or remove them, keeping only the ratio-based final answer.
6. **Qualify perplexity claim**: State "expected <0.1 based on prior work [citations]" rather than implying own measurement.
7. **Soften "3x" claim**: Change to "~2-3x faster" or "approximately 3x."
