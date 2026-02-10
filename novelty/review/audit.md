# Claim Audit -- COLM 2026 Paper

**Date**: 2026-02-09
**Source data**: `benchmarks/results/colm_full_gemma_merged.json`, `benchmarks/results/colm_full_deepseek_merged.json`
**Method**: Median of 3 passes per configuration. Rounding convention: standard (round half up).
**Git SHA**: ee24513

---

## Table 1: TTFT (streaming, batch=1)

### Gemma 3 12B Q4

| Context | Cache | Claimed (ms) | Actual Median (ms) | Match? |
|---------|-------|-------------:|--------------------:|--------|
| 1K  | Cold | 4,007   | 4,006.6  | YES (rounds to 4007) |
| 2K  | Cold | 7,363   | 7,363.1  | YES |
| 4K  | Cold | 15,502  | 15,502.1 | YES |
| 8K  | Cold | 32,944  | 32,943.9 | YES |
| 16K | Cold | 71,132  | 71,131.8 | YES |
| 32K | Cold | 165,189 | 165,188.9| YES |
| 1K  | Warm | 527     | 526.7    | YES |
| 2K  | Warm | 532     | 532.3    | YES |
| 4K  | Warm | 513     | 512.5    | YES -- standard round-half-up gives 513; Python banker's rounding gives 512 |
| 8K  | Warm | 590     | 590.4    | YES |
| 16K | Warm | 808     | 807.6    | YES |
| 32K | Warm | 1,621   | 1,620.6  | YES |
| 1K  | Hot  | 668     | 667.6    | YES |
| 2K  | Hot  | 688     | 688.1    | YES |
| 4K  | Hot  | 709     | 708.8    | YES |
| 8K  | Hot  | 762     | 761.9    | YES |
| 16K | Hot  | 874     | 873.7    | YES |
| 32K | Hot  | 1,276   | 1,275.8  | YES |

### DeepSeek-Coder-V2-Lite Q4

| Context | Cache | Claimed (ms) | Actual Median (ms) | Match? |
|---------|-------|-------------:|--------------------:|--------|
| 1K  | Cold | 1,090   | 1,089.8  | YES |
| 2K  | Cold | 1,884   | 1,883.8  | YES |
| 4K  | Cold | 3,949   | 3,948.7  | YES |
| 8K  | Cold | 8,541   | 8,540.8  | YES |
| 16K | Cold | 19,193  | 19,192.5 | YES -- standard round-half-up gives 19193; Python banker's rounding gives 19192 |
| 32K | Cold | 48,258  | 48,258.4 | YES |
| 1K  | Warm | 217     | 217.1    | YES |
| 2K  | Warm | 285     | 285.3    | YES |
| 4K  | Warm | 252     | 251.8    | YES |
| 8K  | Warm | 307     | 306.5    | YES -- standard round-half-up gives 307; Python banker's rounding gives 306 |
| 16K | Warm | 430     | 429.6    | YES |
| 32K | Warm | 697     | 697.0    | YES |
| 1K  | Hot  | 356     | 356.3    | YES |
| 2K  | Hot  | 376     | 376.1    | YES |
| 4K  | Hot  | 372     | 372.4    | YES |
| 8K  | Hot  | 412     | 411.5    | YES |
| 16K | Hot  | 484     | 483.8    | YES |
| 32K | Hot  | 652     | 651.7    | YES |

**Table 1 verdict**: All 36 values match the source data. Three values (Gemma 4K warm, DeepSeek 16K cold, DeepSeek 8K warm) land on x.5 boundaries where standard round-half-up and Python banker's rounding diverge by 1 ms. The paper uses standard rounding, which is consistent throughout.

---

## Table 2: System TPS (non-streaming, batch=2)

### Gemma 3 12B Q4

| Context | Cache | Claimed (TPS) | Actual Median (TPS) | Match? |
|---------|-------|-------------:|--------------------:|--------|
| 1K  | Cold | 10.2 | 10.2 | YES |
| 1K  | Warm | 22.4 | 22.4 | YES |
| 1K  | Hot  | 22.0 | 22.0 | YES |
| 4K  | Cold | 3.3  | 3.3  | YES |
| 4K  | Warm | 19.8 | 19.8 | YES |
| 4K  | Hot  | 20.0 | 20.0 | YES |
| 16K | Cold | 0.8  | 0.8  | YES |
| 16K | Warm | 13.3 | 13.3 | YES |
| 16K | Hot  | 13.6 | 13.6 | YES |

### DeepSeek-Coder-V2-Lite Q4

| Context | Cache | Claimed (TPS) | Actual Median (TPS) | Match? |
|---------|-------|-------------:|--------------------:|--------|
| 1K  | Cold | 43.6 | 43.6 | YES |
| 1K  | Warm | 64.8 | 64.8 | YES |
| 1K  | Hot  | 65.2 | 65.2 | YES |
| 4K  | Cold | 13.8 | 13.8 | YES |
| 4K  | Warm | 55.1 | 55.1 | YES |
| 4K  | Hot  | 55.8 | 55.8 | YES |
| 16K | Cold | 3.2  | 3.2  | YES |
| 16K | Warm | 28.2 | 28.2 | YES |
| 16K | Hot  | 35.9 | 35.9 | YES |

**Table 2 verdict**: All 18 values match exactly to one decimal place.

---

## Staggered Benchmark (4K cold, batch=2)

| Metric | Claimed | Actual Median | Match? |
|--------|--------:|--------------:|--------|
| Gemma seq wall time  | 38.8 s | 38.8 s (values: 38.5, 39.6, 38.8) | YES |
| Gemma bat wall time  | 38.6 s | 38.6 s (values: 38.3, 39.7, 38.6) | YES |
| DeepSeek seq wall time | 9.5 s | 9.5 s (values: 9.4, 9.6, 9.5) | YES |
| DeepSeek bat wall time | 9.4 s | 9.4 s (values: 9.4, 9.6, 9.3) | YES |

**Staggered verdict**: All 4 values match exactly.

---

## Derived Claims (Abstract, Introduction, Body)

| # | Claim | Location | Calculation | Actual Value | Match? |
|---|-------|----------|-------------|-------------:|--------|
| 1 | 130x TTFT speedup at 32K hot (Gemma) | Abstract | 165,188.9 / 1,275.8 | 129.5x | CLOSE -- rounds to 129x or 130x depending on convention. Paper uses 130x (round half up on 129.5). |
| 2 | 74x TTFT speedup at 32K hot (DeepSeek) | Abstract | 48,258.4 / 651.7 | 74.1x | YES -- rounds to 74x |
| 3 | 88x TTFT speedup at 16K warm (Gemma) | Abstract | 71,131.8 / 807.6 | 88.1x | YES -- rounds to 88x |
| 4 | 45x TTFT speedup at 16K warm (DeepSeek) | Abstract | 19,192.5 / 429.6 | 44.7x | YES -- rounds to 45x |
| 5 | 22.4 system TPS at 1K warm batch=2 (Gemma) | Abstract | Direct from Table 2 | 22.4 | YES -- exact match |
| 6 | 64.8 system TPS at 1K warm batch=2 (DeepSeek) | Abstract | Direct from Table 2 | 64.8 | YES -- exact match |
| 7 | 15.5s TTFT at 4K Gemma cold | Intro | 15,502.1 ms | 15.5 s | YES |
| 8 | 513 ms TTFT at 4K Gemma warm | Intro | 512.5 ms (round-half-up) | 513 ms | YES -- see Table 1 note |
| 9 | 709 ms TTFT at 4K Gemma hot | Intro | 708.8 ms | 709 ms | YES |
| 10 | "77 seconds of dead time" (5 agents x 15.5s) | Intro | 5 x 15,502.1 ms = 77,510 ms | 77.5 s | CLOSE -- paper truncates to "77 seconds". Exact value is 77.5s. Off by 0.5s (0.6%). |
| 11 | 72% memory savings Q4 vs FP16 | Body | group=64: 1 - (64x4+16+16)/(64x16) = 71.9%; group=128: 1 - (128x4+16+16)/(128x16) = 73.4% | 71.9--73.4% | YES -- 72% is within the range depending on group size. Gemma uses group=64 (71.9%), which rounds to 72%. |
| 12 | Gemma warm 102x at 32K | Body | 165,188.9 / 1,620.6 | 101.9x | YES -- rounds to 102x |
| 13 | DeepSeek warm 69x at 32K | Body | 48,258.4 / 697.0 | 69.2x | YES -- rounds to 69x |
| 14 | DeepSeek 3.4x faster per-token cold prefill | Body | Gemma/DeepSeek cold TTFT ratio | 3.4x at 32K, 3.7x at 16K, 3.9x at 4K | SELECTIVE -- the 3.4x figure is the ratio at 32K only. At other context lengths the ratio ranges from 3.7x to 3.9x. The claim picks the most conservative (lowest) point. |
| 15 | DeepSeek 3x faster batched throughput | Body | 64.8 / 22.4 (1K warm sysTPS) | 2.89x | CLOSE -- rounds to 2.9x, paper says 3x. This is a soft approximation (within 4% of stated value). |

---

## Summary of Discrepancies

All 58 table-level data points (36 in Table 1 + 18 in Table 2 + 4 staggered) match the source JSON files exactly or within standard rounding.

Of the 15 derived claims, 12 are exact matches. Three require notes:

1. **Claim 1 (130x)**: The actual ratio is 129.5x. Reporting as "130x" requires rounding 129.5 up, which is defensible under standard round-half-up convention. However, Python `round(129.5) = 130` also agrees here (banker's rounding rounds to even, which is 130). **Verdict: acceptable.**

2. **Claim 10 (77 seconds)**: The exact calculation yields 77.5 seconds. The paper says "77 seconds" (truncated, not rounded). This understates the dead time by 0.5 seconds (0.6%). **Verdict: minor, does not change the argument.** Consider saying "nearly 78 seconds" or "over 77 seconds" for precision.

3. **Claim 15 (3x batched throughput)**: The actual ratio is 2.89x. Rounding to one significant figure gives 3x, which is conventional for approximate claims in running text. **Verdict: acceptable as an approximation, but "2.9x" would be more precise.**

4. **Claim 14 (3.4x per-token prefill)**: This is accurate at 32K context but the ratio varies from 3.4x to 3.9x across context lengths. The claim cherry-picks the most conservative (minimum) value, which is defensible but could be misleading without noting the range. **Verdict: accurate for the stated context length; adding "at 32K" or "3.4--3.9x across context lengths" would be more transparent.**

---

## Rounding Convention Note

Three Table 1 values (Gemma 4K warm = 512.5, DeepSeek 16K cold = 19192.5, DeepSeek 8K warm = 306.5) fall exactly on .5 boundaries. The paper consistently rounds these up (513, 19193, 307), following the standard round-half-up convention rather than Python's default banker's rounding (round-half-to-even). This is internally consistent and should be documented in the benchmark methodology.

---

## Audit Conclusion

The paper's quantitative claims are well-supported by the source data. No fabricated or significantly misrepresented numbers were found. The three "close" derived claims (130x, 77s, 3x) are all within standard rounding or soft-approximation norms for academic papers. The only actionable recommendation is to qualify the "3.4x per-token" claim with its context-length dependency.
