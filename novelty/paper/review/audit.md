# Claim Audit: COLM 2026 Paper
## Agent Memory Below the Prompt

**Audit Date**: 2026-02-04
**Purpose**: Verify every numerical, comparative, existence, and novelty claim in the paper draft

---

## Abstract Claims

### Claim 1: "8--40 seconds of prefill per turn"
- **Type**: Numerical
- **Backup**: Benchmark data from novelty.md Section 5.1 (Cold start scaling table)
- **Calculation**:
  - 5 agents × 4,096 tokens each = 20,480 tokens total
  - At ~500 tok/s: 20,480 / 500 = 40.96 seconds
  - Lower bound (single agent, 4K): 4,096 / 500 = 8.2 seconds
- **Verdict**: VERIFIED (matches benchmark observations)

### Claim 2: "2.0--4.3× end-to-end speedup"
- **Type**: Numerical
- **Backup**: Derived from TTFT + decode times in comparative benchmarks
- **Source**: novelty.md lines 872-893 (multi-turn evaluation)
- **Specific numbers**:
  - 1K context: Cold 3.84s, Hot 2.83s = 1.36× (below claimed range - NEEDS CAVEAT)
  - 4K context: Cold 11.2s, Hot 5.1s = 2.2×
  - Upper bound from multi-turn: "2.0--4.3× improvement" (novelty.md line 889)
- **Verdict**: PARTIALLY VERIFIED (1K is 1.36×, not 2.0×; need to clarify range applies to 2K-8K contexts)

### Claim 3: "81.6× TTFT speedup (hot cache at 16K)"
- **Type**: Numerical
- **Backup**: Table in novelty.md line 927
  - Cold TTFT at 16K: 68,898ms
  - Hot TTFT at 16K: 844ms
  - Ratio: 68,898 / 844 = 81.64×
- **Verdict**: VERIFIED (exact match with 2 decimal precision)

### Claim 4: "1.1--2.0× with disk reload"
- **Type**: Numerical
- **Backup**: Warm cache numbers from TTFT table
  - 1K: 1,756ms cold / 901ms warm = 1.95×
  - 2K: 3,512ms / 1,192ms = 2.95×
  - 4K: 7,024ms / 1,680ms = 4.18×
- **Verdict**: INCORRECT - Actual warm speedups are 1.95× to 10.5× depending on context length. The "1.1--2.0×" claim is too conservative and does not match the data.

### Claim 5: "72% memory savings"
- **Type**: Numerical
- **Backup**: Calculation from Q4 formula in novelty.md lines 366-372
  - FP16: 8,388,608 bytes per layer (at 4K context)
  - Q4: 2,359,296 bytes per layer
  - Savings: (8,388,608 - 2,359,296) / 8,388,608 = 0.7186 = 71.86%
- **Verdict**: VERIFIED (rounds to 72%)

### Claim 6: "4 model architectures"
- **Type**: Existence
- **Backup**: Gemma 3, GPT-OSS Harmony, Llama 3.1, Qwen 2.5 mentioned in system
- **Evidence**:
  - Gemma 3: extensively benchmarked (Section 5)
  - DeepSeek-Coder-V2-Lite: benchmarked (Section 5.1)
  - Llama 3.1, Qwen 2.5: mentioned in model table but NO benchmark data shown
- **Verdict**: PARTIALLY VERIFIED (only 2 models actually benchmarked; 4 "supported" but not all evaluated)

---

## Introduction Claims

### Claim 7: "NVIDIA A100, ~10,000 tokens/second prefill"
- **Type**: Numerical (external hardware spec)
- **Backup**: NEEDS CITATION - Common knowledge but requires official NVIDIA source
- **Verdict**: UNVERIFIED (no citation provided, needs NVIDIA spec sheet or benchmark)

### Claim 8: "M4 Pro, ~500 tokens/second prefill"
- **Type**: Numerical (own hardware)
- **Backup**: Benchmark data from cold start scaling
  - 200 tokens in 388ms = 515 tok/s
  - 1,024 tokens in 2.0s = 512 tok/s
  - 4,096 tokens in 8.1s = 505 tok/s
- **Verdict**: VERIFIED (~500 tok/s is accurate for this hardware)

### Claim 9: "5-agent workflow, 4K tokens each, 40 seconds re-prefill"
- **Type**: Numerical (derived)
- **Backup**: 5 × 4,096 = 20,480 tokens; at 500 tok/s = 40.96s
- **Verdict**: VERIFIED

---

## Background Claims

### Claim 10: "M4 Pro: 273 GB/s memory bandwidth"
- **Type**: Hardware specification
- **Backup**: CRITICAL - Plan states benchmark hardware is M4 Pro (MX2E3LL/A), NOT M4 Max (400 GB/s)
- **Sources needed**: Apple official spec sheet for Mac Mini M4 Pro model MX2E3LL/A
- **Verdict**: NEEDS OFFICIAL CITATION (plan specifies 273 GB/s for benchmark hardware)

### Claim 11: "DGX Spark: 128 GB, 273 GB/s, $3,999"
- **Type**: Hardware specification (external)
- **Backup**: NEEDS CITATION - NVIDIA announcement (January 2026 per plan)
- **Verdict**: UNVERIFIED (need NVIDIA official announcement)

### Claim 12: Cold start prefill times (200 tokens to 32K)
- **Type**: Numerical (benchmark data)
- **Backup**: novelty.md lines 905-911 (cold start scaling table)
  - 200 tokens: 388ms ✓
  - 1,024 tokens: 2.0s ✓
  - 4,096 tokens: 8.1s ✓
  - 8,192 tokens: 16.3s ✓
  - 16,384 tokens: 31.8s ✓
  - 32,768 tokens: 48.0s (marked as "projected")
- **Verdict**: VERIFIED (32K marked as projected, not measured)

---

## System Design Claims

### Claim 13: "256-token block size"
- **Type**: Design parameter
- **Backup**: Universal block size per ADR-002 in novelty.md
- **Verdict**: VERIFIED (design decision documented)

### Claim 14: Memory savings calculation (detailed formula)
- **Type**: Mathematical derivation
- **Backup**: Formula in paper matches novelty.md Section 3.2
  - FP16: 2 × h × d × n × 2 bytes
  - Q4: 2 × h × d × n × 0.5 + 2 × h × d × (n/g) × 2 bytes
  - For h=16, d=128, n=4096, g=64:
    - FP16: 8,388,608 bytes
    - Q4: 2,359,296 bytes
- **Verdict**: VERIFIED (calculation correct)

### Claim 15: "42-layer model (Gemma 3 12B), 352 MB (FP16) vs 99 MB (Q4)"
- **Type**: Numerical (derived)
- **Backup**: 42 × 8,388,608 = 352,321,536 bytes = 335.9 MB (NOT 352 MB)
- **Correction**: Should be 336 MB (FP16) and 94 MB (Q4) for accuracy
- **Verdict**: APPROXIMATELY VERIFIED (rounding to nearest 10 MB acceptable, but less precise than claimed)

### Claim 16: "80% threshold for partial cache reuse"
- **Type**: Design parameter
- **Backup**: Character-level matching algorithm in novelty.md Section 3.3
- **Verdict**: VERIFIED (design decision documented)

---

## Evaluation Claims

### Claim 17: TTFT Table (Table 1 in paper)
- **Type**: Numerical (benchmark data)
- **Backup**: Cross-reference with novelty.md lines 917-927
- **All values match**: Cold, Warm, Hot for 1K, 2K, 4K, 8K, 16K
- **Speedup calculations**:
  - Warm 16K: 68,898 / 6,544 = 10.53× ✓
  - Hot 16K: 68,898 / 844 = 81.64× ✓
- **Verdict**: VERIFIED (all values trace to source)

### Claim 18: "Hot TTFT is roughly constant (650--870ms)"
- **Type**: Observation
- **Backup**: Hot row in Table 1: 650, 702, 758, 810, 844ms
- **Range**: 650 to 844ms (194ms spread, 30% variation)
- **Verdict**: VERIFIED (constant within O(1) expectations; slight increase due to attention over cached state)

### Claim 19: "E2E speedup at 4K: Cold = 11.2s, Warm = 5.9s (1.9×), Hot = 5.1s (2.2×)"
- **Type**: Numerical
- **Backup**: novelty.md lines 893-897 (E2E latency comparison)
  - Cold: 11.2s ✓
  - Warm: 5.9s ✓
  - Hot: 5.1s ✓
  - Speedups: 11.2/5.9 = 1.898× ≈ 1.9× ✓; 11.2/5.1 = 2.196× ≈ 2.2× ✓
- **Verdict**: VERIFIED

### Claim 20: Batched throughput (Table 2)
- **Type**: Numerical
- **Backup**: novelty.md lines 993-1019 (batched serving evaluation)
  - Sequential per-agent TPS: 33.4 ✓
  - Batched per-agent TPS: 24.7 ✓
  - System TPS: 49.4 ✓
  - Speedup: 49.4/33.4 = 1.479× ≈ 1.48× ✓
- **Verdict**: VERIFIED

### Claim 21: "System throughput increases 35%"
- **Type**: Numerical (derived)
- **Calculation**: (49.4 - 33.4) / 33.4 = 0.479 = 47.9% (NOT 35%)
- **Error identified**: Paper claims 35%, but data shows 48% increase
- **Verdict**: INCORRECT (should be "48%" or "nearly 50%")

### Claim 22: Staggered arrivals results
- **Type**: Numerical
- **Backup**: novelty.md lines 1041-1046
  - Sequential User A: 7.0s ✓
  - Sequential User B: 24.5s ✓
  - Batched User A: 7.3s ✓
  - Batched User B: 9.6s ✓
  - User B speedup: 24.5 / 9.6 = 2.552× ≈ 2.6× ✓
  - User A penalty: 7.3 / 7.0 = 1.043× ≈ 4% ✓
- **Verdict**: VERIFIED

### Claim 23: "Net total TTFT: 50.4s (batched) vs 90.8s (sequential)"
- **Type**: Numerical (derived)
- **Calculation Check**:
  - Sequential: 7.0 + 24.5 = 31.5s (NOT 90.8s)
  - Batched: 7.3 + 9.6 = 16.9s (NOT 50.4s)
- **Error identified**: Paper values do not match stated User A/B times
- **Possible source**: Including decode time? Needs clarification
- **Verdict**: INCORRECT or UNCLEAR (numbers don't add up)

---

## Discussion Claims

### Claim 24: "Only 2 models actually benchmarked"
- **Type**: Observation (from audit findings)
- **Evidence**: Evaluation section shows Gemma 3 and DeepSeek-Coder-V2-Lite only
- **Verdict**: VERIFIED (Llama 3.1 and Qwen 2.5 mentioned as "supported" but not evaluated)

### Claim 25: "M5: 3.3--4.1× TTFT improvement"
- **Type**: External claim (future hardware)
- **Backup**: NEEDS CITATION - Plan notes "Apple ML Research, Nov 2025" but no paper title or URL
- **Verdict**: UNVERIFIED (need source or mark as "preliminary benchmarks" with caveat)

### Claim 26: "macOS Tahoe 26.2 RDMA over Thunderbolt 5"
- **Type**: Future feature claim
- **Backup**: NEEDS CITATION - Apple WWDC or official documentation
- **Verdict**: UNVERIFIED (need official source)

### Claim 27: "Thunderbolt 5 <50μs latency"
- **Type**: Hardware specification
- **Backup**: NEEDS CITATION - Intel/Apple Thunderbolt 5 specs
- **Verdict**: UNVERIFIED (need official spec sheet)

---

## Summary

### Verification Statistics
- **Total claims audited**: 27
- **Fully verified**: 15 (56%)
- **Partially verified**: 3 (11%)
- **Incorrect/needs correction**: 4 (15%)
- **Unverified (needs citation)**: 5 (18%)

### Critical Issues

**HIGH PRIORITY:**
1. **Claim 4 (Abstract)**: "1.1--2.0× with disk reload" - Data shows 1.95--10.5×, fix abstract
2. **Claim 21**: "35% throughput increase" - Should be 48% per data
3. **Claim 23**: Net total TTFT numbers don't match user-level numbers - needs clarification
4. **Claim 6**: "4 model architectures" - Only 2 benchmarked; either clarify "supported" vs "evaluated" or remove claim

**MEDIUM PRIORITY:**
5. **Claim 10**: M4 Pro 273 GB/s - Need Apple official spec citation
6. **Claim 11**: DGX Spark specs - Need NVIDIA announcement citation
7. **Claim 15**: Memory calculation slightly off (335.9 MB vs 352 MB claimed)

**LOW PRIORITY:**
8. **Claim 7**: A100 10K tok/s - Common knowledge but needs citation
9. **Claim 25**: M5 TTFT improvement - Need source or caveat
10. **Claims 26-27**: Future features - Need official sources or remove

### Recommended Actions

1. **Fix abstract**: Change "1.1--2.0×" to "1.95--10.5×" (warm speedup range)
2. **Fix throughput claim**: Change "35%" to "48%"
3. **Clarify total TTFT**: Either fix calculation or explain what's included
4. **Add model architecture caveat**: "4 architectures supported (2 benchmarked: Gemma 3, DeepSeek-Coder-V2-Lite)"
5. **Add hardware spec citations**: Apple spec sheet for M4 Pro, NVIDIA announcement for DGX Spark
6. **Resolve M5/Tahoe claims**: Either find sources or remove/mark as preliminary
