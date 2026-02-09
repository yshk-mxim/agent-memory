# Benchmark Crash & Issue Report (2026-02-08)

## Run Summary

| Model | Total | OK | WARN | Staggered | Status |
|-------|-------|----|------|-----------|--------|
| Gemma 3 12B Q4 | 198/198 | 183 | 15 | 6/6 | Complete |
| DeepSeek-Coder-V2-Lite Q4 | 115/204 | 114 | 1 | 0/6 | Killed at 56% |

Result files:
- `benchmarks/results/colm_full_gemma_20260208_152120.json`
- `benchmarks/results/colm_full_deepseek_20260208_152120.json`

---

## BUG 1: Gemma batch2 + cached + short context = empty output

**Severity**: High — 15/198 measurements fail, 100% reproducible

**Pattern**: Concurrent (batch=2) requests with warm or hot cache at 1024-2048 tokens
produce empty output (single space character, 0 TPS, 0 TTFT).

| Config | Passes Failed |
|--------|--------------|
| b2/warm/streaming/1024tok | 3/3 |
| b2/warm/non-streaming/1024tok | 3/3 |
| b2/hot/streaming/1024tok | 3/3 |
| b2/hot/non-streaming/1024tok | 3/3 |
| b2/hot/non-streaming/2048tok | 3/3 |

**What works fine**:
- b2/cold/1024tok: 6/6 OK (sysTPS ~10-12) — cold cache is fine
- b2/warm/2048tok: 6/6 OK (sysTPS ~20-22) — warm + longer context is fine
- b2/hot/streaming/2048tok: 3/3 OK — hot + streaming + 2K is fine
- b2/*/4K+: all OK — 4K and above always work

**Hypothesis**: When both concurrent requests have cache hits at short sequences,
the cache restore + batch assembly race condition causes the engine to produce
degenerate output. Possibly the KV cache is being restored for both sequences
simultaneously and they're interfering with each other's block allocations.

**To investigate**:
1. Add logging in `batch_engine.py` around `_restore_cache()` for batch>1
2. Check if `_extract_cache()` returns valid data when two agents restore simultaneously
3. Check if block pool has enough blocks for two restored 1K caches + decode
4. Test with `--batch-size 2` and manual curl of two 1K warm requests

---

## BUG 2: TPS probe cooldown too tight (5% tolerance)

**Severity**: High — turned a 4-hour DeepSeek benchmark into 16+ hours

**Problem**: `THROTTLE_TPS_TOLERANCE = 0.05` in `colm_full_benchmark.py`.
Baseline TPS is calibrated during warmup when GPU is fresh. Under sustained
inference, TPS permanently drops to ~84-89% of baseline (thermal throttling,
memory pressure, etc.). The 5% tolerance means the probe NEVER passes, and
every measurement hits MAX_COOLDOWN_SECONDS (242s).

**Evidence**:
- Gemma: baseline ~8.6 TPS, sustained ~7.9-8.2 (92-95%). Median cooldown 19s,
  but 36% hit max (72/198). Mixed results.
- DeepSeek: baseline ~26.3 TPS, sustained ~21-23 (82-89%). **100% of 114
  measurements hit max cooldown** (241-243s each). Catastrophic.

**Fix**: Change `THROTTLE_TPS_TOLERANCE` from 0.05 to 0.20. Or better:
calibrate baseline AFTER a warmup burst (not just initial probe), so the
baseline reflects sustained rather than peak performance.

---

## BUG 3: Staggered benchmark metric is apples-to-oranges + no interleaving benefit

**Severity**: High — benchmark design issue + possible server issue

**Results** (Gemma, 4K context, 3 passes averaged):

| Metric | Sequential | Batched |
|--------|-----------|---------|
| User A TTFT | 16,877ms | 16,888ms |
| User B TTFT (from own request) | 16,477ms | 34,067ms |
| Wall time | 38,982ms | 38,884ms |
| System TPS | 3.3 | 3.3 |

### Issue A: Measurement is apples-to-oranges

TTFT is measured per-user from `time.perf_counter()` at HTTP POST to first
SSE content delta (`openai_benchmark.py:244,300`).

- **Sequential** (`run_sequential`, line 126-136): User B's request is sent
  AFTER User A fully completes. User B gets the server all to itself.
  User B TTFT = pure uncontended 4K prefill = ~16.5s.
- **Batched** (`run_batched`, line 186-199): User B's request is sent 2s
  after User A. User B TTFT = time from User B's POST to first token,
  including waiting while server interleaves both users = ~34s.

Sequential User B TTFT is NOT "what User B would experience without batching."
It's User B in total isolation (User A already gone). The fair comparison
for the paper is **User B's wall-clock wait from arrival**, which in the
sequential case would be: User A remaining time + User B prefill.

**Fix for benchmark**: Add `user_b_ttft_from_wall_start` = User B's first
token timestamp minus `t_start_wall` (when User A was sent). Then:
- Sequential: User B wait from wall start = A_e2e + B_ttft ≈ 16.9s + 16.5s ≈ 33.4s
- Batched: User B wait from wall start = B_delay + B_ttft = 2s + 34s ≈ 36s

### Issue B: No interleaving benefit visible

Even with the metric issue, wall time is identical (~38.9s both modes) and
system TPS is identical (3.3). Batched mode should show:
- Lower wall time (interleaved prefill + shared decode)
- Higher system TPS

This suggests **interleaved chunked prefill is not actually helping** at 4K.

**Possible causes**:
1. 4K is just barely above the 2048 chunking threshold (~16 chunks of 256).
   Maybe chunk overhead dominates at this size.
2. Scheduler may be serializing prefills rather than truly interleaving
   (check if User A's decode pauses while User B's chunks run).
3. System TPS of 3.3 is suspiciously low (batch1 cold/4K is ~7 TPS).
   Something may be throttling concurrent decode.

**To investigate**:
1. Fix the metric (add wall-start-relative TTFT for User B)
2. Add server-side logging to confirm chunked prefill interleaving
3. Test with 16K context where interleaving benefit should be larger
4. Compare batch=2 system TPS from Table 2 data vs staggered system TPS

---

## BUG 4: DeepSeek benchmark incomplete (89 measurements + 6 staggered remaining)

**Severity**: Medium — need to resume

**What's done**: All batch1 (108/108) + 7 batch2 cold/streaming measurements.

**What's missing**:
- b2/cold/streaming: 4K pass 2-3, 8K×3, 16K×3 (8 remaining)
- b2/cold/non-streaming: all 15
- b2/warm/streaming: all 15
- b2/warm/non-streaming: all 15
- b2/hot/streaming: all 15
- b2/hot/non-streaming: all 15
- Staggered: all 6

**Resume**: `python benchmarks/colm_full_benchmark.py --port 8399 --resume benchmarks/results/colm_full_deepseek_20260208_152120.json`
(but fix BUG 2 first, or it'll take another 16+ hours)

---

## ISSUE 5: DeepSeek early EOS (soft warning, not a crash)

**Severity**: Low — cosmetic

**What**: b1/hot/streaming/2048tok/pass1 produced 45 of 64 tokens then hit EOS.
Output is perfectly coherent: "I'm here to help you understand the topic
you've asked about, but you didn't provide a specific topic..."

**Action**: No fix needed. The model legitimately finished its response. Could
adjust quality check to not flag early_eos when output is >32 tokens and
structurally sound.

---

## Priority Order

1. **BUG 2** (cooldown) — Fix first, blocks DeepSeek completion
2. **BUG 1** (batch2/cached/short) — Root cause investigation
3. **BUG 3** (staggered comparison) — May be a benchmark design issue, not server bug
4. **BUG 4** (resume DeepSeek) — After fixing BUG 2
5. **ISSUE 5** (early EOS) — Optional cleanup
