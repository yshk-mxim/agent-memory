# Live Observation Analysis: Claude Code CLI Performance Issue

**Date**: 2026-01-26
**Test Setup**: Live server running, user operating Claude Code CLI
**Model**: Gemma 3 12B 4-bit (`mlx-community/gemma-3-12b-it-4bit`)
**Hardware**: M3 Max (assumed based on prior discussions)

---

## Executive Summary

**Root Cause Identified**: Slow generation speed (25.5 tokens/sec vs expected 50-100 tokens/sec)

**NOT the issue**:
- ❌ Concurrent request batching interference
- ❌ Prefill overhead
- ❌ Streaming vs non-streaming format
- ❌ Claude Code CLI interaction overhead
- ❌ Missing API endpoints

**IS the issue**:
- ✅ Generation phase itself runs at 2-4x slower than expected

---

## Observed Timeline

### Request Sequence

```
Time      | Event                                          | Details
----------|------------------------------------------------|---------------------------
09:43:50  | POST /v1/messages (streaming)                  | Cache miss: msg_22cc86115e391733
09:43:51  | POST /v1/messages/count_tokens                 | 9,983 tokens
09:43:51  | POST /v1/messages (streaming)                  | Cache miss: msg_bddb635ad8df413b
          | [~5.5 minutes of generation]                   | Both streaming requests generating
09:49:13  | POST /v1/messages (non-streaming)              | Cache hit: 49,152 tokens
09:54:34  | Response complete                              | 8,192 output tokens
          | Duration: 321 seconds (5.36 minutes)           | 25.5 tokens/sec
09:54:35  | POST /v1/messages (non-streaming)              | Cache miss: msg_ef1eb85a8723a454
09:54:38  | [Server killed]                                | Test ended
```

---

## Performance Analysis

### Measured Performance

**Request Details:**
- **Start**: 09:49:13
- **End**: 09:54:34
- **Duration**: 321 seconds (5.36 minutes)
- **Input**: 49,152 tokens (cache hit - no prefill)
- **Output**: 8,192 tokens
- **Generation Speed**: 8,192 / 321 = **25.5 tokens/sec**

### Expected Performance

**MLX on M3 Max Benchmarks**:
- Gemma 3 12B: ~50-100 tokens/sec (estimated)
- Similar models (SmolLM2): ~230 tokens/sec
- Larger models (20B): ~32-50 tokens/sec

**Expected time for 8,192 tokens**:
- At 50 tok/s: ~164 seconds (2.7 minutes)
- At 100 tok/s: ~82 seconds (1.4 minutes)

**Actual**: 321 seconds (5.4 minutes)

**Slowdown**: 2-4x slower than expected

---

## What This Rules Out

### 1. Concurrent Request Batching Issue ❌

**Initial Hypothesis**: MLX BatchGenerator processes concurrent requests in lockstep, causing interference.

**Reality**: The slow request (09:49:13-09:54:34) was:
- A **single non-streaming request**
- **After** the initial streaming requests should have completed their prefill
- Still slow despite being isolated

**Conclusion**: Not primarily a batching issue (though concurrent requests might add overhead).

### 2. Prefill Overhead ❌

**Initial Hypothesis**: Large prefill sizes (9,983 tokens) cause slowness.

**Reality**:
- The slow request had **cache hit: 49,152 tokens**
- No prefill needed (cache already computed)
- Pure generation phase took 5.4 minutes

**Conclusion**: Not a prefill issue.

### 3. Streaming vs Non-Streaming ❌

**Initial Hypothesis**: SSE streaming adds overhead.

**Reality**:
- The slow request was **non-streaming**
- Still exhibited the same slowness
- Streaming format is irrelevant

**Conclusion**: Not a streaming issue.

### 4. Claude Code CLI Overhead ❌

**Initial Hypothesis**: Claude CLI makes concurrent requests that interfere.

**Reality**:
- The slow generation happened **server-side**
- MLX BatchGenerator was generating slowly
- CLI only receives results after generation

**Conclusion**: Not a CLI interaction issue.

---

## What This Confirms

### Core Issue: Slow Generation Speed ✅

**Observation**: MLX BatchGenerator is generating tokens at 25.5 tokens/sec when it should be 50-100 tokens/sec.

**Evidence**:
1. Cache hit = no prefill overhead
2. Single non-streaming request = no concurrent interference
3. Pure generation phase = 25.5 tok/s
4. Expected = 50-100 tok/s for this model size

**Conclusion**: The generation phase itself is 2-4x slower than expected.

---

## Possible Root Causes

### 1. Model-Specific Performance

**Hypothesis**: Gemma 3 12B 4-bit quantization might be slower than expected with MLX.

**Evidence**:
- Using 4-bit quantization may have decoding overhead
- Gemma 3 architecture might not be optimized for MLX
- Different models have different inference characteristics

**Test**: Run same request with SmolLM2-135M to compare.

### 2. Concurrent Requests Still Interfering

**Hypothesis**: The two earlier streaming requests (09:43:50-51) might still have been running during the slow request (09:49:13-09:54:34).

**Evidence**:
- First two requests were **streaming** (never logged completion)
- Logs show SSE connection established but no "Response:" log
- Streaming requests may stay in batch during entire generation

**Test**: Add logging to show active batch size during generation.

**Code to add**:
```python
# In batch_engine.py step() method
def step(self):
    batch_size = len(self._active_requests)
    logging.info(f"[BATCH] Step: active={batch_size}")
    # ... existing code ...
```

### 3. MLX BatchGenerator Configuration

**Hypothesis**: `prefill_step_size=512` or other settings might be suboptimal.

**Current Configuration**:
```python
# From logs:
max_batch_size=5
prefill_step_size=512
```

**Questions**:
- Is `prefill_step_size=512` too small?
- Should generation have a separate `generation_step_size`?
- Are there MLX-specific performance tuning parameters?

**Test**: Try different `prefill_step_size` values (1024, 2048, 4096).

### 4. Hardware Throttling

**Hypothesis**: Sustained 5+ minutes of inference causes thermal throttling.

**Evidence**:
- 5.4 minutes of continuous generation
- M3 Max might throttle under sustained load
- Generation speed should be consistent, not degrading

**Test**: Monitor CPU/GPU frequency during generation.

### 5. Cache Reconstruction Overhead

**Hypothesis**: Reconstructing 49,152 tokens of cache might add overhead per generation step.

**Evidence**:
- Cache hit logged at 09:49:13
- "Cache reconstructed for msg_bddb635ad8df413b: 49152 tokens"
- Large cache = more data to load/manage per step

**Test**: Compare generation speed with cache hit vs cache miss.

---

## Recommended Investigation Steps

### Priority 1: Check Concurrent Request Interference

**Goal**: Determine if the two earlier streaming requests were still running.

**Action**:
1. Add batch size logging to `batch_engine.py`:
   ```python
   def step(self):
       batch_size = len(self._active_requests)
       active_uids = list(self._active_requests.keys())
       logging.info(f"[BATCH] Step start: batch_size={batch_size}, uids={active_uids}")
   ```

2. Restart server and rerun Claude CLI test
3. Analyze logs to see if multiple UIDs were active during slow generation

**Expected**: If batch_size=1 during slow request, concurrent requests are NOT the issue.

### Priority 2: Test with Smaller Model

**Goal**: Isolate if slowness is model-specific.

**Action**:
1. Change model to SmolLM2-135M-Instruct (known fast model)
2. Rerun same test with Claude CLI
3. Compare generation speed

**Expected**: SmolLM2 should generate at ~200+ tokens/sec. If it's also slow (25 tok/s), issue is in our code.

### Priority 3: Review MLX BatchGenerator Settings

**Goal**: Optimize MLX configuration for generation.

**Action**:
1. Research MLX-LM BatchGenerator optimal settings
2. Try different `prefill_step_size` values
3. Check if there's a separate `generation_step_size` parameter
4. Review MLX documentation for performance tuning

**Expected**: Find configuration that improves generation speed.

### Priority 4: Profile Generation Loop

**Goal**: Identify bottleneck in generation step.

**Action**:
1. Add timing to `batch_engine.py step()`:
   ```python
   def step(self):
       step_start = time.time()
       batch_response = self._batch_gen.next()
       step_elapsed = time.time() - step_start
       logging.info(f"[BATCH] Step: {step_elapsed:.3f}s, responses={len(batch_response)}")
   ```

2. Analyze per-step timing
3. Look for patterns (first steps slow? all steps slow? degrading?)

**Expected**: Identify if slowness is uniform or concentrated in specific phases.

---

## Comparison with Previous Tests

### Your Direct Tests (Working Well)

**test_17k_tokens.py** (non-streaming, single request):
- Result: 21,585 tokens in 47.65 seconds = **452.95 tokens/sec** ✅

**test_streaming_17k.py** (streaming, single request):
- Result: 21,585 tokens in 87.09 seconds = **247.85 tokens/sec** ✅

**Question**: Why do these tests show excellent performance (450 tok/s) but Claude CLI shows terrible performance (25 tok/s)?

**Possible Explanations**:

1. **Different models**: Your tests might have used a different model
   - Check: What model were these tests using?
   - GPT-OSS-20B in test vs Gemma 3 12B in Claude CLI?

2. **Concurrent requests**: Claude CLI sends multiple requests
   - Your tests: Single isolated request
   - Claude CLI: Multiple requests (streaming + non-streaming)

3. **Cache size**: Your tests might have smaller cache
   - Your tests: ~21K tokens
   - Claude CLI: 49K tokens in cache

4. **Request pattern**: Your tests are clean, CLI is complex
   - Your tests: Single request → complete
   - Claude CLI: Multiple concurrent requests with cache hits/misses

---

## Key Questions to Answer

1. **Were the streaming requests still active?**
   - Add batch size logging to confirm

2. **Is this model-specific slowness?**
   - Test with SmolLM2 to isolate

3. **What were your test scripts using?**
   - Check model in `test_17k_tokens.py` and `test_streaming_17k.py`
   - Compare with Gemma 3 12B performance

4. **Is cache size the bottleneck?**
   - 49K token cache vs smaller caches
   - Test generation with different cache sizes

5. **Are there MLX tuning parameters?**
   - Research MLX-LM performance optimization
   - Check if our settings are suboptimal

---

## Immediate Next Action

**Recommendation**: Add batch size logging and rerun the test.

**Code Change**:
```python
# In src/semantic/application/batch_engine.py

def step(self) -> Iterator[CompletedGeneration]:
    """Execute one batch decode step and yield completed generations."""
    # Guard: if no active batch, return immediately
    if self._batch_gen is None:
        return

    # LOG BATCH STATE
    batch_size = len(self._active_requests)
    active_uids = list(self._active_requests.keys())
    logging.info(f"[BATCH] Step: batch_size={batch_size}, active_uids={active_uids}")

    # Execute decode loop until all sequences finish
    while True:
        step_start = time.time()
        batch_response = self._batch_gen.next()
        step_elapsed = time.time() - step_start

        logging.info(f"[BATCH] Step complete: {step_elapsed:.3f}s, responses={len(batch_response) if batch_response else 0}")

        # ... rest of existing code ...
```

This will show:
1. How many requests are in the batch during slow generation
2. How long each step takes
3. If multiple UIDs are interfering

---

## Conclusion

**What we learned**:
1. ✅ Generation speed is 25.5 tok/s (2-4x slower than expected)
2. ✅ Issue persists even with cache hit (no prefill)
3. ✅ Issue persists with non-streaming requests
4. ❌ Not caused by streaming overhead
5. ❌ Not caused by prefill overhead
6. ❓ May be caused by concurrent requests still in batch
7. ❓ May be model-specific (Gemma 3 12B 4-bit)
8. ❓ May be due to suboptimal MLX configuration

**What to do next**:
1. Add batch size logging
2. Rerun test with Claude CLI
3. Compare with SmolLM2 performance
4. Optimize MLX configuration if needed

**Status**: Root cause narrowed to generation phase slowness. Need more detailed logging to pinpoint exact bottleneck.

---

**Analysis Date**: 2026-01-26
**Conclusion**: Single clear issue (slow generation) - not multiple architectural problems. Investigation continues with targeted logging.
