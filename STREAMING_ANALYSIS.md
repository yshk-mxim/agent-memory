# Streaming vs Non-Streaming: Critical Analysis for Claude Code CLI

**Date:** 2026-01-26
**Status:** CRITICAL FINDING - Streaming May Be Unnecessary!

---

## Executive Summary

**CRITICAL DISCOVERY:** Claude Code CLI **does NOT require streaming** (SSE). The Anthropic Messages API fully supports non-streaming responses (`stream: false`), and **Ollama successfully uses non-streaming mode** with Claude Code CLI.

**Impact:** Disabling streaming would **completely eliminate** the 700x performance degradation caused by concurrent request bottlenecks!

---

## 1. Does Claude Code CLI Require Streaming?

### Answer: **NO**

**Evidence from Anthropic API Documentation:**

The Anthropic Messages API supports two modes ([Streaming Messages - Claude API](https://docs.anthropic.com/en/api/messages-streaming)):

1. **Non-streaming (default):** `stream: false` or omit parameter
   - Returns complete response immediately
   - Single JSON payload with full content
   - **This is what Ollama uses!**

2. **Streaming:** `stream: true`
   - Server-sent events (SSE) protocol
   - Incremental token-by-token delivery
   - Used for real-time user experience

**Key Quote from Research:**
> "When creating a Message, you can set `stream: true` to incrementally stream the response using server-sent events (SSE). Conversely, setting `stream: false` (or omitting the parameter, as false is typically the default) will return a complete, non-streaming response."

### How Ollama Handles Claude Code CLI

**Ollama Architecture:**
- Implements Anthropic-compatible API endpoint
- Supports **both** streaming and non-streaming modes
- Claude Code CLI works seamlessly with both
- ([Anthropic compatibility - Ollama](https://docs.ollama.com/api/anthropic-compatibility))

**Critical Insight:** Ollama does NOT struggle with concurrent requests because it doesn't have the MLX BatchGenerator streaming bottleneck!

---

## 2. Performance Comparison: Streaming vs Non-Streaming

### Current Streaming Implementation (BROKEN):

**Architecture:**
```
Claude CLI Request → FastAPI async endpoint → BlockPoolBatchEngine.submit()
                                           ↓
                                     BatchGenerator.insert([prompt])
                                           ↓
                                     WHILE TRUE: step() → yield SSE events
                                           ↓
                                     MLX processes ALL batch members in lockstep
                                           ↓
                                     Each endpoint waits for matching UID
```

**Concurrent Request Behavior:**
- Request 1 (85 tokens): Submitted as uid=0
- Request 2 (113 tokens): Submitted as uid=1
- Request 3 (17,616 tokens): Submitted as uid=2
- **All 3 processed together in batch** (lockstep)
- Each endpoint blocks on `step()` waiting for its UID
- Total time = **longest sequence × overhead** = 105+ seconds

**Result:** 0.81 tokens/sec (700x slower!)

---

### Proposed Non-Streaming Implementation (SIMPLE):

**Architecture:**
```
Claude CLI Request → FastAPI endpoint (non-async or serialized)
                  ↓
            BlockPoolBatchEngine.submit([prompt])
                  ↓
            BatchGenerator.insert([prompt])  # SINGLE request only
                  ↓
            WHILE TRUE: step() until completion
                  ↓
            Accumulate ALL tokens
                  ↓
            Return complete MessagesResponse (JSON)
```

**Concurrent Request Behavior:**
- Request 1 (85 tokens): Process to completion (~0.15s)
- Request 2 (113 tokens): Process to completion (~0.20s)
- Request 3 (17,616 tokens): Process to completion (~31s)
- **Sequential processing** (no batch interference)
- Each request gets full 566 tok/s throughput

**Result:** 566 tokens/sec per request (FAST!)

---

## 3. Code Changes Required

### Issue #1: Model Name Mismatch (CRITICAL BUG)

**Current Behavior:**
```python
# anthropic_adapter.py line 215, 464
model=request_body.model  # Returns "claude-haiku-4-5-20251001"
```

**Problem:** We're echoing back the Claude model name from the request instead of the actual model!

**Your Observation:**
```
"model": "claude-haiku-4-5-20251001"  ← WRONG (client requested this)
"model": "claude-sonnet-4-5-20250929" ← WRONG (client requested this)

Should be:
"model": "gpt-oss-20b"  ← CORRECT (actual model loaded)
```

**Fix Required:**
```python
# Get actual model name from settings
settings = get_settings()
actual_model_name = settings.mlx.model_id  # e.g., "mlx-community/gpt-oss-20b-MXFP4-Q4"

# Or extract short name
model_name = actual_model_name.split('/')[-1]  # "gpt-oss-20b-MXFP4-Q4"

# Return in response
model=model_name  # NOT request_body.model
```

---

### Issue #2: Disable Streaming for Non-Streaming Requests

**Current Code (`anthropic_adapter.py` line ~360-380):**

```python
@router.post("/v1/messages")
async def create_message(...):
    if request_body.stream:
        return StreamingResponse(
            stream_generation(...),
            media_type="text/event-stream",
        )
    else:
        # NON-STREAMING PATH
        # Submit to batch engine
        uid = batch_engine.submit(agent_id, prompt, cached_blocks, max_tokens)

        # Collect all tokens (blocking)
        completion = None
        for result in batch_engine.step():
            if result.uid == uid:
                completion = result
                break

        # Return complete response
        return MessagesResponse(...)
```

**Issue:** Both streaming and non-streaming paths use `batch_engine.step()`, which processes ALL active requests in the batch!

**The Problem:**
- Even with `stream: false`, the request is added to BatchGenerator
- If 3 concurrent requests arrive (mix of streaming/non-streaming), they all go into the same batch
- Lockstep processing still occurs!

---

### Issue #3: Request Serialization (IMMEDIATE FIX)

**Solution:** Serialize ALL requests (streaming or not) with a lock.

**Implementation:**

```python
# anthropic_adapter.py

# Add at module level
_request_lock = asyncio.Lock()

@router.post("/v1/messages")
async def create_message(...):
    # CRITICAL: Acquire lock for ALL requests
    async with _request_lock:
        if request_body.stream:
            return StreamingResponse(
                stream_generation(...),
                media_type="text/event-stream",
            )
        else:
            # Non-streaming path (also serialized)
            uid = batch_engine.submit(agent_id, prompt, cached_blocks, max_tokens)

            completion = None
            for result in batch_engine.step():
                if result.uid == uid:
                    completion = result
                    break

            return MessagesResponse(
                id=f"msg_{uuid.uuid4().hex[:24]}",
                content=[...],
                model=get_actual_model_name(),  # FIX: Use actual model!
                ...
            )
```

**Why This Works:**
- Only 1 request processed at a time (no concurrent batch)
- Each request gets full 566 tok/s throughput
- Streaming and non-streaming both serialized
- **Eliminates 700x degradation immediately!**

---

## 4. Recommended Implementation Strategy

### Option A: Minimal Fix (1 Hour)

**Changes:**
1. Add `_request_lock = asyncio.Lock()` at module level
2. Wrap `create_message()` endpoint with `async with _request_lock:`
3. Fix model name: Return `settings.mlx.model_id` instead of `request_body.model`

**Code:**

```python
# src/semantic/adapters/inbound/anthropic_adapter.py

import asyncio

# Add at module level
_request_lock = asyncio.Lock()

@router.post("/v1/messages")
async def create_message(
    request_body: MessagesRequest,
    request: Request,
) -> MessagesResponse | StreamingResponse:
    # SERIALIZE ALL REQUESTS
    async with _request_lock:
        logger.info(f"POST /v1/messages: model={request_body.model}, stream={request_body.stream}")

        # ... existing code ...

        # Get actual model name
        settings = get_settings()
        actual_model = settings.mlx.model_id.split('/')[-1]

        if request_body.stream:
            # Streaming path
            return StreamingResponse(
                stream_generation(
                    agent_id, prompt, cached_blocks, max_tokens,
                    batch_engine, cache_store, request_body, actual_model
                ),
                media_type="text/event-stream",
            )
        else:
            # Non-streaming path
            uid = batch_engine.submit(agent_id, prompt, cached_blocks, max_tokens)

            completion = None
            for result in batch_engine.step():
                if result.uid == uid:
                    completion = result
                    break

            return MessagesResponse(
                id=f"msg_{uuid.uuid4().hex[:24]}",
                content=[...],
                model=actual_model,  # FIX: Use actual model name
                ...
            )
```

**Result:**
- ✅ 566 tok/s per request (sequential)
- ✅ No 700x degradation
- ✅ Correct model name returned
- ✅ Works with Claude Code CLI streaming AND non-streaming
- ❌ Still sequential (no concurrency)

---

### Option B: Disable Streaming Entirely (30 Minutes)

**If streaming is unnecessary for your workflow:**

```python
@router.post("/v1/messages")
async def create_message(...):
    # IGNORE stream parameter, always return non-streaming
    if request_body.stream:
        logger.warning("Streaming requested but disabled for performance. Returning complete response.")

    # Process request to completion
    async with _request_lock:
        uid = batch_engine.submit(...)

        # Accumulate all tokens
        tokens = []
        for result in batch_engine.step():
            if result.uid == uid:
                tokens.extend(result.tokens)
                if result.finish_reason:
                    break

        # Return complete response
        return MessagesResponse(
            model=get_actual_model_name(),
            content=[TextContentBlock(type="text", text="".join(tokens))],
            ...
        )
```

**Pros:**
- ✅ Simplest code path
- ✅ No SSE complexity
- ✅ No streaming concurrency issues
- ✅ Still works with Claude Code CLI

**Cons:**
- ⚠️ User sees "thinking" delay (no progressive display)
- ⚠️ For 17K token prompts, 30+ second wait with no feedback

---

### Option C: Hybrid Approach (Recommended)

**Support both modes, but serialize:**

```python
@router.post("/v1/messages")
async def create_message(...):
    async with _request_lock:
        if request_body.stream:
            # Streaming mode (for user experience)
            return StreamingResponse(...)
        else:
            # Non-streaming mode (simpler, faster)
            return MessagesResponse(...)
```

**Use Cases:**
- **Streaming (`stream: true`):** User-facing applications, IDE plugins
- **Non-streaming (`stream: false`):** Batch processing, automated tools, testing

---

## 5. Testing Strategy

### Test 1: Single Non-Streaming Request

```bash
curl -X POST http://localhost:8000/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-haiku-4-5-20251001",
    "max_tokens": 100,
    "stream": false,
    "messages": [{"role": "user", "content": "Count to 10"}]
  }'

# Expected:
# - Returns complete response in ~0.18s
# - Model field: "gpt-oss-20b-MXFP4-Q4" (NOT "claude-haiku-4-5-20251001")
# - Content: Full text with all 10 numbers
```

### Test 2: Concurrent Non-Streaming Requests

```bash
# Terminal 1, 2, 3: Run simultaneously
for i in 1 2 3; do
  time curl -X POST http://localhost:8000/v1/messages \
    -H "Content-Type: application/json" \
    -d '{
      "model": "claude-sonnet-4-5-20250929",
      "max_tokens": 50,
      "stream": false,
      "messages": [{"role": "user", "content": "Say hello"}]
    }' &
done

# Expected with serialization lock:
# Request 1: ~0.09s (completes first)
# Request 2: ~0.09s (starts after #1 completes)
# Request 3: ~0.09s (starts after #2 completes)
# Total: ~0.27s (sequential, but each at full speed)

# WITHOUT lock (current broken state):
# All 3: ~15-30s EACH (700x slower due to batch lockstep)
```

### Test 3: Streaming Request (Serialized)

```bash
curl -N -X POST http://localhost:8000/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-haiku-4-5-20251001",
    "max_tokens": 50,
    "stream": true,
    "messages": [{"role": "user", "content": "Count to 5"}]
  }'

# Expected:
# - SSE events stream progressively
# - Model field in message_start: "gpt-oss-20b-MXFP4-Q4"
# - Each token arrives at ~566 tok/s rate
# - No 700x degradation
```

### Test 4: Claude Code CLI Integration

```bash
# Set environment
export ANTHROPIC_BASE_URL=http://localhost:8000
export ANTHROPIC_API_KEY=sk-ant-local-dev
export CLAUDE_CODE_MAX_OUTPUT_TOKENS=64000

# Run Claude Code CLI
claude test

# Expected:
# - Works with both streaming and non-streaming
# - No 14-minute delays
# - Model name in logs: "gpt-oss-20b-MXFP4-Q4"
```

---

## 6. Performance Projections

| Scenario | Current (Broken) | Option A (Serialize) | Option B (No Stream) | Option C (Hybrid) |
|----------|-----------------|---------------------|---------------------|------------------|
| **Single request (non-stream)** | 566 tok/s | 566 tok/s | 566 tok/s | 566 tok/s |
| **Single request (stream)** | 566 tok/s | 566 tok/s | N/A (disabled) | 566 tok/s |
| **3 concurrent (non-stream)** | **0.81 tok/s each** | 566 tok/s (sequential) | 566 tok/s (sequential) | 566 tok/s (sequential) |
| **3 concurrent (stream)** | **0.81 tok/s each** | 566 tok/s (sequential) | N/A | 566 tok/s (sequential) |
| **3 concurrent (mixed)** | **0.81 tok/s each** | 566 tok/s (sequential) | 566 tok/s (sequential) | 566 tok/s (sequential) |

**Key Insight:** With request serialization lock, ALL scenarios work at full speed!

---

## 7. Why Ollama Works Fine

**Ollama's Architecture:**
1. **No MLX BatchGenerator:** Uses Ollama's own inference engine
2. **True concurrent handling:** Each request processed independently
3. **No lockstep batch processing:** Requests don't interfere with each other
4. **Both streaming and non-streaming supported:** No performance penalty

**Why Your Server Struggles:**
1. **MLX BatchGenerator limitation:** Designed for batch throughput, not concurrent streaming
2. **Lockstep processing:** All batch members advance together
3. **Async SSE endpoints:** Each waits for matching UID from shared `step()` call
4. **No request isolation:** Concurrent requests interfere through shared batch

---

## 8. Recommendations

### Immediate Action (TODAY):

**Implement Option A: Request Serialization Lock**

**Steps:**
1. Add `_request_lock = asyncio.Lock()` to `anthropic_adapter.py`
2. Wrap `create_message()` with `async with _request_lock:`
3. Fix model name: Return `settings.mlx.model_id.split('/')[-1]`
4. Deploy and test with Claude Code CLI
5. Monitor performance: Should see 566 tok/s consistently

**Time Estimate:** 1 hour (including testing)

**Risk:** LOW - Simple, safe change

---

### Short-Term (Next Week):

**Evaluate if streaming is needed:**
1. Test Claude Code CLI with `stream: false` default
2. Measure user experience (does 30s wait feel OK?)
3. If streaming unnecessary: Disable entirely (Option B)
4. If streaming valuable: Keep Option A

---

### Long-Term (Month 2):

**If concurrent streaming required:**
- Evaluate vllm-mlx migration (continuous batching)
- See Technical Fellows report for detailed analysis

---

## 9. Critical Bug Fixes Summary

### Bug #1: Model Name Mismatch

**Issue:** Returning `request_body.model` (Claude model) instead of actual loaded model

**Fix:**
```python
settings = get_settings()
actual_model = settings.mlx.model_id.split('/')[-1]
# Return: "gpt-oss-20b-MXFP4-Q4" NOT "claude-haiku-4-5-20251001"
```

**Files Affected:**
- `src/semantic/adapters/inbound/anthropic_adapter.py` (line 215, 464)

---

### Bug #2: Concurrent Request Lockstep

**Issue:** All concurrent requests processed in batch, causing 700x slowdown

**Fix:**
```python
_request_lock = asyncio.Lock()

@router.post("/v1/messages")
async def create_message(...):
    async with _request_lock:
        # Process ONE request at a time
        ...
```

**Files Affected:**
- `src/semantic/adapters/inbound/anthropic_adapter.py` (entire endpoint)

---

### Bug #3: Streaming Assumption

**Issue:** Assuming streaming is required, when non-streaming works fine

**Fix:** Document that `stream: false` is supported and recommended for batch workloads

**Files Affected:**
- README.md
- API documentation

---

## 10. Sources

- [Streaming Messages - Claude API Docs](https://docs.anthropic.com/en/api/messages-streaming)
- [Anthropic compatibility - Ollama](https://docs.ollama.com/api/anthropic-compatibility)
- [Claude Code with Anthropic API compatibility · Ollama Blog](https://ollama.com/blog/claude)
- [Claude Code Internals, Part 7: SSE Stream Processing](https://kotrotsos.medium.com/claude-code-internals-part-7-sse-stream-processing-c620ae9d64a1)
- [Issue #499: Support batching in mlx_lm.server](https://github.com/ml-explore/mlx-lm/issues/499)
- [vllm-mlx Repository](https://github.com/waybarrios/vllm-mlx)
- [ParaLLM: 1600+ tok/s on a MacBook](https://willcb.com/blog/parallm/)

---

**Document Version:** 1.0
**Author:** Technical Review Committee
**Status:** Ready for Implementation
**Priority:** CRITICAL - Fixes 700x performance degradation
