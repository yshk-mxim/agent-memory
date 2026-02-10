# Forensic Analysis: Bugs Found During Development of the Semantic Caching System

**Date**: 2026-02-09
**Scope**: Four critical bugs discovered during development and hardening of the MLX-based semantic caching server on Apple Silicon (M4 Pro, 24 GB unified memory).
**Period**: 2026-02-06 through 2026-02-09

---

## Table of Contents

1. [BUG 1: Thread Safety Race Condition (Batch=2 Cached Empty Output)](#bug-1-thread-safety-race-condition)
2. [BUG 2: Sliding Window Mask Corruption (Chunked Prefill)](#bug-2-sliding-window-mask-corruption)
3. [BUG 3: GQA Mask Shape Mismatch + mx.compile Crash](#bug-3-gqa-mask-shape-mismatch)
4. [BUG 4: Temperature / Tokenizer Interaction (Misattributed Root Cause)](#bug-4-temperature-tokenizer-interaction)
5. [Cross-Cutting Lessons](#cross-cutting-lessons)

---

## BUG 1: Thread Safety Race Condition

**Severity**: Critical (SIGSEGV, data corruption)
**Affected configuration**: batch=2 + warm/hot cache + 1K-2K token context
**Commits**: `371aaa0`, `90f0baf`, `30ae40b`
**Files**: `batch_engine.py`, `safetensors_cache_adapter.py`, `domain/services.py`

### Symptom

When running with batch size 2 and the concurrent scheduler enabled, requests that hit warm or hot cache with context lengths between 1K and 2K tokens produced zero output tokens. The model emitted EOS on the very first decode token, returning an empty response. In more severe cases, the server crashed with SIGSEGV during `update_and_fetch()` or `mx.save_safetensors()`.

The failure was intermittent: the same prompt and cache state could succeed or fail depending on timing. Cold-start requests (no cached context) were unaffected. Single-request mode (batch=1) was unaffected.

### Investigation Timeline

#### Phase 1: Initial Observation (2026-02-09, morning)

During the COLM benchmark suite (`colm_full_benchmark.py`), staggered concurrent requests to Gemma 3 12B Q4 produced empty completions. The benchmark recorded `early_eos=True` for affected requests. Both models (Gemma 3 and DeepSeek-Coder-V2-Lite) exhibited the same pattern, but only when:

- Batch size = 2 (two concurrent sequences in the decode loop)
- The incoming request had a warm or hot cache hit (previously computed KV data)
- Context length was in the 1K-2K token range

#### Phase 2: Wrong Hypothesis H1 -- Lazy Tensor Chain (2026-02-09, 14:01-14:45)

The first hypothesis was that lazy tensor chains were not being materialized after cache reconstruction. The reasoning went:

1. `_reconstruct_cache()` loads cached KV blocks and concatenates them with `mx.concatenate()`, producing lazy tensors.
2. These lazy tensors are inserted into the batch via `batch_gen.insert()`.
3. If the batch cache buffer needs expansion during `update_and_fetch()` (i.e., the reconstructed cache length is not aligned to the 256-token `step` boundary), the `expand_quant` path creates new lazy concatenated buffers.
4. Subsequent in-place scatter writes (`self.keys[i][..., prev:n, :] = q_keys[i]`) create `SliceUpdate` graph nodes referencing the lazy expansion buffers.
5. If these are not materialized before the first decode token is sampled, the model sees garbage values and immediately produces EOS.

This hypothesis led to commit `90f0baf` ("Materialize lazy tensors after non-step-aligned expansion in Q4 batch cache"), which added an unconditional `mx.eval()` call after any expansion in `update_and_fetch()`.

A verification script (`verify_bug1.py`) was run against 5 previously failing test cases, and all 5 passed. The hypothesis appeared confirmed.

**Why H1 was wrong**: The verification script ran requests sequentially, not concurrently. Sequential execution naturally serializes MLX operations onto a single thread, eliminating the actual race condition. The `mx.eval()` call in the fix did help (belt-and-suspenders), but it was not addressing the true root cause. The real trigger required two threads performing MLX operations simultaneously.

#### Phase 3: True Root Cause Discovery (2026-02-09, 15:00-15:38)

Re-examination of the faulthandler stack traces from earlier SIGSEGV crashes revealed the real pattern. Two distinct crash traces showed:

**Crash Pattern A**:
- Thread 1 (scheduler-worker): inside `update_and_fetch()` at `mlx_quantized_extensions.py:274`, performing the model forward pass via `batch_gen.next()`.
- Thread 2 (event loop): inside `mx.save_safetensors()` at `safetensors_cache_adapter.py:209`, called from `_save_to_disk()` during cache eviction or dirty flush.

**Crash Pattern B**:
- Thread 1 (scheduler-worker): inside `_reconstruct_cache()` at `batch_engine.py:1840`, calling `mx.eval()` on concatenated cache tensors.
- Thread 2 (event loop): inside `mx.save_safetensors()` at `safetensors_cache_adapter.py:209`.

Both crashes had the same structure: **two threads calling MLX operations concurrently**. MLX is explicitly not thread-safe, as confirmed by the framework maintainer (Awni Hannun) in upstream issues #2067, #2133, and #3078:

> "MLX is in general not thread safe. There isn't an easy fix for this." -- Issue #2067

The internal `get_command_encoder()` accesses a static `std::unordered_map` without synchronization. Two threads calling `mx.eval()` concurrently corrupt this map, leading to SIGSEGV, data corruption, or (in the empty-output case) silently corrupted attention computations that cause immediate EOS.

### Root Cause

The system has two threads performing MLX operations:

1. **Scheduler-worker thread**: runs the single-threaded inference loop. Calls `batch_gen.next()` (model forward pass, cache operations) and `submit()` (cache reconstruction, chunked prefill, batch insertion).

2. **Event loop thread (asyncio main thread)**: handles HTTP requests. After a streaming response completes, calls `cache_store.save()` which calls `_save_to_disk()` which calls `safetensors_cache_adapter.save()` which calls `mx.save_safetensors()`.

The `mlx_io_lock` had been introduced in commit `30ae40b` (2026-02-08) to protect `_reconstruct_cache()` and `safetensors_cache_adapter.save()/load()`. However, **the model forward pass inside `batch_gen.next()` was not protected by the lock**. This left a window where:

```
Scheduler thread:  batch_gen.next() -> model forward pass -> update_and_fetch()
Event loop thread:                     _save_to_disk() -> mx.save_safetensors()
```

Both threads execute MLX operations concurrently. The Metal command buffer encoding is not thread-safe. The result is either SIGSEGV (hard crash) or silent data corruption (garbage attention weights leading to immediate EOS).

Additionally, `submit()` performs MLX-intensive operations (cache reconstruction, chunked prefill, batch insertion) and was also unprotected.

### Fix

Commit `371aaa0` ("Thread safety -- mlx_io_lock in submit(), RLock, unconditional eval"):

1. **`mlx_io_lock` wraps `batch_gen.next()` in `step_once()`** (`batch_engine.py:1568`):
   ```python
   with mlx_io_lock:
       batch_response = self._batch_gen.next()
   ```
   This ensures the entire model forward pass (including all cache operations triggered by filter/extend/extract inside `_next()`) is serialized against disk I/O.

2. **`mlx_io_lock` wraps the MLX-intensive section of `submit()`** (`batch_engine.py:830-1143`):
   ```python
   mlx_io_lock.acquire()
   try:
       # sampler creation, cache reconstruction, chunked prefill,
       # batch_gen.insert(), all mx.eval() calls
       ...
   finally:
       mlx_io_lock.release()
   ```

3. **`mlx_io_lock` changed from `threading.Lock` to `threading.RLock`** (`domain/services.py:18`):
   ```python
   mlx_io_lock = threading.RLock()
   ```
   This is necessary because `submit()` acquires the lock and then calls `_chunked_prefill()`, which also acquires it (reentrant acquisition on the same thread).

4. **Unconditional `mx.eval()` after non-step-aligned expansion** (retained from the H1 fix in `90f0baf`):
   As belt-and-suspenders, the materialization call remains even though it was not the true root cause. It prevents lazy tensor chain buildup that could compound with future timing-sensitive bugs.

### Validation

- Gemma 3 12B Q4: 15/15 benchmark runs passed (cold + warm cache, varied prompts, max_tokens). 198/198 quality_ok.
- DeepSeek-Coder-V2-Lite 4-bit: 11/11 benchmark runs passed. 198/198 quality_ok.
- Re-ran all 5 original failing test cases from H1 verification -- all passed.
- No SIGSEGV crashes observed across the full benchmark suite.

### Key Lesson

**Sequential verification scripts can mask concurrency bugs.** The H1 verification appeared to confirm a wrong hypothesis because it eliminated the race condition by running requests one at a time. Concurrency bugs require concurrent test harnesses. The faulthandler stack traces (showing two threads in MLX code simultaneously) were the critical evidence.

---

## BUG 2: Sliding Window Mask Corruption

**Severity**: High (silent data corruption, garbage output)
**Affected configuration**: Gemma 3 with chunked prefill enabled (prompts > 2048 tokens)
**Commit**: `ee24513`
**File**: `mlx_quantized_extensions.py`

### Symptom

When chunked prefill was enabled for Gemma 3 12B Q4, prompts longer than 2048 tokens produced corrupted output. The generated text was incoherent -- not simply wrong answers, but garbled token sequences that bore no relationship to the prompt. Shorter prompts (below the chunking threshold) worked correctly. DeepSeek-Coder-V2-Lite was unaffected.

### Investigation Timeline

#### Phase 1: Isolating the Trigger (2026-02-08, morning)

The corruption was first noticed during benchmark runs with long context prompts. Systematic testing narrowed the trigger:

- Prompts under 2048 tokens (below the interleave threshold): correct output.
- Prompts over 2048 tokens (chunked prefill activated): corrupted output.
- Gemma 3: affected. DeepSeek: unaffected.

The difference between the two models pointed toward Gemma 3's hybrid attention architecture. Gemma 3 12B uses a mix of two attention types across its 46 transformer layers:

- **6 global attention layers** (layers 0-5 and periodic thereafter): full causal attention over the entire context.
- **40 sliding window layers**: attention limited to a window of 4096 tokens. Tokens outside the window are masked out.

DeepSeek uses uniform global attention across all layers, which is why it was unaffected.

#### Phase 2: Tracing the Mask Path (2026-02-08, midday)

Chunked prefill processes a long prompt in 256-token chunks rather than all at once. Each chunk runs a model forward pass with a prefix cache already populated from previous chunks. The attention mask for each chunk must correctly represent the causal structure, including the sliding window constraint for windowed layers.

The code path for mask creation during chunked prefill:

1. `_chunked_prefill()` calls `self._model(chunk_tokens, cache=kv_caches)`.
2. Inside the model, each attention layer calls `cache.make_mask(N=chunk_size)`.
3. For `QuantizedKVCache` (the single-sequence cache used during chunked prefill), `make_mask()` delegates to `cache.create_attention_mask()`.

**The bug**: `QuantizedKVCache.make_mask()` in upstream `mlx-lm` calls `self.create_attention_mask(h, cache=self)`, which returns the **string** `"causal"` when `N > 1` and `return_array=False`. This string is a shortcut that tells the model "use standard causal masking." However, it **completely ignores the `window_size` parameter**. Sliding window layers that should restrict attention to a 4096-token window instead receive full causal attention over the entire cached context.

This means that during chunked prefill, sliding window layers compute attention over positions they should never see. The resulting KV cache entries are computed with wrong attention patterns. When subsequent decode steps use these corrupted cache entries, every generated token is garbage.

#### Phase 3: Why BatchQuantizedKVCache Was Not Affected

Our custom `BatchQuantizedKVCache.make_mask()` (used during batch decode, not chunked prefill) was already correct:

```python
def make_mask(self, N=1, return_array=False, **kwargs):
    from mlx_lm.models.base import create_causal_mask
    mask = create_causal_mask(N, offset=self._idx,
                              left_padding=self.left_padding, **kwargs)
    return mask
```

It calls `create_causal_mask()` directly, passing `**kwargs` which includes `window_size`. The function `create_causal_mask()` correctly handles window_size by creating an explicit mask array that zeros out positions outside the window.

The bug was only in the upstream `QuantizedKVCache.make_mask()`, which is used for single-sequence operations like chunked prefill.

### Root Cause

`QuantizedKVCache.make_mask()` in `mlx-lm` v0.30.4 returns the string `"causal"` for multi-token chunks (`N > 1`), ignoring the `window_size` parameter entirely. For models with uniform global attention (like DeepSeek), this is harmless because "causal" is the correct mask. For models with hybrid attention (like Gemma 3, which has 40 out of 46 layers using sliding window attention), this produces incorrect attention patterns during chunked prefill.

The corruption is silent: no error is raised, no warning is logged. The model simply computes attention over the wrong set of positions, producing KV cache entries that are mathematically valid but semantically wrong.

### Fix

Commit `ee24513` ("Patch QuantizedKVCache.make_mask for sliding window attention in chunked prefill"):

A monkeypatch applied during Q4 extension initialization replaces `QuantizedKVCache.make_mask` with a corrected version:

```python
def patched_make_mask(self, N=1, return_array=False, window_size=None, **kwargs):
    from mlx_lm.models.base import create_causal_mask

    if window_size is not None:
        if N == 1 and self.offset <= window_size:
            return None  # Single-token decode within window: no mask needed
        return create_causal_mask(
            N, offset=self.offset, window_size=window_size, **kwargs
        )
    if N == 1:
        return None
    if return_array:
        return create_causal_mask(N, offset=self.offset, **kwargs)
    return "causal"
```

When `window_size` is specified (as it is for Gemma 3's 40 sliding window layers), the function creates an explicit mask array via `create_causal_mask()` instead of returning the string shortcut. When `window_size` is `None` (global attention layers), the original behavior is preserved.

A runtime validation check was also added to `validate_q4_pipeline()`:

```python
test_cache = QuantizedKVCache(group_size=64, bits=4)
test_cache.offset = 1000
mask = test_cache.make_mask(N=512, window_size=256)
if mask is None or (isinstance(mask, str) and mask == "causal"):
    logger.error("[Q4 VALIDATE] QuantizedKVCache.make_mask returned %s "
                 "instead of explicit mask array.", repr(mask))
    return False
```

This catches regression if a future `mlx-lm` update overrides the patch.

### Validation

- 10/10 prompts with context lengths between 1018 and 2482 tokens produced correct output with chunked prefill enabled on Gemma 3.
- The same prompts without the patch produced garbled output (confirmed regression test).
- DeepSeek continued to work correctly (no sliding window layers, so the patch has no effect).

### Key Lesson

**String sentinel returns in APIs are dangerous.** The `make_mask()` method returns either an array or the string `"causal"` -- a polymorphic return type that silently drops parameters. Any caller passing `window_size` to a code path that returns `"causal"` gets incorrect behavior with no indication of failure. The fix was not to change the API contract but to ensure the windowed code path always produces an explicit array.

---

## BUG 3: GQA Mask Shape Mismatch + mx.compile Crash

**Severity**: Critical (hard crash, SIGSEGV)
**Affected configuration**: batch > 1 with GQA models (Gemma 3)
**Commits**: `b2d5617`, `04c814d`, `320d25e`, `50a4388`, `12461c1`
**File**: `mlx_fused_attention.py`, `mlx_quantized_extensions.py`

### Symptom

When batch size exceeded 1, the fused Q4 attention path crashed with a dimension mismatch error during attention computation. Gemma 3 (which uses Grouped Query Attention with `n_repeats = n_q_heads / n_kv_heads > 1`) was affected. DeepSeek-Coder-V2-Lite (which uses Multi-Latent Attention with `n_repeats = 1`) was not affected.

In more extreme cases, even after fixing the shape mismatch, `mx.compile` crashed with SIGSEGV when the batch dimension changed from B=2 to B=1 mid-decode (after one sequence in the batch completed and was filtered out).

### Investigation Timeline

This bug was actually a cluster of three interrelated failures that were discovered and fixed incrementally over 2026-02-07.

#### Sub-bug 3a: GQA 5D Reshape vs 4D Mask (2026-02-07, commit `12461c1`)

The fused Q4 attention function for GQA models reshapes queries from 4D to 5D to align query heads with their corresponding KV head groups:

```python
# Q shape: (B, n_q_heads, L, D)  ->  (B, n_kv_heads, n_repeats, L, D)
queries = mx.reshape(queries, (B, n_kv_heads, n_repeats, L, D))
```

The attention mask from the model is 4D: `(B, 1, L, S)`. For B=1, broadcasting works because the batch dimension is trivially 1. For B>1, the 4D mask cannot broadcast against the 5D query-key score tensor `(B, n_kv_heads, n_repeats, L, S)` -- the ranks differ, and NumPy-style broadcasting cannot reconcile them.

**Initial fix attempt**: Add `mx.expand_dims(mask, axis=2)` to make the mask 5D: `(B, 1, 1, L, S)`. This allows broadcasting along both the `n_kv_heads` and `n_repeats` dimensions.

However, this only works when B=1 because the mask is expanded inside the compiled function which had already traced with a specific batch dimension.

#### Sub-bug 3b: mx.compile Batch Dimension Change (2026-02-07, commits `b2d5617`, `04c814d`)

`mx.compile` traces the computation graph on the first call and caches the compiled representation. When called with `shapeless=False` (the default), any input shape change triggers recompilation. When called with `shapeless=True`, the compiled graph is reused regardless of shape changes.

The problem: for GQA, `shapeless=True` is unsafe because the reshape operation `mx.reshape(queries, (B, n_kv_heads, n_repeats, L, D))` reads `queries.shape` to compute dimensions. With `shapeless=True`, the trace bakes in the shapes from the first call. If the first call had B=2 and a subsequent call has B=1, the reshape uses stale dimensions, producing incorrect tensor shapes that crash Metal.

With `shapeless=False`, B=2 to B=1 transitions trigger recompilation. But recompilation during a batch dimension change (in the middle of `batch_gen.next()`, after `filter()` removes a completed sequence) caused SIGSEGV. The compiled function's cached state was being invalidated while Metal command buffers still referenced it.

**The B=1 split solution**: Instead of compiling for variable batch sizes, the batch is always split into individual sequences before calling the compiled function:

```python
if B == 1:
    return compiled_fn(queries, k0, k1, k2, v0, v1, v2, scale_arr, mask)

# Batch>1: split into per-sequence calls, then stack
parts = []
for i in range(B):
    q_i = queries[i:i+1]
    k_i = tuple(k[i:i+1] for k in q_keys)
    v_i = tuple(v[i:i+1] for v in q_values)
    m_i = mask[i:i+1] if mask is not None else None
    parts.append(compiled_fn(q_i, k_i[0], k_i[1], k_i[2],
                             v_i[0], v_i[1], v_i[2], scale_arr, m_i))
return mx.concatenate(parts, axis=0)
```

The compiled function always sees B=1. Batch dimension never changes. Recompilation only happens on sequence length (L) changes, which is benign.

For GQA (`n_repeats > 1`), `shapeless=False` is used because the reshape reads `queries.shape`. Since B is always 1, the only shape variation is L, which triggers safe recompilation.

For non-GQA (`n_repeats = 1`), `shapeless=True` is safe because there is no reshape operation.

#### Sub-bug 3c: Cascading Metal Assertion Failures (2026-02-07, commits `04c814d`, `50a4388`)

Even after the B=1 split, batch decode produced Metal assertion failures:

1. **"Completed handler after commit call"**: Caused by `mx.async_eval()` scheduling GPU work while the previous command buffer's completion handler was still running. Fixed by replacing all `mx.async_eval()` calls with synchronous `mx.eval()` (commit `04c814d`). This eliminates the CPU/GPU overlap pipeline from upstream `mlx-lm` but prevents the assertion.

2. **"commit command buffer with uncommitted encoder"**: Caused by `filter()` in the batch KV cache. After filtering, the `generation_stream` might have uncommitted command encoders from the forward pass. The next decode step's `mx.eval()` only syncs the default stream (which inside `mx.stream(generation_stream)` IS the generation stream), but the encoder state was not fully committed. Fixed by adding `mx.synchronize()` after `mx.eval()` in `filter()` (commit `50a4388`):

   ```python
   # In BatchQuantizedKVCache.filter():
   mx.eval(*self.keys, *self.values)
   mx.synchronize()  # Sync generation_stream -- ensures encoder committed
   ```

#### Sub-bug 3d: Lazy Tensor Materialization at Transitions (2026-02-07, commits `320d25e`, `b2d5617`)

The `filter()`, `extend()`, `merge()`, and `extract()` operations on `BatchQuantizedKVCache` all produce lazy tensors (via indexing, concatenation, `mx.contiguous()`). If these are not materialized before the next operation that changes the batch dimension, Metal tries to resolve lazy graph nodes with stale shape assumptions.

Each transition point was audited and `mx.eval()` calls were added or restored:

- `filter()`: `mx.eval()` on all keys/values after indexing (line 308)
- `extend()`: `mx.eval()` on all keys/values after concatenation (line 391)
- `merge()`: `mx.eval()` on all keys/values after slice assignments (line 158)
- `extract()`: `mx.eval()` on all keys/values after `mx.contiguous()` (line 433)

Additionally, `clip_residual` (a function called 52 times per forward pass in Gemma 3) was wrapped with `mx.compile` by upstream `mlx-lm`, which interacted badly with the batch decode path. The monkeypatch removes `mx.compile` from `clip_residual` (commit `b2d5617`).

### Root Cause (Composite)

Four interacting failures:

1. **Shape mismatch**: GQA 5D reshape produces tensors incompatible with 4D masks when B > 1.
2. **mx.compile shape tracing**: Compiled functions cache shapes and fail on batch dimension changes.
3. **Async command buffer conflicts**: `mx.async_eval()` races with completion handlers.
4. **Lazy tensor chains**: Unmaterialized tensors at batch dimension transitions crash Metal's graph resolver.

### Fix (Composite)

1. **B=1 split strategy**: Always decompose batch into individual sequences before calling compiled attention. Eliminates shape tracing issues entirely.
2. **`mx.async_eval` replaced with `mx.eval`**: Eliminates async command buffer race conditions.
3. **`mx.synchronize()` after `filter()`**: Ensures Metal command encoders are committed before the next decode step.
4. **`mx.eval()` at all cache transition points**: Materializes lazy tensors before batch dimension changes.
5. **`clip_residual` decompiled**: Removes nested `mx.compile` interaction in the hot path.

### Validation

- 25/25 rounds across 5 separate invocations passed (cold start, warm cache, varied prompts, varied max_tokens).
- Both GQA (Gemma 3, `n_repeats = 4`) and non-GQA (DeepSeek, `n_repeats = 1`) paths validated.
- No Metal assertions observed across the full validation suite.

### Key Lesson

**Compiled GPU kernels and dynamic batch sizes are fundamentally incompatible.** The B=1 split strategy trades a small amount of overhead (loop + concatenation) for complete elimination of an entire class of shape-dependent bugs. When `mx.compile` traces a function, it creates a contract about input shapes; violating that contract can corrupt Metal state in ways that manifest as crashes in unrelated code. The only safe approach is to guarantee the compiled function always sees identical shapes.

---

## BUG 4: Temperature / Tokenizer Interaction (Misattributed Root Cause)

**Severity**: High (all spaces removed from generated text)
**Affected configuration**: DeepSeek-Coder-V2-Lite with `transformers==5.0.0rc1`
**Commit**: `af08cfb`
**Files**: `requirements.txt`, `pyproject.toml`

### Symptom

Generated text from DeepSeek-Coder-V2-Lite had all spaces removed. A prompt asking for a code explanation would produce output like:

```
Thefunctiontakesalistofintegersandreturnsthesumofalleven...
```

instead of:

```
The function takes a list of integers and returns the sum of all even...
```

The output was semantically correct -- the model was producing the right tokens -- but every space character was missing from the decoded text.

### Investigation Timeline

#### Phase 1: Blaming Temperature (2026-02-06, initial hypothesis)

The spacing corruption was first observed during multi-turn dialogue testing with T=0. The initial hypothesis was that T=0 (deterministic greedy decoding) caused some kind of echo loop or degenerate sampling that suppressed space tokens.

Evidence that appeared to support this:
- Switching from T=0 to T=0.1 seemed to improve (but not fully fix) the output.
- T=0 is known to cause deterministic repetition loops in some models.
- The DeepSeek model's `generation_config.json` specifies T=0.3.

The team changed `coordination_service.py` to hardcode T=0.3 (bypassing the environment variable `SEMANTIC_MLX_DEFAULT_TEMPERATURE`) and moved on, believing the issue was resolved.

#### Phase 2: Spacing Corruption Returns (2026-02-06, later)

After hardcoding T=0.3, the spacing corruption persisted in new test runs. This ruled out temperature as the root cause. If T=0 were the problem, T=0.3 should have fixed it completely.

#### Phase 3: Bisecting Dependencies (2026-02-06, evening)

A dependency audit revealed that `mlx-lm==0.30.4` had pulled in `transformers==5.0.0rc1` as a transitive dependency. The `transformers` library had undergone a major version bump from 4.x to 5.0.0rc1.

Testing with `transformers==4.57.6` (pinned manually) eliminated the spacing corruption entirely. Testing with `transformers==5.0.0rc1` reproduced it 100% of the time. Temperature was irrelevant.

#### Phase 4: Root Cause in Transformers v5 (2026-02-06)

The `transformers` v5.0.0rc1 release changed how tokenizers are initialized. From the maintainers:

> "v5 changed to 'manually generate tokenizers' instead of trusting saved config files."

For SentencePiece-based tokenizers (used by DeepSeek and many other models), this change broke the handling of the `\u2581` (lower one eighth block, used as the space marker in SentencePiece vocabulary). The space marker was stripped from token pieces during tokenizer construction, meaning that `tokenizer.decode()` produced text without spaces.

This was not model-specific: any SentencePiece model running under `transformers==5.0.0rc1` would exhibit the same bug. It was filed upstream as [huggingface/transformers#43066](https://github.com/huggingface/transformers/issues/43066), with a fix in [PR #42894](https://github.com/huggingface/transformers/pull/42894).

### Root Cause

`transformers==5.0.0rc1` breaks SentencePiece decode by stripping the `\u2581` space marker from token pieces during tokenizer construction. Round-trip encode/decode loses all spaces. This is a library bug, not a model bug or a temperature bug.

The initial temperature hypothesis was a classic case of **confounding variables**: T=0 did cause observable repetition issues (a real but separate problem), and the spacing bug was present at all temperatures. Switching to T=0.3 improved the repetition issue, making the output seem more correct, but the spacing corruption was still there -- it was just less obvious in shorter outputs at T=0.3 where the model naturally produced fewer space-sensitive constructions.

### Fix

Commit `af08cfb` ("Pin transformers<5.0.0 -- v5.0.0rc1 breaks SentencePiece decode"):

```
# requirements.txt
transformers>=4.47.0,<5.0.0  # v5.0.0rc1 breaks SentencePiece decode (gh #43066)

# pyproject.toml
"transformers>=4.47.0,<5.0.0",  # v5.0.0rc1 breaks SentencePiece decode (gh #43066)
```

Note: `mlx-lm==0.30.4` declares `transformers==5.0.0rc1` as a dependency. pip warns about the conflict but respects our explicit pin. The system works correctly with `transformers==4.57.6`.

The T=0.3 hardcode in `coordination_service.py` was retained because it matches DeepSeek's official generation config and avoids the separate (real) T=0 repetition loop issue. A stale comment on line 614 that blamed temperature for the spacing corruption was corrected to explain the true cause.

### Validation

- DeepSeek-Coder-V2-Lite: all generated text has correct spacing with `transformers<5.0.0`.
- Gemma 3: also uses SentencePiece, was silently affected, now correct.
- T=0.3 works correctly for both models post-fix.

### Key Lesson

**Correlation is not causation, especially with multiple simultaneous bugs.** The spacing corruption and the T=0 repetition loop were two independent bugs that happened to be observed at the same time. Fixing one (T=0 to T=0.3) appeared to fix the other because it changed the output distribution enough to mask the remaining problem. Only systematic dependency bisection revealed the true root cause. Always bisect when a fix "mostly works."

---

## Cross-Cutting Lessons

### 1. MLX Threading is the Fundamental Constraint

Three of the four bugs (BUG 1, BUG 3, and partially BUG 2 via its interaction with chunked prefill) trace back to MLX's lack of thread safety and the challenges of lazy evaluation on GPU. The MLX framework provides no thread-safe API surface whatsoever. The framework maintainers have stated that true thread safety "will take some time."

The single most important architectural decision was routing all MLX inference through a single scheduler-worker thread and serializing cross-thread MLX operations with `mlx_io_lock`. This is not elegant, but it is the only correct approach given the framework's constraints.

### 2. Lazy Evaluation Creates Invisible Dependencies

MLX's lazy evaluation model means that the apparent site of a bug (where the crash occurs) is often far from the actual site (where the lazy graph was incorrectly constructed). BUG 1's empty output appeared to be a sampling issue but was actually a threading issue in graph evaluation. BUG 3's Metal assertions appeared in the attention kernel but were caused by unmaterialized tensors from cache operations.

The defensive strategy is: **materialize at every transition point**. Call `mx.eval()` after every operation that changes the shape semantics of a tensor (filter, extend, merge, extract). The cost is small (the tensors need to be evaluated eventually), and the benefit is that bugs manifest immediately rather than propagating through lazy chains.

### 3. Verification Must Match the Failure Mode

BUG 1's wrong hypothesis (H1, lazy tensor chain) was "confirmed" by a sequential test script. The actual bug was a concurrency race that only manifested under concurrent execution. BUG 4's wrong hypothesis (temperature) was "confirmed" by changing temperature, which coincidentally improved a different issue.

Verification scripts must reproduce the exact conditions of the failure. For concurrency bugs, this means concurrent requests. For timing-sensitive bugs, this means realistic timing (not artificial serialization). For dependency bugs, this means controlled dependency versions.

### 4. Monkeypatching Upstream Code Requires Validation Gates

BUG 2 existed because upstream `mlx-lm`'s `QuantizedKVCache.make_mask()` silently dropped the `window_size` parameter. Our fix was a monkeypatch, which is fragile -- a future `mlx-lm` update could overwrite it. The validation gate in `validate_q4_pipeline()` catches this regression automatically at startup.

Every monkeypatch should have a corresponding validation check that runs at initialization and fails loudly if the patch is not in effect.

### 5. Dependency Management is a Safety Issue

BUG 4 was caused entirely by a transitive dependency (`mlx-lm` pulling in `transformers==5.0.0rc1`). The fix was a version pin. On Apple Silicon / MLX systems where the entire inference stack is Python-native (no CUDA/cuDNN binary isolation), a single bad transitive dependency can silently corrupt model output with no error messages.

Pin all critical dependencies explicitly. Do not trust transitive resolution for inference-critical libraries.
