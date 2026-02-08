# MLX Batch Inference: Threading, Memory, and Tensor Lifecycle Analysis

**Date**: 2026-02-08
**Context**: Debugging batch=2 concurrent inference crashes (SIGSEGV) in the semantic caching server.
**Motivation**: Multiple attempts to fix batch decode crashes led to system reboots. This analysis documents the full top-down/bottom-up understanding before any further code changes.

---

## Table of Contents

1. [MLX Framework Fundamentals](#1-mlx-framework-fundamentals)
   - 1.1 Threading Safety
   - 1.2 Streams Architecture
   - 1.3 Memory Model and Lazy Evaluation
   - 1.4 `mx.compile` Behavior
   - 1.5 Key Guarantees and Non-Guarantees
2. [mlx-lm Batch Architecture](#2-mlx-lm-batch-architecture)
   - 2.1 BatchGenerator Lifecycle
   - 2.2 Batch Dataclass
   - 2.3 BatchKVCache (FP16)
   - 2.4 KVCache (Single Sequence)
   - 2.5 QuantizedKVCache (Single Sequence)
   - 2.6 Data Flow: B=2 Decode Step
   - 2.7 Filter Transition (B=2 -> B=1)
   - 2.8 Extend Transition (B=1 -> B=2)
   - 2.9 Key Shapes During B=2 Decode
3. [Our Code: Thread Map and MLX Operations Audit](#3-our-code-thread-map-and-mlx-operations-audit)
   - 3.1 Thread Map
   - 3.2 Per-Function MLX Operation Inventory
   - 3.3 Shared Data Structures
   - 3.4 Lazy Tensor Creation vs Materialization Map
   - 3.5 In-Place Modification Map
4. [Gap Analysis: Identified Bugs and Risks](#4-gap-analysis-identified-bugs-and-risks)
5. [Crash Forensics](#5-crash-forensics)
6. [Conclusions and Fix Strategy](#6-conclusions-and-fix-strategy)

---

## 1. MLX Framework Fundamentals

### 1.1 Threading Safety

**MLX is NOT thread-safe.** Confirmed by Awni Hannun (MLX maintainer):

> "Mlx is in general not thread safe. There isn't an easy fix for this." — [Issue #2067](https://github.com/ml-explore/mlx/issues/2067)

> "I think it will be a while before we can provide true thread safety...doing it well without taking performance hits will take some time." — [Issue #3078](https://github.com/ml-explore/mlx/issues/3078)

[Issue #2133](https://github.com/ml-explore/mlx/issues/2133) tracks three core unsafe areas:

1. **`StreamContext` context manager** — calls `set_default_stream()` which writes to `Scheduler::default_streams_`, a non-thread-safe `std::unordered_map`. Concurrent `StreamContext` usage from multiple threads corrupts this map or causes incorrect restore on destruction.

2. **Compiler cache** — the C++ compile cache is not thread-safe ([Issue #2086](https://github.com/ml-explore/mlx/issues/2086)).

3. **Graph evaluation** — `eval()` of independent graphs from multiple threads is not safe ([Issue #2067](https://github.com/ml-explore/mlx/issues/2067)). Root cause: `get_command_encoder()` accesses a static `unordered_map` without synchronization. Two threads calling `mx.eval()` concurrently crash because Metal's command buffer encoding is not thread-safe.

**What IS internally locked in MLX:**
- `library_mtx_` — shared lock for Metal library cache lookups
- `kernel_mtx_` — shared lock for kernel/pipeline state lookups
- `stream.fence_mtx` — serializes output map modifications during encoder completion

But these protect internal caches only, not the user-facing API.

**Maintainer's recommended workaround:** Use multiple streams within a single thread, or use multiple processes:

```python
# Safe: multiple streams, single thread
with mx.stream(stream_1):
    output_1 = first_model(input_1)
with mx.stream(stream_2):
    output_2 = second_model(input_2)
mx.eval(output_1, output_2)  # Single eval handles cross-stream deps
```

### 1.2 Streams Architecture

#### What Is a Stream?

A stream is an ordered sequence of operations mapping to a Metal command queue:

```cpp
// Each stream index maps to a dedicated MTL::CommandQueue
void Device::new_queue(int index) {
    auto q = device_->newCommandQueue();
    stream_map_.emplace(index, q);
}
```

Each stream maintains: a Metal command queue, an active command buffer, an active command encoder, an operation counter, and fence tracking for cross-stream synchronization.

#### How `generation_stream` Works

In mlx-lm's `generate.py`:

```python
generation_stream = mx.new_stream(mx.default_device())
```

This creates a **second GPU stream** separate from the default stream. `BatchGenerator.next()` wraps `_next()` in `mx.stream(generation_stream)`, which temporarily changes the default stream via `StreamContext` so all operations route to this stream's Metal command queue.

#### `mx.eval()` vs `mx.synchronize()` — CRITICAL DISTINCTION

**`mx.eval(*args)`**: Evaluates the computation graph for given arrays:
1. Builds a tape of operations via BFS/DFS
2. Dispatches each primitive to its stream's command encoder
3. Inserts fences for cross-stream dependencies
4. Commits command buffers when they fill up
5. Blocks until all arrays reach `available` status

**`mx.synchronize(stream=None)`**: Blocks the calling thread until a **single stream** completes:

```cpp
void synchronize(Stream s) {
    auto cb = d.get_command_buffer(s.index);
    d.end_encoding(s.index);
    d.commit_command_buffer(s.index);
    cb->waitUntilCompleted();  // BLOCKS until GPU done
}

void synchronize() {
    synchronize(default_stream(default_device()));  // DEFAULT STREAM ONLY
}
```

**CRITICAL CORRECTION**: The zero-argument `mx.synchronize()` syncs ONLY the current default stream, NOT "all streams." Our comment in `mlx_quantized_extensions.py:319-324` says "Synchronize ALL Metal streams" — this is **misleading**.

However, inside `BatchGenerator._next()` which runs within `mx.stream(generation_stream)`, the `StreamContext` changes the default stream to `generation_stream`. So `mx.synchronize()` called there syncs the `generation_stream`, which is where decode work runs. Since the default GPU stream has no pending model work at that point, this effectively works. But the reasoning in the comment is wrong.

#### Cross-Stream Dependencies (Automatic)

From documentation: "MLX will automatically insert a dependency between the two streams so that the second operation only starts executing after the first is complete."

Implemented via fences:

```cpp
// In end_encoding():
if (auto it = stream.outputs.find(in); it != stream.outputs.end()) {
    enc.wait_for_fence(it->second->fence);  // Wait for producer
}
enc.signal_fence(fence);  // Signal consumers
```

#### Command Buffer Commit Thresholds

Command buffers are committed when thresholds are exceeded:
- Operation count > `max_ops_per_buffer_` (20 on iPhone, 40 on Pro/base, 50 on Max/Ultra)
- Total data size > `max_mb_per_buffer_` (40-50 MB)

### 1.3 Memory Model and Lazy Evaluation

#### Lazy Evaluation

MLX uses lazy evaluation. `c = a + b` records a computation graph node — no computation happens. Computation only executes when triggered by:

- **Explicit**: `mx.eval(c)` or `mx.async_eval(c)`
- **Implicit**: `print(c)`, `.item()`, `.tolist()`, NumPy conversion, saving operations

Metal buffers are allocated **during evaluation**, not during graph construction.

#### Array Status Lifecycle

Arrays have three states:
- **`unscheduled`**: Output of an unscheduled computation (just a graph node)
- **`evaluated`**: `eval_gpu()` has been called, but GPU execution may not be complete
- **`available`**: GPU execution complete, data safe to read

#### In-Place Operations: `a[3:4] = b`

**MLX does NOT truly modify arrays in place at the graph level.** From documentation:

> "Unlike NumPy, slicing an array creates a copy, not a view."

When you write `a[3:4] = b`:
1. Python calls `__setitem__` which invokes `slice_update()`
2. `slice_update()` creates a **new array** with a `SliceUpdate` primitive
3. The Python variable `a` is rebound to this new array object

```cpp
array slice_update(const array& src, const array& update, ...) {
    return array(src.shape(), src.dtype(),
                 std::make_shared<SliceUpdate>(...),
                 {src, upd});  // Returns NEW array
}
```

This is critical for our KV cache code. In `mlx_quantized_extensions.py:265-267`:
```python
self.keys[i][..., prev : prev + n_tokens, :] = q_keys[i][..., :n_tokens, :]
```
Each assignment creates a new lazy array backed by a `SliceUpdate` graph node referencing the parent buffer. If not materialized before a batch dimension change (B=2→B=1), Metal crashes resolving stale parent references.

### 1.4 `mx.compile` Behavior

`mx.compile(fn)` traces the function's graph on first call, optimizes it, and caches it.

**Recompilation triggers:**
- Shape change (partial retrace)
- Type change (full retrace)
- Number of inputs change (full retrace)

**`shapeless=True`** prevents recompilation when input shapes vary. The compiled graph is reused regardless of shape changes.

**Critical caveat** ([Issue #2607](https://github.com/ml-explore/mlx/issues/2607)):
> "Shape-dependent operations (like `reshape()` using hardcoded dimensions) will fail with different input shapes."

Awni's response: "Use shapeless compilations carefully. Since compilation is not triggered when shapes change, any graphs which are conditional on the input shapes will not work as expected."

This is why our code uses the **B=1 split strategy**: instead of compiling for variable batch sizes, each call always sees B=1 and results are concatenated. This avoids shapeless compilation pitfalls entirely.

### 1.5 Key Guarantees and Non-Guarantees

| Property | Guaranteed? | Details |
|----------|-------------|---------|
| Thread safety | **NO** | Not safe to call any MLX ops from multiple threads |
| Cross-stream auto-deps | **YES** | Fences automatically inserted |
| `mx.eval()` blocks until complete | **YES** | Synchronous eval blocks until available |
| `mx.synchronize()` syncs all streams | **NO** | Only syncs the default/specified stream |
| In-place `a[i] = b` mutates buffer | **NO** | Creates new lazy array via `SliceUpdate` |
| `mx.compile(shapeless=True)` handles all shapes | **NO** | Shape-dependent ops bake in initial shape |
| Slicing creates a view | **NO** | Unlike NumPy, creates a copy |
| Lazy tensors safe across batch size changes | **NO** | Lazy graphs crash if batch dim changes |

**Sources:**
- [MLX Issue #2067 - Thread issues with evaluation](https://github.com/ml-explore/mlx/issues/2067)
- [MLX Issue #3078 - Concurrent inference from separate threads](https://github.com/ml-explore/mlx/issues/3078)
- [MLX Issue #2133 - Thread safety tracking](https://github.com/ml-explore/mlx/issues/2133)
- [MLX Issue #2607 - Shapeless matmul bug](https://github.com/ml-explore/mlx/issues/2607)
- [MLX Issue #1707 - ThreadSanitizer data race](https://github.com/ml-explore/mlx/issues/1707)
- [MLX PR #1969 - Multistream GPU deadlock fix](https://github.com/ml-explore/mlx/pull/1969)

---

## 2. mlx-lm Batch Architecture

Based on analysis of `mlx-lm` v0.30.4, file `mlx_lm/generate.py` and `mlx_lm/models/cache.py`.

### 2.1 BatchGenerator Lifecycle

```python
class BatchGenerator:
    def __init__(self, model, tokenizer, ...):
        self.model = model
        self.active_batch = None
        self.unprocessed_prompts = []
        self.prefill_batch_size = 8       # gate: need this many to prefill
        self.completion_batch_size = 32    # max simultaneous decodes
        self.uid_count = 0
```

**Thread model:** `BatchGenerator` runs on **whatever thread calls `next()`**. There is no internal threading. The `with mx.stream(generation_stream)` in `next()` merely routes Metal commands to the generation stream.

**Key methods:**

- **`insert(prompts, ...)`**: Adds prompts to `unprocessed_prompts` queue, sorted by ascending length (shorter first to minimize padding waste). Assigns unique `uid` per prompt.

- **`next()`**: Wraps `_next()` in `mx.stream(generation_stream)`.

- **`_next()`**: Main loop body — handles both prefill and decode in a single call.

  - **Phase 1 — Maybe prefill new prompts**: If `num_to_add >= prefill_batch_size`, takes prompts from queue, prefills them, extends active batch.
  - **Phase 2 — Decode one token**: Runs `_step()` with current token, `mx.async_eval()` the result.
  - **Phase 3 — Check completions**: `y.tolist()` blocks until values computed. For finished sequences: extract cache, filter batch.

- **`_step(input_tokens, cache, ...)`**: Single forward pass. Input shape `(B, 1)` during decode. Returns `(sampled_tokens, logprobs)`.

- **`close()`**: Syncs `generation_stream`, restores wired limit.

### 2.2 Batch Dataclass

```python
@dataclass
class Batch:
    uids: List[int]              # unique IDs per sequence
    y: mx.array                  # current token IDs, shape (B,)
    logprobs: mx.array           # log-probs, list of B arrays
    max_tokens: List[int]        # max generation length per seq
    num_tokens: List[int]        # tokens generated so far
    cache: List[Any]             # per-layer BatchKVCache objects
    samplers: List[Any]          # per-sequence samplers
    logits_processors: List[Any] # per-sequence logit processors
    tokens: List[mx.array]      # full token history per sequence
```

**Key methods:**

- **`filter(keep_idx)`**: Removes finished sequences. Converts `keep_idx` to `mx.array(keep_idx, mx.int32)`. Indexes `self.y = self.y[keep_idx]` (lazy). Calls `c.filter(keep_idx)` on every cache layer.

- **`extend(other)`**: Merges another Batch. Concatenates `self.y` with `other.y` (lazy). Calls `c.extend(o)` for each cache layer pair.

- **`extract_cache(idx)`**: Returns `[c.extract(idx) for c in self.cache]` — per-layer `KVCache` objects for finished sequences.

### 2.3 BatchKVCache (FP16, Upstream)

```python
class BatchKVCache(_BaseCache):
    step = 256

    def __init__(self, left_padding: List[int]):
        self.keys = None        # shape: (B, n_kv_heads, seq_len, head_dim)
        self.values = None      # shape: same
        self.left_padding = mx.array(left_padding)   # shape: (B,)
        self.offset = mx.array([-l for l in left_padding])  # shape: (B,)
        self._idx = 0           # physical write index (scalar, shared)
```

**Key design:**
- `left_padding`: per-sequence count of leading padding positions
- `offset`: per-sequence logical token count (negative during padded prefill)
- `_idx`: **shared** physical write cursor (scalar int)
- All sequences share the same physical buffer size; shorter sequences have more left-padding

**`update_and_fetch(keys, values)`**: Allocates in multiples of `step=256`. In-place scatter at `[..., prev:self._idx, :]`. Returns full cache up to `_idx`.

**`make_mask(N=1)`**: Always returns an array mask `(B, 1, N, offset+N)` with per-sequence left-padding applied. Unlike single-sequence mode which returns `None` for N=1 decode, batch mode MUST return a mask because each sequence has different padding.

**`filter(batch_indices)`**: Indexes to keep sequences, then **compacts**: if `min_left_pad > 0`, slices off leading padding columns shared by all remaining sequences. `min_left_pad = self.left_padding.min().item()` **forces evaluation** to get a Python scalar.

**`extend(other)`**: Right-justifies both caches (adds left zeros for shorter one), concatenates along batch dimension. After extend: `_idx = max(self._idx, other._idx)`.

**`extract(idx)`**: Slices single sequence, `mx.contiguous()` to create independent copy, sets `offset = keys.shape[2]`.

**`merge(caches)`** (classmethod): Creates `BatchKVCache` from multiple `KVCache` objects. Finds max length, left-pads shorter caches.

### 2.4 KVCache (Single Sequence)

```python
class KVCache(_BaseCache):
    step = 256
    def __init__(self):
        self.keys = None     # (1, n_kv_heads, allocated_len, head_dim)
        self.values = None   # same
        self.offset = 0      # number of valid tokens
```

Standard per-sequence cache. Used outside batch mode and as return type of `extract()`.

### 2.5 QuantizedKVCache (Single Sequence)

```python
class QuantizedKVCache(_BaseCache):
    step = 256
    def __init__(self, group_size=64, bits=8):
        self.keys = None    # tuple of (data, scales, biases) — quantized
        self.values = None  # same
        self.offset = 0
        self.group_size = group_size
        self.bits = bits
```

Stores K/V as quantized tuples. Uses `mx.quantized_matmul()` during attention.

**IMPORTANT: There is NO `BatchQuantizedKVCache` in upstream mlx-lm v0.30.4.** The `_make_cache()` function only handles `KVCache`, `ArraysCache`, `RotatingKVCache`, and `CacheList`. If a model's `make_cache()` returns `QuantizedKVCache`, batch generation raises:

```python
raise ValueError(f"{type(c)} does not yet support batching")
```

**Our `BatchQuantizedKVCache` in `mlx_quantized_extensions.py` is entirely custom code.**

### 2.6 Data Flow: B=2 Decode Step

Setup: `active_batch` has `len(batch) == 2`. Both sequences prefilled. `batch.y` shape `(2,)`.

1. **Read previous tokens**: `y, logprobs = batch.y, batch.logprobs`. Shape: `(2,)`.
2. **Append to token history**: `batch.tokens[i] = mx.concatenate((toks, y[i:i+1]))`.
3. **Reshape for model**: `y[:, None]` → `(2, 1)`.
4. **Model forward pass** (`_step()`):
   - Embedding: `(2, 1)` → `(2, 1, hidden_dim)`
   - `create_attention_mask(h, cache[layer])` → calls `cache.make_mask(N=1)` → array mask `(2, 1, 1, seq_len)` with per-sequence left_padding
   - Per-layer attention: Q/K/V projections, RoPE with per-sequence offsets, `cache.update_and_fetch(keys, values)` appends to buffer, SDPA with mask
   - Final norm + LM head: `(2, 1, vocab_size)` → `(2, vocab_size)` after `[:, -1, :]`
   - Sampling: `(2,)` — one token per sequence
5. **Async eval**: `mx.async_eval(batch.y, batch.logprobs)` — schedules GPU work.
6. **Block on previous**: `y.tolist()` — forces evaluation of previous step's tokens.

### 2.7 Filter Transition (B=2 → B=1)

Scenario: Sequence 0 finishes (stop token), sequence 1 continues.

1. `y = y.tolist()` → `[stop_token, normal_token]`
2. Sequence 0: finished → `end_idx = [0]`; Sequence 1: continues → `keep_idx = [1]`
3. **Extract sequence 0's cache**: `batch.extract_cache(0)`:
   - Per layer: `BatchKVCache.extract(0)` / `BatchQuantizedKVCache.extract(0)`
   - `padding = self.left_padding[0].item()` — forces eval
   - `cache.keys = mx.contiguous(self.keys[0:1, :, padding:self._idx])` — independent copy
   - `mx.eval(*cache.keys, *cache.values)` — materialize (our Q4 version)
4. **Filter batch**: `batch.filter([1])`:
   - `self.y = self.y[mx.array([1])]` — `(2,)` → `(1,)` (lazy)
   - Per layer: `c.filter(mx.array([1]))`
5. **Inside `BatchQuantizedKVCache.filter([1])`**:
   - `self.keys = tuple(k[batch_indices] for k in self.keys)` — lazy indexed views of old B=2 tensors
   - `self.left_padding = self.left_padding[batch_indices]` — lazy
   - `min_pad = self.left_padding.min().item()` — **forces eval of left_padding**
   - If min_pad > 0: slice off leading padding, adjust `_idx`
   - **`mx.eval(*self.keys, *self.values)`** — materialize filtered tensors (our code)
   - **`mx.synchronize()`** — sync `generation_stream` (we are inside `mx.stream(generation_stream)`)
   - **`self._post_filter_evals = 3`** — force eval in next 3 `update_and_fetch()` calls

6. After filter: `active_batch` has B=1. Next decode uses `(1, 1)` input.

**The lazy operation chain**: Without `mx.eval()` after filter, the indexed `k[batch_indices]` tensors are lazy views referencing the old B=2 buffers. When the next forward pass runs with B=1, Metal tries to resolve these stale references with a different batch dimension → crash.

### 2.8 Extend Transition (B=1 → B=2)

Scenario: B=1 decoding, new prompt inserted, next `_next()` adds it.

1. `insert()` adds prompt to `unprocessed_prompts`.
2. In `_next()`: `num_to_add = completion_batch_size - 1` (1 active).
3. If `num_to_add >= prefill_batch_size`, enters prefill loop.
4. **Block on current decode**: `mx.eval(batch.y, batch.logprobs)` — finalize in-flight async decode.
5. **Prefill new prompt**: `_process_prompts(prompts)` → new `Batch` with B=1.
6. **Extend**: `self.active_batch.extend(batch)` → B=1 + B=1 = B=2.
7. **Inside `BatchKVCache.extend(other)`**:
   - Right-justify both caches: shorter cache gets left-padded with zeros
   - `mx.concatenate` along batch dimension (lazy)
   - `_idx = max(self._idx, other._idx)`
   - `left_padding` adjusted for added left padding

### 2.9 Key Shapes During B=2 Decode

| Tensor | Shape | Notes |
|--------|-------|-------|
| `input_tokens` | `(2, 1)` | Batch of 2, seq len 1 |
| Hidden states `h` | `(2, 1, hidden_dim)` | Through all layers |
| Queries `Q` | `(2, n_heads, 1, head_dim)` | After reshape+transpose |
| Keys `K` (from cache) | `(2, n_kv_heads, S, head_dim)` | S = cached length |
| Values `V` (from cache) | `(2, n_kv_heads, S, head_dim)` | Same S |
| Attention mask | `(2, 1, 1, S)` | Per-sequence left_padding |
| Logits | `(2, vocab_size)` | After `[:, -1, :]` |
| Sampled tokens `y` | `(2,)` | One token per sequence |
| `left_padding` | `(2,)` | Per-sequence padding count |
| `offset` | `(2,)` | Per-sequence logical position |
| `_idx` | scalar `int` | Shared physical write cursor |

### 2.10 Critical Design Details

1. **`_idx` is shared, `offset` is per-sequence.** All sequences share the physical write cursor, but each has its own logical offset. Shorter sequences have larger `left_padding`.

2. **`make_mask` always returns an array in batch mode.** The single-sequence optimization of returning `None` for N=1 decode doesn't work in batch mode.

3. **Async eval pipeline:**
   - Step N: `_step()` runs forward pass → `y_n, logprobs_n`
   - `mx.async_eval(y_n, logprobs_n)` — enqueue on GPU
   - Step N+1: `_step()` for next token, GPU still computing step N
   - `y_prev.tolist()` — block until step N ready
   - One step of CPU/GPU overlap.

4. **No thread safety.** `BatchGenerator` has no locks. All calls must be from same thread or externally synchronized.

5. **`filter()` compaction**: When shortest sequence finishes, remaining sequences may all have positive left_padding. Removing shared padding prefix reduces cache size.

6. **`extend()` right-justifies caches**: New (shorter) sequence gets left-padded to align with existing (longer) cache.

---

## 3. Our Code: Thread Map and MLX Operations Audit

### 3.1 Thread Map

The system has four contexts performing MLX operations:

#### Thread A: asyncio event loop (main thread)
- Runs FastAPI/uvicorn server
- Handles HTTP request/response
- Calls `cache_store.save()` / `cache_store.flush_dirty()` from OpenAI adapter after streaming completes (`openai_adapter.py:347-351`)
- Calls `cache_store.save()` after non-streaming generation (`openai_adapter.py:733`)
- `asyncio.to_thread()` pushes `batch_engine.submit()` and `run_step_for_uid()` onto Thread C when no scheduler

#### Thread B: scheduler-worker (daemon thread)
- Created at `scheduler.py:177`: `threading.Thread(target=self._run_loop, daemon=True, name="scheduler-worker")`
- Single-threaded loop: `_accept_requests()` → `_run_decode_step()` → `_process_one_chunk()`
- `_run_decode_step()` → `engine.step_once()` → `batch_gen.next()` (MLX model forward pass)
- `_process_one_chunk()` → `prefill_adapter.process_prefill_chunk()` (MLX model forward pass)
- `_submit_direct()` → `engine.submit()` (reconstructs cache, chunked prefill, batch_gen.insert)

#### Thread C: asyncio thread pool executor (when no scheduler)
- `asyncio.to_thread(batch_engine.submit, ...)` runs `submit()` including `_reconstruct_cache()`, `_chunked_prefill()`, `batch_gen.insert()`
- `asyncio.to_thread(run_step_for_uid, ...)` runs `step()` including `batch_gen.next()`

#### Thread D: SSE streaming response (main event loop context)
- `_stream_via_scheduler()` calls `cache_store.save()` and `cache_store.flush_dirty()` on the asyncio event loop
- Non-scheduler streaming calls `batch_engine.step()` synchronously, blocking the event loop

### 3.2 Per-Function MLX Operation Inventory

#### `BatchQuantizedKVCache.merge()` (mlx_quantized_extensions.py:74)
- **Thread**: Scheduler-worker (called from `batch_gen.insert()` → `_merge_caches()`)
- **MLX ops**: `mx.zeros()` ×6 (allocate buffers), slice assignments ×6 (in-place scatter), `mx.eval()` on all 6 tensors
- **Lazy → eval**: Slice assignments create lazy `SliceUpdate` graphs. `mx.eval()` at line 158 materializes. **CRITICAL — without eval, batch dimension changes crash Metal.**

#### `BatchQuantizedKVCache.update_and_fetch()` (mlx_quantized_extensions.py:214)
- **Thread**: Scheduler-worker (during model forward pass inside `batch_gen.next()`)
- **MLX ops**: `mx.quantize()` ×2, in-place scatter on `self.keys[i]`/`self.values[i]` (6 tensors), conditional `mx.eval()` if `_post_filter_evals > 0`, `tree_map` slice for return value
- **Lazy → eval**: Scatter assignments are lazy. Conditional eval materializes for 3 steps post-filter. Return value `x[..., :self._idx, :]` is lazy — consumed by model attention.
- **Concern**: The `expand_quant` path (line 249) creates lazy concatenated buffers. Subsequent scatter writes into these lazy buffers. Comment says "Do NOT call mx.eval() here." Matches upstream behavior.

#### `BatchQuantizedKVCache.filter()` (mlx_quantized_extensions.py:286)
- **Thread**: Scheduler-worker (during `batch_gen.next()` when sequence completes)
- **MLX ops**: Index slicing ×6, optional sequence-dim slicing ×6, `mx.eval()` on all keys/values, `mx.synchronize()`
- **Lazy → eval**: Index slicing creates lazy views. `mx.eval()` at line 308 materializes. `mx.synchronize()` at line 325 syncs `generation_stream` (we are inside `mx.stream(generation_stream)`).
- **Sets `_post_filter_evals = 3`** for subsequent decode steps.

#### `BatchQuantizedKVCache.extend()` (mlx_quantized_extensions.py:336)
- **Thread**: Scheduler-worker (from `batch_gen.insert()` for staggered arrivals)
- **MLX ops**: Slicing, `mx.concatenate()` ×6, `mx.eval()` on all keys/values, `mx.concatenate()` for offsets/padding
- **Lazy → eval**: Concatenations create lazy tensors. `mx.eval()` at line 391 materializes.

#### `BatchQuantizedKVCache.extract()` (mlx_quantized_extensions.py:406)
- **Thread**: Scheduler-worker (from `batch_gen.next()` → `response.prompt_cache`)
- **MLX ops**: `mx.contiguous()` ×6, `mx.eval()` on all 6 tensors
- **Lazy → eval**: `mx.contiguous()` creates lazy copy. `mx.eval()` at line 433 materializes. Extracted cache is fully independent of batch tensors.

#### `_fused_q4_sdpa()` (mlx_fused_attention.py:115)
- **Thread**: Scheduler-worker (during model forward pass inside `batch_gen.next()`)
- **MLX ops**: Per-sequence: `mx.compile`-wrapped function doing `mx.quantized_matmul` ×2, `mx.softmax`, `mx.where`. For B>1: `mx.concatenate` to recombine.
- **KEY DESIGN**: Always splits batch to B=1 before calling compiled function. Avoids `mx.compile` crashes on batch dimension changes.

#### `MLXPrefillAdapter.process_prefill_chunk()` (mlx_prefill_adapter.py:74)
- **Thread**: Scheduler-worker (from `scheduler._process_one_chunk()`)
- **MLX ops**: `mx.array()`, `model()` (forward pass), `mx.eval(y)`
- **Locks**: Acquires `mlx_io_lock` AND sets `mx.stream(generation_stream)`

#### `SafetensorsCacheAdapter.save()` (safetensors_cache_adapter.py:66)
- **Thread**: Event loop thread (from `cache_store._save_to_disk()`)
- **MLX ops**: `mx.quantize()` (if float data), `mx.save_safetensors()` (disk write)
- **Locks**: Acquires `mlx_io_lock` at line 208
- **Data read**: `block.layer_data["k"]`/`["v"]` tensors from AgentBlocks
- **Concern**: `mx.save_safetensors()` forces evaluation of any lazy tensors. The tensors in `block.layer_data` come from `_extract_cache()` which may store lazy slices.

#### `SafetensorsCacheAdapter.load()` (safetensors_cache_adapter.py:229)
- **Thread**: Scheduler-worker or thread pool executor
- **MLX ops**: `mx.load()` (returns lazy/memory-mapped tensors)
- **Locks**: Acquires `mlx_io_lock` at line 257

#### `BlockPoolBatchEngine.step_once()` (batch_engine.py:1513)
- **Thread**: Scheduler-worker
- **MLX ops**: `batch_gen.next()` (entire model forward pass + cache operations)
- **Locks**: `mlx_io_lock` wraps `batch_gen.next()` at line 1554

#### `BlockPoolBatchEngine._chunked_prefill()` (batch_engine.py:337)
- **Thread**: Scheduler-worker
- **MLX ops**: `mx.array()`, `self._model()` (forward pass), `mx.eval(y)` per chunk, within `mx.stream(generation_stream)`
- **Locks**: **NONE** (no `mlx_io_lock`)
- **CONCERN**: Does NOT acquire `mlx_io_lock`. When called from scheduler-worker, this is safe (single-threaded). When called from `asyncio.to_thread()` (no scheduler), could overlap with other MLX operations.

#### `BlockPoolBatchEngine._reconstruct_cache()` (batch_engine.py:1717)
- **Thread**: Scheduler-worker or thread pool executor
- **MLX ops**: `mx.concatenate()` via `cache_adapter.concatenate_cache_blocks()`, batched `mx.eval()` under `mlx_io_lock`

#### `BlockPoolBatchEngine._extract_cache()` (batch_engine.py:1856)
- **Thread**: Scheduler-worker
- **MLX ops**: Accesses `cache[layer].state`, `mx.quantize()` for float caches, `mx.eval()` for quantization, `slice_cache_tensor()` for chunking
- **Lazy → eval**: `slice_cache_tensor()` creates **lazy slices** stored in `block.layer_data`. These are **NOT evaluated before storage**.
- **Mitigating factor**: For Q4 caches, `extract()` creates contiguous copies first (`mx.contiguous()` + `mx.eval()`), so the slices reference materialized data, not batch cache tensors.

### 3.3 Shared Data Structures Across Threads

| Structure | Accessed by | Protection |
|---|---|---|
| `BatchQuantizedKVCache` (batch KV tensors) | scheduler-worker (via `batch_gen.next()`, `insert()`) | Single scheduler thread; `mlx_io_lock` in `step_once()` |
| `BlockPool.free_list`, `allocated_blocks`, `agent_allocations` | scheduler-worker + event loop | `BlockPool._lock` (threading.Lock) |
| `BatchEngine._active_requests` | scheduler-worker + event loop | `BatchEngine._lock` (threading.RLock) |
| `BatchEngine._agent_blocks` | scheduler-worker + event loop | `BatchEngine._lock` (threading.RLock) |
| `AgentCacheStore._hot_cache`, `_warm_cache` | event loop (save, load) + potentially scheduler via engine | `AgentCacheStore._lock` (threading.RLock) |
| `mlx_io_lock` | scheduler-worker + event loop | Global `threading.Lock` |
| `KVBlock.layer_data` (tensor refs) | scheduler-worker (extract, finalize) + event loop (save) | **Partially protected** — `get_agent_blocks()` takes shallow copy of block refs, but `layer_data` dict references are shared |

### 3.4 Lazy Tensor Creation vs Materialization Map

| Operation | Creates Lazy | Materializes | Location |
|---|---|---|---|
| `merge()` slice assignments | YES | `mx.eval()` at line 158 | mlx_quantized_extensions.py:143-161 |
| `update_and_fetch()` scatter | YES | Conditional `mx.eval()` if `_post_filter_evals > 0` | mlx_quantized_extensions.py:266-274 |
| `update_and_fetch()` return value | YES | By model attention forward pass | mlx_quantized_extensions.py:277 |
| `update_and_fetch()` `expand_quant` | YES | **NOT materialized** (matches upstream) | mlx_quantized_extensions.py:249 |
| `filter()` indexing | YES | `mx.eval()` at line 308 + `mx.synchronize()` | mlx_quantized_extensions.py:291-325 |
| `extend()` concatenations | YES | `mx.eval()` at line 391 | mlx_quantized_extensions.py:388-391 |
| `extract()` `mx.contiguous()` | YES | `mx.eval()` at line 433 | mlx_quantized_extensions.py:418-433 |
| `_chunked_prefill()` model fwd | YES | `mx.eval(y)` per chunk | batch_engine.py:417-418 |
| `process_prefill_chunk()` model fwd | YES | `mx.eval(y)` | mlx_prefill_adapter.py:101-102 |
| `concatenate_cache_blocks()` Q4 | YES | **Deferred** — caller responsible | mlx_cache_adapter.py:78-83 |
| `concatenate_cache_blocks()` float | YES | `mx.eval()` at line 112 | mlx_cache_adapter.py:108-112 |
| `_reconstruct_cache()` batched eval | (collected) | `mx.eval()` under `mlx_io_lock` | batch_engine.py:1846-1848 |
| `_extract_cache()` `slice_cache_tensor()` | YES | **NOT materialized** | batch_engine.py:2010-2011 |
| `_extract_cache()` quantization | YES | `mx.eval()` | batch_engine.py:1920-1937 |
| `safetensors_cache_adapter.save()` | NO | Forces eval of any lazy inputs | safetensors_cache_adapter.py:209 |
| `safetensors_cache_adapter.load()` | YES (memory-mapped) | Deferred until first use | safetensors_cache_adapter.py:258 |

### 3.5 In-Place Modification Map

| Location | Operation | Notes |
|---|---|---|
| `update_and_fetch()` lines 266-267 | `self.keys[i][..., prev:n, :] = q_keys[i]` | In-place scatter. Creates lazy `SliceUpdate`. `_post_filter_evals` forces eval for 3 steps after filter. |
| `update_and_fetch()` lines 246-249 | `self.keys = tree_map(expand_quant, ...)` | Replaces tuple with lazy concatenated tensors. Comment: do NOT eval. |
| `filter()` lines 291-302 | Replaces `self.keys`/`self.values` with indexed versions | Forces eval + synchronize immediately. |
| `extend()` lines 388-395 | Replaces `self.keys`/`self.values` with concatenated versions | Forces eval immediately. |
| `_finalize_sequence()` line 1428 | `block.layer_data = None` | Nulls tensor refs on old blocks. |
| `_extract_cache()` line 2017 | `block.layer_data = {"k": k_chunk, "v": v_chunk}` | Sets lazy tensor slices into block. |
| `submit()` line 684 | `block.layer_data = None` | Clears reconstructed data after loading. |

---

## 4. Gap Analysis: Identified Bugs and Risks

### BUG 1: `_chunked_prefill()` does not acquire `mlx_io_lock` [MEDIUM]

**File**: `batch_engine.py:406-418`

When called from the scheduler-worker thread via `_submit_direct()`, this is safe because the scheduler loop is single-threaded. But when called from the non-scheduler path via `asyncio.to_thread(batch_engine.submit, ...)`, it runs on a thread pool executor **without `mlx_io_lock`** while doing `self._model()` and `mx.eval(y)`.

The prefill adapter (`MLXPrefillAdapter.process_prefill_chunk()`) does acquire `mlx_io_lock`. But `_chunked_prefill()` is a separate, older implementation that predates the adapter.

**Risk**: If two `asyncio.to_thread(submit, ...)` calls overlap (no scheduler), both could run `_chunked_prefill()` simultaneously → concurrent MLX operations → SIGSEGV.

**Mitigating factor**: When scheduler is enabled (the batch=2 mode), all MLX operations go through the scheduler-worker thread. This bug only manifests in the non-scheduler path.

### BUG 2: `mx.synchronize()` comment is misleading [LOW]

**File**: `mlx_quantized_extensions.py:319-324`

The comment says "Synchronize ALL Metal streams" but `mx.synchronize()` only syncs the default stream (which is `generation_stream` inside the `mx.stream()` context). This currently works correctly in practice but would break if pending work existed on a different stream.

**Risk**: Low — no pending work on other streams at this point. But incorrect reasoning could lead to bugs in future changes.

### BUG 3: Lazy tensor slices in `_extract_cache()` stored in `block.layer_data` [LOW-MEDIUM]

**File**: `batch_engine.py:2010-2017`

`slice_cache_tensor()` returns lazy slices of the parent cache tensor. These are stored in `block.layer_data["k"]`/`["v"]`. When `save()` later reads them on a different thread, `mx.save_safetensors()` forces evaluation under `mlx_io_lock`.

**Concern**: The parent tensor (from `response.prompt_cache` → `cache[layer].state`) could theoretically be freed before `save()` runs.

**Mitigating factor**: For Q4 caches (our primary path), `extract()` calls `mx.contiguous()` + `mx.eval()`, making extracted tensors independent of the batch. The slicing in `_extract_cache()` is on these materialized tensors. The lazy slices reference the function-local `state` return value, which persists through the function scope and into `block.layer_data`.

**Risk**: LOW for Q4 path (contiguous copies). MEDIUM for hypothetical float path.

### BUG 4: `get_agent_blocks()` snapshot timing with `layer_data` references [LOW]

**File**: `batch_engine.py:243-288`

`get_agent_blocks()` creates a "deep copy" of blocks but uses `layer_data=block.layer_data` (reference copy). If `_finalize_sequence()` runs on the scheduler-worker and sets `block.layer_data = None` on ORIGINAL blocks, the COPY retains the old reference — this is correct, the snapshot preserves data.

But if `layer_data` contains lazy tensor slices whose parent tensors are freed between snapshot and `save()`, the lazy eval could read freed memory.

**Mitigating factor**: Same as BUG 3 — Q4 path creates independent copies.

### BUG 5: Non-scheduler direct path lacks concurrency protection [HIGH when applicable]

**Files**: `openai_adapter.py:700-716`

When `scheduler is None`, `submit()` and `run_step_for_uid()` are dispatched via `asyncio.to_thread()`. Multiple concurrent HTTP requests can trigger parallel calls, leading to:
- Concurrent `_chunked_prefill()` without `mlx_io_lock`
- Concurrent `batch_gen.insert()` + `batch_gen.next()`

The code logs: "Using direct batch_engine path (no scheduler) -- concurrent requests unsafe."

**Risk**: HIGH if scheduler disabled + multiple concurrent requests. Not applicable in batch=2 mode (requires scheduler).

### DESIGN CONSTRAINT: `_promote_waiting()` blocks promotion during active decodes [PERF]

**File**: `scheduler.py:254-265`

The scheduler comment explains: "Mid-batch insertion (adding a new sequence while another is decoding) corrupts the shared Q4 batch cache state in the engine."

The gate at line 264 (`if self._uid_to_request: return 0`) prevents new requests from being promoted while ANY sequences are decoding. This means:
- Batch utilization is limited — new requests wait for the ENTIRE batch to finish
- B=2 can only happen when both requests arrive before generation starts, OR via the interleaved prefill path (`_promote_to_decode()`)

The interleaved prefill path (`_process_one_chunk()` → `_promote_to_decode()`) bypasses this gate, but only for requests that went through chunked prefill (prompt length > `interleave_threshold`).

---

## 5. Crash Forensics

### Crash Pattern 1: SIGSEGV during B=2→B=1 filter + concurrent save

**Faulthandler output**: Thread 1 at `safetensors_cache_adapter.py:209` (`mx.save_safetensors()`) concurrent with Thread 2 at `batch_engine.py:1840` (`_reconstruct_cache()` → `mx.eval()`).

**Root cause**: Two threads calling MLX operations concurrently. `mlx_io_lock` was not wrapping `batch_gen.next()` at this point. MLX's internal `get_command_encoder()` uses a non-synchronized `unordered_map` → crash.

**Fix applied**: `mlx_io_lock` now wraps `batch_gen.next()` in `step_once()` (line 1554).

### Crash Pattern 2: SIGSEGV at `update_and_fetch()` during B=2 decode

**Faulthandler output**: Thread 1 at `mlx_quantized_extensions.py:274` (scheduler thread, `update_and_fetch()` in model forward pass) concurrent with Thread 2 at `safetensors_cache_adapter.py:209` (event loop thread, `mx.save_safetensors()`).

**Root cause**: Same as Pattern 1 — concurrent MLX ops. Even with `mlx_io_lock` around `_reconstruct_cache()`, the model forward pass inside `batch_gen.next()` wasn't protected.

**Fix applied**: `mlx_io_lock` now wraps the entire `batch_gen.next()` call.

### Crash Pattern 3: System reboot (OOM / Metal crash)

**Symptoms**: Entire machine became unresponsive and rebooted. Occurred when `mlx_io_lock` was added around both `batch_gen.next()` AND `process_prefill_chunk()`, creating potential for Metal command buffer exhaustion or deadlock.

**Hypothesis**: The lock wrapping too broadly caused Metal to accumulate command buffers beyond capacity, or a deadlock between the lock and Metal's internal synchronization caused the GPU to hang, which on Apple Silicon triggers a system watchdog reboot.

**Status**: Needs further investigation. The lock placement was too aggressive. The fundamental question is: can `mlx_io_lock` around `batch_gen.next()` coexist safely with `mlx_io_lock` around `process_prefill_chunk()` on the same thread? Since the scheduler is single-threaded, these should be sequential, not concurrent. The crash may have been caused by something else.

---

## 6. Conclusions and Fix Strategy

### What We Know Works

1. **Single scheduler thread** for all MLX inference — prevents concurrent forward passes.
2. **`mlx_io_lock`** serializes inference (scheduler-worker) vs disk I/O (event loop thread).
3. **`mx.async_eval` → `mx.eval`** patch eliminates async command buffer conflicts.
4. **B=1 split** in fused attention avoids `mx.compile` crashes on batch dimension changes.
5. **`mx.eval()` after merge/filter/extend/extract** materializes lazy tensors, preventing stale graph crashes.
6. **`mx.synchronize()`** after filter syncs `generation_stream`, ensuring command buffers are committed before next decode.

### What Needs Fixing

1. **Misleading `mx.synchronize()` comment** — should say "syncs generation_stream (current default)" not "syncs ALL streams."

2. **`_chunked_prefill()` lacks `mlx_io_lock`** — safe under scheduler (single-threaded), unsafe without scheduler. Should either add the lock or document that the non-scheduler path is unsafe for concurrent requests.

3. **The system reboot crash** needs root-cause analysis. Possible causes:
   - Metal command buffer accumulation under lock contention
   - Interaction between `generation_stream` context and lock held across thread boundaries
   - Wired memory exhaustion from rapid server restart cycles (see Metal GPU Memory Wiring Issue in MEMORY.md)

### Architecture Assessment

The current architecture is **fundamentally sound** when the scheduler is enabled:

- All MLX inference operations route through the single scheduler-worker thread
- `mlx_io_lock` serializes the only cross-thread MLX operations (save/load)
- Lazy tensors are materialized at all transition points (merge, filter, extend, extract)
- The B=1 split strategy avoids `mx.compile` pitfalls

The remaining crash risk comes from:
1. **Thread A (event loop) calling `mx.save_safetensors()` concurrent with Thread B (scheduler) calling `batch_gen.next()`** — both protected by `mlx_io_lock`, so they should not overlap. If crashes still occur, the lock may not be consistently acquired, or there's a code path that bypasses it.
2. **Wired memory exhaustion** from crash cycles — not a code bug, but a Metal runtime limitation.

### Recommended Next Steps

1. Fix the `mx.synchronize()` comment to be accurate.
2. Add `mlx_io_lock` to `_chunked_prefill()` for correctness in the non-scheduler path.
3. Carefully test the batch=2 path with extensive logging to confirm `mlx_io_lock` is actually held during all MLX operations across both threads.
4. Monitor wired memory before testing — reboot if necessary to establish a clean baseline.
5. Test incrementally: single-thread first (confirm batch=2 decode works), then add the save path.
