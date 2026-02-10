# MLX Gotchas for Contributors

This document covers the non-obvious behaviors of the MLX framework that affect this codebase. Read this before modifying any code in `adapters/outbound/mlx_*.py` or `application/batch_engine.py`.

## MLX Is NOT Thread-Safe

MLX has no internal synchronization. Concurrent calls from multiple threads cause data races and Metal assertion failures.

**Upstream references:**
- [Issue #2067](https://github.com/ml-explore/mlx/issues/2067): MLX uses `unordered_map` without synchronization
- [Issue #2133](https://github.com/ml-explore/mlx/issues/2133): Thread-safety discussion
- [Issue #3078](https://github.com/ml-explore/mlx/issues/3078): Metal assertion "Scheduled handler after commit" with concurrent eval

**How we handle this:**
- When the scheduler is enabled, all MLX inference runs on a single thread
- Cross-thread I/O (e.g., saving caches to disk while inference runs) is guarded by `mlx_io_lock` (an `RLock`)
- `submit()` in the batch engine acquires `mlx_io_lock` around the MLX-intensive section
- `_chunked_prefill()` also acquires `mlx_io_lock`

**Rule:** Never call MLX operations from multiple threads concurrently. If you must cross thread boundaries, acquire `mlx_io_lock`.

## mx.synchronize() Syncs Only the Current Default Stream

`mx.synchronize()` does **not** sync all Metal streams. It syncs only the current default stream for the calling thread.

This matters because `mlx-lm` runs decode in a separate `generation_stream`. After `mx.eval()`, tensors may still be pending on the generation stream. The fix:

```python
# WRONG: only syncs current stream
mx.eval(tensor)

# RIGHT: syncs all streams including generation_stream
mx.eval(tensor)
mx.synchronize()
```

We use `mx.synchronize()` in `filter()` to prevent the Metal assertion "commit command buffer with uncommitted encoder." Two distinct Metal assertions exist:

1. **"Completed handler after commit"** -- fixed by replacing `mx.async_eval` with `mx.eval` globally
2. **"commit command buffer with uncommitted encoder"** -- fixed by adding `mx.synchronize()` after eval

## In-Place Assignment Creates Lazy SliceUpdate, Not True Mutation

```python
a[i] = b  # Does NOT mutate a in-place
```

In MLX, `a[i] = b` creates a new lazy `SliceUpdate` node in the computation graph. The original array `a` is not modified. This is fundamentally different from NumPy.

**Consequence:** After slice assignments (e.g., in cache merge/extend operations), you must call `mx.eval()` to materialize the result. Without eval, the lazy graph grows unboundedly.

## MLX Slicing Creates Copies, Not Views

```python
b = a[0:10]  # b is a COPY, not a view of a
```

Unlike NumPy, MLX slicing always creates a copy (a lazy `Slice` node). Modifying `b` will never affect `a`. This means:

- Extracting a sub-tensor from a cache always allocates new memory when evaluated
- You cannot use slice references to share memory between tensors

## mx.compile(shapeless=True) Pitfalls

`mx.compile(shapeless=True)` traces a function once and reuses the compiled graph for any input shape. This avoids recompilation but has a critical limitation:

**Shape-dependent operations bake in the initial shape.** If your compiled function contains ops that depend on tensor dimensions (like reshaping to a specific size), the first call's shapes become permanent.

**Upstream reference:** [Issue #2607](https://github.com/ml-explore/mlx/issues/2607) -- shapeless matmul hardcodes initial dimension.

**Our workaround: B=1 split strategy.** See below.

## B=1 Split Strategy for Batch Decode

When batch size > 1, we do NOT pass the full batch to a compiled function. Instead:

1. Split the batch into individual B=1 inputs
2. Run the compiled function on each B=1 input
3. Concatenate the results

This avoids the `mx.compile(shapeless=True)` crash that occurs when batch dimension changes after the first trace. Both GQA (Gemma) and non-GQA (DeepSeek) paths use this strategy.

**GQA mask broadcast fix:** The fused Q4 attention GQA path reshapes Q to 5D `(B, n_kv_heads, n_repeats, L, D)`. The mask from the model is 4D `(B, 1, L, S)`, which cannot broadcast against 5D when B > 1. Fix: `mx.expand_dims(mask, axis=2)`.

With B=1 split, the compiled function always sees B=1, so the mask broadcast issue is avoided. The fix is still present as a safety measure.

## mx.eval() After merge/filter/extend/extract

Every operation that builds a lazy computation graph must be materialized before the tensors are used across operation boundaries. The following operations in the batch engine and cache store require `mx.eval()` immediately after:

| Operation | Why eval is needed |
|---|---|
| `merge()` | Concatenation of KV tensors creates lazy graph |
| `filter()` | Slice assignments for removing completed sequences |
| `extend()` | Appending new KV entries to cache |
| `_extract_cache()` | Slicing out a subsequence of cached KV data |

Without `mx.eval()`, the lazy graph accumulates, causing:
- Unbounded memory growth
- Metal assertion failures when the graph is eventually evaluated in a different stream context

Always pair these operations with `mx.eval()` and, where cross-stream safety is needed, `mx.synchronize()`.

## Metal GPU Memory Wiring Issue

MLX allocates Metal GPU buffers that the OS is slow to reclaim from killed processes. Repeated crash cycles (especially `kill -9`) accumulate **wired kernel memory** that is only fully reclaimable via system reboot.

On a 24 GB M4 Pro, wired memory has been observed reaching 17-19 GB (70-80%) after repeated crash/restart cycles, leaving insufficient memory for model loading.

**Impact:**
- Gemma 3 12B Q4 (~6.5 GB model weight) needs ~8-10 GB free to load and run inference
- Failed Metal allocations produce cryptic errors, not clear OOM messages
- Zombie python processes holding GPU buffers make it worse

**Prevention:**
- Always use graceful shutdown (`kill -TERM`, not `kill -9`). See [debugging.md](debugging.md) for the full shutdown procedure.
- Allow 5-10 seconds between stop and start for Metal to reclaim buffers
- Monitor wired memory between development cycles: `memory_pressure | head -5`

**Recovery:** If wired memory is critically high, the only fix is a system reboot.

## Native MLX Safetensors I/O

The project uses native MLX I/O instead of the `safetensors.numpy` path:

```python
# Save -- auto-appends .safetensors to path (do NOT include the extension)
mx.save_safetensors("path/to/file", tensors_dict, metadata_dict)

# Load -- returns dict[str, mx.array]
tensors = mx.load("path/to/file.safetensors")

# Load with metadata -- returns (dict, dict), NOT a dict with __metadata__ key
tensors, metadata = mx.load("path/to/file.safetensors", return_metadata=True)
```

Q4 packed format stores `uint32` arrays where each element holds 8 x 4-bit values. A head_dim=256 becomes 32 `uint32` values. `mx.dequantize()` unpacks at inference time. Bfloat16 scales and biases (used by Gemma 3) are preserved natively without needing `ml_dtypes` workarounds.
