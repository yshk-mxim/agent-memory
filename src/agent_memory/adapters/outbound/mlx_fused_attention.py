"""Fused Q4 attention monkeypatch for MLX.

Replaces mlx_lm's 3-dispatch Q4 attention path with:
1. mx.compile-wrapped version (fuses element-wise ops: scale, mask, softmax)
2. Custom Metal kernel for L=1 decode (fuses everything into one dispatch)

Apply via apply_fused_attention_patch() after model loading.
"""

import logging
import os
from functools import partial
from typing import Any

logger = logging.getLogger(__name__)

_patched = False
# Metal decode kernel is disabled by default — benchmarks show compiled SDPA is faster
# because mx.quantized_matmul uses tiled GEMM while the custom kernel loops serially.
# Set SEMANTIC_ENABLE_METAL_DECODE=1 to opt in (useful for further kernel development).
_metal_decode_disabled = os.environ.get("SEMANTIC_ENABLE_METAL_DECODE", "") != "1"


def apply_fused_attention_patch() -> bool:
    """Monkeypatch mlx_lm.models.base with fused Q4 attention.

    Safe to call multiple times (idempotent). Returns True if patch applied.
    Set SEMANTIC_DISABLE_FUSED_ATTN=1 to skip patching entirely (baseline benchmark).
    """
    global _patched
    if _patched:
        return True

    if os.environ.get("SEMANTIC_DISABLE_FUSED_ATTN", "") == "1":
        logger.info("Fused attention patch DISABLED by SEMANTIC_DISABLE_FUSED_ATTN")
        _patched = True  # prevent re-entry
        return False

    try:
        import mlx.core as mx
        import mlx_lm.models.base as base_module
    except ImportError:
        logger.debug("mlx/mlx_lm not available, skipping fused attention patch")
        return False

    _original_q4_sdpa = base_module.quantized_scaled_dot_product_attention

    # --- Compiled Q4 SDPA (per-sequence, B=1) ---
    # mx.compile fuses element-wise ops (scale multiply, mask add, softmax).
    # We always call the compiled function with B=1 by splitting the batch,
    # so mx.compile never sees batch dimension changes (which crash natively).
    _compiled_cache: dict[int, Any] = {}

    def _get_compiled_sdpa(n_repeats: int):
        """Get or create compiled SDPA for this GQA repeat count.

        Always called with B=1 inputs — batch is split before calling.
        GQA uses shapeless=False because the reshape reads queries.shape
        (shapeless=True would bake L from the first trace).
        B=1 is constant so retraces only happen on L changes — safe.
        Non-GQA has no reshape so shapeless=True is fine.
        """
        if n_repeats in _compiled_cache:
            return _compiled_cache[n_repeats]

        if n_repeats > 1:
            @partial(mx.compile, shapeless=False)
            def _inner(queries, k0, k1, k2, v0, v1, v2, scale_arr, mask):
                # B is always 1 here — batch was split by caller
                n_q_heads = queries.shape[1]
                L = queries.shape[2]
                D = queries.shape[3]
                n_kv_heads = n_q_heads // n_repeats
                queries = queries * scale_arr
                queries = mx.reshape(queries, (1, n_kv_heads, n_repeats, L, D))
                k0_ = mx.expand_dims(k0, axis=-3)
                k1_ = mx.expand_dims(k1, axis=-3)
                k2_ = mx.expand_dims(k2, axis=-3)
                v0_ = mx.expand_dims(v0, axis=-3)
                v1_ = mx.expand_dims(v1, axis=-3)
                v2_ = mx.expand_dims(v2, axis=-3)
                scores = mx.quantized_matmul(
                    queries, k0_, k1_, k2_, transpose=True, group_size=64, bits=4,
                )
                if mask is not None:
                    if mask.dtype == mx.bool_:
                        scores = mx.where(mask, scores, mx.finfo(scores.dtype).min)
                    else:
                        scores = scores + mask
                scores = mx.softmax(scores, axis=-1, precise=True)
                out = mx.quantized_matmul(
                    scores, v0_, v1_, v2_, transpose=False, group_size=64, bits=4,
                )
                return mx.reshape(out, (1, n_q_heads, L, -1))
        else:
            @partial(mx.compile, shapeless=True)
            def _inner(queries, k0, k1, k2, v0, v1, v2, scale_arr, mask):
                queries = queries * scale_arr
                scores = mx.quantized_matmul(
                    queries, k0, k1, k2, transpose=True, group_size=64, bits=4,
                )
                if mask is not None:
                    if mask.dtype == mx.bool_:
                        scores = mx.where(mask, scores, mx.finfo(scores.dtype).min)
                    else:
                        scores = scores + mask
                scores = mx.softmax(scores, axis=-1, precise=True)
                return mx.quantized_matmul(
                    scores, v0, v1, v2, transpose=False, group_size=64, bits=4,
                )

        _compiled_cache[n_repeats] = _inner
        return _inner

    def _fused_q4_sdpa(queries, q_keys, q_values, scale, mask, group_size=64, bits=4):
        """Drop-in replacement for quantized_scaled_dot_product_attention.

        Splits batch into individual sequences so the compiled function
        always sees B=1. This avoids mx.compile crashes when batch size
        changes mid-decode (e.g. B=2→B=1 after batch.filter()).
        """
        n_kv_heads = q_keys[0].shape[-3]
        n_q_heads = queries.shape[1]
        n_repeats = n_q_heads // n_kv_heads

        # Handle string mask ("causal") before compiled function
        if isinstance(mask, str):
            L = queries.shape[2]
            N = q_keys[0].shape[-2]
            q_indices = mx.arange(N - L, N)
            k_indices = mx.arange(N)
            mask = q_indices[:, None] >= k_indices[None]

        # For non-Q4 or non-group64, fall back to original
        if group_size != 64 or bits != 4:
            return _original_q4_sdpa(
                queries, q_keys, q_values,
                scale=scale, mask=mask,
                group_size=group_size, bits=bits,
            )

        B = queries.shape[0]
        compiled_fn = _get_compiled_sdpa(n_repeats)

        if B == 1:
            # Single sequence — call compiled function directly
            k0, k1, k2 = q_keys
            v0, v1, v2 = q_values
            scale_arr = mx.array(scale, dtype=queries.dtype)
            return compiled_fn(queries, k0, k1, k2, v0, v1, v2, scale_arr, mask)

        # Batch>1: split into per-sequence calls, then stack.
        # Each compiled call sees B=1 — no batch dim changes ever.
        scale_arr = mx.array(scale, dtype=queries.dtype)
        parts = []
        for i in range(B):
            q_i = queries[i:i+1]
            k_i = tuple(k[i:i+1] for k in q_keys)
            v_i = tuple(v[i:i+1] for v in q_values)
            m_i = mask[i:i+1] if mask is not None else None
            parts.append(compiled_fn(
                q_i, k_i[0], k_i[1], k_i[2],
                v_i[0], v_i[1], v_i[2], scale_arr, m_i,
            ))
        return mx.concatenate(parts, axis=0)

    # --- Metal kernel for L=1 decode ---
    _decode_kernel_cache: dict[tuple, Any] = {}

    def _get_or_create_decode_kernel(
        n_q_heads: int, n_kv_heads: int, k_dim: int, v_dim: int,
    ):
        """Create (or retrieve cached) Metal kernel for this model geometry."""
        key = (n_q_heads, n_kv_heads, k_dim, v_dim)
        if key in _decode_kernel_cache:
            return _decode_kernel_cache[key]

        n_repeats = n_q_heads // n_kv_heads
        threads = 32
        k_packed = k_dim // 8  # uint32 words per K row (4-bit, 8 elements per word)
        v_packed = v_dim // 8
        k_groups = k_dim // 64  # scale/bias groups per K row (group_size=64)
        v_groups = v_dim // 64

        # Dimension-parallel: each thread handles K_DIM/32 and V_DIM/32 dims
        if k_dim % threads != 0 or v_dim % threads != 0:
            logger.warning(
                "Fused decode kernel requires k_dim=%d and v_dim=%d "
                "divisible by %d — skipping",
                k_dim, v_dim, threads,
            )
            _decode_kernel_cache[key] = None
            return None

        dims_per_thread_k = k_dim // threads
        dims_per_thread_v = v_dim // threads

        # Threadgroup memory: shared_q[K_DIM] only
        tg_bytes = k_dim * 4
        if tg_bytes > 32768:
            logger.warning(
                "Fused decode kernel needs %d bytes threadgroup memory "
                "(limit 32768) for k_dim=%d — skipping",
                tg_bytes, k_dim,
            )
            _decode_kernel_cache[key] = None
            return None

        # Header with constants baked for this model
        header = f"""
constant int K_DIM = {k_dim};
constant int V_DIM = {v_dim};
constant int K_PACKED = {k_packed};
constant int V_PACKED = {v_packed};
constant int K_GROUPS = {k_groups};
constant int V_GROUPS = {v_groups};
constant int N_Q_HEADS = {n_q_heads};
constant int N_KV_HEADS = {n_kv_heads};
constant int N_REPEATS = {n_repeats};
constant int THREADS = 32;
constant int DIMS_PER_THREAD_K = {dims_per_thread_k};
constant int DIMS_PER_THREAD_V = {dims_per_thread_v};

// Dequantize one Q4 element from packed array
// Templated on S to handle both float16 (half) and bfloat16 scales/biases
template <typename S>
inline float dequant4(
    const device uint* weights,
    const device S* scales,
    const device S* biases,
    int row_offset_w,  // row * packed_dim
    int row_offset_s,  // row * groups_dim
    int col            // element index within row
) {{
    int packed_idx = col / 8;
    int bit_offset = (col % 8) * 4;
    uint raw = (weights[row_offset_w + packed_idx] >> bit_offset) & 0xFu;
    int grp = col / 64;
    return float(raw) * float(scales[row_offset_s + grp])
         + float(biases[row_offset_s + grp]);
}}
"""
        # Kernel body: dimension-parallel fused decode attention
        # Instead of parallelizing over positions (which requires reducing V_DIM
        # floats across threads — problematic with V_DIM=256 register pressure),
        # we parallelize over dimensions: each thread handles K_DIM/32 K-dims
        # and V_DIM/32 V-dims, processing ALL positions cooperatively.
        # Only cross-thread op: simd_sum of a single float for dot products.
        # Grid: (n_q_heads, B, 1), Threadgroup: (THREADS, 1, 1)
        source = """
    uint tid = thread_position_in_threadgroup.x;
    uint head_idx = threadgroup_position_in_grid.x;
    uint batch_idx = threadgroup_position_in_grid.y;

    int N = int(params[0]);
    float scale = float(scale_in[0]);
    uint kv_head = head_idx / N_REPEATS;

    // Load Q into shared memory (all threads cooperate)
    threadgroup float shared_q[K_DIM];
    int q_base = (batch_idx * N_Q_HEADS + head_idx) * K_DIM;
    for (int i = tid; i < K_DIM; i += THREADS) {
        shared_q[i] = float(queries[q_base + i]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Each thread owns a slice of K and V dimensions
    int k_start = tid * DIMS_PER_THREAD_K;
    int v_start = tid * DIMS_PER_THREAD_V;

    int kv_base = (batch_idx * N_KV_HEADS + kv_head) * N;

    // Online softmax + V accumulation (dimension-parallel)
    // All threads compute the SAME attention score via simd_sum,
    // but each accumulates only its DIMS_PER_THREAD_V dimensions of V.
    float current_max = -INFINITY;
    float current_sum = 0.0f;
    float my_out[DIMS_PER_THREAD_V];
    for (int i = 0; i < DIMS_PER_THREAD_V; i++) my_out[i] = 0.0f;

    for (int pos = 0; pos < N; pos++) {
        // Partial dot product: each thread handles its K-dim slice
        float partial_dot = 0.0f;
        int k_row_w = (kv_base + pos) * K_PACKED;
        int k_row_s = (kv_base + pos) * K_GROUPS;
        for (int d = 0; d < DIMS_PER_THREAD_K; d++) {
            partial_dot += shared_q[k_start + d]
                         * dequant4(k_w, k_s, k_b, k_row_w, k_row_s, k_start + d);
        }
        // simd_sum: all threads get the full dot product (single float)
        float dot = simd_sum(partial_dot) * scale;

        // Online softmax update (identical across all threads)
        float m_new = max(current_max, dot);
        float correction = exp(current_max - m_new);
        float p = exp(dot - m_new);

        // Accumulate weighted V for this thread's dimensions only
        int v_row_w = (kv_base + pos) * V_PACKED;
        int v_row_s = (kv_base + pos) * V_GROUPS;
        for (int i = 0; i < DIMS_PER_THREAD_V; i++) {
            float v_val = dequant4(v_w, v_s, v_b, v_row_w, v_row_s, v_start + i);
            my_out[i] = correction * my_out[i] + p * v_val;
        }
        current_sum = correction * current_sum + p;
        current_max = m_new;
    }

    // Normalize and write (each thread writes its own V-dim slice)
    float inv_sum = 1.0f / current_sum;
    int out_base = (batch_idx * N_Q_HEADS + head_idx) * V_DIM;
    for (int i = 0; i < DIMS_PER_THREAD_V; i++) {
        output[out_base + v_start + i] = half(my_out[i] * inv_sum);
    }
"""
        try:
            kernel = mx.fast.metal_kernel(
                name=f"fused_q4_decode_{k_dim}_{v_dim}_{n_kv_heads}",
                input_names=["queries", "k_w", "k_s", "k_b",
                             "v_w", "v_s", "v_b", "params", "scale_in"],
                output_names=["output"],
                source=source,
                header=header,
                ensure_row_contiguous=True,
            )
            _decode_kernel_cache[key] = kernel
            logger.info(
                "Created fused Q4 decode kernel: k_dim=%d, v_dim=%d, heads=%d/%d",
                k_dim, v_dim, n_q_heads, n_kv_heads,
            )
            return kernel
        except Exception as e:
            logger.warning("Failed to create Metal decode kernel: %s", e)
            _decode_kernel_cache[key] = None
            return None

    def _try_metal_decode(queries, q_keys, q_values, scale, mask, group_size, bits):
        """Try to use the fused Metal kernel for L=1 decode. Returns None if not applicable."""
        B, n_q_heads, L, k_dim = queries.shape
        if L != 1 or bits != 4 or group_size != 64:
            return None

        # Metal kernel hardcodes float16 I/O; bfloat16 queries would be
        # silently truncated (different exponent/mantissa layout).  Fall back
        # to the compiled SDPA path which handles any dtype.
        if queries.dtype == mx.bfloat16:
            return None

        n_kv_heads = q_keys[0].shape[-3]
        N = q_keys[0].shape[-2]
        v_dim_packed = q_values[0].shape[-1]
        v_dim = v_dim_packed * 8  # unpack: 8 elements per uint32

        if N < 1:  # empty cache
            return None

        # k_dim from queries.shape is already the actual head dim (e.g. 192),
        # not the packed dim. Pass directly to kernel creation.
        kernel = _get_or_create_decode_kernel(n_q_heads, n_kv_heads, k_dim, v_dim)
        if kernel is None:
            return None

        orig_dtype = queries.dtype

        # Flatten inputs for the kernel (remove L=1 dimension)
        # queries: [B, n_q_heads, 1, D] -> [B * n_q_heads * D]
        q_flat = queries.reshape(-1).astype(mx.float16)

        # K/V cache arrays: [B, n_kv_heads, N, packed_dim]
        # Flatten to contiguous: [B * n_kv_heads * N * packed_dim]
        k_w = q_keys[0].reshape(-1)
        k_s = q_keys[1].reshape(-1)
        k_b = q_keys[2].reshape(-1)
        v_w = q_values[0].reshape(-1)
        v_s = q_values[1].reshape(-1)
        v_b = q_values[2].reshape(-1)

        params = mx.array([N], dtype=mx.int32)
        scale_in = mx.array([scale], dtype=mx.float32)

        output_shape = (B * n_q_heads * v_dim,)

        try:
            # mx.fast.metal_kernel uses dispatch_threads: grid = total threads
            # (not threadgroup count). Multiply by 32 so each head gets a full
            # 32-thread SIMD group, with threadgroup_position_in_grid.x = head_idx.
            results = kernel(
                inputs=[q_flat, k_w, k_s, k_b, v_w, v_s, v_b, params, scale_in],
                grid=(n_q_heads * 32, B, 1),
                threadgroup=(32, 1, 1),
                output_shapes=[output_shape],
                output_dtypes=[mx.float16],
            )
            # Reshape back to [B, n_q_heads, 1, v_dim] and cast to original dtype
            out = results[0].reshape(B, n_q_heads, 1, v_dim)
            return out.astype(orig_dtype) if orig_dtype != mx.float16 else out
        except Exception as e:
            logger.warning("Metal decode kernel failed: %s, falling back", e)
            return None

    # --- Combined dispatch ---

    def _patched_q4_sdpa(queries, q_keys, q_values, scale, mask, group_size=64, bits=4):
        """Fused Q4 SDPA: Metal kernel for decode, compiled for prefill."""
        if not _metal_decode_disabled:
            result = _try_metal_decode(
                queries, q_keys, q_values, scale, mask, group_size, bits,
            )
            if result is not None:
                return result
        return _fused_q4_sdpa(
            queries, q_keys, q_values, scale, mask, group_size, bits,
        )

    # Apply the SDPA monkeypatch
    base_module.quantized_scaled_dot_product_attention = _patched_q4_sdpa

    # Patch clip_residual in Gemma3 to remove mx.compile.
    # clip_residual is @partial(mx.compile, shapeless=True) and called 52 times
    # per forward pass (2x per layer, 26 layers). When batch.filter() changes
    # B from 2→1 mid-decode, the compiled Metal kernel crashes accessing the
    # nonexistent batch slot. For bfloat16 (Gemma3 Q4), clip_residual is just
    # x + y — compilation has no benefit. For float16, it's a simple clip+cast.
    try:
        import mlx_lm.models.gemma3_text as gemma3_text_module

        def _uncompiled_clip_residual(x, y):
            if x.dtype != mx.float16:
                return x + y
            bound = mx.finfo(mx.float16).max
            return mx.clip(
                x.astype(mx.float32) + y.astype(mx.float32), -bound, bound,
            ).astype(mx.float16)

        gemma3_text_module.clip_residual = _uncompiled_clip_residual
        logger.info("Patched gemma3_text.clip_residual (removed mx.compile for batch compat)")
    except (ImportError, AttributeError):
        logger.debug("gemma3_text not available, skipping clip_residual patch")

    # Replace mx.async_eval with synchronous mx.eval globally.
    # mlx_lm's BatchGenerator._next() calls mx.async_eval(batch.y) then
    # immediately calls extract_cache() and filter(). Any mx.eval on tensors
    # in the async pipeline triggers Metal assertion "Completed handler
    # provided after commit call". Making eval synchronous ensures the
    # command buffer is fully complete before extract/filter.
    # Performance impact: <3% — we lose ~0.3ms Python/GPU overlap per token.
    mx.async_eval = mx.eval
    logger.info("Patched mx.async_eval → mx.eval (sync decode for batch compat)")

    _patched = True
    if _metal_decode_disabled:
        logger.info("Applied fused Q4 attention monkeypatch (compile only, Metal decode disabled)")
    else:
        logger.info("Applied fused Q4 attention monkeypatch (compile + Metal decode)")
    return True
