"""Fused Q4 attention monkeypatch for MLX.

Replaces mlx_lm's 3-dispatch Q4 attention path with:
1. mx.compile-wrapped version (fuses element-wise ops: scale, mask, softmax)
2. Custom Metal kernel for L=1 decode (fuses everything into one dispatch)

Apply via apply_fused_attention_patch() after model loading.
"""

import logging
from functools import partial
from typing import Any

logger = logging.getLogger(__name__)

_patched = False


def apply_fused_attention_patch() -> bool:
    """Monkeypatch mlx_lm.models.base with fused Q4 attention.

    Safe to call multiple times (idempotent). Returns True if patch applied.
    """
    global _patched
    if _patched:
        return True

    try:
        import mlx.core as mx
        import mlx_lm.models.base as base_module
    except ImportError:
        logger.debug("mlx/mlx_lm not available, skipping fused attention patch")
        return False

    _original_q4_sdpa = base_module.quantized_scaled_dot_product_attention

    # --- Compiled Q4 SDPA (general case) ---
    # mx.compile fuses element-wise ops (scale multiply, mask add, softmax)
    # reducing kernel launches from ~5 to ~3 per layer. shapeless=True avoids
    # recompilation when sequence length changes between prefill chunks.
    # We create separate compiled functions per n_repeats value since
    # mx.compile can't trace through data-dependent control flow.
    _compiled_cache: dict[int, Any] = {}

    def _get_compiled_sdpa(n_repeats: int):
        """Get or create compiled SDPA for this GQA repeat count."""
        if n_repeats in _compiled_cache:
            return _compiled_cache[n_repeats]

        if n_repeats > 1:
            @partial(mx.compile, shapeless=True)
            def _inner(queries, k0, k1, k2, v0, v1, v2, scale_arr, mask):
                B, n_q_heads, L, D = queries.shape
                n_kv_heads = n_q_heads // n_repeats
                queries = queries * scale_arr
                queries = mx.reshape(queries, (B, n_kv_heads, n_repeats, L, D))
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
                return mx.reshape(out, (B, n_q_heads, L, -1))
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
        """Drop-in replacement for quantized_scaled_dot_product_attention."""
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

        # Unpack tuples for mx.compile tracing
        k0, k1, k2 = q_keys
        v0, v1, v2 = q_values
        scale_arr = mx.array(scale, dtype=queries.dtype)

        compiled_fn = _get_compiled_sdpa(n_repeats)
        return compiled_fn(queries, k0, k1, k2, v0, v1, v2, scale_arr, mask)

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

        # Threadgroup memory budget check (32KB limit on Apple Silicon)
        # shared_q[K_DIM] + tg_max[THREADS] + tg_sum[THREADS] + tg_out[THREADS*V_DIM]
        tg_bytes = (k_dim + threads + threads + threads * v_dim) * 4
        if tg_bytes > 32768:
            logger.warning(
                "Fused decode kernel needs %d bytes threadgroup memory "
                "(limit 32768) for k_dim=%d, v_dim=%d — skipping",
                tg_bytes, k_dim, v_dim,
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

// Dequantize one Q4 element from packed array
inline float dequant4(
    const device uint* weights,
    const device half* scales,
    const device half* biases,
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
        # Kernel body: online softmax fused decode attention
        # Grid: (n_q_heads, B, 1), Threadgroup: (THREADS, 1, 1)
        # params[0] = N (sequence length, changes each call)
        source = """
    uint tid = thread_position_in_threadgroup.x;
    uint head_idx = threadgroup_position_in_grid.x;
    uint batch_idx = threadgroup_position_in_grid.y;

    // Read N from params
    int N = int(params[0]);
    float scale = float(scale_in[0]);

    uint kv_head = head_idx / N_REPEATS;

    // Shared memory for Q vector and reduction
    threadgroup float shared_q[K_DIM];
    threadgroup float tg_max[THREADS];
    threadgroup float tg_sum[THREADS];
    // Per-thread V accumulator for cross-thread reduction
    // Memory: THREADS * V_DIM * 4 bytes (e.g. 32*128*4 = 16KB)
    threadgroup float tg_out[THREADS * V_DIM];

    // Load Q into shared memory
    int q_base = (batch_idx * N_Q_HEADS + head_idx) * K_DIM;
    for (int i = tid; i < K_DIM; i += THREADS) {
        shared_q[i] = float(queries[q_base + i]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // KV base offsets for this head
    int kv_base = (batch_idx * N_KV_HEADS + kv_head) * N;

    // Phase: Online softmax + V accumulation
    float my_max = -INFINITY;
    float my_sum = 0.0f;
    float my_out[V_DIM];  // in registers
    for (int d = 0; d < V_DIM; d++) my_out[d] = 0.0f;

    for (int pos = tid; pos < N; pos += THREADS) {
        // Compute Q . K[pos] via dequantized dot product
        float dot = 0.0f;
        int k_row_w = (kv_base + pos) * K_PACKED;
        int k_row_s = (kv_base + pos) * K_GROUPS;
        for (int d = 0; d < K_DIM; d++) {
            dot += shared_q[d] * dequant4(k_w, k_s, k_b, k_row_w, k_row_s, d);
        }
        dot *= scale;

        // Online softmax update
        float m_new = max(my_max, dot);
        float correction = exp(my_max - m_new);
        float p = exp(dot - m_new);

        // Update V accumulator with correction
        int v_row_w = (kv_base + pos) * V_PACKED;
        int v_row_s = (kv_base + pos) * V_GROUPS;
        for (int d = 0; d < V_DIM; d++) {
            float v_val = dequant4(v_w, v_s, v_b, v_row_w, v_row_s, d);
            my_out[d] = correction * my_out[d] + p * v_val;
        }
        my_sum = correction * my_sum + p;
        my_max = m_new;
    }

    // Store per-thread results in shared memory for cross-thread reduction
    tg_max[tid] = my_max;
    tg_sum[tid] = my_sum;
    for (int d = 0; d < V_DIM; d++) {
        tg_out[tid * V_DIM + d] = my_out[d];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Reduce: merge online softmax states across threads
    // Thread 0 does the final reduction (simple for THREADS=32)
    if (tid == 0) {
        // Find global max
        float global_max = tg_max[0];
        for (int t = 1; t < THREADS && t < N; t++) {
            global_max = max(global_max, tg_max[t]);
        }

        // Compute corrected sum and output
        float global_sum = 0.0f;
        float final_out[V_DIM];
        for (int d = 0; d < V_DIM; d++) final_out[d] = 0.0f;

        int active_threads = min(THREADS, N);
        for (int t = 0; t < active_threads; t++) {
            float corr = exp(tg_max[t] - global_max);
            float w = corr * tg_sum[t];
            global_sum += w;
            for (int d = 0; d < V_DIM; d++) {
                final_out[d] += corr * tg_out[t * V_DIM + d];
            }
        }

        // Normalize and write output
        int out_base = (batch_idx * N_Q_HEADS + head_idx) * V_DIM;
        float inv_sum = 1.0f / global_sum;
        for (int d = 0; d < V_DIM; d++) {
            output[out_base + d] = half(final_out[d] * inv_sum);
        }
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
            results = kernel(
                inputs=[q_flat, k_w, k_s, k_b, v_w, v_s, v_b, params, scale_in],
                grid=(n_q_heads, B, 1),
                threadgroup=(32, 1, 1),
                output_shapes=[output_shape],
                output_dtypes=[mx.float16],
            )
            # Reshape back to [B, n_q_heads, 1, v_dim]
            return results[0].reshape(B, n_q_heads, 1, v_dim)
        except Exception as e:
            logger.warning("Metal decode kernel failed: %s, falling back", e)
            return None

    # --- Combined dispatch ---

    def _patched_q4_sdpa(queries, q_keys, q_values, scale, mask, group_size=64, bits=4):
        """Fused Q4 SDPA: Metal kernel for decode, compiled for prefill."""
        B, n_q_heads, L, D = queries.shape

        # Try Metal kernel for L=1 decode (single fused dispatch)
        if L == 1:
            result = _try_metal_decode(
                queries, q_keys, q_values, scale, mask, group_size, bits,
            )
            if result is not None:
                return result

        # Fall back to compiled Q4 SDPA (fuses element-wise ops)
        return _fused_q4_sdpa(
            queries, q_keys, q_values, scale, mask, group_size, bits,
        )

    # Apply the monkeypatch
    base_module.quantized_scaled_dot_product_attention = _patched_q4_sdpa
    _patched = True
    logger.info("Applied fused Q4 attention monkeypatch (compile + Metal decode)")
    return True
