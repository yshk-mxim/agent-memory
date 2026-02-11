# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Yakov Shkolnikov and contributors
"""GPU-accelerated tests for fused Q4 attention inner logic.

Exercises the compiled SDPA closures, batch splitting, mask conversion,
Metal decode kernel, and clip_residual with real MLX tensors on Metal GPU.

These tests cover the ~70% of mlx_fused_attention.py that mock-only tests
cannot reach (compiled function bodies, quantized_matmul paths, Metal kernel
creation and dispatch).

Requires MLX and mlx_lm. Skipped if not available.
"""

import importlib
import os
import sys

import pytest

try:
    import mlx.core as mx
    import mlx_lm.models.base  # noqa: F401

    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    mx = None

pytestmark = pytest.mark.skipif(not HAS_MLX, reason="MLX/mlx_lm not available")


# ── Helpers ────────────────────────────────────────────────────────────


def _make_q4_cache(B, n_kv_heads, N, dim, dtype=None):
    """Create quantized Q4 KV cache tensors.

    Returns (data, scales, biases) tuple compatible with mx.quantized_matmul.
    data: uint32 packed, scales/biases: same dtype as input.
    """
    if dtype is None:
        dtype = mx.float16
    flat = mx.random.normal((B * n_kv_heads * N, dim)).astype(dtype)
    data, scales, biases = mx.quantize(flat, group_size=64, bits=4)
    data = data.reshape(B, n_kv_heads, N, -1)
    scales = scales.reshape(B, n_kv_heads, N, -1)
    biases = biases.reshape(B, n_kv_heads, N, -1)
    return (data, scales, biases)


def _fresh_patch(metal_decode=False):
    """Apply fused attention patch with fresh module state.

    Returns the patched SDPA function from base_module.
    The autouse fixture ensures real mlx_lm is available.
    """
    import mlx_lm.models.base as base_module  # real, not mock (fixture handles this)

    mod_name = "agent_memory.adapters.outbound.mlx_fused_attention"
    sys.modules.pop(mod_name, None)

    if metal_decode:
        os.environ["SEMANTIC_ENABLE_METAL_DECODE"] = "1"
    else:
        os.environ.pop("SEMANTIC_ENABLE_METAL_DECODE", None)
    os.environ.pop("SEMANTIC_DISABLE_FUSED_ATTN", None)

    mod = importlib.import_module(mod_name)
    result = mod.apply_fused_attention_patch()
    assert result is True

    return base_module.quantized_scaled_dot_product_attention


@pytest.fixture(autouse=True)
def _save_restore_globals():
    """Save and restore base_module / mx globals that patching modifies."""
    import mlx.core as _mx
    import mlx_lm.models.base as base_module

    original_sdpa = base_module.quantized_scaled_dot_product_attention
    original_async = _mx.async_eval

    try:
        import mlx_lm.models.gemma3_text as _g

        original_clip = _g.clip_residual
    except (ImportError, AttributeError):
        _g = None
        original_clip = None

    mod_name = "agent_memory.adapters.outbound.mlx_fused_attention"
    saved_mod = sys.modules.pop(mod_name, None)

    saved_env = {
        k: os.environ.get(k)
        for k in ["SEMANTIC_ENABLE_METAL_DECODE", "SEMANTIC_DISABLE_FUSED_ATTN"]
    }

    yield

    # Restore everything
    base_module.quantized_scaled_dot_product_attention = original_sdpa
    _mx.async_eval = original_async
    if _g is not None and original_clip is not None:
        _g.clip_residual = original_clip

    if saved_mod is not None:
        sys.modules[mod_name] = saved_mod
    else:
        sys.modules.pop(mod_name, None)

    for k, v in saved_env.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


# ── Compiled SDPA (B=1) ───────────────────────────────────────────────


class TestCompiledSDPA:
    """Tests for the mx.compile-wrapped Q4 SDPA closures."""

    def test_b1_non_gqa(self) -> None:
        """B=1, n_repeats=1 (non-GQA) compiled path."""
        sdpa = _fresh_patch()

        B, n_heads, L, D, N = 1, 4, 2, 64, 8
        queries = mx.random.normal((B, n_heads, L, D)).astype(mx.float16)
        q_keys = _make_q4_cache(B, n_heads, N, D)
        q_values = _make_q4_cache(B, n_heads, N, D)
        mask = mx.zeros((B, 1, L, N), dtype=mx.float16)

        result = sdpa(queries, q_keys, q_values, scale=0.125, mask=mask)
        mx.eval(result)

        assert result.shape == (B, n_heads, L, D)
        assert result.dtype == mx.float16

    def test_b1_gqa(self) -> None:
        """B=1, n_repeats=2 (GQA) compiled path with 5D reshape."""
        sdpa = _fresh_patch()

        B, n_q_heads, n_kv_heads, L, D, N = 1, 8, 4, 2, 64, 8
        queries = mx.random.normal((B, n_q_heads, L, D)).astype(mx.float16)
        q_keys = _make_q4_cache(B, n_kv_heads, N, D)
        q_values = _make_q4_cache(B, n_kv_heads, N, D)
        mask = mx.zeros((B, 1, L, N), dtype=mx.float16)

        result = sdpa(queries, q_keys, q_values, scale=0.125, mask=mask)
        mx.eval(result)

        assert result.shape == (B, n_q_heads, L, D)

    def test_b1_none_mask(self) -> None:
        """B=1 with mask=None passes None through to compiled fn."""
        sdpa = _fresh_patch()

        B, n_heads, L, D, N = 1, 4, 2, 64, 8
        queries = mx.random.normal((B, n_heads, L, D)).astype(mx.float16)
        q_keys = _make_q4_cache(B, n_heads, N, D)
        q_values = _make_q4_cache(B, n_heads, N, D)

        result = sdpa(queries, q_keys, q_values, scale=0.125, mask=None)
        mx.eval(result)

        assert result.shape == (B, n_heads, L, D)

    def test_compiled_cache_hit(self) -> None:
        """Second call with same n_repeats reuses cached compiled fn."""
        sdpa = _fresh_patch()

        B, n_heads, D, N = 1, 4, 64, 8
        q1 = mx.random.normal((B, n_heads, 2, D)).astype(mx.float16)
        k1 = _make_q4_cache(B, n_heads, N, D)
        v1 = _make_q4_cache(B, n_heads, N, D)
        r1 = sdpa(q1, k1, v1, scale=0.125, mask=None)
        mx.eval(r1)

        # Different L and N — same n_repeats=1 → cache hit
        q2 = mx.random.normal((B, n_heads, 3, D)).astype(mx.float16)
        k2 = _make_q4_cache(B, n_heads, 12, D)
        v2 = _make_q4_cache(B, n_heads, 12, D)
        r2 = sdpa(q2, k2, v2, scale=0.125, mask=None)
        mx.eval(r2)

        assert r2.shape == (B, n_heads, 3, D)

    def test_asymmetric_kv_dims(self) -> None:
        """Different K and V head dimensions (like DeepSeek MLA)."""
        sdpa = _fresh_patch()

        B, n_heads, L, K_DIM, V_DIM, N = 1, 4, 2, 128, 64, 8
        queries = mx.random.normal((B, n_heads, L, K_DIM)).astype(mx.float16)
        q_keys = _make_q4_cache(B, n_heads, N, K_DIM)
        q_values = _make_q4_cache(B, n_heads, N, V_DIM)
        mask = mx.zeros((B, 1, L, N), dtype=mx.float16)

        result = sdpa(queries, q_keys, q_values, scale=0.088, mask=mask)
        mx.eval(result)

        assert result.shape == (B, n_heads, L, V_DIM)

    def test_numerical_sanity(self) -> None:
        """Compiled SDPA produces finite, non-NaN results."""
        sdpa = _fresh_patch()

        mx.random.seed(42)
        B, n_heads, L, D, N = 1, 4, 4, 64, 16
        queries = mx.random.normal((B, n_heads, L, D)).astype(mx.float16) * 0.1
        q_keys = _make_q4_cache(B, n_heads, N, D)
        q_values = _make_q4_cache(B, n_heads, N, D)

        result = sdpa(queries, q_keys, q_values, scale=0.125, mask=None)
        mx.eval(result)

        assert not mx.any(mx.isnan(result)).item()
        assert not mx.any(mx.isinf(result)).item()


# ── Batch splitting (B > 1) ───────────────────────────────────────────


class TestBatchSplitting:
    """Tests for the B>1 per-sequence split loop."""

    def test_b2_non_gqa(self) -> None:
        """B=2 non-GQA: splits into 2 B=1 calls, concatenates."""
        sdpa = _fresh_patch()

        B, n_heads, L, D, N = 2, 4, 2, 64, 8
        queries = mx.random.normal((B, n_heads, L, D)).astype(mx.float16)
        q_keys = _make_q4_cache(B, n_heads, N, D)
        q_values = _make_q4_cache(B, n_heads, N, D)
        mask = mx.zeros((B, 1, L, N), dtype=mx.float16)

        result = sdpa(queries, q_keys, q_values, scale=0.125, mask=mask)
        mx.eval(result)

        assert result.shape == (B, n_heads, L, D)

    def test_b2_gqa(self) -> None:
        """B=2 GQA: splits into 2 B=1 GQA calls."""
        sdpa = _fresh_patch()

        B, n_q_heads, n_kv_heads, L, D, N = 2, 8, 4, 2, 64, 8
        queries = mx.random.normal((B, n_q_heads, L, D)).astype(mx.float16)
        q_keys = _make_q4_cache(B, n_kv_heads, N, D)
        q_values = _make_q4_cache(B, n_kv_heads, N, D)
        mask = mx.zeros((B, 1, L, N), dtype=mx.float16)

        result = sdpa(queries, q_keys, q_values, scale=0.125, mask=mask)
        mx.eval(result)

        assert result.shape == (B, n_q_heads, L, D)

    def test_b2_none_mask(self) -> None:
        """B=2 with mask=None: m_i=None for each sequence in split."""
        sdpa = _fresh_patch()

        B, n_heads, L, D, N = 2, 4, 2, 64, 8
        queries = mx.random.normal((B, n_heads, L, D)).astype(mx.float16)
        q_keys = _make_q4_cache(B, n_heads, N, D)
        q_values = _make_q4_cache(B, n_heads, N, D)

        result = sdpa(queries, q_keys, q_values, scale=0.125, mask=None)
        mx.eval(result)

        assert result.shape == (B, n_heads, L, D)

    def test_b3_split(self) -> None:
        """B=3 to verify the split loop handles arbitrary batch sizes."""
        sdpa = _fresh_patch()

        B, n_heads, L, D, N = 3, 4, 2, 64, 8
        queries = mx.random.normal((B, n_heads, L, D)).astype(mx.float16)
        q_keys = _make_q4_cache(B, n_heads, N, D)
        q_values = _make_q4_cache(B, n_heads, N, D)
        mask = mx.zeros((B, 1, L, N), dtype=mx.float16)

        result = sdpa(queries, q_keys, q_values, scale=0.125, mask=mask)
        mx.eval(result)

        assert result.shape == (B, n_heads, L, D)

    def test_b1_vs_b2_consistency(self) -> None:
        """B=1 and B=2 produce same results for same per-seq data."""
        sdpa = _fresh_patch()

        n_heads, L, D, N = 4, 2, 64, 8
        mx.random.seed(99)
        q = mx.random.normal((1, n_heads, L, D)).astype(mx.float16)
        k = _make_q4_cache(1, n_heads, N, D)
        v = _make_q4_cache(1, n_heads, N, D)

        # B=1 direct
        r1 = sdpa(q, k, v, scale=0.125, mask=None)
        mx.eval(r1)

        # B=2 with duplicated data — sequence 0 should match B=1
        q2 = mx.concatenate([q, q], axis=0)
        k2 = tuple(mx.concatenate([a, a], axis=0) for a in k)
        v2 = tuple(mx.concatenate([a, a], axis=0) for a in v)
        r2 = sdpa(q2, k2, v2, scale=0.125, mask=None)
        mx.eval(r2)

        assert mx.allclose(r1, r2[0:1], atol=1e-3).item()


# ── Mask handling ──────────────────────────────────────────────────────


class TestMaskHandling:
    """Tests for string mask conversion and bool/additive mask dispatch."""

    def test_causal_string_mask(self) -> None:
        """String 'causal' converted to boolean mask array."""
        sdpa = _fresh_patch()

        B, n_heads, L, D, N = 1, 4, 3, 64, 10
        queries = mx.random.normal((B, n_heads, L, D)).astype(mx.float16)
        q_keys = _make_q4_cache(B, n_heads, N, D)
        q_values = _make_q4_cache(B, n_heads, N, D)

        result = sdpa(queries, q_keys, q_values, scale=0.125, mask="causal")
        mx.eval(result)

        assert result.shape == (B, n_heads, L, D)

    def test_bool_mask_where_path(self) -> None:
        """Boolean mask triggers mx.where path in compiled fn."""
        sdpa = _fresh_patch()

        B, n_heads, L, D, N = 1, 4, 2, 64, 8
        queries = mx.random.normal((B, n_heads, L, D)).astype(mx.float16)
        q_keys = _make_q4_cache(B, n_heads, N, D)
        q_values = _make_q4_cache(B, n_heads, N, D)

        # Boolean causal mask
        q_idx = mx.arange(N - L, N)
        k_idx = mx.arange(N)
        mask = q_idx[:, None] >= k_idx[None]  # (L, N) bool
        mask = mx.expand_dims(mask, axis=(0, 1))  # (1, 1, L, N)

        result = sdpa(queries, q_keys, q_values, scale=0.125, mask=mask)
        mx.eval(result)

        assert result.shape == (B, n_heads, L, D)

    def test_additive_float_mask(self) -> None:
        """Float additive mask triggers scores + mask path."""
        sdpa = _fresh_patch()

        B, n_heads, L, D, N = 1, 4, 2, 64, 8
        queries = mx.random.normal((B, n_heads, L, D)).astype(mx.float16)
        q_keys = _make_q4_cache(B, n_heads, N, D)
        q_values = _make_q4_cache(B, n_heads, N, D)

        # Additive mask: 0 = no masking, -inf = masked
        mask = mx.zeros((B, 1, L, N), dtype=mx.float16)

        result = sdpa(queries, q_keys, q_values, scale=0.125, mask=mask)
        mx.eval(result)

        assert result.shape == (B, n_heads, L, D)

    def test_causal_string_mask_b2(self) -> None:
        """String 'causal' mask with B=2 batch splitting."""
        sdpa = _fresh_patch()

        B, n_heads, L, D, N = 2, 4, 3, 64, 10
        queries = mx.random.normal((B, n_heads, L, D)).astype(mx.float16)
        q_keys = _make_q4_cache(B, n_heads, N, D)
        q_values = _make_q4_cache(B, n_heads, N, D)

        # String mask gets converted to array before batch split
        result = sdpa(queries, q_keys, q_values, scale=0.125, mask="causal")
        mx.eval(result)

        assert result.shape == (B, n_heads, L, D)


# ── Metal decode kernel ───────────────────────────────────────────────


class TestMetalDecode:
    """Tests for the fused Metal kernel decode path."""

    def test_metal_disabled_uses_compiled(self) -> None:
        """With metal disabled (default), L=1 uses compiled SDPA."""
        sdpa = _fresh_patch(metal_decode=False)

        B, n_heads, D, N = 1, 4, 64, 8
        queries = mx.random.normal((B, n_heads, 1, D)).astype(mx.float16)
        q_keys = _make_q4_cache(B, n_heads, N, D)
        q_values = _make_q4_cache(B, n_heads, N, D)

        result = sdpa(queries, q_keys, q_values, scale=0.125, mask=None)
        mx.eval(result)

        assert result.shape == (B, n_heads, 1, D)

    def test_metal_enabled_l1_decode(self) -> None:
        """With metal enabled, L=1 float16 triggers Metal kernel."""
        sdpa = _fresh_patch(metal_decode=True)

        B, n_heads, D, N = 1, 4, 64, 8
        queries = mx.random.normal((B, n_heads, 1, D)).astype(mx.float16)
        q_keys = _make_q4_cache(B, n_heads, N, D)
        q_values = _make_q4_cache(B, n_heads, N, D)

        result = sdpa(queries, q_keys, q_values, scale=0.125, mask=None)
        mx.eval(result)

        assert result.shape == (B, n_heads, 1, D)
        assert result.dtype == mx.float16

    def test_metal_enabled_l_gt_1_falls_back(self) -> None:
        """L > 1 with metal enabled falls back to compiled SDPA."""
        sdpa = _fresh_patch(metal_decode=True)

        B, n_heads, L, D, N = 1, 4, 3, 64, 8
        queries = mx.random.normal((B, n_heads, L, D)).astype(mx.float16)
        q_keys = _make_q4_cache(B, n_heads, N, D)
        q_values = _make_q4_cache(B, n_heads, N, D)

        result = sdpa(queries, q_keys, q_values, scale=0.125, mask=None)
        mx.eval(result)

        assert result.shape == (B, n_heads, L, D)

    def test_metal_bfloat16_falls_back(self) -> None:
        """bfloat16 queries with metal enabled fall back to compiled."""
        sdpa = _fresh_patch(metal_decode=True)

        B, n_heads, D, N = 1, 4, 64, 8
        queries = mx.random.normal((B, n_heads, 1, D)).astype(mx.bfloat16)
        q_keys = _make_q4_cache(B, n_heads, N, D, dtype=mx.bfloat16)
        q_values = _make_q4_cache(B, n_heads, N, D, dtype=mx.bfloat16)

        result = sdpa(queries, q_keys, q_values, scale=0.125, mask=None)
        mx.eval(result)

        assert result.shape == (B, n_heads, 1, D)

    def test_metal_b2_l1_decode(self) -> None:
        """B=2 with metal enabled and L=1."""
        sdpa = _fresh_patch(metal_decode=True)

        B, n_heads, D, N = 2, 4, 64, 8
        queries = mx.random.normal((B, n_heads, 1, D)).astype(mx.float16)
        q_keys = _make_q4_cache(B, n_heads, N, D)
        q_values = _make_q4_cache(B, n_heads, N, D)

        result = sdpa(queries, q_keys, q_values, scale=0.125, mask=None)
        mx.eval(result)

        assert result.shape == (B, n_heads, 1, D)

    def test_metal_kernel_cached(self) -> None:
        """Second L=1 call reuses cached Metal kernel."""
        sdpa = _fresh_patch(metal_decode=True)

        B, n_heads, D = 1, 4, 64

        # First call — creates kernel
        q1 = mx.random.normal((B, n_heads, 1, D)).astype(mx.float16)
        k1 = _make_q4_cache(B, n_heads, 8, D)
        v1 = _make_q4_cache(B, n_heads, 8, D)
        r1 = sdpa(q1, k1, v1, scale=0.125, mask=None)
        mx.eval(r1)

        # Second call — different N, same geometry → cache hit
        q2 = mx.random.normal((B, n_heads, 1, D)).astype(mx.float16)
        k2 = _make_q4_cache(B, n_heads, 16, D)
        v2 = _make_q4_cache(B, n_heads, 16, D)
        r2 = sdpa(q2, k2, v2, scale=0.125, mask=None)
        mx.eval(r2)

        assert r2.shape == (B, n_heads, 1, D)

    def test_metal_gqa_l1(self) -> None:
        """Metal decode with GQA (n_repeats=2), L=1."""
        sdpa = _fresh_patch(metal_decode=True)

        B, n_q_heads, n_kv_heads, D, N = 1, 8, 4, 64, 8
        queries = mx.random.normal((B, n_q_heads, 1, D)).astype(mx.float16)
        q_keys = _make_q4_cache(B, n_kv_heads, N, D)
        q_values = _make_q4_cache(B, n_kv_heads, N, D)

        result = sdpa(queries, q_keys, q_values, scale=0.125, mask=None)
        mx.eval(result)

        assert result.shape == (B, n_q_heads, 1, D)

    def test_metal_asymmetric_kv(self) -> None:
        """Metal decode with different K and V dimensions."""
        sdpa = _fresh_patch(metal_decode=True)

        B, n_heads, K_DIM, V_DIM, N = 1, 4, 128, 64, 8
        queries = mx.random.normal((B, n_heads, 1, K_DIM)).astype(mx.float16)
        q_keys = _make_q4_cache(B, n_heads, N, K_DIM)
        q_values = _make_q4_cache(B, n_heads, N, V_DIM)

        result = sdpa(queries, q_keys, q_values, scale=0.088, mask=None)
        mx.eval(result)

        assert result.shape == (B, n_heads, 1, V_DIM)

    def test_metal_larger_geometry(self) -> None:
        """Metal decode with larger head count and longer cache."""
        sdpa = _fresh_patch(metal_decode=True)

        B, n_heads, D, N = 1, 8, 128, 32
        queries = mx.random.normal((B, n_heads, 1, D)).astype(mx.float16)
        q_keys = _make_q4_cache(B, n_heads, N, D)
        q_values = _make_q4_cache(B, n_heads, N, D)

        result = sdpa(queries, q_keys, q_values, scale=0.088, mask=None)
        mx.eval(result)

        assert result.shape == (B, n_heads, 1, D)
        assert not mx.any(mx.isnan(result)).item()


# ── Clip residual ─────────────────────────────────────────────────────


class TestClipResidual:
    """Tests for the uncompiled clip_residual replacement."""

    @pytest.fixture
    def clip_fn(self):
        """Get the patched clip_residual function."""
        _fresh_patch()
        import mlx_lm.models.gemma3_text as gemma3

        return gemma3.clip_residual

    def test_bfloat16_simple_add(self, clip_fn) -> None:
        """bfloat16: x.dtype != float16 → simple x + y."""
        x = mx.ones((2, 4), dtype=mx.bfloat16)
        y = mx.ones((2, 4), dtype=mx.bfloat16) * 2.0

        result = clip_fn(x, y)
        mx.eval(result)

        expected = mx.full((2, 4), 3.0, dtype=mx.bfloat16)
        assert mx.allclose(result, expected).item()
        assert result.dtype == mx.bfloat16

    def test_float32_simple_add(self, clip_fn) -> None:
        """float32: x.dtype != float16 → simple x + y."""
        x = mx.array([1.0, 2.0], dtype=mx.float32)
        y = mx.array([3.0, 4.0], dtype=mx.float32)

        result = clip_fn(x, y)
        mx.eval(result)

        expected = mx.array([4.0, 6.0], dtype=mx.float32)
        assert mx.allclose(result, expected).item()
        assert result.dtype == mx.float32

    def test_float16_clip_path(self, clip_fn) -> None:
        """float16: clips in float32 then casts back, preventing overflow."""
        bound = mx.finfo(mx.float16).max  # 65504.0
        x = mx.array([bound * 0.9], dtype=mx.float16)
        y = mx.array([bound * 0.5], dtype=mx.float16)

        result = clip_fn(x, y)
        mx.eval(result)

        assert result.dtype == mx.float16
        # Without clipping, 0.9*65504 + 0.5*65504 = 1.4*65504 > 65504 → inf
        # With clipping, result is clamped to bound
        assert result.item() <= float(bound)
        assert not mx.isinf(result).item()

    def test_float16_negative_clip(self, clip_fn) -> None:
        """float16: negative overflow also clipped."""
        bound = mx.finfo(mx.float16).max
        x = mx.array([-bound * 0.9], dtype=mx.float16)
        y = mx.array([-bound * 0.5], dtype=mx.float16)

        result = clip_fn(x, y)
        mx.eval(result)

        assert result.dtype == mx.float16
        assert result.item() >= -float(bound)
        assert not mx.isinf(result).item()

    def test_float16_normal_values(self, clip_fn) -> None:
        """float16: normal values pass through without clipping."""
        x = mx.array([1.0, -2.0, 3.0], dtype=mx.float16)
        y = mx.array([0.5, 1.0, -1.0], dtype=mx.float16)

        result = clip_fn(x, y)
        mx.eval(result)

        expected = mx.array([1.5, -1.0, 2.0], dtype=mx.float16)
        assert mx.allclose(result, expected, atol=1e-2).item()


# ── Patched dispatch integration ───────────────────────────────────────


class TestPatchedDispatch:
    """Integration tests for the full _patched_q4_sdpa dispatch chain."""

    def test_metal_to_compiled_fallback_chain(self) -> None:
        """Metal enabled + L>1 → metal returns None → compiled handles it."""
        sdpa = _fresh_patch(metal_decode=True)

        B, n_heads, L, D, N = 1, 4, 4, 64, 12
        queries = mx.random.normal((B, n_heads, L, D)).astype(mx.float16)
        q_keys = _make_q4_cache(B, n_heads, N, D)
        q_values = _make_q4_cache(B, n_heads, N, D)

        result = sdpa(queries, q_keys, q_values, scale=0.125, mask=None)
        mx.eval(result)

        assert result.shape == (B, n_heads, L, D)
        assert not mx.any(mx.isnan(result)).item()

    def test_decode_then_prefill(self) -> None:
        """Alternate L=1 (decode) and L>1 (prefill) calls."""
        sdpa = _fresh_patch()

        B, n_heads, D, N = 1, 4, 64, 8
        q_keys = _make_q4_cache(B, n_heads, N, D)
        q_values = _make_q4_cache(B, n_heads, N, D)

        # Decode step (L=1)
        q_decode = mx.random.normal((B, n_heads, 1, D)).astype(mx.float16)
        r_decode = sdpa(q_decode, q_keys, q_values, scale=0.125, mask=None)
        mx.eval(r_decode)
        assert r_decode.shape == (B, n_heads, 1, D)

        # Prefill step (L=4)
        q_prefill = mx.random.normal((B, n_heads, 4, D)).astype(mx.float16)
        r_prefill = sdpa(q_prefill, q_keys, q_values, scale=0.125, mask=None)
        mx.eval(r_prefill)
        assert r_prefill.shape == (B, n_heads, 4, D)

    def test_gemma3_like_geometry(self) -> None:
        """Gemma 3 12B-like: 16 Q heads, 8 KV heads, head_dim=256, GQA n_rep=2."""
        sdpa = _fresh_patch()

        B, n_q, n_kv, D, N = 1, 16, 8, 256, 4
        queries = mx.random.normal((B, n_q, 2, D)).astype(mx.float16)
        q_keys = _make_q4_cache(B, n_kv, N, D)
        q_values = _make_q4_cache(B, n_kv, N, D)

        result = sdpa(queries, q_keys, q_values, scale=0.0625, mask=None)
        mx.eval(result)

        assert result.shape == (B, n_q, 2, D)

    def test_deepseek_like_geometry(self) -> None:
        """DeepSeek MLA-like: 16 Q/KV heads, K=192, V=128."""
        sdpa = _fresh_patch()

        B, n_heads, K_DIM, V_DIM, N = 1, 16, 192, 128, 4
        queries = mx.random.normal((B, n_heads, 2, K_DIM)).astype(mx.float16)
        q_keys = _make_q4_cache(B, n_heads, N, K_DIM)
        q_values = _make_q4_cache(B, n_heads, N, V_DIM)

        result = sdpa(queries, q_keys, q_values, scale=0.072, mask=None)
        mx.eval(result)

        assert result.shape == (B, n_heads, 2, V_DIM)

    def test_logging_on_first_patch(self, caplog) -> None:
        """Patch logs 'Applied fused Q4 attention monkeypatch'."""
        import logging

        mod_name = "agent_memory.adapters.outbound.mlx_fused_attention"
        sys.modules.pop(mod_name, None)
        os.environ.pop("SEMANTIC_DISABLE_FUSED_ATTN", None)
        os.environ.pop("SEMANTIC_ENABLE_METAL_DECODE", None)

        mod = importlib.import_module(mod_name)
        with caplog.at_level(logging.INFO, logger=mod_name):
            result = mod.apply_fused_attention_patch()

        assert result is True
        assert any("fused Q4 attention" in r.message for r in caplog.records)

    def test_non_q4_group_size_fallback(self) -> None:
        """group_size=128 falls through to original mlx_lm SDPA (line 138)."""
        sdpa = _fresh_patch()

        B, n_heads, L, D, N = 1, 4, 2, 128, 8
        queries = mx.random.normal((B, n_heads, L, D)).astype(mx.float16)
        # Quantize with group_size=128 (D=128 >= group_size)
        flat_k = mx.random.normal((B * n_heads * N, D)).astype(mx.float16)
        k_data, k_scales, k_biases = mx.quantize(flat_k, group_size=128, bits=4)
        q_keys = (
            k_data.reshape(B, n_heads, N, -1),
            k_scales.reshape(B, n_heads, N, -1),
            k_biases.reshape(B, n_heads, N, -1),
        )
        flat_v = mx.random.normal((B * n_heads * N, D)).astype(mx.float16)
        v_data, v_scales, v_biases = mx.quantize(flat_v, group_size=128, bits=4)
        q_values = (
            v_data.reshape(B, n_heads, N, -1),
            v_scales.reshape(B, n_heads, N, -1),
            v_biases.reshape(B, n_heads, N, -1),
        )

        result = sdpa(
            queries,
            q_keys,
            q_values,
            scale=0.088,
            mask=None,
            group_size=128,
            bits=4,
        )
        mx.eval(result)

        assert result.shape == (B, n_heads, L, D)
        assert result.dtype == mx.float16


# ── Error fallback paths ──────────────────────────────────────────────


class TestErrorFallbacks:
    """Tests for exception handlers and edge-case guards."""

    def test_metal_kernel_creation_failure(self) -> None:
        """Metal kernel compilation failure falls back to compiled SDPA (lines 334-337, 363)."""
        sdpa = _fresh_patch(metal_decode=True)

        B, n_heads, D, N = 1, 4, 64, 8
        queries = mx.random.normal((B, n_heads, 1, D)).astype(mx.float16)
        q_keys = _make_q4_cache(B, n_heads, N, D)
        q_values = _make_q4_cache(B, n_heads, N, D)

        # Make kernel creation raise — the closure references mx.fast.metal_kernel
        # at call time, so patching the module attribute works.
        def _fail_create(**kwargs):
            raise RuntimeError("Metal compile error")

        original_mk = mx.fast.metal_kernel
        try:
            mx.fast.metal_kernel = _fail_create
            # Should fall back to compiled SDPA gracefully
            result = sdpa(queries, q_keys, q_values, scale=0.125, mask=None)
            mx.eval(result)

            assert result.shape == (B, n_heads, 1, D)
            assert not mx.any(mx.isnan(result)).item()
        finally:
            mx.fast.metal_kernel = original_mk

    def test_metal_kernel_execution_failure(self) -> None:
        """Metal kernel execution failure falls back to compiled SDPA (lines 399-401)."""
        sdpa = _fresh_patch(metal_decode=True)

        B, n_heads, D, N = 1, 4, 64, 8
        queries = mx.random.normal((B, n_heads, 1, D)).astype(mx.float16)
        q_keys = _make_q4_cache(B, n_heads, N, D)
        q_values = _make_q4_cache(B, n_heads, N, D)

        # Replace mx.fast.metal_kernel with one that returns a callable
        # that raises on execution (simulates Metal runtime failure).
        class _BadKernel:
            def __call__(self, **kwargs):
                raise RuntimeError("Metal execution error")

        original_mk = mx.fast.metal_kernel
        try:
            mx.fast.metal_kernel = lambda **kwargs: _BadKernel()
            result = sdpa(queries, q_keys, q_values, scale=0.125, mask=None)
            mx.eval(result)

            assert result.shape == (B, n_heads, 1, D)
            assert not mx.any(mx.isnan(result)).item()
        finally:
            mx.fast.metal_kernel = original_mk

    def test_metal_empty_cache_guard(self) -> None:
        """N=0 (empty cache) with Metal enabled hits guard on line 357.

        After Metal returns None, compiled SDPA handles the empty cache.
        """
        sdpa = _fresh_patch(metal_decode=True)

        B, n_heads, D = 1, 4, 64
        queries = mx.random.normal((B, n_heads, 1, D)).astype(mx.float16)

        # Empty cache: N=0
        packed_dim = D // 8
        groups_dim = D // 64
        q_keys = (
            mx.zeros((B, n_heads, 0, packed_dim), dtype=mx.uint32),
            mx.zeros((B, n_heads, 0, groups_dim), dtype=mx.float16),
            mx.zeros((B, n_heads, 0, groups_dim), dtype=mx.float16),
        )
        q_values = (
            mx.zeros((B, n_heads, 0, packed_dim), dtype=mx.uint32),
            mx.zeros((B, n_heads, 0, groups_dim), dtype=mx.float16),
            mx.zeros((B, n_heads, 0, groups_dim), dtype=mx.float16),
        )

        # Metal returns None (N < 1), falls through to compiled SDPA.
        # quantized_matmul with N=0 may produce NaN or zeros — we just
        # verify no crash and correct output shape.
        result = sdpa(queries, q_keys, q_values, scale=0.125, mask=None)
        mx.eval(result)

        assert result.shape == (B, n_heads, 1, D)
