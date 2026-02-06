"""Validate Metal decode kernel against reference Q4 SDPA.

Systematic validation:
1. Compiled SDPA vs dequantize+manual reference (sanity check)
2. Metal kernel vs compiled SDPA (isolation test)
3. Minimal Metal kernel tests (infrastructure check)

Usage:
    python validate_metal_kernel.py
"""

import sys
import math
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

try:
    import mlx.core as mx
except ImportError:
    logger.error("mlx not available")
    sys.exit(1)


def make_quantized_kv(B, n_kv_heads, N, head_dim, group_size=64, bits=4):
    """Create random quantized KV cache entries."""
    fp = mx.random.normal((B, n_kv_heads, N, head_dim)).astype(mx.float16)
    packed, scales, biases = mx.quantize(fp, group_size=group_size, bits=bits)
    return (packed, scales, biases)


def reference_attention(queries_f32, k_deq_f32, v_deq_f32, scale, n_repeats):
    """Pure reference: dequantized FP32 attention."""
    if n_repeats > 1:
        k_deq_f32 = mx.repeat(k_deq_f32, n_repeats, axis=1)
        v_deq_f32 = mx.repeat(v_deq_f32, n_repeats, axis=1)
    scores = (queries_f32 @ k_deq_f32.transpose(0, 1, 3, 2)) * scale
    weights = mx.softmax(scores, axis=-1)
    return weights @ v_deq_f32


def compare(name, a, b, atol=0.05, rtol=0.1):
    """Compare two arrays, return pass/fail."""
    a_f32 = a.astype(mx.float32)
    b_f32 = b.astype(mx.float32)
    diff = mx.abs(a_f32 - b_f32)
    max_diff = mx.max(diff).item()
    mean_diff = mx.mean(diff).item()
    ref_norm = mx.mean(mx.abs(a_f32)).item()
    rel_error = mean_diff / max(ref_norm, 1e-6)
    passed = max_diff < atol and rel_error < rtol
    status = "PASS" if passed else "FAIL"
    logger.info("  [%s] %s: max=%.6f mean=%.6f rel=%.4f", status, name, max_diff, mean_diff, rel_error)
    if not passed:
        logger.error("    a[:8] = %s", a_f32.reshape(-1)[:8].tolist())
        logger.error("    b[:8] = %s", b_f32.reshape(-1)[:8].tolist())
    return passed


def test_compiled_sdpa_vs_reference():
    """Phase 1: Verify compiled SDPA matches manual reference.

    Uses L=2 queries to bypass the Metal kernel (L=1 only) and test
    only the compiled mx.quantized_matmul path.
    """
    logger.info("=== Phase 1: Compiled SDPA vs Reference ===")

    from semantic.adapters.outbound.mlx_fused_attention import apply_fused_attention_patch
    import mlx_lm.models.base as base_module
    apply_fused_attention_patch()

    results = []
    for name, n_q, n_kv, kdim, N in [
        ("small_no_gqa", 2, 2, 64, 16),
        ("small_gqa", 4, 2, 64, 16),
        ("gemma3", 16, 8, 256, 64),
    ]:
        mx.random.seed(42)
        n_repeats = n_q // n_kv
        scale = 1.0 / math.sqrt(kdim)
        # Use L=2 to bypass Metal kernel (it only handles L=1)
        queries = mx.random.normal((1, n_q, 2, kdim)).astype(mx.float16)
        q_keys = make_quantized_kv(1, n_kv, N, kdim)
        q_values = make_quantized_kv(1, n_kv, N, kdim)
        mx.eval(queries, *q_keys, *q_values)

        # Reference: dequantize + manual attention
        k_deq = mx.dequantize(q_keys[0], q_keys[1], q_keys[2], group_size=64, bits=4)
        v_deq = mx.dequantize(q_values[0], q_values[1], q_values[2], group_size=64, bits=4)
        ref = reference_attention(
            queries.astype(mx.float32), k_deq.astype(mx.float32),
            v_deq.astype(mx.float32), scale, n_repeats,
        ).astype(mx.float16)
        mx.eval(ref)

        # Compiled SDPA: L=2 forces compiled path (skips Metal kernel)
        compiled_out = base_module.quantized_scaled_dot_product_attention(
            queries, q_keys, q_values, scale=scale, mask=None,
            group_size=64, bits=4,
        )
        mx.eval(compiled_out)

        results.append(compare(f"{name}_ref_vs_compiled", ref, compiled_out, atol=0.15, rtol=0.3))

    return all(results)


def test_minimal_metal_kernel():
    """Phase 2: Test that mx.fast.metal_kernel + simd_sum works at all."""
    logger.info("=== Phase 2: Minimal Metal Kernel Tests ===")

    results = []

    # NOTE: mx.fast.metal_kernel uses dispatch_threads semantics:
    # grid = total thread count (NOT threadgroup count).
    # grid=(128,1,1) with threadgroup=(32,1,1) → 4 threadgroups of 32 threads.

    # Test A: Simple copy kernel (no simd operations)
    logger.info("  Test A: Simple copy kernel")
    kernel_copy = mx.fast.metal_kernel(
        name="test_copy",
        input_names=["inp"],
        output_names=["out"],
        source="""
    uint tid = thread_position_in_threadgroup.x;
    uint gid = threadgroup_position_in_grid.x;
    uint idx = gid * 32 + tid;
    out[idx] = inp[idx];
""",
        header="",
        ensure_row_contiguous=True,
    )
    inp = mx.arange(128).astype(mx.float32)
    out = kernel_copy(
        inputs=[inp], grid=(128, 1, 1), threadgroup=(32, 1, 1),
        output_shapes=[(128,)], output_dtypes=[mx.float32],
    )[0]
    mx.eval(out)
    match = mx.array_equal(inp, out).item()
    logger.info("    Copy: %s", "PASS" if match else "FAIL")
    results.append(match)

    # Test B: simd_sum kernel
    logger.info("  Test B: simd_sum kernel")
    kernel_sum = mx.fast.metal_kernel(
        name="test_simd_sum",
        input_names=["inp"],
        output_names=["out"],
        source="""
    uint tid = thread_position_in_threadgroup.x;
    uint gid = threadgroup_position_in_grid.x;
    float val = inp[gid * 32 + tid];
    float total = simd_sum(val);
    if (tid == 0) {
        out[gid] = total;
    }
""",
        header="",
        ensure_row_contiguous=True,
    )
    inp = mx.ones((128,), dtype=mx.float32)
    out = kernel_sum(
        inputs=[inp], grid=(128, 1, 1), threadgroup=(32, 1, 1),
        output_shapes=[(4,)], output_dtypes=[mx.float32],
    )[0]
    mx.eval(out)
    expected = mx.full((4,), 32.0)
    match = mx.allclose(out, expected).item()
    logger.info("    simd_sum(ones): out=%s, expected=%s, %s",
                out.tolist(), expected.tolist(), "PASS" if match else "FAIL")
    results.append(match)

    # Test C: simd_sum returns same value to ALL threads (not just thread 0)
    logger.info("  Test C: simd_sum broadcast to all threads")
    kernel_bcast = mx.fast.metal_kernel(
        name="test_simd_broadcast",
        input_names=["inp"],
        output_names=["out"],
        source="""
    uint tid = thread_position_in_threadgroup.x;
    uint gid = threadgroup_position_in_grid.x;
    float val = inp[gid * 32 + tid];
    float total = simd_sum(val);
    out[gid * 32 + tid] = total;
""",
        header="",
        ensure_row_contiguous=True,
    )
    inp = mx.ones((64,), dtype=mx.float32)
    out = kernel_bcast(
        inputs=[inp], grid=(64, 1, 1), threadgroup=(32, 1, 1),
        output_shapes=[(64,)], output_dtypes=[mx.float32],
    )[0]
    mx.eval(out)
    expected = mx.full((64,), 32.0)
    match = mx.allclose(out, expected).item()
    logger.info("    All threads see same simd_sum: %s (sample: %s)",
                "PASS" if match else "FAIL", out[:4].tolist())
    results.append(match)

    # Test D: dequant4 template with half scales
    logger.info("  Test D: Q4 dequantization")
    # Create a known quantized value: pack [0,1,2,3,4,5,6,7] into one uint32
    # Element i has value i, packed LSB-first: word = 0 | (1<<4) | (2<<8) | ...
    word_val = sum(i << (i * 4) for i in range(8))
    k_w = mx.array([word_val], dtype=mx.uint32)
    k_s = mx.array([2.0], dtype=mx.float16)  # scale=2
    k_b = mx.array([1.0], dtype=mx.float16)  # bias=1
    # Expected: dequant(i) = i * 2 + 1 = [1, 3, 5, 7, 9, 11, 13, 15]

    kernel_deq = mx.fast.metal_kernel(
        name="test_dequant",
        input_names=["k_w", "k_s", "k_b"],
        output_names=["out"],
        source="""
    uint tid = thread_position_in_threadgroup.x;
    if (tid < 8) {
        int packed_idx = tid / 8;
        int bit_offset = (tid % 8) * 4;
        uint raw = (k_w[packed_idx] >> bit_offset) & 0xFu;
        out[tid] = float(raw) * float(k_s[0]) + float(k_b[0]);
    }
""",
        header="",
        ensure_row_contiguous=True,
    )
    out = kernel_deq(
        inputs=[k_w, k_s, k_b], grid=(32, 1, 1), threadgroup=(32, 1, 1),
        output_shapes=[(8,)], output_dtypes=[mx.float32],
    )[0]
    mx.eval(out)
    expected = mx.array([1, 3, 5, 7, 9, 11, 13, 15], dtype=mx.float32)
    match = mx.allclose(out, expected).item()
    logger.info("    Dequant: out=%s, expected=%s, %s",
                out.tolist(), expected.tolist(), "PASS" if match else "FAIL")
    results.append(match)

    return all(results)


def test_metal_decode_kernel():
    """Phase 3: Test actual decode kernel with known inputs."""
    logger.info("=== Phase 3: Metal Decode Kernel ===")

    from semantic.adapters.outbound.mlx_fused_attention import apply_fused_attention_patch
    import mlx_lm.models.base as base_module
    apply_fused_attention_patch()

    results = []
    for name, n_q, n_kv, kdim, vdim, N in [
        ("tiny", 2, 2, 64, 64, 4),
        ("small_gqa", 4, 2, 64, 64, 16),
        ("gemma3", 16, 8, 256, 256, 64),
        ("deepseek", 16, 2, 192, 128, 32),
    ]:
        mx.random.seed(42)
        n_repeats = n_q // n_kv
        scale = 1.0 / math.sqrt(kdim)
        queries = mx.random.normal((1, n_q, 1, kdim)).astype(mx.float16)
        q_keys = make_quantized_kv(1, n_kv, N, kdim)
        q_values = make_quantized_kv(1, n_kv, N, vdim)
        mx.eval(queries, *q_keys, *q_values)

        # Reference
        k_deq = mx.dequantize(q_keys[0], q_keys[1], q_keys[2], group_size=64, bits=4)
        v_deq = mx.dequantize(q_values[0], q_values[1], q_values[2], group_size=64, bits=4)
        ref = reference_attention(
            queries.astype(mx.float32), k_deq.astype(mx.float32),
            v_deq.astype(mx.float32), scale, n_repeats,
        ).astype(mx.float16)
        mx.eval(ref)

        # Metal kernel via patched function
        metal_out = base_module.quantized_scaled_dot_product_attention(
            queries, q_keys, q_values, scale=scale, mask=None,
            group_size=64, bits=4,
        )
        mx.eval(metal_out)

        results.append(compare(f"{name}", ref, metal_out))

    return all(results)


def main():
    phase1 = test_compiled_sdpa_vs_reference()
    logger.info("")
    phase2 = test_minimal_metal_kernel()
    logger.info("")

    if not phase2:
        logger.error("Metal kernel infrastructure broken, skipping Phase 3")
        sys.exit(1)

    phase3 = test_metal_decode_kernel()
    logger.info("")

    logger.info("=" * 60)
    logger.info("Phase 1 (compiled SDPA vs ref): %s", "PASS" if phase1 else "FAIL")
    logger.info("Phase 2 (Metal infrastructure): %s", "PASS" if phase2 else "FAIL")
    logger.info("Phase 3 (Metal decode kernel):  %s", "PASS" if phase3 else "FAIL")

    if phase1 and phase2 and phase3:
        logger.info("All phases passed!")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
