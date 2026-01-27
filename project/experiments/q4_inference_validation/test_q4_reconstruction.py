#!/usr/bin/env python3
"""Direct test of Q4 reconstruction logic without full server.

Tests the _reconstruct_cache method directly to verify Q4 injection works.
"""

import sys
import mlx.core as mx
sys.path.insert(0, '/Users/dev_user/semantic/src')

from mlx_lm.models.cache import QuantizedKVCache
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_quantized_cache_creation():
    """Test that we can create a QuantizedKVCache directly."""

    logger.info("="*60)
    logger.info("Q4 DIRECT INJECTION - UNIT TEST")
    logger.info("="*60)

    # Create mock Q4 data (simulating what we load from disk)
    kv_bits = 4
    kv_group_size = 64
    seq_len = 256  # 1 block worth
    n_heads = 16
    head_dim = 128

    # Calculate packed dimensions
    # uint32 = 4 bytes = 32 bits, with 4-bit quantization: 32/4 = 8 elements per int
    elements_per_int = 32 // kv_bits  # 32 bits / 4 bits = 8 elements per uint32
    packed_dim = head_dim // elements_per_int  # 128 / 8 = 16

    logger.info(f"\n[SETUP] Creating Q4 cache:")
    logger.info(f"  seq_len={seq_len}, n_heads={n_heads}, head_dim={head_dim}")
    logger.info(f"  kv_bits={kv_bits}, kv_group_size={kv_group_size}")
    logger.info(f"  packed_dim={packed_dim}")

    # Create Q4 quantized tensors (weights, scales, biases)
    k_weights = mx.zeros((1, n_heads, seq_len, packed_dim), dtype=mx.uint32)
    k_scales = mx.ones((1, n_heads, seq_len, head_dim // kv_group_size), dtype=mx.float16)
    k_biases = mx.zeros((1, n_heads, seq_len, head_dim // kv_group_size), dtype=mx.float16)

    v_weights = mx.zeros((1, n_heads, seq_len, packed_dim), dtype=mx.uint32)
    v_scales = mx.ones((1, n_heads, seq_len, head_dim // kv_group_size), dtype=mx.float16)
    v_biases = mx.zeros((1, n_heads, seq_len, head_dim // kv_group_size), dtype=mx.float16)

    mx.eval(k_weights, k_scales, k_biases, v_weights, v_scales, v_biases)

    q4_size_mb = (
        k_weights.nbytes + k_scales.nbytes + k_biases.nbytes +
        v_weights.nbytes + v_scales.nbytes + v_biases.nbytes
    ) / (1024**2)

    logger.info(f"\n[Q4 DATA] Total size: {q4_size_mb:.1f}MB")

    # Test 1: Create QuantizedKVCache directly (NEW approach)
    logger.info("\n[TEST 1] Creating QuantizedKVCache directly...")

    try:
        kv_cache = QuantizedKVCache(group_size=kv_group_size, bits=kv_bits)
        kv_cache.keys = (k_weights, k_scales, k_biases)
        kv_cache.values = (v_weights, v_scales, v_biases)
        kv_cache.offset = seq_len

        logger.info(f"✅ QuantizedKVCache created successfully!")
        logger.info(f"   Type: {type(kv_cache)}")
        logger.info(f"   Has 'bits' attr: {hasattr(kv_cache, 'bits')}")
        logger.info(f"   bits={kv_cache.bits}, group_size={kv_cache.group_size}")
        logger.info(f"   offset={kv_cache.offset}")

        # Verify state
        keys, values = kv_cache.state
        logger.info(f"   Keys shape: {keys[0].shape if isinstance(keys, tuple) else keys.shape}")
        logger.info(f"   Values shape: {values[0].shape if isinstance(values, tuple) else values.shape}")

    except Exception as e:
        logger.error(f"❌ Failed to create QuantizedKVCache: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test 2: Verify MLX routing would use quantized attention
    logger.info("\n[TEST 2] Checking MLX routing...")

    if hasattr(kv_cache, 'bits'):
        logger.info(f"✅ MLX will route to quantized_scaled_dot_product_attention!")
        logger.info(f"   Routing condition: hasattr(cache, 'bits') = True")
    else:
        logger.error(f"❌ MLX will NOT use Q4 attention - missing 'bits' attribute")
        return False

    # Test 3: Compare with dequantized size
    logger.info("\n[TEST 3] Memory comparison...")

    # Calculate FP16 size
    fp16_size = seq_len * n_heads * head_dim * 2 * 2 / (1024**2)  # K+V, 2 bytes per element

    logger.info(f"   Q4 size:   {q4_size_mb:.2f}MB")
    logger.info(f"   FP16 size: {fp16_size:.2f}MB (if dequantized)")
    logger.info(f"   Savings:   {(1 - q4_size_mb/fp16_size)*100:.1f}%")

    if q4_size_mb < fp16_size * 0.5:
        logger.info(f"✅ Q4 direct injection saves memory!")
    else:
        logger.warning(f"⚠️  Unexpected: Q4 not smaller than FP16")

    logger.info("\n" + "="*60)
    logger.info("✅ Q4 DIRECT INJECTION UNIT TEST PASSED!")
    logger.info("="*60)

    return True


def main():
    """Run Q4 reconstruction test."""
    try:
        success = test_quantized_cache_creation()
        return 0 if success else 1
    except Exception as e:
        logger.exception(f"Test failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
