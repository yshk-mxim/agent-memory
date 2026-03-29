# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Yakov Shkolnikov and contributors
"""TRT cache round-trip integration tests.

Tests the critical data path:
    FP16 cache -> Q4 quantize -> disk (safetensors) -> load -> dequantize -> FP16

Validates that the TRT quantization adapter, cache adapter, and safetensors
persistence work together end-to-end with acceptable error bounds.
"""

from pathlib import Path

import numpy as np
import pytest

from agent_memory.adapters.outbound.trt_cache_adapter import TRTCacheAdapter
from agent_memory.adapters.outbound.trt_quantization_adapter import TRTQuantizationAdapter

from .conftest import SMOLLM2_HEAD_DIM, SMOLLM2_N_KV_HEADS, SMOLLM2_N_LAYERS

pytestmark = pytest.mark.integration

GROUP_SIZE = 64


class TestQ4FP16RoundTrip:
    """Test Q4 <-> FP16 quantization round-trip via TRT adapters."""

    def test_single_layer_round_trip(self, trt_quantizer: TRTQuantizationAdapter) -> None:
        """Single layer Q4 round-trip should have bounded error."""
        rng = np.random.default_rng(42)
        k = rng.standard_normal((SMOLLM2_N_KV_HEADS, 64, SMOLLM2_HEAD_DIM)).astype(np.float16)

        flat = k.reshape(-1)
        weights, scales, biases = trt_quantizer.quantize(flat, bits=4, group_size=GROUP_SIZE)
        recovered = trt_quantizer.dequantize(weights, scales, biases, bits=4, group_size=GROUP_SIZE)
        recovered = recovered[: flat.size]

        error = np.abs(flat.astype(np.float32) - recovered.astype(np.float32))
        assert error.mean() < 0.10, f"Mean Q4 error {error.mean():.4f}"

    def test_full_cache_round_trip(
        self,
        trt_quantizer: TRTQuantizationAdapter,
        fake_fp16_cache: list,
    ) -> None:
        """Full multi-layer cache Q4 round-trip should preserve data within bounds."""
        for layer_idx, (k, v) in enumerate(fake_fp16_cache[:4]):  # Test 4 layers
            k_flat = k.reshape(-1)
            v_flat = v.reshape(-1)

            # Quantize
            kw, ks, kb = trt_quantizer.quantize(k_flat, bits=4, group_size=GROUP_SIZE)
            vw, vs, vb = trt_quantizer.quantize(v_flat, bits=4, group_size=GROUP_SIZE)

            # Dequantize
            k_rec = trt_quantizer.dequantize(kw, ks, kb, bits=4, group_size=GROUP_SIZE)
            v_rec = trt_quantizer.dequantize(vw, vs, vb, bits=4, group_size=GROUP_SIZE)

            k_err = np.abs(k_flat.astype(np.float32) - k_rec[: k_flat.size].astype(np.float32))
            v_err = np.abs(v_flat.astype(np.float32) - v_rec[: v_flat.size].astype(np.float32))

            assert k_err.mean() < 0.10, f"Layer {layer_idx} K mean error {k_err.mean():.4f}"
            assert v_err.mean() < 0.10, f"Layer {layer_idx} V mean error {v_err.mean():.4f}"


class TestLayoutRoundTrip:
    """Test per-layer <-> stacked 5D layout conversion."""

    def test_stacked_round_trip(
        self,
        trt_cache_adapter: TRTCacheAdapter,
        fake_fp16_cache: list,
    ) -> None:
        """Per-layer -> stacked -> per-layer should be exact."""
        stacked = trt_cache_adapter.per_layer_to_stacked(fake_fp16_cache)
        assert stacked.shape == (
            SMOLLM2_N_LAYERS,
            2,
            SMOLLM2_N_KV_HEADS,
            64,
            SMOLLM2_HEAD_DIM,
        )

        recovered = trt_cache_adapter.stacked_to_per_layer(stacked)
        assert len(recovered) == SMOLLM2_N_LAYERS

        for (k_orig, v_orig), (k_rec, v_rec) in zip(fake_fp16_cache, recovered):
            np.testing.assert_array_equal(k_orig, k_rec)
            np.testing.assert_array_equal(v_orig, v_rec)


class TestDiskPersistenceRoundTrip:
    """Test Q4 quantize -> safetensors disk -> load -> dequantize pipeline."""

    def test_quantize_save_load_dequantize(
        self,
        trt_quantizer: TRTQuantizationAdapter,
        cache_dir: Path,
    ) -> None:
        """Full disk round-trip: FP16 -> Q4 -> safetensors file -> Q4 -> FP16."""
        from safetensors.numpy import load_file, save_file

        rng = np.random.default_rng(99)
        seq_len = 32

        # Create FP16 data for 2 layers
        original_layers = []
        for _ in range(2):
            k = rng.standard_normal((SMOLLM2_N_KV_HEADS, seq_len, SMOLLM2_HEAD_DIM)).astype(
                np.float16
            )
            v = rng.standard_normal((SMOLLM2_N_KV_HEADS, seq_len, SMOLLM2_HEAD_DIM)).astype(
                np.float16
            )
            original_layers.append((k, v))

        # Quantize via CacheQuantizationPort
        tensors: dict[str, np.ndarray] = {}
        for layer_idx, (k, v) in enumerate(original_layers):
            kw, ks, kb = trt_quantizer.quantize(k.reshape(-1), bits=4, group_size=GROUP_SIZE)
            vw, vs, vb = trt_quantizer.quantize(v.reshape(-1), bits=4, group_size=GROUP_SIZE)
            tensors[f"L{layer_idx}_K_weights"] = kw
            tensors[f"L{layer_idx}_K_scales"] = ks
            tensors[f"L{layer_idx}_K_biases"] = kb
            tensors[f"L{layer_idx}_V_weights"] = vw
            tensors[f"L{layer_idx}_V_scales"] = vs
            tensors[f"L{layer_idx}_V_biases"] = vb

        # Save to safetensors file
        cache_file = cache_dir / "trt_roundtrip.safetensors"
        save_file(tensors, str(cache_file))
        assert cache_file.exists()

        # Load back and dequantize
        loaded = load_file(str(cache_file))
        for layer_idx, (k_orig, v_orig) in enumerate(original_layers):
            kw = loaded[f"L{layer_idx}_K_weights"]
            ks = loaded[f"L{layer_idx}_K_scales"]
            kb = loaded[f"L{layer_idx}_K_biases"]
            k_rec = trt_quantizer.dequantize(kw, ks, kb, bits=4, group_size=GROUP_SIZE)
            k_rec = k_rec[: k_orig.size]

            error = np.abs(k_orig.reshape(-1).astype(np.float32) - k_rec.astype(np.float32))
            assert error.mean() < 0.10, f"Layer {layer_idx} K error {error.mean():.4f}"
