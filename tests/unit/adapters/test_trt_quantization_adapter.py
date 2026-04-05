# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Yakov Shkolnikov and contributors
"""Unit tests for TRT quantization adapter — Q4/Q8 round-trip and error bounds."""

import numpy as np
import pytest

from agent_memory.adapters.outbound.trt_quantization_adapter import TRTQuantizationAdapter
from agent_memory.domain.errors import ModelSpecValidationError

pytestmark = pytest.mark.unit

GROUP_SIZE = 64


@pytest.fixture
def adapter() -> TRTQuantizationAdapter:
    return TRTQuantizationAdapter()


def _make_fp16_tensor(size: int = 1024) -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.standard_normal(size).astype(np.float16)


class TestSupportedBits:
    def test_supported_bits(self, adapter: TRTQuantizationAdapter) -> None:
        assert adapter.supported_bits == {4, 8, 16}


class TestQ8RoundTrip:
    def test_round_trip_error_bound(self, adapter: TRTQuantizationAdapter) -> None:
        """Q8 round-trip should have max error < 0.02."""
        original = _make_fp16_tensor(GROUP_SIZE * 16)
        weights, scales, biases = adapter.quantize(original, bits=8, group_size=GROUP_SIZE)
        recovered = adapter.dequantize(weights, scales, biases, bits=8, group_size=GROUP_SIZE)

        # Trim to original size (padding may have been added)
        recovered = recovered[: original.size]
        error = np.abs(original.astype(np.float32) - recovered.astype(np.float32))
        assert error.max() < 0.02, f"Max Q8 error {error.max():.4f} exceeds 0.02"

    def test_round_trip_preserves_shape_info(self, adapter: TRTQuantizationAdapter) -> None:
        original = _make_fp16_tensor(512)
        weights, scales, biases = adapter.quantize(original, bits=8, group_size=GROUP_SIZE)
        assert scales.dtype == np.float16
        assert biases.dtype == np.float16
        assert scales.shape[0] == 512 // GROUP_SIZE


class TestQ4RoundTrip:
    def test_round_trip_error_bound(self, adapter: TRTQuantizationAdapter) -> None:
        """Q4 round-trip mean error should be small; max can be larger for outliers."""
        original = _make_fp16_tensor(GROUP_SIZE * 16)
        weights, scales, biases = adapter.quantize(original, bits=4, group_size=GROUP_SIZE)
        recovered = adapter.dequantize(weights, scales, biases, bits=4, group_size=GROUP_SIZE)
        recovered = recovered[: original.size]

        error = np.abs(original.astype(np.float32) - recovered.astype(np.float32))
        assert error.mean() < 0.10, f"Mean Q4 error {error.mean():.4f} exceeds 0.10"
        assert error.max() < 0.30, f"Max Q4 error {error.max():.4f} exceeds 0.30"

    def test_packed_size(self, adapter: TRTQuantizationAdapter) -> None:
        """Q4 weights should be half the element count in bytes."""
        original = _make_fp16_tensor(1024)
        weights, _, _ = adapter.quantize(original, bits=4, group_size=GROUP_SIZE)
        # 1024 elements -> 512 packed bytes
        assert weights.shape[0] == 512

    def test_q4_less_error_than_random(self, adapter: TRTQuantizationAdapter) -> None:
        """Q4 should be much better than random reconstruction."""
        original = _make_fp16_tensor(GROUP_SIZE * 8)
        weights, scales, biases = adapter.quantize(original, bits=4, group_size=GROUP_SIZE)
        recovered = adapter.dequantize(weights, scales, biases, bits=4, group_size=GROUP_SIZE)
        recovered = recovered[: original.size]

        quant_error = np.mean((original.astype(np.float32) - recovered.astype(np.float32)) ** 2)
        random_error = np.var(original.astype(np.float32))
        # Quantization error should be much less than total variance
        assert quant_error < random_error * 0.1


class TestFP16Passthrough:
    def test_fp16_round_trip_exact(self, adapter: TRTQuantizationAdapter) -> None:
        """bits=16 should be lossless (passthrough)."""
        original = _make_fp16_tensor(256)
        weights, scales, biases = adapter.quantize(original, bits=16, group_size=GROUP_SIZE)
        recovered = adapter.dequantize(weights, scales, biases, bits=16, group_size=GROUP_SIZE)
        np.testing.assert_array_equal(original, recovered)


class TestEdgeCases:
    def test_constant_tensor(self, adapter: TRTQuantizationAdapter) -> None:
        """Constant values (zero range) should not cause division by zero."""
        constant = np.full(GROUP_SIZE, 3.14, dtype=np.float16)
        weights, scales, biases = adapter.quantize(constant, bits=8, group_size=GROUP_SIZE)
        recovered = adapter.dequantize(weights, scales, biases, bits=8, group_size=GROUP_SIZE)
        recovered = recovered[: constant.size]
        assert not np.any(np.isnan(recovered))
        assert not np.any(np.isinf(recovered))

    def test_unsupported_bits_quantize(self, adapter: TRTQuantizationAdapter) -> None:
        with pytest.raises(ModelSpecValidationError, match="Unsupported bits"):
            adapter.quantize(_make_fp16_tensor(64), bits=3, group_size=GROUP_SIZE)

    def test_unsupported_bits_dequantize(self, adapter: TRTQuantizationAdapter) -> None:
        with pytest.raises(ModelSpecValidationError, match="Unsupported bits"):
            adapter.dequantize(
                np.zeros(32, dtype=np.uint8),
                np.ones(1, dtype=np.float16),
                np.zeros(1, dtype=np.float16),
                bits=3,
                group_size=GROUP_SIZE,
            )

    def test_non_multiple_of_group_size(self, adapter: TRTQuantizationAdapter) -> None:
        """Tensor size not divisible by group_size should be padded."""
        original = _make_fp16_tensor(100)  # 100 % 64 != 0
        weights, scales, biases = adapter.quantize(original, bits=8, group_size=GROUP_SIZE)
        recovered = adapter.dequantize(weights, scales, biases, bits=8, group_size=GROUP_SIZE)
        recovered = recovered[: original.size]
        error = np.abs(original.astype(np.float32) - recovered.astype(np.float32))
        assert error.max() < 0.02


class TestPackUnpack4bit:
    def test_pack_unpack_round_trip(self) -> None:
        data = np.array([0, 15, 7, 3, 1, 14, 8, 2], dtype=np.uint8)
        packed = TRTQuantizationAdapter._pack_4bit(data)
        unpacked = TRTQuantizationAdapter._unpack_4bit(packed)
        np.testing.assert_array_equal(data, unpacked[: len(data)])

    def test_pack_size(self) -> None:
        data = np.arange(16, dtype=np.uint8) % 16
        packed = TRTQuantizationAdapter._pack_4bit(data)
        assert packed.shape[0] == 8  # 16 values -> 8 bytes
