# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Yakov Shkolnikov and contributors
"""TRT quantization adapter.

Implements CacheQuantizationPort using numpy for Q4/Q8 <-> FP16
conversion.  Uses the same asymmetric min-max algorithm as MLX's
``mx.quantize()`` for disk-format compatibility: both backends
produce identical Q4 safetensors files.

On Thor with CUDA available, a future optimization could use
torch/cupy for GPU-accelerated quantization (~10x faster for
large caches).  The numpy path is ~1-2ms for typical cache sizes
and is used as the portable default.
"""

import numpy as np
from numpy.typing import NDArray

from agent_memory.domain.errors import ModelSpecValidationError

_SUPPORTED_BITS = {4, 8, 16}
_BITS_FP16 = 16
_BITS_Q4 = 4


class TRTQuantizationAdapter:
    """Asymmetric min-max quantization compatible with MLX Q4/Q8 format.

    Quantization formula (per group of ``group_size`` elements):
        scale = (max - min) / (2^bits - 1)
        bias  = min
        quant = round((x - bias) / scale)

    Dequantization:
        x_approx = quant * scale + bias
    """

    @property
    def supported_bits(self) -> set[int]:
        """Return supported quantization bit widths."""
        return _SUPPORTED_BITS

    def quantize(
        self,
        tensor: NDArray[np.float16],
        bits: int,
        group_size: int,
    ) -> tuple[NDArray[np.uint8], NDArray[np.float16], NDArray[np.float16]]:
        """Quantize FP16 tensor to Q4 or Q8.

        Args:
            tensor: Input FP16 tensor (any shape, last dim divisible by group_size).
            bits: 4 or 8.
            group_size: Elements per quantization group.

        Returns:
            (weights, scales, biases) tuple.
        """
        if bits not in _SUPPORTED_BITS:
            raise ModelSpecValidationError(
                f"Unsupported bits={bits}, must be one of {_SUPPORTED_BITS}"
            )
        if bits == _BITS_FP16:
            # No-op: return tensor as "weights" with unit scales and zero biases
            flat = tensor.reshape(-1)
            n_groups = (flat.shape[0] + group_size - 1) // group_size
            scales = np.ones(n_groups, dtype=np.float16)
            biases = np.zeros(n_groups, dtype=np.float16)
            return tensor.view(np.uint8), scales, biases

        return self._quantize_asymmetric(tensor, bits, group_size)

    def dequantize(
        self,
        weights: NDArray[np.uint8],
        scales: NDArray[np.float16],
        biases: NDArray[np.float16],
        bits: int,
        group_size: int,
    ) -> NDArray[np.float16]:
        """Dequantize Q4/Q8 back to FP16.

        Args:
            weights: Packed quantized weights.
            scales: Per-group scales (float16).
            biases: Per-group biases (float16).
            bits: Quantization bit width.
            group_size: Elements per group.

        Returns:
            Dequantized FP16 tensor.
        """
        if bits not in _SUPPORTED_BITS:
            raise ModelSpecValidationError(
                f"Unsupported bits={bits}, must be one of {_SUPPORTED_BITS}"
            )
        if bits == _BITS_FP16:
            return weights.view(np.float16)

        return self._dequantize_asymmetric(weights, scales, biases, bits, group_size)

    def _quantize_asymmetric(
        self,
        tensor: NDArray[np.float16],
        bits: int,
        group_size: int,
    ) -> tuple[NDArray[np.uint8], NDArray[np.float16], NDArray[np.float16]]:
        """Core asymmetric min-max quantization."""
        flat = tensor.astype(np.float32).reshape(-1)
        n_elements = flat.shape[0]

        # Pad to multiple of group_size
        remainder = n_elements % group_size
        if remainder != 0:
            pad_size = group_size - remainder
            flat = np.concatenate([flat, np.zeros(pad_size, dtype=np.float32)])

        n_groups = flat.shape[0] // group_size
        grouped = flat.reshape(n_groups, group_size)

        # Per-group min/max
        mins = grouped.min(axis=1)
        maxs = grouped.max(axis=1)

        max_val = (1 << bits) - 1
        ranges = maxs - mins
        # Avoid division by zero for constant groups
        ranges = np.where(ranges == 0, 1.0, ranges)
        scales = (ranges / max_val).astype(np.float16)
        biases_arr = mins.astype(np.float16)

        # Quantize
        scales_f32 = scales.astype(np.float32)
        biases_f32 = biases_arr.astype(np.float32)
        quantized = (
            np.round((grouped - biases_f32[:, np.newaxis]) / scales_f32[:, np.newaxis])
            .clip(0, max_val)
            .astype(np.uint8)
        )

        packed = (
            self._pack_4bit(quantized.reshape(-1)) if bits == _BITS_Q4 else quantized.reshape(-1)
        )

        return packed, scales, biases_arr

    def _dequantize_asymmetric(
        self,
        weights: NDArray[np.uint8],
        scales: NDArray[np.float16],
        biases: NDArray[np.float16],
        bits: int,
        group_size: int,
    ) -> NDArray[np.float16]:
        """Core asymmetric dequantization."""
        unpacked = self._unpack_4bit(weights) if bits == _BITS_Q4 else weights

        n_groups = scales.shape[0]
        total_elements = n_groups * group_size

        # Trim to actual group count
        unpacked = unpacked[:total_elements].reshape(n_groups, group_size)

        scales_f32 = scales.astype(np.float32)
        biases_f32 = biases.astype(np.float32)

        dequantized = (
            unpacked.astype(np.float32) * scales_f32[:, np.newaxis] + biases_f32[:, np.newaxis]
        )
        return dequantized.reshape(-1)[:total_elements].astype(np.float16)

    @staticmethod
    def _pack_4bit(data: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """Pack pairs of 4-bit values into single bytes."""
        if data.shape[0] % 2 != 0:
            data = np.concatenate([data, np.zeros(1, dtype=np.uint8)])
        low = data[0::2] & 0x0F
        high = (data[1::2] & 0x0F) << 4
        return (low | high).astype(np.uint8)

    @staticmethod
    def _unpack_4bit(packed: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """Unpack bytes into pairs of 4-bit values."""
        low = packed & 0x0F
        high = (packed >> 4) & 0x0F
        return np.stack([low, high], axis=1).reshape(-1)
