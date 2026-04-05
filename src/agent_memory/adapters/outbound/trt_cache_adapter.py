# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Yakov Shkolnikov and contributors
"""TRT cache operations adapter.

Handles KV cache tensor operations for TensorRT Edge-LLM using numpy.
TRT layout: [numLayers, 2, numKVHeads, seqLen, headDim] as a single
stacked tensor.  Per-layer operations use the standard 4D layout:
[n_kv_heads, seq_len, head_dim] per K/V tensor.
"""

from typing import Any

import numpy as np
from numpy.typing import NDArray

from agent_memory.domain.errors import TRTLayoutError


class TRTCacheAdapter:
    """Adapter for TRT-specific cache tensor operations.

    Handles layout translation between per-layer KV tuples (used by
    the application layer) and the stacked 5D tensor format expected
    by TRT's ``instantiateKVCacheFromTensor`` /
    ``saveKVCacheIntoTensor`` kernels.
    """

    def concatenate_cache_blocks(
        self,
        k_tensors: list[Any],
        v_tensors: list[Any],
    ) -> tuple[NDArray[np.float16], NDArray[np.float16]]:
        """Concatenate K/V tensors from multiple blocks along sequence axis.

        Args:
            k_tensors: Per-block K tensors, each [n_kv_heads, seq_len, head_dim].
            v_tensors: Per-block V tensors, each [n_kv_heads, seq_len, head_dim].

        Returns:
            Tuple of (K, V) with concatenated sequence dimension.
        """
        if not k_tensors:
            raise TRTLayoutError("Cannot concatenate empty tensor list")
        k_cat = np.concatenate(k_tensors, axis=1)
        v_cat = np.concatenate(v_tensors, axis=1)
        return k_cat, v_cat

    def get_sequence_length(self, k_tensor: Any) -> int:
        """Extract sequence length from K tensor.

        Args:
            k_tensor: K tensor with shape [n_kv_heads, seq_len, head_dim].

        Returns:
            Sequence length (axis=1 dimension).
        """
        return int(k_tensor.shape[1])

    def slice_cache_tensor(
        self,
        tensor: Any,
        start_token: int,
        end_token: int,
    ) -> NDArray[np.float16]:
        """Slice cache tensor along sequence axis.

        Args:
            tensor: Cache tensor [n_kv_heads, seq_len, head_dim].
            start_token: Start index (inclusive).
            end_token: End index (exclusive).

        Returns:
            Sliced tensor [n_kv_heads, end-start, head_dim].
        """
        return tensor[:, start_token:end_token, :]  # type: ignore[no-any-return]

    def per_layer_to_stacked(
        self,
        layer_kv_pairs: list[tuple[NDArray[np.float16], NDArray[np.float16]]],
    ) -> NDArray[np.float16]:
        """Convert per-layer (K, V) pairs to TRT stacked 5D tensor.

        Args:
            layer_kv_pairs: List of (K, V) per layer, each [n_kv_heads, seq, head_dim].

        Returns:
            Stacked tensor [n_layers, 2, n_kv_heads, seq_len, head_dim].
        """
        if not layer_kv_pairs:
            raise TRTLayoutError("Cannot stack empty layer list")

        stacked_layers = []
        for k, v in layer_kv_pairs:
            # Stack K and V into [2, n_kv_heads, seq_len, head_dim]
            stacked_layers.append(np.stack([k, v], axis=0))

        # Stack all layers: [n_layers, 2, n_kv_heads, seq_len, head_dim]
        result = np.stack(stacked_layers, axis=0)
        expected_ndim = 5
        if result.ndim != expected_ndim:
            raise TRTLayoutError(
                f"Expected 5D stacked tensor, got {result.ndim}D with shape {result.shape}"
            )
        return result

    def stacked_to_per_layer(
        self,
        stacked: NDArray[np.float16],
    ) -> list[tuple[NDArray[np.float16], NDArray[np.float16]]]:
        """Convert TRT stacked 5D tensor to per-layer (K, V) pairs.

        Args:
            stacked: Tensor [n_layers, 2, n_kv_heads, seq_len, head_dim].

        Returns:
            List of (K, V) per layer, each [n_kv_heads, seq_len, head_dim].
        """
        expected_ndim = 5
        if stacked.ndim != expected_ndim:
            raise TRTLayoutError(
                f"Expected 5D stacked tensor, got {stacked.ndim}D with shape {stacked.shape}"
            )
        kv_axis_size = 2
        if stacked.shape[1] != kv_axis_size:
            raise TRTLayoutError(f"Expected axis 1 size 2 (K,V), got {stacked.shape[1]}")

        result = []
        for layer_idx in range(stacked.shape[0]):
            k = stacked[layer_idx, 0]  # [n_kv_heads, seq_len, head_dim]
            v = stacked[layer_idx, 1]
            result.append((k, v))
        return result
