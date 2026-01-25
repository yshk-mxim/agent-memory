"""MLX cache operations adapter (CRITICAL-1, Sprint 3.5).

Implements CacheOperationsPort for MLX backend, providing tensor
operations (concatenation, slicing, evaluation) needed for block-pool
cache management.

This adapter removes the architecture violation by moving all MLX-specific
code out of the application layer (batch_engine.py) into the adapters layer.
"""

from typing import Any

from semantic.domain.errors import GenerationError


class MLXCacheAdapter:
    """Adapter for MLX-specific cache tensor operations.

    Wraps MLX tensor operations with a clean interface that the application
    layer can use without depending on MLX directly.

    Example:
        >>> adapter = MLXCacheAdapter()
        >>> k_full, v_full = adapter.concatenate_cache_blocks(k_tensors, v_tensors)
        >>> seq_len = adapter.get_sequence_length(k_full)
    """

    def concatenate_cache_blocks(
        self,
        k_tensors: list[Any],
        v_tensors: list[Any],
    ) -> tuple[Any, Any]:
        """Concatenate K/V tensors from multiple blocks along sequence axis.

        Args:
            k_tensors: List of K tensors from blocks (each shape: [n_kv_heads, head_dim, block_tokens])
            v_tensors: List of V tensors from blocks (each shape: [n_kv_heads, head_dim, block_tokens])

        Returns:
            Tuple of (k_full, v_full) concatenated tensors.
            Shape: [n_kv_heads, head_dim, total_seq_len]

        Raises:
            GenerationError: If tensor shapes are incompatible.
        """
        import mlx.core as mx  # Import at runtime to avoid issues in non-MLX environments

        # Validate tensor shapes before concatenation
        if k_tensors:
            expected_k_shape = k_tensors[0].shape[:2]
            expected_v_shape = v_tensors[0].shape[:2]

            for i, (k_t, v_t) in enumerate(zip(k_tensors, v_tensors)):
                if k_t.shape[:2] != expected_k_shape:
                    raise GenerationError(
                        f"K tensor shape mismatch in block {i}: "
                        f"expected {expected_k_shape}, got {k_t.shape[:2]}"
                    )
                if v_t.shape[:2] != expected_v_shape:
                    raise GenerationError(
                        f"V tensor shape mismatch in block {i}: "
                        f"expected {expected_v_shape}, got {v_t.shape[:2]}"
                    )
                if k_t.shape[:2] != v_t.shape[:2]:
                    raise GenerationError(
                        f"K/V shape mismatch in block {i}: "
                        f"K={k_t.shape[:2]}, V={v_t.shape[:2]}"
                    )

        # Concatenate K and V tensors along sequence length axis (axis=2)
        # Shape: (n_kv_heads, head_dim, total_seq_len)
        k_full = mx.concatenate(k_tensors, axis=2)
        v_full = mx.concatenate(v_tensors, axis=2)

        # Force evaluation (MLX lazy evaluation)
        mx.eval(k_full, v_full)

        return k_full, v_full

    def get_sequence_length(self, k_tensor: Any) -> int:
        """Extract sequence length from K tensor.

        Args:
            k_tensor: K tensor with shape [n_kv_heads, head_dim, seq_len]

        Returns:
            Sequence length (axis=2 dimension).
        """
        return int(k_tensor.shape[2])  # Cast to int for type safety

    def slice_cache_tensor(
        self,
        tensor: Any,
        start_token: int,
        end_token: int,
    ) -> Any:
        """Slice cache tensor along sequence axis.

        Args:
            tensor: Cache tensor (K or V) with shape [n_kv_heads, head_dim, seq_len]
            start_token: Start index for slicing (inclusive)
            end_token: End index for slicing (exclusive)

        Returns:
            Sliced tensor with shape [n_kv_heads, head_dim, end_token - start_token]

        Notes:
            Slicing syntax [:, :, start:end] works on MLX arrays natively.
        """
        return tensor[:, :, start_token:end_token]
