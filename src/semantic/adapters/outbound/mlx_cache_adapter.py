"""MLX cache operations adapter."""

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
        """Concatenate K/V tensors from multiple blocks along sequence axis."""
        import mlx.core as mx  # Import at runtime to avoid issues in non-MLX environments

        # Validate tensor shapes before concatenation
        if k_tensors:
            expected_k_shape = k_tensors[0].shape[:2]
            expected_v_shape = v_tensors[0].shape[:2]

            for i, (k_t, v_t) in enumerate(zip(k_tensors, v_tensors, strict=True)):
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

        # Concatenate along sequence axis to form shape [n_kv_heads, head_dim, total_seq_len]
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
        """Slice cache tensor along sequence axis."""
        return tensor[:, :, start_token:end_token]

    def create_batch_generator(
        self,
        model: Any,
        stop_tokens: set[int],
    ) -> Any:
        """Create an MLX BatchGenerator for batched inference."""
        from mlx_lm.server import BatchGenerator  # type: ignore[attr-defined]

        return BatchGenerator(model=model, stop_tokens=stop_tokens)

    def create_sampler(self, temperature: float = 0.0) -> Any:
        """Create an MLX sampler for token sampling."""
        from mlx_lm.sample_utils import make_sampler  # type: ignore[import-not-found]

        return make_sampler(temp=temperature)
