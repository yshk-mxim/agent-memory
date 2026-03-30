# mypy: disable-error-code="assignment,unused-ignore"
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Yakov Shkolnikov and contributors
"""MLX cache operations adapter."""

from typing import Any

from agent_memory.domain.errors import GenerationError


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

        CRITICAL: Supports both float tensors and quantized tuples.
        When quantized, concatenates components separately to avoid
        expensive dequantization overhead (100-500ms saved per cache hit!).

        Args:
            k_tensors: List of K cache blocks (mx.array or quantized tuples)
            v_tensors: List of V cache blocks (mx.array or quantized tuples)

        Returns:
            Tuple of (concatenated_k, concatenated_v)
            - Quantized input → quantized output (tuple)
            - Float input → float output (mx.array)
        """
        import mlx.core as mx  # Import at runtime to avoid issues in non-MLX environments

        # Check if blocks are quantized (tuple of weights, scales, biases)
        if k_tensors and isinstance(k_tensors[0], tuple) and len(k_tensors[0]) == 3:
            # QUANTIZED FORMAT: Concatenate components separately
            # This keeps cache quantized throughout, avoiding dequantization overhead

            k_weights_list = [k[0] for k in k_tensors]
            k_scales_list = [k[1] for k in k_tensors]
            k_biases_list = [k[2] for k in k_tensors]

            v_weights_list = [v[0] for v in v_tensors]
            v_scales_list = [v[1] for v in v_tensors]
            v_biases_list = [v[2] for v in v_tensors]

            # Validate bias consistency - all None or all not None
            k_has_biases = [b is not None for b in k_biases_list]
            v_has_biases = [b is not None for b in v_biases_list]
            if any(k_has_biases) != all(k_has_biases):
                raise GenerationError(
                    f"Inconsistent K biases: some blocks have biases, others don't. "
                    f"Has biases: {k_has_biases}"
                )
            if any(v_has_biases) != all(v_has_biases):
                raise GenerationError(
                    f"Inconsistent V biases: some blocks have biases, others don't. "
                    f"Has biases: {v_has_biases}"
                )

            # Concatenate each component along sequence axis (axis=2)
            k_weights = mx.concatenate(k_weights_list, axis=2)
            k_scales = mx.concatenate(k_scales_list, axis=2)
            k_biases = mx.concatenate(k_biases_list, axis=2) if all(k_has_biases) else None

            v_weights = mx.concatenate(v_weights_list, axis=2)
            v_scales = mx.concatenate(v_scales_list, axis=2)
            v_biases = mx.concatenate(v_biases_list, axis=2) if all(v_has_biases) else None

            # Return LAZY tensors — caller is responsible for mx.eval.
            # _reconstruct_cache batches all layers into a single mx.eval
            # call, eliminating 27 per-layer GPU sync fences.
            k_full = (k_weights, k_scales, k_biases)
            v_full = (v_weights, v_scales, v_biases)

            return k_full, v_full

        # FLOAT FORMAT: Normal concatenation with validation
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
                        f"K/V shape mismatch in block {i}: K={k_t.shape[:2]}, V={v_t.shape[:2]}"
                    )

        # Concatenate along sequence axis to form shape [n_kv_heads, head_dim, total_seq_len]
        k_full = mx.concatenate(k_tensors, axis=2)
        v_full = mx.concatenate(v_tensors, axis=2)

        # Force evaluation (MLX lazy evaluation)
        mx.eval(k_full, v_full)

        return k_full, v_full

    def get_sequence_length(self, k_tensor: Any) -> int:
        """Extract sequence length from K tensor.

        For KVCache tensors (4D: batch, heads, seq, dim): returns shape[2].
        For ArraysCache state tensors (3D: batch, heads, state_dim): returns 0
        (these are SSM states, not KV caches — no sequence axis).

        Args:
            k_tensor: K tensor from cache.state[0].

        Returns:
            Sequence length, or 0 for non-KV cache layers.
        """
        if k_tensor is None:
            return 0

        # Quantized tuple (weights, scales, biases)
        if isinstance(k_tensor, tuple) and len(k_tensor) == 3:
            return int(k_tensor[0].shape[2])

        # 4D tensor (batch, heads, seq_len, head_dim) — KVCache
        ndim = 4
        if hasattr(k_tensor, "ndim") and k_tensor.ndim == ndim:
            return int(k_tensor.shape[2])

        # 3D tensor (batch, heads, state_dim) — ArraysCache SSM state, NOT seq_len
        return 0

    def slice_cache_tensor(
        self,
        tensor: Any,
        start_token: int,
        end_token: int,
    ) -> Any:
        """Slice cache tensor along sequence axis.

        Handles both float tensors and quantized tuples (weights, scales, biases).
        Validates bounds to prevent out-of-range slicing.
        """
        # Check if quantized (tuple of 3 tensors)
        if isinstance(tensor, tuple) and len(tensor) == 3:
            # Quantized format - slice each component
            weights, scales, biases = tensor

            # Validate bounds against weights tensor (authoritative size)
            seq_len = weights.shape[2]
            if start_token < 0 or end_token > seq_len or start_token > end_token:
                raise GenerationError(
                    f"Invalid slice bounds [{start_token}:{end_token}] for tensor "
                    f"with sequence length {seq_len}"
                )

            weights_slice = weights[:, :, start_token:end_token]
            scales_slice = scales[:, :, start_token:end_token]
            biases_slice = biases[:, :, start_token:end_token] if biases is not None else None
            return (weights_slice, scales_slice, biases_slice)

        # 4D float tensor (batch, heads, seq, dim) — KVCache
        ndim_kv = 4
        if hasattr(tensor, "ndim") and tensor.ndim == ndim_kv:
            seq_len = tensor.shape[2]
            if start_token < 0 or end_token > seq_len or start_token > end_token:
                raise GenerationError(
                    f"Invalid slice bounds [{start_token}:{end_token}] for tensor "
                    f"with sequence length {seq_len}"
                )
            return tensor[:, :, start_token:end_token, :]

        # 3D tensor (ArraysCache SSM state) — return as-is (no sequence axis)
        return tensor

    def get_cache_offset(self, layer_cache: Any) -> int:
        """Get the current sequence offset of a cache layer.

        Args:
            layer_cache: A single layer's cache object (KVCache or ArraysCache).

        Returns:
            Number of tokens currently in the cache.
        """
        if hasattr(layer_cache, "offset"):
            return int(layer_cache.offset)
        if hasattr(layer_cache, "size"):
            return int(layer_cache.size())
        return 0

    def trim_cache(self, kv_cache: list[Any], target_length: int) -> None:
        """Trim all cache layers to target sequence length.

        Args:
            kv_cache: List of cache objects (one per layer).
            target_length: Target sequence length.
        """
        import mlx.core as mx

        for layer_cache in kv_cache:
            current = self.get_cache_offset(layer_cache)
            if current > target_length and hasattr(layer_cache, "trim"):
                layer_cache.trim(current - target_length)

    def eval_cache_tensors(self, kv_cache: list[Any]) -> None:
        """Force evaluation of all cache tensors.

        Args:
            kv_cache: List of cache objects.
        """
        import mlx.core as mx

        tensors: list[Any] = []
        for c in kv_cache:
            if hasattr(c, "state"):
                st = c.state
                if isinstance(st, (list, tuple)):
                    tensors.extend(t for t in st if hasattr(t, "shape"))
        if tensors:
            mx.eval(*tensors)

    def extract_kv_from_cache(self, cache: list[Any]) -> list[tuple[Any, Any]]:
        """Extract (K, V) tensor pairs from cache layers.

        Works with both KVCache (.state returns [k, v]) and ArraysCache
        (.state returns SSM state tensors).

        Args:
            cache: List of cache layer objects.

        Returns:
            List of (k_tensor, v_tensor) per layer. For ArraysCache layers,
            returns the state tensors as a pair.
        """
        result = []
        for layer_cache in cache:
            st = layer_cache.state if hasattr(layer_cache, "state") else []
            if isinstance(st, (list, tuple)) and len(st) >= 2:
                result.append((st[0], st[1]))
            else:
                result.append((None, None))
        return result

    def create_batch_generator(
        self,
        model: Any,
        stop_tokens: set[int],
        kv_bits: int | None = 4,
        kv_group_size: int = 64,
    ) -> Any:
        """Create an MLX BatchGenerator for batched inference.

        Args:
            model: MLX model instance
            stop_tokens: Set of token IDs to stop generation
            kv_bits: KV cache quantization bits (4 or 8, None = FP16) -
                stored but not used by BatchGenerator
            kv_group_size: Quantization group size -
                stored but not used by BatchGenerator

        Returns:
            BatchGenerator instance

        Note:
            In mlx-lm 0.30.4+, BatchGenerator does not accept kv_bits parameters.
            We handle quantization manually by creating QuantizedKVCache objects
            when loading from disk (see batch_engine.py).
        """
        from mlx_lm.server import BatchGenerator  # type: ignore[attr-defined]

        return BatchGenerator(
            model=model,
            stop_tokens=stop_tokens,
            # NOTE: kv_bits/kv_group_size not supported in BatchGenerator API
            # We create QuantizedKVCache manually in batch_engine.py instead
        )

    def create_sampler(
        self,
        temperature: float = 0.0,
        top_p: float = 0.0,
        top_k: int = 0,
    ) -> Any:
        """Create an MLX sampler for token sampling."""
        from mlx_lm.sample_utils import make_sampler  # type: ignore[import-not-found]

        return make_sampler(temp=temperature, top_p=top_p, top_k=top_k)
