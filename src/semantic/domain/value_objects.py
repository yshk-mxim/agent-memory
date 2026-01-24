"""Domain value objects (immutable data structures).

Value objects are immutable data structures that represent concepts
from the domain model. They have no identity - two instances with
the same values are considered equal.
"""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GenerationResult:
    """Result of a text generation operation.

    Attributes:
        text: Generated text string.
        tokens: List of generated token IDs.
        cache: Updated KV cache after generation (per-layer).
    """

    text: str
    tokens: list[int]
    cache: list[Any]


@dataclass(frozen=True)
class ModelCacheSpec:
    """Specification for a model's KV cache geometry.

    This value object describes the structure of a model's KV cache,
    including layer count, attention pattern, and block requirements.

    Attributes:
        n_layers: Total number of transformer layers.
        n_kv_heads: Number of key-value attention heads per layer.
        head_dim: Dimension of each attention head.
        block_tokens: Number of tokens per cache block (universal: 256).
        layer_types: Type of each layer ("global" or "sliding_window").
        sliding_window_size: Window size for sliding window layers (e.g., 1024).

    Example:
        >>> # Gemma 3 12B: 8 global + 40 sliding window layers
        >>> spec = ModelCacheSpec.from_model(model)  # type: ignore[arg-type]
        >>> spec.n_layers
        48
        >>> spec.layer_types[:10]
        ['global', 'global', ..., 'sliding_window', ...]
    """

    n_layers: int
    n_kv_heads: int
    head_dim: int
    block_tokens: int
    layer_types: list[str]
    sliding_window_size: int | None = None

    def __post_init__(self) -> None:
        """Validate cache spec invariants."""
        if len(self.layer_types) != self.n_layers:
            raise ValueError(
                f"layer_types length ({len(self.layer_types)}) must equal "
                f"n_layers ({self.n_layers})"
            )

    @classmethod
    def from_model(cls, model: Any) -> "ModelCacheSpec":
        """Extract cache specification from a loaded model.

        Supports 4 model architectures:
        1. Gemma 3 12B: Hybrid (8 global + 40 sliding window, pattern=6)
        2. GPT-OSS-20B: MoE alternating (12 global + 12 sliding window)
        3. Qwen 2.5-14B: Uniform full attention (48 global layers)
        4. Llama 3.1-8B: Uniform full attention (32 global layers)

        Args:
            model: Loaded model object with `args` or `config` attribute.
                Expected attributes:
                - num_hidden_layers or n_layers
                - num_key_value_heads or n_kv_heads
                - head_dim or (hidden_size / n_heads)
                - sliding_window_pattern (optional, for hybrid models)
                - sliding_window (optional, window size)

        Returns:
            ModelCacheSpec with extracted geometry.

        Raises:
            ValueError: If required attributes are missing or invalid.
            AttributeError: If model has neither `args` nor `config`.

        Note:
            This method will be implemented by ML engineer after EXP-001
            validates the actual attribute names across all 4 models.
            Current implementation is a placeholder.
        """
        # Placeholder implementation - ML will fill this in after EXP-001
        raise NotImplementedError(
            "ModelCacheSpec.from_model() will be implemented after EXP-001 "
            "validates model.args attributes across all 4 target models. "
            "See project/experiments/EXP-001-model-args.md for findings."
        )

    def bytes_per_block_per_layer(self) -> int:
        """Calculate memory bytes required for one block at one layer.

        Formula: n_kv_heads × head_dim × 2 (K+V) × 2 (bytes, float16) × block_tokens

        Returns:
            Memory in bytes (e.g., Gemma 3: 2,097,152 bytes = 2 MB).

        Example:
            >>> spec = ModelCacheSpec(
            ...     n_layers=48,
            ...     n_kv_heads=8,
            ...     head_dim=256,
            ...     block_tokens=256,
            ...     layer_types=["global"] * 48,
            ... )
            >>> spec.bytes_per_block_per_layer()
            2097152  # 2 MB
        """
        return (
            self.n_kv_heads
            * self.head_dim
            * 2  # K and V
            * 2  # float16 = 2 bytes
            * self.block_tokens
        )

    def max_blocks_for_layer(self, layer_type: str) -> int | None:
        """Calculate maximum blocks needed for a layer.

        For sliding window layers, blocks are capped based on window size:
        - max_blocks = ceil(sliding_window_size / block_tokens)

        For global attention layers, there is no limit (None).

        Args:
            layer_type: Either "global" or "sliding_window".

        Returns:
            Maximum blocks for this layer type, or None for unlimited.

        Example:
            >>> spec = ModelCacheSpec(
            ...     n_layers=48,
            ...     n_kv_heads=8,
            ...     head_dim=256,
            ...     block_tokens=256,
            ...     layer_types=["sliding_window"] * 48,
            ...     sliding_window_size=512,
            ... )
            >>> spec.max_blocks_for_layer("sliding_window")
            2  # ceil(512 / 256) = 2
            >>> spec.max_blocks_for_layer("global")
            None  # No limit
        """
        if layer_type == "global":
            return None  # No limit for global attention

        if layer_type == "sliding_window" and self.sliding_window_size is not None:
            # Ceiling division: (window + block - 1) // block
            return (self.sliding_window_size + self.block_tokens - 1) // self.block_tokens

        return None


@dataclass(frozen=True)
class CacheKey:
    """Unique identifier for a cache entry.

    Attributes:
        agent_id: Unique agent identifier.
        model_id: Model identifier (for cache invalidation on model swap).
        prefix_hash: Hash of token prefix (for prefix matching).
    """

    agent_id: str
    model_id: str
    prefix_hash: str
