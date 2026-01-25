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
        Gemma 3 12B has 8 global layers followed by 40 sliding window layers::

            spec = ModelCacheSpec.from_model(model)
            assert spec.n_layers == 48
            assert spec.layer_types[0] == "global"
            assert spec.layer_types[8] == "sliding_window"
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
        2. Qwen1.5-MoE-A2.7B: Uniform full attention (24 global layers)
        3. Qwen 2.5-14B: Uniform full attention (48 global layers)
        4. Llama 3.1-8B: Uniform full attention (32 global layers)

        Implementation based on EXP-001 findings (project/experiments/EXP-001-model-args.md).

        Args:
            model: Loaded MLX model from mlx_lm.load() with `args` attribute.
                Expected attributes:
                - num_hidden_layers (layer count)
                - num_key_value_heads (KV head count)
                - hidden_size, num_attention_heads (for computing head_dim)
                - sliding_window (optional, window size)
                - text_config (Gemma 3 only, nested config dict)

        Returns:
            ModelCacheSpec with extracted geometry.

        Raises:
            ValueError: If required attributes are missing or invalid.
            AttributeError: If model lacks `args` attribute.

        Note:
            Block size is fixed to 256 tokens per ADR-002.
            Layer types are detected using a three-tier approach:
            1. Check layer_types attribute (most reliable)
            2. Inspect layer objects for use_sliding
            3. Fallback to model-type heuristics
        """
        args = model.args

        # Step 1: Extract basic attributes (handle Gemma 3 nested config)
        if hasattr(args, "text_config"):
            # Gemma 3: nested config dict
            config = args.text_config
            n_layers = config.get("num_hidden_layers")
            n_kv_heads = config.get("num_key_value_heads")
            num_attention_heads = config.get("num_attention_heads")
            hidden_size = config.get("hidden_size")
            sliding_window = config.get("sliding_window", None)
            model_type = getattr(args, "model_type", "unknown")
        else:
            # Standard models (Qwen, Llama, MoE)
            n_layers = getattr(args, "num_hidden_layers", None)
            n_kv_heads = getattr(args, "num_key_value_heads", None)
            num_attention_heads = getattr(args, "num_attention_heads", None)
            hidden_size = getattr(args, "hidden_size", None)
            sliding_window = getattr(args, "sliding_window", None)
            model_type = getattr(args, "model_type", "unknown")

        # Step 2: Validate required attributes
        if n_layers is None:
            raise ValueError("Cannot extract num_hidden_layers from model.args")
        if n_kv_heads is None:
            raise ValueError("Cannot extract num_key_value_heads from model.args")
        if hidden_size is None or num_attention_heads is None:
            raise ValueError(
                "Cannot compute head_dim: missing hidden_size or num_attention_heads"
            )

        # Step 3: Compute head dimension (ALWAYS compute, never rely on attribute)
        head_dim = hidden_size // num_attention_heads

        # Step 4: Detect layer types (three-tier detection)
        layer_types = cls._detect_layer_types(model, args, model_type, n_layers)

        return cls(
            n_layers=n_layers,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            block_tokens=256,  # Fixed per ADR-002
            layer_types=layer_types,
            sliding_window_size=sliding_window,
        )

    @staticmethod
    def _detect_layer_types(
        model: Any, args: Any, model_type: str, n_layers: int
    ) -> list[str]:
        """Detect layer attention types (global vs sliding_window).

        Three-tier detection approach (EXP-001 findings):
        1. Check layer_types attribute (most reliable)
        2. Inspect layer objects for use_sliding
        3. Fallback to model-type heuristics

        Args:
            model: Loaded MLX model.
            args: Model args/config.
            model_type: Model type string (e.g., 'gemma3', 'llama').
            n_layers: Total number of layers.

        Returns:
            List of layer type strings, one per layer.
            Values: "global" or "sliding_window".

        Note:
            Per EXP-001, Gemma 3 requires heuristic detection as
            layer_types and use_sliding attributes are not present.
        """
        # Tier 1: Check layer_types attribute (e.g., some Llama variants)
        if hasattr(args, "layer_types") and args.layer_types:
            return list(args.layer_types)

        # Tier 2: Inspect layer objects for use_sliding attribute
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            layers = model.model.layers
            detected_types: list[str] = []

            for layer in layers:
                if hasattr(layer, "use_sliding"):
                    layer_type = "sliding_window" if layer.use_sliding else "global"
                    detected_types.append(layer_type)

            if len(detected_types) == n_layers:
                return detected_types

        # Tier 3: Model-type heuristics (for known hybrid models)
        if model_type == "gemma3":
            # Gemma 3 12B: 8 global + 40 sliding window (EXP-001)
            return ["global"] * 8 + ["sliding_window"] * (n_layers - 8)

        # Default: Uniform full attention
        return ["global"] * n_layers

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
