"""Domain value objects (immutable data structures).

Value objects are immutable data structures that represent concepts
from the domain model. They have no identity - two instances with
the same values are considered equal.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from semantic.domain.entities import BLOCK_SIZE_TOKENS
from semantic.domain.errors import ModelSpecValidationError

if TYPE_CHECKING:
    from semantic.domain.entities import AgentBlocks


# Layer Type Detection Strategy Protocol (Issue #10, Sprint 3.5)


class LayerTypeDetectionStrategy(Protocol):
    """Strategy for detecting layer attention types for a specific model architecture.

    This protocol defines the interface for model-specific layer type detection
    strategies. Each concrete strategy implements detection logic for a specific
    model family (Gemma, Llama, Qwen, etc.).

    Created to remove hardcoded model-type conditionals (Issue #10, Sprint 3.5).
    """

    def detect_layer_types(
        self, model: Any, args: Any, n_layers: int
    ) -> list[str] | None:
        """Detect layer types for this model architecture.

        Args:
            model: Loaded MLX model.
            args: Model args/config.
            n_layers: Total number of layers.

        Returns:
            List of layer type strings ("global" or "sliding_window"),
            or None if this strategy cannot detect for this model.
        """
        ...


class Gemma3DetectionStrategy:
    """Layer type detection strategy for Gemma 3 models.

    Gemma 3 12B uses hybrid attention: 8 global + 40 sliding window layers.
    Per EXP-001, layer_types attribute is not present, requiring heuristic.
    """

    def detect_layer_types(
        self, _model: Any, args: Any, n_layers: int
    ) -> list[str] | None:
        """Detect Gemma 3 hybrid layer pattern (8 global + 40 sliding window).

        Args:
            _model: Loaded MLX model (unused, but required by protocol).
            args: Model args/config.
            n_layers: Total number of layers.

        Returns:
            Hybrid layer type list, or None if not Gemma 3.
        """
        model_type = getattr(args, "model_type", "unknown")
        if model_type == "gemma3":
            # Gemma 3 12B: 8 global + 40 sliding window (EXP-001)
            return ["global"] * 8 + ["sliding_window"] * (n_layers - 8)
        return None


class UniformAttentionDetectionStrategy:
    """Layer type detection strategy for models with uniform attention.

    Applies to: Llama, Qwen, and most standard transformer models.
    All layers use global (full) attention.
    """

    def detect_layer_types(
        self, _model: Any, _args: Any, n_layers: int
    ) -> list[str] | None:
        """Detect uniform global attention (default for most models).

        Args:
            _model: Loaded MLX model (unused, but required by protocol).
            _args: Model args/config (unused, but required by protocol).
            n_layers: Total number of layers.

        Returns:
            List of all "global" layer types.
        """
        # Default strategy: all layers use global attention
        return ["global"] * n_layers


@dataclass(frozen=True)
class GenerationResult:
    """Result of a text generation operation.

    Attributes:
        text: Generated text string.
        tokens: List of generated token IDs.
        cache: Updated KV cache after generation (per-layer).
            Sprint 2.5 fix: Each layer has (K, V) tensor tuple.
    """

    text: str
    tokens: list[int]
    cache: list[tuple[Any, Any]]  # Sprint 2.5 fix: More specific than list[Any]


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
            raise ModelSpecValidationError(
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
            ModelSpecValidationError: If required attributes are missing or invalid.
            AttributeError: If model lacks `args` attribute.

        Note:
            Block size is fixed to 256 tokens per ADR-002.
            Layer types are detected using a three-tier approach:
            1. Check layer_types attribute (most reliable)
            2. Inspect layer objects for use_sliding
            3. Fallback to model-type heuristics
        """
        args = model.args

        # Extract attributes (handles Gemma 3 nested config vs standard models)
        attrs = cls._extract_model_attributes(args)

        # Validate all required attributes are present
        cls._validate_model_attributes(attrs)

        # Compute head dimension from attention parameters
        head_dim = cls._compute_head_dim(attrs["hidden_size"], attrs["num_attention_heads"])

        # Detect layer types using three-tier approach
        layer_types = cls._detect_layer_types(model, args, attrs["n_layers"])

        return cls(
            n_layers=attrs["n_layers"],
            n_kv_heads=attrs["n_kv_heads"],
            head_dim=head_dim,
            block_tokens=BLOCK_SIZE_TOKENS,  # Universal constant per ADR-002
            layer_types=layer_types,
            sliding_window_size=attrs["sliding_window"],
        )

    @staticmethod
    def _extract_model_attributes(args: Any) -> dict[str, Any]:
        """Extract model attributes, handling Gemma 3 nested config.

        Args:
            args: Model args/config object.

        Returns:
            Dictionary with extracted attributes:
            - n_layers: Number of hidden layers
            - n_kv_heads: Number of KV attention heads
            - num_attention_heads: Total attention heads
            - hidden_size: Hidden layer dimension
            - sliding_window: Sliding window size (optional)
            - model_type: Model type string

        Note:
            Gemma 3 uses nested text_config dict, while standard models
            (Llama, Qwen) use flat args structure.
        """
        if hasattr(args, "text_config"):
            # Gemma 3: nested config dict
            config = args.text_config
            return {
                "n_layers": config.get("num_hidden_layers"),
                "n_kv_heads": config.get("num_key_value_heads"),
                "num_attention_heads": config.get("num_attention_heads"),
                "hidden_size": config.get("hidden_size"),
                "sliding_window": config.get("sliding_window", None),
                "model_type": getattr(args, "model_type", "unknown"),
            }

        # Standard models (Qwen, Llama, MoE)
        return {
            "n_layers": getattr(args, "num_hidden_layers", None),
            "n_kv_heads": getattr(args, "num_key_value_heads", None),
            "num_attention_heads": getattr(args, "num_attention_heads", None),
            "hidden_size": getattr(args, "hidden_size", None),
            "sliding_window": getattr(args, "sliding_window", None),
            "model_type": getattr(args, "model_type", "unknown"),
        }

    @staticmethod
    def _validate_model_attributes(attrs: dict[str, Any]) -> None:
        """Validate that all required model attributes are present.

        Args:
            attrs: Extracted attribute dictionary from _extract_model_attributes.

        Raises:
            ModelSpecValidationError: If any required attribute is missing.
        """
        if attrs["n_layers"] is None:
            raise ModelSpecValidationError("Cannot extract num_hidden_layers from model.args")
        if attrs["n_kv_heads"] is None:
            raise ModelSpecValidationError("Cannot extract num_key_value_heads from model.args")
        if attrs["hidden_size"] is None or attrs["num_attention_heads"] is None:
            raise ModelSpecValidationError(
                "Cannot compute head_dim: missing hidden_size or num_attention_heads"
            )

    @staticmethod
    def _compute_head_dim(hidden_size: int, num_attention_heads: int) -> int:
        """Compute attention head dimension.

        Args:
            hidden_size: Model hidden layer dimension.
            num_attention_heads: Total number of attention heads.

        Returns:
            Head dimension (hidden_size // num_attention_heads).

        Note:
            Always compute head_dim rather than relying on model attribute,
            as not all models expose head_dim directly.
        """
        return hidden_size // num_attention_heads

    @staticmethod
    def _detect_layer_types(model: Any, args: Any, n_layers: int) -> list[str]:
        """Detect layer attention types (global vs sliding_window).

        Three-tier detection approach (EXP-001 findings):
        1. Check layer_types attribute (most reliable)
        2. Inspect layer objects for use_sliding attribute
        3. Delegate to model-specific detection strategies

        Args:
            model: Loaded MLX model.
            args: Model args/config.
            n_layers: Total number of layers.

        Returns:
            List of layer type strings, one per layer.
            Values: "global" or "sliding_window".

        Note:
            Uses Strategy pattern (Issue #10, Sprint 3.5) to support
            model-specific detection without hardcoded conditionals.
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

        # Tier 3: Model-specific detection strategies (Issue #10 fix)
        # Try each strategy in priority order (most specific first)
        strategies: list[LayerTypeDetectionStrategy] = [
            Gemma3DetectionStrategy(),  # Hybrid models (Gemma 3)
            UniformAttentionDetectionStrategy(),  # Default fallback
        ]

        for strategy in strategies:
            result = strategy.detect_layer_types(model, args, n_layers)
            if result is not None:
                return result

        # Should never reach here (UniformAttentionDetectionStrategy always returns)
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


@dataclass(frozen=True)
class CompletedGeneration:
    """Result of a completed async generation request.

    This value object represents the outcome of a batch generation
    that has finished (either completed naturally or hit a limit).
    Used by GenerationEnginePort.step() to yield completed sequences.

    Attributes:
        uid: Unique identifier for this generation request.
        text: Generated text string.
        blocks: AgentBlocks containing updated cache after generation.
        finish_reason: Reason generation stopped ("stop", "length", "error").
        token_count: Total tokens generated.

    Note:
        Distinct from GenerationResult which is synchronous. CompletedGeneration
        is for async/batching scenarios where requests are submitted and polled.
    """

    uid: str
    text: str
    blocks: "AgentBlocks"  # Sprint 2.5 fix: Use string annotation for forward reference
    finish_reason: str
    token_count: int
