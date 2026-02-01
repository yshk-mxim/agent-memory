"""MLX model specification extraction adapter."""

from typing import Any

from semantic.domain.entities import BLOCK_SIZE_TOKENS
from semantic.domain.errors import ModelSpecValidationError
from semantic.domain.value_objects import ModelCacheSpec


# Layer Type Detection Strategy Protocol
class LayerTypeDetectionStrategy:
    """Strategy for detecting layer attention types for a model architecture."""

    def detect_layer_types(self, model: Any, args: Any, n_layers: int) -> list[str] | None:
        """Detect layer types for this model architecture.

        Args:
            model: Loaded MLX model.
            args: Model args/config.
            n_layers: Total number of layers.

        Returns:
            List of layer type strings, or None if cannot detect.
        """
        raise NotImplementedError


class Gemma3DetectionStrategy(LayerTypeDetectionStrategy):
    """Layer type detection for Gemma 3 models (hybrid attention)."""

    GEMMA3_GLOBAL_LAYERS = 8  # First 8 layers use global attention

    def detect_layer_types(self, _model: Any, args: Any, n_layers: int) -> list[str] | None:
        """Detect Gemma 3 hybrid layer pattern."""
        model_type = getattr(args, "model_type", "unknown")
        if model_type == "gemma3":
            return ["global"] * self.GEMMA3_GLOBAL_LAYERS + ["sliding_window"] * (
                n_layers - self.GEMMA3_GLOBAL_LAYERS
            )
        return None


class UniformAttentionDetectionStrategy(LayerTypeDetectionStrategy):
    """Layer type detection for models with uniform attention."""

    def detect_layer_types(self, _model: Any, _args: Any, n_layers: int) -> list[str] | None:
        """Detect uniform global attention (default)."""
        return ["global"] * n_layers


class MLXModelSpecExtractor:
    """Extracts ModelCacheSpec from loaded MLX models."""

    def __init__(self) -> None:
        """Initialize with detection strategies."""
        self._strategies: list[LayerTypeDetectionStrategy] = [
            Gemma3DetectionStrategy(),
            UniformAttentionDetectionStrategy(),
        ]

    def extract_spec(self, model: Any) -> ModelCacheSpec:
        """Extract cache specification from a loaded MLX model."""
        args = model.args

        attrs = self._extract_model_attributes(args)
        self._validate_model_attributes(attrs)

        head_dim = self._compute_head_dim(attrs["hidden_size"], attrs["num_attention_heads"])
        layer_types = self._detect_layer_types(model, args, attrs["n_layers"])

        return ModelCacheSpec(
            n_layers=attrs["n_layers"],
            n_kv_heads=attrs["n_kv_heads"],
            head_dim=head_dim,
            block_tokens=BLOCK_SIZE_TOKENS,
            layer_types=layer_types,
            sliding_window_size=attrs["sliding_window"],
        )

    def _extract_model_attributes(self, args: Any) -> dict[str, Any]:
        """Extract model attributes, handling Gemma 3 nested config."""
        if hasattr(args, "text_config"):
            config = args.text_config
            return {
                "n_layers": config.get("num_hidden_layers"),
                "n_kv_heads": config.get("num_key_value_heads"),
                "num_attention_heads": config.get("num_attention_heads"),
                "hidden_size": config.get("hidden_size"),
                "sliding_window": config.get("sliding_window", None),
                "model_type": getattr(args, "model_type", "unknown"),
            }

        return {
            "n_layers": getattr(args, "num_hidden_layers", None),
            "n_kv_heads": getattr(args, "num_key_value_heads", None),
            "num_attention_heads": getattr(args, "num_attention_heads", None),
            "hidden_size": getattr(args, "hidden_size", None),
            "sliding_window": getattr(args, "sliding_window", None),
            "model_type": getattr(args, "model_type", "unknown"),
        }

    def _validate_model_attributes(self, attrs: dict[str, Any]) -> None:
        """Validate required model attributes are present."""
        if attrs["n_layers"] is None:
            raise ModelSpecValidationError("Cannot extract num_hidden_layers from model.args")
        if attrs["n_kv_heads"] is None:
            raise ModelSpecValidationError("Cannot extract num_key_value_heads from model.args")
        if attrs["hidden_size"] is None or attrs["num_attention_heads"] is None:
            raise ModelSpecValidationError(
                "Cannot compute head_dim: missing hidden_size or num_attention_heads"
            )

    def _compute_head_dim(self, hidden_size: int, num_attention_heads: int) -> int:
        """Compute attention head dimension."""
        return hidden_size // num_attention_heads

    def _detect_layer_types(self, model: Any, args: Any, n_layers: int) -> list[str]:
        """Detect layer attention types using three-tier approach."""
        # Tier 1: Check layer_types attribute
        if hasattr(args, "layer_types") and args.layer_types:
            return list(args.layer_types)

        # Tier 2: Inspect layer objects
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            layers = model.model.layers
            detected_types: list[str] = []

            for layer in layers:
                if hasattr(layer, "use_sliding"):
                    layer_type = "sliding_window" if layer.use_sliding else "global"
                    detected_types.append(layer_type)

            if len(detected_types) == n_layers:
                return detected_types

        # Tier 3: Strategy-based detection
        for strategy in self._strategies:
            result = strategy.detect_layer_types(model, args, n_layers)
            if result is not None:
                return result

        return ["global"] * n_layers


# Singleton for convenience
_extractor: MLXModelSpecExtractor | None = None


def get_extractor() -> MLXModelSpecExtractor:
    """Get the singleton MLXModelSpecExtractor instance."""
    global _extractor
    if _extractor is None:
        _extractor = MLXModelSpecExtractor()
    return _extractor
