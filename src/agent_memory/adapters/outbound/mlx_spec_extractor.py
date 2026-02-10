"""MLX model specification extraction adapter."""

from typing import Any

from agent_memory.domain.entities import BLOCK_SIZE_TOKENS
from agent_memory.domain.errors import ModelSpecValidationError
from agent_memory.domain.value_objects import ModelCacheSpec


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

        head_dim, v_head_dim = self._extract_head_dims(model, attrs)
        layer_types = self._detect_layer_types(model, args, attrs["n_layers"])

        return ModelCacheSpec(
            n_layers=attrs["n_layers"],
            n_kv_heads=attrs["n_kv_heads"],
            head_dim=head_dim,
            block_tokens=BLOCK_SIZE_TOKENS,
            layer_types=layer_types,
            sliding_window_size=attrs["sliding_window"],
            v_head_dim=v_head_dim,
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

    def _extract_head_dims(self, model: Any, attrs: dict[str, Any]) -> tuple[int, int | None]:
        """Extract K and V head dimensions from the model.

        Returns:
            (k_head_dim, v_head_dim) where v_head_dim is None when K=V
            (symmetric, the common case).

        Handles three cases:
        1. Standard models with attn.head_dim: K=V=head_dim → (head_dim, None)
        2. DeepSeek V2 MLA with asymmetric K/V: K=qk_nope+qk_rope, V=v_head_dim
           → (192, 128) for the Lite variant
        3. Fallback: hidden_size // num_attention_heads → (computed, None)
        """
        for path in [
            lambda m: m.language_model.model.layers,  # Gemma 3 (nested)
            lambda m: m.model.layers,                  # Standard models
            lambda m: m.layers,                        # Direct layer access
        ]:
            try:
                layers = path(model)
                if not layers or len(layers) == 0:
                    continue
                attn = getattr(layers[0], "self_attn", None)
                if attn is None:
                    continue

                # DeepSeek V2 MLA: K and V have different dimensions.
                # K = concat(k_nope[qk_nope_head_dim], k_pe[qk_rope_head_dim])
                # V = values[v_head_dim]
                qk_nope = getattr(attn, "qk_nope_head_dim", None)
                qk_rope = getattr(attn, "qk_rope_head_dim", None)
                v_dim = getattr(attn, "v_head_dim", None)
                if qk_nope is not None and qk_rope is not None and v_dim is not None:
                    k_dim = qk_nope + qk_rope
                    return (k_dim, v_dim if v_dim != k_dim else None)

                # Standard models: symmetric K=V
                if hasattr(attn, "head_dim"):
                    return (attn.head_dim, None)
            except (AttributeError, TypeError):
                continue

        # Fall back to computing from config
        return (attrs["hidden_size"] // attrs["num_attention_heads"], None)

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
