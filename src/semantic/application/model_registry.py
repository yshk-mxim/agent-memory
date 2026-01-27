"""Model lifecycle management with hot-swap support.

Manages loading, unloading, and tracking of ML models. Supports dynamic
model swapping while preserving agent caches on disk.
"""

import gc
import logging
from typing import Any

from semantic.adapters.outbound.mlx_spec_extractor import get_extractor
from semantic.application.ports import ModelLoaderPort
from semantic.domain.errors import ModelNotFoundError
from semantic.domain.value_objects import ModelCacheSpec

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Registry for managing model lifecycle and hot-swapping.

    Responsibilities:
    - Load models on demand via injected ModelLoaderPort
    - Track currently loaded model
    - Unload models and reclaim memory
    - Extract ModelCacheSpec for each model

    Thread Safety:
    - Not thread-safe. Caller must ensure exclusive access during swap.
    - Designed for single-threaded FastAPI lifespan management.

    Architecture:
    - Uses dependency injection to decouple from specific ML frameworks
    - ModelLoaderPort enables swapping MLX for PyTorch, ONNX, etc.

    Example:
        >>> from semantic.adapters.outbound.mlx_model_loader import MLXModelLoader
        >>> loader = MLXModelLoader()
        >>> registry = ModelRegistry(model_loader=loader)
        >>> registry.load_model("mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx")
        >>> spec = registry.get_current_spec()
        >>> registry.unload_model()  # 100% memory reclaimed (per EXP-011)
    """

    def __init__(self, model_loader: ModelLoaderPort) -> None:
        """Initialize registry with injected model loader.

        Args:
            model_loader: Implementation of ModelLoaderPort (e.g., MLXModelLoader)
        """
        self._loader = model_loader
        self._model: Any | None = None  # Framework model object
        self._tokenizer: Any | None = None  # Framework tokenizer object
        self._spec: ModelCacheSpec | None = None  # Cache spec
        self._current_model_id: str | None = None  # HuggingFace model ID

    def load_model(
        self,
        model_id: str,
        kv_bits: int | None = 4,
        kv_group_size: int = 64,
    ) -> tuple[Any, Any]:
        """Load a model and extract its cache spec.

        Args:
            model_id: HuggingFace model ID
                (e.g., "mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx")
            kv_bits: KV cache quantization bits (4 or 8, None = FP16)
            kv_group_size: Quantization group size

        Returns:
            Tuple of (model, tokenizer)

        Raises:
            ModelNotFoundError: If model cannot be loaded

        Example:
            >>> registry.load_model("mlx-community/SmolLM2-135M-Instruct")
            (<Model>, <Tokenizer>)
        """
        logger.info(f"Loading model: {model_id}")

        try:
            # Load model and tokenizer via injected loader (CR-3 fix)
            model, tokenizer = self._loader.load_model(model_id)
        except Exception as e:
            raise ModelNotFoundError(f"Failed to load model {model_id}: {e}") from e

        # Extract cache spec from model
        extractor = get_extractor()
        base_spec = extractor.extract_spec(model)

        # Add quantization settings from caller
        from dataclasses import replace
        spec = replace(
            base_spec,
            kv_bits=kv_bits,
            kv_group_size=kv_group_size,
        )

        logger.info(
            f"Model loaded: {spec.n_layers} layers, "
            f"{spec.n_kv_heads} KV heads, "
            f"{spec.head_dim} head dim, "
            f"kv_bits={spec.kv_bits}, "
            f"kv_group_size={spec.kv_group_size}"
        )

        # Store state
        self._model = model
        self._tokenizer = tokenizer
        self._spec = spec
        self._current_model_id = model_id

        return model, tokenizer

    def unload_model(self) -> None:
        """Unload current model and reclaim memory.

        Uses pattern from EXP-011: del + gc.collect() + loader.clear_cache()
        Achieves 100% memory reclamation.

        Notes:
            - Safe to call even if no model loaded
            - Memory reclaimed immediately (validated by EXP-011)
            - Framework-agnostic via ModelLoaderPort (CR-3 fix)
        """
        if self._model is None:
            logger.debug("No model to unload")
            return

        logger.info(f"Unloading model: {self._current_model_id}")

        # Measure memory before unload (for logging) via injected loader (CR-3 fix)
        mem_before_mb = self._loader.get_active_memory() / (1024 ** 2)

        # Unload (EXP-011 pattern)
        del self._model
        del self._tokenizer
        gc.collect()
        self._loader.clear_cache()  # Framework-agnostic (CR-3 fix)

        # Clear state
        self._model = None
        self._tokenizer = None
        self._spec = None
        old_model_id = self._current_model_id
        self._current_model_id = None

        # Measure memory after unload
        mem_after_mb = self._loader.get_active_memory() / (1024 ** 2)
        reclaimed_mb = mem_before_mb - mem_after_mb

        logger.info(
            f"Model unloaded: {old_model_id}. "
            f"Reclaimed {reclaimed_mb:.2f} MB"
        )

    def get_current(self) -> tuple[Any, Any] | None:
        """Get currently loaded model and tokenizer.

        Returns:
            Tuple of (model, tokenizer) if loaded, None otherwise
        """
        if self._model is None:
            return None
        return (self._model, self._tokenizer)

    def get_current_spec(self) -> ModelCacheSpec | None:
        """Get ModelCacheSpec for currently loaded model.

        Returns:
            ModelCacheSpec if model loaded, None otherwise
        """
        return self._spec

    def get_current_id(self) -> str | None:
        """Get HuggingFace model ID of currently loaded model.

        Returns:
            Model ID string if loaded, None otherwise
        """
        return self._current_model_id

    def is_loaded(self) -> bool:
        """Check if a model is currently loaded.

        Returns:
            True if model loaded, False otherwise
        """
        return self._model is not None
