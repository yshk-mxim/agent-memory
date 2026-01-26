"""MLX implementation of ModelLoaderPort.

Provides model loading and memory management via Apple's MLX framework.
"""

import logging
from typing import Any

import mlx.core as mx
from mlx_lm import load

logger = logging.getLogger(__name__)


class MLXModelLoader:
    """MLX-based model loader for Apple Silicon.

    Implements ModelLoaderPort using MLX framework for optimized
    inference on M-series chips.

    Memory Management:
        - Uses unified memory architecture
        - Memory reclamation via del + gc.collect() + mx.clear_cache()
        - Validated by EXP-011 (100% reclamation)
    """

    def load_model(self, model_id: str) -> tuple[Any, Any]:
        """Load a model using MLX.

        Args:
            model_id: HuggingFace model ID

        Returns:
            Tuple of (MLX model, tokenizer)

        Raises:
            Exception: If model load fails
        """
        logger.debug(f"MLX: Loading model {model_id}")

        model, tokenizer = load(
            model_id,
            tokenizer_config={"trust_remote_code": True},
        )

        logger.debug(f"MLX: Model loaded successfully: {model_id}")
        return model, tokenizer

    def get_active_memory(self) -> int:
        """Get MLX active memory in bytes.

        Returns:
            Bytes currently allocated by MLX
        """
        return mx.get_active_memory()

    def clear_cache(self) -> None:
        """Clear MLX memory cache.

        Frees cached allocations to enable 100% memory reclamation.
        """
        mx.clear_cache()
        logger.debug("MLX: Cache cleared")
