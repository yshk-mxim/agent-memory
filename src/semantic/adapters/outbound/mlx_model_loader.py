"""MLX implementation of ModelLoaderPort.

Provides model loading and memory management via Apple's MLX framework.
"""

import logging
from typing import Any

import mlx.core as mx
from mlx_lm import load

logger = logging.getLogger(__name__)

# Maximum context length for tokenizer (100K tokens for Claude Code CLI)
MAX_CONTEXT_LENGTH = 100000


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
        """Load a model using MLX with extended context support.

        CRITICAL FIX: Override tokenizer model_max_length to support
        long context required by Claude Code CLI (18K+ tokens observed).

        DeepSeek-Coder-V2-Lite architecture supports 163K tokens via
        YaRN RoPE scaling, but default tokenizer limits to 16K.

        Args:
            model_id: HuggingFace model ID

        Returns:
            Tuple of (MLX model, tokenizer with extended context)

        Raises:
            Exception: If model load fails
        """
        logger.info(f"Loading model with extended context: {model_id}")

        # CRITICAL: Override tokenizer max length for long context
        tokenizer_config = {
            "model_max_length": MAX_CONTEXT_LENGTH,
            "truncation_side": "left",  # Keep recent tokens if needed
            "trust_remote_code": True,
        }

        model, tokenizer = load(model_id, tokenizer_config=tokenizer_config)

        # Verify configuration applied
        actual_max = tokenizer.model_max_length
        logger.info(f"Model loaded: {model_id}")
        logger.info(f"Tokenizer max length: {actual_max:,} tokens")

        if actual_max < MAX_CONTEXT_LENGTH:
            logger.warning(
                f"Tokenizer max ({actual_max:,}) < target (100,000). Requests may be truncated."
            )

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
