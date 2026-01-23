"""
MLX Utilities for Persistent Multi-Agent System

Extracted from semantic_isolation_mlx.py (archived January 23, 2026)
Provides basic MLX model management utilities.
"""

import mlx.core as mx
from mlx_lm import load
from typing import Dict
import logging

logger = logging.getLogger(__name__)


class MLXModelLoader:
    """Utilities for loading and managing MLX models"""

    @staticmethod
    def load_model(model_name: str = "mlx-community/gemma-3-12b-it-4bit"):
        """
        Load MLX model and tokenizer

        Args:
            model_name: MLX model identifier

        Returns:
            (model, tokenizer) tuple
        """
        logger.info(f"Loading model: {model_name}")
        model, tokenizer = load(model_name)
        logger.info("Model loaded successfully")
        return model, tokenizer

    @staticmethod
    def get_memory_usage() -> Dict[str, float]:
        """
        Get current MLX memory usage

        Returns:
            Dictionary with active_memory and peak_memory in GB
        """
        active_bytes = mx.metal.get_active_memory()
        peak_bytes = mx.metal.get_peak_memory()

        return {
            'active_memory_bytes': active_bytes,
            'peak_memory_bytes': peak_bytes,
            'active_memory_gb': active_bytes / (1024**3),
            'peak_memory_gb': peak_bytes / (1024**3),
        }

    @staticmethod
    def clear_cache():
        """Clear MLX Metal cache"""
        logger.info("Clearing MLX cache")
        mx.metal.clear_cache()

    @staticmethod
    def set_wired_limit(size_gb: int):
        """
        Set wired memory limit (macOS 15+ only)

        Args:
            size_gb: Memory size in GB to wire
        """
        if hasattr(mx, 'set_wired_limit'):
            logger.info(f"Setting wired memory limit: {size_gb}GB")
            mx.set_wired_limit(size_gb * 1024**3)
        else:
            logger.warning("mx.set_wired_limit not available (requires macOS 15+)")
