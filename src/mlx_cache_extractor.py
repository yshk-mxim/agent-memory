"""
MLX Cache Extractor

Wrapper around mlx_lm that exposes KV cache after generation.
Enables cache inspection, reuse, and persistence for multi-agent systems.
"""

import mlx.core as mx
from mlx_lm import stream_generate
from mlx_lm.models.cache import make_prompt_cache
from typing import Any, Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class MLXCacheExtractor:
    """
    Extracts and manages KV cache from MLX model generation.

    Wraps mlx_lm generation to provide:
    - Cache exposure after generation
    - Cache reuse across generation calls
    - Cache metadata extraction
    """

    def __init__(self, model, tokenizer):
        """
        Initialize cache extractor with model and tokenizer.

        Args:
            model: MLX model instance (from mlx_lm.load)
            tokenizer: Tokenizer instance (from mlx_lm.load)
        """
        self.model = model
        self.tokenizer = tokenizer
        logger.info(f"MLXCacheExtractor initialized for model")

    def generate_with_cache(
        self,
        prompt: str,
        existing_cache: Optional[List[Any]] = None,
        max_tokens: int = 256,
        temperature: float = 0.0,
        **kwargs
    ) -> Tuple[str, List[Any]]:
        """
        Generate text and return both output and the KV cache.

        Args:
            prompt: Input text prompt
            existing_cache: Optional pre-existing cache to continue from
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 = greedy)
            **kwargs: Additional arguments passed to stream_generate

        Returns:
            Tuple of (generated_text, cache)
            - generated_text: str, the generated output
            - cache: List[KVCache], the KV cache state after generation
        """
        # Create or reuse cache
        if existing_cache is None:
            prompt_cache = make_prompt_cache(self.model)
        else:
            prompt_cache = existing_cache

        # Generate with cache
        text = ""
        for response in stream_generate(
            self.model,
            self.tokenizer,
            prompt,
            max_tokens=max_tokens,
            temp=temperature,
            prompt_cache=prompt_cache,
            **kwargs
        ):
            text += response.text

        logger.debug(f"Generated {len(text)} characters, cache has {self.get_cache_info(prompt_cache)['total_tokens']} tokens")

        return text, prompt_cache

    def process_prompt(
        self,
        prompt: str,
        existing_cache: Optional[List[Any]] = None
    ) -> List[Any]:
        """
        Process prompt into cache without generating output.

        Useful for pre-filling system prompts or processing context
        without generating new tokens.

        Args:
            prompt: Input text prompt
            existing_cache: Optional pre-existing cache to extend

        Returns:
            cache: List[KVCache] containing the processed prompt
        """
        # Generate with max_tokens=0 to only process prompt
        _, cache = self.generate_with_cache(
            prompt,
            existing_cache=existing_cache,
            max_tokens=0
        )

        logger.debug(f"Processed prompt into cache: {self.get_cache_info(cache)['total_tokens']} tokens")

        return cache

    def get_cache_info(self, cache: List[Any]) -> Dict[str, Any]:
        """
        Extract metadata about the cache.

        Args:
            cache: List[KVCache] to inspect

        Returns:
            dict with keys:
            - num_layers: Number of transformer layers
            - total_tokens: Total tokens cached (from first layer's offset)
            - memory_bytes: Estimated memory usage in bytes
        """
        if not cache or len(cache) == 0:
            return {
                'num_layers': 0,
                'total_tokens': 0,
                'memory_bytes': 0
            }

        num_layers = len(cache)
        total_tokens = cache[0].offset if hasattr(cache[0], 'offset') else 0
        memory_bytes = self.get_cache_memory_bytes(cache)

        return {
            'num_layers': num_layers,
            'total_tokens': total_tokens,
            'memory_bytes': memory_bytes
        }

    def get_cache_memory_bytes(self, cache: List[Any]) -> int:
        """
        Estimate memory usage of the cache in bytes.

        Calculates based on tensor shapes and data types.

        Args:
            cache: List[KVCache] to measure

        Returns:
            int: Estimated memory usage in bytes
        """
        if not cache or len(cache) == 0:
            return 0

        total_bytes = 0

        for layer_cache in cache:
            # Get state (keys, values) trimmed to actual content
            if hasattr(layer_cache, 'state'):
                keys, values = layer_cache.state

                # Calculate bytes for keys
                if hasattr(keys, 'nbytes'):
                    total_bytes += keys.nbytes

                # Calculate bytes for values
                if hasattr(values, 'nbytes'):
                    total_bytes += values.nbytes

        return total_bytes
