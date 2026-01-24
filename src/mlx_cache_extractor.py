"""
MLX Cache Extractor

Wrapper around mlx_lm that exposes KV cache after generation.
Enables cache inspection, reuse, and persistence for multi-agent systems.
"""

import mlx.core as mx
from mlx_lm import stream_generate
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.sample_utils import make_sampler
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

    def __init__(
        self,
        model,
        tokenizer,
        kv_bits: Optional[int] = None,
        kv_group_size: int = 64
    ):
        """
        Initialize cache extractor with model and tokenizer.

        Args:
            model: MLX model instance (from mlx_lm.load)
            tokenizer: Tokenizer instance (from mlx_lm.load)
            kv_bits: Optional KV cache quantization (2-8 bits, None=no quantization)
            kv_group_size: Group size for quantization (default 64)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.kv_bits = kv_bits
        self.kv_group_size = kv_group_size

        if kv_bits:
            logger.info(
                f"MLXCacheExtractor initialized with {kv_bits}-bit "
                f"quantization (group_size={kv_group_size})"
            )
        else:
            logger.info(f"MLXCacheExtractor initialized (no quantization)")

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
        # Create sampler with temperature (MLX requires sampler, not direct temp param)
        sampler = make_sampler(temperature)

        # Add quantization parameters if specified
        if self.kv_bits is not None:
            kwargs['kv_bits'] = self.kv_bits
            kwargs['kv_group_size'] = self.kv_group_size

        for response in stream_generate(
            self.model,
            self.tokenizer,
            prompt,
            max_tokens=max_tokens,
            sampler=sampler,
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
        # Generate with max_tokens=1 (MLX has bug with 0, we discard the 1 token output)
        _, cache = self.generate_with_cache(
            prompt,
            existing_cache=existing_cache,
            max_tokens=1
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
        Handles both regular and quantized caches.

        Args:
            cache: List[KVCache] or List[QuantizedKVCache] to measure

        Returns:
            int: Estimated memory usage in bytes
        """
        if not cache or len(cache) == 0:
            return 0

        total_bytes = 0

        for layer_cache in cache:
            # Check if this is a quantized cache
            is_quantized = hasattr(layer_cache, 'bits')

            if hasattr(layer_cache, 'state'):
                state = layer_cache.state

                if is_quantized:
                    # Quantized cache: state is (keys, values) where each is
                    # a tuple of (data, scales, biases)
                    keys, values = state

                    # Keys: (data, scales, biases)
                    if isinstance(keys, tuple) and len(keys) == 3:
                        for tensor in keys:
                            if hasattr(tensor, 'nbytes'):
                                total_bytes += tensor.nbytes
                    elif hasattr(keys, 'nbytes'):
                        total_bytes += keys.nbytes

                    # Values: (data, scales, biases)
                    if isinstance(values, tuple) and len(values) == 3:
                        for tensor in values:
                            if hasattr(tensor, 'nbytes'):
                                total_bytes += tensor.nbytes
                    elif hasattr(values, 'nbytes'):
                        total_bytes += values.nbytes
                else:
                    # Regular cache: state is (keys, values) tensors
                    keys, values = state

                    # Calculate bytes for keys
                    if hasattr(keys, 'nbytes'):
                        total_bytes += keys.nbytes

                    # Calculate bytes for values
                    if hasattr(values, 'nbytes'):
                        total_bytes += values.nbytes

        return total_bytes
