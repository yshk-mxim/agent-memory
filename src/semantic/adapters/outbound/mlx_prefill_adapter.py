"""MLX adapter for single-chunk prefill processing.

Implements PrefillChunkPort to process one chunk of a sequence's prefill
at a time, enabling the scheduler to interleave prefill with decode steps.

Mirrors the logic in BatchEngine._chunked_prefill() but exposes it as
a step-at-a-time interface instead of a blocking loop.
"""

import logging
import time
from typing import Any

import mlx.core as mx
from mlx_lm.generate import generation_stream
from mlx_lm.models.cache import QuantizedKVCache

logger = logging.getLogger(__name__)

MIN_CHUNK_DEFAULT = 512
MAX_CHUNK_DEFAULT = 4096


def adaptive_chunk_size(
    cache_pos: int,
    min_chunk: int = MIN_CHUNK_DEFAULT,
    max_chunk: int = MAX_CHUNK_DEFAULT,
) -> int:
    """Calculate optimal chunk size based on current cache position.

    Larger chunks when cache is small (fast), smaller when large
    (memory-efficient).
    """
    if cache_pos < 2000:
        return max_chunk
    elif cache_pos < 8000:
        return max(min_chunk, max_chunk // 2)
    elif cache_pos < 20000:
        return max(min_chunk, max_chunk // 4)
    else:
        return min_chunk


class MLXPrefillAdapter:
    """Processes one prefill chunk at a time using direct MLX model calls.

    The scheduler calls process_prefill_chunk() once per scheduling
    iteration, interleaving with BatchGenerator decode steps.
    """

    def __init__(
        self,
        model: Any,
        kv_bits: int = 4,
        kv_group_size: int = 64,
        min_chunk: int = MIN_CHUNK_DEFAULT,
        max_chunk: int = MAX_CHUNK_DEFAULT,
    ) -> None:
        self._model = model
        self._kv_bits = kv_bits
        self._kv_group_size = kv_group_size
        self._min_chunk = min_chunk
        self._max_chunk = max_chunk

    def init_prefill_caches(self, n_layers: int) -> list[QuantizedKVCache]:
        """Create empty Q4 KV caches for a new prefill sequence."""
        return [
            QuantizedKVCache(group_size=self._kv_group_size, bits=self._kv_bits)
            for _ in range(n_layers)
        ]

    def process_prefill_chunk(
        self,
        tokens: list[int],
        start: int,
        end: int,
        kv_caches: list[QuantizedKVCache],
    ) -> None:
        """Process one chunk of tokens through the model, updating kv_caches.

        This performs a single model forward pass for tokens[start:end],
        updating kv_caches in place. After the forward pass, forces
        evaluation and clears the MLX compute cache to bound memory.

        The forward pass runs on generation_stream (the same Metal command
        stream that BatchGenerator uses for decode). This prevents command
        buffer conflicts when the scheduler interleaves prefill with decode.

        Args:
            tokens: Full token sequence.
            start: Start index of this chunk (inclusive).
            end: End index of this chunk (exclusive).
            kv_caches: KV cache list to update in place.
        """
        t0 = time.time()

        with mx.stream(generation_stream):
            chunk_tokens = mx.array([tokens[start:end]])
            y = self._model(chunk_tokens, cache=kv_caches)
            mx.eval(y)
        mx.clear_cache()

        elapsed_ms = (time.time() - t0) * 1000
        logger.info(
            f"[PREFILL CHUNK] pos {start}->{end} "
            f"({end - start} tokens, {elapsed_ms:.0f}ms)"
        )

    def chunk_size_for_position(self, cache_pos: int) -> int:
        """Return adaptive chunk size based on current cache position."""
        return adaptive_chunk_size(cache_pos, self._min_chunk, self._max_chunk)
