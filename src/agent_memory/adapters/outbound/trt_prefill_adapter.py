# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Yakov Shkolnikov and contributors
"""TRT prefill adapter.

Implements PrefillChunkPort for the TRT backend. Sends prefill chunks
to the subprocess, which processes them through the TRT engine and
returns updated KV caches.
"""

import logging
from typing import Any

import numpy as np

from agent_memory.ports.outbound import ModelBackendPort

logger = logging.getLogger(__name__)

_MIN_CHUNK = 512
_MAX_CHUNK = 2048
_LARGE_CACHE_THRESHOLD = 8192


class TRTPrefillAdapter:
    """Adapter for chunked prefill via TRT subprocess."""

    def __init__(
        self,
        backend: ModelBackendPort,
        min_chunk: int = _MIN_CHUNK,
        max_chunk: int = _MAX_CHUNK,
    ) -> None:
        """Initialize with backend port and chunk size bounds."""
        self._backend = backend
        self._min_chunk = min_chunk
        self._max_chunk = max_chunk

    def init_prefill_caches(self, n_layers: int) -> list[tuple[Any, Any]]:
        """Create empty KV caches for a new prefill sequence.

        Args:
            n_layers: Number of transformer layers.

        Returns:
            List of empty (K, V) numpy array pairs per layer.
        """
        empty_caches = []
        for _ in range(n_layers):
            k = np.zeros((0,), dtype=np.float16)
            v = np.zeros((0,), dtype=np.float16)
            empty_caches.append((k, v))
        return empty_caches

    def process_prefill_chunk(
        self,
        tokens: list[int],
        start: int,
        end: int,
        kv_caches: Any,
    ) -> None:
        """Process one prefill chunk through the TRT engine.

        Args:
            tokens: Full token sequence.
            start: Start index of this chunk (inclusive).
            end: End index of this chunk (exclusive).
            kv_caches: KV cache state — updated in place with subprocess results.
        """
        chunk_tokens = tokens[start:end]

        result = self._backend.generate(
            prompt_tokens=chunk_tokens,
            cache=kv_caches if start > 0 else None,
            max_tokens=0,  # Prefill only, no generation
            temperature=0.0,
        )

        # Update caches in place with result from subprocess
        if result.cache:
            for i, (k, v) in enumerate(result.cache):
                if i < len(kv_caches):
                    kv_caches[i] = (k, v)

    def chunk_size_for_position(self, cache_pos: int) -> int:
        """Return adaptive chunk size based on current cache position.

        Smaller chunks when cache is large (memory-efficient).
        """
        if cache_pos > _LARGE_CACHE_THRESHOLD:
            return self._min_chunk
        return self._max_chunk
