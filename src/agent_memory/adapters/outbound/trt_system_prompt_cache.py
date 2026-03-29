# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Yakov Shkolnikov and contributors
"""TRT system prompt cache.

Wraps TRT's ``genAndSaveSystemPromptKVCache()`` via subprocess command
to pre-compute and cache the KV state for system + tools prompts.
"""

from pathlib import Path
from typing import Any

import numpy as np
import structlog
from safetensors.numpy import load_file, save_file

from agent_memory.domain.errors import TRTEngineError
from agent_memory.ports.outbound import ModelBackendPort

logger = structlog.get_logger(__name__)


class TRTSystemPromptCache:
    """Pre-compute and cache system prompt KV state via TRT subprocess."""

    def __init__(
        self,
        backend: ModelBackendPort,
        cache_dir: Path,
    ) -> None:
        """Initialize with backend port and cache directory."""
        self._backend = backend
        self._cache_dir = cache_dir
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def get_or_create(
        self,
        system_tokens: list[int],
        cache_key: str,
    ) -> list[tuple[Any, Any]]:
        """Get cached system prompt KV or generate it.

        Args:
            system_tokens: Tokenized system prompt.
            cache_key: Unique key for this system prompt configuration.

        Returns:
            List of per-layer (K, V) cache pairs.
        """
        cache_path = self._cache_dir / f"sysprompt_{cache_key}.safetensors"

        if cache_path.exists():
            logger.info("trt_system_prompt_cache_hit", key=cache_key)
            return self._load_cache(cache_path)

        logger.info("trt_system_prompt_cache_miss", key=cache_key, n_tokens=len(system_tokens))

        # Generate system prompt KV cache via subprocess
        result = self._backend.generate(
            prompt_tokens=system_tokens,
            cache=None,
            max_tokens=0,  # Prefill only
            temperature=0.0,
        )

        if not result.cache:
            raise TRTEngineError("System prompt prefill returned no cache")

        self._save_cache(result.cache, cache_path)
        return result.cache

    def _save_cache(self, cache: list[tuple[Any, Any]], path: Path) -> None:
        """Save system prompt cache to disk."""
        tensors = {}
        for layer_idx, (k, v) in enumerate(cache):
            tensors[f"L{layer_idx}_K"] = np.asarray(k, dtype=np.float16)
            tensors[f"L{layer_idx}_V"] = np.asarray(v, dtype=np.float16)

        save_file(tensors, str(path))
        logger.info("trt_system_prompt_cache_saved", path=str(path))

    def _load_cache(self, path: Path) -> list[tuple[Any, Any]]:
        """Load system prompt cache from disk."""
        tensors = load_file(str(path))
        cache: list[tuple[Any, Any]] = []
        layer_idx = 0
        while f"L{layer_idx}_K" in tensors:
            k = tensors[f"L{layer_idx}_K"]
            v = tensors[f"L{layer_idx}_V"]
            cache.append((k, v))
            layer_idx += 1
        return cache
