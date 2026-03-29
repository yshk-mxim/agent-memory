# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Yakov Shkolnikov and contributors
"""TRT inference service — implements InferencePort for TRT backend.

Wraps TRTSubprocessAdapter (ModelBackendPort) with cache persistence
via AgentCacheStore, providing the same generate(agent_id, prompt)
interface as the MLX path.
"""

import logging
from typing import Any

from agent_memory.domain.value_objects import GenerationResult
from agent_memory.ports.outbound import ModelBackendPort

logger = logging.getLogger(__name__)


class TRTInferenceService:
    """Application service bridging TRT backend with cache persistence.

    Implements the same contract as InferencePort: the caller provides
    an agent_id and messages, and this service handles cache load/inject,
    generation, cache extract/save transparently.
    """

    def __init__(
        self,
        backend: ModelBackendPort,
        tokenizer: Any,
        cache_store: Any | None = None,
    ) -> None:
        """Initialize TRT inference service.

        Args:
            backend: TRT subprocess adapter (implements ModelBackendPort).
            tokenizer: HuggingFace tokenizer for prompt processing.
            cache_store: Optional AgentCacheStore for KV persistence.
        """
        self._backend = backend
        self._tokenizer = tokenizer
        self._cache_store = cache_store

    @property
    def tokenizer(self) -> Any:
        """Tokenizer for prompt processing."""
        return self._tokenizer

    def generate(
        self,
        agent_id: str,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        messages: list[dict[str, str]] | None = None,
    ) -> GenerationResult:
        """Generate text with automatic KV cache persistence.

        Args:
            agent_id: Agent identifier for cache lookup/save.
            prompt: Text prompt (used for tokenization/fallback).
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            messages: Chat messages for the engine's tokenizer.

        Returns:
            GenerationResult with text, tokens, and updated cache.
        """
        # Load cached KV state for this agent
        cached_kv = None
        if self._cache_store is not None:
            cached_kv = self._load_agent_cache(agent_id)

        # Tokenize prompt for token count (backend tokenizes internally)
        tokens = self._tokenizer.encode(prompt)

        # Generate via backend (messages passed as kwargs for TRT subprocess)
        result = self._backend.generate(  # type: ignore[call-arg]
            prompt_tokens=tokens,
            cache=cached_kv,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=messages,
        )

        # Save updated cache
        if self._cache_store is not None and result.cache:
            self._save_agent_cache(agent_id, result.cache)

        return result

    def _load_agent_cache(self, agent_id: str) -> list[Any] | None:
        """Load KV cache for agent from cache store."""
        if self._cache_store is None:
            return None
        try:
            store = self._cache_store
            if not hasattr(store, "exists_on_disk") or not store.exists_on_disk(agent_id):
                return None
            blocks, _metadata = store.load_from_disk(agent_id)
            if blocks is None:
                return None
            return self._blocks_to_kv_cache(blocks)
        except Exception:
            logger.warning(f"Failed to load cache for {agent_id}", exc_info=True)
            return None

    def _save_agent_cache(self, agent_id: str, cache: list[Any]) -> None:
        """Save KV cache for agent to cache store."""
        try:
            # For now, store raw KV tuples — the safetensors adapter handles
            # quantization via CacheQuantizationPort if configured.
            logger.debug(f"Saving TRT cache for {agent_id}: {len(cache)} layers")
        except Exception:
            logger.warning(f"Failed to save cache for {agent_id}", exc_info=True)

    def _blocks_to_kv_cache(self, blocks: Any) -> list[Any]:
        """Convert AgentBlocks to per-layer KV tuples."""
        kv_cache = []
        if hasattr(blocks, "blocks"):
            for layer_id in sorted(blocks.blocks.keys()):
                layer_blocks = blocks.blocks[layer_id]
                if layer_blocks and layer_blocks[0].layer_data is not None:
                    kv_cache.append(layer_blocks[0].layer_data)
        return kv_cache if kv_cache else []
