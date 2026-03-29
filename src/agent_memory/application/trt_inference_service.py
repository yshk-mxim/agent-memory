# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Yakov Shkolnikov and contributors
"""TRT inference service — implements InferencePort for TRT backend.

Wraps TRTSubprocessAdapter (ModelBackendPort) with cache persistence,
providing the same generate(agent_id, prompt) interface as the MLX path.
"""

import logging
from typing import Any

from agent_memory.domain.entities import AgentBlocks, KVBlock
from agent_memory.domain.value_objects import GenerationResult
from agent_memory.ports.outbound import ModelBackendPort

logger = logging.getLogger(__name__)


class TRTInferenceService:
    """Application service bridging TRT backend with cache persistence.

    Caller provides agent_id + messages; this service handles
    cache load → inject → generate → extract → save transparently.
    """

    def __init__(
        self,
        backend: ModelBackendPort,
        tokenizer: Any,
        cache_adapter: Any | None = None,
    ) -> None:
        """Initialize TRT inference service.

        Args:
            backend: TRT subprocess adapter (implements ModelBackendPort).
            tokenizer: HuggingFace tokenizer for prompt processing.
            cache_adapter: Cache persistence adapter (TRTSafetensorsCacheAdapter).
        """
        self._backend = backend
        self._tokenizer = tokenizer
        self._cache_adapter = cache_adapter

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
        cached_kv = self._load_agent_cache(agent_id)

        # Tokenize prompt for token count
        tokens = self._tokenizer.encode(prompt)

        # Generate via backend
        result = self._backend.generate(  # type: ignore[call-arg]
            prompt_tokens=tokens,
            cache=cached_kv,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=messages,
        )

        # Save updated cache to disk
        if result.cache:
            self._save_agent_cache(agent_id, result.cache, len(tokens) + len(result.tokens))

        return result

    def _load_agent_cache(self, agent_id: str) -> list[Any] | None:
        """Load KV cache for agent from disk."""
        if self._cache_adapter is None:
            return None
        try:
            if not self._cache_adapter.exists(agent_id):
                return None
            agent_blocks, _metadata = self._cache_adapter.load(agent_id)
            return self._blocks_to_kv_cache(agent_blocks)
        except Exception:
            logger.warning(f"Failed to load cache for {agent_id}", exc_info=True)
            return None

    def _save_agent_cache(
        self,
        agent_id: str,
        cache: list[Any],
        total_tokens: int,
    ) -> None:
        """Save KV cache to disk as AgentBlocks."""
        if self._cache_adapter is None:
            return
        try:
            blocks_dict: dict[int, list[KVBlock]] = {}
            for layer_idx, kv_pair in enumerate(cache):
                block = KVBlock(
                    block_id=layer_idx * 1_000_000,
                    layer_id=layer_idx,
                    token_count=total_tokens,
                    layer_data=kv_pair,
                )
                blocks_dict[layer_idx] = [block]

            agent_blocks = AgentBlocks(
                agent_id=agent_id,
                blocks=blocks_dict,
                total_tokens=total_tokens,
            )

            metadata = {
                "agent_id": agent_id,
                "total_tokens": str(total_tokens),
                "n_layers": str(len(cache)),
            }

            self._cache_adapter.save(agent_id, agent_blocks, metadata)
            logger.debug(
                f"Saved TRT cache for {agent_id}: {len(cache)} layers, {total_tokens} tokens"
            )
        except Exception:
            logger.warning(f"Failed to save cache for {agent_id}", exc_info=True)

    @staticmethod
    def _blocks_to_kv_cache(blocks: AgentBlocks) -> list[Any]:
        """Convert AgentBlocks to per-layer KV tuples for backend injection."""
        kv_cache = []
        for layer_id in sorted(blocks.blocks.keys()):
            layer_blocks = blocks.blocks[layer_id]
            if layer_blocks and layer_blocks[0].layer_data is not None:
                kv_cache.append(layer_blocks[0].layer_data)
        return kv_cache if kv_cache else []
