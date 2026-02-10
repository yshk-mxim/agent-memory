# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Yakov Shkolnikov and contributors
"""Outbound port interfaces (driven adapters).

These ports define the contracts for the application core to interact
with external systems (infrastructure). Implementations are provided by
outbound adapters (MLX backend, disk persistence, etc.).

All interfaces use Protocol (PEP 544) for structural typing, allowing
implicit implementation without inheritance.
"""

from typing import Any, Protocol

from agent_memory.domain.value_objects import CacheKey, GenerationResult, ModelCacheSpec


class ModelBackendPort(Protocol):
    """Port for model inference backend.

    This port defines the contract for LLM inference operations.
    Implementations wrap specific backends (MLX, vLLM, etc.) with
    a unified interface.
    """

    def generate(
        self,
        prompt_tokens: list[int],
        cache: list[Any] | None = None,
        max_tokens: int = 256,
        temperature: float = 0.7,
    ) -> GenerationResult:
        """Generate text from tokenized prompt with optional cache.

        Args:
            prompt_tokens: Pre-tokenized input as list of token IDs.
            cache: Optional pre-built KV cache (from previous generation).
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.

        Returns:
            GenerationResult with generated text, tokens, and updated cache.

        Raises:
            ModelNotFoundError: If model not loaded.
        """
        ...

    def extract_model_spec(self) -> ModelCacheSpec:
        """Extract cache specification from the loaded model.

        Returns:
            ModelCacheSpec describing the model's cache geometry
            (layers, attention pattern, block requirements).

        Raises:
            ModelNotFoundError: If no model is loaded.
        """
        ...


class CachePersistencePort(Protocol):
    """Port for KV cache persistence.

    This port defines the contract for saving and loading KV caches
    to/from disk. Implementations handle serialization formats
    (safetensors, HDF5, etc.).
    """

    def save(
        self,
        agent_id: str,
        cache: list[Any],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Save KV cache to disk.

        Args:
            agent_id: Unique identifier for the agent (used as filename).
            cache: List of KV cache objects (per-layer).
            metadata: Optional metadata (model_id, token_count, etc.).

        Raises:
            CachePersistenceError: If save fails (disk full, permissions, etc.).
        """
        ...

    def load(self, agent_id: str) -> tuple[list[Any], dict[str, Any]]:
        """Load KV cache from disk.

        Args:
            agent_id: Unique identifier for the agent.

        Returns:
            Tuple of (cache, metadata).

        Raises:
            AgentNotFoundError: If cache file does not exist.
            CachePersistenceError: If load fails (corruption, version mismatch).
            IncompatibleCacheError: If cache model_id != current model.
        """
        ...

    def exists(self, agent_id: str) -> bool:
        """Check if cache exists on disk.

        Args:
            agent_id: Unique identifier for the agent.

        Returns:
            True if cache file exists, False otherwise.
        """
        ...

    def delete(self, agent_id: str) -> None:
        """Delete cache from disk.

        Args:
            agent_id: Unique identifier for the agent.

        Raises:
            AgentNotFoundError: If cache file does not exist.
            CachePersistenceError: If deletion fails (permissions, etc.).
        """
        ...

    def list_cached_agents(self) -> list[str]:
        """List all agent IDs with caches on disk.

        Returns:
            List of agent IDs.
        """
        ...


class TokenizerPort(Protocol):
    """Port for tokenization operations.

    This port defines the contract for encoding/decoding text.
    Implementations wrap tokenizer libraries (transformers, tiktoken, etc.).
    """

    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs.

        Args:
            text: Input text string.

        Returns:
            List of token IDs.
        """
        ...

    def decode(self, tokens: list[int]) -> str:
        """Decode token IDs to text.

        Args:
            tokens: List of token IDs.

        Returns:
            Decoded text string.
        """
        ...

    @property
    def eos_token_id(self) -> int:
        """Get the end-of-sequence token ID.

        Returns:
            EOS token ID.
        """
        ...

    @property
    def vocab_size(self) -> int:
        """Get vocabulary size.

        Returns:
            Number of tokens in vocabulary.
        """
        ...


class CacheOperationsPort(Protocol):
    """Port for tensor operations on KV cache."""

    def concatenate_cache_blocks(
        self,
        k_tensors: list[Any],
        v_tensors: list[Any],
    ) -> tuple[Any, Any]:
        """Concatenate K/V tensors from multiple blocks along sequence axis."""
        ...

    def get_sequence_length(self, k_tensor: Any) -> int:
        """Extract sequence length from K tensor.

        Args:
            k_tensor: K tensor with shape [n_kv_heads, head_dim, seq_len]

        Returns:
            Sequence length (axis=2 dimension).
        """
        ...

    def slice_cache_tensor(
        self,
        tensor: Any,
        start_token: int,
        end_token: int,
    ) -> Any:
        """Slice cache tensor along sequence axis.

        Args:
            tensor: Cache tensor (K or V) with shape [n_kv_heads, head_dim, seq_len]
            start_token: Start index for slicing (inclusive)
            end_token: End index for slicing (exclusive)

        Returns:
            Sliced tensor with shape [n_kv_heads, head_dim, end_token - start_token]
        """
        ...


class PrefillChunkPort(Protocol):
    """Port for processing one chunk of a sequence's prefill.

    Used by ConcurrentScheduler to interleave prefill chunks with
    decode steps. Each call processes a single chunk and updates
    the PrefillState's kv_caches in place.
    """

    def init_prefill_caches(self, n_layers: int) -> Any:
        """Create empty KV caches for a new prefill sequence.

        Args:
            n_layers: Number of transformer layers in the model.

        Returns:
            List of per-layer KV cache objects (opaque to application).
        """
        ...

    def process_prefill_chunk(
        self,
        tokens: list[int],
        start: int,
        end: int,
        kv_caches: Any,
    ) -> None:
        """Process one chunk of tokens through the model, updating kv_caches.

        Args:
            tokens: Full token sequence.
            start: Start index of this chunk (inclusive).
            end: End index of this chunk (exclusive).
            kv_caches: KV cache state to update in place.
        """
        ...

    def chunk_size_for_position(self, cache_pos: int) -> int:
        """Return adaptive chunk size based on current cache position.

        Larger chunks when cache is small (fast), smaller when large
        (memory-efficient).

        Args:
            cache_pos: Number of tokens already in the cache.

        Returns:
            Chunk size in tokens.
        """
        ...


class CacheStorePort(Protocol):
    """Port for in-memory cache management."""

    def get(self, cache_key: CacheKey) -> Any | None:
        """Retrieve cache for agent, with prefix matching.

        Args:
            cache_key: Key containing agent_id + token prefix hash.

        Returns:
            AgentBlocks if cache exists (exact or prefix match), None otherwise.

        Notes:
            - Performs trie-based prefix matching
            - Loads from disk if in warm/cold tier (cache miss)
            - Updates LRU on access (move to hot tier)
            - Returns None if agent has no cache

        Example:
            >>> key = CacheKey(agent_id="a1", model_id="gemma-3", prefix_hash="abc123")
            >>> blocks = store.get(key)
            >>> if blocks:
            ...     print(f"Cache hit: {blocks.total_tokens} tokens")
        """
        ...

    def put(self, cache_key: CacheKey, blocks: Any) -> None:
        """Store cache for agent in memory (hot tier).

        Args:
            cache_key: Key containing agent_id + token prefix hash.
            blocks: AgentBlocks to store.

        Raises:
            PoolExhaustedError: If no space for new cache entry.

        Notes:
            - Stores in hot tier (in-memory)
            - Triggers LRU eviction if memory pressure detected
            - Evicted caches move to warm tier (disk)
            - Existing cache for same agent is replaced

        Example:
            >>> key = CacheKey(agent_id="a1", model_id="gemma-3", prefix_hash="abc123")
            >>> store.put(key, agent_blocks)
        """
        ...

    def evict(self, agent_id: str) -> None:
        """Manually evict agent cache from memory to disk.

        Args:
            agent_id: Unique identifier for the agent.

        Notes:
            - Moves from hot tier (memory) to warm tier (disk)
            - Frees blocks via BlockPool.free()
            - Writes cache to disk via CachePersistencePort
            - No-op if agent not in hot tier

        Example:
            >>> store.evict("agent_1")  # Free memory for other agents
        """
        ...

    def delete(self, agent_id: str) -> None:
        """Permanently delete agent cache (memory + disk).

        Args:
            agent_id: Unique identifier for the agent.

        Raises:
            AgentNotFoundError: If agent does not exist.

        Notes:
            - Removes from hot tier (if present)
            - Deletes from disk (warm/cold tiers)
            - Frees all blocks associated with agent
            - Use when agent context is no longer needed

        Example:
            >>> store.delete("agent_1")  # Permanent deletion
        """
        ...
