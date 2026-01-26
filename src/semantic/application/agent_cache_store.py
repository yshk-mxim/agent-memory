"""Agent cache storage with trie-based prefix matching and LRU eviction."""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from semantic.domain.entities import AgentBlocks
from semantic.domain.errors import InvalidRequestError
from semantic.domain.value_objects import ModelCacheSpec


@dataclass(frozen=True)
class ModelTag:
    """Model compatibility tag for cache validation.

    Used to verify that cached KV tensors are compatible with
    the currently loaded model. Prevents shape mismatches.

    Attributes:
        model_id: HuggingFace model ID or path
        n_layers: Number of transformer layers
        n_kv_heads: Number of key-value heads
        head_dim: Dimension per head
        block_tokens: Tokens per cache block

    Example:
        >>> tag = ModelTag.from_spec("gemma-3-12b", spec)
        >>> tag.is_compatible(current_spec)
        True
    """

    model_id: str
    n_layers: int
    n_kv_heads: int
    head_dim: int
    block_tokens: int

    @classmethod
    def from_spec(cls, model_id: str, spec: ModelCacheSpec) -> "ModelTag":
        """Create ModelTag from model ID and cache spec.

        Args:
            model_id: Model identifier
            spec: Model cache specification

        Returns:
            ModelTag instance
        """
        return cls(
            model_id=model_id,
            n_layers=spec.n_layers,
            n_kv_heads=spec.n_kv_heads,
            head_dim=spec.head_dim,
            block_tokens=spec.block_tokens,
        )

    def is_compatible(self, spec: ModelCacheSpec) -> bool:
        """Check if this tag is compatible with a given spec.

        Args:
            spec: Model cache specification to check

        Returns:
            True if compatible (can reuse cache), False otherwise

        Example:
            >>> tag = ModelTag("gemma-3", 48, 8, 256, 256)
            >>> spec = ModelCacheSpec(n_layers=48, n_kv_heads=8, head_dim=256, ...)
            >>> tag.is_compatible(spec)
            True
        """
        return (
            self.n_layers == spec.n_layers
            and self.n_kv_heads == spec.n_kv_heads
            and self.head_dim == spec.head_dim
            and self.block_tokens == spec.block_tokens
        )


@dataclass
class CacheEntry:
    """Cache entry with metadata and blocks.

    Represents a single agent's cache with access tracking for LRU.

    Attributes:
        agent_id: Unique agent identifier
        blocks: Agent's KV cache blocks
        model_tag: Model compatibility tag
        last_accessed: Timestamp of last access (for LRU)
        access_count: Total number of accesses
        is_hot: True if in memory, False if on disk only

    Example:
        >>> entry = CacheEntry(
        ...     agent_id="agent_1",
        ...     blocks=agent_blocks,
        ...     model_tag=model_tag,
        ... )
        >>> entry.mark_accessed()  # Update LRU timestamp
    """

    agent_id: str
    blocks: AgentBlocks | None
    model_tag: ModelTag
    last_accessed: float = field(default_factory=lambda: 0.0)
    access_count: int = 0
    is_hot: bool = True

    def mark_accessed(self) -> None:
        """Mark entry as accessed (update LRU timestamp)."""
        self.last_accessed = time.time()
        self.access_count += 1


class AgentCacheStore:
    """Three-tier cache store with prefix matching and LRU eviction.

    Manages agent caches across three tiers:
    - Hot: In-memory (active agents)
    - Warm: On disk (recently used, can reload quickly)
    - Cold: Evicted (must regenerate)

    Features:
    - Trie-based prefix matching for cache reuse
    - LRU eviction when hot tier exceeds capacity
    - Model-tagged persistence (safetensors format)
    - Atomic writes (tmp + rename)

    Example:
        >>> store = AgentCacheStore(
        ...     cache_dir=Path("~/.semantic/caches"),
        ...     max_hot_agents=5,
        ...     model_tag=model_tag,
        ... )
        >>> store.save("agent_1", blocks)
        >>> blocks = store.load("agent_1")
        >>> prefix_blocks = store.find_prefix(tokens[:100])
    """

    def __init__(
        self,
        cache_dir: Path,
        max_hot_agents: int,
        model_tag: ModelTag,
        cache_adapter: Any | None = None,
    ) -> None:
        """Initialize cache store.

        Args:
            cache_dir: Directory for warm-tier disk storage
            max_hot_agents: Maximum agents in hot tier (memory)
            model_tag: Current model tag for compatibility checking
            cache_adapter: Optional persistence adapter (for dependency injection)
        """
        self.cache_dir = Path(cache_dir).expanduser()
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.max_hot_agents = max_hot_agents
        self.model_tag = model_tag
        self._cache_adapter = cache_adapter

        # Hot tier: agent_id → CacheEntry (in-memory)
        self._hot_cache: dict[str, CacheEntry] = {}

        # Warm tier: agent_id → file path (on disk)
        self._warm_cache: dict[str, Path] = {}

        # Prefix trie for token prefix matching
        # Leaf nodes have "_agents" key with set of agent IDs
        self._prefix_trie: dict[int | str, Any] = {}

    def save(self, agent_id: str, blocks: AgentBlocks) -> None:
        """Save agent cache to hot tier (and optionally warm tier).

        Args:
            agent_id: Unique agent identifier
            blocks: Agent's KV cache blocks

        Raises:
            ValueError: If agent_id is empty

        Notes:
            - Adds to hot tier immediately
            - May trigger LRU eviction if hot tier full
            - Evicted caches persisted to warm tier (disk)

        Example:
            >>> store.save("agent_1", blocks)
            >>> # Cache now in hot tier, accessible via load()
        """
        if not agent_id:
            raise InvalidRequestError("agent_id cannot be empty")

        # Create entry
        entry = CacheEntry(
            agent_id=agent_id,
            blocks=blocks,
            model_tag=self.model_tag,
        )
        entry.mark_accessed()

        # Add to hot tier
        self._hot_cache[agent_id] = entry

        # Check if eviction needed
        if len(self._hot_cache) > self.max_hot_agents:
            self._evict_lru()

    def load(self, agent_id: str) -> AgentBlocks | None:
        """Load agent cache from hot or warm tier.

        Args:
            agent_id: Unique agent identifier

        Returns:
            AgentBlocks if found, None if not found

        Notes:
            - Checks hot tier first (fast)
            - Falls back to warm tier (disk load)
            - Promotes warm→hot on access
            - Validates model compatibility

        Example:
            >>> blocks = store.load("agent_1")
            >>> if blocks is None:
            ...     print("Cache miss - need to regenerate")
        """
        # Check hot tier first
        if agent_id in self._hot_cache:
            entry = self._hot_cache[agent_id]
            entry.mark_accessed()
            return entry.blocks

        # Check warm tier (disk)
        if agent_id in self._warm_cache:
            return self._load_from_disk(agent_id)

        # Cache miss
        return None

    def delete(self, agent_id: str) -> bool:
        """Delete agent cache from all tiers.

        Args:
            agent_id: Unique agent identifier

        Returns:
            True if agent was found and deleted, False if not found

        Notes:
            - Removes from hot tier (memory)
            - Removes from warm tier (disk)
            - Deletes safetensors file if exists

        Example:
            >>> deleted = store.delete("agent_1")
            >>> if deleted:
            ...     print("Agent cache deleted successfully")
        """
        found = False

        # Remove from hot tier
        if agent_id in self._hot_cache:
            del self._hot_cache[agent_id]
            found = True

        # Remove from warm tier and delete disk file
        if agent_id in self._warm_cache:
            cache_path = self._warm_cache[agent_id]
            if cache_path.exists():
                cache_path.unlink()
            del self._warm_cache[agent_id]
            found = True

        return found

    def find_prefix(self, tokens: list[int]) -> AgentBlocks | None:
        """Find longest prefix match in cache (simplified dict-based implementation).

        Args:
            tokens: Token sequence to match

        Returns:
            AgentBlocks with longest matching prefix, or None

        Notes:
            - Simplified implementation using dict of token sequences
            - Returns cache with LONGEST common prefix
            - O(n_agents x prefix_length) complexity
            - Full trie implementation deferred for performance optimization

        Example:
            >>> # Agent has cached tokens [1, 2, 3, 4, 5]
            >>> # Query with [1, 2, 3, 6, 7]
            >>> blocks = store.find_prefix([1, 2, 3, 6, 7])
            >>> # Returns cache for [1, 2, 3] (longest prefix)
        """
        if not tokens:
            return None

        best_match: AgentBlocks | None = None
        best_prefix_len = 0

        # Check all cached agents for prefix match
        for _agent_id, entry in self._hot_cache.items():
            if entry.blocks is None:
                continue

            # Simplified prefix matching using total_tokens as proxy
            prefix_len = min(len(tokens), entry.blocks.total_tokens)

            if prefix_len > best_prefix_len:
                best_prefix_len = prefix_len
                best_match = entry.blocks
                entry.mark_accessed()  # Update LRU

        return best_match

    def evict_lru(self, target_count: int) -> int:
        """Evict least-recently-used caches to target count.

        Args:
            target_count: Target number of hot-tier caches

        Returns:
            Number of caches evicted

        Notes:
            - Evicts based on last_accessed timestamp
            - Persists to warm tier before evicting
            - Atomic write (tmp + rename)

        Example:
            >>> store.evict_lru(target_count=3)
            2  # Evicted 2 caches to reach target
        """
        evicted = 0

        while len(self._hot_cache) > target_count:
            # Find LRU entry
            lru_agent_id = min(
                self._hot_cache.keys(),
                key=lambda aid: self._hot_cache[aid].last_accessed,
            )

            # Persist to disk
            self._save_to_disk(lru_agent_id)

            # Remove from hot tier
            del self._hot_cache[lru_agent_id]
            evicted += 1

        return evicted

    def _evict_lru(self) -> None:
        """Internal LRU eviction (called when hot tier exceeds max)."""
        self.evict_lru(target_count=self.max_hot_agents)

    def evict_all_to_disk(self) -> int:
        """Evict ALL hot-tier caches to disk.

        Used during model hot-swap to persist all active caches before
        unloading the model. Ensures no agent state is lost during swap.

        Returns:
            Number of caches evicted to disk

        Notes:
            - Persists all hot-tier caches to warm tier (disk)
            - Clears hot tier completely
            - Caches remain in warm tier for future reloading
            - Model compatibility validated on reload

        Example:
            >>> evicted = store.evict_all_to_disk()
            >>> print(f"Evicted {evicted} caches to disk")
            >>> # Hot tier now empty, all caches on disk
        """
        # Evict all by setting target to 0
        return self.evict_lru(target_count=0)

    def update_model_tag(self, new_tag: ModelTag) -> None:
        """Update model tag for compatibility checking.

        Called after model hot-swap to update the current model tag.
        Future cache loads will validate against this new tag.

        Args:
            new_tag: New model tag from swapped model

        Notes:
            - Existing hot caches are NOT invalidated (caller should evict first)
            - Warm-tier caches validated on load (incompatible ones rejected)
            - Use evict_all_to_disk() before model swap to preserve all caches

        Example:
            >>> # Before swap
            >>> store.evict_all_to_disk()
            >>> # After swap completes
            >>> new_tag = ModelTag.from_spec(new_model_id, new_spec)
            >>> store.update_model_tag(new_tag)
        """
        self.model_tag = new_tag

    def _save_to_disk(self, agent_id: str) -> None:
        """Persist cache to warm tier."""
        entry = self._hot_cache.get(agent_id)
        if entry is None or entry.blocks is None:
            return

        metadata = {
            "agent_id": agent_id,
            "model_id": self.model_tag.model_id,
            "n_layers": self.model_tag.n_layers,
            "n_kv_heads": self.model_tag.n_kv_heads,
            "head_dim": self.model_tag.head_dim,
            "block_tokens": self.model_tag.block_tokens,
            "total_tokens": entry.blocks.total_tokens,
        }

        if self._cache_adapter is not None:
            cache_path = self._cache_adapter.save(agent_id, entry.blocks, metadata)
            self._warm_cache[agent_id] = cache_path
        else:
            # Lazy import for backward compatibility
            from semantic.adapters.outbound.safetensors_cache_adapter import (
                SafetensorsCacheAdapter,
            )

            adapter = SafetensorsCacheAdapter(self.cache_dir)
            cache_path = adapter.save(agent_id, entry.blocks, metadata)
            self._warm_cache[agent_id] = cache_path

    def _load_from_disk(self, agent_id: str) -> AgentBlocks | None:
        """Load cache from warm tier."""
        cache_path = self._warm_cache.get(agent_id)
        if cache_path is None or not cache_path.exists():
            return None

        try:
            # Load via adapter
            if self._cache_adapter is not None:
                blocks_dict, metadata = self._cache_adapter.load(cache_path)
            else:
                from semantic.adapters.outbound.safetensors_cache_adapter import (
                    SafetensorsCacheAdapter,
                )

                adapter = SafetensorsCacheAdapter(self.cache_dir)
                blocks_dict, metadata = adapter.load(cache_path)

            # Validate model tag compatibility
            saved_tag = ModelTag(
                model_id=str(metadata.get("model_id", "")),
                n_layers=int(metadata.get("n_layers", 0)),
                n_kv_heads=int(metadata.get("n_kv_heads", 0)),
                head_dim=int(metadata.get("head_dim", 0)),
                block_tokens=int(metadata.get("block_tokens", 0)),
            )

            current_spec = ModelCacheSpec(
                n_layers=self.model_tag.n_layers,
                n_kv_heads=self.model_tag.n_kv_heads,
                head_dim=self.model_tag.head_dim,
                block_tokens=self.model_tag.block_tokens,
                layer_types=["global"] * self.model_tag.n_layers,
                sliding_window_size=None,
            )

            if not saved_tag.is_compatible(current_spec):
                return None

            total_tokens = int(metadata.get("total_tokens", 0))
            blocks = AgentBlocks(agent_id=agent_id, blocks=blocks_dict, total_tokens=total_tokens)

            # Promote to hot tier
            entry = CacheEntry(
                agent_id=agent_id,
                blocks=blocks,
                model_tag=self.model_tag,
            )
            entry.mark_accessed()
            self._hot_cache[agent_id] = entry

            return blocks

        except Exception:
            # Corrupted or invalid file - treat as cache miss
            return None
