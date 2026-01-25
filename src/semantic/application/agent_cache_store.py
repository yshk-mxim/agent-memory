"""Agent cache storage with trie-based prefix matching and LRU eviction.

Implements three-tier cache lifecycle:
- Hot tier: In-memory caches for active agents
- Warm tier: Disk-persisted caches (safetensors format)
- Cold tier: Evicted (can be reloaded from warm tier)

NEW-4 Part 2: AgentCacheStore skeleton with interfaces defined.
Days 5-7: Full implementation of persistence, prefix matching, LRU eviction.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from semantic.domain.entities import AgentBlocks
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
    blocks: Optional[AgentBlocks]
    model_tag: ModelTag
    last_accessed: float = field(default_factory=lambda: 0.0)
    access_count: int = 0
    is_hot: bool = True

    def mark_accessed(self) -> None:
        """Mark entry as accessed (update LRU timestamp).

        Updates last_accessed to current time and increments access_count.
        Used by LRU eviction policy.
        """
        import time

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
    ) -> None:
        """Initialize cache store.

        Args:
            cache_dir: Directory for warm-tier disk storage
            max_hot_agents: Maximum agents in hot tier (memory)
            model_tag: Current model tag for compatibility checking
        """
        self.cache_dir = Path(cache_dir).expanduser()
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.max_hot_agents = max_hot_agents
        self.model_tag = model_tag

        # Hot tier: agent_id → CacheEntry (in-memory)
        self._hot_cache: dict[str, CacheEntry] = {}

        # Warm tier: agent_id → file path (on disk)
        self._warm_cache: dict[str, Path] = {}

        # Prefix trie (Day 6 implementation)
        # Maps token prefixes to agent IDs for cache reuse
        self._prefix_trie: dict[Any, Any] = {}  # TODO: Implement trie structure

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
            raise ValueError("agent_id cannot be empty")

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

    def load(self, agent_id: str) -> Optional[AgentBlocks]:
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

    def find_prefix(self, tokens: list[int]) -> Optional[AgentBlocks]:
        """Find longest prefix match in cache (Day 6 implementation).

        Args:
            tokens: Token sequence to match

        Returns:
            AgentBlocks with longest matching prefix, or None

        Notes:
            - Uses trie structure for O(prefix_length) lookup
            - Returns cache with LONGEST common prefix
            - Useful for continuing conversations

        Example:
            >>> # Agent has cached tokens [1, 2, 3, 4, 5]
            >>> # Query with [1, 2, 3, 6, 7]
            >>> blocks = store.find_prefix([1, 2, 3, 6, 7])
            >>> # Returns cache for [1, 2, 3] (longest prefix)
        """
        # TODO Day 6: Implement trie-based prefix matching
        return None

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

    def _save_to_disk(self, agent_id: str) -> None:
        """Persist cache to warm tier (Day 7 implementation).

        Args:
            agent_id: Agent to persist

        Notes:
            - Saves in safetensors format
            - Atomic write (tmp + rename)
            - Includes model tag for validation
        """
        # TODO Day 7: Implement safetensors persistence
        cache_path = self.cache_dir / f"{agent_id}.safetensors"
        self._warm_cache[agent_id] = cache_path

    def _load_from_disk(self, agent_id: str) -> Optional[AgentBlocks]:
        """Load cache from warm tier (Day 7 implementation).

        Args:
            agent_id: Agent to load

        Returns:
            AgentBlocks if valid, None if incompatible or missing

        Notes:
            - Validates model tag compatibility
            - Promotes to hot tier on successful load
        """
        # TODO Day 7: Implement safetensors loading
        return None
