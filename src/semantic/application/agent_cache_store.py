"""Agent cache storage with trie-based prefix matching and LRU eviction."""

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from semantic.domain.entities import AgentBlocks
from semantic.domain.errors import InvalidRequestError
from semantic.domain.value_objects import ModelCacheSpec

logger = logging.getLogger(__name__)


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
        dirty: True if modified since last disk sync (write-behind)

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
    dirty: bool = False  # Write-behind: needs disk sync

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

        # Thread-safe lock for cache access
        self._lock = threading.RLock()

        # Hot tier: agent_id → CacheEntry (in-memory)
        self._hot_cache: dict[str, CacheEntry] = {}

        # Warm tier: agent_id → file path (on disk)
        self._warm_cache: dict[str, Path] = {}

        # Future: trie-based prefix matching for O(log n) lookup

    def save(self, agent_id: str, blocks: AgentBlocks) -> None:
        """Save agent cache to hot tier with write-behind persistence.

        Args:
            agent_id: Unique agent identifier
            blocks: Agent's KV cache blocks

        Raises:
            ValueError: If agent_id is empty

        Notes:
            - Adds to hot tier immediately with dirty=True
            - Disk write deferred until eviction or explicit flush
            - Saves 50-100ms per call vs immediate persistence
            - May trigger LRU eviction if hot tier full

        Example:
            >>> store.save("agent_1", blocks)
            >>> # Cache now in hot tier (memory), disk write deferred
        """
        if not agent_id:
            raise InvalidRequestError("agent_id cannot be empty")

        # Create entry with dirty flag (write-behind)
        entry = CacheEntry(
            agent_id=agent_id,
            blocks=blocks,
            model_tag=self.model_tag,
            dirty=True,  # Mark for eventual disk sync
        )
        entry.mark_accessed()

        with self._lock:
            # Add to hot tier
            self._hot_cache[agent_id] = entry

            # NO disk write here — deferred to eviction or flush
            # This eliminates 50-100ms of synchronous I/O per save

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
            - Falls back to warm tier (disk load) if:
              - Not in hot tier, OR
              - Hot tier entry has empty/cleared blocks
            - Promotes warm→hot on access
            - Validates model compatibility

        Example:
            >>> blocks = store.load("agent_1")
            >>> if blocks is None:
            ...     print("Cache miss - need to regenerate")
        """
        with self._lock:
            # Check hot tier first
            if agent_id in self._hot_cache:
                entry = self._hot_cache[agent_id]

                # CRITICAL: Check if blocks were cleared by batch_engine
                # batch_engine clears layer_data after reconstruction to free Q4 memory,
                # but this also clears the shared reference in hot_cache.
                # If blocks are empty, fall back to disk.
                if entry.blocks is not None and entry.blocks.blocks:
                    # Check if at least one block has actual data
                    has_data = False
                    for layer_blocks in entry.blocks.blocks.values():
                        if layer_blocks and any(b.layer_data is not None for b in layer_blocks):
                            has_data = True
                            break

                    if has_data:
                        entry.mark_accessed()
                        return entry.blocks

                # Blocks were cleared - fall through to disk load

            # Check warm tier (disk)
            if agent_id in self._warm_cache:
                return self._load_from_disk(agent_id)

            # Cache miss
            return None

    def delete(self, agent_id: str, keep_disk: bool = False) -> bool:
        """Delete agent cache from all tiers.

        Args:
            agent_id: Unique agent identifier
            keep_disk: If True, only evict from hot tier and keep disk file
                      for warm cache reload. If False, fully delete.

        Returns:
            True if agent was found and deleted, False if not found

        Notes:
            - Removes from hot tier (memory)
            - If keep_disk=False: Removes from warm tier and deletes disk file
            - If keep_disk=True: Flushes to disk and keeps for warm reload

        Example:
            >>> deleted = store.delete("agent_1")  # Full delete
            >>> store.delete("agent_1", keep_disk=True)  # Evict to disk only
        """
        with self._lock:
            found = False

            # CRITICAL: Flush dirty cache to disk BEFORE deletion
            # This ensures warm cache test can reload from disk
            if agent_id in self._hot_cache:
                entry = self._hot_cache[agent_id]
                if entry.dirty and entry.blocks is not None:
                    self._save_to_disk(agent_id)
                    logger.debug(f"Flushed dirty cache to disk before eviction: {agent_id}")
                found = True

            # Remove from hot tier
            if agent_id in self._hot_cache:
                del self._hot_cache[agent_id]

            # If keep_disk=False, remove from warm tier and delete disk file
            if not keep_disk and agent_id in self._warm_cache:
                cache_path = self._warm_cache[agent_id]
                if cache_path.exists():
                    cache_path.unlink()
                del self._warm_cache[agent_id]

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

        with self._lock:
            best_match: AgentBlocks | None = None
            best_prefix_len = 0
            best_entry: CacheEntry | None = None

            for _agent_id, entry in self._hot_cache.items():
                if entry.blocks is None:
                    continue

                prefix_len = entry.blocks.common_prefix_length(tokens)

                if prefix_len > best_prefix_len:
                    best_prefix_len = prefix_len
                    best_match = entry.blocks
                    best_entry = entry

            if best_entry is not None:
                best_entry.mark_accessed()

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
        with self._lock:
            evicted = 0

            while len(self._hot_cache) > target_count:
                # Find LRU entry
                lru_agent_id = min(
                    self._hot_cache.keys(),
                    key=lambda aid: self._hot_cache[aid].last_accessed,
                )

                # Persist to disk (flushes dirty entries)
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

    def invalidate_hot(self, agent_id: str) -> None:
        """Remove agent from hot tier after blocks are cleared.

        Called by batch_engine after clearing Q4 blocks to free memory.
        The cache remains in warm tier (disk) for future loads.

        This prevents the hot cache from holding a stale reference to
        cleared blocks, which would cause unnecessary has_data checks
        on every load.

        Args:
            agent_id: Agent whose hot cache entry should be invalidated

        Notes:
            - Only removes from hot tier (memory reference)
            - Warm tier (disk) is NOT affected
            - Future loads will reload from disk
        """
        with self._lock:
            if agent_id in self._hot_cache:
                del self._hot_cache[agent_id]

    def _save_to_disk(self, agent_id: str) -> None:
        """Persist cache to warm tier and clear dirty flag."""
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
            "token_sequence": entry.blocks.token_sequence,
            "prompt_text": entry.blocks.prompt_text,
        }

        if self._cache_adapter is None:
            raise InvalidRequestError("CacheAdapter is required - dependency not injected")
        cache_path = self._cache_adapter.save(agent_id, entry.blocks, metadata)
        self._warm_cache[agent_id] = cache_path

        # Clear dirty flag after successful disk write
        entry.dirty = False

    def flush_dirty(self) -> int:
        """Flush all dirty (unsaved) caches to disk.

        Returns:
            Number of caches flushed.

        Notes:
            Call this on server shutdown to ensure no data loss.
            With write-behind, dirty caches exist only in memory until
            eviction or explicit flush.

        Example:
            >>> store.flush_dirty()
            3  # Flushed 3 dirty caches to disk
        """
        with self._lock:
            flushed = 0
            for agent_id, entry in self._hot_cache.items():
                if entry.dirty and entry.blocks is not None:
                    self._save_to_disk(agent_id)
                    flushed += 1
            return flushed

    def _load_from_disk(self, agent_id: str) -> AgentBlocks | None:
        """Load cache from warm tier."""
        cache_path = self._warm_cache.get(agent_id)
        if cache_path is None or not cache_path.exists():
            return None

        try:
            if self._cache_adapter is None:
                raise InvalidRequestError("CacheAdapter is required - dependency not injected")
            blocks_dict, metadata = self._cache_adapter.load(cache_path)

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
            token_seq_raw = metadata.get("token_sequence", "[]")
            if isinstance(token_seq_raw, str):
                try:
                    token_sequence = json.loads(token_seq_raw)
                    if not isinstance(token_sequence, list):
                        token_sequence = []
                except (json.JSONDecodeError, ValueError):
                    token_sequence = []
            else:
                token_sequence = token_seq_raw if isinstance(token_seq_raw, list) else []
            prompt_text = str(metadata.get("prompt_text", ""))
            blocks = AgentBlocks(
                agent_id=agent_id,
                blocks=blocks_dict,
                total_tokens=total_tokens,
                token_sequence=token_sequence,
                prompt_text=prompt_text,
            )

            # Promote to hot tier
            entry = CacheEntry(
                agent_id=agent_id,
                blocks=blocks,
                model_tag=self.model_tag,
            )
            entry.mark_accessed()
            self._hot_cache[agent_id] = entry

            return blocks

        except (OSError, IOError) as e:
            logger.warning(f"Failed to load cache for {agent_id} from {cache_path}: {e}")
            return None
        except (ValueError, KeyError) as e:
            logger.warning(f"Invalid cache format for {agent_id} at {cache_path}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error loading cache for {agent_id}: {e}", exc_info=True)
            return None

    def list_all_agents(self) -> list[dict[str, Any]]:
        """List all cached agents across hot, warm, and cold tiers.

        Returns union of hot (in-memory) and warm (on-disk) agents with metadata.

        Returns:
            List of agent metadata dicts with keys:
                - agent_id: Agent identifier
                - tier: "hot" or "warm"
                - tokens: Total tokens cached (0 if warm)
                - last_accessed: Unix timestamp (0 if warm)
                - access_count: Number of accesses (0 if warm)
                - dirty: True if has unsaved changes (False if warm)
                - model_id: Model identifier from tag
                - file_size_bytes: Disk size (0 if hot-only)

        Example:
            >>> agents = store.list_all_agents()
            >>> hot = [a for a in agents if a["tier"] == "hot"]
            >>> warm = [a for a in agents if a["tier"] == "warm"]
        """
        with self._lock:
            result = []

            # Hot tier agents
            for agent_id, entry in self._hot_cache.items():
                tokens = entry.blocks.total_tokens if entry.blocks else 0
                file_size = 0
                cache_path = self.cache_dir / f"{agent_id}.safetensors"
                if cache_path.exists():
                    file_size = cache_path.stat().st_size

                result.append({
                    "agent_id": agent_id,
                    "tier": "hot",
                    "tokens": tokens,
                    "last_accessed": entry.last_accessed,
                    "access_count": entry.access_count,
                    "dirty": entry.dirty,
                    "model_id": entry.model_tag.model_id,
                    "file_size_bytes": file_size,
                })

            # Warm tier agents (on disk, not in memory)
            warm_ids = set(self._warm_cache.keys()) - set(self._hot_cache.keys())
            for agent_id in warm_ids:
                cache_path = self._warm_cache[agent_id]
                file_size = 0
                if cache_path.exists():
                    file_size = cache_path.stat().st_size

                result.append({
                    "agent_id": agent_id,
                    "tier": "warm",
                    "tokens": 0,  # Unknown without loading
                    "last_accessed": 0.0,
                    "access_count": 0,
                    "dirty": False,
                    "model_id": self.model_tag.model_id,
                    "file_size_bytes": file_size,
                })

            return result

    def get_agent_metadata(self, agent_id: str) -> dict[str, Any] | None:
        """Get agent cache metadata without loading tensors.

        Performs header-only read for warm-tier agents (no tensor deserialization).

        Args:
            agent_id: Agent identifier

        Returns:
            Metadata dict or None if not found. Keys:
                - agent_id: Agent identifier
                - tier: "hot" or "warm"
                - tokens: Total tokens cached
                - last_accessed: Unix timestamp
                - access_count: Number of accesses
                - dirty: True if has unsaved changes
                - model_id: Model identifier
                - file_size_bytes: Disk size (0 if hot-only)
                - prompt_preview: First 100 chars of prompt (if available)

        Example:
            >>> meta = store.get_agent_metadata("agent_123")
            >>> if meta:
            ...     print(f"{meta['agent_id']}: {meta['tokens']} tokens")
        """
        with self._lock:
            # Check hot tier first
            if agent_id in self._hot_cache:
                entry = self._hot_cache[agent_id]
                tokens = entry.blocks.total_tokens if entry.blocks else 0
                prompt_preview = ""
                if entry.blocks and entry.blocks.prompt_text:
                    prompt_preview = entry.blocks.prompt_text[:100]

                file_size = 0
                cache_path = self.cache_dir / f"{agent_id}.safetensors"
                if cache_path.exists():
                    file_size = cache_path.stat().st_size

                return {
                    "agent_id": agent_id,
                    "tier": "hot",
                    "tokens": tokens,
                    "last_accessed": entry.last_accessed,
                    "access_count": entry.access_count,
                    "dirty": entry.dirty,
                    "model_id": entry.model_tag.model_id,
                    "file_size_bytes": file_size,
                    "prompt_preview": prompt_preview,
                }

            # Check warm tier (header-only read)
            if agent_id in self._warm_cache:
                cache_path = self._warm_cache[agent_id]
                if not cache_path.exists():
                    return None

                try:
                    # Use cache adapter for header-only read
                    if self._cache_adapter:
                        metadata = self._cache_adapter.get_cache_file_metadata(agent_id)
                        if metadata:
                            file_size = cache_path.stat().st_size
                            return {
                                "agent_id": agent_id,
                                "tier": "warm",
                                "tokens": metadata.get("total_tokens", 0),
                                "last_accessed": 0.0,
                                "access_count": 0,
                                "dirty": False,
                                "model_id": metadata.get("model_id", self.model_tag.model_id),
                                "file_size_bytes": file_size,
                                "prompt_preview": metadata.get("prompt_text", "")[:100],
                            }
                except Exception as e:
                    logger.warning(f"Failed to read metadata for {agent_id}: {e}")
                    return None

            return None
