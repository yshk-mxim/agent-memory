"""Shared prefix cache for system+tools KV state reuse.

When multiple agents share the same system prompt and tool definitions,
the tokenized prefix is identical.  This cache stores the KV state for
that static prefix so that new agents skip the expensive prefill for
the common portion.

Architecture layer: application service.
No MLX / infrastructure imports — KV caches are stored as opaque objects.
"""

import hashlib
import logging
import threading
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

MAX_PREFIX_ENTRIES = 8  # Bound memory usage


@dataclass
class PrefixEntry:
    """Cached KV state for a static prefix."""

    prefix_hash: str
    kv_caches: Any
    n_tokens: int
    token_sequence: list[int]
    hit_count: int = 0


class SharedPrefixCache:
    """In-memory cache of KV state for static prompt prefixes.

    Keyed by MD5 hash of the system+tools text.  When a new agent
    arrives with the same system+tools, the cached KV state is cloned
    and used as the starting point, skipping prefill for the common
    portion.

    Thread-safe.  Entries are evicted LRU when the cache exceeds
    MAX_PREFIX_ENTRIES.
    """

    def __init__(self, max_entries: int = MAX_PREFIX_ENTRIES) -> None:
        self._max_entries = max_entries
        self._lock = threading.Lock()
        self._entries: dict[str, PrefixEntry] = {}

    @staticmethod
    def compute_hash(system_text: str, tools_text: str) -> str:
        """Compute a stable hash for a system+tools combination."""
        payload = f"{system_text}\x00{tools_text}"
        return hashlib.md5(payload.encode()).hexdigest()

    def get(self, prefix_hash: str) -> PrefixEntry | None:
        """Look up cached prefix KV state.

        Returns PrefixEntry if found, None otherwise.
        The caller must clone the KV caches before mutating them.
        """
        with self._lock:
            entry = self._entries.get(prefix_hash)
            if entry is not None:
                entry.hit_count += 1
                logger.debug(
                    f"[PREFIX CACHE HIT] hash={prefix_hash[:8]}, "
                    f"tokens={entry.n_tokens}, hits={entry.hit_count}"
                )
            return entry

    def put(
        self,
        prefix_hash: str,
        kv_caches: Any,
        n_tokens: int,
        token_sequence: list[int],
    ) -> None:
        """Store KV state for a static prefix.

        If the cache is full, evicts the least-hit entry.
        """
        with self._lock:
            if prefix_hash in self._entries:
                return  # Already cached

            if len(self._entries) >= self._max_entries:
                self._evict_least_used()

            self._entries[prefix_hash] = PrefixEntry(
                prefix_hash=prefix_hash,
                kv_caches=kv_caches,
                n_tokens=n_tokens,
                token_sequence=token_sequence,
            )
            logger.info(
                f"[PREFIX CACHE STORE] hash={prefix_hash[:8]}, "
                f"tokens={n_tokens}, entries={len(self._entries)}"
            )

    def clear(self) -> None:
        """Clear all cached prefixes (e.g. on model swap)."""
        with self._lock:
            count = len(self._entries)
            self._entries.clear()
            if count:
                logger.info(f"[PREFIX CACHE CLEAR] Cleared {count} entries")

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._entries)

    def _evict_least_used(self) -> None:
        """Evict the entry with the lowest hit count."""
        if not self._entries:
            return
        lru_hash = min(self._entries, key=lambda h: self._entries[h].hit_count)
        evicted = self._entries.pop(lru_hash)
        logger.debug(
            f"[PREFIX CACHE EVICT] hash={lru_hash[:8]}, "
            f"hits={evicted.hit_count}"
        )
