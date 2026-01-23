"""
Cache Persistence

Agent-level KV cache file management using safetensors format.
Wraps mlx_lm's save_prompt_cache/load_prompt_cache with agent-specific utilities.
"""

import os
import json
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import logging

from mlx_lm.models.cache import save_prompt_cache, load_prompt_cache

logger = logging.getLogger(__name__)


class CachePersistence:
    """
    Manages persistent storage of agent KV caches.

    Provides agent-level file management on top of mlx_lm's cache save/load:
    - Agent-specific file paths
    - Metadata storage (agent_id, timestamp, model, tokens)
    - Directory management
    - Cache listing and cleanup
    """

    def __init__(self, cache_dir: str = "~/.agent_caches"):
        """
        Initialize cache persistence manager.

        Args:
            cache_dir: Directory to store cache files (default: ~/.agent_caches)
        """
        self.cache_dir = Path(cache_dir).expanduser()
        self._ensure_cache_dir()
        logger.info(f"CachePersistence initialized with cache_dir: {self.cache_dir}")

    def _ensure_cache_dir(self):
        """Create cache directory if it doesn't exist."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_path(self, agent_id: str) -> Path:
        """Get file path for agent's cache file."""
        return self.cache_dir / f"{agent_id}.safetensors"

    def save_agent_cache(
        self,
        agent_id: str,
        cache: List[Any],
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Save agent's KV cache to disk.

        Args:
            agent_id: Unique identifier for the agent
            cache: List[KVCache] to save
            metadata: Optional metadata dict (agent_id, model, timestamp, etc.)
        """
        cache_path = self._get_cache_path(agent_id)

        # Prepare metadata
        full_metadata = metadata or {}
        full_metadata.update({
            'agent_id': agent_id,
            'timestamp': datetime.now().isoformat(),
        })

        # Extract cache size if not provided
        if 'cache_tokens' not in full_metadata and cache:
            if hasattr(cache[0], 'offset'):
                full_metadata['cache_tokens'] = str(cache[0].offset)

        # Convert all metadata values to strings (safetensors requirement)
        string_metadata = {k: str(v) for k, v in full_metadata.items()}

        # Save using mlx_lm's save_prompt_cache
        save_prompt_cache(str(cache_path), cache, string_metadata)

        logger.info(f"Saved agent cache: {agent_id} ({full_metadata.get('cache_tokens', 0)} tokens) -> {cache_path}")

    def load_agent_cache(self, agent_id: str) -> Tuple[List[Any], Dict[str, str]]:
        """
        Load agent's KV cache from disk.

        Args:
            agent_id: Unique identifier for the agent

        Returns:
            Tuple of (cache, metadata)
            - cache: List[KVCache] loaded from disk
            - metadata: Dict[str, str] of metadata

        Raises:
            FileNotFoundError: If cache file doesn't exist
        """
        cache_path = self._get_cache_path(agent_id)

        if not cache_path.exists():
            raise FileNotFoundError(f"Cache file not found for agent: {agent_id} (expected: {cache_path})")

        # Load using mlx_lm's load_prompt_cache
        cache, metadata = load_prompt_cache(str(cache_path), return_metadata=True)

        logger.info(f"Loaded agent cache: {agent_id} ({metadata.get('cache_tokens', 0)} tokens) <- {cache_path}")

        return cache, metadata

    def agent_cache_exists(self, agent_id: str) -> bool:
        """
        Check if cache file exists for agent.

        Args:
            agent_id: Unique identifier for the agent

        Returns:
            bool: True if cache file exists
        """
        return self._get_cache_path(agent_id).exists()

    def list_cached_agents(self) -> List[Dict[str, Any]]:
        """
        List all cached agents with metadata.

        Returns:
            List of dicts, each containing:
            - agent_id: str
            - file_path: str
            - file_size: int (bytes)
            - modified_time: str (ISO format)
            - metadata: Dict[str, str] (from safetensors)
        """
        cached_agents = []

        for cache_file in self.cache_dir.glob("*.safetensors"):
            agent_id = cache_file.stem

            # Get file stats
            stat = cache_file.stat()

            # Try to load metadata without loading full cache
            try:
                _, metadata = load_prompt_cache(str(cache_file), return_metadata=True)
            except Exception as e:
                logger.warning(f"Failed to load metadata for {agent_id}: {e}")
                metadata = {}

            cached_agents.append({
                'agent_id': agent_id,
                'file_path': str(cache_file),
                'file_size': stat.st_size,
                'modified_time': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                'metadata': metadata
            })

        return sorted(cached_agents, key=lambda x: x['modified_time'], reverse=True)

    def delete_agent_cache(self, agent_id: str) -> bool:
        """
        Delete agent's cache file.

        Args:
            agent_id: Unique identifier for the agent

        Returns:
            bool: True if file was deleted, False if it didn't exist
        """
        cache_path = self._get_cache_path(agent_id)

        if cache_path.exists():
            cache_path.unlink()
            logger.info(f"Deleted agent cache: {agent_id}")
            return True
        else:
            logger.warning(f"Cache file not found for deletion: {agent_id}")
            return False

    def get_cache_disk_usage(self) -> Dict[str, Any]:
        """
        Report disk space used by cached agents.

        Returns:
            dict with keys:
            - total_bytes: Total disk space used
            - total_mb: Total disk space in MB
            - num_agents: Number of cached agents
            - per_agent: List of (agent_id, bytes) tuples
        """
        agents = []
        total_bytes = 0

        for cache_file in self.cache_dir.glob("*.safetensors"):
            size = cache_file.stat().st_size
            agents.append((cache_file.stem, size))
            total_bytes += size

        return {
            'total_bytes': total_bytes,
            'total_mb': total_bytes / (1024 * 1024),
            'num_agents': len(agents),
            'per_agent': sorted(agents, key=lambda x: x[1], reverse=True)
        }
