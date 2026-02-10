# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Yakov Shkolnikov and contributors
"""Unit tests for cache corruption detection in SafetensorsCacheAdapter.

Verifies that the adapter correctly detects and rejects:
- Truncated files
- Corrupted JSON headers
- Zero-byte files
- Path traversal attempts
"""

import struct
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Mock MLX modules before importing adapter (adapter imports mlx at load time)
sys.modules.setdefault("mlx", MagicMock())
sys.modules.setdefault("mlx.core", MagicMock())

from agent_memory.domain.errors import AgentNotFoundError, CachePersistenceError

pytestmark = pytest.mark.unit


@pytest.fixture
def cache_dir(tmp_path: Path) -> Path:
    """Create a temporary cache directory."""
    d = tmp_path / "cache"
    d.mkdir()
    return d


@pytest.fixture
def adapter(cache_dir: Path):
    """Create adapter instance."""
    from agent_memory.adapters.outbound.safetensors_cache_adapter import SafetensorsCacheAdapter

    return SafetensorsCacheAdapter(cache_dir)


class TestPathTraversalPrevention:
    def test_rejects_empty_agent_id(self, adapter) -> None:
        with pytest.raises(CachePersistenceError, match="agent_id cannot be empty"):
            adapter._validate_agent_id("")

    def test_rejects_path_traversal_dotdot(self, adapter) -> None:
        with pytest.raises(CachePersistenceError, match="Invalid agent_id"):
            adapter._validate_agent_id("../../etc/passwd")

    def test_rejects_slashes(self, adapter) -> None:
        with pytest.raises(CachePersistenceError, match="Invalid agent_id"):
            adapter._validate_agent_id("path/to/file")

    def test_rejects_backslashes(self, adapter) -> None:
        with pytest.raises(CachePersistenceError, match="Invalid agent_id"):
            adapter._validate_agent_id("path\\to\\file")

    def test_rejects_dots(self, adapter) -> None:
        with pytest.raises(CachePersistenceError, match="Invalid agent_id"):
            adapter._validate_agent_id("agent.name")

    def test_rejects_spaces(self, adapter) -> None:
        with pytest.raises(CachePersistenceError, match="Invalid agent_id"):
            adapter._validate_agent_id("agent name")

    def test_rejects_too_long_id(self, adapter) -> None:
        with pytest.raises(CachePersistenceError, match="too long"):
            adapter._validate_agent_id("a" * 257)

    def test_accepts_valid_alphanumeric(self, adapter) -> None:
        adapter._validate_agent_id("agent_123")  # Should not raise

    def test_accepts_hyphens(self, adapter) -> None:
        adapter._validate_agent_id("agent-123-abc")  # Should not raise

    def test_accepts_underscores(self, adapter) -> None:
        adapter._validate_agent_id("agent_123_abc")  # Should not raise

    def test_accepts_max_length(self, adapter) -> None:
        adapter._validate_agent_id("a" * 256)  # Should not raise


class TestTruncatedFile:
    def test_truncated_header_less_than_8_bytes(self, cache_dir: Path, adapter) -> None:
        cache_path = cache_dir / "test_agent.safetensors"
        cache_path.write_bytes(b"\x00\x01\x02")  # Only 3 bytes

        with pytest.raises(CachePersistenceError, match="truncated header"):
            adapter.load(cache_path)

    def test_file_not_found(self, cache_dir: Path, adapter) -> None:
        cache_path = cache_dir / "nonexistent.safetensors"
        with pytest.raises(AgentNotFoundError, match="Cache not found"):
            adapter.load(cache_path)


class TestCorruptedJsonHeader:
    def test_invalid_json_in_header(self, cache_dir: Path, adapter) -> None:
        cache_path = cache_dir / "test_agent.safetensors"
        invalid_json = b"not valid json at all"
        header_size = struct.pack("<Q", len(invalid_json))
        cache_path.write_bytes(header_size + invalid_json)

        with pytest.raises(CachePersistenceError, match="Corrupted cache metadata"):
            adapter.load(cache_path)

    def test_zero_byte_header_size(self, cache_dir: Path, adapter) -> None:
        """Zero-length header produces empty bytes which json.loads rejects."""
        cache_path = cache_dir / "test_agent.safetensors"
        header_size = struct.pack("<Q", 0)
        cache_path.write_bytes(header_size)

        with pytest.raises(CachePersistenceError, match="Corrupted cache metadata"):
            adapter.load(cache_path)


class TestZeroByteFile:
    def test_empty_file(self, cache_dir: Path, adapter) -> None:
        cache_path = cache_dir / "test_agent.safetensors"
        cache_path.write_bytes(b"")

        with pytest.raises(CachePersistenceError, match="truncated header"):
            adapter.load(cache_path)


class TestExistsAndDelete:
    def test_exists_returns_false_for_missing(self, adapter) -> None:
        assert adapter.exists("nonexistent_agent") is False

    def test_delete_nonexistent_is_noop(self, adapter) -> None:
        adapter.delete("nonexistent_agent")  # Should not raise

    def test_list_cached_agents_empty(self, adapter) -> None:
        assert adapter.list_cached_agents() == []
