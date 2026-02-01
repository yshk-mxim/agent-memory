"""Unit tests for AgentCacheStore (Day 4 skeleton).

Tests ModelTag, CacheEntry, and AgentCacheStore interfaces.
Full implementation tests added in Days 5-7.
"""

import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from semantic.application.agent_cache_store import (
    AgentCacheStore,
    CacheEntry,
    ModelTag,
)
from semantic.domain.entities import AgentBlocks
from semantic.domain.errors import InvalidRequestError
from semantic.domain.value_objects import ModelCacheSpec


@pytest.fixture
def spec() -> ModelCacheSpec:
    """Create test model cache spec."""
    return ModelCacheSpec(
        n_layers=12,
        n_kv_heads=4,
        head_dim=64,
        block_tokens=256,
        layer_types=["global"] * 12,
        sliding_window_size=None,
    )


@pytest.fixture
def model_tag(spec: ModelCacheSpec) -> ModelTag:
    """Create test model tag."""
    return ModelTag.from_spec("test-model", spec)


@pytest.fixture
def agent_blocks() -> AgentBlocks:
    """Create test agent blocks with non-empty layer data."""
    from semantic.domain.entities import KVBlock

    block = KVBlock(block_id=0, layer_id=0, token_count=256, layer_data="fake_kv")
    return AgentBlocks(
        agent_id="test_agent",
        blocks={0: [block]},
        total_tokens=256,
    )


@pytest.fixture
def mock_cache_adapter() -> MagicMock:
    """Create a mock cache adapter for disk persistence."""
    adapter = MagicMock()
    adapter.save.return_value = Path("/tmp/claude/fake_cache.safetensors")
    return adapter


class TestModelTag:
    """Tests for ModelTag dataclass."""

    def test_create_from_spec(self, spec: ModelCacheSpec) -> None:
        """Should create ModelTag from spec."""
        tag = ModelTag.from_spec("gemma-3-12b", spec)

        assert tag.model_id == "gemma-3-12b"
        assert tag.n_layers == 12
        assert tag.n_kv_heads == 4
        assert tag.head_dim == 64
        assert tag.block_tokens == 256

    def test_is_compatible_matching_spec(self, spec: ModelCacheSpec) -> None:
        """Should return True for matching spec."""
        tag = ModelTag.from_spec("test-model", spec)

        assert tag.is_compatible(spec) is True

    def test_is_compatible_different_n_layers(self, spec: ModelCacheSpec) -> None:
        """Should return False if n_layers differs."""
        tag = ModelTag.from_spec("test-model", spec)

        different_spec = ModelCacheSpec(
            n_layers=24,  # Different!
            n_kv_heads=4,
            head_dim=64,
            block_tokens=256,
            layer_types=["global"] * 24,
            sliding_window_size=None,
        )

        assert tag.is_compatible(different_spec) is False

    def test_is_compatible_different_n_kv_heads(self, spec: ModelCacheSpec) -> None:
        """Should return False if n_kv_heads differs."""
        tag = ModelTag.from_spec("test-model", spec)

        different_spec = ModelCacheSpec(
            n_layers=12,
            n_kv_heads=8,  # Different!
            head_dim=64,
            block_tokens=256,
            layer_types=["global"] * 12,
            sliding_window_size=None,
        )

        assert tag.is_compatible(different_spec) is False

    def test_is_frozen(self) -> None:
        """ModelTag should be immutable (frozen dataclass)."""
        tag = ModelTag("test", 12, 4, 64, 256)

        with pytest.raises(AttributeError):
            tag.n_layers = 24  # type: ignore[misc]


class TestCacheEntry:
    """Tests for CacheEntry dataclass."""

    def test_create_entry(self, agent_blocks: AgentBlocks, model_tag: ModelTag) -> None:
        """Should create cache entry with defaults."""
        entry = CacheEntry(
            agent_id="test_agent",
            blocks=agent_blocks,
            model_tag=model_tag,
        )

        assert entry.agent_id == "test_agent"
        assert entry.blocks is agent_blocks
        assert entry.model_tag is model_tag
        assert entry.access_count == 0
        assert entry.is_hot is True

    def test_mark_accessed_updates_timestamp(
        self, agent_blocks: AgentBlocks, model_tag: ModelTag
    ) -> None:
        """Should update last_accessed timestamp."""
        entry = CacheEntry(
            agent_id="test_agent",
            blocks=agent_blocks,
            model_tag=model_tag,
        )

        initial_time = entry.last_accessed
        entry.mark_accessed()

        assert entry.last_accessed > initial_time
        assert entry.access_count == 1

    def test_mark_accessed_increments_count(
        self, agent_blocks: AgentBlocks, model_tag: ModelTag
    ) -> None:
        """Should increment access_count on each call."""
        entry = CacheEntry(
            agent_id="test_agent",
            blocks=agent_blocks,
            model_tag=model_tag,
        )

        entry.mark_accessed()
        entry.mark_accessed()
        entry.mark_accessed()

        assert entry.access_count == 3


class TestAgentCacheStoreInit:
    """Tests for AgentCacheStore initialization."""

    def test_create_with_valid_params(self, model_tag: ModelTag) -> None:
        """Should create store with valid parameters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = AgentCacheStore(
                cache_dir=Path(tmpdir),
                max_hot_agents=5,
                model_tag=model_tag,
            )

            assert store.cache_dir == Path(tmpdir)
            assert store.max_hot_agents == 5
            assert store.model_tag is model_tag

    def test_creates_cache_dir_if_not_exists(self, model_tag: ModelTag) -> None:
        """Should create cache directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "nested" / "cache"

            store = AgentCacheStore(
                cache_dir=cache_path,
                max_hot_agents=5,
                model_tag=model_tag,
            )

            assert store.cache_dir == cache_path
            assert cache_path.exists()
            assert cache_path.is_dir()


class TestAgentCacheStoreSave:
    """Tests for AgentCacheStore.save()."""

    def test_save_adds_to_hot_tier(
        self, agent_blocks: AgentBlocks, model_tag: ModelTag, mock_cache_adapter: MagicMock
    ) -> None:
        """Should add cache entry to hot tier."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = AgentCacheStore(
                cache_dir=Path(tmpdir),
                max_hot_agents=5,
                model_tag=model_tag,
                cache_adapter=mock_cache_adapter,
            )

            store.save("agent_1", agent_blocks)

            # Verify in hot cache
            assert "agent_1" in store._hot_cache
            entry = store._hot_cache["agent_1"]
            assert entry.agent_id == "agent_1"
            assert entry.blocks is agent_blocks

    def test_save_marks_entry_accessed(
        self, agent_blocks: AgentBlocks, model_tag: ModelTag, mock_cache_adapter: MagicMock
    ) -> None:
        """Should mark entry as accessed on save."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = AgentCacheStore(
                cache_dir=Path(tmpdir),
                max_hot_agents=5,
                model_tag=model_tag,
                cache_adapter=mock_cache_adapter,
            )

            store.save("agent_1", agent_blocks)

            entry = store._hot_cache["agent_1"]
            assert entry.access_count > 0
            assert entry.last_accessed > 0

    def test_save_rejects_empty_agent_id(
        self, agent_blocks: AgentBlocks, model_tag: ModelTag
    ) -> None:
        """Should reject empty agent_id."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = AgentCacheStore(
                cache_dir=Path(tmpdir),
                max_hot_agents=5,
                model_tag=model_tag,
            )

            with pytest.raises(InvalidRequestError, match="agent_id cannot be empty"):
                store.save("", agent_blocks)

    def test_save_triggers_eviction_when_full(
        self, agent_blocks: AgentBlocks, model_tag: ModelTag, mock_cache_adapter: MagicMock
    ) -> None:
        """Should trigger LRU eviction when hot tier exceeds max."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = AgentCacheStore(
                cache_dir=Path(tmpdir),
                max_hot_agents=3,  # Small limit
                model_tag=model_tag,
                cache_adapter=mock_cache_adapter,
            )

            # Add 4 agents (exceeds limit of 3)
            store.save("agent_1", agent_blocks)
            store.save("agent_2", agent_blocks)
            store.save("agent_3", agent_blocks)
            store.save("agent_4", agent_blocks)

            # Should evict LRU, keeping only 3 in hot tier
            assert len(store._hot_cache) == 3


class TestAgentCacheStoreLoad:
    """Tests for AgentCacheStore.load()."""

    def test_load_from_hot_tier(
        self, agent_blocks: AgentBlocks, model_tag: ModelTag, mock_cache_adapter: MagicMock
    ) -> None:
        """Should load cache from hot tier."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = AgentCacheStore(
                cache_dir=Path(tmpdir),
                max_hot_agents=5,
                model_tag=model_tag,
                cache_adapter=mock_cache_adapter,
            )

            store.save("agent_1", agent_blocks)
            loaded = store.load("agent_1")

            assert loaded is agent_blocks

    def test_load_updates_access_time(
        self, agent_blocks: AgentBlocks, model_tag: ModelTag, mock_cache_adapter: MagicMock
    ) -> None:
        """Should update access time on load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = AgentCacheStore(
                cache_dir=Path(tmpdir),
                max_hot_agents=5,
                model_tag=model_tag,
                cache_adapter=mock_cache_adapter,
            )

            store.save("agent_1", agent_blocks)
            entry = store._hot_cache["agent_1"]
            initial_access = entry.last_accessed

            time.sleep(0.01)  # Small delay
            store.load("agent_1")

            assert entry.last_accessed > initial_access

    def test_load_returns_none_for_missing_agent(self, model_tag: ModelTag) -> None:
        """Should return None if agent not found."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = AgentCacheStore(
                cache_dir=Path(tmpdir),
                max_hot_agents=5,
                model_tag=model_tag,
            )

            loaded = store.load("nonexistent_agent")

            assert loaded is None


class TestAgentCacheStoreEviction:
    """Tests for AgentCacheStore LRU eviction."""

    def test_evict_lru_removes_oldest(
        self, agent_blocks: AgentBlocks, model_tag: ModelTag, mock_cache_adapter: MagicMock
    ) -> None:
        """Should evict least-recently-used entry."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = AgentCacheStore(
                cache_dir=Path(tmpdir),
                max_hot_agents=5,
                model_tag=model_tag,
                cache_adapter=mock_cache_adapter,
            )

            # Add 3 agents
            store.save("agent_1", agent_blocks)
            time.sleep(0.01)
            store.save("agent_2", agent_blocks)
            time.sleep(0.01)
            store.save("agent_3", agent_blocks)

            # Evict to 1 (should keep agent_3, the most recent)
            evicted = store.evict_lru(target_count=1)

            assert evicted == 2
            assert len(store._hot_cache) == 1
            assert "agent_3" in store._hot_cache

    def test_evict_lru_returns_count(
        self, agent_blocks: AgentBlocks, model_tag: ModelTag, mock_cache_adapter: MagicMock
    ) -> None:
        """Should return number of evicted entries."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = AgentCacheStore(
                cache_dir=Path(tmpdir),
                max_hot_agents=5,
                model_tag=model_tag,
                cache_adapter=mock_cache_adapter,
            )

            store.save("agent_1", agent_blocks)
            store.save("agent_2", agent_blocks)
            store.save("agent_3", agent_blocks)

            evicted = store.evict_lru(target_count=1)

            assert evicted == 2


class TestAgentCacheStorePrefixMatching:
    """Tests for prefix matching (Day 6 implementation)."""

    def test_find_prefix_returns_none_when_empty(self, model_tag: ModelTag) -> None:
        """find_prefix() should return None when no caches exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = AgentCacheStore(
                cache_dir=Path(tmpdir),
                max_hot_agents=5,
                model_tag=model_tag,
            )

            result = store.find_prefix([1, 2, 3])

            assert result is None

    def test_find_prefix_returns_longest_match(
        self, agent_blocks: AgentBlocks, model_tag: ModelTag, mock_cache_adapter: MagicMock
    ) -> None:
        """find_prefix() should return cache with longest prefix."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = AgentCacheStore(
                cache_dir=Path(tmpdir),
                max_hot_agents=5,
                model_tag=model_tag,
                cache_adapter=mock_cache_adapter,
            )

            # Save two agents with different cache sizes
            # (In stub, just save empty blocks - real implementation would have tokens)
            store.save("agent_1", agent_blocks)
            store.save("agent_2", agent_blocks)

            # Manually set total_tokens for testing (bypassing validation)
            store._hot_cache["agent_1"].blocks.total_tokens = 10  # type: ignore[union-attr]
            store._hot_cache["agent_2"].blocks.total_tokens = 100  # type: ignore[union-attr]

            # Query should return longest match
            result = store.find_prefix([1, 2, 3, 4, 5])

            # Simplified implementation returns agent with most tokens
            assert result is not None
            assert result.total_tokens == 100  # agent_2 has longest cache
