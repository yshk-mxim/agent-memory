"""Unit tests for cache invalidation during model hot-swap."""

import sys
from unittest.mock import MagicMock, Mock

# Mock MLX modules
sys.modules["mlx"] = MagicMock()
sys.modules["mlx.core"] = MagicMock()
sys.modules["mlx.utils"] = MagicMock()
sys.modules["mlx_lm"] = MagicMock()

from semantic.application.agent_cache_store import AgentCacheStore, CacheEntry, ModelTag
from semantic.domain.entities import AgentBlocks
from semantic.domain.value_objects import ModelCacheSpec


class TestEvictAllToDisk:
    """Test evict_all_to_disk() method for model hot-swap."""

    def test_evict_all_to_disk_with_hot_caches(self, tmp_path):
        """Evicting all hot caches persists them to disk."""
        # Setup
        spec = ModelCacheSpec(
            n_layers=24, n_kv_heads=8, head_dim=128, block_tokens=16, layer_types=["global"] * 24
        )
        tag = ModelTag.from_spec("test-model", spec)

        # Mock cache adapter
        mock_adapter = Mock()
        mock_adapter.save.side_effect = (
            lambda aid, blocks, metadata: tmp_path / f"{aid}.safetensors"
        )

        store = AgentCacheStore(
            cache_dir=tmp_path,
            max_hot_agents=5,
            model_tag=tag,
            cache_adapter=mock_adapter,
        )

        # Add 3 hot caches
        for i in range(3):
            blocks = AgentBlocks(
                agent_id=f"agent_{i}",
                blocks={},
                total_tokens=0,  # Empty blocks
            )
            store.save(f"agent_{i}", blocks)

        assert len(store._hot_cache) == 3

        # Execute evict_all
        evicted_count = store.evict_all_to_disk()

        # Verify
        assert evicted_count == 3
        assert len(store._hot_cache) == 0  # Hot tier cleared
        assert len(store._warm_cache) == 3  # All in warm tier
        # Write-behind: save() defers disk write, only eviction flushes (3 calls)
        assert mock_adapter.save.call_count == 3

    def test_evict_all_to_disk_with_empty_hot_tier(self, tmp_path):
        """Evicting when hot tier is empty returns 0."""
        spec = ModelCacheSpec(
            n_layers=24, n_kv_heads=8, head_dim=128, block_tokens=16, layer_types=["global"] * 24
        )
        tag = ModelTag.from_spec("test-model", spec)

        store = AgentCacheStore(
            cache_dir=tmp_path,
            max_hot_agents=5,
            model_tag=tag,
        )

        # No hot caches
        evicted_count = store.evict_all_to_disk()

        # Verify
        assert evicted_count == 0
        assert len(store._hot_cache) == 0

    def test_evict_all_preserves_agent_data(self, tmp_path):
        """Evicted caches remain in warm tier for reloading."""
        spec = ModelCacheSpec(
            n_layers=24, n_kv_heads=8, head_dim=128, block_tokens=16, layer_types=["global"] * 24
        )
        tag = ModelTag.from_spec("test-model", spec)

        # Mock adapter
        mock_adapter = Mock()
        mock_adapter.save.side_effect = (
            lambda aid, blocks, metadata: tmp_path / f"{aid}.safetensors"
        )

        store = AgentCacheStore(
            cache_dir=tmp_path,
            max_hot_agents=5,
            model_tag=tag,
            cache_adapter=mock_adapter,
        )

        # Create blocks
        blocks = AgentBlocks(
            agent_id="agent_1",
            blocks={},
            total_tokens=0,
        )

        # Save and evict
        store.save("agent_1", blocks)
        store.evict_all_to_disk()

        # Verify: hot tier empty, warm tier has entry
        assert len(store._hot_cache) == 0
        assert "agent_1" in store._warm_cache
        # Write-behind: save() defers disk write, only eviction flushes (1 call)
        assert mock_adapter.save.call_count == 1


class TestUpdateModelTag:
    """Test update_model_tag() for post-swap cache compatibility."""

    def test_update_model_tag_changes_current_tag(self, tmp_path):
        """Updating model tag changes the store's current tag."""
        old_spec = ModelCacheSpec(
            n_layers=24, n_kv_heads=8, head_dim=128, block_tokens=16, layer_types=["global"] * 24
        )
        old_tag = ModelTag.from_spec("old-model", old_spec)

        store = AgentCacheStore(
            cache_dir=tmp_path,
            max_hot_agents=5,
            model_tag=old_tag,
        )

        assert store.model_tag.model_id == "old-model"

        # Update to new tag
        new_spec = ModelCacheSpec(
            n_layers=32, n_kv_heads=16, head_dim=128, block_tokens=16, layer_types=["global"] * 32
        )
        new_tag = ModelTag.from_spec("new-model", new_spec)
        store.update_model_tag(new_tag)

        # Verify
        assert store.model_tag.model_id == "new-model"
        assert store.model_tag.n_layers == 32
        assert store.model_tag.n_kv_heads == 16

    def test_incompatible_cache_rejected_after_tag_update(self, tmp_path):
        """Updating to incompatible tag changes validation."""
        old_spec = ModelCacheSpec(
            n_layers=24, n_kv_heads=8, head_dim=128, block_tokens=16, layer_types=["global"] * 24
        )
        old_tag = ModelTag.from_spec("old-model", old_spec)

        store = AgentCacheStore(
            cache_dir=tmp_path,
            max_hot_agents=5,
            model_tag=old_tag,
        )

        # Update to incompatible model (different n_layers)
        new_spec = ModelCacheSpec(
            n_layers=32,  # Different!
            n_kv_heads=8,
            head_dim=128,
            block_tokens=16,
            layer_types=["global"] * 32,
        )
        new_tag = ModelTag.from_spec("new-model", new_spec)
        store.update_model_tag(new_tag)

        # Verify tag updated correctly
        assert store.model_tag.model_id == "new-model"
        assert store.model_tag.n_layers == 32

    def test_compatible_cache_reloaded_after_tag_update(self, tmp_path):
        """Updating to compatible tag allows cache reloading."""
        old_spec = ModelCacheSpec(
            n_layers=24, n_kv_heads=8, head_dim=128, block_tokens=16, layer_types=["global"] * 24
        )
        old_tag = ModelTag.from_spec("old-model-v1", old_spec)

        store = AgentCacheStore(
            cache_dir=tmp_path,
            max_hot_agents=5,
            model_tag=old_tag,
        )

        # Update to compatible model (same spec, different model_id)
        new_spec = ModelCacheSpec(
            n_layers=24,  # Same
            n_kv_heads=8,  # Same
            head_dim=128,  # Same
            block_tokens=16,  # Same
            layer_types=["global"] * 24,
        )
        new_tag = ModelTag.from_spec("old-model-v2", new_spec)  # Different ID, same spec
        store.update_model_tag(new_tag)

        # Verify tag updated
        assert store.model_tag.model_id == "old-model-v2"
        assert store.model_tag.n_layers == 24


class TestModelTagCompatibility:
    """Test ModelTag.is_compatible() validation logic."""

    def test_identical_specs_are_compatible(self):
        """Identical specs are compatible."""
        tag = ModelTag(model_id="model-a", n_layers=24, n_kv_heads=8, head_dim=128, block_tokens=16)
        spec = ModelCacheSpec(
            n_layers=24, n_kv_heads=8, head_dim=128, block_tokens=16, layer_types=["global"] * 24
        )

        assert tag.is_compatible(spec)

    def test_different_n_layers_incompatible(self):
        """Different n_layers makes caches incompatible."""
        tag = ModelTag(model_id="model-a", n_layers=24, n_kv_heads=8, head_dim=128, block_tokens=16)
        spec = ModelCacheSpec(
            n_layers=32,  # Different
            n_kv_heads=8,
            head_dim=128,
            block_tokens=16,
            layer_types=["global"] * 32,
        )

        assert not tag.is_compatible(spec)

    def test_different_n_kv_heads_incompatible(self):
        """Different n_kv_heads makes caches incompatible."""
        tag = ModelTag(model_id="model-a", n_layers=24, n_kv_heads=8, head_dim=128, block_tokens=16)
        spec = ModelCacheSpec(
            n_layers=24,
            n_kv_heads=16,  # Different
            head_dim=128,
            block_tokens=16,
            layer_types=["global"] * 24,
        )

        assert not tag.is_compatible(spec)

    def test_different_head_dim_incompatible(self):
        """Different head_dim makes caches incompatible."""
        tag = ModelTag(model_id="model-a", n_layers=24, n_kv_heads=8, head_dim=128, block_tokens=16)
        spec = ModelCacheSpec(
            n_layers=24,
            n_kv_heads=8,
            head_dim=256,  # Different
            block_tokens=16,
            layer_types=["global"] * 24,
        )

        assert not tag.is_compatible(spec)

    def test_different_block_tokens_incompatible(self):
        """Different block_tokens makes caches incompatible."""
        tag = ModelTag(model_id="model-a", n_layers=24, n_kv_heads=8, head_dim=128, block_tokens=16)
        spec = ModelCacheSpec(
            n_layers=24,
            n_kv_heads=8,
            head_dim=128,
            block_tokens=32,  # Different
            layer_types=["global"] * 24,
        )

        assert not tag.is_compatible(spec)

    def test_model_id_difference_does_not_affect_compatibility(self):
        """Model ID difference doesn't affect compatibility (only spec matters)."""
        tag = ModelTag(
            model_id="gemma-3-12b-v1", n_layers=24, n_kv_heads=8, head_dim=128, block_tokens=16
        )
        spec = ModelCacheSpec(
            n_layers=24, n_kv_heads=8, head_dim=128, block_tokens=16, layer_types=["global"] * 24
        )

        # Same spec, different model_id -> compatible
        assert tag.is_compatible(spec)


class TestDiskLoadRejectsIncompatible:
    """Verify _load_from_disk rejects cache when model tag doesn't match."""

    def _make_store_with_cache(self, tmp_path, tag, adapter=None):
        """Helper to create a store with one saved cache entry."""
        from semantic.domain.entities import KVBlock

        if adapter is None:
            adapter = Mock()
            adapter.save.side_effect = (
                lambda aid, blocks, metadata: tmp_path / f"{aid}.safetensors"
            )

        store = AgentCacheStore(
            cache_dir=tmp_path,
            max_hot_agents=5,
            model_tag=tag,
            cache_adapter=adapter,
        )

        block = KVBlock(block_id=0, layer_id=0, token_count=16, layer_data="fake")
        blocks = AgentBlocks(
            agent_id="agent_1",
            blocks={0: [block]},
            total_tokens=16,
            token_sequence=[1, 2, 3],
        )
        store.save("agent_1", blocks)
        return store

    def test_disk_load_rejects_n_layers_mismatch(self, tmp_path):
        """Disk load returns None when n_layers differs."""
        from pathlib import Path

        original_tag = ModelTag(
            model_id="model", n_layers=24, n_kv_heads=8, head_dim=128, block_tokens=16,
        )

        # Create an in-memory adapter that returns metadata with original tag
        saved_data = {}

        class FakeAdapter:
            def save(self, agent_id, blocks, metadata):
                path = tmp_path / f"{agent_id}.safetensors"
                path.write_bytes(b"x")
                saved_data[agent_id] = (blocks, metadata)
                return path

            def load(self, path):
                agent_id = Path(path).stem
                blocks_obj, metadata = saved_data[agent_id]
                return blocks_obj.blocks, metadata

        store = self._make_store_with_cache(tmp_path, original_tag, FakeAdapter())
        store.invalidate_hot("agent_1")

        # Change to model with different n_layers
        store.model_tag = ModelTag(
            model_id="model", n_layers=32, n_kv_heads=8, head_dim=128, block_tokens=16,
        )
        assert store.load("agent_1") is None

    def test_disk_load_rejects_n_kv_heads_mismatch(self, tmp_path):
        """Disk load returns None when n_kv_heads differs."""
        from pathlib import Path

        original_tag = ModelTag(
            model_id="model", n_layers=24, n_kv_heads=8, head_dim=128, block_tokens=16,
        )

        saved_data = {}

        class FakeAdapter:
            def save(self, agent_id, blocks, metadata):
                path = tmp_path / f"{agent_id}.safetensors"
                path.write_bytes(b"x")
                saved_data[agent_id] = (blocks, metadata)
                return path

            def load(self, path):
                agent_id = Path(path).stem
                blocks_obj, metadata = saved_data[agent_id]
                return blocks_obj.blocks, metadata

        store = self._make_store_with_cache(tmp_path, original_tag, FakeAdapter())
        store.invalidate_hot("agent_1")

        store.model_tag = ModelTag(
            model_id="model", n_layers=24, n_kv_heads=16, head_dim=128, block_tokens=16,
        )
        assert store.load("agent_1") is None

    def test_disk_load_rejects_head_dim_mismatch(self, tmp_path):
        """Disk load returns None when head_dim differs."""
        from pathlib import Path

        original_tag = ModelTag(
            model_id="model", n_layers=24, n_kv_heads=8, head_dim=128, block_tokens=16,
        )

        saved_data = {}

        class FakeAdapter:
            def save(self, agent_id, blocks, metadata):
                path = tmp_path / f"{agent_id}.safetensors"
                path.write_bytes(b"x")
                saved_data[agent_id] = (blocks, metadata)
                return path

            def load(self, path):
                agent_id = Path(path).stem
                blocks_obj, metadata = saved_data[agent_id]
                return blocks_obj.blocks, metadata

        store = self._make_store_with_cache(tmp_path, original_tag, FakeAdapter())
        store.invalidate_hot("agent_1")

        store.model_tag = ModelTag(
            model_id="model", n_layers=24, n_kv_heads=8, head_dim=256, block_tokens=16,
        )
        assert store.load("agent_1") is None

    def test_disk_load_rejects_block_tokens_mismatch(self, tmp_path):
        """Disk load returns None when block_tokens differs."""
        from pathlib import Path

        original_tag = ModelTag(
            model_id="model", n_layers=24, n_kv_heads=8, head_dim=128, block_tokens=16,
        )

        saved_data = {}

        class FakeAdapter:
            def save(self, agent_id, blocks, metadata):
                path = tmp_path / f"{agent_id}.safetensors"
                path.write_bytes(b"x")
                saved_data[agent_id] = (blocks, metadata)
                return path

            def load(self, path):
                agent_id = Path(path).stem
                blocks_obj, metadata = saved_data[agent_id]
                return blocks_obj.blocks, metadata

        store = self._make_store_with_cache(tmp_path, original_tag, FakeAdapter())
        store.invalidate_hot("agent_1")

        store.model_tag = ModelTag(
            model_id="model", n_layers=24, n_kv_heads=8, head_dim=128, block_tokens=32,
        )
        assert store.load("agent_1") is None


class TestInvalidateHot:
    """Test invalidate_hot() preserves warm tier."""

    def test_invalidate_hot_preserves_warm_tier(self, tmp_path):
        """invalidate_hot() removes from hot but keeps warm entry intact."""
        tag = ModelTag(
            model_id="model", n_layers=24, n_kv_heads=8, head_dim=128, block_tokens=16,
        )
        mock_adapter = Mock()
        mock_adapter.save.side_effect = (
            lambda aid, blocks, metadata: tmp_path / f"{aid}.safetensors"
        )

        store = AgentCacheStore(
            cache_dir=tmp_path,
            max_hot_agents=5,
            model_tag=tag,
            cache_adapter=mock_adapter,
        )

        blocks = AgentBlocks(agent_id="agent_1", blocks={}, total_tokens=0)
        store.save("agent_1", blocks)

        assert "agent_1" in store._hot_cache

        # Evict to warm tier first (write-behind: save() doesn't write to disk)
        store.evict_all_to_disk()
        assert "agent_1" in store._warm_cache
        assert "agent_1" not in store._hot_cache

        # Simulate reload to hot tier with a proper CacheEntry
        store._hot_cache["agent_1"] = CacheEntry(
            agent_id="agent_1",
            blocks=blocks,
            model_tag=tag,
            dirty=False,
        )
        store.invalidate_hot("agent_1")

        assert "agent_1" not in store._hot_cache
        assert "agent_1" in store._warm_cache

    def test_invalidate_hot_noop_for_missing_agent(self, tmp_path):
        """invalidate_hot() on non-existent agent is a no-op."""
        tag = ModelTag(
            model_id="model", n_layers=24, n_kv_heads=8, head_dim=128, block_tokens=16,
        )
        store = AgentCacheStore(
            cache_dir=tmp_path, max_hot_agents=5, model_tag=tag,
        )

        # Should not raise
        store.invalidate_hot("nonexistent")
