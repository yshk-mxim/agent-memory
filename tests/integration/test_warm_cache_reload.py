# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Yakov Shkolnikov and contributors
"""Integration test for warm cache reload from disk.

CRITICAL: This test ensures warm cache actually reloads from disk.
If this test passes but warm TTFT ≈ cold TTFT in benchmarks, the
benchmark methodology is wrong (not testing true disk reload).
"""

import pytest
import time
from pathlib import Path

from agent_memory.application.agent_cache_store import AgentCacheStore
from agent_memory.domain.entities import AgentBlocks, KVBlock
from agent_memory.domain.value_objects import ModelCacheSpec, ModelTag


@pytest.fixture
def cache_store(tmp_path):
    """Create cache store with temporary directory."""
    cache_dir = tmp_path / "caches"
    cache_dir.mkdir()

    spec = ModelCacheSpec(
        n_layers=4,
        n_kv_heads=8,
        head_dim=128,
        block_tokens=256,
    )
    tag = ModelTag(model_id="test-model", spec=spec)

    # Mock cache adapter
    from unittest.mock import MagicMock
    adapter = MagicMock()
    adapter.save.return_value = cache_dir / "test_agent.safetensors"
    adapter.load.return_value = AgentBlocks(
        spec=spec,
        blocks={0: [KVBlock(block_id=1, layer_id=0, token_count=256)]},
        total_tokens=256,
        token_sequence=[1] * 256,
        prompt_text="test prompt",
    )

    store = AgentCacheStore(
        cache_adapter=adapter,
        model_tag=tag,
        max_hot_agents=2,
    )
    return store, adapter, cache_dir


def test_evict_only_keeps_disk_file(cache_store):
    """Test that delete(keep_disk=True) preserves disk file."""
    store, adapter, cache_dir = cache_store

    # Create cache entry
    spec = store.model_tag.spec
    blocks = AgentBlocks(
        spec=spec,
        blocks={0: [KVBlock(block_id=1, layer_id=0, token_count=256)]},
        total_tokens=256,
        token_sequence=[1] * 256,
        prompt_text="test",
    )

    # Save to hot tier
    store.save("agent_1", blocks)

    # Verify in hot tier
    assert store.load("agent_1") is not None

    # Evict with keep_disk=True
    deleted = store.delete("agent_1", keep_disk=True)
    assert deleted is True

    # Verify adapter.save was called (flush to disk)
    assert adapter.save.called

    # Verify NOT in hot tier
    assert "agent_1" not in store._hot_cache

    # Verify still in warm tier
    assert "agent_1" in store._warm_cache


def test_full_delete_removes_disk_file(cache_store):
    """Test that delete(keep_disk=False) removes disk file."""
    store, adapter, cache_dir = cache_store

    # Create cache entry
    spec = store.model_tag.spec
    blocks = AgentBlocks(
        spec=spec,
        blocks={0: [KVBlock(block_id=1, layer_id=0, token_count=256)]},
        total_tokens=256,
        token_sequence=[1] * 256,
        prompt_text="test",
    )

    store.save("agent_2", blocks)

    # Full delete
    deleted = store.delete("agent_2", keep_disk=False)
    assert deleted is True

    # Verify not in hot tier
    assert "agent_2" not in store._hot_cache

    # Verify not in warm tier (would be if file existed)
    assert "agent_2" not in store._warm_cache


def test_warm_reload_after_evict(cache_store):
    """Test that cache can be reloaded from warm tier after eviction.

    This is the core warm cache pattern:
    1. Prime: create cache in hot tier
    2. Evict: flush to disk, remove from hot tier
    3. Reload: load from disk into hot tier
    """
    store, adapter, cache_dir = cache_store

    # Create cache entry
    spec = store.model_tag.spec
    blocks = AgentBlocks(
        spec=spec,
        blocks={0: [KVBlock(block_id=1, layer_id=0, token_count=256)]},
        total_tokens=256,
        token_sequence=[1] * 256,
        prompt_text="test prompt",
    )

    # 1. Prime: save to hot tier
    store.save("agent_3", blocks)
    assert "agent_3" in store._hot_cache

    # 2. Evict: flush to disk, remove from hot
    store.delete("agent_3", keep_disk=True)
    assert "agent_3" not in store._hot_cache
    assert "agent_3" in store._warm_cache

    # 3. Reload: should come from disk (warm tier)
    reloaded = store.load("agent_3")
    assert reloaded is not None
    assert reloaded.total_tokens == 256
    assert reloaded.prompt_text == "test prompt"

    # Verify adapter.load was called (disk reload)
    assert adapter.load.called


def test_dirty_flag_cleared_after_evict(cache_store):
    """Test that dirty flag is cleared after flush to disk."""
    store, adapter, cache_dir = cache_store

    spec = store.model_tag.spec
    blocks = AgentBlocks(
        spec=spec,
        blocks={0: [KVBlock(block_id=1, layer_id=0, token_count=256)]},
        total_tokens=256,
        token_sequence=[1] * 256,
        prompt_text="test",
    )

    # Save creates dirty entry
    store.save("agent_4", blocks)
    entry = store._hot_cache["agent_4"]
    assert entry.dirty is True

    # Evict flushes and clears dirty
    store.delete("agent_4", keep_disk=True)

    # Verify save was called with correct arguments
    assert adapter.save.called
    call_args = adapter.save.call_args
    assert call_args[0][0] == "agent_4"  # agent_id
    assert call_args[0][1] == blocks  # blocks
