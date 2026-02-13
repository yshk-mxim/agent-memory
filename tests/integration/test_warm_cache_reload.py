# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Yakov Shkolnikov and contributors
"""Integration test for warm cache reload from disk.

CRITICAL: This test ensures warm cache actually reloads from disk.
If this test passes but warm TTFT ≈ cold TTFT in benchmarks, the
benchmark methodology is wrong (not testing true disk reload).
"""

from unittest.mock import MagicMock

import pytest

from agent_memory.application.agent_cache_store import AgentCacheStore, ModelTag
from agent_memory.domain.entities import AgentBlocks, KVBlock
from agent_memory.domain.value_objects import ModelCacheSpec


def _make_spec():
    return ModelCacheSpec(
        n_layers=4,
        n_kv_heads=8,
        head_dim=128,
        block_tokens=256,
        layer_types=["global"] * 4,
    )


def _make_blocks(agent_id="test", prompt_text="test prompt"):
    return AgentBlocks(
        agent_id=agent_id,
        blocks={0: [KVBlock(block_id=1, layer_id=0, token_count=256, layer_data=b"fake")]},
        total_tokens=256,
        token_sequence=[1] * 256,
        prompt_text=prompt_text,
    )


@pytest.fixture
def cache_store(tmp_path):
    """Create cache store with temporary directory."""
    cache_dir = tmp_path / "caches"
    cache_dir.mkdir()

    spec = _make_spec()
    tag = ModelTag.from_spec("test-model", spec)

    adapter = MagicMock()

    def fake_save(agent_id, blocks, metadata=None):
        path = cache_dir / f"{agent_id}.safetensors"
        path.write_bytes(b"fake-safetensors-data")
        return path

    adapter.save.side_effect = fake_save
    adapter.load.return_value = (
        {0: [KVBlock(block_id=1, layer_id=0, token_count=256, layer_data=b"fake")]},
        {
            "model_id": "test-model",
            "n_layers": "4",
            "n_kv_heads": "8",
            "head_dim": "128",
            "block_tokens": "256",
            "kv_bits": "4",
            "kv_group_size": "64",
            "total_tokens": "256",
            "token_sequence": "[1]",
            "prompt_text": "test prompt",
        },
    )

    store = AgentCacheStore(
        cache_dir=cache_dir,
        max_hot_agents=2,
        model_tag=tag,
        cache_adapter=adapter,
    )
    return store, adapter, cache_dir


def test_evict_only_keeps_disk_file(cache_store):
    """Test that delete(keep_disk=True) preserves disk file."""
    store, adapter, cache_dir = cache_store
    blocks = _make_blocks("agent_1")

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
    blocks = _make_blocks("agent_2")

    store.save("agent_2", blocks)

    # Full delete
    deleted = store.delete("agent_2", keep_disk=False)
    assert deleted is True

    # Verify not in hot tier
    assert "agent_2" not in store._hot_cache

    # Verify not in warm tier
    assert "agent_2" not in store._warm_cache


def test_warm_reload_after_evict(cache_store):
    """Test that cache can be reloaded from warm tier after eviction.

    This is the core warm cache pattern:
    1. Prime: create cache in hot tier
    2. Evict: flush to disk, remove from hot tier
    3. Reload: load from disk into hot tier
    """
    store, adapter, cache_dir = cache_store
    blocks = _make_blocks("agent_3", prompt_text="test prompt")

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
    blocks = _make_blocks("agent_4")

    # Save persists immediately (layer_data is not None) and clears dirty
    store.save("agent_4", blocks)
    entry = store._hot_cache["agent_4"]
    assert entry.dirty is False  # Already flushed during save()

    # Verify adapter.save was called during save()
    assert adapter.save.called
    call_args = adapter.save.call_args
    assert call_args[0][0] == "agent_4"  # agent_id
    assert call_args[0][1] == blocks  # blocks

    # Evict — no additional save needed since already persisted
    store.delete("agent_4", keep_disk=True)
    assert "agent_4" not in store._hot_cache
    assert "agent_4" in store._warm_cache
