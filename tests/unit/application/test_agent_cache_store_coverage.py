# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Yakov Shkolnikov and contributors
"""Coverage tests for AgentCacheStore — list_all_agents, get_agent_metadata,
flush_dirty, _scan_cache_directory, delete, invalidate_hot, evict_all_to_disk,
_load_from_disk error paths, CacheMetrics.
"""

from unittest.mock import MagicMock

import pytest

from agent_memory.application.agent_cache_store import (
    AgentCacheStore,
    CacheEntry,
    CacheMetrics,
    ModelTag,
)
from agent_memory.domain.entities import AgentBlocks, KVBlock

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tag(**overrides):
    defaults = dict(
        model_id="test-model",
        n_layers=12,
        n_kv_heads=4,
        head_dim=128,
        block_tokens=256,
    )
    defaults.update(overrides)
    return ModelTag(**defaults)


def _make_blocks(agent_id="agent_1", total_tokens=100, with_data=True):
    layer_data = {"fake": True} if with_data else None
    blocks = {
        0: [KVBlock(block_id=0, layer_id=0, token_count=total_tokens, layer_data=layer_data)],
    }
    return AgentBlocks(
        agent_id=agent_id,
        blocks=blocks,
        total_tokens=total_tokens,
        token_sequence=[1, 2, 3],
        prompt_text="Hello world test prompt " * 10,
    )


def _make_store(tmp_path, cache_adapter=None, max_hot=5, tag=None):
    if tag is None:
        tag = _make_tag()
    return AgentCacheStore(
        cache_dir=tmp_path,
        max_hot_agents=max_hot,
        model_tag=tag,
        cache_adapter=cache_adapter,
    )


# ===========================================================================
# CacheMetrics
# ===========================================================================


class TestCacheMetrics:
    def test_hit_rate_zero_total(self):
        m = CacheMetrics()
        assert m.hit_rate() == 0.0

    def test_warm_hit_rate_zero_total(self):
        m = CacheMetrics()
        assert m.warm_hit_rate() == 0.0

    def test_hit_rate_with_hits(self):
        m = CacheMetrics(hot_hits=3, warm_hits=2, misses=5)
        assert m.hit_rate() == pytest.approx(0.5)

    def test_warm_hit_rate_with_hits(self):
        m = CacheMetrics(hot_hits=3, warm_hits=2, misses=5)
        assert m.warm_hit_rate() == pytest.approx(0.2)


# ===========================================================================
# _scan_cache_directory
# ===========================================================================


class TestScanCacheDirectory:
    def test_scan_finds_safetensors(self, tmp_path):
        (tmp_path / "agent_1.safetensors").write_bytes(b"fake")
        (tmp_path / "agent_2.safetensors").write_bytes(b"fake")
        store = _make_store(tmp_path)
        assert "agent_1" in store._warm_cache
        assert "agent_2" in store._warm_cache

    def test_scan_empty_dir(self, tmp_path):
        store = _make_store(tmp_path)
        assert len(store._warm_cache) == 0

    def test_scan_nonexistent_dir(self, tmp_path):
        missing = tmp_path / "missing"
        # AgentCacheStore.__init__ creates the dir, so we test
        # that no error occurs
        store = _make_store(missing)
        assert len(store._warm_cache) == 0


# ===========================================================================
# list_all_agents
# ===========================================================================


class TestListAllAgents:
    def test_hot_only_agent(self, tmp_path):
        store = _make_store(tmp_path)
        entry = CacheEntry(
            agent_id="hot_1",
            blocks=_make_blocks("hot_1", total_tokens=50),
            model_tag=store.model_tag,
            dirty=True,
        )
        entry.mark_accessed()
        store._hot_cache["hot_1"] = entry

        agents = store.list_all_agents()
        assert len(agents) == 1
        a = agents[0]
        assert a["agent_id"] == "hot_1"
        assert a["tier"] == "hot"
        assert a["tokens"] == 50
        assert a["dirty"] is True
        assert a["file_size_bytes"] == 0  # No disk file

    def test_warm_only_agent(self, tmp_path):
        sf = tmp_path / "warm_1.safetensors"
        sf.write_bytes(b"x" * 42)
        store = _make_store(tmp_path)
        # scan already populated warm_cache

        agents = store.list_all_agents()
        assert len(agents) == 1
        a = agents[0]
        assert a["agent_id"] == "warm_1"
        assert a["tier"] == "warm"
        assert a["tokens"] == 0
        assert a["last_accessed"] == 0.0
        assert a["file_size_bytes"] == 42

    def test_agent_in_both_hot_and_warm(self, tmp_path):
        sf = tmp_path / "both_1.safetensors"
        sf.write_bytes(b"x" * 10)
        store = _make_store(tmp_path)

        entry = CacheEntry(
            agent_id="both_1",
            blocks=_make_blocks("both_1"),
            model_tag=store.model_tag,
        )
        store._hot_cache["both_1"] = entry

        agents = store.list_all_agents()
        # Should appear once as hot (warm_ids - hot_ids)
        assert len(agents) == 1
        assert agents[0]["tier"] == "hot"
        assert agents[0]["file_size_bytes"] == 10

    def test_empty_store(self, tmp_path):
        store = _make_store(tmp_path)
        assert store.list_all_agents() == []

    def test_file_size_from_disk(self, tmp_path):
        data = b"a" * 1024
        (tmp_path / "agent_x.safetensors").write_bytes(data)
        store = _make_store(tmp_path)
        entry = CacheEntry(
            agent_id="agent_x",
            blocks=_make_blocks("agent_x"),
            model_tag=store.model_tag,
        )
        store._hot_cache["agent_x"] = entry
        agents = store.list_all_agents()
        assert agents[0]["file_size_bytes"] == 1024


# ===========================================================================
# get_agent_metadata
# ===========================================================================


class TestGetAgentMetadata:
    def test_hot_tier_agent(self, tmp_path):
        store = _make_store(tmp_path)
        blocks = _make_blocks("agent_h", total_tokens=200)
        entry = CacheEntry(agent_id="agent_h", blocks=blocks, model_tag=store.model_tag)
        entry.mark_accessed()
        store._hot_cache["agent_h"] = entry

        meta = store.get_agent_metadata("agent_h")
        assert meta is not None
        assert meta["tier"] == "hot"
        assert meta["tokens"] == 200
        assert len(meta["prompt_preview"]) <= 100

    def test_warm_tier_with_adapter(self, tmp_path):
        adapter = MagicMock()
        adapter.get_cache_file_metadata.return_value = {
            "total_tokens": 500,
            "model_id": "test-model",
            "prompt_text": "Short preview",
        }
        sf = tmp_path / "warm_a.safetensors"
        sf.write_bytes(b"x" * 100)
        store = _make_store(tmp_path, cache_adapter=adapter)

        meta = store.get_agent_metadata("warm_a")
        assert meta is not None
        assert meta["tier"] == "warm"
        assert meta["tokens"] == 500

    def test_warm_tier_file_missing(self, tmp_path):
        store = _make_store(tmp_path)
        # Manually add to warm cache with nonexistent path
        store._warm_cache["gone"] = tmp_path / "gone.safetensors"
        assert store.get_agent_metadata("gone") is None

    def test_warm_tier_adapter_exception(self, tmp_path):
        adapter = MagicMock()
        adapter.get_cache_file_metadata.side_effect = RuntimeError("boom")
        sf = tmp_path / "err_a.safetensors"
        sf.write_bytes(b"x")
        store = _make_store(tmp_path, cache_adapter=adapter)

        assert store.get_agent_metadata("err_a") is None

    def test_not_found(self, tmp_path):
        store = _make_store(tmp_path)
        assert store.get_agent_metadata("nonexistent") is None


# ===========================================================================
# flush_dirty
# ===========================================================================


class TestFlushDirty:
    def test_one_dirty_entry(self, tmp_path):
        adapter = MagicMock()
        adapter.save.return_value = tmp_path / "d1.safetensors"
        store = _make_store(tmp_path, cache_adapter=adapter)

        entry = CacheEntry(
            agent_id="d1",
            blocks=_make_blocks("d1"),
            model_tag=store.model_tag,
            dirty=True,
        )
        store._hot_cache["d1"] = entry
        assert store.flush_dirty() == 1

    def test_no_dirty_entries(self, tmp_path):
        store = _make_store(tmp_path)
        entry = CacheEntry(
            agent_id="clean",
            blocks=_make_blocks("clean"),
            model_tag=store.model_tag,
            dirty=False,
        )
        store._hot_cache["clean"] = entry
        assert store.flush_dirty() == 0

    def test_entry_with_none_blocks_skipped(self, tmp_path):
        store = _make_store(tmp_path)
        entry = CacheEntry(
            agent_id="no_blocks",
            blocks=None,
            model_tag=store.model_tag,
            dirty=True,
        )
        store._hot_cache["no_blocks"] = entry
        assert store.flush_dirty() == 0


# ===========================================================================
# delete
# ===========================================================================


class TestDelete:
    def test_keep_disk_true(self, tmp_path):
        sf = tmp_path / "del1.safetensors"
        sf.write_bytes(b"data")
        adapter = MagicMock()
        adapter.save.return_value = sf
        store = _make_store(tmp_path, cache_adapter=adapter)

        entry = CacheEntry(
            agent_id="del1",
            blocks=_make_blocks("del1"),
            model_tag=store.model_tag,
            dirty=False,
        )
        store._hot_cache["del1"] = entry

        found = store.delete("del1", keep_disk=True)
        assert found is True
        assert "del1" not in store._hot_cache
        assert sf.exists()  # File untouched

    def test_keep_disk_false(self, tmp_path):
        sf = tmp_path / "del2.safetensors"
        sf.write_bytes(b"data")
        store = _make_store(tmp_path)
        store._warm_cache["del2"] = sf

        entry = CacheEntry(
            agent_id="del2",
            blocks=_make_blocks("del2"),
            model_tag=store.model_tag,
            dirty=False,
        )
        store._hot_cache["del2"] = entry

        found = store.delete("del2", keep_disk=False)
        assert found is True
        assert "del2" not in store._hot_cache
        assert "del2" not in store._warm_cache
        assert not sf.exists()

    def test_agent_not_found(self, tmp_path):
        store = _make_store(tmp_path)
        assert store.delete("nope") is False


# ===========================================================================
# invalidate_hot
# ===========================================================================


class TestInvalidateHot:
    def test_dirty_agent_flushed_then_removed(self, tmp_path):
        adapter = MagicMock()
        adapter.save.return_value = tmp_path / "inv1.safetensors"
        store = _make_store(tmp_path, cache_adapter=adapter)

        entry = CacheEntry(
            agent_id="inv1",
            blocks=_make_blocks("inv1"),
            model_tag=store.model_tag,
            dirty=True,
        )
        store._hot_cache["inv1"] = entry
        store.invalidate_hot("inv1")

        assert "inv1" not in store._hot_cache
        adapter.save.assert_called_once()

    def test_clean_agent_removed_without_save(self, tmp_path):
        adapter = MagicMock()
        store = _make_store(tmp_path, cache_adapter=adapter)

        entry = CacheEntry(
            agent_id="inv2",
            blocks=_make_blocks("inv2"),
            model_tag=store.model_tag,
            dirty=False,
        )
        store._hot_cache["inv2"] = entry
        store.invalidate_hot("inv2")

        assert "inv2" not in store._hot_cache
        adapter.save.assert_not_called()

    def test_agent_not_in_hot_silent(self, tmp_path):
        store = _make_store(tmp_path)
        store.invalidate_hot("ghost")  # Should not raise


# ===========================================================================
# evict_all_to_disk
# ===========================================================================


class TestEvictAllToDisk:
    def test_multiple_hot_agents(self, tmp_path):
        adapter = MagicMock()
        adapter.save.return_value = tmp_path / "x.safetensors"
        store = _make_store(tmp_path, cache_adapter=adapter)

        for i in range(3):
            aid = f"ea_{i}"
            entry = CacheEntry(
                agent_id=aid,
                blocks=_make_blocks(aid),
                model_tag=store.model_tag,
            )
            store._hot_cache[aid] = entry

        evicted = store.evict_all_to_disk()
        assert evicted == 3
        assert len(store._hot_cache) == 0

    def test_empty_store(self, tmp_path):
        store = _make_store(tmp_path)
        assert store.evict_all_to_disk() == 0


# ===========================================================================
# _load_from_disk error paths
# ===========================================================================


class TestLoadFromDiskErrors:
    def test_incompatible_model_tag(self, tmp_path):
        adapter = MagicMock()
        # Return metadata with mismatched n_layers
        adapter.load.return_value = (
            {},
            {
                "model_id": "test-model",
                "n_layers": "999",
                "n_kv_heads": "4",
                "head_dim": "128",
                "block_tokens": "256",
                "total_tokens": "100",
                "token_sequence": "[]",
            },
        )
        sf = tmp_path / "bad_tag.safetensors"
        sf.write_bytes(b"data")
        store = _make_store(tmp_path, cache_adapter=adapter)

        result = store._load_from_disk("bad_tag")
        assert result is None

    def test_malformed_json_token_sequence(self, tmp_path):
        tag = _make_tag()
        adapter = MagicMock()
        adapter.load.return_value = (
            {},
            {
                "model_id": tag.model_id,
                "n_layers": str(tag.n_layers),
                "n_kv_heads": str(tag.n_kv_heads),
                "head_dim": str(tag.head_dim),
                "block_tokens": str(tag.block_tokens),
                "total_tokens": "0",
                "token_sequence": "not-json{{{",
            },
        )
        sf = tmp_path / "bad_json.safetensors"
        sf.write_bytes(b"data")
        store = _make_store(tmp_path, cache_adapter=adapter)

        result = store._load_from_disk("bad_json")
        # Should not raise, falls back to empty list (total_tokens=0 with empty blocks is valid)
        assert result is not None
        assert result.token_sequence == []

    def test_no_cache_adapter_raises(self, tmp_path):
        store = _make_store(tmp_path, cache_adapter=None)
        sf = tmp_path / "no_adapt.safetensors"
        sf.write_bytes(b"data")
        store._warm_cache["no_adapt"] = sf

        # _load_from_disk should catch and return None (InvalidRequestError → generic Exception path)
        result = store._load_from_disk("no_adapt")
        assert result is None

    def test_os_error_during_load(self, tmp_path):
        adapter = MagicMock()
        adapter.load.side_effect = OSError("disk failure")
        sf = tmp_path / "os_err.safetensors"
        sf.write_bytes(b"data")
        store = _make_store(tmp_path, cache_adapter=adapter)

        result = store._load_from_disk("os_err")
        assert result is None

    def test_value_error_during_load(self, tmp_path):
        adapter = MagicMock()
        adapter.load.side_effect = ValueError("bad format")
        sf = tmp_path / "val_err.safetensors"
        sf.write_bytes(b"data")
        store = _make_store(tmp_path, cache_adapter=adapter)

        result = store._load_from_disk("val_err")
        assert result is None
