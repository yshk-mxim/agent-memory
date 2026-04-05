# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Yakov Shkolnikov and contributors
"""Unit tests for TRT safetensors cache adapter — save/load, validation, listing."""

from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from agent_memory.adapters.outbound.trt_safetensors_cache_adapter import (
    TRTSafetensorsCacheAdapter,
)
from agent_memory.domain.entities import AgentBlocks, KVBlock
from agent_memory.domain.errors import AgentNotFoundError, CachePersistenceError

pytestmark = pytest.mark.unit

N_LAYERS = 2
SEQ_LEN = 64
N_KV_HEADS = 4
HEAD_DIM = 32


def _make_fp16_blocks(agent_id: str, n_layers: int = N_LAYERS) -> AgentBlocks:
    """Build AgentBlocks with raw FP16 K/V arrays."""
    rng = np.random.default_rng(42)
    blocks: dict[int, list[KVBlock]] = {}
    for layer_id in range(n_layers):
        k = rng.standard_normal((N_KV_HEADS, SEQ_LEN, HEAD_DIM)).astype(np.float16)
        v = rng.standard_normal((N_KV_HEADS, SEQ_LEN, HEAD_DIM)).astype(np.float16)
        block = KVBlock(
            block_id=layer_id * 1_000_000,
            layer_id=layer_id,
            token_count=SEQ_LEN,
            layer_data=(k, v),
        )
        blocks[layer_id] = [block]
    return AgentBlocks(agent_id=agent_id, blocks=blocks, total_tokens=SEQ_LEN)


def _make_q4_blocks(agent_id: str, n_layers: int = N_LAYERS) -> AgentBlocks:
    """Build AgentBlocks with pre-quantized (weights, scales, biases) tuples."""
    rng = np.random.default_rng(99)
    blocks: dict[int, list[KVBlock]] = {}
    n_elements = N_KV_HEADS * SEQ_LEN * HEAD_DIM
    n_groups = n_elements // 64
    for layer_id in range(n_layers):
        kw = rng.integers(0, 16, size=(n_elements // 2,), dtype=np.uint8)
        ks = rng.standard_normal((n_groups,)).astype(np.float16)
        kb = rng.standard_normal((n_groups,)).astype(np.float16)
        vw = rng.integers(0, 16, size=(n_elements // 2,), dtype=np.uint8)
        vs = rng.standard_normal((n_groups,)).astype(np.float16)
        vb = rng.standard_normal((n_groups,)).astype(np.float16)
        block = KVBlock(
            block_id=layer_id * 1_000_000,
            layer_id=layer_id,
            token_count=SEQ_LEN,
            layer_data=((kw, ks, kb), (vw, vs, vb)),
        )
        blocks[layer_id] = [block]
    return AgentBlocks(agent_id=agent_id, blocks=blocks, total_tokens=SEQ_LEN)


@pytest.fixture
def cache_dir(tmp_path: Path) -> Path:
    return tmp_path / "caches"


@pytest.fixture
def adapter(cache_dir: Path) -> TRTSafetensorsCacheAdapter:
    return TRTSafetensorsCacheAdapter(cache_dir=cache_dir, quantizer=None)


class TestSaveLoadFP16:
    def test_round_trip_fp16(self, adapter: TRTSafetensorsCacheAdapter) -> None:
        agent_blocks = _make_fp16_blocks("agent-fp16")
        metadata = {"total_tokens": str(SEQ_LEN), "model_id": "test-model"}

        path = adapter.save("agent-fp16", agent_blocks, metadata)
        assert path.exists()
        assert path.suffix == ".safetensors"

        loaded_blocks, loaded_meta = adapter.load("agent-fp16")

        assert loaded_blocks.agent_id == "agent-fp16"
        assert loaded_blocks.total_tokens == SEQ_LEN
        assert len(loaded_blocks.blocks) == N_LAYERS

        # Verify tensor data is preserved
        for layer_id in range(N_LAYERS):
            orig_k, orig_v = agent_blocks.blocks[layer_id][0].layer_data
            loaded_k, loaded_v = loaded_blocks.blocks[layer_id][0].layer_data
            np.testing.assert_array_equal(loaded_k, orig_k)
            np.testing.assert_array_equal(loaded_v, orig_v)

    def test_metadata_round_trip(self, adapter: TRTSafetensorsCacheAdapter) -> None:
        agent_blocks = _make_fp16_blocks("agent-meta")
        metadata = {"total_tokens": "64", "model_id": "smollm2"}

        adapter.save("agent-meta", agent_blocks, metadata)
        _, loaded_meta = adapter.load("agent-meta")

        assert loaded_meta["total_tokens"] == "64"
        assert loaded_meta["model_id"] == "smollm2"

    def test_save_without_metadata(self, adapter: TRTSafetensorsCacheAdapter) -> None:
        agent_blocks = _make_fp16_blocks("agent-nometa")
        path = adapter.save("agent-nometa", agent_blocks)
        assert path.exists()

        loaded_blocks, loaded_meta = adapter.load("agent-nometa")
        assert loaded_blocks.agent_id == "agent-nometa"


class TestSaveLoadQ4:
    def test_round_trip_q4_pre_quantized(self, adapter: TRTSafetensorsCacheAdapter) -> None:
        agent_blocks = _make_q4_blocks("agent-q4")
        metadata = {"total_tokens": str(SEQ_LEN)}

        path = adapter.save("agent-q4", agent_blocks, metadata)
        assert path.exists()

        loaded_blocks, loaded_meta = adapter.load("agent-q4")
        assert loaded_blocks.agent_id == "agent-q4"
        assert len(loaded_blocks.blocks) == N_LAYERS

        # Verify quantized structure: ((kw, ks, kb), (vw, vs, vb))
        for layer_id in range(N_LAYERS):
            data = loaded_blocks.blocks[layer_id][0].layer_data
            assert isinstance(data, tuple)
            assert len(data) == 2
            k_tuple, v_tuple = data
            assert isinstance(k_tuple, tuple) and len(k_tuple) == 3
            assert isinstance(v_tuple, tuple) and len(v_tuple) == 3

    def test_q4_tensor_values_preserved(self, adapter: TRTSafetensorsCacheAdapter) -> None:
        agent_blocks = _make_q4_blocks("agent-q4-vals")
        metadata = {"total_tokens": str(SEQ_LEN)}
        adapter.save("agent-q4-vals", agent_blocks, metadata)

        loaded_blocks, _ = adapter.load("agent-q4-vals")

        for layer_id in range(N_LAYERS):
            orig_data = agent_blocks.blocks[layer_id][0].layer_data
            loaded_data = loaded_blocks.blocks[layer_id][0].layer_data

            (orig_kw, orig_ks, orig_kb), (orig_vw, orig_vs, orig_vb) = orig_data
            (loaded_kw, loaded_ks, loaded_kb), (loaded_vw, loaded_vs, loaded_vb) = loaded_data

            np.testing.assert_array_equal(loaded_kw, orig_kw)
            np.testing.assert_array_equal(loaded_ks, orig_ks)
            np.testing.assert_array_equal(loaded_kb, orig_kb)
            np.testing.assert_array_equal(loaded_vw, orig_vw)
            np.testing.assert_array_equal(loaded_vs, orig_vs)
            np.testing.assert_array_equal(loaded_vb, orig_vb)


class TestSaveWithQuantizer:
    def test_fp16_data_quantized_on_save(self, cache_dir: Path) -> None:
        mock_quantizer = Mock()
        # quantize returns (weights, scales, biases)
        n_elements = N_KV_HEADS * SEQ_LEN * HEAD_DIM
        n_groups = n_elements // 64
        mock_quantizer.quantize.return_value = (
            np.zeros((n_elements // 2,), dtype=np.uint8),
            np.zeros((n_groups,), dtype=np.float16),
            np.zeros((n_groups,), dtype=np.float16),
        )

        adapter = TRTSafetensorsCacheAdapter(cache_dir=cache_dir, quantizer=mock_quantizer)
        agent_blocks = _make_fp16_blocks("agent-quant")
        adapter.save("agent-quant", agent_blocks)

        # quantize called twice per layer (K + V), times N_LAYERS
        assert mock_quantizer.quantize.call_count == N_LAYERS * 2


class TestValidateAgentId:
    def test_valid_ids(self, adapter: TRTSafetensorsCacheAdapter) -> None:
        # Should not raise
        adapter._validate_agent_id("agent-123")
        adapter._validate_agent_id("agent_abc")
        adapter._validate_agent_id("ABC-123_def")
        adapter._validate_agent_id("a")

    def test_empty_id_raises(self, adapter: TRTSafetensorsCacheAdapter) -> None:
        with pytest.raises(CachePersistenceError, match="Invalid agent_id length"):
            adapter._validate_agent_id("")

    def test_too_long_id_raises(self, adapter: TRTSafetensorsCacheAdapter) -> None:
        with pytest.raises(CachePersistenceError, match="Invalid agent_id length"):
            adapter._validate_agent_id("a" * 257)

    def test_max_length_id_ok(self, adapter: TRTSafetensorsCacheAdapter) -> None:
        adapter._validate_agent_id("a" * 256)  # Should not raise

    def test_invalid_characters_raises(self, adapter: TRTSafetensorsCacheAdapter) -> None:
        with pytest.raises(CachePersistenceError, match="Invalid agent_id characters"):
            adapter._validate_agent_id("agent/../../etc")

    def test_spaces_raise(self, adapter: TRTSafetensorsCacheAdapter) -> None:
        with pytest.raises(CachePersistenceError, match="Invalid agent_id characters"):
            adapter._validate_agent_id("agent 123")

    def test_dots_raise(self, adapter: TRTSafetensorsCacheAdapter) -> None:
        with pytest.raises(CachePersistenceError, match="Invalid agent_id characters"):
            adapter._validate_agent_id("agent.id")


class TestExistsDeleteList:
    def test_exists_false_when_no_file(self, adapter: TRTSafetensorsCacheAdapter) -> None:
        assert adapter.exists("nonexistent") is False

    def test_exists_true_after_save(self, adapter: TRTSafetensorsCacheAdapter) -> None:
        agent_blocks = _make_fp16_blocks("agent-exists")
        adapter.save("agent-exists", agent_blocks)
        assert adapter.exists("agent-exists") is True

    def test_delete_removes_file(self, adapter: TRTSafetensorsCacheAdapter) -> None:
        agent_blocks = _make_fp16_blocks("agent-del")
        adapter.save("agent-del", agent_blocks)
        assert adapter.exists("agent-del") is True

        adapter.delete("agent-del")
        assert adapter.exists("agent-del") is False

    def test_delete_nonexistent_is_noop(self, adapter: TRTSafetensorsCacheAdapter) -> None:
        adapter.delete("no-such-agent")  # Should not raise

    def test_list_cached_agents_empty(self, adapter: TRTSafetensorsCacheAdapter) -> None:
        assert adapter.list_cached_agents() == []

    def test_list_cached_agents_after_saves(self, adapter: TRTSafetensorsCacheAdapter) -> None:
        for name in ["alpha", "bravo", "charlie"]:
            agent_blocks = _make_fp16_blocks(name)
            adapter.save(name, agent_blocks)

        agents = sorted(adapter.list_cached_agents())
        assert agents == ["alpha", "bravo", "charlie"]

    def test_list_after_delete(self, adapter: TRTSafetensorsCacheAdapter) -> None:
        for name in ["x", "y"]:
            agent_blocks = _make_fp16_blocks(name)
            adapter.save(name, agent_blocks)

        adapter.delete("x")
        assert adapter.list_cached_agents() == ["y"]


class TestLoadNonexistent:
    def test_load_missing_agent_raises(self, adapter: TRTSafetensorsCacheAdapter) -> None:
        with pytest.raises(AgentNotFoundError, match="No cache for agent"):
            adapter.load("missing-agent")


class TestCacheDirCreation:
    def test_creates_directory_on_init(self, tmp_path: Path) -> None:
        new_dir = tmp_path / "nested" / "cache" / "dir"
        assert not new_dir.exists()
        TRTSafetensorsCacheAdapter(cache_dir=new_dir)
        assert new_dir.exists()
