# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Yakov Shkolnikov and contributors
"""Memory leak detection tests using tracemalloc.

Verifies that BlockPool and AgentCacheStore operations do not leak
memory over many allocate/free cycles. Uses tracemalloc to measure
net memory growth and asserts growth stays within bounds.
"""

import tracemalloc
from pathlib import Path
from unittest.mock import MagicMock

from agent_memory.domain.entities import AgentBlocks, KVBlock
from agent_memory.domain.services import BlockPool
from agent_memory.domain.value_objects import ModelCacheSpec


def _make_spec(n_layers: int = 4, n_kv_heads: int = 4, head_dim: int = 64) -> ModelCacheSpec:
    return ModelCacheSpec(
        n_layers=n_layers,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        block_tokens=256,
        layer_types=["global"] * n_layers,
    )


class TestBlockPoolMemoryLeaks:
    """Verify BlockPool doesn't leak memory on allocate/free cycles."""

    def test_allocate_free_cycle_no_growth(self) -> None:
        """1000 allocate/free cycles should not grow memory significantly."""
        spec = _make_spec()
        pool = BlockPool(spec=spec, total_blocks=100)

        tracemalloc.start()
        snapshot_before = tracemalloc.take_snapshot()

        for i in range(1000):
            agent_id = f"agent_{i % 10}"
            blocks = pool.allocate(n_blocks=5, layer_id=0, agent_id=agent_id)
            pool.free(blocks, agent_id=agent_id)

        snapshot_after = tracemalloc.take_snapshot()
        tracemalloc.stop()

        stats = snapshot_after.compare_to(snapshot_before, "lineno")
        total_growth = sum(s.size_diff for s in stats if s.size_diff > 0)

        assert total_growth < 1_000_000, (
            f"Memory grew {total_growth / 1024:.1f} KB over 1000 alloc/free cycles"
        )

    def test_pool_invariant_after_many_cycles(self) -> None:
        """Pool accounting must be exact after many operations."""
        spec = _make_spec()
        total = 200
        pool = BlockPool(spec=spec, total_blocks=total)

        for i in range(500):
            agent_id = f"agent_{i % 5}"
            blocks = pool.allocate(n_blocks=10, layer_id=i % spec.n_layers, agent_id=agent_id)
            pool.free(blocks, agent_id=agent_id)

        assert pool.allocated_block_count() == 0
        assert pool.available_blocks() == total
        assert pool.used_memory() + pool.available_memory() == pool.total_memory()
        assert len(pool.agent_allocations) == 0

    def test_layer_data_cleared_on_free(self) -> None:
        """Freed blocks must have layer_data = None to release GPU memory."""
        spec = _make_spec()
        pool = BlockPool(spec=spec, total_blocks=50)

        blocks = pool.allocate(n_blocks=10, layer_id=0, agent_id="test")
        for b in blocks:
            b.layer_data = bytearray(1024)  # Simulate tensor data

        freed_blocks = list(blocks)  # Keep references
        pool.free(blocks, agent_id="test")

        for b in freed_blocks:
            assert b.layer_data is None, f"Block {b.block_id} layer_data not cleared"

    def test_agent_allocation_cleanup(self) -> None:
        """Empty agent entries should be removed from agent_allocations."""
        spec = _make_spec()
        pool = BlockPool(spec=spec, total_blocks=100)

        for i in range(50):
            agent_id = f"agent_{i}"
            blocks = pool.allocate(n_blocks=2, layer_id=0, agent_id=agent_id)
            pool.free(blocks, agent_id=agent_id)

        assert len(pool.agent_allocations) == 0, (
            f"Expected 0 agent entries, got {len(pool.agent_allocations)}"
        )

    def test_free_agent_blocks_cleanup(self) -> None:
        """free_agent_blocks should remove agent entry entirely."""
        spec = _make_spec()
        pool = BlockPool(spec=spec, total_blocks=100)

        for layer in range(spec.n_layers):
            pool.allocate(n_blocks=5, layer_id=layer, agent_id="agent_x")

        freed = pool.free_agent_blocks("agent_x")
        assert freed == 5 * spec.n_layers
        assert "agent_x" not in pool.agent_allocations
        assert pool.allocated_block_count() == 0
        assert pool.available_blocks() == 100

    def test_multi_agent_interleaved_no_leak(self) -> None:
        """Interleaved multi-agent alloc/free must maintain pool invariant."""
        spec = _make_spec()
        pool = BlockPool(spec=spec, total_blocks=500)

        agent_blocks: dict[str, list[KVBlock]] = {}

        for cycle in range(100):
            agent_id = f"agent_{cycle % 10}"

            # Free previous blocks if agent already has some
            if agent_id in agent_blocks:
                pool.free(agent_blocks[agent_id], agent_id=agent_id)
                del agent_blocks[agent_id]

            # Allocate new blocks
            blocks = pool.allocate(n_blocks=3, layer_id=cycle % spec.n_layers, agent_id=agent_id)
            agent_blocks[agent_id] = blocks

        # Free all remaining
        for agent_id, blocks in agent_blocks.items():
            pool.free(blocks, agent_id=agent_id)

        assert pool.allocated_block_count() == 0
        assert pool.available_blocks() == 500
        assert len(pool.agent_allocations) == 0


class TestAgentCacheStoreMemoryLeaks:
    """Verify AgentCacheStore doesn't leak hot cache entries."""

    def _make_agent_blocks(self, agent_id: str, n_layers: int = 2) -> AgentBlocks:
        blocks: dict[int, list[KVBlock]] = {}
        for layer in range(n_layers):
            blocks[layer] = [
                KVBlock(
                    block_id=layer * 10,
                    layer_id=layer,
                    token_count=10,
                    layer_data=b"fake",
                    metadata={"agent_id": agent_id},
                )
            ]
        return AgentBlocks(
            agent_id=agent_id,
            blocks=blocks,
            total_tokens=10,
            token_sequence=[1, 2, 3],
            prompt_text="test",
        )

    def test_hot_cache_bounded_by_max(self) -> None:
        """Hot cache must not exceed max_hot_agents."""
        from agent_memory.application.agent_cache_store import AgentCacheStore, ModelTag

        tag = ModelTag(model_id="test", n_layers=2, n_kv_heads=4, head_dim=64, block_tokens=256)
        store = AgentCacheStore(
            cache_dir=Path("/tmp/claude/test_mem_leak"),
            max_hot_agents=5,
            model_tag=tag,
            cache_adapter=MagicMock(),
        )

        for i in range(20):
            store.save(f"agent_{i}", self._make_agent_blocks(f"agent_{i}"))

        assert len(store._hot_cache) <= 5, (
            f"Hot cache has {len(store._hot_cache)} entries, max is 5"
        )

    def test_delete_removes_from_hot_cache(self) -> None:
        """Deleting an agent should remove it from hot cache."""
        from agent_memory.application.agent_cache_store import AgentCacheStore, ModelTag

        tag = ModelTag(model_id="test", n_layers=2, n_kv_heads=4, head_dim=64, block_tokens=256)
        store = AgentCacheStore(
            cache_dir=Path("/tmp/claude/test_mem_delete"),
            max_hot_agents=10,
            model_tag=tag,
            cache_adapter=MagicMock(),
        )

        for i in range(5):
            store.save(f"agent_{i}", self._make_agent_blocks(f"agent_{i}"))

        for i in range(5):
            store.delete(f"agent_{i}")

        assert len(store._hot_cache) == 0, (
            f"Hot cache should be empty after deleting all, has {len(store._hot_cache)}"
        )

    def test_evict_all_drains_hot_cache(self) -> None:
        """evict_all_to_disk should leave hot cache empty."""
        from agent_memory.application.agent_cache_store import AgentCacheStore, ModelTag

        tag = ModelTag(model_id="test", n_layers=2, n_kv_heads=4, head_dim=64, block_tokens=256)
        mock_adapter = MagicMock()
        store = AgentCacheStore(
            cache_dir=Path("/tmp/claude/test_mem_evict"),
            max_hot_agents=10,
            model_tag=tag,
            cache_adapter=mock_adapter,
        )

        for i in range(8):
            store.save(f"agent_{i}", self._make_agent_blocks(f"agent_{i}"))

        evicted = store.evict_all_to_disk()
        assert evicted == 8
        assert len(store._hot_cache) == 0


class TestBlockPoolOrphanDetection:
    """Verify no orphaned blocks exist after operations."""

    def test_no_orphans_after_mixed_operations(self) -> None:
        """All allocated blocks must be tracked by an agent."""
        spec = _make_spec()
        pool = BlockPool(spec=spec, total_blocks=200)

        # Allocate for several agents
        for i in range(10):
            pool.allocate(n_blocks=5, layer_id=0, agent_id=f"agent_{i}")

        # Free some agents
        for i in range(0, 10, 2):
            pool.free_agent_blocks(f"agent_{i}")

        # Verify: all allocated blocks are tracked
        allocated_ids = set(pool.allocated_blocks.keys())
        tracked_ids: set[int] = set()
        for block_ids in pool.agent_allocations.values():
            tracked_ids.update(block_ids)

        orphans = allocated_ids - tracked_ids
        assert len(orphans) == 0, f"Found {len(orphans)} orphaned blocks: {orphans}"

        # Verify counts
        assert pool.allocated_block_count() == 25  # 5 agents x 5 blocks
        assert pool.available_blocks() == 175

    def test_no_orphans_after_partial_free(self) -> None:
        """Freeing some blocks from an agent leaves no orphans."""
        spec = _make_spec()
        pool = BlockPool(spec=spec, total_blocks=100)

        blocks = pool.allocate(n_blocks=10, layer_id=0, agent_id="agent_a")
        pool.free(blocks[:5], agent_id="agent_a")

        allocated_ids = set(pool.allocated_blocks.keys())
        tracked_ids = set(pool.agent_allocations.get("agent_a", set()))

        assert allocated_ids == tracked_ids
        assert len(tracked_ids) == 5
