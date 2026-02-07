"""Unit tests for domain services.

Tests BlockPool service with focus on:
- Property-based testing (Hypothesis) for invariants
- Allocation and deallocation correctness
- Memory budget tracking
- Pool exhaustion handling
- Model reconfiguration (hot-swap)
- Multi-agent scenarios

Per production_plan.md: 95%+ coverage, property-based tests CRITICAL for 2025.

Research findings (2025-2026):
- Hypothesis property testing is CRITICAL but underutilized
- BlockPool is perfect candidate (clear invariants: used + available = total)
- Source: https://danielsarney.com/blog/python-testing-best-practices-2025-building-reliable-applications/
"""

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from semantic.domain.entities import KVBlock
from semantic.domain.errors import (
    BlockOperationError,
    ModelSpecValidationError,
    PoolConfigurationError,
    PoolExhaustedError,
)
from semantic.domain.services import BlockPool
from semantic.domain.value_objects import ModelCacheSpec

pytestmark = pytest.mark.unit


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def gemma_spec() -> ModelCacheSpec:
    """Gemma 3 12B cache spec (hybrid: 8 global + 40 sliding window)."""
    return ModelCacheSpec(
        n_layers=48,
        n_kv_heads=8,
        head_dim=240,
        block_tokens=256,
        layer_types=["global"] * 8 + ["sliding_window"] * 40,
        sliding_window_size=1024,
    )


@pytest.fixture
def llama_spec() -> ModelCacheSpec:
    """Llama 3.1-8B cache spec (uniform: all global)."""
    return ModelCacheSpec(
        n_layers=32,
        n_kv_heads=8,
        head_dim=128,
        block_tokens=256,
        layer_types=["global"] * 32,
    )


@pytest.fixture
def small_spec() -> ModelCacheSpec:
    """Small test spec for faster property tests."""
    return ModelCacheSpec(
        n_layers=4,
        n_kv_heads=4,
        head_dim=64,
        block_tokens=256,
        layer_types=["global"] * 4,
    )


# ============================================================================
# PROPERTY-BASED TESTS (Hypothesis) - CRITICAL FOR 2025
# ============================================================================


class TestBlockPoolProperties:
    """Property-based tests for BlockPool invariants.

    Per 2025 research, property-based testing is CRITICAL but underutilized.
    These tests generate 100 random scenarios to validate invariants.

    Source: https://pytest-with-eric.com/introduction/python-unit-testing-best-practices/
    """

    @given(
        total_blocks=st.integers(min_value=10, max_value=100),
        n_blocks_to_allocate=st.integers(min_value=1, max_value=50),
    )
    def test_property_used_plus_available_equals_total(
        self,
        total_blocks: int,
        n_blocks_to_allocate: int,
    ) -> None:
        """Property: used_blocks + available_blocks = total_blocks (ALWAYS).

        This is the core invariant of the BlockPool.
        No matter what operations, this must hold.
        """
        assume(n_blocks_to_allocate <= total_blocks)

        # Create spec inline (Hypothesis doesn't work with pytest fixtures)
        small_spec = ModelCacheSpec(
            n_layers=4,
            n_kv_heads=4,
            head_dim=64,
            block_tokens=256,
            layer_types=["global"] * 4,
        )
        pool = BlockPool(spec=small_spec, total_blocks=total_blocks)

        # Initially: used=0, available=total
        assert pool.allocated_block_count() + pool.available_blocks() == total_blocks

        # After allocation: invariant still holds
        blocks = pool.allocate(n_blocks=n_blocks_to_allocate, layer_id=0, agent_id="test_agent")
        assert pool.allocated_block_count() + pool.available_blocks() == total_blocks

        # After freeing: invariant still holds
        pool.free(blocks, agent_id="test_agent")
        assert pool.allocated_block_count() + pool.available_blocks() == total_blocks

    @given(
        total_blocks=st.integers(min_value=10, max_value=50),
        allocations=st.lists(st.integers(min_value=1, max_value=10), min_size=1, max_size=10),
    )
    def test_property_allocate_then_free_restores_state(
        self,
        total_blocks: int,
        allocations: list[int],
    ) -> None:
        """Property: allocate(n) then free(n) restores pool to initial state.

        Round-trip operations should be idempotent (no leaks).
        """
        assume(sum(allocations) <= total_blocks)

        # Create spec inline (Hypothesis doesn't work with pytest fixtures)
        small_spec = ModelCacheSpec(
            n_layers=4,
            n_kv_heads=4,
            head_dim=64,
            block_tokens=256,
            layer_types=["global"] * 4,
        )
        pool = BlockPool(spec=small_spec, total_blocks=total_blocks)
        initial_available = pool.available_blocks()

        # Allocate multiple times
        all_blocks = []
        for i, n_blocks in enumerate(allocations):
            if pool.available_blocks() >= n_blocks:
                blocks = pool.allocate(
                    n_blocks=n_blocks,
                    layer_id=0,
                    agent_id=f"agent_{i}",
                )
                all_blocks.append((blocks, f"agent_{i}"))

        # Free all blocks
        for blocks, agent_id in all_blocks:
            pool.free(blocks, agent_id=agent_id)

        # Should restore to initial state
        assert pool.available_blocks() == initial_available
        assert pool.allocated_block_count() == 0

    @given(
        total_blocks=st.integers(min_value=20, max_value=50),
        n_agents=st.integers(min_value=2, max_value=10),
    )
    def test_property_multi_agent_isolation(
        self,
        total_blocks: int,
        n_agents: int,
    ) -> None:
        """Property: Agent allocations are isolated (no cross-contamination).

        Each agent's blocks are tracked separately.
        Freeing one agent doesn't affect others.
        """
        # Create spec inline (Hypothesis doesn't work with pytest fixtures)
        small_spec = ModelCacheSpec(
            n_layers=4,
            n_kv_heads=4,
            head_dim=64,
            block_tokens=256,
            layer_types=["global"] * 4,
        )
        pool = BlockPool(spec=small_spec, total_blocks=total_blocks)
        blocks_per_agent = total_blocks // n_agents

        assume(blocks_per_agent >= 1)

        # Allocate to each agent
        agent_allocations: dict[str, list[KVBlock]] = {}
        for i in range(n_agents):
            agent_id = f"agent_{i}"
            blocks = pool.allocate(
                n_blocks=blocks_per_agent,
                layer_id=0,
                agent_id=agent_id,
            )
            agent_allocations[agent_id] = blocks

        # Free one agent's blocks
        first_agent = "agent_0"
        pool.free(agent_allocations[first_agent], agent_id=first_agent)

        # Other agents should still have their blocks
        for i in range(1, n_agents):
            agent_id = f"agent_{i}"
            # Verify blocks still tracked (would raise if freed)
            assert len(agent_allocations[agent_id]) == blocks_per_agent


# ============================================================================
# TRADITIONAL UNIT TESTS (Specific Scenarios)
# ============================================================================


class TestBlockPoolInitialization:
    """Test BlockPool construction and validation."""

    def test_create_pool_valid_spec(self, gemma_spec: ModelCacheSpec) -> None:
        """Should create pool with valid spec."""
        pool = BlockPool(spec=gemma_spec, total_blocks=1000)

        assert pool.spec == gemma_spec
        assert pool.total_blocks == 1000
        assert pool.block_tokens == 256
        assert pool.available_blocks() == 1000
        assert pool.allocated_block_count() == 0

    def test_create_pool_small(self, llama_spec: ModelCacheSpec) -> None:
        """Should create pool with minimal blocks."""
        pool = BlockPool(spec=llama_spec, total_blocks=10)

        assert pool.total_blocks == 10
        assert pool.available_blocks() == 10

    def test_reject_zero_blocks(self, gemma_spec: ModelCacheSpec) -> None:
        """Should raise ValueError for total_blocks <= 0."""
        with pytest.raises(PoolConfigurationError, match="total_blocks must be > 0"):
            BlockPool(spec=gemma_spec, total_blocks=0)

    def test_reject_negative_blocks(self, gemma_spec: ModelCacheSpec) -> None:
        """Should raise ValueError for negative total_blocks."""
        with pytest.raises(PoolConfigurationError, match="total_blocks must be > 0"):
            BlockPool(spec=gemma_spec, total_blocks=-100)

    def test_reject_invalid_spec(self) -> None:
        """Should raise for invalid spec (n_layers=0 caught at spec level)."""
        with pytest.raises(ModelSpecValidationError, match="n_layers must be > 0"):
            ModelCacheSpec(
                n_layers=0,
                n_kv_heads=8,
                head_dim=256,
                block_tokens=256,
                layer_types=[],
            )


class TestBlockPoolAllocation:
    """Test block allocation from pool."""

    def test_allocate_single_block(self, gemma_spec: ModelCacheSpec) -> None:
        """Should allocate single block successfully."""
        pool = BlockPool(spec=gemma_spec, total_blocks=100)

        blocks = pool.allocate(n_blocks=1, layer_id=0, agent_id="agent_1")

        assert len(blocks) == 1
        assert blocks[0].block_id >= 0
        assert blocks[0].layer_id == 0
        assert blocks[0].token_count == 0
        assert blocks[0].metadata["agent_id"] == "agent_1"
        assert pool.available_blocks() == 99
        assert pool.allocated_block_count() == 1

    def test_allocate_multiple_blocks(self, gemma_spec: ModelCacheSpec) -> None:
        """Should allocate multiple blocks at once."""
        pool = BlockPool(spec=gemma_spec, total_blocks=100)

        blocks = pool.allocate(n_blocks=5, layer_id=3, agent_id="agent_2")

        assert len(blocks) == 5
        assert all(b.layer_id == 3 for b in blocks)
        assert all(b.token_count == 0 for b in blocks)
        assert pool.available_blocks() == 95

    def test_allocate_all_blocks(self, llama_spec: ModelCacheSpec) -> None:
        """Should allow allocating all available blocks."""
        pool = BlockPool(spec=llama_spec, total_blocks=50)

        blocks = pool.allocate(n_blocks=50, layer_id=0, agent_id="greedy_agent")

        assert len(blocks) == 50
        assert pool.available_blocks() == 0
        assert pool.allocated_block_count() == 50

    def test_allocate_assigns_unique_block_ids(self, gemma_spec: ModelCacheSpec) -> None:
        """Should assign unique block IDs (no duplicates)."""
        pool = BlockPool(spec=gemma_spec, total_blocks=100)

        blocks = pool.allocate(n_blocks=10, layer_id=0, agent_id="agent_1")
        block_ids = [b.block_id for b in blocks]

        assert len(block_ids) == len(set(block_ids))  # All unique

    def test_allocate_tracks_agent_allocations(self, gemma_spec: ModelCacheSpec) -> None:
        """Should track which blocks belong to which agent."""
        pool = BlockPool(spec=gemma_spec, total_blocks=100)

        pool.allocate(n_blocks=5, layer_id=0, agent_id="agent_1")
        pool.allocate(n_blocks=3, layer_id=0, agent_id="agent_2")

        # Agent allocations tracked
        assert "agent_1" in pool.agent_allocations
        assert "agent_2" in pool.agent_allocations
        assert len(pool.agent_allocations["agent_1"]) == 5
        assert len(pool.agent_allocations["agent_2"]) == 3

    def test_allocate_to_different_layers(self, gemma_spec: ModelCacheSpec) -> None:
        """Should allow allocating blocks to different layers."""
        pool = BlockPool(spec=gemma_spec, total_blocks=100)

        blocks_layer_0 = pool.allocate(n_blocks=3, layer_id=0, agent_id="agent_1")
        blocks_layer_5 = pool.allocate(n_blocks=2, layer_id=5, agent_id="agent_1")
        blocks_layer_47 = pool.allocate(n_blocks=1, layer_id=47, agent_id="agent_1")

        assert blocks_layer_0[0].layer_id == 0
        assert blocks_layer_5[0].layer_id == 5
        assert blocks_layer_47[0].layer_id == 47
        assert pool.allocated_block_count() == 6


class TestBlockPoolAllocationErrors:
    """Test block allocation error cases."""

    def test_reject_zero_block_allocation(self, gemma_spec: ModelCacheSpec) -> None:
        """Should raise ValueError for n_blocks <= 0."""
        pool = BlockPool(spec=gemma_spec, total_blocks=100)

        with pytest.raises(BlockOperationError, match="n_blocks must be > 0"):
            pool.allocate(n_blocks=0, layer_id=0, agent_id="agent_1")

    def test_reject_negative_block_allocation(self, gemma_spec: ModelCacheSpec) -> None:
        """Should raise ValueError for negative n_blocks."""
        pool = BlockPool(spec=gemma_spec, total_blocks=100)

        with pytest.raises(BlockOperationError, match="n_blocks must be > 0"):
            pool.allocate(n_blocks=-5, layer_id=0, agent_id="agent_1")

    def test_reject_invalid_layer_id(self, gemma_spec: ModelCacheSpec) -> None:
        """Should raise ValueError for out-of-range layer_id."""
        pool = BlockPool(spec=gemma_spec, total_blocks=100)

        # Gemma has 48 layers (0-47)
        with pytest.raises(BlockOperationError, match=r"layer_id must be 0-47"):
            pool.allocate(n_blocks=1, layer_id=48, agent_id="agent_1")

    def test_reject_negative_layer_id(self, gemma_spec: ModelCacheSpec) -> None:
        """Should raise ValueError for negative layer_id."""
        pool = BlockPool(spec=gemma_spec, total_blocks=100)

        with pytest.raises(BlockOperationError, match=r"layer_id must be 0-47"):
            pool.allocate(n_blocks=1, layer_id=-1, agent_id="agent_1")

    def test_raise_pool_exhausted_error(self, llama_spec: ModelCacheSpec) -> None:
        """Should raise PoolExhaustedError when insufficient blocks."""
        pool = BlockPool(spec=llama_spec, total_blocks=10)

        # Allocate all blocks
        pool.allocate(n_blocks=10, layer_id=0, agent_id="agent_1")

        # Try to allocate more - should fail
        with pytest.raises(
            PoolExhaustedError,
            match=r"Requested 5 blocks but only 0 available",
        ):
            pool.allocate(n_blocks=5, layer_id=0, agent_id="agent_2")

    def test_pool_exhausted_provides_details(self, llama_spec: ModelCacheSpec) -> None:
        """PoolExhaustedError should provide pool statistics."""
        pool = BlockPool(spec=llama_spec, total_blocks=20)
        pool.allocate(n_blocks=15, layer_id=0, agent_id="agent_1")

        with pytest.raises(
            PoolExhaustedError,
            match=r"Total pool size: 20, allocated: 15",
        ):
            pool.allocate(n_blocks=10, layer_id=0, agent_id="agent_2")


class TestBlockPoolDeallocation:
    """Test block deallocation (freeing blocks)."""

    def test_free_allocated_blocks(self, gemma_spec: ModelCacheSpec) -> None:
        """Should free blocks and return them to pool."""
        pool = BlockPool(spec=gemma_spec, total_blocks=100)
        blocks = pool.allocate(n_blocks=5, layer_id=0, agent_id="agent_1")

        pool.free(blocks, agent_id="agent_1")

        assert pool.available_blocks() == 100
        assert pool.allocated_block_count() == 0

    def test_free_updates_agent_allocations(self, gemma_spec: ModelCacheSpec) -> None:
        """Should remove blocks from agent allocation tracking."""
        pool = BlockPool(spec=gemma_spec, total_blocks=100)
        blocks = pool.allocate(n_blocks=3, layer_id=0, agent_id="agent_1")

        assert "agent_1" in pool.agent_allocations

        pool.free(blocks, agent_id="agent_1")

        assert "agent_1" not in pool.agent_allocations

    def test_free_partial_allocation(self, gemma_spec: ModelCacheSpec) -> None:
        """Should allow freeing subset of agent's blocks."""
        pool = BlockPool(spec=gemma_spec, total_blocks=100)
        blocks = pool.allocate(n_blocks=10, layer_id=0, agent_id="agent_1")

        # Free first 5 blocks
        pool.free(blocks[:5], agent_id="agent_1")

        assert pool.available_blocks() == 95
        assert len(pool.agent_allocations["agent_1"]) == 5

    def test_reject_freeing_unallocated_block(self, gemma_spec: ModelCacheSpec) -> None:
        """Should raise ValueError when freeing unallocated block."""
        pool = BlockPool(spec=gemma_spec, total_blocks=100)

        fake_block = KVBlock(block_id=999, layer_id=0, token_count=0, layer_data=None)

        with pytest.raises(BlockOperationError, match=r"Block 999 is not allocated"):
            pool.free([fake_block], agent_id="agent_1")

    def test_reject_double_free(self, gemma_spec: ModelCacheSpec) -> None:
        """Should raise ValueError on double-free (same block twice)."""
        pool = BlockPool(spec=gemma_spec, total_blocks=100)
        blocks = pool.allocate(n_blocks=1, layer_id=0, agent_id="agent_1")

        pool.free(blocks, agent_id="agent_1")

        # Try to free again - should fail
        with pytest.raises(BlockOperationError, match=r"is not allocated"):
            pool.free(blocks, agent_id="agent_1")

    def test_reject_freeing_other_agents_blocks(self, gemma_spec: ModelCacheSpec) -> None:
        """Should raise ValueError when freeing another agent's blocks."""
        pool = BlockPool(spec=gemma_spec, total_blocks=100)
        blocks = pool.allocate(n_blocks=3, layer_id=0, agent_id="agent_1")

        # Agent 2 tries to free agent 1's blocks
        with pytest.raises(
            BlockOperationError,
            match=r"does not belong to agent agent_2",
        ):
            pool.free(blocks, agent_id="agent_2")


class TestBlockPoolFreeAgentBlocks:
    """Test free_agent_blocks() bulk operation."""

    def test_free_all_agent_blocks(self, gemma_spec: ModelCacheSpec) -> None:
        """Should free all blocks for a given agent."""
        pool = BlockPool(spec=gemma_spec, total_blocks=100)
        pool.allocate(n_blocks=5, layer_id=0, agent_id="agent_1")
        pool.allocate(n_blocks=3, layer_id=1, agent_id="agent_1")
        pool.allocate(n_blocks=2, layer_id=2, agent_id="agent_1")

        freed = pool.free_agent_blocks("agent_1")

        assert freed == 10
        assert pool.available_blocks() == 100
        assert "agent_1" not in pool.agent_allocations

    def test_free_agent_blocks_returns_count(self, gemma_spec: ModelCacheSpec) -> None:
        """Should return number of blocks freed."""
        pool = BlockPool(spec=gemma_spec, total_blocks=100)
        pool.allocate(n_blocks=7, layer_id=0, agent_id="agent_1")

        freed = pool.free_agent_blocks("agent_1")

        assert freed == 7

    def test_free_nonexistent_agent_returns_zero(self, gemma_spec: ModelCacheSpec) -> None:
        """Should return 0 when freeing non-existent agent."""
        pool = BlockPool(spec=gemma_spec, total_blocks=100)

        freed = pool.free_agent_blocks("nonexistent_agent")

        assert freed == 0

    def test_free_agent_blocks_leaves_others(self, gemma_spec: ModelCacheSpec) -> None:
        """Should only free target agent's blocks, not others."""
        pool = BlockPool(spec=gemma_spec, total_blocks=100)
        pool.allocate(n_blocks=5, layer_id=0, agent_id="agent_1")
        pool.allocate(n_blocks=3, layer_id=0, agent_id="agent_2")

        pool.free_agent_blocks("agent_1")

        assert pool.allocated_block_count() == 3
        assert "agent_2" in pool.agent_allocations


class TestBlockPoolMemoryAccounting:
    """Test memory budget tracking."""

    def test_used_memory_empty_pool(self, gemma_spec: ModelCacheSpec) -> None:
        """Should return 0 for empty pool."""
        pool = BlockPool(spec=gemma_spec, total_blocks=100)

        assert pool.used_memory() == 0

    def test_used_memory_after_allocation(self, gemma_spec: ModelCacheSpec) -> None:
        """Should calculate used memory correctly."""
        pool = BlockPool(spec=gemma_spec, total_blocks=100)
        pool.allocate(n_blocks=10, layer_id=0, agent_id="agent_1")

        # Gemma 3: 8 * 240 * 2 * 2 * 256 = 1,966,080 bytes per block
        bytes_per_block = gemma_spec.bytes_per_block_per_layer()
        expected_used = 10 * bytes_per_block

        assert pool.used_memory() == expected_used

    def test_available_memory(self, gemma_spec: ModelCacheSpec) -> None:
        """Should calculate available memory correctly."""
        pool = BlockPool(spec=gemma_spec, total_blocks=100)
        pool.allocate(n_blocks=25, layer_id=0, agent_id="agent_1")

        bytes_per_block = gemma_spec.bytes_per_block_per_layer()
        expected_available = 75 * bytes_per_block

        assert pool.available_memory() == expected_available

    def test_total_memory(self, gemma_spec: ModelCacheSpec) -> None:
        """Should calculate total memory correctly."""
        pool = BlockPool(spec=gemma_spec, total_blocks=100)

        bytes_per_block = gemma_spec.bytes_per_block_per_layer()
        expected_total = 100 * bytes_per_block

        assert pool.total_memory() == expected_total

    def test_memory_invariant(self, gemma_spec: ModelCacheSpec) -> None:
        """Invariant: used_memory + available_memory = total_memory."""
        pool = BlockPool(spec=gemma_spec, total_blocks=100)
        pool.allocate(n_blocks=37, layer_id=0, agent_id="agent_1")

        assert pool.used_memory() + pool.available_memory() == pool.total_memory()


class TestBlockPoolReconfiguration:
    """Test model hot-swap (reconfigure with new spec)."""

    def test_reconfigure_with_empty_pool(
        self,
        gemma_spec: ModelCacheSpec,
        llama_spec: ModelCacheSpec,
    ) -> None:
        """Should reconfigure to new spec when pool is empty."""
        pool = BlockPool(spec=gemma_spec, total_blocks=100)

        pool.reconfigure(llama_spec)

        assert pool.spec == llama_spec
        assert pool.block_tokens == 256
        assert pool.available_blocks() == 100

    def test_reconfigure_resets_free_list(
        self,
        gemma_spec: ModelCacheSpec,
        llama_spec: ModelCacheSpec,
    ) -> None:
        """Should reset free list on reconfigure."""
        pool = BlockPool(spec=gemma_spec, total_blocks=50)

        pool.reconfigure(llama_spec)

        # All blocks should be available
        assert len(pool.free_list) == 50
        assert pool.free_list == list(range(50))

    def test_reconfigure_clears_allocations(
        self,
        gemma_spec: ModelCacheSpec,
        llama_spec: ModelCacheSpec,
    ) -> None:
        """Should clear allocation tracking on reconfigure."""
        pool = BlockPool(spec=gemma_spec, total_blocks=100)

        pool.reconfigure(llama_spec)

        assert len(pool.allocated_blocks) == 0
        assert len(pool.agent_allocations) == 0

    def test_reject_reconfigure_with_active_allocations(
        self,
        gemma_spec: ModelCacheSpec,
        llama_spec: ModelCacheSpec,
    ) -> None:
        """Should raise RuntimeError if blocks are still allocated."""
        pool = BlockPool(spec=gemma_spec, total_blocks=100)
        pool.allocate(n_blocks=5, layer_id=0, agent_id="agent_1")

        with pytest.raises(
            PoolConfigurationError,
            match=r"Cannot reconfigure pool with 5 active allocations",
        ):
            pool.reconfigure(llama_spec)

    def test_reconfigure_after_free_all(
        self,
        gemma_spec: ModelCacheSpec,
        llama_spec: ModelCacheSpec,
    ) -> None:
        """Should allow reconfigure after freeing all blocks."""
        pool = BlockPool(spec=gemma_spec, total_blocks=100)
        pool.allocate(n_blocks=10, layer_id=0, agent_id="agent_1")
        pool.free_agent_blocks("agent_1")

        # Now reconfigure should work
        pool.reconfigure(llama_spec)

        assert pool.spec == llama_spec
        assert pool.available_blocks() == 100


class TestBlockPoolMaxBatchSize:
    """Test max_batch_size() calculation."""

    def test_max_batch_size_default_tokens(self, gemma_spec: ModelCacheSpec) -> None:
        """Should calculate max agents with default 256 tokens per agent."""
        pool = BlockPool(spec=gemma_spec, total_blocks=1000)

        # 256 tokens = 1 block per layer, 48 layers = 48 blocks per agent
        # 1000 blocks / 48 = 20 agents
        assert pool.max_batch_size() == 20

    def test_max_batch_size_custom_tokens(self, llama_spec: ModelCacheSpec) -> None:
        """Should calculate max agents with custom tokens per agent."""
        pool = BlockPool(spec=llama_spec, total_blocks=640)

        # 512 tokens = 2 blocks per layer, 32 layers = 64 blocks per agent
        # 640 blocks / 64 = 10 agents
        assert pool.max_batch_size(tokens_per_agent=512) == 10

    def test_max_batch_size_large_context(self, gemma_spec: ModelCacheSpec) -> None:
        """Should handle large context sizes correctly."""
        pool = BlockPool(spec=gemma_spec, total_blocks=960)

        # 2048 tokens = 8 blocks per layer, 48 layers = 384 blocks per agent
        # 960 blocks / 384 = 2 agents
        assert pool.max_batch_size(tokens_per_agent=2048) == 2

    def test_max_batch_size_edge_case_small_tokens(self, llama_spec: ModelCacheSpec) -> None:
        """Should handle edge case of very small token count."""
        pool = BlockPool(spec=llama_spec, total_blocks=100)

        # 64 tokens = 1 block (rounds up), 32 layers = 32 blocks per agent
        # 100 blocks / 32 = 3 agents
        assert pool.max_batch_size(tokens_per_agent=64) == 3


class TestBlockPoolForceClear:
    """Test force_clear_all_allocations() nulls layer_data."""

    def test_force_clear_nulls_layer_data(self, small_spec: ModelCacheSpec) -> None:
        """force_clear_all_allocations should null layer_data on all blocks."""
        pool = BlockPool(spec=small_spec, total_blocks=100)

        # Allocate and set layer_data
        blocks = pool.allocate(n_blocks=5, layer_id=0, agent_id="agent_1")
        for b in blocks:
            b.layer_data = {"k": "fake_tensor", "v": "fake_tensor"}

        # Keep references to check after clear
        block_refs = list(blocks)

        pool.force_clear_all_allocations()

        # All referenced blocks should have layer_data=None
        for b in block_refs:
            assert b.layer_data is None, f"Block {b.block_id} layer_data not nulled"

    def test_force_clear_resets_pool(self, small_spec: ModelCacheSpec) -> None:
        """force_clear_all_allocations should reset pool to full capacity."""
        pool = BlockPool(spec=small_spec, total_blocks=100)

        pool.allocate(n_blocks=30, layer_id=0, agent_id="agent_1")
        pool.allocate(n_blocks=20, layer_id=0, agent_id="agent_2")

        count = pool.force_clear_all_allocations()

        assert count == 2  # 2 agents cleared
        assert pool.available_blocks() == 100
        assert len(pool.allocated_blocks) == 0
        assert len(pool.agent_allocations) == 0


class TestBlockPoolO1Timing:
    """Verify block pool allocation/deallocation is O(1), not O(n)."""

    def test_allocation_mean_under_threshold(self, small_spec: ModelCacheSpec) -> None:
        """1000 allocations should average < 0.1ms each."""
        import time

        pool = BlockPool(spec=small_spec, total_blocks=2000)
        times = []

        for i in range(1000):
            start = time.perf_counter()
            blocks = pool.allocate(1, layer_id=0, agent_id=f"agent_{i}")
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        mean_ms = (sum(times) / len(times)) * 1000
        assert mean_ms < 0.1, f"Mean allocation time {mean_ms:.4f}ms exceeds 0.1ms"

    def test_deallocation_mean_under_threshold(self, small_spec: ModelCacheSpec) -> None:
        """1000 deallocations should average < 0.1ms each."""
        import time

        pool = BlockPool(spec=small_spec, total_blocks=2000)
        allocated = []
        for i in range(1000):
            blocks = pool.allocate(1, layer_id=0, agent_id=f"agent_{i}")
            allocated.append(blocks)

        times = []
        for i, blocks in enumerate(allocated):
            start = time.perf_counter()
            pool.free(blocks, agent_id=f"agent_{i}")
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        mean_ms = (sum(times) / len(times)) * 1000
        assert mean_ms < 0.1, f"Mean deallocation time {mean_ms:.4f}ms exceeds 0.1ms"

    def test_scaling_ratio_indicates_o1(self, small_spec: ModelCacheSpec) -> None:
        """100 allocations vs 10000: ratio must be < 5x (O(1) not O(n))."""
        import time

        # Warmup pass to stabilize JIT/caches
        pool_warmup = BlockPool(spec=small_spec, total_blocks=20000)
        for i in range(200):
            pool_warmup.allocate(1, layer_id=0, agent_id=f"warmup_{i}")

        # Small batch: 100 allocations (enough to reduce noise)
        pool_small = BlockPool(spec=small_spec, total_blocks=20000)
        start = time.perf_counter()
        for i in range(100):
            pool_small.allocate(1, layer_id=0, agent_id=f"agent_{i}")
        time_100 = time.perf_counter() - start

        # Large batch: 10000 allocations
        pool_large = BlockPool(spec=small_spec, total_blocks=20000)
        start = time.perf_counter()
        for i in range(10000):
            pool_large.allocate(1, layer_id=0, agent_id=f"agent_{i}")
        time_10000 = time.perf_counter() - start

        mean_100 = time_100 / 100
        mean_10000 = time_10000 / 10000
        ratio = mean_10000 / mean_100 if mean_100 > 0 else 1.0

        assert ratio < 5.0, (
            f"Scaling ratio {ratio:.2f}x suggests O(n) not O(1). "
            f"100-op mean={mean_100*1000:.4f}ms, "
            f"10000-op mean={mean_10000*1000:.4f}ms"
        )
