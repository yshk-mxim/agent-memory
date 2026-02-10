# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Yakov Shkolnikov and contributors
"""Unit tests for cache growth correctness.

Verifies block allocation counts match formula predictions:
- Prompt of L tokens → ceil(L / block_tokens) blocks per layer
- After generating G tokens → ceil((L + G) / block_tokens) blocks
- Pool invariant: allocated + free == total after every operation
"""

import math

import pytest

from agent_memory.domain.services import BlockPool
from agent_memory.domain.value_objects import ModelCacheSpec

pytestmark = pytest.mark.unit

BLOCK_TOKENS = 256


@pytest.fixture
def spec() -> ModelCacheSpec:
    return ModelCacheSpec(
        n_layers=4,
        n_kv_heads=4,
        head_dim=64,
        block_tokens=BLOCK_TOKENS,
        layer_types=["global"] * 4,
    )


class TestBlockCountFromPromptLength:
    """Verify prompt of L tokens requires ceil(L / block_tokens) blocks per layer."""

    @pytest.mark.parametrize(
        "n_tokens, expected_blocks",
        [
            (1, 1),          # 1 token needs 1 block
            (256, 1),        # Exact block boundary
            (257, 2),        # 1 past boundary needs 2 blocks
            (512, 2),        # Exact 2 blocks
            (1000, 4),       # ceil(1000/256) = 4
            (2048, 8),       # Exact 8 blocks
            (2049, 9),       # 1 past → 9 blocks
        ],
    )
    def test_blocks_per_layer_matches_formula(
        self, spec: ModelCacheSpec, n_tokens: int, expected_blocks: int
    ) -> None:
        """Number of blocks allocated per layer matches ceil(L / block_tokens)."""
        computed = math.ceil(n_tokens / BLOCK_TOKENS)
        assert computed == expected_blocks

        pool = BlockPool(spec=spec, total_blocks=1000)
        blocks = pool.allocate(n_blocks=computed, layer_id=0, agent_id="test")

        assert len(blocks) == expected_blocks

    def test_total_blocks_across_layers(self, spec: ModelCacheSpec) -> None:
        """Total blocks = ceil(L / block_tokens) * n_layers."""
        n_tokens = 1000
        blocks_per_layer = math.ceil(n_tokens / BLOCK_TOKENS)  # 4
        total_expected = blocks_per_layer * spec.n_layers  # 4 * 4 = 16

        pool = BlockPool(spec=spec, total_blocks=1000)
        total_allocated = 0
        for layer_id in range(spec.n_layers):
            blocks = pool.allocate(n_blocks=blocks_per_layer, layer_id=layer_id, agent_id="test")
            total_allocated += len(blocks)

        assert total_allocated == total_expected


class TestCacheGrowthDuringGeneration:
    """After generating G tokens, verify ceil((L + G) / block_tokens) blocks needed."""

    def test_growth_within_same_block(self, spec: ModelCacheSpec) -> None:
        """Generating tokens within same block doesn't require new allocation."""
        # Start with 100 tokens = 1 block per layer
        # Generate 50 more = 150 total, still 1 block
        prompt_blocks = math.ceil(100 / BLOCK_TOKENS)  # 1
        total_blocks = math.ceil(150 / BLOCK_TOKENS)  # 1
        assert prompt_blocks == total_blocks  # No growth needed

    def test_growth_crossing_block_boundary(self, spec: ModelCacheSpec) -> None:
        """Generating tokens past block boundary needs new block."""
        # Start with 250 tokens = 1 block per layer
        # Generate 10 more = 260 total, needs 2 blocks
        prompt_blocks = math.ceil(250 / BLOCK_TOKENS)  # 1
        total_blocks = math.ceil(260 / BLOCK_TOKENS)  # 2
        new_blocks_needed = total_blocks - prompt_blocks  # 1

        pool = BlockPool(spec=spec, total_blocks=1000)
        initial = pool.allocate(n_blocks=prompt_blocks, layer_id=0, agent_id="test")
        growth = pool.allocate(n_blocks=new_blocks_needed, layer_id=0, agent_id="test")

        assert len(initial) == 1
        assert len(growth) == 1

    @pytest.mark.parametrize(
        "prompt_tokens, gen_tokens",
        [
            (100, 200),   # 100→300: 1→2 blocks
            (256, 1),     # 256→257: 1→2 blocks
            (500, 500),   # 500→1000: 2→4 blocks
            (1000, 1048), # 1000→2048: 4→8 blocks
        ],
    )
    def test_parametrized_growth(
        self, spec: ModelCacheSpec, prompt_tokens: int, gen_tokens: int
    ) -> None:
        """Verify growth formula for various prompt+generation combinations."""
        initial_blocks = math.ceil(prompt_tokens / BLOCK_TOKENS)
        final_blocks = math.ceil((prompt_tokens + gen_tokens) / BLOCK_TOKENS)
        growth = final_blocks - initial_blocks

        assert growth >= 0
        assert final_blocks == math.ceil((prompt_tokens + gen_tokens) / BLOCK_TOKENS)


class TestPoolInvariantAfterOperations:
    """Pool invariant: allocated + free == total after every operation."""

    def test_invariant_after_allocate(self, spec: ModelCacheSpec) -> None:
        """Invariant holds after each allocation."""
        total = 100
        pool = BlockPool(spec=spec, total_blocks=total)

        for i in range(10):
            pool.allocate(n_blocks=1, layer_id=0, agent_id=f"agent_{i}")
            assert pool.allocated_block_count() + pool.available_blocks() == total

    def test_invariant_after_free(self, spec: ModelCacheSpec) -> None:
        """Invariant holds after each deallocation."""
        total = 100
        pool = BlockPool(spec=spec, total_blocks=total)

        allocated = []
        for i in range(10):
            blocks = pool.allocate(n_blocks=1, layer_id=0, agent_id=f"agent_{i}")
            allocated.append((blocks, f"agent_{i}"))

        for blocks, agent_id in allocated:
            pool.free(blocks, agent_id=agent_id)
            assert pool.allocated_block_count() + pool.available_blocks() == total

    def test_invariant_after_mixed_operations(self, spec: ModelCacheSpec) -> None:
        """Invariant holds after interleaved allocate/free operations."""
        total = 200
        pool = BlockPool(spec=spec, total_blocks=total)

        # Allocate 5, free 2, allocate 3, free all
        batch1 = []
        for i in range(5):
            blocks = pool.allocate(n_blocks=1, layer_id=0, agent_id=f"a_{i}")
            batch1.append(blocks)
        assert pool.allocated_block_count() + pool.available_blocks() == total

        pool.free(batch1[0], agent_id="a_0")
        pool.free(batch1[1], agent_id="a_1")
        assert pool.allocated_block_count() + pool.available_blocks() == total

        batch2 = []
        for i in range(3):
            blocks = pool.allocate(n_blocks=1, layer_id=0, agent_id=f"b_{i}")
            batch2.append(blocks)
        assert pool.allocated_block_count() + pool.available_blocks() == total

        # Free everything — must use correct agent_id (BlockPool validates ownership)
        for i, b in enumerate(batch1[2:], start=2):
            pool.free(b, agent_id=f"a_{i}")
        for i, b in enumerate(batch2):
            pool.free(b, agent_id=f"b_{i}")
        assert pool.allocated_block_count() + pool.available_blocks() == total
        assert pool.allocated_block_count() == 0

    def test_invariant_after_10k_operations(self, spec: ModelCacheSpec) -> None:
        """Invariant holds after 10K allocate+free cycles."""
        total = 20000
        pool = BlockPool(spec=spec, total_blocks=total)

        for i in range(10000):
            blocks = pool.allocate(n_blocks=1, layer_id=0, agent_id=f"agent_{i}")
            pool.free(blocks, agent_id=f"agent_{i}")

        assert pool.allocated_block_count() + pool.available_blocks() == total
        assert pool.allocated_block_count() == 0
