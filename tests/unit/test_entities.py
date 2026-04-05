# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Yakov Shkolnikov and contributors
"""Unit tests for domain entities.

Tests KVBlock and AgentBlocks entities with focus on:
- Validation logic in __post_init__
- State query methods (is_full, is_empty, num_blocks, etc.)
- Mutation methods (add_block, remove_block)
- Invariant enforcement

Per production_plan.md: 95%+ coverage, mypy --strict compliant.
"""

import pytest

from agent_memory.domain.entities import AgentBlocks, KVBlock
from agent_memory.domain.errors import AgentBlocksValidationError, BlockValidationError

pytestmark = pytest.mark.unit


class TestKVBlock:
    """Test suite for KVBlock entity."""

    def test_create_valid_block(self) -> None:
        """Should create block with valid attributes."""
        block = KVBlock(
            block_id=42,
            layer_id=0,
            token_count=128,
            layer_data={"k": "data", "v": "data"},
            metadata={"agent_id": "agent_1"},
        )

        assert block.block_id == 42
        assert block.layer_id == 0
        assert block.token_count == 128
        assert block.layer_data == {"k": "data", "v": "data"}
        assert block.metadata == {"agent_id": "agent_1"}

    def test_create_block_with_defaults(self) -> None:
        """Should create block with default metadata."""
        block = KVBlock(block_id=0, layer_id=0, token_count=0, layer_data=None)

        assert block.metadata == {}

    def test_is_full_true(self) -> None:
        """Should return True when block has 256 tokens."""
        block = KVBlock(block_id=0, layer_id=0, token_count=256, layer_data=None)

        assert block.is_full() is True

    def test_is_full_false(self) -> None:
        """Should return False when block has less than 256 tokens."""
        block = KVBlock(block_id=0, layer_id=0, token_count=255, layer_data=None)

        assert block.is_full() is False

    def test_is_empty_true(self) -> None:
        """Should return True when block has 0 tokens."""
        block = KVBlock(block_id=0, layer_id=0, token_count=0, layer_data=None)

        assert block.is_empty() is True

    def test_is_empty_false(self) -> None:
        """Should return False when block has tokens."""
        block = KVBlock(block_id=0, layer_id=0, token_count=1, layer_data=None)

        assert block.is_empty() is False

    def test_reject_negative_block_id(self) -> None:
        """Should raise ValueError for negative block_id."""
        with pytest.raises(BlockValidationError, match="block_id must be >= 0"):
            KVBlock(block_id=-1, layer_id=0, token_count=0, layer_data=None)

    def test_reject_negative_layer_id(self) -> None:
        """Should raise ValueError for negative layer_id."""
        with pytest.raises(BlockValidationError, match="layer_id must be >= 0"):
            KVBlock(block_id=0, layer_id=-1, token_count=0, layer_data=None)

    def test_reject_negative_token_count(self) -> None:
        """Should raise ValueError for negative token_count."""
        with pytest.raises(BlockValidationError, match="token_count must be 0-256"):
            KVBlock(block_id=0, layer_id=0, token_count=-1, layer_data=None)

    def test_reject_token_count_exceeds_256(self) -> None:
        """Should raise ValueError when token_count > 256."""
        with pytest.raises(BlockValidationError, match="token_count must be 0-256"):
            KVBlock(block_id=0, layer_id=0, token_count=257, layer_data=None)

    def test_accept_boundary_values(self) -> None:
        """Should accept boundary values for all attributes."""
        block = KVBlock(block_id=0, layer_id=0, token_count=0, layer_data=None)
        assert block.block_id == 0

        block = KVBlock(block_id=9999, layer_id=0, token_count=0, layer_data=None)
        assert block.block_id == 9999

        block = KVBlock(block_id=0, layer_id=0, token_count=256, layer_data=None)
        assert block.token_count == 256


class TestAgentBlocks:
    """Test suite for AgentBlocks entity."""

    def test_create_empty_agent(self) -> None:
        """Should create agent with no blocks."""
        agent = AgentBlocks(agent_id="agent_1", blocks={}, total_tokens=0)

        assert agent.agent_id == "agent_1"
        assert agent.blocks == {}
        assert agent.total_tokens == 0
        assert agent.num_blocks() == 0
        assert agent.num_layers() == 0

    def test_create_agent_with_blocks(self) -> None:
        """Should create agent with pre-allocated blocks."""
        # All layers store the same sequence, so token counts must match
        block_l0_a = KVBlock(block_id=1, layer_id=0, token_count=256, layer_data=None)
        block_l0_b = KVBlock(block_id=2, layer_id=0, token_count=128, layer_data=None)
        block_l1_a = KVBlock(block_id=3, layer_id=1, token_count=256, layer_data=None)
        block_l1_b = KVBlock(block_id=4, layer_id=1, token_count=128, layer_data=None)

        agent = AgentBlocks(
            agent_id="agent_1",
            blocks={0: [block_l0_a, block_l0_b], 1: [block_l1_a, block_l1_b]},
            total_tokens=384,
        )

        assert agent.num_blocks() == 4
        assert agent.num_layers() == 2
        assert agent.total_tokens == 384

    def test_num_blocks_counts_all_layers(self) -> None:
        """Should count blocks across all layers."""
        # Each layer has one block with 256 tokens (same sequence replicated)
        block1 = KVBlock(block_id=1, layer_id=0, token_count=256, layer_data=None)
        block2 = KVBlock(block_id=2, layer_id=1, token_count=256, layer_data=None)
        block3 = KVBlock(block_id=3, layer_id=2, token_count=256, layer_data=None)

        agent = AgentBlocks(
            agent_id="agent_1",
            blocks={0: [block1], 1: [block2], 2: [block3]},
            total_tokens=256,
        )

        assert agent.num_blocks() == 3
        assert agent.num_layers() == 3

    def test_blocks_for_layer_exists(self) -> None:
        """Should return blocks for existing layer."""
        block1 = KVBlock(block_id=1, layer_id=0, token_count=256, layer_data=None)
        block2 = KVBlock(block_id=2, layer_id=0, token_count=128, layer_data=None)

        agent = AgentBlocks(
            agent_id="agent_1",
            blocks={0: [block1, block2]},
            total_tokens=384,
        )

        blocks = agent.blocks_for_layer(0)
        assert len(blocks) == 2
        assert blocks[0].block_id == 1
        assert blocks[1].block_id == 2

    def test_blocks_for_layer_not_exists(self) -> None:
        """Should return empty list for non-existent layer."""
        agent = AgentBlocks(agent_id="agent_1", blocks={}, total_tokens=0)

        blocks = agent.blocks_for_layer(99)
        assert blocks == []

    def test_reject_empty_agent_id(self) -> None:
        """Should raise ValueError for empty agent_id."""
        with pytest.raises(AgentBlocksValidationError, match="agent_id cannot be empty"):
            AgentBlocks(agent_id="", blocks={}, total_tokens=0)

    def test_reject_negative_total_tokens(self) -> None:
        """Should raise ValueError for negative total_tokens."""
        with pytest.raises(AgentBlocksValidationError, match="total_tokens must be >= 0"):
            AgentBlocks(agent_id="agent_1", blocks={}, total_tokens=-1)

    def test_reject_mismatched_total_tokens(self) -> None:
        """Should raise ValueError when total_tokens doesn't match sum."""
        block = KVBlock(block_id=1, layer_id=0, token_count=256, layer_data=None)

        with pytest.raises(AgentBlocksValidationError, match=r"total_tokens .* doesn't match sum"):
            AgentBlocks(
                agent_id="agent_1",
                blocks={0: [block]},
                total_tokens=100,  # Wrong! Should be 256
            )

    def test_create_with_metadata(self) -> None:
        """Should store optional metadata."""
        agent = AgentBlocks(
            agent_id="agent_1",
            blocks={},
            total_tokens=0,
            metadata={"model": "gemma-3", "created_at": "2026-01-24"},
        )

        assert agent.metadata["model"] == "gemma-3"
        assert agent.metadata["created_at"] == "2026-01-24"

    def test_create_with_default_metadata(self) -> None:
        """Should default to empty metadata dict."""
        agent = AgentBlocks(agent_id="agent_1", blocks={}, total_tokens=0)

        assert agent.metadata == {}

    def test_detach_for_prefix_cache_copies_blocks(self) -> None:
        """detach_for_prefix_cache() creates independent block copies."""
        fake_tensor = {"k": [1, 2, 3], "v": [4, 5, 6]}
        block = KVBlock(block_id=42, layer_id=0, token_count=100, layer_data=fake_tensor)
        agent = AgentBlocks(
            agent_id="original",
            blocks={0: [block]},
            total_tokens=100,
            token_sequence=[1, 2, 3],
            prompt_text="hello world",
        )

        detached = agent.detach_for_prefix_cache()

        # Detached has different identity
        assert detached.agent_id == "_prefix_cache"
        assert detached is not agent

        # Blocks are independent objects
        assert detached.blocks[0][0] is not agent.blocks[0][0]
        assert detached.blocks[0][0].block_id >= 10_000_000  # High IDs = not in pool

        # Tensor data is shared (same reference — immutable)
        assert detached.blocks[0][0].layer_data is agent.blocks[0][0].layer_data

        # Token sequence is a copy
        assert detached.token_sequence == [1, 2, 3]
        assert detached.token_sequence is not agent.token_sequence

        # Prompt text preserved
        assert detached.prompt_text == "hello world"
        assert detached.total_tokens == 100

    def test_detach_survives_original_clear(self) -> None:
        """Clearing original blocks' layer_data doesn't affect detached copy."""
        fake_tensor = {"k": [1, 2, 3], "v": [4, 5, 6]}
        block = KVBlock(block_id=42, layer_id=0, token_count=100, layer_data=fake_tensor)
        agent = AgentBlocks(
            agent_id="original",
            blocks={0: [block]},
            total_tokens=100,
        )

        detached = agent.detach_for_prefix_cache()

        # Simulate what submit() does after reconstruction
        block.layer_data = None

        # Detached still has the tensor data
        assert detached.blocks[0][0].layer_data is not None
        assert detached.blocks[0][0].layer_data == {"k": [1, 2, 3], "v": [4, 5, 6]}

    def test_detach_empty_blocks(self) -> None:
        """detach_for_prefix_cache() works on empty blocks."""
        agent = AgentBlocks(agent_id="empty", blocks={}, total_tokens=0)
        detached = agent.detach_for_prefix_cache()
        assert detached.blocks == {}
        assert detached.total_tokens == 0

    def test_trim_to_prefix_exact_block_boundary(self) -> None:
        """trim_to_prefix at exact block boundary keeps only full blocks."""
        blocks = {
            0: [
                KVBlock(block_id=1, layer_id=0, token_count=256, layer_data="a"),
                KVBlock(block_id=2, layer_id=0, token_count=256, layer_data="b"),
                KVBlock(block_id=3, layer_id=0, token_count=100, layer_data="c"),
            ],
        }
        agent = AgentBlocks(
            agent_id="test", blocks=blocks, total_tokens=612,
            token_sequence=list(range(612)),
        )
        trimmed = agent.trim_to_prefix(512)  # Exactly 2 blocks
        assert trimmed.total_tokens == 512
        assert len(trimmed.blocks[0]) == 2
        assert trimmed.token_sequence == list(range(512))

    def test_trim_to_prefix_partial_block(self) -> None:
        """trim_to_prefix with partial block reduces token_count."""
        blocks = {
            0: [
                KVBlock(block_id=1, layer_id=0, token_count=256, layer_data="a"),
                KVBlock(block_id=2, layer_id=0, token_count=256, layer_data="b"),
            ],
        }
        agent = AgentBlocks(
            agent_id="test", blocks=blocks, total_tokens=512,
            token_sequence=list(range(512)),
        )
        trimmed = agent.trim_to_prefix(300)  # 1 full + 44 partial
        assert trimmed.total_tokens == 300
        assert len(trimmed.blocks[0]) == 2
        assert trimmed.blocks[0][0].token_count == 256  # Full block
        assert trimmed.blocks[0][1].token_count == 44   # Partial
        assert len(trimmed.token_sequence) == 300

    def test_trim_to_prefix_multi_layer(self) -> None:
        """trim_to_prefix works across layers."""
        blocks = {
            0: [KVBlock(block_id=1, layer_id=0, token_count=256, layer_data="a"),
                KVBlock(block_id=2, layer_id=0, token_count=100, layer_data="b")],
            1: [KVBlock(block_id=3, layer_id=1, token_count=256, layer_data="c"),
                KVBlock(block_id=4, layer_id=1, token_count=100, layer_data="d")],
        }
        agent = AgentBlocks(agent_id="test", blocks=blocks, total_tokens=356)
        trimmed = agent.trim_to_prefix(256)
        assert trimmed.total_tokens == 256
        for layer_id in (0, 1):
            assert len(trimmed.blocks[layer_id]) == 1
            assert trimmed.blocks[layer_id][0].token_count == 256

    def test_trim_to_prefix_zero(self) -> None:
        """trim_to_prefix(0) returns empty blocks."""
        block = KVBlock(block_id=1, layer_id=0, token_count=256, layer_data="a")
        agent = AgentBlocks(agent_id="test", blocks={0: [block]}, total_tokens=256)
        trimmed = agent.trim_to_prefix(0)
        assert trimmed.total_tokens == 0
        assert trimmed.blocks == {}

    def test_detach_multi_layer(self) -> None:
        """detach_for_prefix_cache() handles multiple layers."""
        blocks = {
            0: [KVBlock(block_id=1, layer_id=0, token_count=256, layer_data={"k": "a"})],
            1: [KVBlock(block_id=2, layer_id=1, token_count=256, layer_data={"k": "b"})],
        }
        agent = AgentBlocks(agent_id="multi", blocks=blocks, total_tokens=256)
        detached = agent.detach_for_prefix_cache()

        assert len(detached.blocks) == 2
        assert detached.blocks[0][0].layer_data == {"k": "a"}
        assert detached.blocks[1][0].layer_data == {"k": "b"}
        # All block_ids are negative and unique
        ids = [b.block_id for layer in detached.blocks.values() for b in layer]
        assert all(i >= 10_000_000 for i in ids)
        assert len(set(ids)) == len(ids)  # Unique
