# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Yakov Shkolnikov and contributors
"""Integration tests for BlockPoolBatchEngine.

These tests use real MLX models and validate end-to-end functionality.
Requires Apple Silicon (M1/M2/M3/M4) and mlx/mlx_lm installed.
"""

import pytest

from agent_memory.application.batch_engine import BlockPoolBatchEngine
from agent_memory.domain.errors import InvalidRequestError, PoolExhaustedError
from agent_memory.domain.services import BlockPool

# MLX tests will run - models will be downloaded on first run


@pytest.fixture(scope="module")
def model_and_tokenizer():
    """Load SmolLM2-135M model and tokenizer (once per module).

    Note: This is a real model load - takes ~5-10 seconds.
    Scope is 'module' to avoid reloading for each test.
    """
    from mlx_lm import load

    model, tokenizer = load("mlx-community/SmolLM2-135M-Instruct")
    return model, tokenizer


@pytest.fixture
def spec(model_and_tokenizer):
    """Extract ModelCacheSpec from loaded model."""
    from agent_memory.adapters.outbound.mlx_spec_extractor import get_extractor

    model, _ = model_and_tokenizer
    return get_extractor().extract_spec(model)


@pytest.fixture
def pool(spec):
    """Create BlockPool with 100 blocks for testing."""
    return BlockPool(spec=spec, total_blocks=100)


@pytest.fixture
def engine(model_and_tokenizer, pool, spec):
    """Create BlockPoolBatchEngine for testing."""
    from agent_memory.adapters.outbound.mlx_cache_adapter import MLXCacheAdapter

    model, tokenizer = model_and_tokenizer
    cache_adapter = MLXCacheAdapter()

    return BlockPoolBatchEngine(
        model=model,
        tokenizer=tokenizer,
        pool=pool,
        spec=spec,
        cache_adapter=cache_adapter,
    )


@pytest.mark.integration
class TestBlockPoolBatchEngineIntegration:
    """Integration tests with real MLX model."""

    def test_single_agent_fresh_generation(self, engine) -> None:
        """Should generate text for single agent with no cache."""
        # Submit generation request
        uid = engine.submit(
            agent_id="test_agent",
            prompt="Hello",
            max_tokens=20,
        )

        # Execute generation (call step() repeatedly until completion)
        completions = []
        for completion in engine.step():
            completions.append(completion)

        # Verify completion
        assert len(completions) == 1, "Should yield exactly one completion"
        completion = completions[0]

        assert completion.uid == uid, "UID should match"
        assert len(completion.text) > 0, "Should generate non-empty text"
        assert completion.finish_reason in ["stop", "length"], "Should finish normally"
        assert completion.token_count > 0, "Should have tokens"
        assert completion.blocks.total_tokens > 0, "Should have blocks"

    @pytest.mark.skip(reason="Cache reconstruction needs KVCache objects from reconstructed tensors")
    def test_single_agent_with_cache_resume(self, engine, pool) -> None:
        """Should resume generation from cached state.

        Currently skipped because _reconstruct_cache() returns (K, V) tuples,
        but MLX BatchGenerator.insert() expects cache objects with .size() method.
        """
        # First generation
        engine.submit(agent_id="test_agent", prompt="Hello", max_tokens=10)
        completions1 = []
        for completion in engine.step():
            completions1.append(completion)
        cached_blocks = completions1[0].blocks

        # Record initial token count
        initial_tokens = cached_blocks.total_tokens

        # Resume with cache
        engine.submit(
            agent_id="test_agent",
            prompt=" world",
            cache=cached_blocks,
            max_tokens=10,
        )
        completions2 = []
        for completion in engine.step():
            completions2.append(completion)

        # Verify cache was used (blocks exist, generation continues)
        assert len(completions2) == 1, "Should complete second generation"
        assert completions2[0].blocks.total_tokens > initial_tokens, "Should add more tokens"

    def test_multi_agent_variable_lengths(self, engine) -> None:
        """Should handle 3 agents with different prompt lengths."""
        # Submit 3 requests with different lengths
        uid1 = engine.submit(agent_id="agent_1", prompt="Hi", max_tokens=10)
        uid2 = engine.submit(agent_id="agent_2", prompt="Hello world", max_tokens=10)
        uid3 = engine.submit(
            agent_id="agent_3",
            prompt="This is a longer prompt with more tokens",
            max_tokens=10,
        )

        # Execute all generations (call step() repeatedly)
        completions = []
        for completion in engine.step():
            completions.append(completion)

        # Verify all completed
        assert len(completions) == 3, "Should complete all 3 agents"

        # Verify UIDs match
        completion_uids = {c.uid for c in completions}
        expected_uids = {uid1, uid2, uid3}
        assert completion_uids == expected_uids, "All UIDs should match"

        # Verify all have text
        for completion in completions:
            assert len(completion.text) > 0, f"Agent {completion.uid} should generate text"
            assert completion.blocks.total_tokens > 0, f"Agent {completion.uid} should have blocks"

    def test_no_memory_leaks(self, engine, pool) -> None:
        """Should not leak blocks across multiple generations."""
        initial_available = pool.available_blocks()

        # Run 10 generations
        for i in range(10):
            engine.submit(agent_id=f"agent_{i}", prompt=f"Test {i}", max_tokens=10)
            completions = []
            for completion in engine.step():
                completions.append(completion)
            assert len(completions) == 1, f"Generation {i} should complete"

            # Free blocks after each generation
            for completion in completions:
                for layer_blocks in completion.blocks.blocks.values():
                    pool.free(layer_blocks, completion.blocks.agent_id)

        # Verify no memory leak
        final_available = pool.available_blocks()
        assert final_available == initial_available, (
            f"Memory leak detected: started with {initial_available}, ended with {final_available}"
        )

    def test_pool_exhaustion_error(self, engine, pool) -> None:
        """Should raise PoolExhaustedError when no blocks available."""
        # Exhaust the pool by not freeing blocks
        allocated_agents = []
        try:
            # Keep allocating until pool exhausted
            for i in range(200):  # More than pool capacity
                uid = engine.submit(agent_id=f"agent_{i}", prompt="Test", max_tokens=10)
                allocated_agents.append(uid)
        except PoolExhaustedError:
            # Expected - pool exhausted
            assert len(allocated_agents) > 0, "Should allocate some before exhaustion"
            return

        # If we get here, pool wasn't exhausted
        pytest.fail("Pool should have been exhausted")

    def test_empty_prompt_rejection(self, engine) -> None:
        """Should reject empty prompt with InvalidRequestError."""
        with pytest.raises(InvalidRequestError, match="Prompt cannot be empty"):
            engine.submit(agent_id="test", prompt="", max_tokens=10)
