"""Integration tests for BlockPoolBatchEngine.

These tests use real MLX models and validate end-to-end functionality.
Requires Apple Silicon (M1/M2/M3/M4) and mlx/mlx_lm installed.
"""

import pytest

from semantic.application.batch_engine import BlockPoolBatchEngine
from semantic.domain.services import BlockPool
from semantic.domain.value_objects import ModelCacheSpec


@pytest.fixture(scope="module")
def model_and_tokenizer():
    """Load SmolLM2-135M model and tokenizer (once per module).

    Note: This is a real model load - takes ~5-10 seconds.
    Scope is 'module' to avoid reloading for each test.
    """
    # TODO: Day 9 implementation
    # from mlx_lm import load
    # model, tokenizer = load("mlx-community/SmolLM2-135M-Instruct")
    # return model, tokenizer
    pytest.skip("Integration tests require MLX model (implement Day 9)")


@pytest.fixture
def spec(model_and_tokenizer):
    """Extract ModelCacheSpec from loaded model."""
    # TODO: Day 9 implementation
    # model, _ = model_and_tokenizer
    # return ModelCacheSpec.from_model(model)
    pytest.skip("Integration tests require MLX model (implement Day 9)")


@pytest.fixture
def pool(spec):
    """Create BlockPool with 100 blocks for testing."""
    return BlockPool(spec=spec, total_blocks=100)


@pytest.fixture
def engine(model_and_tokenizer, pool, spec):
    """Create BlockPoolBatchEngine for testing."""
    model, tokenizer = model_and_tokenizer
    return BlockPoolBatchEngine(
        model=model,
        tokenizer=tokenizer,
        pool=pool,
        spec=spec,
    )


@pytest.mark.integration
class TestBlockPoolBatchEngineIntegration:
    """Integration tests with real MLX model."""

    def test_single_agent_fresh_generation(self, engine) -> None:
        """Should generate text for single agent with no cache."""
        # TODO: Day 9 implementation
        pytest.skip("Implement after BlockPoolBatchEngine.step() is done (Day 8)")

    def test_single_agent_with_cache_resume(self, engine, pool) -> None:
        """Should resume generation from cached state."""
        # TODO: Day 9 implementation
        pytest.skip("Implement after cache reconstruction is done (Day 7)")

    def test_multi_agent_variable_lengths(self, engine) -> None:
        """Should handle 3 agents with different prompt lengths."""
        # TODO: Day 9 implementation
        pytest.skip("Implement after BlockPoolBatchEngine.step() is done (Day 8)")

    def test_no_memory_leaks(self, engine, pool) -> None:
        """Should not leak blocks across multiple generations."""
        # TODO: Day 9 implementation
        # Run 10 generations, verify pool size stable
        pytest.skip("Implement after full engine is complete (Day 9)")

    def test_pool_exhaustion_error(self, engine, pool) -> None:
        """Should raise PoolExhaustedError when no blocks available."""
        # TODO: Day 9 implementation
        pytest.skip("Implement after BlockPoolBatchEngine.submit() handles exhaustion (Day 6)")

    def test_empty_prompt_rejection(self, engine) -> None:
        """Should reject empty prompt with InvalidRequestError."""
        # TODO: Day 9 implementation
        # This test can actually run now (validation is implemented)
        pytest.skip("Can implement immediately - validation already works")
