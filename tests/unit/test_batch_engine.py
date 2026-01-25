"""Unit tests for BlockPoolBatchEngine.

Tests the batch inference engine with mocked dependencies.
No MLX required - uses fake models and tokenizers.
"""

import pytest

from semantic.application.batch_engine import BlockPoolBatchEngine
from semantic.domain.errors import InvalidRequestError, ModelNotFoundError
from semantic.domain.services import BlockPool
from semantic.domain.value_objects import ModelCacheSpec


class FakeModel:
    """Fake MLX model for testing."""

    pass


class FakeTokenizer:
    """Fake MLX tokenizer for testing."""

    def encode(self, text: str) -> list[int]:
        """Fake tokenization - returns list of ints based on text length."""
        # Simple: 1 token per character (for testing block allocation)
        return list(range(len(text)))


@pytest.fixture
def model() -> FakeModel:
    """Create fake model."""
    return FakeModel()


@pytest.fixture
def tokenizer() -> FakeTokenizer:
    """Create fake tokenizer."""
    return FakeTokenizer()


@pytest.fixture
def spec() -> ModelCacheSpec:
    """Create model cache spec for testing."""
    return ModelCacheSpec(
        n_layers=12,
        n_kv_heads=4,
        head_dim=64,
        block_tokens=256,
        layer_types=["global"] * 12,
        sliding_window_size=None,
    )


@pytest.fixture
def pool(spec: ModelCacheSpec) -> BlockPool:
    """Create block pool with 100 blocks."""
    return BlockPool(spec=spec, total_blocks=100)


class TestBlockPoolBatchEngineInit:
    """Tests for BlockPoolBatchEngine.__init__()."""

    def test_create_engine_with_valid_inputs(
        self, model: FakeModel, tokenizer: FakeTokenizer, pool: BlockPool, spec: ModelCacheSpec
    ) -> None:
        """Should create engine with valid inputs."""
        engine = BlockPoolBatchEngine(
            model=model,
            tokenizer=tokenizer,
            pool=pool,
            spec=spec,
        )

        assert engine._model is model
        assert engine._tokenizer is tokenizer
        assert engine._pool is pool
        assert engine._spec is spec
        assert engine._batch_gen is None  # Lazy initialization
        assert engine._active_requests == {}
        assert engine._agent_blocks == {}

    def test_reject_none_model(self, tokenizer, pool, spec) -> None:
        """Should raise ModelNotFoundError if model is None."""
        with pytest.raises(
            ModelNotFoundError, match="Model must be loaded before creating engine"
        ):
            BlockPoolBatchEngine(
                model=None,
                tokenizer=tokenizer,
                pool=pool,
                spec=spec,
            )

    def test_reject_none_tokenizer(self, model, pool, spec) -> None:
        """Should raise ModelNotFoundError if tokenizer is None."""
        with pytest.raises(
            ModelNotFoundError, match="Tokenizer must be loaded before creating engine"
        ):
            BlockPoolBatchEngine(
                model=model,
                tokenizer=None,
                pool=pool,
                spec=spec,
            )

    def test_reject_none_pool(self, model, tokenizer, spec) -> None:
        """Should raise InvalidRequestError if pool is None."""
        with pytest.raises(InvalidRequestError, match="BlockPool is required"):
            BlockPoolBatchEngine(
                model=model,
                tokenizer=tokenizer,
                pool=None,  # type: ignore[arg-type]
                spec=spec,
            )

    def test_reject_none_spec(self, model, tokenizer, pool) -> None:
        """Should raise InvalidRequestError if spec is None."""
        with pytest.raises(InvalidRequestError, match="ModelCacheSpec is required"):
            BlockPoolBatchEngine(
                model=model,
                tokenizer=tokenizer,
                pool=pool,
                spec=None,  # type: ignore[arg-type]
            )


class TestBlockPoolBatchEngineSubmit:
    """Tests for BlockPoolBatchEngine.submit()."""

    @pytest.fixture
    def engine(self, model, tokenizer, pool, spec):
        """Create engine for testing."""
        return BlockPoolBatchEngine(
            model=model,
            tokenizer=tokenizer,
            pool=pool,
            spec=spec,
        )

    def test_reject_empty_prompt(self, engine) -> None:
        """Should raise InvalidRequestError if prompt is empty."""
        with pytest.raises(InvalidRequestError, match="Prompt cannot be empty"):
            engine.submit(agent_id="test", prompt="", max_tokens=100)

    def test_reject_zero_max_tokens(self, engine) -> None:
        """Should raise InvalidRequestError if max_tokens is 0."""
        with pytest.raises(InvalidRequestError, match="max_tokens must be positive"):
            engine.submit(agent_id="test", prompt="Hello", max_tokens=0)

    def test_reject_negative_max_tokens(self, engine) -> None:
        """Should raise InvalidRequestError if max_tokens is negative."""
        with pytest.raises(InvalidRequestError, match="max_tokens must be positive"):
            engine.submit(agent_id="test", prompt="Hello", max_tokens=-1)

    def test_reject_empty_agent_id(self, engine) -> None:
        """Should raise InvalidRequestError if agent_id is empty."""
        with pytest.raises(InvalidRequestError, match="agent_id cannot be empty"):
            engine.submit(agent_id="", prompt="Hello", max_tokens=100)

    @pytest.mark.skip(reason="Requires BatchGenerator mock (Day 6)")
    def test_submit_without_cache_allocates_blocks(self, engine, pool) -> None:
        """Should allocate blocks when no cache provided."""
        # This test requires mocking BatchGenerator.insert()
        # Will be implemented on Day 6
        pass

    @pytest.mark.skip(reason="Cache reconstruction not implemented (Day 7)")
    def test_submit_with_cache_reconstructs(self, engine) -> None:
        """Should reconstruct cache from blocks when cache provided."""
        # Will be implemented after _reconstruct_cache() is done
        pass


class TestBlockPoolBatchEngineStep:
    """Tests for BlockPoolBatchEngine.step()."""

    @pytest.fixture
    def engine(self, model, tokenizer, pool, spec):
        """Create engine for testing."""
        return BlockPoolBatchEngine(
            model=model,
            tokenizer=tokenizer,
            pool=pool,
            spec=spec,
        )

    def test_step_with_no_batch_returns_immediately(self, engine) -> None:
        """Should return empty iterator if no batch created."""
        completions = list(engine.step())
        assert completions == []

    @pytest.mark.skip(reason="step() not implemented (Day 8)")
    def test_step_executes_decode_and_yields_completions(self, engine) -> None:
        """Should execute decode and yield CompletedGeneration."""
        # Will be implemented on Day 8
        pass


class TestBlockPoolBatchEngineCacheReconstruction:
    """Tests for BlockPoolBatchEngine._reconstruct_cache()."""

    @pytest.fixture
    def engine(self, model, tokenizer, pool, spec):
        """Create engine for testing."""
        return BlockPoolBatchEngine(
            model=model,
            tokenizer=tokenizer,
            pool=pool,
            spec=spec,
        )

    @pytest.mark.skip(reason="_reconstruct_cache() not implemented (Day 7)")
    def test_reconstruct_cache_from_single_block(self, engine) -> None:
        """Should reconstruct cache from single block."""
        # Will be implemented on Day 7
        pass

    @pytest.mark.skip(reason="_reconstruct_cache() not implemented (Day 7)")
    def test_reconstruct_cache_from_multiple_blocks(self, engine) -> None:
        """Should reconstruct cache from multiple blocks."""
        # Will be implemented on Day 7
        pass


class TestBlockPoolBatchEngineCacheExtraction:
    """Tests for BlockPoolBatchEngine._extract_cache()."""

    @pytest.fixture
    def engine(self, model, tokenizer, pool, spec):
        """Create engine for testing."""
        return BlockPoolBatchEngine(
            model=model,
            tokenizer=tokenizer,
            pool=pool,
            spec=spec,
        )

    @pytest.mark.skip(reason="_extract_cache() not implemented (Day 8)")
    def test_extract_cache_converts_to_blocks(self, engine) -> None:
        """Should convert KVCache back to blocks."""
        # Will be implemented on Day 8
        pass
