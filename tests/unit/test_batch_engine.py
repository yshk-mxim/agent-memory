"""Unit tests for BlockPoolBatchEngine.

Tests the batch inference engine with mocked dependencies.
No MLX required - uses fake models and tokenizers.
"""

from typing import Any

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

    @property
    def eos_token_ids(self) -> list[int]:
        """Return fake EOS token IDs."""
        return [2]  # Common EOS token ID


class FakeBatchGenerator:
    """Fake BatchGenerator for testing (mimics mlx_lm.BatchGenerator).

    Simulates batch inference without requiring MLX/GPU.
    Tracks inserted sequences and simulates completion.
    """

    def __init__(self, model: Any, tokenizer: Any) -> None:
        """Initialize fake batch generator."""
        self._model = model
        self._tokenizer = tokenizer
        self._sequences: dict[str, dict[str, Any]] = {}  # uid → sequence data
        self._next_uid = 0

    def insert(
        self,
        prompts: list[list[int]],
        max_tokens: int = 256,
        caches: list[Any] | None = None,
    ) -> list[str]:
        """Insert sequences into batch (fake implementation).

        Args:
            prompts: List of tokenized prompts (list of token IDs).
            max_tokens: Maximum tokens to generate per sequence.
            caches: Optional list of KV caches (one per prompt).

        Returns:
            List of UIDs (one per prompt).
        """
        uids = []
        for i, prompt_tokens in enumerate(prompts):
            # Generate fake UID
            uid = f"fake_uid_{self._next_uid}"
            self._next_uid += 1

            # Store sequence data
            self._sequences[uid] = {
                "prompt_tokens": prompt_tokens,
                "max_tokens": max_tokens,
                "cache": caches[i] if caches else None,
                "generated_tokens": 0,
                "finished": False,
                "finish_reason": None,
                "text": f"Generated text for prompt {i}",  # Fake output
            }

            uids.append(uid)

        return uids

    def next(self) -> Any:
        """Execute one decode step (fake implementation).

        Returns:
            Fake BatchResponse with finished sequences.
        """
        # Simulate: mark all sequences as finished after first next() call
        finished = []

        for uid, seq in self._sequences.items():
            if not seq["finished"]:
                # Simulate generation complete
                seq["finished"] = True
                seq["finish_reason"] = "stop"  # Simulate EOS token
                seq["generated_tokens"] = 10  # Fake token count

                # Create fake finished sequence
                finished.append({
                    "uid": uid,
                    "text": seq["text"],
                    "finish_reason": seq["finish_reason"],
                    "token_count": len(seq["prompt_tokens"]) + seq["generated_tokens"],
                })

        # Return fake BatchResponse
        return FakeBatchResponse(finished=finished)

    def extract_cache(self, uid: str) -> list[tuple[Any, Any]]:
        """Extract cache for a sequence (fake implementation).

        Args:
            uid: Sequence UID.

        Returns:
            Fake cache (empty list for now).
        """
        if uid not in self._sequences:
            raise KeyError(f"UID {uid} not found in batch")

        # Return fake cache (empty for now - Day 8 will implement properly)
        return []

    def remove(self, uid: str) -> None:
        """Remove sequence from batch (fake implementation).

        Args:
            uid: Sequence UID.
        """
        if uid in self._sequences:
            del self._sequences[uid]


class FakeBatchResponse:
    """Fake BatchResponse (mimics mlx_lm BatchResponse)."""

    def __init__(self, finished: list[dict[str, Any]]) -> None:
        """Initialize fake batch response.

        Args:
            finished: List of finished sequence data.
        """
        self._finished = finished

    @property
    def finished(self) -> list[Any]:
        """Return list of finished sequences."""
        return [FakeGenerationResponse(**seq) for seq in self._finished]


class FakeGenerationResponse:
    """Fake GenerationResponse (mimics mlx_lm GenerationResponse)."""

    def __init__(self, uid: str, text: str, finish_reason: str, token_count: int) -> None:
        """Initialize fake generation response."""
        self.uid = uid
        self.text = text
        self.finish_reason = finish_reason
        self.token_count = token_count


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
            batch_gen_factory=FakeBatchGenerator,  # Inject fake for testing
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

    def test_submit_without_cache_allocates_blocks(self, engine, pool) -> None:
        """Should allocate blocks when no cache provided."""
        # Submit request (should allocate blocks)
        initial_available = pool.available_blocks()
        uid = engine.submit(agent_id="test_agent", prompt="Hello world", max_tokens=50)

        # Verify UID returned
        assert uid is not None
        assert isinstance(uid, str)
        assert uid.startswith("fake_uid_")  # From FakeBatchGenerator

        # Verify blocks allocated (prompt is 11 characters = 11 tokens in FakeTokenizer)
        # Need 1 block (11 tokens / 256 block_tokens = 0.04 → rounds up to 1)
        expected_blocks_allocated = 1
        assert pool.available_blocks() == initial_available - expected_blocks_allocated

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
            batch_gen_factory=FakeBatchGenerator,  # Inject fake for testing
        )

    def test_step_with_no_batch_returns_immediately(self, engine) -> None:
        """Should return empty iterator if no batch created."""
        completions = list(engine.step())
        assert completions == []

    def test_step_executes_decode_and_yields_completions(self, engine) -> None:
        """Should execute decode and yield CompletedGeneration."""
        # Submit a request first
        uid = engine.submit(agent_id="test_agent", prompt="Hello", max_tokens=50)

        # Execute step (should yield completion)
        completions = list(engine.step())

        # Verify we got one completion
        assert len(completions) == 1

        completion = completions[0]
        assert completion.uid == uid
        assert completion.text == "Generated text for prompt 0"  # From FakeBatchGenerator
        assert completion.finish_reason == "stop"
        assert completion.token_count > 0

        # Verify tracking cleaned up
        assert uid not in engine._active_requests


    def test_step_with_multiple_submissions(self, engine) -> None:
        """Should handle multiple concurrent submissions."""
        # Submit 3 requests
        uid1 = engine.submit(agent_id="agent_1", prompt="First", max_tokens=50)
        uid2 = engine.submit(agent_id="agent_2", prompt="Second", max_tokens=50)
        uid3 = engine.submit(agent_id="agent_3", prompt="Third", max_tokens=50)

        # Execute step (FakeBatchGenerator completes all in one step)
        completions = list(engine.step())

        # Verify all 3 completed
        assert len(completions) == 3

        # Verify UIDs match
        returned_uids = {c.uid for c in completions}
        expected_uids = {uid1, uid2, uid3}
        assert returned_uids == expected_uids

        # Verify all tracking cleaned up
        assert len(engine._active_requests) == 0


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
            batch_gen_factory=FakeBatchGenerator,  # Inject fake for testing
        )

    @pytest.mark.skip(reason="Requires MLX arrays - deferred to integration tests (Day 9)")
    def test_reconstruct_cache_from_single_block(self, engine) -> None:
        """Should reconstruct cache from single block."""
        # Requires real mlx.array objects with proper KV cache format
        # Will be implemented in integration tests (Day 9)
        pass

    @pytest.mark.skip(reason="Requires MLX arrays - deferred to integration tests (Day 9)")
    def test_reconstruct_cache_from_multiple_blocks(self, engine) -> None:
        """Should reconstruct cache from multiple blocks."""
        # Requires real mlx.array objects with proper KV cache format
        # Will be implemented in integration tests (Day 9)
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
            batch_gen_factory=FakeBatchGenerator,  # Inject fake for testing
        )

    @pytest.mark.skip(reason="_extract_cache() not implemented (Day 8)")
    def test_extract_cache_converts_to_blocks(self, engine) -> None:
        """Should convert KVCache back to blocks."""
        # Will be implemented on Day 8
        pass
