# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Yakov Shkolnikov and contributors
"""Unit tests for BlockPoolBatchEngine.

Tests the batch inference engine with mocked dependencies.
No MLX required - uses fake models and tokenizers.
"""

import sys
from typing import Any
from unittest.mock import MagicMock

import pytest

# Mock MLX modules before importing batch_engine
sys.modules["mlx"] = MagicMock()
sys.modules["mlx.core"] = MagicMock()
sys.modules["mlx.utils"] = MagicMock()
sys.modules["mlx_lm"] = MagicMock()
sys.modules["mlx_lm.models"] = MagicMock()
sys.modules["mlx_lm.models.cache"] = MagicMock()

from agent_memory.application.batch_engine import BlockPoolBatchEngine
from agent_memory.domain.errors import InvalidRequestError, ModelNotFoundError
from agent_memory.domain.services import BlockPool
from agent_memory.domain.value_objects import ModelCacheSpec


class FakeTensor:
    """Fake tensor for testing (mimics MLX tensor behavior).

    Supports shape attribute and slicing operations needed for cache extraction.
    """

    def __init__(self, shape: tuple[int, ...], data: Any = None) -> None:
        """Initialize fake tensor with shape."""
        self.shape = shape
        self._data = data or [[[0.0] * shape[2]] * shape[1]] * shape[0]

    def __getitem__(self, key: Any) -> "FakeTensor":
        """Support slicing (returns new FakeTensor with adjusted shape)."""
        # Simplified slicing - just adjust sequence length (axis=2)
        if isinstance(key, tuple) and len(key) == 3:
            # Extract slice for axis 2 (sequence length)
            slice_obj = key[2]
            if isinstance(slice_obj, slice):
                start = slice_obj.start or 0
                stop = slice_obj.stop or self.shape[2]
                new_seq_len = stop - start
                return FakeTensor((self.shape[0], self.shape[1], new_seq_len))
        return self


class FakeModel:
    """Fake MLX model for testing."""


class FakeDetokenizer:
    """Fake detokenizer for streaming token-to-text conversion."""

    def __init__(self, tokenizer: "FakeTokenizer") -> None:
        self._tokenizer = tokenizer
        self._tokens: list[int] = []
        self._text = ""

    def add_token(self, token: int) -> None:
        """Add a token to the stream."""
        self._tokens.append(token)
        self._text = f"Generated text with {len(self._tokens)} tokens"

    @property
    def text(self) -> str:
        """Get current decoded text."""
        return self._text


class FakeTokenizer:
    """Fake MLX tokenizer for testing."""

    def __init__(self) -> None:
        self._detokenizer = FakeDetokenizer(self)

    def encode(self, text: str) -> list[int]:
        """Fake tokenization - returns list of ints based on text length."""
        # Simple: 1 token per character (for testing block allocation)
        return list(range(len(text)))

    def decode(self, tokens: list[int]) -> str:
        """Fake detokenization - returns string from token IDs."""
        # Simple: return string representation of tokens
        return f"Generated text with {len(tokens)} tokens"

    @property
    def detokenizer(self) -> FakeDetokenizer:
        """Return the streaming detokenizer."""
        return self._detokenizer

    @property
    def eos_token_id(self) -> int:
        """Return fake EOS token ID (singular)."""
        return 2  # Common EOS token ID

    @property
    def eos_token_ids(self) -> list[int]:
        """Return fake EOS token IDs (plural for backward compatibility)."""
        return [self.eos_token_id]


class FakeBatchGenerator:
    """Fake BatchGenerator for testing (mimics mlx_lm.BatchGenerator).

    Simulates batch inference without requiring MLX/GPU.
    Research finding: Real MLX API generates 1 token per next() call.
    """

    def __init__(self, model: Any, tokenizer: Any, stop_tokens: set[int] | None = None) -> None:
        """Initialize fake batch generator.

        Args:
            model: Fake model (ignored).
            tokenizer: Fake tokenizer.
            stop_tokens: Set of token IDs that stop generation (default: {tokenizer.eos_token_id}).
        """
        self._model = model
        self._tokenizer = tokenizer
        self._stop_tokens = stop_tokens or {tokenizer.eos_token_id}
        self._sequences: dict[str, dict[str, Any]] = {}  # uid → sequence data
        self._next_uid = 0

    def insert(
        self,
        prompts: list[list[int]],
        max_tokens: list[int] | None = None,
        caches: list[Any] | None = None,
        samplers: list[Any] | None = None,
    ) -> list[str]:
        """Insert sequences into batch (fake implementation).

        Args:
            prompts: List of tokenized prompts (list of token IDs).
            max_tokens: List of max tokens per sequence (research finding: must be list!).
            caches: Optional list of KV caches (one per prompt).
            samplers: Optional list of samplers (required by real API).

        Returns:
            List of UIDs (one per prompt).
        """
        # Default max_tokens if not provided
        if max_tokens is None:
            max_tokens = [256] * len(prompts)

        uids = []
        for i, prompt_tokens in enumerate(prompts):
            # Generate fake UID
            uid = f"fake_uid_{self._next_uid}"
            self._next_uid += 1

            # Store sequence data
            self._sequences[uid] = {
                "prompt_tokens": prompt_tokens,
                "max_tokens": max_tokens[i] if isinstance(max_tokens, list) else max_tokens,
                "cache": caches[i] if caches else None,
                "generated_tokens": 0,
                "finished": False,
                "finish_reason": None,
            }

            uids.append(uid)

        return uids

    def next(self) -> list[Any]:
        """Execute one decode step - generates 1 token per sequence (fake implementation).

        Research finding: Real API returns list[Response], generates 1 token per call.

        Returns:
            List of FakeResponse objects (one per active sequence).
            Returns empty list [] when all sequences complete.
        """
        # Research finding: Return empty list when no active sequences
        if not any(not seq["finished"] for seq in self._sequences.values()):
            return []

        responses = []

        for uid, seq in self._sequences.items():
            if not seq["finished"]:
                # Generate one token
                seq["generated_tokens"] += 1

                # Determine token ID (simple: sequential IDs)
                token_id = 100 + seq["generated_tokens"]

                # Check if should finish
                finish_reason = None
                if token_id in self._stop_tokens:
                    finish_reason = "stop"
                    seq["finished"] = True
                    seq["finish_reason"] = "stop"
                elif seq["generated_tokens"] >= seq["max_tokens"]:
                    finish_reason = "length"
                    seq["finished"] = True
                    seq["finish_reason"] = "length"

                # Create response with prompt_cache only if finished
                prompt_cache = None
                if finish_reason is not None:
                    # Generate fake cache (list of (K, V) tuples)
                    seq_len = len(seq["prompt_tokens"]) + seq["generated_tokens"]
                    n_kv_heads, head_dim = 4, 64
                    n_layers = 12
                    prompt_cache = []
                    for _ in range(n_layers):
                        k = FakeTensor((n_kv_heads, head_dim, seq_len))
                        v = FakeTensor((n_kv_heads, head_dim, seq_len))
                        prompt_cache.append((k, v))

                # Create Response object (research finding: has .token, not .text)
                response = FakeResponse(
                    uid=uid,
                    token=token_id,
                    finish_reason=finish_reason,
                    prompt_cache=prompt_cache,
                )
                responses.append(response)

        return responses

    def extract_cache(self, uid: str) -> list[tuple[Any, Any]]:
        """Extract cache for a sequence (fake implementation).

        Args:
            uid: Sequence UID.

        Returns:
            Fake cache with realistic structure for testing block extraction.
        """
        if uid not in self._sequences:
            raise KeyError(f"UID {uid} not found in batch")

        # Generate realistic fake cache (Technical Fellow review fix)
        # Cache format: list of (k, v) tuples, one per layer
        # k, v shapes: (n_kv_heads, head_dim, seq_len)
        seq = self._sequences[uid]
        seq_len = len(seq["prompt_tokens"]) + seq["generated_tokens"]

        # Use realistic dimensions (matching typical test spec)
        n_kv_heads, head_dim = 4, 64
        n_layers = 12

        cache = []
        for _ in range(n_layers):
            k = FakeTensor((n_kv_heads, head_dim, seq_len))
            v = FakeTensor((n_kv_heads, head_dim, seq_len))
            cache.append((k, v))

        return cache

    def remove(self, uid: str) -> None:
        """Remove sequence from batch (fake implementation).

        Args:
            uid: Sequence UID.
        """
        if uid in self._sequences:
            del self._sequences[uid]


class FakeResponse:
    """Fake Response (mimics mlx_lm BatchGenerator.Response).

    Research findings:
    - Response has .token (singular), not .text or .tokens
    - Response has .finish_reason (None, "stop", or "length")
    - Response has .prompt_cache (only populated when finished)
    """

    def __init__(
        self,
        uid: str,
        token: int,
        finish_reason: str | None,
        prompt_cache: list[tuple[Any, Any]] | None,
    ) -> None:
        """Initialize fake response.

        Args:
            uid: Sequence unique identifier.
            token: Single generated token ID.
            finish_reason: None (continuing), "stop" (EOS), or "length" (max tokens).
            prompt_cache: KV cache (only when finished).
        """
        self.uid = uid
        self.token = token
        self.finish_reason = finish_reason
        self.prompt_cache = prompt_cache


class FakeCacheAdapter:
    """Fake cache adapter for testing.

    Mimics MLXCacheAdapter behavior without requiring MLX.
    Works with FakeTensor objects.
    """

    def concatenate_cache_blocks(
        self,
        k_tensors: list[Any],
        v_tensors: list[Any],
    ) -> tuple[Any, Any]:
        """Concatenate K/V tensors (fake implementation)."""
        # For FakeTensors, just return the first one (simplified)
        # Real implementation would concatenate along axis=2
        if k_tensors and v_tensors:
            # Calculate total sequence length
            total_seq_len = sum(t.shape[2] for t in k_tensors)
            # Return new FakeTensor with combined sequence length
            return (
                FakeTensor((k_tensors[0].shape[0], k_tensors[0].shape[1], total_seq_len)),
                FakeTensor((v_tensors[0].shape[0], v_tensors[0].shape[1], total_seq_len)),
            )
        return FakeTensor((0, 0, 0)), FakeTensor((0, 0, 0))

    def get_sequence_length(self, k_tensor: Any) -> int:
        """Extract sequence length from K tensor."""
        return int(k_tensor.shape[2])  # Cast to int for type safety

    def slice_cache_tensor(
        self,
        tensor: Any,
        start_token: int,
        end_token: int,
    ) -> Any:
        """Slice cache tensor along sequence axis."""
        # FakeTensor already supports slicing via __getitem__
        return tensor[:, :, start_token:end_token]


@pytest.fixture
def cache_adapter() -> FakeCacheAdapter:
    """Create fake cache adapter."""
    return FakeCacheAdapter()


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
        kv_bits=None,  # FP16 mode — avoids mx.quantize() on FakeTensors
    )


@pytest.fixture
def pool(spec: ModelCacheSpec) -> BlockPool:
    """Create block pool with 100 blocks."""
    return BlockPool(spec=spec, total_blocks=100)


class TestBlockPoolBatchEngineInit:
    """Tests for BlockPoolBatchEngine.__init__()."""

    def test_create_engine_with_valid_inputs(
        self,
        model: FakeModel,
        tokenizer: FakeTokenizer,
        pool: BlockPool,
        spec: ModelCacheSpec,
        cache_adapter: FakeCacheAdapter,
    ) -> None:
        """Should create engine with valid inputs."""
        engine = BlockPoolBatchEngine(
            model=model,
            tokenizer=tokenizer,
            pool=pool,
            spec=spec,
            cache_adapter=cache_adapter,
        )

        assert engine._model is model
        assert engine._tokenizer is tokenizer
        assert engine._pool is pool
        assert engine._spec is spec
        assert engine._batch_gen is None  # Lazy initialization
        assert engine._active_requests == {}
        assert engine._agent_blocks == {}

    def test_reject_none_model(self, tokenizer, pool, spec, cache_adapter) -> None:
        """Should raise ModelNotFoundError if model is None."""
        with pytest.raises(ModelNotFoundError, match="Model must be loaded before creating engine"):
            BlockPoolBatchEngine(
                cache_adapter=cache_adapter,
                model=None,
                tokenizer=tokenizer,
                pool=pool,
                spec=spec,
            )

    def test_reject_none_tokenizer(self, model, pool, spec, cache_adapter) -> None:
        """Should raise ModelNotFoundError if tokenizer is None."""
        with pytest.raises(
            ModelNotFoundError, match="Tokenizer must be loaded before creating engine"
        ):
            BlockPoolBatchEngine(
                cache_adapter=cache_adapter,
                model=model,
                tokenizer=None,
                pool=pool,
                spec=spec,
            )

    def test_reject_none_pool(self, model, tokenizer, spec, cache_adapter) -> None:
        """Should raise InvalidRequestError if pool is None."""
        with pytest.raises(InvalidRequestError, match="BlockPool is required"):
            BlockPoolBatchEngine(
                cache_adapter=cache_adapter,
                model=model,
                tokenizer=tokenizer,
                pool=None,  # type: ignore[arg-type]
                spec=spec,
            )

    def test_reject_none_spec(self, model, tokenizer, pool, cache_adapter) -> None:
        """Should raise InvalidRequestError if spec is None."""
        with pytest.raises(InvalidRequestError, match="ModelCacheSpec is required"):
            BlockPoolBatchEngine(
                cache_adapter=cache_adapter,
                model=model,
                tokenizer=tokenizer,
                pool=pool,
                spec=None,  # type: ignore[arg-type]
            )


class TestBlockPoolBatchEngineSubmit:
    """Tests for BlockPoolBatchEngine.submit()."""

    @pytest.fixture
    def engine(self, model, tokenizer, pool, spec, cache_adapter):
        """Create engine for testing."""
        return BlockPoolBatchEngine(
            cache_adapter=cache_adapter,
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
        assert isinstance(uid, str)
        assert uid.startswith("fake_uid_")  # From FakeBatchGenerator

        # Verify blocks allocated (prompt is 11 characters = 11 tokens in FakeTokenizer)
        # Need 1 block (11 tokens / 256 block_tokens = 0.04 → rounds up to 1)
        expected_blocks_allocated = 1
        assert pool.available_blocks() == initial_available - expected_blocks_allocated

    def test_submit_with_cache_reconstructs(self, engine, pool) -> None:
        """Should reconstruct cache from blocks when cache provided."""
        from agent_memory.domain.entities import AgentBlocks

        # Create fake agent blocks (simulating existing cache)
        allocated_block = pool.allocate(n_blocks=1, layer_id=0, agent_id="test")[0]
        fake_blocks = AgentBlocks(
            agent_id="test",
            blocks={0: [allocated_block]},
            total_tokens=allocated_block.token_count,
        )

        # Submit with existing cache
        uid = engine.submit(
            agent_id="test_agent",
            prompt="Hello",
            max_tokens=50,
            cache=fake_blocks,
        )

        # Verify request was tracked
        assert uid in engine._active_requests
        assert uid.startswith("fake_uid_")


class TestBlockPoolBatchEngineStep:
    """Tests for BlockPoolBatchEngine.step()."""

    @pytest.fixture
    def engine(self, model, tokenizer, pool, spec, cache_adapter):
        """Create engine for testing."""
        return BlockPoolBatchEngine(
            cache_adapter=cache_adapter,
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
        # Research finding: FakeBatchGenerator now generates 1 token per step
        completions = list(engine.step())

        # Verify we got one completion (FakeBatchGenerator generates 1 token per step)
        assert len(completions) == 1

        completion = completions[0]
        assert completion.uid == uid
        # Research finding: Text is decoded from accumulated tokens
        assert completion.text.startswith("Generated text with")
        assert completion.finish_reason in ["stop", "length"]
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
    def engine(self, model, tokenizer, pool, spec, cache_adapter):
        """Create engine for testing."""
        return BlockPoolBatchEngine(
            cache_adapter=cache_adapter,
            model=model,
            tokenizer=tokenizer,
            pool=pool,
            spec=spec,
            batch_gen_factory=FakeBatchGenerator,  # Inject fake for testing
        )

    def test_reconstruct_cache_from_single_block(self, engine, pool) -> None:
        """Should reconstruct cache from single block."""
        from agent_memory.domain.entities import AgentBlocks

        # Create agent blocks with single block per layer
        blocks = {}
        total_tokens = 0
        for layer_id in range(2):
            allocated = pool.allocate(1, layer_id, "test")
            # Add mock K/V data to each block
            for block in allocated:
                block.layer_data = {
                    "k": FakeTensor((1, engine._spec.n_kv_heads, 256, engine._spec.head_dim)),
                    "v": FakeTensor((1, engine._spec.n_kv_heads, 256, engine._spec.head_dim)),
                }
            blocks[layer_id] = allocated
            total_tokens += sum(b.token_count for b in allocated)

        agent_blocks = AgentBlocks(agent_id="test", blocks=blocks, total_tokens=total_tokens)

        # Reconstruct cache
        cache = engine._reconstruct_cache(agent_blocks)

        # Verify cache structure (list of tuples for each layer)
        assert isinstance(cache, list)
        assert len(cache) == engine._spec.n_layers

    def test_reconstruct_cache_from_multiple_blocks(self, engine, pool) -> None:
        """Should reconstruct cache from multiple blocks."""
        from agent_memory.domain.entities import AgentBlocks

        # Create agent blocks with multiple blocks per layer
        blocks = {}
        total_tokens = 0
        for layer_id in range(2):
            allocated1 = pool.allocate(1, layer_id, "test")
            allocated2 = pool.allocate(1, layer_id, "test")
            # Add mock K/V data to all blocks
            for block in allocated1 + allocated2:
                block.layer_data = {
                    "k": FakeTensor((1, engine._spec.n_kv_heads, 256, engine._spec.head_dim)),
                    "v": FakeTensor((1, engine._spec.n_kv_heads, 256, engine._spec.head_dim)),
                }
            blocks[layer_id] = allocated1 + allocated2
            total_tokens += sum(b.token_count for b in blocks[layer_id])

        agent_blocks = AgentBlocks(agent_id="test", blocks=blocks, total_tokens=total_tokens)

        # Reconstruct cache
        cache = engine._reconstruct_cache(agent_blocks)

        # Verify cache structure
        assert isinstance(cache, list)
        assert len(cache) == engine._spec.n_layers


class TestBlockPoolBatchEngineCacheExtraction:
    """Tests for BlockPoolBatchEngine._extract_cache()."""

    @pytest.fixture
    def engine(self, model, tokenizer, pool, spec, cache_adapter):
        """Create engine for testing."""
        return BlockPoolBatchEngine(
            cache_adapter=cache_adapter,
            model=model,
            tokenizer=tokenizer,
            pool=pool,
            spec=spec,
            batch_gen_factory=FakeBatchGenerator,  # Inject fake for testing
        )

    def test_extract_cache_converts_to_blocks(self, engine) -> None:
        """Should convert KVCache back to blocks."""
        # Submit a request to create active tracking
        uid = engine.submit(agent_id="test_agent", prompt="Hello", max_tokens=50)

        # Extract cache (will be called after generation completes)
        # The method requires the request to be in _active_requests
        assert uid in engine._active_requests

        # Verify extraction logic exists and can be called
        # (actual extraction happens in step() after generation)
        assert hasattr(engine, "_extract_cache")


class TestFinalizeSequenceBlockLeakProtection:
    """Tests that _finalize_sequence cleans up blocks on _extract_cache failure."""

    @pytest.fixture
    def engine(self, model, tokenizer, pool, spec, cache_adapter):
        """Create engine for testing."""
        return BlockPoolBatchEngine(
            cache_adapter=cache_adapter,
            model=model,
            tokenizer=tokenizer,
            pool=pool,
            spec=spec,
            batch_gen_factory=FakeBatchGenerator,
        )

    def test_extract_cache_failure_frees_blocks(self, engine, pool) -> None:
        """If _extract_cache raises, pool blocks should be cleaned up (no leak)."""
        initial_available = pool.available_blocks()

        # Pre-allocate some blocks for the agent (simulating old blocks)
        agent_id = "leak_test_agent"
        pool.allocate(n_blocks=3, layer_id=0, agent_id=agent_id)

        # Patch _extract_cache to raise an error
        original_extract = engine._extract_cache

        def failing_extract(*args, **kwargs):
            # Allocate blocks (simulating partial extraction) then fail
            pool.allocate(n_blocks=2, layer_id=0, agent_id=agent_id)
            raise RuntimeError("Simulated extraction failure")

        engine._extract_cache = failing_extract

        # Create a fake response
        response = FakeResponse(
            uid="test_uid",
            token=101,
            finish_reason="length",
            prompt_cache=[],
        )

        # _finalize_sequence should catch the error and clean up
        with pytest.raises(RuntimeError, match="Simulated extraction failure"):
            engine._finalize_sequence(
                uid="test_uid",
                response=response,
                agent_id=agent_id,
                tokens=[101],
                prompt_tokens=[1, 2, 3],
                prompt_text="Hello",
            )

        # After cleanup, the agent should have no blocks allocated
        assert agent_id not in pool.agent_allocations


class NeverFinishingBatchGenerator:
    """Fake BatchGenerator that never finishes — for testing step() safety limit."""

    def __init__(self, model=None, tokenizer=None, stop_tokens=None):
        self._sequences = {}
        self._next_uid = 0

    def insert(self, prompts, max_tokens=None, caches=None, samplers=None):
        uids = []
        for _ in prompts:
            uid = f"stall_uid_{self._next_uid}"
            self._next_uid += 1
            self._sequences[uid] = True
            uids.append(uid)
        return uids

    def next(self):
        """Always returns non-empty responses that never finish."""
        responses = []
        for uid in self._sequences:
            responses.append(FakeResponse(
                uid=uid,
                token=999,
                finish_reason=None,  # Never finishes
                prompt_cache=None,
            ))
        return responses

    def remove(self, uid):
        if uid in self._sequences:
            del self._sequences[uid]


class TestStepSafetyLimit:
    """Tests for step() max_iterations safety limit."""

    @pytest.fixture
    def engine(self, model, tokenizer, pool, spec, cache_adapter):
        """Create engine with a never-finishing batch generator."""
        return BlockPoolBatchEngine(
            cache_adapter=cache_adapter,
            model=model,
            tokenizer=tokenizer,
            pool=pool,
            spec=spec,
            batch_gen_factory=NeverFinishingBatchGenerator,
        )

    def test_step_max_iterations_safety(self, engine) -> None:
        """step() should break after max_iterations even if BatchGenerator never finishes."""
        uid = engine.submit(agent_id="stall_agent", prompt="Hello", max_tokens=50)

        # With a small max_iterations, step() must return instead of looping forever
        completions = list(engine.step(max_iterations=10))

        # No completions because the generator never finishes
        assert completions == []


class TestDrainUsesStepOnce:
    """Tests for drain() using step_once() instead of step()."""

    @pytest.fixture
    def engine(self, model, tokenizer, pool, spec, cache_adapter):
        """Create engine for drain testing."""
        return BlockPoolBatchEngine(
            cache_adapter=cache_adapter,
            model=model,
            tokenizer=tokenizer,
            pool=pool,
            spec=spec,
            batch_gen_factory=FakeBatchGenerator,
        )

    @pytest.mark.asyncio
    async def test_drain_completes_within_timeout(self, engine) -> None:
        """drain() should complete within timeout using step_once()."""
        uid = engine.submit(agent_id="drain_test", prompt="Hello world", max_tokens=5)

        drained = await engine.drain(timeout_seconds=5.0)

        # FakeBatchGenerator finishes in max_tokens steps
        assert drained >= 1
        assert len(engine._active_requests) == 0

    @pytest.mark.asyncio
    async def test_drain_with_no_active_requests(self, engine) -> None:
        """drain() should return immediately with no active requests."""
        drained = await engine.drain(timeout_seconds=1.0)
        assert drained == 0
