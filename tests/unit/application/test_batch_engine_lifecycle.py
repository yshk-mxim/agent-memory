"""Unit tests for BatchEngine drain and shutdown lifecycle methods."""

import sys
import time
from unittest.mock import MagicMock, Mock, patch

import pytest

# Mock MLX modules before imports
sys.modules["mlx"] = MagicMock()
sys.modules["mlx.core"] = MagicMock()
sys.modules["mlx.utils"] = MagicMock()
sys.modules["mlx_lm"] = MagicMock()

from agent_memory.application.batch_engine import BlockPoolBatchEngine
from agent_memory.domain.errors import GenerationError
from agent_memory.domain.value_objects import ModelCacheSpec


class TestBatchEngineDrain:
    """Test drain() method for graceful shutdown."""

    async def test_drain_with_no_active_requests(self):
        """Drain with no active requests returns immediately."""
        # Setup
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_pool = MagicMock()
        mock_spec = ModelCacheSpec(
            n_layers=24, n_kv_heads=8, head_dim=128, block_tokens=16, layer_types=["global"] * 24
        )
        mock_adapter = MagicMock()

        engine = BlockPoolBatchEngine(
            model=mock_model,
            tokenizer=mock_tokenizer,
            pool=mock_pool,
            spec=mock_spec,
            cache_adapter=mock_adapter,
        )

        # Execute - should return immediately
        start_time = time.time()
        await engine.drain(timeout_seconds=5.0)
        elapsed = time.time() - start_time

        # Verify - drain should be instant
        assert elapsed < 1.0  # Should finish in <1s

    @patch("agent_memory.application.batch_engine.time.sleep")
    async def test_drain_waits_for_active_requests_to_complete(self, mock_sleep):
        """Drain waits until all active requests finish."""
        # Setup
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.decode.return_value = "Generated text"
        mock_pool = MagicMock()
        mock_pool.free = MagicMock()
        mock_spec = ModelCacheSpec(
            n_layers=24, n_kv_heads=8, head_dim=128, block_tokens=16, layer_types=["global"] * 24
        )
        mock_adapter = MagicMock()

        engine = BlockPoolBatchEngine(
            model=mock_model,
            tokenizer=mock_tokenizer,
            pool=mock_pool,
            spec=mock_spec,
            cache_adapter=mock_adapter,
        )

        # Manually inject an active request to simulate in-flight work
        # Format: (agent_id, generated_tokens, detokenizer, prompt_tokens, prompt_text)
        engine._active_requests = {"uid_123": ("agent_1", [42, 43, 44], mock_tokenizer, [1, 2, 3], "test prompt")}

        # Create a mock batch generator that finishes the request on second next() call
        mock_batch_gen = MagicMock()

        # First call: request still running
        response_1 = Mock()
        response_1.uid = "uid_123"
        response_1.token = 45
        response_1.finish_reason = None  # Still running

        # Second call: request finishes
        response_2 = Mock()
        response_2.uid = "uid_123"
        response_2.token = 46
        response_2.finish_reason = "stop"  # Finished
        response_2.prompt_cache = []  # Empty cache for simplicity

        # Third call: empty (all done)
        mock_batch_gen.next.side_effect = [[response_1], [response_2], []]

        engine._batch_gen = mock_batch_gen

        # Execute drain
        await engine.drain(timeout_seconds=10.0)

        # Verify - requests should be cleared
        assert len(engine._active_requests) == 0
        assert mock_batch_gen.next.call_count >= 2  # At least 2 steps needed

    @patch("agent_memory.application.batch_engine.time.time")
    async def test_drain_raises_on_timeout(self, mock_time):
        """Drain raises GenerationError if timeout exceeded."""
        # Setup
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_pool = MagicMock()
        mock_spec = ModelCacheSpec(
            n_layers=24, n_kv_heads=8, head_dim=128, block_tokens=16, layer_types=["global"] * 24
        )
        mock_adapter = MagicMock()

        engine = BlockPoolBatchEngine(
            model=mock_model,
            tokenizer=mock_tokenizer,
            pool=mock_pool,
            spec=mock_spec,
            cache_adapter=mock_adapter,
        )

        # Inject stuck requests (agent_id, generated_tokens, detokenizer, prompt_tokens, prompt_text)
        engine._active_requests = {
            "uid_1": ("agent_1", [], mock_tokenizer, [], "prompt 1"),
            "uid_2": ("agent_2", [], mock_tokenizer, [], "prompt 2"),
            "uid_3": ("agent_3", [], mock_tokenizer, [], "prompt 3"),
        }

        # Mock batch generator that never completes
        mock_batch_gen = MagicMock()
        response = Mock()
        response.uid = "uid_1"
        response.token = 42
        response.finish_reason = None  # Never finishes
        mock_batch_gen.next.return_value = [response]
        engine._batch_gen = mock_batch_gen

        # Mock time to simulate timeout
        # Start at 0, then jump to 31s to trigger timeout
        # Provide enough values to cover all time.time() calls
        mock_time.side_effect = [0.0, 31.0, 31.0, 31.0, 31.0]

        # Execute and verify timeout
        with pytest.raises(GenerationError) as exc_info:
            await engine.drain(timeout_seconds=30.0)

        assert "Drain timeout after 30.0s" in str(exc_info.value)
        assert "Still pending" in str(exc_info.value)


class TestBatchEngineShutdown:
    """Test shutdown() method for resource cleanup."""

    def test_shutdown_clears_all_state(self):
        """Shutdown clears batch gen, requests, and references."""
        # Setup
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_pool = MagicMock()
        mock_spec = ModelCacheSpec(
            n_layers=24, n_kv_heads=8, head_dim=128, block_tokens=16, layer_types=["global"] * 24
        )
        mock_adapter = MagicMock()

        engine = BlockPoolBatchEngine(
            model=mock_model,
            tokenizer=mock_tokenizer,
            pool=mock_pool,
            spec=mock_spec,
            cache_adapter=mock_adapter,
        )

        # Populate state (agent_id, generated_tokens, detokenizer, prompt_tokens, prompt_text)
        engine._batch_gen = MagicMock()
        engine._active_requests = {"uid_1": ("agent_1", [], None, [], "")}
        engine._agent_blocks = {"agent_1": MagicMock()}

        # Execute shutdown
        engine.shutdown()

        # Verify all state cleared
        assert engine._batch_gen is None
        assert len(engine._active_requests) == 0
        assert len(engine._agent_blocks) == 0
        assert engine._model is None
        assert engine._tokenizer is None

    def test_shutdown_warns_if_active_requests_exist(self, caplog):
        """Shutdown warns if called before drain."""
        # Setup
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_pool = MagicMock()
        mock_spec = ModelCacheSpec(
            n_layers=24, n_kv_heads=8, head_dim=128, block_tokens=16, layer_types=["global"] * 24
        )
        mock_adapter = MagicMock()

        engine = BlockPoolBatchEngine(
            model=mock_model,
            tokenizer=mock_tokenizer,
            pool=mock_pool,
            spec=mock_spec,
            cache_adapter=mock_adapter,
        )

        # Add active requests (agent_id, generated_tokens, detokenizer, prompt_tokens, prompt_text)
        engine._active_requests = {
            "uid_1": ("agent_1", [], None, [], ""),
            "uid_2": ("agent_2", [], None, [], ""),
        }

        # Execute shutdown
        with caplog.at_level("WARNING"):
            engine.shutdown()

        # Verify warning logged
        assert "Shutdown with 2 active requests" in caplog.text
        assert "possible loss" in caplog.text

    def test_shutdown_is_idempotent(self):
        """Calling shutdown multiple times is safe."""
        # Setup
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_pool = MagicMock()
        mock_spec = ModelCacheSpec(
            n_layers=24, n_kv_heads=8, head_dim=128, block_tokens=16, layer_types=["global"] * 24
        )
        mock_adapter = MagicMock()

        engine = BlockPoolBatchEngine(
            model=mock_model,
            tokenizer=mock_tokenizer,
            pool=mock_pool,
            spec=mock_spec,
            cache_adapter=mock_adapter,
        )

        # Call shutdown multiple times
        engine.shutdown()
        engine.shutdown()
        engine.shutdown()

        # Should not raise or error
        assert engine._batch_gen is None


class TestDrainShutdownIntegration:
    """Test drain → shutdown sequence for model hot-swap."""

    @patch("agent_memory.application.batch_engine.time.sleep")
    async def test_drain_then_shutdown_clears_all_state(self, mock_sleep):
        """Typical hot-swap sequence: drain active requests then shutdown."""
        # Setup
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.decode.return_value = "text"
        mock_pool = MagicMock()
        mock_pool.free = MagicMock()
        mock_spec = ModelCacheSpec(
            n_layers=24, n_kv_heads=8, head_dim=128, block_tokens=16, layer_types=["global"] * 24
        )
        mock_adapter = MagicMock()

        engine = BlockPoolBatchEngine(
            model=mock_model,
            tokenizer=mock_tokenizer,
            pool=mock_pool,
            spec=mock_spec,
            cache_adapter=mock_adapter,
        )

        # Inject active request (agent_id, generated_tokens, detokenizer, prompt_tokens, prompt_text)
        engine._active_requests = {"uid_1": ("agent_1", [42], mock_tokenizer, [1], "test")}

        # Mock batch gen to finish request
        mock_batch_gen = MagicMock()
        response = Mock()
        response.uid = "uid_1"
        response.token = 43
        response.finish_reason = "stop"
        response.prompt_cache = []
        mock_batch_gen.next.side_effect = [[response], []]
        engine._batch_gen = mock_batch_gen

        # Execute drain → shutdown
        await engine.drain(timeout_seconds=10.0)
        engine.shutdown()

        # Verify clean state
        assert len(engine._active_requests) == 0
        assert engine._batch_gen is None
        assert engine._model is None
        assert engine._tokenizer is None
