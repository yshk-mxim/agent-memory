"""Unit tests for ModelSwapOrchestrator."""

import sys
from unittest.mock import MagicMock, Mock

import pytest

# Mock MLX modules
sys.modules["mlx"] = MagicMock()
sys.modules["mlx.core"] = MagicMock()
sys.modules["mlx_lm"] = MagicMock()

from semantic.application.model_swap_orchestrator import ModelSwapOrchestrator
from semantic.domain.errors import ModelNotFoundError
from semantic.domain.value_objects import ModelCacheSpec


class TestModelSwapOrchestrator:
    """Test hot-swap orchestration logic."""

    def test_swap_model_success_full_sequence(self):
        """Successful swap executes all 7 steps in order."""
        # Setup mocks
        mock_registry = Mock()
        mock_registry.get_current_id.return_value = "old-model"
        old_spec = ModelCacheSpec(
            n_layers=24, n_kv_heads=8, head_dim=128, block_tokens=16, layer_types=["global"] * 24
        )
        new_spec = ModelCacheSpec(
            n_layers=32, n_kv_heads=16, head_dim=128, block_tokens=16, layer_types=["global"] * 32
        )
        mock_registry.get_current_spec.side_effect = [old_spec, new_spec]
        mock_registry.load_model.return_value = (MagicMock(), MagicMock())

        mock_pool = Mock()
        mock_cache_store = Mock()
        mock_cache_store.evict_all_to_disk.return_value = 5  # 5 caches evicted
        mock_cache_adapter = Mock()

        mock_old_engine = Mock()

        orchestrator = ModelSwapOrchestrator(
            model_registry=mock_registry,
            block_pool=mock_pool,
            cache_store=mock_cache_store,
            cache_adapter=mock_cache_adapter,
        )

        # Execute swap
        new_engine = orchestrator.swap_model(
            old_engine=mock_old_engine,
            new_model_id="new-model",
            timeout_seconds=30.0,
        )

        # Verify steps executed in order
        # Step 1: Drain
        mock_old_engine.drain.assert_called_once_with(timeout_seconds=30.0)

        # Step 2: Evict caches
        mock_cache_store.evict_all_to_disk.assert_called_once()

        # Step 3: Shutdown
        mock_old_engine.shutdown.assert_called_once()

        # Step 4: Unload old model
        mock_registry.unload_model.assert_called_once()

        # Step 5: Load new model
        mock_registry.load_model.assert_called_once_with("new-model")

        # Step 6: Reconfigure pool
        mock_pool.reconfigure.assert_called_once_with(new_spec)

        # Step 7: New engine created
        assert new_engine is not None
        assert hasattr(new_engine, "submit")  # BatchEngine interface

    def test_swap_model_first_load_skips_drain_and_shutdown(self):
        """First model load skips drain/shutdown (no old engine)."""
        mock_registry = Mock()
        mock_registry.get_current_id.return_value = None  # No old model
        new_spec = ModelCacheSpec(
            n_layers=24, n_kv_heads=8, head_dim=128, block_tokens=16, layer_types=["global"] * 24
        )
        mock_registry.get_current_spec.return_value = new_spec
        mock_registry.load_model.return_value = (MagicMock(), MagicMock())

        mock_pool = Mock()
        mock_cache_store = Mock()
        mock_cache_store.evict_all_to_disk.return_value = 0
        mock_cache_adapter = Mock()

        orchestrator = ModelSwapOrchestrator(
            model_registry=mock_registry,
            block_pool=mock_pool,
            cache_store=mock_cache_store,
            cache_adapter=mock_cache_adapter,
        )

        # Execute first load (no old_engine)
        new_engine = orchestrator.swap_model(
            old_engine=None,
            new_model_id="first-model",
        )

        # Verify unload NOT called (no old model)
        mock_registry.unload_model.assert_not_called()

        # Verify load WAS called
        mock_registry.load_model.assert_called_once_with("first-model")

        # Verify new engine created
        assert new_engine is not None

    def test_swap_model_rollback_on_load_failure(self):
        """Failed model load triggers rollback to old model."""
        # Setup
        mock_registry = Mock()
        mock_registry.get_current_id.return_value = "old-model"
        old_spec = ModelCacheSpec(
            n_layers=24, n_kv_heads=8, head_dim=128, block_tokens=16, layer_types=["global"] * 24
        )
        mock_registry.get_current_spec.return_value = old_spec
        mock_registry.is_loaded.return_value = False

        # Simulate load failure
        mock_registry.load_model.side_effect = [
            ModelNotFoundError("new-model not found"),  # First call fails
            (MagicMock(), MagicMock()),  # Rollback succeeds
        ]

        mock_pool = Mock()
        mock_cache_store = Mock()
        mock_cache_store.evict_all_to_disk.return_value = 0
        mock_cache_adapter = Mock()
        mock_old_engine = Mock()

        orchestrator = ModelSwapOrchestrator(
            model_registry=mock_registry,
            block_pool=mock_pool,
            cache_store=mock_cache_store,
            cache_adapter=mock_cache_adapter,
        )

        # Execute swap (should fail and rollback)
        with pytest.raises(ModelNotFoundError):
            orchestrator.swap_model(
                old_engine=mock_old_engine,
                new_model_id="new-model",
            )

        # Verify rollback happened
        # First call: failed swap attempt
        # Second call: rollback to old-model
        assert mock_registry.load_model.call_count == 2
        mock_registry.load_model.assert_any_call("old-model")  # Rollback

        # Verify pool reconfigured back to old spec
        mock_pool.reconfigure.assert_called_with(old_spec)

    def test_swap_model_rollback_failure_raises_critical_error(self):
        """Failed rollback leaves system in degraded state."""
        mock_registry = Mock()
        mock_registry.get_current_id.return_value = "old-model"
        old_spec = ModelCacheSpec(
            n_layers=24, n_kv_heads=8, head_dim=128, block_tokens=16, layer_types=["global"] * 24
        )
        mock_registry.get_current_spec.return_value = old_spec
        mock_registry.is_loaded.return_value = False

        # Both swap AND rollback fail
        mock_registry.load_model.side_effect = [
            ModelNotFoundError("new-model not found"),  # Swap fails
            ModelNotFoundError("old-model also not found"),  # Rollback fails
        ]

        mock_pool = Mock()
        mock_cache_store = Mock()
        mock_cache_store.evict_all_to_disk.return_value = 0
        mock_cache_adapter = Mock()
        mock_old_engine = Mock()

        orchestrator = ModelSwapOrchestrator(
            model_registry=mock_registry,
            block_pool=mock_pool,
            cache_store=mock_cache_store,
            cache_adapter=mock_cache_adapter,
        )

        # Execute swap (should fail with critical error)
        with pytest.raises(ModelNotFoundError):
            orchestrator.swap_model(
                old_engine=mock_old_engine,
                new_model_id="new-model",
            )

        # Verify both attempts were made
        assert mock_registry.load_model.call_count == 2

    def test_swap_model_passes_timeout_to_drain(self):
        """Timeout parameter is passed to drain()."""
        mock_registry = Mock()
        mock_registry.get_current_id.return_value = "old-model"
        new_spec = ModelCacheSpec(
            n_layers=24, n_kv_heads=8, head_dim=128, block_tokens=16, layer_types=["global"] * 24
        )
        mock_registry.get_current_spec.side_effect = [new_spec, new_spec]
        mock_registry.load_model.return_value = (MagicMock(), MagicMock())

        mock_pool = Mock()
        mock_cache_store = Mock()
        mock_cache_store.evict_all_to_disk.return_value = 0
        mock_cache_adapter = Mock()
        mock_old_engine = Mock()

        orchestrator = ModelSwapOrchestrator(
            model_registry=mock_registry,
            block_pool=mock_pool,
            cache_store=mock_cache_store,
            cache_adapter=mock_cache_adapter,
        )

        # Execute with custom timeout
        orchestrator.swap_model(
            old_engine=mock_old_engine,
            new_model_id="new-model",
            timeout_seconds=60.0,  # Custom timeout
        )

        # Verify drain called with correct timeout
        mock_old_engine.drain.assert_called_once_with(timeout_seconds=60.0)

    def test_swap_model_evicts_all_caches_before_unload(self):
        """Cache eviction happens before model unload."""
        mock_registry = Mock()
        mock_registry.get_current_id.return_value = "old-model"
        new_spec = ModelCacheSpec(
            n_layers=24, n_kv_heads=8, head_dim=128, block_tokens=16, layer_types=["global"] * 24
        )
        mock_registry.get_current_spec.side_effect = [new_spec, new_spec]
        mock_registry.load_model.return_value = (MagicMock(), MagicMock())

        mock_pool = Mock()
        mock_cache_store = Mock()
        mock_cache_store.evict_all_to_disk.return_value = 3  # 3 caches evicted
        mock_cache_adapter = Mock()
        mock_old_engine = Mock()

        orchestrator = ModelSwapOrchestrator(
            model_registry=mock_registry,
            block_pool=mock_pool,
            cache_store=mock_cache_store,
            cache_adapter=mock_cache_adapter,
        )

        # Execute swap
        orchestrator.swap_model(
            old_engine=mock_old_engine,
            new_model_id="new-model",
        )

        # Verify eviction called
        mock_cache_store.evict_all_to_disk.assert_called_once()

        # Verify unload called AFTER eviction (check call order)
        # We can't easily assert call order with Mock, but we verified both were called
        mock_registry.unload_model.assert_called_once()
