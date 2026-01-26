"""Unit tests for model swap rollback and error recovery."""

import sys
from unittest.mock import MagicMock, Mock

import pytest

# Mock MLX modules
sys.modules["mlx"] = MagicMock()
sys.modules["mlx.core"] = MagicMock()
sys.modules["mlx_lm"] = MagicMock()

from semantic.application.model_swap_orchestrator import ModelSwapOrchestrator
from semantic.domain.errors import ModelNotFoundError, PoolConfigurationError
from semantic.domain.value_objects import ModelCacheSpec


class TestSwapRollback:
    """Test rollback behavior when model swap fails."""

    def test_rollback_on_model_load_failure(self):
        """Failed model load triggers rollback to old model."""
        # Setup mocks
        mock_registry = Mock()
        mock_registry.get_current_id.return_value = "old-model"
        old_spec = ModelCacheSpec(
            n_layers=24, n_kv_heads=8, head_dim=128, block_tokens=16, layer_types=["global"] * 24
        )
        mock_registry.get_current_spec.return_value = old_spec
        mock_registry.is_loaded.return_value = False

        # Simulate load failure then successful rollback
        mock_registry.load_model.side_effect = [
            ModelNotFoundError("new-model not found"),  # First call fails
            (MagicMock(), MagicMock()),  # Rollback succeeds
        ]

        mock_pool = Mock()
        mock_cache_store = Mock()
        mock_cache_store.evict_all_to_disk.return_value = 3
        mock_cache_adapter = Mock()
        mock_old_engine = Mock()

        orchestrator = ModelSwapOrchestrator(
            model_registry=mock_registry,
            block_pool=mock_pool,
            cache_store=mock_cache_store,
            cache_adapter=mock_cache_adapter,
        )

        # Execute swap (should fail and rollback)
        with pytest.raises(ModelNotFoundError) as exc_info:
            orchestrator.swap_model(
                old_engine=mock_old_engine,
                new_model_id="new-model",
            )

        # Verify rollback attempted
        assert mock_registry.load_model.call_count == 2
        # First call: new-model (failed)
        # Second call: old-model (rollback)
        mock_registry.load_model.assert_any_call("old-model")

        # Verify pool reconfigured back to old spec
        mock_pool.reconfigure.assert_called_with(old_spec)

    def test_rollback_on_pool_reconfiguration_failure(self):
        """Failed pool reconfiguration triggers rollback."""
        mock_registry = Mock()
        mock_registry.get_current_id.return_value = "old-model"
        old_spec = ModelCacheSpec(
            n_layers=24, n_kv_heads=8, head_dim=128, block_tokens=16, layer_types=["global"] * 24
        )
        new_spec = ModelCacheSpec(
            n_layers=32, n_kv_heads=16, head_dim=128, block_tokens=16, layer_types=["global"] * 32
        )
        mock_registry.get_current_spec.side_effect = [old_spec, new_spec]
        mock_registry.load_model.side_effect = [
            (MagicMock(), MagicMock()),  # New model loads successfully
            (MagicMock(), MagicMock()),  # Rollback loads old model
        ]
        mock_registry.is_loaded.return_value = True

        mock_pool = Mock()
        # First reconfigure (new model) fails, second reconfigure (rollback) succeeds
        mock_pool.reconfigure.side_effect = [
            PoolConfigurationError("Active allocations exist"),  # Swap fails
            None,  # Rollback succeeds
        ]

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
        with pytest.raises(PoolConfigurationError):
            orchestrator.swap_model(
                old_engine=mock_old_engine,
                new_model_id="new-model",
            )

        # Verify rollback: unloaded new model, reloaded old model
        assert mock_registry.unload_model.call_count >= 1
        assert mock_pool.reconfigure.call_count == 2  # Failed attempt + rollback

    def test_rollback_failure_raises_critical_error(self):
        """Failed rollback raises original error."""
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

        # Execute swap (both swap and rollback should fail)
        with pytest.raises(ModelNotFoundError) as exc_info:
            orchestrator.swap_model(
                old_engine=mock_old_engine,
                new_model_id="new-model",
            )

        # Verify both attempts were made
        assert mock_registry.load_model.call_count == 2

    def test_no_rollback_on_first_model_load(self):
        """First model load (no old model) doesn't attempt rollback on failure."""
        mock_registry = Mock()
        mock_registry.get_current_id.return_value = None  # No old model
        mock_registry.get_current_spec.return_value = None
        mock_registry.load_model.side_effect = ModelNotFoundError("first-model not found")

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

        # Execute first load (should fail without rollback)
        with pytest.raises(ModelNotFoundError):
            orchestrator.swap_model(
                old_engine=None,
                new_model_id="first-model",
            )

        # Verify no rollback attempted (only one load_model call)
        assert mock_registry.load_model.call_count == 1

    def test_rollback_unloads_failed_new_model_before_reload(self):
        """Rollback unloads failed new model before reloading old model."""
        mock_registry = Mock()
        mock_registry.get_current_id.return_value = "old-model"
        old_spec = ModelCacheSpec(
            n_layers=24, n_kv_heads=8, head_dim=128, block_tokens=16, layer_types=["global"] * 24
        )
        new_spec = ModelCacheSpec(
            n_layers=32, n_kv_heads=16, head_dim=128, block_tokens=16, layer_types=["global"] * 32
        )
        mock_registry.get_current_spec.side_effect = [old_spec, new_spec]
        mock_registry.is_loaded.return_value = True  # New model is loaded but swap fails

        # New model loads, but subsequent step fails
        mock_registry.load_model.side_effect = [
            (MagicMock(), MagicMock()),  # New model loads
            (MagicMock(), MagicMock()),  # Rollback reloads old model
        ]

        mock_pool = Mock()
        mock_pool.reconfigure.side_effect = Exception("Reconfigure failed")

        mock_cache_store = Mock()
        mock_cache_store.evict_all_to_disk.return_value = 0
        mock_cache_store.update_model_tag.side_effect = [None, None]  # Both tag updates succeed
        mock_cache_adapter = Mock()
        mock_old_engine = Mock()

        orchestrator = ModelSwapOrchestrator(
            model_registry=mock_registry,
            block_pool=mock_pool,
            cache_store=mock_cache_store,
            cache_adapter=mock_cache_adapter,
        )

        # Execute swap (should fail after loading new model)
        with pytest.raises(Exception):
            orchestrator.swap_model(
                old_engine=mock_old_engine,
                new_model_id="new-model",
            )

        # Verify unload called during rollback
        assert mock_registry.unload_model.call_count >= 1

    def test_successful_swap_no_rollback(self):
        """Successful swap doesn't trigger rollback."""
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
        mock_cache_store.evict_all_to_disk.return_value = 0
        mock_cache_adapter = Mock()
        mock_old_engine = Mock()

        orchestrator = ModelSwapOrchestrator(
            model_registry=mock_registry,
            block_pool=mock_pool,
            cache_store=mock_cache_store,
            cache_adapter=mock_cache_adapter,
        )

        # Execute successful swap
        new_engine = orchestrator.swap_model(
            old_engine=mock_old_engine,
            new_model_id="new-model",
        )

        # Verify: only one load_model call (new model, no rollback)
        assert mock_registry.load_model.call_count == 1
        mock_registry.load_model.assert_called_once_with("new-model")

        # Verify new engine created
        assert new_engine is not None
