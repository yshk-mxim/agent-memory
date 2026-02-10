# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Yakov Shkolnikov and contributors
"""Unit tests for model swap rollback and error recovery."""

import sys
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest

# Mock MLX modules
sys.modules["mlx"] = MagicMock()
sys.modules["mlx.core"] = MagicMock()
sys.modules["mlx.utils"] = MagicMock()
sys.modules["mlx_lm"] = MagicMock()

from agent_memory.application.batch_engine import BlockPoolBatchEngine
from agent_memory.application.model_swap_orchestrator import ModelSwapOrchestrator
from agent_memory.domain.errors import ModelNotFoundError, PoolConfigurationError
from agent_memory.domain.value_objects import ModelCacheSpec


class TestSwapRollback:
    """Test rollback behavior when model swap fails."""

    async def test_rollback_on_model_load_failure(self):
        """Failed model load triggers rollback to old model."""
        mock_registry = Mock()
        mock_registry.get_current_id.return_value = "old-model"
        old_spec = ModelCacheSpec(
            n_layers=24, n_kv_heads=8, head_dim=128, block_tokens=16, layer_types=["global"] * 24
        )
        mock_registry.get_current_spec.return_value = old_spec
        mock_registry.is_loaded.return_value = False

        mock_registry.load_model.side_effect = [
            ModelNotFoundError("new-model not found"),
            (MagicMock(), MagicMock()),
        ]

        mock_pool = Mock()
        mock_cache_store = Mock()
        mock_cache_store.evict_all_to_disk.return_value = 3
        mock_cache_adapter = Mock()
        mock_old_engine = Mock()
        mock_old_engine.drain = AsyncMock()

        orchestrator = ModelSwapOrchestrator(
            model_registry=mock_registry,
            block_pool=mock_pool,
            cache_store=mock_cache_store,
            cache_adapter=mock_cache_adapter,
        )

        with pytest.raises(ModelNotFoundError):
            await orchestrator.swap_model(
                old_engine=mock_old_engine,
                new_model_id="new-model",
            )

        assert mock_registry.load_model.call_count == 2
        mock_registry.load_model.assert_any_call("old-model")
        mock_pool.reconfigure.assert_called_with(old_spec)

    async def test_rollback_on_pool_reconfiguration_failure(self):
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
            (MagicMock(), MagicMock()),
            (MagicMock(), MagicMock()),
        ]
        mock_registry.is_loaded.return_value = True

        mock_pool = Mock()
        mock_pool.reconfigure.side_effect = [
            PoolConfigurationError("Active allocations exist"),
            None,
        ]

        mock_cache_store = Mock()
        mock_cache_store.evict_all_to_disk.return_value = 0
        mock_cache_adapter = Mock()
        mock_old_engine = Mock()
        mock_old_engine.drain = AsyncMock()

        orchestrator = ModelSwapOrchestrator(
            model_registry=mock_registry,
            block_pool=mock_pool,
            cache_store=mock_cache_store,
            cache_adapter=mock_cache_adapter,
        )

        with pytest.raises(PoolConfigurationError):
            await orchestrator.swap_model(
                old_engine=mock_old_engine,
                new_model_id="new-model",
            )

        assert mock_registry.unload_model.call_count >= 1
        assert mock_pool.reconfigure.call_count == 2

    async def test_rollback_failure_raises_critical_error(self):
        """Failed rollback raises original error."""
        mock_registry = Mock()
        mock_registry.get_current_id.return_value = "old-model"
        old_spec = ModelCacheSpec(
            n_layers=24, n_kv_heads=8, head_dim=128, block_tokens=16, layer_types=["global"] * 24
        )
        mock_registry.get_current_spec.return_value = old_spec
        mock_registry.is_loaded.return_value = False

        mock_registry.load_model.side_effect = [
            ModelNotFoundError("new-model not found"),
            ModelNotFoundError("old-model also not found"),
        ]

        mock_pool = Mock()
        mock_cache_store = Mock()
        mock_cache_store.evict_all_to_disk.return_value = 0
        mock_cache_adapter = Mock()
        mock_old_engine = Mock()
        mock_old_engine.drain = AsyncMock()

        orchestrator = ModelSwapOrchestrator(
            model_registry=mock_registry,
            block_pool=mock_pool,
            cache_store=mock_cache_store,
            cache_adapter=mock_cache_adapter,
        )

        with pytest.raises(ModelNotFoundError):
            await orchestrator.swap_model(
                old_engine=mock_old_engine,
                new_model_id="new-model",
            )

        assert mock_registry.load_model.call_count == 2

    async def test_no_rollback_on_first_model_load(self):
        """First model load (no old model) doesn't attempt rollback on failure."""
        mock_registry = Mock()
        mock_registry.get_current_id.return_value = None
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

        with pytest.raises(ModelNotFoundError):
            await orchestrator.swap_model(
                old_engine=None,
                new_model_id="first-model",
            )

        assert mock_registry.load_model.call_count == 1

    async def test_rollback_unloads_failed_new_model_before_reload(self):
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
        mock_registry.is_loaded.return_value = True

        mock_registry.load_model.side_effect = [
            (MagicMock(), MagicMock()),
            (MagicMock(), MagicMock()),
        ]

        mock_pool = Mock()
        mock_pool.reconfigure.side_effect = Exception("Reconfigure failed")

        mock_cache_store = Mock()
        mock_cache_store.evict_all_to_disk.return_value = 0
        mock_cache_store.update_model_tag.side_effect = [None, None]
        mock_cache_adapter = Mock()
        mock_old_engine = Mock()
        mock_old_engine.drain = AsyncMock()

        orchestrator = ModelSwapOrchestrator(
            model_registry=mock_registry,
            block_pool=mock_pool,
            cache_store=mock_cache_store,
            cache_adapter=mock_cache_adapter,
        )

        with pytest.raises(Exception):
            await orchestrator.swap_model(
                old_engine=mock_old_engine,
                new_model_id="new-model",
            )

        assert mock_registry.unload_model.call_count >= 1

    async def test_successful_swap_no_rollback(self):
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
        mock_old_engine.drain = AsyncMock()

        orchestrator = ModelSwapOrchestrator(
            model_registry=mock_registry,
            block_pool=mock_pool,
            cache_store=mock_cache_store,
            cache_adapter=mock_cache_adapter,
        )

        new_engine = await orchestrator.swap_model(
            old_engine=mock_old_engine,
            new_model_id="new-model",
        )

        assert mock_registry.load_model.call_count == 1
        mock_registry.load_model.assert_called_once_with("new-model")
        assert isinstance(new_engine, BlockPoolBatchEngine)
