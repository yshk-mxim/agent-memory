# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Yakov Shkolnikov and contributors
"""Unit tests for ModelSwapOrchestrator."""

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
from agent_memory.domain.errors import ModelNotFoundError
from agent_memory.domain.value_objects import ModelCacheSpec


class TestModelSwapOrchestrator:
    """Test hot-swap orchestration logic."""

    async def test_swap_model_success_full_sequence(self):
        """Successful swap executes all steps in order."""
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
        mock_cache_store.evict_all_to_disk.return_value = 5
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
            timeout_seconds=30.0,
        )

        mock_old_engine.drain.assert_awaited_once_with(timeout_seconds=30.0)
        mock_cache_store.evict_all_to_disk.assert_called_once()
        mock_old_engine.shutdown.assert_called_once()
        mock_registry.unload_model.assert_called_once()
        mock_registry.load_model.assert_called_once_with("new-model")
        mock_pool.reconfigure.assert_called_once_with(new_spec)
        assert isinstance(new_engine, BlockPoolBatchEngine)

    async def test_swap_model_first_load_skips_drain_and_shutdown(self):
        """First model load skips drain/shutdown (no old engine)."""
        mock_registry = Mock()
        mock_registry.get_current_id.return_value = None
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

        new_engine = await orchestrator.swap_model(
            old_engine=None,
            new_model_id="first-model",
        )

        mock_registry.unload_model.assert_not_called()
        mock_registry.load_model.assert_called_once_with("first-model")
        assert isinstance(new_engine, BlockPoolBatchEngine)

    async def test_swap_model_rollback_on_load_failure(self):
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
        mock_registry.load_model.assert_any_call("old-model")
        mock_pool.reconfigure.assert_called_with(old_spec)

    async def test_swap_model_rollback_failure_raises_critical_error(self):
        """Failed rollback leaves system in degraded state."""
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

    async def test_swap_model_passes_timeout_to_drain(self):
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
        mock_old_engine.drain = AsyncMock()

        orchestrator = ModelSwapOrchestrator(
            model_registry=mock_registry,
            block_pool=mock_pool,
            cache_store=mock_cache_store,
            cache_adapter=mock_cache_adapter,
        )

        await orchestrator.swap_model(
            old_engine=mock_old_engine,
            new_model_id="new-model",
            timeout_seconds=60.0,
        )

        mock_old_engine.drain.assert_awaited_once_with(timeout_seconds=60.0)

    async def test_swap_model_evicts_all_caches_before_unload(self):
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

        await orchestrator.swap_model(
            old_engine=mock_old_engine,
            new_model_id="new-model",
        )

        mock_cache_store.evict_all_to_disk.assert_called_once()
        mock_registry.unload_model.assert_called_once()
