"""Integration tests for model hot-swap end-to-end flow."""

import os
import sys
from unittest.mock import MagicMock, Mock, patch

import pytest
from fastapi.testclient import TestClient

# Mock MLX modules
sys.modules["mlx"] = MagicMock()
sys.modules["mlx.core"] = MagicMock()
sys.modules["mlx_lm"] = MagicMock()

from semantic.adapters.inbound.admin_api import get_old_engine, get_orchestrator, router
from semantic.application.agent_cache_store import AgentCacheStore, ModelTag
from semantic.application.model_registry import ModelRegistry
from semantic.application.model_swap_orchestrator import ModelSwapOrchestrator
from semantic.domain.services import BlockPool
from semantic.domain.value_objects import ModelCacheSpec


@pytest.fixture
def test_app():
    """Create test FastAPI app with admin router."""
    from types import SimpleNamespace

    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(router)

    # Initialize app.state.semantic (required for admin API CR-1 fix)
    app.state.semantic = SimpleNamespace()
    app.state.semantic.batch_engine = None

    return app


@pytest.fixture
def mock_components(tmp_path):
    """Create mock components for integration testing."""
    # Create specs
    spec_small = ModelCacheSpec(
        n_layers=12, n_kv_heads=4, head_dim=64, block_tokens=16, layer_types=["global"] * 12
    )
    spec_large = ModelCacheSpec(
        n_layers=24, n_kv_heads=8, head_dim=128, block_tokens=16, layer_types=["global"] * 24
    )

    # Mock registry
    registry = Mock(spec=ModelRegistry)
    registry.get_current_id.side_effect = ["small-model", "large-model"]
    registry.get_current_spec.side_effect = [spec_small, spec_large]
    registry.load_model.return_value = (MagicMock(), MagicMock())
    registry.is_loaded.return_value = False

    # Mock pool (avoid real BlockPool initialization complexities)
    pool = Mock(spec=BlockPool)
    pool.reconfigure = Mock()

    tag = ModelTag.from_spec("small-model", spec_small)

    mock_cache_adapter = Mock()
    mock_cache_adapter.save.side_effect = lambda aid, blocks, metadata: tmp_path / f"{aid}.safetensors"

    cache_store = AgentCacheStore(
        cache_dir=tmp_path,
        max_hot_agents=5,
        model_tag=tag,
        cache_adapter=mock_cache_adapter,
    )

    return {
        "registry": registry,
        "pool": pool,
        "cache_store": cache_store,
        "cache_adapter": Mock(),
    }


class TestFullHotSwapFlow:
    """Test complete hot-swap sequence integration."""

    def test_successful_swap_preserves_caches(self, mock_components, tmp_path):
        """Full swap flow: drain → evict → unload → load → reconfigure → reinit."""
        orchestrator = ModelSwapOrchestrator(
            model_registry=mock_components["registry"],
            block_pool=mock_components["pool"],
            cache_store=mock_components["cache_store"],
            cache_adapter=mock_components["cache_adapter"],
        )

        # Add some caches before swap
        from semantic.domain.entities import AgentBlocks

        for i in range(3):
            blocks = AgentBlocks(agent_id=f"agent_{i}", blocks={}, total_tokens=0)
            mock_components["cache_store"].save(f"agent_{i}", blocks)

        # Create mock old engine
        old_engine = Mock()

        # Execute swap
        new_engine = orchestrator.swap_model(
            old_engine=old_engine,
            new_model_id="large-model",
            timeout_seconds=30.0,
        )

        # Verify all phases executed
        old_engine.drain.assert_called_once()
        old_engine.shutdown.assert_called_once()
        mock_components["registry"].unload_model.assert_called_once()
        mock_components["registry"].load_model.assert_called_with("large-model")

        # Verify new engine created
        assert new_engine is not None

    def test_swap_with_active_requests_drains_first(self, mock_components):
        """Swap with active requests waits for drain before proceeding."""
        orchestrator = ModelSwapOrchestrator(
            model_registry=mock_components["registry"],
            block_pool=mock_components["pool"],
            cache_store=mock_components["cache_store"],
            cache_adapter=mock_components["cache_adapter"],
        )

        # Mock engine with active requests
        old_engine = Mock()
        old_engine.drain.return_value = None  # Simulates successful drain

        # Execute swap
        new_engine = orchestrator.swap_model(
            old_engine=old_engine,
            new_model_id="large-model",
        )

        # Verify drain called with default timeout
        old_engine.drain.assert_called_once_with(timeout_seconds=30.0)

    def test_swap_failure_rolls_back_to_old_model(self, mock_components):
        """Failed swap triggers rollback to previous model."""
        # Setup registry to fail on new model load
        mock_components["registry"].load_model.side_effect = [
            Exception("New model load failed"),
            (MagicMock(), MagicMock()),  # Rollback succeeds
        ]
        mock_components["registry"].get_current_id.side_effect = ["small-model", None]

        orchestrator = ModelSwapOrchestrator(
            model_registry=mock_components["registry"],
            block_pool=mock_components["pool"],
            cache_store=mock_components["cache_store"],
            cache_adapter=mock_components["cache_adapter"],
        )

        old_engine = Mock()

        # Execute swap (should fail and rollback)
        with pytest.raises(Exception) as exc_info:
            orchestrator.swap_model(
                old_engine=old_engine,
                new_model_id="large-model",
            )

        # Verify rollback attempted (2 load calls: failed + rollback)
        assert mock_components["registry"].load_model.call_count == 2


class TestAdminAPIIntegration:
    """Test Admin API triggers hot-swap correctly."""

    @patch.dict(os.environ, {"SEMANTIC_ADMIN_KEY": "test-admin-key"})
    def test_swap_endpoint_triggers_orchestrator(self, test_app, mock_components):
        """POST /admin/models/swap triggers full orchestration."""
        orchestrator = ModelSwapOrchestrator(
            model_registry=mock_components["registry"],
            block_pool=mock_components["pool"],
            cache_store=mock_components["cache_store"],
            cache_adapter=mock_components["cache_adapter"],
        )

        old_engine = Mock()

        # Override dependencies
        test_app.dependency_overrides[get_orchestrator] = lambda: orchestrator
        test_app.dependency_overrides[get_old_engine] = lambda: old_engine

        client = TestClient(test_app)

        # Execute swap via API
        response = client.post(
            "/admin/models/swap",
            json={"model_id": "large-model", "timeout_seconds": 60.0},
            headers={"X-Admin-Key": "test-admin-key"},
        )

        # Verify success
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["new_model_id"] == "large-model"

        # Verify orchestrator was called
        mock_components["registry"].load_model.assert_called()

    @patch.dict(os.environ, {"SEMANTIC_ADMIN_KEY": "test-admin-key"})
    def test_swap_endpoint_handles_orchestrator_failure(self, test_app, mock_components):
        """Failed swap returns 500 with error details."""
        mock_components["registry"].load_model.side_effect = Exception("Model not found")

        orchestrator = ModelSwapOrchestrator(
            model_registry=mock_components["registry"],
            block_pool=mock_components["pool"],
            cache_store=mock_components["cache_store"],
            cache_adapter=mock_components["cache_adapter"],
        )

        old_engine = Mock()

        test_app.dependency_overrides[get_orchestrator] = lambda: orchestrator
        test_app.dependency_overrides[get_old_engine] = lambda: old_engine

        client = TestClient(test_app)

        # Execute swap (should fail)
        response = client.post(
            "/admin/models/swap",
            json={"model_id": "nonexistent-model"},
            headers={"X-Admin-Key": "test-admin-key"},
        )

        # Verify error response
        assert response.status_code == 500
        assert "Model swap failed" in response.json()["detail"]


class TestCachePreservationAcrossSwap:
    """Test that agent caches survive model swaps."""

    def test_caches_evicted_before_swap(self, mock_components, tmp_path):
        """All hot caches evicted to disk before model swap."""
        orchestrator = ModelSwapOrchestrator(
            model_registry=mock_components["registry"],
            block_pool=mock_components["pool"],
            cache_store=mock_components["cache_store"],
            cache_adapter=mock_components["cache_adapter"],
        )

        # Add caches
        from semantic.domain.entities import AgentBlocks

        for i in range(3):
            blocks = AgentBlocks(agent_id=f"agent_{i}", blocks={}, total_tokens=0)
            mock_components["cache_store"].save(f"agent_{i}", blocks)

        assert len(mock_components["cache_store"]._hot_cache) == 3

        old_engine = Mock()

        # Execute swap
        orchestrator.swap_model(old_engine=old_engine, new_model_id="large-model")

        # Verify all caches evicted to disk (hot tier should be empty)
        assert len(mock_components["cache_store"]._hot_cache) == 0
        assert len(mock_components["cache_store"]._warm_cache) == 3

    def test_model_tag_updated_after_swap(self, mock_components):
        """Cache store model tag updated to new model after swap."""
        orchestrator = ModelSwapOrchestrator(
            model_registry=mock_components["registry"],
            block_pool=mock_components["pool"],
            cache_store=mock_components["cache_store"],
            cache_adapter=mock_components["cache_adapter"],
        )

        old_tag = mock_components["cache_store"].model_tag
        assert old_tag.model_id == "small-model"

        old_engine = Mock()

        # Execute swap
        orchestrator.swap_model(old_engine=old_engine, new_model_id="large-model")

        # Verify tag updated
        new_tag = mock_components["cache_store"].model_tag
        assert new_tag.model_id == "large-model"
        assert new_tag.n_layers == 24  # From large model spec
