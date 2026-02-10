# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Yakov Shkolnikov and contributors
"""Unit tests for Admin API endpoints."""

import os
import sys
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from fastapi import FastAPI, status
from fastapi.testclient import TestClient

# Mock MLX modules
sys.modules["mlx"] = MagicMock()
sys.modules["mlx.core"] = MagicMock()
sys.modules["mlx.utils"] = MagicMock()
sys.modules["mlx_lm"] = MagicMock()

from agent_memory.adapters.inbound.admin_api import (
    get_old_engine,
    get_orchestrator,
    get_registry,
    router,
    verify_admin_key,
)
from agent_memory.domain.value_objects import ModelCacheSpec


@pytest.fixture
def app():
    """Create FastAPI app with admin router."""
    app = FastAPI()
    app.include_router(router)

    # Initialize app.state.agent_memory (required for admin API)
    from types import SimpleNamespace

    app.state.agent_memory = SimpleNamespace()
    app.state.agent_memory.batch_engine = None  # Will be updated by swap

    return app


@pytest.fixture
def client(app):
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def mock_orchestrator():
    """Create mock ModelSwapOrchestrator."""
    mock = Mock()
    mock._registry = Mock()
    mock._registry.get_current_id.return_value = "old-model"
    mock.swap_model = AsyncMock(return_value=Mock())  # Async swap returns engine
    return mock


@pytest.fixture
def mock_registry():
    """Create mock ModelRegistry."""
    mock = Mock()
    mock.get_current_id.return_value = "test-model"
    mock.get_current_spec.return_value = ModelCacheSpec(
        n_layers=24,
        n_kv_heads=8,
        head_dim=128,
        block_tokens=16,
        layer_types=["global"] * 24,
    )
    return mock


class TestAdminAuthentication:
    """Test admin key authentication."""

    def test_verify_admin_key_success(self):
        """Valid admin key allows access."""
        with patch.dict(os.environ, {"SEMANTIC_ADMIN_KEY": "secret123"}):
            # Should not raise
            verify_admin_key(x_admin_key="secret123")

    def test_verify_admin_key_missing_header(self):
        """Missing X-Admin-Key header returns 401."""
        with patch.dict(os.environ, {"SEMANTIC_ADMIN_KEY": "secret123"}):
            with pytest.raises(Exception) as exc_info:
                verify_admin_key(x_admin_key=None)
            assert "401" in str(exc_info.value)

    def test_verify_admin_key_invalid_key(self):
        """Invalid admin key returns 401."""
        with patch.dict(os.environ, {"SEMANTIC_ADMIN_KEY": "secret123"}):
            with pytest.raises(Exception) as exc_info:
                verify_admin_key(x_admin_key="wrong_key")
            assert "401" in str(exc_info.value)

    def test_verify_admin_key_not_configured(self):
        """Missing SEMANTIC_ADMIN_KEY env var returns 500."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(Exception) as exc_info:
                verify_admin_key(x_admin_key="any_key")
            assert "500" in str(exc_info.value)


class TestSwapModelEndpoint:
    """Test POST /admin/models/swap endpoint."""

    @patch.dict(os.environ, {"SEMANTIC_ADMIN_KEY": "admin123"})
    def test_swap_model_success(self, client, app, mock_orchestrator):
        """Successful swap returns 200 with details."""
        # Setup dependency override
        mock_new_engine = Mock()
        mock_orchestrator.swap_model.return_value = mock_new_engine
        app.dependency_overrides[get_orchestrator] = lambda: mock_orchestrator
        app.dependency_overrides[get_old_engine] = lambda: Mock()  # old_engine

        # Verify initial state
        assert app.state.agent_memory.batch_engine is None

        # Execute request
        response = client.post(
            "/admin/models/swap",
            json={"model_id": "new-model", "timeout_seconds": 30.0},
            headers={"X-Admin-Key": "admin123"},
        )

        # Verify response
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["status"] == "success"
        assert data["old_model_id"] == "old-model"
        assert data["new_model_id"] == "new-model"
        assert "successfully" in data["message"]

        # Verify orchestrator called
        mock_orchestrator.swap_model.assert_called_once()

        # CRITICAL: Verify app.state updated (CR-1 fix verification)
        assert app.state.agent_memory.batch_engine is mock_new_engine

    @patch.dict(os.environ, {"SEMANTIC_ADMIN_KEY": "admin123"})
    def test_swap_model_missing_auth(self, client, app):
        """Request without X-Admin-Key returns 401."""
        # Provide dependencies even though auth will fail first
        app.dependency_overrides[get_orchestrator] = lambda: Mock()
        app.dependency_overrides[get_old_engine] = lambda: Mock()

        response = client.post(
            "/admin/models/swap",
            json={"model_id": "new-model"},
        )

        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    @patch.dict(os.environ, {"SEMANTIC_ADMIN_KEY": "admin123"})
    def test_swap_model_invalid_auth(self, client, app):
        """Request with wrong admin key returns 401."""
        # Provide dependencies even though auth will fail first
        app.dependency_overrides[get_orchestrator] = lambda: Mock()
        app.dependency_overrides[get_old_engine] = lambda: Mock()

        response = client.post(
            "/admin/models/swap",
            json={"model_id": "new-model"},
            headers={"X-Admin-Key": "wrong_key"},
        )

        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    @patch.dict(os.environ, {"SEMANTIC_ADMIN_KEY": "admin123"})
    def test_swap_model_orchestrator_failure(self, client, app, mock_orchestrator):
        """Failed swap returns 500 with error details."""
        # Setup dependency override
        app.dependency_overrides[get_orchestrator] = lambda: mock_orchestrator
        app.dependency_overrides[get_old_engine] = lambda: Mock()

        # Simulate swap failure
        mock_orchestrator.swap_model.side_effect = Exception("Model not found")

        # Execute request
        response = client.post(
            "/admin/models/swap",
            json={"model_id": "nonexistent-model"},
            headers={"X-Admin-Key": "admin123"},
        )

        # Verify error response
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert "Model swap failed" in response.json()["detail"]

    @patch.dict(os.environ, {"SEMANTIC_ADMIN_KEY": "admin123"})
    def test_swap_model_custom_timeout(self, client, app, mock_orchestrator):
        """Custom timeout is passed to orchestrator."""
        app.dependency_overrides[get_orchestrator] = lambda: mock_orchestrator
        app.dependency_overrides[get_old_engine] = lambda: Mock()

        # Execute with custom timeout
        response = client.post(
            "/admin/models/swap",
            json={"model_id": "new-model", "timeout_seconds": 60.0},
            headers={"X-Admin-Key": "admin123"},
        )

        assert response.status_code == status.HTTP_200_OK

        # Verify timeout passed
        call_args = mock_orchestrator.swap_model.call_args
        assert call_args.kwargs["timeout_seconds"] == 60.0

    def test_swap_lock_exists(self):
        """Verify global swap lock exists for thread safety (CR-2)."""
        import asyncio

        from agent_memory.adapters.inbound.admin_api import _swap_lock

        # Verify lock exists and is an asyncio.Lock
        assert isinstance(_swap_lock, asyncio.Lock)

        # NOTE: Testing actual concurrent behavior in unit tests is complex.
        # The lock mechanism is verified to exist. Integration tests should
        # validate actual serialization behavior under load.


class TestGetCurrentModelEndpoint:
    """Test GET /admin/models/current endpoint."""

    @patch.dict(os.environ, {"SEMANTIC_ADMIN_KEY": "admin123"})
    def test_get_current_model_success(self, client, app, mock_registry):
        """Returns current model details."""
        app.dependency_overrides[get_registry] = lambda: mock_registry

        response = client.get(
            "/admin/models/current",
            headers={"X-Admin-Key": "admin123"},
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["model_id"] == "test-model"
        assert data["n_layers"] == 24
        assert data["n_kv_heads"] == 8
        assert data["head_dim"] == 128
        assert data["block_tokens"] == 16

    @patch.dict(os.environ, {"SEMANTIC_ADMIN_KEY": "admin123"})
    def test_get_current_model_no_model_loaded(self, client, app):
        """Returns null when no model loaded."""
        mock_registry = Mock()
        mock_registry.get_current_id.return_value = None
        mock_registry.get_current_spec.return_value = None
        app.dependency_overrides[get_registry] = lambda: mock_registry

        response = client.get(
            "/admin/models/current",
            headers={"X-Admin-Key": "admin123"},
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["model_id"] is None
        assert data["n_layers"] is None

    @patch.dict(os.environ, {"SEMANTIC_ADMIN_KEY": "admin123"})
    def test_get_current_model_missing_auth(self, client, app):
        """Request without auth returns 401."""
        # Provide dependencies even though auth will fail first
        app.dependency_overrides[get_registry] = lambda: Mock()

        response = client.get("/admin/models/current")

        assert response.status_code == status.HTTP_401_UNAUTHORIZED


class TestGetAvailableModelsEndpoint:
    """Test GET /admin/models/available endpoint."""

    @patch.dict(os.environ, {"SEMANTIC_ADMIN_KEY": "admin123"})
    def test_get_available_models_success(self, client):
        """Returns list of supported models."""
        response = client.get(
            "/admin/models/available",
            headers={"X-Admin-Key": "admin123"},
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "models" in data
        assert isinstance(data["models"], list)
        assert len(data["models"]) > 0
        # Verify some expected models
        assert "mlx-community/Qwen2.5-14B-Instruct-4bit" in data["models"]
        assert "mlx-community/Llama-3.1-8B-Instruct-4bit" in data["models"]

    @patch.dict(os.environ, {"SEMANTIC_ADMIN_KEY": "admin123"})
    def test_get_available_models_missing_auth(self, client):
        """Request without auth returns 401."""
        response = client.get("/admin/models/available")

        assert response.status_code == status.HTTP_401_UNAUTHORIZED
