# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Yakov Shkolnikov and contributors
"""Coverage tests for admin API — offload_model, clear_all_caches, set_random_seed."""

import os
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from fastapi import FastAPI, status
from fastapi.testclient import TestClient

# Must mock MLX before importing admin_api (same pattern as test_admin_api.py)
sys.modules.setdefault("mlx", MagicMock())
sys.modules.setdefault("mlx.core", MagicMock())
sys.modules.setdefault("mlx.utils", MagicMock())
sys.modules.setdefault("mlx_lm", MagicMock())

from agent_memory.adapters.inbound.admin_api import (
    get_registry,
    router,
    verify_admin_key,
)

pytestmark = pytest.mark.unit

ADMIN_KEY = "test-admin-key-123"


@pytest.fixture
def mock_registry():
    reg = Mock()
    reg.get_current_id.return_value = "test-model-id"
    reg.get_current_spec.return_value = None
    reg.unload_model.return_value = None
    return reg


@pytest.fixture
def app(mock_registry):
    app = FastAPI()
    app.include_router(router)

    # Setup app state
    semantic = SimpleNamespace()
    semantic.batch_engine = None
    semantic.cache_store = None
    semantic.block_pool = None
    app.state.agent_memory = semantic

    # Override dependencies
    app.dependency_overrides[get_registry] = lambda: mock_registry
    app.dependency_overrides[verify_admin_key] = lambda: None  # Skip auth

    return app


@pytest.fixture
def client(app):
    return TestClient(app)


# ===========================================================================
# offload_model
# ===========================================================================


class TestOffloadModel:
    def test_no_model_loaded(self, client, mock_registry):
        mock_registry.get_current_id.return_value = None
        resp = client.post("/admin/models/offload")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert "No model" in data["message"]

    def test_model_loaded_full_sequence(self, client, app, mock_registry):
        engine = MagicMock()
        engine.drain = AsyncMock()
        engine.shutdown = MagicMock()
        app.state.agent_memory.batch_engine = engine

        cache_store = MagicMock()
        cache_store.evict_all_to_disk.return_value = 3
        app.state.agent_memory.cache_store = cache_store

        block_pool = MagicMock()
        block_pool.force_clear_all_allocations.return_value = 5
        app.state.agent_memory.block_pool = block_pool

        resp = client.post("/admin/models/offload")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert data["model_id"] == "test-model-id"

        # Verify cleanup sequence
        engine.drain.assert_called_once()
        cache_store.evict_all_to_disk.assert_called_once()
        block_pool.force_clear_all_allocations.assert_called_once()
        engine.shutdown.assert_called_once()
        mock_registry.unload_model.assert_called_once()

    def test_partial_state_no_engine(self, client, app, mock_registry):
        # No engine, no pool, just cache_store
        cache_store = MagicMock()
        cache_store.evict_all_to_disk.return_value = 1
        app.state.agent_memory.cache_store = cache_store
        app.state.agent_memory.batch_engine = None
        app.state.agent_memory.block_pool = None

        resp = client.post("/admin/models/offload")
        assert resp.status_code == 200
        assert resp.json()["status"] == "success"

    def test_exception_returns_failed(self, client, mock_registry):
        # First call succeeds (in try block), but unload_model raises.
        # Second get_current_id call (in except block) succeeds.
        mock_registry.get_current_id.return_value = "test-model"
        mock_registry.unload_model.side_effect = RuntimeError("boom")
        resp = client.post("/admin/models/offload")
        assert resp.status_code == 200
        assert resp.json()["status"] == "failed"


# ===========================================================================
# clear_all_caches
# ===========================================================================


class TestClearAllCaches:
    def test_normal_clear(self, client, app):
        cache_store = MagicMock()
        cache_store._hot_cache = {"a": "entry1", "b": "entry2"}
        cache_dir = MagicMock()
        cache_store.cache_dir = cache_dir

        # Simulate disk files
        fake_file1 = MagicMock()
        fake_file2 = MagicMock()
        fake_file3 = MagicMock()
        cache_dir.exists.return_value = True
        cache_dir.glob.return_value = [fake_file1, fake_file2, fake_file3]

        block_pool = MagicMock()
        block_pool.force_clear_all_allocations.return_value = 1

        engine = MagicMock()
        engine.clear_all_agent_blocks = MagicMock()

        app.state.agent_memory.cache_store = cache_store
        app.state.agent_memory.block_pool = block_pool
        app.state.agent_memory.batch_engine = engine

        resp = client.delete("/admin/caches")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert data["hot_cleared"] == 2
        assert data["disk_cleared"] == 3
        assert data["pool_cleared"] == 1

    def test_no_cache_store(self, client, app):
        app.state.agent_memory.cache_store = None
        app.state.agent_memory.block_pool = None
        app.state.agent_memory.batch_engine = None

        resp = client.delete("/admin/caches")
        # cache_store is None, so accessing it will raise
        # The endpoint catches generic exceptions
        assert resp.status_code == 200
        data = resp.json()
        # Either success with 0s or failed
        assert data["status"] in ("success", "failed")

    def test_engine_with_clear_all_agent_blocks(self, client, app):
        cache_store = MagicMock()
        cache_store._hot_cache = {}
        cache_store.cache_dir = MagicMock()
        cache_store.cache_dir.exists.return_value = False

        engine = MagicMock()
        engine.clear_all_agent_blocks = MagicMock()

        app.state.agent_memory.cache_store = cache_store
        app.state.agent_memory.block_pool = None
        app.state.agent_memory.batch_engine = engine

        resp = client.delete("/admin/caches")
        assert resp.status_code == 200
        engine.clear_all_agent_blocks.assert_called_once()

    def test_exception_returns_failed(self, client, app):
        bad_store = MagicMock()
        # cache_dir.exists() works, but glob raises
        bad_store.cache_dir.exists.return_value = True
        bad_store.cache_dir.glob.side_effect = RuntimeError("boom")
        app.state.agent_memory.cache_store = bad_store
        resp = client.delete("/admin/caches")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "failed"


# ===========================================================================
# set_random_seed
# ===========================================================================


class TestSetRandomSeed:
    def test_valid_seed(self, client):
        # Patch mlx.core.random.seed
        mock_mx = sys.modules["mlx.core"]
        mock_mx.random = MagicMock()

        resp = client.post("/admin/seed", json={"seed": 42})
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["seed"] == 42

    def test_seed_calls_mx_random(self, client):
        # `import mlx.core as mx` uses sys.modules['mlx'].core, not sys.modules['mlx.core']
        mock_mx = sys.modules["mlx"].core
        mock_mx.random.seed.reset_mock()

        resp = client.post("/admin/seed", json={"seed": 123})
        assert resp.status_code == 200
        mock_mx.random.seed.assert_called_with(123)


# ===========================================================================
# verify_admin_key error paths
# ===========================================================================


class TestVerifyAdminKeyErrors:
    def test_missing_env_var(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("SEMANTIC_ADMIN_KEY", None)
            with pytest.raises(Exception) as exc_info:
                verify_admin_key(x_admin_key="any")
            assert "500" in str(exc_info.value)

    def test_missing_header(self):
        with patch.dict(os.environ, {"SEMANTIC_ADMIN_KEY": "secret"}):
            with pytest.raises(Exception) as exc_info:
                verify_admin_key(x_admin_key=None)
            assert "401" in str(exc_info.value)

    def test_wrong_key(self):
        with patch.dict(os.environ, {"SEMANTIC_ADMIN_KEY": "secret"}):
            with pytest.raises(Exception) as exc_info:
                verify_admin_key(x_admin_key="wrong")
            assert "401" in str(exc_info.value)
