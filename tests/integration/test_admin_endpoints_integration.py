# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Yakov Shkolnikov and contributors
"""Integration tests for admin API endpoints with real cache store."""

import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from agent_memory.adapters.inbound.admin_api import (
    get_registry,
    router,
    verify_admin_key,
)


@pytest.fixture
def mock_registry():
    reg = Mock()
    reg.get_current_id.return_value = "test-model"
    reg.get_current_spec.return_value = None
    reg.unload_model.return_value = None
    return reg


@pytest.fixture
def app_with_cache(tmp_path, mock_registry):
    """Create app with real AgentCacheStore pointed at tmp_path."""
    from agent_memory.application.agent_cache_store import AgentCacheStore, ModelTag

    tag = ModelTag(
        model_id="test-model",
        n_layers=12,
        n_kv_heads=4,
        head_dim=128,
        block_tokens=256,
    )
    cache_store = AgentCacheStore(
        cache_dir=tmp_path,
        max_hot_agents=5,
        model_tag=tag,
    )

    app = FastAPI()
    app.include_router(router)

    semantic = SimpleNamespace()
    semantic.batch_engine = None
    semantic.cache_store = cache_store
    semantic.block_pool = None
    app.state.agent_memory = semantic

    app.dependency_overrides[get_registry] = lambda: mock_registry
    app.dependency_overrides[verify_admin_key] = lambda: None

    return app


@pytest.fixture
def client(app_with_cache):
    return TestClient(app_with_cache)


class TestOffloadIntegration:
    def test_offload_with_engine(self, client, app_with_cache, mock_registry):
        engine = MagicMock()
        engine.drain = AsyncMock()
        engine.shutdown = MagicMock()
        app_with_cache.state.agent_memory.batch_engine = engine

        resp = client.post("/admin/models/offload")
        assert resp.status_code == 200
        assert resp.json()["status"] == "success"
        engine.drain.assert_called_once()


class TestClearCachesIntegration:
    def test_clear_with_real_files(self, client, app_with_cache, tmp_path):
        # Create some fake safetensors files
        for i in range(3):
            (tmp_path / f"agent_{i}.safetensors").write_bytes(b"fake data")

        resp = client.delete("/admin/caches")
        assert resp.status_code == 200
        data = resp.json()
        assert data["disk_cleared"] == 3

        # Verify files are gone
        remaining = list(tmp_path.glob("*.safetensors"))
        assert len(remaining) == 0


class TestSeedIntegration:
    def test_seed_response(self, client):
        mock_mx = sys.modules.get("mlx.core") or MagicMock()
        mock_mx.random = MagicMock()
        sys.modules["mlx.core"] = mock_mx

        resp = client.post("/admin/seed", json={"seed": 7})
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok", "seed": 7}
