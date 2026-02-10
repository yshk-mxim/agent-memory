# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Yakov Shkolnikov and contributors
"""Integration tests for 3-tier health endpoints.

Tests verify Kubernetes-compatible health endpoints:
- /health/live - Liveness probe (always 200)
- /health/ready - Readiness probe (503 when not ready)
- /health/startup - Startup probe (503 during initialization)
"""

import pytest
from fastapi.testclient import TestClient

from agent_memory.application.batch_engine import BlockPoolBatchEngine
from agent_memory.domain.services import BlockPool
from agent_memory.domain.value_objects import ModelCacheSpec
from agent_memory.entrypoints.api_server import create_app


@pytest.fixture
def test_app():
    """Create test FastAPI app without full MLX initialization."""
    app = create_app()

    # Initialize minimal state for testing (without MLX)
    class MockAppState:
        def __init__(self):
            self.shutting_down = False
            self.agent_memory = type(
                "obj",
                (object,),
                {
                    "block_pool": None,
                    "batch_engine": None,
                    "cache_store": None,
                },
            )()

    app.state = MockAppState()

    return app


@pytest.mark.integration
def test_health_live_always_200(test_app):
    """Test that /health/live always returns 200.

    Liveness probe should ALWAYS return 200 as long as process is running.
    It should never return 503, even during shutdown.
    """
    client = TestClient(test_app)

    # Normal operation
    response = client.get("/health/live")
    assert response.status_code == 200
    assert response.json() == {"status": "alive"}

    # During shutdown
    test_app.state.shutting_down = True
    response = client.get("/health/live")
    assert response.status_code == 200
    assert response.json() == {"status": "alive"}

    # Even without pool
    test_app.state.agent_memory.block_pool = None
    response = client.get("/health/live")
    assert response.status_code == 200
    assert response.json() == {"status": "alive"}

    print("\n✅ /health/live always returns 200")


@pytest.mark.integration
def test_health_ready_503_when_pool_exhausted(test_app):
    """Test that /health/ready returns 503 when pool is >90% utilized."""
    client = TestClient(test_app)

    # Create a pool with 10 blocks
    spec = ModelCacheSpec(
        n_layers=2,
        n_kv_heads=8,
        head_dim=128,
        block_tokens=256,
        layer_types=["global"] * 2,
        sliding_window_size=0,
    )
    pool = BlockPool(spec=spec, total_blocks=10)
    test_app.state.agent_memory.block_pool = pool

    # Pool mostly empty - should be ready
    response = client.get("/health/ready")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ready"
    assert "pool_utilization" in data

    # Allocate 9 blocks (90% utilization) - should be ready
    allocated = pool.allocate(n_blocks=9, layer_id=0, agent_id="test")
    response = client.get("/health/ready")
    assert response.status_code == 200

    # Allocate 1 more block (100% utilization) - should be not ready
    allocated2 = pool.allocate(n_blocks=1, layer_id=0, agent_id="test2")
    response = client.get("/health/ready")
    assert response.status_code == 503
    data = response.json()
    assert data["status"] == "not_ready"
    assert data["reason"] == "pool_near_exhaustion"
    assert data["pool_utilization"] > 90

    # Free blocks - should be ready again
    pool.free(allocated, "test")
    pool.free(allocated2, "test2")
    response = client.get("/health/ready")
    assert response.status_code == 200

    print("\n✅ /health/ready returns 503 when pool exhausted")


@pytest.mark.integration
def test_health_ready_503_when_shutting_down(test_app):
    """Test that /health/ready returns 503 when server is shutting down."""
    client = TestClient(test_app)

    # Create a pool
    spec = ModelCacheSpec(
        n_layers=2,
        n_kv_heads=8,
        head_dim=128,
        block_tokens=256,
        layer_types=["global"] * 2,
        sliding_window_size=0,
    )
    pool = BlockPool(spec=spec, total_blocks=100)
    test_app.state.agent_memory.block_pool = pool

    # Normal operation - should be ready
    test_app.state.shutting_down = False
    response = client.get("/health/ready")
    assert response.status_code == 200
    assert response.json()["status"] == "ready"

    # Shutdown flag set - should be not ready
    test_app.state.shutting_down = True
    response = client.get("/health/ready")
    assert response.status_code == 503
    data = response.json()
    assert data["status"] == "not_ready"
    assert data["reason"] == "shutting_down"

    print("\n✅ /health/ready returns 503 when shutting down")


@pytest.mark.integration
def test_health_startup_503_until_model_loaded(test_app):
    """Test that /health/startup returns 503 until model is loaded."""
    client = TestClient(test_app)

    # No batch engine - still starting
    test_app.state.agent_memory.batch_engine = None
    response = client.get("/health/startup")
    assert response.status_code == 503
    data = response.json()
    assert data["status"] == "starting"
    assert data["reason"] == "model_loading"

    # Create a mock batch engine
    spec = ModelCacheSpec(
        n_layers=2,
        n_kv_heads=8,
        head_dim=128,
        block_tokens=256,
        layer_types=["global"] * 2,
        sliding_window_size=0,
    )
    pool = BlockPool(spec=spec, total_blocks=100)

    def fake_batch_gen_factory(model, tokenizer):
        class FakeBatchGen:
            def insert(self, prompts, max_tokens, caches=None, samplers=None):
                return ["fake-uid"]

            def next(self):
                return []

        return FakeBatchGen()

    batch_engine = BlockPoolBatchEngine(
        model={"fake": "model"},
        tokenizer={"encode": lambda x: [1, 2, 3], "eos_token_id": 0},
        pool=pool,
        spec=spec,
        cache_adapter=type(
            "obj",
            (object,),
            {
                "create_batch_generator": lambda *args, **kwargs: None,
                "create_sampler": lambda *args, **kwargs: None,
            },
        )(),
        batch_gen_factory=fake_batch_gen_factory,
    )

    test_app.state.agent_memory.batch_engine = batch_engine

    # Batch engine exists - started
    response = client.get("/health/startup")
    assert response.status_code == 200
    assert response.json() == {"status": "started"}

    print("\n✅ /health/startup returns 503 until model loaded")
