# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Yakov Shkolnikov and contributors
"""Integration tests for Prometheus metrics.

Tests verify that metrics are:
- Exposed via /metrics endpoint in Prometheus format
- Automatically collected for requests
- Not self-tracking the /metrics endpoint
- Tracking pool utilization and active agents
"""

import pytest
from fastapi.testclient import TestClient

from agent_memory.domain.services import BlockPool
from agent_memory.domain.value_objects import ModelCacheSpec
from agent_memory.entrypoints.api_server import create_app


@pytest.fixture
def test_app():
    """Create test FastAPI app with metrics."""
    app = create_app()

    # Initialize minimal state
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
def test_metrics_endpoint_exists(test_app):
    """Test /metrics endpoint returns Prometheus format.

    Expected behavior:
    - Returns 200 OK
    - Content-Type is text/plain
    - Contains metric names and HELP/TYPE comments
    """
    client = TestClient(test_app)

    response = client.get("/metrics")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")

    # Should contain metric names
    content = response.text
    assert "semantic_request_total" in content
    assert "semantic_request_duration_seconds" in content
    assert "semantic_pool_utilization_ratio" in content
    assert "semantic_agents_active" in content
    assert "semantic_cache_hit_total" in content

    # Should have HELP and TYPE comments
    assert "# HELP" in content
    assert "# TYPE" in content

    print("\n✅ /metrics endpoint working")


@pytest.mark.integration
def test_request_counter_increments(test_app):
    """Test request_total counter increments on requests.

    Expected behavior:
    - Counter starts at 0
    - Increments with each request
    - Labels include method, path, status_code
    """
    client = TestClient(test_app)

    # Make a request
    response = client.get("/")
    assert response.status_code == 200

    # Check metrics
    metrics_response = client.get("/metrics")
    content = metrics_response.text

    # Should have incremented request_total
    assert 'semantic_request_total{method="GET",path="/",status_code="200"}' in content

    print("\n✅ request_total counter working")


@pytest.mark.integration
def test_request_histogram_records(test_app):
    """Test request_duration_seconds histogram records latency.

    Expected behavior:
    - Histogram buckets present
    - Labels include method and path
    - Records timing for each request
    """
    client = TestClient(test_app)

    # Make a request
    response = client.get("/")
    assert response.status_code == 200

    # Check metrics
    metrics_response = client.get("/metrics")
    content = metrics_response.text

    # Should have recorded histogram
    assert "semantic_request_duration_seconds_bucket" in content
    assert 'method="GET",path="/"' in content
    assert "semantic_request_duration_seconds_count" in content
    assert "semantic_request_duration_seconds_sum" in content

    print("\n✅ request_duration histogram working")


@pytest.mark.integration
def test_metrics_endpoint_not_tracked(test_app):
    """Test /metrics endpoint doesn't track itself.

    Expected behavior:
    - /metrics path not in request_total
    - Prevents infinite metrics growth
    - Skip_paths configuration working
    """
    client = TestClient(test_app)

    # Get metrics multiple times
    for _ in range(5):
        response = client.get("/metrics")
        assert response.status_code == 200

    # Check metrics one final time
    metrics_response = client.get("/metrics")
    content = metrics_response.text

    # /metrics should not appear in request_total
    assert 'path="/metrics"' not in content

    print("\n✅ /metrics endpoint not self-tracking")


@pytest.mark.integration
def test_pool_utilization_metric(test_app):
    """Test pool_utilization_ratio gauge updates.

    Expected behavior:
    - Gauge starts at 0.0 (no pool)
    - Updates when pool is allocated
    - Reflects actual utilization ratio
    """
    client = TestClient(test_app)

    # Create a small pool
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

    # Trigger health check (updates metric)
    response = client.get("/health/ready")
    assert response.status_code == 200

    # Check metrics
    metrics_response = client.get("/metrics")
    content = metrics_response.text

    # Should have pool utilization (0.0 since no blocks allocated)
    assert "semantic_pool_utilization_ratio 0.0" in content

    # Allocate some blocks (50%)
    pool.allocate(n_blocks=5, layer_id=0, agent_id="test")

    # Trigger health check again
    response = client.get("/health/ready")

    # Check metrics again
    metrics_response = client.get("/metrics")
    content = metrics_response.text

    # Should now show 0.5 utilization
    assert "semantic_pool_utilization_ratio 0.5" in content

    print("\n✅ pool_utilization metric working")
