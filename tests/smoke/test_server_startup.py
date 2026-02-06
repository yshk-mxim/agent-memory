"""Smoke tests for server lifecycle.

These tests verify basic server functionality:
- Server starts successfully
- Health endpoint responds
- Model loads correctly
- Graceful shutdown works
"""

import httpx
import pytest


@pytest.mark.smoke
def test_server_starts_successfully(live_server: str):
    """Test that server starts within 60 seconds.

    Verifies:
    - Server process starts
    - Model loads successfully
    - Health endpoint becomes available

    Performance target: <60s startup time
    """
    # live_server fixture already waited for server to start
    # If we got here, server started successfully
    assert live_server.startswith("http://localhost:")

    # Verify health endpoint responds
    response = httpx.get(f"{live_server}/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "pool" in data  # Enhanced health check includes pool metrics


@pytest.mark.smoke
def test_health_endpoint_responds(live_server: str):
    """Test that health endpoint returns correct status.

    Verifies:
    - GET /health returns 200
    - Response JSON has correct structure
    """
    response = httpx.get(f"{live_server}/health")

    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "ok"


@pytest.mark.smoke
def test_model_loads_correctly(live_server: str):
    """Test that model loads and server is ready for inference.

    Verifies:
    - Server started (implying model loaded)
    - Root endpoint responds with API info
    """
    response = httpx.get(f"{live_server}/")

    assert response.status_code == 200
    data = response.json()
    assert "name" in data
    assert "version" in data
    assert "endpoints" in data
    assert data["name"] == "Semantic Caching API"


@pytest.mark.smoke
def test_graceful_shutdown_works(live_server: str):
    """Test that server shuts down gracefully.

    Verifies:
    - Server responds before shutdown
    - Shutdown completes successfully

    Note: Actual shutdown tested by live_server fixture teardown.
    This test just verifies server is running before teardown.
    """
    # Verify server is running
    response = httpx.get(f"{live_server}/health")
    assert response.status_code == 200

    # Fixture teardown will test graceful shutdown
    # If teardown fails or hangs, test will fail
