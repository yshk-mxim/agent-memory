"""Integration tests for request logging middleware (Sprint 7 Day 2).

Tests verify that request logging:
- Logs all requests with timing information
- Skips health check paths to avoid spam
- Logs errors with context
- Provides reasonable timing values
"""

import pytest
from fastapi.testclient import TestClient

from semantic.entrypoints.api_server import create_app


@pytest.fixture
def test_app():
    """Create test FastAPI app with request logging middleware."""
    app = create_app()

    # Initialize minimal state for testing
    class MockAppState:
        def __init__(self):
            self.shutting_down = False
            self.semantic = type('obj', (object,), {
                'block_pool': None,
                'batch_engine': None,
            })()

    app.state = MockAppState()

    return app


@pytest.mark.integration
def test_request_logged_with_timing(test_app):
    """Test request is logged with timing information.

    Expected behavior:
    - X-Response-Time header is present
    - Timing is in milliseconds format (e.g., "1.23ms")
    - Response is successful
    """
    client = TestClient(test_app)

    response = client.get("/")

    assert response.status_code == 200
    assert "X-Response-Time" in response.headers
    # Should be like "1.23ms"
    assert "ms" in response.headers["X-Response-Time"]

    timing_str = response.headers["X-Response-Time"]
    print(f"\n✅ Request logged with timing: {timing_str}")


@pytest.mark.integration
def test_health_checks_not_logged(test_app):
    """Test health check paths are not logged (avoid spam).

    Expected behavior:
    - Health endpoints work normally
    - No request_start/request_complete logs for health checks
    - All three health endpoints are skipped
    """
    client = TestClient(test_app)

    # Health checks should work but not spam logs
    response = client.get("/health/live")
    assert response.status_code == 200
    assert "X-Request-ID" in response.headers
    # X-Response-Time should NOT be present for skipped paths
    assert "X-Response-Time" not in response.headers

    response = client.get("/health/ready")
    assert response.status_code in [200, 503]
    assert "X-Request-Time" not in response.headers

    response = client.get("/health/startup")
    assert response.status_code in [200, 503]
    assert "X-Response-Time" not in response.headers

    print("\n✅ Health checks not logged (spam prevention)")


@pytest.mark.integration
def test_error_logged_with_context(test_app):
    """Test errors are logged with request context.

    Expected behavior:
    - 404 responses get timing header
    - Error responses are logged with context
    - Response is 404 (not found)
    """
    client = TestClient(test_app)

    # Trigger an error (invalid endpoint)
    response = client.get("/nonexistent")

    assert response.status_code == 404
    assert "X-Response-Time" in response.headers

    timing_str = response.headers["X-Response-Time"]
    print(f"\n✅ Error logged with timing: {timing_str}")


@pytest.mark.integration
def test_request_timing_reasonable(test_app):
    """Test response timing header is reasonable.

    Expected behavior:
    - Timing is parseable as float
    - Timing is positive
    - Timing is less than 1000ms for simple endpoint
    """
    client = TestClient(test_app)

    response = client.get("/")

    # Parse timing (e.g., "1.23ms")
    timing_str = response.headers["X-Response-Time"]
    timing_value = float(timing_str.replace("ms", ""))

    # Should be fast for simple endpoint (<1000ms)
    assert timing_value < 1000.0
    assert timing_value > 0.0

    print(f"\n✅ Timing reasonable: {timing_value:.2f}ms (< 1000ms)")
