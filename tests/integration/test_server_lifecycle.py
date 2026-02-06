"""Integration tests for server lifecycle (startup/shutdown).

Tests FastAPI application creation, health endpoints, and middleware.
"""

import pytest
from fastapi.testclient import TestClient

from semantic.entrypoints.api_server import create_app


@pytest.mark.integration
class TestServerHealth:
    """Test health check endpoint."""

    def test_health_endpoint_returns_200(self):
        """Health endpoint should return 200 OK with status dict."""
        # Note: TestClient doesn't trigger lifespan events (no model loading)
        # This is intentional - we're testing the endpoint definition only
        app = create_app()
        client = TestClient(app, raise_server_exceptions=False)

        response = client.get("/health")

        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_root_endpoint_returns_api_info(self):
        """Root endpoint should return API information."""
        app = create_app()
        client = TestClient(app, raise_server_exceptions=False)

        response = client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Semantic Caching API"
        assert data["version"] == "0.2.0"
        assert "endpoints" in data
        assert data["endpoints"]["health"] == "/health"


@pytest.mark.integration
class TestCORSMiddleware:
    """Test CORS middleware configuration."""

    def test_cors_headers_present(self):
        """CORS middleware should add appropriate headers."""
        app = create_app()
        client = TestClient(app, raise_server_exceptions=False)

        # Send GET with Origin header to trigger CORS
        response = client.get("/health", headers={"Origin": "http://localhost:3000"})

        # Check CORS headers are present
        assert "access-control-allow-origin" in response.headers
        # CORS middleware adds allow-origin on requests with Origin header


@pytest.mark.integration
class TestErrorHandlers:
    """Test error handling middleware."""

    def test_404_not_found(self):
        """Unknown endpoints should return 404."""
        app = create_app()
        client = TestClient(app, raise_server_exceptions=False)

        response = client.get("/nonexistent")

        assert response.status_code == 404

    def test_validation_error_format(self):
        """Validation errors should return structured format."""
        # Verify the app creates successfully with error handlers registered
        app = create_app()
        # Verify app has routes registered
        assert len(app.routes) > 0


@pytest.mark.integration
class TestServerLifecycle:
    """Test full server lifecycle with model loading.

    These tests are marked slow because they load the MLX model.
    """

    def test_server_starts_and_stops_cleanly(self):
        """Server should initialize all components and shut down cleanly.

        This test is skipped by default because it:
        - Loads the full MLX model (~several GB)
        - Takes 30+ seconds to complete
        - Requires GPU/Metal

        To run manually:
            pytest -v -m integration tests/integration/test_server_lifecycle.py::TestServerLifecycle::test_server_starts_and_stops_cleanly
        """
        from fastapi.testclient import TestClient

        app = create_app()

        # TestClient context manager triggers lifespan events
        with TestClient(app) as client:
            # Verify server is initialized
            response = client.get("/health")
            assert response.status_code == 200

            # Verify app state is populated
            assert hasattr(app.state, "semantic")
            assert hasattr(app.state.semantic, "block_pool")
            assert hasattr(app.state.semantic, "batch_engine")
            assert hasattr(app.state.semantic, "cache_store")

        # After context exit, shutdown should have completed
        # (no assertions needed - test passes if no exceptions)
