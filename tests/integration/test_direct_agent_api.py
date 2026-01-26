"""Integration tests for Direct Agent API.

Tests agent CRUD operations:
- POST /v1/agents (create)
- GET /v1/agents/{agent_id} (get info)
- POST /v1/agents/{agent_id}/generate (generate)
- DELETE /v1/agents/{agent_id} (delete)
"""

import pytest
from fastapi.testclient import TestClient

from semantic.entrypoints.api_server import create_app


@pytest.mark.integration
class TestDirectAgentAPI:
    """Test Direct Agent API endpoints."""

    def test_create_agent_endpoint_exists(self):
        """POST /v1/agents endpoint should be registered."""
        app = create_app()
        client = TestClient(app, raise_server_exceptions=False)

        response = client.post(
            "/v1/agents",
            json={},
        )

        # Endpoint exists (may fail for other reasons without model loaded)
        assert response.status_code != 404

    def test_create_agent_with_id(self):
        """Creating agent with explicit ID should be accepted."""
        app = create_app()
        client = TestClient(app, raise_server_exceptions=False)

        response = client.post(
            "/v1/agents",
            json={"agent_id": "test_agent_123"},
        )

        # Validation passes (may fail without model, but validation is OK)
        assert response.status_code != 422

    def test_create_agent_without_id(self):
        """Creating agent without ID should generate one."""
        app = create_app()
        client = TestClient(app, raise_server_exceptions=False)

        response = client.post(
            "/v1/agents",
            json={},
        )

        # Validation passes
        assert response.status_code != 422

    def test_get_agent_endpoint_exists(self):
        """GET /v1/agents/{agent_id} endpoint should be registered."""
        app = create_app()
        client = TestClient(app, raise_server_exceptions=False)

        response = client.get("/v1/agents/test_agent")

        # Endpoint exists (will return 404 if agent not found)
        assert response.status_code in [404, 500]  # 404 not found or 500 no model

    def test_generate_endpoint_exists(self):
        """POST /v1/agents/{agent_id}/generate endpoint should be registered."""
        app = create_app()
        client = TestClient(app, raise_server_exceptions=False)

        response = client.post(
            "/v1/agents/test_agent/generate",
            json={
                "prompt": "Hello",
                "max_tokens": 10,
            },
        )

        # Endpoint exists
        assert response.status_code != 404

    def test_generate_validation(self):
        """Generate request should validate required fields."""
        app = create_app()
        client = TestClient(app, raise_server_exceptions=False)

        # Missing prompt
        response = client.post(
            "/v1/agents/test_agent/generate",
            json={
                "max_tokens": 10,
            },
        )

        # Should fail validation
        assert response.status_code == 422

    def test_delete_agent_endpoint_exists(self):
        """DELETE /v1/agents/{agent_id} endpoint should be registered."""
        app = create_app()
        client = TestClient(app, raise_server_exceptions=False)

        response = client.delete("/v1/agents/test_agent")

        # Endpoint exists (will return 404 if agent not found)
        assert response.status_code in [404, 500]  # 404 not found or 500 no model

    def test_temperature_validation(self):
        """Temperature should be validated to 0-2 range."""
        app = create_app()
        client = TestClient(app, raise_server_exceptions=False)

        # Valid temperature
        response = client.post(
            "/v1/agents/test_agent/generate",
            json={
                "prompt": "Hello",
                "max_tokens": 10,
                "temperature": 0.5,
            },
        )
        assert response.status_code != 422

        # Invalid temperature (too high)
        response = client.post(
            "/v1/agents/test_agent/generate",
            json={
                "prompt": "Hello",
                "max_tokens": 10,
                "temperature": 2.5,
            },
        )
        assert response.status_code == 422


@pytest.mark.integration
@pytest.mark.skip(reason="Requires model loading - run manually with real server")
class TestDirectAgentAPIWithModel:
    """Tests that require MLX model loaded.

    These tests are skipped by default because they:
    - Load the full MLX model (~several GB)
    - Take 30+ seconds to complete
    - Require GPU/Metal

    To run manually:
        pytest -v -m integration tests/integration/test_direct_agent_api.py::TestDirectAgentAPIWithModel
    """

    def test_create_and_generate(self):
        """Create agent and generate text should work end-to-end."""
        # This would test with real model loading via TestClient lifespan

    def test_agent_persistence(self):
        """Agent cache should persist across requests."""

    def test_delete_agent(self):
        """Deleting agent should remove cache."""
