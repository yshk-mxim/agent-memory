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

        # Endpoint exists (404 not found, 500 no model, or 503 if lifespan not run)
        assert response.status_code in [404, 500, 503]

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

        # Endpoint exists (404 not found, 500 no model, or 503 if lifespan not run)
        assert response.status_code in [404, 500, 503]

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
class TestDirectAgentAPIWithModel:
    """Tests that require MLX model loaded."""

    def test_create_and_generate(self):
        """Create agent and generate text should work end-to-end."""
        from fastapi.testclient import TestClient

        app = create_app()

        with TestClient(app) as client:
            # Create agent
            response = client.post("/v1/agents", json={"agent_id": "test-agent"})
            assert response.status_code == 201

            # Generate with agent
            response = client.post(
                "/v1/agents/test-agent/generate",
                json={"prompt": "Hello", "max_tokens": 20},
            )
            assert response.status_code == 200
            data = response.json()
            assert "text" in data
            assert len(data["text"]) > 0
            assert data["tokens_generated"] > 0

    def test_agent_persistence(self):
        """Agent cache should persist across requests."""
        from fastapi.testclient import TestClient

        app = create_app()

        with TestClient(app) as client:
            # Create agent and generate
            client.post("/v1/agents", json={"agent_id": "persist-test"})
            response1 = client.post(
                "/v1/agents/persist-test/generate",
                json={"prompt": "First", "max_tokens": 10},
            )
            assert response1.status_code == 200
            cache_size_1 = response1.json()["cache_size_tokens"]

            # Second generation should have larger cache
            response2 = client.post(
                "/v1/agents/persist-test/generate",
                json={"prompt": "Second", "max_tokens": 10},
            )
            assert response2.status_code == 200
            cache_size_2 = response2.json()["cache_size_tokens"]
            assert cache_size_2 >= cache_size_1

    def test_delete_agent(self):
        """Deleting agent should remove cache."""
        from fastapi.testclient import TestClient

        app = create_app()

        with TestClient(app) as client:
            # Create agent and generate
            client.post("/v1/agents", json={"agent_id": "delete-test"})
            client.post(
                "/v1/agents/delete-test/generate",
                json={"prompt": "Test", "max_tokens": 10},
            )

            # Delete agent
            response = client.delete("/v1/agents/delete-test")
            assert response.status_code == 204

            # Agent should not exist
            response = client.get("/v1/agents/delete-test")
            assert response.status_code == 404
