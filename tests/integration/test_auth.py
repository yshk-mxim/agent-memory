"""Integration tests for authentication middleware.

Tests API key validation:
- Missing API key (401)
- Invalid API key (401)
- Valid API key (allowed)
- Public endpoints (no auth required)
"""

import os

import pytest
from fastapi.testclient import TestClient

from semantic.entrypoints.api_server import create_app


@pytest.mark.integration
class TestAuthentication:
    """Test authentication middleware."""

    def test_health_endpoint_no_auth_required(self):
        """Health endpoint should not require authentication."""
        # Set API key to enable auth globally
        os.environ["ANTHROPIC_API_KEY"] = "test-key-12345"

        app = create_app()
        client = TestClient(app, raise_server_exceptions=False)

        # No API key provided
        response = client.get("/health")

        # Should succeed without auth
        assert response.status_code == 200

        # Cleanup
        del os.environ["ANTHROPIC_API_KEY"]

    def test_root_endpoint_no_auth_required(self):
        """Root endpoint should not require authentication."""
        os.environ["ANTHROPIC_API_KEY"] = "test-key-12345"

        app = create_app()
        client = TestClient(app, raise_server_exceptions=False)

        response = client.get("/")

        # Should succeed without auth
        assert response.status_code == 200

        del os.environ["ANTHROPIC_API_KEY"]

    def test_api_endpoint_missing_key(self):
        """API endpoints should reject requests without API key."""
        os.environ["ANTHROPIC_API_KEY"] = "test-key-12345"

        app = create_app()
        client = TestClient(app, raise_server_exceptions=False)

        # No API key in request
        response = client.post(
            "/v1/messages",
            json={
                "model": "claude-sonnet-4-5-20250929",
                "max_tokens": 10,
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )

        # Should return 401 Unauthorized
        assert response.status_code == 401
        data = response.json()
        assert "error" in data
        assert data["error"]["type"] == "authentication_error"
        assert "Missing API key" in data["error"]["message"]

        del os.environ["ANTHROPIC_API_KEY"]

    def test_api_endpoint_invalid_key(self):
        """API endpoints should reject requests with invalid API key."""
        os.environ["ANTHROPIC_API_KEY"] = "valid-key-12345"

        app = create_app()
        client = TestClient(app, raise_server_exceptions=False)

        # Invalid API key
        response = client.post(
            "/v1/messages",
            json={
                "model": "claude-sonnet-4-5-20250929",
                "max_tokens": 10,
                "messages": [{"role": "user", "content": "Hello"}],
            },
            headers={"x-api-key": "wrong-key"},
        )

        # Should return 401 Unauthorized
        assert response.status_code == 401
        data = response.json()
        assert "error" in data
        assert data["error"]["type"] == "authentication_error"
        assert "Invalid API key" in data["error"]["message"]

        del os.environ["ANTHROPIC_API_KEY"]

    def test_api_endpoint_valid_key(self):
        """API endpoints should accept requests with valid API key."""
        os.environ["ANTHROPIC_API_KEY"] = "valid-key-12345"

        app = create_app()
        client = TestClient(app, raise_server_exceptions=False)

        # Valid API key
        response = client.post(
            "/v1/messages",
            json={
                "model": "claude-sonnet-4-5-20250929",
                "max_tokens": 10,
                "messages": [{"role": "user", "content": "Hello"}],
            },
            headers={"x-api-key": "valid-key-12345"},
        )

        # Should NOT return 401 (may return other errors without model loaded)
        assert response.status_code != 401

        del os.environ["ANTHROPIC_API_KEY"]

    def test_alternative_header_name(self):
        """Should accept anthropic-api-key header as alternative."""
        os.environ["ANTHROPIC_API_KEY"] = "valid-key-12345"

        app = create_app()
        client = TestClient(app, raise_server_exceptions=False)

        # Use alternative header name
        response = client.post(
            "/v1/messages",
            json={
                "model": "claude-sonnet-4-5-20250929",
                "max_tokens": 10,
                "messages": [{"role": "user", "content": "Hello"}],
            },
            headers={"anthropic-api-key": "valid-key-12345"},
        )

        # Should NOT return 401
        assert response.status_code != 401

        del os.environ["ANTHROPIC_API_KEY"]

    def test_multiple_valid_keys(self):
        """Should support comma-separated list of valid keys."""
        os.environ["ANTHROPIC_API_KEY"] = "key1,key2,key3"

        app = create_app()
        client = TestClient(app, raise_server_exceptions=False)

        # Try each key
        for key in ["key1", "key2", "key3"]:
            response = client.post(
                "/v1/messages",
                json={
                    "model": "claude-sonnet-4-5-20250929",
                    "max_tokens": 10,
                    "messages": [{"role": "user", "content": "Hello"}],
                },
                headers={"x-api-key": key},
            )

            # Should NOT return 401
            assert response.status_code != 401

        del os.environ["ANTHROPIC_API_KEY"]

    def test_auth_disabled_when_no_env_var(self):
        """Authentication should be disabled if ANTHROPIC_API_KEY not set."""
        # Ensure no API key in environment
        if "ANTHROPIC_API_KEY" in os.environ:
            del os.environ["ANTHROPIC_API_KEY"]

        app = create_app()
        client = TestClient(app, raise_server_exceptions=False)

        # No API key in request
        response = client.post(
            "/v1/messages",
            json={
                "model": "claude-sonnet-4-5-20250929",
                "max_tokens": 10,
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )

        # Should NOT return 401 (auth disabled)
        assert response.status_code != 401

    def test_openai_endpoint_auth(self):
        """OpenAI endpoint should also require authentication."""
        os.environ["ANTHROPIC_API_KEY"] = "test-key-12345"

        app = create_app()
        client = TestClient(app, raise_server_exceptions=False)

        # No API key
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )

        # Should return 401
        assert response.status_code == 401

        del os.environ["ANTHROPIC_API_KEY"]

    def test_direct_agent_endpoint_auth(self):
        """Direct Agent endpoint should also require authentication."""
        os.environ["ANTHROPIC_API_KEY"] = "test-key-12345"

        app = create_app()
        client = TestClient(app, raise_server_exceptions=False)

        # No API key
        response = client.post(
            "/v1/agents",
            json={},
        )

        # Should return 401
        assert response.status_code == 401

        del os.environ["ANTHROPIC_API_KEY"]
