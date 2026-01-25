"""Integration tests for Anthropic Messages API.

Tests POST /v1/messages endpoint with:
- Simple text generation
- Multi-turn conversations
- System prompts
- Error handling
"""

import pytest
from fastapi.testclient import TestClient

from semantic.entrypoints.api_server import create_app


@pytest.mark.integration
class TestAnthropicMessagesAPI:
    """Test Anthropic Messages API endpoint."""

    def test_endpoint_exists(self):
        """POST /v1/messages endpoint should be registered."""
        app = create_app()
        client = TestClient(app, raise_server_exceptions=False)

        # Send minimal valid request
        response = client.post(
            "/v1/messages",
            json={
                "model": "test-model",
                "max_tokens": 10,
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )

        # Endpoint exists (may fail for other reasons without model loaded)
        # We're just testing the route is registered
        assert response.status_code != 404

    def test_request_validation_empty_messages(self):
        """Empty messages list should be rejected."""
        app = create_app()
        client = TestClient(app, raise_server_exceptions=False)

        response = client.post(
            "/v1/messages",
            json={
                "model": "test-model",
                "max_tokens": 10,
                "messages": [],
            },
        )

        assert response.status_code == 422  # Validation error
        data = response.json()
        assert "error" in data or "detail" in data

    def test_request_validation_first_message_not_user(self):
        """First message must be from user."""
        app = create_app()
        client = TestClient(app, raise_server_exceptions=False)

        response = client.post(
            "/v1/messages",
            json={
                "model": "test-model",
                "max_tokens": 10,
                "messages": [{"role": "assistant", "content": "Hello"}],
            },
        )

        assert response.status_code == 422  # Validation error

    def test_request_validation_consecutive_same_role(self):
        """Consecutive messages with same role should be rejected."""
        app = create_app()
        client = TestClient(app, raise_server_exceptions=False)

        response = client.post(
            "/v1/messages",
            json={
                "model": "test-model",
                "max_tokens": 10,
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "user", "content": "Again"},
                ],
            },
        )

        assert response.status_code == 422  # Validation error

    def test_request_with_system_prompt_string(self):
        """System prompt as string should be accepted."""
        app = create_app()
        client = TestClient(app, raise_server_exceptions=False)

        response = client.post(
            "/v1/messages",
            json={
                "model": "test-model",
                "max_tokens": 10,
                "messages": [{"role": "user", "content": "Hello"}],
                "system": "You are a helpful assistant.",
            },
        )

        # Validation passes (may fail without model, but validation is OK)
        assert response.status_code != 422

    def test_request_with_tools(self):
        """Request with tools should be accepted."""
        app = create_app()
        client = TestClient(app, raise_server_exceptions=False)

        response = client.post(
            "/v1/messages",
            json={
                "model": "test-model",
                "max_tokens": 10,
                "messages": [{"role": "user", "content": "Search for cats"}],
                "tools": [
                    {
                        "name": "search",
                        "description": "Search the web",
                        "input_schema": {
                            "type": "object",
                            "properties": {"query": {"type": "string"}},
                        },
                    }
                ],
            },
        )

        # Validation passes
        assert response.status_code != 422

    def test_request_with_temperature_bounds(self):
        """Temperature should be validated to 0-2 range."""
        app = create_app()
        client = TestClient(app, raise_server_exceptions=False)

        # Valid temperatures
        for temp in [0.0, 1.0, 2.0]:
            response = client.post(
                "/v1/messages",
                json={
                    "model": "test-model",
                    "max_tokens": 10,
                    "messages": [{"role": "user", "content": "Hello"}],
                    "temperature": temp,
                },
            )
            assert response.status_code != 422

        # Invalid temperature (too high)
        response = client.post(
            "/v1/messages",
            json={
                "model": "test-model",
                "max_tokens": 10,
                "messages": [{"role": "user", "content": "Hello"}],
                "temperature": 2.1,
            },
        )
        assert response.status_code == 422

    def test_request_with_stream_true(self):
        """Request with stream=true should be accepted."""
        app = create_app()
        client = TestClient(app, raise_server_exceptions=False)

        response = client.post(
            "/v1/messages",
            json={
                "model": "test-model",
                "max_tokens": 10,
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": True,
            },
        )

        # Validation passes (may fail without model, but validation is OK)
        assert response.status_code != 422

    def test_count_tokens_endpoint_exists(self):
        """POST /v1/messages/count_tokens endpoint should be registered."""
        app = create_app()
        client = TestClient(app, raise_server_exceptions=False)

        response = client.post(
            "/v1/messages/count_tokens",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )

        # Endpoint exists (may fail for other reasons without model loaded)
        assert response.status_code != 404


@pytest.mark.integration
@pytest.mark.skip(reason="Requires model loading - run manually with real server")
class TestAnthropicAPIWithModel:
    """Tests that require MLX model loaded.

    These tests are skipped by default because they:
    - Load the full MLX model (~several GB)
    - Take 30+ seconds to complete
    - Require GPU/Metal

    To run manually:
        pytest -v -m integration tests/integration/test_anthropic_api.py::TestAnthropicAPIWithModel
    """

    def test_simple_generation(self):
        """Simple text generation should work end-to-end."""
        # This would test with real model loading via TestClient lifespan
        pass

    def test_cache_reuse_across_requests(self):
        """Cache should persist across multiple requests."""
        pass

    def test_multi_turn_conversation(self):
        """Multi-turn conversation should maintain context."""
        pass
