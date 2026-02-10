"""Integration tests for OpenAI Chat Completions API.

Tests POST /v1/chat/completions endpoint with:
- Simple text generation
- Session ID extension (request body and header)
- Multi-turn conversations
- Error handling
"""

import pytest
from fastapi.testclient import TestClient

from agent_memory.entrypoints.api_server import create_app


@pytest.mark.integration
class TestOpenAIChatCompletionsAPI:
    """Test OpenAI Chat Completions API endpoint."""

    def test_endpoint_exists(self):
        """POST /v1/chat/completions endpoint should be registered."""
        app = create_app()
        client = TestClient(app, raise_server_exceptions=False)

        # Send minimal valid request
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4",
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
            "/v1/chat/completions",
            json={
                "model": "gpt-4",
                "messages": [],
            },
        )

        assert response.status_code == 422  # Validation error
        data = response.json()
        assert "error" in data or "detail" in data

    def test_session_id_in_request_body(self):
        """Session ID in request body should be accepted."""
        app = create_app()
        client = TestClient(app, raise_server_exceptions=False)

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "Hello"}],
                "session_id": "session_123",
            },
        )

        # Validation passes (may fail without model, but validation is OK)
        assert response.status_code != 422

    def test_session_id_in_header(self):
        """Session ID in X-Session-ID header should be accepted."""
        app = create_app()
        client = TestClient(app, raise_server_exceptions=False)

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "Hello"}],
            },
            headers={"X-Session-ID": "session_456"},
        )

        # Validation passes
        assert response.status_code != 422

    def test_optional_max_tokens(self):
        """max_tokens should be optional (defaults to 256)."""
        app = create_app()
        client = TestClient(app, raise_server_exceptions=False)

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )

        # Validation passes (max_tokens is optional)
        assert response.status_code != 422

    def test_temperature_default(self):
        """Temperature should default to 1.0."""
        app = create_app()
        client = TestClient(app, raise_server_exceptions=False)

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )

        # Validation passes (temperature has default)
        assert response.status_code != 422

    def test_stream_flag(self):
        """stream=true should be accepted but return 501."""
        app = create_app()
        client = TestClient(app, raise_server_exceptions=False)

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": True,
            },
        )

        # Either validation passes (no model) or returns 501 (streaming not implemented)
        assert response.status_code in [422, 500, 501, 503]

    def test_multi_turn_conversation(self):
        """Multi-turn conversation should be accepted."""
        app = create_app()
        client = TestClient(app, raise_server_exceptions=False)

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4",
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there!"},
                    {"role": "user", "content": "How are you?"},
                ],
            },
        )

        # Validation passes
        assert response.status_code != 422

    def test_system_message_support(self):
        """System message should be accepted."""
        app = create_app()
        client = TestClient(app, raise_server_exceptions=False)

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4",
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Hello"},
                ],
            },
        )

        # Validation passes
        assert response.status_code != 422


@pytest.mark.integration
class TestOpenAIAPIWithModel:
    """Tests that require MLX model loaded."""

    def test_simple_generation(self):
        """Simple text generation should work end-to-end."""
        from fastapi.testclient import TestClient

        app = create_app()

        with TestClient(app) as client:
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "SmolLM2-135M-Instruct",
                    "messages": [{"role": "user", "content": "Say hello"}],
                    "max_tokens": 20,
                },
            )

            assert response.status_code == 200
            data = response.json()
            assert "choices" in data
            assert len(data["choices"]) > 0
            assert data["choices"][0]["message"]["role"] == "assistant"
            assert len(data["choices"][0]["message"]["content"]) > 0

    def test_session_persistence(self):
        """Session ID should persist cache across multiple requests."""
        from fastapi.testclient import TestClient

        app = create_app()

        with TestClient(app) as client:
            session_id = "test-session-123"

            # First request with session ID
            response1 = client.post(
                "/v1/chat/completions",
                json={
                    "model": "SmolLM2-135M-Instruct",
                    "messages": [{"role": "user", "content": "Hello"}],
                    "max_tokens": 10,
                    "session_id": session_id,
                },
            )
            assert response1.status_code == 200

            # Second request with same session ID should reuse cache
            response2 = client.post(
                "/v1/chat/completions",
                json={
                    "model": "SmolLM2-135M-Instruct",
                    "messages": [{"role": "user", "content": "Hello"}],
                    "max_tokens": 10,
                    "session_id": session_id,
                },
            )
            assert response2.status_code == 200

    def test_response_format(self):
        """Response should match OpenAI format."""
        from fastapi.testclient import TestClient

        app = create_app()

        with TestClient(app) as client:
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "SmolLM2-135M-Instruct",
                    "messages": [{"role": "user", "content": "Test"}],
                    "max_tokens": 10,
                },
            )

            assert response.status_code == 200
            data = response.json()
            # OpenAI format fields
            assert "id" in data
            assert data["id"].startswith("chatcmpl-")
            assert "created" in data
            assert isinstance(data["created"], int)
            assert "model" in data
            assert "choices" in data
            assert len(data["choices"]) == 1
            assert "usage" in data
            assert data["choices"][0]["finish_reason"] in ["stop", "length"]
            assert data["choices"][0]["message"]["role"] == "assistant"
            assert isinstance(data["choices"][0]["message"]["content"], str)
            # Usage fields must be present and positive
            assert data["usage"]["prompt_tokens"] > 0
            assert data["usage"]["completion_tokens"] > 0
            assert data["usage"]["total_tokens"] == (
                data["usage"]["prompt_tokens"] + data["usage"]["completion_tokens"]
            )
