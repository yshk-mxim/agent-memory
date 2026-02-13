# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Yakov Shkolnikov and contributors
"""Integration tests for Anthropic Messages API.

Tests POST /v1/messages endpoint with:
- Simple text generation
- Multi-turn conversations
- System prompts
- Error handling
"""

import pytest
from fastapi.testclient import TestClient

from agent_memory.entrypoints.api_server import create_app


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
class TestAnthropicAPIWithModel:
    """Tests that require MLX model loaded.

    These tests load the full MLX model and verify end-to-end functionality.
    """

    def test_simple_generation(self):
        """Simple text generation should work end-to-end."""
        from fastapi.testclient import TestClient

        app = create_app()

        with TestClient(app) as client:
            response = client.post(
                "/v1/messages",
                headers={"Content-Type": "application/json"},
                json={
                    "model": "SmolLM2-135M-Instruct",
                    "messages": [{"role": "user", "content": "Say hello"}],
                    "max_tokens": 20,
                },
            )

            assert response.status_code == 200
            data = response.json()
            # Content block validation
            assert "content" in data
            assert len(data["content"]) > 0
            assert data["content"][0]["type"] == "text"
            assert isinstance(data["content"][0]["text"], str)
            assert len(data["content"][0]["text"]) > 0
            # Stop reason must be a valid Anthropic reason
            assert data["stop_reason"] in ("end_turn", "max_tokens")
            # Usage fields
            assert "usage" in data
            assert data["usage"]["input_tokens"] > 0
            assert data["usage"]["output_tokens"] > 0
            # Model and ID fields
            assert "id" in data
            assert "model" in data

    def test_cache_reuse_across_requests(self):
        """Cache should persist across multiple requests."""
        from fastapi.testclient import TestClient

        app = create_app()

        with TestClient(app) as client:
            # First request creates cache
            response1 = client.post(
                "/v1/messages",
                json={
                    "model": "SmolLM2-135M-Instruct",
                    "messages": [{"role": "user", "content": "Hello"}],
                    "max_tokens": 10,
                },
            )
            assert response1.status_code == 200
            usage1 = response1.json()["usage"]
            assert usage1["cache_creation_input_tokens"] > 0

            # Second request reuses cache
            response2 = client.post(
                "/v1/messages",
                json={
                    "model": "SmolLM2-135M-Instruct",
                    "messages": [{"role": "user", "content": "Hello"}],
                    "max_tokens": 10,
                },
            )
            assert response2.status_code == 200
            usage2 = response2.json()["usage"]
            assert usage2["cache_read_input_tokens"] > 0

    def test_multi_turn_conversation(self):
        """Multi-turn conversation should maintain context."""
        from fastapi.testclient import TestClient

        app = create_app()

        with TestClient(app) as client:
            response = client.post(
                "/v1/messages",
                json={
                    "model": "SmolLM2-135M-Instruct",
                    "messages": [
                        {"role": "user", "content": "Hi"},
                        {"role": "assistant", "content": "Hello"},
                        {"role": "user", "content": "How are you?"},
                    ],
                    "max_tokens": 20,
                },
            )
            assert response.status_code == 200
            data = response.json()
            assert len(data["content"]) > 0
            assert data["content"][0]["type"] == "text"
            assert data["stop_reason"] in ("end_turn", "max_tokens")
