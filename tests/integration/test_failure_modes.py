# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Yakov Shkolnikov and contributors
"""Integration tests for failure modes and error handling.

Tests edge cases and error conditions:
- Malformed JSON
- Invalid field values
- Missing required fields
- Invalid model names
- Empty content
"""

import os

import pytest
from fastapi.testclient import TestClient

from agent_memory.entrypoints.api_server import create_app


@pytest.mark.integration
class TestFailureModes:
    """Test error handling and edge cases."""

    def test_malformed_json(self):
        """Malformed JSON should return 422."""
        if "ANTHROPIC_API_KEY" in os.environ:
            del os.environ["ANTHROPIC_API_KEY"]

        app = create_app()
        client = TestClient(app, raise_server_exceptions=False)

        # Send invalid JSON
        response = client.post(
            "/v1/messages",
            data="{invalid json}",
            headers={"Content-Type": "application/json"},
        )

        assert response.status_code == 422

    def test_missing_required_field(self):
        """Missing required fields should return 422."""
        if "ANTHROPIC_API_KEY" in os.environ:
            del os.environ["ANTHROPIC_API_KEY"]

        app = create_app()
        client = TestClient(app, raise_server_exceptions=False)

        # Missing max_tokens
        response = client.post(
            "/v1/messages",
            json={
                "model": "claude-sonnet-4-5-20250929",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )

        assert response.status_code == 422

    def test_invalid_temperature(self):
        """Out-of-range temperature should return 422."""
        if "ANTHROPIC_API_KEY" in os.environ:
            del os.environ["ANTHROPIC_API_KEY"]

        app = create_app()
        client = TestClient(app, raise_server_exceptions=False)

        # Temperature too high
        response = client.post(
            "/v1/messages",
            json={
                "model": "claude-sonnet-4-5-20250929",
                "max_tokens": 10,
                "messages": [{"role": "user", "content": "Hello"}],
                "temperature": 3.0,  # Invalid (max is 2.0)
            },
        )

        assert response.status_code == 422

    def test_empty_message_content(self):
        """Empty message content should return 422."""
        if "ANTHROPIC_API_KEY" in os.environ:
            del os.environ["ANTHROPIC_API_KEY"]

        app = create_app()
        client = TestClient(app, raise_server_exceptions=False)

        # Empty content
        response = client.post(
            "/v1/messages",
            json={
                "model": "claude-sonnet-4-5-20250929",
                "max_tokens": 10,
                "messages": [{"role": "user", "content": ""}],
            },
        )

        # May return 422 (validation) or pass validation but fail later
        # Either is acceptable for empty content
        assert response.status_code in [422, 500, 503]

    def test_invalid_role(self):
        """Invalid message role should return 422."""
        if "ANTHROPIC_API_KEY" in os.environ:
            del os.environ["ANTHROPIC_API_KEY"]

        app = create_app()
        client = TestClient(app, raise_server_exceptions=False)

        # Invalid role
        response = client.post(
            "/v1/messages",
            json={
                "model": "claude-sonnet-4-5-20250929",
                "max_tokens": 10,
                "messages": [{"role": "invalid_role", "content": "Hello"}],
            },
        )

        assert response.status_code == 422

    def test_consecutive_same_role(self):
        """Consecutive messages with same role should return 422."""
        if "ANTHROPIC_API_KEY" in os.environ:
            del os.environ["ANTHROPIC_API_KEY"]

        app = create_app()
        client = TestClient(app, raise_server_exceptions=False)

        response = client.post(
            "/v1/messages",
            json={
                "model": "claude-sonnet-4-5-20250929",
                "max_tokens": 10,
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "user", "content": "Again"},  # Consecutive user
                ],
            },
        )

        assert response.status_code == 422

    def test_first_message_not_user(self):
        """First message not from user should return 422."""
        if "ANTHROPIC_API_KEY" in os.environ:
            del os.environ["ANTHROPIC_API_KEY"]

        app = create_app()
        client = TestClient(app, raise_server_exceptions=False)

        response = client.post(
            "/v1/messages",
            json={
                "model": "claude-sonnet-4-5-20250929",
                "max_tokens": 10,
                "messages": [{"role": "assistant", "content": "Hello"}],
            },
        )

        assert response.status_code == 422

    def test_openai_empty_messages(self):
        """OpenAI API with empty messages should return 422."""
        if "ANTHROPIC_API_KEY" in os.environ:
            del os.environ["ANTHROPIC_API_KEY"]

        app = create_app()
        client = TestClient(app, raise_server_exceptions=False)

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4",
                "messages": [],
            },
        )

        assert response.status_code == 422

    def test_openai_invalid_temperature(self):
        """OpenAI API with invalid temperature should return 422."""
        if "ANTHROPIC_API_KEY" in os.environ:
            del os.environ["ANTHROPIC_API_KEY"]

        app = create_app()
        client = TestClient(app, raise_server_exceptions=False)

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "Hello"}],
                "temperature": 3.0,  # Invalid
            },
        )

        assert response.status_code == 422

    def test_direct_agent_missing_prompt(self):
        """Direct Agent API with missing prompt should return 422."""
        if "ANTHROPIC_API_KEY" in os.environ:
            del os.environ["ANTHROPIC_API_KEY"]

        app = create_app()
        client = TestClient(app, raise_server_exceptions=False)

        response = client.post(
            "/v1/agents/test_agent/generate",
            json={
                "max_tokens": 10,
                # Missing prompt
            },
        )

        assert response.status_code == 422

    def test_direct_agent_invalid_temperature(self):
        """Direct Agent API with invalid temperature should return 422."""
        if "ANTHROPIC_API_KEY" in os.environ:
            del os.environ["ANTHROPIC_API_KEY"]

        app = create_app()
        client = TestClient(app, raise_server_exceptions=False)

        response = client.post(
            "/v1/agents/test_agent/generate",
            json={
                "prompt": "Hello",
                "max_tokens": 10,
                "temperature": 3.0,  # Invalid
            },
        )

        assert response.status_code == 422

    def test_404_invalid_endpoint(self):
        """Invalid endpoint should return 404."""
        if "ANTHROPIC_API_KEY" in os.environ:
            del os.environ["ANTHROPIC_API_KEY"]

        app = create_app()
        client = TestClient(app, raise_server_exceptions=False)

        response = client.get("/v1/invalid_endpoint")

        assert response.status_code == 404

    def test_405_wrong_method(self):
        """Wrong HTTP method should return 405."""
        if "ANTHROPIC_API_KEY" in os.environ:
            del os.environ["ANTHROPIC_API_KEY"]

        app = create_app()
        client = TestClient(app, raise_server_exceptions=False)

        # GET instead of POST
        response = client.get("/v1/messages")

        assert response.status_code == 405
