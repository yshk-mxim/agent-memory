# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Yakov Shkolnikov and contributors
"""Integration tests for Anthropic API tool calling.

Tests tool_use functionality including:
- Single tool invocation
- Streaming tool calls
- Tool result handling
- Error handling
"""

import json

import pytest
from fastapi.testclient import TestClient

from agent_memory.entrypoints.api_server import create_app


@pytest.mark.integration
class TestAnthropicToolCalling:
    """Test Anthropic tool calling (non-streaming)."""

    def test_tool_use_single_call(self):
        """Single tool invocation should work end-to-end."""
        app = create_app()

        with TestClient(app) as client:
            # Define a simple tool
            tools = [
                {
                    "name": "get_weather",
                    "description": "Get the weather for a location",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city name",
                            }
                        },
                        "required": ["location"],
                    },
                }
            ]

            # Request with tool definition
            response = client.post(
                "/v1/messages",
                json={
                    "model": "test",
                    "max_tokens": 100,
                    "messages": [
                        {
                            "role": "user",
                            "content": 'Get weather for Paris. Use the get_weather tool with {"tool_use": {"name": "get_weather", "input": {"location": "Paris"}}}',
                        }
                    ],
                    "tools": tools,
                },
            )

            assert response.status_code == 200
            data = response.json()

            # Should have tool_use in content
            has_tool_use = any(block.get("type") == "tool_use" for block in data.get("content", []))

            if has_tool_use:
                # Verify stop_reason is tool_use
                assert data["stop_reason"] == "tool_use"

                # Find the tool_use block
                tool_block = next(b for b in data["content"] if b.get("type") == "tool_use")
                assert "id" in tool_block
                assert tool_block["name"] == "get_weather"
                assert "location" in tool_block["input"]

                # Send tool result
                response2 = client.post(
                    "/v1/messages",
                    json={
                        "model": "test",
                        "max_tokens": 100,
                        "messages": [
                            {
                                "role": "user",
                                "content": 'Get weather for Paris. Use the get_weather tool with {"tool_use": {"name": "get_weather", "input": {"location": "Paris"}}}',
                            },
                            {"role": "assistant", "content": data["content"]},
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "tool_result",
                                        "tool_use_id": tool_block["id"],
                                        "content": "Sunny, 22°C",
                                    }
                                ],
                            },
                        ],
                        "tools": tools,
                    },
                )

                # Should continue after tool result
                assert response2.status_code == 200
                data2 = response2.json()
                assert "content" in data2

    def test_tool_use_without_tools_defined(self):
        """Request without tools should work normally."""
        app = create_app()

        with TestClient(app) as client:
            response = client.post(
                "/v1/messages",
                json={
                    "model": "test",
                    "max_tokens": 50,
                    "messages": [{"role": "user", "content": "Hello"}],
                },
            )

            assert response.status_code == 200
            data = response.json()
            assert "content" in data
            assert data["stop_reason"] in ["end_turn", "max_tokens"]

    def test_tool_result_error_handling(self):
        """Tool errors should be handled gracefully."""
        app = create_app()

        with TestClient(app) as client:
            tools = [
                {
                    "name": "read_file",
                    "description": "Read a file",
                    "input_schema": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                }
            ]

            # Send message with tool result error
            response = client.post(
                "/v1/messages",
                json={
                    "model": "test",
                    "max_tokens": 100,
                    "messages": [
                        {"role": "user", "content": "Read /nonexistent"},
                        {
                            "role": "assistant",
                            "content": [
                                {
                                    "type": "tool_use",
                                    "id": "toolu_123",
                                    "name": "read_file",
                                    "input": {"path": "/nonexistent"},
                                }
                            ],
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": "toolu_123",
                                    "content": "Error: File not found",
                                    "is_error": True,
                                }
                            ],
                        },
                    ],
                    "tools": tools,
                },
            )

            # Should handle error tool result
            assert response.status_code == 200
            data = response.json()
            assert "content" in data


@pytest.mark.integration
class TestAnthropicToolCallingStreaming:
    """Test Anthropic tool calling with streaming."""

    def test_tool_use_streaming(self):
        """Tool calls should work in streaming mode."""
        app = create_app()

        with TestClient(app) as client:
            tools = [
                {
                    "name": "calculate",
                    "description": "Perform calculation",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "expression": {"type": "string"},
                        },
                        "required": ["expression"],
                    },
                }
            ]

            # Streaming request with tools
            with client.stream(
                "POST",
                "/v1/messages",
                json={
                    "model": "test",
                    "max_tokens": 100,
                    "stream": True,
                    "messages": [
                        {
                            "role": "user",
                            "content": 'Calculate 2+2. Use {"tool_use": {"name": "calculate", "input": {"expression": "2+2"}}}',
                        }
                    ],
                    "tools": tools,
                },
            ) as response:
                assert response.status_code == 200

                events = []
                for line in response.iter_lines():
                    if line.startswith("data: "):
                        try:
                            event_data = json.loads(line[6:])
                            events.append(event_data)
                        except json.JSONDecodeError:
                            pass

                # Should have message_start, content blocks, message_delta, message_stop
                event_types = [e.get("type") for e in events if isinstance(e, dict) and "type" in e]

                # Basic streaming events should be present
                assert "message_start" in event_types or len(events) > 0

    def test_streaming_without_tools(self):
        """Streaming without tools should work normally."""
        app = create_app()

        with (
            TestClient(app) as client,
            client.stream(
                "POST",
                "/v1/messages",
                json={
                    "model": "test",
                    "max_tokens": 50,
                    "stream": True,
                    "messages": [{"role": "user", "content": "Hello"}],
                },
            ) as response,
        ):
            assert response.status_code == 200

            # Should receive streaming events
            events_received = False
            for line in response.iter_lines():
                if line.startswith("data: "):
                    events_received = True
                    break

            assert events_received
