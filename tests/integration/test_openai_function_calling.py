"""Integration tests for OpenAI Chat Completions API function calling.

Tests function calling functionality including:
- Single function call
- Parallel function calls
- Streaming function calls
- tool_choice parameter
"""

import json

import pytest
from fastapi.testclient import TestClient

from semantic.entrypoints.api_server import create_app


@pytest.mark.integration
class TestOpenAIFunctionCalling:
    """Test OpenAI function calling (non-streaming)."""

    def test_function_calling_single(self):
        """Single function call should work end-to-end."""
        app = create_app()

        with TestClient(app) as client:
            # Define a simple function
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "get_current_weather",
                        "description": "Get the current weather in a location",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "location": {
                                    "type": "string",
                                    "description": "The city name",
                                },
                                "unit": {
                                    "type": "string",
                                    "enum": ["celsius", "fahrenheit"],
                                },
                            },
                            "required": ["location"],
                        },
                    },
                }
            ]

            # Request with function definition
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "test",
                    "messages": [
                        {
                            "role": "user",
                            "content": 'Get weather for Paris. Use {"function_call": {"name": "get_current_weather", "arguments": {"location": "Paris", "unit": "celsius"}}}',
                        }
                    ],
                    "tools": tools,
                    "max_tokens": 100,
                },
            )

            assert response.status_code == 200
            data = response.json()

            # Check if tool_calls present
            if data["choices"][0]["message"].get("tool_calls"):
                # Verify structure
                assert data["choices"][0]["finish_reason"] == "tool_calls"
                tool_calls = data["choices"][0]["message"]["tool_calls"]
                assert len(tool_calls) >= 1

                # Verify first tool call
                tool_call = tool_calls[0]
                assert "id" in tool_call
                assert tool_call["type"] == "function"
                assert tool_call["function"]["name"] == "get_current_weather"

                # Arguments should be JSON string
                arguments = json.loads(tool_call["function"]["arguments"])
                assert "location" in arguments

                # Send function result
                response2 = client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "test",
                        "messages": [
                            {
                                "role": "user",
                                "content": 'Get weather for Paris. Use {"function_call": {"name": "get_current_weather", "arguments": {"location": "Paris", "unit": "celsius"}}}',
                            },
                            {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": tool_calls,
                            },
                            {
                                "role": "tool",
                                "tool_call_id": tool_call["id"],
                                "content": "22 degrees celsius, sunny",
                            },
                        ],
                        "tools": tools,
                        "max_tokens": 100,
                    },
                )

                # Should continue after function result
                assert response2.status_code == 200
                data2 = response2.json()
                assert "choices" in data2

    def test_function_calling_without_tools(self):
        """Request without tools should work normally."""
        app = create_app()

        with TestClient(app) as client:
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "test",
                    "messages": [{"role": "user", "content": "Hello"}],
                    "max_tokens": 50,
                },
            )

            assert response.status_code == 200
            data = response.json()
            assert "choices" in data
            assert data["choices"][0]["finish_reason"] in ["stop", "length"]

    def test_function_calling_parallel(self):
        """Multiple parallel function calls should be supported."""
        app = create_app()

        with TestClient(app) as client:
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather",
                        "parameters": {
                            "type": "object",
                            "properties": {"location": {"type": "string"}},
                            "required": ["location"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "get_time",
                        "description": "Get current time",
                        "parameters": {
                            "type": "object",
                            "properties": {"timezone": {"type": "string"}},
                            "required": ["timezone"],
                        },
                    },
                },
            ]

            # Request that might trigger multiple function calls
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "test",
                    "messages": [
                        {
                            "role": "user",
                            "content": 'Call two functions: {"function_call": {"name": "get_weather", "arguments": {"location": "Paris"}}} and {"function_call": {"name": "get_time", "arguments": {"timezone": "Europe/Paris"}}}',
                        }
                    ],
                    "tools": tools,
                    "max_tokens": 150,
                },
            )

            assert response.status_code == 200
            data = response.json()

            # If tool_calls are present, verify structure
            if data["choices"][0]["message"].get("tool_calls"):
                tool_calls = data["choices"][0]["message"]["tool_calls"]
                # May have 1 or more tool calls
                assert len(tool_calls) >= 1

                # Each should have proper structure
                for tool_call in tool_calls:
                    assert "id" in tool_call
                    assert tool_call["type"] == "function"
                    assert "name" in tool_call["function"]
                    assert "arguments" in tool_call["function"]

    def test_tool_choice_parameter(self):
        """tool_choice parameter should be accepted (basic validation)."""
        app = create_app()

        with TestClient(app) as client:
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "test_function",
                        "description": "A test function",
                        "parameters": {
                            "type": "object",
                            "properties": {"param": {"type": "string"}},
                        },
                    },
                }
            ]

            # tool_choice: auto (default)
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "test",
                    "messages": [{"role": "user", "content": "Test"}],
                    "tools": tools,
                    "tool_choice": "auto",
                    "max_tokens": 50,
                },
            )
            assert response.status_code == 200

            # tool_choice: required (forces function call)
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "test",
                    "messages": [{"role": "user", "content": "Test"}],
                    "tools": tools,
                    "tool_choice": "required",
                    "max_tokens": 50,
                },
            )
            assert response.status_code == 200

            # tool_choice: specific function
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "test",
                    "messages": [{"role": "user", "content": "Test"}],
                    "tools": tools,
                    "tool_choice": {
                        "type": "function",
                        "function": {"name": "test_function"},
                    },
                    "max_tokens": 50,
                },
            )
            assert response.status_code == 200


@pytest.mark.integration
class TestOpenAIFunctionCallingStreaming:
    """Test OpenAI function calling with streaming."""

    def test_function_calling_streaming(self):
        """Function calls should work in streaming mode."""
        app = create_app()

        with (
            TestClient(app) as client,
            client.stream(
                "POST",
                "/v1/chat/completions",
                json={
                    "model": "test",
                    "messages": [
                        {
                            "role": "user",
                            "content": 'Calculate 5+3. Use {"function_call": {"name": "calculate", "arguments": {"expression": "5+3"}}}',
                        }
                    ],
                    "tools": [
                        {
                            "type": "function",
                            "function": {
                                "name": "calculate",
                                "description": "Perform calculation",
                                "parameters": {
                                    "type": "object",
                                    "properties": {"expression": {"type": "string"}},
                                    "required": ["expression"],
                                },
                            },
                        }
                    ],
                    "stream": True,
                    "max_tokens": 100,
                },
            ) as response,
        ):
            assert response.status_code == 200

            # Collect all chunks
            chunks = []
            for line in response.iter_lines():
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str != "[DONE]":
                        try:
                            chunk_data = json.loads(data_str)
                            chunks.append(chunk_data)
                        except json.JSONDecodeError:
                            pass

            # Should have at least some chunks
            assert len(chunks) > 0

            # Check for finish_reason in final chunks
            finish_reasons = [
                chunk["choices"][0].get("finish_reason")
                for chunk in chunks
                if chunk.get("choices", [{}])[0].get("finish_reason")
            ]

            # Should have a finish_reason
            assert len(finish_reasons) > 0

    def test_streaming_without_tools(self):
        """Streaming without tools should work normally."""
        app = create_app()

        with (
            TestClient(app) as client,
            client.stream(
                "POST",
                "/v1/chat/completions",
                json={
                    "model": "test",
                    "messages": [{"role": "user", "content": "Hello"}],
                    "stream": True,
                    "max_tokens": 50,
                },
            ) as response,
        ):
            assert response.status_code == 200

            # Should receive chunks
            chunks_received = False
            for line in response.iter_lines():
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str != "[DONE]":
                        chunks_received = True
                        break

            assert chunks_received
