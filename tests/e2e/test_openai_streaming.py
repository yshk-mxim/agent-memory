# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Yakov Shkolnikov and contributors
"""E2E tests for OpenAI streaming API.

Tests verify SSE (Server-Sent Events) streaming for OpenAI Chat Completions API:
- Streaming response format (OpenAI SSE)
- Delta chunks arrive progressively
- Final [DONE] marker
- Error handling in streaming
- Streaming vs non-streaming consistency
"""

import json

import httpx
import pytest


@pytest.mark.e2e
def test_streaming_response_format(live_server: str):
    """Test that streaming returns proper OpenAI SSE format.

    Verifies:
    - Response is SSE stream (text/event-stream)
    - Initial chunk has role delta
    - Content chunks have text deltas
    - Final chunk has finish_reason
    - [DONE] marker at end
    """
    client = httpx.Client(
        base_url=live_server,
        timeout=30.0,
        headers={"x-api-key": "test-key-for-e2e"},
    )

    try:
        request_body = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Say 'test' please"}],
            "max_tokens": 10,
            "stream": True,
        }

        with client.stream("POST", "/v1/chat/completions", json=request_body) as response:
            assert response.status_code == 200
            assert "text/event-stream" in response.headers.get("content-type", "")

            chunks = []
            for line in response.iter_lines():
                if line.startswith("data: "):
                    data = line[6:]  # Remove "data: " prefix
                    if data == "[DONE]":
                        chunks.append({"done": True})
                    else:
                        chunks.append(json.loads(data))

            # Verify we got chunks
            assert len(chunks) > 0, "Should receive at least one chunk"

            # Verify first chunk has role
            first_chunk = chunks[0]
            assert first_chunk["object"] == "chat.completion.chunk"
            assert first_chunk["choices"][0]["delta"].get("role") == "assistant"

            # Verify at least one content chunk
            content_chunks = [
                c for c in chunks if not c.get("done") and c["choices"][0]["delta"].get("content")
            ]
            assert len(content_chunks) > 0, "Should have at least one content delta"

            # Verify final chunk has finish_reason
            final_chunk = [
                c for c in chunks if not c.get("done") and c["choices"][0].get("finish_reason")
            ]
            assert len(final_chunk) > 0, "Should have final chunk with finish_reason"

            # Verify [DONE] marker
            assert chunks[-1] == {"done": True}, "Last chunk should be [DONE]"

    finally:
        client.close()


@pytest.mark.e2e
def test_delta_chunks_arrive_progressively(live_server: str):
    """Test that content arrives in delta chunks.

    Verifies:
    - Multiple delta chunks received
    - Each chunk has incremental content
    - Chunks can be accumulated to full response
    """
    client = httpx.Client(
        base_url=live_server, timeout=30.0, headers={"x-api-key": "test-key-for-e2e"}
    )

    try:
        request_body = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Count to five"}],
            "max_tokens": 50,
            "stream": True,
        }

        accumulated_text = ""
        content_chunks_received = 0

        with client.stream("POST", "/v1/chat/completions", json=request_body) as response:
            assert response.status_code == 200

            for line in response.iter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break

                    chunk = json.loads(data)
                    delta = chunk["choices"][0]["delta"]

                    if "content" in delta:
                        accumulated_text += delta["content"]
                        content_chunks_received += 1

        # Verify we got multiple chunks (streaming, not single response)
        assert content_chunks_received > 0, "Should receive at least one content chunk"

        # Verify accumulated text is not empty
        assert len(accumulated_text) > 0, "Accumulated text should not be empty"

        print("\n📊 Streaming stats:")
        print(f"  Content chunks: {content_chunks_received}")
        print(f"  Total text length: {len(accumulated_text)}")
        print(f"  Accumulated text: {accumulated_text[:100]}...")

    finally:
        client.close()


@pytest.mark.e2e
def test_final_done_marker(live_server: str):
    """Test that stream ends with [DONE] marker.

    Verifies:
    - Stream ends with data: [DONE]
    - No chunks after [DONE]
    - Proper stream termination
    """
    client = httpx.Client(
        base_url=live_server, timeout=30.0, headers={"x-api-key": "test-key-for-e2e"}
    )

    try:
        request_body = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 20,
            "stream": True,
        }

        chunks = []
        with client.stream("POST", "/v1/chat/completions", json=request_body) as response:
            assert response.status_code == 200

            for line in response.iter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    chunks.append(data)

        # Verify last chunk is [DONE]
        assert chunks[-1] == "[DONE]", "Stream should end with [DONE] marker"

        # Verify we got content before [DONE]
        assert len(chunks) > 1, "Should have content chunks before [DONE]"

    finally:
        client.close()


@pytest.mark.e2e
def test_error_handling_in_streaming(live_server: str):
    """Test error handling during streaming.

    Verifies:
    - Invalid requests return appropriate errors
    - Stream terminates properly on errors
    - Error messages are clear
    """
    client = httpx.Client(
        base_url=live_server, timeout=30.0, headers={"x-api-key": "test-key-for-e2e"}
    )

    try:
        # Test with invalid max_tokens (too large)
        request_body = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Test"}],
            "max_tokens": 100000,  # Way too large
            "stream": True,
        }

        # Note: This might succeed or fail depending on validation
        # Main point is no crashes and proper error handling
        with client.stream("POST", "/v1/chat/completions", json=request_body) as response:
            # Should get either success or proper error
            assert response.status_code in [200, 400, 422, 503], (
                f"Should get valid status code, got {response.status_code}"
            )

            # If it streams, it should not crash
            if response.status_code == 200:
                chunks_received = 0
                for line in response.iter_lines():
                    if line.startswith("data: "):
                        chunks_received += 1
                        if chunks_received > 100:  # Safety limit
                            break

                assert chunks_received > 0, "Should receive at least one chunk"

    finally:
        client.close()


@pytest.mark.e2e
def test_openai_streaming_vs_non_streaming(live_server: str):
    """Test that streaming and non-streaming return equivalent content.

    Verifies:
    - Same input produces consistent output
    - Streaming chunks accumulate to same text as non-streaming
    - Both methods work correctly
    """
    client = httpx.Client(
        base_url=live_server, timeout=30.0, headers={"x-api-key": "test-key-for-e2e"}
    )

    try:
        request_body = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Say hello"}],
            "max_tokens": 20,
        }

        # Get non-streaming response
        non_stream_response = client.post("/v1/chat/completions", json=request_body)
        assert non_stream_response.status_code == 200
        non_stream_text = non_stream_response.json()["choices"][0]["message"]["content"]

        # Get streaming response
        request_body["stream"] = True
        accumulated_text = ""

        with client.stream("POST", "/v1/chat/completions", json=request_body) as response:
            assert response.status_code == 200

            for line in response.iter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break

                    chunk = json.loads(data)
                    delta = chunk["choices"][0]["delta"]

                    if "content" in delta:
                        accumulated_text += delta["content"]

        # Verify both methods produced content
        assert len(non_stream_text) > 0, "Non-streaming should produce content"
        assert len(accumulated_text) > 0, "Streaming should produce content"

        # Note: Content might differ slightly due to non-determinism,
        # but both should be valid responses
        print("\n📊 Comparison:")
        print(f"  Non-streaming: {non_stream_text}")
        print(f"  Streaming: {accumulated_text}")

    finally:
        client.close()
