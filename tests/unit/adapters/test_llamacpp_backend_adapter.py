# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Yakov Shkolnikov and contributors
"""Unit tests for llama.cpp backend adapter — mock HTTP, slot API, errors."""

import json
from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest

from agent_memory.adapters.outbound.llamacpp_backend_adapter import LlamaCppBackendAdapter
from agent_memory.domain.errors import GenerationError

pytestmark = pytest.mark.unit


def _make_http_response(body: dict, status: int = 200) -> MagicMock:
    """Create a mock HTTP response with JSON body."""
    resp = MagicMock()
    resp.read.return_value = json.dumps(body).encode()
    resp.status = status
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    return resp


def _make_adapter(**kwargs) -> LlamaCppBackendAdapter:
    return LlamaCppBackendAdapter(
        base_url="http://localhost:8001",
        model_id="test-model",
        timeout_s=10.0,
        n_slots=4,
        **kwargs,
    )


class TestGenerate:
    def test_generate_returns_result(self) -> None:
        adapter = _make_adapter()
        mock_resp = _make_http_response({
            "choices": [{"message": {"content": "Hello world"}}],
            "usage": {"completion_tokens": 2},
        })

        with patch("agent_memory.adapters.outbound.llamacpp_backend_adapter.urlopen", return_value=mock_resp):
            result = adapter.generate(
                prompt_tokens=[],
                messages=[{"role": "user", "content": "Hi"}],
            )

        assert result.text == "Hello world"
        assert result.tokens == [0, 1]
        assert result.cache == []

    def test_generate_sends_cache_prompt_true(self) -> None:
        adapter = _make_adapter()
        mock_resp = _make_http_response({
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"completion_tokens": 1},
        })

        with patch("agent_memory.adapters.outbound.llamacpp_backend_adapter.urlopen", return_value=mock_resp) as mock_urlopen:
            adapter.generate(
                prompt_tokens=[],
                messages=[{"role": "user", "content": "test"}],
            )

        # Verify cache_prompt is in the request body
        call_args = mock_urlopen.call_args
        req = call_args[0][0]
        body = json.loads(req.data)
        assert body["cache_prompt"] is True

    def test_generate_pins_slot_from_session_id(self) -> None:
        adapter = _make_adapter()
        mock_resp = _make_http_response({
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"completion_tokens": 1},
        })

        with patch("agent_memory.adapters.outbound.llamacpp_backend_adapter.urlopen", return_value=mock_resp) as mock_urlopen:
            adapter.generate(
                prompt_tokens=[],
                messages=[{"role": "user", "content": "test"}],
                session_id="agent-123",
            )

        body = json.loads(mock_urlopen.call_args[0][0].data)
        # Slot assignment is now handled by llama-server (auto-picks best slot
        # for prompt cache reuse), not forced via id_slot in request body.
        assert "id_slot" not in body

    def test_generate_includes_stop_sequences(self) -> None:
        adapter = _make_adapter()
        mock_resp = _make_http_response({
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"completion_tokens": 1},
        })

        with patch("agent_memory.adapters.outbound.llamacpp_backend_adapter.urlopen", return_value=mock_resp) as mock_urlopen:
            adapter.generate(
                prompt_tokens=[],
                messages=[{"role": "user", "content": "test"}],
                stop_sequences=["<|endoftext|>", "\n\n"],
            )

        body = json.loads(mock_urlopen.call_args[0][0].data)
        assert body["stop"] == ["<|endoftext|>", "\n\n"]

    def test_generate_raises_on_no_choices(self) -> None:
        adapter = _make_adapter()
        mock_resp = _make_http_response({"choices": []})

        with patch("agent_memory.adapters.outbound.llamacpp_backend_adapter.urlopen", return_value=mock_resp):
            with pytest.raises(GenerationError, match="no choices"):
                adapter.generate(prompt_tokens=[])

    def test_generate_raises_on_timeout(self) -> None:
        adapter = _make_adapter()

        with patch(
            "agent_memory.adapters.outbound.llamacpp_backend_adapter.urlopen",
            side_effect=TimeoutError("timed out"),
        ):
            with pytest.raises(GenerationError, match="request failed"):
                adapter.generate(prompt_tokens=[])

    def test_generate_default_messages(self) -> None:
        adapter = _make_adapter()
        mock_resp = _make_http_response({
            "choices": [{"message": {"content": "Hi there"}}],
            "usage": {"completion_tokens": 2},
        })

        with patch("agent_memory.adapters.outbound.llamacpp_backend_adapter.urlopen", return_value=mock_resp) as mock_urlopen:
            adapter.generate(prompt_tokens=[])

        body = json.loads(mock_urlopen.call_args[0][0].data)
        assert body["messages"] == [
            {"role": "system", "content": "/no_think"},
            {"role": "user", "content": "Hello"},
        ]


class TestGenerateStream:
    def test_stream_yields_chunks(self) -> None:
        adapter = _make_adapter()

        sse_data = (
            b'data: {"choices":[{"delta":{"content":"Hello"}}]}\n'
            b'\n'
            b'data: {"choices":[{"delta":{"content":" world"}}]}\n'
            b'\n'
            b'data: [DONE]\n'
        )
        mock_resp = MagicMock()
        mock_resp.__iter__ = lambda self: iter(sse_data.split(b"\n"))

        with patch("agent_memory.adapters.outbound.llamacpp_backend_adapter.urlopen", return_value=mock_resp):
            chunks = list(adapter.generate_stream(
                messages=[{"role": "user", "content": "Hi"}],
            ))

        assert len(chunks) == 2
        assert chunks[0]["choices"][0]["delta"]["content"] == "Hello"
        assert chunks[1]["choices"][0]["delta"]["content"] == " world"

    def test_stream_raises_on_connection_error(self) -> None:
        adapter = _make_adapter()

        with patch(
            "agent_memory.adapters.outbound.llamacpp_backend_adapter.urlopen",
            side_effect=ConnectionError("refused"),
        ):
            with pytest.raises(GenerationError, match="stream request failed"):
                list(adapter.generate_stream(
                    messages=[{"role": "user", "content": "Hi"}],
                ))


class TestExtractModelSpec:
    def test_returns_spec_on_healthy_server(self) -> None:
        adapter = _make_adapter()
        mock_resp = _make_http_response({"status": "ok"})

        with patch("agent_memory.adapters.outbound.llamacpp_backend_adapter.urlopen", return_value=mock_resp):
            spec = adapter.extract_model_spec()

        assert spec.n_layers == 64
        assert spec.kv_format == "fp"
        assert spec.kv_bits is None

    def test_raises_on_health_failure(self) -> None:
        adapter = _make_adapter()

        with patch(
            "agent_memory.adapters.outbound.llamacpp_backend_adapter.urlopen",
            side_effect=TimeoutError("unreachable"),
        ):
            with pytest.raises(GenerationError, match="health check failed"):
                adapter.extract_model_spec()


class TestSlotSave:
    def test_save_slot_calls_correct_url(self) -> None:
        adapter = _make_adapter()
        mock_resp = _make_http_response({
            "id_slot": 0,
            "filename": "session-abc.bin",
            "n_saved": 512,
            "n_written": 40000000,
        })

        with patch("agent_memory.adapters.outbound.llamacpp_backend_adapter.urlopen", return_value=mock_resp) as mock_urlopen:
            result = adapter.save_slot(0, "session-abc.bin")

        req = mock_urlopen.call_args[0][0]
        assert "slots/0?action=save" in req.full_url
        body = json.loads(req.data)
        assert body["filename"] == "session-abc.bin"
        assert result["n_saved"] == 512

    def test_save_slot_raises_on_error(self) -> None:
        adapter = _make_adapter()

        with patch(
            "agent_memory.adapters.outbound.llamacpp_backend_adapter.urlopen",
            side_effect=TimeoutError("timeout"),
        ):
            with pytest.raises(GenerationError, match="slot save failed"):
                adapter.save_slot(0, "test.bin")


class TestSlotRestore:
    def test_restore_slot_calls_correct_url(self) -> None:
        adapter = _make_adapter()
        mock_resp = _make_http_response({
            "id_slot": 1,
            "filename": "session-abc.bin",
            "n_restored": 512,
            "n_read": 40000000,
        })

        with patch("agent_memory.adapters.outbound.llamacpp_backend_adapter.urlopen", return_value=mock_resp) as mock_urlopen:
            result = adapter.restore_slot(1, "session-abc.bin")

        req = mock_urlopen.call_args[0][0]
        assert "slots/1?action=restore" in req.full_url
        assert result["n_restored"] == 512

    def test_restore_slot_raises_on_error(self) -> None:
        adapter = _make_adapter()

        with patch(
            "agent_memory.adapters.outbound.llamacpp_backend_adapter.urlopen",
            side_effect=TimeoutError("timeout"),
        ):
            with pytest.raises(GenerationError, match="slot restore failed"):
                adapter.restore_slot(0, "test.bin")


class TestSlotErase:
    def test_erase_slot_calls_correct_url(self) -> None:
        adapter = _make_adapter()
        mock_resp = _make_http_response({
            "id_slot": 0,
            "n_erased": 1024,
        })

        with patch("agent_memory.adapters.outbound.llamacpp_backend_adapter.urlopen", return_value=mock_resp) as mock_urlopen:
            result = adapter.erase_slot(0)

        req = mock_urlopen.call_args[0][0]
        assert "slots/0?action=erase" in req.full_url
        assert result["n_erased"] == 1024
