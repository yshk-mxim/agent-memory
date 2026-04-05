# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Yakov Shkolnikov and contributors
"""Unit tests for TRT subprocess adapter — mock subprocess, JSON protocol, errors."""

import json
from io import BytesIO
from unittest.mock import MagicMock, Mock, patch

import pytest

from agent_memory.adapters.outbound.trt_subprocess_adapter import TRTSubprocessAdapter
from agent_memory.domain.errors import TRTEngineError, TRTSubprocessError

pytestmark = pytest.mark.unit


def _make_stdout(*lines: dict) -> BytesIO:
    """Create a BytesIO mimicking subprocess stdout with NDJSON lines."""
    data = b""
    for line in lines:
        data += json.dumps(line).encode() + b"\n"
    return BytesIO(data)


class TestStart:
    def test_start_reads_ready(self) -> None:
        adapter = TRTSubprocessAdapter(
            llm_inference_bin="/fake/bin",
            engine_path="/fake/engine",
        )

        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.stdout = _make_stdout({"status": "ready"})
        mock_proc.stdin = BytesIO()
        mock_proc.stderr = BytesIO()

        with patch("subprocess.Popen", return_value=mock_proc):
            adapter.start()

        assert adapter._process is mock_proc

    def test_start_raises_on_bad_ready(self) -> None:
        adapter = TRTSubprocessAdapter(
            llm_inference_bin="/fake/bin",
            engine_path="/fake/engine",
        )

        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.stdout = _make_stdout({"status": "error", "msg": "bad engine"})
        mock_proc.stdin = BytesIO()
        mock_proc.stderr = BytesIO()

        with patch("subprocess.Popen", return_value=mock_proc):
            with pytest.raises(TRTEngineError, match="did not report ready"):
                adapter.start()

    def test_start_raises_on_missing_binary(self) -> None:
        adapter = TRTSubprocessAdapter(
            llm_inference_bin="/nonexistent/bin",
            engine_path="/fake/engine",
        )

        with patch("subprocess.Popen", side_effect=FileNotFoundError("not found")):
            with pytest.raises(TRTEngineError, match="not found"):
                adapter.start()


class TestStop:
    def test_stop_terminates_process(self) -> None:
        adapter = TRTSubprocessAdapter(
            llm_inference_bin="/fake/bin",
            engine_path="/fake/engine",
        )

        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.stdin = BytesIO()
        mock_proc.stdout = _make_stdout({"status": "shutdown"})

        adapter._process = mock_proc
        adapter.stop()

        mock_proc.terminate.assert_called_once()
        assert adapter._process is None

    def test_stop_noop_when_not_started(self) -> None:
        adapter = TRTSubprocessAdapter(
            llm_inference_bin="/fake/bin",
            engine_path="/fake/engine",
        )
        adapter.stop()  # Should not raise


class TestGenerate:
    def test_generate_sends_correct_command(self) -> None:
        adapter = TRTSubprocessAdapter(
            llm_inference_bin="/fake/bin",
            engine_path="/fake/engine",
        )

        stdin_buf = BytesIO()
        response = {
            "text": "Hello world",
            "tokens": [101, 102, 103],
            "finish_reason": "stop",
        }

        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.stdin = stdin_buf
        mock_proc.stdout = _make_stdout(response)

        adapter._process = mock_proc

        result = adapter.generate(
            prompt_tokens=[1, 2, 3],
            max_tokens=10,
            temperature=0.5,
        )

        assert result.text == "Hello world"
        assert result.tokens == [101, 102, 103]

        # Verify command was sent correctly
        stdin_buf.seek(0)
        sent = json.loads(stdin_buf.read().decode())
        assert sent["cmd"] == "generate"
        assert sent["tokens"] == [1, 2, 3]
        assert sent["max_tokens"] == 10

    def test_generate_raises_on_error_response(self) -> None:
        adapter = TRTSubprocessAdapter(
            llm_inference_bin="/fake/bin",
            engine_path="/fake/engine",
        )

        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.stdin = BytesIO()
        mock_proc.stdout = _make_stdout({"error": "OOM"})

        adapter._process = mock_proc

        with pytest.raises(TRTEngineError, match="OOM"):
            adapter.generate(prompt_tokens=[1, 2, 3])

    def test_generate_raises_when_not_running(self) -> None:
        adapter = TRTSubprocessAdapter(
            llm_inference_bin="/fake/bin",
            engine_path="/fake/engine",
        )

        with pytest.raises(TRTSubprocessError, match="not running"):
            adapter.generate(prompt_tokens=[1, 2, 3])


class TestExtractModelSpec:
    def test_extract_spec(self) -> None:
        adapter = TRTSubprocessAdapter(
            llm_inference_bin="/fake/bin",
            engine_path="/fake/engine",
        )

        spec_response = {
            "n_layers": 32,
            "n_kv_heads": 8,
            "head_dim": 128,
            "block_tokens": 256,
        }

        stdin_buf = BytesIO()
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.stdin = stdin_buf
        mock_proc.stdout = _make_stdout(spec_response)

        adapter._process = mock_proc

        spec = adapter.extract_model_spec()

        assert spec.n_layers == 32
        assert spec.n_kv_heads == 8
        assert spec.head_dim == 128
        assert spec.kv_format == "fp"
        assert spec.kv_bits is None
        assert spec.layer_types == ["global"] * 32


class TestProtocolEdgeCases:
    def test_empty_stdout_raises(self) -> None:
        adapter = TRTSubprocessAdapter(
            llm_inference_bin="/fake/bin",
            engine_path="/fake/engine",
        )

        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.stdin = BytesIO()
        mock_proc.stdout = BytesIO(b"")  # Empty
        mock_proc.stderr = BytesIO(b"segfault")

        adapter._process = mock_proc

        with pytest.raises(TRTSubprocessError, match="no output"):
            adapter._read_response()

    def test_invalid_json_raises(self) -> None:
        adapter = TRTSubprocessAdapter(
            llm_inference_bin="/fake/bin",
            engine_path="/fake/engine",
        )

        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.stdin = BytesIO()
        mock_proc.stdout = BytesIO(b"not json\n")

        adapter._process = mock_proc

        with pytest.raises(TRTSubprocessError, match="Invalid JSON"):
            adapter._read_response()

    def test_broken_pipe_on_send(self) -> None:
        adapter = TRTSubprocessAdapter(
            llm_inference_bin="/fake/bin",
            engine_path="/fake/engine",
        )

        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.stdin = Mock()
        mock_proc.stdin.write.side_effect = BrokenPipeError("broken")

        adapter._process = mock_proc

        with pytest.raises(TRTSubprocessError, match="Failed to send"):
            adapter._send_command({"cmd": "test"})
