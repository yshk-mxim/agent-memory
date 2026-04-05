# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Yakov Shkolnikov and contributors
"""Unit tests for llama.cpp model loader — subprocess lifecycle, health, slots."""

import json
import signal
import subprocess
from unittest.mock import MagicMock, Mock, call, patch

import pytest

from agent_memory.adapters.outbound.llamacpp_model_loader import (
    LlamaCppModelLoader,
    _HEALTH_TIMEOUT_S,
)
from agent_memory.domain.errors import GenerationError, ModelNotFoundError

pytestmark = pytest.mark.unit


def _make_loader(**kwargs) -> LlamaCppModelLoader:
    return LlamaCppModelLoader(
        server_binary="/usr/local/bin/llama-server",
        port=8001,
        **kwargs,
    )


def _make_profile(gguf_path: str = "/models/test.gguf") -> dict:
    """Return a minimal model profile dict with llamacpp section."""
    return {
        "llamacpp": {
            "gguf_path": gguf_path,
            "tokenizer_id": "test-org/test-model",
            "ctx_size": 65536,
            "n_slots": 2,
            "n_gpu_layers": 99,
            "extra_args": [],
        },
    }


def _mock_health_response(status: str = "ok") -> MagicMock:
    """Create a mock urlopen context manager returning JSON health."""
    resp = MagicMock()
    resp.read.return_value = json.dumps({"status": status}).encode()
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    return resp


# ── load_model happy path ──────────────────────────────────────


class TestLoadModelHappyPath:
    """load_model starts server, waits for health, returns (adapter, tokenizer)."""

    @pytest.mark.skipif(
        "mlx" in __import__("sys").modules,
        reason="MLX mock conflict during test collection",
    )
    @patch("agent_memory.adapters.outbound.llamacpp_model_loader.urlopen")
    @patch("subprocess.Popen")
    def test_load_returns_adapter_and_tokenizer(
        self, mock_popen, mock_urlopen, tmp_path
    ) -> None:
        # Create a real GGUF file so Path.exists() passes
        gguf = tmp_path / "test.gguf"
        gguf.write_bytes(b"\x00" * 1024)

        profile = _make_profile(gguf_path=str(gguf))

        # Subprocess mock
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None  # still running
        mock_popen.return_value = mock_proc

        # Health check mock
        mock_urlopen.return_value = _mock_health_response("ok")

        loader = _make_loader()

        # Patch at source modules (imports are deferred inside load_model)
        with (
            patch(
                "agent_memory.adapters.config.settings.load_model_profile",
                return_value=profile,
            ),
            patch(
                "agent_memory.adapters.outbound.llamacpp_backend_adapter.LlamaCppBackendAdapter",
            ) as MockAdapter,
            patch(
                "transformers.AutoTokenizer",
            ) as MockTokenizer,
        ):
            MockAdapter.return_value = MagicMock(name="adapter")
            mock_tok = MagicMock(name="tokenizer")
            MockTokenizer.from_pretrained.return_value = mock_tok

            adapter, tokenizer = loader.load_model("test-model")

        # Verify subprocess was started
        mock_popen.assert_called_once()
        cmd = mock_popen.call_args[0][0]
        assert str(gguf) in cmd

        # Verify health check was polled
        mock_urlopen.assert_called()

        # Verify adapter and tokenizer returned
        assert adapter is not None
        assert tokenizer is mock_tok


# ── load_model with missing GGUF ───────────────────────────────


class TestLoadModelMissingGGUF:
    """load_model raises ModelNotFoundError when GGUF file is absent."""

    def test_missing_gguf_raises_model_not_found(self) -> None:
        profile = _make_profile(gguf_path="/nonexistent/path.gguf")
        loader = _make_loader()

        with patch(
            "agent_memory.adapters.config.settings.load_model_profile",
            return_value=profile,
        ):
            with pytest.raises(ModelNotFoundError, match="GGUF not found"):
                loader.load_model("bad-model")

    def test_empty_gguf_path_raises_model_not_found(self) -> None:
        profile = _make_profile(gguf_path="")
        loader = _make_loader()

        with patch(
            "agent_memory.adapters.config.settings.load_model_profile",
            return_value=profile,
        ):
            with pytest.raises(ModelNotFoundError, match="GGUF not found"):
                loader.load_model("bad-model")


# ── clear_cache ────────────────────────────────────────────────


class TestClearCache:
    """clear_cache delegates to _stop_server."""

    def test_clear_cache_calls_stop_server(self) -> None:
        loader = _make_loader()
        loader._stop_server = MagicMock()

        loader.clear_cache()

        loader._stop_server.assert_called_once()


# ── _stop_server ───────────────────────────────────────────────


class TestStopServer:
    """_stop_server sends SIGTERM then SIGKILL on timeout."""

    def test_stop_sends_sigterm_then_waits(self) -> None:
        loader = _make_loader()
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None  # still running
        mock_proc.pid = 12345
        mock_proc.wait.return_value = 0  # exits cleanly after SIGTERM
        loader._process = mock_proc

        with patch("os.killpg") as mock_killpg, patch("os.getpgid", return_value=12345):
            loader._stop_server()

        mock_killpg.assert_called_once_with(12345, signal.SIGTERM)
        mock_proc.wait.assert_called_once_with(timeout=5)
        assert loader._process is None

    def test_stop_sends_sigkill_on_timeout(self) -> None:
        loader = _make_loader()
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None  # still running
        mock_proc.pid = 12345
        # First wait times out, second (after SIGKILL) succeeds
        mock_proc.wait.side_effect = [
            subprocess.TimeoutExpired(cmd="llama-server", timeout=5),
            0,
        ]
        loader._process = mock_proc

        with patch("os.killpg") as mock_killpg, patch("os.getpgid", return_value=12345):
            loader._stop_server()

        # Should have sent both SIGTERM and SIGKILL
        assert mock_killpg.call_count == 2
        mock_killpg.assert_any_call(12345, signal.SIGTERM)
        mock_killpg.assert_any_call(12345, signal.SIGKILL)
        assert loader._process is None

    def test_stop_noop_when_no_process(self) -> None:
        loader = _make_loader()
        loader._process = None

        # Should not raise
        loader._stop_server()

    def test_stop_noop_when_already_exited(self) -> None:
        loader = _make_loader()
        mock_proc = MagicMock()
        mock_proc.poll.return_value = 0  # already exited
        loader._process = mock_proc

        loader._stop_server()

        assert loader._process is None


# ── _wait_for_health ───────────────────────────────────────────


class TestWaitForHealth:
    """_wait_for_health raises GenerationError on timeout."""

    @patch("agent_memory.adapters.outbound.llamacpp_model_loader.urlopen")
    @patch("agent_memory.adapters.outbound.llamacpp_model_loader.time")
    def test_timeout_raises_generation_error(self, mock_time, mock_urlopen) -> None:
        loader = _make_loader()
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None  # still running
        loader._process = mock_proc

        # Simulate time advancing past deadline on every call
        mock_time.monotonic.side_effect = [0.0, _HEALTH_TIMEOUT_S + 1.0]

        with pytest.raises(GenerationError, match="failed to become healthy"):
            loader._wait_for_health()

    @patch("agent_memory.adapters.outbound.llamacpp_model_loader.urlopen")
    @patch("agent_memory.adapters.outbound.llamacpp_model_loader.time")
    def test_process_crash_raises_generation_error(
        self, mock_time, mock_urlopen
    ) -> None:
        loader = _make_loader()
        mock_proc = MagicMock()
        mock_proc.poll.return_value = 1  # crashed
        mock_proc.returncode = 1
        mock_proc.stderr = MagicMock()
        mock_proc.stderr.read.return_value = b"segfault"
        loader._process = mock_proc

        # Time within deadline
        mock_time.monotonic.side_effect = [0.0, 1.0]

        with pytest.raises(GenerationError, match="exited with code 1"):
            loader._wait_for_health()


# ── save_all_slots ─────────────────────────────────────────────


class TestSaveAllSlots:
    """save_all_slots saves available slots, ignores errors on empty ones."""

    @patch("agent_memory.adapters.outbound.llamacpp_model_loader.urlopen")
    def test_saves_available_slots(self, mock_urlopen) -> None:
        loader = _make_loader()

        # Slot 0: saved, Slot 1: saved, Slot 2: empty (error), Slot 3: saved
        responses = []
        for n_saved in [512, 256, None, 128]:
            if n_saved is not None:
                resp = MagicMock()
                resp.read.return_value = json.dumps({"n_saved": n_saved}).encode()
                resp.__enter__ = lambda s: s
                resp.__exit__ = MagicMock(return_value=False)
                responses.append(resp)
            else:
                responses.append(Exception("slot empty"))

        mock_urlopen.side_effect = responses

        saved = loader.save_all_slots(n_slots=4)

        assert saved == 3
        assert mock_urlopen.call_count == 4

    @patch("agent_memory.adapters.outbound.llamacpp_model_loader.urlopen")
    def test_all_slots_empty_returns_zero(self, mock_urlopen) -> None:
        loader = _make_loader()
        mock_urlopen.side_effect = Exception("all empty")

        saved = loader.save_all_slots(n_slots=4)

        assert saved == 0

    @patch("agent_memory.adapters.outbound.llamacpp_model_loader.urlopen")
    def test_zero_saved_tokens_not_counted(self, mock_urlopen) -> None:
        loader = _make_loader()
        resp = MagicMock()
        resp.read.return_value = json.dumps({"n_saved": 0}).encode()
        resp.__enter__ = lambda s: s
        resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = resp

        saved = loader.save_all_slots(n_slots=2)

        assert saved == 0


# ── is_running property ────────────────────────────────────────


class TestIsRunning:
    """is_running reflects subprocess state."""

    def test_running_when_process_alive(self) -> None:
        loader = _make_loader()
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None  # still running
        loader._process = mock_proc

        assert loader.is_running is True

    def test_not_running_when_no_process(self) -> None:
        loader = _make_loader()
        loader._process = None

        assert loader.is_running is False

    def test_not_running_when_process_exited(self) -> None:
        loader = _make_loader()
        mock_proc = MagicMock()
        mock_proc.poll.return_value = 0  # exited
        loader._process = mock_proc

        assert loader.is_running is False


# ── --slot-save-path in startup command ──────────────────────────


class TestSlotSavePath:
    """_start_server passes --slot-save-path to llama-server."""

    @patch("subprocess.Popen")
    @patch("shutil.which", return_value="/usr/local/bin/llama-server")
    def test_slot_save_path_in_cmd(self, mock_which, mock_popen, tmp_path) -> None:
        slot_dir = str(tmp_path / "slots")
        loader = _make_loader(slot_save_path=slot_dir)

        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_popen.return_value = mock_proc

        loader._start_server(gguf_path="/models/test.gguf")

        cmd = mock_popen.call_args[0][0]
        assert "--slot-save-path" in cmd
        # The expanded path should be in the command
        idx = cmd.index("--slot-save-path")
        assert cmd[idx + 1] == slot_dir  # already expanded by Path.expanduser()

    @patch("subprocess.Popen")
    @patch("shutil.which", return_value="/usr/local/bin/llama-server")
    def test_slot_save_dir_created(self, mock_which, mock_popen, tmp_path) -> None:
        slot_dir = tmp_path / "deep" / "nested" / "slots"
        loader = _make_loader(slot_save_path=str(slot_dir))

        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_popen.return_value = mock_proc

        loader._start_server(gguf_path="/models/test.gguf")

        assert slot_dir.exists()


# ── restore_all_slots ────────────────────────────────────────────


class TestRestoreAllSlots:
    """restore_all_slots restores available slot files for a model."""

    @patch("agent_memory.adapters.outbound.llamacpp_model_loader.urlopen")
    def test_restores_existing_files(self, mock_urlopen, tmp_path) -> None:
        loader = _make_loader(slot_save_path=str(tmp_path))
        loader._current_model_id = "model-a"

        # Create slot files for slots 0 and 1
        (tmp_path / "model-a_slot_0.bin").write_bytes(b"\x00" * 100)
        (tmp_path / "model-a_slot_1.bin").write_bytes(b"\x00" * 100)

        resp = MagicMock()
        resp.read.return_value = json.dumps({"n_restored": 500}).encode()
        resp.__enter__ = lambda s: s
        resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = resp

        restored = loader.restore_all_slots(n_slots=4, model_id="model-a")

        assert restored == 2
        assert mock_urlopen.call_count == 2

    @patch("agent_memory.adapters.outbound.llamacpp_model_loader.urlopen")
    def test_skips_missing_files(self, mock_urlopen, tmp_path) -> None:
        loader = _make_loader(slot_save_path=str(tmp_path))

        # No slot files on disk
        restored = loader.restore_all_slots(n_slots=4, model_id="model-a")

        assert restored == 0
        mock_urlopen.assert_not_called()

    @patch("agent_memory.adapters.outbound.llamacpp_model_loader.urlopen")
    def test_handles_restore_failure(self, mock_urlopen, tmp_path) -> None:
        loader = _make_loader(slot_save_path=str(tmp_path))
        (tmp_path / "model-a_slot_0.bin").write_bytes(b"\x00" * 100)

        mock_urlopen.side_effect = Exception("restore failed")

        restored = loader.restore_all_slots(n_slots=4, model_id="model-a")

        assert restored == 0

    @patch("agent_memory.adapters.outbound.llamacpp_model_loader.urlopen")
    def test_uses_model_tagged_filenames(self, mock_urlopen, tmp_path) -> None:
        loader = _make_loader(slot_save_path=str(tmp_path))
        (tmp_path / "model-b_slot_0.bin").write_bytes(b"\x00" * 100)

        resp = MagicMock()
        resp.read.return_value = json.dumps({"n_restored": 100}).encode()
        resp.__enter__ = lambda s: s
        resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = resp

        restored = loader.restore_all_slots(n_slots=4, model_id="model-b")

        assert restored == 1
        # Verify model-tagged filename in request body
        req_arg = mock_urlopen.call_args[0][0]
        body = json.loads(req_arg.data)
        assert body["filename"] == "model-b_slot_0.bin"
