# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Yakov Shkolnikov and contributors
"""TRT subprocess adapter.

Manages the ``llm_inference`` binary via ``subprocess.Popen`` with NDJSON
over stdin/stdout for control commands and safetensors temp files in
``/dev/shm`` for KV cache data transfer.

The subprocess is persistent (not per-request) — started once, reused
for all inference calls, and stopped on shutdown.
"""

import contextlib
import json
import os
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Any

import numpy as np
import structlog
from numpy.typing import NDArray
from safetensors.numpy import load_file, save_file

from agent_memory.domain.entities import BLOCK_SIZE_TOKENS
from agent_memory.domain.errors import TRTEngineError, TRTSubprocessError
from agent_memory.domain.value_objects import GenerationResult, ModelCacheSpec

logger = structlog.get_logger(__name__)


class TRTSubprocessAdapter:
    r"""Adapter wrapping ``llm_inference`` binary via subprocess.

    Protocol (NDJSON over stdin/stdout):
        Request:  ``{"cmd": "generate", "tokens": [...], "max_tokens": N, ...}\n``
        Response: ``{"text": "...", "tokens": [...], "finish_reason": "stop"}\n``

    KV cache injection/extraction uses safetensors files in shared memory.
    """

    def __init__(
        self,
        llm_inference_bin: str,
        engine_path: str,
        timeout_s: float = 30.0,
        shm_dir: str = "/dev/shm",  # noqa: S108
    ) -> None:
        """Initialize adapter.

        Args:
            llm_inference_bin: Path to the llm_inference binary.
            engine_path: Path to the TRT engine directory.
            timeout_s: Timeout for subprocess commands.
            shm_dir: Shared memory directory for temp files.
        """
        self._bin = llm_inference_bin
        self._engine_path = engine_path
        self._timeout_s = timeout_s
        self._shm_dir = Path(shm_dir)
        self._process: subprocess.Popen[bytes] | None = None
        self._lock = threading.Lock()  # Serialize NDJSON command/response pairs

    def start(self) -> None:
        """Start the llm_inference subprocess."""
        if self._process is not None and self._process.poll() is None:
            logger.warning("subprocess_already_running")
            return

        cmd = [self._bin, "--engine-path", self._engine_path, "--mode", "interactive"]
        try:
            self._process = subprocess.Popen(  # noqa: S603
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except FileNotFoundError as e:
            raise TRTEngineError(f"llm_inference binary not found: {self._bin}") from e
        except OSError as e:
            raise TRTEngineError(f"Failed to start subprocess: {e}") from e

        logger.info("trt_subprocess_started", pid=self._process.pid)

        # Wait for ready signal
        resp = self._read_response()
        if resp.get("status") != "ready":
            raise TRTEngineError(f"Subprocess did not report ready: {resp}")

    def stop(self) -> None:
        """Stop the subprocess gracefully."""
        if self._process is None:
            return

        with contextlib.suppress(TRTSubprocessError):
            self._send_command({"cmd": "shutdown"})

        try:
            self._process.terminate()
            self._process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait()
        finally:
            # Close file handles to avoid ResourceWarning
            for pipe in (self._process.stdin, self._process.stdout, self._process.stderr):
                if pipe is not None:
                    with contextlib.suppress(OSError):
                        pipe.close()
            logger.info("trt_subprocess_stopped")
            self._process = None

    def generate(
        self,
        prompt_tokens: list[int],
        cache: list[Any] | None = None,
        max_tokens: int = 256,
        temperature: float = 0.7,
        messages: list[dict[str, str]] | None = None,
        top_p: float = 0.95,
        top_k: int = 40,
        stop_sequences: list[str] | None = None,
        repetition_penalty: float = 1.0,
        **kwargs: Any,  # noqa: ARG002
    ) -> GenerationResult:
        """Generate text via TRT subprocess.

        Args:
            prompt_tokens: Pre-tokenized input (used as fallback text).
            cache: Optional KV cache (list of per-layer (K,V) tuples).
                When provided, injected into the engine before generation.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            messages: Chat messages for the engine's tokenizer. If not
                provided, prompt_tokens are passed as raw token IDs.
            top_p: Top-p (nucleus) sampling parameter.
            top_k: Top-k sampling parameter.
            stop_sequences: Stop strings for early output truncation.
            repetition_penalty: Repetition penalty (1.0 = no penalty).

        Returns:
            GenerationResult with text, tokens, and updated cache.
        """
        self._ensure_running()

        with self._lock:
            # Inject prior KV cache if provided
            inject_path = None
            if cache is not None:
                inject_path = self._write_cache_to_shm(cache)
                self._send_command({"cmd": "inject_cache", "input_path": str(inject_path)})
                inject_resp = self._read_response()
                inject_path.unlink(missing_ok=True)
                if "error" in inject_resp:
                    raise TRTEngineError(f"Cache inject failed: {inject_resp['error']}")

            # Build generate command
            extract_path = self._shm_dir / f"kv_out_{os.getpid()}.safetensors"
            cmd: dict[str, Any] = {
                "cmd": "generate",
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "extract_cache": True,
                "kv_cache_path": str(extract_path),
            }

            if stop_sequences:
                cmd["stop_sequences"] = stop_sequences
            if repetition_penalty != 1.0:
                cmd["repetition_penalty"] = repetition_penalty

            if messages is not None:
                cmd["messages"] = messages
            else:
                cmd["tokens"] = prompt_tokens

            self._send_command(cmd)
            resp = self._read_response()

        if "error" in resp:
            raise TRTEngineError(f"Generation failed: {resp['error']}")

        # Read extracted cache from safetensors
        updated_cache: list[tuple[Any, Any]] = []
        cache_file = Path(resp.get("kv_cache_path", str(extract_path)))
        if cache_file.exists():
            updated_cache = self._read_cache_from_shm(cache_file)
            cache_file.unlink(missing_ok=True)

        return GenerationResult(
            text=resp.get("text", ""),
            tokens=resp.get("tokens", []),
            cache=updated_cache,
        )

    def extract_model_spec(self) -> ModelCacheSpec:
        """Query subprocess for model geometry.

        Returns:
            ModelCacheSpec with kv_format='fp' and kv_bits=None (FP16).
        """
        self._ensure_running()
        self._send_command({"cmd": "get_model_spec"})
        resp = self._read_response()

        if "error" in resp:
            raise TRTEngineError(f"Failed to get model spec: {resp['error']}")

        n_layers = resp["n_layers"]
        return ModelCacheSpec(
            n_layers=n_layers,
            n_kv_heads=resp["n_kv_heads"],
            head_dim=resp["head_dim"],
            block_tokens=resp.get("block_tokens", BLOCK_SIZE_TOKENS),
            layer_types=["global"] * n_layers,
            kv_format="fp",
            kv_bits=None,
        )

    def _ensure_running(self) -> None:
        """Check subprocess is alive, restart if needed."""
        if self._process is None or self._process.poll() is not None:
            exit_code = self._process.returncode if self._process else None
            raise TRTSubprocessError(
                f"Subprocess not running (exit code: {exit_code}). Call start() first."
            )

    def _send_command(self, cmd: dict[str, Any]) -> None:
        """Send NDJSON command to subprocess stdin."""
        if self._process is None or self._process.stdin is None:
            raise TRTSubprocessError("Subprocess stdin not available")

        line = json.dumps(cmd) + "\n"
        try:
            self._process.stdin.write(line.encode())
            self._process.stdin.flush()
        except (BrokenPipeError, OSError) as e:
            raise TRTSubprocessError(f"Failed to send command: {e}") from e

    def _read_response(self) -> dict[str, Any]:
        """Read one NDJSON line from subprocess stdout."""
        if self._process is None or self._process.stdout is None:
            raise TRTSubprocessError("Subprocess stdout not available")

        try:
            line = self._process.stdout.readline()
        except OSError as e:
            raise TRTSubprocessError(f"Failed to read response: {e}") from e

        if not line:
            stderr = ""
            if self._process.stderr:
                stderr = self._process.stderr.read().decode(errors="replace")
            raise TRTSubprocessError(
                f"Subprocess produced no output (may have crashed). stderr: {stderr}"
            )

        try:
            return json.loads(line)  # type: ignore[no-any-return]
        except json.JSONDecodeError as e:
            raise TRTSubprocessError(f"Invalid JSON from subprocess: {line!r}") from e

    def _write_cache_to_shm(
        self,
        cache: list[Any],
    ) -> Path:
        """Write KV cache to a temp safetensors file in shared memory."""
        tensors: dict[str, NDArray[np.float16]] = {}
        for layer_idx, (k, v) in enumerate(cache):
            tensors[f"L{layer_idx}_K"] = np.asarray(k, dtype=np.float16)
            tensors[f"L{layer_idx}_V"] = np.asarray(v, dtype=np.float16)

        fd, path_str = tempfile.mkstemp(
            suffix=".safetensors",
            dir=str(self._shm_dir),
        )
        os.close(fd)
        path = Path(path_str)
        save_file(tensors, str(path))
        return path

    def _read_cache_from_shm(self, path: Path) -> list[tuple[Any, Any]]:
        """Read KV cache from safetensors file in shared memory."""
        tensors = load_file(str(path))
        cache: list[tuple[Any, Any]] = []
        layer_idx = 0
        while f"L{layer_idx}_K" in tensors:
            k = tensors[f"L{layer_idx}_K"]
            v = tensors[f"L{layer_idx}_V"]
            cache.append((k, v))
            layer_idx += 1
        return cache
