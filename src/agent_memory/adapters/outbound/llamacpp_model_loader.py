# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Yakov Shkolnikov and contributors
"""llama.cpp model loader — manages llama-server subprocess lifecycle.

Implements ``ModelLoaderPort`` so the existing ``ModelRegistry`` and
``ModelSwapOrchestrator`` infrastructure can swap llama.cpp models
exactly like MLX models.

No llama.cpp code changes required.  The loader starts/stops the
standard llama-server binary with different GGUF files.

Architecture (hexagonal):
    ModelLoaderPort  ←  LlamaCppModelLoader  →  llama-server subprocess
                                              →  config/models/*.toml
"""

import json
import logging
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

from agent_memory.domain.errors import GenerationError

logger = logging.getLogger(__name__)

# Health check polling
_HEALTH_POLL_INTERVAL_S = 0.5
_HEALTH_TIMEOUT_S = 60.0


class LlamaCppModelLoader:
    """Manages llama-server subprocess lifecycle.

    Implements ``ModelLoaderPort`` protocol:
    - ``load_model(model_id)`` → start llama-server with GGUF, return (adapter, tokenizer)
    - ``unload_model()`` (via ``clear_cache``) → stop llama-server
    - ``get_active_memory()`` → estimate from GGUF file size
    - ``clear_cache()`` → kill process (memory freed by OS)

    The loader looks up model configuration from TOML files in
    ``config/models/``. Each TOML has a ``[llamacpp]`` section with
    ``gguf_path``, ``tokenizer_id``, ``ctx_size``, ``n_slots``.
    """

    def __init__(
        self,
        server_binary: str = "llama-server",
        port: int = 8001,
        cache_type_k: str = "q8_0",
        cache_type_v: str = "q8_0",
        timeout_s: float = 120.0,
        slot_save_path: str = "~/.agent_memory/llamacpp_slots",
    ) -> None:
        self._server_binary = server_binary
        self._port = port
        self._cache_type_k = cache_type_k
        self._cache_type_v = cache_type_v
        self._timeout_s = timeout_s
        self._slot_save_path = slot_save_path
        self._process: subprocess.Popen | None = None
        self._current_model_id: str | None = None
        self._current_n_slots: int = 0
        self._gguf_size_bytes: int = 0

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self._port}"

    @property
    def is_running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    @property
    def current_n_slots(self) -> int:
        """Number of slots configured for the currently loaded model."""
        return self._current_n_slots

    # ── ModelLoaderPort ─────────────────────────────────────────

    def load_model(self, model_id: str) -> tuple[Any, Any]:
        """Start llama-server with the GGUF for ``model_id``.

        Looks up model config from ``config/models/*.toml``, starts
        the server subprocess, waits for health, then returns
        a ``(LlamaCppBackendAdapter, tokenizer)`` tuple.

        Args:
            model_id: Model identifier matching a TOML config name
                (e.g. "gemma-4-26b-a4b").

        Returns:
            Tuple of (LlamaCppBackendAdapter, AutoTokenizer).

        Raises:
            ModelNotFoundError: If config or GGUF not found.
            GenerationError: If server fails to start.
        """
        from agent_memory.adapters.config.settings import load_model_profile
        from agent_memory.adapters.outbound.llamacpp_backend_adapter import (
            LlamaCppBackendAdapter,
        )
        from agent_memory.domain.errors import ModelNotFoundError

        # Stop existing server if running
        if self.is_running:
            self._stop_server()

        # Load model config from TOML
        profile = load_model_profile(model_id=model_id)
        llamacpp_cfg = profile.get("llamacpp", {})

        gguf_path = str(Path(llamacpp_cfg.get("gguf_path", "")).expanduser())
        if not gguf_path or not Path(gguf_path).exists():
            raise ModelNotFoundError(
                f"GGUF not found for {model_id}: {gguf_path}. Check config/models/ TOML files."
            )

        tokenizer_id = llamacpp_cfg.get("tokenizer_id", model_id)
        ctx_size = llamacpp_cfg.get("ctx_size", 131072)
        n_slots = llamacpp_cfg.get("n_slots", 4)
        n_gpu_layers = llamacpp_cfg.get("n_gpu_layers", 99)
        extra_args = llamacpp_cfg.get("extra_args", [])

        # Expand ~ in extra_args that are file paths (e.g. --model-draft ~/models/...)
        _path_flags = {"--model-draft", "-md"}
        extra_args = list(extra_args)
        for i, arg in enumerate(extra_args):
            if i > 0 and extra_args[i - 1] in _path_flags:
                extra_args[i] = str(Path(arg).expanduser())

        # Chat template override (enables better tool calling)
        chat_template_file = llamacpp_cfg.get("chat_template_file", "")
        if chat_template_file:
            # Resolve relative paths from project root
            tpl_path = Path(chat_template_file)
            if not tpl_path.is_absolute():
                tpl_path = Path(__file__).resolve().parents[4] / chat_template_file
            if tpl_path.exists():
                extra_args = list(extra_args) + ["--jinja", "--chat-template-file", str(tpl_path)]
                logger.info("using chat template: %s", tpl_path)

        self._gguf_size_bytes = Path(gguf_path).stat().st_size
        self._current_n_slots = n_slots

        # Start llama-server
        self._start_server(
            gguf_path=gguf_path,
            ctx_size=ctx_size,
            n_slots=n_slots,
            n_gpu_layers=n_gpu_layers,
            extra_args=extra_args,
        )
        self._current_model_id = model_id

        # Wait for health
        self._wait_for_health()

        # Create adapter (HTTP client to the running server)
        adapter = LlamaCppBackendAdapter(
            base_url=self.base_url,
            model_id=model_id,
            timeout_s=self._timeout_s,
            n_slots=n_slots,
        )

        # Load tokenizer from HuggingFace
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_id,
            trust_remote_code=True,
        )
        tokenizer.model_max_length = ctx_size

        logger.info(
            "llamacpp_model_loaded",
            extra={
                "model_id": model_id,
                "gguf": gguf_path,
                "ctx_size": ctx_size,
                "n_slots": n_slots,
                "port": self._port,
            },
        )

        return adapter, tokenizer

    def get_active_memory(self) -> int:
        """Estimate active memory from GGUF file size."""
        return self._gguf_size_bytes

    def clear_cache(self) -> None:
        """Stop llama-server — OS reclaims all memory."""
        self._stop_server()

    # ── Slot cache save/restore ─────────────────────────────────

    def save_all_slots(self, n_slots: int, model_id: str | None = None) -> int:
        """Save all active slot KV caches (e.g. before model swap or shutdown).

        Args:
            n_slots: Number of slots to save.
            model_id: Model ID for filename tagging. Falls back to current model.

        Returns:
            Number of slots successfully saved.
        """
        mid = model_id or self._current_model_id or "unknown"
        saved = 0
        for slot_id in range(n_slots):
            try:
                url = f"{self.base_url}/slots/{slot_id}?action=save"
                filename = f"{mid}_slot_{slot_id}.bin"
                data = json.dumps({"filename": filename}).encode()
                req = Request(url, data=data, headers={"Content-Type": "application/json"})
                with urlopen(req, timeout=10) as resp:  # noqa: S310
                    result = json.loads(resp.read())
                    if result.get("n_saved", 0) > 0:
                        saved += 1
            except Exception:
                # Slot may be empty — that's OK
                pass
        logger.info("saved %d/%d slot caches for %s", saved, n_slots, mid)
        return saved

    def restore_all_slots(self, n_slots: int, model_id: str | None = None) -> int:
        """Restore slot KV caches from disk (best effort).

        Args:
            n_slots: Number of slots to restore.
            model_id: Model ID for filename tagging. Falls back to current model.

        Returns:
            Number of slots successfully restored.
        """
        mid = model_id or self._current_model_id or "unknown"
        save_dir = Path(self._slot_save_path).expanduser()
        restored = 0
        for slot_id in range(n_slots):
            filename = f"{mid}_slot_{slot_id}.bin"
            slot_path = save_dir / filename
            if not slot_path.exists():
                continue
            try:
                url = f"{self.base_url}/slots/{slot_id}?action=restore"
                data = json.dumps({"filename": filename}).encode()
                req = Request(url, data=data, headers={"Content-Type": "application/json"})
                with urlopen(req, timeout=10) as resp:  # noqa: S310
                    result = json.loads(resp.read())
                    if result.get("n_restored", 0) > 0:
                        restored += 1
            except Exception:
                pass  # File may be stale or corrupt — OK
        logger.info("restored %d/%d slot caches for %s", restored, n_slots, mid)
        return restored

    # ── Process management ──────────────────────────────────────

    def _start_server(
        self,
        gguf_path: str,
        ctx_size: int = 131072,
        n_slots: int = 4,
        n_gpu_layers: int = 99,
        extra_args: list[str] | None = None,
    ) -> None:
        """Start llama-server subprocess."""
        binary = self._server_binary
        if not Path(binary).is_absolute():
            resolved = shutil.which(binary)
            if resolved is None:
                raise GenerationError(f"llama-server binary not found: {binary}")
            binary = resolved

        # Ensure slot save directory exists
        slot_save_dir = Path(self._slot_save_path).expanduser()
        slot_save_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            binary,
            "-m",
            gguf_path,
            "--port",
            str(self._port),
            "--ctx-size",
            str(ctx_size),
            "-np",
            str(n_slots),
            "-ngl",
            str(n_gpu_layers),
            "--cache-type-k",
            self._cache_type_k,
            "--cache-type-v",
            self._cache_type_v,
            "--slot-save-path",
            str(slot_save_dir),
            "--log-disable",
            # flash attention (-fa on) comes from extra_args in model TOML config
        ]
        if extra_args:
            cmd.extend(extra_args)

        logger.info("starting llama-server: %s", " ".join(cmd))

        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            preexec_fn=os.setsid,  # noqa: PLW1509 — own process group for clean kill
        )

    def _stop_server(self) -> None:
        """Stop llama-server gracefully.

        Kills the entire process group, which frees both the main model
        and any draft model (spec decode) loaded via --model-draft.
        No separate cleanup needed — OS reclaims all memory on process exit.
        """
        if self._process is None:
            return

        if self._process.poll() is not None:
            self._process = None
            return

        logger.info("stopping llama-server (pid=%d)", self._process.pid)

        # SIGTERM → wait 5s → SIGKILL
        try:
            os.killpg(os.getpgid(self._process.pid), signal.SIGTERM)
            self._process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            logger.warning("llama-server did not exit, sending SIGKILL")
            os.killpg(os.getpgid(self._process.pid), signal.SIGKILL)
            self._process.wait(timeout=3)
        except ProcessLookupError:
            pass  # Already dead

        self._process = None
        self._current_model_id = None
        self._current_n_slots = 0
        logger.info("llama-server stopped")

    def _wait_for_health(self) -> None:
        """Poll /health until llama-server is ready."""
        url = f"{self.base_url}/health"
        deadline = time.monotonic() + _HEALTH_TIMEOUT_S

        while time.monotonic() < deadline:
            # Check process hasn't crashed
            if self._process and self._process.poll() is not None:
                stderr = ""
                if self._process.stderr:
                    stderr = self._process.stderr.read().decode(errors="replace")[-500:]
                raise GenerationError(
                    f"llama-server exited with code {self._process.returncode}: {stderr}"
                )

            try:
                req = Request(url)  # noqa: S310
                with urlopen(req, timeout=2) as resp:  # noqa: S310
                    health = json.loads(resp.read())
                    if health.get("status") == "ok":
                        logger.info("llama-server healthy")
                        return
            except (URLError, TimeoutError, OSError):
                pass

            time.sleep(_HEALTH_POLL_INTERVAL_S)

        raise GenerationError(f"llama-server failed to become healthy within {_HEALTH_TIMEOUT_S}s")
