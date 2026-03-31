# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Yakov Shkolnikov and contributors
"""llama.cpp backend adapter.

Implements ModelBackendPort by calling a llama-server's OpenAI-compatible
API over HTTP.  Adds slot-level KV cache save/restore via llama.cpp's
``/slots/{id}?action=save|restore|erase`` endpoints.

llama-server runs independently — agent-memory does not manage its
lifecycle.  Start it with::

    llama-server -m model.gguf --port 8001 \\
        --slot-save-path ~/.agent_memory/llamacpp_slots \\
        --cache-type-k q4_0 --cache-type-v q4_0 \\
        -np 4 --ctx-size 65536
"""

import json
import uuid
import logging
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from agent_memory.domain.errors import GenerationError
from agent_memory.domain.value_objects import GenerationResult, ModelCacheSpec

logger = logging.getLogger(__name__)


class LlamaCppBackendAdapter:
    """Adapter for llama-server via OpenAI-compatible HTTP API.

    Satisfies ModelBackendPort protocol.  Extends it with slot-level
    KV cache persistence unique to llama.cpp.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8001",
        model_id: str = "qwen3-coder-next",
        timeout_s: float = 120.0,
        n_slots: int = 4,
        disable_thinking: bool = True,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model_id = model_id
        self._timeout_s = timeout_s
        self._n_slots = n_slots
        self._disable_thinking = disable_thinking

    # ── Thinking suppression ────────────────────────────────────

    def _apply_no_think(self, messages: list[dict]) -> list[dict]:
        """Append /no_think to the last user message to disable Qwen3 thinking.

        Qwen3's chat template checks for /no_think in user turns to skip
        the <think>...</think> reasoning block. This works across all
        llama.cpp versions without needing chat_template_kwargs support.
        """
        if not self._disable_thinking or not messages:
            return messages
        messages = [m.copy() for m in messages]
        for i in reversed(range(len(messages))):
            if messages[i].get("role") == "user":
                content = messages[i].get("content", "")
                if isinstance(content, str) and "/no_think" not in content:
                    messages[i]["content"] = content + " /no_think"
                break
        return messages

    # ── ModelBackendPort: generate ──────────────────────────────

    def generate(
        self,
        prompt_tokens: list[int],  # noqa: ARG002
        cache: list[Any] | None = None,  # noqa: ARG002
        max_tokens: int = 256,
        temperature: float = 0.7,
        messages: list[dict[str, str]] | None = None,
        top_p: float = 0.95,
        top_k: int = 40,
        stop_sequences: list[str] | None = None,
        session_id: str | None = None,
        openai_tools: list[dict] | None = None,
    ) -> GenerationResult:
        """Generate text via llama-server's OpenAI-compatible API.

        Args:
            prompt_tokens: Not used (llama-server tokenizes from messages).
            cache: Not used (llama-server manages its own KV cache).
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            messages: OpenAI-format chat messages (already converted by caller).
            top_p: Top-p sampling.
            top_k: Top-k sampling.
            stop_sequences: Stop strings.
            session_id: Optional session ID for slot pinning.
            openai_tools: OpenAI-format tool definitions for native function calling.

        Returns:
            GenerationResult with text, token count, and tool_calls if the
            model chose to call a tool.
        """
        if not messages:
            messages = [{"role": "user", "content": "Hello"}]

        messages = self._apply_no_think(messages)

        body: dict[str, Any] = {
            "model": self._model_id,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "stream": False,
            "cache_prompt": True,
        }

        if stop_sequences:
            body["stop"] = stop_sequences

        if openai_tools:
            body["tools"] = openai_tools
            body["tool_choice"] = "auto"

        # Pin to a slot based on session ID for KV cache reuse
        if session_id and self._n_slots > 0:
            body["id_slot"] = hash(session_id) % self._n_slots

        url = f"{self._base_url}/v1/chat/completions"
        data = json.dumps(body).encode()
        req = Request(url, data=data, headers={"Content-Type": "application/json"})  # noqa: S310

        try:
            with urlopen(req, timeout=self._timeout_s) as resp:  # noqa: S310
                result = json.loads(resp.read())
        except HTTPError as e:
            body = e.read().decode(errors="replace")
            raise GenerationError(f"llama.cpp request failed: {e} — {body}") from e
        except (URLError, TimeoutError) as e:
            raise GenerationError(f"llama.cpp request failed: {e}") from e

        choices = result.get("choices", [])
        if not choices:
            raise GenerationError("llama.cpp returned no choices")

        message = choices[0].get("message", {})
        text = message.get("content") or ""
        usage = result.get("usage", {})
        completion_tokens = usage.get("completion_tokens", 0)

        # Convert OpenAI tool_calls → list[dict] for GenerationResult
        raw_tool_calls = message.get("tool_calls")
        tool_calls: list[dict] | None = None
        if raw_tool_calls:
            tool_calls = []
            for tc in raw_tool_calls:
                fn = tc.get("function", {})
                try:
                    arguments = json.loads(fn.get("arguments", "{}"))
                except json.JSONDecodeError:
                    arguments = {}
                tool_calls.append({
                    "id": tc.get("id", f"toolu_{uuid.uuid4().hex[:24]}"),
                    "name": fn.get("name", ""),
                    "input": arguments,
                })

        return GenerationResult(
            text=text,
            tokens=list(range(completion_tokens)),
            cache=[],  # llama.cpp manages its own KV cache
            tool_calls=tool_calls,
        )

    # ── Streaming ───────────────────────────────────────────────

    def generate_stream(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        stop_sequences: list[str] | None = None,
        session_id: str | None = None,
    ) -> Any:
        """Stream generation via llama-server SSE.

        Yields parsed JSON chunks from the streaming response.

        Args:
            messages: Chat messages.
            max_tokens: Maximum tokens.
            temperature: Sampling temperature.
            top_p: Top-p sampling.
            stop_sequences: Stop strings.
            session_id: Optional session ID for slot pinning.

        Yields:
            Parsed JSON chunks from llama-server's streaming response.
        """
        messages = self._apply_no_think(messages)

        body: dict[str, Any] = {
            "model": self._model_id,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": True,
            "cache_prompt": True,
        }

        if stop_sequences:
            body["stop"] = stop_sequences
        if session_id and self._n_slots > 0:
            body["id_slot"] = hash(session_id) % self._n_slots

        url = f"{self._base_url}/v1/chat/completions"
        data = json.dumps(body).encode()
        req = Request(url, data=data, headers={"Content-Type": "application/json"})  # noqa: S310

        try:
            resp = urlopen(req, timeout=self._timeout_s)  # noqa: S310
        except (HTTPError, URLError, TimeoutError, ConnectionError, OSError) as e:
            raise GenerationError(f"llama.cpp stream request failed: {e}") from e

        for line_bytes in resp:
            line = line_bytes.decode().strip()
            if not line or not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            try:
                yield json.loads(payload)
            except json.JSONDecodeError:
                continue

    # ── ModelBackendPort: extract_model_spec ────────────────────

    def extract_model_spec(self) -> ModelCacheSpec:
        """Query llama-server health and return approximate ModelCacheSpec.

        llama-server doesn't expose full model geometry via API,
        so we return reasonable defaults.  The spec is used by
        agent-memory's cache store for block sizing — with llama.cpp
        managing its own KV cache, these values are advisory.
        """
        url = f"{self._base_url}/health"
        req = Request(url)  # noqa: S310

        try:
            with urlopen(req, timeout=self._timeout_s) as resp:  # noqa: S310
                health = json.loads(resp.read())
                if health.get("status") != "ok":
                    logger.warning("llama-server not ready: %s", health)
        except (HTTPError, URLError, TimeoutError) as e:
            raise GenerationError(f"llama.cpp health check failed: {e}") from e

        # Advisory spec — llama.cpp manages KV cache internally
        return ModelCacheSpec(
            n_layers=64,
            n_kv_heads=2,
            head_dim=128,
            block_tokens=256,
            layer_types=["global"] * 64,
            kv_format="fp",
            kv_bits=None,
        )

    # ── Slot-level KV cache persistence ─────────────────────────

    def save_slot(self, slot_id: int, filename: str) -> dict[str, Any]:
        """Save a slot's KV cache to disk.

        Requires llama-server started with ``--slot-save-path``.

        Args:
            slot_id: Slot index (0-based).
            filename: Filename relative to --slot-save-path.

        Returns:
            Response dict with n_saved, n_written, timings.
        """
        return self._slot_action(slot_id, "save", filename)

    def restore_slot(self, slot_id: int, filename: str) -> dict[str, Any]:
        """Restore a slot's KV cache from disk.

        Args:
            slot_id: Slot index (0-based).
            filename: Filename relative to --slot-save-path.

        Returns:
            Response dict with n_restored, n_read, timings.
        """
        return self._slot_action(slot_id, "restore", filename)

    def erase_slot(self, slot_id: int) -> dict[str, Any]:
        """Erase a slot's KV cache.

        Args:
            slot_id: Slot index (0-based).

        Returns:
            Response dict with n_erased.
        """
        return self._slot_action(slot_id, "erase")

    def _slot_action(
        self,
        slot_id: int,
        action: str,
        filename: str | None = None,
    ) -> dict[str, Any]:
        """Execute a slot management action.

        Args:
            slot_id: Slot index.
            action: "save", "restore", or "erase".
            filename: Required for save/restore.

        Returns:
            Parsed JSON response.
        """
        url = f"{self._base_url}/slots/{slot_id}?action={action}"
        body = json.dumps({"filename": filename}).encode() if filename else b"{}"
        req = Request(  # noqa: S310
            url,
            data=body,
            headers={"Content-Type": "application/json"},
        )

        try:
            with urlopen(req, timeout=self._timeout_s) as resp:  # noqa: S310
                return json.loads(resp.read())
        except (HTTPError, URLError, TimeoutError) as e:
            raise GenerationError(
                f"llama.cpp slot {action} failed (slot {slot_id}): {e}"
            ) from e
