# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Yakov Shkolnikov and contributors
r"""llama.cpp backend adapter.

Implements ModelBackendPort by calling a llama-server's OpenAI-compatible
API over HTTP.  Adds slot-level KV cache save/restore via llama.cpp's
``/slots/{id}?action=save|restore|erase`` endpoints.

llama-server runs independently — agent-memory does not manage its
lifecycle.  Start it with::

    llama-server -m model.gguf --port 8001 \
        --slot-save-path ~/.agent_memory/llamacpp_slots \
        --cache-type-k q8_0 --cache-type-v q8_0 \
        -np 2 --ctx-size 131072
"""

import json
import logging
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from agent_memory.adapters.outbound.tool_call_parsers import create_parser_for_model
from agent_memory.adapters.outbound.tool_call_parsers.llama_server_native import (
    extract_from_openai_tool_calls,
)
from agent_memory.application.text_cleaning import strip_thinking_tags
from agent_memory.application.tool_call_parsing import ToolCallParserChain
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
        tool_parser: ToolCallParserChain | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model_id = model_id
        self._timeout_s = timeout_s
        self._n_slots = n_slots
        self._disable_thinking = disable_thinking
        self._tool_parser = tool_parser or create_parser_for_model(model_id)
        self.slot_tracker: Any = None  # Optional SlotTracker for usage tracking
        self._capture_path: str | None = None  # Set to enable traffic capture

    # ── Thinking suppression ────────────────────────────────────

    @property
    def _is_gemma(self) -> bool:
        return "gemma" in self._model_id.lower()

    @property
    def _is_qwen(self) -> bool:
        return "qwen" in self._model_id.lower()

    def _apply_no_think(
        self, messages: list[dict], disable_thinking: bool | None = None
    ) -> list[dict]:
        """Suppress thinking for models that support it.

        - Qwen3: Prepend /no_think to the system message.
        - Gemma 4: Uses --reasoning off server flag (not handled here).

        Args:
            messages: Chat messages to modify.
            disable_thinking: Per-request override. Defaults to instance setting.
        """
        should_disable = (
            disable_thinking if disable_thinking is not None else self._disable_thinking
        )
        if not should_disable or not messages or self._is_gemma:
            return messages
        messages = [m.copy() for m in messages]
        for i, msg in enumerate(messages):
            if msg.get("role") == "system":
                content = msg.get("content", "")
                if isinstance(content, str) and not content.startswith("/no_think"):
                    messages[i]["content"] = "/no_think\n" + content
                return messages
        # No system message — insert one at the front
        messages.insert(0, {"role": "system", "content": "/no_think"})
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
        disable_thinking: bool = True,
        model: str | None = None,  # noqa: ARG002 — used by router, ignored here
        **kwargs: Any,  # noqa: ARG002
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
            session_id: Optional session ID for slot tracking.
            openai_tools: OpenAI-format tool definitions for native function calling.

        Returns:
            GenerationResult with text, token count, and tool_calls if the
            model chose to call a tool.
        """
        if not messages:
            messages = [{"role": "user", "content": "Hello"}]

        messages = self._apply_no_think(messages, disable_thinking=disable_thinking)
        logger.info(
            "llamacpp generate: disable_thinking=%s n_msgs=%d", disable_thinking, len(messages)
        )

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

        # Pass tools to llama-server for grammar-constrained generation.
        # b8665+ has a dedicated Gemma 4 parser (PR #21418) that handles
        # parallel tool calls correctly. If native parsing fails or the
        # server is older, our text-based parser chain catches it below.
        if openai_tools:
            body["tools"] = openai_tools

        # Gemma 4 thinking suppression: handled by --reasoning off server flag
        # (chat_template_kwargs {"enable_thinking": false} is unreliable).

        # Let llama-server auto-assign the optimal slot. With --cache-prompt,
        # the server picks a slot with the longest matching prompt prefix,
        # which is strictly better than forcing a slot via hash (which causes
        # thrashing when Claude Code sends parallel requests).

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

        # Capture raw traffic for regression test fixtures
        if self._capture_path:
            self._write_capture(messages, text, message.get("tool_calls"), result.get("usage"))

        # Strip thinking/channel tags (canonical implementation)
        text = strip_thinking_tags(text)

        usage = result.get("usage", {})
        completion_tokens = usage.get("completion_tokens", 0)
        prompt_tokens_count = usage.get("prompt_tokens", 0)
        cached_tokens = usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)
        timings = result.get("timings", {})
        logger.info(
            "llamacpp response: content=%r tool_calls=%r gen=%d prompt=%d cached=%d "
            "prefill=%.0ft/s gen=%.1ft/s",
            text[:100] if text else None,
            message.get("tool_calls"),
            completion_tokens,
            prompt_tokens_count,
            cached_tokens,
            timings.get("prompt_per_second", 0),
            timings.get("predicted_per_second", 0),
        )

        # Extract tool calls: check native API field first, then parse from text
        tool_calls: list[dict] | None = None
        raw_tool_calls = message.get("tool_calls")
        if raw_tool_calls:
            parsed = extract_from_openai_tool_calls(raw_tool_calls)
            if parsed:
                tool_calls = [{"id": tc.id, "name": tc.name, "input": tc.input} for tc in parsed]
        elif text:
            remaining, parsed = self._tool_parser.parse(text)
            if parsed:
                tool_calls = [{"id": tc.id, "name": tc.name, "input": tc.input} for tc in parsed]
                text = remaining

        # Track slot usage for LRU-LFU slot persistence
        # Slot ID comes from llama-server response (auto-assigned), not from hash
        resp_slot_id = result.get("id_slot")
        if self.slot_tracker and session_id and resp_slot_id is not None:
            self.slot_tracker.mark_used(
                resp_slot_id, session_id, prompt_tokens_count + completion_tokens
            )

        return GenerationResult(
            text=text,
            tokens=list(range(completion_tokens)),
            cache=[],  # llama.cpp manages its own KV cache
            tool_calls=tool_calls,
        )

    # ── Traffic capture for regression tests ─────────────────────

    def enable_capture(self, path: str) -> None:
        """Enable raw traffic capture to a JSONL file.

        Each line is a JSON object with the messages sent to the model
        and the raw (pre-stripping) response. Use these as fixtures for
        parser regression tests.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._capture_path = path
        logger.info("traffic capture enabled: %s", path)

    def _write_capture(
        self,
        messages: list[dict],
        raw_content: str,
        raw_tool_calls: list | None,
        usage: dict | None,
    ) -> None:
        """Append one request/response pair to the capture file."""
        record = {
            "ts": time.time(),
            "model_id": self._model_id,
            "messages": messages,
            "raw_content": raw_content,
            "raw_tool_calls": raw_tool_calls,
            "usage": usage,
        }
        try:
            with open(self._capture_path, "a") as f:  # noqa: PTH123
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception:
            logger.warning("failed to write capture", exc_info=True)

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
            session_id: Optional session ID for slot tracking.

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
        # Gemma 4 thinking: handled by --reasoning off server flag
        # Slot assignment: let llama-server auto-pick (see generate() comment)

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
            raise GenerationError(f"llama.cpp slot {action} failed (slot {slot_id}): {e}") from e
