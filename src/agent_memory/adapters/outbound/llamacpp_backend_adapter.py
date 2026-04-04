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
        --cache-type-k q8_0 --cache-type-v q8_0 \\
        -np 2 --ctx-size 131072
"""

import json
import re
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

    @property
    def _is_gemma(self) -> bool:
        return "gemma" in self._model_id.lower()

    @property
    def _is_qwen(self) -> bool:
        return "qwen" in self._model_id.lower()

    def _apply_no_think(self, messages: list[dict], disable_thinking: bool | None = None) -> list[dict]:
        """Suppress thinking for models that support it.

        - Qwen3: Prepend /no_think to the system message.
        - Gemma 4: Uses chat_template_kwargs (handled in request body, not here).

        Args:
            messages: Chat messages to modify.
            disable_thinking: Per-request override. Defaults to instance setting.
        """
        should_disable = disable_thinking if disable_thinking is not None else self._disable_thinking
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
            session_id: Optional session ID for slot pinning.
            openai_tools: OpenAI-format tool definitions for native function calling.

        Returns:
            GenerationResult with text, token count, and tool_calls if the
            model chose to call a tool.
        """
        if not messages:
            messages = [{"role": "user", "content": "Hello"}]

        messages = self._apply_no_think(messages, disable_thinking=disable_thinking)
        logger.info("llamacpp generate: disable_thinking=%s n_msgs=%d",
                    disable_thinking, len(messages))

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

        # NOTE: We do NOT pass tools to llama-server. The Jinja chat template
        # already injects tool definitions into the prompt. Passing tools to
        # llama-server activates its built-in tool parser, which chokes on
        # multiple tool calls. Instead we parse tool calls from text output.

        # Gemma 4: disable thinking via chat_template_kwargs
        if self._is_gemma and (disable_thinking if disable_thinking is not None else self._disable_thinking):
            body["chat_template_kwargs"] = {"enable_thinking": False}

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
        logger.info("llamacpp response: content=%r reasoning=%r tool_calls=%r tokens=%d",
                    text[:100] if text else None,
                    str(message.get("reasoning_content", ""))[:100],
                    message.get("tool_calls"),
                    completion_tokens)

        # Extract tool calls from text output (model generates JSON per template)
        # Also check OpenAI tool_calls in case llama-server parsed them natively
        tool_calls: list[dict] | None = None
        raw_tool_calls = message.get("tool_calls")
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
        elif text:
            # Parse tool calls from model text output (JSON objects with name+parameters)
            parsed = self._extract_tool_calls_from_text(text)
            if parsed:
                tool_calls = parsed
                # Remove tool call JSON from the text content
                text = self._strip_tool_calls_from_text(text)

        return GenerationResult(
            text=text,
            tokens=list(range(completion_tokens)),
            cache=[],  # llama.cpp manages its own KV cache
            tool_calls=tool_calls,
        )

    # ── Tool call extraction from text ──────────────────────────
    #
    # Three formats handled:
    #   1. JSON (template-instructed): {"name": "tool", "parameters": {...}}
    #   2. Gemma native: call:ToolName{key: "value"}  (concatenated, no newlines)
    #   3. Qwen native: <tool_call>{"name": "tool", "arguments": {...}}</tool_call>

    # Gemma: call:Name{...} or call:namespace:Name {...}
    # Captures the last colon-separated segment as the tool name
    _GEMMA_CALL_RE = re.compile(r'call:([\w:]+)')

    # Qwen: <tool_call>...</tool_call>
    _QWEN_CALL_RE = re.compile(r'<tool_call>\s*(.*?)\s*</tool_call>', re.DOTALL)

    def _extract_tool_calls_from_text(self, text: str) -> list[dict] | None:
        """Extract tool calls from model text output."""
        results: list[dict] = []

        # --- Format 1: JSON lines — {"name": ..., "parameters": ...}
        for line in text.split("\n"):
            line = line.strip()
            if not line or not line.startswith("{"):
                continue
            try:
                obj = json.loads(line)
                if "name" in obj and ("parameters" in obj or "arguments" in obj):
                    results.append({
                        "id": f"toolu_{uuid.uuid4().hex[:24]}",
                        "name": obj["name"],
                        "input": obj.get("parameters") or obj.get("arguments", {}),
                    })
            except json.JSONDecodeError:
                continue
        if results:
            return results

        # --- Format 2: Gemma native — call:Name{...}call:Name{...}
        if "call:" in text:
            results = self._parse_gemma_native_calls(text)
            if results:
                return results

        # --- Format 3: Qwen native — <tool_call>JSON</tool_call>
        for match in self._QWEN_CALL_RE.finditer(text):
            try:
                obj = json.loads(match.group(1))
                name = obj.get("name", "")
                args = obj.get("arguments") or obj.get("parameters", {})
                if isinstance(args, str):
                    args = json.loads(args)
                if name:
                    results.append({
                        "id": f"toolu_{uuid.uuid4().hex[:24]}",
                        "name": name,
                        "input": args,
                    })
            except (json.JSONDecodeError, TypeError):
                continue
        return results or None

    def _parse_gemma_native_calls(self, text: str) -> list[dict]:
        """Parse Gemma's call:Name{...} format with balanced brace matching.

        Handles variants:
          - call:ToolName{key: "val"}         (simple)
          - call:ns:ToolName{key: "val"}      (namespace prefix)
          - call:ns:ToolName {key: "val"}     (space before brace)
          - call:A{...}call:B{...}            (concatenated)
        """
        results = []
        # Find all call: positions
        for m in self._GEMMA_CALL_RE.finditer(text):
            raw_name = m.group(1)
            # Use the last colon-separated segment as tool name
            # e.g. "agent_memory:Agent" → "Agent"
            name = raw_name.rsplit(":", 1)[-1] if ":" in raw_name else raw_name
            # Find the opening brace (may have whitespace/newline between name and {)
            rest = text[m.end():]
            rest_stripped = rest.lstrip(" \t\n")
            if not rest_stripped or rest_stripped[0] != "{":
                continue
            raw_args = self._extract_balanced_braces(rest_stripped)
            if raw_args is None:
                continue
            parsed = self._parse_jslike_object(raw_args)
            if parsed is not None:
                results.append({
                    "id": f"toolu_{uuid.uuid4().hex[:24]}",
                    "name": name,
                    "input": parsed,
                })
        return results

    @staticmethod
    def _extract_balanced_braces(s: str) -> str | None:
        """Extract content within balanced { } from start of string."""
        if not s or s[0] != "{":
            return None
        depth = 0
        for i, ch in enumerate(s):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return s[: i + 1]
        return None  # Unbalanced

    @staticmethod
    def _parse_jslike_object(raw: str) -> dict | None:
        """Parse JS-like object syntax into a Python dict.

        Gemma outputs {key: "value"} — not valid JSON. Attempts:
        1. Direct JSON parse
        2. Quote unquoted keys, fix single quotes
        3. Manual key-value extraction (flat only)
        """
        # Direct JSON
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

        # Fix unquoted keys and brackets: {key: "val"} → {"key": "val"}
        # Also handle nested arrays: [{key: "val"}] → [{"key": "val"}]
        fixed = re.sub(r'(?<=[{,\[])\s*(\w+)\s*:', r' "\1":', raw)
        fixed = fixed.replace("'", '"')
        try:
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass

        # Flat key-value extraction (last resort)
        try:
            result = {}
            for kv in re.finditer(r'(\w+)\s*:\s*(?:"([^"]*)"|([\w.+\-/]+))', raw):
                key = kv.group(1)
                val = kv.group(2) if kv.group(2) is not None else kv.group(3)
                result[key] = val
            return result if result else None
        except Exception:
            return None

    def _strip_tool_calls_from_text(self, text: str) -> str:
        """Remove tool calls from text (all formats)."""
        # Strip Gemma native: call:Name{...} (with optional namespace and whitespace)
        # Rebuild text by removing matched call: regions
        result_chars = list(text)
        # Mark regions to remove (reverse order to preserve indices)
        regions_to_remove = []
        for m in self._GEMMA_CALL_RE.finditer(text):
            rest = text[m.end():]
            rest_stripped = rest.lstrip(" \t\n")
            skip = len(rest) - len(rest_stripped)
            balanced = self._extract_balanced_braces(rest_stripped)
            if balanced:
                start = m.start()
                end = m.end() + skip + len(balanced)
                regions_to_remove.append((start, end))
        for start, end in reversed(regions_to_remove):
            result_chars[start:end] = []
        text = "".join(result_chars)

        # Strip Qwen native: <tool_call>...</tool_call>
        text = self._QWEN_CALL_RE.sub("", text)

        # Strip JSON tool call lines
        lines = []
        for line in text.split("\n"):
            stripped = line.strip()
            if stripped.startswith("{"):
                try:
                    obj = json.loads(stripped)
                    if "name" in obj and ("parameters" in obj or "arguments" in obj):
                        continue
                except json.JSONDecodeError:
                    pass
            lines.append(line)
        return "\n".join(lines).strip()

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
        if self._is_gemma and self._disable_thinking:
            body["chat_template_kwargs"] = {"enable_thinking": False}
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
