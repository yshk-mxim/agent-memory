# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Yakov Shkolnikov and contributors
"""vLLM backend adapter.

Implements ModelBackendPort by calling a vLLM server's OpenAI-compatible
API over HTTP. No subprocess management — vLLM runs independently
(e.g., Docker on Thor: `vllm serve nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4`).

KV cache is managed internally by vLLM (PagedAttention). agent-memory
provides session management, conversation persistence to disk, and
prompt prefix caching at the application layer.
"""

import json
import logging
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from agent_memory.domain.errors import GenerationError
from agent_memory.domain.value_objects import GenerationResult, ModelCacheSpec

logger = logging.getLogger(__name__)


class VLLMBackendAdapter:
    """Adapter for vLLM server via OpenAI-compatible HTTP API.

    Satisfies ModelBackendPort protocol.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:5000",
        model_id: str = "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4",
        timeout_s: float = 120.0,
    ) -> None:
        """Initialize vLLM backend adapter.

        Args:
            base_url: vLLM server URL.
            model_id: Model name for API requests.
            timeout_s: HTTP request timeout in seconds.
        """
        self._base_url = base_url.rstrip("/")
        self._model_id = model_id
        self._timeout_s = timeout_s

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
    ) -> GenerationResult:
        """Generate text via vLLM's OpenAI-compatible API.

        Args:
            prompt_tokens: Not used (vLLM tokenizes from messages).
            cache: Not used (vLLM manages its own KV cache).
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            messages: Chat messages.
            top_p: Top-p sampling.
            top_k: Top-k sampling.
            stop_sequences: Stop strings.

        Returns:
            GenerationResult with text and token count.
        """
        if not messages:
            messages = [{"role": "user", "content": "Hello"}]

        body: dict[str, Any] = {
            "model": self._model_id,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": False,
        }

        if stop_sequences:
            body["stop"] = stop_sequences
        if top_k > 0:
            body["extra_body"] = {"top_k": top_k}

        url = f"{self._base_url}/v1/chat/completions"
        data = json.dumps(body).encode()
        req = Request(url, data=data, headers={"Content-Type": "application/json"})  # noqa: S310

        try:
            with urlopen(req, timeout=self._timeout_s) as resp:  # noqa: S310
                result = json.loads(resp.read())
        except (HTTPError, URLError, TimeoutError) as e:
            raise GenerationError(f"vLLM request failed: {e}") from e

        choices = result.get("choices", [])
        if not choices:
            raise GenerationError("vLLM returned no choices")

        text = choices[0].get("message", {}).get("content", "")
        usage = result.get("usage", {})
        completion_tokens = usage.get("completion_tokens", 0)

        return GenerationResult(
            text=text,
            tokens=list(range(completion_tokens)),
            cache=[],  # vLLM manages its own KV cache
        )

    def generate_stream(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        stop_sequences: list[str] | None = None,
    ) -> Any:
        """Stream generation via vLLM SSE.

        Yields raw SSE chunks from vLLM for forwarding to clients.

        Args:
            messages: Chat messages.
            max_tokens: Maximum tokens.
            temperature: Sampling temperature.
            top_p: Top-p sampling.
            stop_sequences: Stop strings.

        Yields:
            Parsed JSON chunks from vLLM's streaming response.
        """
        body: dict[str, Any] = {
            "model": self._model_id,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": True,
        }

        if stop_sequences:
            body["stop"] = stop_sequences

        url = f"{self._base_url}/v1/chat/completions"
        data = json.dumps(body).encode()
        req = Request(url, data=data, headers={"Content-Type": "application/json"})  # noqa: S310

        try:
            resp = urlopen(req, timeout=self._timeout_s)  # noqa: S310
        except (HTTPError, URLError, TimeoutError) as e:
            raise GenerationError(f"vLLM stream request failed: {e}") from e

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

    def extract_model_spec(self) -> ModelCacheSpec:
        """Query vLLM for model info.

        Returns approximate ModelCacheSpec. vLLM doesn't expose full
        model geometry via API, so values are based on known Nemotron
        3 Super architecture.
        """
        url = f"{self._base_url}/v1/models"
        req = Request(url, headers={"Content-Type": "application/json"})  # noqa: S310

        try:
            with urlopen(req, timeout=self._timeout_s) as resp:  # noqa: S310
                json.loads(resp.read())  # Validate response is JSON
        except (HTTPError, URLError, TimeoutError) as e:
            raise GenerationError(f"vLLM models query failed: {e}") from e

        # Nemotron 3 Super 120B-A12B architecture
        return ModelCacheSpec(
            n_layers=80,
            n_kv_heads=8,
            head_dim=128,
            block_tokens=256,
            layer_types=["global"] * 80,
            kv_format="fp",
            kv_bits=None,
        )
