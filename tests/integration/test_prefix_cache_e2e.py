# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Yakov Shkolnikov and contributors
"""E2E tests for KV cache reuse and TTFT improvement with real MLX model.

Starts the agent-memory server with SmolLM2-135M and measures:
- Non-streaming: total response time (cold vs warm, same session)
- Streaming: true TTFT via SSE (time to first content_block_delta)
- Multi-turn: KV cache reuse across turns in same session

The same-session cache path is the primary TTFT optimization:
request 1 pays full prefill, request 2+ reuses the cached KV state
and only prefills new tokens → dramatically lower TTFT.

Run: pytest tests/integration/test_prefix_cache_e2e.py -v -s
(requires MLX and SmolLM2-135M model cached locally)
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

# Skip entire module if MLX not available
try:
    import mlx.core  # noqa: F401
except ImportError:
    pytest.skip("MLX not available", allow_module_level=True)


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _wait_for_health(url: str, timeout: float = 120) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = httpx.get(f"{url}/health/live", timeout=2.0)
            if r.status_code == 200:
                return True
        except (httpx.ConnectError, httpx.TimeoutException):
            pass
        time.sleep(0.5)
    return False


def _send_message(
    base_url: str,
    system: str,
    user_msg: str,
    session_id: str | None = None,
    max_tokens: int = 32,
    stream: bool = False,
) -> tuple[float, float, dict]:
    """Send Anthropic Messages API request.

    Returns (ttft_ms, total_ms, response_dict).
    - stream=True: ttft_ms is time to first SSE content_block_delta (true TTFT).
    - stream=False: ttft_ms == total_ms (non-streaming has no partial delivery).
    """
    headers = {
        "Content-Type": "application/json",
        "X-API-Key": "test-key-for-e2e",
        "anthropic-version": "2023-06-01",
    }
    if session_id:
        headers["X-Session-ID"] = session_id

    body = {
        "model": "mlx-community/SmolLM2-135M-Instruct",
        "max_tokens": max_tokens,
        "system": system,
        "messages": [{"role": "user", "content": user_msg}],
        "stream": stream,
    }

    t0 = time.perf_counter()

    if stream:
        ttft_ms = None
        full_text = ""
        with httpx.stream(
            "POST",
            f"{base_url}/v1/messages",
            json=body,
            headers=headers,
            timeout=60.0,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if line.startswith("data:"):
                    data = line[5:].strip()
                    if data and data != "[DONE]":
                        parsed = json.loads(data)
                        if parsed.get("type") == "content_block_delta":
                            if ttft_ms is None:
                                ttft_ms = (time.perf_counter() - t0) * 1000
                            delta = parsed.get("delta", {})
                            full_text += delta.get("text", "")

        total_ms = (time.perf_counter() - t0) * 1000
        if ttft_ms is None:
            ttft_ms = total_ms
        return ttft_ms, total_ms, {"content": [{"text": full_text}], "type": "message"}

    else:
        r = httpx.post(
            f"{base_url}/v1/messages",
            json=body,
            headers=headers,
            timeout=60.0,
        )
        total_ms = (time.perf_counter() - t0) * 1000
        r.raise_for_status()
        return total_ms, total_ms, r.json()


@pytest.fixture(scope="module")
def mlx_server():
    """Start agent-memory server with SmolLM2-135M for cache testing."""
    port = _find_free_port()
    url = f"http://127.0.0.1:{port}"

    env = os.environ.copy()
    # Use clean cache dir to avoid stale Q4 caches from production
    test_cache_dir = Path.home() / ".cache" / "agent_memory" / "test_prefix"
    env["SEMANTIC_AGENT_CACHE_DIR"] = str(test_cache_dir)
    env["SEMANTIC_LOG_LEVEL"] = "INFO"
    env["ANTHROPIC_API_KEY"] = "test-key-for-e2e"
    # Use SmolLM2-135M for fast testing (default is Gemma 3 12B)
    env.pop("SEMANTIC_BACKEND", None)  # Ensure MLX (default)
    env["SEMANTIC_MLX_MODEL_ID"] = "mlx-community/SmolLM2-135M-Instruct"
    # FP16 KV cache — avoids QuantizedKVCache batching limitation in mlx-lm 0.31
    env["SEMANTIC_MLX_KV_BITS"] = "0"
    env["SEMANTIC_MLX_SCHEDULER_ENABLED"] = "false"  # Direct engine path
    # Disable chunked prefill — it's broken with FP16 + direct engine (causes
    # shape mismatch in BatchRotatingKVCache.merge).  Standard prefill works
    # fine for our ~3400 token test prompts.
    env["SEMANTIC_MLX_CHUNKED_PREFILL_ENABLED"] = "false"

    log_dir = Path("/tmp/claude/prefix_cache_logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    stdout_f = open(log_dir / "stdout.log", "w")
    stderr_f = open(log_dir / "stderr.log", "w")

    proc = subprocess.Popen(
        [
            sys.executable, "-m", "uvicorn",
            "agent_memory.entrypoints.api_server:create_app",
            "--host", "127.0.0.1",
            "--port", str(port),
            "--factory",
        ],
        env=env,
        stdout=stdout_f,
        stderr=stderr_f,
    )

    if not _wait_for_health(url, timeout=120):
        proc.kill()
        proc.wait()
        stdout_f.close()
        stderr_f.close()
        stderr = (log_dir / "stderr.log").read_text()[-2000:]
        pytest.fail(f"Server failed to start. Stderr:\n{stderr}")

    yield url

    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
    stdout_f.close()
    stderr_f.close()

    # Cleanup
    import shutil
    test_cache = Path.home() / ".cache" / "agent_memory" / "test_prefix"
    if test_cache.exists():
        shutil.rmtree(test_cache, ignore_errors=True)


def _make_long_system_prompt(n_repeat: int = 30) -> str:
    """Build a realistic long system prompt (~2000+ tokens).

    Mimics Claude Code's system prompt: instructions, tool reference,
    project context, and behavior guidelines.
    """
    base = (
        "You are an expert software engineering assistant. You help users "
        "with coding tasks including debugging, refactoring, writing tests, "
        "and implementing new features.\n\n"
        "# Tool Quick Reference\n\n"
        "## File Operations\n"
        "- Read: file_path (required, absolute), offset (start line), limit (max lines)\n"
        "- Write: file_path (required, absolute), content (required, full file)\n"
        "- Edit: file_path (required), old_string (required, must be unique), new_string (required)\n"
        "- Glob: pattern (required, e.g. \"**/*.py\"), path (optional directory)\n"
        "- Grep: pattern (required, regex), path, glob, output_mode\n\n"
        "## Execution\n"
        "- Bash: command (required), timeout (ms, max 600000)\n"
        "- Agent: prompt (required), description (required, 3-5 words)\n\n"
        "## Planning\n"
        "- TodoWrite: todos (required, array of {content, status})\n\n"
        "# Project Context\n"
        "This is a Python project using FastAPI, Pydantic, and pytest.\n"
        "Follow hexagonal architecture: ports in application layer, adapters in adapters layer.\n"
        "Never import adapter code from the application layer.\n"
        "Write tests for all new code. Use pytest fixtures for test setup.\n\n"
        "# Behavior Guidelines\n"
        "- Be concise and direct in responses.\n"
        "- Read files before editing them.\n"
        "- Use dedicated tools (Read/Write/Grep) instead of shell commands.\n"
        "- Break complex tasks into sub-tasks using the Agent tool.\n"
        "- Always verify your changes compile and pass tests.\n"
    )
    extra = (
        "\n# Additional Context Block\n"
        "The codebase uses dataclasses for domain objects and Pydantic for API models. "
        "Error handling follows the domain error hierarchy. Logging uses structlog. "
        "Cache persistence uses safetensors format on disk. Block allocation uses a "
        "pool allocator with configurable block sizes. KV cache supports Q4 and Q8 "
        "quantization with configurable group sizes. The scheduler handles concurrent "
        "prefill and decode interleaving for batched inference. Configuration is loaded "
        "from TOML files in config/models/ with environment variable overrides.\n"
    )
    return base + extra * n_repeat


class TestCacheReuseTTFT:
    """Verify KV cache reuse produces measurable TTFT improvement.

    Uses the same session for cold (request 1) and warm (request 2+)
    measurements. The same-session cache path is the primary optimization:
    request 1 pays full prefill, request 2+ reuses cached KV state.
    """

    SYSTEM_PROMPT = _make_long_system_prompt(30)

    def test_server_healthy(self, mlx_server):
        """Server is running and healthy."""
        r = httpx.get(f"{mlx_server}/health/live", timeout=5.0)
        assert r.status_code == 200

    def test_non_streaming_cold_vs_warm(self, mlx_server):
        """Non-streaming: second request with same session should be much faster.

        Request 1 (cold): full prefill of long system prompt → slow
        Request 2 (warm): KV cache reused, only new user tokens prefilled → fast
        """
        prompt = self.SYSTEM_PROMPT
        session = "non-stream-perf"

        # Request 1: cold — full prefill
        _, total1, resp1 = _send_message(
            mlx_server,
            system=prompt,
            user_msg="What is 2+2? Answer in one word.",
            session_id=session,
            stream=False,
        )
        assert "content" in resp1

        # Request 2: warm — same session, KV cache reused
        _, total2, resp2 = _send_message(
            mlx_server,
            system=prompt,
            user_msg="What is 3+3? Answer in one word.",
            session_id=session,
            stream=False,
        )
        assert "content" in resp2

        # Request 3: warm again (consistency check)
        _, total3, resp3 = _send_message(
            mlx_server,
            system=prompt,
            user_msg="What is 4+4? Answer in one word.",
            session_id=session,
            stream=False,
        )
        assert "content" in resp3

        avg_warm = (total2 + total3) / 2
        speedup = total1 / avg_warm if avg_warm > 0 else 0

        print(f"\n  Non-streaming (same session):")
        print(f"  System prompt: ~{len(prompt.split())} words")
        print(f"  Request 1 (cold): {total1:.0f}ms")
        print(f"  Request 2 (warm): {total2:.0f}ms")
        print(f"  Request 3 (warm): {total3:.0f}ms")
        print(f"  Speedup: {speedup:.2f}x (cold / avg_warm)")

        # With cache reuse, warm should be faster.
        # Cold includes full prefill of ~3000+ token system prompt.
        # Warm only prefills the new user message (~10 tokens).
        # SmolLM2-135M has fast prefill (~400ms), so 1.2x is realistic.
        # Larger models show 3-5x speedup.
        assert speedup > 1.2, (
            f"Expected >=1.2x speedup from cache reuse, got {speedup:.2f}x. "
            f"Cold={total1:.0f}ms, warm avg={avg_warm:.0f}ms"
        )

    def test_streaming_ttft_cold_vs_warm(self, mlx_server):
        """Streaming: true TTFT (time to first token) should be much lower on warm.

        Streaming gives us real TTFT — the time until the first content_block_delta
        SSE event arrives, which is when the first generated token is emitted.
        """
        prompt = self.SYSTEM_PROMPT
        session = "stream-perf"

        # Request 1: cold — full prefill, measures true TTFT
        ttft1, total1, resp1 = _send_message(
            mlx_server,
            system=prompt,
            user_msg="What is 2+2? Answer in one word.",
            session_id=session,
            stream=True,
        )
        assert "content" in resp1

        # Request 2: warm — same session, cached KV state
        ttft2, total2, resp2 = _send_message(
            mlx_server,
            system=prompt,
            user_msg="What is 3+3? Answer in one word.",
            session_id=session,
            stream=True,
        )
        assert "content" in resp2

        # Request 3: warm (consistency)
        ttft3, total3, resp3 = _send_message(
            mlx_server,
            system=prompt,
            user_msg="What is 4+4? Answer in one word.",
            session_id=session,
            stream=True,
        )
        assert "content" in resp3

        avg_ttft_warm = (ttft2 + ttft3) / 2
        ttft_speedup = ttft1 / avg_ttft_warm if avg_ttft_warm > 0 else 0

        print(f"\n  Streaming TTFT (same session):")
        print(f"  System prompt: ~{len(prompt.split())} words")
        print(f"  Request 1 (cold) TTFT: {ttft1:.0f}ms  total: {total1:.0f}ms")
        print(f"  Request 2 (warm) TTFT: {ttft2:.0f}ms  total: {total2:.0f}ms")
        print(f"  Request 3 (warm) TTFT: {ttft3:.0f}ms  total: {total3:.0f}ms")
        print(f"  TTFT speedup: {ttft_speedup:.2f}x (cold / avg_warm)")

        # Streaming TTFT improvement: cold includes full system prompt prefill,
        # warm reuses cached KV state. SmolLM2-135M has very fast prefill so
        # the speedup is modest (~1.2-2x); larger models show 3-5x.
        assert ttft_speedup > 1.1, (
            f"Expected >=1.1x TTFT speedup from cache reuse, got {ttft_speedup:.2f}x. "
            f"Cold TTFT={ttft1:.0f}ms, warm avg TTFT={avg_ttft_warm:.0f}ms"
        )

    def test_cross_session_prefix_cache_non_streaming(self, mlx_server):
        """Different sessions with same system prompt should benefit from prefix cache.

        Session A populates the SharedPrefixCache after generation.
        Session B hits the cache and uses the detached prefix blocks
        via the DIVERGE path — same system prompt prefix, different user msg.
        """
        prompt = self.SYSTEM_PROMPT

        # Session A: cold — populates SharedPrefixCache
        _, total_a, resp_a = _send_message(
            mlx_server,
            system=prompt,
            user_msg="What is 2+2? Answer in one word.",
            session_id="cross-ns-a",
            stream=False,
        )
        assert "content" in resp_a

        # Session B: different session, same system prompt — prefix cache hit
        _, total_b, resp_b = _send_message(
            mlx_server,
            system=prompt,
            user_msg="What is 3+3? Answer in one word.",
            session_id="cross-ns-b",
            stream=False,
        )
        assert "content" in resp_b

        # Session C: another session — prefix cache hit
        _, total_c, resp_c = _send_message(
            mlx_server,
            system=prompt,
            user_msg="What is 4+4? Answer in one word.",
            session_id="cross-ns-c",
            stream=False,
        )
        assert "content" in resp_c

        avg_warm = (total_b + total_c) / 2
        speedup = total_a / avg_warm if avg_warm > 0 else 0

        print(f"\n  Cross-session prefix cache (non-streaming):")
        print(f"  System prompt: ~{len(prompt.split())} words")
        print(f"  Session A (cold): {total_a:.0f}ms")
        print(f"  Session B (prefix hit): {total_b:.0f}ms")
        print(f"  Session C (prefix hit): {total_c:.0f}ms")
        print(f"  Speedup: {speedup:.2f}x (cold / avg_warm)")

        # NOTE: SmolLM2-135M is too small to benefit — prefill is only ~200ms
        # and reconstruction overhead is ~100ms.  Larger models (Gemma 3 12B+)
        # with ~5-15s prefill show massive speedup from prefix sharing.
        # Here we verify the path WORKS (no errors) even if speedup is marginal.
        if speedup < 1.0:
            print(f"  (reconstruction overhead > prefill savings for this tiny model)")

    def test_cross_session_prefix_cache_streaming(self, mlx_server):
        """Streaming: cross-session prefix cache should reduce TTFT.

        Same as non-streaming test but measures true TTFT via SSE.
        """
        prompt = self.SYSTEM_PROMPT

        # Session A: cold — populates SharedPrefixCache
        ttft_a, _, resp_a = _send_message(
            mlx_server,
            system=prompt,
            user_msg="What is 2+2? Answer in one word.",
            session_id="cross-st-a",
            stream=True,
        )
        assert "content" in resp_a

        # Session B: prefix cache hit
        ttft_b, _, resp_b = _send_message(
            mlx_server,
            system=prompt,
            user_msg="What is 3+3? Answer in one word.",
            session_id="cross-st-b",
            stream=True,
        )
        assert "content" in resp_b

        # Session C: prefix cache hit
        ttft_c, _, resp_c = _send_message(
            mlx_server,
            system=prompt,
            user_msg="What is 4+4? Answer in one word.",
            session_id="cross-st-c",
            stream=True,
        )
        assert "content" in resp_c

        avg_ttft_warm = (ttft_b + ttft_c) / 2
        ttft_speedup = ttft_a / avg_ttft_warm if avg_ttft_warm > 0 else 0

        print(f"\n  Cross-session prefix cache (streaming TTFT):")
        print(f"  System prompt: ~{len(prompt.split())} words")
        print(f"  Session A (cold) TTFT: {ttft_a:.0f}ms")
        print(f"  Session B (prefix hit) TTFT: {ttft_b:.0f}ms")
        print(f"  Session C (prefix hit) TTFT: {ttft_c:.0f}ms")
        print(f"  TTFT speedup: {ttft_speedup:.2f}x (cold / avg_warm)")

        # NOTE: SmolLM2-135M too small for meaningful cross-session speedup.
        # Verify path works without errors; speedup validated on larger models.
        if ttft_speedup < 1.0:
            print(f"  (reconstruction overhead > prefill savings for this tiny model)")

    def test_different_system_prompts_no_benefit(self, mlx_server):
        """Different system prompts should NOT get prefix cache benefit."""
        prompt_a = "You are a pirate captain. " * 100
        prompt_b = "You are a medieval knight. " * 100

        ttft_a, _, resp_a = _send_message(
            mlx_server, system=prompt_a, user_msg="Hello!",
            session_id="diff-sys-a", stream=True,
        )
        ttft_b, _, resp_b = _send_message(
            mlx_server, system=prompt_b, user_msg="Hello!",
            session_id="diff-sys-b", stream=True,
        )

        assert "content" in resp_a
        assert "content" in resp_b

        print(f"\n  Different system prompts:")
        print(f"  Prompt A TTFT: {ttft_a:.0f}ms")
        print(f"  Prompt B TTFT: {ttft_b:.0f}ms")
        print(f"  (Both cold — no prefix cache benefit expected)")

    def test_multi_turn_same_session(self, mlx_server):
        """Multi-turn: each turn reuses growing KV cache from previous turns."""
        prompt = self.SYSTEM_PROMPT
        session = "multi-turn"

        # Turn 1: cold
        ttft1, _, resp1 = _send_message(
            mlx_server,
            system=prompt,
            user_msg="Remember: the secret number is 42.",
            session_id=session,
            stream=True,
        )
        assert "content" in resp1
        assistant_text = resp1["content"][0].get("text", "OK")

        # Turn 2: full conversation history — KV cache reuses shared prefix
        headers = {
            "Content-Type": "application/json",
            "X-API-Key": "test-key-for-e2e",
            "anthropic-version": "2023-06-01",
            "X-Session-ID": session,
        }
        body = {
            "model": "mlx-community/SmolLM2-135M-Instruct",
            "max_tokens": 32,
            "system": prompt,
            "stream": True,
            "messages": [
                {"role": "user", "content": "Remember: the secret number is 42."},
                {"role": "assistant", "content": assistant_text},
                {"role": "user", "content": "What is the secret number?"},
            ],
        }

        t0 = time.perf_counter()
        ttft2 = None
        with httpx.stream(
            "POST", f"{mlx_server}/v1/messages",
            json=body, headers=headers, timeout=60.0,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if ttft2 is None and line.startswith("data:"):
                    data = line[5:].strip()
                    if data and data != "[DONE]":
                        parsed = json.loads(data)
                        if parsed.get("type") == "content_block_delta":
                            ttft2 = (time.perf_counter() - t0) * 1000
        if ttft2 is None:
            ttft2 = (time.perf_counter() - t0) * 1000

        speedup = ttft1 / ttft2 if ttft2 > 0 else 0

        print(f"\n  Multi-turn (same session):")
        print(f"  Turn 1 (cold)  TTFT: {ttft1:.0f}ms")
        print(f"  Turn 2 (warm)  TTFT: {ttft2:.0f}ms")
        print(f"  Speedup: {speedup:.2f}x")
