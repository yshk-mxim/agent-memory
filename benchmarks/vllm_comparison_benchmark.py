# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Yakov Shkolnikov and contributors
"""Head-to-head comparison benchmark: agent-memory vs vllm-mlx.

Measures TTFT and TPS under identical conditions for both backends:
  - Cold (no cache): First request, cache cleared/fresh server
  - Prefix-cached (same session): Identical prompt repeated
  - Post-restart: Server restarted, same prompt resent
  - Concurrent (batch=2): Two simultaneous requests

Both backends serve OpenAI-compatible /v1/chat/completions, so we use
the same streaming client and corpus for fair comparison.

Key differentiator: agent-memory persists KV cache to disk (safetensors),
so post-restart warm TTFT is fast.  vllm-mlx uses in-memory prefix cache
only — restart = cold start.

Usage:
    # Run full comparison (starts/stops servers automatically)
    PYTHONUNBUFFERED=1 python benchmarks/vllm_comparison_benchmark.py

    # Specific model
    PYTHONUNBUFFERED=1 python benchmarks/vllm_comparison_benchmark.py \\
        --model gemma --contexts 1024 2048 4096 --passes 6

    # Skip agent-memory (only benchmark vllm-mlx)
    PYTHONUNBUFFERED=1 python benchmarks/vllm_comparison_benchmark.py \\
        --vllm-only

Prerequisites:
    # vllm-mlx requires transformers>=5.0.0, agent-memory pins <5.0.0.
    # They CANNOT coexist in the same Python environment.
    # Option A: Separate venv for vllm-mlx
    #   python -m venv ~/.venvs/vllm-mlx
    #   source ~/.venvs/vllm-mlx/bin/activate
    #   pip install vllm-mlx
    # Option B: Run this script with --am-only or --vllm-only from
    #   the appropriate environment.
    # The script manages servers via subprocess, so the vllm-mlx binary
    # just needs to be on PATH (e.g. ~/.venvs/vllm-mlx/bin/vllm-mlx).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))

from capability_benchmark import ScenarioResult, compute_stats
from openai_benchmark import OpenAIStreamingClient

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RESULTS_DIR = Path(__file__).resolve().parent / "results"
CORPUS_PATH = Path(__file__).resolve().parent / "data" / "prefill_corpus.txt"
PADDING_TEXT = (
    "The system implements a block-based KV cache architecture where each "
    "block stores a fixed number of token key-value pairs. Blocks are "
    "allocated from a shared pool and assigned per-layer per-agent. "
)

OUTPUT_TOKENS = 64
DEFAULT_PASSES = 3
DEFAULT_PORT_AM = 8399  # agent-memory
DEFAULT_PORT_VLLM = 8398  # vllm-mlx (different port to avoid conflicts)
TEMPERATURE = 0.0
COOLDOWN_SECONDS = 15  # Fixed cooldown between measurements
STARTUP_TIMEOUT = 180  # seconds

MODEL_IDS = {
    "gemma": "mlx-community/gemma-3-12b-it-4bit",
    "deepseek": "mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx",
    "llama": "mlx-community/Llama-3.1-8B-Instruct-4bit",
}

# agent-memory cache budgets
AM_CACHE_BUDGET = {
    "gemma": "8192",
    "deepseek": "4096",
    "llama": "8192",
}

ALL_CONTEXTS = [1024, 2048, 4096, 8192, 16384]
ADMIN_KEY = "benchmark"

# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------


def load_corpus() -> str:
    if CORPUS_PATH.exists():
        text = CORPUS_PATH.read_text(encoding="utf-8")
        if len(text) > 10000:
            return text
    return PADDING_TEXT * 5000


def build_messages(corpus: str, target_tokens: int, offset: int = 0) -> list[dict[str, str]]:
    chars_needed = target_tokens * 4
    if len(corpus) >= chars_needed + 1000:
        max_start = len(corpus) - chars_needed
        start = offset % max_start if max_start > 0 else 0
        content = corpus[start : start + chars_needed]
    else:
        content = (PADDING_TEXT + " ") * (chars_needed // len(PADDING_TEXT) + 1)
        content = content[:chars_needed]
    return [
        {
            "role": "user",
            "content": (
                f"Here is some text:\n\n{content}\n\n"
                "What are the main topics and themes discussed above?"
            ),
        },
    ]


# ---------------------------------------------------------------------------
# Server management
# ---------------------------------------------------------------------------


def kill_port(port: int) -> None:
    """Kill any process on the given port."""
    try:
        result = subprocess.run(
            ["lsof", "-ti", f":{port}"],
            capture_output=True, text=True, timeout=5,
        )
        pids = result.stdout.strip().split()
        for pid in pids:
            if pid:
                os.kill(int(pid), signal.SIGTERM)
        if pids:
            time.sleep(3)
            # Force kill if still alive
            result2 = subprocess.run(
                ["lsof", "-ti", f":{port}"],
                capture_output=True, text=True, timeout=5,
            )
            for pid in result2.stdout.strip().split():
                if pid:
                    try:
                        os.kill(int(pid), signal.SIGKILL)
                    except ProcessLookupError:
                        pass
            time.sleep(2)
    except Exception:
        pass


def wait_for_health(url: str, timeout: float = STARTUP_TIMEOUT) -> bool:
    """Poll health endpoint until ready."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = httpx.get(url, timeout=3.0)
            if r.status_code == 200:
                return True
        except (httpx.ConnectError, httpx.ReadTimeout, httpx.ConnectTimeout):
            pass
        time.sleep(2.0)
    return False


class AgentMemoryServer:
    """Manage agent-memory server lifecycle."""

    def __init__(self, port: int, model_key: str):
        self.port = port
        self.model_key = model_key
        self.model_id = MODEL_IDS[model_key]
        self.proc: subprocess.Popen | None = None
        self.base_url = f"http://127.0.0.1:{port}"

    def start(self) -> bool:
        kill_port(self.port)
        env = os.environ.copy()
        env.update({
            "SEMANTIC_MLX_MODEL_ID": self.model_id,
            "SEMANTIC_MLX_MAX_BATCH_SIZE": "2",
            "SEMANTIC_MLX_SCHEDULER_ENABLED": "true",
            "SEMANTIC_MLX_CHUNKED_PREFILL_ENABLED": "true",
            "SEMANTIC_MLX_CHUNKED_PREFILL_THRESHOLD": "2048",
            "SEMANTIC_MLX_CHUNKED_PREFILL_MIN_CHUNK": "512",
            "SEMANTIC_MLX_CHUNKED_PREFILL_MAX_CHUNK": "4096",
            "SEMANTIC_MLX_PREFILL_STEP_SIZE": "256",
            "SEMANTIC_MLX_KV_BITS": "4",
            "SEMANTIC_MLX_MAX_CONTEXT_LENGTH": "100000",
            "SEMANTIC_MLX_CACHE_BUDGET_MB": AM_CACHE_BUDGET.get(self.model_key, "8192"),
            "SEMANTIC_SERVER_LOG_LEVEL": "WARNING",
            "SEMANTIC_ADMIN_KEY": ADMIN_KEY,
            "SEMANTIC_API_KEY": "",
        })
        self.proc = subprocess.Popen(
            [
                sys.executable, "-m", "agent_memory.entrypoints.cli",
                "serve", "--port", str(self.port), "--log-level", "WARNING",
            ],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(Path(__file__).resolve().parents[1]),
        )
        print(f"  Starting agent-memory on port {self.port}...")
        if wait_for_health(f"{self.base_url}/health/startup"):
            print(f"  agent-memory ready on port {self.port}")
            return True
        print(f"  FAILED: agent-memory did not start within {STARTUP_TIMEOUT}s")
        return False

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.send_signal(signal.SIGTERM)
            try:
                self.proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=5)
        self.proc = None
        time.sleep(5)  # GPU memory release
        kill_port(self.port)

    def is_alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None


class VllmMlxServer:
    """Manage vllm-mlx server lifecycle."""

    def __init__(self, port: int, model_key: str, continuous_batching: bool = False,
                 cache_memory_percent: float = 0.40):
        self.port = port
        self.model_key = model_key
        self.model_id = MODEL_IDS[model_key]
        self.continuous_batching = continuous_batching
        self.cache_memory_percent = cache_memory_percent
        self.proc: subprocess.Popen | None = None
        self.base_url = f"http://127.0.0.1:{port}"

    def start(self) -> bool:
        kill_port(self.port)
        # Use vllm-mlx binary (may be in a separate venv).
        # Check VLLM_MLX_BIN env var, fallback to PATH lookup.
        vllm_bin = os.environ.get("VLLM_MLX_BIN", "vllm-mlx")
        cmd = [
            vllm_bin,
            "serve", self.model_id,
            "--port", str(self.port),
            "--enable-prefix-cache",  # Ensure prefix caching is on
            "--cache-memory-percent", str(self.cache_memory_percent),
        ]
        if self.continuous_batching:
            cmd.extend(["--continuous-batching", "--use-paged-cache"])
        self.proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        print(f"  Starting vllm-mlx on port {self.port} "
              f"(batching={'on' if self.continuous_batching else 'off'})...")
        if wait_for_health(f"{self.base_url}/health"):
            print(f"  vllm-mlx ready on port {self.port}")
            return True
        print(f"  FAILED: vllm-mlx did not start within {STARTUP_TIMEOUT}s")
        return False

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.send_signal(signal.SIGTERM)
            try:
                self.proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=5)
        self.proc = None
        time.sleep(5)
        kill_port(self.port)

    def is_alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None


# ---------------------------------------------------------------------------
# Cache management
# ---------------------------------------------------------------------------


async def clear_am_caches(base_url: str) -> None:
    """Clear all agent-memory caches via admin API."""
    async with httpx.AsyncClient(timeout=30) as client:
        try:
            await client.delete(
                f"{base_url}/admin/caches",
                headers={"X-Admin-Key": ADMIN_KEY},
            )
        except Exception:
            pass
    # Also clean cache files on disk
    cache_dir = Path(__file__).resolve().parents[1] / "cache"
    if cache_dir.exists():
        for f in cache_dir.glob("*.safetensors"):
            try:
                f.unlink()
            except OSError:
                pass


async def evict_am_agent(base_url: str, agent_id: str) -> None:
    """Evict agent from hot cache but keep disk file (warm state)."""
    async with httpx.AsyncClient(timeout=30) as client:
        try:
            await client.delete(
                f"{base_url}/v1/agents/{agent_id}",
                headers={"X-Admin-Key": ADMIN_KEY},
                params={"evict_only": "true"},
            )
        except Exception:
            pass
    await asyncio.sleep(1.5)  # Wait for disk write


# ---------------------------------------------------------------------------
# Quality check
# ---------------------------------------------------------------------------


def quality_ok(text: str) -> bool:
    """Basic quality check: non-empty, reasonable length, no obvious issues."""
    if not text or len(text.strip()) < 10:
        return False
    words = text.split()
    if len(words) < 5:
        return False
    # Check for excessive repetition
    if len(set(words)) < len(words) * 0.15:
        return False
    return True


# ---------------------------------------------------------------------------
# Measurement functions
# ---------------------------------------------------------------------------


async def measure_streaming_ttft(
    base_url: str,
    messages: list[dict],
    session_id: str | None = None,
) -> dict[str, Any]:
    """Single streaming request, return TTFT, TPS, output text."""
    client = OpenAIStreamingClient(base_url)
    body = {
        "model": "default",
        "messages": messages,
        "max_tokens": OUTPUT_TOKENS,
        "temperature": TEMPERATURE,
        "stream": True,
    }
    r = await client.send_and_measure(body, session_id=session_id)
    return {
        "ttft_ms": round(r.ttft_ms, 1),
        "e2e_ms": round(r.e2e_ms, 1),
        "decode_tps": round(r.decode_tps, 1),
        "output_tokens": r.output_tokens,
        "raw_output": r.raw_output[:500],
        "quality_ok": quality_ok(r.raw_output),
        "error": r.error,
    }


async def measure_concurrent_ttft(
    base_url: str,
    messages_a: list[dict],
    messages_b: list[dict],
    sid_a: str,
    sid_b: str,
) -> dict[str, Any]:
    """Two simultaneous streaming requests, return per-request and aggregate metrics."""
    client_a = OpenAIStreamingClient(base_url)
    client_b = OpenAIStreamingClient(base_url)
    body_a = {
        "model": "default",
        "messages": messages_a,
        "max_tokens": OUTPUT_TOKENS,
        "temperature": TEMPERATURE,
        "stream": True,
    }
    body_b = {
        "model": "default",
        "messages": messages_b,
        "max_tokens": OUTPUT_TOKENS,
        "temperature": TEMPERATURE,
        "stream": True,
    }
    t_start = time.perf_counter()
    r_a, r_b = await asyncio.gather(
        client_a.send_and_measure(body_a, session_id=sid_a),
        client_b.send_and_measure(body_b, session_id=sid_b),
    )
    wall_ms = (time.perf_counter() - t_start) * 1000

    return {
        "wall_ms": round(wall_ms, 1),
        "user_a_ttft_ms": round(r_a.ttft_ms, 1),
        "user_b_ttft_ms": round(r_b.ttft_ms, 1),
        "avg_ttft_ms": round((r_a.ttft_ms + r_b.ttft_ms) / 2, 1),
        "user_a_tps": round(r_a.decode_tps, 1),
        "user_b_tps": round(r_b.decode_tps, 1),
        "total_output_tokens": r_a.output_tokens + r_b.output_tokens,
        "quality_ok": quality_ok(r_a.raw_output) and quality_ok(r_b.raw_output),
        "error_a": r_a.error,
        "error_b": r_b.error,
    }


# ---------------------------------------------------------------------------
# Test scenarios
# ---------------------------------------------------------------------------


async def test_cold(
    backend: str,
    base_url: str,
    corpus: str,
    context: int,
    pass_id: int,
) -> dict[str, Any]:
    """Cold start: clear caches, measure first request."""
    if backend == "agent-memory":
        await clear_am_caches(base_url)
    # vllm-mlx: no admin API to clear prefix cache — use unique session
    sid = f"cold_{backend}_{context}_{pass_id}_{int(time.time())}"
    messages = build_messages(corpus, context, offset=pass_id * 1000)
    result = await measure_streaming_ttft(base_url, messages, session_id=sid)
    result["scenario"] = "cold"
    result["backend"] = backend
    result["context_tokens"] = context
    result["pass_id"] = pass_id
    return result


async def test_prefix_cached(
    backend: str,
    base_url: str,
    corpus: str,
    context: int,
    pass_id: int,
) -> dict[str, Any]:
    """Prefix-cached: send same prompt twice, measure second request.

    For agent-memory: clear → prime (cold) → measure (should hit warm/hot cache)
    For vllm-mlx: prime → measure (prefix cache hit within same session)
    """
    if backend == "agent-memory":
        await clear_am_caches(base_url)
    offset = pass_id * 1000 + 50000  # Different offset from cold test
    messages = build_messages(corpus, context, offset=offset)
    sid = f"prefix_{backend}_{context}_{pass_id}_{int(time.time())}"

    # Prime request (cold)
    await measure_streaming_ttft(base_url, messages, session_id=sid)
    await asyncio.sleep(2)  # Let cache settle

    if backend == "agent-memory":
        # Evict from hot to warm (disk-backed), then measure reload
        await evict_am_agent(base_url, f"oai_{sid}")

    # Measure cached request (same prompt, same session)
    result = await measure_streaming_ttft(base_url, messages, session_id=sid)
    result["scenario"] = "prefix_cached"
    result["backend"] = backend
    result["context_tokens"] = context
    result["pass_id"] = pass_id
    return result


async def test_post_restart(
    backend: str,
    server_factory,
    corpus: str,
    context: int,
    pass_id: int,
) -> dict[str, Any]:
    """Post-restart: prime cache, restart server, resend same prompt.

    This is the KEY differentiator:
    - agent-memory: warm TTFT (loads from disk safetensors)
    - vllm-mlx: cold TTFT (no disk persistence, cache lost on restart)
    """
    server = server_factory()
    if not server.start():
        return {"error": "Server failed to start", "scenario": "post_restart",
                "backend": backend, "context_tokens": context, "pass_id": pass_id}

    offset = pass_id * 1000 + 80000
    messages = build_messages(corpus, context, offset=offset)
    sid = f"restart_{backend}_{context}_{pass_id}_{int(time.time())}"

    # Prime request
    prime_result = await measure_streaming_ttft(server.base_url, messages, session_id=sid)
    if prime_result.get("error"):
        server.stop()
        return {"error": f"Prime failed: {prime_result['error']}", "scenario": "post_restart",
                "backend": backend, "context_tokens": context, "pass_id": pass_id}

    await asyncio.sleep(3)  # Let cache flush to disk

    # Restart server
    print(f"    Restarting {backend}...")
    server.stop()
    time.sleep(3)

    server2 = server_factory()
    if not server2.start():
        return {"error": "Server failed to restart", "scenario": "post_restart",
                "backend": backend, "context_tokens": context, "pass_id": pass_id}

    # Measure same prompt after restart
    result = await measure_streaming_ttft(server2.base_url, messages, session_id=sid)
    result["scenario"] = "post_restart"
    result["backend"] = backend
    result["context_tokens"] = context
    result["pass_id"] = pass_id
    result["prime_ttft_ms"] = prime_result["ttft_ms"]

    server2.stop()
    return result


async def test_concurrent(
    backend: str,
    base_url: str,
    corpus: str,
    context: int,
    pass_id: int,
) -> dict[str, Any]:
    """Concurrent batch=2: two simultaneous requests."""
    if backend == "agent-memory":
        await clear_am_caches(base_url)

    sid_a = f"conc_a_{backend}_{context}_{pass_id}_{int(time.time())}"
    sid_b = f"conc_b_{backend}_{context}_{pass_id}_{int(time.time())}"
    messages_a = build_messages(corpus, context, offset=pass_id * 1000 + 20000)
    messages_b = build_messages(corpus, context, offset=pass_id * 1000 + 25000)

    result = await measure_concurrent_ttft(base_url, messages_a, messages_b, sid_a, sid_b)
    result["scenario"] = "concurrent"
    result["backend"] = backend
    result["context_tokens"] = context
    result["pass_id"] = pass_id
    return result


# ---------------------------------------------------------------------------
# Main benchmark runner
# ---------------------------------------------------------------------------


async def run_comparison(
    model_key: str,
    contexts: list[int],
    passes: int,
    port_am: int,
    port_vllm: int,
    skip_am: bool = False,
    skip_vllm: bool = False,
    skip_restart: bool = False,
    skip_concurrent: bool = False,
    cache_memory_percent: float = 0.40,
) -> dict[str, Any]:
    """Run full comparison benchmark."""
    corpus = load_corpus()
    print(f"  Corpus size: {len(corpus):,} chars ({len(corpus) // 4:,} est. tokens)")

    model_id = MODEL_IDS[model_key]
    results: dict[str, Any] = {
        "metadata": {
            "benchmark": "vllm_comparison",
            "timestamp_start": datetime.now(UTC).isoformat(),
            "model_id": model_id,
            "model_key": model_key,
            "contexts": contexts,
            "passes": passes,
            "output_tokens": OUTPUT_TOKENS,
            "temperature": TEMPERATURE,
            "machine": {
                "os": platform.system(),
                "arch": platform.machine(),
                "python": platform.python_version(),
            },
        },
        "measurements": [],
    }

    backends_to_test = []
    if not skip_am:
        backends_to_test.append("agent-memory")
    if not skip_vllm:
        backends_to_test.append("vllm-mlx")

    for backend in backends_to_test:
        print(f"\n{'='*60}")
        print(f"  BACKEND: {backend}")
        print(f"{'='*60}")

        if backend == "agent-memory":
            server = AgentMemoryServer(port_am, model_key)
            if not server.start():
                print(f"  SKIP {backend}: server failed to start")
                continue
            base_url = server.base_url
        else:
            # vllm-mlx: use --continuous-batching to enable prefix caching.
            # SimpleEngine (default) has NO text prefix caching in v0.2.6.
            server = VllmMlxServer(port_vllm, model_key, continuous_batching=True, cache_memory_percent=cache_memory_percent)
            if not server.start():
                print(f"  SKIP {backend}: server failed to start")
                continue
            base_url = server.base_url

        # --- Cold + Cached TTFT (per-context interleaved) ---
        # Run cold→cached for each context before moving to the next.
        # This prevents FP16 prefix cache eviction: vllm-mlx's LRU cache
        # cannot hold all contexts simultaneously in FP16 (e.g., 1K+2K+4K+
        # 8K+16K × 3 passes ≈ 93K tokens × 128 KB/tok ≈ 11.6 GB at FP16).
        # Interleaving ensures each context's cold prefill is still in cache
        # when the cached test runs immediately after.
        print(f"\n  --- Cold + Cached TTFT ({backend}, per-context) ---")
        for ctx in contexts:
            # Cold passes
            for p in range(passes):
                print(f"    cold/{ctx}tok/pass{p}...", end=" ", flush=True)
                r = await test_cold(backend, base_url, corpus, ctx, p)
                q = "OK" if r.get("quality_ok") else "FAIL"
                print(f"TTFT={r.get('ttft_ms', 0):.0f}ms TPS={r.get('decode_tps', 0):.1f} Q={q}")
                results["measurements"].append(r)
                await asyncio.sleep(COOLDOWN_SECONDS)
            # Cached passes (prefix should still be in cache)
            for p in range(passes):
                print(f"    cached/{ctx}tok/pass{p}...", end=" ", flush=True)
                r = await test_prefix_cached(backend, base_url, corpus, ctx, p)
                q = "OK" if r.get("quality_ok") else "FAIL"
                print(f"TTFT={r.get('ttft_ms', 0):.0f}ms TPS={r.get('decode_tps', 0):.1f} Q={q}")
                results["measurements"].append(r)
                await asyncio.sleep(COOLDOWN_SECONDS)

        # Stop server
        server.stop()

        # --- Concurrent (batch=2) ---
        if not skip_concurrent:
            print(f"\n  --- Concurrent batch=2 ({backend}) ---")
            if backend == "agent-memory":
                # agent-memory already has scheduler, same server config works
                server = AgentMemoryServer(port_am, model_key)
            else:
                server = VllmMlxServer(port_vllm, model_key, continuous_batching=True, cache_memory_percent=cache_memory_percent)

            if server.start():
                base_url = server.base_url
                for ctx in contexts:
                    if ctx > 16384:
                        print(f"    SKIP {ctx}tok (OOM risk for batch=2)")
                        continue
                    for p in range(passes):
                        print(f"    concurrent/{ctx}tok/pass{p}...", end=" ", flush=True)
                        r = await test_concurrent(backend, base_url, corpus, ctx, p)
                        q = "OK" if r.get("quality_ok") else "FAIL"
                        print(f"avg_TTFT={r.get('avg_ttft_ms', 0):.0f}ms Q={q}")
                        results["measurements"].append(r)
                        await asyncio.sleep(COOLDOWN_SECONDS)
                server.stop()

    # --- Post-restart test (THE key differentiator) ---
    if not skip_restart:
        print(f"\n{'='*60}")
        print("  POST-RESTART COMPARISON (persistence test)")
        print(f"{'='*60}")
        # Use subset of contexts for restart test (expensive: 2 server starts per test)
        restart_contexts = [c for c in contexts if c <= 4096][:3]
        for ctx in restart_contexts:
            for p in range(min(passes, 2)):  # Fewer passes (restart is slow)
                for backend in backends_to_test:
                    print(f"    restart/{backend}/{ctx}tok/pass{p}...", end=" ", flush=True)
                    if backend == "agent-memory":
                        factory = lambda _p=port_am, _m=model_key: AgentMemoryServer(_p, _m)
                    else:
                        factory = lambda _p=port_vllm, _m=model_key: VllmMlxServer(_p, _m)
                    r = await test_post_restart(backend, factory, corpus, ctx, p)
                    if r.get("error"):
                        print(f"ERROR: {r['error']}")
                    else:
                        q = "OK" if r.get("quality_ok") else "FAIL"
                        prime = r.get("prime_ttft_ms", 0)
                        print(f"TTFT={r.get('ttft_ms', 0):.0f}ms "
                              f"(prime={prime:.0f}ms) Q={q}")
                    results["measurements"].append(r)
                    await asyncio.sleep(5)

    results["metadata"]["timestamp_end"] = datetime.now(UTC).isoformat()
    return results


# ---------------------------------------------------------------------------
# Summary and reporting
# ---------------------------------------------------------------------------


def print_summary(results: dict[str, Any]) -> None:
    """Print comparison summary table."""
    measurements = results["measurements"]
    if not measurements:
        print("No measurements to summarize.")
        return

    print(f"\n{'='*70}")
    print("  COMPARISON SUMMARY")
    print(f"{'='*70}")

    # Group by scenario, backend, context
    from collections import defaultdict
    groups: dict[tuple, list] = defaultdict(list)
    for m in measurements:
        key = (m.get("scenario", ""), m.get("backend", ""), m.get("context_tokens", 0))
        groups[key].append(m)

    scenarios = sorted(set(m.get("scenario", "") for m in measurements))
    backends = sorted(set(m.get("backend", "") for m in measurements))
    contexts = sorted(set(m.get("context_tokens", 0) for m in measurements))

    for scenario in scenarios:
        print(f"\n  --- {scenario.upper()} ---")
        header = f"  {'Context':>8s}"
        for b in backends:
            header += f"  {b:>16s}"
        if len(backends) == 2:
            header += f"  {'Speedup':>10s}"
        print(header)
        print("  " + "-" * len(header))

        for ctx in contexts:
            row = f"  {ctx:>8d}"
            values = {}
            for b in backends:
                key = (scenario, b, ctx)
                ms = groups.get(key, [])
                if scenario == "concurrent":
                    ttfts = [m.get("avg_ttft_ms", 0) for m in ms if not m.get("error")]
                else:
                    ttfts = [m.get("ttft_ms", 0) for m in ms if not m.get("error")]
                if ttfts:
                    median = sorted(ttfts)[len(ttfts) // 2]
                    values[b] = median
                    row += f"  {median:>13.0f}ms"
                else:
                    row += f"  {'N/A':>16s}"
            if len(backends) == 2 and all(b in values for b in backends):
                am_val = values.get("agent-memory", 0)
                vllm_val = values.get("vllm-mlx", 0)
                if am_val > 0 and vllm_val > 0:
                    ratio = vllm_val / am_val
                    winner = "am" if am_val < vllm_val else "vllm"
                    row += f"  {ratio:>7.1f}x ({winner})"
            print(row)

    # Post-restart highlight
    restart_ms = [m for m in measurements if m.get("scenario") == "post_restart"]
    if restart_ms:
        print(f"\n  --- POST-RESTART HIGHLIGHT ---")
        for b in backends:
            bms = [m for m in restart_ms if m.get("backend") == b and not m.get("error")]
            if bms:
                ttfts = [m["ttft_ms"] for m in bms]
                primes = [m.get("prime_ttft_ms", 0) for m in bms]
                med_ttft = sorted(ttfts)[len(ttfts) // 2]
                med_prime = sorted(primes)[len(primes) // 2]
                ratio = med_prime / med_ttft if med_ttft > 0 else 0
                print(f"  {b:>16s}: prime={med_prime:.0f}ms → restart={med_ttft:.0f}ms "
                      f"(speedup: {ratio:.1f}x)")


def save_results(results: dict[str, Any], model_key: str) -> Path:
    """Save results to JSON."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = RESULTS_DIR / f"vllm_comparison_{model_key}_{ts}.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved: {path}")
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Head-to-head benchmark: agent-memory vs vllm-mlx"
    )
    parser.add_argument(
        "--model", choices=list(MODEL_IDS.keys()), default="gemma",
        help="Model to benchmark (default: gemma)",
    )
    parser.add_argument(
        "--contexts", type=int, nargs="+", default=None,
        help=f"Context lengths (default: {ALL_CONTEXTS})",
    )
    parser.add_argument(
        "--passes", type=int, default=DEFAULT_PASSES,
        help=f"Passes per configuration (default: {DEFAULT_PASSES})",
    )
    parser.add_argument(
        "--port-am", type=int, default=DEFAULT_PORT_AM,
        help=f"agent-memory port (default: {DEFAULT_PORT_AM})",
    )
    parser.add_argument(
        "--port-vllm", type=int, default=DEFAULT_PORT_VLLM,
        help=f"vllm-mlx port (default: {DEFAULT_PORT_VLLM})",
    )
    parser.add_argument("--am-only", action="store_true", help="Only benchmark agent-memory")
    parser.add_argument("--vllm-only", action="store_true", help="Only benchmark vllm-mlx")
    parser.add_argument("--skip-restart", action="store_true", help="Skip post-restart test")
    parser.add_argument("--skip-concurrent", action="store_true", help="Skip batch=2 test")
    parser.add_argument(
        "--cache-memory-percent", type=float, default=0.40,
        help="vllm-mlx KV cache memory fraction (default: 0.40 = 40%% of RAM)",
    )

    args = parser.parse_args()
    contexts = args.contexts or ALL_CONTEXTS

    print(f"{'#'*60}")
    print(f"  vllm-mlx vs agent-memory Comparison Benchmark")
    print(f"  Model: {MODEL_IDS[args.model]}")
    print(f"  Contexts: {contexts}")
    print(f"  Passes: {args.passes}")
    print(f"{'#'*60}")

    results = asyncio.run(run_comparison(
        model_key=args.model,
        contexts=contexts,
        passes=args.passes,
        port_am=args.port_am,
        port_vllm=args.port_vllm,
        skip_am=args.vllm_only,
        skip_vllm=args.am_only,
        skip_restart=args.skip_restart,
        skip_concurrent=args.skip_concurrent,
        cache_memory_percent=args.cache_memory_percent,
    ))

    print_summary(results)
    save_results(results, args.model)


if __name__ == "__main__":
    main()
