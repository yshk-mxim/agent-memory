# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Yakov Shkolnikov and contributors
"""Capability benchmark suite for agent-memory server.

Measures: TTFT, ITL, TPOT, E2E, decode TPS, memory pressure.
Configs: single-request, batched, unchunked prefill.
Context lengths: 200, 2K, 8K, 32K tokens + memory pressure to 64K.

Usage:
    python benchmarks/capability_benchmark.py
    python benchmarks/capability_benchmark.py --quick
    python benchmarks/capability_benchmark.py --config single
    python benchmarks/capability_benchmark.py --pressure-only
    python benchmarks/capability_benchmark.py --resume benchmarks/results/capability_*.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import platform
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ScenarioResult:
    """Single run result with all metrics."""

    scenario: str
    config: str
    ttft_ms: float = 0.0
    e2e_ms: float = 0.0
    itl_mean_ms: float = 0.0
    itl_p95_ms: float = 0.0
    itl_p99_ms: float = 0.0
    tpot_ms: float = 0.0
    decode_tps: float = 0.0
    overall_tps: float = 0.0
    output_tokens: int = 0
    input_tokens: int = 0
    cache_created: int = 0
    cache_read: int = 0
    memory_before_mb: float = 0.0
    memory_after_mb: float = 0.0
    peak_memory_mb: float = 0.0
    raw_output: str = ""
    error: str | None = None


@dataclass
class ScenarioStats:
    """Aggregated stats across multiple runs."""

    mean: float = 0.0
    median: float = 0.0
    p95: float = 0.0
    p99: float = 0.0


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def percentile(values: list[float], pct: float) -> float:
    """Compute percentile using linear interpolation."""
    if not values:
        return 0.0
    s = sorted(values)
    k = (pct / 100) * (len(s) - 1)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return s[int(k)]
    return s[f] * (c - k) + s[c] * (k - f)


def compute_stats(values: list[float]) -> dict[str, float]:
    """Compute mean, median, P95, P99."""
    if not values:
        return {"mean": 0, "median": 0, "p95": 0, "p99": 0}
    return {
        "mean": sum(values) / len(values),
        "median": percentile(values, 50),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
    }


# ---------------------------------------------------------------------------
# Prompt factory
# ---------------------------------------------------------------------------

PADDING_TEXT = (
    "The system implements a block-based KV cache architecture where each "
    "block stores a fixed number of token key-value pairs. Blocks are "
    "allocated from a shared pool and assigned per-layer per-agent. When "
    "an agent's cache exceeds the hot tier capacity the least recently "
    "used agent is evicted to disk via safetensors serialization. On "
    "subsequent requests the cache is loaded from disk and reconstructed "
    "into quantized KV blocks. This design enables efficient memory "
    "management while preserving semantic context across conversations. "
    "The prefill phase processes input tokens in adaptive chunks to bound "
    "peak memory usage during long-context inference on Apple Silicon. "
)


class PromptFactory:
    """Build prompts at approximate target token counts."""

    def __init__(self, base_url: str) -> None:
        self.base = base_url
        self._padding = PADDING_TEXT

    def build_messages(self, target_tokens: int) -> list[dict]:
        """Build messages list targeting approximate token count."""
        # Rough estimate: 1 token ≈ 4 chars for English prose
        chars_needed = max(target_tokens * 4, 100)
        repeats = (chars_needed // len(self._padding)) + 1
        content = (self._padding * repeats)[:chars_needed]
        return [{"role": "user", "content": content}]

    def build_request(self, target_tokens: int, max_tokens: int = 128) -> dict:
        """Build full Anthropic API request body."""
        return {
            "model": "default",
            "max_tokens": max_tokens,
            "messages": self.build_messages(target_tokens),
            "temperature": 0.0,
            "top_p": 1.0,
        }

    def build_followup_request(
        self,
        original_messages: list[dict],
        assistant_response: str,
        max_tokens: int = 128,
    ) -> dict:
        """Build a multi-turn follow-up that extends the conversation.

        The server detects that the new prompt's token prefix matches the
        cached sequence and skips re-prefilling the cached portion.
        """
        messages = list(original_messages) + [
            {"role": "assistant", "content": assistant_response},
            {"role": "user", "content": "Continue explaining in more detail."},
        ]
        return {
            "model": "default",
            "max_tokens": max_tokens,
            "messages": messages,
            "temperature": 0.0,
            "top_p": 1.0,
        }


# ---------------------------------------------------------------------------
# Request client (non-streaming — server generates all tokens before response)
# ---------------------------------------------------------------------------


class RequestClient:
    """Async HTTP client that measures E2E latency from non-streaming requests.

    The server generates all tokens in a single blocking call before returning,
    so streaming does not provide real ITL. Non-streaming gives cleaner metrics.
    """

    def __init__(self, base_url: str) -> None:
        self.base = base_url
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(300.0))

    async def close(self) -> None:
        await self.client.aclose()

    async def send_and_measure(
        self,
        body: dict,
        session_id: str | None = None,
    ) -> ScenarioResult:
        """Send a non-streaming request and measure E2E latency."""
        headers: dict[str, str] = {}
        if session_id:
            headers["X-Session-ID"] = session_id

        # Ensure non-streaming
        request_body = {**body}
        request_body.pop("stream", None)

        t_start = time.perf_counter()
        try:
            resp = await self.client.post(
                f"{self.base}/v1/messages",
                json=request_body,
                headers=headers,
            )
        except Exception as exc:
            t_end = time.perf_counter()
            return ScenarioResult(
                scenario="",
                config="",
                e2e_ms=(t_end - t_start) * 1000,
                error=str(exc),
            )

        t_end = time.perf_counter()
        e2e_s = t_end - t_start

        if resp.status_code != 200:
            return ScenarioResult(
                scenario="",
                config="",
                e2e_ms=e2e_s * 1000,
                error=f"HTTP {resp.status_code}: {resp.text[:200]}",
            )

        data = resp.json()
        usage = data.get("usage", {})
        output_tokens = usage.get("output_tokens", 0)
        input_tokens = usage.get("input_tokens", 0)
        cache_created = usage.get("cache_creation_input_tokens", 0)
        cache_read = usage.get("cache_read_input_tokens", 0)

        # Extract raw output text
        content = data.get("content", [])
        raw_output = ""
        for block in content:
            if block.get("type") == "text":
                raw_output += block.get("text", "")

        # Compute TPS (all time is E2E since non-streaming)
        tps = (output_tokens / e2e_s) if e2e_s > 0 and output_tokens > 0 else 0

        return ScenarioResult(
            scenario="",
            config="",
            e2e_ms=e2e_s * 1000,
            ttft_ms=e2e_s * 1000,  # No streaming granularity
            tpot_ms=(e2e_s / output_tokens * 1000) if output_tokens else 0,
            decode_tps=tps,
            overall_tps=tps,
            output_tokens=output_tokens,
            input_tokens=input_tokens,
            cache_created=cache_created,
            cache_read=cache_read,
            raw_output=raw_output,
        )


# ---------------------------------------------------------------------------
# Memory probe
# ---------------------------------------------------------------------------


class MemoryProbe:
    """Query /debug/memory endpoint."""

    def __init__(self, base_url: str) -> None:
        self.base = base_url

    async def snapshot(self) -> dict[str, float]:
        async with httpx.AsyncClient(timeout=10.0) as c:
            resp = await c.get(f"{self.base}/debug/memory")
            if resp.status_code == 200:
                return resp.json()
        return {}


# ---------------------------------------------------------------------------
# Server manager
# ---------------------------------------------------------------------------


class ServerManager:
    """Start/stop agent-memory server subprocess."""

    def __init__(self, port: int = 8399) -> None:
        self.port = port
        self.proc: subprocess.Popen | None = None

    def start(self, env_overrides: dict[str, str] | None = None) -> None:
        """Start server with given env overrides."""
        env = os.environ.copy()
        if env_overrides:
            env.update(env_overrides)
        # Ensure empty API key for benchmark access
        env.setdefault("SEMANTIC_API_KEY", "")

        self.proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "agent_memory.entrypoints.cli",
                "serve",
                "--port",
                str(self.port),
                "--log-level",
                "WARNING",
            ],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(Path(__file__).resolve().parents[1]),
        )
        self._wait_for_startup()

    def _wait_for_startup(self, timeout: float = 180.0) -> None:
        """Poll health endpoint until server is ready."""
        deadline = time.time() + timeout
        url = f"http://127.0.0.1:{self.port}/health/startup"
        while time.time() < deadline:
            try:
                r = httpx.get(url, timeout=2.0)
                if r.status_code == 200:
                    data = r.json()
                    if data.get("status") == "started":
                        print(f"  Server ready on port {self.port}")
                        return
            except (httpx.ConnectError, httpx.ReadTimeout):
                pass
            time.sleep(2.0)
        raise TimeoutError(f"Server did not start within {timeout}s")

    def is_alive(self) -> bool:
        """Check if server process is still running."""
        return self.proc is not None and self.proc.poll() is None

    def stop(self) -> None:
        """Gracefully stop server."""
        if self.proc and self.proc.poll() is None:
            self.proc.send_signal(signal.SIGTERM)
            try:
                self.proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=5)
        self.proc = None


# ---------------------------------------------------------------------------
# Config definitions
# ---------------------------------------------------------------------------

CONFIG_A: dict[str, str] = {
    "SEMANTIC_MLX_MAX_BATCH_SIZE": "1",
    "SEMANTIC_MLX_SCHEDULER_ENABLED": "false",
    "SEMANTIC_MLX_CHUNKED_PREFILL_ENABLED": "true",
    "SEMANTIC_MLX_CHUNKED_PREFILL_THRESHOLD": "2048",
    "SEMANTIC_MLX_CHUNKED_PREFILL_MIN_CHUNK": "512",
    "SEMANTIC_MLX_KV_BITS": "4",
    "SEMANTIC_MLX_MAX_CONTEXT_LENGTH": "100000",
    "SEMANTIC_SERVER_LOG_LEVEL": "WARNING",
}

CONFIG_B: dict[str, str] = {
    **CONFIG_A,
    "SEMANTIC_MLX_MAX_BATCH_SIZE": "2",
    "SEMANTIC_MLX_SCHEDULER_ENABLED": "true",
}

CONFIG_C: dict[str, str] = {
    **CONFIG_A,
    "SEMANTIC_MLX_CHUNKED_PREFILL_ENABLED": "false",
    "SEMANTIC_MLX_CHUNKED_PREFILL_THRESHOLD": "16384",
}

CONTEXT_SIZES = {
    "short": 200,
    "medium": 2000,
    "long": 8000,
    "xl": 32000,
}

PRESSURE_SIZES = [32000, 48000, 64000]


# ---------------------------------------------------------------------------
# Benchmark suite
# ---------------------------------------------------------------------------


class BenchmarkSuite:
    """Orchestrates all benchmark scenarios across server configs."""

    def __init__(
        self,
        port: int = 8399,
        runs: int = 3,
        quick: bool = False,
        output_path: str | None = None,
        resume_path: str | None = None,
    ) -> None:
        self.port = port
        self.runs = runs
        self.quick = quick
        self.base_url = f"http://127.0.0.1:{port}"
        self.server = ServerManager(port=port)
        self.prompt = PromptFactory(self.base_url)
        self.memory = MemoryProbe(self.base_url)

        # Results storage
        self.results: dict[str, Any] = {
            "metadata": self._build_metadata(),
            "configs": {},
        }
        self.output_path = output_path or self._default_output_path()

        # Resume support
        if resume_path and Path(resume_path).exists():
            with open(resume_path) as f:
                self.results = json.load(f)
            print(f"[RESUME] Loaded {resume_path}")

    def _build_metadata(self) -> dict[str, Any]:
        try:
            sha = subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=str(Path(__file__).resolve().parent),
                text=True,
            ).strip()
        except Exception:
            sha = "unknown"
        return {
            "timestamp": datetime.now(UTC).isoformat(),
            "model_id": "mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx",
            "machine": {
                "os": platform.system(),
                "os_version": platform.release(),
                "chip": platform.machine(),
            },
            "git_sha": sha,
            "runs_per_scenario": self.runs,
        }

    def _default_output_path(self) -> str:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = Path(__file__).resolve().parent / "results"
        results_dir.mkdir(exist_ok=True)
        return str(results_dir / f"capability_{ts}.json")

    def _save_results(self) -> None:
        """Incrementally save results to JSON."""
        Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, "w") as f:
            json.dump(self.results, f, indent=2, default=str)

    def _is_completed(self, config_name: str, scenario_name: str) -> bool:
        """Check if scenario already completed (for resume)."""
        cfg = self.results.get("configs", {}).get(config_name, {})
        sc = cfg.get("scenarios", {}).get(scenario_name, {})
        runs = sc.get("runs", [])
        return len(runs) >= self.runs

    def _store_scenario(
        self,
        config_name: str,
        scenario_name: str,
        run_results: list[ScenarioResult],
    ) -> None:
        """Store scenario results and compute stats."""
        if config_name not in self.results["configs"]:
            self.results["configs"][config_name] = {"scenarios": {}}

        runs_data = [asdict(r) for r in run_results]

        # Compute stats for key metrics
        stats: dict[str, Any] = {}
        for metric in [
            "ttft_ms",
            "e2e_ms",
            "itl_mean_ms",
            "tpot_ms",
            "decode_tps",
            "overall_tps",
        ]:
            values = [r[metric] for r in runs_data if not r.get("error")]
            stats[metric] = compute_stats(values)

        self.results["configs"][config_name]["scenarios"][scenario_name] = {
            "runs": runs_data,
            "stats": stats,
        }
        self._save_results()

    # --- Warmup ---

    async def warmup_and_stabilize(self) -> None:
        """Memory warmup: pressure fill, stabilize, clear.

        Sends progressively larger requests to exercise the MLX compute
        path and push unified memory toward operating pressure.  Failures
        are non-fatal — the benchmark continues with whatever pressure
        was achieved.
        """
        client = RequestClient(self.base_url)
        warmup_sizes = [2000, 8000, 16000]
        if self.quick:
            warmup_sizes = [2000]

        try:
            for tokens in warmup_sizes:
                label = f"{tokens // 1000}K"
                print(f"[WARMUP] Sending {label}-token request...")
                body = self.prompt.build_request(target_tokens=tokens, max_tokens=16)
                result = await client.send_and_measure(body, session_id=f"warmup_{tokens}")
                if result.error:
                    print(f"[WARMUP] {label} failed: {result.error[:120]}")
                    break
                print(f"[WARMUP] {label} OK — E2E={result.e2e_ms:.0f}ms")

            mem = await self.memory.snapshot()
            peak = mem.get("peak_memory_mb", 0)
            print(f"[WARMUP] Peak memory: {peak:.0f} MB")

            if not self.quick:
                print("[WARMUP] Stabilizing for 60s (OS memory compressor)...")
                await asyncio.sleep(60)
            else:
                print("[WARMUP] Quick mode — skipping stabilization wait")
                await asyncio.sleep(2)

            # Clear warmup caches
            async with httpx.AsyncClient(timeout=10.0) as c:
                for tokens in warmup_sizes:
                    await c.delete(f"{self.base_url}/v1/agents/sess_warmup_{tokens}")

            mem_after = await self.memory.snapshot()
            active = mem_after.get("active_memory_mb", 0)
            print(f"[WARMUP] Post-clear memory: {active:.0f} MB")
            print("[WARMUP] Baseline established. Starting benchmarks.\n")
        finally:
            await client.close()

    # --- Cache cleanup ---

    async def _delete_agent(self, session_id: str) -> bool:
        """Delete a benchmark agent's cached KV data."""
        agent_id = f"sess_{session_id}"
        async with httpx.AsyncClient(timeout=10.0) as c:
            resp = await c.delete(f"{self.base_url}/v1/agents/{agent_id}")
            return resp.status_code == 204

    async def _cleanup_scenario_caches(
        self, scenario_name: str, sid_base: str, warm: bool = False
    ) -> None:
        """Delete all agent caches created by a scenario."""
        sids = []
        if warm:
            sids.append(sid_base)  # prime + measurement share same sid
        else:
            sids.append(f"{sid_base}_warmup")
            for i in range(self.runs):
                sids.append(f"{sid_base}_run{i}")
        for sid in sids:
            try:
                await self._delete_agent(sid)
            except Exception:
                pass

    # --- Single scenario runner ---

    async def _run_scenario(
        self,
        config_name: str,
        scenario_name: str,
        target_tokens: int,
        max_tokens: int = 128,
        session_id: str | None = None,
        warm: bool = False,
    ) -> list[ScenarioResult]:
        """Run a single scenario N times, return results."""
        if self._is_completed(config_name, scenario_name):
            print(f"  [{scenario_name}] Already completed, skipping")
            return []

        sse = RequestClient(self.base_url)
        results: list[ScenarioResult] = []
        body = self.prompt.build_request(target_tokens, max_tokens)
        sid = session_id or f"bench_{scenario_name}"

        try:
            # If warm scenario, do a cold prime first, then build follow-up
            if warm:
                print(f"  [{scenario_name}] Priming cache...")
                try:
                    prime_result = await sse.send_and_measure(body, session_id=sid)
                except (httpx.ConnectError, httpx.RemoteProtocolError) as e:
                    print(f"  [{scenario_name}] PRIME CRASHED: {type(e).__name__}")
                    return results
                # Build multi-turn follow-up that extends the conversation.
                # The server matches the cached token prefix and skips
                # re-prefilling the original prompt — only new tokens
                # (assistant response + follow-up question) need processing.
                original_messages = self.prompt.build_messages(target_tokens)
                assistant_text = prime_result.raw_output or "Understood."
                body = self.prompt.build_followup_request(
                    original_messages, assistant_text, max_tokens
                )

            # Warmup run (not measured)
            if not warm:
                print(f"  [{scenario_name}] Warmup run...")
                try:
                    await sse.send_and_measure(body, session_id=f"{sid}_warmup")
                except (httpx.ConnectError, httpx.RemoteProtocolError) as e:
                    print(f"  [{scenario_name}] WARMUP CRASHED: {type(e).__name__}")
                    return results

            for i in range(self.runs):
                try:
                    mem_before = await self.memory.snapshot()
                except Exception:
                    mem_before = {}

                run_sid = sid if warm else f"{sid}_run{i}"
                try:
                    result = await sse.send_and_measure(body, session_id=run_sid)
                except (httpx.ConnectError, httpx.RemoteProtocolError) as e:
                    print(
                        f"  [{scenario_name}] run {i + 1}/{self.runs} "
                        f"SERVER CRASHED: {type(e).__name__}"
                    )
                    break

                try:
                    mem_after = await self.memory.snapshot()
                except Exception:
                    mem_after = {}

                result.scenario = scenario_name
                result.config = config_name
                result.memory_before_mb = mem_before.get("active_memory_mb", 0)
                result.memory_after_mb = mem_after.get("active_memory_mb", 0)
                result.peak_memory_mb = mem_after.get("peak_memory_mb", 0)

                results.append(result)
                print(
                    f"  [{scenario_name}] run {i + 1}/{self.runs} "
                    f"TTFT={result.ttft_ms:.0f}ms "
                    f"E2E={result.e2e_ms:.0f}ms "
                    f"TPS={result.decode_tps:.1f}"
                )
        finally:
            await sse.close()

        self._store_scenario(config_name, scenario_name, results)
        return results

    # --- Concurrent scenario runner ---

    async def _run_concurrent(
        self,
        config_name: str,
        scenario_name: str,
        target_tokens: int,
        n_concurrent: int = 2,
        max_tokens: int = 128,
    ) -> list[ScenarioResult]:
        """Run N concurrent requests, measure system throughput."""
        if self._is_completed(config_name, scenario_name):
            print(f"  [{scenario_name}] Already completed, skipping")
            return []

        sse = RequestClient(self.base_url)
        all_results: list[ScenarioResult] = []

        try:
            for run_i in range(self.runs):
                body = self.prompt.build_request(target_tokens, max_tokens)
                t_wall_start = time.perf_counter()

                coros = [
                    sse.send_and_measure(body, session_id=f"{scenario_name}_r{run_i}_c{j}")
                    for j in range(n_concurrent)
                ]
                try:
                    results = await asyncio.gather(*coros)
                except (httpx.ConnectError, httpx.RemoteProtocolError) as e:
                    print(
                        f"  [{scenario_name}] run {run_i + 1}/{self.runs} "
                        f"SERVER CRASHED: {type(e).__name__}"
                    )
                    break
                t_wall_end = time.perf_counter()
                wall_s = t_wall_end - t_wall_start

                total_out = sum(r.output_tokens for r in results)
                system_tps = total_out / wall_s if wall_s > 0 else 0

                # Store first result with system-level metrics
                main = results[0]
                main.scenario = scenario_name
                main.config = config_name
                main.overall_tps = system_tps
                all_results.append(main)

                print(
                    f"  [{scenario_name}] run {run_i + 1}/{self.runs} "
                    f"wall={wall_s * 1000:.0f}ms "
                    f"system_tps={system_tps:.1f} "
                    f"total_out={total_out}"
                )
        finally:
            await sse.close()

        self._store_scenario(config_name, scenario_name, all_results)
        return all_results

    # --- Interleave stall test ---

    async def _run_interleave_stall(self, config_name: str) -> list[ScenarioResult]:
        """Send A(2K) then B(8K) 0.5s later. Measure A's ITL stability."""
        scenario_name = "interleave_stall"
        if self._is_completed(config_name, scenario_name):
            print(f"  [{scenario_name}] Already completed, skipping")
            return []

        sse = RequestClient(self.base_url)
        all_results: list[ScenarioResult] = []

        try:
            body_a = self.prompt.build_request(2000, 128)
            body_b = self.prompt.build_request(8000, 128)

            for run_i in range(self.runs):
                task_a = asyncio.create_task(
                    sse.send_and_measure(body_a, session_id=f"stall_a_{run_i}")
                )
                await asyncio.sleep(0.5)  # Let A begin decoding
                task_b = asyncio.create_task(
                    sse.send_and_measure(body_b, session_id=f"stall_b_{run_i}")
                )
                try:
                    result_a, result_b = await asyncio.gather(task_a, task_b)
                except (httpx.ConnectError, httpx.RemoteProtocolError) as e:
                    print(
                        f"  [{scenario_name}] run {run_i + 1}/{self.runs} "
                        f"SERVER CRASHED: {type(e).__name__}"
                    )
                    break
                result_a.scenario = scenario_name
                result_a.config = config_name
                all_results.append(result_a)

                print(
                    f"  [{scenario_name}] run {run_i + 1}/{self.runs} "
                    f"A_ITL_p99={result_a.itl_p99_ms:.1f}ms "
                    f"B_TTFT={result_b.ttft_ms:.0f}ms"
                )
        finally:
            await sse.close()

        self._store_scenario(config_name, scenario_name, all_results)
        return all_results

    # --- Config runners ---

    async def run_config(self, config_name: str, env: dict[str, str]) -> None:
        """Run all scenarios for a given config."""
        sizes = CONTEXT_SIZES
        if self.quick:
            sizes = {"short": 200, "medium": 2000}

        print(f"\n{'=' * 70}")
        print(f"  Config: {config_name}")
        print(f"{'=' * 70}")

        # Cold scenarios
        for label, tokens in sizes.items():
            scenario_name = f"cold_{label}"
            sid = f"bench_{scenario_name}"
            await self._run_scenario(config_name, scenario_name, tokens)
            await self._cleanup_scenario_caches(scenario_name, sid)
            if not self.server.is_alive():
                print(f"  [!] Server crashed — skipping remaining {config_name} scenarios")
                return

        # Warm scenarios
        for label, tokens in sizes.items():
            scenario_name = f"warm_{label}"
            sid = f"bench_{scenario_name}"
            await self._run_scenario(config_name, scenario_name, tokens, warm=True)
            await self._cleanup_scenario_caches(scenario_name, sid, warm=True)
            if not self.server.is_alive():
                print(f"  [!] Server crashed — skipping remaining {config_name} scenarios")
                return

    async def run_config_b_extras(self) -> None:
        """Batch-only concurrent scenarios."""
        print("\n  --- Batch concurrent scenarios ---")
        await self._run_concurrent("batched", "concurrent_2x_medium", 2000)
        if not self.server.is_alive():
            return

        if not self.quick:
            await self._run_concurrent("batched", "concurrent_2x_long", 8000)
            if not self.server.is_alive():
                return
            await self._run_interleave_stall("batched")

    async def run_chunked_comparison(self) -> None:
        """Run chunked scenarios for later comparison with unchunked."""
        for label, tokens in [("medium", 2000), ("long", 8000)]:
            scenario = f"chunked_{label}"
            await self._run_scenario("single", scenario, tokens)
            await self._cleanup_scenario_caches(scenario, f"bench_{scenario}")
            if not self.server.is_alive():
                return

    async def run_unchunked_comparison(self) -> None:
        """Run unchunked at <10K contexts for comparison with chunked."""
        for label, tokens in [("medium", 2000), ("long", 8000)]:
            scenario = f"unchunked_{label}"
            await self._run_scenario("unchunked", scenario, tokens)
            await self._cleanup_scenario_caches(scenario, f"bench_{scenario}")
            if not self.server.is_alive():
                return

    async def run_memory_pressure(self) -> None:
        """Memory pressure test: 32K → 64K single-threaded."""
        print(f"\n{'=' * 70}")
        print("  Memory Pressure Test")
        print(f"{'=' * 70}")

        for tokens in PRESSURE_SIZES:
            scenario = f"pressure_{tokens // 1000}k"
            if self._is_completed("pressure", scenario):
                print(f"  [{scenario}] Already completed, skipping")
                continue

            sse = RequestClient(self.base_url)
            try:
                body = self.prompt.build_request(tokens, max_tokens=16)
                try:
                    mem_before = await self.memory.snapshot()
                except Exception:
                    mem_before = {}
                try:
                    result = await sse.send_and_measure(body, session_id=f"pressure_{tokens}")
                except (httpx.ConnectError, httpx.RemoteProtocolError) as e:
                    print(f"  [{scenario}] SERVER CRASHED: {type(e).__name__}")
                    break
                try:
                    mem_after = await self.memory.snapshot()
                except Exception:
                    mem_after = {}

                result.scenario = scenario
                result.config = "pressure"
                result.memory_before_mb = mem_before.get("active_memory_mb", 0)
                result.memory_after_mb = mem_after.get("active_memory_mb", 0)
                result.peak_memory_mb = mem_after.get("peak_memory_mb", 0)

                status_str = "OK" if not result.error else f"ERR: {result.error}"
                print(
                    f"  [{scenario}] "
                    f"TTFT={result.ttft_ms:.0f}ms "
                    f"peak={result.peak_memory_mb:.0f}MB "
                    f"{status_str}"
                )
                self._store_scenario("pressure", scenario, [result])
            finally:
                await sse.close()

    # --- Full suite ---

    async def run_all(self) -> None:
        """Run complete benchmark suite."""
        print(f"\nResults will be saved to: {self.output_path}\n")

        # Config A: Single request
        print("[CONFIG A] Starting server (single, batch=1, scheduler=off)...")
        self.server.start(CONFIG_A)
        try:
            await self.warmup_and_stabilize()
            await self.run_config("single", CONFIG_A)

            if not self.quick and self.server.is_alive():
                await self.run_chunked_comparison()
            if not self.quick and self.server.is_alive():
                await self.run_memory_pressure()
        finally:
            self.server.stop()
            print("[CONFIG A] Server stopped.\n")
            self._save_results()

        # Config B: Batched
        print("[CONFIG B] Starting server (batched, batch=2, scheduler=on)...")
        self.server.start(CONFIG_B)
        try:
            # Brief warmup (no full stabilization needed — OS already warmed)
            sse = RequestClient(self.base_url)
            try:
                body = self.prompt.build_request(2000, 16)
                await sse.send_and_measure(body, session_id="warmup_b")
            except (httpx.ConnectError, httpx.RemoteProtocolError):
                print("  [!] Warmup failed — server may have crashed")
            finally:
                await sse.close()

            if self.server.is_alive():
                await self.run_config("batched", CONFIG_B)
            if self.server.is_alive():
                await self.run_config_b_extras()
        finally:
            self.server.stop()
            print("[CONFIG B] Server stopped.\n")
            self._save_results()

        # Config C: Unchunked (only 32K comparison)
        if not self.quick:
            print("[CONFIG C] Starting server (unchunked, chunked_prefill=disabled)...")
            self.server.start(CONFIG_C)
            try:
                sse = RequestClient(self.base_url)
                try:
                    body = self.prompt.build_request(2000, 16)
                    await sse.send_and_measure(body, session_id="warmup_c")
                except (httpx.ConnectError, httpx.RemoteProtocolError):
                    print("  [!] Warmup failed — server may have crashed")
                finally:
                    await sse.close()

                if self.server.is_alive():
                    await self.run_unchunked_comparison()
            finally:
                self.server.stop()
                print("[CONFIG C] Server stopped.\n")

        self._save_results()
        self._print_summary()
        print(f"\nResults saved to: {self.output_path}")

    async def run_single_config(self, config_name: str) -> None:
        """Run a single named config."""
        configs = {
            "single": CONFIG_A,
            "batched": CONFIG_B,
            "unchunked": CONFIG_C,
        }
        if config_name not in configs:
            print(f"Unknown config: {config_name}")
            return

        env = configs[config_name]
        print(f"[{config_name.upper()}] Starting server...")
        self.server.start(env)
        try:
            await self.warmup_and_stabilize()
            await self.run_config(config_name, env)
            if config_name == "batched":
                await self.run_config_b_extras()
        finally:
            self.server.stop()

        self._save_results()
        self._print_summary()
        print(f"\nResults saved to: {self.output_path}")

    async def run_pressure_only(self) -> None:
        """Run only memory pressure tests."""
        print("[PRESSURE] Starting server...")
        self.server.start(CONFIG_A)
        try:
            await self.warmup_and_stabilize()
            await self.run_memory_pressure()
        finally:
            self.server.stop()

        self._save_results()
        self._print_pressure_table()
        print(f"\nResults saved to: {self.output_path}")

    # --- Output formatting ---

    def _print_summary(self) -> None:
        """Print summary tables to console."""
        for config_name, cfg_data in self.results.get("configs", {}).items():
            scenarios = cfg_data.get("scenarios", {})
            if not scenarios:
                continue

            if config_name == "pressure":
                self._print_pressure_table()
                continue

            print(f"\n{'═' * 90}")
            print(f"  Config: {config_name}")
            print(f"{'═' * 90}")
            print(
                f"{'Scenario':<22} │ {'Input':>6} │ {'Output':>6} │ "
                f"{'E2E':>9} │ {'TPOT':>7} │ "
                f"{'TPS':>7} │ {'CacheR':>6} │ {'Peak':>8}"
            )
            print(
                f"{'':22} │ {'(tok)':>6} │ {'(tok)':>6} │ "
                f"{'med(ms)':>9} │ {'(ms)':>7} │ "
                f"{'(tok/s)':>7} │ {'(tok)':>6} │ {'Mem(MB)':>8}"
            )
            print(
                f"{'─' * 22}┼{'─' * 8}┼{'─' * 8}┼{'─' * 11}┼{'─' * 9}┼{'─' * 9}┼{'─' * 8}┼{'─' * 10}"
            )

            for sc_name, sc_data in scenarios.items():
                stats = sc_data.get("stats", {})
                runs = sc_data.get("runs", [])
                if not runs:
                    continue

                input_tok = runs[0].get("input_tokens", 0)
                output_tok = runs[0].get("output_tokens", 0)
                peak_mem = max(r.get("peak_memory_mb", 0) for r in runs)
                tpot_med = stats.get("tpot_ms", {}).get("median", 0)
                e2e_med = stats.get("e2e_ms", {}).get("median", 0)
                tps_med = stats.get("decode_tps", {}).get("median", 0)
                cache_read = runs[0].get("cache_read", 0)

                print(
                    f"{sc_name:<22} │ {input_tok:>6} │ {output_tok:>6} │ "
                    f"{e2e_med:>9.0f} │ {tpot_med:>7.1f} │ "
                    f"{tps_med:>7.1f} │ {cache_read:>6} │ {peak_mem:>8.0f}"
                )

    def _print_pressure_table(self) -> None:
        """Print memory pressure results."""
        pressure = self.results.get("configs", {}).get("pressure", {})
        scenarios = pressure.get("scenarios", {})
        if not scenarios:
            return

        print(f"\n{'═' * 72}")
        print("  Memory Pressure Test (single-threaded, chunked prefill 512)")
        print(f"{'═' * 72}")
        print(
            f"{'Context (tok)':>14} │ {'TTFT (ms)':>10} │ "
            f"{'Peak Mem (MB)':>14} │ {'Active (MB)':>12} │ {'Status':>8}"
        )
        print(f"{'─' * 14}┼{'─' * 12}┼{'─' * 16}┼{'─' * 14}┼{'─' * 10}")

        for sc_name in sorted(scenarios.keys()):
            runs = scenarios[sc_name].get("runs", [])
            if not runs:
                continue
            r = runs[0]
            tokens = r.get("input_tokens", 0)
            ttft = r.get("ttft_ms", 0)
            peak = r.get("peak_memory_mb", 0)
            active = r.get("memory_after_mb", 0)
            err = r.get("error")
            st = "ERR" if err else "OK"
            print(f"{tokens:>14,} │ {ttft:>10,.0f} │ {peak:>14,.0f} │ {active:>12,.0f} │ {st:>8}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Capability benchmark suite for agent-memory server"
    )
    parser.add_argument("--quick", action="store_true", help="Quick mode: 1 run, skip xl/pressure")
    parser.add_argument(
        "--config", choices=["single", "batched", "unchunked"], help="Run only a specific config"
    )
    parser.add_argument(
        "--pressure-only", action="store_true", help="Run only memory pressure tests"
    )
    parser.add_argument(
        "--runs", type=int, default=3, help="Number of runs per scenario (default: 3)"
    )
    parser.add_argument("--port", type=int, default=8399, help="Server port (default: 8399)")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file path")
    parser.add_argument(
        "--resume", type=str, default=None, help="Resume from a previous results JSON file"
    )
    args = parser.parse_args()

    runs = 1 if args.quick else args.runs
    suite = BenchmarkSuite(
        port=args.port,
        runs=runs,
        quick=args.quick,
        output_path=args.output,
        resume_path=args.resume,
    )

    if args.pressure_only:
        asyncio.run(suite.run_pressure_only())
    elif args.config:
        asyncio.run(suite.run_single_config(args.config))
    else:
        asyncio.run(suite.run_all())


if __name__ == "__main__":
    main()
