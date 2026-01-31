"""OpenAI Chat Completions benchmark suite for semantic server.

Benchmarks the /v1/chat/completions endpoint with:
- Cold start latency across context lengths
- Warm cache reuse (session mode)
- Multi-turn conversation extend
- Output length scaling (decode throughput)
- Concurrent request throughput
- Prefix sharing isolation

Supports external servers (LM Studio) for direct comparison.

Usage:
    # Against semantic server (manages server lifecycle)
    python benchmarks/openai_benchmark.py --quick
    python benchmarks/openai_benchmark.py --context-mode session --runs 3

    # Against LM Studio (external, user manages server)
    python benchmarks/openai_benchmark.py --external --base-url http://127.0.0.1:1234

    # Comparison mode
    python benchmarks/openai_benchmark.py --compare \
        --semantic-url http://127.0.0.1:8399 \
        --lmstudio-url http://127.0.0.1:1234
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))

from capability_benchmark import (
    MemoryProbe,
    ScenarioResult,
    ServerManager,
    compute_stats,
    PADDING_TEXT,
)

RESULTS_DIR = Path(__file__).resolve().parent / "results"
OUTPUT_TOKENS = 64
RUNS_DEFAULT = 3
RUNS_QUICK = 1
PORT = 8399

OPENAI_BENCH_ENV: dict[str, str] = {
    "SEMANTIC_MLX_MAX_BATCH_SIZE": "2",
    "SEMANTIC_MLX_SCHEDULER_ENABLED": "true",
    "SEMANTIC_MLX_CHUNKED_PREFILL_ENABLED": "true",
    "SEMANTIC_MLX_CHUNKED_PREFILL_THRESHOLD": "2048",
    "SEMANTIC_MLX_CHUNKED_PREFILL_MIN_CHUNK": "512",
    "SEMANTIC_MLX_CHUNKED_PREFILL_MAX_CHUNK": "4096",
    "SEMANTIC_MLX_PREFILL_STEP_SIZE": "256",
    "SEMANTIC_MLX_DEFAULT_TEMPERATURE": "0.0",
    "SEMANTIC_MLX_KV_BITS": "4",
    "SEMANTIC_MLX_MAX_CONTEXT_LENGTH": "100000",
    "SEMANTIC_SERVER_LOG_LEVEL": "WARNING",
    "SEMANTIC_API_KEY": "",
}


# ---------------------------------------------------------------------------
# Prompt factory (OpenAI format)
# ---------------------------------------------------------------------------

class OpenAIPromptFactory:
    """Build prompts in OpenAI chat format at target token counts."""

    def __init__(self) -> None:
        self._padding = PADDING_TEXT

    def build_messages(self, target_tokens: int) -> list[dict[str, str]]:
        chars_needed = max(target_tokens * 4, 100)
        repeats = (chars_needed // len(self._padding)) + 1
        content = (self._padding * repeats)[:chars_needed]
        return [{"role": "user", "content": content}]

    def build_system_messages(
        self, system_tokens: int, user_content: str
    ) -> list[dict[str, str]]:
        chars_needed = max(system_tokens * 4, 100)
        repeats = (chars_needed // len(self._padding)) + 1
        sys_content = (self._padding * repeats)[:chars_needed]
        return [
            {"role": "system", "content": sys_content},
            {"role": "user", "content": user_content},
        ]

    def build_request(
        self, target_tokens: int, max_tokens: int = OUTPUT_TOKENS
    ) -> dict[str, Any]:
        return {
            "model": "default",
            "messages": self.build_messages(target_tokens),
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "stream": False,
        }

    def build_followup(
        self,
        original_messages: list[dict[str, str]],
        assistant_text: str,
        max_tokens: int = OUTPUT_TOKENS,
    ) -> dict[str, Any]:
        messages = list(original_messages) + [
            {"role": "assistant", "content": assistant_text},
            {"role": "user", "content": "Continue explaining in more detail."},
        ]
        return {
            "model": "default",
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "stream": False,
        }

    def build_system_request(
        self,
        system_tokens: int,
        user_content: str,
        max_tokens: int = OUTPUT_TOKENS,
    ) -> dict[str, Any]:
        return {
            "model": "default",
            "messages": self.build_system_messages(system_tokens, user_content),
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "stream": False,
        }


# ---------------------------------------------------------------------------
# OpenAI request client (non-streaming)
# ---------------------------------------------------------------------------

class OpenAIRequestClient:
    """Non-streaming client for /v1/chat/completions with latency measurement."""

    def __init__(self, base_url: str) -> None:
        self.url = f"{base_url}/v1/chat/completions"
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(300.0))

    async def close(self) -> None:
        await self.client.aclose()

    async def send_and_measure(
        self,
        body: dict[str, Any],
        session_id: str | None = None,
    ) -> ScenarioResult:
        headers: dict[str, str] = {}
        if session_id:
            headers["X-Session-ID"] = session_id

        request_body = {**body}
        request_body["stream"] = False

        t_start = time.perf_counter()
        try:
            resp = await self.client.post(
                self.url, json=request_body, headers=headers
            )
        except Exception as exc:
            t_end = time.perf_counter()
            return ScenarioResult(
                scenario="", config="",
                e2e_ms=(t_end - t_start) * 1000,
                error=str(exc),
            )

        t_end = time.perf_counter()
        e2e_s = t_end - t_start

        if resp.status_code != 200:
            return ScenarioResult(
                scenario="", config="",
                e2e_ms=e2e_s * 1000,
                error=f"HTTP {resp.status_code}: {resp.text[:200]}",
            )

        data = resp.json()
        usage = data.get("usage", {})
        output_tokens = usage.get("completion_tokens", 0)
        input_tokens = usage.get("prompt_tokens", 0)

        raw_output = ""
        choices = data.get("choices", [])
        if choices:
            msg = choices[0].get("message", {})
            raw_output = msg.get("content", "") or ""

        tps = (output_tokens / e2e_s) if e2e_s > 0 and output_tokens > 0 else 0

        return ScenarioResult(
            scenario="", config="",
            e2e_ms=e2e_s * 1000,
            ttft_ms=e2e_s * 1000,
            tpot_ms=(e2e_s / output_tokens * 1000) if output_tokens else 0,
            decode_tps=tps,
            overall_tps=tps,
            output_tokens=output_tokens,
            input_tokens=input_tokens,
            raw_output=raw_output,
        )


# ---------------------------------------------------------------------------
# OpenAI streaming client (for TTFT measurement)
# ---------------------------------------------------------------------------

class OpenAIStreamingClient:
    """SSE streaming client for /v1/chat/completions with TTFT measurement."""

    def __init__(self, base_url: str) -> None:
        self.url = f"{base_url}/v1/chat/completions"

    async def send_and_measure(
        self,
        body: dict[str, Any],
        session_id: str | None = None,
    ) -> ScenarioResult:
        headers: dict[str, str] = {}
        if session_id:
            headers["X-Session-ID"] = session_id

        request_body = {**body, "stream": True}
        t_start = time.perf_counter()
        ttft = 0.0
        output_tokens = 0
        input_tokens = 0
        text = ""
        delta_count = 0

        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(300.0)
            ) as client:
                async with client.stream(
                    "POST", self.url,
                    json=request_body, headers=headers,
                ) as resp:
                    if resp.status_code != 200:
                        t_end = time.perf_counter()
                        return ScenarioResult(
                            scenario="", config="",
                            e2e_ms=(t_end - t_start) * 1000,
                            error=f"HTTP {resp.status_code}",
                        )

                    buffer = ""
                    async for chunk in resp.aiter_text():
                        buffer += chunk.replace("\r\n", "\n")
                        while "\n\n" in buffer:
                            event_str, buffer = buffer.split("\n\n", 1)
                            for line in event_str.strip().split("\n"):
                                if not line.startswith("data:"):
                                    continue
                                payload = line[5:].strip()
                                if payload == "[DONE]":
                                    continue
                                try:
                                    parsed = json.loads(payload)
                                except json.JSONDecodeError:
                                    continue

                                # Extract usage from final chunk (if present)
                                chunk_usage = parsed.get("usage", {})
                                if chunk_usage:
                                    output_tokens = chunk_usage.get(
                                        "completion_tokens", output_tokens
                                    )
                                    input_tokens = chunk_usage.get(
                                        "prompt_tokens", input_tokens
                                    )

                                choices = parsed.get("choices", [])
                                if not choices:
                                    continue
                                delta = choices[0].get("delta", {})
                                content = delta.get("content", "")
                                if content:
                                    if ttft == 0.0:
                                        ttft = (
                                            time.perf_counter() - t_start
                                        ) * 1000
                                    text += content
                                    delta_count += 1

        except Exception as exc:
            t_end = time.perf_counter()
            return ScenarioResult(
                scenario="", config="",
                e2e_ms=(t_end - t_start) * 1000,
                error=str(exc),
            )

        t_end = time.perf_counter()
        e2e_ms = (t_end - t_start) * 1000

        # If server didn't provide token count, use delta count
        if output_tokens == 0:
            output_tokens = delta_count

        decode_ms = e2e_ms - ttft if ttft > 0 else e2e_ms
        decode_tps = (
            (output_tokens / (decode_ms / 1000))
            if decode_ms > 0 and output_tokens > 0
            else 0
        )

        return ScenarioResult(
            scenario="", config="",
            ttft_ms=ttft,
            e2e_ms=e2e_ms,
            tpot_ms=(decode_ms / output_tokens) if output_tokens else 0,
            decode_tps=decode_tps,
            overall_tps=decode_tps,
            output_tokens=output_tokens,
            input_tokens=input_tokens,
            raw_output=text,
        )


# ---------------------------------------------------------------------------
# Benchmark suite
# ---------------------------------------------------------------------------

CONTEXT_SIZES = {"short": 200, "medium": 2000, "long": 8000, "xl": 32000}
CONTEXT_SIZES_QUICK = {"short": 200, "medium": 2000}

OUTPUT_LENGTHS = [16, 64, 128, 256, 512]
OUTPUT_LENGTHS_QUICK = [16, 64, 256]


class OpenAIBenchmarkSuite:
    """OpenAI /v1/chat/completions benchmark with LM Studio comparison."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8399",
        port: int = PORT,
        runs: int = RUNS_DEFAULT,
        quick: bool = False,
        external: bool = False,
        context_mode: str = "full",
        output_path: str | None = None,
        server_label: str = "semantic",
    ) -> None:
        self.base_url = base_url
        self.port = port
        self.runs = runs
        self.quick = quick
        self.external = external
        self.context_mode = context_mode
        self.server_label = server_label
        self.prompt = OpenAIPromptFactory()

        self.server: ServerManager | None = None
        if not external:
            self.server = ServerManager(port=port)

        self.memory: MemoryProbe | None = None
        if not external:
            self.memory = MemoryProbe(base_url)

        self.results: dict[str, Any] = {
            "metadata": self._build_metadata(),
            "experiments": {},
        }

        self.output_path = output_path or self._default_output_path()

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
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "server": self.server_label,
            "base_url": self.base_url,
            "context_mode": self.context_mode,
            "api": "openai",
            "endpoint": "/v1/chat/completions",
            "machine": {
                "os": platform.system(),
                "os_version": platform.release(),
                "chip": platform.machine(),
            },
            "git_sha": sha,
            "runs_per_scenario": self.runs,
            "quick": self.quick,
        }

    def _default_output_path(self) -> str:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        return str(RESULTS_DIR / f"openai_{self.server_label}_{ts}.json")

    def _save(self) -> None:
        Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, "w") as f:
            json.dump(self.results, f, indent=2, default=str)

    async def _delete_agent(self, agent_id: str) -> None:
        if self.external:
            return
        async with httpx.AsyncClient(timeout=10.0) as c:
            try:
                await c.delete(f"{self.base_url}/v1/agents/{agent_id}")
            except Exception:
                pass

    async def _memory_snapshot(self) -> dict[str, float]:
        if self.memory is None:
            return {}
        try:
            return await self.memory.snapshot()
        except Exception:
            return {}

    def _store_experiment(
        self, name: str, scenarios: dict[str, Any]
    ) -> None:
        self.results["experiments"][name] = scenarios
        self._save()

    # --- Warmup ---

    async def _warmup(self) -> None:
        client = OpenAIRequestClient(self.base_url)
        try:
            body = self.prompt.build_request(1000, 16)
            result = await client.send_and_measure(body)
            if result.error:
                print(f"  [WARMUP] Failed: {result.error[:100]}")
            else:
                print(
                    f"  [WARMUP] OK — E2E={result.e2e_ms:.0f}ms "
                    f"TPS={result.decode_tps:.1f}"
                )
        finally:
            await client.close()
        await asyncio.sleep(2)

    # --- Experiment 1: Cold Start Latency ---

    async def exp_cold_start(self) -> None:
        print("\n  === Experiment 1: Cold Start Latency ===")
        sizes = CONTEXT_SIZES_QUICK if self.quick else CONTEXT_SIZES
        scenarios: dict[str, Any] = {}

        for label, tokens in sizes.items():
            scenario_name = f"cold_{label}"
            print(f"\n  [{scenario_name}]")

            # Streaming-only measurement: TTFT from first delta,
            # decode TPS from (output_tokens / (E2E - TTFT))
            stream_client = OpenAIStreamingClient(self.base_url)
            body = self.prompt.build_request(tokens, OUTPUT_TOKENS)
            runs_data: list[dict[str, Any]] = []

            for i in range(self.runs):
                sid = f"cold_{label}_r{i}" if self.context_mode == "session" else None
                sr = await stream_client.send_and_measure(body, session_id=sid)
                sr.scenario = scenario_name
                sr.config = self.server_label
                mem = await self._memory_snapshot()
                sr.peak_memory_mb = mem.get("peak_memory_mb", 0)
                runs_data.append(asdict(sr))

                print(
                    f"    run {i + 1}/{self.runs} "
                    f"TTFT={sr.ttft_ms:.0f}ms "
                    f"E2E={sr.e2e_ms:.0f}ms "
                    f"TPS={sr.decode_tps:.1f} "
                    f"out={sr.output_tokens}"
                )

                # Cleanup agent cache to ensure next run is cold
                if sid:
                    await self._delete_agent(f"oai_{sid}")

            # Compute stats
            stats: dict[str, Any] = {}
            for metric in ["ttft_ms", "e2e_ms", "decode_tps", "peak_memory_mb"]:
                vals = [r[metric] for r in runs_data if not r.get("error")]
                stats[metric] = compute_stats(vals)

            scenarios[scenario_name] = {"runs": runs_data, "stats": stats}

        self._store_experiment("1_cold_start", scenarios)

    # --- Experiment 2: Warm Cache ---

    async def exp_warm_cache(self) -> None:
        if self.external:
            print("\n  === Experiment 2: Warm Cache — SKIPPED (external server) ===")
            return
        if self.context_mode != "session":
            print("\n  === Experiment 2: Warm Cache — SKIPPED (requires session mode) ===")
            return

        print("\n  === Experiment 2: Warm Cache (session mode) ===")
        scenarios: dict[str, Any] = {}
        client = OpenAIRequestClient(self.base_url)

        try:
            for i in range(self.runs):
                sid = f"warm_test_r{i}"
                body = self.prompt.build_request(2000, OUTPUT_TOKENS)

                # Cold prime
                cold = await client.send_and_measure(body, session_id=sid)

                # Warm hit (same session, same messages)
                warm = await client.send_and_measure(body, session_id=sid)

                speedup = cold.e2e_ms / warm.e2e_ms if warm.e2e_ms > 0 else 0

                print(
                    f"    run {i + 1}/{self.runs} "
                    f"cold={cold.e2e_ms:.0f}ms "
                    f"warm={warm.e2e_ms:.0f}ms "
                    f"speedup={speedup:.1f}x"
                )

                scenarios[f"warm_r{i}"] = {
                    "cold": asdict(cold),
                    "warm": asdict(warm),
                    "speedup": speedup,
                }

                await self._delete_agent(f"oai_{sid}")
        finally:
            await client.close()

        # Aggregate stats
        cold_e2es = [
            s["cold"]["e2e_ms"] for s in scenarios.values()
            if not s["cold"].get("error")
        ]
        warm_e2es = [
            s["warm"]["e2e_ms"] for s in scenarios.values()
            if not s["warm"].get("error")
        ]
        speedups = [s["speedup"] for s in scenarios.values() if s["speedup"] > 0]

        scenarios["stats"] = {
            "cold_e2e_ms": compute_stats(cold_e2es),
            "warm_e2e_ms": compute_stats(warm_e2es),
            "speedup": compute_stats(speedups),
        }

        self._store_experiment("2_warm_cache", scenarios)

    # --- Experiment 3: Multi-turn Extend ---

    async def exp_multi_turn(self) -> None:
        print("\n  === Experiment 3: Multi-turn Extend ===")
        scenarios: dict[str, Any] = {}
        client = OpenAIRequestClient(self.base_url)
        n_turns = 3

        try:
            for i in range(self.runs):
                sid = f"multiturn_r{i}" if self.context_mode == "session" else None
                turn_results: list[dict[str, Any]] = []
                messages = self.prompt.build_messages(2000)
                body: dict[str, Any] = {
                    "model": "default",
                    "messages": messages,
                    "max_tokens": OUTPUT_TOKENS,
                    "temperature": 0.0,
                    "stream": False,
                }

                for turn in range(n_turns):
                    r = await client.send_and_measure(body, session_id=sid)
                    r.scenario = f"turn_{turn}"
                    r.config = self.server_label
                    turn_results.append(asdict(r))

                    print(
                        f"    run {i + 1}/{self.runs} turn {turn + 1} "
                        f"E2E={r.e2e_ms:.0f}ms "
                        f"TPS={r.decode_tps:.1f} "
                        f"out={r.output_tokens}"
                    )

                    if r.error:
                        break

                    # Build follow-up with assistant response
                    assistant_text = r.raw_output or "Understood."
                    messages = messages + [
                        {"role": "assistant", "content": assistant_text},
                        {"role": "user", "content": "Continue explaining in more detail."},
                    ]
                    body = {
                        "model": "default",
                        "messages": messages,
                        "max_tokens": OUTPUT_TOKENS,
                        "temperature": 0.0,
                        "stream": False,
                    }

                scenarios[f"multiturn_r{i}"] = {"turns": turn_results}

                if sid:
                    await self._delete_agent(f"oai_{sid}")
        finally:
            await client.close()

        # Compute per-turn stats
        turn_stats: dict[str, Any] = {}
        for turn in range(n_turns):
            e2es = []
            tpses = []
            for key, val in scenarios.items():
                if key.startswith("multiturn_r"):
                    turns = val.get("turns", [])
                    if turn < len(turns) and not turns[turn].get("error"):
                        e2es.append(turns[turn]["e2e_ms"])
                        tpses.append(turns[turn]["decode_tps"])
            turn_stats[f"turn_{turn}"] = {
                "e2e_ms": compute_stats(e2es),
                "decode_tps": compute_stats(tpses),
            }
        scenarios["stats"] = turn_stats

        self._store_experiment("3_multi_turn", scenarios)

    # --- Experiment 4: Output Length Scaling ---

    async def exp_output_scaling(self) -> None:
        print("\n  === Experiment 4: Output Length Scaling ===")
        lengths = OUTPUT_LENGTHS_QUICK if self.quick else OUTPUT_LENGTHS
        scenarios: dict[str, Any] = {}
        client = OpenAIRequestClient(self.base_url)

        try:
            for max_out in lengths:
                scenario_name = f"output_{max_out}"
                print(f"\n  [{scenario_name}]")
                run_data: list[dict[str, Any]] = []

                for i in range(self.runs):
                    sid = f"outscale_{max_out}_r{i}" if self.context_mode == "session" else None
                    body = self.prompt.build_request(1000, max_out)
                    r = await client.send_and_measure(body, session_id=sid)
                    r.scenario = scenario_name
                    r.config = self.server_label
                    run_data.append(asdict(r))

                    print(
                        f"    run {i + 1}/{self.runs} "
                        f"E2E={r.e2e_ms:.0f}ms "
                        f"TPS={r.decode_tps:.1f} "
                        f"out={r.output_tokens}"
                    )

                    if sid:
                        await self._delete_agent(f"oai_{sid}")

                stats: dict[str, Any] = {}
                for metric in ["e2e_ms", "tpot_ms", "decode_tps"]:
                    vals = [r[metric] for r in run_data if not r.get("error")]
                    stats[metric] = compute_stats(vals)

                scenarios[scenario_name] = {"runs": run_data, "stats": stats}
        finally:
            await client.close()

        self._store_experiment("4_output_scaling", scenarios)

    # --- Experiment 5: Concurrent Requests ---

    async def exp_concurrent(self) -> None:
        if self.external:
            print("\n  === Experiment 5: Concurrent Requests — SKIPPED (external) ===")
            return

        print("\n  === Experiment 5: Concurrent Requests (batch=2) ===")
        scenarios: dict[str, Any] = {}

        for i in range(self.runs):
            body = self.prompt.build_request(2000, OUTPUT_TOKENS)

            clients = [
                OpenAIRequestClient(self.base_url),
                OpenAIRequestClient(self.base_url),
            ]

            sids = [
                f"concurrent_r{i}_c0" if self.context_mode == "session" else None,
                f"concurrent_r{i}_c1" if self.context_mode == "session" else None,
            ]

            t_wall_start = time.perf_counter()
            try:
                results = await asyncio.gather(
                    clients[0].send_and_measure(body, session_id=sids[0]),
                    clients[1].send_and_measure(body, session_id=sids[1]),
                )
            except Exception as exc:
                print(f"    run {i + 1}/{self.runs} ERROR: {exc}")
                for c in clients:
                    await c.close()
                continue
            t_wall_end = time.perf_counter()
            wall_ms = (t_wall_end - t_wall_start) * 1000

            for c in clients:
                await c.close()

            total_out = sum(r.output_tokens for r in results)
            wall_s = wall_ms / 1000
            system_tps = total_out / wall_s if wall_s > 0 else 0

            print(
                f"    run {i + 1}/{self.runs} "
                f"wall={wall_ms:.0f}ms "
                f"sysTPS={system_tps:.1f} "
                f"total_out={total_out}"
            )

            scenarios[f"concurrent_r{i}"] = {
                "wall_ms": wall_ms,
                "system_tps": system_tps,
                "total_output_tokens": total_out,
                "per_request": [asdict(r) for r in results],
            }

            for sid in sids:
                if sid:
                    await self._delete_agent(f"oai_{sid}")

        # Aggregate stats
        wall_vals = [s["wall_ms"] for s in scenarios.values() if isinstance(s, dict) and "wall_ms" in s]
        tps_vals = [s["system_tps"] for s in scenarios.values() if isinstance(s, dict) and "system_tps" in s]
        scenarios["stats"] = {
            "wall_ms": compute_stats(wall_vals),
            "system_tps": compute_stats(tps_vals),
        }

        self._store_experiment("5_concurrent", scenarios)

    # --- Experiment 6: Prefix Sharing ---

    async def exp_prefix_sharing(self) -> None:
        if self.external:
            print("\n  === Experiment 6: Prefix Sharing — SKIPPED (external) ===")
            return

        print("\n  === Experiment 6: Prefix Sharing ===")
        scenarios: dict[str, Any] = {}
        client = OpenAIRequestClient(self.base_url)

        try:
            for i in range(self.runs):
                # Two prompts with shared system prefix, different user messages
                body_a = self.prompt.build_system_request(
                    1000, "What is the maximum block size?", OUTPUT_TOKENS
                )
                body_b = self.prompt.build_system_request(
                    1000, "How does eviction work?", OUTPUT_TOKENS
                )

                if self.context_mode == "session":
                    sid = f"prefix_r{i}"
                    ra = await client.send_and_measure(body_a, session_id=sid)
                    rb = await client.send_and_measure(body_b, session_id=sid)
                    await self._delete_agent(f"oai_{sid}")
                else:
                    # Full context mode: no session_id, different hashes
                    ra = await client.send_and_measure(body_a)
                    rb = await client.send_and_measure(body_b)

                print(
                    f"    run {i + 1}/{self.runs} "
                    f"A_E2E={ra.e2e_ms:.0f}ms "
                    f"B_E2E={rb.e2e_ms:.0f}ms"
                )

                scenarios[f"prefix_r{i}"] = {
                    "prompt_a": asdict(ra),
                    "prompt_b": asdict(rb),
                }
        finally:
            await client.close()

        # Stats
        a_e2es = [
            s["prompt_a"]["e2e_ms"] for s in scenarios.values()
            if isinstance(s, dict) and "prompt_a" in s
            and not s["prompt_a"].get("error")
        ]
        b_e2es = [
            s["prompt_b"]["e2e_ms"] for s in scenarios.values()
            if isinstance(s, dict) and "prompt_b" in s
            and not s["prompt_b"].get("error")
        ]
        scenarios["stats"] = {
            "prompt_a_e2e_ms": compute_stats(a_e2es),
            "prompt_b_e2e_ms": compute_stats(b_e2es),
        }

        self._store_experiment("6_prefix_sharing", scenarios)

    # --- Full suite ---

    async def run_all(
        self, experiments: list[int] | None = None
    ) -> None:
        print(f"\nOpenAI Benchmark Suite ({self.server_label})")
        print(f"  Base URL: {self.base_url}")
        print(f"  Context mode: {self.context_mode}")
        print(f"  Runs: {self.runs}")
        print(f"  Output: {self.output_path}")

        if not self.external:
            print("\n[SERVER] Starting semantic server...")
            assert self.server is not None
            self.server.start(OPENAI_BENCH_ENV)

        try:
            await self._warmup()

            exp_map: dict[int, Any] = {
                1: self.exp_cold_start,
                2: self.exp_warm_cache,
                3: self.exp_multi_turn,
                4: self.exp_output_scaling,
                5: self.exp_concurrent,
                6: self.exp_prefix_sharing,
            }

            targets = experiments or sorted(exp_map.keys())
            for exp_num in targets:
                if exp_num in exp_map:
                    await exp_map[exp_num]()
                else:
                    print(f"\n  Unknown experiment: {exp_num}")

        finally:
            if not self.external and self.server is not None:
                self.server.stop()
                print("\n[SERVER] Stopped.")

        self._save()
        self._print_summary()
        print(f"\nResults saved to: {self.output_path}")

    # --- Summary ---

    def _print_summary(self) -> None:
        print(f"\n{'=' * 80}")
        print(f"  OpenAI Benchmark Results: {self.server_label}")
        print(f"{'=' * 80}")

        # Experiment 1: Cold start
        exp1 = self.results.get("experiments", {}).get("1_cold_start", {})
        if exp1:
            print(f"\n  {'Scenario':<20} {'TTFT':>8} {'E2E':>8} {'TPS':>7} {'Out':>5} {'Peak':>8}")
            print(f"  {'─' * 20} {'─' * 8} {'─' * 8} {'─' * 7} {'─' * 5} {'─' * 8}")
            for name, data in sorted(exp1.items()):
                if name == "stats" or not isinstance(data, dict) or "stats" not in data:
                    continue
                stats = data["stats"]
                ttft = stats.get("ttft_ms", {}).get("median", 0)
                e2e = stats.get("e2e_ms", {}).get("median", 0)
                tps = stats.get("decode_tps", {}).get("median", 0)
                peak = stats.get("peak_memory_mb", {}).get("median", 0)
                runs = data.get("runs", [])
                out = runs[0].get("output_tokens", 0) if runs else 0
                print(f"  {name:<20} {ttft:>8.0f} {e2e:>8.0f} {tps:>7.1f} {out:>5} {peak:>8.0f}")

        # Experiment 2: Warm cache
        exp2 = self.results.get("experiments", {}).get("2_warm_cache", {})
        if exp2 and "stats" in exp2:
            stats = exp2["stats"]
            cold_med = stats.get("cold_e2e_ms", {}).get("median", 0)
            warm_med = stats.get("warm_e2e_ms", {}).get("median", 0)
            speedup_med = stats.get("speedup", {}).get("median", 0)
            print(f"\n  Warm Cache: cold={cold_med:.0f}ms warm={warm_med:.0f}ms speedup={speedup_med:.1f}x")

        # Experiment 3: Multi-turn
        exp3 = self.results.get("experiments", {}).get("3_multi_turn", {})
        if exp3 and "stats" in exp3:
            print(f"\n  Multi-turn:")
            for turn_name, turn_stats in sorted(exp3["stats"].items()):
                e2e = turn_stats.get("e2e_ms", {}).get("median", 0)
                tps = turn_stats.get("decode_tps", {}).get("median", 0)
                print(f"    {turn_name}: E2E={e2e:.0f}ms TPS={tps:.1f}")

        # Experiment 4: Output scaling
        exp4 = self.results.get("experiments", {}).get("4_output_scaling", {})
        if exp4:
            print(f"\n  {'Output Len':<15} {'E2E':>8} {'TPOT':>8} {'TPS':>7}")
            print(f"  {'─' * 15} {'─' * 8} {'─' * 8} {'─' * 7}")
            for name, data in sorted(exp4.items()):
                if not isinstance(data, dict) or "stats" not in data:
                    continue
                stats = data["stats"]
                e2e = stats.get("e2e_ms", {}).get("median", 0)
                tpot = stats.get("tpot_ms", {}).get("median", 0)
                tps = stats.get("decode_tps", {}).get("median", 0)
                print(f"  {name:<15} {e2e:>8.0f} {tpot:>8.1f} {tps:>7.1f}")

        # Experiment 5: Concurrent
        exp5 = self.results.get("experiments", {}).get("5_concurrent", {})
        if exp5 and "stats" in exp5:
            stats = exp5["stats"]
            wall = stats.get("wall_ms", {}).get("median", 0)
            tps = stats.get("system_tps", {}).get("median", 0)
            print(f"\n  Concurrent (batch=2): wall={wall:.0f}ms sysTPS={tps:.1f}")


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def compare_results(
    semantic_path: str, lmstudio_path: str
) -> None:
    with open(semantic_path) as f:
        sem = json.load(f)
    with open(lmstudio_path) as f:
        lms = json.load(f)

    print(f"\n{'=' * 90}")
    print("  OpenAI Benchmark Comparison: Semantic vs LM Studio")
    print(f"{'=' * 90}")

    # Compare cold start
    sem_cold = sem.get("experiments", {}).get("1_cold_start", {})
    lms_cold = lms.get("experiments", {}).get("1_cold_start", {})

    if sem_cold and lms_cold:
        print(
            f"\n  {'Scenario':<20} │ "
            f"{'Semantic':>24} │ "
            f"{'LM Studio':>24} │ "
            f"{'Delta':>8}"
        )
        print(
            f"  {'':20} │ "
            f"{'TTFT':>8} {'E2E':>7} {'TPS':>7} │ "
            f"{'TTFT':>8} {'E2E':>7} {'TPS':>7} │ "
            f"{'TPS %':>8}"
        )
        print(f"  {'─' * 20}┼{'─' * 26}┼{'─' * 26}┼{'─' * 10}")

        all_scenarios = set(sem_cold.keys()) | set(lms_cold.keys())
        for name in sorted(all_scenarios):
            if name == "stats":
                continue
            sem_data = sem_cold.get(name, {})
            lms_data = lms_cold.get(name, {})

            sem_stats = sem_data.get("stats", {}) if isinstance(sem_data, dict) else {}
            lms_stats = lms_data.get("stats", {}) if isinstance(lms_data, dict) else {}

            sem_ttft = sem_stats.get("ttft_ms", {}).get("median", 0)
            sem_e2e = sem_stats.get("e2e_ms", {}).get("median", 0)
            sem_tps = sem_stats.get("decode_tps", {}).get("median", 0)
            lms_ttft = lms_stats.get("ttft_ms", {}).get("median", 0)
            lms_e2e = lms_stats.get("e2e_ms", {}).get("median", 0)
            lms_tps = lms_stats.get("decode_tps", {}).get("median", 0)

            delta = ""
            if sem_tps > 0 and lms_tps > 0:
                pct = ((sem_tps - lms_tps) / lms_tps) * 100
                delta = f"{pct:+.1f}%"

            print(
                f"  {name:<20} │ "
                f"{sem_ttft:>8.0f} {sem_e2e:>7.0f} {sem_tps:>7.1f} │ "
                f"{lms_ttft:>8.0f} {lms_e2e:>7.0f} {lms_tps:>7.1f} │ "
                f"{delta:>8}"
            )

    # Compare output scaling
    sem_out = sem.get("experiments", {}).get("4_output_scaling", {})
    lms_out = lms.get("experiments", {}).get("4_output_scaling", {})

    if sem_out and lms_out:
        print(
            f"\n  {'Output':<15} │ "
            f"{'Semantic TPS':>12} │ "
            f"{'LMStudio TPS':>12} │ "
            f"{'Delta':>8}"
        )
        print(f"  {'─' * 15}┼{'─' * 14}┼{'─' * 14}┼{'─' * 10}")

        all_out = set(sem_out.keys()) | set(lms_out.keys())
        for name in sorted(all_out):
            if name == "stats":
                continue
            sd = sem_out.get(name, {})
            ld = lms_out.get(name, {})
            s_stats = sd.get("stats", {}) if isinstance(sd, dict) else {}
            l_stats = ld.get("stats", {}) if isinstance(ld, dict) else {}
            s_tps = s_stats.get("decode_tps", {}).get("median", 0)
            l_tps = l_stats.get("decode_tps", {}).get("median", 0)

            delta = ""
            if s_tps > 0 and l_tps > 0:
                pct = ((s_tps - l_tps) / l_tps) * 100
                delta = f"{pct:+.1f}%"

            print(
                f"  {name:<15} │ "
                f"{s_tps:>12.1f} │ "
                f"{l_tps:>12.1f} │ "
                f"{delta:>8}"
            )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="OpenAI benchmark suite for /v1/chat/completions"
    )
    parser.add_argument(
        "--base-url", type=str, default="http://127.0.0.1:8399",
        help="Server base URL (default: http://127.0.0.1:8399)"
    )
    parser.add_argument(
        "--port", type=int, default=PORT,
        help=f"Server port for managed server (default: {PORT})"
    )
    parser.add_argument(
        "--external", action="store_true",
        help="External server (don't manage lifecycle)"
    )
    parser.add_argument(
        "--context-mode", choices=["full", "session"], default="full",
        help="Context mode: full (stateless) or session (with X-Session-ID)"
    )
    parser.add_argument(
        "--runs", type=int, default=None,
        help=f"Runs per scenario (default: {RUNS_DEFAULT}, quick: {RUNS_QUICK})"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick mode: fewer scenarios, 1 run"
    )
    parser.add_argument(
        "--experiment", nargs="+", type=int, choices=[1, 2, 3, 4, 5, 6],
        help="Run specific experiments"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output JSON file path"
    )
    parser.add_argument(
        "--label", type=str, default=None,
        help="Server label for results (default: auto-detect)"
    )

    # Comparison mode
    parser.add_argument(
        "--compare", action="store_true",
        help="Compare two result files"
    )
    parser.add_argument(
        "--semantic-results", type=str,
        help="Path to semantic server results JSON"
    )
    parser.add_argument(
        "--lmstudio-results", type=str,
        help="Path to LM Studio results JSON"
    )

    args = parser.parse_args()

    # Comparison mode
    if args.compare:
        if not args.semantic_results or not args.lmstudio_results:
            parser.error("--compare requires --semantic-results and --lmstudio-results")
        compare_results(args.semantic_results, args.lmstudio_results)
        return

    # Benchmark mode
    runs = args.runs
    if runs is None:
        runs = RUNS_QUICK if args.quick else RUNS_DEFAULT

    label = args.label
    if label is None:
        if args.external:
            if "1234" in args.base_url:
                label = "lmstudio"
            else:
                label = "external"
        else:
            label = "semantic"

    suite = OpenAIBenchmarkSuite(
        base_url=args.base_url,
        port=args.port,
        runs=runs,
        quick=args.quick,
        external=args.external,
        context_mode=args.context_mode,
        output_path=args.output,
        server_label=label,
    )

    asyncio.run(suite.run_all(args.experiment))


if __name__ == "__main__":
    main()
