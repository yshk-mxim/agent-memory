# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Yakov Shkolnikov and contributors
"""Profiling benchmark suite for optimal parameter discovery.

Runs 9 experiments across all tunable parameters to find the optimal
configuration for a given model on the user's hardware.

Experiments:
    A: Prefill step size sweep (cold start)
    B: Chunked prefill tuning (cold start, long context)
    C: Batch size vs throughput (concurrent requests)
    D: Batch window tuning (staggered arrivals)
    E: Output length scaling (decode throughput)
    F: Cache hit ratio impact (cold/warm/hot)
    G: LRU eviction pressure (agent cycling)
    H: Asymmetric batch workloads (mismatched lengths)
    I: KV bits comparison (Q4 vs Q8 vs FP16)

Usage:
    python benchmarks/profiling_benchmark.py
    python benchmarks/profiling_benchmark.py --quick
    python benchmarks/profiling_benchmark.py --experiment A
    python benchmarks/profiling_benchmark.py --experiment A C E --quick
"""

from __future__ import annotations

import argparse
import asyncio
import json
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

# Ensure benchmarks directory is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Reuse infrastructure from capability_benchmark
from capability_benchmark import (
    MemoryProbe,
    PromptFactory,
    RequestClient,
    ScenarioResult,
    ServerManager,
    compute_stats,
)

RESULTS_DIR = Path(__file__).resolve().parent / "results"
OUTPUT_TOKENS = 64
RUNS_DEFAULT = 5
RUNS_QUICK = 2
PORT = 8399


# ---------------------------------------------------------------------------
# Base environment (shared across experiments)
# ---------------------------------------------------------------------------

BASE_ENV: dict[str, str] = {
    "SEMANTIC_MLX_CHUNKED_PREFILL_ENABLED": "true",
    "SEMANTIC_MLX_CHUNKED_PREFILL_THRESHOLD": "2048",
    "SEMANTIC_MLX_CHUNKED_PREFILL_MIN_CHUNK": "512",
    "SEMANTIC_MLX_CHUNKED_PREFILL_MAX_CHUNK": "4096",
    "SEMANTIC_MLX_KV_BITS": "4",
    "SEMANTIC_MLX_MAX_CONTEXT_LENGTH": "100000",
    "SEMANTIC_MLX_MAX_BATCH_SIZE": "1",
    "SEMANTIC_MLX_SCHEDULER_ENABLED": "false",
    "SEMANTIC_MLX_PREFILL_STEP_SIZE": "256",
    "SEMANTIC_SERVER_LOG_LEVEL": "WARNING",
    "SEMANTIC_API_KEY": "",
}


# ---------------------------------------------------------------------------
# Streaming client (for TTFT measurement)
# ---------------------------------------------------------------------------


class StreamingClient:
    """Async HTTP client that measures TTFT via SSE streaming."""

    def __init__(self, base_url: str) -> None:
        self.base = base_url

    async def send_and_measure(
        self,
        body: dict,
        session_id: str | None = None,
    ) -> ScenarioResult:
        """Send a streaming request and measure TTFT and E2E.

        Token counts come from the server's message_delta usage field,
        NOT from counting SSE events.
        """
        headers: dict[str, str] = {}
        if session_id:
            headers["X-Session-ID"] = session_id

        request_body = {**body, "stream": True}
        t_start = time.perf_counter()
        ttft = 0.0
        delta_count = 0  # SSE text_delta events (for TTFT only)
        output_tokens = 0  # Authoritative count from message_delta
        input_tokens = 0
        cache_created = 0
        cache_read = 0
        text = ""

        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as client:
                async with client.stream(
                    "POST",
                    f"{self.base}/v1/messages",
                    json=request_body,
                    headers=headers,
                ) as resp:
                    if resp.status_code != 200:
                        t_end = time.perf_counter()
                        return ScenarioResult(
                            scenario="",
                            config="",
                            e2e_ms=(t_end - t_start) * 1000,
                            error=f"HTTP {resp.status_code}",
                        )

                    buffer = ""
                    async for chunk in resp.aiter_text():
                        buffer += chunk.replace("\r\n", "\n")
                        while "\n\n" in buffer:
                            event_str, buffer = buffer.split("\n\n", 1)
                            event_type = None
                            event_data = None
                            for line in event_str.strip().split("\n"):
                                if line.startswith("event:"):
                                    event_type = line[6:].strip()
                                elif line.startswith("data:"):
                                    event_data = line[5:].strip()

                            if not event_data:
                                continue

                            if event_type == "content_block_delta":
                                parsed = json.loads(event_data)
                                delta = parsed.get("delta", {})
                                if delta.get("type") == "text_delta":
                                    if ttft == 0.0:
                                        ttft = (time.perf_counter() - t_start) * 1000
                                    delta_count += 1
                                    text += delta.get("text", "")

                            elif event_type == "message_start":
                                parsed = json.loads(event_data)
                                msg = parsed.get("message", {})
                                usage = msg.get("usage", {})
                                input_tokens = usage.get("input_tokens", 0)
                                cache_created = usage.get("cache_creation_input_tokens", 0)
                                cache_read = usage.get("cache_read_input_tokens", 0)

                            elif event_type == "message_delta":
                                parsed = json.loads(event_data)
                                usage = parsed.get("usage", {})
                                out = usage.get("output_tokens", 0)
                                if out > 0:
                                    output_tokens = out

        except Exception as exc:
            t_end = time.perf_counter()
            return ScenarioResult(
                scenario="",
                config="",
                e2e_ms=(t_end - t_start) * 1000,
                error=str(exc),
            )

        t_end = time.perf_counter()
        e2e_ms = (t_end - t_start) * 1000
        decode_ms = e2e_ms - ttft if ttft > 0 else e2e_ms
        decode_tps = (
            (output_tokens / (decode_ms / 1000)) if decode_ms > 0 and output_tokens > 0 else 0
        )

        return ScenarioResult(
            scenario="",
            config="",
            ttft_ms=ttft,
            e2e_ms=e2e_ms,
            tpot_ms=(e2e_ms / output_tokens * 1000) if output_tokens else 0,
            decode_tps=decode_tps,
            overall_tps=decode_tps,
            output_tokens=output_tokens,
            input_tokens=input_tokens,
            cache_created=cache_created,
            cache_read=cache_read,
            raw_output=text,
        )


# ---------------------------------------------------------------------------
# Experiment definitions
# ---------------------------------------------------------------------------


@dataclass
class ExperimentScenario:
    """Single scenario within an experiment."""

    name: str
    target_tokens: int
    max_tokens: int = OUTPUT_TOKENS
    session_id: str | None = None
    warm: bool = False
    concurrent: int = 1
    stagger_ms: float = 0.0


@dataclass
class Experiment:
    """A named experiment with env overrides and scenario list."""

    name: str
    description: str
    env_overrides: dict[str, str]
    scenarios: list[ExperimentScenario]


def build_experiments(quick: bool = False) -> dict[str, list[Experiment]]:
    """Build the full experiment matrix."""
    experiments: dict[str, list[Experiment]] = {}

    # --- Experiment A: Prefill Step Size Sweep ---
    input_lengths = [200, 1000, 2000, 4000, 8000, 16000, 32000]
    step_sizes = [128, 256, 512, 1024, 2048]
    if quick:
        input_lengths = [200, 2000, 8000]
        step_sizes = [256, 512, 1024]

    exp_a = []
    for step in step_sizes:
        scenarios = [
            ExperimentScenario(
                name=f"prefill_step{step}_input{tok}",
                target_tokens=tok,
            )
            for tok in input_lengths
        ]
        exp_a.append(
            Experiment(
                name=f"A_step{step}",
                description=f"Prefill step size = {step}",
                env_overrides={
                    **BASE_ENV,
                    "SEMANTIC_MLX_PREFILL_STEP_SIZE": str(step),
                },
                scenarios=scenarios,
            )
        )
    experiments["A"] = exp_a

    # --- Experiment B: Chunked Prefill Tuning ---
    long_inputs = [4000, 8000, 16000, 32000]
    min_chunks = [256, 512, 1024]
    max_chunks = [2048, 4096, 8192]
    if quick:
        long_inputs = [4000, 16000]
        min_chunks = [512]
        max_chunks = [2048, 4096]

    exp_b = []
    for min_c in min_chunks:
        for max_c in max_chunks:
            if min_c >= max_c:
                continue
            scenarios = [
                ExperimentScenario(
                    name=f"chunk_min{min_c}_max{max_c}_input{tok}",
                    target_tokens=tok,
                )
                for tok in long_inputs
            ]
            exp_b.append(
                Experiment(
                    name=f"B_min{min_c}_max{max_c}",
                    description=f"Chunk bounds = [{min_c}, {max_c}]",
                    env_overrides={
                        **BASE_ENV,
                        "SEMANTIC_MLX_CHUNKED_PREFILL_MIN_CHUNK": str(min_c),
                        "SEMANTIC_MLX_CHUNKED_PREFILL_MAX_CHUNK": str(max_c),
                    },
                    scenarios=scenarios,
                )
            )
    experiments["B"] = exp_b

    # --- Experiment C: Batch Size vs Throughput ---
    batch_sizes = [1, 2, 3, 4]
    batch_inputs = [1000, 4000]
    if quick:
        batch_sizes = [1, 2, 3]
        batch_inputs = [1000]

    exp_c = []
    for bs in batch_sizes:
        scenarios = [
            ExperimentScenario(
                name=f"batch{bs}_input{tok}",
                target_tokens=tok,
                concurrent=bs,
            )
            for tok in batch_inputs
        ]
        exp_c.append(
            Experiment(
                name=f"C_batch{bs}",
                description=f"Batch size = {bs}",
                env_overrides={
                    **BASE_ENV,
                    "SEMANTIC_MLX_MAX_BATCH_SIZE": str(bs),
                    "SEMANTIC_MLX_SCHEDULER_ENABLED": "true" if bs > 1 else "false",
                },
                scenarios=scenarios,
            )
        )
    experiments["C"] = exp_c

    # --- Experiment D: Batch Window Tuning ---
    windows = [5, 10, 25, 50, 100]
    if quick:
        windows = [10, 50]

    exp_d = []
    for w in windows:
        scenarios = [
            ExperimentScenario(
                name=f"window{w}_stagger0",
                target_tokens=2000,
                concurrent=2,
                stagger_ms=0,
            ),
            ExperimentScenario(
                name=f"window{w}_stagger{w // 2}",
                target_tokens=2000,
                concurrent=2,
                stagger_ms=w / 2,
            ),
        ]
        exp_d.append(
            Experiment(
                name=f"D_window{w}",
                description=f"Batch window = {w}ms",
                env_overrides={
                    **BASE_ENV,
                    "SEMANTIC_MLX_MAX_BATCH_SIZE": "2",
                    "SEMANTIC_MLX_SCHEDULER_ENABLED": "true",
                    "SEMANTIC_AGENT_BATCH_WINDOW_MS": str(w),
                },
                scenarios=scenarios,
            )
        )
    experiments["D"] = exp_d

    # --- Experiment E: Output Length Scaling ---
    output_lengths = [16, 32, 64, 128, 256, 512]
    if quick:
        output_lengths = [16, 64, 256]

    scenarios_e = [
        ExperimentScenario(
            name=f"output{out}",
            target_tokens=1000,
            max_tokens=out,
        )
        for out in output_lengths
    ]
    experiments["E"] = [
        Experiment(
            name="E_output_scaling",
            description="Output length scaling",
            env_overrides={**BASE_ENV},
            scenarios=scenarios_e,
        )
    ]

    # --- Experiment F: Cache Hit Ratio ---
    experiments["F"] = [
        Experiment(
            name="F_cache_ratio",
            description="Cold/warm/hot cache ratios",
            env_overrides={**BASE_ENV},
            scenarios=[
                ExperimentScenario(name="cold_only", target_tokens=2000),
                ExperimentScenario(
                    name="warm_only",
                    target_tokens=2000,
                    warm=True,
                    session_id="cache_warm",
                ),
                ExperimentScenario(name="cold_warm_alt", target_tokens=2000),
            ],
        )
    ]

    # --- Experiment G: LRU Eviction Pressure ---
    agent_counts = [1, 3, 5, 10]
    if quick:
        agent_counts = [1, 5]

    exp_g = []
    for n in agent_counts:
        scenarios = [
            ExperimentScenario(
                name=f"agents{n}_cycle",
                target_tokens=1000,
                max_tokens=32,
            ),
        ]
        exp_g.append(
            Experiment(
                name=f"G_agents{n}",
                description=f"Max agents = {n}",
                env_overrides={
                    **BASE_ENV,
                    "SEMANTIC_AGENT_MAX_AGENTS_IN_MEMORY": str(n),
                },
                scenarios=scenarios,
            )
        )
    experiments["G"] = exp_g

    # --- Experiment H: Asymmetric Batch ---
    pairs = [(200, 8000), (500, 16000), (1000, 32000)]
    if quick:
        pairs = [(200, 8000)]

    scenarios_h = [
        ExperimentScenario(
            name=f"asym_{short}_{long}",
            target_tokens=long,
            concurrent=2,
        )
        for short, long in pairs
    ]
    experiments["H"] = [
        Experiment(
            name="H_asymmetric",
            description="Asymmetric batch workloads",
            env_overrides={
                **BASE_ENV,
                "SEMANTIC_MLX_MAX_BATCH_SIZE": "2",
                "SEMANTIC_MLX_SCHEDULER_ENABLED": "true",
            },
            scenarios=scenarios_h,
        )
    ]

    # --- Experiment I: KV Bits Comparison ---
    kv_configs = [("4", "Q4"), ("8", "Q8")]
    if not quick:
        kv_configs.append(("", "FP16"))

    exp_i_inputs = [2000, 8000]
    if quick:
        exp_i_inputs = [2000]

    exp_i = []
    for bits_str, label in kv_configs:
        scenarios = [
            ExperimentScenario(
                name=f"kv{label}_input{tok}",
                target_tokens=tok,
            )
            for tok in exp_i_inputs
        ]
        env = {**BASE_ENV}
        if bits_str:
            env["SEMANTIC_MLX_KV_BITS"] = bits_str
        else:
            env["SEMANTIC_MLX_KV_BITS"] = ""
        exp_i.append(
            Experiment(
                name=f"I_kv{label}",
                description=f"KV bits = {label}",
                env_overrides=env,
                scenarios=scenarios,
            )
        )
    experiments["I"] = exp_i

    return experiments


# ---------------------------------------------------------------------------
# Profiling benchmark runner
# ---------------------------------------------------------------------------


class ProfilingBenchmark:
    """Orchestrates all profiling experiments."""

    def __init__(
        self,
        port: int = PORT,
        runs: int = RUNS_DEFAULT,
        quick: bool = False,
        output_dir: str | None = None,
    ) -> None:
        self.port = port
        self.runs = runs
        self.quick = quick
        self.base_url = f"http://127.0.0.1:{port}"
        self.server = ServerManager(port=port)
        self.prompt = PromptFactory(self.base_url)
        self.memory = MemoryProbe(self.base_url)

        self.output_dir = Path(output_dir) if output_dir else RESULTS_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.json_path = self.output_dir / f"profiling_{ts}.json"
        self.summary_path = self.output_dir / f"profiling_{ts}_summary.txt"

        self.results: dict[str, Any] = {
            "metadata": self._build_metadata(),
            "experiments": {},
        }

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
            "quick": self.quick,
        }

    def _save(self) -> None:
        with open(self.json_path, "w") as f:
            json.dump(self.results, f, indent=2, default=str)

    async def _delete_agent(self, session_id: str) -> None:
        async with httpx.AsyncClient(timeout=10.0) as c:
            try:
                await c.delete(f"{self.base_url}/v1/agents/sess_{session_id}")
            except Exception:
                pass

    async def _warmup(self) -> None:
        """Quick warmup to stabilize Metal runtime."""
        client = RequestClient(self.base_url)
        try:
            body = self.prompt.build_request(1000, 16)
            result = await client.send_and_measure(body, session_id="warmup")
            if result.error:
                print(f"    [WARMUP] Failed: {result.error[:100]}")
            else:
                print(f"    [WARMUP] OK — E2E={result.e2e_ms:.0f}ms")
            await self._delete_agent("warmup")
        finally:
            await client.close()
        await asyncio.sleep(2)

    async def _run_single(
        self,
        scenario: ExperimentScenario,
        run_idx: int,
    ) -> ScenarioResult:
        """Run a single scenario once using non-streaming API."""
        sid = scenario.session_id or f"prof_{scenario.name}_r{run_idx}"

        # Clean previous cache for this session
        await self._delete_agent(sid)

        client = RequestClient(self.base_url)
        body = self.prompt.build_request(scenario.target_tokens, scenario.max_tokens)

        try:
            mem_before = await self.memory.snapshot()
        except Exception:
            mem_before = {}

        try:
            if scenario.warm:
                prime = await client.send_and_measure(body, session_id=sid)
                if prime.error:
                    return ScenarioResult(
                        scenario=scenario.name,
                        config="",
                        error=f"Prime failed: {prime.error}",
                    )
                original_messages = self.prompt.build_messages(scenario.target_tokens)
                body = self.prompt.build_followup_request(
                    original_messages,
                    prime.raw_output or "Understood.",
                    scenario.max_tokens,
                )

            if scenario.concurrent > 1:
                result = await self._run_concurrent(scenario, body, run_idx)
            else:
                result = await client.send_and_measure(body, session_id=sid)

        except Exception as exc:
            return ScenarioResult(
                scenario=scenario.name,
                config="",
                error=str(exc),
            )
        finally:
            await client.close()

        try:
            mem_after = await self.memory.snapshot()
        except Exception:
            mem_after = {}

        result.scenario = scenario.name
        result.memory_before_mb = mem_before.get("active_memory_mb", 0)
        result.memory_after_mb = mem_after.get("active_memory_mb", 0)
        result.peak_memory_mb = mem_after.get("peak_memory_mb", 0)

        await self._delete_agent(sid)
        return result

    async def _run_concurrent(
        self,
        scenario: ExperimentScenario,
        body: dict,
        run_idx: int,
    ) -> ScenarioResult:
        """Run concurrent requests and return total system throughput."""
        n = scenario.concurrent

        # For asymmetric experiments, vary input lengths
        bodies: list[dict] = []
        if "asym_" in scenario.name:
            parts = scenario.name.split("_")
            short_tok = int(parts[1])
            long_tok = int(parts[2])
            bodies = [
                self.prompt.build_request(short_tok, scenario.max_tokens),
                self.prompt.build_request(long_tok, scenario.max_tokens),
            ]
        else:
            bodies = [body] * n

        # Each concurrent request gets its own RequestClient
        clients = [RequestClient(self.base_url) for _ in range(n)]

        async def send_with_stagger(idx: int) -> ScenarioResult:
            if scenario.stagger_ms > 0 and idx > 0:
                await asyncio.sleep(scenario.stagger_ms / 1000)
            sid = f"prof_{scenario.name}_r{run_idx}_c{idx}"
            return await clients[idx].send_and_measure(
                bodies[min(idx, len(bodies) - 1)],
                session_id=sid,
            )

        t_wall_start = time.perf_counter()
        results = await asyncio.gather(*[send_with_stagger(i) for i in range(n)])
        t_wall_end = time.perf_counter()
        wall_ms = (t_wall_end - t_wall_start) * 1000

        # Close all clients
        for c in clients:
            await c.close()

        total_output = sum(r.output_tokens for r in results)
        wall_s = wall_ms / 1000
        system_tps = total_output / wall_s if wall_s > 0 else 0

        # Build result with system-level metrics
        main = results[0]
        main.output_tokens = total_output
        main.overall_tps = system_tps
        main.decode_tps = system_tps
        main.e2e_ms = wall_ms

        # Cleanup concurrent sessions
        for i in range(n):
            await self._delete_agent(f"prof_{scenario.name}_r{run_idx}_c{i}")

        return main

    async def _run_cache_ratio_experiment(
        self, exp: Experiment, run_idx: int
    ) -> list[ScenarioResult]:
        """Special handler for Experiment F (cache hit ratio)."""
        results: list[ScenarioResult] = []
        client = RequestClient(self.base_url)

        try:
            # Scenario 1: 100% cold (5 different agents)
            cold_results: list[ScenarioResult] = []
            for i in range(5):
                sid = f"cold_only_{run_idx}_{i}"
                await self._delete_agent(sid)
                body = self.prompt.build_request(2000, OUTPUT_TOKENS)
                r = await client.send_and_measure(body, session_id=sid)
                cold_results.append(r)
                await self._delete_agent(sid)

            cold_e2e = [r.e2e_ms for r in cold_results]
            cold_tps = [r.decode_tps for r in cold_results if r.decode_tps > 0]
            results.append(
                ScenarioResult(
                    scenario="F_cold_only",
                    config=exp.name,
                    e2e_ms=sum(cold_e2e) / len(cold_e2e),
                    decode_tps=sum(cold_tps) / len(cold_tps) if cold_tps else 0,
                    output_tokens=sum(r.output_tokens for r in cold_results),
                )
            )

            # Scenario 2: 100% warm (same agent, multi-turn)
            sid = f"warm_only_{run_idx}"
            await self._delete_agent(sid)
            body = self.prompt.build_request(2000, OUTPUT_TOKENS)

            prime = await client.send_and_measure(body, session_id=sid)
            if not prime.error:
                warm_results: list[ScenarioResult] = []
                for i in range(5):
                    msgs = self.prompt.build_messages(2000)
                    followup = self.prompt.build_followup_request(
                        msgs,
                        prime.raw_output or "Understood.",
                        OUTPUT_TOKENS,
                    )
                    r = await client.send_and_measure(followup, session_id=sid)
                    warm_results.append(r)

                warm_e2e = [r.e2e_ms for r in warm_results]
                warm_tps = [r.decode_tps for r in warm_results if r.decode_tps > 0]
                results.append(
                    ScenarioResult(
                        scenario="F_warm_only",
                        config=exp.name,
                        e2e_ms=sum(warm_e2e) / len(warm_e2e),
                        decode_tps=sum(warm_tps) / len(warm_tps) if warm_tps else 0,
                        output_tokens=sum(r.output_tokens for r in warm_results),
                    )
                )
            await self._delete_agent(sid)

            # Scenario 3: Hot path (10 sequential turns)
            sid = f"hot_path_{run_idx}"
            await self._delete_agent(sid)
            body = self.prompt.build_request(2000, 16)
            hot_results: list[ScenarioResult] = []
            prev_response = ""
            msgs = self.prompt.build_messages(2000)

            for i in range(10):
                if i == 0:
                    r = await client.send_and_measure(body, session_id=sid)
                else:
                    followup = self.prompt.build_followup_request(
                        msgs,
                        prev_response or "Continue.",
                        16,
                    )
                    r = await client.send_and_measure(followup, session_id=sid)
                hot_results.append(r)
                prev_response = r.raw_output or "Continue."

            hot_e2e = [r.e2e_ms for r in hot_results]
            hot_tps = [r.decode_tps for r in hot_results if r.decode_tps > 0]
            results.append(
                ScenarioResult(
                    scenario="F_hot_path",
                    config=exp.name,
                    e2e_ms=sum(hot_e2e) / len(hot_e2e),
                    ttft_ms=hot_e2e[0] if hot_e2e else 0,
                    decode_tps=sum(hot_tps) / len(hot_tps) if hot_tps else 0,
                    output_tokens=sum(r.output_tokens for r in hot_results),
                )
            )
            await self._delete_agent(sid)

        finally:
            await client.close()

        return results

    async def _run_eviction_experiment(self, exp: Experiment, run_idx: int) -> list[ScenarioResult]:
        """Special handler for Experiment G (LRU eviction)."""
        max_agents = int(exp.env_overrides.get("SEMANTIC_AGENT_MAX_AGENTS_IN_MEMORY", "5"))
        n_agents = max_agents + 2
        results: list[ScenarioResult] = []
        client = RequestClient(self.base_url)

        try:
            # Phase 1: Fill cache with n_agents
            fill_times = []
            for i in range(n_agents):
                sid = f"evict_{run_idx}_agent{i}"
                body = self.prompt.build_request(1000, 32)
                r = await client.send_and_measure(body, session_id=sid)
                fill_times.append(r.e2e_ms)

            results.append(
                ScenarioResult(
                    scenario=f"G_fill_{max_agents}",
                    config=exp.name,
                    e2e_ms=sum(fill_times) / len(fill_times),
                )
            )

            try:
                mem = await self.memory.snapshot()
            except Exception:
                mem = {}

            results[-1].peak_memory_mb = mem.get("peak_memory_mb", 0)

            # Phase 2: Revisit evicted agents (forces disk load)
            reload_times = []
            for i in range(min(2, n_agents)):
                sid = f"evict_{run_idx}_agent{i}"
                msgs = self.prompt.build_messages(1000)
                body = self.prompt.build_followup_request(msgs, "Previous response.", 32)
                r = await client.send_and_measure(body, session_id=sid)
                reload_times.append(r.e2e_ms)

            if reload_times:
                results.append(
                    ScenarioResult(
                        scenario=f"G_reload_{max_agents}",
                        config=exp.name,
                        e2e_ms=sum(reload_times) / len(reload_times),
                    )
                )

            # Cleanup
            for i in range(n_agents):
                await self._delete_agent(f"evict_{run_idx}_agent{i}")

        finally:
            await client.close()

        return results

    async def run_experiment_group(self, exp_key: str, exp_list: list[Experiment]) -> None:
        """Run all sub-experiments for a given experiment key."""
        print(f"\n{'=' * 70}")
        print(f"  Experiment {exp_key}")
        print(f"{'=' * 70}")

        exp_results: dict[str, Any] = {}

        for exp in exp_list:
            print(f"\n  --- {exp.name}: {exp.description} ---")

            # Start server with this experiment's config
            try:
                self.server.start(exp.env_overrides)
            except TimeoutError:
                print(f"    SERVER FAILED TO START — skipping {exp.name}")
                continue

            try:
                await self._warmup()
                scenario_results: dict[str, Any] = {}

                # Special handlers for F and G
                if exp_key == "F":
                    for run_i in range(self.runs):
                        sub_results = await self._run_cache_ratio_experiment(exp, run_i)
                        for r in sub_results:
                            key = r.scenario
                            if key not in scenario_results:
                                scenario_results[key] = {"runs": []}
                            scenario_results[key]["runs"].append(asdict(r))
                            print(f"    {key} run {run_i + 1}: E2E={r.e2e_ms:.0f}ms")

                elif exp_key == "G":
                    for run_i in range(self.runs):
                        sub_results = await self._run_eviction_experiment(exp, run_i)
                        for r in sub_results:
                            key = r.scenario
                            if key not in scenario_results:
                                scenario_results[key] = {"runs": []}
                            scenario_results[key]["runs"].append(asdict(r))
                            print(
                                f"    {key} run {run_i + 1}: "
                                f"E2E={r.e2e_ms:.0f}ms "
                                f"peak={r.peak_memory_mb:.0f}MB"
                            )

                else:
                    for scenario in exp.scenarios:
                        scenario_runs = []
                        for run_i in range(self.runs):
                            if not self.server.is_alive():
                                print("    SERVER CRASHED — aborting")
                                break
                            r = await self._run_single(scenario, run_i)
                            scenario_runs.append(asdict(r))
                            tps_label = "TPS" if scenario.concurrent <= 1 else "sysTPS"
                            status = (
                                f"TTFT={r.ttft_ms:.0f}ms "
                                f"E2E={r.e2e_ms:.0f}ms "
                                f"{tps_label}={r.decode_tps:.1f} "
                                f"out={r.output_tokens} "
                                f"peak={r.peak_memory_mb:.0f}MB"
                            )
                            if r.error:
                                status = f"ERR: {r.error[:60]}"
                            print(f"    [{scenario.name}] run {run_i + 1}/{self.runs} {status}")

                        # Compute stats
                        stats: dict[str, Any] = {}
                        for metric in [
                            "ttft_ms",
                            "e2e_ms",
                            "tpot_ms",
                            "decode_tps",
                            "overall_tps",
                            "peak_memory_mb",
                        ]:
                            values = [r[metric] for r in scenario_runs if not r.get("error")]
                            stats[metric] = compute_stats(values)

                        scenario_results[scenario.name] = {
                            "runs": scenario_runs,
                            "stats": stats,
                        }

                # Compute aggregate stats for F and G
                for key in scenario_results:
                    if "stats" not in scenario_results[key]:
                        runs = scenario_results[key]["runs"]
                        stats = {}
                        for metric in [
                            "e2e_ms",
                            "ttft_ms",
                            "decode_tps",
                            "overall_tps",
                            "peak_memory_mb",
                            "output_tokens",
                        ]:
                            values = [r.get(metric, 0) for r in runs if not r.get("error")]
                            stats[metric] = compute_stats(values)
                        scenario_results[key]["stats"] = stats

                exp_results[exp.name] = {
                    "description": exp.description,
                    "env": exp.env_overrides,
                    "scenarios": scenario_results,
                }

            finally:
                self.server.stop()
                print(f"    Server stopped for {exp.name}")

        self.results["experiments"][exp_key] = exp_results
        self._save()

    async def run_all(self, experiment_keys: list[str] | None = None) -> None:
        """Run all or selected experiments."""
        all_experiments = build_experiments(self.quick)

        keys = experiment_keys or sorted(all_experiments.keys())
        print(f"\nProfiling benchmark: experiments {', '.join(keys)}")
        print(f"Runs per scenario: {self.runs}")
        print(f"Results: {self.json_path}\n")

        for key in keys:
            if key not in all_experiments:
                print(f"Unknown experiment: {key}")
                continue
            await self.run_experiment_group(key, all_experiments[key])

        self._generate_summary()
        print(f"\nResults saved to: {self.json_path}")
        print(f"Summary saved to: {self.summary_path}")

    def _generate_summary(self) -> None:
        """Generate human-readable summary tables."""
        lines: list[str] = []
        lines.append("=" * 90)
        lines.append("  PROFILING BENCHMARK SUMMARY")
        lines.append(f"  Model: {self.results['metadata']['model_id']}")
        lines.append(f"  Date: {self.results['metadata']['timestamp']}")
        lines.append(f"  Runs per scenario: {self.runs}")
        lines.append("=" * 90)

        for exp_key, exp_data in sorted(self.results.get("experiments", {}).items()):
            lines.append(f"\n{'─' * 90}")
            lines.append(f"  Experiment {exp_key}")
            lines.append(f"{'─' * 90}")

            for exp_name, exp_info in sorted(exp_data.items()):
                desc = exp_info.get("description", "")
                lines.append(f"\n  {exp_name}: {desc}")
                lines.append(f"  {'Scenario':<40} {'TTFT':>8} {'E2E':>8} {'TPS':>7} {'Peak MB':>8}")
                lines.append(f"  {'─' * 40} {'─' * 8} {'─' * 8} {'─' * 7} {'─' * 8}")

                scenarios = exp_info.get("scenarios", {})
                for sc_name, sc_data in sorted(scenarios.items()):
                    stats = sc_data.get("stats", {})
                    ttft = stats.get("ttft_ms", {}).get("median", 0)
                    e2e = stats.get("e2e_ms", {}).get("median", 0)
                    tps = stats.get("decode_tps", {}).get("median", 0)
                    peak = stats.get("peak_memory_mb", {}).get("median", 0)
                    lines.append(
                        f"  {sc_name:<40} {ttft:>8.0f} {e2e:>8.0f} {tps:>7.1f} {peak:>8.0f}"
                    )

        # Find optimal per experiment
        lines.append(f"\n{'=' * 90}")
        lines.append("  OPTIMAL PARAMETER RECOMMENDATIONS")
        lines.append(f"{'=' * 90}")

        # Experiment A: best prefill step size
        exp_a = self.results.get("experiments", {}).get("A", {})
        if exp_a:
            best_step = None
            best_ttft = float("inf")
            for exp_name, exp_info in exp_a.items():
                for sc_name, sc_data in exp_info.get("scenarios", {}).items():
                    ttft_med = (
                        sc_data.get("stats", {}).get("ttft_ms", {}).get("median", float("inf"))
                    )
                    if ttft_med < best_ttft:
                        best_ttft = ttft_med
                        best_step = exp_name
            if best_step:
                lines.append(f"  Prefill step: {best_step} (TTFT={best_ttft:.0f}ms)")

        # Experiment C: best batch size
        exp_c = self.results.get("experiments", {}).get("C", {})
        if exp_c:
            best_batch = None
            best_tps = 0.0
            for exp_name, exp_info in exp_c.items():
                for sc_name, sc_data in exp_info.get("scenarios", {}).items():
                    tps_med = sc_data.get("stats", {}).get("overall_tps", {}).get("median", 0)
                    if tps_med > best_tps:
                        best_tps = tps_med
                        best_batch = exp_name
            if best_batch:
                lines.append(f"  Batch size: {best_batch} (system TPS={best_tps:.1f})")

        summary = "\n".join(lines)
        with open(self.summary_path, "w") as f:
            f.write(summary)
        print(summary)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Profiling benchmark for optimal parameter discovery"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: fewer scenarios, 2 runs each",
    )
    parser.add_argument(
        "--experiment",
        nargs="+",
        choices=list("ABCDEFGHI"),
        help="Run specific experiments (e.g., --experiment A C E)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=None,
        help="Runs per scenario (default: 5, quick: 2)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=PORT,
        help=f"Server port (default: {PORT})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results",
    )
    args = parser.parse_args()

    runs = args.runs
    if runs is None:
        runs = RUNS_QUICK if args.quick else RUNS_DEFAULT

    suite = ProfilingBenchmark(
        port=args.port,
        runs=runs,
        quick=args.quick,
        output_dir=args.output_dir,
    )

    asyncio.run(suite.run_all(args.experiment))


if __name__ == "__main__":
    main()
