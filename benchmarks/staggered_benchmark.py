#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Yakov Shkolnikov and contributors
"""Staggered arrivals benchmark for batched vs sequential serving.

Tests User A arriving at t=0 and User B arriving at t=2s, both with 4K context.
Compares:
- Sequential: A completes, then B starts (no batching benefit)
- Batched: B joins while A is running (batching benefit)

Usage:
    # Against managed agent-memory server
    python benchmarks/staggered_benchmark.py

    # Against running server
    python benchmarks/staggered_benchmark.py --external --base-url http://localhost:8399

    # Custom stagger delay
    python benchmarks/staggered_benchmark.py --stagger-delay 3.0

    # More runs for statistical confidence
    python benchmarks/staggered_benchmark.py --runs 5
"""

import argparse
import asyncio
import json
import platform
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import statistics

import httpx

# Reuse infrastructure
import sys
sys.path.insert(0, str(Path(__file__).parent))

from openai_benchmark import (
    OpenAIStreamingClient,
    OpenAIPromptFactory,
    ScenarioResult,
    ServerManager,
    OPENAI_BENCH_ENV,
    PORT,
)

RESULTS_DIR = Path(__file__).parent / "results"
DEFAULT_CONTEXT = 4096
DEFAULT_OUTPUT = 64
DEFAULT_RUNS = 3
DEFAULT_STAGGER_DELAY = 2.0  # seconds


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


async def _delete_agent(base_url: str, agent_id: str) -> None:
    """Delete agent from server."""
    async with httpx.AsyncClient(timeout=10.0) as c:
        try:
            await c.delete(f"{base_url}/v1/agents/{agent_id}")
        except Exception:
            pass


async def _wait_for_server(base_url: str, timeout: float = 300) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            async with httpx.AsyncClient(timeout=5.0) as c:
                r = await c.get(f"{base_url}/v1/models")
                if r.status_code == 200:
                    return True
        except Exception:
            pass
        await asyncio.sleep(2)
    return False


@dataclass
class StaggeredResult:
    """Result from a staggered benchmark run."""
    mode: str  # "sequential" or "batched"
    run_id: int
    user_a_ttft_ms: float
    user_a_e2e_ms: float
    user_b_ttft_ms: float
    user_b_e2e_ms: float
    user_b_start_delay_ms: float  # Actual delay between A and B starts
    user_b_wait_ms: float  # B's wait from scenario start (delay + ttft)
    total_wall_time_ms: float  # Total time from A start to B completion
    user_a_tps: float
    user_b_tps: float
    system_tps: float  # Total tokens / total wall time
    error: str = ""


async def run_sequential(
    base_url: str,
    context_tokens: int,
    output_tokens: int,
    run_id: int,
) -> StaggeredResult:
    """Sequential serving: User A completes, then User B starts.

    This is the baseline - no batching benefit.
    """
    client = OpenAIStreamingClient(base_url)
    factory = OpenAIPromptFactory()

    # User A
    body_a = factory.build_request(context_tokens, output_tokens)
    body_a["stream"] = True
    sid_a = f"stagger_seq_a_{run_id}"

    t_start_wall = time.perf_counter()
    result_a = await client.send_and_measure(body_a, session_id=sid_a)
    t_a_done = time.perf_counter()
    await _delete_agent(base_url, f"oai_{sid_a}")

    # User B (starts after A completes)
    body_b = factory.build_request(context_tokens, output_tokens)
    body_b["stream"] = True
    sid_b = f"stagger_seq_b_{run_id}"

    t_start_b = time.perf_counter()
    result_b = await client.send_and_measure(body_b, session_id=sid_b)
    t_end_wall = time.perf_counter()
    await _delete_agent(base_url, f"oai_{sid_b}")

    # Calculate metrics
    total_wall_ms = (t_end_wall - t_start_wall) * 1000
    user_b_delay_ms = (t_start_b - t_start_wall) * 1000
    total_tokens = result_a.output_tokens + result_b.output_tokens
    system_tps = (total_tokens / (total_wall_ms / 1000)) if total_wall_ms > 0 else 0

    return StaggeredResult(
        mode="sequential",
        run_id=run_id,
        user_a_ttft_ms=result_a.ttft_ms,
        user_a_e2e_ms=result_a.e2e_ms,
        user_b_ttft_ms=result_b.ttft_ms,
        user_b_e2e_ms=result_b.e2e_ms,
        user_b_start_delay_ms=user_b_delay_ms,
        user_b_wait_ms=user_b_delay_ms + result_b.ttft_ms,
        total_wall_time_ms=total_wall_ms,
        user_a_tps=result_a.decode_tps,
        user_b_tps=result_b.decode_tps,
        system_tps=system_tps,
        error=result_a.error or result_b.error,
    )


async def run_batched(
    base_url: str,
    context_tokens: int,
    output_tokens: int,
    stagger_delay: float,
    run_id: int,
) -> StaggeredResult:
    """Batched serving: User B joins while User A is running.

    This tests the batching benefit - B should see faster service
    compared to sequential because it shares compute with A.
    """
    client = OpenAIStreamingClient(base_url)
    factory = OpenAIPromptFactory()

    # Prepare both requests
    body_a = factory.build_request(context_tokens, output_tokens)
    body_a["stream"] = True
    sid_a = f"stagger_batch_a_{run_id}"

    body_b = factory.build_request(context_tokens, output_tokens)
    body_b["stream"] = True
    sid_b = f"stagger_batch_b_{run_id}"

    # Launch both with stagger
    t_start_wall = time.perf_counter()

    async def launch_a():
        return await client.send_and_measure(body_a, session_id=sid_a)

    async def launch_b():
        await asyncio.sleep(stagger_delay)
        t_b_start = time.perf_counter()
        result = await client.send_and_measure(body_b, session_id=sid_b)
        return result, t_b_start

    # Run concurrently
    result_a, (result_b, t_b_start) = await asyncio.gather(launch_a(), launch_b())
    t_end_wall = time.perf_counter()

    # Cleanup
    await _delete_agent(base_url, f"oai_{sid_a}")
    await _delete_agent(base_url, f"oai_{sid_b}")

    # Calculate metrics
    total_wall_ms = (t_end_wall - t_start_wall) * 1000
    user_b_delay_ms = (t_b_start - t_start_wall) * 1000
    total_tokens = result_a.output_tokens + result_b.output_tokens
    system_tps = (total_tokens / (total_wall_ms / 1000)) if total_wall_ms > 0 else 0

    return StaggeredResult(
        mode="batched",
        run_id=run_id,
        user_a_ttft_ms=result_a.ttft_ms,
        user_a_e2e_ms=result_a.e2e_ms,
        user_b_ttft_ms=result_b.ttft_ms,
        user_b_e2e_ms=result_b.e2e_ms,
        user_b_start_delay_ms=user_b_delay_ms,
        user_b_wait_ms=user_b_delay_ms + result_b.ttft_ms,
        total_wall_time_ms=total_wall_ms,
        user_a_tps=result_a.decode_tps,
        user_b_tps=result_b.decode_tps,
        system_tps=system_tps,
        error=result_a.error or result_b.error,
    )


async def run_benchmark(
    base_url: str,
    context_tokens: int,
    output_tokens: int,
    stagger_delay: float,
    runs: int,
) -> dict[str, Any]:
    """Run full staggered benchmark suite."""
    sequential_results = []
    batched_results = []

    print(f"\n{'='*80}")
    print(f"Staggered Arrivals Benchmark")
    print(f"{'='*80}")
    print(f"Context: {context_tokens} tokens")
    print(f"Output: {output_tokens} tokens")
    print(f"Stagger delay: {stagger_delay}s")
    print(f"Runs: {runs}")
    print(f"Base URL: {base_url}\n")

    # Run sequential tests
    print("Running sequential tests (User A completes, then User B starts)...")
    for i in range(runs):
        print(f"  Run {i+1}/{runs}...", end=" ", flush=True)
        result = await run_sequential(base_url, context_tokens, output_tokens, i)
        sequential_results.append(result)
        if result.error:
            print(f"ERROR: {result.error}")
        else:
            print(f"Wall: {result.total_wall_time_ms:.0f}ms, B E2E: {result.user_b_e2e_ms:.0f}ms")
        await asyncio.sleep(1)  # Brief pause between runs

    # Run batched tests
    print("\nRunning batched tests (User B joins while User A is running)...")
    for i in range(runs):
        print(f"  Run {i+1}/{runs}...", end=" ", flush=True)
        result = await run_batched(base_url, context_tokens, output_tokens, stagger_delay, i)
        batched_results.append(result)
        if result.error:
            print(f"ERROR: {result.error}")
        else:
            print(f"Wall: {result.total_wall_time_ms:.0f}ms, B E2E: {result.user_b_e2e_ms:.0f}ms")
        await asyncio.sleep(1)

    # Compute statistics
    def compute_stats(results: list[StaggeredResult]) -> dict[str, float]:
        if not results or any(r.error for r in results):
            return {}

        return {
            "user_a_ttft_ms_median": statistics.median(r.user_a_ttft_ms for r in results),
            "user_a_e2e_ms_median": statistics.median(r.user_a_e2e_ms for r in results),
            "user_b_ttft_ms_median": statistics.median(r.user_b_ttft_ms for r in results),
            "user_b_e2e_ms_median": statistics.median(r.user_b_e2e_ms for r in results),
            "total_wall_time_ms_median": statistics.median(r.total_wall_time_ms for r in results),
            "system_tps_median": statistics.median(r.system_tps for r in results),
            "user_b_speedup": (
                statistics.median(r.user_b_e2e_ms for r in sequential_results) /
                statistics.median(r.user_b_e2e_ms for r in batched_results)
                if batched_results else 0
            ),
        }

    seq_stats = compute_stats(sequential_results)
    batch_stats = compute_stats(batched_results)

    # Print summary
    print(f"\n{'='*80}")
    print("RESULTS SUMMARY (medians)")
    print(f"{'='*80}")

    if seq_stats:
        print(f"\nSequential:")
        print(f"  User A TTFT: {seq_stats['user_a_ttft_ms_median']:.0f}ms")
        print(f"  User A E2E:  {seq_stats['user_a_e2e_ms_median']:.0f}ms")
        print(f"  User B TTFT: {seq_stats['user_b_ttft_ms_median']:.0f}ms")
        print(f"  User B E2E:  {seq_stats['user_b_e2e_ms_median']:.0f}ms")
        print(f"  Wall time:   {seq_stats['total_wall_time_ms_median']:.0f}ms")
        print(f"  System TPS:  {seq_stats['system_tps_median']:.1f} tok/s")

    if batch_stats:
        print(f"\nBatched:")
        print(f"  User A TTFT: {batch_stats['user_a_ttft_ms_median']:.0f}ms")
        print(f"  User A E2E:  {batch_stats['user_a_e2e_ms_median']:.0f}ms")
        print(f"  User B TTFT: {batch_stats['user_b_ttft_ms_median']:.0f}ms")
        print(f"  User B E2E:  {batch_stats['user_b_e2e_ms_median']:.0f}ms")
        print(f"  Wall time:   {batch_stats['total_wall_time_ms_median']:.0f}ms")
        print(f"  System TPS:  {batch_stats['system_tps_median']:.1f} tok/s")

    if seq_stats and batch_stats:
        speedup = batch_stats['user_b_speedup']
        print(f"\nUser B Speedup (batched vs sequential): {speedup:.2f}x")
        wall_speedup = seq_stats['total_wall_time_ms_median'] / batch_stats['total_wall_time_ms_median']
        print(f"Total Wall Time Speedup: {wall_speedup:.2f}x")
        tps_gain = (batch_stats['system_tps_median'] - seq_stats['system_tps_median']) / seq_stats['system_tps_median'] * 100
        print(f"System Throughput Gain: {tps_gain:+.1f}%")

    # Return full results
    # Query server for model identity
    model_id = "unknown"
    try:
        r = httpx.get(f"{base_url}/v1/models", timeout=5.0)
        if r.status_code == 200:
            models = r.json().get("data", [])
            if models:
                model_id = models[0].get("id", "unknown")
    except Exception:
        pass

    return {
        "benchmark": "staggered_arrivals",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_id": model_id,
        "git_sha": _git_sha(),
        "system": {
            "platform": platform.system(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "config": {
            "context_tokens": context_tokens,
            "output_tokens": output_tokens,
            "stagger_delay_s": stagger_delay,
            "runs": runs,
        },
        "sequential": {
            "stats": seq_stats,
            "raw_results": [
                {
                    "run_id": r.run_id,
                    "user_a_ttft_ms": r.user_a_ttft_ms,
                    "user_a_e2e_ms": r.user_a_e2e_ms,
                    "user_b_ttft_ms": r.user_b_ttft_ms,
                    "user_b_e2e_ms": r.user_b_e2e_ms,
                    "total_wall_time_ms": r.total_wall_time_ms,
                    "system_tps": r.system_tps,
                    "error": r.error,
                }
                for r in sequential_results
            ],
        },
        "batched": {
            "stats": batch_stats,
            "raw_results": [
                {
                    "run_id": r.run_id,
                    "user_a_ttft_ms": r.user_a_ttft_ms,
                    "user_a_e2e_ms": r.user_a_e2e_ms,
                    "user_b_ttft_ms": r.user_b_ttft_ms,
                    "user_b_e2e_ms": r.user_b_e2e_ms,
                    "user_b_start_delay_ms": r.user_b_start_delay_ms,
                    "total_wall_time_ms": r.total_wall_time_ms,
                    "system_tps": r.system_tps,
                    "error": r.error,
                }
                for r in batched_results
            ],
        },
    }


async def main():
    parser = argparse.ArgumentParser(
        description="Staggered arrivals benchmark for batched vs sequential serving"
    )
    parser.add_argument(
        "--external",
        action="store_true",
        help="Use external server (skip server management)",
    )
    parser.add_argument(
        "--base-url",
        default=f"http://127.0.0.1:{PORT}",
        help="Server base URL",
    )
    parser.add_argument(
        "--context-tokens",
        type=int,
        default=DEFAULT_CONTEXT,
        help=f"Context length (default: {DEFAULT_CONTEXT})",
    )
    parser.add_argument(
        "--output-tokens",
        type=int,
        default=DEFAULT_OUTPUT,
        help=f"Output tokens (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--stagger-delay",
        type=float,
        default=DEFAULT_STAGGER_DELAY,
        help=f"Delay in seconds before User B starts (default: {DEFAULT_STAGGER_DELAY})",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=DEFAULT_RUNS,
        help=f"Number of runs per test (default: {DEFAULT_RUNS})",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        help="Custom output filename (default: auto-generated)",
    )

    args = parser.parse_args()

    server_mgr = None
    try:
        # Start server if not external
        if not args.external:
            print("Starting agent-memory server...")
            server_mgr = ServerManager(PORT, env=OPENAI_BENCH_ENV)
            server_mgr.start()
            if not await _wait_for_server(args.base_url):
                print("ERROR: Server failed to start")
                return 1
            print("Server ready")

        # Run benchmark
        results = await run_benchmark(
            args.base_url,
            args.context_tokens,
            args.output_tokens,
            args.stagger_delay,
            args.runs,
        )

        # Save results
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        if args.output_file:
            output_path = RESULTS_DIR / args.output_file
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = RESULTS_DIR / f"staggered_{args.context_tokens}_{timestamp}.json"

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to: {output_path}")
        return 0

    except KeyboardInterrupt:
        print("\nBenchmark interrupted")
        return 1
    finally:
        if server_mgr:
            server_mgr.stop()


if __name__ == "__main__":
    exit(asyncio.run(main()))
