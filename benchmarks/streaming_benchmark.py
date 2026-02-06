#!/usr/bin/env python3
"""Streaming vs Non-streaming benchmark with cache state comparison.

Tests all combinations of:
- Modes: streaming, non-streaming
- Cache states: cold, warm, hot (multi-turn extend)
- Context lengths: 1024, 2048, 4096, 8192, 16384, 32768
- Batch sizes: 1, 2

Usage:
    # Full sweep (starts server automatically)
    python benchmarks/streaming_benchmark.py

    # Specific context lengths
    python benchmarks/streaming_benchmark.py --contexts 1024 4096 16384

    # Single batch size
    python benchmarks/streaming_benchmark.py --batch-sizes 1

    # Against running server (skip server management)
    python benchmarks/streaming_benchmark.py --external --base-url http://localhost:8399
"""

import argparse
import asyncio
import json
import platform
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

# Reuse infrastructure from openai_benchmark
import sys
sys.path.insert(0, str(Path(__file__).parent))

from openai_benchmark import (
    OpenAIStreamingClient,
    OpenAIRequestClient,
    OpenAIPromptFactory,
    ScenarioResult,
    ServerManager,
    compute_stats,
    OPENAI_BENCH_ENV,
    PORT,
    PADDING_TEXT,
)

RESULTS_DIR = Path(__file__).parent / "results"

DEFAULT_CONTEXTS = [1024, 2048, 4096, 8192, 16384, 32768]
DEFAULT_OUTPUTS = [64]
DEFAULT_BATCH_SIZES = [1, 2]
DEFAULT_RUNS = 3
OUTPUT_TOKENS = 64


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


async def _delete_agent(base_url: str, agent_id: str, evict_only: bool = False) -> None:
    """Delete or evict agent from server.

    Args:
        base_url: Server base URL
        agent_id: Agent identifier to delete
        evict_only: If True, evict from hot tier but keep disk file (for warm cache test)
    """
    async with httpx.AsyncClient(timeout=10.0) as c:
        try:
            url = f"{base_url}/v1/agents/{agent_id}"
            if evict_only:
                url += "?evict_only=true"
            await c.delete(url)
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


# ---------------------------------------------------------------------------
# Core benchmark functions
# ---------------------------------------------------------------------------

async def run_streaming_cold(
    base_url: str, context_tokens: int, output_tokens: int, run_id: str,
) -> dict[str, Any]:
    """Cold start with streaming — measures TTFT + decode TPS."""
    client = OpenAIStreamingClient(base_url)
    factory = OpenAIPromptFactory()
    body = factory.build_request(context_tokens, output_tokens)
    sid = f"stream_cold_{context_tokens}_{run_id}"
    body["stream"] = True
    r = await client.send_and_measure(body, session_id=sid)
    await _delete_agent(base_url, f"oai_{sid}")
    return {
        "mode": "streaming", "cache_state": "cold",
        "context_tokens": context_tokens,
        **_result_dict(r),
    }


async def run_nonstreaming_cold(
    base_url: str, context_tokens: int, output_tokens: int, run_id: str,
) -> dict[str, Any]:
    """Cold start without streaming — measures E2E only."""
    client = OpenAIRequestClient(base_url)
    factory = OpenAIPromptFactory()
    body = factory.build_request(context_tokens, output_tokens)
    sid = f"nonstream_cold_{context_tokens}_{run_id}"
    r = await client.send_and_measure(body, session_id=sid)
    await _delete_agent(base_url, f"oai_{sid}")
    return {
        "mode": "non-streaming", "cache_state": "cold",
        "context_tokens": context_tokens,
        **_result_dict(r),
    }


async def run_streaming_warm(
    base_url: str, context_tokens: int, output_tokens: int, run_id: str,
) -> dict[str, Any]:
    """Warm cache with streaming — prime then measure with stream.

    Tests true disk reload: prime creates cache, evict from hot tier,
    measure reloads from disk (safetensors).
    """
    prime_client = OpenAIRequestClient(base_url)
    measure_client = OpenAIStreamingClient(base_url)
    factory = OpenAIPromptFactory()
    body = factory.build_request(context_tokens, output_tokens)
    sid = f"stream_warm_{context_tokens}_{run_id}"

    # Prime (cold hit, populates cache)
    await prime_client.send_and_measure(body, session_id=sid)

    # Evict from hot tier but keep disk file (for warm cache reload)
    await _delete_agent(base_url, f"oai_{sid}", evict_only=True)
    await asyncio.sleep(1.0)  # Allow disk write to complete

    # Measure (warm hit from disk reload, streaming)
    body["stream"] = True
    r = await measure_client.send_and_measure(body, session_id=sid)

    # Full cleanup after measurement
    await _delete_agent(base_url, f"oai_{sid}")
    return {
        "mode": "streaming", "cache_state": "warm",
        "context_tokens": context_tokens,
        **_result_dict(r),
    }


async def run_nonstreaming_warm(
    base_url: str, context_tokens: int, output_tokens: int, run_id: str,
) -> dict[str, Any]:
    """Warm cache without streaming — prime then measure.

    Tests true disk reload: prime creates cache, evict from hot tier,
    measure reloads from disk (safetensors).
    """
    client = OpenAIRequestClient(base_url)
    factory = OpenAIPromptFactory()
    body = factory.build_request(context_tokens, output_tokens)
    sid = f"nonstream_warm_{context_tokens}_{run_id}"

    # Prime
    await client.send_and_measure(body, session_id=sid)

    # Evict from hot tier but keep disk file (for warm cache reload)
    await _delete_agent(base_url, f"oai_{sid}", evict_only=True)
    await asyncio.sleep(1.0)  # Allow disk write to complete

    # Measure
    r = await client.send_and_measure(body, session_id=sid)

    # Full cleanup after measurement
    await _delete_agent(base_url, f"oai_{sid}")
    return {
        "mode": "non-streaming", "cache_state": "warm",
        "context_tokens": context_tokens,
        **_result_dict(r),
    }


async def run_streaming_hot(
    base_url: str, context_tokens: int, output_tokens: int, run_id: str,
) -> dict[str, Any]:
    """Hot (multi-turn extend) with streaming — 3 turns, measure 3rd."""
    prime_client = OpenAIRequestClient(base_url)
    measure_client = OpenAIStreamingClient(base_url)
    factory = OpenAIPromptFactory()
    body = factory.build_request(context_tokens, output_tokens)
    sid = f"stream_hot_{context_tokens}_{run_id}"

    # Turn 1 (cold)
    r1 = await prime_client.send_and_measure(body, session_id=sid)
    await asyncio.sleep(0.3)

    # Turn 2 (extend)
    followup1 = factory.build_followup(
        body["messages"], r1.raw_text if hasattr(r1, "raw_text") else "I see.",
        output_tokens,
    )
    await prime_client.send_and_measure(followup1, session_id=sid)
    await asyncio.sleep(0.3)

    # Turn 3 (hot, streaming measurement)
    followup2 = factory.build_followup(
        followup1["messages"], "Understood.",
        output_tokens,
    )
    followup2["stream"] = True
    r = await measure_client.send_and_measure(followup2, session_id=sid)
    await _delete_agent(base_url, f"oai_{sid}")
    return {
        "mode": "streaming", "cache_state": "hot",
        "context_tokens": context_tokens,
        **_result_dict(r),
    }


async def run_nonstreaming_hot(
    base_url: str, context_tokens: int, output_tokens: int, run_id: str,
) -> dict[str, Any]:
    """Hot (multi-turn extend) without streaming — 3 turns, measure 3rd."""
    client = OpenAIRequestClient(base_url)
    factory = OpenAIPromptFactory()
    body = factory.build_request(context_tokens, output_tokens)
    sid = f"nonstream_hot_{context_tokens}_{run_id}"

    # Turn 1
    r1 = await client.send_and_measure(body, session_id=sid)
    await asyncio.sleep(0.3)

    # Turn 2
    followup1 = factory.build_followup(
        body["messages"], r1.raw_text if hasattr(r1, "raw_text") else "I see.",
        output_tokens,
    )
    await client.send_and_measure(followup1, session_id=sid)
    await asyncio.sleep(0.3)

    # Turn 3
    followup2 = factory.build_followup(
        followup1["messages"], "Understood.",
        output_tokens,
    )
    r = await client.send_and_measure(followup2, session_id=sid)
    await _delete_agent(base_url, f"oai_{sid}")
    return {
        "mode": "non-streaming", "cache_state": "hot",
        "context_tokens": context_tokens,
        **_result_dict(r),
    }


async def run_concurrent_pair(
    base_url: str, context_tokens: int, output_tokens: int, run_id: str,
    streaming: bool = False,
) -> dict[str, Any]:
    """Two concurrent requests (batch=2) — streaming or non-streaming."""
    factory = OpenAIPromptFactory()
    mode_label = "streaming" if streaming else "non-streaming"

    if streaming:
        clients = [OpenAIStreamingClient(base_url) for _ in range(2)]
    else:
        clients = [OpenAIRequestClient(base_url) for _ in range(2)]

    bodies = []
    sids = []
    for i in range(2):
        body = factory.build_request(context_tokens, output_tokens)
        if streaming:
            body["stream"] = True
        bodies.append(body)
        sids.append(f"concurrent_{mode_label}_{context_tokens}_{run_id}_{i}")

    t_start = time.perf_counter()
    results = await asyncio.gather(
        clients[0].send_and_measure(bodies[0], session_id=sids[0]),
        clients[1].send_and_measure(bodies[1], session_id=sids[1]),
    )
    wall_ms = (time.perf_counter() - t_start) * 1000

    # Cleanup
    for sid in sids:
        await _delete_agent(base_url, f"oai_{sid}")

    total_output = sum(r.output_tokens for r in results)
    avg_e2e = sum(r.e2e_ms for r in results) / 2
    avg_ttft = sum(r.ttft_ms for r in results) / 2 if streaming else 0
    system_tps = total_output / (wall_ms / 1000) if wall_ms > 0 else 0

    return {
        "mode": mode_label, "cache_state": "concurrent_2x",
        "context_tokens": context_tokens,
        "wall_ms": round(wall_ms, 1),
        "avg_e2e_ms": round(avg_e2e, 1),
        "avg_ttft_ms": round(avg_ttft, 1),
        "total_output_tokens": total_output,
        "system_tps": round(system_tps, 1),
        "per_request_tps": round(system_tps / 2, 1),
    }


def _result_dict(r: ScenarioResult) -> dict[str, Any]:
    """Extract key metrics from ScenarioResult."""
    return {
        "ttft_ms": round(r.ttft_ms, 1),
        "e2e_ms": round(r.e2e_ms, 1),
        "decode_tps": round(r.decode_tps, 1),
        "tpot_ms": round(r.tpot_ms, 1),
        "output_tokens": r.output_tokens,
        "input_tokens": r.input_tokens,
        "peak_memory_mb": round(r.peak_memory_mb, 1),
    }


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

async def run_sweep(
    base_url: str,
    contexts: list[int],
    batch_sizes: list[int],
    runs: int,
    output_tokens: int,
    external: bool = False,
) -> dict[str, Any]:
    """Run the full streaming benchmark sweep."""

    all_results: list[dict[str, Any]] = []
    timestamp = datetime.now(timezone.utc).isoformat()

    test_functions = {
        ("streaming", "cold"): run_streaming_cold,
        ("non-streaming", "cold"): run_nonstreaming_cold,
        ("streaming", "warm"): run_streaming_warm,
        ("non-streaming", "warm"): run_nonstreaming_warm,
        ("streaming", "hot"): run_streaming_hot,
        ("non-streaming", "hot"): run_nonstreaming_hot,
    }

    for batch_size in batch_sizes:
        print(f"\n{'='*60}")
        print(f"  BATCH SIZE = {batch_size}")
        print(f"{'='*60}")

        for ctx in sorted(contexts):
            print(f"\n--- Context: {ctx} tokens ---")

            for (mode, cache_state), func in test_functions.items():
                label = f"{mode}/{cache_state}/{ctx}tok"
                results_for_scenario = []

                for run_i in range(runs):
                    run_id = f"b{batch_size}_r{run_i}_{int(time.time())}"
                    try:
                        print(f"  {label} run {run_i+1}/{runs}...", end=" ", flush=True)
                        r = await func(base_url, ctx, output_tokens, run_id)
                        r["batch_size"] = batch_size
                        r["run"] = run_i
                        results_for_scenario.append(r)
                        e2e = r.get("e2e_ms", 0)
                        ttft = r.get("ttft_ms", 0)
                        tps = r.get("decode_tps", 0)
                        print(f"E2E={e2e:.0f}ms TTFT={ttft:.0f}ms TPS={tps:.1f}")
                    except Exception as e:
                        print(f"ERROR: {e}")
                        results_for_scenario.append({
                            "mode": mode, "cache_state": cache_state,
                            "context_tokens": ctx, "batch_size": batch_size,
                            "run": run_i, "error": str(e),
                        })

                all_results.extend(results_for_scenario)

            # Concurrent tests for batch >= 2
            if batch_size >= 2 and not external:
                for streaming in [True, False]:
                    mode_label = "streaming" if streaming else "non-streaming"
                    label = f"concurrent_2x/{mode_label}/{ctx}tok"
                    for run_i in range(runs):
                        run_id = f"b{batch_size}_r{run_i}_{int(time.time())}"
                        try:
                            print(f"  {label} run {run_i+1}/{runs}...", end=" ", flush=True)
                            r = await run_concurrent_pair(
                                base_url, ctx, output_tokens, run_id, streaming=streaming,
                            )
                            r["batch_size"] = batch_size
                            r["run"] = run_i
                            all_results.append(r)
                            wall = r.get("wall_ms", 0)
                            sys_tps = r.get("system_tps", 0)
                            print(f"wall={wall:.0f}ms sysTPS={sys_tps:.1f}")
                        except Exception as e:
                            print(f"ERROR: {e}")
                            all_results.append({
                                "mode": mode_label,
                                "cache_state": "concurrent_2x",
                                "context_tokens": ctx,
                                "batch_size": batch_size,
                                "run": run_i,
                                "error": str(e),
                            })

    # Query server for model identity
    model_id = "unknown"
    try:
        async with httpx.AsyncClient(timeout=5.0) as c:
            r = await c.get(f"{base_url}/v1/models")
            if r.status_code == 200:
                models = r.json().get("data", [])
                if models:
                    model_id = models[0].get("id", "unknown")
    except Exception:
        pass

    # Build output document
    output = {
        "benchmark": "streaming_comparison",
        "timestamp": timestamp,
        "model_id": model_id,
        "git_sha": _git_sha(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "settings": {
            "contexts": contexts,
            "batch_sizes": batch_sizes,
            "runs_per_scenario": runs,
            "output_tokens": output_tokens,
        },
        "results": all_results,
    }

    return output


def save_results(data: dict[str, Any]) -> Path:
    """Save results to JSON file."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = RESULTS_DIR / f"streaming_{ts}.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    return path


def print_summary(data: dict[str, Any]) -> None:
    """Print a human-readable summary table."""
    results = [r for r in data["results"] if "error" not in r]
    if not results:
        print("\nNo successful results to summarize.")
        return

    print(f"\n{'='*80}")
    print("  STREAMING BENCHMARK SUMMARY")
    print(f"{'='*80}")

    # Group by (batch_size, context_tokens, cache_state)
    groups: dict[tuple, list[dict]] = {}
    for r in results:
        if r.get("cache_state") == "concurrent_2x":
            continue
        key = (r["batch_size"], r["context_tokens"], r["cache_state"])
        groups.setdefault(key, [])
        groups[key].append(r)

    # Print comparison tables
    for (bs, ctx, cs), entries in sorted(groups.items()):
        streaming = [e for e in entries if e["mode"] == "streaming"]
        nonstreaming = [e for e in entries if e["mode"] == "non-streaming"]

        if streaming and nonstreaming:
            s_e2e = sum(e["e2e_ms"] for e in streaming) / len(streaming)
            s_ttft = sum(e["ttft_ms"] for e in streaming) / len(streaming)
            s_tps = sum(e["decode_tps"] for e in streaming) / len(streaming)
            ns_e2e = sum(e["e2e_ms"] for e in nonstreaming) / len(nonstreaming)

            overhead = ((s_e2e - ns_e2e) / ns_e2e * 100) if ns_e2e > 0 else 0
            print(f"\n  batch={bs} ctx={ctx:>5} {cs:>5}: "
                  f"stream E2E={s_e2e:>7.0f}ms TTFT={s_ttft:>6.0f}ms TPS={s_tps:>5.1f} | "
                  f"non-stream E2E={ns_e2e:>7.0f}ms | "
                  f"overhead={overhead:>+.1f}%")

    # Concurrent summary
    concurrent = [r for r in results if r.get("cache_state") == "concurrent_2x"]
    if concurrent:
        print(f"\n  --- Concurrent (batch=2) ---")
        for r in concurrent:
            print(f"  ctx={r['context_tokens']:>5} {r['mode']:>14}: "
                  f"wall={r['wall_ms']:>7.0f}ms sysTPS={r['system_tps']:>5.1f}")


async def main():
    parser = argparse.ArgumentParser(description="Streaming vs Non-streaming benchmark")
    parser.add_argument("--contexts", nargs="+", type=int, default=DEFAULT_CONTEXTS)
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=DEFAULT_BATCH_SIZES)
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    parser.add_argument("--output-tokens", type=int, default=OUTPUT_TOKENS)
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--external", action="store_true", help="Use running server")
    parser.add_argument("--base-url", type=str, default=None)
    args = parser.parse_args()

    base_url = args.base_url or f"http://127.0.0.1:{args.port}"
    server = None

    if not args.external:
        # Start server with appropriate config
        env = dict(OPENAI_BENCH_ENV)
        env["SEMANTIC_MLX_MAX_BATCH_SIZE"] = str(max(args.batch_sizes))
        env["SEMANTIC_MLX_SCHEDULER_ENABLED"] = "true"
        env["SEMANTIC_MLX_KV_BITS"] = "4"

        server = ServerManager(port=args.port)
        print(f"Starting server on port {args.port}...")
        server.start(env_overrides=env)

        print("Waiting for server to be ready...")
        if not await _wait_for_server(base_url):
            print("ERROR: Server did not start in time")
            server.stop()
            return

        # Small warmup request
        print("Warmup request...")
        warmup_client = OpenAIRequestClient(base_url)
        factory = OpenAIPromptFactory()
        warmup_body = factory.build_request(100, 1)
        try:
            await warmup_client.send_and_measure(warmup_body, session_id="warmup")
            await _delete_agent(base_url, "oai_warmup")
        except Exception as e:
            print(f"Warmup failed: {e}")

    try:
        data = await run_sweep(
            base_url=base_url,
            contexts=args.contexts,
            batch_sizes=args.batch_sizes,
            runs=args.runs,
            output_tokens=args.output_tokens,
            external=args.external,
        )

        path = save_results(data)
        print(f"\nResults saved to: {path}")
        print_summary(data)

    finally:
        if server:
            server.stop()


if __name__ == "__main__":
    asyncio.run(main())
