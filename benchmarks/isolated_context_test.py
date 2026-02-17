#!/usr/bin/env python3
"""
Isolated single-context vllm-mlx test.

Runs cold→prefix-cached for a SINGLE context length with a fresh server,
proving prefix cache works when there's no FP16 accumulation from other
context lengths.

Usage:
    # vllm-mlx isolated 8K test
    VLLM_MLX_BIN=$HOME/.venvs/vllm-mlx/bin/vllm-mlx \
        python benchmarks/isolated_context_test.py --context 8192 --passes 3

    # vllm-mlx isolated 16K test
    VLLM_MLX_BIN=$HOME/.venvs/vllm-mlx/bin/vllm-mlx \
        python benchmarks/isolated_context_test.py --context 16384 --passes 3
"""

import argparse
import asyncio
import json
import os
import signal
import subprocess
import sys
import time

import aiohttp

# Reuse corpus loading from the comparison benchmark
sys.path.insert(0, os.path.dirname(__file__))
from vllm_comparison_benchmark import (
    load_corpus,
    build_messages,
    measure_streaming_ttft,
    kill_port,
    wait_for_health,
    COOLDOWN_SECONDS,
)

MODEL_ID = "mlx-community/Llama-3.1-8B-Instruct-4bit"
PORT = 8399


async def run_isolated_test(context: int, passes: int, cache_pct: float) -> dict:
    """Run cold→cached for a single context on a fresh vllm-mlx server."""
    corpus = load_corpus()

    kill_port(PORT)
    vllm_bin = os.environ.get("VLLM_MLX_BIN", "vllm-mlx")
    cmd = [
        vllm_bin, "serve", MODEL_ID,
        "--port", str(PORT),
        "--enable-prefix-cache",
        "--cache-memory-percent", str(cache_pct),
        "--continuous-batching", "--use-paged-cache",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print(f"  Starting vllm-mlx on port {PORT} (cache={cache_pct*100:.0f}%)...")

    if not wait_for_health(f"http://127.0.0.1:{PORT}/health"):
        print("  FAILED: server did not start")
        proc.kill()
        return {}
    print(f"  vllm-mlx ready on port {PORT}")

    base_url = f"http://127.0.0.1:{PORT}"
    results = {"context": context, "passes": passes, "cold": [], "cached": []}

    # Cold passes
    for p in range(passes):
        sid = f"iso_cold_{context}_{p}_{int(time.time())}"
        messages = build_messages(corpus, context, offset=p * 1000)
        r = await measure_streaming_ttft(base_url, messages, session_id=sid)
        ttft = r.get("ttft_ms", 0)
        tps = r.get("decode_tps", 0)
        q = "OK" if r.get("quality_ok") else "FAIL"
        print(f"    cold/{context}tok/pass{p}... TTFT={ttft:.0f}ms TPS={tps:.1f} Q={q}")
        results["cold"].append(r)
        await asyncio.sleep(COOLDOWN_SECONDS)

    # Cached passes (prime + measure, self-contained)
    for p in range(passes):
        offset = p * 1000 + 50000
        messages = build_messages(corpus, context, offset=offset)
        sid = f"iso_prefix_{context}_{p}_{int(time.time())}"

        # Prime (cold)
        await measure_streaming_ttft(base_url, messages, session_id=sid)
        await asyncio.sleep(2)

        # Measure (should hit prefix cache)
        r = await measure_streaming_ttft(base_url, messages, session_id=sid)
        ttft = r.get("ttft_ms", 0)
        tps = r.get("decode_tps", 0)
        q = "OK" if r.get("quality_ok") else "FAIL"
        print(f"    cached/{context}tok/pass{p}... TTFT={ttft:.0f}ms TPS={tps:.1f} Q={q}")
        results["cached"].append(r)
        await asyncio.sleep(COOLDOWN_SECONDS)

    # Cleanup
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)

    return results


def main():
    parser = argparse.ArgumentParser(description="Isolated single-context vllm-mlx test")
    parser.add_argument("--context", type=int, required=True, help="Context length to test")
    parser.add_argument("--passes", type=int, default=3, help="Number of passes")
    parser.add_argument("--cache-memory-percent", type=float, default=0.40,
                        help="FP16 prefix cache memory fraction")
    args = parser.parse_args()

    print(f"{'#'*60}")
    print(f"  Isolated vllm-mlx test: {args.context} tokens, {args.passes} passes")
    print(f"  Cache memory: {args.cache_memory_percent*100:.0f}%")
    print(f"{'#'*60}")

    results = asyncio.run(run_isolated_test(args.context, args.passes, args.cache_memory_percent))

    # Save results
    ts = time.strftime("%Y%m%d_%H%M%S")
    outpath = f"benchmarks/results/isolated_vllm_{args.context}_{ts}.json"
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    with open(outpath, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to {outpath}")


if __name__ == "__main__":
    main()
