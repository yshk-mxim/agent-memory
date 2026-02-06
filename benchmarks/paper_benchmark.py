#!/usr/bin/env python3
"""
Paper Benchmark Suite - Replicates all benchmarks from semantic_colm2026 paper.

This script runs the complete benchmark suite as described in the paper:
- Context lengths: 1K, 2K, 4K, 8K, 16K, 32K tokens
- Cache states: Cold, Warm, Hot
- Metrics: TTFT, E2E latency, tokens generated, memory usage

Results are saved with full statistics (min, max, median, average, std).

Usage:
    python benchmarks/paper_benchmark.py

    # Quick test with fewer runs
    python benchmarks/paper_benchmark.py --runs 1

    # Specific contexts only
    python benchmarks/paper_benchmark.py --contexts 1024 4096 16384
"""

import argparse
import asyncio
import gc
import json
import os
import signal
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

# Configuration
MODEL_ID = "mlx-community/gemma-3-12b-it-4bit"
SERVER_PORT = 8000
BASE_URL = f"http://localhost:{SERVER_PORT}"
OUTPUT_TOKENS = 64
RUNS_PER_CONFIG = 3
COOLDOWN_BETWEEN_RUNS = 5  # seconds
CONTEXT_LENGTHS = [1024, 2048, 4096, 8192, 16384, 32768]

# Padding text for context generation
PADDING_TEXT = """
The quick brown fox jumps over the lazy dog. This is a sample text used for padding.
Machine learning models require substantial context to demonstrate their capabilities.
Natural language processing has advanced significantly in recent years.
Transformer architectures have revolutionized the field of artificial intelligence.
Attention mechanisms allow models to focus on relevant parts of the input.
"""

RESULTS_DIR = Path(__file__).parent / "results"


@dataclass
class BenchmarkResult:
    """Single benchmark measurement."""
    context_tokens: int
    cache_state: str  # cold, warm, hot
    run_id: int
    ttft_ms: float  # Time to first token
    e2e_ms: float  # End-to-end latency
    tokens_generated: int
    decode_tps: float  # Tokens per second during decode
    memory_mb: float  # Memory usage estimate
    cache_size_mb: float = 0.0  # Cache file size on disk
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class AggregatedStats:
    """Statistics across multiple runs."""
    context_tokens: int
    cache_state: str
    n_runs: int
    ttft_min: float
    ttft_max: float
    ttft_median: float
    ttft_mean: float
    ttft_std: float
    e2e_min: float
    e2e_max: float
    e2e_median: float
    e2e_mean: float
    e2e_std: float
    tokens_min: int
    tokens_max: int
    tokens_median: float
    decode_tps_mean: float
    memory_mb_mean: float
    cache_size_mb_mean: float = 0.0


def generate_padding(target_tokens: int) -> str:
    """Generate padding text to reach target token count (approximate)."""
    # Rough estimate: 4 chars per token
    target_chars = target_tokens * 4
    padding = PADDING_TEXT * (target_chars // len(PADDING_TEXT) + 1)
    return padding[:target_chars]


def build_messages(context_tokens: int) -> list[dict]:
    """Build chat messages with specified context length."""
    padding = generate_padding(context_tokens - 50)  # Reserve tokens for structure
    return [
        {"role": "system", "content": "You are a helpful assistant. Be concise."},
        {"role": "user", "content": f"Here is some context:\n\n{padding}\n\nNow, briefly summarize what you understand."}
    ]


def build_followup(messages: list[dict], response: str, output_tokens: int) -> dict:
    """Build follow-up request for multi-turn."""
    new_messages = messages + [
        {"role": "assistant", "content": response},
        {"role": "user", "content": "Please continue with more details."}
    ]
    return {
        "model": "gemma",
        "messages": new_messages,
        "max_tokens": output_tokens,
        "temperature": 0.0,
    }


class SemanticClient:
    """Client for semantic server OpenAI-compatible API."""

    def __init__(self, base_url: str):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=600.0)

    async def close(self):
        await self.client.aclose()

    async def health_check(self) -> bool:
        try:
            r = await self.client.get(f"{self.base_url}/health")
            return r.status_code == 200
        except Exception:
            return False

    async def delete_agent(self, agent_id: str, evict_only: bool = False) -> None:
        """Delete or evict agent."""
        try:
            url = f"{self.base_url}/v1/agents/{agent_id}"
            if evict_only:
                url += "?evict_only=true"
            await self.client.delete(url)
        except Exception:
            pass

    async def send_streaming(
        self,
        messages: list[dict],
        session_id: str,
        output_tokens: int = OUTPUT_TOKENS,
    ) -> BenchmarkResult:
        """Send streaming request and measure TTFT + decode."""
        body = {
            "model": "gemma",
            "messages": messages,
            "max_tokens": output_tokens,
            "temperature": 0.0,
            "stream": True,
            "session_id": session_id,
        }

        start = time.perf_counter()
        ttft = None
        tokens = 0
        first_token_time = None

        async with self.client.stream(
            "POST",
            f"{self.base_url}/v1/chat/completions",
            json=body,
        ) as response:
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                    if chunk.get("choices", [{}])[0].get("delta", {}).get("content"):
                        if ttft is None:
                            ttft = (time.perf_counter() - start) * 1000
                            first_token_time = time.perf_counter()
                        tokens += 1
                except json.JSONDecodeError:
                    pass

        end = time.perf_counter()
        e2e = (end - start) * 1000

        # Calculate decode TPS (excluding TTFT)
        if first_token_time and tokens > 1:
            decode_time = end - first_token_time
            decode_tps = (tokens - 1) / decode_time if decode_time > 0 else 0
        else:
            decode_tps = 0

        return BenchmarkResult(
            context_tokens=0,  # Filled by caller
            cache_state="",  # Filled by caller
            run_id=0,  # Filled by caller
            ttft_ms=ttft or e2e,
            e2e_ms=e2e,
            tokens_generated=tokens,
            decode_tps=decode_tps,
            memory_mb=0,  # Filled later
        )

    async def send_non_streaming(
        self,
        messages: list[dict],
        session_id: str,
        output_tokens: int = OUTPUT_TOKENS,
    ) -> tuple[float, int, str]:
        """Send non-streaming request, return (e2e_ms, tokens, response_text)."""
        body = {
            "model": "gemma",
            "messages": messages,
            "max_tokens": output_tokens,
            "temperature": 0.0,
            "stream": False,
            "session_id": session_id,
        }

        start = time.perf_counter()
        r = await self.client.post(
            f"{self.base_url}/v1/chat/completions",
            json=body,
        )
        e2e = (time.perf_counter() - start) * 1000

        data = r.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        tokens = data.get("usage", {}).get("completion_tokens", len(content.split()))

        return e2e, tokens, content


def get_memory_usage() -> float:
    """Get current memory usage of semantic server process."""
    try:
        result = subprocess.run(
            ["lsof", f"-ti:{SERVER_PORT}"],
            capture_output=True,
            text=True
        )
        if result.stdout.strip():
            pid = result.stdout.strip().split('\n')[0]
            ps_result = subprocess.run(
                ["ps", "-o", "rss=", "-p", pid],
                capture_output=True,
                text=True
            )
            if ps_result.stdout.strip():
                return int(ps_result.stdout.strip()) / 1024  # Convert to MB
    except Exception:
        pass
    return 0.0


def get_cache_size() -> float:
    """Get total size of cache files in MB."""
    cache_dir = Path.home() / ".semantic" / "caches"
    if not cache_dir.exists():
        return 0.0
    total_bytes = sum(f.stat().st_size for f in cache_dir.glob("*.safetensors"))
    return total_bytes / (1024 * 1024)


def kill_all_servers():
    """Kill any running semantic servers."""
    try:
        subprocess.run(["pkill", "-9", "-f", "semantic serve"],
                      capture_output=True, timeout=5)
    except Exception:
        pass
    try:
        result = subprocess.run(["lsof", f"-ti:{SERVER_PORT}"],
                               capture_output=True, text=True)
        for pid in result.stdout.strip().split('\n'):
            if pid:
                os.kill(int(pid), signal.SIGKILL)
    except Exception:
        pass
    time.sleep(2)


def start_server() -> subprocess.Popen:
    """Start semantic server and wait for ready."""
    print(f"Starting semantic server with {MODEL_ID}...")

    # Clear cache directory
    cache_dir = Path.home() / ".semantic" / "caches"
    if cache_dir.exists():
        for f in cache_dir.glob("*.safetensors"):
            try:
                f.unlink()
            except Exception:
                pass
        print(f"Cleared cache directory: {cache_dir}")

    log_file = open("/tmp/claude/semantic_benchmark.log", "w")
    proc = subprocess.Popen(
        ["semantic", "serve", "--model", MODEL_ID, "--port", str(SERVER_PORT)],
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )

    # Wait for server
    print("Waiting for server to be ready...")
    for i in range(120):
        try:
            r = httpx.get(f"{BASE_URL}/health", timeout=2.0)
            if r.status_code == 200:
                print(f"Server ready after {i+1} seconds")
                return proc
        except Exception:
            pass
        time.sleep(1)
        if i % 10 == 9:
            print(f"  Still waiting... ({i+1}s)")

    raise RuntimeError("Server failed to start within 120 seconds")


async def run_cold_benchmark(
    client: SemanticClient,
    context_tokens: int,
    run_id: int,
) -> BenchmarkResult:
    """Run cold cache benchmark - no prior cache."""
    session_id = f"cold_{context_tokens}_{run_id}_{int(time.time())}"
    agent_id = f"oai_{session_id}"

    # Ensure clean state
    await client.delete_agent(agent_id)

    messages = build_messages(context_tokens)
    result = await client.send_streaming(messages, session_id)

    # Cleanup
    await client.delete_agent(agent_id)

    result.context_tokens = context_tokens
    result.cache_state = "cold"
    result.run_id = run_id
    result.memory_mb = get_memory_usage()
    result.cache_size_mb = get_cache_size()

    return result


async def run_warm_benchmark(
    client: SemanticClient,
    context_tokens: int,
    run_id: int,
) -> BenchmarkResult:
    """Run warm cache benchmark - cache loaded from disk."""
    session_id = f"warm_{context_tokens}_{run_id}_{int(time.time())}"
    agent_id = f"oai_{session_id}"

    # Ensure clean state
    await client.delete_agent(agent_id)

    messages = build_messages(context_tokens)

    # Prime: create cache (cold hit)
    await client.send_non_streaming(messages, session_id)

    # Evict from hot tier but keep disk file
    await client.delete_agent(agent_id, evict_only=True)
    await asyncio.sleep(1.5)  # Allow disk write to complete

    # Measure: reload from disk (warm hit)
    result = await client.send_streaming(messages, session_id)

    # Full cleanup
    await client.delete_agent(agent_id)

    result.context_tokens = context_tokens
    result.cache_state = "warm"
    result.run_id = run_id
    result.memory_mb = get_memory_usage()
    result.cache_size_mb = get_cache_size()

    return result


async def run_hot_benchmark(
    client: SemanticClient,
    context_tokens: int,
    run_id: int,
) -> BenchmarkResult:
    """Run hot cache benchmark - cache in memory, multi-turn extend."""
    session_id = f"hot_{context_tokens}_{run_id}_{int(time.time())}"
    agent_id = f"oai_{session_id}"

    # Ensure clean state
    await client.delete_agent(agent_id)

    messages = build_messages(context_tokens)

    # Turn 1: Cold start
    _, _, response1 = await client.send_non_streaming(messages, session_id)
    await asyncio.sleep(0.5)

    # Turn 2: Extend cache
    followup1 = build_followup(messages, response1, OUTPUT_TOKENS)
    _, _, response2 = await client.send_non_streaming(
        followup1["messages"], session_id
    )
    await asyncio.sleep(0.5)

    # Turn 3: Hot measurement (cache fully in memory)
    followup2 = build_followup(followup1["messages"], response2, OUTPUT_TOKENS)
    result = await client.send_streaming(followup2["messages"], session_id)

    # Cleanup
    await client.delete_agent(agent_id)

    result.context_tokens = context_tokens
    result.cache_state = "hot"
    result.run_id = run_id
    result.memory_mb = get_memory_usage()
    result.cache_size_mb = get_cache_size()

    return result


def compute_aggregate_stats(results: list[BenchmarkResult]) -> AggregatedStats:
    """Compute statistics across multiple runs."""
    ttfts = [r.ttft_ms for r in results]
    e2es = [r.e2e_ms for r in results]
    tokens = [r.tokens_generated for r in results]
    tps = [r.decode_tps for r in results]
    memory = [r.memory_mb for r in results]
    cache_sizes = [r.cache_size_mb for r in results]

    return AggregatedStats(
        context_tokens=results[0].context_tokens,
        cache_state=results[0].cache_state,
        n_runs=len(results),
        ttft_min=min(ttfts),
        ttft_max=max(ttfts),
        ttft_median=statistics.median(ttfts),
        ttft_mean=statistics.mean(ttfts),
        ttft_std=statistics.stdev(ttfts) if len(ttfts) > 1 else 0,
        e2e_min=min(e2es),
        e2e_max=max(e2es),
        e2e_median=statistics.median(e2es),
        e2e_mean=statistics.mean(e2es),
        e2e_std=statistics.stdev(e2es) if len(e2es) > 1 else 0,
        tokens_min=min(tokens),
        tokens_max=max(tokens),
        tokens_median=statistics.median(tokens),
        decode_tps_mean=statistics.mean(tps),
        memory_mb_mean=statistics.mean(memory),
        cache_size_mb_mean=statistics.mean(cache_sizes) if cache_sizes else 0.0,
    )


def verify_ordering(stats_by_context: dict[int, dict[str, AggregatedStats]]) -> list[str]:
    """Verify that results follow expected ordering: cold > warm > hot."""
    issues = []

    for ctx, states in stats_by_context.items():
        cold = states.get("cold")
        warm = states.get("warm")
        hot = states.get("hot")

        if cold and warm:
            if warm.ttft_median >= cold.ttft_median:
                issues.append(f"Context {ctx}: warm TTFT ({warm.ttft_median:.0f}ms) >= cold ({cold.ttft_median:.0f}ms)")

        if warm and hot:
            if hot.ttft_median >= warm.ttft_median:
                issues.append(f"Context {ctx}: hot TTFT ({hot.ttft_median:.0f}ms) >= warm ({warm.ttft_median:.0f}ms)")

        if cold and hot:
            if hot.ttft_median >= cold.ttft_median * 0.5:
                # Hot should be significantly faster than cold
                pass  # This is expected at low context lengths

    return issues


def print_table(stats_list: list[AggregatedStats]):
    """Print results in table format."""
    print("\n" + "=" * 115)
    print(f"{'Context':>8} {'State':>6} {'TTFT min':>10} {'TTFT med':>10} {'TTFT max':>10} "
          f"{'E2E med':>10} {'Tokens':>8} {'TPS':>8} {'Memory':>10} {'Cache':>10}")
    print("=" * 115)

    for s in sorted(stats_list, key=lambda x: (x.context_tokens,
                                                {"cold": 0, "warm": 1, "hot": 2}.get(x.cache_state, 3))):
        print(f"{s.context_tokens:>8} {s.cache_state:>6} {s.ttft_min:>10.0f} {s.ttft_median:>10.0f} "
              f"{s.ttft_max:>10.0f} {s.e2e_median:>10.0f} {s.tokens_median:>8.0f} "
              f"{s.decode_tps_mean:>8.1f} {s.memory_mb_mean:>10.0f} {s.cache_size_mb_mean:>10.1f}")
    print("=" * 115)


async def main():
    parser = argparse.ArgumentParser(description="Paper benchmark suite")
    parser.add_argument("--runs", type=int, default=RUNS_PER_CONFIG,
                       help=f"Runs per configuration (default: {RUNS_PER_CONFIG})")
    parser.add_argument("--contexts", type=int, nargs="+", default=CONTEXT_LENGTHS,
                       help=f"Context lengths to test (default: {CONTEXT_LENGTHS})")
    parser.add_argument("--output", type=str, default=None,
                       help="Output JSON file path")
    args = parser.parse_args()

    print("=" * 70)
    print("SEMANTIC PAPER BENCHMARK SUITE")
    print("=" * 70)
    print(f"Model: {MODEL_ID}")
    print(f"Context lengths: {args.contexts}")
    print(f"Runs per config: {args.runs}")
    print(f"Output tokens: {OUTPUT_TOKENS}")
    print()

    # Kill any existing servers
    print("Killing any existing servers...")
    kill_all_servers()

    # Start fresh server
    server_proc = start_server()

    try:
        client = SemanticClient(BASE_URL)

        # Verify server is healthy
        if not await client.health_check():
            raise RuntimeError("Server health check failed")

        all_results: list[BenchmarkResult] = []
        stats_by_context: dict[int, dict[str, AggregatedStats]] = {}

        for context_tokens in args.contexts:
            print(f"\n{'='*70}")
            print(f"CONTEXT: {context_tokens} tokens")
            print(f"{'='*70}")

            stats_by_context[context_tokens] = {}

            # Run Cold benchmarks
            print(f"\n[COLD] Running {args.runs} iterations...")
            cold_results = []
            for run_id in range(args.runs):
                print(f"  Run {run_id + 1}/{args.runs}...", end=" ", flush=True)
                result = await run_cold_benchmark(client, context_tokens, run_id)
                cold_results.append(result)
                all_results.append(result)
                print(f"TTFT={result.ttft_ms:.0f}ms, E2E={result.e2e_ms:.0f}ms, "
                      f"tokens={result.tokens_generated}")

                # Cooldown between runs
                gc.collect()
                await asyncio.sleep(COOLDOWN_BETWEEN_RUNS)

            cold_stats = compute_aggregate_stats(cold_results)
            stats_by_context[context_tokens]["cold"] = cold_stats
            print(f"  Cold TTFT: {cold_stats.ttft_median:.0f}ms (median), "
                  f"range [{cold_stats.ttft_min:.0f}, {cold_stats.ttft_max:.0f}]")

            # Run Warm benchmarks
            print(f"\n[WARM] Running {args.runs} iterations...")
            warm_results = []
            for run_id in range(args.runs):
                print(f"  Run {run_id + 1}/{args.runs}...", end=" ", flush=True)
                result = await run_warm_benchmark(client, context_tokens, run_id)
                warm_results.append(result)
                all_results.append(result)
                print(f"TTFT={result.ttft_ms:.0f}ms, E2E={result.e2e_ms:.0f}ms, "
                      f"tokens={result.tokens_generated}")

                gc.collect()
                await asyncio.sleep(COOLDOWN_BETWEEN_RUNS)

            warm_stats = compute_aggregate_stats(warm_results)
            stats_by_context[context_tokens]["warm"] = warm_stats
            print(f"  Warm TTFT: {warm_stats.ttft_median:.0f}ms (median), "
                  f"range [{warm_stats.ttft_min:.0f}, {warm_stats.ttft_max:.0f}]")

            # Run Hot benchmarks
            print(f"\n[HOT] Running {args.runs} iterations...")
            hot_results = []
            for run_id in range(args.runs):
                print(f"  Run {run_id + 1}/{args.runs}...", end=" ", flush=True)
                result = await run_hot_benchmark(client, context_tokens, run_id)
                hot_results.append(result)
                all_results.append(result)
                print(f"TTFT={result.ttft_ms:.0f}ms, E2E={result.e2e_ms:.0f}ms, "
                      f"tokens={result.tokens_generated}")

                gc.collect()
                await asyncio.sleep(COOLDOWN_BETWEEN_RUNS)

            hot_stats = compute_aggregate_stats(hot_results)
            stats_by_context[context_tokens]["hot"] = hot_stats
            print(f"  Hot TTFT: {hot_stats.ttft_median:.0f}ms (median), "
                  f"range [{hot_stats.ttft_min:.0f}, {hot_stats.ttft_max:.0f}]")

            # Verify ordering for this context
            speedup_warm = cold_stats.ttft_median / warm_stats.ttft_median if warm_stats.ttft_median > 0 else 0
            speedup_hot = cold_stats.ttft_median / hot_stats.ttft_median if hot_stats.ttft_median > 0 else 0
            print(f"\n  Speedups: warm={speedup_warm:.2f}x, hot={speedup_hot:.2f}x")

        await client.close()

        # Print summary table
        all_stats = []
        for ctx_stats in stats_by_context.values():
            all_stats.extend(ctx_stats.values())
        print_table(all_stats)

        # Verify ordering
        print("\n" + "=" * 70)
        print("VERIFICATION")
        print("=" * 70)
        issues = verify_ordering(stats_by_context)
        if issues:
            print("⚠ Ordering issues detected:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print("✓ All results follow expected ordering (cold > warm > hot)")

        # Save results
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        output_file = args.output or str(RESULTS_DIR / f"paper_benchmark_{timestamp}.json")

        output_data = {
            "metadata": {
                "model": MODEL_ID,
                "contexts": args.contexts,
                "runs_per_config": args.runs,
                "output_tokens": OUTPUT_TOKENS,
                "timestamp": datetime.now().isoformat(),
            },
            "raw_results": [asdict(r) for r in all_results],
            "aggregated_stats": [asdict(s) for s in all_stats],
            "verification_issues": issues,
        }

        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {output_file}")

        # Print paper-format table (Table 1 from paper)
        print("\n" + "=" * 70)
        print("TABLE 1: TTFT (ms) across context lengths and cache states")
        print("=" * 70)
        print(f"{'Cache State':>12}", end="")
        for ctx in args.contexts:
            print(f" {ctx//1024}K".rjust(8), end="")
        print()
        print("-" * 70)

        for state in ["cold", "warm", "hot"]:
            print(f"{state.capitalize():>12}", end="")
            for ctx in args.contexts:
                if ctx in stats_by_context and state in stats_by_context[ctx]:
                    ttft = stats_by_context[ctx][state].ttft_median
                    print(f"{ttft:>8.0f}", end="")
                else:
                    print(f"{'N/A':>8}", end="")
            print()
        print("=" * 70)

    finally:
        print("\nShutting down server...")
        server_proc.terminate()
        try:
            server_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server_proc.kill()
        kill_all_servers()


if __name__ == "__main__":
    asyncio.run(main())
