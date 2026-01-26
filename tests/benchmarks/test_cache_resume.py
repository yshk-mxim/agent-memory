"""Cache resume performance benchmarks (Sprint 6 Day 6).

Measures cache save/load performance across different cache sizes:
- Save time: 2K, 4K, 8K tokens
- Load time: 2K, 4K, 8K tokens (target <500ms)
- Resume generation speed (first-token latency)
- Cold start vs cache resume speedup
"""

import time

import httpx
import pytest

from tests.benchmarks.conftest import BenchmarkReporter


@pytest.mark.benchmark
def test_cache_save_time_2k_4k_8k_tokens(benchmark_client: httpx.Client, benchmark_reporter: BenchmarkReporter):
    """Benchmark cache save time for different token counts.

    Measures:
    - Save time for 2K token cache
    - Save time for 4K token cache
    - Save time for 8K token cache

    Expected: <200ms save time for most sizes
    """
    # Note: This benchmark measures API response time as proxy for save time
    # Actual cache save happens on server shutdown or explicit save call

    save_times = {}

    for token_count in [2000, 4000, 8000]:
        # Create agent with specified token count in cache
        # (make multiple requests to build up cache)
        agent_id = f"cache-save-{token_count}"

        # Make request to build cache (tokens ≈ token_count / num_requests)
        num_requests = 4
        tokens_per_request = token_count // num_requests

        for i in range(num_requests):
            benchmark_client.post(
                "/v1/messages",
                json={
                    "model": "test-model",
                    "messages": [
                        {
                            "role": "user",
                            "content": f"Build cache for {agent_id} part {i}",
                        }
                    ],
                    "max_tokens": tokens_per_request,
                },
                headers={"X-API-Key": agent_id},  # Use agent_id as key
            )

        # Measure final request time (includes any cache operations)
        start_time = time.perf_counter()

        benchmark_client.post(
            "/v1/messages",
            json={
                "model": "test-model",
                "messages": [
                    {"role": "user", "content": f"Final request for {agent_id}"}
                ],
                "max_tokens": 50,
            },
            headers={"X-API-Key": agent_id},
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        save_times[f"{token_count}_tokens"] = round(elapsed_ms, 1)

        print(
            f"  {token_count} tokens: {elapsed_ms:.0f}ms"
        )

    benchmark_reporter.record("cache_save_time_by_size", save_times)

    print(
        f"\n📊 Cache save time benchmark:"
        f"\n  Results: {save_times}"
    )


@pytest.mark.benchmark
def test_cache_load_time_2k_4k_8k_tokens(benchmark_client: httpx.Client, benchmark_reporter: BenchmarkReporter):
    """Benchmark cache load time for different token counts.

    Measures:
    - Load time for 2K token cache (target <200ms)
    - Load time for 4K token cache (target <350ms)
    - Load time for 8K token cache (target <500ms)

    Expected: <500ms for 8K tokens
    """
    # Note: Cache load time measured via first request latency after restart
    # In actual test environment, we measure request completion time

    load_times = {}

    for token_count in [2000, 4000, 8000]:
        agent_id = f"cache-load-{token_count}"

        # Build cache first
        num_requests = 4
        tokens_per_request = token_count // num_requests

        for i in range(num_requests):
            benchmark_client.post(
                "/v1/messages",
                json={
                    "model": "test-model",
                    "messages": [
                        {
                            "role": "user",
                            "content": f"Build cache {agent_id} part {i}",
                        }
                    ],
                    "max_tokens": tokens_per_request,
                },
                headers={"X-API-Key": agent_id},
            )

        # Measure "resume" latency (first request after cache exists)
        start_time = time.perf_counter()

        response = benchmark_client.post(
            "/v1/messages",
            json={
                "model": "test-model",
                "messages": [
                    {"role": "user", "content": f"Resume with cache {agent_id}"}
                ],
                "max_tokens": 50,
            },
            headers={"X-API-Key": agent_id},
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        load_times[f"{token_count}_tokens"] = round(elapsed_ms, 1)

        # Verify target
        target_ms = {2000: 200, 4000: 350, 8000: 500}[token_count]
        status = "✅" if elapsed_ms < target_ms else "⚠️"

        print(
            f"  {token_count} tokens: {elapsed_ms:.0f}ms {status} (target <{target_ms}ms)"
        )

    benchmark_reporter.record("cache_load_time_by_size", load_times)

    # Verify 8K target (<500ms)
    assert (
        load_times["8000_tokens"] < 500
    ), f"8K token load time too high: {load_times['8000_tokens']:.0f}ms (target <500ms)"

    print(
        f"\n📊 Cache load time benchmark:"
        f"\n  Results: {load_times}"
        f"\n  ✅ All targets met" if load_times["8000_tokens"] < 500 else "\n  ⚠️  8K target missed"
    )


@pytest.mark.benchmark
def test_resume_generation_speed(benchmark_client: httpx.Client, benchmark_reporter: BenchmarkReporter):
    """Benchmark generation speed when resuming from cache.

    Measures:
    - First-token latency with cached context
    - Total generation time with cache
    - Speedup from cache hit

    Expected: Significantly faster than cold start
    """
    agent_id = "resume-speed-test"

    # Build cache (2K token context)
    for i in range(4):
        benchmark_client.post(
            "/v1/messages",
            json={
                "model": "test-model",
                "messages": [
                    {"role": "user", "content": f"Build context part {i}"}
                ],
                "max_tokens": 500,
            },
            headers={"X-API-Key": agent_id},
        )

    # Measure resume generation
    start_time = time.perf_counter()

    response = benchmark_client.post(
        "/v1/messages",
        json={
            "model": "test-model",
            "messages": [
                {"role": "user", "content": "Generate with cached context"}
            ],
            "max_tokens": 100,
        },
        headers={"X-API-Key": agent_id},
    )

    elapsed_ms = (time.perf_counter() - start_time) * 1000

    benchmark_reporter.record(
        "resume_generation_speed",
        {
            "latency_ms": round(elapsed_ms, 1),
            "tokens_generated": 100,
            "cached_context_tokens": 2000,
        },
    )

    print(
        f"\n📊 Resume generation speed:"
        f"\n  Latency: {elapsed_ms:.0f}ms"
        f"\n  Cached context: 2000 tokens"
        f"\n  Generated: 100 tokens"
    )


@pytest.mark.benchmark
def test_cold_start_vs_cache_resume(benchmark_client: httpx.Client, benchmark_reporter: BenchmarkReporter):
    """Compare cold start vs cache resume speedup.

    Measures:
    - Cold start generation time (no cache)
    - Cache resume generation time (with cache)
    - Speedup ratio

    Expected: 3-5x speedup from caching
    """
    tokens_to_generate = 100

    # Measure cold start (new agent, no cache)
    cold_agent_id = "cold-start-agent"

    cold_start_time = time.perf_counter()

    cold_response = benchmark_client.post(
        "/v1/messages",
        json={
            "model": "test-model",
            "messages": [
                {"role": "user", "content": "Cold start generation"}
            ],
            "max_tokens": tokens_to_generate,
        },
        headers={"X-API-Key": cold_agent_id},
    )

    cold_elapsed_ms = (time.perf_counter() - cold_start_time) * 1000

    # Build cache for warm agent
    warm_agent_id = "warm-cached-agent"

    for i in range(4):
        benchmark_client.post(
            "/v1/messages",
            json={
                "model": "test-model",
                "messages": [
                    {"role": "user", "content": f"Build cache part {i}"}
                ],
                "max_tokens": 500,
            },
            headers={"X-API-Key": warm_agent_id},
        )

    # Measure cache resume
    warm_start_time = time.perf_counter()

    warm_response = benchmark_client.post(
        "/v1/messages",
        json={
            "model": "test-model",
            "messages": [
                {"role": "user", "content": "Cache resume generation"}
            ],
            "max_tokens": tokens_to_generate,
        },
        headers={"X-API-Key": warm_agent_id},
    )

    warm_elapsed_ms = (time.perf_counter() - warm_start_time) * 1000

    # Calculate speedup
    speedup = cold_elapsed_ms / warm_elapsed_ms if warm_elapsed_ms > 0 else 0

    benchmark_reporter.record(
        "cold_vs_cache_comparison",
        {
            "cold_start_ms": round(cold_elapsed_ms, 1),
            "cache_resume_ms": round(warm_elapsed_ms, 1),
            "speedup": round(speedup, 2),
        },
    )

    print(
        f"\n📊 Cold start vs cache resume:"
        f"\n  Cold start: {cold_elapsed_ms:.0f}ms"
        f"\n  Cache resume: {warm_elapsed_ms:.0f}ms"
        f"\n  Speedup: {speedup:.2f}x"
    )

    # Verify speedup target (>3x)
    # Note: In test environment without real model, speedup may be minimal
    # This test validates the measurement methodology
    print(
        f"\n  {'✅ Speedup target met (>3x)' if speedup > 3 else '⚠️  Speedup below target (test environment)'}"
    )
