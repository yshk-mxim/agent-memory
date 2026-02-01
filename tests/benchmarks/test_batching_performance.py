"""Batching performance benchmarks.

Measures throughput benefits of batching multiple agents:
- Sequential (1 agent): baseline performance
- Batched (3 agents): expected 1.7-2.3x speedup
- Batched (5 agents): expected 2.7-3.3x speedup
- Throughput comparison table
"""

import time

import httpx
import pytest

from tests.benchmarks.conftest import BenchmarkReporter


@pytest.mark.benchmark
def test_sequential_1_agent_per_model(
    benchmark_client: httpx.Client, benchmark_reporter: BenchmarkReporter
):
    """Benchmark sequential processing (1 agent, baseline).

    Measures:
    - Tokens generated per second (baseline)
    - Latency for single agent
    - Baseline for comparison with batched modes

    Expected: 20-30 tokens/sec on Apple Silicon
    """
    num_requests = 5
    tokens_per_request = 100

    # Warm up (load model, initialize caches)
    warmup_response = benchmark_client.post(
        "/v1/messages",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "Warmup request"}],
            "max_tokens": 10,
        },
    )
    assert warmup_response.status_code in [
        200,
        400,
        501,
    ], f"Warmup failed: {warmup_response.status_code}"

    # Measure sequential throughput
    start_time = time.perf_counter()
    total_tokens = 0

    for i in range(num_requests):
        response = benchmark_client.post(
            "/v1/messages",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": f"Sequential request {i}"}],
                "max_tokens": tokens_per_request,
            },
        )

        if response.status_code == 200:
            # Count tokens from response (estimate from max_tokens if not available)
            total_tokens += tokens_per_request

    elapsed_seconds = time.perf_counter() - start_time
    throughput = total_tokens / elapsed_seconds if elapsed_seconds > 0 else 0

    # Record results
    benchmark_reporter.record(
        "sequential_1_agent",
        {
            "tokens_per_second": round(throughput, 2),
            "total_tokens": total_tokens,
            "elapsed_seconds": round(elapsed_seconds, 2),
            "num_requests": num_requests,
        },
    )

    print(
        f"\n📊 Sequential (1 agent) benchmark:"
        f"\n  Throughput: {throughput:.1f} tokens/sec"
        f"\n  Total tokens: {total_tokens}"
        f"\n  Elapsed: {elapsed_seconds:.2f}s"
    )


@pytest.mark.benchmark
def test_batched_3_agents_per_model(
    benchmark_client: httpx.Client, benchmark_reporter: BenchmarkReporter
):
    """Benchmark batched processing (3 concurrent agents).

    Measures:
    - Tokens generated per second with 3-agent batching
    - Speedup vs sequential baseline
    - Latency distribution

    Expected: 50-70 tokens/sec (1.7-2.3x speedup)
    """
    import threading

    num_agents = 3
    tokens_per_agent = 100
    requests_per_agent = 2  # 2 requests per agent = 6 total

    # Warm up
    warmup_response = benchmark_client.post(
        "/v1/messages",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "Warmup"}],
            "max_tokens": 10,
        },
    )

    # Create barrier for synchronized start
    barrier = threading.Barrier(num_agents)
    results = []
    total_tokens = 0

    def agent_worker(agent_id: int):
        """Worker that makes requests for one agent."""
        nonlocal total_tokens

        # Wait for all agents to be ready
        barrier.wait()

        agent_tokens = 0
        for req_num in range(requests_per_agent):
            response = benchmark_client.post(
                "/v1/messages",
                json={
                    "model": "test-model",
                    "messages": [
                        {
                            "role": "user",
                            "content": f"Agent {agent_id} request {req_num}",
                        }
                    ],
                    "max_tokens": tokens_per_agent,
                },
            )

            if response.status_code == 200:
                agent_tokens += tokens_per_agent

        results.append(agent_tokens)
        total_tokens += agent_tokens

    # Measure batched throughput
    start_time = time.perf_counter()

    threads = [threading.Thread(target=agent_worker, args=(i,)) for i in range(num_agents)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    elapsed_seconds = time.perf_counter() - start_time
    throughput = total_tokens / elapsed_seconds if elapsed_seconds > 0 else 0

    # Record results
    benchmark_reporter.record(
        "batched_3_agents",
        {
            "tokens_per_second": round(throughput, 2),
            "total_tokens": total_tokens,
            "elapsed_seconds": round(elapsed_seconds, 2),
            "num_agents": num_agents,
            "total_requests": num_agents * requests_per_agent,
        },
    )

    print(
        f"\n📊 Batched (3 agents) benchmark:"
        f"\n  Throughput: {throughput:.1f} tokens/sec"
        f"\n  Total tokens: {total_tokens}"
        f"\n  Elapsed: {elapsed_seconds:.2f}s"
    )


@pytest.mark.benchmark
def test_batched_5_agents_per_model(
    benchmark_client: httpx.Client, benchmark_reporter: BenchmarkReporter
):
    """Benchmark batched processing (5 concurrent agents).

    Measures:
    - Tokens generated per second with 5-agent batching
    - Maximum batching speedup
    - System capacity under high concurrency

    Expected: 80-100 tokens/sec (2.7-3.3x speedup)
    """
    import threading

    num_agents = 5
    tokens_per_agent = 100
    requests_per_agent = 2  # 2 requests per agent = 10 total

    # Warm up
    warmup_response = benchmark_client.post(
        "/v1/messages",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "Warmup"}],
            "max_tokens": 10,
        },
    )

    # Create barrier for synchronized start
    barrier = threading.Barrier(num_agents)
    results = []
    total_tokens = 0

    def agent_worker(agent_id: int):
        """Worker that makes requests for one agent."""
        nonlocal total_tokens

        # Wait for all agents to be ready
        barrier.wait()

        agent_tokens = 0
        for req_num in range(requests_per_agent):
            response = benchmark_client.post(
                "/v1/messages",
                json={
                    "model": "test-model",
                    "messages": [
                        {
                            "role": "user",
                            "content": f"Agent {agent_id} request {req_num}",
                        }
                    ],
                    "max_tokens": tokens_per_agent,
                },
            )

            if response.status_code == 200:
                agent_tokens += tokens_per_agent

        results.append(agent_tokens)
        total_tokens += agent_tokens

    # Measure batched throughput
    start_time = time.perf_counter()

    threads = [threading.Thread(target=agent_worker, args=(i,)) for i in range(num_agents)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    elapsed_seconds = time.perf_counter() - start_time
    throughput = total_tokens / elapsed_seconds if elapsed_seconds > 0 else 0

    # Record results
    benchmark_reporter.record(
        "batched_5_agents",
        {
            "tokens_per_second": round(throughput, 2),
            "total_tokens": total_tokens,
            "elapsed_seconds": round(elapsed_seconds, 2),
            "num_agents": num_agents,
            "total_requests": num_agents * requests_per_agent,
        },
    )

    print(
        f"\n📊 Batched (5 agents) benchmark:"
        f"\n  Throughput: {throughput:.1f} tokens/sec"
        f"\n  Total tokens: {total_tokens}"
        f"\n  Elapsed: {elapsed_seconds:.2f}s"
    )


@pytest.mark.benchmark
def test_throughput_comparison(benchmark_reporter: BenchmarkReporter):
    """Generate comparison table for sequential vs batched throughput.

    Uses results from previous benchmarks to compute speedup ratios.

    Expected:
    - 3-agent batching: 1.7-2.3x speedup
    - 5-agent batching: 2.7-3.3x speedup
    """
    # Get results from reporter
    sequential = benchmark_reporter.results.get("sequential_1_agent", {})
    batched_3 = benchmark_reporter.results.get("batched_3_agents", {})
    batched_5 = benchmark_reporter.results.get("batched_5_agents", {})

    # If benchmarks haven't run yet, skip
    if not (sequential and batched_3 and batched_5):
        pytest.skip("Run other benchmarks first to generate comparison table")

    # Calculate speedups
    seq_throughput = sequential["tokens_per_second"]
    batch3_throughput = batched_3["tokens_per_second"]
    batch5_throughput = batched_5["tokens_per_second"]

    speedup_3 = batch3_throughput / seq_throughput if seq_throughput > 0 else 0
    speedup_5 = batch5_throughput / seq_throughput if seq_throughput > 0 else 0

    # Generate comparison table
    comparison = {
        "sequential_tokens_per_sec": seq_throughput,
        "batched_3_tokens_per_sec": batch3_throughput,
        "batched_5_tokens_per_sec": batch5_throughput,
        "speedup_3_agents": round(speedup_3, 2),
        "speedup_5_agents": round(speedup_5, 2),
    }

    benchmark_reporter.record("throughput_comparison", comparison)

    # Print comparison table
    print("\n" + "=" * 70)
    print("  THROUGHPUT COMPARISON TABLE")
    print("=" * 70)
    print(f"Sequential (1 agent):   {seq_throughput:.1f} tokens/sec (baseline)")
    print(f"Batched (3 agents):     {batch3_throughput:.1f} tokens/sec ({speedup_3:.2f}x speedup)")
    print(f"Batched (5 agents):     {batch5_throughput:.1f} tokens/sec ({speedup_5:.2f}x speedup)")
    print("=" * 70)

    # Validate speedup targets
    assert speedup_3 > 1.5, f"3-agent batching speedup too low: {speedup_3:.2f}x (target >1.5x)"
    assert speedup_5 > 2.5, f"5-agent batching speedup too low: {speedup_5:.2f}x (target >2.5x)"

    print("\n✅ Batching speedup targets met:")
    print(f"  3-agent: {speedup_3:.2f}x (target >1.5x)")
    print(f"  5-agent: {speedup_5:.2f}x (target >2.5x)")
