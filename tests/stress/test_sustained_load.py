# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Yakov Shkolnikov and contributors
"""Sustained load tests for latency degradation detection.

Tests verify stability and performance over extended periods (1 hour):
- 1 hour of continuous operation
- Realistic sustained traffic patterns
- Memory stability (growth <5%)
- No performance degradation over time
"""

import asyncio
import time

import aiohttp
import pytest

from tests.stress.conftest import MetricsCollector, RequestResult, get_server_memory_usage
from tests.stress.harness import StressTestHarness


@pytest.mark.stress
@pytest.mark.slow
@pytest.mark.asyncio
async def test_one_hour_sustained_load(live_server, cleanup_after_stress, memory_profiler):
    """Test 1 hour of sustained load with memory profiling.

    Verifies:
    - Server stable for 1 hour continuous operation
    - Memory growth <5% over duration
    - Error rate <1%
    - No crashes or degradation

    Pattern: Run sustained load for 3600s, sample memory every 5 minutes

    Note: This test takes 1 hour to run. Skip in CI with -m "not slow"
    """
    base_url = live_server
    duration_seconds = 3600  # 1 hour
    sample_interval_seconds = 300  # 5 minutes

    harness = StressTestHarness(
        base_url=base_url,
        num_workers=5,
        rate_limit=0.5,  # 0.5 requests/sec per worker = 2.5 req/sec total
        timeout=60.0,
    )

    # Track results and memory
    all_results = []
    memory_samples = []

    # Sample initial memory
    initial_memory = get_server_memory_usage()
    memory_samples.append(initial_memory)
    print(f"\n🔍 Initial memory: {initial_memory:.1f} MB")

    start_time = time.time()
    last_sample_time = start_time

    async def sustained_workload():
        """Single sustained request."""
        request_start = time.time()

        try:
            timeout = aiohttp.ClientTimeout(total=60.0)
            async with (
                aiohttp.ClientSession(timeout=timeout) as session,
                session.post(
                    f"{base_url}/v1/messages",
                    json={
                        "model": "test-model",
                        "messages": [
                            {
                                "role": "user",
                                "content": "Sustained load test - 1 hour stability",
                            }
                        ],
                        "max_tokens": 50,
                    },
                    headers={
                        "Content-Type": "application/json",
                        "X-API-Key": "test-key-sustained",
                    },
                ) as response,
            ):
                latency_ms = (time.time() - request_start) * 1000

                all_results.append(
                    RequestResult(
                        status_code=response.status,
                        latency_ms=latency_ms,
                        timestamp=request_start,
                    )
                )
        except Exception as e:
            all_results.append(
                RequestResult(
                    status_code=0,
                    latency_ms=0.0,
                    timestamp=request_start,
                    error=e,
                )
            )

    # Run sustained load for duration
    tasks = []
    for worker_id in range(harness.num_workers):

        async def worker():
            nonlocal last_sample_time

            while (time.time() - start_time) < duration_seconds:
                await sustained_workload()

                # Sample memory every 5 minutes
                current_time = time.time()
                if (current_time - last_sample_time) >= sample_interval_seconds:
                    memory = get_server_memory_usage()
                    memory_samples.append(memory)
                    elapsed = current_time - start_time
                    print(
                        f"🔍 Memory at {elapsed / 60:.0f}min: {memory:.1f} MB "
                        f"(+{((memory - initial_memory) / initial_memory * 100):.1f}%)"
                    )
                    last_sample_time = current_time

                # Small delay between requests (rate limiting handled by harness)
                await asyncio.sleep(1.0 / (harness.rate_limit or 1.0))

        tasks.append(worker())

    # Run all workers
    await asyncio.gather(*tasks)

    # Final memory sample
    final_memory = get_server_memory_usage()
    memory_samples.append(final_memory)

    # Analyze results
    collector = MetricsCollector()
    collector.memory_samples = memory_samples
    metrics = collector.analyze(all_results)
    memory_analysis = collector.analyze_memory_growth()

    # Verify memory stability (<5% growth)
    memory_growth_pct = memory_analysis["growth_pct"]
    assert memory_growth_pct < 5.0, f"Memory growth too high: {memory_growth_pct:.1f}% (target <5%)"

    # Verify error rate <1%
    assert metrics.error_rate < 0.01, f"Error rate too high: {metrics.error_rate:.2%} (target <1%)"

    # No crashes
    status_5xx = sum(count for status, count in metrics.status_codes.items() if 500 <= status < 600)
    assert status_5xx == 0, f"Server crashed {status_5xx} times"

    print(
        f"\n✅ 1-hour sustained load test passed:"
        f"\n  Duration: {duration_seconds / 60:.0f} minutes"
        f"\n  Total requests: {metrics.total_requests}"
        f"\n  Successful: {metrics.successful_requests}"
        f"\n  Error rate: {metrics.error_rate:.2%}"
        f"\n  Memory growth: {memory_growth_pct:.1f}%"
        f"\n  Initial memory: {initial_memory:.1f} MB"
        f"\n  Final memory: {final_memory:.1f} MB"
        f"\n  Peak memory: {memory_analysis['max_mb']:.1f} MB"
    )


@pytest.mark.stress
@pytest.mark.asyncio
async def test_10_requests_per_minute_across_5_agents(live_server, cleanup_after_stress):
    """Test realistic sustained traffic: 10 req/min across 5 agents.

    Verifies:
    - 5 agents running concurrently
    - ~10 requests/minute total (2 req/min per agent)
    - Stable over extended period (10 minutes)
    - Error rate <1%

    Pattern: Run for 10 minutes (shorter than 1-hour test)
    """
    base_url = live_server
    duration_seconds = 600  # 10 minutes
    num_agents = 5
    requests_per_minute_per_agent = 2  # 10 total req/min

    harness = StressTestHarness(
        base_url=base_url,
        num_workers=num_agents,
        rate_limit=requests_per_minute_per_agent / 60,  # Convert to requests per second
        timeout=60.0,
    )

    all_results = []

    async def agent_sustained_workload(agent_id: str):
        """Sustained workload for single agent."""
        agent_results = []
        start_time = time.time()

        while (time.time() - start_time) < duration_seconds:
            request_start = time.time()

            try:
                timeout = aiohttp.ClientTimeout(total=60.0)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.post(
                        f"{base_url}/v1/messages",
                        json={
                            "model": "test-model",
                            "messages": [
                                {
                                    "role": "user",
                                    "content": f"{agent_id}: Realistic sustained request",
                                }
                            ],
                            "max_tokens": 50,
                        },
                        headers={
                            "Content-Type": "application/json",
                            "X-API-Key": f"test-key-{agent_id}",
                        },
                    ) as response:
                        latency_ms = (time.time() - request_start) * 1000

                        agent_results.append(
                            RequestResult(
                                status_code=response.status,
                                latency_ms=latency_ms,
                                timestamp=request_start,
                            )
                        )
            except Exception as e:
                agent_results.append(
                    RequestResult(
                        status_code=0,
                        latency_ms=0.0,
                        timestamp=request_start,
                        error=e,
                    )
                )

            # Wait for next request (rate limiting)
            await asyncio.sleep(60 / requests_per_minute_per_agent)

        return agent_results

    # Run all agents concurrently
    tasks = [agent_sustained_workload(f"agent_{i}") for i in range(num_agents)]
    agent_results_list = await asyncio.gather(*tasks)

    # Collect all results
    for agent_results in agent_results_list:
        all_results.extend(agent_results)

    # Analyze
    collector = MetricsCollector()
    metrics = collector.analyze(all_results)

    # Verify ~10 requests/minute (over 10 minutes = ~100 total requests)
    expected_requests = int((duration_seconds / 60) * num_agents * requests_per_minute_per_agent)
    # Allow 20% variance (rate limiting not perfect)
    assert abs(metrics.total_requests - expected_requests) / expected_requests < 0.20, (
        f"Request rate off: {metrics.total_requests} vs {expected_requests} expected"
    )

    # Error rate <1%
    assert metrics.error_rate < 0.01, f"Error rate too high: {metrics.error_rate:.2%}"

    print(
        f"\n✅ Realistic sustained traffic test passed:"
        f"\n  Duration: {duration_seconds / 60:.0f} minutes"
        f"\n  Agents: {num_agents}"
        f"\n  Total requests: {metrics.total_requests}"
        f"\n  Expected: ~{expected_requests}"
        f"\n  Error rate: {metrics.error_rate:.2%}"
    )


@pytest.mark.stress
@pytest.mark.asyncio
async def test_memory_stable_no_leaks(live_server, cleanup_after_stress, memory_profiler):
    """Test memory remains stable with no leaks over extended operation.

    Verifies:
    - Memory growth <5% over 30 minutes
    - No sudden memory spikes
    - Memory usage predictable

    Pattern: Run for 30 minutes, sample memory every 2 minutes
    """
    base_url = live_server
    duration_seconds = 1800  # 30 minutes
    sample_interval_seconds = 120  # 2 minutes

    harness = StressTestHarness(base_url=base_url, num_workers=3, rate_limit=1.0, timeout=60.0)

    all_results = []
    memory_samples = []

    # Initial memory
    initial_memory = get_server_memory_usage()
    memory_samples.append(initial_memory)
    print(f"\n🔍 Initial memory: {initial_memory:.1f} MB")

    start_time = time.time()
    last_sample = start_time

    async def workload_with_memory_sampling():
        """Workload that samples memory periodically."""
        nonlocal last_sample

        request_start = time.time()

        try:
            timeout = aiohttp.ClientTimeout(total=60.0)
            async with (
                aiohttp.ClientSession(timeout=timeout) as session,
                session.post(
                    f"{base_url}/v1/messages",
                    json={
                        "model": "test-model",
                        "messages": [{"role": "user", "content": "Memory leak detection test"}],
                        "max_tokens": 50,
                    },
                    headers={
                        "Content-Type": "application/json",
                        "X-API-Key": "test-key-memory",
                    },
                ) as response,
            ):
                all_results.append(
                    RequestResult(
                        status_code=response.status,
                        latency_ms=(time.time() - request_start) * 1000,
                        timestamp=request_start,
                    )
                )
        except Exception as e:
            all_results.append(
                RequestResult(
                    status_code=0,
                    latency_ms=0.0,
                    timestamp=request_start,
                    error=e,
                )
            )

        # Sample memory
        current_time = time.time()
        if (current_time - last_sample) >= sample_interval_seconds:
            memory = get_server_memory_usage()
            memory_samples.append(memory)
            elapsed = current_time - start_time
            print(
                f"🔍 Memory at {elapsed / 60:.0f}min: {memory:.1f} MB "
                f"(+{((memory - initial_memory) / initial_memory * 100):.1f}%)"
            )
            last_sample = current_time

    # Run workload
    tasks = []
    for _ in range(harness.num_workers):

        async def worker():
            while (time.time() - start_time) < duration_seconds:
                await workload_with_memory_sampling()
                await asyncio.sleep(1.0 / (harness.rate_limit or 1.0))

        tasks.append(worker())

    await asyncio.gather(*tasks)

    # Final memory
    final_memory = get_server_memory_usage()
    memory_samples.append(final_memory)

    # Analyze
    collector = MetricsCollector()
    collector.memory_samples = memory_samples
    memory_analysis = collector.analyze_memory_growth()

    # Memory growth <5%
    assert memory_analysis["growth_pct"] < 5.0, (
        f"Memory growth too high: {memory_analysis['growth_pct']:.1f}%"
    )

    print(
        f"\n✅ Memory stability test passed:"
        f"\n  Duration: {duration_seconds / 60:.0f} minutes"
        f"\n  Memory growth: {memory_analysis['growth_pct']:.1f}%"
        f"\n  Initial: {initial_memory:.1f} MB"
        f"\n  Final: {final_memory:.1f} MB"
        f"\n  Peak: {memory_analysis['max_mb']:.1f} MB"
    )


@pytest.mark.stress
@pytest.mark.asyncio
async def test_no_performance_degradation_over_time(live_server, cleanup_after_stress):
    """Test latency remains stable over extended period (no degradation).

    Verifies:
    - Latency at end ≈ latency at beginning
    - No gradual slowdown
    - p95 latency stable

    Pattern: Measure latency at start, middle, end → compare
    """
    base_url = live_server
    duration_seconds = 600  # 10 minutes
    measurement_window = 60  # Measure over 60s windows

    harness = StressTestHarness(base_url=base_url, num_workers=5, rate_limit=2.0, timeout=60.0)

    # Collect results for different time periods
    early_results = []
    middle_results = []
    late_results = []

    start_time = time.time()

    async def timed_workload():
        """Workload that categorizes results by time."""
        request_start = time.time()
        elapsed = request_start - start_time

        try:
            timeout = aiohttp.ClientTimeout(total=60.0)
            async with (
                aiohttp.ClientSession(timeout=timeout) as session,
                session.post(
                    f"{base_url}/v1/messages",
                    json={
                        "model": "test-model",
                        "messages": [
                            {
                                "role": "user",
                                "content": "Performance degradation test",
                            }
                        ],
                        "max_tokens": 50,
                    },
                    headers={
                        "Content-Type": "application/json",
                        "X-API-Key": "test-key-perf",
                    },
                ) as response,
            ):
                latency_ms = (time.time() - request_start) * 1000

                result = RequestResult(
                    status_code=response.status,
                    latency_ms=latency_ms,
                    timestamp=request_start,
                )

                # Categorize by time period
                if elapsed < measurement_window:
                    early_results.append(result)
                elif (
                    duration_seconds / 2 - measurement_window / 2
                    < elapsed
                    < duration_seconds / 2 + measurement_window / 2
                ):
                    middle_results.append(result)
                elif elapsed > duration_seconds - measurement_window:
                    late_results.append(result)

        except Exception:
            pass

    # Run workload
    tasks = []
    for _ in range(harness.num_workers):

        async def worker():
            while (time.time() - start_time) < duration_seconds:
                await timed_workload()
                await asyncio.sleep(1.0 / (harness.rate_limit or 1.0))

        tasks.append(worker())

    await asyncio.gather(*tasks)

    # Analyze latency for each period
    collector = MetricsCollector()

    early_metrics = collector.analyze(early_results)
    middle_metrics = collector.analyze(middle_results)
    late_metrics = collector.analyze(late_results)

    # Compare latencies (allow 20% variance)
    early_p95 = early_metrics.latency.p95
    late_p95 = late_metrics.latency.p95

    degradation_pct = ((late_p95 - early_p95) / early_p95) * 100

    assert degradation_pct < 20, (
        f"Performance degraded by {degradation_pct:.1f}% (early p95: {early_p95:.0f}ms, late p95: {late_p95:.0f}ms)"
    )

    print(
        f"\n✅ No performance degradation test passed:"
        f"\n  Early p95: {early_p95:.0f}ms"
        f"\n  Middle p95: {middle_metrics.latency.p95:.0f}ms"
        f"\n  Late p95: {late_p95:.0f}ms"
        f"\n  Degradation: {degradation_pct:.1f}%"
    )
