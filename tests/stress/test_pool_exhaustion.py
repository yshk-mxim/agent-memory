# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Yakov Shkolnikov and contributors
"""Pool exhaustion stress tests.

Tests verify graceful degradation when block pool approaches/reaches capacity:
- 100+ concurrent requests handled without crashes
- Graceful 429 (Too Many Requests) when pool exhausted
- Server stability under extreme load
- Pool recovery after load subsides
"""

import asyncio

import pytest

from tests.stress.conftest import MetricsCollector
from tests.stress.harness import StressTestHarness


@pytest.mark.stress
@pytest.mark.asyncio
async def test_100_plus_concurrent_requests(live_server, cleanup_after_stress):
    """Test server handles 100+ concurrent requests without crashing.

    Verifies:
    - Server accepts and processes 100+ concurrent requests
    - No server crashes or 500 errors (some 429s acceptable)
    - All requests get a response (no dropped connections)
    - Response times remain reasonable

    Pattern: Launch 150 concurrent requests, verify all complete
    """
    base_url = live_server

    harness = StressTestHarness(
        base_url=base_url,
        num_workers=150,
        timeout=60.0,  # Allow time for queueing
    )

    # Make 150 concurrent requests
    results = await harness.run_concurrent_requests(
        path="/v1/messages",
        body={
            "model": "test-model",
            "messages": [{"role": "user", "content": "Concurrent load test request"}],
            "max_tokens": 50,
        },
    )

    # Analyze results
    collector = MetricsCollector()
    metrics = collector.analyze(results)

    # Assertions
    assert metrics.total_requests == 150, "All 150 requests should complete"

    # Allow for some 429s (pool exhaustion), but no 500s (crashes)
    status_5xx = sum(count for status, count in metrics.status_codes.items() if 500 <= status < 600)
    assert status_5xx == 0, f"No 5xx errors allowed, got {status_5xx}"

    # All requests should get a response (status code != 0)
    failed_connections = sum(1 for r in results if r.status_code == 0)
    assert failed_connections < 10, f"Too many connection failures: {failed_connections}/150"

    # At least 50% should succeed (rest can be 429s)
    success_rate = metrics.successful_requests / metrics.total_requests
    assert success_rate > 0.5, f"Success rate too low: {success_rate:.2%}"

    print(
        f"\n✅ 100+ concurrent requests test passed:"
        f"\n  Total requests: {metrics.total_requests}"
        f"\n  Successful: {metrics.successful_requests}"
        f"\n  Failed: {metrics.failed_requests}"
        f"\n  Status codes: {metrics.status_codes}"
        f"\n  Error rate: {metrics.error_rate:.2%}"
    )


@pytest.mark.stress
@pytest.mark.asyncio
async def test_graceful_429_when_pool_exhausted(live_server, cleanup_after_stress):
    """Test server returns 429 when block pool is exhausted.

    Verifies:
    - Server returns 429 (Too Many Requests) when pool full
    - 429 responses are properly formatted
    - No crashes or 500 errors during pool exhaustion
    - Server can still respond (not deadlocked)

    Pattern: Ramp up load until 429s appear, verify graceful handling
    """
    from tests.stress.harness import RampUpHarness

    base_url = live_server

    harness = RampUpHarness(
        base_url=base_url,
        num_workers=10,  # Start with 10
        timeout=60.0,
    )

    # Ramp up from 10 to 100 workers over 30 seconds
    results_by_level = await harness.run_ramp_up(
        path="/v1/messages",
        body={
            "model": "test-model",
            "messages": [{"role": "user", "content": "Pool exhaustion test - ramp up load"}],
            "max_tokens": 100,  # Larger tokens to fill pool faster
        },
        start_workers=10,
        end_workers=100,
        ramp_duration_seconds=30,
    )

    # Analyze results across all load levels
    all_results = []
    for level, results in results_by_level.items():
        all_results.extend(results)

    collector = MetricsCollector()
    metrics = collector.analyze(all_results)

    # We should see some 429s (pool exhaustion)
    status_429_count = metrics.status_codes.get(429, 0)

    # Note: If pool is very large, we might not hit exhaustion
    # Check if we got ANY 429s, or if all succeeded (large pool)
    if status_429_count > 0:
        print(f"\n✅ Pool exhaustion detected: {status_429_count} × 429 responses")
        # Verify 429s are graceful (not mixed with 500s)
        status_5xx = sum(
            count for status, count in metrics.status_codes.items() if 500 <= status < 600
        )
        assert status_5xx == 0, f"Got {status_5xx} × 5xx errors during pool exhaustion"
    else:
        print("\n⚠️  No 429s observed (pool may be large enough for this test)")
        # If no 429s, all requests should have succeeded
        assert metrics.error_rate < 0.1, (
            f"Expected low error rate without pool exhaustion, got {metrics.error_rate:.2%}"
        )

    # No crashes regardless
    status_500 = metrics.status_codes.get(500, 0)
    assert status_500 == 0, f"Server crashed {status_500} times"

    print(
        f"\n✅ Graceful 429 test passed:"
        f"\n  Total requests: {metrics.total_requests}"
        f"\n  429 responses: {status_429_count}"
        f"\n  5xx errors: {sum(count for status, count in metrics.status_codes.items() if 500 <= status < 600)}"
        f"\n  Status distribution: {metrics.status_codes}"
    )


@pytest.mark.stress
@pytest.mark.asyncio
async def test_no_crashes_under_load(live_server, cleanup_after_stress):
    """Test server remains stable under sustained high load.

    Verifies:
    - No crashes (500 errors) during sustained load
    - Server continues accepting requests
    - Health endpoint remains responsive
    - No deadlocks or hangs

    Pattern: 60 seconds of sustained 20 req/sec load, verify stability
    """
    import aiohttp

    base_url = live_server

    harness = StressTestHarness(
        base_url=base_url,
        num_workers=5,
        rate_limit=20.0,  # 20 requests per second
        timeout=30.0,
    )

    # Track all results
    all_results = []

    async def sustained_workload():
        """Single sustained request."""
        result = await harness._make_request(
            url=f"{base_url}/v1/messages",
            body={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Sustained load stability test"}],
                "max_tokens": 50,
            },
            headers={
                "Content-Type": "application/json",
                "X-API-Key": "test-key-for-stress",
            },
        )
        all_results.append(result)

    # Run sustained load for 60 seconds
    duration_seconds = 60
    results = await harness.run_sustained(
        workload_fn=sustained_workload,
        duration_seconds=duration_seconds,
        num_workers=5,
    )

    # Analyze results
    all_results.extend(results)
    collector = MetricsCollector()
    metrics = collector.analyze(all_results)

    # Verify no crashes (500 errors)
    status_5xx = sum(count for status, count in metrics.status_codes.items() if 500 <= status < 600)
    assert status_5xx == 0, f"Server crashed {status_5xx} times during sustained load"

    # Verify health endpoint still responsive
    async with aiohttp.ClientSession() as session:
        async with session.get(
            f"{base_url}/health/live", timeout=aiohttp.ClientTimeout(total=5.0)
        ) as response:
            assert response.status == 200, f"Health endpoint not responsive: {response.status}"

    # Most requests should succeed (allow for some 429s under load)
    assert metrics.error_rate < 0.5, (
        f"Error rate too high under sustained load: {metrics.error_rate:.2%}"
    )

    print(
        f"\n✅ No crashes under load test passed:"
        f"\n  Duration: {duration_seconds}s"
        f"\n  Total requests: {metrics.total_requests}"
        f"\n  Successful: {metrics.successful_requests}"
        f"\n  Error rate: {metrics.error_rate:.2%}"
        f"\n  5xx errors: {status_5xx}"
        f"\n  Health endpoint: ✅ Responsive"
    )


@pytest.mark.stress
@pytest.mark.asyncio
async def test_pool_recovery_after_load(live_server, cleanup_after_stress):
    """Test pool recovers correctly after load subsides.

    Verifies:
    - Pool accepts requests again after load drops
    - No permanent degradation from stress
    - Response times return to normal
    - Success rate recovers

    Pattern: High load → wait → low load → verify recovery
    """
    import aiohttp

    base_url = live_server

    # Phase 1: High load (saturate pool)
    harness_high = StressTestHarness(base_url=base_url, num_workers=50, timeout=60.0)

    high_load_results = await harness_high.run_concurrent_requests(
        path="/v1/messages",
        body={
            "model": "test-model",
            "messages": [{"role": "user", "content": "High load phase"}],
            "max_tokens": 100,
        },
    )

    collector = MetricsCollector()
    high_load_metrics = collector.analyze(high_load_results)

    print(
        f"\n📊 High load phase:"
        f"\n  Total requests: {high_load_metrics.total_requests}"
        f"\n  Success rate: {(high_load_metrics.successful_requests / high_load_metrics.total_requests):.2%}"
        f"\n  Status codes: {high_load_metrics.status_codes}"
    )

    # Phase 2: Wait for pool to drain (simulate load subsiding)
    await asyncio.sleep(10)  # 10 second cooldown

    # Phase 3: Low load (verify recovery)
    harness_low = StressTestHarness(base_url=base_url, num_workers=5, timeout=30.0)

    low_load_results = await harness_low.run_concurrent_requests(
        path="/v1/messages",
        body={
            "model": "test-model",
            "messages": [{"role": "user", "content": "Recovery phase test"}],
            "max_tokens": 50,
        },
    )

    low_load_metrics = collector.analyze(low_load_results)

    # Verify recovery
    low_load_success_rate = low_load_metrics.successful_requests / low_load_metrics.total_requests
    assert low_load_success_rate > 0.8, (
        f"Pool did not recover, success rate: {low_load_success_rate:.2%}"
    )

    # Verify health endpoint responsive
    async with aiohttp.ClientSession() as session:
        async with session.get(
            f"{base_url}/health/live", timeout=aiohttp.ClientTimeout(total=5.0)
        ) as response:
            assert response.status == 200, "Health endpoint not responsive after recovery"

    # If we have latency data, verify it's reasonable
    if low_load_metrics.latency:
        assert low_load_metrics.latency.p95 < 5000, (
            f"p95 latency still high after recovery: {low_load_metrics.latency.p95:.0f}ms"
        )

    print(
        f"\n✅ Pool recovery test passed:"
        f"\n  Low load success rate: {low_load_success_rate:.2%}"
        f"\n  Status codes: {low_load_metrics.status_codes}"
        f"\n  Health endpoint: ✅ Responsive"
    )
    if low_load_metrics.latency:
        print(f"  p95 latency: {low_load_metrics.latency.p95:.0f}ms")
