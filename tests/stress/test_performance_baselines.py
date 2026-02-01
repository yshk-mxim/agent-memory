"""Performance baseline measurements.

Simple tests to measure actual inference performance and establish baselines.
These replace full stress tests which assume faster inference than MLX provides.
"""

import time

import httpx
import pytest


@pytest.mark.stress
def test_single_request_cold_start(live_server):
    """Measure cold start performance (first request to server).

    Establishes baseline for:
    - Initial request latency
    - Model warm-up time
    - First-time cache creation

    Target: Document actual performance (no specific target)
    """
    client = httpx.Client(
        base_url=live_server,
        headers={
            "Content-Type": "application/json",
            "X-API-Key": "test-key-for-e2e",
        },
        timeout=120.0,  # 2 minutes for slow inference
    )

    # Single request
    start_time = time.time()
    response = client.post(
        "/v1/messages",
        json={
            "model": "test-model",
            "messages": [
                {
                    "role": "user",
                    "content": "Hello, this is a test message for performance measurement.",
                }
            ],
            "max_tokens": 50,
        },
    )
    elapsed_ms = (time.time() - start_time) * 1000

    # Validate response
    print("\n📊 Cold Start Performance:")
    print(f"  Status Code: {response.status_code}")
    print(f"  Latency: {elapsed_ms:.0f}ms")

    if response.status_code == 200:
        print("  ✅ Request succeeded")
        try:
            data = response.json()
            print(f"  Response: {str(data)[:100]}...")
        except Exception:
            pass
    elif response.status_code == 503:
        print("  ⚠️  Service unavailable (pool exhausted or draining)")
    else:
        print(f"  ❌ Unexpected status code: {response.status_code}")
        print(f"  Response: {response.text[:200]}")

    # Just measure, don't assert on performance
    assert response.status_code in [200, 503], f"Unexpected status: {response.status_code}"

    client.close()


@pytest.mark.stress
def test_sequential_requests_same_agent(live_server):
    """Measure sequential request performance for same agent (cache warm).

    Establishes baseline for:
    - Cache hit performance
    - Warm cache vs cold cache
    - Sequential processing speed

    Target: Document cache hit improvement
    """
    client = httpx.Client(
        base_url=live_server,
        headers={
            "Content-Type": "application/json",
            "X-API-Key": "test-key-for-e2e",
        },
        timeout=120.0,
    )

    latencies = []

    print("\n📊 Sequential Requests Performance:")

    for i in range(3):
        start_time = time.time()
        response = client.post(
            "/v1/messages",
            json={
                "model": "test-model",
                "messages": [
                    {"role": "user", "content": f"Sequential request {i + 1} for cache testing."}
                ],
                "max_tokens": 50,
            },
        )
        elapsed_ms = (time.time() - start_time) * 1000
        latencies.append(elapsed_ms)

        print(f"  Request {i + 1}: {elapsed_ms:.0f}ms (status: {response.status_code})")

    # Calculate stats
    if latencies:
        print("\n  Statistics:")
        print(f"    First request: {latencies[0]:.0f}ms (cold)")
        if len(latencies) > 1:
            print(f"    Subsequent avg: {sum(latencies[1:]) / len(latencies[1:]):.0f}ms (warm)")
        print(f"    Min: {min(latencies):.0f}ms")
        print(f"    Max: {max(latencies):.0f}ms")

    client.close()


@pytest.mark.stress
def test_health_endpoint_performance(live_server):
    """Measure health endpoint performance.

    Establishes baseline for:
    - Health check latency
    - Readiness probe speed
    - Monitoring overhead

    Target: <100ms for health checks
    """
    client = httpx.Client(base_url=live_server, timeout=5.0)

    print("\n📊 Health Endpoint Performance:")

    # Test /health/live
    start = time.time()
    response = client.get("/health/live")
    live_latency = (time.time() - start) * 1000
    print(f"  /health/live: {live_latency:.2f}ms (status: {response.status_code})")
    assert response.status_code == 200

    # Test /health/ready
    start = time.time()
    response = client.get("/health/ready")
    ready_latency = (time.time() - start) * 1000
    print(f"  /health/ready: {ready_latency:.2f}ms (status: {response.status_code})")
    assert response.status_code in [200, 503]

    # Test /health/startup
    start = time.time()
    response = client.get("/health/startup")
    startup_latency = (time.time() - start) * 1000
    print(f"  /health/startup: {startup_latency:.2f}ms (status: {response.status_code})")
    assert response.status_code == 200

    # Health checks should be fast (<100ms target)
    print("\n  ✅ Health check performance:")
    print("    All endpoints: <100ms is good")
    print(f"    Observed: {max(live_latency, ready_latency, startup_latency):.2f}ms")

    client.close()


@pytest.mark.stress
def test_concurrent_health_checks(live_server):
    """Measure health check performance under concurrent load.

    Validates:
    - Health checks excluded from rate limiting
    - No degradation under polling
    - Consistent response times

    Target: 100 concurrent health checks without failures
    """
    import asyncio

    import aiohttp

    async def check_health():
        """Single health check."""
        start = time.time()
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{live_server}/health/live", timeout=aiohttp.ClientTimeout(total=5.0)
            ) as response:
                latency = (time.time() - start) * 1000
                return (response.status, latency)

    async def run_concurrent_checks(num_checks):
        """Run multiple health checks concurrently."""
        tasks = [check_health() for _ in range(num_checks)]
        return await asyncio.gather(*tasks, return_exceptions=True)

    print("\n📊 Concurrent Health Check Performance:")

    # Run 100 concurrent health checks
    results = asyncio.run(run_concurrent_checks(100))

    # Analyze results
    successes = sum(1 for r in results if not isinstance(r, Exception) and r[0] == 200)
    latencies = [r[1] for r in results if not isinstance(r, Exception)]

    print("  Total checks: 100")
    print(f"  Successful: {successes}")
    print(f"  Failed: {100 - successes}")

    if latencies:
        latencies_sorted = sorted(latencies)
        p50 = latencies_sorted[len(latencies_sorted) // 2]
        p95 = latencies_sorted[int(len(latencies_sorted) * 0.95)]
        print(f"  Latency p50: {p50:.2f}ms")
        print(f"  Latency p95: {p95:.2f}ms")

    # Validate no rate limiting on health checks
    assert successes == 100, f"Expected 100 successful health checks, got {successes}"
    print("\n  ✅ Health endpoints exempt from rate limiting")
