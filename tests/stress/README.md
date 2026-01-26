# Stress Testing Framework

**Purpose**: Validate system behavior under high load, concurrent requests, and sustained operation.

This directory contains stress tests that push the semantic caching server to its limits, validating:
- Pool exhaustion handling (graceful 429 responses)
- Concurrent multi-agent scenarios (10+ agents, 50+ rapid requests)
- Sustained load over time (1-hour stability tests)
- Memory stability (no leaks during sustained operation)
- Performance degradation analysis

---

## Test Categories

### 1. Pool Exhaustion Tests (`test_pool_exhaustion.py`)

**Goal**: Validate graceful degradation when block pool approaches/reaches capacity.

Tests:
- `test_100_plus_concurrent_requests` - Validates server handles >100 concurrent requests
- `test_graceful_429_when_pool_exhausted` - Returns 429 (Too Many Requests) when pool full
- `test_no_crashes_under_load` - Server remains stable under extreme load
- `test_pool_recovery_after_load` - Pool correctly recovers after load subsides

**Pattern**:
```python
@pytest.mark.stress
async def test_pool_exhaustion(stress_client, metrics_collector):
    harness = StressTestHarness(
        base_url="http://localhost:8000",
        num_workers=150,
        requests_per_worker=1
    )

    results = await harness.run_concurrent_requests(
        path="/v1/messages",
        body={"model": "test-model", "messages": [...], "max_tokens": 100}
    )

    metrics = metrics_collector.analyze(results)
    assert metrics.error_rate < 0.05  # <5% errors
    assert 429 in metrics.status_codes  # Graceful degradation
```

### 2. Concurrent Agent Stress (`test_concurrent_agents.py`)

**Goal**: Validate multi-agent scenarios with rapid, concurrent requests.

Tests:
- `test_10_agents_50_rapid_requests` - 10 agents × 50 requests each (500 total)
- `test_cache_isolation_under_load` - No cache leakage under stress
- `test_latency_remains_acceptable` - p95 latency <2s under load
- `test_cache_hit_rate_high` - >80% cache hits in multi-turn conversations

**Pattern**:
```python
@pytest.mark.stress
async def test_concurrent_agents(stress_client, metrics_collector):
    num_agents = 10
    requests_per_agent = 50

    harness = StressTestHarness(
        base_url="http://localhost:8000",
        num_workers=num_agents
    )

    async def agent_workload(agent_id: str):
        results = []
        for i in range(requests_per_agent):
            response = await stress_client.post(
                "/v1/messages",
                json={
                    "model": "test-model",
                    "messages": [{"role": "user", "content": f"Agent {agent_id} request {i}"}],
                    "max_tokens": 50,
                }
            )
            results.append(response)
        return results

    all_results = await harness.run_agent_workloads(agent_workload, num_agents)

    metrics = metrics_collector.analyze_latency(all_results)
    assert metrics.p95_latency < 2000  # <2s p95 latency
```

### 3. Sustained Load Tests (`test_sustained_load.py`)

**Goal**: Validate stability and performance over extended periods (1 hour).

Tests:
- `test_one_hour_sustained_load` - 1 hour of continuous operation
- `test_10_requests_per_minute_across_5_agents` - Realistic sustained traffic
- `test_memory_stable_no_leaks` - Memory growth <5% over 1 hour
- `test_no_performance_degradation_over_time` - Latency stable over time

**Pattern**:
```python
@pytest.mark.stress
@pytest.mark.slow
async def test_sustained_load(stress_client, metrics_collector):
    duration_seconds = 3600  # 1 hour
    requests_per_minute = 10
    num_agents = 5

    harness = StressTestHarness(
        base_url="http://localhost:8000",
        num_workers=num_agents,
        rate_limit=requests_per_minute / 60  # requests per second
    )

    memory_samples = []
    start_time = time.time()

    async def sustained_workload():
        while time.time() - start_time < duration_seconds:
            # Make request
            response = await stress_client.post(...)

            # Sample memory every 5 minutes
            if int(time.time() - start_time) % 300 == 0:
                memory_samples.append(get_server_memory_usage())

            await asyncio.sleep(60 / requests_per_minute)

    await harness.run_sustained(sustained_workload, duration_seconds)

    # Validate memory stability
    memory_growth = (memory_samples[-1] - memory_samples[0]) / memory_samples[0]
    assert memory_growth < 0.05  # <5% growth
```

### 4. Realistic Conversations (`test_realistic_conversations.py`)

**Goal**: Simulate real-world usage patterns with bursty traffic and varying request sizes.

Test:
- `test_realistic_conversation_patterns` - 5 agents with different usage patterns

**Patterns**:
- Agent 1: Short queries (10-50 tokens) - frequent
- Agent 2: Long context (500-1000 tokens) - occasional
- Agent 3: Rapid-fire questions - bursty
- Agent 4-5: Intermittent usage - sporadic

---

## Fixtures (in `conftest.py`)

### `stress_client`

Async HTTP client using `aiohttp.ClientSession` for concurrent requests.

```python
@pytest.fixture
async def stress_client():
    """Async HTTP client for stress testing."""
    async with aiohttp.ClientSession() as session:
        yield session
```

### `metrics_collector`

Collects and analyzes test metrics (latency, errors, throughput).

```python
@pytest.fixture
def metrics_collector():
    """Metrics collection and analysis."""
    return MetricsCollector()
```

**Collected Metrics**:
- Latency: p50, p95, p99, max
- Throughput: requests/sec, tokens/sec
- Error rate: 4xx, 5xx status codes
- Status code distribution
- Cache hit rate (if available)

### `cleanup_after_stress`

Resets server state after stress tests.

```python
@pytest.fixture
async def cleanup_after_stress():
    """Clean up server state after stress tests."""
    yield
    # Reset pools, clear caches, etc.
```

---

## StressTestHarness Class

**File**: `harness.py`

**Purpose**: Manage concurrent request execution, rate limiting, and metrics collection.

**Key Methods**:

```python
class StressTestHarness:
    def __init__(
        self,
        base_url: str,
        num_workers: int,
        rate_limit: float | None = None,
        timeout: float = 30.0
    ):
        """Initialize stress test harness.

        Args:
            base_url: Server base URL
            num_workers: Number of concurrent workers
            rate_limit: Optional requests per second limit
            timeout: Request timeout in seconds
        """

    async def run_concurrent_requests(
        self,
        path: str,
        body: dict,
        num_requests: int | None = None
    ) -> list[RequestResult]:
        """Execute concurrent requests.

        Args:
            path: API endpoint path
            body: Request body
            num_requests: Total requests (default: num_workers)

        Returns:
            List of RequestResult objects with timing and status
        """

    async def run_agent_workloads(
        self,
        agent_fn: Callable[[str], Awaitable[list]],
        num_agents: int
    ) -> dict[str, list]:
        """Run parallel agent workloads.

        Args:
            agent_fn: Async function that executes agent workload
            num_agents: Number of concurrent agents

        Returns:
            Dict mapping agent_id to list of results
        """

    async def run_sustained(
        self,
        workload_fn: Callable[[], Awaitable[None]],
        duration_seconds: int,
        num_workers: int | None = None
    ) -> SustainedLoadMetrics:
        """Run sustained load test.

        Args:
            workload_fn: Async workload function
            duration_seconds: Test duration
            num_workers: Number of concurrent workers

        Returns:
            Metrics from sustained load test
        """
```

**Data Classes**:

```python
@dataclass
class RequestResult:
    """Single request result."""
    status_code: int
    latency_ms: float
    timestamp: float
    error: Exception | None = None

@dataclass
class SustainedLoadMetrics:
    """Metrics from sustained load test."""
    total_requests: int
    successful_requests: int
    error_rate: float
    avg_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    throughput_rps: float
    memory_samples: list[float]
```

---

## Running Stress Tests

### Run All Stress Tests

```bash
pytest tests/stress/ -v -m stress
```

### Run Specific Test Suite

```bash
pytest tests/stress/test_pool_exhaustion.py -v
pytest tests/stress/test_concurrent_agents.py -v
pytest tests/stress/test_sustained_load.py -v
```

### Skip Slow Tests (1-hour sustained load)

```bash
pytest tests/stress/ -v -m "stress and not slow"
```

### Run Only Fast Stress Tests

```bash
pytest tests/stress/test_pool_exhaustion.py -v
pytest tests/stress/test_concurrent_agents.py -v
```

---

## Configuration

### Environment Variables

- `SEMANTIC_STRESS_TEST_DURATION`: Override sustained load duration (default: 3600s)
- `SEMANTIC_STRESS_TEST_WORKERS`: Override concurrent workers (default: varies by test)
- `SEMANTIC_STRESS_TEST_RATE_LIMIT`: Override rate limit (requests/sec)

### Pytest Markers

```python
# Mark test as stress test
@pytest.mark.stress

# Mark test as slow (>5 minutes)
@pytest.mark.slow

# Mark test as requiring high resources
@pytest.mark.resource_intensive
```

### CI Configuration

In `pyproject.toml`:

```toml
[tool.pytest.ini_options]
markers = [
    "stress: Stress tests (high load, concurrency)",
    "slow: Slow tests (>5 minutes)",
    "resource_intensive: Tests requiring significant resources"
]

# Skip stress tests in CI by default
addopts = "-m 'not stress'"
```

---

## Interpreting Results

### Successful Test Criteria

**Pool Exhaustion Tests**:
- ✅ Server returns 429 when pool >95% utilized
- ✅ No crashes or 500 errors
- ✅ Pool recovers after load subsides
- ✅ Error rate <5% overall

**Concurrent Agent Tests**:
- ✅ All 10 agents complete all 50 requests
- ✅ No cache leakage between agents
- ✅ p95 latency <2s under load
- ✅ Cache hit rate >80% (multi-turn)

**Sustained Load Tests**:
- ✅ 1 hour of stable operation
- ✅ Memory growth <5%
- ✅ No performance degradation over time
- ✅ Error rate <1%

**Realistic Conversation Tests**:
- ✅ All agents complete workloads
- ✅ Mixed latency patterns acceptable
- ✅ No deadlocks or hangs
- ✅ Cache behavior realistic

### Common Issues

**High Error Rates**:
- Check pool sizing (`SEMANTIC_CACHE_BUDGET_MB`)
- Verify rate limiting configuration
- Check for resource exhaustion (CPU, memory)

**Memory Growth**:
- Profile with `memory_profiler`
- Check cache eviction policies
- Look for reference cycles

**Performance Degradation**:
- Check for lock contention
- Profile with `py-spy`
- Verify batch engine efficiency

**Timeouts**:
- Increase request timeout
- Check model loading time
- Verify server capacity

---

## Adding New Stress Tests

### Template

```python
import pytest
from tests.stress.harness import StressTestHarness, RequestResult

@pytest.mark.stress
async def test_your_stress_scenario(stress_client, metrics_collector):
    """Test description.

    Verifies:
    - What you're testing
    - Expected behavior
    - Performance targets
    """
    # Setup
    harness = StressTestHarness(
        base_url="http://localhost:8000",
        num_workers=10
    )

    # Execute
    results = await harness.run_concurrent_requests(
        path="/v1/messages",
        body={"model": "test-model", "messages": [...], "max_tokens": 100}
    )

    # Analyze
    metrics = metrics_collector.analyze(results)

    # Assert
    assert metrics.error_rate < 0.05
    assert metrics.p95_latency < 2000
```

### Best Practices

1. **Use realistic request patterns** - Don't just hammer with identical requests
2. **Measure actual server state** - Check pool utilization, memory, cache state
3. **Test graceful degradation** - Validate behavior at 80%, 90%, 100% capacity
4. **Clean up after tests** - Reset server state, clear caches
5. **Document performance targets** - Specify expected latency, error rates
6. **Use appropriate timeouts** - Allow for model loading, batch processing
7. **Profile memory carefully** - Sample at regular intervals during sustained tests
8. **Test recovery** - Validate server returns to normal after stress

---

## Performance Targets (Sprint 6)

### Pool Exhaustion
- Graceful 429 at >95% pool utilization: ✅ Required
- No crashes under 100+ concurrent requests: ✅ Required
- Pool recovery after load: <30s

### Concurrent Agents
- 10 agents × 50 requests: All complete successfully
- p95 latency under load: <2s
- Cache hit rate (multi-turn): >80%

### Sustained Load
- Duration: 1 hour stable operation
- Memory growth: <5%
- Performance: No degradation over time
- Error rate: <1%

### Realistic Conversations
- Mixed agent patterns: All complete
- Latency distribution: Reasonable for workload
- Cache behavior: Realistic hit rates

---

**Last Updated**: 2026-01-25 (Sprint 6 Day 3)
**Framework Version**: 1.0.0
**Dependencies**: aiohttp, pytest-asyncio, psutil
