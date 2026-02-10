"""Stress test fixtures for agent-memory server.

Provides fixtures for:
- Async HTTP client (aiohttp)
- Metrics collection and analysis
- Server state cleanup after stress tests

Imports E2E fixtures for server lifecycle management.
"""

import asyncio
import statistics
from dataclasses import dataclass, field
from pathlib import Path

import aiohttp
import psutil
import pytest

# Import E2E fixtures for server lifecycle
pytest_plugins = ["tests.e2e.conftest"]


@dataclass
class RequestResult:
    """Result from a single HTTP request."""

    status_code: int
    latency_ms: float
    timestamp: float
    response_body: dict | None = None
    error: Exception | None = None


@dataclass
class LatencyMetrics:
    """Latency statistics from request results."""

    p50: float
    p95: float
    p99: float
    max: float
    mean: float
    min: float
    total_requests: int


@dataclass
class LoadMetrics:
    """Comprehensive load test metrics."""

    total_requests: int
    successful_requests: int
    failed_requests: int
    error_rate: float
    status_codes: dict[int, int] = field(default_factory=dict)
    latency: LatencyMetrics | None = None
    throughput_rps: float = 0.0
    duration_seconds: float = 0.0


class MetricsCollector:
    """Collects and analyzes stress test metrics."""

    def __init__(self):
        """Initialize metrics collector."""
        self.results: list[RequestResult] = []
        self.memory_samples: list[float] = []

    def add_result(self, result: RequestResult) -> None:
        """Add a request result to the collector."""
        self.results.append(result)

    def add_memory_sample(self, memory_mb: float) -> None:
        """Add a memory usage sample."""
        self.memory_samples.append(memory_mb)

    def analyze(self, results: list[RequestResult] | None = None) -> LoadMetrics:
        """Analyze request results and compute metrics.

        Args:
            results: Optional list of results to analyze (default: use collected results)

        Returns:
            LoadMetrics with comprehensive statistics
        """
        data = results if results is not None else self.results

        if not data:
            return LoadMetrics(
                total_requests=0,
                successful_requests=0,
                failed_requests=0,
                error_rate=1.0,
            )

        # Count successes/failures
        total = len(data)
        successful = sum(1 for r in data if 200 <= r.status_code < 300)
        failed = total - successful
        error_rate = failed / total if total > 0 else 0.0

        # Status code distribution
        status_codes: dict[int, int] = {}
        for r in data:
            status_codes[r.status_code] = status_codes.get(r.status_code, 0) + 1

        # Latency analysis
        latencies = [r.latency_ms for r in data if r.error is None]
        latency_metrics = None
        if latencies:
            latencies_sorted = sorted(latencies)
            latency_metrics = LatencyMetrics(
                p50=self._percentile(latencies_sorted, 50),
                p95=self._percentile(latencies_sorted, 95),
                p99=self._percentile(latencies_sorted, 99),
                max=max(latencies),
                mean=statistics.mean(latencies),
                min=min(latencies),
                total_requests=len(latencies),
            )

        # Throughput (if we have timestamps)
        duration = 0.0
        throughput = 0.0
        if data:
            timestamps = [r.timestamp for r in data]
            duration = max(timestamps) - min(timestamps)
            if duration > 0:
                throughput = total / duration

        return LoadMetrics(
            total_requests=total,
            successful_requests=successful,
            failed_requests=failed,
            error_rate=error_rate,
            status_codes=status_codes,
            latency=latency_metrics,
            throughput_rps=throughput,
            duration_seconds=duration,
        )

    def analyze_memory_growth(self) -> dict[str, float]:
        """Analyze memory growth over time.

        Returns:
            Dict with memory statistics (growth percentage, max, min, mean)
        """
        if len(self.memory_samples) < 2:
            return {"growth_pct": 0.0, "max_mb": 0.0, "min_mb": 0.0, "mean_mb": 0.0}

        initial = self.memory_samples[0]
        final = self.memory_samples[-1]
        growth_pct = ((final - initial) / initial) * 100 if initial > 0 else 0.0

        return {
            "growth_pct": growth_pct,
            "max_mb": max(self.memory_samples),
            "min_mb": min(self.memory_samples),
            "mean_mb": statistics.mean(self.memory_samples),
        }

    @staticmethod
    def _percentile(sorted_data: list[float], percentile: int) -> float:
        """Calculate percentile from sorted data.

        Args:
            sorted_data: Sorted list of values
            percentile: Percentile to calculate (0-100)

        Returns:
            Percentile value
        """
        if not sorted_data:
            return 0.0
        index = int((percentile / 100) * len(sorted_data))
        index = min(index, len(sorted_data) - 1)
        return sorted_data[index]


@pytest.fixture
async def stress_client():
    """Async HTTP client for stress testing.

    Yields:
        aiohttp.ClientSession configured for stress testing
    """
    timeout = aiohttp.ClientTimeout(total=60)  # 60s timeout for stress tests
    async with aiohttp.ClientSession(timeout=timeout) as session:
        yield session


@pytest.fixture
def metrics_collector():
    """Metrics collector for stress test analysis.

    Returns:
        MetricsCollector instance for collecting and analyzing metrics
    """
    return MetricsCollector()


@pytest.fixture
async def cleanup_after_stress():
    """Clean up server state after stress tests.

    Runs after each stress test to:
    - Clear test cache directories
    - Reset server state (if possible)
    - Ensure clean state for next test
    """
    yield

    # Cleanup test cache directory
    test_cache_dir = Path.home() / ".cache" / "agent_memory" / "test"
    if test_cache_dir.exists():
        import shutil

        shutil.rmtree(test_cache_dir)


def get_server_memory_usage(process_name: str = "python") -> float:
    """Get current server memory usage in MB.

    Args:
        process_name: Name of the process to monitor (default: "python")

    Returns:
        Memory usage in MB, or 0.0 if process not found
    """
    for proc in psutil.process_iter(["name", "memory_info"]):
        try:
            if proc.info["name"] == process_name:
                return proc.info["memory_info"].rss / 1024 / 1024  # Convert to MB
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return 0.0


@pytest.fixture
def memory_profiler():
    """Memory profiling fixture for sustained load tests.

    Returns:
        Function to sample memory at regular intervals
    """

    async def sample_memory_periodically(
        interval_seconds: int, duration_seconds: int
    ) -> list[float]:
        """Sample server memory at regular intervals.

        Args:
            interval_seconds: Sampling interval
            duration_seconds: Total sampling duration

        Returns:
            List of memory samples in MB
        """
        samples = []
        start_time = asyncio.get_event_loop().time()

        while (asyncio.get_event_loop().time() - start_time) < duration_seconds:
            memory_mb = get_server_memory_usage()
            samples.append(memory_mb)
            await asyncio.sleep(interval_seconds)

        return samples

    return sample_memory_periodically
