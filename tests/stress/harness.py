"""Stress test harness for managing concurrent load tests (Sprint 6 Day 3).

Provides StressTestHarness class for:
- Concurrent request execution (100+ workers)
- Rate limiting and backoff
- Agent workload simulation
- Sustained load testing
- Metrics collection
"""

import asyncio
import time
from typing import Any, Awaitable, Callable

import aiohttp

from tests.stress.conftest import RequestResult


class StressTestHarness:
    """Harness for executing stress tests with concurrent workers."""

    def __init__(
        self,
        base_url: str,
        num_workers: int = 10,
        rate_limit: float | None = None,
        timeout: float = 30.0,
    ):
        """Initialize stress test harness.

        Args:
            base_url: Server base URL (e.g., "http://localhost:8000")
            num_workers: Number of concurrent workers
            rate_limit: Optional requests per second limit (None = no limit)
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.num_workers = num_workers
        self.rate_limit = rate_limit
        self.timeout = timeout

        # Rate limiting state
        self._last_request_time = 0.0
        self._rate_limit_lock = asyncio.Lock()

    async def run_concurrent_requests(
        self,
        path: str,
        body: dict[str, Any],
        num_requests: int | None = None,
        headers: dict[str, str] | None = None,
    ) -> list[RequestResult]:
        """Execute concurrent HTTP requests.

        Args:
            path: API endpoint path (e.g., "/v1/messages")
            body: Request body (JSON)
            num_requests: Total requests to make (default: num_workers)
            headers: Optional HTTP headers

        Returns:
            List of RequestResult objects with timing and status
        """
        total_requests = num_requests or self.num_workers
        url = f"{self.base_url}{path}"

        # Default headers
        if headers is None:
            headers = {
                "Content-Type": "application/json",
                "X-API-Key": "test-key-for-stress",
            }

        # Create shared session for all requests (more efficient than per-request sessions)
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            # Create tasks for concurrent execution
            tasks = [
                self._make_request_with_session(session, url, body, headers)
                for _ in range(total_requests)
            ]

            # Execute all tasks concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)

        # Convert exceptions to RequestResult objects
        processed_results = []
        for result in results:
            if isinstance(result, RequestResult):
                processed_results.append(result)
            elif isinstance(result, Exception):
                processed_results.append(
                    RequestResult(
                        status_code=0,
                        latency_ms=0.0,
                        timestamp=time.time(),
                        error=result,
                    )
                )

        return processed_results

    async def run_agent_workloads(
        self,
        agent_fn: Callable[[str], Awaitable[list[RequestResult]]],
        num_agents: int,
    ) -> dict[str, list[RequestResult]]:
        """Run parallel agent workloads.

        Args:
            agent_fn: Async function that executes agent workload
                      Takes agent_id (str), returns list of RequestResult
            num_agents: Number of concurrent agents

        Returns:
            Dict mapping agent_id to list of RequestResult objects
        """
        # Create agent tasks
        agent_ids = [f"agent_{i}" for i in range(num_agents)]
        tasks = [agent_fn(agent_id) for agent_id in agent_ids]

        # Execute all agents concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Map results to agent IDs
        agent_results: dict[str, list[RequestResult]] = {}
        for agent_id, result in zip(agent_ids, results):
            if isinstance(result, list):
                agent_results[agent_id] = result
            elif isinstance(result, Exception):
                # Agent failed entirely
                agent_results[agent_id] = [
                    RequestResult(
                        status_code=0,
                        latency_ms=0.0,
                        timestamp=time.time(),
                        error=result,
                    )
                ]

        return agent_results

    async def run_sustained(
        self,
        workload_fn: Callable[[], Awaitable[None]],
        duration_seconds: int,
        num_workers: int | None = None,
    ) -> list[RequestResult]:
        """Run sustained load test.

        Args:
            workload_fn: Async workload function (runs continuously)
            duration_seconds: Test duration in seconds
            num_workers: Number of concurrent workers (default: self.num_workers)

        Returns:
            List of all RequestResult objects from sustained test
        """
        workers = num_workers or self.num_workers
        results: list[RequestResult] = []

        # Create worker tasks
        tasks = [
            self._sustained_worker(workload_fn, duration_seconds, results)
            for _ in range(workers)
        ]

        # Run all workers until duration expires
        await asyncio.gather(*tasks, return_exceptions=True)

        return results

    async def run_bursty_load(
        self,
        path: str,
        body: dict[str, Any],
        burst_size: int,
        burst_interval_seconds: float,
        num_bursts: int,
        headers: dict[str, str] | None = None,
    ) -> list[RequestResult]:
        """Run bursty load pattern (simulate bursty traffic).

        Args:
            path: API endpoint path
            body: Request body
            burst_size: Number of requests per burst
            burst_interval_seconds: Time between bursts
            num_bursts: Total number of bursts
            headers: Optional HTTP headers

        Returns:
            List of RequestResult objects from all bursts
        """
        all_results: list[RequestResult] = []

        for burst_num in range(num_bursts):
            # Execute burst
            burst_results = await self.run_concurrent_requests(
                path=path, body=body, num_requests=burst_size, headers=headers
            )
            all_results.extend(burst_results)

            # Wait before next burst (unless this is the last burst)
            if burst_num < num_bursts - 1:
                await asyncio.sleep(burst_interval_seconds)

        return all_results

    async def _make_request_with_session(
        self,
        session: aiohttp.ClientSession,
        url: str,
        body: dict[str, Any],
        headers: dict[str, str],
    ) -> RequestResult:
        """Make a single HTTP request with timing using shared session.

        Args:
            session: Shared aiohttp.ClientSession
            url: Full URL
            body: Request body
            headers: HTTP headers

        Returns:
            RequestResult with timing and status
        """
        # Apply rate limiting if configured
        if self.rate_limit is not None:
            await self._apply_rate_limit()

        start_time = time.time()
        timestamp = start_time

        try:
            async with session.post(url, json=body, headers=headers) as response:
                status_code = response.status
                response_body = None

                try:
                    response_body = await response.json()
                except Exception:
                    # Response might not be JSON
                    pass

                latency_ms = (time.time() - start_time) * 1000

                return RequestResult(
                    status_code=status_code,
                    latency_ms=latency_ms,
                    timestamp=timestamp,
                    response_body=response_body,
                    error=None,
                )

        except asyncio.TimeoutError as e:
            return RequestResult(
                status_code=0,
                latency_ms=(time.time() - start_time) * 1000,
                timestamp=timestamp,
                error=e,
            )
        except Exception as e:
            return RequestResult(
                status_code=0,
                latency_ms=(time.time() - start_time) * 1000,
                timestamp=timestamp,
                error=e,
            )

    async def _make_request(
        self, url: str, body: dict[str, Any], headers: dict[str, str]
    ) -> RequestResult:
        """Make a single HTTP request with timing (creates own session).

        DEPRECATED: Use _make_request_with_session for better performance with concurrent requests.

        Args:
            url: Full URL
            body: Request body
            headers: HTTP headers

        Returns:
            RequestResult with timing and status
        """
        # Create session just for this request (less efficient, kept for backward compat)
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            return await self._make_request_with_session(session, url, body, headers)

    async def _apply_rate_limit(self) -> None:
        """Apply rate limiting to requests."""
        if self.rate_limit is None:
            return

        async with self._rate_limit_lock:
            now = time.time()
            time_since_last = now - self._last_request_time

            # Calculate required delay
            min_interval = 1.0 / self.rate_limit
            if time_since_last < min_interval:
                delay = min_interval - time_since_last
                await asyncio.sleep(delay)

            self._last_request_time = time.time()

    async def _sustained_worker(
        self,
        workload_fn: Callable[[], Awaitable[None]],
        duration_seconds: int,
        results: list[RequestResult],
    ) -> None:
        """Worker for sustained load testing.

        Args:
            workload_fn: Async workload function
            duration_seconds: Duration to run
            results: List to append results to
        """
        start_time = time.time()

        while (time.time() - start_time) < duration_seconds:
            try:
                await workload_fn()
            except Exception as e:
                # Log error but continue
                results.append(
                    RequestResult(
                        status_code=0,
                        latency_ms=0.0,
                        timestamp=time.time(),
                        error=e,
                    )
                )

    async def measure_pool_utilization(
        self, health_url: str | None = None
    ) -> float:
        """Measure current pool utilization from health endpoint.

        Args:
            health_url: Health endpoint URL (default: base_url/health)

        Returns:
            Pool utilization as percentage (0.0-1.0), or -1.0 if unavailable
        """
        url = health_url or f"{self.base_url}/health"

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        # Assume health endpoint returns pool stats
                        # Adjust based on actual API
                        if "pool" in data:
                            used = data["pool"].get("used_blocks", 0)
                            total = data["pool"].get("total_blocks", 1)
                            return used / total if total > 0 else 0.0
        except Exception:
            pass

        return -1.0  # Unavailable


class RampUpHarness(StressTestHarness):
    """Stress test harness with gradual ramp-up of load.

    Useful for testing behavior at different load levels (80%, 90%, 100%).
    """

    async def run_ramp_up(
        self,
        path: str,
        body: dict[str, Any],
        start_workers: int,
        end_workers: int,
        ramp_duration_seconds: int,
        headers: dict[str, str] | None = None,
    ) -> dict[int, list[RequestResult]]:
        """Run load test with gradual ramp-up.

        Args:
            path: API endpoint path
            body: Request body
            start_workers: Initial number of workers
            end_workers: Final number of workers
            ramp_duration_seconds: Duration to ramp from start to end
            headers: Optional HTTP headers

        Returns:
            Dict mapping worker_count to list of RequestResult objects
        """
        results_by_level: dict[int, list[RequestResult]] = {}

        # Calculate ramp steps
        num_steps = min(10, end_workers - start_workers)  # Max 10 steps
        step_size = (end_workers - start_workers) / num_steps
        step_duration = ramp_duration_seconds / num_steps

        for step in range(num_steps + 1):
            worker_count = int(start_workers + step * step_size)

            # Execute load at this level
            step_results = await self.run_concurrent_requests(
                path=path, body=body, num_requests=worker_count, headers=headers
            )

            results_by_level[worker_count] = step_results

            # Wait before next step (unless last step)
            if step < num_steps:
                await asyncio.sleep(step_duration)

        return results_by_level
