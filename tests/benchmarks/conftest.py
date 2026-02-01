"""Benchmark fixtures for semantic caching server (Sprint 6 Day 5).

Provides fixtures for:
- Benchmark server (module-scoped for performance)
- Benchmark result reporting
- Performance measurement utilities
"""

import json
import subprocess
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest


class BenchmarkReporter:
    """Reporter for benchmark results."""

    def __init__(self, output_file: str | None = None):
        """Initialize benchmark reporter.

        Args:
            output_file: Path to save results (default: benchmark_results.json)
        """
        self.output_file = output_file or "benchmark_results.json"
        self.results: dict[str, Any] = {}
        self.metadata: dict[str, Any] = {
            "timestamp": time.time(),
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

    def record(self, metric_name: str, value: Any) -> None:
        """Record a benchmark result.

        Args:
            metric_name: Name of the metric
            value: Measured value (number, dict, list, etc.)
        """
        self.results[metric_name] = value

    def add_metadata(self, key: str, value: Any) -> None:
        """Add metadata to the report.

        Args:
            key: Metadata key
            value: Metadata value
        """
        self.metadata[key] = value

    def save(self) -> None:
        """Save results to JSON file."""
        output = {"metadata": self.metadata, "results": self.results}

        output_path = Path(self.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)

        print(f"\n📊 Benchmark results saved to: {output_path}")

    def print_summary(self) -> None:
        """Print benchmark summary to console."""
        print("\n" + "=" * 70)
        print("  BENCHMARK RESULTS SUMMARY")
        print("=" * 70)

        for metric_name, value in self.results.items():
            if isinstance(value, dict):
                print(f"\n{metric_name}:")
                for k, v in value.items():
                    print(f"  {k}: {v}")
            elif isinstance(value, (int, float)):
                print(f"{metric_name}: {value}")
            else:
                print(f"{metric_name}: {value}")

        print("=" * 70 + "\n")


@pytest.fixture(scope="module")
def benchmark_server() -> Iterator[str]:
    """Start a live server for benchmarking (module-scoped for performance).

    Starts the server once per test module and shares it across all benchmarks
    in that module for efficiency.

    Yields:
        Server URL (e.g., "http://localhost:8000")
    """
    import os

    # Find free port
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.listen(1)
        port = s.getsockname()[1]

    server_url = f"http://localhost:{port}"

    # Setup environment
    env = os.environ.copy()
    env["SEMANTIC_CACHE_DIR"] = str(Path.home() / ".cache" / "semantic" / "benchmark")
    env["SEMANTIC_LOG_LEVEL"] = "WARNING"
    env["ANTHROPIC_API_KEY"] = "test-key-benchmark"

    # Start server
    process = subprocess.Popen(
        [
            "python",
            "-m",
            "semantic.entrypoints.cli",
            "serve",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        close_fds=True,  # Ensure file descriptors are closed
    )

    # Wait for server to be ready (allow 60s for model loading)
    start_time = time.time()
    timeout = 60
    server_ready = False

    while time.time() - start_time < timeout:
        try:
            response = httpx.get(f"{server_url}/health", timeout=2.0)
            if response.status_code == 200:
                server_ready = True
                print(f"\n✅ Benchmark server ready at {server_url}")
                break
        except (httpx.ConnectError, httpx.TimeoutException):
            time.sleep(0.5)

    if not server_ready:
        stdout, stderr = process.communicate(timeout=5)
        process.kill()
        raise RuntimeError(
            f"Benchmark server failed to start within {timeout}s\n"
            f"STDOUT:\n{stdout}\nSTDERR:\n{stderr}"
        )

    yield server_url

    # Teardown: clean up resources
    # Close pipes to prevent resource warnings
    if process.stdout:
        process.stdout.close()
    if process.stderr:
        process.stderr.close()

    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()

    print("\n✅ Benchmark server shut down")


@pytest.fixture
def benchmark_reporter():
    """Benchmark result reporter.

    Returns:
        BenchmarkReporter instance for recording results
    """
    reporter = BenchmarkReporter()
    yield reporter
    # Optionally auto-save results
    # reporter.save()


@pytest.fixture
def benchmark_client(benchmark_server: str) -> Iterator[httpx.Client]:
    """HTTP client configured for benchmark server.

    Args:
        benchmark_server: Server URL from fixture

    Yields:
        Configured HTTP client
    """
    client = httpx.Client(
        base_url=benchmark_server,
        headers={
            "Content-Type": "application/json",
            "X-API-Key": "test-key-benchmark",
        },
        timeout=60.0,
    )
    try:
        yield client
    finally:
        client.close()


@pytest.fixture(scope="module")
def cleanup_benchmark_caches():
    """Clean up benchmark cache directories before and after module.

    Yields:
        None
    """
    benchmark_cache_dir = Path.home() / ".cache" / "semantic" / "benchmark"

    # Cleanup before
    if benchmark_cache_dir.exists():
        import shutil

        shutil.rmtree(benchmark_cache_dir)

    yield

    # Cleanup after
    if benchmark_cache_dir.exists():
        import shutil

        shutil.rmtree(benchmark_cache_dir)
