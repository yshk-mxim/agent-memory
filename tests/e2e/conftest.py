"""E2E test fixtures for full-stack testing.

This module provides fixtures for end-to-end testing with real components:
- live_server: Starts FastAPI server in subprocess
- test_client: HTTP client for making requests
- cleanup_caches: Cleans up test cache directories
"""

import os
import socket
import subprocess
import time
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest


def find_free_port() -> int:
    """Find a free port for the test server."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


@pytest.fixture(scope="function")
def cleanup_caches() -> Iterator[None]:
    """Clean up test cache directories before and after tests.

    Ensures test isolation by removing cached data.
    """
    test_cache_dir = Path.home() / ".cache" / "agent_memory" / "test"

    # Cleanup before test
    if test_cache_dir.exists():
        import shutil

        shutil.rmtree(test_cache_dir)

    yield

    # Cleanup after test
    if test_cache_dir.exists():
        import shutil

        shutil.rmtree(test_cache_dir)


@pytest.fixture(scope="function")
def live_server(cleanup_caches) -> Iterator[str]:
    """Start a live FastAPI server for E2E testing.

    Starts the server in a subprocess, waits for it to be ready,
    yields the server URL, and tears down on completion.

    Returns:
        Server URL (e.g., "http://localhost:8001")

    Example:
        def test_server_health(live_server):
            response = httpx.get(f"{live_server}/health")
            assert response.status_code == 200
    """
    # Find free port
    port = find_free_port()
    server_url = f"http://localhost:{port}"

    # Set test environment variables
    env = os.environ.copy()
    env["SEMANTIC_CACHE_DIR"] = str(Path.home() / ".cache" / "agent_memory" / "test")
    env["SEMANTIC_LOG_LEVEL"] = "WARNING"  # Reduce noise in tests
    # Support all test API keys (comma-separated for multiple agents)
    test_keys = [
        "test-key-for-e2e",
        "test-key-for-stress",
        "test-key-sustained",
        "test-key-memory",
        "test-key-perf",
    ]
    # Add agent-specific keys (agent_0 through agent_9)
    test_keys.extend([f"test-key-agent_{i}" for i in range(10)])
    env["ANTHROPIC_API_KEY"] = ",".join(test_keys)

    # Create log files for debugging
    log_dir = Path("/tmp/claude") / "e2e_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_log = log_dir / f"server_{port}_stdout.log"
    stderr_log = log_dir / f"server_{port}_stderr.log"

    # Open log files (will be closed in finally block)
    stdout_file = open(stdout_log, "w")
    stderr_file = open(stderr_log, "w")

    # Start server in subprocess
    process = subprocess.Popen(
        [
            "python",
            "-m",
            "uvicorn",
            "agent_memory.entrypoints.api_server:create_app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--factory",
        ],
        env=env,
        stdout=stdout_file,
        stderr=stderr_file,
        text=True,
        close_fds=False,  # Don't close log file descriptors
    )

    # Wait for server to start (poll health endpoint)
    start_time = time.time()
    timeout = 60  # 60 seconds for model loading
    server_ready = False

    while time.time() - start_time < timeout:
        try:
            response = httpx.get(f"{server_url}/health/live", timeout=2.0)
            if response.status_code == 200:
                server_ready = True
                break
        except (httpx.ConnectError, httpx.TimeoutException):
            time.sleep(0.5)

    if not server_ready:
        # Server failed to start - read logs
        process.kill()
        process.wait()
        stdout_file.close()
        stderr_file.close()

        # Read log files
        stdout_content = stdout_log.read_text() if stdout_log.exists() else ""
        stderr_content = stderr_log.read_text() if stderr_log.exists() else ""

        raise RuntimeError(
            f"Server failed to start within {timeout}s\n"
            f"Logs saved to: {log_dir}\n"
            f"STDOUT:\n{stdout_content}\n"
            f"STDERR:\n{stderr_content}"
        )

    # Yield server URL to test
    try:
        yield server_url
    finally:
        # Teardown: stop server and clean up resources
        # Close log files to prevent resource warnings
        try:
            stdout_file.close()
        except OSError:
            pass
        try:
            stderr_file.close()
        except OSError:
            pass

        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


@pytest.fixture(scope="module")
def server_ready() -> None:
    """Skip all tests if the live server at localhost:8000 is not reachable.

    Used by test suites that expect a manually-started server (e.g. test_live_api,
    test_gui_pages) rather than the auto-managed live_server fixture.
    """
    try:
        resp = httpx.get("http://localhost:8000/health/ready", timeout=5.0)
        if resp.status_code != 200:
            pytest.skip("Server not ready at localhost:8000")
    except httpx.HTTPError:
        pytest.skip("Server not reachable at localhost:8000")


@pytest.fixture(scope="function")
def test_client(live_server: str) -> Iterator[httpx.Client]:
    """HTTP client configured for E2E testing.

    Args:
        live_server: Server URL from live_server fixture

    Yields:
        Configured httpx.Client with authentication headers

    Example:
        def test_api_endpoint(test_client):
            response = test_client.post("/v1/messages", json={...})
            assert response.status_code == 200
    """
    client = httpx.Client(
        base_url=live_server,
        headers={
            "Content-Type": "application/json",
            "X-API-Key": "test-key-for-e2e",
        },
        timeout=30.0,  # 30 second timeout for E2E requests
    )
    try:
        yield client
    finally:
        client.close()
