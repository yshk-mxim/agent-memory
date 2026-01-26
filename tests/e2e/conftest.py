"""E2E test fixtures for full-stack testing.

This module provides fixtures for end-to-end testing with real components:
- live_server: Starts FastAPI server in subprocess
- test_client: HTTP client for making requests
- cleanup_caches: Cleans up test cache directories
"""

import multiprocessing
import os
import socket
import subprocess
import time
from pathlib import Path
from typing import Iterator

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
    test_cache_dir = Path.home() / ".cache" / "semantic" / "test"

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
    env["SEMANTIC_CACHE_DIR"] = str(Path.home() / ".cache" / "semantic" / "test")
    env["SEMANTIC_LOG_LEVEL"] = "WARNING"  # Reduce noise in tests
    env["ANTHROPIC_API_KEY"] = "test-key-for-e2e"

    # Start server in subprocess
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

    # Wait for server to start (poll health endpoint)
    start_time = time.time()
    timeout = 60  # 60 seconds for model loading
    server_ready = False

    while time.time() - start_time < timeout:
        try:
            response = httpx.get(f"{server_url}/health", timeout=2.0)
            if response.status_code == 200:
                server_ready = True
                break
        except (httpx.ConnectError, httpx.TimeoutException):
            time.sleep(0.5)

    if not server_ready:
        # Server failed to start - capture logs
        stdout, stderr = process.communicate(timeout=5)
        process.kill()
        raise RuntimeError(
            f"Server failed to start within {timeout}s\n"
            f"STDOUT:\n{stdout}\n"
            f"STDERR:\n{stderr}"
        )

    # Yield server URL to test
    try:
        yield server_url
    finally:
        # Teardown: stop server and clean up resources
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
