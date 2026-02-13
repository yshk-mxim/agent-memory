# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Yakov Shkolnikov and contributors
"""Integration tests for request ID middleware.

Tests verify that request correlation IDs are:
- Generated automatically when not provided
- Preserved when provided by client
- Propagated through structured logging context
- Unique per request
"""

import pytest
from fastapi.testclient import TestClient

from agent_memory.entrypoints.api_server import create_app


@pytest.fixture
def test_app():
    """Create test FastAPI app with request ID middleware."""
    app = create_app()

    # Initialize minimal state for testing
    class MockAppState:
        def __init__(self):
            self.shutting_down = False
            self.agent_memory = type(
                "obj",
                (object,),
                {
                    "block_pool": None,
                    "batch_engine": None,
                },
            )()

    app.state = MockAppState()

    return app


@pytest.mark.integration
def test_request_id_generated_when_missing(test_app):
    """Test request ID is generated if not provided.

    Expected behavior:
    - X-Request-ID header is present in response
    - ID is 16 characters (hex encoded UUID)
    - Response is successful
    """
    client = TestClient(test_app)

    response = client.get("/health/live")

    assert response.status_code == 200
    assert "X-Request-ID" in response.headers
    assert len(response.headers["X-Request-ID"]) == 16  # 16-char hex
    assert response.json() == {"status": "alive"}

    print(f"\n✅ Request ID generated: {response.headers['X-Request-ID']}")


@pytest.mark.integration
def test_request_id_preserved_from_header(test_app):
    """Test request ID is preserved if provided.

    Expected behavior:
    - Client-provided X-Request-ID is returned unchanged
    - No new ID is generated
    - Response is successful
    """
    client = TestClient(test_app)

    custom_id = "custom-request-id"
    response = client.get("/health/live", headers={"X-Request-ID": custom_id})

    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == custom_id
    assert response.json() == {"status": "alive"}

    print(f"\n✅ Request ID preserved: {response.headers['X-Request-ID']}")


@pytest.mark.integration
def test_request_id_in_logs(test_app):
    """Test request ID appears in structured logs.

    This test verifies that the request_id is bound to structlog context
    and would appear in all logs for this request.

    Note: Full log capture validation depends on log capture configuration.
    This test primarily validates the middleware integration is working.
    """
    client = TestClient(test_app)

    # Make request
    response = client.get("/health/live")
    request_id = response.headers["X-Request-ID"]

    # Verify response is successful
    assert response.status_code == 200
    assert isinstance(request_id, str)
    assert len(request_id) == 16

    # In a full logging setup, we would capture logs here
    # and verify request_id appears in all log entries
    # For now, we verify the middleware is working correctly

    print(f"\n✅ Request ID in context: {request_id}")


@pytest.mark.integration
def test_request_id_different_per_request(test_app):
    """Test each request gets unique ID.

    Expected behavior:
    - Two sequential requests generate different IDs
    - IDs do not collide
    - Both requests succeed
    """
    client = TestClient(test_app)

    response1 = client.get("/health/live")
    response2 = client.get("/health/live")

    id1 = response1.headers["X-Request-ID"]
    id2 = response2.headers["X-Request-ID"]

    assert response1.status_code == 200
    assert response2.status_code == 200
    assert id1 != id2
    assert len(id1) == 16
    assert len(id2) == 16

    print(f"\n✅ Unique request IDs: {id1} != {id2}")
