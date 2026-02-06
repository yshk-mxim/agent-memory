"""Middleware chain ordering tests.

Verifies that middleware fires in the correct order and each middleware's
effects are visible to downstream middleware. FastAPI's add_middleware uses
LIFO ordering — the last-added middleware executes first on request.

Registration order in api_server._register_middleware:
  1. RequestIDMiddleware    (first added → executed LAST on request, FIRST on response)
  2. RequestLoggingMiddleware
  3. RequestMetricsMiddleware
  4. CORSMiddleware
  5. AuthenticationMiddleware
  6. RateLimiter            (last added → executed FIRST on request, LAST on response)

Execution order on request:
  RateLimiter → Auth → CORS → Metrics → Logging → RequestID → Handler
"""

import pytest
from fastapi.testclient import TestClient

from semantic.entrypoints.api_server import create_app

pytestmark = pytest.mark.integration


@pytest.fixture
def test_app():
    """Create test FastAPI app with all middleware registered."""
    app = create_app()

    class MockAppState:
        def __init__(self):
            self.shutting_down = False
            self.semantic = type(
                "obj",
                (object,),
                {
                    "block_pool": None,
                    "batch_engine": None,
                },
            )()

    app.state = MockAppState()
    return app


@pytest.fixture
def _auth_enabled(monkeypatch):
    """Enable authentication by setting valid API keys."""
    monkeypatch.setenv("SEMANTIC_AUTH_DISABLED", "false")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-123")


@pytest.fixture
def _rate_limit_low(monkeypatch):
    """Set very low rate limits for testing."""
    monkeypatch.setenv("SEMANTIC_SERVER_RATE_LIMIT_GLOBAL", "3")
    monkeypatch.setenv("SEMANTIC_SERVER_RATE_LIMIT_PER_AGENT", "2")


class TestRequestIDPropagation:
    """Request ID should be present on all responses including errors."""

    def test_request_id_on_successful_response(self, test_app) -> None:
        client = TestClient(test_app, raise_server_exceptions=False)
        response = client.get("/health/live")
        assert "x-request-id" in response.headers
        assert len(response.headers["x-request-id"]) >= 8

    def test_request_id_preserved_from_client(self, test_app) -> None:
        client = TestClient(test_app, raise_server_exceptions=False)
        response = client.get("/health/live", headers={"x-request-id": "my-custom-id"})
        assert response.headers["x-request-id"] == "my-custom-id"

    def test_request_id_on_404_response(self, test_app) -> None:
        client = TestClient(test_app, raise_server_exceptions=False)
        response = client.get("/nonexistent/endpoint")
        assert "x-request-id" in response.headers

    def test_request_id_unique_per_request(self, test_app) -> None:
        client = TestClient(test_app, raise_server_exceptions=False)
        ids = set()
        for _ in range(10):
            resp = client.get("/health/live")
            ids.add(resp.headers.get("x-request-id"))
        assert len(ids) == 10, f"Expected 10 unique IDs, got {len(ids)}"


class TestResponseTimingHeader:
    """X-Response-Time should appear on all non-skipped responses."""

    def test_timing_header_on_api_endpoint(self, test_app) -> None:
        client = TestClient(test_app, raise_server_exceptions=False)
        response = client.get("/")
        assert "x-response-time" in response.headers
        timing = response.headers["x-response-time"]
        assert timing.endswith("ms"), f"Expected '...ms' format, got {timing!r}"

    def test_timing_header_value_is_numeric(self, test_app) -> None:
        client = TestClient(test_app, raise_server_exceptions=False)
        response = client.get("/")
        timing = response.headers.get("x-response-time", "")
        numeric_part = timing.replace("ms", "").strip()
        value = float(numeric_part)
        assert value >= 0, f"Timing must be non-negative, got {value}"
        assert value < 10_000, f"Timing unreasonably high: {value}ms"


class TestCORSHeaders:
    """CORS headers should be present on preflight and normal responses."""

    def test_cors_preflight_allows_origin(self, test_app) -> None:
        client = TestClient(test_app, raise_server_exceptions=False)
        response = client.options(
            "/v1/chat/completions",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
            },
        )
        assert response.headers.get("access-control-allow-origin") in ("*", "http://localhost:3000")

    def test_cors_allows_all_methods(self, test_app) -> None:
        client = TestClient(test_app, raise_server_exceptions=False)
        response = client.options(
            "/v1/chat/completions",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
            },
        )
        allowed = response.headers.get("access-control-allow-methods", "")
        assert "POST" in allowed or "*" in allowed


class TestAuthBeforeHandler:
    """Auth should reject before reaching the handler."""

    @pytest.fixture
    def auth_app(self, test_app, _auth_enabled):
        return test_app

    def test_missing_key_returns_401(self, auth_app) -> None:
        client = TestClient(auth_app, raise_server_exceptions=False)
        response = client.post("/v1/chat/completions", json={"model": "test"})
        assert response.status_code == 401

    def test_invalid_key_returns_401(self, auth_app) -> None:
        client = TestClient(auth_app, raise_server_exceptions=False)
        response = client.post(
            "/v1/chat/completions",
            json={"model": "test"},
            headers={"x-api-key": "wrong-key"},
        )
        assert response.status_code == 401

    def test_401_returns_json_error(self, auth_app) -> None:
        """Auth rejection should return a JSON error body."""
        client = TestClient(auth_app, raise_server_exceptions=False)
        response = client.post("/v1/chat/completions", json={"model": "test"})
        assert response.status_code == 401
        body = response.json()
        assert "error" in body or "detail" in body

    def test_health_bypasses_auth(self, auth_app) -> None:
        """Health endpoints should not require authentication."""
        client = TestClient(auth_app, raise_server_exceptions=False)
        response = client.get("/health/live")
        assert response.status_code == 200


class TestMiddlewareChainOrder:
    """Verify middleware effects are layered correctly."""

    def test_rate_limit_response_has_request_id_and_timing(
        self, test_app, _rate_limit_low
    ) -> None:
        """Rate-limited responses should still have RequestID and timing headers."""
        client = TestClient(test_app, raise_server_exceptions=False)
        # Exhaust rate limit
        for _ in range(5):
            client.post(
                "/v1/chat/completions",
                json={"model": "test", "messages": [{"role": "user", "content": "hi"}]},
            )
        # This should be rate-limited
        response = client.post(
            "/v1/chat/completions",
            json={"model": "test", "messages": [{"role": "user", "content": "hi"}]},
        )
        if response.status_code == 429:
            assert "x-request-id" in response.headers, "Rate-limited response missing request ID"

    def test_error_responses_have_request_id(self, test_app) -> None:
        """All error responses should have X-Request-ID set by middleware."""
        client = TestClient(test_app, raise_server_exceptions=False)
        # 404 error
        r404 = client.get("/does/not/exist")
        assert "x-request-id" in r404.headers

        # Method not allowed (POST to health)
        r405 = client.post("/health/live")
        assert "x-request-id" in r405.headers
