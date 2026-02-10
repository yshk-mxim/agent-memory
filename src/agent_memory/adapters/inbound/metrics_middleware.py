"""Metrics collection middleware for Prometheus.

Automatically collects request metrics for all HTTP requests.
"""

import time
from collections.abc import Callable

import structlog
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from agent_memory.adapters.inbound.metrics import (
    request_duration_seconds,
    request_total,
)

logger = structlog.get_logger(__name__)


class RequestMetricsMiddleware(BaseHTTPMiddleware):
    """Middleware to collect request metrics for Prometheus.

    Collects:
    - request_total (counter by method, path, status)
    - request_duration_seconds (histogram by method, path)

    Example:
        app.add_middleware(RequestMetricsMiddleware)
    """

    def __init__(self, app, skip_paths: set[str] | None = None):
        """Initialize metrics middleware.

        Args:
            app: FastAPI application instance
            skip_paths: Set of paths to skip metrics (e.g., /metrics itself)
        """
        super().__init__(app)
        self.skip_paths = skip_paths or set()

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Collect metrics for this request.

        Args:
            request: Incoming HTTP request
            call_next: Next middleware/handler in chain

        Returns:
            Response from handler
        """
        # Skip metrics for certain paths
        if request.url.path in self.skip_paths:
            return await call_next(request)

        # Start timer
        start_time = time.time()

        # Process request
        try:
            response = await call_next(request)
        except Exception:
            # Still record metrics on error
            duration = time.time() - start_time
            request_duration_seconds.labels(method=request.method, path=request.url.path).observe(
                duration
            )
            request_total.labels(
                method=request.method,
                path=request.url.path,
                status_code="500",  # Assume 500 for unhandled exception
            ).inc()
            raise

        # Record metrics
        duration = time.time() - start_time
        request_duration_seconds.labels(method=request.method, path=request.url.path).observe(
            duration
        )
        request_total.labels(
            method=request.method, path=request.url.path, status_code=str(response.status_code)
        ).inc()

        return response
