"""Request ID middleware for correlation tracking.

Generates unique request IDs and propagates them through the request lifecycle
using structlog contextvars.
"""

import uuid
from collections.abc import Callable

import structlog
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = structlog.get_logger(__name__)


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Middleware to generate and track request correlation IDs.

    Adds X-Request-ID header to all responses and propagates ID through
    structlog contextvars for logging context.

    Example:
        app.add_middleware(RequestIDMiddleware)
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with correlation ID.

        Args:
            request: Incoming HTTP request
            call_next: Next middleware/handler in chain

        Returns:
            Response with X-Request-ID header
        """
        # Extract or generate request ID
        request_id = request.headers.get("x-request-id")
        if not request_id:
            request_id = uuid.uuid4().hex[:16]

        # Bind to structlog context for all logs in this request
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(
            request_id=request_id,
            method=request.method,
            path=request.url.path,
        )

        # Process request
        response = await call_next(request)

        # Add request ID to response headers
        response.headers["X-Request-ID"] = request_id

        return response
