"""Request logging middleware for observability.

Logs all HTTP requests with timing, status, and contextual information.
"""

import time
from collections.abc import Callable

import structlog
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = structlog.get_logger(__name__)

# HTTP status code threshold for warning vs info logs
HTTP_ERROR_THRESHOLD = 400


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware to log all HTTP requests with timing and context.

    Logs request start, request end with duration, and any errors.
    Uses structlog for structured output with request_id context.

    Example:
        app.add_middleware(RequestLoggingMiddleware)
    """

    def __init__(self, app, skip_paths: set[str] | None = None):
        """Initialize request logging middleware.

        Args:
            app: FastAPI application instance
            skip_paths: Set of paths to skip logging (e.g., health checks)
        """
        super().__init__(app)
        self.skip_paths = skip_paths or set()

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with logging.

        Args:
            request: Incoming HTTP request
            call_next: Next middleware/handler in chain

        Returns:
            Response from handler
        """
        # Skip logging for certain paths (avoid log spam)
        if request.url.path in self.skip_paths:
            return await call_next(request)

        # Log request start
        start_time = time.time()

        logger.info(
            "request_start",
            method=request.method,
            path=request.url.path,
            query=str(request.url.query) if request.url.query else None,
            client_host=request.client.host if request.client else None,
        )

        # Process request and capture any errors
        try:
            response = await call_next(request)
        except Exception as exc:
            duration_ms = (time.time() - start_time) * 1000
            logger.error(
                "request_error",
                duration_ms=round(duration_ms, 2),
                error_type=type(exc).__name__,
                error_message=str(exc),
                exc_info=True,
            )
            raise

        # Log request completion
        duration_ms = (time.time() - start_time) * 1000

        log_method = logger.info if response.status_code < HTTP_ERROR_THRESHOLD else logger.warning
        log_method(
            "request_complete",
            status_code=response.status_code,
            duration_ms=round(duration_ms, 2),
        )

        # Add timing header for debugging
        response.headers["X-Response-Time"] = f"{duration_ms:.2f}ms"

        return response
