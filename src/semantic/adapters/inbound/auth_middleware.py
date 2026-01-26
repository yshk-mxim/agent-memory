"""Authentication middleware for API key validation.

Validates ANTHROPIC_API_KEY from request headers against configured keys.
Returns 401 Unauthorized for invalid or missing keys.
"""

import logging
import os
from collections.abc import Callable
from typing import ClassVar

from fastapi import Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class AuthenticationMiddleware(BaseHTTPMiddleware):
    """Middleware for API key authentication.

    Validates x-api-key header against ANTHROPIC_API_KEY environment variable.
    Supports multiple valid keys (comma-separated in env var).

    Example:
        app.add_middleware(AuthenticationMiddleware)
    """

    # Endpoints that don't require authentication
    PUBLIC_ENDPOINTS: ClassVar[set[str]] = {
        "/",
        "/health",
        "/docs",
        "/redoc",
        "/openapi.json",
    }

    def __init__(self, app):
        """Initialize authentication middleware.

        Args:
            app: FastAPI application instance
        """
        super().__init__(app)
        self._valid_keys = self._load_valid_keys()

        if not self._valid_keys:
            logger.warning(
                "No ANTHROPIC_API_KEY configured - authentication disabled! "
                "Set ANTHROPIC_API_KEY environment variable to enable."
            )
        else:
            logger.info(
                f"Authentication enabled with {len(self._valid_keys)} valid key(s)"
            )

    def _load_valid_keys(self) -> set[str]:
        """Load valid API keys from environment.

        Returns:
            Set of valid API keys

        Notes:
            - Reads ANTHROPIC_API_KEY environment variable
            - Supports comma-separated list for multiple keys
            - Empty/missing env var disables authentication
        """
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            return set()

        # Support comma-separated list of keys
        keys = {key.strip() for key in api_key.split(",")}
        return {key for key in keys if key}  # Filter empty strings

    async def dispatch(
        self, request: Request, call_next: Callable
    ) -> Response:
        """Process request with authentication check.

        Args:
            request: Incoming HTTP request
            call_next: Next middleware/handler in chain

        Returns:
            Response from handler or 401 error
        """
        # Skip authentication for public endpoints
        if request.url.path in self.PUBLIC_ENDPOINTS:
            return await call_next(request)

        # Skip authentication for all health endpoints
        if request.url.path.startswith("/health/"):
            return await call_next(request)

        # Skip authentication if no keys configured (development mode)
        if not self._valid_keys:
            logger.debug(f"{request.method} {request.url.path} - Auth disabled")
            return await call_next(request)

        # Extract API key from header
        api_key = request.headers.get("x-api-key")

        # Also check anthropic-api-key header (alternative)
        if not api_key:
            api_key = request.headers.get("anthropic-api-key")

        # Validate API key
        if not api_key:
            logger.warning(
                f"{request.method} {request.url.path} - Missing API key"
            )
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={
                    "error": {
                        "type": "authentication_error",
                        "message": "Missing API key. Provide x-api-key header.",
                    }
                },
            )

        if api_key not in self._valid_keys:
            logger.warning(
                f"{request.method} {request.url.path} - Invalid API key"
            )
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={
                    "error": {
                        "type": "authentication_error",
                        "message": "Invalid API key.",
                    }
                },
            )

        # Valid key - proceed with request
        logger.debug(
            f"{request.method} {request.url.path} - Authenticated"
        )
        return await call_next(request)


def create_auth_middleware() -> AuthenticationMiddleware:
    """Factory function for creating authentication middleware.

    Returns:
        AuthenticationMiddleware instance

    Example:
        >>> app.add_middleware(create_auth_middleware())
    """
    # Note: FastAPI add_middleware expects a class, not an instance
    # This function is here for future use if we need to pass config
    return AuthenticationMiddleware
