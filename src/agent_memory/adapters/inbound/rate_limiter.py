# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Yakov Shkolnikov and contributors
"""Rate limiting middleware for API requests.

Implements per-agent and global rate limiting using sliding window algorithm.
Returns 429 Too Many Requests with Retry-After header when limits exceeded.
"""

import asyncio
import logging
import time
from collections import defaultdict, deque
from collections.abc import Callable

from fastapi import Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class RateLimiter(BaseHTTPMiddleware):
    """Middleware for rate limiting API requests.

    Uses sliding window algorithm to track requests per agent and globally.
    Limits can be configured independently.

    Example:
        app.add_middleware(
            RateLimiter,
            requests_per_minute_per_agent=60,
            requests_per_minute_global=1000
        )
    """

    def __init__(
        self,
        app,
        requests_per_minute_per_agent: int = 60,
        requests_per_minute_global: int = 1000,
        window_size_seconds: int = 60,
    ):
        """Initialize rate limiter.

        Args:
            app: FastAPI application instance
            requests_per_minute_per_agent: Max requests per agent per minute
            requests_per_minute_global: Max global requests per minute
            window_size_seconds: Size of sliding window in seconds
        """
        super().__init__(app)
        self.requests_per_minute_per_agent = requests_per_minute_per_agent
        self.requests_per_minute_global = requests_per_minute_global
        self.window_size_seconds = window_size_seconds

        # Track requests per agent: agent_id → deque of timestamps
        self._agent_requests: dict[str, deque[float]] = defaultdict(deque)

        # Track global requests: deque of timestamps
        self._global_requests: deque[float] = deque()

        # Thread safety: asyncio lock for concurrent request handling
        # Without this, deque operations can corrupt state under concurrent access
        self._lock = asyncio.Lock()

        # Counter for periodic stale entry cleanup (every N requests)
        self._cleanup_counter = 0
        self._cleanup_interval = 100  # Clean up every 100 requests

        logger.info(
            f"Rate limiter enabled: {requests_per_minute_per_agent} req/min per agent, "
            f"{requests_per_minute_global} req/min global"
        )

    def _clean_old_requests(self, requests: deque[float], now: float) -> None:
        """Remove requests outside the sliding window.

        Args:
            requests: Deque of request timestamps
            now: Current timestamp
        """
        window_start = now - self.window_size_seconds
        while requests and requests[0] < window_start:
            requests.popleft()

    def _cleanup_stale_agents(self, now: float) -> int:
        """Remove empty agent entries to prevent memory leaks.

        Called periodically to clean up agents with no recent requests.
        This prevents unbounded growth of _agent_requests dict.

        Args:
            now: Current timestamp

        Returns:
            Number of stale entries removed
        """
        stale_agents = []
        window_start = now - self.window_size_seconds

        for agent_id, requests in self._agent_requests.items():
            # Clean old requests first
            while requests and requests[0] < window_start:
                requests.popleft()

            # Mark for removal if empty
            if not requests:
                stale_agents.append(agent_id)

        # Remove stale entries
        for agent_id in stale_agents:
            del self._agent_requests[agent_id]

        if stale_agents:
            logger.debug(f"Cleaned up {len(stale_agents)} stale rate limit entries")

        return len(stale_agents)

    def _is_rate_limited_agent(self, agent_id: str, now: float) -> tuple[bool, float]:
        """Check if agent has exceeded rate limit.

        Args:
            agent_id: Agent identifier
            now: Current timestamp

        Returns:
            Tuple of (is_limited, retry_after_seconds)
        """
        requests = self._agent_requests[agent_id]
        self._clean_old_requests(requests, now)

        if len(requests) >= self.requests_per_minute_per_agent:
            # Rate limited - calculate retry time
            oldest_request = requests[0]
            retry_after = oldest_request + self.window_size_seconds - now
            return True, max(1, int(retry_after))

        return False, 0

    def _is_rate_limited_global(self, now: float) -> tuple[bool, float]:
        """Check if global rate limit exceeded.

        Args:
            now: Current timestamp

        Returns:
            Tuple of (is_limited, retry_after_seconds)
        """
        self._clean_old_requests(self._global_requests, now)

        if len(self._global_requests) >= self.requests_per_minute_global:
            # Rate limited - calculate retry time
            oldest_request = self._global_requests[0]
            retry_after = oldest_request + self.window_size_seconds - now
            return True, max(1, int(retry_after))

        return False, 0

    def _extract_agent_id(self, request: Request) -> str | None:
        """Extract agent ID from request.

        Tries multiple sources:
        1. X-Agent-ID header
        2. Path parameter (for /v1/agents/{agent_id})
        3. Request body (for create_agent, generate)

        Args:
            request: Incoming HTTP request

        Returns:
            Agent ID if found, None otherwise
        """
        # Try header
        agent_id = request.headers.get("x-agent-id")
        if agent_id:
            return agent_id

        # Try path parameter
        if "agent_id" in request.path_params:
            return request.path_params["agent_id"]

        # For now, return None - body parsing would require async read
        # which is complex in middleware
        return None

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with rate limiting check.

        Args:
            request: Incoming HTTP request
            call_next: Next middleware/handler in chain

        Returns:
            Response from handler or 429 error
        """
        # Skip rate limiting for health endpoints
        if request.url.path.startswith("/health/"):
            return await call_next(request)

        now = time.time()

        # Thread safety: acquire lock for all rate limit checks and updates
        # This prevents deque corruption under concurrent async access
        async with self._lock:
            # Check global rate limit
            is_global_limited, global_retry_after = self._is_rate_limited_global(now)
            if is_global_limited:
                logger.warning(f"{request.method} {request.url.path} - Global rate limit exceeded")
                return JSONResponse(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    headers={"Retry-After": str(global_retry_after)},
                    content={
                        "error": {
                            "type": "rate_limit_error",
                            "message": (
                                f"Global rate limit exceeded. Retry after {global_retry_after}s."
                            ),
                        }
                    },
                )

            # Extract agent ID (if available)
            agent_id = self._extract_agent_id(request)

            # Check per-agent rate limit (if agent ID available)
            if agent_id:
                is_agent_limited, agent_retry_after = self._is_rate_limited_agent(agent_id, now)
                if is_agent_limited:
                    logger.warning(
                        f"{request.method} {request.url.path} - "
                        f"Agent {agent_id} rate limit exceeded"
                    )
                    return JSONResponse(
                        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                        headers={"Retry-After": str(agent_retry_after)},
                        content={
                            "error": {
                                "type": "rate_limit_error",
                                "message": (
                                    f"Agent rate limit exceeded. Retry after {agent_retry_after}s."
                                ),
                            }
                        },
                    )

            # Not rate limited - record request and proceed
            self._global_requests.append(now)
            if agent_id:
                self._agent_requests[agent_id].append(now)

            # Periodic cleanup of stale agent entries (prevents memory leak)
            self._cleanup_counter += 1
            if self._cleanup_counter >= self._cleanup_interval:
                self._cleanup_counter = 0
                self._cleanup_stale_agents(now)

            agent_info = (
                f", agent: {len(self._agent_requests[agent_id])}/"
                f"{self.requests_per_minute_per_agent})"
                if agent_id
                else ")"
            )
            logger.debug(
                f"{request.method} {request.url.path} - "
                f"Rate limit OK (global: {len(self._global_requests)}/"
                f"{self.requests_per_minute_global}{agent_info}"
            )

        return await call_next(request)
