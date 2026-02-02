"""Shared helper functions for inbound adapters.

Contains common functionality used by multiple API adapters to avoid code duplication.
"""

import json
import logging
import time
from typing import Any

from fastapi import HTTPException, Request, status

logger = logging.getLogger(__name__)

# Default timeout for batch engine step operations (5 minutes)
STEP_TIMEOUT_SECONDS = 300


def get_semantic_state(request: Request) -> Any:
    """Safely get semantic state from request, raising clear error if not initialized.

    Args:
        request: FastAPI request object

    Returns:
        The semantic state object

    Raises:
        HTTPException: If semantic state is not initialized
    """
    if not hasattr(request.app.state, "semantic") or request.app.state.semantic is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Server is still initializing. Please retry in a few seconds.",
        )
    return request.app.state.semantic


def get_coordination_service(request: Request) -> Any:
    """Safely get coordination service from request, raising clear error if not initialized.

    Args:
        request: FastAPI request object

    Returns:
        The coordination service object

    Raises:
        HTTPException: If coordination service is not initialized
    """
    if (
        not hasattr(request.app.state, "coordination_service")
        or request.app.state.coordination_service is None
    ):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Coordination service is not available. Server may still be initializing.",
        )
    return request.app.state.coordination_service


def run_step_for_uid(
    batch_engine: Any,
    uid: str,
    timeout_seconds: float = STEP_TIMEOUT_SECONDS,
) -> Any:
    """Run batch engine step until we get result for uid, with timeout.

    This is a blocking helper meant to run in a thread executor.

    Args:
        batch_engine: The batch engine instance
        uid: The unique ID to wait for
        timeout_seconds: Maximum time to wait (default: 5 minutes)

    Returns:
        The completion result for the given uid, or None if not found/timeout

    Raises:
        TimeoutError: If generation exceeds timeout
    """
    start_time = time.monotonic()

    for result in batch_engine.step():
        if result.uid == uid:
            return result

        elapsed = time.monotonic() - start_time
        if elapsed > timeout_seconds:
            logger.error(f"Generation timeout after {elapsed:.1f}s for uid={uid}")
            raise TimeoutError(f"Generation timed out after {timeout_seconds}s")

    return None


def try_parse_json_at(text: str, start: int) -> tuple[dict[str, Any] | None, int]:
    """Try to parse a JSON object starting at the given position.

    Uses a bracket-counting approach to find the end of the JSON object,
    then validates with json.loads().

    Args:
        text: The full text
        start: Starting position (should be at '{')

    Returns:
        Tuple of (parsed_dict or None, end_position)
    """
    if start >= len(text) or text[start] != "{":
        return None, start

    depth = 0
    in_string = False
    escape_next = False
    end = start

    for i, char in enumerate(text[start:], start):
        if escape_next:
            escape_next = False
            continue

        if char == "\\" and in_string:
            escape_next = True
            continue

        if char == '"' and not escape_next:
            in_string = not in_string
            continue

        if in_string:
            continue

        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    else:
        return None, start

    try:
        json_str = text[start:end]
        return json.loads(json_str), end
    except json.JSONDecodeError:
        return None, start
