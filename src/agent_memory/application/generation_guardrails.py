# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Yakov Shkolnikov and contributors
"""Generation guardrails — detect and break LLM failure loops.

When a local model retries the same failing tool call repeatedly (e.g.,
Edit with wrong old_string), the client wastes tokens and wall-clock time
on each retry.  These guardrails scan the conversation history and return
corrective hints to inject before the next generation.

Architecture layer: application (pure logic, no I/O).
Called by: inbound adapters before message conversion / generation.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# After this many consecutive errors from the same tool, inject a hint.
TOOL_RETRY_THRESHOLD = 3

# Tool-specific corrective hints.
_TOOL_HINTS: dict[str, str] = {
    "Edit": (
        "You have repeatedly failed to edit this file because old_string "
        "does not match the actual file content. STOP retrying. Use the "
        "Read tool to read the file first, then copy the EXACT text from "
        "the Read output as your old_string."
    ),
    "Write": (
        "The Write tool has failed repeatedly. Check that the file_path "
        "is correct and the parent directory exists."
    ),
}

_DEFAULT_HINT = (
    "This tool call has failed {count} times in a row with the same error. "
    "Try a different approach."
)


def detect_tool_retry_loop(
    messages: list[Any],
) -> str | None:
    """Scan Anthropic-format messages for repeated tool failures.

    Walks the conversation tail looking for consecutive assistant tool_use
    followed by user tool_result with is_error=True for the same tool.

    Args:
        messages: List of Anthropic Message objects or dicts with
            role/content fields.

    Returns:
        Corrective hint string if a retry loop is detected, None otherwise.
    """
    consecutive_errors = 0
    tool_name: str | None = None

    # Walk backward through messages
    for msg in reversed(messages):
        role = msg.role if hasattr(msg, "role") else msg.get("role", "")
        content = msg.content if hasattr(msg, "content") else msg.get("content", "")

        if isinstance(content, str):
            break  # Plain text message — not part of a tool loop

        if not isinstance(content, list):
            break

        # Check for tool_result with is_error
        has_error = False
        has_tool_use = False
        current_tool_name: str | None = None

        for block in content:
            btype = block.type if hasattr(block, "type") else block.get("type", "")

            if btype == "tool_result":
                is_error = (
                    block.is_error
                    if hasattr(block, "is_error")
                    else block.get("is_error", False)
                )
                if is_error:
                    has_error = True

            elif btype == "tool_use":
                has_tool_use = True
                current_tool_name = (
                    block.name
                    if hasattr(block, "name")
                    else block.get("name", "")
                )

        if role == "user" and has_error:
            consecutive_errors += 1
        elif role == "assistant" and has_tool_use:
            if tool_name is None:
                tool_name = current_tool_name
            elif current_tool_name != tool_name:
                break  # Different tool — not a retry loop
        else:
            break  # Non-tool message — end of loop region

    if consecutive_errors >= TOOL_RETRY_THRESHOLD and tool_name:
        hint = _TOOL_HINTS.get(tool_name)
        if hint is None:
            hint = _DEFAULT_HINT.format(count=consecutive_errors)
        return hint

    return None
