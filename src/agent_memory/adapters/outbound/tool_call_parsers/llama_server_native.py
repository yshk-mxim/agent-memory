# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Yakov Shkolnikov and contributors
"""llama-server native tool call extractor.

When llama-server's built-in tool calling handler works correctly
(e.g., for Qwen 2.5, Llama 3), it returns tool calls in the OpenAI
``tool_calls`` field of the response message. This module extracts
``ParsedToolCall`` objects from that field.

This is NOT a text parser — it reads structured API response fields.
It's called separately from the text-based parser chain.
"""

import json
import uuid
from typing import Any

from agent_memory.domain.value_objects import ParsedToolCall


def _generate_tool_id() -> str:
    return f"toolu_{uuid.uuid4().hex[:24]}"


def extract_from_openai_tool_calls(
    raw_tool_calls: list[dict[str, Any]],
) -> list[ParsedToolCall]:
    """Convert OpenAI-format tool_calls to ParsedToolCall objects.

    Args:
        raw_tool_calls: List of OpenAI tool call dicts with
            ``function.name`` and ``function.arguments`` fields.

    Returns:
        List of validated ParsedToolCall objects.
    """
    results: list[ParsedToolCall] = []
    for tc in raw_tool_calls:
        fn = tc.get("function", {})
        name = fn.get("name", "")
        if not name:
            continue

        args = fn.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except (json.JSONDecodeError, TypeError):
                args = {}
        if not isinstance(args, dict):
            args = {}

        results.append(ParsedToolCall(
            id=tc.get("id", _generate_tool_id()),
            name=name,
            input=args,
        ))

    return results
