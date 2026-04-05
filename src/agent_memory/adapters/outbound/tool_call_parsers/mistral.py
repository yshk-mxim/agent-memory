# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Yakov Shkolnikov and contributors
"""Mistral tool call parser.

Mistral models use the format:
    [TOOL_CALLS][{"name": "func", "arguments": {"key": "value"}}]

The content after [TOOL_CALLS] is a JSON array of tool call objects.
"""

import json
import uuid

from agent_memory.domain.value_objects import ParsedToolCall

_MARKER = "[TOOL_CALLS]"


def _generate_tool_id() -> str:
    return f"toolu_{uuid.uuid4().hex[:24]}"


class MistralToolCallParser:
    """Parser for Mistral's [TOOL_CALLS] format."""

    def parse(self, text: str) -> tuple[str, list[ParsedToolCall]]:
        if _MARKER not in text:
            return text, []

        idx = text.index(_MARKER)
        before = text[:idx].strip()
        after = text[idx + len(_MARKER) :].strip()

        if not after or after[0] != "[":
            return text, []

        try:
            tool_array = json.loads(after)
        except (json.JSONDecodeError, TypeError):
            return text, []

        if not isinstance(tool_array, list):
            return text, []

        calls: list[ParsedToolCall] = []
        for obj in tool_array:
            if not isinstance(obj, dict):
                continue
            name = obj.get("name", "")
            if not name:
                continue
            args = obj.get("arguments") or obj.get("parameters", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except (json.JSONDecodeError, TypeError):
                    args = {}
            if not isinstance(args, dict):
                args = {}
            calls.append(
                ParsedToolCall(
                    id=_generate_tool_id(),
                    name=name,
                    input=args,
                )
            )

        return before, calls
