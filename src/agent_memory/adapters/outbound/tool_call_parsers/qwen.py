# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Yakov Shkolnikov and contributors
"""Qwen / Hermes / Functionary tool call parser.

These model families use XML-like tags for tool calls:
    <tool_call>{"name": "func", "arguments": {"key": "value"}}</tool_call>

The inner content is valid JSON with either "arguments" (string or dict)
or "parameters" as the arguments key.
"""

import json
import re
import uuid

from agent_memory.domain.value_objects import ParsedToolCall

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


def _generate_tool_id() -> str:
    return f"toolu_{uuid.uuid4().hex[:24]}"


class QwenToolCallParser:
    """Parser for <tool_call>JSON</tool_call> format."""

    def parse(self, text: str) -> tuple[str, list[ParsedToolCall]]:
        if "<tool_call>" not in text:
            return text, []

        calls: list[ParsedToolCall] = []
        remaining = text

        for match in _TOOL_CALL_RE.finditer(text):
            try:
                obj = json.loads(match.group(1))
            except (json.JSONDecodeError, TypeError):
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

            calls.append(ParsedToolCall(
                id=_generate_tool_id(),
                name=name,
                input=args,
            ))

        if calls:
            remaining = _TOOL_CALL_RE.sub("", text).strip()

        return remaining, calls
