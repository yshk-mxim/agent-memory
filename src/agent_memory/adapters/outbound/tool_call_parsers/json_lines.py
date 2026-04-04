# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Yakov Shkolnikov and contributors
"""Universal JSON-based tool call parser (fallback).

Handles any model that outputs tool calls as JSON objects in text:
    {"name": "func", "parameters": {"key": "value"}}     — Jinja template / generic
    {"name": "func", "arguments": {"key": "value"}}       — OpenAI function calling
    {"tool_use": {"name": "func", "input": {"key": ...}}} — Anthropic proxy format
    {"function_call": {"name": "func", "arguments": {}}}  — OpenAI legacy

Also handles JSON inside markdown code blocks (```json ... ```) which
instruction-following models sometimes produce.
"""

import json
import re
import uuid
from typing import Any

from agent_memory.domain.value_objects import ParsedToolCall

# Strip markdown code blocks: ```json ... ``` or ``` ... ```
_CODE_BLOCK_RE = re.compile(r"```(?:json)?\s*\n?(.*?)\n?\s*```", re.DOTALL)


def _generate_tool_id() -> str:
    return f"toolu_{uuid.uuid4().hex[:24]}"


def _try_parse_json_at(text: str, start: int) -> tuple[dict | None, int]:
    """Parse a JSON object starting at position. Returns (obj, end_pos)."""
    if start >= len(text) or text[start] != "{":
        return None, start

    depth = 0
    in_string = False
    escape_next = False

    for i in range(start, len(text)):
        ch = text[i]
        if escape_next:
            escape_next = False
            continue
        if ch == "\\":
            escape_next = True
            continue
        if ch == '"' and not escape_next:
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    obj = json.loads(text[start : i + 1])
                    if isinstance(obj, dict):
                        return obj, i + 1
                except (json.JSONDecodeError, ValueError):
                    pass
                return None, i + 1

    return None, len(text)


def _extract_tool_call(obj: dict[str, Any]) -> ParsedToolCall | None:
    """Try to extract a tool call from a JSON object in any known format."""
    # Format 1: {"name": ..., "parameters": ...}
    if "name" in obj and "parameters" in obj:
        name = obj["name"]
        args = obj["parameters"]
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except (json.JSONDecodeError, TypeError):
                args = {}
        if name and isinstance(args, dict):
            return ParsedToolCall(id=_generate_tool_id(), name=name, input=args)

    # Format 2: {"name": ..., "arguments": ...}
    if "name" in obj and "arguments" in obj:
        name = obj["name"]
        args = obj["arguments"]
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except (json.JSONDecodeError, TypeError):
                args = {}
        if name and isinstance(args, dict):
            return ParsedToolCall(id=_generate_tool_id(), name=name, input=args)

    # Format 3: {"tool_use": {"name": ..., "input": ...}}
    if "tool_use" in obj and isinstance(obj["tool_use"], dict):
        tool_data = obj["tool_use"]
        name = tool_data.get("name", "")
        args = tool_data.get("input", {})
        if name and isinstance(args, dict):
            return ParsedToolCall(id=_generate_tool_id(), name=name, input=args)

    # Format 4: {"function_call": {"name": ..., "arguments": ...}}
    if "function_call" in obj and isinstance(obj["function_call"], dict):
        fc = obj["function_call"]
        name = fc.get("name", "")
        args = fc.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except (json.JSONDecodeError, TypeError):
                args = {}
        if name and isinstance(args, dict):
            return ParsedToolCall(id=_generate_tool_id(), name=name, input=args)

    return None


class JsonLinesToolCallParser:
    """Universal fallback parser for JSON tool calls in text."""

    def parse(self, text: str) -> tuple[str, list[ParsedToolCall]]:
        calls: list[ParsedToolCall] = []
        regions_to_remove: list[tuple[int, int]] = []

        # First, extract JSON from markdown code blocks
        expanded_text = text
        code_block_replacements: list[tuple[int, int, str]] = []
        for match in _CODE_BLOCK_RE.finditer(text):
            code_block_replacements.append(
                (match.start(), match.end(), match.group(1).strip())
            )

        # Scan text for JSON objects
        search_text = text
        # Also scan inside code blocks
        for _start, _end, content in code_block_replacements:
            pos = 0
            while pos < len(content):
                idx = content.find("{", pos)
                if idx == -1:
                    break
                obj, end = _try_parse_json_at(content, idx)
                if obj is not None:
                    tc = _extract_tool_call(obj)
                    if tc is not None:
                        calls.append(tc)
                        # Mark the entire code block for removal
                        if (_start, _end) not in [
                            (s, e) for s, e, *_ in regions_to_remove
                        ]:
                            regions_to_remove.append((_start, _end))
                    pos = end
                else:
                    pos = idx + 1

        # Scan line by line for JSON objects (outside code blocks)
        for line in text.split("\n"):
            stripped = line.strip()
            if not stripped or not stripped.startswith("{"):
                continue
            try:
                obj = json.loads(stripped)
            except (json.JSONDecodeError, ValueError):
                continue
            if not isinstance(obj, dict):
                continue
            tc = _extract_tool_call(obj)
            if tc is not None:
                # Avoid duplicates from code block scanning
                if not any(c.name == tc.name and c.input == tc.input for c in calls):
                    calls.append(tc)
                    # Find and mark this line for removal
                    idx = text.find(stripped)
                    if idx >= 0:
                        regions_to_remove.append((idx, idx + len(stripped)))

        if not calls:
            return text, []

        # Remove matched regions (reverse order)
        chars = list(text)
        for start, end in sorted(set(regions_to_remove), reverse=True):
            chars[start:end] = []
        remaining = "".join(chars).strip()

        return remaining, calls
