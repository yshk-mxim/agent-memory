# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Yakov Shkolnikov and contributors
"""Gemma 4 native tool call parser.

Gemma 4 models (26B MoE, 31B Dense) use special tokens for tool calls:
    <|tool_call>call:Name{key: "value"}<tool_call|>

llama-server strips the special tokens before returning text, leaving:
    call:Name{key: "value"}

Variants observed in production:
    call:Read{file_path: "/foo/bar.py"}              — simple
    call:agent_memory:Agent{task: "..."}              — namespace prefix
    call:TaskCreate{tasks: [{description: "..."}]}    — nested arrays
    call:A{x: "1"}call:B{y: "2"}                     — concatenated
    call:Name {key: "val"}                            — space before brace

The arguments use JavaScript-like object syntax (unquoted keys, mixed
quotes) which requires a state-machine parser, not simple json.loads().
"""

import json
import re
import uuid
from typing import Any

from agent_memory.domain.value_objects import ParsedToolCall

# Matches call:Name or call:namespace:Name
_CALL_RE = re.compile(r"call:([\w:]+)")


def _generate_tool_id() -> str:
    return f"toolu_{uuid.uuid4().hex[:24]}"


class GemmaToolCallParser:
    """Parser for Gemma 4's native call:Name{...} format."""

    def parse(self, text: str) -> tuple[str, list[ParsedToolCall]]:
        """Extract Gemma-native tool calls from text."""
        if "call:" not in text:
            return text, []

        calls: list[ParsedToolCall] = []
        regions_to_remove: list[tuple[int, int]] = []

        for m in _CALL_RE.finditer(text):
            raw_name = m.group(1)
            # Use last colon-separated segment as tool name
            # e.g. "agent_memory:Agent" → "Agent"
            name = raw_name.rsplit(":", 1)[-1] if ":" in raw_name else raw_name

            # Find opening brace (may have whitespace between name and {)
            rest = text[m.end() :]
            stripped = rest.lstrip(" \t\n")
            if not stripped or stripped[0] != "{":
                continue

            whitespace_len = len(rest) - len(stripped)
            raw_args = _extract_balanced(stripped, "{", "}")
            if raw_args is None:
                continue

            parsed = _parse_jslike(raw_args)
            if parsed is None:
                continue

            if not isinstance(parsed, dict):
                continue

            calls.append(
                ParsedToolCall(
                    id=_generate_tool_id(),
                    name=name,
                    input=parsed,
                )
            )

            # Track region for removal
            start = m.start()
            end = m.end() + whitespace_len + len(raw_args)
            regions_to_remove.append((start, end))

        if not calls:
            return text, []

        # Strip matched regions from text (reverse order to preserve indices)
        chars = list(text)
        for start, end in reversed(regions_to_remove):
            chars[start:end] = []
        remaining = "".join(chars).strip()

        return remaining, calls


def _extract_balanced(s: str, open_ch: str, close_ch: str) -> str | None:
    """Extract content within balanced delimiters from start of string.

    Recognizes standard quotes AND Gemma 4's ``<|"|>`` string delimiters
    so that ``{`` / ``}`` inside strings don't break brace depth tracking.
    """
    if not s or s[0] != open_ch:
        return None
    depth = 0
    in_string: str | bool = False
    in_gemma_string = False
    escape_next = False
    i = 0
    n = len(s)
    while i < n:
        # Check for Gemma 4 string delimiter <|"|>
        if not in_string and s[i : i + 5] == '<|"|>':
            in_gemma_string = not in_gemma_string
            i += 5
            continue
        if in_gemma_string:
            i += 1
            continue

        ch = s[i]
        if escape_next:
            escape_next = False
            i += 1
            continue
        if ch == "\\":
            escape_next = True
            i += 1
            continue
        if ch in ('"', "'") and not in_string:
            in_string = ch
            i += 1
            continue
        if in_string:
            if ch == in_string:
                in_string = False
            i += 1
            continue
        if ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return s[: i + 1]
        i += 1
    return None  # Unbalanced


def _replace_gemma_strings(s: str) -> str:
    """Replace Gemma 4 ``<|"|>content<|"|>`` with ``"escaped_content"``.

    Escapes double-quotes and backslashes inside the delimiters so the
    result is valid JSON string syntax.  Mirrors llama.cpp's
    ``escape_json_string_inner()`` in the Gemma 4 AST mapper.
    """
    delim = '<|"|>'
    dlen = len(delim)  # 5
    result: list[str] = []
    i = 0
    while i < len(s):
        start = s.find(delim, i)
        if start == -1:
            result.append(s[i:])
            break
        # Copy text before the opening delimiter
        result.append(s[i:start])
        end = s.find(delim, start + dlen)
        if end == -1:
            # Unmatched opening delimiter — treat rest as content
            result.append('"')
            inner = s[start + dlen :]
            result.append(inner.replace("\\", "\\\\").replace('"', '\\"'))
            result.append('"')
            i = len(s)
            break
        inner = s[start + dlen : end]
        result.append('"')
        result.append(inner.replace("\\", "\\\\").replace('"', '\\"'))
        result.append('"')
        i = end + dlen
    return "".join(result)


def _parse_jslike(s: str) -> Any:
    """Parse JavaScript-like object/array syntax into Python objects.

    Handles:
    - Valid JSON (fast path via json.loads)
    - Unquoted keys: {key: "value"} → {"key": "value"}
    - Single-quoted strings: {'key': 'value'}
    - Bare values: {debug: true, count: 42}
    - Nested objects and arrays
    - Gemma's <|"|> delimiter remnants

    State-machine approach: process character by character, building
    tokens and inferring structure from context.
    """
    # Fast path: try valid JSON first
    try:
        return json.loads(s)
    except (json.JSONDecodeError, ValueError):
        pass

    # Clean up Gemma <|"|> string delimiters.
    # Must escape inner double-quotes before converting to JSON strings,
    # matching llama.cpp's escape_json_string_inner() behavior.
    s = _replace_gemma_strings(s)

    # Fix unquoted keys and try JSON again
    fixed = _fix_jslike_to_json(s)
    if fixed is not None:
        try:
            return json.loads(fixed)
        except (json.JSONDecodeError, ValueError):
            pass

    # Last resort: manual flat key-value extraction
    return _extract_flat_kvs(s)


def _fix_jslike_to_json(s: str) -> str | None:
    """Convert JS-like object notation to valid JSON.

    Walks the string character by character, quoting unquoted keys
    and converting single quotes to double quotes.
    """
    result: list[str] = []
    i = 0
    n = len(s)

    while i < n:
        ch = s[i]

        # Skip whitespace
        if ch in " \t\n\r":
            result.append(ch)
            i += 1
            continue

        # String literals
        if ch in ('"', "'"):
            end, converted = _read_string(s, i)
            if end is None:
                return None
            result.append(converted)
            i = end
            continue

        # Structural characters
        if ch in "{}[]:,":
            result.append(ch)
            i += 1
            continue

        # Unquoted identifier (key or bare value)
        if ch.isalpha() or ch == "_" or ch == "$":
            end = i
            while end < n and (s[end].isalnum() or s[end] in "_$.-/"):
                end += 1

            word = s[i:end]

            # Check if this is a key (followed by ':') or a bare value
            rest = s[end:].lstrip()
            if rest and rest[0] == ":":
                # It's a key — quote it
                result.append(f'"{word}"')
            # Bare value: true/false/null stay as-is, others get quoted
            elif word in ("true", "false", "null"):
                result.append(word)
            else:
                result.append(f'"{word}"')
            i = end
            continue

        # Numbers (including negative)
        if ch.isdigit() or (ch == "-" and i + 1 < n and s[i + 1].isdigit()):
            end = i + 1
            while end < n and (s[end].isdigit() or s[end] in ".eE+-"):
                end += 1
            result.append(s[i:end])
            i = end
            continue

        # Unknown character — pass through
        result.append(ch)
        i += 1

    return "".join(result)


def _read_string(s: str, start: int) -> tuple[int | None, str]:
    """Read a string literal starting at s[start], return (end_pos, json_string)."""
    quote = s[start]
    chars: list[str] = ['"']  # Always output double quotes
    i = start + 1
    n = len(s)

    while i < n:
        ch = s[i]
        if ch == "\\":
            if i + 1 < n:
                chars.append(ch)
                chars.append(s[i + 1])
                i += 2
                continue
            return None, ""
        if ch == quote:
            chars.append('"')
            return i + 1, "".join(chars)
        if ch == '"' and quote == "'":
            # Escape double quotes inside single-quoted strings
            chars.append('\\"')
        else:
            chars.append(ch)
        i += 1

    return None, ""  # Unterminated string


def _extract_flat_kvs(s: str) -> dict[str, str] | None:
    """Last-resort flat key-value extraction via regex."""
    result: dict[str, str] = {}
    for kv in re.finditer(r'(\w+)\s*:\s*(?:"([^"]*)"|([\w.+\-/]+))', s):
        key = kv.group(1)
        val = kv.group(2) if kv.group(2) is not None else kv.group(3)
        result[key] = val
    return result if result else None
