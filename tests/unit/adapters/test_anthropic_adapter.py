# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Yakov Shkolnikov and contributors
"""Unit tests for Anthropic adapter pure functions.

Tests parse_tool_calls, generate_agent_id_from_tokens, messages_to_prompt,
and messages_to_chat_dicts — no MLX or FastAPI needed.
"""

import sys
from unittest.mock import MagicMock

import pytest

# Mock MLX modules before importing the adapter
sys.modules.setdefault("mlx", MagicMock())
sys.modules.setdefault("mlx.core", MagicMock())
sys.modules.setdefault("mlx.utils", MagicMock())
sys.modules.setdefault("mlx_lm", MagicMock())

from agent_memory.adapters.inbound.anthropic_adapter import (
    _anthropic_to_openai_tools,
    generate_agent_id_from_tokens,
    parse_tool_calls,
)

pytestmark = pytest.mark.unit


class TestGenerateAgentId:
    """Tests for generate_agent_id_from_tokens."""

    def test_format(self) -> None:
        """Should return msg_{16-char-hex}."""
        agent_id = generate_agent_id_from_tokens([1, 2, 3, 4, 5])
        assert agent_id.startswith("msg_")
        assert len(agent_id) == 20  # "msg_" + 16 hex chars

    def test_deterministic(self) -> None:
        """Same tokens should produce same ID."""
        tokens = list(range(200))
        id1 = generate_agent_id_from_tokens(tokens)
        id2 = generate_agent_id_from_tokens(tokens)
        assert id1 == id2

    def test_uses_first_100_tokens(self) -> None:
        """Tokens beyond 100 should not affect the ID."""
        base = list(range(100))
        id1 = generate_agent_id_from_tokens(base + [999])
        id2 = generate_agent_id_from_tokens(base + [888])
        assert id1 == id2

    def test_different_prefixes_differ(self) -> None:
        """Different first-100 tokens should produce different IDs."""
        id1 = generate_agent_id_from_tokens([1, 2, 3])
        id2 = generate_agent_id_from_tokens([4, 5, 6])
        assert id1 != id2

    def test_empty_tokens(self) -> None:
        """Empty token list should still produce valid ID."""
        agent_id = generate_agent_id_from_tokens([])
        assert agent_id.startswith("msg_")
        assert len(agent_id) == 20

    def test_single_token(self) -> None:
        agent_id = generate_agent_id_from_tokens([42])
        assert agent_id.startswith("msg_")


class TestParseToolCalls:
    """Tests for parse_tool_calls."""

    def test_no_tool_calls(self) -> None:
        """Text without tool calls should return unchanged."""
        text = "Hello, how can I help you?"
        remaining, calls = parse_tool_calls(text)
        assert remaining == text
        assert calls == []

    def test_single_tool_call(self) -> None:
        """Should extract a single tool call."""
        text = (
            "Let me read the file. "
            '{"tool_use": {"name": "read_file", "input": {"path": "test.py"}}}'
        )
        remaining, calls = parse_tool_calls(text)
        assert len(calls) == 1
        assert calls[0]["name"] == "read_file"
        assert calls[0]["input"] == {"path": "test.py"}
        assert "read_file" not in remaining
        assert "Let me read the file." in remaining

    def test_multiple_tool_calls(self) -> None:
        """Should extract multiple tool calls."""
        text = (
            '{"tool_use": {"name": "tool_a", "input": {"x": 1}}} '
            "and "
            '{"tool_use": {"name": "tool_b", "input": {"y": 2}}}'
        )
        remaining, calls = parse_tool_calls(text)
        assert len(calls) == 2
        names = {c["name"] for c in calls}
        assert names == {"tool_a", "tool_b"}

    def test_nested_json_in_input(self) -> None:
        """Should handle nested JSON in tool input."""
        text = '{"tool_use": {"name": "write", "input": {"data": {"nested": true}}}}'
        remaining, calls = parse_tool_calls(text)
        assert len(calls) == 1
        assert calls[0]["input"]["data"]["nested"] is True

    def test_tool_call_with_surrounding_text(self) -> None:
        """Should preserve text around tool calls."""
        text = 'Before {"tool_use": {"name": "test", "input": {}}} After'
        remaining, calls = parse_tool_calls(text)
        assert len(calls) == 1
        assert "Before" in remaining
        assert "After" in remaining

    def test_missing_name_field(self) -> None:
        """Tool call without name should be skipped."""
        text = '{"tool_use": {"input": {"x": 1}}}'
        remaining, calls = parse_tool_calls(text)
        assert calls == []
        # Text should still contain the JSON since it wasn't recognized
        assert "tool_use" in remaining

    def test_missing_input_field(self) -> None:
        """Tool call without input should be skipped."""
        text = '{"tool_use": {"name": "test"}}'
        remaining, calls = parse_tool_calls(text)
        assert calls == []

    def test_tool_use_not_dict(self) -> None:
        """tool_use value that isn't a dict should be skipped."""
        text = '{"tool_use": "not a dict"}'
        remaining, calls = parse_tool_calls(text)
        assert calls == []

    def test_empty_text(self) -> None:
        remaining, calls = parse_tool_calls("")
        assert remaining == ""
        assert calls == []


class TestAnthropicToOpenaiTools:
    """Tests for _anthropic_to_openai_tools with description hints and overrides."""

    def _make_edit_tool(self) -> dict:
        return {
            "name": "Edit",
            "description": "Performs exact string replacements in files.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "The absolute path to the file to modify",
                    },
                    "old_string": {
                        "type": "string",
                        "description": "The text to replace",
                    },
                    "new_string": {
                        "type": "string",
                        "description": "The text to replace it with",
                    },
                },
                "required": ["file_path", "old_string", "new_string"],
            },
        }

    def test_edit_tool_description_hint_appended(self) -> None:
        """Edit tool should get CRITICAL hint appended to description."""
        tools = [self._make_edit_tool()]
        result = _anthropic_to_openai_tools(tools)
        desc = result[0]["function"]["description"]
        assert "CRITICAL" in desc
        assert "character-for-character" in desc

    def test_edit_old_string_description_overridden(self) -> None:
        """old_string description should be overridden to be unambiguous."""
        tools = [self._make_edit_tool()]
        result = _anthropic_to_openai_tools(tools)
        props = result[0]["function"]["parameters"]["properties"]
        assert "EXACT text currently in the file" in props["old_string"]["description"]

    def test_edit_new_string_description_overridden(self) -> None:
        """new_string description should be overridden to be unambiguous."""
        tools = [self._make_edit_tool()]
        result = _anthropic_to_openai_tools(tools)
        props = result[0]["function"]["parameters"]["properties"]
        assert "replacement" in props["new_string"]["description"]

    def test_original_schema_not_mutated(self) -> None:
        """Original tool schema should not be modified (deep copy)."""
        tool = self._make_edit_tool()
        original_desc = tool["input_schema"]["properties"]["old_string"]["description"]
        _anthropic_to_openai_tools([tool])
        assert tool["input_schema"]["properties"]["old_string"]["description"] == original_desc

    def test_non_edit_tool_unchanged(self) -> None:
        """Non-Edit tools should pass through without modification."""
        tool = {
            "name": "Read",
            "description": "Read a file",
            "input_schema": {
                "type": "object",
                "properties": {"file_path": {"type": "string", "description": "Path"}},
            },
        }
        result = _anthropic_to_openai_tools([tool])
        assert result[0]["function"]["description"] == "Read a file"
        assert result[0]["function"]["parameters"]["properties"]["file_path"]["description"] == "Path"

    def test_file_path_description_preserved(self) -> None:
        """file_path description should not be overridden (no override defined)."""
        tools = [self._make_edit_tool()]
        result = _anthropic_to_openai_tools(tools)
        props = result[0]["function"]["parameters"]["properties"]
        assert props["file_path"]["description"] == "The absolute path to the file to modify"
