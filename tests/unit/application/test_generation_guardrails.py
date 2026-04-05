# SPDX-License-Identifier: MIT
"""Tests for generation_guardrails — tool retry loop detection."""

from agent_memory.application.generation_guardrails import (
    TOOL_RETRY_THRESHOLD,
    detect_tool_retry_loop,
)


def _make_assistant_tool_use(tool_name: str = "Edit") -> dict:
    return {
        "role": "assistant",
        "content": [
            {
                "type": "tool_use",
                "id": "toolu_abc",
                "name": tool_name,
                "input": {"old_string": "x", "new_string": "y", "file_path": "/f"},
            }
        ],
    }


def _make_user_tool_error(error_text: str = "String to replace not found") -> dict:
    return {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": "toolu_abc",
                "content": error_text,
                "is_error": True,
            }
        ],
    }


def _make_user_tool_success(text: str = "OK") -> dict:
    return {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": "toolu_abc",
                "content": text,
                "is_error": False,
            }
        ],
    }


class TestDetectToolRetryLoop:
    def test_no_loop_returns_none(self):
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        assert detect_tool_retry_loop(messages) is None

    def test_below_threshold_returns_none(self):
        messages = []
        for _ in range(TOOL_RETRY_THRESHOLD - 1):
            messages.append(_make_assistant_tool_use())
            messages.append(_make_user_tool_error())
        assert detect_tool_retry_loop(messages) is None

    def test_at_threshold_returns_hint(self):
        messages = []
        for _ in range(TOOL_RETRY_THRESHOLD):
            messages.append(_make_assistant_tool_use())
            messages.append(_make_user_tool_error())
        hint = detect_tool_retry_loop(messages)
        assert hint is not None
        assert "Read" in hint
        assert "old_string" in hint

    def test_above_threshold_returns_hint(self):
        messages = []
        for _ in range(TOOL_RETRY_THRESHOLD + 2):
            messages.append(_make_assistant_tool_use())
            messages.append(_make_user_tool_error())
        assert detect_tool_retry_loop(messages) is not None

    def test_success_breaks_loop(self):
        messages = []
        # 5 errors then a success then 2 errors
        for _ in range(5):
            messages.append(_make_assistant_tool_use())
            messages.append(_make_user_tool_error())
        messages.append(_make_assistant_tool_use())
        messages.append(_make_user_tool_success())
        messages.append(_make_assistant_tool_use())
        messages.append(_make_user_tool_error())
        messages.append(_make_assistant_tool_use())
        messages.append(_make_user_tool_error())
        # Only 2 errors since the success — below threshold
        assert detect_tool_retry_loop(messages) is None

    def test_different_tool_breaks_loop(self):
        messages = []
        messages.append(_make_assistant_tool_use("Edit"))
        messages.append(_make_user_tool_error())
        messages.append(_make_assistant_tool_use("Edit"))
        messages.append(_make_user_tool_error())
        messages.append(_make_assistant_tool_use("Write"))  # Different tool
        messages.append(_make_user_tool_error())
        # Different tools — not a retry loop
        assert detect_tool_retry_loop(messages) is None

    def test_generic_tool_gets_default_hint(self):
        messages = []
        for _ in range(TOOL_RETRY_THRESHOLD):
            messages.append(_make_assistant_tool_use("SomeCustomTool"))
            messages.append(_make_user_tool_error("Generic error"))
        hint = detect_tool_retry_loop(messages)
        assert hint is not None
        assert "different approach" in hint

    def test_empty_messages(self):
        assert detect_tool_retry_loop([]) is None
