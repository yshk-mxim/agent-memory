# SPDX-License-Identifier: MIT
"""Tests for strip_thinking_tags — canonical tag/channel stripping."""

from agent_memory.application.text_cleaning import strip_thinking_tags


class TestGemmaChannelMarkers:
    def test_channel_thought(self):
        text = "<|channel>thought\n<channel|>The answer is 42."
        assert strip_thinking_tags(text) == "The answer is 42."

    def test_channel_tool_code(self):
        text = '<|channel>tool_code\n<channel|>{"name": "Read"}'
        assert strip_thinking_tags(text) == '{"name": "Read"}'

    def test_channel_without_newline(self):
        text = "<|channel>thought<channel|>result"
        assert strip_thinking_tags(text) == "result"

    def test_multiple_channel_markers(self):
        text = "<|channel>thought\n<channel|>first <|channel>tool\n<channel|>second"
        result = strip_thinking_tags(text)
        assert "<|channel>" not in result
        assert "<channel|>" not in result


class TestGemmaThinking:
    def test_end_of_thought(self):
        text = "<start_of_thought>reasoning here<end_of_thought>answer"
        assert strip_thinking_tags(text) == "answer"

    def test_slash_end_of_thought(self):
        text = "<start_of_thought>reasoning</end_of_thought>answer"
        assert strip_thinking_tags(text) == "answer"

    def test_orphaned_start(self):
        text = "<start_of_thought>incomplete thinking"
        result = strip_thinking_tags(text)
        assert "<start_of_thought>" not in result


class TestQwenThinking:
    def test_think_tags(self):
        text = "<think>internal reasoning</think>The answer is 42."
        assert strip_thinking_tags(text) == "The answer is 42."

    def test_orphaned_think(self):
        text = "<think>incomplete thinking"
        result = strip_thinking_tags(text)
        assert "<think>" not in result


class TestBareThoughtLabels:
    def test_thought_label_stripped(self):
        text = "thought\nThe answer is 42."
        assert strip_thinking_tags(text) == "The answer is 42."

    def test_underscore_thought_label(self):
        text = "_thought\nThe answer is 42."
        assert strip_thinking_tags(text) == "The answer is 42."

    def test_no_thought_label_unchanged(self):
        text = "The answer is 42."
        assert strip_thinking_tags(text) == "The answer is 42."


class TestQwen35ThinkingProcess:
    def test_thinking_process_prefix(self):
        text = "Thinking Process:\n1. First step\n2. Second step\nThe answer is 42."
        result = strip_thinking_tags(text)
        assert result == "The answer is 42."


class TestCombined:
    def test_channel_plus_thinking(self):
        """Channel markers stripped first, then thinking tags."""
        text = "<|channel>thought\n<channel|><start_of_thought>reasoning<end_of_thought>answer"
        assert strip_thinking_tags(text) == "answer"

    def test_no_tags_unchanged(self):
        text = "Plain text with no special tags."
        assert strip_thinking_tags(text) == text
