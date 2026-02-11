# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Yakov Shkolnikov and contributors
"""Unit tests for tokenization utilities.

Verifies tokenize_with_chat_template behavior for different model families
using mock tokenizers.
"""

from unittest.mock import MagicMock

import pytest

from agent_memory.application.tokenization import (
    _is_gpt_oss_tokenizer,
    tokenize_with_chat_template,
)

pytestmark = pytest.mark.unit


def _make_tokenizer(
    chat_template: str | None = None,
    apply_result_tokens: list[int] | None = None,
    apply_result_text: str | None = None,
) -> MagicMock:
    """Create a mock tokenizer with configurable chat template behavior."""
    tok = MagicMock()
    tok.chat_template = chat_template
    tok.encode.return_value = [99, 98, 97]  # fallback tokens

    if chat_template is not None and apply_result_tokens is not None:

        def side_effect(messages, **kwargs):
            if kwargs.get("tokenize", True):
                return apply_result_tokens
            return apply_result_text or "templated text"

        tok.apply_chat_template.side_effect = side_effect

    return tok


# ── _is_gpt_oss_tokenizer() ────────────────────────────────────────


class TestIsGptOssTokenizer:
    def test_harmony_format_detected(self) -> None:
        tok = _make_tokenizer("<|channel|>reasoning<|start|>template")
        assert _is_gpt_oss_tokenizer(tok) is True

    def test_partial_markers_not_detected(self) -> None:
        tok = _make_tokenizer("<|channel|>only no start")
        assert _is_gpt_oss_tokenizer(tok) is False

    def test_none_template(self) -> None:
        tok = _make_tokenizer(None)
        assert _is_gpt_oss_tokenizer(tok) is False

    def test_empty_template(self) -> None:
        tok = _make_tokenizer("")
        assert _is_gpt_oss_tokenizer(tok) is False

    def test_no_chat_template_attr(self) -> None:
        tok = MagicMock(spec=[])
        assert _is_gpt_oss_tokenizer(tok) is False


# ── tokenize_with_chat_template() ──────────────────────────────────


class TestTokenizeWithChatTemplate:
    def test_no_chat_template_uses_fallback(self) -> None:
        tok = _make_tokenizer(None)
        tokens, text = tokenize_with_chat_template(
            tok, [{"role": "user", "content": "hi"}], "fallback"
        )
        tok.encode.assert_called_once_with("fallback")
        assert tokens == [99, 98, 97]
        assert text == "fallback"

    def test_empty_chat_template_uses_fallback(self) -> None:
        tok = _make_tokenizer("")
        tokens, text = tokenize_with_chat_template(
            tok, [{"role": "user", "content": "hi"}], "fallback"
        )
        tok.encode.assert_called_once_with("fallback")
        assert text == "fallback"

    def test_valid_template_returns_tokens_and_text(self) -> None:
        tok = _make_tokenizer(
            chat_template="simple {{message}} template",
            apply_result_tokens=[10, 20, 30],
            apply_result_text="<turn>user: hi</turn>",
        )
        messages = [{"role": "user", "content": "hi"}]
        tokens, text = tokenize_with_chat_template(tok, messages, "fallback")

        assert tokens == [10, 20, 30]
        assert text == "<turn>user: hi</turn>"
        # apply_chat_template should be called twice (tokens + text)
        assert tok.apply_chat_template.call_count == 2

    def test_gemma_merges_consecutive_user_messages(self) -> None:
        tok = _make_tokenizer(
            chat_template="<start_of_turn>user\n{{msg}}<end_of_turn>",
            apply_result_tokens=[1, 2],
            apply_result_text="merged text",
        )
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "user", "content": "How are you?"},
        ]
        tokens, text = tokenize_with_chat_template(tok, messages, "fb")
        # The merged messages should have been passed to apply_chat_template
        call_args = tok.apply_chat_template.call_args_list[0]
        passed_messages = call_args[0][0]
        # Two user messages merged into one
        assert len(passed_messages) == 1
        assert "Hello" in passed_messages[0]["content"]
        assert "How are you?" in passed_messages[0]["content"]

    def test_gemma_system_merged_into_user(self) -> None:
        tok = _make_tokenizer(
            chat_template="<start_of_turn>user\n{{msg}}<end_of_turn>",
            apply_result_tokens=[1, 2],
            apply_result_text="text",
        )
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hi"},
        ]
        tokens, _ = tokenize_with_chat_template(tok, messages, "fb")
        call_args = tok.apply_chat_template.call_args_list[0]
        passed_messages = call_args[0][0]
        # system + user merged: system becomes user role
        assert len(passed_messages) == 1
        assert passed_messages[0]["role"] == "user"
        assert "You are helpful" in passed_messages[0]["content"]
        assert "Hi" in passed_messages[0]["content"]

    def test_gemma_system_followed_by_assistant(self) -> None:
        """Gemma: system followed by non-user gets role changed to user."""
        tok = _make_tokenizer(
            chat_template="<start_of_turn>user\n{{msg}}<end_of_turn>",
            apply_result_tokens=[1, 2],
            apply_result_text="text",
        )
        messages = [
            {"role": "system", "content": "Rules"},
            {"role": "assistant", "content": "Sure"},
        ]
        tokens, _ = tokenize_with_chat_template(tok, messages, "fb")
        call_args = tok.apply_chat_template.call_args_list[0]
        passed_messages = call_args[0][0]
        # system becomes user, assistant stays separate
        assert len(passed_messages) == 2
        assert passed_messages[0]["role"] == "user"
        assert passed_messages[1]["role"] == "assistant"

    def test_deepseek_merges_consecutive_user_messages(self) -> None:
        tok = _make_tokenizer(
            chat_template="{% set 'User: ' %}{% set 'Assistant: ' %}",
            apply_result_tokens=[5, 6],
            apply_result_text="ds text",
        )
        messages = [
            {"role": "user", "content": "A"},
            {"role": "user", "content": "B"},
        ]
        tokens, text = tokenize_with_chat_template(tok, messages, "fb")
        call_args = tok.apply_chat_template.call_args_list[0]
        passed_messages = call_args[0][0]
        assert len(passed_messages) == 1
        assert passed_messages[0]["content"] == "A\nB"

    def test_deepseek_no_system_merge(self) -> None:
        """DeepSeek does NOT merge system into user (only Gemma does)."""
        tok = _make_tokenizer(
            chat_template="{% set 'User: ' %}{% set 'Assistant: ' %}",
            apply_result_tokens=[5, 6],
            apply_result_text="ds text",
        )
        messages = [
            {"role": "system", "content": "Rules"},
            {"role": "user", "content": "Hi"},
        ]
        tokens, _ = tokenize_with_chat_template(tok, messages, "fb")
        call_args = tok.apply_chat_template.call_args_list[0]
        passed_messages = call_args[0][0]
        # system stays separate for DeepSeek
        assert len(passed_messages) == 2
        assert passed_messages[0]["role"] == "system"

    def test_gpt_oss_adds_reasoning_effort(self) -> None:
        tok = _make_tokenizer(
            chat_template="<|channel|>reasoning<|start|>template",
            apply_result_tokens=[7, 8],
            apply_result_text="gpt oss text",
        )
        messages = [{"role": "user", "content": "hi"}]
        tokens, text = tokenize_with_chat_template(tok, messages, "fb")
        # Check that reasoning_effort=low was passed
        call_kwargs = tok.apply_chat_template.call_args_list[0][1]
        assert call_kwargs.get("reasoning_effort") == "low"

    def test_template_exception_falls_back(self) -> None:
        tok = MagicMock()
        tok.chat_template = "valid template"
        tok.apply_chat_template.side_effect = RuntimeError("template error")
        tok.encode.return_value = [99, 98]

        tokens, text = tokenize_with_chat_template(
            tok, [{"role": "user", "content": "hi"}], "fallback text"
        )
        assert tokens == [99, 98]
        assert text == "fallback text"

    def test_non_list_tokens_falls_back(self) -> None:
        """If apply_chat_template returns non-list tokens, fall back."""
        tok = MagicMock()
        tok.chat_template = "valid template"

        def side_effect(messages, **kwargs):
            if kwargs.get("tokenize", True):
                return "not a list"  # invalid token result
            return "templated text"

        tok.apply_chat_template.side_effect = side_effect
        tok.encode.return_value = [99]

        tokens, text = tokenize_with_chat_template(
            tok, [{"role": "user", "content": "hi"}], "fallback"
        )
        assert tokens == [99]
        assert text == "fallback"

    def test_non_str_text_falls_back(self) -> None:
        """If apply_chat_template returns non-str text, fall back."""
        tok = MagicMock()
        tok.chat_template = "valid template"

        def side_effect(messages, **kwargs):
            if kwargs.get("tokenize", True):
                return [1, 2, 3]
            return [1, 2, 3]  # not a string

        tok.apply_chat_template.side_effect = side_effect
        tok.encode.return_value = [99]

        tokens, text = tokenize_with_chat_template(
            tok, [{"role": "user", "content": "hi"}], "fallback"
        )
        assert tokens == [99]
        assert text == "fallback"

    def test_mixed_roles_not_merged_for_plain_template(self) -> None:
        """Non-DeepSeek/non-Gemma template: no message merging."""
        tok = _make_tokenizer(
            chat_template="plain {{message}} template",
            apply_result_tokens=[1, 2],
            apply_result_text="text",
        )
        messages = [
            {"role": "user", "content": "A"},
            {"role": "user", "content": "B"},
        ]
        tokens, text = tokenize_with_chat_template(tok, messages, "fb")
        # Messages should be passed as-is (no merging)
        call_args = tok.apply_chat_template.call_args_list[0]
        passed_messages = call_args[0][0]
        assert len(passed_messages) == 2

    def test_empty_messages_no_crash(self) -> None:
        tok = _make_tokenizer(
            chat_template="<start_of_turn>user<end_of_turn>",
            apply_result_tokens=[],
            apply_result_text="",
        )
        tokens, text = tokenize_with_chat_template(tok, [], "fb")
        assert isinstance(tokens, list)
