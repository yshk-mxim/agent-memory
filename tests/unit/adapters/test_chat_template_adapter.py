# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Yakov Shkolnikov and contributors
"""Unit tests for ChatTemplateAdapter.

Verifies model family detection and template configuration from tokenizer
chat template strings.
"""

from unittest.mock import MagicMock

import pytest

from agent_memory.adapters.outbound.chat_template_adapter import ChatTemplateAdapter

pytestmark = pytest.mark.unit


@pytest.fixture
def adapter():
    return ChatTemplateAdapter()


def _make_tokenizer(chat_template: str | None = None) -> MagicMock:
    tok = MagicMock()
    tok.chat_template = chat_template
    return tok


# ── is_deepseek() ──────────────────────────────────────────────────


class TestIsDeepSeek:

    def test_deepseek_template_detected(self, adapter: ChatTemplateAdapter) -> None:
        tok = _make_tokenizer("{% if 'User: ' in message %}some 'User: ' and 'Assistant: ' template{% endif %}")
        assert adapter.is_deepseek(tok) is True

    def test_gemma_template_not_deepseek(self, adapter: ChatTemplateAdapter) -> None:
        tok = _make_tokenizer("<start_of_turn>user\n{{message}}<end_of_turn>")
        assert adapter.is_deepseek(tok) is False

    def test_none_template(self, adapter: ChatTemplateAdapter) -> None:
        tok = _make_tokenizer(None)
        assert adapter.is_deepseek(tok) is False

    def test_empty_template(self, adapter: ChatTemplateAdapter) -> None:
        tok = _make_tokenizer("")
        assert adapter.is_deepseek(tok) is False

    def test_no_chat_template_attr(self, adapter: ChatTemplateAdapter) -> None:
        tok = MagicMock(spec=[])  # no attributes
        assert adapter.is_deepseek(tok) is False


# ── needs_message_merging() ─────────────────────────────────────────


class TestNeedsMessageMerging:

    def test_llama_header_detected(self, adapter: ChatTemplateAdapter) -> None:
        tok = _make_tokenizer("<|start_header_id|>system<|end_header_id|>")
        assert adapter.needs_message_merging(tok) is True

    def test_chatml_detected(self, adapter: ChatTemplateAdapter) -> None:
        tok = _make_tokenizer("<|im_start|>system\n{{message}}<|im_end|>")
        assert adapter.needs_message_merging(tok) is True

    def test_gemma_detected(self, adapter: ChatTemplateAdapter) -> None:
        tok = _make_tokenizer("<start_of_turn>user\n{{message}}<end_of_turn>")
        assert adapter.needs_message_merging(tok) is True

    def test_deepseek_detected(self, adapter: ChatTemplateAdapter) -> None:
        tok = _make_tokenizer("template with 'User: ' and 'Assistant: ' labels")
        assert adapter.needs_message_merging(tok) is True

    def test_harmony_detected(self, adapter: ChatTemplateAdapter) -> None:
        tok = _make_tokenizer("<|channel|>reasoning<|start|>")
        assert adapter.needs_message_merging(tok) is True

    def test_plain_template_no_merge(self, adapter: ChatTemplateAdapter) -> None:
        tok = _make_tokenizer("{{message}}")
        assert adapter.needs_message_merging(tok) is False

    def test_none_template_no_merge(self, adapter: ChatTemplateAdapter) -> None:
        tok = _make_tokenizer(None)
        assert adapter.needs_message_merging(tok) is False


# ── get_template_kwargs() ───────────────────────────────────────────


class TestGetTemplateKwargs:

    def test_base_kwargs_always_present(self, adapter: ChatTemplateAdapter) -> None:
        tok = _make_tokenizer("{{message}}")
        kwargs = adapter.get_template_kwargs(tok)
        assert kwargs["tokenize"] is True
        assert kwargs["add_generation_prompt"] is True
        assert "reasoning_effort" not in kwargs

    def test_harmony_adds_reasoning_effort(self, adapter: ChatTemplateAdapter) -> None:
        tok = _make_tokenizer("<|channel|>reasoning<|start|>some template")
        kwargs = adapter.get_template_kwargs(tok)
        assert kwargs["reasoning_effort"] == "low"
        assert kwargs["tokenize"] is True

    def test_deepseek_no_reasoning_effort(self, adapter: ChatTemplateAdapter) -> None:
        tok = _make_tokenizer("'User: ' and 'Assistant: ' labels")
        kwargs = adapter.get_template_kwargs(tok)
        assert "reasoning_effort" not in kwargs


# ── Static helpers ──────────────────────────────────────────────────


class TestStaticHelpers:

    def test_is_harmony_format_true(self) -> None:
        assert ChatTemplateAdapter._is_harmony_format("<|channel|>r<|start|>") is True

    def test_is_harmony_format_partial(self) -> None:
        assert ChatTemplateAdapter._is_harmony_format("<|channel|>only") is False

    def test_is_harmony_format_empty(self) -> None:
        assert ChatTemplateAdapter._is_harmony_format("") is False

    def test_is_deepseek_format_true(self) -> None:
        assert ChatTemplateAdapter._is_deepseek_format("'User: ' and 'Assistant: '") is True

    def test_is_deepseek_format_wrong_quotes(self) -> None:
        # Must be single-quoted User:/Assistant: (matches Jinja template string)
        assert ChatTemplateAdapter._is_deepseek_format("User: and Assistant:") is False

    def test_is_deepseek_format_empty(self) -> None:
        assert ChatTemplateAdapter._is_deepseek_format("") is False
