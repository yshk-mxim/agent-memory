# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Yakov Shkolnikov and contributors
"""Chat template adapter — model-specific tokenizer behavior.

Detects model family from tokenizer chat template strings and provides
appropriate configuration for message formatting and tokenization.
"""

from typing import Any

import structlog

logger = structlog.get_logger(__name__)


class ChatTemplateAdapter:
    """Determines chat template behavior based on model tokenizer.

    Detects model family (Llama, ChatML, GPT-OSS Harmony) from tokenizer
    chat template strings and provides model-appropriate configuration for
    message merging and template kwargs.
    """

    def is_deepseek(self, tokenizer: Any) -> bool:
        """Check if the tokenizer belongs to a DeepSeek model.

        DeepSeek models require special handling:
        - Assistant-role priming on first turn
        - T=0 (greedy) sampling for proper spacing and deterministic output
        - English-only instructions (bilingual Chinese/English model)
        - Speaker name prefix in assistant messages for turn coordination

        Args:
            tokenizer: The tokenizer to check

        Returns:
            True if DeepSeek model, False otherwise
        """
        chat_template = getattr(tokenizer, "chat_template", "") or ""
        return self._is_deepseek_format(chat_template)

    def needs_message_merging(self, tokenizer: Any) -> bool:
        """Whether consecutive same-role messages should be merged.

        Llama 3.1, ChatML, Gemma, and DeepSeek models require strict user/assistant
        alternation. GPT-OSS performs better with merged messages.
        """
        chat_template = getattr(tokenizer, "chat_template", "") or ""

        if "<|start_header_id|>" in chat_template:
            return True

        if "<|im_start|>" in chat_template:
            return True

        if self._is_harmony_format(chat_template):
            return True

        # Gemma: Uses <start_of_turn> markers and requires strict alternation
        # Consecutive user messages cause "Conversation roles must alternate" error
        if "<start_of_turn>" in chat_template and "<end_of_turn>" in chat_template:
            logger.info("Gemma detected: will merge consecutive messages")
            return True

        # DeepSeek: template uses 'User: ' and 'Assistant: ' labels
        # Consecutive user messages create "User:\n\nUser:\n\nUser:" which
        # confuses the model and causes it to generate Chinese or EOS
        if self._is_deepseek_format(chat_template):
            logger.info("DeepSeek detected: will merge consecutive messages")
            return True

        return False

    def get_template_kwargs(self, tokenizer: Any) -> dict[str, Any]:
        """Extra kwargs for tokenizer.apply_chat_template().

        Returns base kwargs plus model-specific overrides
        (e.g., reasoning_effort=low for GPT-OSS Harmony format).
        """
        chat_template = getattr(tokenizer, "chat_template", "") or ""

        kwargs: dict[str, Any] = {
            "tokenize": True,
            "add_generation_prompt": True,
        }

        if self._is_harmony_format(chat_template):
            kwargs["reasoning_effort"] = "low"
            logger.info("GPT-OSS Harmony: using reasoning_effort=low")

        return kwargs

    @staticmethod
    def _is_harmony_format(chat_template: str) -> bool:
        """Check if template uses GPT-OSS Harmony format."""
        return "<|channel|>" in chat_template and "<|start|>" in chat_template

    @staticmethod
    def _is_deepseek_format(chat_template: str) -> bool:
        """Check if template uses DeepSeek format (User:/Assistant: labels).

        DeepSeek-Coder-V2 uses simple role labels like:
          'User: ' + content + '\n\n'
          'Assistant: ' + content + eos_token

        Consecutive user messages create "User:\n\nUser:\n\n" which confuses
        the model, causing Chinese generation or immediate EOS.
        """
        # Check for the specific DeepSeek pattern with proper quotes
        return ("'User: '" in chat_template and "'Assistant: '" in chat_template)
