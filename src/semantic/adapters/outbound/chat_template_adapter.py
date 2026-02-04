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

    def needs_message_merging(self, tokenizer: Any) -> bool:
        """Whether consecutive same-role messages should be merged.

        Llama 3.1 and ChatML models require strict user/assistant
        alternation. GPT-OSS performs better with merged messages.
        Gemma handles consecutive messages natively.
        """
        chat_template = getattr(tokenizer, "chat_template", "") or ""

        if "<|start_header_id|>" in chat_template:
            return True

        if "<|im_start|>" in chat_template:
            return True

        if self._is_harmony_format(chat_template):
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
