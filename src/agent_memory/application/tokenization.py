# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Yakov Shkolnikov and contributors
"""Tokenization utilities for chat template handling.

Provides tokenize_with_chat_template for the application layer, avoiding
direct adapter-to-application import violations.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def _is_gpt_oss_tokenizer(tokenizer: Any) -> bool:
    """Check if tokenizer uses GPT-OSS Harmony format.

    GPT-OSS models have <|channel|> markers in their chat template.
    """
    chat_template = getattr(tokenizer, "chat_template", "") or ""
    return "<|channel|>" in chat_template and "<|start|>" in chat_template


def tokenize_with_chat_template(
    tokenizer: Any,
    chat_messages: list[dict[str, str]],
    fallback_text: str,
) -> tuple[list[int], str]:
    """Tokenize using model's native chat template when available.

    Models like Gemma3 use special turn markers (<start_of_turn>user, etc.)
    that are critical for proper identity handling in multi-turn conversations.
    Falls back to raw text tokenization if no chat template is available.

    For GPT-OSS models, uses reasoning_effort="low" to minimize analysis
    channel output and prevent the model from getting stuck in analysis mode.

    Merges consecutive same-role messages for DeepSeek/Llama models
    that require strict user/assistant alternation. Without this, consecutive
    "User:" labels confuse the model and cause broken tokenization.

    Args:
        tokenizer: HuggingFace-compatible tokenizer.
        chat_messages: List of {"role": ..., "content": ...} dicts.
        fallback_text: Pre-formatted text to use if no chat template.

    Returns:
        Tuple of (token_ids, prompt_text) where prompt_text is the ACTUAL
        templated text that was tokenized (for cache prefix matching).
    """
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        try:
            # Merge consecutive USER messages for models requiring alternation.
            # DeepSeek and Gemma templates require strict user/assistant alternation.
            # Only merge "user" role, NOT "assistant" — assistant messages
            # are the agent's own voice and must remain separate.
            chat_template = getattr(tokenizer, "chat_template", "") or ""
            is_deepseek = ("'User: '" in chat_template and "'Assistant: '" in chat_template)
            is_gemma = ("<start_of_turn>" in chat_template and "<end_of_turn>" in chat_template)

            messages_for_template = chat_messages
            if (is_deepseek or is_gemma) and chat_messages:
                merged = []
                for msg in chat_messages:
                    if not merged:
                        merged.append(dict(msg))
                        continue
                    prev = merged[-1]
                    if is_gemma and prev["role"] == "system" and msg["role"] == "user":
                        prev["role"] = "user"
                        prev["content"] = prev["content"] + "\n" + msg["content"]
                    elif is_gemma and prev["role"] == "system":
                        prev["role"] = "user"
                        merged.append(dict(msg))
                    elif msg["role"] == "user" and prev["role"] == "user":
                        prev["content"] = prev["content"] + "\n" + msg["content"]
                    else:
                        merged.append(dict(msg))
                messages_for_template = merged
                model_type = "DeepSeek" if is_deepseek else "Gemma" if is_gemma else "Unknown"
                logger.info(
                    f"{model_type} tokenization: merged {len(chat_messages)} messages → "
                    f"{len(merged)} (collapsed consecutive user messages)"
                )

            template_kwargs: dict[str, Any] = {
                "tokenize": True,
                "add_generation_prompt": True,
            }
            if _is_gpt_oss_tokenizer(tokenizer):
                template_kwargs["reasoning_effort"] = "low"
                logger.debug("GPT-OSS detected, using reasoning_effort=low")

            # Get the ACTUAL templated text that will be tokenized
            # to ensure prompt_text matches tokens for cache prefix matching
            template_kwargs_text = template_kwargs.copy()
            template_kwargs_text["tokenize"] = False
            templated_text = tokenizer.apply_chat_template(messages_for_template, **template_kwargs_text)

            tokens = tokenizer.apply_chat_template(messages_for_template, **template_kwargs)
            logger.info(f"Template result: tokens_type={type(tokens).__name__}, tokens_is_list={isinstance(tokens, list)}, text_type={type(templated_text).__name__}, text_is_str={isinstance(templated_text, str)}")
            if isinstance(tokens, list) and isinstance(templated_text, str):
                logger.info(f"Template applied: {len(tokens)} tokens, {len(templated_text)} chars")
                return tokens, templated_text
            else:
                logger.warning(f"Template check failed: returning fallback (tokens={type(tokens).__name__}, text={type(templated_text).__name__})")
        except Exception as e:
            logger.warning(f"Chat template exception: {e}, falling back to raw text tokenization")
    return tokenizer.encode(fallback_text), fallback_text
