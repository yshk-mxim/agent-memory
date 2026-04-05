# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Yakov Shkolnikov and contributors
"""Model output text cleaning — strip thinking/channel tags and leaked tokens.

Canonical location for all tag stripping logic. Adapters and services
should import from here, not reimplement.

Architecture layer: application (pure logic, no I/O).
"""

from __future__ import annotations

import re

# Pre-compiled patterns for channel markers
_CHANNEL_START_RE = re.compile(r"<\|channel\>\w*\n?")
_CHANNEL_END_RE = re.compile(r"<channel\|>")
_BARE_THOUGHT_RE = re.compile(r"(?m)^_?thought\s*$\n?")

# HTML formatting tokens that Gemma 4 leaks into tool call arguments.
# These are dedicated vocabulary tokens that the model generates randomly
# in JSON string values during grammar-constrained tool calling.
# See: https://github.com/ggml-org/llama.cpp/issues/21316
_LEAKED_HTML_RE = re.compile(
    r"</?(?:strong|em|b|i|u|mark|del|ins|sub|sup|small|big)>"
)


def strip_thinking_tags(text: str) -> str:
    """Strip reasoning/channel tags from model output.

    Canonical implementation — all other modules should import this.

    Handles:
    - Gemma 4 channel markers: <|channel>thought / <channel|>
    - Gemma 4 thinking: <start_of_thought>...<end_of_thought>
    - Qwen3: <think>...</think>
    - Qwen3.5 non-tag: "Thinking Process:" prefix
    - Bare "thought" labels (model mimics Claude thinking format)

    Returns the final answer after stripping, or original text if no tags.
    """
    # Gemma 4 channel markers (e.g. "<|channel>thought\n<channel|>...")
    text = _CHANNEL_START_RE.sub("", text)
    text = _CHANNEL_END_RE.sub("", text)

    # Gemma 4 thinking (handles both <end_of_thought> and </end_of_thought>)
    for end_tag in ("<end_of_thought>", "</end_of_thought>"):
        if end_tag in text:
            return text.rsplit(end_tag, 1)[-1].strip()
    if "<start_of_thought>" in text:
        return text.replace("<start_of_thought>", "").strip()

    # Qwen3 format
    if "</think>" in text:
        return text.rsplit("</think>", 1)[-1].strip()
    if text.startswith("<think>"):
        return text.replace("<think>", "").strip()

    # Qwen3.5 non-tag: "Thinking Process:" prefix
    if text.startswith("Thinking Process:"):
        lines = text.split("\n")
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped and i > 0:
                is_reasoning = stripped[0].isdigit() or stripped.startswith(("*", "-", "Thinking"))
                if not is_reasoning:
                    return "\n".join(lines[i:]).strip()

    # Bare "thought" labels
    cleaned = _BARE_THOUGHT_RE.sub("", text)
    if cleaned != text:
        return cleaned.strip()
    return text


def strip_leaked_html(text: str) -> str:
    """Remove HTML formatting tokens leaked by Gemma 4 into tool call args.

    Gemma 4's vocabulary includes single-token HTML tags (``<strong>``,
    ``</strong>``, etc.) from web training data.  During grammar-constrained
    tool calling, these tokens occasionally leak into JSON string values,
    corrupting parameters like ``old_string`` and causing Edit failures.

    Only removes formatting-only tags — structural HTML (div, p, span,
    table, etc.) is left alone since those could appear in legitimate
    file content being edited.
    """
    return _LEAKED_HTML_RE.sub("", text)
