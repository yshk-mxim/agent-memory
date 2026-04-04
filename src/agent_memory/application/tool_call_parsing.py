# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Yakov Shkolnikov and contributors
"""Tool call parsing protocol and chain.

Defines the ``ToolCallParser`` protocol that model-specific parsers
implement, and ``ToolCallParserChain`` which tries parsers in priority
order until one extracts tool calls.

Architecture (hexagonal):
    Protocol lives in application layer (it's a port).
    Parser implementations are outbound adapters (model-specific wire formats).
    ParsedToolCall is a domain value object.
"""

import logging
from typing import Protocol

from agent_memory.domain.value_objects import ParsedToolCall

logger = logging.getLogger(__name__)


class ToolCallParser(Protocol):
    """Single-format parser for extracting tool calls from model text output.

    Implementations are model-specific adapters that know how to parse
    one particular tool call format (Gemma native, Qwen XML, JSON, etc.).
    """

    def parse(self, text: str) -> tuple[str, list[ParsedToolCall]]:
        """Extract tool calls from text.

        Args:
            text: Model output text that may contain tool calls.

        Returns:
            Tuple of (remaining_text_with_calls_stripped, parsed_calls).
            If no tool calls found, returns (text, []).
        """
        ...


class ToolCallParserChain:
    """Tries parsers in priority order. First one that extracts calls wins.

    Each model gets an ordered list of parsers via the factory.
    The JSON fallback is always last.

    Example:
        chain = ToolCallParserChain([GemmaToolCallParser(), JsonLinesToolCallParser()])
        remaining, calls = chain.parse(model_output_text)
    """

    def __init__(self, parsers: list[ToolCallParser]) -> None:
        if not parsers:
            raise ValueError("ToolCallParserChain requires at least one parser")
        self._parsers = parsers

    def parse(self, text: str) -> tuple[str, list[ParsedToolCall]]:
        """Try each parser in order. First one that finds calls wins."""
        for parser in self._parsers:
            remaining, calls = parser.parse(text)
            if calls:
                parser_name = type(parser).__name__
                logger.info(
                    "tool_calls_parsed: parser=%s count=%d",
                    parser_name,
                    len(calls),
                )
                return remaining, calls
        return text, []
