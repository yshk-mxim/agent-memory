# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Yakov Shkolnikov and contributors
"""Model-aware tool call parser package.

Provides ``create_parser_for_model()`` factory that returns the right
``ToolCallParserChain`` for any model_id. Extend by adding entries
to ``_MODEL_PARSERS``.
"""

from agent_memory.adapters.outbound.tool_call_parsers.gemma import (
    GemmaToolCallParser,
)
from agent_memory.adapters.outbound.tool_call_parsers.json_lines import (
    JsonLinesToolCallParser,
)
from agent_memory.adapters.outbound.tool_call_parsers.mistral import (
    MistralToolCallParser,
)
from agent_memory.adapters.outbound.tool_call_parsers.qwen import (
    QwenToolCallParser,
)
from agent_memory.application.tool_call_parsing import ToolCallParserChain

__all__ = [
    "GemmaToolCallParser",
    "JsonLinesToolCallParser",
    "MistralToolCallParser",
    "QwenToolCallParser",
    "ToolCallParserChain",
    "create_parser_for_model",
]

# Model family → parser chain factory (ordered by priority within each chain).
# JsonLinesToolCallParser is always last as the universal fallback.
_MODEL_PARSERS: dict[str, type] = {
    "gemma": GemmaToolCallParser,
    "qwen": QwenToolCallParser,
    "hermes": QwenToolCallParser,  # Same <tool_call> tag format
    "functionary": QwenToolCallParser,
    "mistral": MistralToolCallParser,
}


def create_parser_for_model(model_id: str) -> ToolCallParserChain:
    """Factory: returns appropriate parser chain for the model.

    The chain always includes ``JsonLinesToolCallParser`` as a fallback.
    For known model families, the native parser is tried first.

    Args:
        model_id: Model identifier (e.g. "gemma-4-26b-a4b", "qwen3-coder-next").

    Returns:
        ToolCallParserChain with model-appropriate parsers.
    """
    model_lower = model_id.lower()
    fallback = JsonLinesToolCallParser()

    for family, parser_cls in _MODEL_PARSERS.items():
        if family in model_lower:
            return ToolCallParserChain([parser_cls(), fallback])

    # Unknown model — JSON fallback only
    return ToolCallParserChain([fallback])
