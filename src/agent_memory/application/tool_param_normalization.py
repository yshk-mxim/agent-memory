# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Yakov Shkolnikov and contributors
"""Normalize model-hallucinated tool parameters to match actual tool schemas.

Models (especially smaller ones like Gemma 4 26B MoE) produce tool calls
with wrong parameter names — "instructions" instead of "prompt" for Agent,
"search_query" instead of "query" for WebSearch, etc.

This module centralizes all such normalization in one place rather than
scattering ad-hoc fixes across parsers.

Architecture layer: application (pure logic, no I/O).
"""

from __future__ import annotations

from typing import Any


# Known parameter name mismatches models produce.
# Mapping: tool_name → {hallucinated_param: canonical_param}
_PARAM_ALIASES: dict[str, dict[str, str]] = {
    "Agent": {
        "instructions": "prompt",
        "task": "prompt",
        "input": "prompt",
    },
    "WebSearch": {
        "search_query": "query",
    },
    "WebFetch": {
        "link": "url",
    },
    "AskUserQuestion": {
        "question": "questions",
    },
    "Edit": {
        "search": "old_string",
        "replace": "new_string",
        "path": "file_path",
    },
    "Write": {
        "path": "file_path",
        "text": "content",
    },
    "Read": {
        "path": "file_path",
    },
    "Glob": {
        "glob": "pattern",
    },
    "Grep": {
        "regex": "pattern",
        "search": "pattern",
        "directory": "path",
    },
}

# Tools that require a "description" field — auto-generate if missing.
_NEEDS_DESCRIPTION = frozenset({"Agent"})


def normalize_tool_params(name: str, params: dict[str, Any]) -> dict[str, Any]:
    """Normalize hallucinated parameter names to match actual tool schemas.

    Args:
        name: Tool name (e.g. "Agent", "WebSearch").
        params: Raw parameters from parsed tool call.

    Returns:
        Parameters with canonical names.
    """
    aliases = _PARAM_ALIASES.get(name)
    if aliases:
        normalized: dict[str, Any] = {}
        for k, v in params.items():
            normalized[aliases.get(k, k)] = v
        params = normalized

    if name in _NEEDS_DESCRIPTION and "description" not in params:
        prompt_preview = str(params.get("prompt", ""))[:50]
        params["description"] = prompt_preview or "agent task"

    # AskUserQuestion: models generate question(str) instead of questions(list)
    if name == "AskUserQuestion":
        params = _normalize_ask_user_question(params)

    return params


def _normalize_ask_user_question(params: dict[str, Any]) -> dict[str, Any]:
    """Restructure AskUserQuestion params into expected schema.

    Models generate: {"question": "...", "options": [...], "multiSelect": bool}
    Expected:        {"questions": [{"question": "...", "options": [...], ...}]}
    """
    questions = params.get("questions")

    # Already correct format
    if isinstance(questions, list) and questions:
        return params

    # Single question string → wrap in questions array
    if isinstance(questions, str):
        q_obj: dict[str, Any] = {"question": questions}
        if "options" in params:
            q_obj["options"] = params.pop("options")
        if "multiSelect" in params:
            q_obj["multiSelect"] = params.pop("multiSelect")
        params["questions"] = [q_obj]
        return params

    # question (singular) at top level → restructure
    question = params.pop("question", None)
    if question:
        q_obj = {"question": question}
        if "options" in params:
            q_obj["options"] = params.pop("options")
        if "multiSelect" in params:
            q_obj["multiSelect"] = params.pop("multiSelect")
        params["questions"] = [q_obj]

    return params
