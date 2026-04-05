# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Yakov Shkolnikov and contributors
"""Compress Claude Code tool definitions for local model context efficiency.

Claude Code sends ~15 tools with full JSON schemas (~6,000-10,000 tokens).
This module compresses them to ~800-1,200 tokens while preserving the
information local models need for reliable tool calling.

Architecture layer: inbound adapter (transforms client wire format).
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Hand-written compact descriptions for Claude Code's known tools.
# These are much shorter than the full JSON schemas Claude Code sends
# while preserving the critical info: name, purpose, required params.
KNOWN_TOOLS: dict[str, str] = {
    "Read": (
        "Read file contents. "
        "Params: file_path (str, required, absolute path), "
        "offset (int, start line), limit (int, max lines)"
    ),
    "Write": (
        "Write/create file (overwrites). "
        "Params: file_path (str, required, absolute path), "
        "content (str, required, full file content)"
    ),
    "Edit": (
        "Replace exact string in file. "
        "Params: file_path (str, required), old_string (str, required, must be unique), "
        "new_string (str, required), replace_all (bool, default false)"
    ),
    "Bash": (
        "Run shell command. "
        "Params: command (str, required), timeout (int, ms, max 600000), "
        "description (str, short summary)"
    ),
    "Glob": (
        "Find files by glob pattern. "
        'Params: pattern (str, required, e.g. "**/*.py"), path (str, directory)'
    ),
    "Grep": (
        "Search file contents with regex. "
        "Params: pattern (str, required), path (str), glob (str), "
        'output_mode ("files_with_matches"|"content"|"count"), '
        "-n (bool, line numbers), -A/-B/-C (int, context lines)"
    ),
    "Agent": (
        "Launch subagent for complex tasks. "
        "Params: prompt (str, required), description (str, required, 3-5 words), "
        "subagent_type (str)"
    ),
    "TodoWrite": (
        "Track task progress. "
        "Params: todos (array, required, each: {content: str, status: "
        '"pending"|"in_progress"|"completed", activeForm: str})'
    ),
    "WebSearch": "Search the web. Params: query (str, required)",
    "WebFetch": ("Fetch URL content. Params: url (str, required), prompt (str, required)"),
    "AskUserQuestion": ("Ask user a question. Params: question (str, required)"),
    "NotebookEdit": (
        "Edit Jupyter notebook cell. "
        "Params: notebook_path (str, required), cell_index (int, required), "
        "new_source (str, required)"
    ),
    "Skill": ("Invoke a skill/slash command. Params: skill (str, required), args (str)"),
    "ToolSearch": ("Fetch deferred tool schemas. Params: query (str, required), max_results (int)"),
}

# Instruction header (kept minimal to not waste context)
_HEADER = (
    "You have access to the following tools. "
    "To call a tool, output a JSON object on its own line:\n"
    '{"name": "tool_name", "parameters": {"param": "value"}}\n'
    "Multiple calls = multiple JSON objects, one per line. "
    "Do NOT wrap in markdown code blocks."
)


def compress_tool_definitions(tools: list[dict[str, Any]]) -> str:
    """Compress tool definitions to ~800-1200 tokens.

    For known Claude Code tools, uses pre-written compact descriptions.
    For unknown tools, extracts name + required params from the JSON schema.

    Args:
        tools: List of tool dicts (Anthropic format with name, description,
            input_schema) or OpenAI format (type: function, function: {...}).

    Returns:
        Compact multi-line string suitable for injection into system message.
    """
    lines = [_HEADER, "\nAvailable tools:"]
    seen: set[str] = set()

    for tool in tools:
        name, desc_parts = _extract_tool_info(tool)
        if not name or name in seen:
            continue
        seen.add(name)

        if name in KNOWN_TOOLS:
            lines.append(f"- {name}: {KNOWN_TOOLS[name]}")
        elif desc_parts:
            lines.append(f"- {name}: {desc_parts}")
        else:
            lines.append(f"- {name}")

    n_known = sum(1 for t in seen if t in KNOWN_TOOLS)
    n_unknown = len(seen) - n_known
    if n_unknown:
        logger.debug(
            "Compressed %d tools (%d known, %d from schema)",
            len(seen),
            n_known,
            n_unknown,
        )

    return "\n".join(lines)


def _extract_tool_info(tool: dict[str, Any]) -> tuple[str, str]:
    """Extract name and compact param string from a tool definition.

    Handles both Anthropic format and OpenAI function format.

    Returns:
        (name, compact_description_with_params)
    """
    # Anthropic format: {"name": ..., "description": ..., "input_schema": ...}
    name = tool.get("name", "")
    schema = tool.get("input_schema", {})
    description = tool.get("description", "")

    # OpenAI format: {"type": "function", "function": {"name": ..., ...}}
    if not name and "function" in tool:
        fn = tool["function"]
        name = fn.get("name", "")
        schema = fn.get("parameters", {})
        description = fn.get("description", "")

    if not name:
        return "", ""

    # For known tools, we'll use the pre-written description
    if name in KNOWN_TOOLS:
        return name, ""

    # Unknown tool: build compact description from schema
    parts: list[str] = []
    if description:
        # First sentence only
        first_sentence = description.split(". ")[0].split("\n")[0]
        if len(first_sentence) > 120:
            first_sentence = first_sentence[:117] + "..."
        parts.append(first_sentence + ".")

    if isinstance(schema, dict) and schema.get("properties"):
        props = schema["properties"]
        required = set(schema.get("required", []))
        param_strs: list[str] = []
        for pname in props:
            ptype = props[pname].get("type", "any")
            req = ", required" if pname in required else ""
            param_strs.append(f"{pname} ({ptype}{req})")
        if param_strs:
            parts.append("Params: " + ", ".join(param_strs))

    return name, " ".join(parts)
