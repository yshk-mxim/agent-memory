# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Yakov Shkolnikov and contributors
"""Template resolver for cross-phase context injection.

Resolves ${phase_name.messages[agent_key]} placeholders in
initial_prompt_template fields by substituting formatted
message transcripts from prior phases.
"""

from __future__ import annotations

import re

_TEMPLATE_PATTERN = re.compile(r"\$\{(\w+)\.messages\[(\w+)\]\}")


def resolve_template(
    template: str,
    phase_messages: dict[str, list[dict[str, str]]],
    agent_display_names: dict[str, str] | None = None,
) -> str:
    """Replace ${phase.messages[agent]} with formatted transcripts.

    The agent_key in brackets controls perspective: the named agent's
    messages are labeled "You" while others keep their display names.

    Args:
        template: Template string with ${phase.messages[agent]} placeholders.
        phase_messages: Map of phase_name -> list of message dicts.
            Each message dict should have 'sender_name' and 'content' keys.
        agent_display_names: Map of agent_key -> display_name (e.g. "alice" -> "Alice").
            Required for perspective-aware formatting.

    Returns:
        Resolved string with placeholders replaced by transcripts.
    """

    def _replacer(match: re.Match[str]) -> str:
        phase_name = match.group(1)
        agent_key = match.group(2)
        messages = phase_messages.get(phase_name, [])
        perspective_name = (agent_display_names or {}).get(agent_key)
        return _format_transcript(messages, perspective_name)

    return _TEMPLATE_PATTERN.sub(_replacer, template)


def _format_transcript(
    messages: list[dict[str, str]],
    perspective_name: str | None = None,
) -> str:
    """Format messages as a readable transcript.

    Args:
        messages: List of message dicts with 'sender_name' and 'content'.
        perspective_name: If set, this agent's messages are labeled "You"
            so the model maintains first-person identity.
    """
    if not messages:
        return "(no messages yet)"
    lines = []
    for msg in messages:
        sender = msg.get("sender_name", "Unknown")
        content = msg.get("content", "")
        if sender == "System":
            continue
        label = "You" if perspective_name and sender == perspective_name else sender
        lines.append(f"{label}: {content}")
    return "\n".join(lines)


def has_template_refs(text: str) -> bool:
    """Check if text contains any ${phase.messages[agent]} references."""
    return bool(_TEMPLATE_PATTERN.search(text))


def extract_phase_refs(template: str) -> set[str]:
    """Extract all phase names referenced in a template."""
    return {match.group(1) for match in _TEMPLATE_PATTERN.finditer(template)}
