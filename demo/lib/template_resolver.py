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
) -> str:
    """Replace ${phase.messages[agent]} with formatted transcripts.

    Args:
        template: Template string with ${phase.messages[agent]} placeholders.
        phase_messages: Map of phase_name -> list of message dicts.
            Each message dict should have 'sender_name' and 'content' keys.

    Returns:
        Resolved string with placeholders replaced by transcripts.
    """

    def _replacer(match: re.Match[str]) -> str:
        phase_name = match.group(1)
        messages = phase_messages.get(phase_name, [])
        return _format_transcript(messages)

    return _TEMPLATE_PATTERN.sub(_replacer, template)


def _format_transcript(messages: list[dict[str, str]]) -> str:
    """Format messages as a readable transcript."""
    if not messages:
        return "(no messages yet)"
    lines = []
    for msg in messages:
        sender = msg.get("sender_name", "Unknown")
        content = msg.get("content", "")
        lines.append(f"{sender}: {content}")
    return "\n".join(lines)


def has_template_refs(text: str) -> bool:
    """Check if text contains any ${phase.messages[agent]} references."""
    return bool(_TEMPLATE_PATTERN.search(text))


def extract_phase_refs(template: str) -> set[str]:
    """Extract all phase names referenced in a template."""
    return {match.group(1) for match in _TEMPLATE_PATTERN.finditer(template)}
