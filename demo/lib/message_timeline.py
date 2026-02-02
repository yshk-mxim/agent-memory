"""Shared message timeline rendering.

Displays coordination messages with agent colors
and scrollable containers.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

DEFAULT_COLORS = [
    "#FF6B6B",
    "#4ECDC4",
    "#45B7D1",
    "#96CEB4",
    "#FFEAA7",
    "#DDA0DD",
    "#98D8C8",
    "#F7DC6F",
    "#BB8FCE",
    "#85C1E9",
    "#F0B27A",
    "#AED6F1",
]


def render_message(
    msg: dict[str, Any],
    agent_colors: dict[str, str],
) -> None:
    """Render a single message with sender name and color indicator."""
    sender_id = msg.get("sender_id", "")
    sender_name = msg.get("sender_name", "Unknown")
    content = msg.get("content", "")
    turn = msg.get("turn_number", 0)

    if sender_id == "system":
        st.markdown(f"*[Turn {turn}] {content}*")
    else:
        color = agent_colors.get(sender_name, "#888888")
        st.markdown(
            f'<span style="color:{color}; font-weight:bold;">'
            f"[Turn {turn}] {sender_name}:</span> {content}",
            unsafe_allow_html=True,
        )


def render_timeline(
    messages: list[dict[str, Any]],
    agent_colors: dict[str, str],
    *,
    max_visible: int = 50,
    height: int = 400,
) -> None:
    """Render a scrollable message timeline.

    Args:
        messages: List of message dicts with sender_name, content, etc.
        agent_colors: Map of agent display_name to hex color.
        max_visible: Maximum messages to show (most recent).
        height: Container height in pixels.
    """
    if not messages:
        st.info("No messages yet. Run turns to start the conversation.")
        return

    visible = messages[-max_visible:] if len(messages) > max_visible else messages

    with st.container(height=height):
        for msg in visible:
            render_message(msg, agent_colors)
            st.divider()


def build_agent_colors(
    agents: dict[str, Any],
) -> dict[str, str]:
    """Build agent display_name -> color map from scenario agents.

    Uses agent-specified colors if available, falls back to defaults.
    """
    colors: dict[str, str] = {}
    default_idx = 0
    for agent in agents.values():
        has_attrs = hasattr(agent, "display_name")
        name = agent.display_name if has_attrs else agent.get("display_name", "")
        color = agent.color if has_attrs else agent.get("color", "")
        if color:
            colors[name] = color
        else:
            colors[name] = DEFAULT_COLORS[default_idx % len(DEFAULT_COLORS)]
            default_idx += 1
    return colors
