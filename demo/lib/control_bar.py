"""Shared control bar for coordination session management.

Provides reusable button groups for running turns, rounds,
and managing sessions.
"""

from __future__ import annotations

import streamlit as st


def render_round_controls(
    *,
    is_executing: bool,
    is_active: bool,
    key_prefix: str,
    auto_rounds: int = 3,
) -> str | None:
    """Render Run Turns / Run Round buttons.

    Returns:
        Action string: 'run_turns', 'run_round', or None.
    """
    col1, col2, col3 = st.columns(3)
    disabled = is_executing or not is_active

    with col1:
        if st.button(
            f"Run {auto_rounds} Turns",
            disabled=disabled,
            use_container_width=True,
            type="primary",
            key=f"{key_prefix}_run_turns",
        ):
            return "run_turns"

    with col2:
        if st.button(
            "Run Round",
            disabled=disabled,
            use_container_width=True,
            key=f"{key_prefix}_run_round",
        ):
            return "run_round"

    with col3:
        if st.button(
            "Refresh",
            use_container_width=True,
            key=f"{key_prefix}_refresh",
        ):
            return "refresh"

    return None


def render_session_actions(
    *,
    key_prefix: str,
) -> str | None:
    """Render Delete / Reset buttons.

    Returns:
        Action string: 'delete', 'reset', or None.
    """
    col1, col2 = st.columns(2)

    with col1:
        if st.button(
            "Delete Session",
            use_container_width=True,
            type="secondary",
            key=f"{key_prefix}_delete",
        ):
            return "delete"

    with col2:
        if st.button(
            "Reset",
            use_container_width=True,
            key=f"{key_prefix}_reset",
        ):
            return "reset"

    return None
