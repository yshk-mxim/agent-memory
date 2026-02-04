"""Multi-Agent Coordination Page.

Dynamic session creation with configurable agents.
Uses shared ScenarioRenderer for session rendering.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is importable (Streamlit pages don't inherit project root on sys.path)
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import logging
from concurrent.futures import ThreadPoolExecutor

import streamlit as st
from demo.lib import api_client
from demo.lib.control_bar import render_round_controls
from demo.lib.message_timeline import DEFAULT_COLORS, render_timeline

logger = logging.getLogger(__name__)

SERVER_URL = "http://127.0.0.1:8000"

TOPOLOGIES = {
    "turn_by_turn": "Turn by Turn",
    "round_robin": "Round Robin",
    "broadcast": "Broadcast",
}

DEBATE_FORMATS = {
    "free_form": "Free Form",
    "structured": "Structured",
    "socratic": "Socratic",
    "devils_advocate": "Devil's Advocate",
    "parliamentary": "Parliamentary",
}

DECISION_MODES = {
    "none": "None",
    "majority_vote": "Majority Vote",
    "ranked_choice": "Ranked Choice",
    "consensus": "Consensus",
}

_MAX_LABEL_AGENTS = 3
_ASCII_A = 65
_DEFAULT_TURN_ROUNDS = 3


def init_state() -> None:
    """Initialize coordination session state."""
    if "coord_init" in st.session_state:
        return
    st.session_state.coord_init = True
    st.session_state.coord_sessions = {}
    st.session_state.coord_active = None
    st.session_state.coord_executor = ThreadPoolExecutor(max_workers=4)


def discover_sessions() -> None:
    """Repopulate sessions dict from server."""
    try:
        sessions = api_client.list_sessions(SERVER_URL)
        for status in sessions:
            sid = status["session_id"]
            if sid in st.session_state.coord_sessions:
                continue
            agents = status.get("agent_states", [])
            names = [a["display_name"] for a in agents]
            label = " & ".join(names[:_MAX_LABEL_AGENTS])
            if len(names) > _MAX_LABEL_AGENTS:
                label += f" +{len(names) - _MAX_LABEL_AGENTS}"
            color_map = {n: DEFAULT_COLORS[i % len(DEFAULT_COLORS)] for i, n in enumerate(names)}
            st.session_state.coord_sessions[sid] = {
                "status": status,
                "messages": [],
                "executing": False,
                "label": label,
                "colors": color_map,
            }
    except (KeyError, TypeError):
        logger.debug("Failed to discover sessions", exc_info=True)


def _register_session(sid: str) -> None:
    """Register a newly created session in session state."""
    status = api_client.get_session_status(SERVER_URL, sid)
    if not status:
        return
    names = [a["display_name"] for a in status.get("agent_states", [])]
    label = " & ".join(names[:_MAX_LABEL_AGENTS])
    if len(names) > _MAX_LABEL_AGENTS:
        label += f" +{len(names) - _MAX_LABEL_AGENTS}"
    color_map = {n: DEFAULT_COLORS[idx % len(DEFAULT_COLORS)] for idx, n in enumerate(names)}
    st.session_state.coord_sessions[sid] = {
        "status": status,
        "messages": [],
        "executing": False,
        "label": label,
        "colors": color_map,
    }
    st.session_state.coord_active = sid
    st.rerun()


def _render_creation_form(has_sessions: bool) -> None:
    """Render the session creation form in the sidebar."""
    with st.expander("New Session", expanded=not has_sessions):
        topology = st.selectbox(
            "Topology",
            list(TOPOLOGIES.keys()),
            format_func=lambda x: TOPOLOGIES[x],
            key="coord_topo",
        )
        debate_format = st.selectbox(
            "Format",
            list(DEBATE_FORMATS.keys()),
            format_func=lambda x: DEBATE_FORMATS[x],
            key="coord_fmt",
        )
        decision_mode = st.selectbox(
            "Decision",
            list(DECISION_MODES.keys()),
            format_func=lambda x: DECISION_MODES[x],
            key="coord_dec",
        )
        prompt = st.text_area(
            "Topic",
            value="Should AI development be open source?",
            height=80,
            key="coord_prompt",
        )
        max_turns = st.number_input(
            "Max Turns (0=unlimited)",
            0,
            100,
            10,
            key="coord_mt",
        )

        st.subheader("Agents")
        n = st.number_input("Count", 2, 10, 3, key="coord_n")
        agents = []
        for i in range(n):
            with st.expander(f"Agent {i + 1}", expanded=(i == 0)):
                default_name = f"Agent {chr(_ASCII_A + i)}"
                name = st.text_input("Name", default_name, key=f"coord_a{i}_n")
                role = st.selectbox(
                    "Role",
                    ["participant", "moderator", "critic", "advocate"],
                    key=f"coord_a{i}_r",
                )
                sp = st.text_area("System Prompt", "", height=60, key=f"coord_a{i}_sp")
                lc = st.radio(
                    "Lifecycle",
                    ["ephemeral", "permanent"],
                    key=f"coord_a{i}_lc",
                    horizontal=True,
                )
                agents.append(
                    {
                        "display_name": name,
                        "role": role,
                        "system_prompt": sp,
                        "lifecycle": lc,
                    }
                )

        if st.button("Create", type="primary", use_container_width=True, key="coord_create"):
            with st.spinner("Creating..."):
                resp = api_client.create_session(
                    SERVER_URL,
                    topology=topology,
                    debate_format=debate_format,
                    decision_mode=decision_mode,
                    agents=agents,
                    initial_prompt=prompt,
                    max_turns=max_turns,
                )
                if resp:
                    _register_session(resp["session_id"])


def render_sidebar() -> None:
    """Render sidebar with session list and creation form."""
    with st.sidebar:
        st.title("Coordination")

        # Connection check
        stats = api_client.get_agent_stats(SERVER_URL)
        if stats is not None:
            st.success("Server connected")
        else:
            st.error("Server not reachable. Start with: `semantic serve`")

        sessions = st.session_state.coord_sessions
        active = st.session_state.coord_active

        if sessions:
            sids = list(sessions.keys())
            labels = [sessions[s]["label"] for s in sids]
            idx = sids.index(active) if active in sids else 0

            sel = st.selectbox(
                "Session",
                options=range(len(sids)),
                format_func=lambda i: f"{labels[i]} ({sids[i][:8]}...)",
                index=idx,
                key="coord_sel",
            )
            if sids[sel] != active:
                st.session_state.coord_active = sids[sel]
                st.rerun()

            data = sessions[sids[sel]]
            status = data["status"]
            st.markdown(f"**Turn:** {status['current_turn']}")
            active_text = "Yes" if status["is_active"] else "No"
            st.markdown(f"**Active:** {active_text}")

            delete_clicked = st.button(
                "Delete",
                use_container_width=True,
                key="coord_del",
            )
            if delete_clicked and api_client.delete_session(SERVER_URL, sids[sel]):
                del st.session_state.coord_sessions[sids[sel]]
                remaining = list(st.session_state.coord_sessions.keys())
                st.session_state.coord_active = remaining[0] if remaining else None
                st.rerun()

            st.divider()

        _render_creation_form(bool(sessions))


def render_session(sid: str) -> None:
    """Render active session timeline and controls."""
    data = st.session_state.coord_sessions[sid]
    messages = api_client.get_session_messages(SERVER_URL, sid)
    data["messages"] = messages

    st.subheader(f"Session: {data['label']}")
    render_timeline(messages, data["colors"], height=500)

    action = render_round_controls(
        is_executing=data["executing"],
        is_active=data["status"].get("is_active", False),
        key_prefix=f"coord_{sid}",
        auto_rounds=3,
    )

    if action == "run_turns":
        data["executing"] = True
        with st.spinner("Generating..."):
            n_agents = len(data["status"].get("agent_states", []))
            api_client.execute_turns(SERVER_URL, sid, _DEFAULT_TURN_ROUNDS * n_agents)
        data["executing"] = False
        data["status"] = api_client.get_session_status(SERVER_URL, sid) or data["status"]
        st.rerun()
    elif action == "run_round":
        data["executing"] = True
        with st.spinner("Running round..."):
            api_client.execute_round(SERVER_URL, sid)
        data["executing"] = False
        data["status"] = api_client.get_session_status(SERVER_URL, sid) or data["status"]
        st.rerun()
    elif action == "refresh":
        st.rerun()


def main() -> None:
    """Multi-Agent Coordination demo page."""
    st.set_page_config(
        page_title="Coordination",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    init_state()
    discover_sessions()

    st.title("Multi-Agent Coordination")
    st.markdown("Create and manage multi-agent coordination sessions.")
    st.divider()

    render_sidebar()

    active = st.session_state.coord_active
    if active and active in st.session_state.coord_sessions:
        render_session(active)
    else:
        st.info("Create a coordination session using the sidebar.")


if __name__ == "__main__":
    main()
