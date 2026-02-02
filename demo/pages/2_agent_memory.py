"""Agent Memory Inspector - View and manage cached agents across tiers.

Uses shared api_client for all server communication.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import datetime
import time
from typing import Any

import streamlit as st
from demo.lib import api_client

SERVER_URL = "http://localhost:8000"

_BYTES_PER_KB = 1024
_BYTES_PER_MB = 1024 * 1024
_BYTES_PER_GB = 1024 * 1024 * 1024
_AGENT_ID_DISPLAY_LEN = 32
_AUTO_REFRESH_SECONDS = 5.0
_DELETE_DELAY_SECONDS = 0.5


def init_state() -> None:
    """Initialize memory inspector session state."""
    if "mem_init" in st.session_state:
        return
    st.session_state.mem_init = True
    st.session_state.mem_selected = None
    st.session_state.mem_auto = True
    st.session_state.mem_last = 0.0


def detect_source(agent_id: str) -> str:
    """Detect agent source from ID prefix."""
    if agent_id.startswith("oai_"):
        return "OpenAI"
    if agent_id.startswith("coord_"):
        return "Coordination"
    if agent_id.startswith("sess_"):
        return "Anthropic"
    return "Direct"


def fmt_bytes(b: int) -> str:
    """Format byte count as human-readable string."""
    if b < _BYTES_PER_KB:
        return f"{b} B"
    if b < _BYTES_PER_MB:
        return f"{b / _BYTES_PER_KB:.1f} KB"
    if b < _BYTES_PER_GB:
        return f"{b / _BYTES_PER_MB:.1f} MB"
    return f"{b / _BYTES_PER_GB:.2f} GB"


def render_sidebar() -> tuple[str | None, str | None]:
    """Returns (tier_filter, source_filter)."""
    with st.sidebar:
        st.title("Memory Inspector")

        stats = api_client.get_agent_stats(SERVER_URL)
        if stats is not None:
            st.success("Server connected")
            c1, c2 = st.columns(2)
            with c1:
                st.metric("Total", stats["total_count"])
                st.metric("Hot", stats["hot_count"])
            with c2:
                st.metric("Warm", stats["warm_count"])
                st.metric("Pool", f"{stats['pool_utilization_pct']:.1f}%")
            st.metric("Cache Size", f"{stats['total_cache_size_mb']:.1f} MB")
        else:
            st.error("Server not reachable. Start with: `semantic serve`")

        st.divider()

        tier = st.selectbox(
            "Tier",
            [None, "hot", "warm"],
            format_func=lambda x: "All" if x is None else x.capitalize(),
            key="mem_tier",
        )
        source = st.selectbox(
            "Source",
            [None, "OpenAI", "Coordination", "Anthropic", "Direct"],
            format_func=lambda x: "All" if x is None else x,
            key="mem_src",
        )

        st.divider()
        auto = st.checkbox("Auto-refresh (5s)", value=st.session_state.mem_auto, key="mem_auto_cb")
        st.session_state.mem_auto = auto

        if st.button("Refresh", use_container_width=True, key="mem_ref"):
            st.session_state.mem_last = time.time()
            st.rerun()

    return tier, source


def render_table(agents: list[dict[str, Any]], tier_f: str | None, source_f: str | None) -> None:
    """Render agent table with filtering."""
    filtered = agents
    if tier_f:
        filtered = [a for a in filtered if a["tier"] == tier_f]
    if source_f:
        filtered = [a for a in filtered if detect_source(a["agent_id"]) == source_f]

    if not filtered:
        st.info("No agents match filters.")
        return

    st.subheader(f"Agents ({len(filtered)})")
    for agent in filtered:
        aid = agent["agent_id"]
        cols = st.columns([3, 1, 1, 1, 1, 1])
        with cols[0]:
            suffix = "..." if len(aid) > _AGENT_ID_DISPLAY_LEN else ""
            label = aid[:_AGENT_ID_DISPLAY_LEN] + suffix
            if st.button(label, key=f"m_sel_{aid}", use_container_width=True):
                st.session_state.mem_selected = aid
                st.rerun()
        cols[1].markdown(f"{'Hot' if agent['tier'] == 'hot' else 'Warm'}")
        cols[2].markdown(detect_source(aid))
        cols[3].markdown(f"{agent['tokens']:,}" if agent["tokens"] > 0 else "-")
        size = fmt_bytes(agent["file_size_bytes"]) if agent["file_size_bytes"] > 0 else "-"
        cols[4].markdown(size)
        with cols[5]:
            if st.button("Del", key=f"m_del_{aid}") and api_client.delete_agent(SERVER_URL, aid):
                st.session_state.mem_selected = None
                time.sleep(_DELETE_DELAY_SECONDS)
                st.rerun()
        st.divider()


def render_detail(aid: str, agents: list[dict[str, Any]]) -> None:
    """Render detailed view for a single agent."""
    agent = next((a for a in agents if a["agent_id"] == aid), None)
    if not agent:
        st.session_state.mem_selected = None
        return

    st.subheader(f"Detail: {aid}")
    if st.button("Close", key="m_close"):
        st.session_state.mem_selected = None
        st.rerun()

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**Tier:** {agent['tier']}")
        st.markdown(f"**Source:** {detect_source(aid)}")
        st.markdown(f"**Tokens:** {agent['tokens']:,}" if agent["tokens"] > 0 else "**Tokens:** -")
        st.markdown(f"**Model:** {agent.get('model_id', '?')}")
    with c2:
        size_text = (
            f"**Size:** {fmt_bytes(agent['file_size_bytes'])}"
            if agent["file_size_bytes"] > 0
            else "**Size:** -"
        )
        st.markdown(size_text)
        if agent.get("last_accessed", 0) > 0:
            dt = datetime.datetime.fromtimestamp(agent["last_accessed"])
            st.markdown(f"**Accessed:** {dt:%Y-%m-%d %H:%M:%S}")
        st.markdown(f"**Accesses:** {agent.get('access_count', 0)}")
        st.markdown(f"**Dirty:** {'Yes' if agent.get('dirty') else 'No'}")

    delete_clicked = st.button("Delete Agent", type="primary", key=f"m_ddel_{aid}")
    if delete_clicked and api_client.delete_agent(SERVER_URL, aid):
        st.session_state.mem_selected = None
        st.rerun()


def main() -> None:
    """Agent Memory Inspector demo page."""
    st.set_page_config(
        page_title="Memory Inspector",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    init_state()

    elapsed = time.time() - st.session_state.mem_last
    if st.session_state.mem_auto and elapsed > _AUTO_REFRESH_SECONDS:
        st.session_state.mem_last = time.time()
        st.rerun()

    st.title("Agent Memory Inspector")
    st.markdown("Monitor agent caches across hot and warm tiers.")
    st.divider()

    tier_f, source_f = render_sidebar()
    agents = api_client.get_agent_list(SERVER_URL)

    if not agents:
        st.info("No cached agents. Create agents or run coordination sessions.")
        return

    selected = st.session_state.mem_selected
    if selected and selected in [a["agent_id"] for a in agents]:
        render_detail(selected, agents)
    else:
        render_table(agents, tier_f, source_f)


if __name__ == "__main__":
    main()
