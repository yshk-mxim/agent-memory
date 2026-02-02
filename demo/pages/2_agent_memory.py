"""Agent Memory Inspector - View and manage cached agents across tiers."""

import time
from typing import Any

import httpx
import streamlit as st

SERVER_URL = "http://localhost:8000"


def init_memory_state() -> None:
    """Initialize memory inspector session state."""
    if "memory_initialized" in st.session_state:
        return

    st.session_state.memory_initialized = True
    st.session_state.memory_selected_agent = None
    st.session_state.memory_auto_refresh = True
    st.session_state.memory_last_refresh = 0.0


def get_agent_list() -> dict[str, Any] | None:
    """Fetch list of all cached agents from server."""
    try:
        with httpx.Client(timeout=5.0) as client:
            resp = client.get(f"{SERVER_URL}/v1/agents/list")
            if resp.status_code == 200:
                return resp.json()
            else:
                st.error(f"Failed to fetch agents: {resp.status_code}")
                return None
    except Exception as e:
        st.error(f"Connection error: {e}")
        return None


def get_agent_stats() -> dict[str, Any] | None:
    """Fetch aggregate cache statistics from server."""
    try:
        with httpx.Client(timeout=5.0) as client:
            resp = client.get(f"{SERVER_URL}/v1/agents/stats")
            if resp.status_code == 200:
                return resp.json()
            else:
                st.error(f"Failed to fetch stats: {resp.status_code}")
                return None
    except Exception as e:
        st.error(f"Connection error: {e}")
        return None


def delete_agent(agent_id: str) -> bool:
    """Delete an agent and its cache."""
    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.delete(f"{SERVER_URL}/v1/agents/{agent_id}")
            if resp.status_code == 204:
                return True
            else:
                st.error(f"Failed to delete agent: {resp.status_code}")
                return False
    except Exception as e:
        st.error(f"Connection error: {e}")
        return False


def detect_agent_source(agent_id: str) -> str:
    """Detect agent source from ID prefix.

    Returns:
        "OpenAI", "Coordination", "Anthropic", or "Direct"
    """
    if agent_id.startswith("oai_"):
        return "OpenAI"
    elif agent_id.startswith("coord_"):
        return "Coordination"
    elif agent_id.startswith("sess_"):
        return "Anthropic"
    else:
        return "Direct"


def format_bytes(bytes_val: int) -> str:
    """Format bytes as human-readable size."""
    if bytes_val < 1024:
        return f"{bytes_val} B"
    elif bytes_val < 1024 * 1024:
        return f"{bytes_val / 1024:.1f} KB"
    elif bytes_val < 1024 * 1024 * 1024:
        return f"{bytes_val / (1024 * 1024):.1f} MB"
    else:
        return f"{bytes_val / (1024 * 1024 * 1024):.2f} GB"


def render_sidebar() -> tuple[str | None, str | None, str | None]:
    """Render sidebar with filters and controls.

    Returns:
        Tuple of (tier_filter, source_filter, lifecycle_filter)
    """
    with st.sidebar:
        st.title("Memory Inspector")

        # Server status
        st.subheader("Server Status")
        stats = get_agent_stats()
        if stats:
            st.success("Connected", icon="✅")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Agents", stats["total_count"])
                st.metric("Hot Tier", stats["hot_count"])
            with col2:
                st.metric("Warm Tier", stats["warm_count"])
                st.metric("Pool Usage", f"{stats['pool_utilization_pct']:.1f}%")
            st.metric("Cache Size", f"{stats['total_cache_size_mb']:.1f} MB")

            if stats["dirty_count"] > 0:
                st.warning(f"⚠️ {stats['dirty_count']} unsaved changes", icon="⚠️")
        else:
            st.error("Disconnected", icon="❌")

        st.divider()

        # Filters
        st.subheader("Filters")
        tier_filter = st.selectbox(
            "Tier",
            options=[None, "hot", "warm"],
            format_func=lambda x: "All" if x is None else x.capitalize(),
            key="memory_tier_filter",
        )

        source_filter = st.selectbox(
            "Source",
            options=[None, "OpenAI", "Coordination", "Anthropic", "Direct"],
            format_func=lambda x: "All" if x is None else x,
            key="memory_source_filter",
        )

        lifecycle_filter = st.selectbox(
            "Lifecycle",
            options=[None, "ephemeral", "permanent"],
            format_func=lambda x: "All" if x is None else x.capitalize(),
            key="memory_lifecycle_filter",
        )

        st.divider()

        # Bulk actions
        st.subheader("Bulk Actions")
        if st.button("Clear All Hot Caches", use_container_width=True, key="memory_bulk_clear"):
            st.warning("Bulk clear not yet implemented")

        if st.button("Flush Dirty to Disk", use_container_width=True, key="memory_bulk_flush"):
            st.warning("Flush not yet implemented")

        st.divider()

        # Auto-refresh
        auto_refresh = st.checkbox(
            "Auto-refresh (5s)",
            value=st.session_state.memory_auto_refresh,
            key="memory_auto_refresh_checkbox",
        )
        st.session_state.memory_auto_refresh = auto_refresh

        if st.button("🔄 Refresh Now", use_container_width=True, key="memory_refresh_now"):
            st.session_state.memory_last_refresh = time.time()
            st.rerun()

    return tier_filter, source_filter, lifecycle_filter


def render_agent_table(agents: list[dict], tier_filter: str | None, source_filter: str | None) -> None:
    """Render agent table with selection."""
    # Apply filters
    filtered = agents
    if tier_filter:
        filtered = [a for a in filtered if a["tier"] == tier_filter]
    if source_filter:
        filtered = [a for a in filtered if detect_agent_source(a["agent_id"]) == source_filter]

    if not filtered:
        st.info("No agents match the current filters.", icon="ℹ️")
        return

    st.subheader(f"Agents ({len(filtered)})")

    # Table headers
    cols = st.columns([3, 1, 1, 1, 1, 1])
    cols[0].markdown("**Agent ID**")
    cols[1].markdown("**Tier**")
    cols[2].markdown("**Source**")
    cols[3].markdown("**Tokens**")
    cols[4].markdown("**Disk Size**")
    cols[5].markdown("**Actions**")

    st.divider()

    # Agent rows
    for agent in filtered:
        agent_id = agent["agent_id"]
        tier = agent["tier"]
        source = detect_agent_source(agent_id)
        tokens = agent["tokens"]
        file_size = agent["file_size_bytes"]

        cols = st.columns([3, 1, 1, 1, 1, 1])

        # Agent ID (clickable)
        with cols[0]:
            if st.button(
                agent_id[:32] + ("..." if len(agent_id) > 32 else ""),
                key=f"memory_select_{agent_id}",
                use_container_width=True,
            ):
                st.session_state.memory_selected_agent = agent_id
                st.rerun()

        # Tier badge
        with cols[1]:
            if tier == "hot":
                st.markdown("🔥 **Hot**")
            else:
                st.markdown("💾 Warm")

        # Source
        with cols[2]:
            source_emoji = {
                "OpenAI": "🤖",
                "Coordination": "🤝",
                "Anthropic": "🧠",
                "Direct": "⚡",
            }
            st.markdown(f"{source_emoji.get(source, '❓')} {source}")

        # Tokens
        with cols[3]:
            if tokens > 0:
                st.markdown(f"{tokens:,}")
            else:
                st.markdown("—")

        # Disk size
        with cols[4]:
            if file_size > 0:
                st.markdown(format_bytes(file_size))
            else:
                st.markdown("—")

        # Delete button
        with cols[5]:
            if st.button("🗑️", key=f"memory_delete_{agent_id}"):
                if delete_agent(agent_id):
                    st.success(f"Deleted {agent_id[:12]}...")
                    st.session_state.memory_selected_agent = None
                    time.sleep(1)
                    st.rerun()

        st.divider()


def render_agent_detail(agent_id: str, agents: list[dict]) -> None:
    """Render detailed view for selected agent."""
    # Find agent in list
    agent = next((a for a in agents if a["agent_id"] == agent_id), None)
    if not agent:
        st.error(f"Agent {agent_id} not found")
        st.session_state.memory_selected_agent = None
        return

    st.subheader(f"Agent Detail: {agent_id}")

    # Close button
    if st.button("✕ Close", key="memory_close_detail"):
        st.session_state.memory_selected_agent = None
        st.rerun()

    st.divider()

    # Metadata grid
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Tier:**")
        tier_emoji = "🔥" if agent["tier"] == "hot" else "💾"
        st.markdown(f"{tier_emoji} {agent['tier'].capitalize()}")

        st.markdown("**Source:**")
        source = detect_agent_source(agent_id)
        st.markdown(source)

        st.markdown("**Tokens:**")
        if agent["tokens"] > 0:
            st.markdown(f"{agent['tokens']:,}")
        else:
            st.markdown("Unknown (warm tier)")

        st.markdown("**Model:**")
        st.markdown(agent.get("model_id", "Unknown"))

    with col2:
        st.markdown("**Disk Size:**")
        if agent["file_size_bytes"] > 0:
            st.markdown(format_bytes(agent["file_size_bytes"]))
        else:
            st.markdown("Not persisted")

        st.markdown("**Last Accessed:**")
        if agent["last_accessed"] > 0:
            import datetime
            dt = datetime.datetime.fromtimestamp(agent["last_accessed"])
            st.markdown(dt.strftime("%Y-%m-%d %H:%M:%S"))
        else:
            st.markdown("Unknown")

        st.markdown("**Access Count:**")
        st.markdown(str(agent["access_count"]))

        st.markdown("**Dirty:**")
        if agent.get("dirty", False):
            st.markdown("⚠️ Yes (unsaved)")
        else:
            st.markdown("✅ No")

    st.divider()

    # Actions
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Delete Agent", use_container_width=True, type="primary", key=f"memory_detail_delete_{agent_id}"):
            if delete_agent(agent_id):
                st.success("Agent deleted")
                st.session_state.memory_selected_agent = None
                time.sleep(1)
                st.rerun()

    with col2:
        if agent["tier"] == "hot":
            if st.button("💾 Evict to Disk", use_container_width=True, key=f"memory_detail_evict_{agent_id}"):
                st.warning("Manual eviction not yet implemented")


def main() -> None:
    """Main entry point."""
    st.set_page_config(
        page_title="Agent Memory Inspector",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    init_memory_state()

    # Auto-refresh logic
    if st.session_state.memory_auto_refresh:
        current_time = time.time()
        if current_time - st.session_state.memory_last_refresh > 5.0:
            st.session_state.memory_last_refresh = current_time
            st.rerun()

    # Page title
    st.title("🧠 Agent Memory Inspector")
    st.markdown(
        "Monitor and manage agent caches across hot (in-memory) and warm (on-disk) tiers."
    )

    st.divider()

    # Render sidebar and get filters
    tier_filter, source_filter, lifecycle_filter = render_sidebar()

    # Fetch agent list
    data = get_agent_list()
    if not data:
        st.warning("Unable to fetch agent list. Is the server running?", icon="⚠️")
        return

    agents = data.get("agents", [])

    if not agents:
        st.info("No cached agents found. Create agents via API or run coordination sessions.", icon="ℹ️")
        return

    # Render selected agent detail or table
    selected = st.session_state.memory_selected_agent
    if selected and selected in [a["agent_id"] for a in agents]:
        render_agent_detail(selected, agents)
    else:
        render_agent_table(agents, tier_filter, source_filter)


if __name__ == "__main__":
    main()
