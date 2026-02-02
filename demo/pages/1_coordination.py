"""Multi-Agent Coordination Page.

This page demonstrates structured multi-agent debates and discussions with
various topologies (turn-by-turn, broadcast), debate formats (free-form,
structured, Socratic), and decision-making modes (voting, consensus).
"""

import time
from concurrent.futures import ThreadPoolExecutor

import httpx
import streamlit as st

SERVER_URL = "http://127.0.0.1:8000"

# Configuration options
TOPOLOGIES = {
    "turn_by_turn": "Turn by Turn (sequential)",
    "round_robin": "Round Robin (cycle through all)",
    "broadcast": "Broadcast (all see all)",
}

DEBATE_FORMATS = {
    "free_form": "Free Form (unstructured)",
    "structured": "Structured (with evidence)",
    "socratic": "Socratic (question-driven)",
    "devils_advocate": "Devil's Advocate (critical)",
    "parliamentary": "Parliamentary (formal)",
}

DECISION_MODES = {
    "none": "None (no voting)",
    "majority_vote": "Majority Vote",
    "ranked_choice": "Ranked Choice",
    "consensus": "Consensus Building",
}


def init_coordination_state() -> None:
    """Initialize coordination-specific session state for multi-session support."""
    if "coord_initialized" in st.session_state:
        return

    st.session_state.coord_initialized = True
    # Multi-session support: dict of session_id -> session data
    st.session_state.coord_sessions = {}  # {session_id: {status, messages, executing, streaming_text, label, agent_names}}
    st.session_state.coord_active_session = None  # Currently selected session_id
    st.session_state.coord_executor = ThreadPoolExecutor(max_workers=4)


def discover_existing_sessions() -> None:
    """Fetch existing sessions from server and populate coord_sessions dict."""
    try:
        with httpx.Client(timeout=5.0) as client:
            resp = client.get(f"{SERVER_URL}/v1/coordination/sessions")
            if resp.status_code == 200:
                data = resp.json()
                sessions = data.get("sessions", [])

                # Populate coord_sessions dict for each server session
                for session_status in sessions:
                    session_id = session_status["session_id"]

                    # Skip if already in our dict
                    if session_id in st.session_state.coord_sessions:
                        continue

                    # Build label from agent names
                    agent_states = session_status.get("agent_states", [])
                    agent_names = [a["display_name"] for a in agent_states]
                    label = " & ".join(agent_names[:3])  # First 3 names
                    if len(agent_names) > 3:
                        label += f" +{len(agent_names) - 3}"

                    # Add to dict
                    st.session_state.coord_sessions[session_id] = {
                        "status": session_status,
                        "messages": [],  # Will be fetched on demand
                        "executing": False,
                        "streaming_text": "",
                        "label": label,
                        "agent_names": agent_names,
                    }
    except Exception:
        pass  # Silently ignore - server may not be running yet


def create_coordination_session(
    topology: str,
    debate_format: str,
    decision_mode: str,
    agents: list[dict],
    initial_prompt: str,
    max_turns: int,
) -> dict | None:
    """Create a new coordination session via API.

    Args:
        topology: Communication topology.
        debate_format: Debate structure.
        decision_mode: Decision-making mode.
        agents: List of agent configs (display_name, role, system_prompt).
        initial_prompt: Topic/question for discussion.
        max_turns: Maximum turns (0 = unlimited).

    Returns:
        Session response dict, or None if error.
    """
    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.post(
                f"{SERVER_URL}/v1/coordination/sessions",
                json={
                    "topology": topology,
                    "debate_format": debate_format,
                    "decision_mode": decision_mode,
                    "agents": agents,
                    "initial_prompt": initial_prompt,
                    "max_turns": max_turns,
                },
            )
            if resp.status_code == 201:
                return resp.json()
            else:
                st.error(f"Failed to create session: {resp.status_code} {resp.text}")
                return None
    except Exception as e:
        st.error(f"Connection error: {e}")
        return None


def execute_turn(session_id: str) -> dict | None:
    """Execute the next turn in a coordination session.

    Args:
        session_id: Session identifier.

    Returns:
        Response dict with message and status, or None if error.
    """
    try:
        with httpx.Client(timeout=120.0) as client:
            resp = client.post(f"{SERVER_URL}/v1/coordination/sessions/{session_id}/turn")
            if resp.status_code == 200:
                return resp.json()
            else:
                st.error(f"Failed to execute turn: {resp.status_code} {resp.text}")
                return None
    except Exception as e:
        st.error(f"Connection error: {e}")
        return None


def execute_round(session_id: str) -> dict | None:
    """Execute a full round (all agents speak once).

    Args:
        session_id: Session identifier.

    Returns:
        Response dict with messages and status, or None if error.
    """
    try:
        with httpx.Client(timeout=300.0) as client:
            resp = client.post(f"{SERVER_URL}/v1/coordination/sessions/{session_id}/round")
            if resp.status_code == 200:
                return resp.json()
            else:
                st.error(f"Failed to execute round: {resp.status_code} {resp.text}")
                return None
    except Exception as e:
        st.error(f"Connection error: {e}")
        return None


def get_session_status(session_id: str) -> dict | None:
    """Get status of a coordination session.

    Args:
        session_id: Session identifier.

    Returns:
        Session status dict, or None if error.
    """
    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(f"{SERVER_URL}/v1/coordination/sessions/{session_id}")
            if resp.status_code == 200:
                return resp.json()
            else:
                return None
    except Exception:
        return None


def get_session_messages(session_id: str) -> list[dict]:
    """Get all messages in a coordination session.

    Args:
        session_id: Session identifier.

    Returns:
        List of message dicts.
    """
    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(f"{SERVER_URL}/v1/coordination/sessions/{session_id}/messages")
            if resp.status_code == 200:
                data = resp.json()
                return data.get("messages", [])
            else:
                return []
    except Exception:
        return []


def delete_coordination_session(session_id: str) -> bool:
    """Delete a coordination session.

    Args:
        session_id: Session identifier.

    Returns:
        True if successfully deleted, False otherwise.
    """
    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.delete(f"{SERVER_URL}/v1/coordination/sessions/{session_id}")
            if resp.status_code == 200:
                return True
            else:
                st.error(f"Failed to delete session: {resp.status_code} {resp.text}")
                return False
    except Exception as e:
        st.error(f"Connection error: {e}")
        return False


def stream_turn(session_id: str) -> tuple[str, str, str]:
    """Stream a single turn via SSE, yielding tokens as they arrive.

    Args:
        session_id: Session identifier.

    Returns:
        Tuple of (agent_id, agent_name, full_text).
    """
    agent_id = ""
    agent_name = ""
    full_text = ""

    try:
        with httpx.Client(timeout=120.0) as client:
            with client.stream(
                "POST",
                f"{SERVER_URL}/v1/coordination/sessions/{session_id}/turn/stream"
            ) as resp:
                if resp.status_code != 200:
                    return "", "", f"[Error: HTTP {resp.status_code}]"

                for line in resp.iter_lines():
                    if not line.startswith("event: "):
                        continue

                    event_type = line[7:]  # Remove "event: " prefix
                    next_line = next(resp.iter_lines(), "")

                    if not next_line.startswith("data: "):
                        continue

                    data_json = next_line[6:]  # Remove "data: " prefix

                    try:
                        data = json.loads(data_json)
                    except json.JSONDecodeError:
                        continue

                    if event_type == "turn_start":
                        agent_id = data.get("agent_id", "")
                        agent_name = data.get("agent_name", "")
                    elif event_type == "token":
                        full_text = data.get("accumulated", full_text)
                        # Update session state for real-time display
                        if session_id in st.session_state.coord_sessions:
                            st.session_state.coord_sessions[session_id]["streaming_text"] = full_text
                    elif event_type == "turn_complete":
                        full_text = data.get("content", full_text)
                    elif event_type == "error":
                        return "", "", f"[Error: {data.get('error', 'Unknown')}]"

    except Exception as e:
        return "", "", f"[Error: {e}]"

    return agent_id, agent_name, full_text


def render_sidebar() -> None:
    """Render the multi-session coordination sidebar."""
    with st.sidebar:
        st.title("Coordination Sessions")

        sessions = st.session_state.coord_sessions
        active = st.session_state.coord_active_session

        # Session selector
        if sessions:
            session_ids = list(sessions.keys())
            session_labels = [sessions[sid]["label"] for sid in session_ids]

            # Find current index
            current_idx = 0
            if active and active in session_ids:
                current_idx = session_ids.index(active)

            selected_idx = st.selectbox(
                "Active Session",
                options=range(len(session_ids)),
                format_func=lambda i: f"{session_labels[i]} ({session_ids[i][:8]}...)",
                index=current_idx,
                key="coord_sidebar_session_selector",
            )

            selected_id = session_ids[selected_idx]
            if selected_id != active:
                st.session_state.coord_active_session = selected_id
                st.rerun()

            # Session info panel
            session_data = sessions[selected_id]
            status = session_data["status"]

            st.markdown(f"**Turn:** {status['current_turn']}")
            st.markdown(
                f"**Status:** {'🟢 Active' if status['is_active'] else '🔴 Completed'}"
            )

            if status.get("next_speaker"):
                agent_states = status.get("agent_states", [])
                next_name = next(
                    (
                        a["display_name"]
                        for a in agent_states
                        if a["agent_id"] == status["next_speaker"]
                    ),
                    status["next_speaker"],
                )
                st.markdown(f"**Next:** {next_name}")

            st.divider()

            # Agent statistics
            st.subheader("Agents")
            agent_states = status.get("agent_states", [])
            for agent in agent_states:
                lifecycle = agent.get("lifecycle", "ephemeral")
                lifecycle_emoji = "💾" if lifecycle == "permanent" else "⚡"
                st.markdown(
                    f"{lifecycle_emoji} **{agent['display_name']}** ({agent['role']}): "
                    f"{agent['message_count']} msgs"
                )

            st.divider()

            # Delete session button
            if st.button(
                "Delete Session",
                type="secondary",
                use_container_width=True,
                key="coord_sidebar_delete",
            ):
                if delete_coordination_session(selected_id):
                    del st.session_state.coord_sessions[selected_id]
                    if st.session_state.coord_active_session == selected_id:
                        st.session_state.coord_active_session = (
                            list(sessions.keys())[0] if len(sessions) > 1 else None
                        )
                    st.success("Session deleted")
                    st.rerun()

            st.divider()

        # Collapsible creation form
        with st.expander("➕ Create New Session", expanded=(not sessions)):
            topology = st.selectbox(
                "Topology",
                options=list(TOPOLOGIES.keys()),
                format_func=lambda x: TOPOLOGIES[x],
                key="coord_new_topology",
            )

            debate_format = st.selectbox(
                "Debate Format",
                options=list(DEBATE_FORMATS.keys()),
                format_func=lambda x: DEBATE_FORMATS[x],
                key="coord_new_debate_format",
            )

            decision_mode = st.selectbox(
                "Decision Mode",
                options=list(DECISION_MODES.keys()),
                format_func=lambda x: DECISION_MODES[x],
                key="coord_new_decision_mode",
            )

            initial_prompt = st.text_area(
                "Topic / Question",
                value="Should AI development be open source?",
                height=100,
                key="coord_new_prompt",
            )

            max_turns = st.number_input(
                "Max Turns (0 = unlimited)",
                min_value=0,
                max_value=100,
                value=10,
                key="coord_new_max_turns",
            )

            st.divider()

            st.subheader("Configure Agents")
            num_agents = st.number_input(
                "Number of Agents",
                min_value=2,
                max_value=7,
                value=3,
                key="coord_new_num_agents",
            )

            agents = []
            for i in range(num_agents):
                with st.expander(f"Agent {i+1}", expanded=(i == 0)):
                    agent_name = st.text_input(
                        "Name",
                        value=f"Agent {chr(65+i)}",
                        key=f"coord_new_agent_{i}_name",
                    )
                    agent_role = st.selectbox(
                        "Role",
                        options=["participant", "moderator", "critic", "advocate"],
                        key=f"coord_new_agent_{i}_role",
                    )
                    agent_prompt = st.text_area(
                        "System Prompt (optional)",
                        value="",
                        height=80,
                        key=f"coord_new_agent_{i}_prompt",
                    )
                    lifecycle = st.radio(
                        "Memory Lifecycle",
                        options=["ephemeral", "permanent"],
                        index=0,
                        help="Ephemeral: session-only (STM). Permanent: long-term memory (future).",
                        key=f"coord_new_agent_{i}_lifecycle",
                        horizontal=True,
                    )
                    agents.append(
                        {
                            "display_name": agent_name,
                            "role": agent_role,
                            "system_prompt": agent_prompt,
                            "lifecycle": lifecycle,
                        }
                    )

            st.divider()

            if st.button(
                "Create Session",
                type="primary",
                use_container_width=True,
                key="coord_new_create_btn",
            ):
                with st.spinner("Creating coordination session..."):
                    response = create_coordination_session(
                        topology=topology,
                        debate_format=debate_format,
                        decision_mode=decision_mode,
                        agents=agents,
                        initial_prompt=initial_prompt,
                        max_turns=max_turns,
                    )

                    if response:
                        session_id = response["session_id"]
                        status = get_session_status(session_id)

                        if status:
                            agent_states = status.get("agent_states", [])
                            agent_names = [a["display_name"] for a in agent_states]
                            label = " & ".join(agent_names[:3])
                            if len(agent_names) > 3:
                                label += f" +{len(agent_names) - 3}"

                            st.session_state.coord_sessions[session_id] = {
                                "status": status,
                                "messages": [],
                                "executing": False,
                                "streaming_text": "",
                                "label": label,
                                "agent_names": agent_names,
                            }
                            st.session_state.coord_active_session = session_id
                            st.success("Session created!")
                            st.rerun()
                        else:
                            st.error("Failed to fetch session status")


def render_timeline(session_id: str) -> None:
    """Render the conversation timeline for a specific session."""
    if session_id not in st.session_state.coord_sessions:
        return

    session_data = st.session_state.coord_sessions[session_id]

    # Fetch latest messages
    messages = get_session_messages(session_id)
    session_data["messages"] = messages

    # Display messages
    st.subheader(f"Conversation: {session_data['label']}")

    if not messages:
        st.info("No messages yet. Click 'Next Turn' or 'Stream Turn' to start.", icon="💬")
        return

    # Message display
    timeline_container = st.container(height=500)
    with timeline_container:
        for msg in messages:
            sender_name = msg["sender_name"]
            content = msg["content"]
            turn = msg["turn_number"]
            is_system = msg["sender_id"] == "system"

            if is_system:
                st.markdown(
                    f"**[Turn {turn}] 🔔 System:** {content}",
                    help="Initial prompt or system message",
                )
            else:
                # Agent message with colored badge
                st.markdown(
                    f"**[Turn {turn}] {sender_name}:** {content}",
                )
            st.divider()


def render_controls(session_id: str) -> None:
    """Render control buttons for session management with streaming support."""
    if session_id not in st.session_state.coord_sessions:
        return

    session_data = st.session_state.coord_sessions[session_id]
    status = session_data["status"]
    is_active = status.get("is_active", False)
    is_executing = session_data["executing"]

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        next_turn_disabled = is_executing or not is_active
        if st.button(
            "⏭️ Next Turn",
            disabled=next_turn_disabled,
            use_container_width=True,
            key=f"coord_{session_id}_next_turn",
        ):
            session_data["executing"] = True
            with st.spinner("Executing turn..."):
                result = execute_turn(session_id)
                if result:
                    session_data["status"] = result.get("session_status")
                    st.success(f"Turn complete: {result['message']['sender_name']}")
                session_data["executing"] = False
                st.rerun()

    with col2:
        stream_turn_disabled = is_executing or not is_active
        if st.button(
            "▶️ Stream Turn",
            disabled=stream_turn_disabled,
            use_container_width=True,
            type="primary",
            key=f"coord_{session_id}_stream_turn",
        ):
            session_data["executing"] = True
            # Show streaming area
            stream_container = st.empty()

            with stream_container.container():
                st.markdown("**Streaming response...**")
                text_placeholder = st.empty()

                # Stream the turn
                agent_id, agent_name, full_text = stream_turn(session_id)

                if agent_name:
                    text_placeholder.markdown(f"**{agent_name}:** {full_text}")
                    st.success(f"✅ {agent_name} completed")
                else:
                    st.error(full_text)  # Error message

            session_data["executing"] = False
            # Update status
            updated_status = get_session_status(session_id)
            if updated_status:
                session_data["status"] = updated_status
            time.sleep(1)  # Brief pause to show result
            st.rerun()

    with col3:
        run_round_disabled = is_executing or not is_active
        if st.button(
            "🔄 Run Round",
            disabled=run_round_disabled,
            use_container_width=True,
            key=f"coord_{session_id}_run_round",
        ):
            session_data["executing"] = True
            with st.spinner("Executing round..."):
                result = execute_round(session_id)
                if result:
                    session_data["status"] = result.get("session_status")
                    num_messages = len(result.get("messages", []))
                    st.success(f"Round complete: {num_messages} messages generated")
                session_data["executing"] = False
                st.rerun()

    with col4:
        if st.button(
            "🔄 Refresh",
            use_container_width=True,
            key=f"coord_{session_id}_refresh",
        ):
            st.rerun()

    # Status message
    if is_executing:
        st.info("⏳ Generating... Please wait.", icon="⏳")
    elif not is_active:
        st.warning("Session has reached max turns or been ended.", icon="⚠️")


def main() -> None:
    """Main page entry point."""
    st.set_page_config(
        page_title="Multi-Agent Coordination",
        page_icon="🤝",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    init_coordination_state()
    discover_existing_sessions()

    # Page title
    st.title("🤝 Multi-Agent Coordination")
    st.markdown(
        "Create structured debates, discussions, and decision-making sessions "
        "with multiple agents using various topologies and debate formats."
    )

    st.divider()

    # Sidebar for session config
    render_sidebar()

    # Main content - session-scoped rendering
    active = st.session_state.coord_active_session
    if active and active in st.session_state.coord_sessions:
        render_timeline(active)
        st.divider()
        render_controls(active)
    else:
        st.info(
            "👈 Create a coordination session using the sidebar to get started.",
            icon="ℹ️",
        )


if __name__ == "__main__":
    main()
