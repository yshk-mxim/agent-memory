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
    st.session_state.coord_sessions = {}  # {session_id: {status, messages, executing, streaming_text}}
    st.session_state.coord_active_tab = None  # Currently selected tab
    st.session_state.coord_executor = ThreadPoolExecutor(max_workers=4)
    st.session_state.coord_new_session_config = None  # Config for creating new session


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
    """Render the session configuration sidebar."""
    with st.sidebar:
        st.title("Coordination Session")

        # Check if session is active
        if st.session_state.coord_session_id:
            st.success(f"Active Session: {st.session_state.coord_session_id[:12]}...")

            # Session controls
            if st.button("End Session", type="secondary", use_container_width=True):
                st.session_state.coord_session_id = None
                st.session_state.coord_session_status = None
                st.session_state.coord_messages = []
                st.rerun()

            st.divider()

            # Display session status
            if st.session_state.coord_session_status:
                status = st.session_state.coord_session_status
                st.markdown(f"**Turn:** {status['current_turn']}")
                st.markdown(
                    f"**Status:** {'🟢 Active' if status['is_active'] else '🔴 Completed'}"
                )
                if status["next_speaker"]:
                    # Find agent name
                    agent_states = status.get("agent_states", [])
                    next_name = next(
                        (
                            a["display_name"]
                            for a in agent_states
                            if a["agent_id"] == status["next_speaker"]
                        ),
                        status["next_speaker"],
                    )
                    st.markdown(f"**Next Speaker:** {next_name}")

                st.divider()

                # Agent statistics
                st.subheader("Agents")
                for agent in agent_states:
                    st.markdown(
                        f"**{agent['display_name']}** ({agent['role']}): "
                        f"{agent['message_count']} messages"
                    )

        else:
            # Session creation form
            st.subheader("Create New Session")

            topology = st.selectbox(
                "Topology",
                options=list(TOPOLOGIES.keys()),
                format_func=lambda x: TOPOLOGIES[x],
            )

            debate_format = st.selectbox(
                "Debate Format",
                options=list(DEBATE_FORMATS.keys()),
                format_func=lambda x: DEBATE_FORMATS[x],
            )

            decision_mode = st.selectbox(
                "Decision Mode",
                options=list(DECISION_MODES.keys()),
                format_func=lambda x: DECISION_MODES[x],
            )

            initial_prompt = st.text_area(
                "Topic / Question",
                value="Should AI development be open source?",
                height=100,
            )

            max_turns = st.number_input(
                "Max Turns (0 = unlimited)",
                min_value=0,
                max_value=100,
                value=10,
            )

            st.divider()

            # Agent configuration
            st.subheader("Configure Agents")

            num_agents = st.number_input(
                "Number of Agents",
                min_value=2,
                max_value=5,
                value=3,
            )

            agents = []
            for i in range(num_agents):
                with st.expander(f"Agent {i+1}", expanded=(i == 0)):
                    agent_name = st.text_input(
                        "Name",
                        value=f"Agent {chr(65+i)}",  # A, B, C, ...
                        key=f"agent_{i}_name",
                    )
                    agent_role = st.selectbox(
                        "Role",
                        options=["participant", "moderator", "critic", "advocate"],
                        key=f"agent_{i}_role",
                    )
                    agent_prompt = st.text_area(
                        "System Prompt (optional)",
                        value="",
                        height=80,
                        key=f"agent_{i}_prompt",
                    )
                    agents.append(
                        {
                            "display_name": agent_name,
                            "role": agent_role,
                            "system_prompt": agent_prompt,
                        }
                    )

            st.divider()

            # Create button
            if st.button("Create Session", type="primary", use_container_width=True):
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
                        st.session_state.coord_session_id = session_id
                        st.session_state.coord_messages = []

                        # Fetch actual session status from server
                        status = get_session_status(session_id)
                        if status:
                            st.session_state.coord_session_status = status
                        else:
                            # Fallback initialization if status fetch fails
                            st.session_state.coord_session_status = {
                                "is_active": True,
                                "current_turn": 0,
                                "next_speaker": None,
                                "agent_states": [],
                            }

                        st.success("Session created!")
                        st.rerun()


def render_timeline() -> None:
    """Render the conversation timeline."""
    if not st.session_state.coord_session_id:
        st.info(
            "👈 Create a coordination session using the sidebar to get started.",
            icon="ℹ️",
        )
        return

    # Fetch latest messages
    messages = get_session_messages(st.session_state.coord_session_id)
    st.session_state.coord_messages = messages

    # Display messages
    st.subheader("Conversation Timeline")

    if not messages:
        st.info("No messages yet. Click 'Next Turn' or 'Run Round' to start.", icon="💬")
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


def render_controls() -> None:
    """Render control buttons for session management with streaming support."""
    if not st.session_state.coord_session_id:
        return

    status = st.session_state.coord_session_status
    is_active = status and status.get("is_active", False)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        next_turn_disabled = st.session_state.coord_executing or not is_active
        if st.button(
            "⏭️ Next Turn",
            disabled=next_turn_disabled,
            use_container_width=True,
        ):
            st.session_state.coord_executing = True
            with st.spinner("Executing turn..."):
                result = execute_turn(st.session_state.coord_session_id)
                if result:
                    st.session_state.coord_session_status = result.get("session_status")
                    st.success(f"Turn complete: {result['message']['sender_name']}")
                st.session_state.coord_executing = False
                st.rerun()

    with col2:
        stream_turn_disabled = st.session_state.coord_executing or not is_active
        if st.button(
            "▶️ Stream Turn",
            disabled=stream_turn_disabled,
            use_container_width=True,
            type="primary",
        ):
            st.session_state.coord_executing = True
            # Show streaming area
            stream_container = st.empty()

            with stream_container.container():
                st.markdown("**Streaming response...**")
                text_placeholder = st.empty()

                # Stream the turn
                agent_id, agent_name, full_text = stream_turn(st.session_state.coord_session_id)

                if agent_name:
                    text_placeholder.markdown(f"**{agent_name}:** {full_text}")
                    st.success(f"✅ {agent_name} completed")
                else:
                    st.error(full_text)  # Error message

            st.session_state.coord_executing = False
            time.sleep(1)  # Brief pause to show result
            st.rerun()

    with col3:
        run_round_disabled = st.session_state.coord_executing or not is_active
        if st.button(
            "🔄 Run Round",
            disabled=run_round_disabled,
            use_container_width=True,
        ):
            st.session_state.coord_executing = True
            with st.spinner("Executing round..."):
                result = execute_round(st.session_state.coord_session_id)
                if result:
                    st.session_state.coord_session_status = result.get("session_status")
                    num_messages = len(result.get("messages", []))
                    st.success(f"Round complete: {num_messages} messages generated")
                st.session_state.coord_executing = False
                st.rerun()

    with col4:
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()

    # Status message
    if st.session_state.coord_executing:
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

    # Page title
    st.title("🤝 Multi-Agent Coordination")
    st.markdown(
        "Create structured debates, discussions, and decision-making sessions "
        "with multiple agents using various topologies and debate formats."
    )

    st.divider()

    # Sidebar for session config
    render_sidebar()

    # Main content
    render_timeline()

    st.divider()

    # Control buttons
    render_controls()


if __name__ == "__main__":
    main()
