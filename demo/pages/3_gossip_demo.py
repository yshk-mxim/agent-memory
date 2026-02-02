"""Gossip Demo - Three-session scenario with cross-session memory.

Demonstrates:
- Session 1: Alice & Bob gossip about Eve (private)
- Session 2: Alice & Eve gossip about Bob (private)
- Session 3: The Reunion - all three with cross-session context
"""

import time
from typing import Any

import httpx
import streamlit as st

SERVER_URL = "http://localhost:8000"

# Agent configurations (permanent lifecycle for long-term memory)
ALICE = {
    "display_name": "Alice",
    "role": "participant",
    "system_prompt": "You are Alice. Witty, loves gossip and inside jokes. Keep responses under 4 sentences.",
    "lifecycle": "permanent",
}

BOB = {
    "display_name": "Bob",
    "role": "participant",
    "system_prompt": "You are Bob. Friendly but a bit oblivious. Keep responses under 4 sentences.",
    "lifecycle": "permanent",
}

EVE = {
    "display_name": "Eve",
    "role": "participant",
    "system_prompt": "You are Eve. Perceptive and curious. Notices when people hide things. Keep responses under 4 sentences.",
    "lifecycle": "permanent",
}


def init_gossip_state() -> None:
    """Initialize gossip demo session state."""
    if "gossip_initialized" in st.session_state:
        return

    st.session_state.gossip_initialized = True
    st.session_state.gossip_session_1 = None  # Alice & Bob
    st.session_state.gossip_session_2 = None  # Alice & Eve
    st.session_state.gossip_session_3 = None  # The Reunion
    st.session_state.gossip_executing = False


def create_session(agents: list[dict], initial_prompt: str, max_turns: int = 10) -> str | None:
    """Create a coordination session."""
    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.post(
                f"{SERVER_URL}/v1/coordination/sessions",
                json={
                    "topology": "round_robin",
                    "debate_format": "open",
                    "decision_mode": "consensus",
                    "agents": agents,
                    "initial_prompt": initial_prompt,
                    "max_turns": max_turns,
                },
            )
            if resp.status_code == 201:
                data = resp.json()
                return data["session_id"]
            else:
                st.error(f"Failed to create session: {resp.status_code}")
                return None
    except Exception as e:
        st.error(f"Connection error: {e}")
        return None


def get_session_messages(session_id: str) -> list[dict]:
    """Get all messages in a session."""
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


def execute_turns(session_id: str, num_turns: int) -> bool:
    """Execute multiple turns in a session."""
    try:
        for _ in range(num_turns):
            with httpx.Client(timeout=120.0) as client:
                resp = client.post(f"{SERVER_URL}/v1/coordination/sessions/{session_id}/turn")
                if resp.status_code != 200:
                    st.error(f"Turn execution failed: {resp.status_code}")
                    return False
        return True
    except Exception as e:
        st.error(f"Execution error: {e}")
        return False


def format_messages_for_perspective(messages: list[dict], perspective_agent: str) -> str:
    """Format messages from a specific agent's perspective.

    Args:
        messages: List of message dicts
        perspective_agent: Agent name whose perspective we're formatting for

    Returns:
        Formatted string of conversation from that agent's perspective
    """
    lines = []
    for msg in messages:
        if msg["sender_id"] == "system":
            continue
        sender = msg["sender_name"]
        content = msg["content"]
        if sender == perspective_agent:
            lines.append(f"You said: {content}")
        else:
            lines.append(f"{sender} said: {content}")
    return "\n".join(lines)


def create_reunion_prompt(session_1_id: str, session_2_id: str) -> str:
    """Create reunion prompt with cross-session context.

    Fetches messages from Sessions 1 and 2 and composes a prompt that:
    1. Provides each agent with context from their private conversations
    2. Instructs agents to be socially aware - don't repeat private gossip
    3. Encourages inside jokes and allusions without betraying trust
    """
    s1_messages = get_session_messages(session_1_id)
    s2_messages = get_session_messages(session_2_id)

    alice_s1_context = format_messages_for_perspective(s1_messages, "Alice")
    alice_s2_context = format_messages_for_perspective(s2_messages, "Alice")
    bob_s1_context = format_messages_for_perspective(s1_messages, "Bob")
    eve_s2_context = format_messages_for_perspective(s2_messages, "Eve")

    return f"""The three of you are meeting together for the first time.

IMPORTANT RULE: Each agent has private knowledge from earlier conversations,
but you must ONLY say things you'd be comfortable saying in front of everyone.
If someone said something nasty or embarrassing about another person in private,
you may laugh or hint at inside jokes, but you must NOT repeat the private
gossip directly. You are socially aware — you know what's appropriate to say
when everyone is present.

Context that Alice remembers (from private chat with Bob):
{alice_s1_context}

Context that Alice remembers (from private chat with Eve):
{alice_s2_context}

Context that Bob remembers (from private chat with Alice):
{bob_s1_context}

Context that Eve remembers (from private chat with Alice):
{eve_s2_context}

Now you're all in the same room. Be natural — reference shared experiences
and inside jokes, but don't betray anyone's trust. Start the conversation!"""


def render_session_panel(title: str, session_id: str | None, agents_str: str) -> None:
    """Render a session panel with messages."""
    st.subheader(title)
    st.markdown(f"**Participants:** {agents_str}")

    if session_id:
        messages = get_session_messages(session_id)
        if messages:
            with st.container(height=300):
                for msg in messages:
                    if msg["sender_id"] == "system":
                        st.markdown(f"*{msg['content']}*")
                    else:
                        st.markdown(f"**{msg['sender_name']}:** {msg['content']}")
                    st.divider()
        else:
            st.info("No messages yet. Run turns to start conversation.", icon="💬")
    else:
        st.info("Session not created yet.", icon="ℹ️")


def render_memory_panel() -> None:
    """Render memory inspection panel."""
    st.subheader("Agent Memory Status")

    try:
        with httpx.Client(timeout=5.0) as client:
            resp = client.get(f"{SERVER_URL}/v1/agents/list")
            if resp.status_code == 200:
                data = resp.json()
                agents = data.get("agents", [])

                # Filter to Alice, Bob, Eve
                gossip_agents = [
                    a for a in agents
                    if any(name in a["agent_id"] for name in ["Alice", "Bob", "Eve"])
                ]

                if gossip_agents:
                    cols = st.columns(3)
                    for idx, agent in enumerate(gossip_agents[:3]):
                        with cols[idx]:
                            st.markdown(f"**{agent['agent_id'][:20]}...**")
                            st.markdown(f"Tier: {agent['tier']}")
                            st.markdown(f"Tokens: {agent['tokens']:,}")
                            st.markdown(f"Lifecycle: {agent.get('lifecycle', 'ephemeral')}")
                else:
                    st.info("No agent caches found yet.", icon="ℹ️")
            else:
                st.error("Failed to fetch agent list")
    except Exception as e:
        st.error(f"Connection error: {e}")


def main() -> None:
    """Main entry point."""
    st.set_page_config(
        page_title="Gossip Demo",
        page_icon="🗣️",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    init_gossip_state()

    # Page title
    st.title("🗣️ Gossip Demo: Cross-Session Memory")
    st.markdown(
        "Three agents, three sessions. Watch how Alice carries private context "
        "from separate conversations into a group reunion — but stays socially aware."
    )

    st.divider()

    # Phase 1 & 2: Private conversations
    col1, col2 = st.columns(2)

    with col1:
        st.header("Phase 1: Alice & Bob")
        render_session_panel(
            "Private Gossip Session",
            st.session_state.gossip_session_1,
            "Alice, Bob"
        )

        col_a, col_b = st.columns(2)
        with col_a:
            if st.button(
                "Create Session 1",
                disabled=st.session_state.gossip_session_1 is not None,
                use_container_width=True,
                key="gossip_create_s1",
            ):
                session_id = create_session(
                    agents=[ALICE, BOB],
                    initial_prompt="You're having a private chat. Alice, start by asking Bob what he thinks about Eve.",
                    max_turns=6,
                )
                if session_id:
                    st.session_state.gossip_session_1 = session_id
                    st.success("Session 1 created!")
                    st.rerun()

        with col_b:
            if st.button(
                "Run 3 Turns",
                disabled=st.session_state.gossip_session_1 is None or st.session_state.gossip_executing,
                use_container_width=True,
                key="gossip_run_s1",
            ):
                st.session_state.gossip_executing = True
                with st.spinner("Generating conversation..."):
                    if execute_turns(st.session_state.gossip_session_1, 3):
                        st.success("3 turns complete!")
                st.session_state.gossip_executing = False
                st.rerun()

    with col2:
        st.header("Phase 2: Alice & Eve")
        render_session_panel(
            "Private Gossip Session",
            st.session_state.gossip_session_2,
            "Alice, Eve"
        )

        col_a, col_b = st.columns(2)
        with col_a:
            if st.button(
                "Create Session 2",
                disabled=st.session_state.gossip_session_2 is not None,
                use_container_width=True,
                key="gossip_create_s2",
            ):
                session_id = create_session(
                    agents=[ALICE, EVE],
                    initial_prompt="You're having a private chat. Alice, start by asking Eve what she thinks about Bob.",
                    max_turns=6,
                )
                if session_id:
                    st.session_state.gossip_session_2 = session_id
                    st.success("Session 2 created!")
                    st.rerun()

        with col_b:
            if st.button(
                "Run 3 Turns",
                disabled=st.session_state.gossip_session_2 is None or st.session_state.gossip_executing,
                use_container_width=True,
                key="gossip_run_s2",
            ):
                st.session_state.gossip_executing = True
                with st.spinner("Generating conversation..."):
                    if execute_turns(st.session_state.gossip_session_2, 3):
                        st.success("3 turns complete!")
                st.session_state.gossip_executing = False
                st.rerun()

    st.divider()

    # Phase 3: The Reunion
    st.header("Phase 3: The Reunion 🎉")
    render_session_panel(
        "All Three Together",
        st.session_state.gossip_session_3,
        "Alice, Bob, Eve"
    )

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button(
            "Create Reunion Session",
            disabled=(
                st.session_state.gossip_session_1 is None
                or st.session_state.gossip_session_2 is None
                or st.session_state.gossip_session_3 is not None
            ),
            use_container_width=True,
            type="primary",
            key="gossip_create_s3",
        ):
            # Create reunion prompt with cross-session context
            reunion_prompt = create_reunion_prompt(
                st.session_state.gossip_session_1,
                st.session_state.gossip_session_2,
            )

            session_id = create_session(
                agents=[ALICE, BOB, EVE],
                initial_prompt=reunion_prompt,
                max_turns=9,
            )
            if session_id:
                st.session_state.gossip_session_3 = session_id
                st.success("Reunion session created with cross-session context!")
                st.rerun()

    with col_b:
        if st.button(
            "Run 3 Turns",
            disabled=st.session_state.gossip_session_3 is None or st.session_state.gossip_executing,
            use_container_width=True,
            key="gossip_run_s3",
        ):
            st.session_state.gossip_executing = True
            with st.spinner("Generating reunion conversation..."):
                if execute_turns(st.session_state.gossip_session_3, 3):
                    st.success("3 turns complete!")
            st.session_state.gossip_executing = False
            st.rerun()

    st.divider()

    # Memory panel
    render_memory_panel()

    st.divider()

    # Reset button
    if st.button("🔄 Reset All Sessions", use_container_width=True, key="gossip_reset"):
        st.session_state.gossip_session_1 = None
        st.session_state.gossip_session_2 = None
        st.session_state.gossip_session_3 = None
        st.success("All sessions reset!")
        st.rerun()


if __name__ == "__main__":
    main()
