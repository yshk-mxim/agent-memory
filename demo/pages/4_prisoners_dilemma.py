"""Prisoner's Dilemma Demo - Game theory with optional communication.

Demonstrates:
- Two prisoners interrogated separately by warden
- Optional communication toggle (The Yard)
- Cross-session context for warden
- Outcome parsing from final decisions
"""

import re
from typing import Any

import httpx
import streamlit as st

SERVER_URL = "http://localhost:8000"

# Agent configurations (permanent lifecycle)
WARDEN = {
    "display_name": "Warden",
    "role": "moderator",
    "system_prompt": (
        "You are a pragmatic corrections officer. Explain the deal plainly and logically. "
        "Keep responses under 3 sentences. You want confessions."
    ),
    "lifecycle": "permanent",
}

MARCO = {
    "display_name": "Marco",
    "role": "participant",
    "system_prompt": (
        "You are Marco, 34, blue-collar worker, first offense. Anxious about your family. "
        "Keep responses under 3 sentences. You're scared but trying to stay calm."
    ),
    "lifecycle": "permanent",
}

DANNY = {
    "display_name": "Danny",
    "role": "participant",
    "system_prompt": (
        "You are Danny, 41, street-smart, been through the system before. "
        "Keep responses under 3 sentences. You know how this game works."
    ),
    "lifecycle": "permanent",
}

# The Deal
DEAL_TEXT = """**The Deal:**
- Both stay quiet: 2 years each
- One talks, one quiet: Talker goes free, quiet gets 10 years
- Both talk: 5 years each"""


def init_dilemma_state() -> None:
    """Initialize prisoner's dilemma session state."""
    if "dilemma_initialized" in st.session_state:
        return

    st.session_state.dilemma_initialized = True
    st.session_state.dilemma_session_1 = None  # Warden & Marco
    st.session_state.dilemma_session_2 = None  # Warden & Danny
    st.session_state.dilemma_session_3 = None  # Marco & Danny (The Yard)
    st.session_state.dilemma_communication_enabled = False
    st.session_state.dilemma_executing = False


def create_session(agents: list[dict], initial_prompt: str, max_turns: int = 6) -> str | None:
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


def format_messages_for_warden(messages: list[dict], prisoner_name: str) -> str:
    """Format interrogation from warden's perspective."""
    lines = [f"During your interrogation with {prisoner_name}:"]
    for msg in messages:
        if msg["sender_id"] == "system":
            continue
        sender = msg["sender_name"]
        content = msg["content"]
        lines.append(f"{sender}: {content}")
    return "\n".join(lines)


def create_warden_session_2_prompt(session_1_id: str) -> str:
    """Create Warden's second interrogation prompt with context from first."""
    s1_messages = get_session_messages(session_1_id)
    warden_context = format_messages_for_warden(s1_messages, "Marco")

    return f"""You are the Warden interrogating Danny (the second prisoner).

{DEAL_TEXT}

You've already interrogated Marco. Here's what happened:

{warden_context}

Now you're with Danny. Use your experience from Marco's interrogation to persuade
Danny to confess. Be strategic."""


def create_yard_prompt(session_1_id: str, session_2_id: str) -> str:
    """Create The Yard prompt with prisoner context."""
    s1_messages = get_session_messages(session_1_id)
    s2_messages = get_session_messages(session_2_id)

    # Extract what each prisoner knows
    marco_lines = []
    danny_lines = []
    for msg in s1_messages:
        if msg["sender_name"] == "Marco":
            marco_lines.append(f"Marco: {msg['content']}")
    for msg in s2_messages:
        if msg["sender_name"] == "Danny":
            danny_lines.append(f"Danny: {msg['content']}")

    marco_context = "\n".join(marco_lines) if marco_lines else "Marco was interrogated."
    danny_context = "\n".join(danny_lines) if danny_lines else "Danny was interrogated."

    return f"""Marco and Danny meet briefly in the prison yard.

{DEAL_TEXT}

What Marco knows from his interrogation:
{marco_context}

What Danny knows from his interrogation:
{danny_context}

You have a few minutes to talk before guards separate you. What do you say?"""


def parse_decision(messages: list[dict], prisoner_name: str) -> str | None:
    """Try to parse final decision from prisoner's messages.

    Looks for keywords: "talk", "confess", "stay quiet", "keep quiet", "stay silent"
    """
    prisoner_messages = [
        msg["content"]
        for msg in messages
        if msg["sender_name"] == prisoner_name
    ]

    if not prisoner_messages:
        return None

    # Check last few messages for decision keywords
    last_messages = " ".join(prisoner_messages[-3:]).lower()

    # Talk keywords
    if any(word in last_messages for word in ["i'll talk", "i'll confess", "i confess", "going to talk"]):
        return "TALK"

    # Quiet keywords
    if any(word in last_messages for word in ["stay quiet", "stay silent", "keep quiet", "won't talk", "not talking"]):
        return "QUIET"

    return None


def calculate_outcome(marco_decision: str | None, danny_decision: str | None) -> tuple[str, str] | None:
    """Calculate outcome based on decisions.

    Returns:
        Tuple of (marco_sentence, danny_sentence) or None if decisions unclear
    """
    if not marco_decision or not danny_decision:
        return None

    if marco_decision == "QUIET" and danny_decision == "QUIET":
        return ("2 years", "2 years")
    elif marco_decision == "TALK" and danny_decision == "QUIET":
        return ("Free", "10 years")
    elif marco_decision == "QUIET" and danny_decision == "TALK":
        return ("10 years", "Free")
    elif marco_decision == "TALK" and danny_decision == "TALK":
        return ("5 years", "5 years")

    return None


def render_interrogation_room(
    title: str,
    session_id: str | None,
    agents_str: str,
    button_prefix: str,
    disabled: bool = False,
) -> None:
    """Render an interrogation room panel."""
    st.subheader(title)
    st.markdown(f"**In the room:** {agents_str}")

    if session_id:
        messages = get_session_messages(session_id)
        if messages:
            with st.container(height=250):
                for msg in messages:
                    if msg["sender_id"] == "system":
                        st.markdown(f"*{msg['content']}*")
                    else:
                        st.markdown(f"**{msg['sender_name']}:** {msg['content']}")
                    st.divider()
        else:
            st.info("No conversation yet. Run turns to start.", icon="💬")
    else:
        st.info("Session not created yet.", icon="ℹ️")

    # Controls
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button(
            "Create Session",
            disabled=session_id is not None or disabled,
            use_container_width=True,
            key=f"{button_prefix}_create",
        ):
            return "CREATE"

    with col_b:
        if st.button(
            "Run 2 Turns",
            disabled=session_id is None or st.session_state.dilemma_executing,
            use_container_width=True,
            key=f"{button_prefix}_run",
        ):
            return "RUN"

    return None


def main() -> None:
    """Main entry point."""
    st.set_page_config(
        page_title="Prisoner's Dilemma",
        page_icon="⚖️",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    init_dilemma_state()

    # Page title
    st.title("⚖️ Prisoner's Dilemma: To Talk or Not To Talk")
    st.markdown(
        "Two prisoners, one warden, and a classic game theory dilemma. "
        "Watch how communication changes the outcome."
    )

    st.divider()

    # Communication toggle
    col_toggle, col_deal = st.columns([1, 2])
    with col_toggle:
        comm_enabled = st.toggle(
            "Enable Prisoner Communication (The Yard)",
            value=st.session_state.dilemma_communication_enabled,
            key="dilemma_comm_toggle",
        )
        if comm_enabled != st.session_state.dilemma_communication_enabled:
            st.session_state.dilemma_communication_enabled = comm_enabled
            st.rerun()

    with col_deal:
        st.markdown(DEAL_TEXT)

    st.divider()

    # Interrogation rooms
    col1, col2 = st.columns(2)

    with col1:
        st.header("Interrogation Room A")
        action_1 = render_interrogation_room(
            "Warden & Marco",
            st.session_state.dilemma_session_1,
            "Warden, Marco",
            "room1",
        )

        if action_1 == "CREATE":
            session_id = create_session(
                agents=[WARDEN, MARCO],
                initial_prompt=f"{DEAL_TEXT}\n\nWarden, explain the deal to Marco. Marco, you're nervous but trying to think clearly.",
                max_turns=8,
            )
            if session_id:
                st.session_state.dilemma_session_1 = session_id
                st.success("Session 1 created!")
                st.rerun()

        elif action_1 == "RUN":
            st.session_state.dilemma_executing = True
            with st.spinner("Running interrogation..."):
                if execute_turns(st.session_state.dilemma_session_1, 2):
                    st.success("2 turns complete!")
            st.session_state.dilemma_executing = False
            st.rerun()

    with col2:
        st.header("Interrogation Room B")
        action_2 = render_interrogation_room(
            "Warden & Danny",
            st.session_state.dilemma_session_2,
            "Warden, Danny",
            "room2",
            disabled=st.session_state.dilemma_session_1 is None,
        )

        if action_2 == "CREATE":
            # Warden's second interrogation includes context from first
            prompt = create_warden_session_2_prompt(st.session_state.dilemma_session_1)
            session_id = create_session(
                agents=[WARDEN, DANNY],
                initial_prompt=prompt,
                max_turns=8,
            )
            if session_id:
                st.session_state.dilemma_session_2 = session_id
                st.success("Session 2 created with warden's context!")
                st.rerun()

        elif action_2 == "RUN":
            st.session_state.dilemma_executing = True
            with st.spinner("Running interrogation..."):
                if execute_turns(st.session_state.dilemma_session_2, 2):
                    st.success("2 turns complete!")
            st.session_state.dilemma_executing = False
            st.rerun()

    st.divider()

    # The Yard (if communication enabled)
    if st.session_state.dilemma_communication_enabled:
        st.header("The Yard 🌤️")
        st.markdown("**Prisoners briefly meet outside**")

        if st.session_state.dilemma_session_3:
            messages = get_session_messages(st.session_state.dilemma_session_3)
            if messages:
                with st.container(height=250):
                    for msg in messages:
                        if msg["sender_id"] == "system":
                            st.markdown(f"*{msg['content']}*")
                        else:
                            st.markdown(f"**{msg['sender_name']}:** {msg['content']}")
                        st.divider()
            else:
                st.info("No conversation yet.", icon="💬")
        else:
            st.info("The Yard session not created yet.", icon="ℹ️")

        col_a, col_b = st.columns(2)
        with col_a:
            if st.button(
                "Create Yard Session",
                disabled=(
                    st.session_state.dilemma_session_1 is None
                    or st.session_state.dilemma_session_2 is None
                    or st.session_state.dilemma_session_3 is not None
                ),
                use_container_width=True,
                type="primary",
                key="yard_create",
            ):
                prompt = create_yard_prompt(
                    st.session_state.dilemma_session_1,
                    st.session_state.dilemma_session_2,
                )
                session_id = create_session(
                    agents=[MARCO, DANNY],
                    initial_prompt=prompt,
                    max_turns=6,
                )
                if session_id:
                    st.session_state.dilemma_session_3 = session_id
                    st.success("Yard session created!")
                    st.rerun()

        with col_b:
            if st.button(
                "Run 2 Turns",
                disabled=st.session_state.dilemma_session_3 is None or st.session_state.dilemma_executing,
                use_container_width=True,
                key="yard_run",
            ):
                st.session_state.dilemma_executing = True
                with st.spinner("Running yard conversation..."):
                    if execute_turns(st.session_state.dilemma_session_3, 2):
                        st.success("2 turns complete!")
                st.session_state.dilemma_executing = False
                st.rerun()

        st.divider()

    # Outcome display
    st.header("Outcome Analysis")

    if st.session_state.dilemma_session_1 and st.session_state.dilemma_session_2:
        # Try to parse decisions
        s1_messages = get_session_messages(st.session_state.dilemma_session_1)
        s2_messages = get_session_messages(st.session_state.dilemma_session_2)

        # Also check yard messages if communication was enabled
        yard_messages = []
        if st.session_state.dilemma_communication_enabled and st.session_state.dilemma_session_3:
            yard_messages = get_session_messages(st.session_state.dilemma_session_3)

        # Parse decisions (prefer yard messages if they exist)
        marco_decision = parse_decision(yard_messages + s1_messages, "Marco")
        danny_decision = parse_decision(yard_messages + s2_messages, "Danny")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Marco's Decision")
            if marco_decision:
                if marco_decision == "TALK":
                    st.error("🗣️ TALK / CONFESS", icon="🗣️")
                else:
                    st.success("🤐 STAY QUIET", icon="🤐")
            else:
                st.info("Decision unclear", icon="❓")

        with col2:
            st.subheader("Danny's Decision")
            if danny_decision:
                if danny_decision == "TALK":
                    st.error("🗣️ TALK / CONFESS", icon="🗣️")
                else:
                    st.success("🤐 STAY QUIET", icon="🤐")
            else:
                st.info("Decision unclear", icon="❓")

        with col3:
            st.subheader("Sentences")
            outcome = calculate_outcome(marco_decision, danny_decision)
            if outcome:
                marco_sentence, danny_sentence = outcome
                st.markdown(f"**Marco:** {marco_sentence}")
                st.markdown(f"**Danny:** {danny_sentence}")

                # Nash equilibrium analysis
                if marco_decision == "TALK" and danny_decision == "TALK":
                    st.warning("Nash Equilibrium: Both confess (suboptimal)", icon="⚠️")
                elif marco_decision == "QUIET" and danny_decision == "QUIET":
                    st.success("Optimal Outcome: Cooperation succeeded!", icon="🎉")
            else:
                st.info("Awaiting decisions...", icon="⏳")
    else:
        st.info("Complete both interrogations to see outcome analysis.", icon="ℹ️")

    st.divider()

    # Reset button
    if st.button("🔄 Reset Scenario", use_container_width=True):
        st.session_state.dilemma_session_1 = None
        st.session_state.dilemma_session_2 = None
        st.session_state.dilemma_session_3 = None
        st.success("Scenario reset!")
        st.rerun()


if __name__ == "__main__":
    main()
