"""Scenario renderer — renders any ScenarioSpec as a Streamlit UI.

This is the core shared component. Each demo page loads a YAML scenario
and passes the resulting ScenarioSpec to this renderer.
"""

from __future__ import annotations

import logging

import streamlit as st

from demo.lib import api_client
from demo.lib.control_bar import render_round_controls
from demo.lib.message_timeline import build_agent_colors, render_timeline
from demo.lib.template_resolver import (
    extract_phase_refs,
    has_template_refs,
    resolve_template,
)
from semantic.domain.scenario import PhaseSpec, ScenarioSpec

logger = logging.getLogger(__name__)

_MAX_MEMORY_COLUMNS = 4


def _state_key(scenario_id: str, suffix: str) -> str:
    return f"scn_{scenario_id}_{suffix}"


def _phase_key(scenario_id: str, phase_name: str, suffix: str) -> str:
    return f"scn_{scenario_id}_{phase_name}_{suffix}"


class ScenarioRenderer:
    """Renders a ScenarioSpec as a complete Streamlit page."""

    def __init__(self, spec: ScenarioSpec, base_url: str) -> None:
        """Initialize renderer with scenario spec and API base URL."""
        self.spec = spec
        self.base_url = base_url
        self.agent_colors = build_agent_colors(spec.agents)
        self._init_state()

    def _init_state(self) -> None:
        """Initialize session_state for this scenario."""
        key = _state_key(self.spec.id, "initialized")
        if key in st.session_state:
            return
        st.session_state[key] = True
        for phase in self.spec.phases:
            sid = self.spec.id
            st.session_state[_phase_key(sid, phase.name, "session_id")] = None
            st.session_state[_phase_key(sid, phase.name, "messages")] = []
            st.session_state[_phase_key(sid, phase.name, "executing")] = False

    def render(self) -> None:
        """Main render entry point."""
        self._render_sidebar()
        self._render_phases()
        if self.spec.ui.show_memory_panel:
            st.divider()
            self._render_memory_panel()

    def _render_sidebar(self) -> None:
        """Render scenario info and agent list in sidebar."""
        with st.sidebar:
            st.title(self.spec.title)
            if self.spec.description:
                st.markdown(self.spec.description)

            # Connection check
            stats = api_client.get_agent_stats(self.base_url)
            if stats is not None:
                st.success("Server connected")
            else:
                st.error("Server not reachable. Start with: `semantic serve`")
            st.divider()

            st.subheader("Agents")
            for agent in self.spec.agents.values():
                color = self.agent_colors.get(agent.display_name, "#888")
                st.markdown(
                    f'<span style="color:{color};">&#9679;</span> '
                    f"**{agent.display_name}** ({agent.role})",
                    unsafe_allow_html=True,
                )

            st.divider()

            if self.spec.ui.show_run_all:
                all_done = all(
                    st.session_state.get(_phase_key(self.spec.id, p.name, "session_id"))
                    for p in self.spec.phases
                )
                col_run, col_reset = st.columns(2)
                with col_run:
                    if st.button(
                        "Run All" if not all_done else "Re-run All",
                        use_container_width=True,
                        type="primary",
                        key=_state_key(self.spec.id, "run_all"),
                    ):
                        self._reset_all()
                        self._run_all_phases()
                        st.rerun()
                with col_reset:
                    if st.button(
                        "Reset All",
                        use_container_width=True,
                        key=_state_key(self.spec.id, "reset_all"),
                    ):
                        self._reset_all()
                        st.rerun()

    def _render_phases(self) -> None:
        """Render phases based on ui.layout."""
        layout = self.spec.ui.layout
        phases = self.spec.phases

        if layout == "columns" and len(phases) > 1:
            self._render_columns(phases)
        elif layout == "tabs" and len(phases) > 1:
            self._render_tabs(phases)
        else:
            for phase in phases:
                self._render_single_phase(phase)
                st.divider()

    def _render_columns(self, phases: tuple[PhaseSpec, ...]) -> None:
        """Render phases in column grid, preserving phase order.

        Consecutive narrow phases (≤column_count agents) share a row.
        Wide phases and row breaks render full-width with a divider.
        """
        cols_per_row = self.spec.ui.column_count
        narrow_batch: list[PhaseSpec] = []

        for phase in phases:
            if len(phase.agents) <= cols_per_row:
                narrow_batch.append(phase)
                if len(narrow_batch) == cols_per_row:
                    cols = st.columns(cols_per_row)
                    for col, p in zip(cols, narrow_batch, strict=False):
                        with col:
                            self._render_single_phase(p)
                    narrow_batch = []
            else:
                # Flush any pending narrow phases first
                if narrow_batch:
                    cols = st.columns(len(narrow_batch))
                    for col, p in zip(cols, narrow_batch, strict=False):
                        with col:
                            self._render_single_phase(p)
                    narrow_batch = []
                st.divider()
                self._render_single_phase(phase)

        # Flush remaining narrow phases
        if narrow_batch:
            cols = st.columns(len(narrow_batch))
            for col, p in zip(cols, narrow_batch, strict=False):
                with col:
                    self._render_single_phase(p)

    def _render_tabs(self, phases: tuple[PhaseSpec, ...]) -> None:
        """Render phases as tabs."""
        tab_labels = [p.label for p in phases]
        tabs = st.tabs(tab_labels)
        for tab, phase in zip(tabs, phases, strict=False):
            with tab:
                self._render_single_phase(phase)

    def _render_single_phase(self, phase: PhaseSpec) -> None:
        """Render one phase: header, create/run, timeline, controls."""
        sid = self.spec.id
        pname = phase.name
        session_key = _phase_key(sid, pname, "session_id")
        exec_key = _phase_key(sid, pname, "executing")
        session_id = st.session_state.get(session_key)
        is_executing = st.session_state.get(exec_key, False)

        st.subheader(phase.label)

        # Show agents for this phase
        phase_agent_names = [
            self.spec.agents[k].display_name for k in phase.agents if k in self.spec.agents
        ]
        st.caption(f"Participants: {', '.join(phase_agent_names)}")

        if session_id is None:
            # Check if template deps are satisfied
            can_create = self._can_create_phase(phase)
            if st.button(
                "Start",
                disabled=not can_create or is_executing,
                use_container_width=True,
                type="primary",
                key=_phase_key(sid, pname, "create_btn"),
            ):
                new_id = self._create_phase_session(phase)
                if new_id:
                    st.session_state[session_key] = new_id
                    st.rerun()
            if not can_create:
                st.caption("Waiting for prior phases to complete.")
        else:
            # Fetch and display messages
            messages = api_client.get_session_messages(self.base_url, session_id)
            st.session_state[_phase_key(sid, pname, "messages")] = messages

            render_timeline(
                messages,
                self.agent_colors,
                max_visible=self.spec.ui.max_visible_messages,
                height=350,
            )

            # Controls
            action = render_round_controls(
                is_executing=is_executing,
                is_active=True,
                key_prefix=_phase_key(sid, pname, "ctrl"),
                auto_rounds=phase.auto_rounds,
            )

            if action == "run_turns":
                st.session_state[exec_key] = True
                count = phase.auto_rounds * len(phase.agents)
                self._stream_turns(session_id, count)
                st.session_state[exec_key] = False
                st.rerun()
            elif action == "run_round":
                st.session_state[exec_key] = True
                self._stream_turns(session_id, len(phase.agents))
                st.session_state[exec_key] = False
                st.rerun()
            elif action == "refresh":
                st.rerun()

    def _can_create_phase(self, phase: PhaseSpec) -> bool:
        """Check if a phase's dependencies are satisfied.

        Checks both template refs (for ephemeral agents) and prior phase
        completion (for permanent agents needing KV cache carryover).
        """
        sid = self.spec.id

        # Template dependencies (explicit ${phase.messages[agent]} refs)
        all_deps: set[str] = set()
        all_deps.update(extract_phase_refs(phase.initial_prompt_template))
        for tmpl in phase.per_agent_prompt_templates.values():
            all_deps.update(extract_phase_refs(tmpl))
        for dep_name in all_deps:
            dep_key = _phase_key(sid, dep_name, "session_id")
            if not st.session_state.get(dep_key):
                return False
            msg_key = _phase_key(sid, dep_name, "messages")
            if not st.session_state.get(msg_key):
                return False

        # Prior phase dependencies (permanent agents with prior history)
        phase_idx = next(
            (i for i, p in enumerate(self.spec.phases) if p.name == phase.name), 0
        )
        for agent_key in phase.agents:
            agent = self.spec.agents.get(agent_key)
            if not agent or agent.lifecycle != "permanent":
                continue
            for prior_phase in self.spec.phases[:phase_idx]:
                if agent_key not in prior_phase.agents:
                    continue
                prior_key = _phase_key(sid, prior_phase.name, "messages")
                if not st.session_state.get(prior_key):
                    return False

        return True

    def _build_agent_display_names(self) -> dict[str, str]:
        """Build agent_key -> display_name mapping for template resolution."""
        return {key: agent.display_name for key, agent in self.spec.agents.items()}

    def _fetch_dep_messages(self, phase: PhaseSpec) -> dict[str, list[dict[str, str]]]:
        """Fetch messages from dependency phases referenced by templates."""
        phase_messages: dict[str, list[dict[str, str]]] = {}
        all_templates = [phase.initial_prompt_template]
        all_templates.extend(phase.per_agent_prompt_templates.values())
        dep_names: set[str] = set()
        for tmpl in all_templates:
            dep_names.update(extract_phase_refs(tmpl))
        for dep_name in dep_names:
            dep_session = st.session_state.get(_phase_key(self.spec.id, dep_name, "session_id"))
            if dep_session:
                msgs = api_client.get_session_messages(self.base_url, dep_session)
                phase_messages[dep_name] = msgs
        return phase_messages

    def _stable_agent_id(self, agent_key: str) -> str:
        """Generate a stable agent_id from scenario + agent key."""
        return f"scn_{self.spec.id}_{agent_key}"

    def _collect_prior_messages(
        self, phase: PhaseSpec
    ) -> dict[str, list[dict[str, str]]]:
        """Collect prior phase messages for each permanent agent in this phase.

        For each permanent agent, finds all completed prior phases where the
        agent participated and gathers their messages. This enables KV cache
        prefix matching across phases.
        """
        sid = self.spec.id
        prior: dict[str, list[dict[str, str]]] = {}
        phase_idx = next(
            (i for i, p in enumerate(self.spec.phases) if p.name == phase.name), 0
        )

        for agent_key in phase.agents:
            agent = self.spec.agents.get(agent_key)
            if not agent or agent.lifecycle != "permanent":
                continue

            agent_id = self._stable_agent_id(agent_key)
            agent_msgs: list[dict[str, str]] = []

            # Iterate prior phases in order
            for prior_phase in self.spec.phases[:phase_idx]:
                if agent_key not in prior_phase.agents:
                    continue
                stored = st.session_state.get(
                    _phase_key(sid, prior_phase.name, "messages"), []
                )
                for msg in stored:
                    sender_name = msg.get("sender_name", "")
                    content = msg.get("content", "")
                    if not content:
                        continue
                    # Map sender_name to stable agent_id for prompt role mapping
                    if sender_name == "System":
                        sender_id = "system"
                    else:
                        sender_key = self._agent_key_by_name(sender_name)
                        sender_id = (
                            self._stable_agent_id(sender_key)
                            if sender_key
                            else msg.get("sender_id", "system")
                        )
                    agent_msgs.append({
                        "sender_id": sender_id,
                        "sender_name": sender_name,
                        "content": content,
                    })

            if agent_msgs:
                prior[agent_id] = agent_msgs

        return prior

    def _agent_key_by_name(self, display_name: str) -> str | None:
        """Look up agent key by display name."""
        for key, agent in self.spec.agents.items():
            if agent.display_name == display_name:
                return key
        return None

    def _create_phase_session(self, phase: PhaseSpec) -> str | None:
        """Create a coordination session for a phase."""
        agent_names = self._build_agent_display_names()
        phase_messages = self._fetch_dep_messages(phase)

        # Resolve shared initial prompt template (for ephemeral agents like Analyst)
        initial_prompt = phase.initial_prompt
        if has_template_refs(phase.initial_prompt_template):
            initial_prompt = resolve_template(
                phase.initial_prompt_template,
                phase_messages,
                agent_names,
            )

        # Resolve per-agent private prompt templates
        per_agent_prompts: dict[str, str] | None = None
        if phase.per_agent_prompt_templates:
            per_agent_prompts = {}
            for agent_key, tmpl in phase.per_agent_prompt_templates.items():
                agent = self.spec.agents.get(agent_key)
                if not agent:
                    continue
                resolved = resolve_template(tmpl, phase_messages, agent_names)
                if resolved.strip():
                    per_agent_prompts[agent.display_name] = resolved

        # Build agent configs with stable IDs
        agent_configs = []
        for agent_key in phase.agents:
            agent = self.spec.agents.get(agent_key)
            if not agent:
                continue
            agent_configs.append(
                {
                    "agent_id": self._stable_agent_id(agent_key),
                    "display_name": agent.display_name,
                    "role": agent.role,
                    "system_prompt": agent.system_prompt,
                    "lifecycle": agent.lifecycle,
                }
            )

        # Collect prior messages for permanent agents (cross-phase KV cache)
        prior_agent_messages = self._collect_prior_messages(phase)

        result = api_client.create_session(
            self.base_url,
            topology=phase.topology,
            debate_format=phase.debate_format,
            decision_mode=phase.decision_mode,
            agents=agent_configs,
            initial_prompt=initial_prompt,
            max_turns=phase.max_turns,
            per_agent_prompts=per_agent_prompts,
            persistent_cache_prefix=self.spec.id,
            prior_agent_messages=prior_agent_messages or None,
        )
        if result:
            return result.get("session_id")
        st.error("Failed to create session. Is the server running?")
        return None

    def _render_memory_panel(self) -> None:
        """Show per-agent cache statistics."""
        st.subheader("Agent Memory")
        agents = api_client.get_agent_list(self.base_url)
        if not agents:
            st.info("No cached agents.")
            return

        cols = st.columns(min(len(self.spec.agents), _MAX_MEMORY_COLUMNS))
        for idx, (_key, agent) in enumerate(self.spec.agents.items()):
            col = cols[idx % len(cols)]
            # Find matching caches
            matching = [a for a in agents if agent.display_name in a.get("agent_id", "")]
            with col:
                color = self.agent_colors.get(agent.display_name, "#888")
                st.markdown(
                    f'<span style="color:{color}; font-weight:bold;">{agent.display_name}</span>',
                    unsafe_allow_html=True,
                )
                if matching:
                    total_tokens = sum(a.get("tokens", 0) for a in matching)
                    tiers = {a.get("tier", "?") for a in matching}
                    st.caption(f"{total_tokens:,} tokens | {', '.join(tiers)} | {agent.lifecycle}")
                else:
                    st.caption("No cache")

    def _stream_turns(self, session_id: str, count: int) -> None:
        """Stream multiple turns, showing tokens as they arrive."""
        for _i in range(count):
            placeholder = st.empty()
            agent_name = ""
            accumulated = ""
            color = "#888"

            for event_type, data in api_client.stream_turn(
                self.base_url,
                session_id,
            ):
                if event_type == "turn_start":
                    agent_name = data.get("agent_name", "")
                    color = self.agent_colors.get(agent_name, "#888")
                    accumulated = ""
                elif event_type == "token":
                    accumulated = data.get("accumulated", accumulated)
                    placeholder.markdown(
                        f'<span style="color:{color}; font-weight:bold;">'
                        f"{agent_name}:</span> {accumulated}\u258c",
                        unsafe_allow_html=True,
                    )
                elif event_type == "turn_complete":
                    content = data.get("content", accumulated)
                    placeholder.markdown(
                        f'<span style="color:{color}; font-weight:bold;">'
                        f"{agent_name}:</span> {content}",
                        unsafe_allow_html=True,
                    )
                elif event_type == "error":
                    placeholder.error(data.get("error", "Stream error"))
                    return

    def _run_all_phases(self) -> None:
        """Execute all phases sequentially: create session, run turns, fetch messages."""
        status = st.sidebar.empty()
        for idx, phase in enumerate(self.spec.phases, 1):
            sid = self.spec.id
            pname = phase.name
            session_key = _phase_key(sid, pname, "session_id")

            status.info(f"Phase {idx}/{len(self.spec.phases)}: {phase.label}...")

            # Create session (resolves templates from prior phases)
            new_id = self._create_phase_session(phase)
            if not new_id:
                status.error(f"Failed to create session for {phase.label}")
                return
            st.session_state[session_key] = new_id

            # Run all turns for this phase
            turn_count = phase.auto_rounds * len(phase.agents)
            for _t in range(turn_count):
                api_client.execute_turn(self.base_url, new_id)

            # Fetch and store messages (needed for template resolution in later phases)
            messages = api_client.get_session_messages(self.base_url, new_id)
            st.session_state[_phase_key(sid, pname, "messages")] = messages

        status.success("All phases complete!")

    def _reset_all(self) -> None:
        """Delete all phase sessions, persistent caches, and reset state."""
        for phase in self.spec.phases:
            session_key = _phase_key(self.spec.id, phase.name, "session_id")
            session_id = st.session_state.get(session_key)
            if session_id:
                api_client.delete_session(self.base_url, session_id)
            st.session_state[session_key] = None
            st.session_state[_phase_key(self.spec.id, phase.name, "messages")] = []
            st.session_state[_phase_key(self.spec.id, phase.name, "executing")] = False
        # Clear persistent KV caches for permanent agents
        api_client.delete_persistent_caches(self.base_url, self.spec.id)
