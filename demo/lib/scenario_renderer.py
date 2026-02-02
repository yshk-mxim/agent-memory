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

            # Global reset
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
        """Render phases in column grid."""
        cols_per_row = self.spec.ui.column_count
        # Split phases that fit in columns vs full-width
        column_phases = [p for p in phases if len(p.agents) <= cols_per_row]
        wide_phases = [p for p in phases if len(p.agents) > cols_per_row]

        # Render column-width phases in rows
        for i in range(0, len(column_phases), cols_per_row):
            batch = column_phases[i : i + cols_per_row]
            cols = st.columns(len(batch))
            for col, phase in zip(cols, batch, strict=False):
                with col:
                    self._render_single_phase(phase)

        # Render wide phases full-width
        for phase in wide_phases:
            st.divider()
            self._render_single_phase(phase)

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
                deps = extract_phase_refs(phase.initial_prompt_template)
                st.caption(f"Requires: {', '.join(deps)}")
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
                with st.spinner("Generating..."):
                    count = phase.auto_rounds * len(phase.agents)
                    api_client.execute_turns(self.base_url, session_id, count)
                st.session_state[exec_key] = False
                st.rerun()
            elif action == "run_round":
                st.session_state[exec_key] = True
                with st.spinner("Running round..."):
                    api_client.execute_round(self.base_url, session_id)
                st.session_state[exec_key] = False
                st.rerun()
            elif action == "refresh":
                st.rerun()

    def _can_create_phase(self, phase: PhaseSpec) -> bool:
        """Check if a phase's template dependencies are satisfied."""
        if not has_template_refs(phase.initial_prompt_template):
            return True
        deps = extract_phase_refs(phase.initial_prompt_template)
        for dep_name in deps:
            dep_key = _phase_key(self.spec.id, dep_name, "session_id")
            if not st.session_state.get(dep_key):
                return False
            # Check that dependency has messages
            msg_key = _phase_key(self.spec.id, dep_name, "messages")
            if not st.session_state.get(msg_key):
                return False
        return True

    def _create_phase_session(self, phase: PhaseSpec) -> str | None:
        """Create a coordination session for a phase."""
        # Resolve template if needed
        initial_prompt = phase.initial_prompt
        if has_template_refs(phase.initial_prompt_template):
            phase_messages: dict[str, list[dict[str, str]]] = {}
            for dep_name in extract_phase_refs(phase.initial_prompt_template):
                dep_session = st.session_state.get(_phase_key(self.spec.id, dep_name, "session_id"))
                if dep_session:
                    msgs = api_client.get_session_messages(self.base_url, dep_session)
                    phase_messages[dep_name] = msgs
            initial_prompt = resolve_template(phase.initial_prompt_template, phase_messages)

        # Build agent configs
        agent_configs = []
        for agent_key in phase.agents:
            agent = self.spec.agents.get(agent_key)
            if not agent:
                continue
            agent_configs.append(
                {
                    "display_name": agent.display_name,
                    "role": agent.role,
                    "system_prompt": agent.system_prompt,
                    "lifecycle": agent.lifecycle,
                }
            )

        result = api_client.create_session(
            self.base_url,
            topology=phase.topology,
            debate_format=phase.debate_format,
            decision_mode=phase.decision_mode,
            agents=agent_configs,
            initial_prompt=initial_prompt,
            max_turns=phase.max_turns,
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

    def _reset_all(self) -> None:
        """Delete all phase sessions and reset state."""
        for phase in self.spec.phases:
            session_key = _phase_key(self.spec.id, phase.name, "session_id")
            session_id = st.session_state.get(session_key)
            if session_id:
                api_client.delete_session(self.base_url, session_id)
            st.session_state[session_key] = None
            st.session_state[_phase_key(self.spec.id, phase.name, "messages")] = []
            st.session_state[_phase_key(self.spec.id, phase.name, "executing")] = False
