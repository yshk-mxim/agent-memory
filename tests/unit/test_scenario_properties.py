# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Yakov Shkolnikov and contributors
"""Property-based tests for scenario domain, Pydantic models, and template resolver."""

from __future__ import annotations

import re

import pytest
from demo.lib.template_resolver import has_template_refs, resolve_template
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from agent_memory.adapters.config.scenario_models import (
    AgentSpecModel,
    InteractionEdgeModel,
    PhaseSpecModel,
    ScenarioSpecModel,
)
from agent_memory.domain.scenario import (
    AgentSpec,
    InteractionEdge,
    PhaseSpec,
    ScenarioSpec,
)

pytestmark = [pytest.mark.unit, pytest.mark.property]

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

valid_key = st.from_regex(r"[a-z][a-z0-9_]{0,9}", fullmatch=True)
valid_name = st.text(
    min_size=1,
    max_size=50,
    alphabet=st.characters(categories=("L", "N", "P", "Z")),
)
valid_role = st.sampled_from(["participant", "observer", "moderator", "judge", "narrator"])
valid_lifecycle = st.sampled_from(["ephemeral", "permanent"])
valid_color = st.from_regex(r"#[0-9A-Fa-f]{6}", fullmatch=True) | st.just("")
valid_channel = st.sampled_from(["public", "whisper", "observe"])

_KEY_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,29}$")


# ---------------------------------------------------------------------------
# Property 1: AgentSpec round-trips through Pydantic
# ---------------------------------------------------------------------------


class TestAgentSpecRoundTrip:
    @given(
        key=valid_key,
        display_name=valid_name,
        role=valid_role,
        system_prompt=st.text(max_size=100),
        lifecycle=valid_lifecycle,
        color=valid_color,
    )
    @settings(max_examples=100)
    def test_agent_spec_survives_pydantic_round_trip(
        self,
        key: str,
        display_name: str,
        role: str,
        system_prompt: str,
        lifecycle: str,
        color: str,
    ) -> None:
        model = AgentSpecModel(
            display_name=display_name,
            role=role,
            system_prompt=system_prompt,
            lifecycle=lifecycle,
            color=color,
        )
        domain = model.to_domain(key)

        assert domain.key == key
        assert domain.display_name == display_name
        assert domain.role == role
        assert domain.system_prompt == system_prompt
        assert domain.lifecycle == lifecycle
        assert domain.color == color
        assert isinstance(domain, AgentSpec)


# ---------------------------------------------------------------------------
# Property 2: ScenarioSpec with N agents and M phases always validates
# ---------------------------------------------------------------------------


class TestScenarioSpecConstruction:
    @given(
        n_agents=st.integers(min_value=1, max_value=10),
        n_phases=st.integers(min_value=1, max_value=5),
        data=st.data(),
    )
    @settings(max_examples=100)
    def test_valid_agents_and_phases_always_construct(
        self,
        n_agents: int,
        n_phases: int,
        data: st.DataObject,
    ) -> None:
        agent_keys = [f"agent{i}" for i in range(n_agents)]
        agents: dict[str, AgentSpec] = {}
        for ak in agent_keys:
            display = data.draw(valid_name, label=f"name_{ak}")
            agents[ak] = AgentSpec(key=ak, display_name=display)

        phases: list[PhaseSpec] = []
        for p_idx in range(n_phases):
            subset_size = data.draw(
                st.integers(min_value=1, max_value=n_agents),
                label=f"phase{p_idx}_size",
            )
            chosen = data.draw(
                st.lists(
                    st.sampled_from(agent_keys),
                    min_size=subset_size,
                    max_size=subset_size,
                ),
                label=f"phase{p_idx}_agents",
            )
            phases.append(
                PhaseSpec(
                    name=f"phase{p_idx}",
                    label=f"Phase {p_idx}",
                    agents=tuple(chosen),
                )
            )

        spec = ScenarioSpec(
            id="prop-test",
            title="Property Test",
            description="Generated scenario",
            agents=agents,
            phases=tuple(phases),
        )

        assert len(spec.agents) == n_agents
        assert len(spec.phases) == n_phases
        for phase in spec.phases:
            for agent_ref in phase.agents:
                assert agent_ref in spec.agents


# ---------------------------------------------------------------------------
# Property 3: Template resolver is idempotent on resolved output
# ---------------------------------------------------------------------------


class TestTemplateResolverIdempotent:
    @given(
        text=st.text(
            min_size=0,
            max_size=200,
            alphabet=st.characters(blacklist_characters="$"),
        ),
    )
    @settings(max_examples=100)
    def test_resolved_text_without_refs_is_identity(self, text: str) -> None:
        result = resolve_template(text, {})
        assert result == text

    @given(
        text=st.text(
            min_size=0,
            max_size=200,
            alphabet=st.characters(blacklist_characters="$"),
        ),
    )
    @settings(max_examples=100)
    def test_double_resolve_is_idempotent(self, text: str) -> None:
        once = resolve_template(text, {})
        twice = resolve_template(once, {})
        assert once == twice


# ---------------------------------------------------------------------------
# Property 4: Invalid keys rejected by Pydantic
# ---------------------------------------------------------------------------


_INVALID_KEY_CHARS = st.characters(
    whitelist_categories=("Lu", "Nd", "P", "S", "Z"),
)

_INVALID_KEYS = (
    st.text(min_size=1, max_size=30, alphabet=_INVALID_KEY_CHARS)
    | st.just("")
    | st.from_regex(r"[0-9][a-z0-9_]*", fullmatch=True)
    | st.from_regex(r"[A-Z][a-zA-Z0-9_]*", fullmatch=True)
    | st.from_regex(r"[a-z][a-z0-9_]{30,40}", fullmatch=True)
)


class TestInvalidKeysRejected:
    @given(bad_key=_INVALID_KEYS)
    @settings(max_examples=100)
    def test_phase_name_rejects_invalid_keys(self, bad_key: str) -> None:
        assume(not _KEY_PATTERN.match(bad_key))
        with pytest.raises(ValidationError):
            PhaseSpecModel(
                name=bad_key,
                label="Test Phase",
                agents=["a"],
            )


# ---------------------------------------------------------------------------
# Property 5: has_template_refs matches iff pattern exists
# ---------------------------------------------------------------------------


_WORD = st.from_regex(r"\w+", fullmatch=True).filter(lambda s: len(s) <= 20)


class TestHasTemplateRefsProperty:
    @given(phase=_WORD, agent=_WORD, prefix=st.text(max_size=30), suffix=st.text(max_size=30))
    @settings(max_examples=100)
    def test_true_when_pattern_present(
        self,
        phase: str,
        agent: str,
        prefix: str,
        suffix: str,
    ) -> None:
        template = f"{prefix}${{{phase}.messages[{agent}]}}{suffix}"
        assert has_template_refs(template) is True

    @given(
        text=st.text(
            min_size=0,
            max_size=200,
            alphabet=st.characters(blacklist_characters="$"),
        ),
    )
    @settings(max_examples=100)
    def test_false_when_no_dollar_sign(self, text: str) -> None:
        assert has_template_refs(text) is False

    @given(
        text=st.text(min_size=0, max_size=200),
    )
    @settings(max_examples=100)
    def test_result_matches_regex_search(self, text: str) -> None:
        pattern = re.compile(r"\$\{(\w+)\.messages\[(\w+)\]\}")
        expected = bool(pattern.search(text))
        assert has_template_refs(text) is expected


# ---------------------------------------------------------------------------
# Property: InteractionEdge round-trips through Pydantic
# ---------------------------------------------------------------------------


class TestInteractionEdgeRoundTrip:
    @given(
        from_agent=valid_key,
        to_agent=valid_key,
        channel=valid_channel,
    )
    @settings(max_examples=100)
    def test_edge_survives_pydantic_round_trip(
        self,
        from_agent: str,
        to_agent: str,
        channel: str,
    ) -> None:
        model = InteractionEdgeModel(
            from_agent=from_agent,
            to_agent=to_agent,
            channel=channel,
        )
        domain = model.to_domain()

        assert domain == InteractionEdge(
            from_agent=from_agent,
            to_agent=to_agent,
            channel=channel,
        )


# ---------------------------------------------------------------------------
# Property: ScenarioSpecModel round-trips through Pydantic to domain
# ---------------------------------------------------------------------------


class TestScenarioSpecModelRoundTrip:
    @given(
        n_agents=st.integers(min_value=1, max_value=5),
        n_phases=st.integers(min_value=1, max_value=3),
        data=st.data(),
    )
    @settings(max_examples=100)
    def test_pydantic_to_domain_round_trip(
        self,
        n_agents: int,
        n_phases: int,
        data: st.DataObject,
    ) -> None:
        agent_keys = [f"agent{i}" for i in range(n_agents)]
        agents_dict: dict[str, AgentSpecModel] = {}
        for ak in agent_keys:
            display = data.draw(valid_name, label=f"display_{ak}")
            role = data.draw(valid_role, label=f"role_{ak}")
            agents_dict[ak] = AgentSpecModel(
                display_name=display,
                role=role,
            )

        phases_list: list[PhaseSpecModel] = []
        for p_idx in range(n_phases):
            chosen = data.draw(
                st.lists(
                    st.sampled_from(agent_keys),
                    min_size=1,
                    max_size=n_agents,
                ),
                label=f"phase{p_idx}_agents",
            )
            phases_list.append(
                PhaseSpecModel(
                    name=f"phase{p_idx}",
                    label=f"Phase {p_idx}",
                    agents=chosen,
                )
            )

        model = ScenarioSpecModel(
            id="roundtrip-test",
            title="Round Trip",
            description="Testing",
            agents=agents_dict,
            phases=phases_list,
        )
        domain = model.to_domain()

        assert domain.id == "roundtrip-test"
        assert len(domain.agents) == n_agents
        assert len(domain.phases) == n_phases
        for ak in agent_keys:
            assert ak in domain.agents
            assert domain.agents[ak].key == ak
        for phase, phase_model in zip(domain.phases, phases_list):
            assert phase.name == phase_model.name
            assert set(phase.agents) == set(phase_model.agents)
