"""Unit tests for scenario domain dataclasses.

Tests InteractionEdge, AgentSpec, PhaseSpec, OutcomeRule, PayoffMatrix,
UIHints, and ScenarioSpec. All frozen dataclasses with no external dependencies.
"""

from __future__ import annotations

import pytest

from semantic.domain.scenario import (
    AgentSpec,
    InteractionEdge,
    OutcomeRule,
    PayoffMatrix,
    PhaseSpec,
    ScenarioSpec,
    UIHints,
)

pytestmark = pytest.mark.unit


class TestInteractionEdge:
    def test_defaults(self) -> None:
        edge = InteractionEdge(from_agent="a", to_agent="b")
        assert edge.from_agent == "a"
        assert edge.to_agent == "b"
        assert edge.channel == "public"

    def test_custom_channel(self) -> None:
        edge = InteractionEdge(from_agent="a", to_agent="b", channel="private")
        assert edge.channel == "private"

    def test_frozen(self) -> None:
        edge = InteractionEdge(from_agent="a", to_agent="b")
        with pytest.raises(AttributeError):
            edge.channel = "private"  # type: ignore[misc]

    def test_equality(self) -> None:
        edge_a = InteractionEdge(from_agent="x", to_agent="y", channel="public")
        edge_b = InteractionEdge(from_agent="x", to_agent="y", channel="public")
        assert edge_a == edge_b

    def test_inequality(self) -> None:
        edge_a = InteractionEdge(from_agent="x", to_agent="y")
        edge_b = InteractionEdge(from_agent="y", to_agent="x")
        assert edge_a != edge_b

    def test_hashable(self) -> None:
        edge_a = InteractionEdge(from_agent="x", to_agent="y")
        edge_b = InteractionEdge(from_agent="x", to_agent="y")
        assert hash(edge_a) == hash(edge_b)
        assert len({edge_a, edge_b}) == 1


class TestAgentSpec:
    def test_defaults(self) -> None:
        agent = AgentSpec(key="alice", display_name="Alice")
        assert agent.key == "alice"
        assert agent.display_name == "Alice"
        assert agent.role == "participant"
        assert agent.system_prompt == ""
        assert agent.lifecycle == "ephemeral"
        assert agent.color == ""

    def test_all_fields(self) -> None:
        agent = AgentSpec(
            key="mod",
            display_name="Moderator",
            role="moderator",
            system_prompt="Keep order.",
            lifecycle="persistent",
            color="#ff0000",
        )
        assert agent.role == "moderator"
        assert agent.system_prompt == "Keep order."
        assert agent.lifecycle == "persistent"
        assert agent.color == "#ff0000"

    def test_frozen(self) -> None:
        agent = AgentSpec(key="a", display_name="A")
        with pytest.raises(AttributeError):
            agent.role = "moderator"  # type: ignore[misc]

    def test_equality(self) -> None:
        a = AgentSpec(key="k", display_name="K")
        b = AgentSpec(key="k", display_name="K")
        assert a == b
        assert hash(a) == hash(b)


class TestPhaseSpec:
    def test_defaults(self) -> None:
        phase = PhaseSpec(name="debate", label="Debate", agents=("a", "b"))
        assert phase.name == "debate"
        assert phase.label == "Debate"
        assert phase.agents == ("a", "b")
        assert phase.topology == "round_robin"
        assert phase.debate_format == "free_form"
        assert phase.decision_mode == "none"
        assert phase.initial_prompt == ""
        assert phase.initial_prompt_template == ""
        assert phase.max_turns == 0
        assert phase.auto_rounds == 3
        assert phase.interactions is None

    def test_all_fields(self) -> None:
        edges = (
            InteractionEdge(from_agent="a", to_agent="b"),
            InteractionEdge(from_agent="b", to_agent="a", channel="private"),
        )
        phase = PhaseSpec(
            name="vote",
            label="Voting Round",
            agents=("a", "b", "c"),
            topology="star",
            debate_format="structured",
            decision_mode="majority",
            initial_prompt="Begin voting.",
            initial_prompt_template="{agent} votes now.",
            max_turns=10,
            auto_rounds=5,
            interactions=edges,
        )
        assert phase.topology == "star"
        assert phase.debate_format == "structured"
        assert phase.decision_mode == "majority"
        assert phase.max_turns == 10
        assert phase.auto_rounds == 5
        assert phase.interactions is not None
        assert len(phase.interactions) == 2
        assert phase.interactions[1].channel == "private"

    def test_frozen(self) -> None:
        phase = PhaseSpec(name="p", label="P", agents=("x",))
        with pytest.raises(AttributeError):
            phase.max_turns = 5  # type: ignore[misc]

    def test_nested_interactions(self) -> None:
        edge = InteractionEdge(from_agent="sender", to_agent="receiver")
        phase = PhaseSpec(
            name="chat",
            label="Chat",
            agents=("sender", "receiver"),
            interactions=(edge,),
        )
        assert phase.interactions == (edge,)
        assert phase.interactions[0].from_agent == "sender"


class TestOutcomeRule:
    def test_defaults(self) -> None:
        rule = OutcomeRule(type="vote")
        assert rule.type == "vote"
        assert rule.pattern == ""
        assert rule.choices is None
        assert rule.display == "table"

    def test_all_fields(self) -> None:
        rule = OutcomeRule(
            type="regex",
            pattern=r"CHOICE:\s*(\w+)",
            choices=("cooperate", "defect"),
            display="card",
        )
        assert rule.pattern == r"CHOICE:\s*(\w+)"
        assert rule.choices == ("cooperate", "defect")
        assert rule.display == "card"

    def test_frozen(self) -> None:
        rule = OutcomeRule(type="vote")
        with pytest.raises(AttributeError):
            rule.type = "regex"  # type: ignore[misc]


class TestPayoffMatrix:
    def test_construction(self) -> None:
        matrix = PayoffMatrix(
            labels=("cooperate", "defect"),
            matrix=(("3,3", "0,5"), ("5,0", "1,1")),
        )
        assert matrix.labels == ("cooperate", "defect")
        assert matrix.matrix[0][0] == "3,3"
        assert matrix.matrix[1][1] == "1,1"

    def test_frozen(self) -> None:
        matrix = PayoffMatrix(labels=("a",), matrix=(("1",),))
        with pytest.raises(AttributeError):
            matrix.labels = ("b",)  # type: ignore[misc]

    def test_equality(self) -> None:
        a = PayoffMatrix(labels=("x",), matrix=(("v",),))
        b = PayoffMatrix(labels=("x",), matrix=(("v",),))
        assert a == b
        assert hash(a) == hash(b)


class TestUIHints:
    def test_defaults(self) -> None:
        ui = UIHints()
        assert ui.layout == "auto"
        assert ui.column_count == 2
        assert ui.show_memory_panel is False
        assert ui.show_interaction_graph is False
        assert ui.max_visible_messages == 50
        assert ui.phase_controls == "per_phase"

    def test_all_fields(self) -> None:
        ui = UIHints(
            layout="grid",
            column_count=3,
            show_memory_panel=True,
            show_interaction_graph=True,
            max_visible_messages=100,
            phase_controls="global",
        )
        assert ui.layout == "grid"
        assert ui.column_count == 3
        assert ui.show_memory_panel is True
        assert ui.show_interaction_graph is True
        assert ui.max_visible_messages == 100
        assert ui.phase_controls == "global"

    def test_frozen(self) -> None:
        ui = UIHints()
        with pytest.raises(AttributeError):
            ui.layout = "grid"  # type: ignore[misc]


class TestScenarioSpec:
    @staticmethod
    def _make_agents() -> dict[str, AgentSpec]:
        return {
            "alice": AgentSpec(key="alice", display_name="Alice"),
            "bob": AgentSpec(key="bob", display_name="Bob", role="moderator"),
        }

    @staticmethod
    def _make_phases() -> tuple[PhaseSpec, ...]:
        return (
            PhaseSpec(name="intro", label="Introduction", agents=("alice", "bob")),
            PhaseSpec(
                name="debate",
                label="Debate",
                agents=("alice", "bob"),
                max_turns=10,
            ),
        )

    def test_minimal(self) -> None:
        spec = ScenarioSpec(
            id="test-scenario",
            title="Test",
            description="A test scenario.",
            agents=self._make_agents(),
            phases=self._make_phases(),
        )
        assert spec.id == "test-scenario"
        assert spec.title == "Test"
        assert spec.description == "A test scenario."
        assert len(spec.agents) == 2
        assert len(spec.phases) == 2
        assert spec.ui == UIHints()
        assert spec.outcome is None
        assert spec.payoff is None

    def test_all_fields(self) -> None:
        outcome = OutcomeRule(type="vote", choices=("yes", "no"))
        payoff = PayoffMatrix(
            labels=("yes", "no"),
            matrix=(("1,1", "0,2"), ("2,0", "0,0")),
        )
        ui = UIHints(layout="grid", show_interaction_graph=True)

        spec = ScenarioSpec(
            id="full",
            title="Full Scenario",
            description="All fields set.",
            agents=self._make_agents(),
            phases=self._make_phases(),
            ui=ui,
            outcome=outcome,
            payoff=payoff,
        )
        assert spec.ui.layout == "grid"
        assert spec.outcome is not None
        assert spec.outcome.type == "vote"
        assert spec.payoff is not None
        assert spec.payoff.labels == ("yes", "no")

    def test_frozen(self) -> None:
        spec = ScenarioSpec(
            id="s",
            title="T",
            description="D",
            agents=self._make_agents(),
            phases=self._make_phases(),
        )
        with pytest.raises(AttributeError):
            spec.title = "Changed"  # type: ignore[misc]

    def test_agents_dict_access(self) -> None:
        agents = self._make_agents()
        spec = ScenarioSpec(
            id="s",
            title="T",
            description="D",
            agents=agents,
            phases=self._make_phases(),
        )
        assert "alice" in spec.agents
        assert spec.agents["bob"].role == "moderator"

    def test_phase_ordering_preserved(self) -> None:
        phases = self._make_phases()
        spec = ScenarioSpec(
            id="s",
            title="T",
            description="D",
            agents=self._make_agents(),
            phases=phases,
        )
        assert spec.phases[0].name == "intro"
        assert spec.phases[1].name == "debate"
        assert spec.phases[1].max_turns == 10

    def test_nested_phase_with_interactions(self) -> None:
        edge = InteractionEdge(from_agent="alice", to_agent="bob")
        phase = PhaseSpec(
            name="talk",
            label="Talk",
            agents=("alice", "bob"),
            interactions=(edge,),
        )
        spec = ScenarioSpec(
            id="nested",
            title="Nested",
            description="With interactions.",
            agents=self._make_agents(),
            phases=(phase,),
        )
        assert spec.phases[0].interactions is not None
        assert spec.phases[0].interactions[0].to_agent == "bob"
