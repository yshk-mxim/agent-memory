# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Yakov Shkolnikov and contributors
from __future__ import annotations

import pytest
from pydantic import ValidationError

from agent_memory.adapters.config.scenario_models import (
    AgentSpecModel,
    InteractionEdgeModel,
    OutcomeRuleModel,
    PayoffMatrixModel,
    PhaseSpecModel,
    ScenarioSpecModel,
    UIHintsModel,
)
from agent_memory.domain.scenario import (
    AgentSpec,
    InteractionEdge,
    OutcomeRule,
    PayoffMatrix,
    PhaseSpec,
    ScenarioSpec,
    UIHints,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _minimal_agent(**overrides: object) -> dict:
    base = {"display_name": "Alice"}
    base.update(overrides)
    return base


def _minimal_phase(**overrides: object) -> dict:
    base = {"name": "debate", "label": "Debate", "agents": ["alice"]}
    base.update(overrides)
    return base


def _minimal_scenario(**overrides: object) -> dict:
    base: dict = {
        "id": "prisoners-dilemma",
        "title": "Prisoner's Dilemma",
        "agents": {"alice": _minimal_agent()},
        "phases": [_minimal_phase()],
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# AgentSpecModel
# ---------------------------------------------------------------------------


class TestAgentSpecModel:
    def test_valid_defaults(self):
        model = AgentSpecModel(display_name="Alice")
        assert model.role == "participant"
        assert model.system_prompt == ""
        assert model.lifecycle == "ephemeral"
        assert model.color == ""

    def test_to_domain_round_trip(self):
        model = AgentSpecModel(
            display_name="Bob",
            role="judge",
            system_prompt="You are the judge.",
            lifecycle="permanent",
            color="#ff0000",
        )
        domain = model.to_domain("bob")
        assert isinstance(domain, AgentSpec)
        assert domain.key == "bob"
        assert domain.display_name == "Bob"
        assert domain.role == "judge"
        assert domain.system_prompt == "You are the judge."
        assert domain.lifecycle == "permanent"
        assert domain.color == "#ff0000"

    def test_display_name_empty_rejected(self):
        with pytest.raises(ValidationError, match="display_name"):
            AgentSpecModel(display_name="")

    def test_display_name_too_long(self):
        with pytest.raises(ValidationError, match="display_name"):
            AgentSpecModel(display_name="x" * 51)

    def test_display_name_max_length_accepted(self):
        model = AgentSpecModel(display_name="x" * 50)
        assert len(model.display_name) == 50

    @pytest.mark.parametrize("bad_role", ["admin", "Player", "OBSERVER", ""])
    def test_invalid_role_rejected(self, bad_role: str):
        with pytest.raises(ValidationError, match="role"):
            AgentSpecModel(display_name="A", role=bad_role)

    @pytest.mark.parametrize("role", ["participant", "observer", "narrator", "judge", "moderator"])
    def test_valid_roles(self, role: str):
        model = AgentSpecModel(display_name="A", role=role)
        assert model.role == role

    def test_system_prompt_max_length(self):
        model = AgentSpecModel(display_name="A", system_prompt="p" * 4000)
        assert len(model.system_prompt) == 4000

    def test_system_prompt_too_long(self):
        with pytest.raises(ValidationError, match="system_prompt"):
            AgentSpecModel(display_name="A", system_prompt="p" * 4001)

    @pytest.mark.parametrize("bad_lc", ["temp", "Ephemeral", ""])
    def test_invalid_lifecycle_rejected(self, bad_lc: str):
        with pytest.raises(ValidationError, match="lifecycle"):
            AgentSpecModel(display_name="A", lifecycle=bad_lc)


# ---------------------------------------------------------------------------
# InteractionEdgeModel
# ---------------------------------------------------------------------------


class TestInteractionEdgeModel:
    def test_valid_defaults(self):
        model = InteractionEdgeModel(from_agent="a", to_agent="b")
        assert model.channel == "public"

    def test_to_domain_round_trip(self):
        model = InteractionEdgeModel(from_agent="alice", to_agent="bob", channel="whisper")
        domain = model.to_domain()
        assert isinstance(domain, InteractionEdge)
        assert domain.from_agent == "alice"
        assert domain.to_agent == "bob"
        assert domain.channel == "whisper"

    @pytest.mark.parametrize("channel", ["public", "whisper", "observe"])
    def test_valid_channels(self, channel: str):
        model = InteractionEdgeModel(from_agent="a", to_agent="b", channel=channel)
        assert model.channel == channel

    @pytest.mark.parametrize("bad_ch", ["private", "broadcast", "Public", ""])
    def test_invalid_channel_rejected(self, bad_ch: str):
        with pytest.raises(ValidationError, match="channel"):
            InteractionEdgeModel(from_agent="a", to_agent="b", channel=bad_ch)


# ---------------------------------------------------------------------------
# PhaseSpecModel
# ---------------------------------------------------------------------------


class TestPhaseSpecModel:
    def test_valid_with_defaults(self):
        model = PhaseSpecModel(**_minimal_phase())
        assert model.topology == "round_robin"
        assert model.debate_format == "free_form"
        assert model.decision_mode == "none"
        assert model.initial_prompt == ""
        assert model.initial_prompt_template == ""
        assert model.max_turns == 0
        assert model.auto_rounds == 3
        assert model.interactions is None

    def test_to_domain_round_trip(self):
        model = PhaseSpecModel(
            name="round_one",
            label="Round 1",
            agents=["alice", "bob"],
            topology="star",
            max_turns=10,
            auto_rounds=5,
            interactions=[
                InteractionEdgeModel(from_agent="alice", to_agent="bob", channel="whisper")
            ],
        )
        domain = model.to_domain()
        assert isinstance(domain, PhaseSpec)
        assert domain.name == "round_one"
        assert domain.agents == ("alice", "bob")
        assert domain.max_turns == 10
        assert domain.auto_rounds == 5
        assert domain.interactions is not None
        assert len(domain.interactions) == 1
        assert domain.interactions[0].channel == "whisper"

    def test_to_domain_no_interactions(self):
        model = PhaseSpecModel(**_minimal_phase())
        domain = model.to_domain()
        assert domain.interactions is None

    @pytest.mark.parametrize(
        "bad_name", ["", "A", "1start", "has space", "UPPER", "-dash", "a" * 31]
    )
    def test_invalid_name_rejected(self, bad_name: str):
        with pytest.raises(ValidationError, match="name"):
            PhaseSpecModel(**_minimal_phase(name=bad_name))

    def test_name_max_30_chars_accepted(self):
        valid_name = "a" + "b" * 29
        model = PhaseSpecModel(**_minimal_phase(name=valid_name))
        assert model.name == valid_name

    def test_label_empty_rejected(self):
        with pytest.raises(ValidationError, match="label"):
            PhaseSpecModel(**_minimal_phase(label=""))

    def test_label_too_long(self):
        with pytest.raises(ValidationError, match="label"):
            PhaseSpecModel(**_minimal_phase(label="L" * 81))

    def test_agents_empty_accepted(self):
        model = PhaseSpecModel(**_minimal_phase(agents=[]))
        assert model.agents == []

    def test_agents_too_many_rejected(self):
        with pytest.raises(ValidationError, match="agents"):
            PhaseSpecModel(**_minimal_phase(agents=[f"a{i}" for i in range(31)]))

    def test_agents_max_30_accepted(self):
        agents_list = [f"a{i}" for i in range(30)]
        model = PhaseSpecModel(**_minimal_phase(agents=agents_list))
        assert len(model.agents) == 30

    def test_max_turns_negative_rejected(self):
        with pytest.raises(ValidationError, match="max_turns"):
            PhaseSpecModel(**_minimal_phase(max_turns=-1))

    def test_max_turns_over_limit_rejected(self):
        with pytest.raises(ValidationError, match="max_turns"):
            PhaseSpecModel(**_minimal_phase(max_turns=1001))

    def test_max_turns_boundary_values(self):
        assert PhaseSpecModel(**_minimal_phase(max_turns=0)).max_turns == 0
        assert PhaseSpecModel(**_minimal_phase(max_turns=1000)).max_turns == 1000

    def test_auto_rounds_zero_rejected(self):
        with pytest.raises(ValidationError, match="auto_rounds"):
            PhaseSpecModel(**_minimal_phase(auto_rounds=0))

    def test_auto_rounds_over_limit_rejected(self):
        with pytest.raises(ValidationError, match="auto_rounds"):
            PhaseSpecModel(**_minimal_phase(auto_rounds=51))

    def test_initial_prompt_max_length(self):
        model = PhaseSpecModel(**_minimal_phase(initial_prompt="x" * 50000))
        assert len(model.initial_prompt) == 50000

    def test_initial_prompt_too_long(self):
        with pytest.raises(ValidationError, match="initial_prompt"):
            PhaseSpecModel(**_minimal_phase(initial_prompt="x" * 50001))

    def test_initial_prompt_template_too_long(self):
        with pytest.raises(ValidationError, match="initial_prompt_template"):
            PhaseSpecModel(**_minimal_phase(initial_prompt_template="x" * 50001))


# ---------------------------------------------------------------------------
# OutcomeRuleModel
# ---------------------------------------------------------------------------


class TestOutcomeRuleModel:
    def test_valid_with_defaults(self):
        model = OutcomeRuleModel(type="summary")
        assert model.pattern == ""
        assert model.choices is None
        assert model.display == "table"

    def test_to_domain_round_trip(self):
        model = OutcomeRuleModel(
            type="parse_choice",
            pattern=r"I choose (\w+)",
            choices=["cooperate", "defect"],
            display="matrix",
        )
        domain = model.to_domain()
        assert isinstance(domain, OutcomeRule)
        assert domain.type == "parse_choice"
        assert domain.pattern == r"I choose (\w+)"
        assert domain.choices == ("cooperate", "defect")
        assert domain.display == "matrix"

    def test_to_domain_no_choices(self):
        domain = OutcomeRuleModel(type="summary").to_domain()
        assert domain.choices is None

    @pytest.mark.parametrize("valid_type", ["parse_choice", "vote_tally", "summary"])
    def test_valid_types(self, valid_type: str):
        model = OutcomeRuleModel(type=valid_type)
        assert model.type == valid_type

    @pytest.mark.parametrize("bad_type", ["unknown", "ParseChoice", ""])
    def test_invalid_type_rejected(self, bad_type: str):
        with pytest.raises(ValidationError, match="type"):
            OutcomeRuleModel(type=bad_type)

    @pytest.mark.parametrize("bad_display", ["chart", "Graph", ""])
    def test_invalid_display_rejected(self, bad_display: str):
        with pytest.raises(ValidationError, match="display"):
            OutcomeRuleModel(type="summary", display=bad_display)


# ---------------------------------------------------------------------------
# PayoffMatrixModel
# ---------------------------------------------------------------------------


class TestPayoffMatrixModel:
    def test_valid_construction(self):
        model = PayoffMatrixModel(
            labels=["cooperate", "defect"],
            matrix=[["3,3", "0,5"], ["5,0", "1,1"]],
        )
        assert len(model.labels) == 2
        assert len(model.matrix) == 2

    def test_to_domain_round_trip(self):
        model = PayoffMatrixModel(
            labels=["cooperate", "defect"],
            matrix=[["3,3", "0,5"], ["5,0", "1,1"]],
        )
        domain = model.to_domain()
        assert isinstance(domain, PayoffMatrix)
        assert domain.labels == ("cooperate", "defect")
        assert domain.matrix == (("3,3", "0,5"), ("5,0", "1,1"))

    def test_labels_fewer_than_two_rejected(self):
        with pytest.raises(ValidationError, match="labels"):
            PayoffMatrixModel(labels=["only_one"], matrix=[["1"], ["2"]])

    def test_matrix_fewer_than_two_rows_rejected(self):
        with pytest.raises(ValidationError, match="matrix"):
            PayoffMatrixModel(labels=["a", "b"], matrix=[["1,1"]])


# ---------------------------------------------------------------------------
# UIHintsModel
# ---------------------------------------------------------------------------


class TestUIHintsModel:
    def test_defaults(self):
        model = UIHintsModel()
        assert model.layout == "auto"
        assert model.column_count == 2
        assert model.show_memory_panel is False
        assert model.show_interaction_graph is False
        assert model.max_visible_messages == 50
        assert model.phase_controls == "per_phase"

    def test_to_domain_round_trip(self):
        model = UIHintsModel(
            layout="tabs",
            column_count=3,
            show_memory_panel=True,
            show_interaction_graph=True,
            max_visible_messages=200,
            phase_controls="global",
        )
        domain = model.to_domain()
        assert isinstance(domain, UIHints)
        assert domain.layout == "tabs"
        assert domain.column_count == 3
        assert domain.show_memory_panel is True
        assert domain.show_interaction_graph is True
        assert domain.max_visible_messages == 200
        assert domain.phase_controls == "global"

    @pytest.mark.parametrize("layout", ["auto", "columns", "tabs", "single"])
    def test_valid_layouts(self, layout: str):
        model = UIHintsModel(layout=layout)
        assert model.layout == layout

    @pytest.mark.parametrize("bad_layout", ["grid", "Auto", ""])
    def test_invalid_layout_rejected(self, bad_layout: str):
        with pytest.raises(ValidationError, match="layout"):
            UIHintsModel(layout=bad_layout)

    def test_column_count_below_min_rejected(self):
        with pytest.raises(ValidationError, match="column_count"):
            UIHintsModel(column_count=0)

    def test_column_count_above_max_rejected(self):
        with pytest.raises(ValidationError, match="column_count"):
            UIHintsModel(column_count=5)

    def test_column_count_boundaries(self):
        assert UIHintsModel(column_count=1).column_count == 1
        assert UIHintsModel(column_count=4).column_count == 4

    def test_max_visible_messages_below_min_rejected(self):
        with pytest.raises(ValidationError, match="max_visible_messages"):
            UIHintsModel(max_visible_messages=0)

    def test_max_visible_messages_above_max_rejected(self):
        with pytest.raises(ValidationError, match="max_visible_messages"):
            UIHintsModel(max_visible_messages=501)

    def test_max_visible_messages_boundaries(self):
        assert UIHintsModel(max_visible_messages=1).max_visible_messages == 1
        assert UIHintsModel(max_visible_messages=500).max_visible_messages == 500

    @pytest.mark.parametrize("bad_pc", ["none", "PerPhase", ""])
    def test_invalid_phase_controls_rejected(self, bad_pc: str):
        with pytest.raises(ValidationError, match="phase_controls"):
            UIHintsModel(phase_controls=bad_pc)


# ---------------------------------------------------------------------------
# ScenarioSpecModel
# ---------------------------------------------------------------------------


class TestScenarioSpecModel:
    def test_valid_minimal(self):
        model = ScenarioSpecModel(**_minimal_scenario())
        assert model.id == "prisoners-dilemma"
        assert model.title == "Prisoner's Dilemma"
        assert model.description == ""
        assert model.outcome is None
        assert model.payoff is None

    def test_defaults(self):
        model = ScenarioSpecModel(**_minimal_scenario())
        assert isinstance(model.ui, UIHintsModel)
        assert model.ui.layout == "auto"
        assert model.description == ""

    def test_to_domain_round_trip(self):
        data = _minimal_scenario(
            description="A classic game theory scenario.",
            outcome={"type": "parse_choice", "choices": ["cooperate", "defect"]},
            payoff={
                "labels": ["cooperate", "defect"],
                "matrix": [["3,3", "0,5"], ["5,0", "1,1"]],
            },
        )
        model = ScenarioSpecModel(**data)
        domain = model.to_domain()

        assert isinstance(domain, ScenarioSpec)
        assert domain.id == "prisoners-dilemma"
        assert domain.title == "Prisoner's Dilemma"
        assert domain.description == "A classic game theory scenario."
        assert "alice" in domain.agents
        assert isinstance(domain.agents["alice"], AgentSpec)
        assert len(domain.phases) == 1
        assert isinstance(domain.ui, UIHints)
        assert domain.outcome is not None
        assert domain.outcome.choices == ("cooperate", "defect")
        assert domain.payoff is not None
        assert domain.payoff.labels == ("cooperate", "defect")

    def test_to_domain_without_optional_fields(self):
        domain = ScenarioSpecModel(**_minimal_scenario()).to_domain()
        assert domain.outcome is None
        assert domain.payoff is None

    # --- ID validation ---

    @pytest.mark.parametrize("bad_id", ["", "A", "1start", "has space", "UPPER", "a" * 51])
    def test_invalid_id_rejected(self, bad_id: str):
        with pytest.raises(ValidationError, match="id"):
            ScenarioSpecModel(**_minimal_scenario(id=bad_id))

    def test_id_with_hyphens_accepted(self):
        model = ScenarioSpecModel(**_minimal_scenario(id="my-scenario-1"))
        assert model.id == "my-scenario-1"

    def test_id_max_50_chars_accepted(self):
        long_id = "a" + "b" * 49
        model = ScenarioSpecModel(**_minimal_scenario(id=long_id))
        assert model.id == long_id

    # --- Title validation ---

    def test_title_empty_rejected(self):
        with pytest.raises(ValidationError, match="title"):
            ScenarioSpecModel(**_minimal_scenario(title=""))

    def test_title_too_long(self):
        with pytest.raises(ValidationError, match="title"):
            ScenarioSpecModel(**_minimal_scenario(title="T" * 101))

    def test_title_max_length_accepted(self):
        model = ScenarioSpecModel(**_minimal_scenario(title="T" * 100))
        assert len(model.title) == 100

    # --- Description validation ---

    def test_description_max_length(self):
        model = ScenarioSpecModel(**_minimal_scenario(description="d" * 500))
        assert len(model.description) == 500

    def test_description_too_long(self):
        with pytest.raises(ValidationError, match="description"):
            ScenarioSpecModel(**_minimal_scenario(description="d" * 501))

    # --- Phases validation ---

    def test_phases_empty_rejected(self):
        with pytest.raises(ValidationError, match="phases"):
            ScenarioSpecModel(**_minimal_scenario(phases=[]))

    def test_phases_too_many_rejected(self):
        phases = [_minimal_phase(name=f"phase{i}") for i in range(21)]
        with pytest.raises(ValidationError, match="phases"):
            ScenarioSpecModel(**_minimal_scenario(phases=phases))

    def test_phases_max_20_accepted(self):
        phases = [_minimal_phase(name=f"phase{i}") for i in range(20)]
        model = ScenarioSpecModel(**_minimal_scenario(phases=phases))
        assert len(model.phases) == 20

    # --- Cross-validation: phase agent refs ---

    def test_phase_referencing_nonexistent_agent_rejected(self):
        data = _minimal_scenario(phases=[_minimal_phase(agents=["alice", "bob"])])
        with pytest.raises(ValidationError, match=r"bob.*not found in agents"):
            ScenarioSpecModel(**data)

    def test_phase_agents_all_present_accepted(self):
        data = _minimal_scenario(
            agents={
                "alice": _minimal_agent(),
                "bob": _minimal_agent(display_name="Bob"),
            },
            phases=[_minimal_phase(agents=["alice", "bob"])],
        )
        model = ScenarioSpecModel(**data)
        assert len(model.agents) == 2

    def test_multiple_phases_cross_validated(self):
        data = _minimal_scenario(
            agents={
                "alice": _minimal_agent(),
                "bob": _minimal_agent(display_name="Bob"),
            },
            phases=[
                _minimal_phase(name="phase_a", agents=["alice"]),
                _minimal_phase(name="phase_b", agents=["bob"]),
                _minimal_phase(name="phase_c", agents=["alice", "bob"]),
            ],
        )
        model = ScenarioSpecModel(**data)
        assert len(model.phases) == 3

    def test_cross_validation_error_message_lists_available_keys(self):
        data = _minimal_scenario(phases=[_minimal_phase(agents=["ghost"])])
        with pytest.raises(ValidationError, match="Available:"):
            ScenarioSpecModel(**data)

    # --- Empty agents dict with phases ---

    def test_empty_agents_with_phase_referencing_agents_rejected(self):
        data = _minimal_scenario(agents={}, phases=[_minimal_phase(agents=["alice"])])
        with pytest.raises(ValidationError, match=r"alice.*not found"):
            ScenarioSpecModel(**data)

    # --- Multi-agent to_domain ---

    def test_to_domain_preserves_agent_keys(self):
        data = _minimal_scenario(
            agents={
                "alice": _minimal_agent(),
                "bob": _minimal_agent(display_name="Bob"),
            },
            phases=[_minimal_phase(agents=["alice", "bob"])],
        )
        domain = ScenarioSpecModel(**data).to_domain()
        assert domain.agents["alice"].key == "alice"
        assert domain.agents["bob"].key == "bob"
