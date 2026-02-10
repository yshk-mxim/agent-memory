# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Yakov Shkolnikov and contributors
from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from agent_memory.adapters.config.scenario_loader import discover_scenarios, load_scenario
from agent_memory.domain.scenario import (
    AgentSpec,
    OutcomeRule,
    PayoffMatrix,
    PhaseSpec,
    ScenarioSpec,
    UIHints,
)

pytestmark = pytest.mark.unit

DEMO_DIR = Path(__file__).resolve().parents[3] / "demo" / "scenarios"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MINIMAL_YAML = {
    "id": "test-scenario",
    "title": "Test Scenario",
    "description": "A minimal scenario for testing",
    "agents": {
        "alice": {"display_name": "Alice", "role": "participant"},
    },
    "phases": [
        {"name": "main", "label": "Main Phase", "agents": ["alice"]},
    ],
}


def _write_yaml(path: Path, data: object) -> Path:
    path.write_text(yaml.dump(data, default_flow_style=False))
    return path


# ---------------------------------------------------------------------------
# load_scenario: real YAML files
# ---------------------------------------------------------------------------

class TestLoadScenarioRealFiles:
    def test_gossip_yaml(self):
        spec = load_scenario(DEMO_DIR / "gossip.yaml")

        assert isinstance(spec, ScenarioSpec)
        assert spec.id == "gossip"
        assert spec.title == "The Gossip Network"
        assert "alice" in spec.agents
        assert "bob" in spec.agents
        assert "eve" in spec.agents
        for agent in spec.agents.values():
            assert isinstance(agent, AgentSpec)
            assert agent.lifecycle == "permanent"
        assert len(spec.phases) == 3
        phase_names = [p.name for p in spec.phases]
        assert phase_names == ["alice_bob", "alice_eve", "reunion"]
        assert spec.ui.layout == "columns"
        assert spec.ui.show_memory_panel is True

    def test_prisoners_dilemma_yaml(self):
        spec = load_scenario(DEMO_DIR / "prisoners_dilemma.yaml")

        assert isinstance(spec, ScenarioSpec)
        assert spec.id == "prisoners-dilemma"
        assert "warden" in spec.agents
        assert "marco" in spec.agents
        assert "danny" in spec.agents
        assert spec.agents["warden"].role == "moderator"
        assert "analyst" in spec.agents
        assert spec.agents["analyst"].lifecycle == "ephemeral"
        assert len(spec.phases) == 5
        phase_names = [p.name for p in spec.phases]
        assert phase_names == [
            "interrogation_marco",
            "interrogation_danny",
            "the_yard",
            "final_reckoning",
            "outcome_analysis",
        ]
        assert spec.outcome is not None
        assert isinstance(spec.outcome, OutcomeRule)
        assert spec.outcome.type == "parse_choice"
        assert spec.outcome.choices == ("keep_silent", "confess")
        assert spec.payoff is not None
        assert isinstance(spec.payoff, PayoffMatrix)
        assert spec.payoff.labels == ("Keep Silent", "Confess")
        assert len(spec.payoff.matrix) == 2

    def test_real_files_produce_frozen_domain_objects(self):
        spec = load_scenario(DEMO_DIR / "gossip.yaml")

        with pytest.raises(AttributeError):
            spec.id = "mutated"  # type: ignore[misc]
        for phase in spec.phases:
            assert isinstance(phase, PhaseSpec)
            assert isinstance(phase.agents, tuple)
        assert isinstance(spec.ui, UIHints)


# ---------------------------------------------------------------------------
# load_scenario: temp YAML (minimal valid)
# ---------------------------------------------------------------------------

class TestLoadScenarioTempFile:
    def test_minimal_valid_scenario(self, tmp_path: Path):
        yaml_path = _write_yaml(tmp_path / "minimal.yaml", MINIMAL_YAML)
        spec = load_scenario(yaml_path)

        assert spec.id == "test-scenario"
        assert spec.title == "Test Scenario"
        assert spec.description == "A minimal scenario for testing"
        assert len(spec.agents) == 1
        assert spec.agents["alice"].display_name == "Alice"
        assert len(spec.phases) == 1
        assert spec.phases[0].name == "main"
        assert spec.outcome is None
        assert spec.payoff is None

    def test_scenario_with_all_optional_fields(self, tmp_path: Path):
        full_data = {
            **MINIMAL_YAML,
            "outcome": {
                "type": "vote_tally",
                "choices": ["yes", "no"],
                "display": "table",
            },
            "payoff": {
                "labels": ["yes", "no"],
                "matrix": [["win/win", "lose/win"], ["win/lose", "lose/lose"]],
            },
            "ui": {
                "layout": "tabs",
                "column_count": 3,
                "show_memory_panel": True,
                "show_interaction_graph": True,
            },
        }
        yaml_path = _write_yaml(tmp_path / "full.yaml", full_data)
        spec = load_scenario(yaml_path)

        assert spec.outcome is not None
        assert spec.outcome.type == "vote_tally"
        assert spec.payoff is not None
        assert spec.ui.layout == "tabs"
        assert spec.ui.column_count == 3


# ---------------------------------------------------------------------------
# load_scenario: invalid inputs
# ---------------------------------------------------------------------------

class TestLoadScenarioInvalid:
    def test_nonexistent_file_raises_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_scenario(Path("/nonexistent/path/scenario.yaml"))

    def test_malformed_yaml_raises(self, tmp_path: Path):
        bad_yaml = tmp_path / "bad.yaml"
        bad_yaml.write_text("{{{{not: valid: yaml: [")
        with pytest.raises(yaml.YAMLError):
            load_scenario(bad_yaml)

    def test_missing_required_id(self, tmp_path: Path):
        data = {k: v for k, v in MINIMAL_YAML.items() if k != "id"}
        yaml_path = _write_yaml(tmp_path / "no_id.yaml", data)
        with pytest.raises(ValidationError, match="id"):
            load_scenario(yaml_path)

    def test_missing_required_title(self, tmp_path: Path):
        data = {k: v for k, v in MINIMAL_YAML.items() if k != "title"}
        yaml_path = _write_yaml(tmp_path / "no_title.yaml", data)
        with pytest.raises(ValidationError, match="title"):
            load_scenario(yaml_path)

    def test_missing_required_phases(self, tmp_path: Path):
        data = {k: v for k, v in MINIMAL_YAML.items() if k != "phases"}
        yaml_path = _write_yaml(tmp_path / "no_phases.yaml", data)
        with pytest.raises(ValidationError, match="phases"):
            load_scenario(yaml_path)

    def test_empty_phases_list(self, tmp_path: Path):
        data = {**MINIMAL_YAML, "phases": []}
        yaml_path = _write_yaml(tmp_path / "empty_phases.yaml", data)
        with pytest.raises(ValidationError, match="phases"):
            load_scenario(yaml_path)

    def test_phase_references_unknown_agent(self, tmp_path: Path):
        data = {
            **MINIMAL_YAML,
            "phases": [
                {"name": "main", "label": "Main", "agents": ["alice", "ghost"]},
            ],
        }
        yaml_path = _write_yaml(tmp_path / "bad_ref.yaml", data)
        with pytest.raises(ValidationError, match=r"ghost.*not found"):
            load_scenario(yaml_path)

    def test_invalid_id_format(self, tmp_path: Path):
        data = {**MINIMAL_YAML, "id": "INVALID ID!"}
        yaml_path = _write_yaml(tmp_path / "bad_id.yaml", data)
        with pytest.raises(ValidationError, match="id"):
            load_scenario(yaml_path)

    def test_empty_yaml_file(self, tmp_path: Path):
        empty = tmp_path / "empty.yaml"
        empty.write_text("")
        with pytest.raises((ValidationError, TypeError, AttributeError)):
            load_scenario(empty)

    def test_yaml_with_wrong_structure(self, tmp_path: Path):
        yaml_path = _write_yaml(tmp_path / "list.yaml", ["not", "a", "dict"])
        with pytest.raises((ValidationError, TypeError, AttributeError)):
            load_scenario(yaml_path)


# ---------------------------------------------------------------------------
# discover_scenarios: real directory
# ---------------------------------------------------------------------------

class TestDiscoverScenariosRealDir:
    def test_finds_demo_scenarios(self):
        scenarios = discover_scenarios(DEMO_DIR)

        assert isinstance(scenarios, dict)
        assert "gossip" in scenarios
        assert "prisoners_dilemma" in scenarios
        for scenario_id, path in scenarios.items():
            assert path.suffix == ".yaml"
            assert path.exists()
            assert path.stem == scenario_id

    def test_returns_sorted_by_filename(self):
        scenarios = discover_scenarios(DEMO_DIR)
        keys = list(scenarios.keys())
        assert keys == sorted(keys)


# ---------------------------------------------------------------------------
# discover_scenarios: temp directories
# ---------------------------------------------------------------------------

class TestDiscoverScenariosTempDir:
    def test_empty_directory(self, tmp_path: Path):
        scenarios = discover_scenarios(tmp_path)
        assert scenarios == {}

    def test_nonexistent_directory(self, tmp_path: Path):
        scenarios = discover_scenarios(tmp_path / "does_not_exist")
        assert scenarios == {}

    def test_ignores_non_yaml_files(self, tmp_path: Path):
        _write_yaml(tmp_path / "valid.yaml", MINIMAL_YAML)
        (tmp_path / "readme.md").write_text("# Not a scenario")
        (tmp_path / "config.json").write_text("{}")
        (tmp_path / "script.py").write_text("print('hello')")
        (tmp_path / "notes.txt").write_text("just notes")

        scenarios = discover_scenarios(tmp_path)

        assert len(scenarios) == 1
        assert "valid" in scenarios

    def test_multiple_yaml_files(self, tmp_path: Path):
        for name in ["alpha", "beta", "gamma"]:
            data = {**MINIMAL_YAML, "id": name}
            _write_yaml(tmp_path / f"{name}.yaml", data)

        scenarios = discover_scenarios(tmp_path)

        assert len(scenarios) == 3
        assert list(scenarios.keys()) == ["alpha", "beta", "gamma"]

    def test_ignores_yml_extension(self, tmp_path: Path):
        _write_yaml(tmp_path / "scenario.yml", MINIMAL_YAML)

        scenarios = discover_scenarios(tmp_path)
        assert scenarios == {}

    def test_ignores_subdirectory_yaml_files(self, tmp_path: Path):
        _write_yaml(tmp_path / "top_level.yaml", MINIMAL_YAML)
        nested = tmp_path / "subdir"
        nested.mkdir()
        _write_yaml(nested / "nested.yaml", MINIMAL_YAML)

        scenarios = discover_scenarios(tmp_path)

        assert len(scenarios) == 1
        assert "top_level" in scenarios
