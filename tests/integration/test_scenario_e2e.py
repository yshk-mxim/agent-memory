"""End-to-end integration tests for the scenario pipeline.

Validates the full flow: YAML -> Pydantic -> Domain -> Template Resolution
without requiring a running server or network access.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

import pytest
from demo.lib.template_resolver import (
    extract_phase_refs,
    has_template_refs,
    resolve_template,
)

from agent_memory.adapters.config.scenario_loader import discover_scenarios, load_scenario
from agent_memory.domain.scenario import PhaseSpec, ScenarioSpec

pytestmark = pytest.mark.integration

SCENARIOS_DIR = Path(__file__).resolve().parents[2] / "demo" / "scenarios"


class TestGossipFullPipeline:
    """YAML -> Pydantic -> Domain -> Template for gossip.yaml."""

    @pytest.fixture(scope="class")
    def gossip(self) -> ScenarioSpec:
        return load_scenario(SCENARIOS_DIR / "gossip.yaml")

    def test_three_agents(self, gossip: ScenarioSpec) -> None:
        assert set(gossip.agents.keys()) == {"alice", "bob", "eve"}

    def test_three_phases(self, gossip: ScenarioSpec) -> None:
        assert len(gossip.phases) == 3
        phase_names = [p.name for p in gossip.phases]
        assert phase_names == ["alice_bob", "alice_eve", "reunion"]

    def test_phase_agent_membership(self, gossip: ScenarioSpec) -> None:
        alice_bob, alice_eve, reunion = gossip.phases
        assert alice_bob.agents == ("alice", "bob")
        assert alice_eve.agents == ("alice", "eve")
        assert reunion.agents == ("alice", "bob", "eve")

    def test_reunion_has_per_agent_templates(self, gossip: ScenarioSpec) -> None:
        reunion = gossip.phases[2]
        assert reunion.per_agent_prompt_templates
        assert set(reunion.per_agent_prompt_templates.keys()) == {"alice", "bob", "eve"}
        for tmpl in reunion.per_agent_prompt_templates.values():
            assert has_template_refs(tmpl)

    def test_reunion_per_agent_templates_ref_prior_phases(self, gossip: ScenarioSpec) -> None:
        reunion = gossip.phases[2]
        all_refs: set[str] = set()
        for tmpl in reunion.per_agent_prompt_templates.values():
            all_refs.update(extract_phase_refs(tmpl))
        assert all_refs == {"alice_bob", "alice_eve"}

    def test_reunion_per_agent_templates_resolve_with_perspective(
        self,
        gossip: ScenarioSpec,
    ) -> None:
        reunion = gossip.phases[2]
        phase_messages = {
            "alice_bob": [
                {"sender_name": "Alice", "content": "Bob is so forgetful."},
                {"sender_name": "Bob", "content": "Don't tell Eve I said this."},
            ],
            "alice_eve": [
                {"sender_name": "Alice", "content": "Eve, you won't believe this."},
                {"sender_name": "Eve", "content": "Tell me everything!"},
            ],
        }
        agent_names = {k: a.display_name for k, a in gossip.agents.items()}

        # Alice sees her own messages as "You"
        alice_resolved = resolve_template(
            reunion.per_agent_prompt_templates["alice"],
            phase_messages,
            agent_names,
        )
        assert "You: Bob is so forgetful." in alice_resolved
        assert "Bob: Don't tell Eve I said this." in alice_resolved
        assert "You: Eve, you won't believe this." in alice_resolved
        assert "${" not in alice_resolved

        # Bob sees his messages as "You", Alice's as "Alice"
        bob_resolved = resolve_template(
            reunion.per_agent_prompt_templates["bob"],
            phase_messages,
            agent_names,
        )
        assert "Alice: Bob is so forgetful." in bob_resolved
        assert "You: Don't tell Eve I said this." in bob_resolved

        # Eve sees her messages as "You", Alice's as "Alice"
        eve_resolved = resolve_template(
            reunion.per_agent_prompt_templates["eve"],
            phase_messages,
            agent_names,
        )
        assert "Alice: Eve, you won't believe this." in eve_resolved
        assert "You: Tell me everything!" in eve_resolved

    def test_non_template_phases_have_initial_prompt(self, gossip: ScenarioSpec) -> None:
        alice_bob, alice_eve, _ = gossip.phases
        assert alice_bob.initial_prompt
        assert alice_eve.initial_prompt
        assert not has_template_refs(alice_bob.initial_prompt)


class TestPrisonersDilemmaFullPipeline:
    """YAML -> Pydantic -> Domain -> Template for prisoners_dilemma.yaml."""

    @pytest.fixture(scope="class")
    def pd_spec(self) -> ScenarioSpec:
        return load_scenario(SCENARIOS_DIR / "prisoners_dilemma.yaml")

    def test_agents(self, pd_spec: ScenarioSpec) -> None:
        assert set(pd_spec.agents.keys()) == {"warden", "marco", "danny", "analyst"}

    def test_warden_is_moderator(self, pd_spec: ScenarioSpec) -> None:
        assert pd_spec.agents["warden"].role == "moderator"

    def test_marco_and_danny_are_participants(self, pd_spec: ScenarioSpec) -> None:
        assert pd_spec.agents["marco"].role == "participant"
        assert pd_spec.agents["danny"].role == "participant"

    def test_interrogation_danny_uses_kv_cache_not_templates(self, pd_spec: ScenarioSpec) -> None:
        danny_phase = pd_spec.phases[1]
        assert danny_phase.name == "interrogation_danny"
        # With persistent KV cache, danny phase uses plain prompt (no template refs)
        refs = extract_phase_refs(danny_phase.initial_prompt_template)
        assert refs == set()
        assert danny_phase.initial_prompt  # Has plain initial prompt

    def test_outcome_analysis_queries_prisoners_directly(self, pd_spec: ScenarioSpec) -> None:
        outcome_phase = pd_spec.phases[4]
        assert outcome_phase.name == "outcome_analysis"
        # New design: analyst queries prisoners directly (no template injection)
        assert outcome_phase.agents == ("marco", "danny", "analyst")
        assert "marco" in outcome_phase.per_agent_prompt_templates
        assert "danny" in outcome_phase.per_agent_prompt_templates
        assert "analyst" in outcome_phase.per_agent_prompt_templates

    def test_payoff_matrix_present(self, pd_spec: ScenarioSpec) -> None:
        assert pd_spec.payoff is not None
        assert pd_spec.payoff.labels == ("Keep Silent", "Confess")
        assert len(pd_spec.payoff.matrix) == 2

    def test_outcome_rule_present(self, pd_spec: ScenarioSpec) -> None:
        assert pd_spec.outcome is not None
        assert pd_spec.outcome.type == "parse_choice"
        assert pd_spec.outcome.display == "matrix"

    def test_the_yard_has_plain_prompt(self, pd_spec: ScenarioSpec) -> None:
        yard = pd_spec.phases[2]
        assert yard.name == "the_yard"
        assert yard.initial_prompt
        assert not has_template_refs(yard.initial_prompt)

    def test_final_reckoning_has_per_agent_prompts(self, pd_spec: ScenarioSpec) -> None:
        reckoning = pd_spec.phases[3]
        assert reckoning.name == "final_reckoning"
        assert reckoning.initial_prompt  # Shared scene-setting
        assert not has_template_refs(reckoning.initial_prompt)
        # Per-agent instructions: only warden needs special instructions
        assert "warden" in reckoning.per_agent_prompt_templates
        for tmpl in reckoning.per_agent_prompt_templates.values():
            assert not has_template_refs(tmpl)


class TestTemplateResolutionWithRealisticMessages:
    """Template resolution with multi-turn, realistic message data."""

    def test_multi_turn_transcript_format(self) -> None:
        template = "Prior context:\n${debate.messages[judge]}"
        messages = [
            {"sender_name": "Alice", "content": "I believe the answer is 42."},
            {"sender_name": "Bob", "content": "No, it should be 7."},
            {"sender_name": "Alice", "content": "Let me explain my reasoning..."},
            {"sender_name": "Bob", "content": "Fair point, but consider this."},
        ]
        resolved = resolve_template(template, {"debate": messages})

        lines = resolved.split("\n")
        assert lines[0] == "Prior context:"
        assert lines[1] == "Alice: I believe the answer is 42."
        assert lines[2] == "Bob: No, it should be 7."
        assert lines[3] == "Alice: Let me explain my reasoning..."
        assert lines[4] == "Bob: Fair point, but consider this."

    def test_special_characters_preserved(self) -> None:
        template = "${chat.messages[x]}"
        messages = [
            {"sender_name": "Bot", "content": "Price: $50 (25% off!) & free shipping"},
            {"sender_name": "User", "content": 'What about <html> tags & "quotes"?'},
        ]
        resolved = resolve_template(template, {"chat": messages})

        assert "$50" in resolved
        assert "<html>" in resolved
        assert '"quotes"' in resolved

    def test_empty_prior_phase_yields_placeholder(self) -> None:
        template = "History:\n${empty_phase.messages[agent]}"
        resolved = resolve_template(template, {"empty_phase": []})
        assert resolved == "History:\n(no messages yet)"

    def test_multiple_phase_refs_in_single_template(self) -> None:
        template = "Phase A:\n${phase_a.messages[x]}\n\nPhase B:\n${phase_b.messages[y]}"
        phase_messages = {
            "phase_a": [{"sender_name": "Host", "content": "Welcome"}],
            "phase_b": [{"sender_name": "Host", "content": "Goodbye"}],
        }
        resolved = resolve_template(template, phase_messages)

        assert "Phase A:\nHost: Welcome" in resolved
        assert "Phase B:\nHost: Goodbye" in resolved


class TestCrossPhaseDependencyChecking:
    """Verify extract_phase_refs correctly identifies dependencies."""

    def test_linear_dependency_chain(self) -> None:
        phase_a = PhaseSpec(
            name="phase_a",
            label="A",
            agents=("x",),
            initial_prompt="Start here.",
        )
        phase_b = PhaseSpec(
            name="phase_b",
            label="B",
            agents=("x",),
            initial_prompt_template="Prior: ${phase_a.messages[x]}",
        )
        phase_c = PhaseSpec(
            name="phase_c",
            label="C",
            agents=("x",),
            initial_prompt_template=(
                "From A: ${phase_a.messages[x]}\nFrom B: ${phase_b.messages[x]}"
            ),
        )

        assert extract_phase_refs(phase_a.initial_prompt) == set()
        assert extract_phase_refs(phase_b.initial_prompt_template) == {"phase_a"}
        assert extract_phase_refs(phase_c.initial_prompt_template) == {"phase_a", "phase_b"}

    def test_self_reference_detected(self) -> None:
        template = "Recap: ${my_phase.messages[agent]}"
        assert extract_phase_refs(template) == {"my_phase"}

    def test_no_dependencies_for_plain_prompt(self) -> None:
        phase = PhaseSpec(
            name="standalone",
            label="Standalone",
            agents=("a",),
            initial_prompt="No dependencies here.",
        )
        assert extract_phase_refs(phase.initial_prompt) == set()
        assert not has_template_refs(phase.initial_prompt)

    def test_gossip_reunion_depends_on_both_private_phases(self) -> None:
        gossip = load_scenario(SCENARIOS_DIR / "gossip.yaml")
        reunion = gossip.phases[2]
        # Dependencies are now in per_agent_prompt_templates, not initial_prompt_template
        all_deps: set[str] = set()
        all_deps.update(extract_phase_refs(reunion.initial_prompt_template))
        for tmpl in reunion.per_agent_prompt_templates.values():
            all_deps.update(extract_phase_refs(tmpl))
        prior_phase_names = {p.name for p in gossip.phases[:2]}
        assert all_deps == prior_phase_names


class TestScenarioDiscoveryPipeline:
    """discover_scenarios -> load_scenario -> validate ScenarioSpec."""

    def test_discovers_yaml_files(self) -> None:
        discovered = discover_scenarios(SCENARIOS_DIR)
        assert len(discovered) >= 2
        assert "gossip" in discovered
        assert "prisoners_dilemma" in discovered

    def test_discovered_paths_exist(self) -> None:
        discovered = discover_scenarios(SCENARIOS_DIR)
        for stem, path in discovered.items():
            assert path.exists(), f"{stem} path does not exist: {path}"
            assert path.suffix == ".yaml"

    def test_all_loadable_scenarios_are_valid_specs(self) -> None:
        discovered = discover_scenarios(SCENARIOS_DIR)
        loaded_count = 0
        for stem, path in discovered.items():
            try:
                spec = load_scenario(path)
            except Exception:
                logger.debug("Skipping %s: load failed", stem, exc_info=True)
                continue
            loaded_count += 1
            assert isinstance(spec, ScenarioSpec), f"{stem} is not a ScenarioSpec"
            assert spec.id, f"{stem} has empty id"
            assert spec.title, f"{stem} has empty title"
            assert len(spec.phases) >= 1, f"{stem} has no phases"
            for phase in spec.phases:
                for agent_key in phase.agents:
                    assert agent_key in spec.agents, (
                        f"{stem}: phase '{phase.name}' refs unknown agent '{agent_key}'"
                    )
        assert loaded_count >= 2

    def test_empty_directory_returns_empty_dict(self, tmp_path: Path) -> None:
        assert discover_scenarios(tmp_path) == {}

    def test_nonexistent_directory_returns_empty_dict(self, tmp_path: Path) -> None:
        assert discover_scenarios(tmp_path / "nonexistent") == {}
