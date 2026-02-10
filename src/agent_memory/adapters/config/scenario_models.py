# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Yakov Shkolnikov and contributors
"""Pydantic validation models for scenario YAML files.

Validates YAML input and converts to frozen domain dataclasses.
"""

from __future__ import annotations

from typing import Self

from pydantic import BaseModel, Field, model_validator

from agent_memory.domain.scenario import (
    AgentSpec,
    InteractionEdge,
    OutcomeRule,
    PayoffMatrix,
    PhaseSpec,
    ScenarioSpec,
    UIHints,
)


class AgentSpecModel(BaseModel):
    """Validated agent definition."""

    key: str = Field(default="", description="Set from dict key during parent validation")
    display_name: str = Field(..., min_length=1, max_length=50)
    role: str = Field(
        default="participant",
        pattern=r"^(participant|observer|narrator|judge|moderator)$",
    )
    system_prompt: str = Field(default="", max_length=4000)
    lifecycle: str = Field(default="ephemeral", pattern=r"^(ephemeral|permanent)$")
    color: str = Field(default="")

    def to_domain(self, key: str) -> AgentSpec:
        """Convert to frozen domain AgentSpec."""
        return AgentSpec(
            key=key,
            display_name=self.display_name,
            role=self.role,
            system_prompt=self.system_prompt,
            lifecycle=self.lifecycle,
            color=self.color,
        )


class InteractionEdgeModel(BaseModel):
    """Validated interaction edge."""

    from_agent: str
    to_agent: str
    channel: str = Field(default="public", pattern=r"^(public|whisper|observe)$")

    def to_domain(self) -> InteractionEdge:
        """Convert to frozen domain InteractionEdge."""
        return InteractionEdge(
            from_agent=self.from_agent,
            to_agent=self.to_agent,
            channel=self.channel,
        )


class PhaseSpecModel(BaseModel):
    """Validated phase definition."""

    name: str = Field(..., pattern=r"^[a-z][a-z0-9_]{0,29}$")
    label: str = Field(..., min_length=1, max_length=80)
    agents: list[str] = Field(..., min_length=0, max_length=30)
    topology: str = Field(default="round_robin")
    debate_format: str = Field(default="free_form")
    decision_mode: str = Field(default="none")
    initial_prompt: str = Field(default="", max_length=50000)
    initial_prompt_template: str = Field(default="", max_length=50000)
    per_agent_prompt_templates: dict[str, str] = Field(default_factory=dict)
    max_turns: int = Field(default=0, ge=0, le=1000)
    auto_rounds: int = Field(default=3, ge=1, le=50)
    interactions: list[InteractionEdgeModel] | None = None

    def to_domain(self) -> PhaseSpec:
        """Convert to frozen domain PhaseSpec."""
        edges = None
        if self.interactions is not None:
            edges = tuple(e.to_domain() for e in self.interactions)
        return PhaseSpec(
            name=self.name,
            label=self.label,
            agents=tuple(self.agents),
            topology=self.topology,
            debate_format=self.debate_format,
            decision_mode=self.decision_mode,
            initial_prompt=self.initial_prompt,
            initial_prompt_template=self.initial_prompt_template,
            per_agent_prompt_templates=dict(self.per_agent_prompt_templates),
            max_turns=self.max_turns,
            auto_rounds=self.auto_rounds,
            interactions=edges,
        )


class OutcomeRuleModel(BaseModel):
    """Validated outcome rule."""

    type: str = Field(..., pattern=r"^(parse_choice|vote_tally|summary)$")
    pattern: str = Field(default="")
    choices: list[str] | None = None
    display: str = Field(default="table", pattern=r"^(table|matrix|text)$")

    def to_domain(self) -> OutcomeRule:
        """Convert to frozen domain OutcomeRule."""
        return OutcomeRule(
            type=self.type,
            pattern=self.pattern,
            choices=tuple(self.choices) if self.choices else None,
            display=self.display,
        )


class PayoffMatrixModel(BaseModel):
    """Validated payoff matrix."""

    labels: list[str] = Field(..., min_length=2)
    matrix: list[list[str]] = Field(..., min_length=2)

    def to_domain(self) -> PayoffMatrix:
        """Convert to frozen domain PayoffMatrix."""
        return PayoffMatrix(
            labels=tuple(self.labels),
            matrix=tuple(tuple(row) for row in self.matrix),
        )


class UIHintsModel(BaseModel):
    """Validated UI hints."""

    layout: str = Field(default="auto", pattern=r"^(auto|columns|tabs|single)$")
    column_count: int = Field(default=2, ge=1, le=4)
    show_memory_panel: bool = Field(default=False)
    show_interaction_graph: bool = Field(default=False)
    show_run_all: bool = Field(default=False)
    max_visible_messages: int = Field(default=50, ge=1, le=500)
    phase_controls: str = Field(default="per_phase", pattern=r"^(per_phase|global)$")

    def to_domain(self) -> UIHints:
        """Convert to frozen domain UIHints."""
        return UIHints(
            layout=self.layout,
            column_count=self.column_count,
            show_memory_panel=self.show_memory_panel,
            show_interaction_graph=self.show_interaction_graph,
            show_run_all=self.show_run_all,
            max_visible_messages=self.max_visible_messages,
            phase_controls=self.phase_controls,
        )


class ScenarioSpecModel(BaseModel):
    """Validated scenario specification. Top-level model for YAML parsing."""

    id: str = Field(..., pattern=r"^[a-z][a-z0-9_-]{0,49}$")
    title: str = Field(..., min_length=1, max_length=100)
    description: str = Field(default="", max_length=500)
    agents: dict[str, AgentSpecModel] = Field(default_factory=dict)
    phases: list[PhaseSpecModel] = Field(..., min_length=1, max_length=20)
    ui: UIHintsModel = Field(default_factory=UIHintsModel)
    outcome: OutcomeRuleModel | None = None
    payoff: PayoffMatrixModel | None = None

    @model_validator(mode="after")
    def validate_phase_agent_refs(self) -> Self:
        """Every agent key referenced in phases must exist in agents dict."""
        agent_keys = set(self.agents.keys())
        for phase in self.phases:
            for agent_ref in phase.agents:
                if agent_ref not in agent_keys:
                    msg = (
                        f"Phase '{phase.name}' references agent '{agent_ref}' "
                        f"not found in agents dict. Available: {sorted(agent_keys)}"
                    )
                    raise ValueError(msg)
        return self

    def to_domain(self) -> ScenarioSpec:
        """Convert validated Pydantic model to frozen domain dataclass."""
        agents = {key: model.to_domain(key) for key, model in self.agents.items()}
        phases = tuple(p.to_domain() for p in self.phases)
        return ScenarioSpec(
            id=self.id,
            title=self.title,
            description=self.description,
            agents=agents,
            phases=phases,
            ui=self.ui.to_domain(),
            outcome=self.outcome.to_domain() if self.outcome else None,
            payoff=self.payoff.to_domain() if self.payoff else None,
        )
