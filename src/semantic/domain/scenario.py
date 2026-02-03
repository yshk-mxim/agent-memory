"""Scenario configuration domain model.

Pure Python dataclasses defining the structure of YAML-driven
multi-agent coordination scenarios. No external dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class InteractionEdge:
    """Directed communication edge between agents."""

    from_agent: str
    to_agent: str
    channel: str = "public"


@dataclass(frozen=True)
class AgentSpec:
    """Agent definition within a scenario."""

    key: str
    display_name: str
    role: str = "participant"
    system_prompt: str = ""
    lifecycle: str = "ephemeral"
    color: str = ""


@dataclass(frozen=True)
class PhaseSpec:
    """A phase (session) within a scenario."""

    name: str
    label: str
    agents: tuple[str, ...]
    topology: str = "round_robin"
    debate_format: str = "free_form"
    decision_mode: str = "none"
    initial_prompt: str = ""
    initial_prompt_template: str = ""
    per_agent_prompt_templates: dict[str, str] = field(default_factory=dict)
    max_turns: int = 0
    auto_rounds: int = 3
    interactions: tuple[InteractionEdge, ...] | None = None


@dataclass(frozen=True)
class OutcomeRule:
    """How to parse/display session outcomes."""

    type: str
    pattern: str = ""
    choices: tuple[str, ...] | None = None
    display: str = "table"


@dataclass(frozen=True)
class PayoffMatrix:
    """Game-theory payoff matrix for outcome display."""

    labels: tuple[str, ...]
    matrix: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class UIHints:
    """UI rendering hints for the scenario."""

    layout: str = "auto"
    column_count: int = 2
    show_memory_panel: bool = False
    show_interaction_graph: bool = False
    show_run_all: bool = False
    max_visible_messages: int = 50
    phase_controls: str = "per_phase"


@dataclass(frozen=True)
class ScenarioSpec:
    """Complete scenario specification."""

    id: str
    title: str
    description: str
    agents: dict[str, AgentSpec]
    phases: tuple[PhaseSpec, ...]
    ui: UIHints = field(default_factory=UIHints)
    outcome: OutcomeRule | None = None
    payoff: PayoffMatrix | None = None
