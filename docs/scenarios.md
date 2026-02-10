# Scenario Authoring Guide

YAML-driven configuration for multi-agent demo scenarios.

## Scenario Configuration System

Scenarios are YAML files in `demo/scenarios/`. Each file declares agents, phases, prompts, and UI layout. The YAML is validated by Pydantic models (`adapters/config/scenario_models.py`) and converted to frozen domain dataclasses (`domain/scenario.py`). The `ScenarioRenderer` drives the Streamlit UI from the resulting `ScenarioSpec`.

Loader path: `YAML file -> ScenarioSpecModel (Pydantic) -> ScenarioSpec (domain)`

## YAML Schema Reference

### Top-Level

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `id` | str | yes | -- | Pattern: `^[a-z][a-z0-9_-]{0,49}$` |
| `title` | str | yes | -- | 1-100 chars |
| `description` | str | no | `""` | Max 500 chars |
| `agents` | dict | yes | -- | Map of agent key -> AgentSpec |
| `phases` | list | yes | -- | 1-20 PhaseSpec objects, ordered |
| `ui` | UIHints | no | defaults | UI rendering hints |
| `outcome` | OutcomeRule | no | null | How to parse/display outcomes |
| `payoff` | PayoffMatrix | no | null | Game-theory payoff display |

### AgentSpec

Keyed by a short identifier (e.g., `alice`, `warden`) in the `agents` dict.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `display_name` | str | required | Shown in UI, 1-50 chars |
| `role` | str | `"participant"` | `participant`, `observer`, `narrator`, `judge`, or `moderator` |
| `system_prompt` | str | `""` | Injected into agent context, max 4000 chars |
| `lifecycle` | str | `"ephemeral"` | `ephemeral` (cleared between phases) or `permanent` (persists) |
| `color` | str | `""` | Hex color for timeline (e.g., `"#FF6B6B"`) |

### PhaseSpec

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | str | required | Pattern: `^[a-z][a-z0-9_]{0,29}$` |
| `label` | str | required | Display label, 1-80 chars |
| `agents` | list[str] | required | Participating agent keys, 1-30 |
| `topology` | str | `"round_robin"` | Turn-taking strategy |
| `debate_format` | str | `"free_form"` | Conversation structure |
| `decision_mode` | str | `"none"` | How decisions are reached |
| `initial_prompt` | str | `""` | Static opening prompt, max 50000 chars |
| `initial_prompt_template` | str | `""` | Template with cross-phase refs (see below) |
| `max_turns` | int | `0` | Turn limit (0 = unlimited), max 1000 |
| `auto_rounds` | int | `3` | Turns per "Run Round" click, 1-50 |
| `interactions` | list | null | InteractionEdge objects |

**InteractionEdge**: `from_agent` (str), `to_agent` (str), `channel` (str, default `"public"` -- one of `public`, `whisper`, `observe`).

### UIHints

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `layout` | str | `"auto"` | `auto`, `columns`, `tabs`, or `single` |
| `column_count` | int | `2` | 1-4, used when layout is `columns` |
| `show_memory_panel` | bool | `false` | Per-agent cache stats below phases |
| `show_interaction_graph` | bool | `false` | Agent interaction diagram |
| `max_visible_messages` | int | `50` | Timeline message limit, 1-500 |
| `phase_controls` | str | `"per_phase"` | `per_phase` or `global` |

### OutcomeRule

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `type` | str | required | `parse_choice`, `vote_tally`, or `summary` |
| `pattern` | str | `""` | Regex to extract choices from messages |
| `choices` | list[str] | null | Valid choice values |
| `display` | str | `"table"` | `table`, `matrix`, or `text` |

### PayoffMatrix

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `labels` | list[str] | required | Row/column labels, min 2 |
| `matrix` | list[list[str]] | required | Payoff cells, min 2x2 |

## Template Syntax

Use `initial_prompt_template` to inject message history from prior phases:

```
${phase_name.messages[agent_key]}
```

Resolves to a formatted transcript from `phase_name`. The renderer blocks phase creation until all referenced phases have messages.

**Example** from the gossip scenario reunion phase:

```yaml
initial_prompt_template: |
  Alice remembers (from private chat with Bob):
  ${alice_bob.messages[alice]}

  Bob remembers (from private chat with Alice):
  ${alice_bob.messages[bob]}

  Now you're all in the same room. Be natural.
```

Regex used: `\$\{(\w+)\.messages\[(\w+)\]\}`. The helper `extract_phase_refs()` returns dependent phase names so the UI enforces ordering.

## Agent Roles

| Role | Behavior |
|------|----------|
| `participant` | Active conversant, takes turns, generates responses |
| `observer` | Receives messages but does not speak |
| `moderator` | Guides discussion, enforces rules, may inject prompts |
| `judge` | Evaluates outcomes, renders verdicts |
| `narrator` | Scene-setting context, not an in-character participant |

## Creating a New Scenario

**Step 1** -- Copy an existing YAML:

```bash
cp demo/scenarios/gossip.yaml demo/scenarios/my_scenario.yaml
```

**Step 2** -- Define agents and phases. All agent keys referenced in `phases[].agents` must exist in the top-level `agents` dict (enforced by Pydantic validator).

**Step 3** -- Set UI hints (`layout`, `column_count`, `show_memory_panel`, etc.).

**Step 4** -- Optionally add `outcome` and `payoff` for game-theory scenarios. See `demo/scenarios/prisoners_dilemma.yaml` for an example.

**Step 5** -- Create the page loader at `demo/pages/<N>_my_scenario.py`:

```python
"""My Scenario - brief description."""

from pathlib import Path

import streamlit as st
from demo.lib.scenario_renderer import ScenarioRenderer
from agent_memory.adapters.config.scenario_loader import load_scenario

BASE_URL = "http://localhost:8000"
SCENARIO_PATH = Path(__file__).resolve().parent.parent / "scenarios" / "my_scenario.yaml"


def main() -> None:
    st.set_page_config(page_title="My Scenario", layout="wide")
    spec = load_scenario(SCENARIO_PATH)
    ScenarioRenderer(spec, BASE_URL).render()


if __name__ == "__main__":
    main()
```

## Running Tests

```bash
# Domain dataclasses + Pydantic validation
pytest tests/unit/test_scenario_domain.py tests/unit/adapters/test_scenario_models.py -v

# Property-based tests (Hypothesis)
pytest tests/unit/test_scenario_properties.py -v --hypothesis-show-statistics

# End-to-end scenario loading
pytest tests/integration/test_scenario_e2e.py -v

# All scenario tests
pytest tests/unit/test_scenario_domain.py \
       tests/unit/adapters/test_scenario_models.py \
       tests/unit/test_scenario_properties.py \
       tests/integration/test_scenario_e2e.py -v
```

## See Also

- [Testing Guide](testing.md)
- [Developer Guide](developer-guide.md)
- [Architecture Overview](architecture.md)
