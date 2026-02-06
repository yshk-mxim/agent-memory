"""YAML scenario loader.

Loads scenario YAML files, validates via Pydantic, and returns
frozen domain dataclasses.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from semantic.adapters.config.scenario_models import ScenarioSpecModel
from semantic.domain.scenario import ScenarioSpec


def load_scenario(path: Path) -> ScenarioSpec:
    """Load and validate a YAML scenario file.

    Args:
        path: Path to the YAML file.

    Returns:
        Validated ScenarioSpec domain object.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        yaml.YAMLError: If the file contains invalid YAML.
        pydantic.ValidationError: If the content fails schema validation.
    """
    raw = yaml.safe_load(path.read_text())
    model = ScenarioSpecModel.model_validate(raw)
    return model.to_domain()


def discover_scenarios(directory: Path) -> dict[str, Path]:
    """Find all *.yaml scenario files in a directory.

    Returns:
        Dict mapping scenario ID (from filename stem) to file path.
    """
    scenarios: dict[str, Path] = {}
    if not directory.is_dir():
        return scenarios
    for path in sorted(directory.glob("*.yaml")):
        scenarios[path.stem] = path
    return scenarios
