"""Gossip Demo - YAML-driven three-phase scenario.

Three friends, two private conversations, one awkward reunion.
Cross-session context injected via template resolver.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import logging

import streamlit as st
from demo.lib.scenario_renderer import ScenarioRenderer

from agent_memory.adapters.config.scenario_loader import load_scenario

logger = logging.getLogger(__name__)

BASE_URL = "http://localhost:8000"
SCENARIO_PATH = Path(__file__).resolve().parent.parent / "scenarios" / "gossip.yaml"


def main() -> None:
    """Gossip Network scenario demo page."""
    st.set_page_config(
        page_title="Gossip Network",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    try:
        spec = load_scenario(SCENARIO_PATH)
    except Exception:
        logger.exception("Failed to load gossip scenario from %s", SCENARIO_PATH)
        st.error(f"Failed to load scenario: {SCENARIO_PATH}")
        return

    renderer = ScenarioRenderer(spec, BASE_URL)
    renderer.render()


if __name__ == "__main__":
    main()
