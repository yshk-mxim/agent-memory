"""Prisoner's Dilemma Demo - YAML-driven game theory scenario.

Two suspects, one warden, classic prisoner's dilemma with optional yard session.
"""

from __future__ import annotations

import logging
from pathlib import Path

import streamlit as st
from demo.lib.scenario_renderer import ScenarioRenderer

from semantic.adapters.config.scenario_loader import load_scenario

logger = logging.getLogger(__name__)

BASE_URL = "http://localhost:8000"
SCENARIO_PATH = Path(__file__).resolve().parent.parent / "scenarios" / "prisoners_dilemma.yaml"


def main() -> None:
    """Prisoner's Dilemma scenario demo page."""
    st.set_page_config(
        page_title="Prisoner's Dilemma",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    try:
        spec = load_scenario(SCENARIO_PATH)
    except Exception:
        logger.exception("Failed to load PD scenario from %s", SCENARIO_PATH)
        st.error(f"Failed to load scenario: {SCENARIO_PATH}")
        return

    renderer = ScenarioRenderer(spec, BASE_URL)
    renderer.render()


if __name__ == "__main__":
    main()
