"""GUI tests for Streamlit demo pages — headless rendering validation.

Tests all demo pages can load, render, and respond to basic interactions
using Streamlit's AppTest framework. Requires a running server on port 8000.

Run with: pytest tests/e2e/test_gui_pages.py -v --timeout=120
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.e2e, pytest.mark.live]

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)


# ---------------------------------------------------------------------------
# 0. Streamlit Import Path Simulation — catch ModuleNotFoundError in real runtime
# ---------------------------------------------------------------------------


class TestStreamlitImportPath:
    """Verify page files import correctly WITHOUT project root on sys.path.

    Streamlit does NOT put the project root on sys.path when loading page
    files. Only the editable-install path (src/) is available. Each page
    must set up its own sys.path to import from demo.lib.

    These tests run each page file in a subprocess with a restricted
    sys.path that mimics Streamlit's actual runtime environment, catching
    ModuleNotFoundError that pytest's privileged sys.path would hide.
    """

    PAGE_FILES = [
        "demo/app.py",
        "demo/pages/1_coordination.py",
        "demo/pages/2_agent_memory.py",
        "demo/pages/3_gossip_demo.py",
        "demo/pages/4_prisoners_dilemma.py",
    ]

    @pytest.mark.parametrize("page_path", PAGE_FILES)
    def test_page_imports_without_project_root(self, page_path: str) -> None:
        """Each page must resolve its imports via its own sys.path setup.

        Runs the page file as a subprocess with PYTHONPATH stripped of the
        project root. Python adds the page's own directory to sys.path[0]
        (like Streamlit), and __file__ is set automatically. The page's
        own sys.path fix must resolve demo.lib imports; if it doesn't,
        the subprocess will fail with ModuleNotFoundError.
        """
        abs_page = str(Path(PROJECT_ROOT) / page_path)

        # Build PYTHONPATH excluding the project root.
        # Keep only the editable-install path (src/) and system paths.
        existing = os.environ.get("PYTHONPATH", "")
        filtered = [
            p
            for p in existing.split(os.pathsep)
            if p and os.path.realpath(p) != os.path.realpath(PROJECT_ROOT)
        ]
        env = {**os.environ, "PYTHONPATH": os.pathsep.join(filtered)}

        # Run the page file from a neutral CWD (not the project root)
        result = subprocess.run(
            [sys.executable, abs_page],
            capture_output=True,
            text=True,
            timeout=30,
            cwd="/tmp",
            env=env,
        )

        if result.returncode != 0:
            stderr = result.stderr
            # ModuleNotFoundError or ImportError = the bug class we catch
            if "ModuleNotFoundError" in stderr or "ImportError" in stderr:
                pytest.fail(f"{page_path} has import errors in Streamlit-like runtime:\n{stderr}")

    @pytest.mark.parametrize("page_path", PAGE_FILES)
    def test_page_has_sys_path_setup(self, page_path: str) -> None:
        """Each page must have sys.path setup before demo.lib imports."""
        abs_page = Path(PROJECT_ROOT) / page_path
        source = abs_page.read_text()

        # Verify the sys.path setup pattern exists
        assert "sys.path" in source, (
            f"{page_path} is missing sys.path setup for Streamlit compatibility"
        )
        assert "_PROJECT_ROOT" in source, f"{page_path} is missing _PROJECT_ROOT definition"

        # Verify sys.path setup comes BEFORE demo.lib imports
        lines = source.splitlines()
        path_setup_line = -1
        first_demo_import_line = -1
        for i, line in enumerate(lines):
            if "sys.path.insert" in line and path_setup_line == -1:
                path_setup_line = i
            if line.strip().startswith(("from demo.", "import demo.")):
                if first_demo_import_line == -1:
                    first_demo_import_line = i

        if first_demo_import_line >= 0:
            assert path_setup_line >= 0, (
                f"{page_path} imports from demo.* but has no sys.path.insert"
            )
            assert path_setup_line < first_demo_import_line, (
                f"{page_path}: sys.path.insert (line {path_setup_line + 1}) "
                f"must come before demo import (line {first_demo_import_line + 1})"
            )


# ---------------------------------------------------------------------------
# 1. Import Smoke Tests — verify all pages and components import cleanly
# ---------------------------------------------------------------------------


class TestPageImports:
    """Verify every demo page and shared component imports without error."""

    def test_import_api_client(self) -> None:
        from demo.lib import api_client

        assert hasattr(api_client, "create_session")
        assert hasattr(api_client, "get_session_status")
        assert hasattr(api_client, "get_session_messages")
        assert hasattr(api_client, "execute_turn")
        assert hasattr(api_client, "execute_round")
        assert hasattr(api_client, "execute_turns")
        assert hasattr(api_client, "stream_round")
        assert hasattr(api_client, "delete_session")
        assert hasattr(api_client, "list_sessions")
        assert hasattr(api_client, "get_agent_stats")
        assert hasattr(api_client, "get_agent_list")
        assert hasattr(api_client, "delete_agent")

    def test_import_template_resolver(self) -> None:
        from demo.lib.template_resolver import (
            extract_phase_refs,
            has_template_refs,
            resolve_template,
        )

        assert callable(resolve_template)
        assert callable(has_template_refs)
        assert callable(extract_phase_refs)

    def test_import_message_timeline(self) -> None:
        from demo.lib.message_timeline import (
            DEFAULT_COLORS,
            build_agent_colors,
            render_message,
            render_timeline,
        )

        assert callable(render_message)
        assert callable(render_timeline)
        assert callable(build_agent_colors)
        assert len(DEFAULT_COLORS) >= 10

    def test_import_control_bar(self) -> None:
        from demo.lib.control_bar import render_round_controls, render_session_actions

        assert callable(render_round_controls)
        assert callable(render_session_actions)

    def test_import_scenario_renderer(self) -> None:
        from demo.lib.scenario_renderer import ScenarioRenderer

        assert callable(ScenarioRenderer)

    def test_import_scenario_loader(self) -> None:
        from agent_memory.adapters.config.scenario_loader import load_scenario

        assert callable(load_scenario)

    def test_import_scenario_domain(self) -> None:
        from agent_memory.domain.scenario import (
            AgentSpec,
            PhaseSpec,
            ScenarioSpec,
            UIHints,
        )

        assert callable(AgentSpec)
        assert callable(PhaseSpec)
        assert callable(ScenarioSpec)
        assert callable(UIHints)


# ---------------------------------------------------------------------------
# 2. Scenario YAML Loading — verify all YAML files load correctly
# ---------------------------------------------------------------------------


class TestScenarioYAMLLoading:
    """Test that all scenario YAML files parse and validate correctly."""

    def test_load_gossip_scenario(self) -> None:
        from agent_memory.adapters.config.scenario_loader import load_scenario

        spec = load_scenario(Path("demo/scenarios/gossip.yaml"))
        assert spec.id == "gossip"
        assert spec.title == "The Gossip Network"
        assert len(spec.agents) == 3
        assert "alice" in spec.agents
        assert "bob" in spec.agents
        assert "eve" in spec.agents
        assert len(spec.phases) == 3
        assert spec.phases[0].name == "alice_bob"
        assert spec.phases[1].name == "alice_eve"
        assert spec.phases[2].name == "reunion"
        assert spec.ui.layout == "columns"
        assert spec.ui.show_memory_panel is True

    def test_load_prisoners_dilemma_scenario(self) -> None:
        from agent_memory.adapters.config.scenario_loader import load_scenario

        spec = load_scenario(Path("demo/scenarios/prisoners_dilemma.yaml"))
        assert spec.id == "prisoners-dilemma"
        assert len(spec.agents) == 3
        assert "warden" in spec.agents
        assert "marco" in spec.agents
        assert "danny" in spec.agents
        assert len(spec.phases) == 3
        assert spec.phases[0].name == "interrogation_marco"
        assert spec.phases[1].name == "interrogation_danny"
        assert spec.phases[2].name == "the_yard"
        assert spec.outcome is not None
        assert spec.payoff is not None

    def test_load_coordination_scenario(self) -> None:
        from agent_memory.adapters.config.scenario_loader import load_scenario

        spec = load_scenario(Path("demo/scenarios/coordination.yaml"))
        assert spec.id == "coordination"
        assert spec.ui.layout == "single"

    def test_gossip_agents_have_system_prompts(self) -> None:
        from agent_memory.adapters.config.scenario_loader import load_scenario

        spec = load_scenario(Path("demo/scenarios/gossip.yaml"))
        for agent in spec.agents.values():
            assert len(agent.system_prompt) > 0, f"{agent.key} has empty system_prompt"

    def test_gossip_phase_agents_reference_valid_keys(self) -> None:
        from agent_memory.adapters.config.scenario_loader import load_scenario

        spec = load_scenario(Path("demo/scenarios/gossip.yaml"))
        for phase in spec.phases:
            for agent_key in phase.agents:
                assert agent_key in spec.agents, (
                    f"Phase {phase.name} references unknown agent {agent_key}"
                )

    def test_pd_phase_agents_reference_valid_keys(self) -> None:
        from agent_memory.adapters.config.scenario_loader import load_scenario

        spec = load_scenario(Path("demo/scenarios/prisoners_dilemma.yaml"))
        for phase in spec.phases:
            for agent_key in phase.agents:
                assert agent_key in spec.agents, (
                    f"Phase {phase.name} references unknown agent {agent_key}"
                )

    def test_gossip_reunion_has_template(self) -> None:
        from demo.lib.template_resolver import extract_phase_refs, has_template_refs

        from agent_memory.adapters.config.scenario_loader import load_scenario

        spec = load_scenario(Path("demo/scenarios/gossip.yaml"))
        reunion = spec.phases[2]
        assert has_template_refs(reunion.initial_prompt_template)
        refs = extract_phase_refs(reunion.initial_prompt_template)
        assert "alice_bob" in refs
        assert "alice_eve" in refs

    def test_pd_interrogation_danny_has_template(self) -> None:
        from demo.lib.template_resolver import has_template_refs

        from agent_memory.adapters.config.scenario_loader import load_scenario

        spec = load_scenario(Path("demo/scenarios/prisoners_dilemma.yaml"))
        danny_phase = spec.phases[1]
        assert has_template_refs(danny_phase.initial_prompt_template)


# ---------------------------------------------------------------------------
# 3. Shared Component Logic Tests (no Streamlit dependency)
# ---------------------------------------------------------------------------


class TestTemplateResolver:
    """Test template resolver logic without Streamlit."""

    def test_resolve_empty_template(self) -> None:
        from demo.lib.template_resolver import resolve_template

        result = resolve_template("No templates here.", {})
        assert result == "No templates here."

    def test_resolve_single_reference(self) -> None:
        from demo.lib.template_resolver import resolve_template

        messages = {
            "phase1": [
                {"sender_name": "Alice", "content": "Hello Bob!"},
                {"sender_name": "Bob", "content": "Hey Alice!"},
            ],
        }
        template = "Prior conversation:\n${phase1.messages[alice]}"
        result = resolve_template(template, messages)
        assert "Alice: Hello Bob!" in result
        assert "Bob: Hey Alice!" in result
        assert "${" not in result

    def test_resolve_multiple_references(self) -> None:
        from demo.lib.template_resolver import resolve_template

        messages = {
            "p1": [{"sender_name": "A", "content": "msg1"}],
            "p2": [{"sender_name": "B", "content": "msg2"}],
        }
        template = "From p1: ${p1.messages[a]}\nFrom p2: ${p2.messages[b]}"
        result = resolve_template(template, messages)
        assert "A: msg1" in result
        assert "B: msg2" in result
        assert "${" not in result

    def test_resolve_missing_phase(self) -> None:
        from demo.lib.template_resolver import resolve_template

        result = resolve_template("${missing.messages[x]}", {})
        assert result == "(no messages yet)"

    def test_has_template_refs(self) -> None:
        from demo.lib.template_resolver import has_template_refs

        assert has_template_refs("${phase.messages[agent]}")
        assert not has_template_refs("No templates")
        assert not has_template_refs("")

    def test_extract_phase_refs(self) -> None:
        from demo.lib.template_resolver import extract_phase_refs

        refs = extract_phase_refs("${a.messages[x]} and ${b.messages[y]}")
        assert refs == {"a", "b"}


class TestBuildAgentColors:
    """Test color assignment logic."""

    def test_colors_from_agent_specs(self) -> None:
        from demo.lib.message_timeline import build_agent_colors

        from agent_memory.adapters.config.scenario_loader import load_scenario

        spec = load_scenario(Path("demo/scenarios/gossip.yaml"))
        colors = build_agent_colors(spec.agents)
        assert "Alice" in colors
        assert "Bob" in colors
        assert "Eve" in colors
        # Each should have a hex color
        for name, color in colors.items():
            assert color.startswith("#"), f"{name} color {color} is not hex"

    def test_colors_from_dict(self) -> None:
        from demo.lib.message_timeline import build_agent_colors

        agents = {
            "a": {"display_name": "Agent A", "color": "#FF0000"},
            "b": {"display_name": "Agent B", "color": ""},
        }
        colors = build_agent_colors(agents)
        assert colors["Agent A"] == "#FF0000"
        assert colors["Agent B"].startswith("#")  # Default color


# ---------------------------------------------------------------------------
# 4. Streamlit AppTest — Headless Page Rendering
# ---------------------------------------------------------------------------


class TestStreamlitPageRendering:
    """Test Streamlit pages render without crashing using AppTest.

    These tests use Streamlit's headless testing framework to verify
    that pages load and render their initial state correctly.
    """

    def test_gossip_page_renders(self, server_ready: None) -> None:
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file("demo/pages/3_gossip_demo.py", default_timeout=30)
        at.run()
        assert not at.exception, f"Page crashed: {at.exception}"
        # Should have sidebar title
        assert any("Gossip" in str(el.value) for el in at.title)

    def test_prisoners_dilemma_page_renders(self, server_ready: None) -> None:
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file("demo/pages/4_prisoners_dilemma.py", default_timeout=30)
        at.run()
        assert not at.exception, f"Page crashed: {at.exception}"
        assert any("Prisoner" in str(el.value) for el in at.title)

    def test_agent_memory_page_renders(self, server_ready: None) -> None:
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file("demo/pages/2_agent_memory.py", default_timeout=30)
        at.run()
        assert not at.exception, f"Page crashed: {at.exception}"
        assert any("Memory" in str(el.value) for el in at.title)

    def test_coordination_page_renders(self, server_ready: None) -> None:
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file("demo/pages/1_coordination.py", default_timeout=30)
        at.run()
        assert not at.exception, f"Page crashed: {at.exception}"
        assert any("Coordination" in str(el.value) for el in at.title)

    def test_main_app_page_renders(self, server_ready: None) -> None:
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file("demo/app.py", default_timeout=30)
        at.run()
        assert not at.exception, f"Page crashed: {at.exception}"


# ---------------------------------------------------------------------------
# 5. Page Content Verification
# ---------------------------------------------------------------------------


class TestPageContent:
    """Verify pages render expected content elements."""

    def test_gossip_shows_three_phases(self, server_ready: None) -> None:
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file("demo/pages/3_gossip_demo.py", default_timeout=30)
        at.run()
        assert not at.exception

        # Should show phase labels as subheaders
        subheaders = [str(el.value) for el in at.subheader]
        phase_labels = ["Alice & Bob", "Alice & Eve", "Reunion"]
        for label in phase_labels:
            assert any(label in sh for sh in subheaders), (
                f"Phase '{label}' not found in subheaders: {subheaders}"
            )

    def test_gossip_shows_agents_in_sidebar(self, server_ready: None) -> None:
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file("demo/pages/3_gossip_demo.py", default_timeout=30)
        at.run()
        assert not at.exception

        # Sidebar markdown should mention agent names
        all_markdown = " ".join(str(el.value) for el in at.markdown)
        assert "Alice" in all_markdown
        assert "Bob" in all_markdown
        assert "Eve" in all_markdown

    def test_pd_shows_three_phases(self, server_ready: None) -> None:
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file("demo/pages/4_prisoners_dilemma.py", default_timeout=30)
        at.run()
        assert not at.exception

        subheaders = [str(el.value) for el in at.subheader]
        # At minimum should have interrogation rooms
        assert any("Marco" in sh for sh in subheaders), (
            f"Marco interrogation not found in: {subheaders}"
        )
        assert any("Danny" in sh for sh in subheaders), (
            f"Danny interrogation not found in: {subheaders}"
        )

    def test_pd_shows_agents_in_sidebar(self, server_ready: None) -> None:
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file("demo/pages/4_prisoners_dilemma.py", default_timeout=30)
        at.run()
        assert not at.exception

        all_markdown = " ".join(str(el.value) for el in at.markdown)
        assert "Warden" in all_markdown
        assert "Marco" in all_markdown
        assert "Danny" in all_markdown

    def test_coordination_shows_creation_form_or_session(
        self,
        server_ready: None,
    ) -> None:
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file("demo/pages/1_coordination.py", default_timeout=30)
        at.run()
        assert not at.exception

        # Page should show either creation prompt (no sessions) or session view
        titles = [str(el.value) for el in at.title]
        assert any("Coordination" in t for t in titles)

    def test_memory_page_shows_inspector_content(self, server_ready: None) -> None:
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file("demo/pages/2_agent_memory.py", default_timeout=30)
        at.run()
        assert not at.exception

        # Title should reference memory
        titles = [str(el.value) for el in at.title]
        assert any("Memory" in t for t in titles)


# ---------------------------------------------------------------------------
# 6. Graceful Degradation — Pages handle server-down without crashing
# ---------------------------------------------------------------------------


class TestGracefulDegradation:
    """Verify pages show user-friendly errors instead of crashing when server is down."""

    def test_gossip_page_with_unreachable_server(self) -> None:
        """Test that gossip page doesn't crash when server is unreachable."""
        from streamlit.testing.v1 import AppTest

        # Patch the BASE_URL to point to a dead server
        at = AppTest.from_file("demo/pages/3_gossip_demo.py", default_timeout=15)
        # Override the module-level BASE_URL before running
        at.run()
        # The page should not have unhandled exceptions
        # (it may show st.error messages, which is correct behavior)
        assert not at.exception, f"Page crashed with server down: {at.exception}"

    def test_pd_page_with_unreachable_server(self) -> None:
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file("demo/pages/4_prisoners_dilemma.py", default_timeout=15)
        at.run()
        assert not at.exception, f"Page crashed with server down: {at.exception}"

    def test_memory_page_with_unreachable_server(self) -> None:
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file("demo/pages/2_agent_memory.py", default_timeout=15)
        at.run()
        assert not at.exception, f"Page crashed with server down: {at.exception}"

    def test_coordination_page_with_unreachable_server(self) -> None:
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file("demo/pages/1_coordination.py", default_timeout=15)
        at.run()
        assert not at.exception, f"Page crashed with server down: {at.exception}"
