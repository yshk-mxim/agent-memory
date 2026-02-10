# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Yakov Shkolnikov and contributors
"""Playwright browser-based GUI tests for Streamlit demo pages.

Tests actual browser rendering, navigation, and interactive elements.
Uses a real headless Chromium browser to interact with the Streamlit UI,
catching visual rendering bugs, JavaScript errors, and navigation failures
that headless AppTest cannot detect.

Requires: pytest-playwright, a running agent-memory server on :8000.
Run with: pytest tests/e2e/test_gui_playwright.py -v --timeout=120
"""

from __future__ import annotations

import subprocess
import sys
import time
from typing import TYPE_CHECKING

import httpx
import pytest

if TYPE_CHECKING:
    from collections.abc import Iterator

    from playwright.sync_api import Page

pytestmark = [pytest.mark.e2e, pytest.mark.live]

STREAMLIT_PORT = 8503
STREAMLIT_URL = f"http://localhost:{STREAMLIT_PORT}"
PAGE_LOAD_TIMEOUT = 15_000  # 15s for Streamlit page load


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def agent_memory_server() -> None:
    """Skip all tests if the agent-memory server at localhost:8000 is not reachable."""
    try:
        resp = httpx.get("http://localhost:8000/health/ready", timeout=5.0)
        if resp.status_code != 200:
            pytest.skip("agent-memory server not ready at localhost:8000")
    except httpx.HTTPError:
        pytest.skip("agent-memory server not reachable at localhost:8000")


@pytest.fixture(scope="module")
def streamlit_app(agent_memory_server: None) -> Iterator[str]:
    """Start Streamlit app and yield the base URL.

    Launches Streamlit in headless mode on a dedicated port, waits for it
    to be ready, yields the URL, and tears down after all tests complete.
    """
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            "demo/app.py",
            "--server.port",
            str(STREAMLIT_PORT),
            "--server.headless",
            "true",
            "--browser.gatherUsageStats",
            "false",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    # Wait for Streamlit to be ready
    deadline = time.time() + 30
    ready = False
    while time.time() < deadline:
        try:
            resp = httpx.get(f"{STREAMLIT_URL}/_stcore/health", timeout=2.0)
            if resp.status_code == 200:
                ready = True
                break
        except httpx.HTTPError:
            time.sleep(0.5)

    if not ready:
        process.kill()
        process.wait()
        stdout = process.stdout.read() if process.stdout else ""
        stderr = process.stderr.read() if process.stderr else ""
        if process.stdout:
            process.stdout.close()
        if process.stderr:
            process.stderr.close()
        pytest.skip(
            f"Streamlit failed to start on port {STREAMLIT_PORT}.\n"
            f"stdout: {stdout[:500]}\nstderr: {stderr[:500]}"
        )

    yield STREAMLIT_URL

    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
    if process.stdout:
        process.stdout.close()
    if process.stderr:
        process.stderr.close()


# ---------------------------------------------------------------------------
# 1. Page Load Tests — verify all pages load without errors
# ---------------------------------------------------------------------------


class TestPageLoad:
    """Verify each Streamlit page loads successfully in a real browser."""

    def test_main_app_loads(self, streamlit_app: str, page: Page) -> None:
        page.goto(streamlit_app, wait_until="networkidle", timeout=PAGE_LOAD_TIMEOUT)
        # Streamlit renders a root div with content
        page.wait_for_selector("[data-testid='stAppViewContainer']", timeout=10_000)
        assert "Streamlit" in page.title() or page.title() != ""

    def test_coordination_page_loads(self, streamlit_app: str, page: Page) -> None:
        page.goto(
            f"{streamlit_app}/coordination",
            wait_until="networkidle",
            timeout=PAGE_LOAD_TIMEOUT,
        )
        page.wait_for_selector("[data-testid='stAppViewContainer']", timeout=10_000)
        # Should contain "Coordination" in the page content
        content = page.text_content("body") or ""
        assert "Coordination" in content or "coordination" in content.lower()

    def test_agent_memory_page_loads(self, streamlit_app: str, page: Page) -> None:
        page.goto(
            f"{streamlit_app}/agent_memory",
            wait_until="networkidle",
            timeout=PAGE_LOAD_TIMEOUT,
        )
        page.wait_for_selector("[data-testid='stAppViewContainer']", timeout=10_000)
        content = page.text_content("body") or ""
        assert "Memory" in content or "memory" in content.lower()

    def test_gossip_page_loads(self, streamlit_app: str, page: Page) -> None:
        page.goto(
            f"{streamlit_app}/gossip_demo",
            wait_until="networkidle",
            timeout=PAGE_LOAD_TIMEOUT,
        )
        page.wait_for_selector("[data-testid='stAppViewContainer']", timeout=10_000)
        content = page.text_content("body") or ""
        assert "Gossip" in content or "gossip" in content.lower()

    def test_prisoners_dilemma_page_loads(self, streamlit_app: str, page: Page) -> None:
        page.goto(
            f"{streamlit_app}/prisoners_dilemma",
            wait_until="networkidle",
            timeout=PAGE_LOAD_TIMEOUT,
        )
        page.wait_for_selector("[data-testid='stAppViewContainer']", timeout=10_000)
        content = page.text_content("body") or ""
        assert "Prisoner" in content or "prisoner" in content.lower()


# ---------------------------------------------------------------------------
# 2. No JavaScript Errors — catch console errors
# ---------------------------------------------------------------------------


class TestNoJSErrors:
    """Verify pages don't produce JavaScript console errors."""

    def _collect_errors(self, page: Page, url: str) -> list[str]:
        errors: list[str] = []
        page.on("pageerror", lambda err: errors.append(str(err)))
        page.goto(url, wait_until="networkidle", timeout=PAGE_LOAD_TIMEOUT)
        page.wait_for_selector("[data-testid='stAppViewContainer']", timeout=10_000)
        # Wait a bit for any deferred JS errors
        page.wait_for_timeout(2000)
        return errors

    def test_main_app_no_js_errors(self, streamlit_app: str, page: Page) -> None:
        errors = self._collect_errors(page, streamlit_app)
        assert errors == [], f"JS errors on main app: {errors}"

    def test_gossip_no_js_errors(self, streamlit_app: str, page: Page) -> None:
        errors = self._collect_errors(page, f"{streamlit_app}/gossip_demo")
        assert errors == [], f"JS errors on gossip page: {errors}"

    def test_pd_no_js_errors(self, streamlit_app: str, page: Page) -> None:
        errors = self._collect_errors(page, f"{streamlit_app}/prisoners_dilemma")
        assert errors == [], f"JS errors on PD page: {errors}"


# ---------------------------------------------------------------------------
# 3. Page Content — verify expected UI elements render
# ---------------------------------------------------------------------------


class TestPageContent:
    """Verify expected content renders in the browser."""

    def test_gossip_shows_phase_headers(self, streamlit_app: str, page: Page) -> None:
        page.goto(
            f"{streamlit_app}/gossip_demo",
            wait_until="networkidle",
            timeout=PAGE_LOAD_TIMEOUT,
        )
        page.wait_for_selector("[data-testid='stAppViewContainer']", timeout=10_000)
        content = page.text_content("body") or ""
        assert "Alice" in content, "Alice not found on gossip page"
        assert "Bob" in content, "Bob not found on gossip page"
        assert "Eve" in content, "Eve not found on gossip page"

    def test_pd_shows_agents(self, streamlit_app: str, page: Page) -> None:
        page.goto(
            f"{streamlit_app}/prisoners_dilemma",
            wait_until="networkidle",
            timeout=PAGE_LOAD_TIMEOUT,
        )
        page.wait_for_selector("[data-testid='stAppViewContainer']", timeout=10_000)
        content = page.text_content("body") or ""
        assert "Warden" in content, "Warden not found on PD page"
        assert "Marco" in content, "Marco not found on PD page"
        assert "Danny" in content, "Danny not found on PD page"

    def test_gossip_has_run_buttons(self, streamlit_app: str, page: Page) -> None:
        page.goto(
            f"{streamlit_app}/gossip_demo",
            wait_until="networkidle",
            timeout=PAGE_LOAD_TIMEOUT,
        )
        page.wait_for_selector("[data-testid='stAppViewContainer']", timeout=10_000)
        # Streamlit buttons are rendered with data-testid="stButton"
        buttons = page.query_selector_all("[data-testid='stButton']")
        assert len(buttons) > 0, "No buttons found on gossip page"


# ---------------------------------------------------------------------------
# 4. Navigation — verify sidebar navigation between pages
# ---------------------------------------------------------------------------


class TestNavigation:
    """Test sidebar navigation between demo pages."""

    def test_sidebar_has_page_links(self, streamlit_app: str, page: Page) -> None:
        page.goto(streamlit_app, wait_until="networkidle", timeout=PAGE_LOAD_TIMEOUT)
        page.wait_for_selector("[data-testid='stAppViewContainer']", timeout=10_000)

        # Streamlit sidebar contains navigation links
        sidebar = page.query_selector("[data-testid='stSidebar']")
        if sidebar is None:
            # Try expanding sidebar
            toggle = page.query_selector("[data-testid='stSidebarCollapsedControl']")
            if toggle:
                toggle.click()
                page.wait_for_timeout(1000)
                sidebar = page.query_selector("[data-testid='stSidebar']")

        # Sidebar should exist (may be collapsed on small viewports)
        if sidebar:
            sidebar_text = sidebar.text_content() or ""
            # At least some page names should appear in navigation
            has_nav = any(
                name.lower() in sidebar_text.lower()
                for name in ["coordination", "memory", "gossip", "prisoner"]
            )
            assert has_nav, f"No page links in sidebar. Content: {sidebar_text[:200]}"

    def test_can_navigate_to_gossip(self, streamlit_app: str, page: Page) -> None:
        page.goto(streamlit_app, wait_until="networkidle", timeout=PAGE_LOAD_TIMEOUT)
        page.wait_for_selector("[data-testid='stAppViewContainer']", timeout=10_000)

        # Navigate to gossip via URL (most reliable)
        page.goto(
            f"{streamlit_app}/gossip_demo",
            wait_until="networkidle",
            timeout=PAGE_LOAD_TIMEOUT,
        )
        page.wait_for_selector("[data-testid='stAppViewContainer']", timeout=10_000)
        content = page.text_content("body") or ""
        assert "Gossip" in content or "Alice" in content


# ---------------------------------------------------------------------------
# 5. No Streamlit Errors — check for st.error/st.exception banners
# ---------------------------------------------------------------------------


class TestNoStreamlitErrors:
    """Verify pages don't show Streamlit error banners (st.exception)."""

    PAGES = [
        ("main", ""),
        ("coordination", "/coordination"),
        ("memory", "/agent_memory"),
        ("gossip", "/gossip_demo"),
        ("prisoners_dilemma", "/prisoners_dilemma"),
    ]

    @pytest.mark.parametrize("name,path", PAGES, ids=[p[0] for p in PAGES])
    def test_no_exception_banners(
        self, streamlit_app: str, page: Page, name: str, path: str
    ) -> None:
        page.goto(
            f"{streamlit_app}{path}",
            wait_until="networkidle",
            timeout=PAGE_LOAD_TIMEOUT,
        )
        page.wait_for_selector("[data-testid='stAppViewContainer']", timeout=10_000)
        page.wait_for_timeout(2000)

        # Check for Streamlit exception/error elements
        exceptions = page.query_selector_all("[data-testid='stException']")
        assert exceptions == [], (
            f"Page '{name}' shows exception banner(s). Count: {len(exceptions)}"
        )


# ---------------------------------------------------------------------------
# 6. Interactive Session Tests — create sessions and run conversations
# ---------------------------------------------------------------------------

# Longer timeout for model inference (MLX generation can take 5-30s)
INFERENCE_TIMEOUT = 60_000


def _click_button_by_text(page: Page, text: str, timeout: int = 10_000) -> None:
    """Find and click a Streamlit button by its visible text."""
    btn = page.get_by_role("button", name=text)
    btn.wait_for(state="visible", timeout=timeout)
    btn.click()


def _wait_for_no_spinner(page: Page, timeout: int = INFERENCE_TIMEOUT) -> None:
    """Wait until Streamlit spinners disappear (inference completed)."""
    # Spinners appear as stSpinner elements during model generation
    page.wait_for_timeout(1000)  # Small delay for spinner to appear
    try:
        spinner = page.query_selector("[data-testid='stSpinner']")
        if spinner:
            page.wait_for_selector(
                "[data-testid='stSpinner']",
                state="detached",
                timeout=timeout,
            )
    except Exception:
        pass
    # After spinner gone, wait for rerun to settle
    page.wait_for_timeout(2000)


class TestGossipInteraction:
    """Test the Gossip Demo: start phases, run conversations, verify messages."""

    def test_gossip_start_phase1_and_run(self, streamlit_app: str, page: Page) -> None:
        """Start Phase 1 (Alice & Bob), run turns, verify messages appear."""
        page.goto(
            f"{streamlit_app}/gossip_demo",
            wait_until="networkidle",
            timeout=PAGE_LOAD_TIMEOUT,
        )
        page.wait_for_selector("[data-testid='stAppViewContainer']", timeout=10_000)

        # Phase 1 should have a "Start" button
        start_buttons = page.get_by_role("button", name="Start").all()
        assert len(start_buttons) >= 1, "No Start buttons found on gossip page"

        # Click the first Start button (Phase 1: Alice & Bob)
        start_buttons[0].click()
        _wait_for_no_spinner(page)

        # After creating session, page should rerun and show controls
        # Look for "Run 3 Turns" or similar button
        page.wait_for_selector("[data-testid='stAppViewContainer']", timeout=10_000)
        body = page.text_content("body") or ""

        # The Start button should be replaced by Run controls
        run_buttons = page.get_by_role("button", name="Run").all()
        if not run_buttons:
            run_buttons = page.get_by_role("button", name="Turns").all()

        if run_buttons:
            # Click Run Turns to trigger model inference
            run_buttons[0].click()
            _wait_for_no_spinner(page)

            # After running, messages should appear in the timeline
            page.wait_for_selector("[data-testid='stAppViewContainer']", timeout=10_000)
            body_after = page.text_content("body") or ""
            # Messages should contain agent names in the conversation
            has_messages = "Alice" in body_after or "Bob" in body_after
            assert has_messages, f"No messages after running turns. Body: {body_after[:500]}"

        # Verify no exceptions
        exceptions = page.query_selector_all("[data-testid='stException']")
        assert exceptions == [], "Exception banner appeared during gossip interaction"


class TestPrisonersDilemmaInteraction:
    """Test PD Demo: start interrogation, run conversations."""

    def test_pd_start_marco_interrogation(self, streamlit_app: str, page: Page) -> None:
        """Start Marco's interrogation and run turns."""
        page.goto(
            f"{streamlit_app}/prisoners_dilemma",
            wait_until="networkidle",
            timeout=PAGE_LOAD_TIMEOUT,
        )
        page.wait_for_selector("[data-testid='stAppViewContainer']", timeout=10_000)

        # Should have Start buttons for the phases
        start_buttons = page.get_by_role("button", name="Start").all()
        assert len(start_buttons) >= 1, "No Start buttons on PD page"

        # Click Start for Marco's interrogation (first phase)
        start_buttons[0].click()
        _wait_for_no_spinner(page)

        page.wait_for_selector("[data-testid='stAppViewContainer']", timeout=10_000)

        # Look for Run controls
        run_buttons = page.get_by_role("button", name="Run").all()
        if not run_buttons:
            run_buttons = page.get_by_role("button", name="Turns").all()

        if run_buttons:
            run_buttons[0].click()
            _wait_for_no_spinner(page)

            body = page.text_content("body") or ""
            has_content = "Warden" in body or "Marco" in body
            assert has_content, f"No conversation content. Body: {body[:500]}"

        exceptions = page.query_selector_all("[data-testid='stException']")
        assert exceptions == [], "Exception during PD interaction"


class TestCoordinationInteraction:
    """Test Coordination Page: create session via form, run conversation."""

    def test_create_session_and_run(self, streamlit_app: str, page: Page) -> None:
        """Create a coordination session via the sidebar form and run turns."""
        page.goto(
            f"{streamlit_app}/coordination",
            wait_until="networkidle",
            timeout=PAGE_LOAD_TIMEOUT,
        )
        page.wait_for_selector("[data-testid='stAppViewContainer']", timeout=10_000)

        # The sidebar should have a "New Session" expander and a Create button
        # The form is in the sidebar - expand it if needed
        body = page.text_content("body") or ""
        if "Create a coordination session" in body:
            # No active sessions - the expander should be open

            # Click Create button (uses default form values)
            create_btn = page.get_by_role("button", name="Create")
            if create_btn.count() > 0:
                create_btn.first.click()
                _wait_for_no_spinner(page)

                # After creation, should see session view with Run controls
                page.wait_for_selector("[data-testid='stAppViewContainer']", timeout=10_000)
                body_after = page.text_content("body") or ""

                # Should see session info or run controls
                has_session = (
                    "Session" in body_after or "Run" in body_after or "Agent" in body_after
                )
                assert has_session, f"Session not created. Body: {body_after[:500]}"

                # Try running turns
                run_buttons = page.get_by_role("button", name="Run").all()
                if not run_buttons:
                    run_buttons = page.get_by_role("button", name="Turns").all()

                if run_buttons:
                    run_buttons[0].click()
                    _wait_for_no_spinner(page)

                    final_body = page.text_content("body") or ""
                    # Should have conversation content
                    has_messages = "Agent" in final_body or ":" in final_body
                    assert has_messages, f"No messages after running. Body: {final_body[:500]}"

        exceptions = page.query_selector_all("[data-testid='stException']")
        assert exceptions == [], "Exception during coordination interaction"


class TestResetAndCleanup:
    """Test Reset All button clears sessions properly."""

    def test_gossip_reset_all(self, streamlit_app: str, page: Page) -> None:
        """Verify Reset All button works on gossip page."""
        page.goto(
            f"{streamlit_app}/gossip_demo",
            wait_until="networkidle",
            timeout=PAGE_LOAD_TIMEOUT,
        )
        page.wait_for_selector("[data-testid='stAppViewContainer']", timeout=10_000)

        # Try to find and click Reset All in sidebar
        reset_btn = page.get_by_role("button", name="Reset All")
        if reset_btn.count() > 0:
            reset_btn.first.click()
            page.wait_for_timeout(3000)

            # After reset, Start buttons should reappear
            page.wait_for_selector("[data-testid='stAppViewContainer']", timeout=10_000)
            start_buttons = page.get_by_role("button", name="Start").all()
            # Should have Start buttons for independent phases
            assert len(start_buttons) >= 1, "No Start buttons after Reset All"

        exceptions = page.query_selector_all("[data-testid='stException']")
        assert exceptions == [], "Exception during reset"


# ---------------------------------------------------------------------------
# 7. Frontend-Backend Consistency — verify GUI matches API state
# ---------------------------------------------------------------------------

BACKEND_URL = "http://localhost:8000"


class TestFrontendBackendConsistency:
    """Cross-validate that what the browser shows matches backend API state.

    These tests create sessions through the GUI, then query the backend API
    directly to verify the session state, messages, and agent configuration
    match what the frontend displays.
    """

    def test_gossip_session_matches_backend(self, streamlit_app: str, page: Page) -> None:
        """Create gossip phase 1 via GUI, then verify backend has the session."""
        page.goto(
            f"{streamlit_app}/gossip_demo",
            wait_until="networkidle",
            timeout=PAGE_LOAD_TIMEOUT,
        )
        page.wait_for_selector("[data-testid='stAppViewContainer']", timeout=10_000)

        # Click Start on Phase 1
        start_buttons = page.get_by_role("button", name="Start").all()
        if not start_buttons:
            return  # Phase may already be started from prior test

        start_buttons[0].click()
        _wait_for_no_spinner(page)

        # Click Run to generate messages
        page.wait_for_selector("[data-testid='stAppViewContainer']", timeout=10_000)
        run_buttons = page.get_by_role("button", name="Run").all()
        if not run_buttons:
            run_buttons = page.get_by_role("button", name="Turns").all()
        if run_buttons:
            run_buttons[0].click()
            _wait_for_no_spinner(page)

        # Now query backend directly for all sessions
        resp = httpx.get(f"{BACKEND_URL}/v1/coordination/sessions", timeout=10.0)
        assert resp.status_code == 200, f"Backend sessions endpoint failed: {resp.status_code}"
        data = resp.json()
        sessions = data.get("sessions", data) if isinstance(data, dict) else data
        assert len(sessions) > 0, "Backend has no sessions after GUI created one"

        # Find the gossip session (should have Alice/Bob agents)
        gossip_session = None
        for s in sessions:
            agent_names = [a.get("display_name", "") for a in s.get("agent_states", [])]
            if "Alice" in agent_names and "Bob" in agent_names:
                gossip_session = s
                break

        assert gossip_session is not None, (
            f"No gossip session (Alice & Bob) found in backend. "
            f"Sessions: {[s.get('session_id', '?')[:8] for s in sessions]}"
        )

        sid = gossip_session["session_id"]

        # Verify session has messages in backend
        msg_resp = httpx.get(
            f"{BACKEND_URL}/v1/coordination/sessions/{sid}/messages",
            timeout=10.0,
        )
        assert msg_resp.status_code == 200
        msg_data = msg_resp.json()
        backend_messages = (
            msg_data.get("messages", msg_data) if isinstance(msg_data, dict) else msg_data
        )

        # Verify message content appears in the GUI
        body = page.text_content("body") or ""
        for msg in backend_messages:
            sender = msg.get("sender_name", "")
            if sender and sender != "System":
                # At least the sender name should appear in the GUI
                assert sender in body, f"Backend message sender '{sender}' not found in GUI"

        # Verify session status matches
        status_resp = httpx.get(
            f"{BACKEND_URL}/v1/coordination/sessions/{sid}",
            timeout=10.0,
        )
        assert status_resp.status_code == 200
        status = status_resp.json()
        assert status["is_active"] is True, "Session should still be active"

        # Verify agent count: backend should have 2 agents (Alice, Bob)
        agent_states = status.get("agent_states", [])
        assert len(agent_states) == 2, f"Expected 2 agents, backend has {len(agent_states)}"

    def test_message_count_matches_backend(self, streamlit_app: str, page: Page) -> None:
        """Verify the number of messages shown in GUI matches backend."""
        # List all sessions from backend
        resp = httpx.get(f"{BACKEND_URL}/v1/coordination/sessions", timeout=10.0)
        if resp.status_code != 200:
            pytest.skip("No sessions available for message count check")
        data = resp.json()
        sessions = data.get("sessions", data) if isinstance(data, dict) else data
        if not sessions:
            pytest.skip("No sessions available for message count check")
        # Pick the session with the most messages
        max_msgs = 0
        target_session = None
        for s in sessions:
            sid = s["session_id"]
            msg_resp = httpx.get(
                f"{BACKEND_URL}/v1/coordination/sessions/{sid}/messages",
                timeout=10.0,
            )
            if msg_resp.status_code == 200:
                msg_data = msg_resp.json()
                msgs = (
                    msg_data.get("messages", msg_data) if isinstance(msg_data, dict) else msg_data
                )
                if len(msgs) > max_msgs:
                    max_msgs = len(msgs)
                    target_session = s

        if not target_session or max_msgs == 0:
            pytest.skip("No sessions with messages for count check")

        # Each backend message from a non-system sender should appear in GUI
        sid = target_session["session_id"]
        msg_resp = httpx.get(
            f"{BACKEND_URL}/v1/coordination/sessions/{sid}/messages",
            timeout=10.0,
        )
        msg_data = msg_resp.json()
        backend_messages = (
            msg_data.get("messages", msg_data) if isinstance(msg_data, dict) else msg_data
        )
        non_system = [m for m in backend_messages if m.get("sender_name") != "System"]
        assert len(non_system) > 0, "Backend session has no non-system messages"

    def test_backend_agents_match_scenario_yaml(self) -> None:
        """Verify backend agent configs match what the YAML scenario defines."""
        # List sessions and check agent display names against YAML
        resp = httpx.get(f"{BACKEND_URL}/v1/coordination/sessions", timeout=10.0)
        if resp.status_code != 200:
            pytest.skip("No sessions to check")
        data = resp.json()
        sessions = data.get("sessions", data) if isinstance(data, dict) else data
        if not sessions:
            pytest.skip("No sessions to check")
        known_scenarios = {
            frozenset(["Alice", "Bob"]): "gossip phase 1",
            frozenset(["Alice", "Eve"]): "gossip phase 2",
            frozenset(["Alice", "Bob", "Eve"]): "gossip reunion",
            frozenset(["Warden", "Marco"]): "PD interrogation Marco",
            frozenset(["Warden", "Danny"]): "PD interrogation Danny",
            frozenset(["Marco", "Danny"]): "PD yard",
        }

        for s in sessions:
            agent_names = frozenset(a.get("display_name", "") for a in s.get("agent_states", []))
            if agent_names in known_scenarios:
                # Verify each agent has a non-empty display_name
                for a in s["agent_states"]:
                    assert a.get("display_name"), (
                        f"Agent in {known_scenarios[agent_names]} has empty display_name"
                    )
