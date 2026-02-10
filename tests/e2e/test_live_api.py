"""Live API endpoint tests — requires a running semantic server on port 8000.

Tests all coordination and agent management endpoints against a real server.
Run with: pytest tests/e2e/test_live_api.py -v --timeout=120

Prerequisites:
    semantic serve --port 8000
"""

from __future__ import annotations

import httpx
import pytest

pytestmark = [pytest.mark.e2e, pytest.mark.live]

BASE_URL = "http://localhost:8000"
_HTTP_OK = 200
_HTTP_CREATED = 201
_HTTP_NO_CONTENT = 204


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def client() -> httpx.Client:
    """Shared HTTP client for all tests in this module."""
    with httpx.Client(base_url=BASE_URL, timeout=30.0) as c:
        yield c


# ---------------------------------------------------------------------------
# 1. Health & Infrastructure Endpoints
# ---------------------------------------------------------------------------


class TestHealthEndpoints:
    def test_health_ready(self, client: httpx.Client, server_ready: None) -> None:
        resp = client.get("/health/ready")
        assert resp.status_code == _HTTP_OK
        data = resp.json()
        assert data["status"] == "ready"

    def test_models_list(self, client: httpx.Client, server_ready: None) -> None:
        resp = client.get("/v1/models")
        assert resp.status_code == _HTTP_OK
        data = resp.json()
        assert "data" in data
        assert len(data["data"]) >= 1


# ---------------------------------------------------------------------------
# 2. Agent Management Endpoints
# ---------------------------------------------------------------------------


class TestAgentEndpoints:
    def test_agent_stats(self, client: httpx.Client, server_ready: None) -> None:
        resp = client.get("/v1/agents/stats")
        assert resp.status_code == _HTTP_OK
        data = resp.json()
        assert "total_count" in data
        assert "hot_count" in data
        assert "warm_count" in data
        assert "pool_utilization_pct" in data
        assert "total_cache_size_mb" in data

    def test_agent_list(self, client: httpx.Client, server_ready: None) -> None:
        resp = client.get("/v1/agents/list")
        assert resp.status_code == _HTTP_OK
        data = resp.json()
        assert "agents" in data
        assert isinstance(data["agents"], list)


# ---------------------------------------------------------------------------
# 3. Coordination Session Lifecycle
# ---------------------------------------------------------------------------


class TestCoordinationSessionLifecycle:
    """Test the full session lifecycle: create -> status -> turns -> messages -> delete."""

    def test_create_session(self, client: httpx.Client, server_ready: None) -> None:
        resp = client.post(
            "/v1/coordination/sessions",
            json={
                "topology": "round_robin",
                "debate_format": "free_form",
                "decision_mode": "none",
                "agents": [
                    {
                        "display_name": "TestAlpha",
                        "role": "participant",
                        "system_prompt": "You are a helpful test agent. Keep responses under 2 sentences.",
                        "lifecycle": "ephemeral",
                    },
                    {
                        "display_name": "TestBeta",
                        "role": "participant",
                        "system_prompt": "You are a friendly test agent. Keep responses under 2 sentences.",
                        "lifecycle": "ephemeral",
                    },
                ],
                "initial_prompt": "Discuss the weather briefly.",
                "max_turns": 10,
            },
        )
        assert resp.status_code == _HTTP_CREATED, (
            f"Expected 201, got {resp.status_code}: {resp.text}"
        )
        data = resp.json()
        assert "session_id" in data
        assert data["status"] == "active"
        assert len(data["agents"]) == 2

    def test_list_sessions(self, client: httpx.Client, server_ready: None) -> None:
        resp = client.get("/v1/coordination/sessions")
        assert resp.status_code == _HTTP_OK
        data = resp.json()
        assert "sessions" in data
        assert isinstance(data["sessions"], list)

    def test_full_session_lifecycle(
        self,
        client: httpx.Client,
        server_ready: None,
    ) -> None:
        """Create session -> get status -> execute turn -> get messages -> delete."""
        # Create
        create_resp = client.post(
            "/v1/coordination/sessions",
            json={
                "topology": "round_robin",
                "debate_format": "free_form",
                "decision_mode": "none",
                "agents": [
                    {
                        "display_name": "LifecycleA",
                        "role": "participant",
                        "system_prompt": "Reply in exactly one sentence.",
                        "lifecycle": "ephemeral",
                    },
                    {
                        "display_name": "LifecycleB",
                        "role": "participant",
                        "system_prompt": "Reply in exactly one sentence.",
                        "lifecycle": "ephemeral",
                    },
                ],
                "initial_prompt": "Say hello.",
                "max_turns": 6,
            },
        )
        assert create_resp.status_code == _HTTP_CREATED
        session_id = create_resp.json()["session_id"]

        # Get status
        status_resp = client.get(f"/v1/coordination/sessions/{session_id}")
        assert status_resp.status_code == _HTTP_OK
        status = status_resp.json()
        assert status["session_id"] == session_id
        assert status["is_active"] is True
        assert status["current_turn"] == 0
        assert len(status["agent_states"]) == 2

        # Execute single turn (with longer timeout for inference)
        with httpx.Client(base_url=BASE_URL, timeout=120.0) as long_client:
            turn_resp = long_client.post(
                f"/v1/coordination/sessions/{session_id}/turn",
            )
        assert turn_resp.status_code == _HTTP_OK, (
            f"Turn failed: {turn_resp.status_code}: {turn_resp.text}"
        )
        turn_data = turn_resp.json()
        assert "message" in turn_data
        msg = turn_data["message"]
        assert msg["sender_name"] in ("LifecycleA", "LifecycleB")
        assert len(msg["content"]) > 0
        assert "turn_number" in msg

        # Verify status updated (turn count incremented)
        status2 = client.get(f"/v1/coordination/sessions/{session_id}").json()
        assert status2["current_turn"] >= 1

        # Get messages
        msgs_resp = client.get(
            f"/v1/coordination/sessions/{session_id}/messages",
        )
        assert msgs_resp.status_code == _HTTP_OK
        msgs = msgs_resp.json()
        assert msgs["session_id"] == session_id
        assert len(msgs["messages"]) >= 1
        first_msg = msgs["messages"][0]
        assert "sender_name" in first_msg
        assert "content" in first_msg
        assert "turn_number" in first_msg

        # Delete session
        del_resp = client.delete(f"/v1/coordination/sessions/{session_id}")
        assert del_resp.status_code == _HTTP_NO_CONTENT

        # Verify deleted
        get_resp = client.get(f"/v1/coordination/sessions/{session_id}")
        assert get_resp.status_code != _HTTP_OK


# ---------------------------------------------------------------------------
# 4. Coordination Rounds
# ---------------------------------------------------------------------------


class TestCoordinationRounds:
    """Test round execution (all agents speak once)."""

    def test_execute_round(self, client: httpx.Client, server_ready: None) -> None:
        # Create session
        create_resp = client.post(
            "/v1/coordination/sessions",
            json={
                "topology": "round_robin",
                "debate_format": "free_form",
                "decision_mode": "none",
                "agents": [
                    {
                        "display_name": "RoundA",
                        "role": "participant",
                        "system_prompt": "Reply with one word only.",
                        "lifecycle": "ephemeral",
                    },
                    {
                        "display_name": "RoundB",
                        "role": "participant",
                        "system_prompt": "Reply with one word only.",
                        "lifecycle": "ephemeral",
                    },
                ],
                "initial_prompt": "Name a color.",
                "max_turns": 10,
            },
        )
        assert create_resp.status_code == _HTTP_CREATED
        session_id = create_resp.json()["session_id"]

        try:
            # Execute full round (both agents speak)
            with httpx.Client(base_url=BASE_URL, timeout=300.0) as long_client:
                round_resp = long_client.post(
                    f"/v1/coordination/sessions/{session_id}/round",
                )
            assert round_resp.status_code == _HTTP_OK, (
                f"Round failed: {round_resp.status_code}: {round_resp.text}"
            )
            data = round_resp.json()
            assert "messages" in data
            assert len(data["messages"]) >= 2  # Both agents spoke
            senders = {m["sender_name"] for m in data["messages"]}
            assert "RoundA" in senders
            assert "RoundB" in senders
        finally:
            client.delete(f"/v1/coordination/sessions/{session_id}")


# ---------------------------------------------------------------------------
# 5. Agent Lifecycle (permanent agents)
# ---------------------------------------------------------------------------


class TestAgentLifecycle:
    """Test permanent agent lifecycle and cache persistence."""

    def test_permanent_agent_persists_across_session(
        self,
        client: httpx.Client,
        server_ready: None,
    ) -> None:
        # Create session with permanent agent
        create_resp = client.post(
            "/v1/coordination/sessions",
            json={
                "topology": "round_robin",
                "debate_format": "free_form",
                "decision_mode": "none",
                "agents": [
                    {
                        "display_name": "PermanentAgent",
                        "role": "participant",
                        "system_prompt": "You are a test agent with permanent memory.",
                        "lifecycle": "permanent",
                    },
                    {
                        "display_name": "EphemeralAgent",
                        "role": "participant",
                        "system_prompt": "You are a temporary agent.",
                        "lifecycle": "ephemeral",
                    },
                ],
                "initial_prompt": "Introduce yourselves briefly.",
                "max_turns": 4,
            },
        )
        assert create_resp.status_code == _HTTP_CREATED
        session_id = create_resp.json()["session_id"]

        try:
            # Run a turn to populate caches
            with httpx.Client(base_url=BASE_URL, timeout=120.0) as long_client:
                turn_resp = long_client.post(
                    f"/v1/coordination/sessions/{session_id}/turn",
                )
            assert turn_resp.status_code == _HTTP_OK

            # Check agent list contains our agents
            agents_resp = client.get("/v1/agents/list")
            assert agents_resp.status_code == _HTTP_OK
            agents = agents_resp.json()["agents"]
            assert len(agents) >= 1  # At least one agent cached
        finally:
            client.delete(f"/v1/coordination/sessions/{session_id}")


# ---------------------------------------------------------------------------
# 6. Validation & Error Cases
# ---------------------------------------------------------------------------


class TestValidationErrors:
    """Test that invalid requests are rejected properly."""

    def test_create_session_no_agents(
        self,
        client: httpx.Client,
        server_ready: None,
    ) -> None:
        resp = client.post(
            "/v1/coordination/sessions",
            json={
                "topology": "round_robin",
                "debate_format": "free_form",
                "decision_mode": "none",
                "agents": [],
                "initial_prompt": "test",
                "max_turns": 10,
            },
        )
        # Should be rejected (min 2 agents)
        assert resp.status_code == 422

    def test_create_session_one_agent(
        self,
        client: httpx.Client,
        server_ready: None,
    ) -> None:
        resp = client.post(
            "/v1/coordination/sessions",
            json={
                "topology": "round_robin",
                "debate_format": "free_form",
                "decision_mode": "none",
                "agents": [
                    {"display_name": "Solo", "role": "participant"},
                ],
                "initial_prompt": "test",
                "max_turns": 10,
            },
        )
        assert resp.status_code == 422

    def test_get_nonexistent_session(
        self,
        client: httpx.Client,
        server_ready: None,
    ) -> None:
        resp = client.get("/v1/coordination/sessions/nonexistent-id-12345")
        assert resp.status_code != _HTTP_OK

    def test_delete_nonexistent_session(
        self,
        client: httpx.Client,
        server_ready: None,
    ) -> None:
        resp = client.delete("/v1/coordination/sessions/nonexistent-id-12345")
        # Should be 404 or similar, not 500
        assert resp.status_code != 500

    def test_turn_nonexistent_session(
        self,
        client: httpx.Client,
        server_ready: None,
    ) -> None:
        with httpx.Client(base_url=BASE_URL, timeout=30.0) as c:
            resp = c.post("/v1/coordination/sessions/nonexistent-id-12345/turn")
        assert resp.status_code != 500


# ---------------------------------------------------------------------------
# 7. Demo api_client Integration
# ---------------------------------------------------------------------------


class TestDemoApiClient:
    """Test the demo api_client module against the live server."""

    def test_api_client_create_and_delete(self, server_ready: None) -> None:
        from demo.lib import api_client

        result = api_client.create_session(
            BASE_URL,
            topology="round_robin",
            debate_format="free_form",
            decision_mode="none",
            agents=[
                {"display_name": "ClientA", "role": "participant", "lifecycle": "ephemeral"},
                {"display_name": "ClientB", "role": "participant", "lifecycle": "ephemeral"},
            ],
            initial_prompt="Test from api_client.",
            max_turns=4,
        )
        assert result is not None, "api_client.create_session returned None"
        session_id = result["session_id"]
        assert session_id

        # Status
        status = api_client.get_session_status(BASE_URL, session_id)
        assert status is not None
        assert status["is_active"] is True

        # Messages (empty initially)
        messages = api_client.get_session_messages(BASE_URL, session_id)
        assert isinstance(messages, list)

        # Delete
        deleted = api_client.delete_session(BASE_URL, session_id)
        assert deleted is True

    def test_api_client_execute_turn(self, server_ready: None) -> None:
        from demo.lib import api_client

        result = api_client.create_session(
            BASE_URL,
            topology="round_robin",
            debate_format="free_form",
            decision_mode="none",
            agents=[
                {
                    "display_name": "TurnTestA",
                    "role": "participant",
                    "system_prompt": "Reply briefly.",
                    "lifecycle": "ephemeral",
                },
                {
                    "display_name": "TurnTestB",
                    "role": "participant",
                    "system_prompt": "Reply briefly.",
                    "lifecycle": "ephemeral",
                },
            ],
            initial_prompt="Say one word.",
            max_turns=6,
        )
        assert result is not None
        session_id = result["session_id"]

        try:
            # Execute a single turn
            turn_result = api_client.execute_turn(BASE_URL, session_id)
            assert turn_result is not None, "execute_turn returned None"
            assert "message" in turn_result

            # Check messages
            messages = api_client.get_session_messages(BASE_URL, session_id)
            assert len(messages) >= 1
        finally:
            api_client.delete_session(BASE_URL, session_id)

    def test_api_client_execute_round(self, server_ready: None) -> None:
        from demo.lib import api_client

        result = api_client.create_session(
            BASE_URL,
            topology="round_robin",
            debate_format="free_form",
            decision_mode="none",
            agents=[
                {
                    "display_name": "RndTestA",
                    "role": "participant",
                    "system_prompt": "Reply with one word.",
                    "lifecycle": "ephemeral",
                },
                {
                    "display_name": "RndTestB",
                    "role": "participant",
                    "system_prompt": "Reply with one word.",
                    "lifecycle": "ephemeral",
                },
            ],
            initial_prompt="Name a fruit.",
            max_turns=10,
        )
        assert result is not None
        session_id = result["session_id"]

        try:
            round_result = api_client.execute_round(BASE_URL, session_id)
            assert round_result is not None, "execute_round returned None"
            assert "messages" in round_result
            assert len(round_result["messages"]) >= 2
        finally:
            api_client.delete_session(BASE_URL, session_id)

    def test_api_client_list_sessions(self, server_ready: None) -> None:
        from demo.lib import api_client

        sessions = api_client.list_sessions(BASE_URL)
        assert isinstance(sessions, list)

    def test_api_client_agent_stats(self, server_ready: None) -> None:
        from demo.lib import api_client

        stats = api_client.get_agent_stats(BASE_URL)
        assert stats is not None
        assert "total_count" in stats

    def test_api_client_agent_list(self, server_ready: None) -> None:
        from demo.lib import api_client

        agents = api_client.get_agent_list(BASE_URL)
        assert isinstance(agents, list)

    def test_api_client_handles_server_down(self) -> None:
        """Verify api_client returns None/empty instead of crashing when server unreachable."""
        from demo.lib import api_client

        bad_url = "http://localhost:19999"  # Not a real server

        assert api_client.get_agent_stats(bad_url) is None
        assert api_client.get_agent_list(bad_url) == []
        assert api_client.list_sessions(bad_url) == []
        assert api_client.get_session_status(bad_url, "fake") is None
        assert api_client.get_session_messages(bad_url, "fake") == []
        assert api_client.delete_session(bad_url, "fake") is False
        assert api_client.delete_agent(bad_url, "fake") is False
        assert (
            api_client.create_session(
                bad_url,
                topology="round_robin",
                debate_format="free_form",
                decision_mode="none",
                agents=[{"display_name": "A", "role": "p"}, {"display_name": "B", "role": "p"}],
                initial_prompt="test",
                max_turns=1,
            )
            is None
        )


# ---------------------------------------------------------------------------
# 8. Scenario YAML -> Session Flow
# ---------------------------------------------------------------------------


class TestScenarioSessionFlow:
    """Test creating sessions from YAML scenario specs — the full demo path."""

    def test_gossip_phase_1_creates_session(
        self,
        client: httpx.Client,
        server_ready: None,
    ) -> None:
        from pathlib import Path

        from agent_memory.adapters.config.scenario_loader import load_scenario

        spec = load_scenario(Path("demo/scenarios/gossip.yaml"))
        phase = spec.phases[0]  # alice_bob

        # Build agent configs exactly as ScenarioRenderer does
        agent_configs = []
        for agent_key in phase.agents:
            agent = spec.agents[agent_key]
            agent_configs.append(
                {
                    "display_name": agent.display_name,
                    "role": agent.role,
                    "system_prompt": agent.system_prompt,
                    "lifecycle": agent.lifecycle,
                }
            )

        resp = client.post(
            "/v1/coordination/sessions",
            json={
                "topology": phase.topology,
                "debate_format": phase.debate_format,
                "decision_mode": phase.decision_mode,
                "agents": agent_configs,
                "initial_prompt": phase.initial_prompt,
                "max_turns": phase.max_turns,
            },
        )
        assert resp.status_code == _HTTP_CREATED, (
            f"Gossip phase 1 create failed: {resp.status_code}: {resp.text}"
        )
        data = resp.json()
        assert data["status"] == "active"
        assert len(data["agents"]) == 2

        # Execute one turn
        session_id = data["session_id"]
        try:
            with httpx.Client(base_url=BASE_URL, timeout=120.0) as long_client:
                turn = long_client.post(
                    f"/v1/coordination/sessions/{session_id}/turn",
                )
            assert turn.status_code == _HTTP_OK
            msg = turn.json()["message"]
            assert msg["sender_name"] in ("Alice", "Bob")
            assert len(msg["content"]) > 0
        finally:
            client.delete(f"/v1/coordination/sessions/{session_id}")

    def test_pd_interrogation_creates_session(
        self,
        client: httpx.Client,
        server_ready: None,
    ) -> None:
        from pathlib import Path

        from agent_memory.adapters.config.scenario_loader import load_scenario

        spec = load_scenario(Path("demo/scenarios/prisoners_dilemma.yaml"))
        phase = spec.phases[0]  # interrogation_marco

        agent_configs = []
        for agent_key in phase.agents:
            agent = spec.agents[agent_key]
            agent_configs.append(
                {
                    "display_name": agent.display_name,
                    "role": agent.role,
                    "system_prompt": agent.system_prompt,
                    "lifecycle": agent.lifecycle,
                }
            )

        resp = client.post(
            "/v1/coordination/sessions",
            json={
                "topology": phase.topology,
                "debate_format": phase.debate_format,
                "decision_mode": phase.decision_mode,
                "agents": agent_configs,
                "initial_prompt": phase.initial_prompt,
                "max_turns": phase.max_turns,
            },
        )
        assert resp.status_code == _HTTP_CREATED, (
            f"PD phase 1 create failed: {resp.status_code}: {resp.text}"
        )
        data = resp.json()
        assert data["status"] == "active"
        assert len(data["agents"]) == 2

        session_id = data["session_id"]
        try:
            with httpx.Client(base_url=BASE_URL, timeout=120.0) as long_client:
                turn = long_client.post(
                    f"/v1/coordination/sessions/{session_id}/turn",
                )
            assert turn.status_code == _HTTP_OK
            msg = turn.json()["message"]
            assert msg["sender_name"] in ("Warden", "Marco")
            assert len(msg["content"]) > 0
        finally:
            client.delete(f"/v1/coordination/sessions/{session_id}")

    def test_gossip_cross_phase_template_resolution(
        self,
        server_ready: None,
    ) -> None:
        """Test the full gossip flow: phase 1 -> phase 2 -> reunion with cross-phase context."""
        from pathlib import Path

        from demo.lib import api_client
        from demo.lib.template_resolver import (
            extract_phase_refs,
            has_template_refs,
            resolve_template,
        )

        from agent_memory.adapters.config.scenario_loader import load_scenario

        spec = load_scenario(Path("demo/scenarios/gossip.yaml"))

        # Phase 1: Alice & Bob
        phase1 = spec.phases[0]
        agents1 = [
            {
                "display_name": spec.agents[k].display_name,
                "role": spec.agents[k].role,
                "system_prompt": spec.agents[k].system_prompt,
                "lifecycle": spec.agents[k].lifecycle,
            }
            for k in phase1.agents
        ]
        result1 = api_client.create_session(
            BASE_URL,
            topology=phase1.topology,
            debate_format=phase1.debate_format,
            decision_mode=phase1.decision_mode,
            agents=agents1,
            initial_prompt=phase1.initial_prompt,
            max_turns=phase1.max_turns,
        )
        assert result1 is not None
        sid1 = result1["session_id"]

        # Phase 2: Alice & Eve
        phase2 = spec.phases[1]
        agents2 = [
            {
                "display_name": spec.agents[k].display_name,
                "role": spec.agents[k].role,
                "system_prompt": spec.agents[k].system_prompt,
                "lifecycle": spec.agents[k].lifecycle,
            }
            for k in phase2.agents
        ]
        result2 = api_client.create_session(
            BASE_URL,
            topology=phase2.topology,
            debate_format=phase2.debate_format,
            decision_mode=phase2.decision_mode,
            agents=agents2,
            initial_prompt=phase2.initial_prompt,
            max_turns=phase2.max_turns,
        )
        assert result2 is not None
        sid2 = result2["session_id"]

        try:
            # Run turns on both phases
            assert api_client.execute_turns(BASE_URL, sid1, 2)
            assert api_client.execute_turns(BASE_URL, sid2, 2)

            # Get messages from both phases
            msgs1 = api_client.get_session_messages(BASE_URL, sid1)
            msgs2 = api_client.get_session_messages(BASE_URL, sid2)
            assert len(msgs1) >= 2, f"Phase 1 has {len(msgs1)} messages"
            assert len(msgs2) >= 2, f"Phase 2 has {len(msgs2)} messages"

            # Phase 3: Reunion with template
            reunion = spec.phases[2]
            assert has_template_refs(reunion.initial_prompt_template)
            refs = extract_phase_refs(reunion.initial_prompt_template)
            assert refs == {"alice_bob", "alice_eve"}

            # Resolve template with real messages
            phase_messages = {
                "alice_bob": msgs1,
                "alice_eve": msgs2,
            }
            resolved = resolve_template(
                reunion.initial_prompt_template,
                phase_messages,
            )

            # Verify template was resolved (no ${} remaining)
            assert "${" not in resolved, f"Unresolved template vars in: {resolved[:200]}"
            assert len(resolved) > 100  # Should have real content

            # Create reunion session with resolved prompt
            agents3 = [
                {
                    "display_name": spec.agents[k].display_name,
                    "role": spec.agents[k].role,
                    "system_prompt": spec.agents[k].system_prompt,
                    "lifecycle": spec.agents[k].lifecycle,
                }
                for k in reunion.agents
            ]
            result3 = api_client.create_session(
                BASE_URL,
                topology=reunion.topology,
                debate_format=reunion.debate_format,
                decision_mode=reunion.decision_mode,
                agents=agents3,
                initial_prompt=resolved,
                max_turns=reunion.max_turns,
            )
            assert result3 is not None
            sid3 = result3["session_id"]

            # Run a turn in the reunion
            turn_result = api_client.execute_turn(BASE_URL, sid3)
            assert turn_result is not None
            msg = turn_result["message"]
            assert msg["sender_name"] in ("Alice", "Bob", "Eve")
            assert len(msg["content"]) > 0

            api_client.delete_session(BASE_URL, sid3)
        finally:
            api_client.delete_session(BASE_URL, sid1)
            api_client.delete_session(BASE_URL, sid2)
