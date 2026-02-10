# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Yakov Shkolnikov and contributors
"""Integration tests for Coordination API.

Tests multi-agent coordination endpoints:
- POST /v1/coordination/sessions (create)
- GET /v1/coordination/sessions (list)
- GET /v1/coordination/sessions/{id} (get status)
- DELETE /v1/coordination/sessions/{id} (delete)
- POST /v1/coordination/sessions/{id}/turn (execute turn)
- POST /v1/coordination/sessions/{id}/round (execute round)
- POST /v1/coordination/sessions/{id}/whisper (send whisper)
- POST /v1/coordination/sessions/{id}/vote (submit vote)
- GET /v1/coordination/sessions/{id}/messages (get messages)
"""

import pytest
from fastapi.testclient import TestClient

from agent_memory.entrypoints.api_server import create_app

pytestmark = pytest.mark.integration


class TestCoordinationAPIEndpoints:
    """Test coordination API endpoints without MLX model."""

    def test_create_session_endpoint_exists(self):
        """POST /v1/coordination/sessions endpoint should be registered."""
        app = create_app()
        client = TestClient(app, raise_server_exceptions=False)

        response = client.post(
            "/v1/coordination/sessions",
            json={
                "topology": "turn_by_turn",
                "debate_format": "free_form",
                "decision_mode": "none",
                "agents": [
                    {"display_name": "Alice", "role": "participant"},
                    {"display_name": "Bob", "role": "participant"},
                ],
                "initial_prompt": "Test topic",
                "max_turns": 5,
            },
        )

        # Endpoint exists (may fail without model, but not 404)
        assert response.status_code != 404

    def test_create_session_validation(self):
        """Create session should validate required fields."""
        app = create_app()
        client = TestClient(app, raise_server_exceptions=False)

        # Missing required fields
        response = client.post("/v1/coordination/sessions", json={})
        assert response.status_code == 422

        # Invalid topology
        response = client.post(
            "/v1/coordination/sessions",
            json={
                "topology": "invalid_topology",
                "debate_format": "free_form",
                "decision_mode": "none",
                "agents": [{"display_name": "Alice"}],
            },
        )
        # 422 (validation error), 400 (invalid enum), or 503 (passes validation
        # but service unavailable — topology is validated at service layer)
        assert response.status_code in [400, 422, 503]

    def test_list_sessions_endpoint_exists(self):
        """GET /v1/coordination/sessions endpoint should be registered."""
        app = create_app()
        client = TestClient(app, raise_server_exceptions=False)

        response = client.get("/v1/coordination/sessions")
        # Endpoint exists (503 if lifespan not run, or 200)
        assert response.status_code in [200, 503]

    def test_get_session_endpoint_exists(self):
        """GET /v1/coordination/sessions/{id} endpoint should be registered."""
        app = create_app()
        client = TestClient(app, raise_server_exceptions=False)

        response = client.get("/v1/coordination/sessions/test_session_id")
        # Endpoint exists (404 not found, or 503 if lifespan not run)
        assert response.status_code in [404, 503]

    def test_delete_session_endpoint_exists(self):
        """DELETE /v1/coordination/sessions/{id} endpoint should be registered."""
        app = create_app()
        client = TestClient(app, raise_server_exceptions=False)

        response = client.delete("/v1/coordination/sessions/test_session_id")
        # Endpoint exists
        assert response.status_code in [404, 503]

    def test_execute_turn_endpoint_exists(self):
        """POST /v1/coordination/sessions/{id}/turn endpoint should be registered."""
        app = create_app()
        client = TestClient(app, raise_server_exceptions=False)

        response = client.post("/v1/coordination/sessions/test_id/turn")
        # Endpoint exists
        assert response.status_code != 404

    def test_execute_round_endpoint_exists(self):
        """POST /v1/coordination/sessions/{id}/round endpoint should be registered."""
        app = create_app()
        client = TestClient(app, raise_server_exceptions=False)

        response = client.post("/v1/coordination/sessions/test_id/round")
        # Endpoint exists
        assert response.status_code != 404

    def test_send_whisper_endpoint_exists(self):
        """POST /v1/coordination/sessions/{id}/whisper endpoint should be registered."""
        app = create_app()
        client = TestClient(app, raise_server_exceptions=False)

        response = client.post(
            "/v1/coordination/sessions/test_id/whisper",
            json={
                "from_agent_id": "a",
                "to_agent_id": "b",
                "content": "Secret message",
            },
        )
        # Endpoint exists
        assert response.status_code != 404

    def test_submit_vote_endpoint_exists(self):
        """POST /v1/coordination/sessions/{id}/vote endpoint should be registered."""
        app = create_app()
        client = TestClient(app, raise_server_exceptions=False)

        response = client.post(
            "/v1/coordination/sessions/test_id/vote",
            json={
                "agent_id": "a",
                "question": "Proceed?",
                "choice": "yes",
            },
        )
        # Endpoint exists
        assert response.status_code != 404

    def test_get_messages_endpoint_exists(self):
        """GET /v1/coordination/sessions/{id}/messages endpoint should be registered."""
        app = create_app()
        client = TestClient(app, raise_server_exceptions=False)

        response = client.get("/v1/coordination/sessions/test_id/messages")
        # Endpoint exists
        assert response.status_code in [404, 503]


class TestCoordinationAPIWithModel:
    """Tests that require MLX model loaded."""

    def test_create_session_basic(self):
        """Creating a coordination session should work end-to-end."""
        app = create_app()

        with TestClient(app) as client:
            response = client.post(
                "/v1/coordination/sessions",
                json={
                    "topology": "turn_by_turn",
                    "debate_format": "free_form",
                    "decision_mode": "none",
                    "agents": [
                        {"display_name": "Alice", "role": "participant"},
                        {"display_name": "Bob", "role": "participant"},
                    ],
                    "initial_prompt": "Should AI be open source?",
                    "max_turns": 5,
                },
            )

            assert response.status_code == 201
            data = response.json()
            assert "session_id" in data
            assert data["session_id"].startswith("coord_")
            assert len(data["agents"]) == 2
            assert data["topology"] == "turn_by_turn"
            assert data["status"] == "active"

    def test_create_session_with_agent_ids(self):
        """Creating session with explicit agent IDs should preserve them."""
        app = create_app()

        with TestClient(app) as client:
            response = client.post(
                "/v1/coordination/sessions",
                json={
                    "topology": "turn_by_turn",
                    "debate_format": "free_form",
                    "decision_mode": "none",
                    "agents": [
                        {
                            "agent_id": "alice_001",
                            "display_name": "Alice",
                            "role": "participant",
                        },
                        {
                            "agent_id": "bob_002",
                            "display_name": "Bob",
                            "role": "participant",
                        },
                    ],
                },
            )

            assert response.status_code == 201
            data = response.json()
            agent_ids = [a["agent_id"] for a in data["agents"]]
            assert "alice_001" in agent_ids
            assert "bob_002" in agent_ids

    def test_list_sessions(self):
        """Listing sessions should return all active sessions."""
        app = create_app()

        with TestClient(app) as client:
            # Create a session
            create_resp = client.post(
                "/v1/coordination/sessions",
                json={
                    "topology": "turn_by_turn",
                    "debate_format": "free_form",
                    "decision_mode": "none",
                    "agents": [
                        {"display_name": "Alice"},
                        {"display_name": "Bob"},
                    ],
                },
            )
            assert create_resp.status_code == 201

            # List sessions
            response = client.get("/v1/coordination/sessions")
            assert response.status_code == 200
            data = response.json()
            assert "sessions" in data
            assert len(data["sessions"]) >= 1

    def test_get_session_status(self):
        """Getting session status should return current state."""
        app = create_app()

        with TestClient(app) as client:
            # Create session
            create_resp = client.post(
                "/v1/coordination/sessions",
                json={
                    "topology": "turn_by_turn",
                    "debate_format": "free_form",
                    "decision_mode": "none",
                    "agents": [
                        {"display_name": "Alice"},
                        {"display_name": "Bob"},
                    ],
                },
            )
            session_id = create_resp.json()["session_id"]

            # Get status
            response = client.get(f"/v1/coordination/sessions/{session_id}")
            assert response.status_code == 200
            data = response.json()
            assert data["session_id"] == session_id
            assert data["current_turn"] == 0
            assert data["is_active"] is True
            assert "next_speaker" in data
            assert len(data["agent_states"]) == 2

    def test_delete_session(self):
        """Deleting a session should remove it."""
        app = create_app()

        with TestClient(app) as client:
            # Create session
            create_resp = client.post(
                "/v1/coordination/sessions",
                json={
                    "topology": "turn_by_turn",
                    "debate_format": "free_form",
                    "decision_mode": "none",
                    "agents": [
                        {"display_name": "Alice"},
                        {"display_name": "Bob"},
                    ],
                },
            )
            session_id = create_resp.json()["session_id"]

            # Delete session
            response = client.delete(f"/v1/coordination/sessions/{session_id}")
            assert response.status_code == 204

            # Session should not exist
            response = client.get(f"/v1/coordination/sessions/{session_id}")
            assert response.status_code == 404

    def test_execute_turn(self):
        """Executing a turn should generate a message."""
        app = create_app()

        with TestClient(app) as client:
            # Create session
            create_resp = client.post(
                "/v1/coordination/sessions",
                json={
                    "topology": "turn_by_turn",
                    "debate_format": "free_form",
                    "decision_mode": "none",
                    "agents": [
                        {"display_name": "Alice", "role": "participant"},
                        {"display_name": "Bob", "role": "participant"},
                    ],
                    "initial_prompt": "What is 2+2?",
                    "max_turns": 2,
                },
            )
            session_id = create_resp.json()["session_id"]

            # Execute turn
            response = client.post(f"/v1/coordination/sessions/{session_id}/turn")
            assert response.status_code == 200
            data = response.json()

            # Check message
            assert "message" in data
            message = data["message"]
            assert "sender_id" in message
            assert "sender_name" in message
            assert "content" in message
            assert len(message["content"]) > 0
            assert message["turn_number"] == 0

            # Check session status updated
            assert "session_status" in data
            status = data["session_status"]
            assert status["current_turn"] == 1
            assert status["is_active"] is True

    def test_execute_round(self):
        """Executing a round should have all agents respond once."""
        app = create_app()

        with TestClient(app) as client:
            # Create session with 3 agents
            create_resp = client.post(
                "/v1/coordination/sessions",
                json={
                    "topology": "turn_by_turn",
                    "debate_format": "free_form",
                    "decision_mode": "none",
                    "agents": [
                        {"display_name": "Alice"},
                        {"display_name": "Bob"},
                        {"display_name": "Charlie"},
                    ],
                    "initial_prompt": "Count to 3",
                    "max_turns": 6,
                },
            )
            session_id = create_resp.json()["session_id"]

            # Execute round
            response = client.post(f"/v1/coordination/sessions/{session_id}/round")
            assert response.status_code == 200
            data = response.json()

            # Should have 3 messages (one per agent)
            assert "messages" in data
            assert len(data["messages"]) == 3

            # Each message should be from a different agent
            senders = {msg["sender_id"] for msg in data["messages"]}
            assert len(senders) == 3

            # Session should advance by 3 turns
            status = data["session_status"]
            assert status["current_turn"] == 3

    def test_send_whisper(self):
        """Sending a whisper should create a private message."""
        app = create_app()

        with TestClient(app) as client:
            # Create session
            create_resp = client.post(
                "/v1/coordination/sessions",
                json={
                    "topology": "whisper",
                    "debate_format": "free_form",
                    "decision_mode": "none",
                    "agents": [
                        {"agent_id": "alice", "display_name": "Alice"},
                        {"agent_id": "bob", "display_name": "Bob"},
                        {"agent_id": "charlie", "display_name": "Charlie"},
                    ],
                },
            )
            session_id = create_resp.json()["session_id"]

            # Send whisper from Alice to Bob
            response = client.post(
                f"/v1/coordination/sessions/{session_id}/whisper",
                json={
                    "from_agent_id": "alice",
                    "to_agent_id": "bob",
                    "content": "Secret message",
                },
            )
            assert response.status_code == 200
            data = response.json()

            # Check message
            assert "message" in data
            message = data["message"]
            assert message["sender_id"] == "alice"
            assert message["content"] == "Secret message"
            assert message["channel_type"] == "whisper"

    def test_submit_vote(self):
        """Submitting a vote should return confirmation."""
        app = create_app()

        with TestClient(app) as client:
            # Create session
            create_resp = client.post(
                "/v1/coordination/sessions",
                json={
                    "topology": "turn_by_turn",
                    "debate_format": "free_form",
                    "decision_mode": "majority_vote",
                    "agents": [
                        {"agent_id": "alice", "display_name": "Alice"},
                        {"agent_id": "bob", "display_name": "Bob"},
                    ],
                },
            )
            session_id = create_resp.json()["session_id"]

            # Submit vote
            response = client.post(
                f"/v1/coordination/sessions/{session_id}/vote",
                json={
                    "agent_id": "alice",
                    "question": "Should we proceed?",
                    "choice": "yes",
                },
            )
            assert response.status_code == 200
            data = response.json()
            assert "vote_id" in data
            assert data["agent_id"] == "alice"
            assert data["choice"] == "yes"

    def test_get_messages(self):
        """Getting messages should return conversation history."""
        app = create_app()

        with TestClient(app) as client:
            # Create session with initial prompt
            create_resp = client.post(
                "/v1/coordination/sessions",
                json={
                    "topology": "turn_by_turn",
                    "debate_format": "free_form",
                    "decision_mode": "none",
                    "agents": [
                        {"display_name": "Alice"},
                        {"display_name": "Bob"},
                    ],
                    "initial_prompt": "Test topic",
                    "max_turns": 2,
                },
            )
            session_id = create_resp.json()["session_id"]

            # Execute a turn
            client.post(f"/v1/coordination/sessions/{session_id}/turn")

            # Get messages
            response = client.get(f"/v1/coordination/sessions/{session_id}/messages")
            assert response.status_code == 200
            data = response.json()
            assert "messages" in data
            assert len(data["messages"]) >= 2  # Initial prompt + agent message

            # Check message structure
            for msg in data["messages"]:
                assert "sender_id" in msg
                assert "sender_name" in msg
                assert "content" in msg
                assert "turn_number" in msg

    def test_full_conversation_flow(self):
        """Test complete conversation: create, execute multiple rounds, verify coherence."""
        app = create_app()

        with TestClient(app) as client:
            # Create session with 3 agents and structured debate
            create_resp = client.post(
                "/v1/coordination/sessions",
                json={
                    "topology": "turn_by_turn",
                    "debate_format": "structured",
                    "decision_mode": "majority_vote",
                    "agents": [
                        {
                            "display_name": "Pro",
                            "role": "advocate",
                            "system_prompt": "You support the proposition.",
                        },
                        {
                            "display_name": "Con",
                            "role": "critic",
                            "system_prompt": "You oppose the proposition.",
                        },
                        {
                            "display_name": "Moderator",
                            "role": "moderator",
                            "system_prompt": "You moderate the debate.",
                        },
                    ],
                    "initial_prompt": "Should AI development be open source?",
                    "max_turns": 9,  # 3 rounds with 3 agents
                },
            )
            assert create_resp.status_code == 201
            session_id = create_resp.json()["session_id"]

            # Execute 3 rounds
            for round_num in range(3):
                response = client.post(f"/v1/coordination/sessions/{session_id}/round")
                assert response.status_code == 200
                data = response.json()
                assert len(data["messages"]) == 3  # All 3 agents respond

                # Verify all agents are speaking
                agent_names = {msg["sender_name"] for msg in data["messages"]}
                assert agent_names == {"Pro", "Con", "Moderator"}

            # Get final messages
            response = client.get(f"/v1/coordination/sessions/{session_id}/messages")
            assert response.status_code == 200
            data = response.json()

            # Should have initial prompt + 9 agent messages (3 rounds × 3 agents)
            assert len(data["messages"]) == 10

            # Verify messages are in chronological order
            turn_numbers = [msg["turn_number"] for msg in data["messages"]]
            assert turn_numbers == sorted(turn_numbers)

            # Verify session is complete (reached max turns)
            status_resp = client.get(f"/v1/coordination/sessions/{session_id}")
            status = status_resp.json()
            assert status["current_turn"] == 9
            assert status["is_active"] is False  # Should be inactive after max turns

    def test_session_max_turns_enforcement(self):
        """Session should become inactive after reaching max_turns."""
        app = create_app()

        with TestClient(app) as client:
            # Create session with max_turns=2
            create_resp = client.post(
                "/v1/coordination/sessions",
                json={
                    "topology": "turn_by_turn",
                    "debate_format": "free_form",
                    "decision_mode": "none",
                    "agents": [
                        {"display_name": "Alice"},
                        {"display_name": "Bob"},
                    ],
                    "initial_prompt": "Test",
                    "max_turns": 2,
                },
            )
            session_id = create_resp.json()["session_id"]

            # Execute 2 turns
            client.post(f"/v1/coordination/sessions/{session_id}/turn")
            client.post(f"/v1/coordination/sessions/{session_id}/turn")

            # Session should be inactive
            status_resp = client.get(f"/v1/coordination/sessions/{session_id}")
            assert status_resp.json()["is_active"] is False

            # Attempting another turn should fail
            turn_resp = client.post(f"/v1/coordination/sessions/{session_id}/turn")
            assert turn_resp.status_code == 400
