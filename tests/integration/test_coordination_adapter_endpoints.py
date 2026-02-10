# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Yakov Shkolnikov and contributors
"""Integration tests for coordination adapter REST endpoints."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from agent_memory.adapters.inbound.coordination_adapter import router
from agent_memory.domain.coordination import (
    AgentRole,
    Channel,
    ChannelMessage,
    CoordinationSession,
    DebateFormat,
    DecisionMode,
    Topology,
    TurnDirective,
)
from agent_memory.domain.errors import SessionNotFoundError


def _make_session(session_id="sess_int"):
    agents = {
        "a": AgentRole(agent_id="a", display_name="Alice", role="participant"),
        "b": AgentRole(agent_id="b", display_name="Bob", role="critic"),
    }
    public_channel = Channel(
        channel_id=f"{session_id}_public",
        channel_type="public",
        participant_ids=frozenset(["a", "b"]),
    )
    return CoordinationSession(
        session_id=session_id,
        topology=Topology.TURN_BY_TURN,
        debate_format=DebateFormat.FREE_FORM,
        decision_mode=DecisionMode.NONE,
        agents=agents,
        channels={public_channel.channel_id: public_channel},
        turn_order=["a", "b"],
    )


@pytest.fixture
def mock_service():
    svc = MagicMock()
    session = _make_session()
    svc.get_session.return_value = session
    svc.list_sessions.return_value = [session]
    svc.get_next_turn.return_value = TurnDirective(
        session_id="sess_int",
        agent_id="a",
        turn_number=0,
        visible_messages=[],
        system_instruction="You are Alice.",
    )
    svc.create_session = AsyncMock(return_value=session)
    svc.delete_session = AsyncMock()
    svc.delete_persistent_caches.return_value = 0

    msg = ChannelMessage(
        message_id="m1",
        channel_id="sess_int_public",
        sender_id="a",
        content="Test reply",
        turn_number=0,
    )
    svc.execute_turn = AsyncMock(return_value=msg)
    svc.execute_round = AsyncMock(return_value=[msg])

    whisper_msg = ChannelMessage(
        message_id="w1",
        channel_id="sess_int_whisper",
        sender_id="a",
        content="Whisper text",
        turn_number=0,
        visible_to=frozenset(["a", "b"]),
    )
    svc.add_whisper.return_value = whisper_msg

    return svc


@pytest.fixture
def client(mock_service):
    app = FastAPI()
    app.include_router(router)
    semantic = SimpleNamespace()
    app.state.agent_memory = semantic
    app.state.coordination_service = mock_service

    with patch(
        "agent_memory.adapters.inbound.coordination_adapter.get_coordination_service",
        return_value=mock_service,
    ):
        yield TestClient(app)


class TestFullLifecycle:
    def test_create_list_status_turn_messages_delete(self, client, mock_service):
        # Create
        resp = client.post(
            "/v1/coordination/sessions",
            json={
                "agents": [
                    {"display_name": "Alice"},
                    {"display_name": "Bob"},
                ],
                "initial_prompt": "Test",
            },
        )
        assert resp.status_code == 201
        data = resp.json()
        session_id = data["session_id"]

        # List
        resp = client.get("/v1/coordination/sessions")
        assert resp.status_code == 200
        assert len(resp.json()["sessions"]) >= 1

        # Status
        resp = client.get(f"/v1/coordination/sessions/{session_id}")
        assert resp.status_code == 200

        # Execute turn
        resp = client.post(f"/v1/coordination/sessions/{session_id}/turn")
        assert resp.status_code == 200

        # Get messages
        resp = client.get(f"/v1/coordination/sessions/{session_id}/messages")
        assert resp.status_code == 200

        # Delete
        resp = client.delete(f"/v1/coordination/sessions/{session_id}")
        assert resp.status_code == 204


class TestWhisperFlow:
    def test_whisper(self, client, mock_service):
        resp = client.post(
            "/v1/coordination/sessions/sess_int/whisper",
            json={
                "from_agent_id": "a",
                "to_agent_id": "b",
                "content": "Private message",
            },
        )
        assert resp.status_code == 200
        assert resp.json()["message"]["content"] == "Whisper text"


class TestErrorResponses:
    def test_session_not_found_404(self, client, mock_service):
        mock_service.get_session.side_effect = SessionNotFoundError("gone")
        resp = client.get("/v1/coordination/sessions/missing")
        assert resp.status_code == 404

    def test_delete_not_found_404(self, client, mock_service):
        mock_service.delete_session = AsyncMock(
            side_effect=SessionNotFoundError("gone")
        )
        resp = client.delete("/v1/coordination/sessions/missing")
        assert resp.status_code == 404

    def test_invalid_topology_400(self, client, mock_service):
        resp = client.post(
            "/v1/coordination/sessions",
            json={
                "topology": "invalid",
                "agents": [{"display_name": "Alice"}],
            },
        )
        assert resp.status_code == 400
