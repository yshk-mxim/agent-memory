# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Yakov Shkolnikov and contributors
"""Coverage tests for coordination_adapter.py — SSE generators and endpoint coverage."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from agent_memory.adapters.inbound.coordination_adapter import (
    router,
    stream_round_events,
    stream_turn_events,
)
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
from agent_memory.domain.errors import CoordinationError, SessionNotFoundError
from agent_memory.domain.value_objects import StreamDelta

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_session(session_id="sess_1"):
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
        session_id="sess_1",
        agent_id="a",
        turn_number=0,
        visible_messages=[],
        system_instruction="You are Alice.",
    )
    return svc


@pytest.fixture
def app(mock_service):
    app = FastAPI()
    app.include_router(router)

    # Mock app state
    semantic = SimpleNamespace()
    app.state.agent_memory = semantic
    app.state.coordination_service = mock_service

    # Patch get_coordination_service
    with patch(
        "agent_memory.adapters.inbound.coordination_adapter.get_coordination_service",
        return_value=mock_service,
    ):
        yield app


@pytest.fixture
def client(app):
    with patch(
        "agent_memory.adapters.inbound.coordination_adapter.get_coordination_service",
        return_value=app.state.coordination_service,
    ):
        yield TestClient(app)


# ===========================================================================
# stream_turn_events
# ===========================================================================


class TestStreamTurnEvents:
    async def test_normal_stream(self, mock_service):
        async def fake_turn_stream(session_id):
            yield StreamDelta(text="Hello", token_count=1)
            yield StreamDelta(text="Hello world", token_count=2, finish_reason="stop")
            yield StreamDelta(text="Hello world", token_count=2, finish_reason="cleaned")

        mock_service.execute_turn_stream = fake_turn_stream

        events = []
        async for event in stream_turn_events(mock_service, "sess_1"):
            events.append(event)

        assert events[0]["event"] == "turn_start"
        # Should have token events + turn_complete
        assert any(e["event"] == "token" for e in events)
        assert events[-1]["event"] == "turn_complete"

    async def test_cleaned_delta_used_for_complete(self, mock_service):
        async def fake_turn_stream(session_id):
            yield StreamDelta(text="Raw stuff", token_count=1, finish_reason="stop")
            yield StreamDelta(text="Cleaned text", token_count=1, finish_reason="cleaned")

        mock_service.execute_turn_stream = fake_turn_stream

        events = []
        async for event in stream_turn_events(mock_service, "sess_1"):
            events.append(event)

        complete = events[-1]
        data = json.loads(complete["data"])
        assert data["content"] == "Cleaned text"

    async def test_empty_accumulated_text(self, mock_service):
        async def fake_turn_stream(session_id):
            yield StreamDelta(text="", token_count=0, finish_reason="cleaned")

        mock_service.execute_turn_stream = fake_turn_stream

        events = []
        async for event in stream_turn_events(mock_service, "sess_1"):
            events.append(event)

        assert events[-1]["event"] == "turn_complete"

    async def test_coordination_error_yields_error_event(self, mock_service):
        mock_service.get_next_turn.side_effect = CoordinationError("test error")

        events = []
        async for event in stream_turn_events(mock_service, "sess_1"):
            events.append(event)

        assert len(events) == 1
        assert events[0]["event"] == "error"
        data = json.loads(events[0]["data"])
        assert "test error" in data["error"]


# ===========================================================================
# stream_round_events
# ===========================================================================


class TestStreamRoundEvents:
    async def test_two_agents(self, mock_service):
        async def fake_round_stream(session_id):
            yield ("a", "Alice", StreamDelta(text="Hi", token_count=1))
            yield ("b", "Bob", StreamDelta(text="Hey", token_count=1))

        mock_service.execute_round_stream = fake_round_stream

        events = []
        async for event in stream_round_events(mock_service, "sess_1"):
            events.append(event)

        assert any(e["event"] == "token" for e in events)
        assert events[-1]["event"] == "round_complete"

    async def test_coordination_error(self, mock_service):
        async def failing_stream(session_id):
            raise CoordinationError("round failed")
            yield  # Make it a generator

        mock_service.execute_round_stream = failing_stream

        events = []
        async for event in stream_round_events(mock_service, "sess_1"):
            events.append(event)

        assert events[0]["event"] == "error"

    async def test_empty_stream(self, mock_service):
        async def empty_stream(session_id):
            return
            yield

        mock_service.execute_round_stream = empty_stream

        events = []
        async for event in stream_round_events(mock_service, "sess_1"):
            events.append(event)

        assert len(events) == 1
        assert events[0]["event"] == "round_complete"


# ===========================================================================
# REST endpoints — create / list / get / delete
# ===========================================================================


class TestCreateSession:
    def test_valid_request(self, client, mock_service):
        session = _make_session("sess_new")
        mock_service.create_session = AsyncMock(return_value=session)

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

    def test_invalid_topology(self, client, mock_service):
        resp = client.post(
            "/v1/coordination/sessions",
            json={
                "topology": "invalid_topology",
                "agents": [{"display_name": "Alice"}],
            },
        )
        assert resp.status_code == 400


class TestListSessions:
    def test_empty_list(self, client, mock_service):
        mock_service.list_sessions.return_value = []
        resp = client.get("/v1/coordination/sessions")
        assert resp.status_code == 200
        assert resp.json()["sessions"] == []

    def test_one_session(self, client, mock_service):
        resp = client.get("/v1/coordination/sessions")
        assert resp.status_code == 200
        assert len(resp.json()["sessions"]) == 1


class TestGetSessionStatus:
    def test_valid_session(self, client, mock_service):
        resp = client.get("/v1/coordination/sessions/sess_1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["session_id"] == "sess_1"

    def test_not_found(self, client, mock_service):
        mock_service.get_session.side_effect = SessionNotFoundError("not found")
        resp = client.get("/v1/coordination/sessions/missing")
        assert resp.status_code == 404


class TestDeleteSession:
    def test_valid(self, client, mock_service):
        mock_service.delete_session = AsyncMock()
        resp = client.delete("/v1/coordination/sessions/sess_1")
        assert resp.status_code == 204

    def test_not_found(self, client, mock_service):
        mock_service.delete_session = AsyncMock(side_effect=SessionNotFoundError("not found"))
        resp = client.delete("/v1/coordination/sessions/missing")
        assert resp.status_code == 404


# ===========================================================================
# execute_turn / execute_round
# ===========================================================================


class TestExecuteTurn:
    def test_normal_flow(self, client, mock_service):
        msg = ChannelMessage(
            message_id="m1",
            channel_id="sess_1_public",
            sender_id="a",
            content="Hello",
            turn_number=0,
        )
        mock_service.execute_turn = AsyncMock(return_value=msg)

        resp = client.post("/v1/coordination/sessions/sess_1/turn")
        assert resp.status_code == 200
        data = resp.json()
        assert data["message"]["content"] == "Hello"

    def test_error(self, client, mock_service):
        mock_service.execute_turn = AsyncMock(side_effect=CoordinationError("turn error"))
        resp = client.post("/v1/coordination/sessions/sess_1/turn")
        assert resp.status_code == 400


class TestExecuteRound:
    def test_normal_flow(self, client, mock_service):
        msg = ChannelMessage(
            message_id="m1",
            channel_id="sess_1_public",
            sender_id="a",
            content="Hi",
            turn_number=0,
        )
        mock_service.execute_round = AsyncMock(return_value=[msg])

        resp = client.post("/v1/coordination/sessions/sess_1/round")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["messages"]) == 1

    def test_error(self, client, mock_service):
        mock_service.execute_round = AsyncMock(side_effect=CoordinationError("round error"))
        resp = client.post("/v1/coordination/sessions/sess_1/round")
        assert resp.status_code == 400


# ===========================================================================
# whisper / vote / messages / delete_persistent_caches
# ===========================================================================


class TestSendWhisper:
    def test_normal_flow(self, client, mock_service):
        msg = ChannelMessage(
            message_id="w1",
            channel_id="sess_1_whisper",
            sender_id="a",
            content="Secret",
            turn_number=0,
            visible_to=frozenset(["a", "b"]),
        )
        mock_service.add_whisper.return_value = msg

        resp = client.post(
            "/v1/coordination/sessions/sess_1/whisper",
            json={
                "from_agent_id": "a",
                "to_agent_id": "b",
                "content": "Secret message",
            },
        )
        assert resp.status_code == 200
        assert resp.json()["message"]["content"] == "Secret"


class TestSubmitVote:
    def test_returns_vote_id(self, client, mock_service):
        resp = client.post(
            "/v1/coordination/sessions/sess_1/vote",
            json={
                "agent_id": "a",
                "question": "Should we?",
                "choice": "yes",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "vote_id" in data
        assert data["agent_id"] == "a"
        assert data["choice"] == "yes"


class TestGetMessages:
    def test_normal_public_messages(self, client, mock_service):
        session = mock_service.get_session.return_value
        public_channel = next(c for c in session.channels.values() if c.channel_type == "public")
        public_channel.add_message(sender_id="a", content="Hello", turn_number=0)
        public_channel.add_message(sender_id="b", content="Hi", turn_number=1)

        resp = client.get("/v1/coordination/sessions/sess_1/messages")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["messages"]) == 2

    def test_private_messages_filtered(self, client, mock_service):
        session = mock_service.get_session.return_value
        public_channel = next(c for c in session.channels.values() if c.channel_type == "public")
        public_channel.add_message(sender_id="a", content="Public", turn_number=0)
        public_channel.add_message(
            sender_id="system",
            content="Private",
            turn_number=0,
            visible_to=frozenset(["a"]),
        )

        resp = client.get("/v1/coordination/sessions/sess_1/messages")
        assert resp.status_code == 200
        data = resp.json()
        # Only public messages (empty visible_to) should appear
        assert len(data["messages"]) == 1
        assert data["messages"][0]["content"] == "Public"

    def test_no_public_channel(self, client, mock_service):
        session = mock_service.get_session.return_value
        session.channels = {}  # No channels at all

        resp = client.get("/v1/coordination/sessions/sess_1/messages")
        assert resp.status_code == 200
        assert resp.json()["messages"] == []

    def test_agent_not_in_session(self, client, mock_service):
        session = mock_service.get_session.return_value
        public_channel = next(c for c in session.channels.values() if c.channel_type == "public")
        public_channel.add_message(sender_id="unknown_agent", content="From unknown", turn_number=0)

        resp = client.get("/v1/coordination/sessions/sess_1/messages")
        assert resp.status_code == 200
        data = resp.json()
        # Unknown agent gets sender_name="System" (fallback)
        assert data["messages"][-1]["sender_name"] == "System"

    def test_session_not_found(self, client, mock_service):
        mock_service.get_session.side_effect = SessionNotFoundError("gone")
        resp = client.get("/v1/coordination/sessions/missing/messages")
        assert resp.status_code == 404


class TestDeletePersistentCaches:
    def test_returns_204(self, client, mock_service):
        mock_service.delete_persistent_caches.return_value = 2
        resp = client.delete("/v1/coordination/caches/my_prefix")
        assert resp.status_code == 204
