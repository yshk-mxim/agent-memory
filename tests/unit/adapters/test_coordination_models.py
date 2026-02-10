# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Yakov Shkolnikov and contributors
"""Unit tests for coordination request/response models.

Validates Pydantic field constraints, defaults, and model construction.
"""

import pytest
from pydantic import ValidationError

from agent_memory.adapters.inbound.coordination_models import (
    AgentRoleConfig,
    AgentStateResponse,
    ChannelMessageResponse,
    CreateSessionRequest,
    CreateSessionResponse,
    ExecuteRoundResponse,
    ExecuteTurnResponse,
    MessageListResponse,
    SessionListResponse,
    SessionStatusResponse,
    TallyResponse,
    VoteRequest,
    VoteResponse,
    WhisperRequest,
    WhisperResponse,
)

pytestmark = pytest.mark.unit


# ── AgentRoleConfig ─────────────────────────────────────────────────


class TestAgentRoleConfig:

    def test_valid_minimal(self) -> None:
        cfg = AgentRoleConfig(display_name="Alice")
        assert cfg.display_name == "Alice"
        assert cfg.agent_id is None
        assert cfg.role == "participant"
        assert cfg.system_prompt == ""
        assert cfg.lifecycle == "ephemeral"

    def test_valid_full(self) -> None:
        cfg = AgentRoleConfig(
            agent_id="a1",
            display_name="Bob",
            role="critic",
            system_prompt="Challenge everything.",
            lifecycle="permanent",
        )
        assert cfg.agent_id == "a1"
        assert cfg.lifecycle == "permanent"

    def test_display_name_empty_rejected(self) -> None:
        with pytest.raises(ValidationError):
            AgentRoleConfig(display_name="")

    def test_display_name_too_long_rejected(self) -> None:
        with pytest.raises(ValidationError):
            AgentRoleConfig(display_name="A" * 51)

    def test_display_name_max_length_accepted(self) -> None:
        cfg = AgentRoleConfig(display_name="A" * 50)
        assert len(cfg.display_name) == 50

    def test_lifecycle_invalid_rejected(self) -> None:
        with pytest.raises(ValidationError):
            AgentRoleConfig(display_name="Alice", lifecycle="transient")

    def test_lifecycle_ephemeral(self) -> None:
        cfg = AgentRoleConfig(display_name="Alice", lifecycle="ephemeral")
        assert cfg.lifecycle == "ephemeral"

    def test_lifecycle_permanent(self) -> None:
        cfg = AgentRoleConfig(display_name="Alice", lifecycle="permanent")
        assert cfg.lifecycle == "permanent"

    def test_system_prompt_max_length(self) -> None:
        cfg = AgentRoleConfig(display_name="Alice", system_prompt="x" * 2000)
        assert len(cfg.system_prompt) == 2000

    def test_system_prompt_too_long(self) -> None:
        with pytest.raises(ValidationError):
            AgentRoleConfig(display_name="Alice", system_prompt="x" * 2001)


# ── CreateSessionRequest ────────────────────────────────────────────


class TestCreateSessionRequest:

    def test_valid_minimal(self) -> None:
        req = CreateSessionRequest(
            agents=[AgentRoleConfig(display_name="Alice")],
        )
        assert req.topology == "turn_by_turn"
        assert req.debate_format == "free_form"
        assert req.decision_mode == "none"
        assert len(req.agents) == 1
        assert req.max_turns == 0
        assert req.initial_prompt == ""
        assert req.persistent_cache_prefix == ""

    def test_agents_empty_rejected(self) -> None:
        with pytest.raises(ValidationError):
            CreateSessionRequest(agents=[])

    def test_agents_max_accepted(self) -> None:
        agents = [AgentRoleConfig(display_name=f"Agent{i}") for i in range(30)]
        req = CreateSessionRequest(agents=agents)
        assert len(req.agents) == 30

    def test_agents_over_max_rejected(self) -> None:
        agents = [AgentRoleConfig(display_name=f"Agent{i}") for i in range(31)]
        with pytest.raises(ValidationError):
            CreateSessionRequest(agents=agents)

    def test_max_turns_negative_rejected(self) -> None:
        with pytest.raises(ValidationError):
            CreateSessionRequest(
                agents=[AgentRoleConfig(display_name="A")],
                max_turns=-1,
            )

    def test_max_turns_at_limit(self) -> None:
        req = CreateSessionRequest(
            agents=[AgentRoleConfig(display_name="A")],
            max_turns=1000,
        )
        assert req.max_turns == 1000

    def test_max_turns_over_limit_rejected(self) -> None:
        with pytest.raises(ValidationError):
            CreateSessionRequest(
                agents=[AgentRoleConfig(display_name="A")],
                max_turns=1001,
            )

    def test_per_agent_prompts_default_empty(self) -> None:
        req = CreateSessionRequest(
            agents=[AgentRoleConfig(display_name="A")],
        )
        assert req.per_agent_prompts == {}

    def test_prior_agent_messages_default_empty(self) -> None:
        req = CreateSessionRequest(
            agents=[AgentRoleConfig(display_name="A")],
        )
        assert req.prior_agent_messages == {}


# ── WhisperRequest ──────────────────────────────────────────────────


class TestWhisperRequest:

    def test_valid(self) -> None:
        req = WhisperRequest(
            from_agent_id="a",
            to_agent_id="b",
            content="Hello",
        )
        assert req.from_agent_id == "a"
        assert req.content == "Hello"

    def test_content_empty_rejected(self) -> None:
        with pytest.raises(ValidationError):
            WhisperRequest(from_agent_id="a", to_agent_id="b", content="")

    def test_content_max_length(self) -> None:
        req = WhisperRequest(
            from_agent_id="a",
            to_agent_id="b",
            content="x" * 5000,
        )
        assert len(req.content) == 5000

    def test_content_over_max_rejected(self) -> None:
        with pytest.raises(ValidationError):
            WhisperRequest(
                from_agent_id="a",
                to_agent_id="b",
                content="x" * 5001,
            )


# ── VoteRequest ─────────────────────────────────────────────────────


class TestVoteRequest:

    def test_valid_with_ranking(self) -> None:
        req = VoteRequest(
            agent_id="a",
            question="Best?",
            choice="yes",
            ranking=["yes", "no"],
        )
        assert req.ranking == ["yes", "no"]

    def test_ranking_defaults_empty(self) -> None:
        req = VoteRequest(agent_id="a", question="Q", choice="A")
        assert req.ranking == []


# ── Response models ─────────────────────────────────────────────────


class TestResponseModels:

    def test_create_session_response(self) -> None:
        resp = CreateSessionResponse(
            session_id="s1",
            agents=[AgentRoleConfig(display_name="Alice")],
            topology="turn_by_turn",
            debate_format="free_form",
            decision_mode="none",
            status="active",
        )
        assert resp.session_id == "s1"
        assert resp.status == "active"

    def test_channel_message_response(self) -> None:
        resp = ChannelMessageResponse(
            message_id="m1",
            sender_id="a",
            sender_name="Alice",
            content="Hello",
            turn_number=1,
            channel_type="public",
        )
        assert resp.is_interrupted is False

    def test_agent_state_response(self) -> None:
        resp = AgentStateResponse(
            agent_id="a",
            display_name="Alice",
            role="participant",
            message_count=5,
            lifecycle="ephemeral",
        )
        assert resp.message_count == 5

    def test_session_status_response(self) -> None:
        resp = SessionStatusResponse(
            session_id="s1",
            current_turn=3,
            is_active=True,
            next_speaker="a",
            agent_states=[],
        )
        assert resp.next_speaker == "a"

    def test_execute_turn_response(self) -> None:
        msg = ChannelMessageResponse(
            message_id="m1", sender_id="a", sender_name="Alice",
            content="Hello", turn_number=1, channel_type="public",
        )
        status = SessionStatusResponse(
            session_id="s1", current_turn=1, is_active=True,
            next_speaker="b", agent_states=[],
        )
        resp = ExecuteTurnResponse(message=msg, session_status=status)
        assert resp.message.message_id == "m1"

    def test_execute_round_response(self) -> None:
        status = SessionStatusResponse(
            session_id="s1", current_turn=3, is_active=True,
            next_speaker=None, agent_states=[],
        )
        resp = ExecuteRoundResponse(messages=[], session_status=status)
        assert resp.messages == []

    def test_whisper_response(self) -> None:
        msg = ChannelMessageResponse(
            message_id="m1", sender_id="a", sender_name="Alice",
            content="psst", turn_number=1, channel_type="whisper",
        )
        resp = WhisperResponse(message=msg)
        assert resp.message.channel_type == "whisper"

    def test_vote_response(self) -> None:
        resp = VoteResponse(vote_id="v1", agent_id="a", choice="yes")
        assert resp.vote_id == "v1"

    def test_tally_response(self) -> None:
        resp = TallyResponse(
            question="Best?",
            total_votes=3,
            results={"yes": 2, "no": 1},
            winner="yes",
            tied=False,
        )
        assert resp.winner == "yes"

    def test_session_list_response(self) -> None:
        resp = SessionListResponse(sessions=[])
        assert resp.sessions == []

    def test_message_list_response(self) -> None:
        resp = MessageListResponse(session_id="s1", messages=[])
        assert resp.session_id == "s1"
