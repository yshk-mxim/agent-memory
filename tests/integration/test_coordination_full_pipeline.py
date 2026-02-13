# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Yakov Shkolnikov and contributors
"""Integration tests for CoordinationService full pipeline.

Uses integration conftest's FakeBatchGenerator and MLX mocks to test
session lifecycle, turn execution, and cache handling end-to-end.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_memory.application.coordination_service import CoordinationService
from agent_memory.domain.coordination import (
    AgentLifecycle,
    AgentRole,
    DebateFormat,
    DecisionMode,
    Topology,
)
from agent_memory.domain.errors import SessionNotFoundError


@pytest.fixture
def mock_cache_store():
    store = MagicMock()
    store.load.return_value = None
    store.save.return_value = None
    store.delete.return_value = None
    store.list_all_agents.return_value = []
    return store


@pytest.fixture
def mock_engine():
    engine = MagicMock()
    tokenizer = MagicMock()
    tokenizer.encode.return_value = [1, 2, 3]
    tokenizer.chat_template = None
    engine.tokenizer = tokenizer
    engine.get_agent_blocks.return_value = MagicMock(total_tokens=50)
    return engine


@pytest.fixture
def mock_scheduler():
    scheduler = AsyncMock()
    completion = MagicMock()
    completion.text = "I think we should proceed."
    completion.token_count = 8
    completion.finish_reason = "stop"
    completion.blocks = MagicMock(total_tokens=50)
    scheduler.submit_and_wait.return_value = completion
    return scheduler


@pytest.fixture
def service(mock_scheduler, mock_cache_store, mock_engine):
    return CoordinationService(
        scheduler=mock_scheduler,
        cache_store=mock_cache_store,
        engine=mock_engine,
    )


@pytest.fixture
def sample_agents():
    return [
        AgentRole(agent_id="alice", display_name="Alice", role="participant"),
        AgentRole(agent_id="bob", display_name="Bob", role="critic"),
    ]


# ── Session lifecycle ───────────────────────────────────────────────


class TestSessionLifecycle:
    @pytest.mark.asyncio
    async def test_create_and_execute_turn(self, service, sample_agents) -> None:
        """Create session → execute_turn → message recorded in channel."""
        session = await service.create_session(
            topology=Topology.TURN_BY_TURN,
            debate_format=DebateFormat.FREE_FORM,
            decision_mode=DecisionMode.NONE,
            agents=sample_agents,
            initial_prompt="Should AI be regulated?",
        )

        with patch(
            "agent_memory.application.chat_completion_service.generate_chat_completion",
            new_callable=AsyncMock,
            return_value={
                "text": "I believe regulation is needed.",
                "token_count": 6,
                "finish_reason": "stop",
                "blocks": MagicMock(),
            },
        ):
            message = await service.execute_turn(session.session_id)

        assert message.sender_id == "alice"
        assert "regulation" in message.content

        # Message should be recorded in public channel
        public = next(c for c in session.channels.values() if c.channel_type == "public")
        agent_msgs = [m for m in public.messages if m.sender_id != "system"]
        assert len(agent_msgs) == 1

    @pytest.mark.asyncio
    async def test_execute_round_all_agents_speak(self, service, sample_agents) -> None:
        """execute_round → both agents produce messages."""
        session = await service.create_session(
            topology=Topology.TURN_BY_TURN,
            debate_format=DebateFormat.FREE_FORM,
            decision_mode=DecisionMode.NONE,
            agents=sample_agents,
            initial_prompt="Topic",
        )

        call_count = 0

        async def mock_gen(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return {
                "text": f"Response {call_count}",
                "token_count": 3,
                "finish_reason": "stop",
                "blocks": MagicMock(),
            }

        with patch(
            "agent_memory.application.chat_completion_service.generate_chat_completion",
            side_effect=mock_gen,
        ):
            messages = await service.execute_round(session.session_id)

        assert len(messages) == 2
        senders = {m.sender_id for m in messages}
        assert senders == {"alice", "bob"}

    @pytest.mark.asyncio
    async def test_delete_session_clears_ephemeral(
        self, service, sample_agents, mock_cache_store
    ) -> None:
        session = await service.create_session(
            topology=Topology.TURN_BY_TURN,
            debate_format=DebateFormat.FREE_FORM,
            decision_mode=DecisionMode.NONE,
            agents=sample_agents,
        )

        await service.delete_session(session.session_id)

        # Ephemeral caches should be deleted
        assert mock_cache_store.delete.call_count == 2  # alice + bob

        with pytest.raises(SessionNotFoundError):
            service.get_session(session.session_id)


# ── Persistent cache handling ───────────────────────────────────────


class TestPersistentCacheHandling:
    @pytest.mark.asyncio
    async def test_persistent_agents_use_persist_key(
        self, mock_scheduler, mock_cache_store, mock_engine
    ) -> None:
        svc = CoordinationService(
            scheduler=mock_scheduler,
            cache_store=mock_cache_store,
            engine=mock_engine,
        )

        agents = [
            AgentRole(
                agent_id="warden",
                display_name="Warden",
                role="moderator",
                lifecycle=AgentLifecycle.PERMANENT,
            ),
            AgentRole(
                agent_id="prisoner",
                display_name="Prisoner",
                role="participant",
                lifecycle=AgentLifecycle.EPHEMERAL,
            ),
        ]

        session = await svc.create_session(
            topology=Topology.TURN_BY_TURN,
            debate_format=DebateFormat.FREE_FORM,
            decision_mode=DecisionMode.NONE,
            agents=agents,
            persistent_cache_prefix="pd_game",
        )

        # Persistent agent gets persist_ key
        warden_key = svc._resolve_cache_key(session.session_id, "warden")
        assert warden_key == "persist_pd_game_warden"

        # Ephemeral agent gets session-scoped key
        prisoner_key = svc._resolve_cache_key(session.session_id, "prisoner")
        assert prisoner_key.startswith("coord_")

    @pytest.mark.asyncio
    async def test_delete_preserves_persistent_caches(
        self, mock_scheduler, mock_cache_store, mock_engine
    ) -> None:
        svc = CoordinationService(
            scheduler=mock_scheduler,
            cache_store=mock_cache_store,
            engine=mock_engine,
        )

        agents = [
            AgentRole(
                agent_id="warden",
                display_name="Warden",
                role="moderator",
                lifecycle=AgentLifecycle.PERMANENT,
            ),
            AgentRole(
                agent_id="prisoner",
                display_name="Prisoner",
                role="participant",
            ),
        ]

        session = await svc.create_session(
            topology=Topology.TURN_BY_TURN,
            debate_format=DebateFormat.FREE_FORM,
            decision_mode=DecisionMode.NONE,
            agents=agents,
            persistent_cache_prefix="pd_game",
        )

        await svc.delete_session(session.session_id)

        # Only ephemeral (prisoner) should be deleted, not persistent (warden)
        deleted_keys = [call.args[0] for call in mock_cache_store.delete.call_args_list]
        assert len(deleted_keys) == 1
        assert "prisoner" in deleted_keys[0]
        assert not any("warden" in k for k in deleted_keys)

    @pytest.mark.asyncio
    async def test_delete_persistent_caches_explicit(
        self, mock_scheduler, mock_cache_store, mock_engine
    ) -> None:
        svc = CoordinationService(
            scheduler=mock_scheduler,
            cache_store=mock_cache_store,
            engine=mock_engine,
        )

        mock_cache_store.list_all_agents.return_value = [
            {"agent_id": "persist_pd_game_warden"},
            {"agent_id": "persist_pd_game_prisoner"},
            {"agent_id": "coord_other_agent"},
        ]

        count = svc.delete_persistent_caches("pd_game")

        assert count == 2
        deleted_keys = [call.args[0] for call in mock_cache_store.delete.call_args_list]
        assert "persist_pd_game_warden" in deleted_keys
        assert "persist_pd_game_prisoner" in deleted_keys
