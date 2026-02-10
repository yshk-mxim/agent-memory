# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Yakov Shkolnikov and contributors
"""Integration tests for coordination service streaming."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_memory.application.coordination_service import CoordinationService
from agent_memory.domain.coordination import (
    AgentRole,
    DebateFormat,
    DecisionMode,
    Topology,
)
from agent_memory.domain.errors import CoordinationError
from agent_memory.domain.value_objects import CompletedGeneration, StreamDelta


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_engine():
    engine = MagicMock()
    tokenizer = MagicMock()
    tokenizer.encode.return_value = [1, 2, 3, 4, 5]
    tokenizer.chat_template = None
    engine.tokenizer = tokenizer
    engine.get_agent_blocks.return_value = None
    return engine


@pytest.fixture
def mock_cache_store():
    store = MagicMock()
    store.load.return_value = None
    store.save.return_value = None
    store.list_all_agents.return_value = []
    return store


@pytest.fixture
def agents():
    return [
        AgentRole(agent_id="a", display_name="Alice", role="participant"),
        AgentRole(agent_id="b", display_name="Bob", role="critic"),
    ]


async def _setup_service_and_session(mock_engine, mock_cache_store, agents, scheduler=None):
    svc = CoordinationService(
        scheduler=scheduler,
        cache_store=mock_cache_store,
        engine=mock_engine,
    )
    session = await svc.create_session(
        topology=Topology.TURN_BY_TURN,
        debate_format=DebateFormat.FREE_FORM,
        decision_mode=DecisionMode.NONE,
        agents=agents,
        initial_prompt="Discuss AI safety",
    )
    return svc, session


# ===========================================================================
# Turn streaming
# ===========================================================================


class TestTurnStreaming:
    async def test_full_sse_sequence(self, mock_engine, mock_cache_store, agents):
        """Stream turn yields tokens and ends with cleaned delta."""
        mock_scheduler = AsyncMock()

        async def fake_stream(**kwargs):
            yield StreamDelta(text="I", token_count=1)
            yield StreamDelta(text="I think", token_count=2)
            yield StreamDelta(text="I think so", token_count=3, finish_reason="stop")

        mock_scheduler.submit_and_stream = fake_stream

        svc, session = await _setup_service_and_session(
            mock_engine, mock_cache_store, agents, scheduler=mock_scheduler
        )

        collected = []
        async for delta in svc.execute_turn_stream(session.session_id):
            collected.append(delta)

        # Should have raw deltas + final cleaned delta
        assert len(collected) >= 3
        assert collected[-1].finish_reason == "cleaned"
        assert isinstance(collected[-1].text, str)

    async def test_token_accumulation(self, mock_engine, mock_cache_store, agents):
        """New text extracted correctly from accumulated deltas."""
        mock_scheduler = AsyncMock()

        async def fake_stream(**kwargs):
            yield StreamDelta(text="Hello", token_count=1)
            yield StreamDelta(text="Hello world", token_count=2)
            yield StreamDelta(text="Hello world!", token_count=3, finish_reason="stop")

        mock_scheduler.submit_and_stream = fake_stream

        svc, session = await _setup_service_and_session(
            mock_engine, mock_cache_store, agents, scheduler=mock_scheduler
        )

        texts = []
        async for delta in svc.execute_turn_stream(session.session_id):
            texts.append(delta.text)

        # Raw deltas should show accumulation
        assert "Hello" in texts
        assert "Hello world" in texts

    async def test_direct_fallback(self, mock_engine, mock_cache_store, agents):
        """No scheduler → uses _generate_direct."""
        completion = CompletedGeneration(
            uid="uid_1", text="Direct answer", blocks=None,
            finish_reason="stop", token_count=2,
        )
        mock_engine.submit.return_value = "uid_1"
        mock_engine.step.return_value = [completion]

        svc, session = await _setup_service_and_session(
            mock_engine, mock_cache_store, agents, scheduler=None
        )

        collected = []
        async for delta in svc.execute_turn_stream(session.session_id):
            collected.append(delta)

        assert len(collected) >= 2
        texts = [d.text for d in collected]
        assert "Direct answer" in texts


# ===========================================================================
# Round streaming
# ===========================================================================


class TestRoundStreaming:
    async def test_multiple_agents(self, mock_engine, mock_cache_store, agents):
        mock_scheduler = AsyncMock()
        call_count = 0

        async def fake_stream(**kwargs):
            nonlocal call_count
            call_count += 1
            yield StreamDelta(
                text=f"Reply from agent {call_count}",
                token_count=1,
                finish_reason="stop",
            )

        mock_scheduler.submit_and_stream = fake_stream

        svc, session = await _setup_service_and_session(
            mock_engine, mock_cache_store, agents, scheduler=mock_scheduler
        )

        collected = []
        async for agent_id, name, delta in svc.execute_round_stream(session.session_id):
            collected.append((agent_id, name, delta))

        assert len(collected) >= 2
        agent_ids = {item[0] for item in collected}
        assert "a" in agent_ids
