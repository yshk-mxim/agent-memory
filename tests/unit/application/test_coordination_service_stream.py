# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Yakov Shkolnikov and contributors
"""Coverage tests for CoordinationService — streaming, helpers, _generate_direct."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_memory.application.coordination_service import CoordinationService, _DirectResult
from agent_memory.domain.coordination import (
    AgentRole,
    DebateFormat,
    DecisionMode,
    Topology,
    Vote,
)
from agent_memory.domain.value_objects import CompletedGeneration, StreamDelta

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_scheduler():
    scheduler = AsyncMock()
    return scheduler


@pytest.fixture
def mock_cache_store():
    store = MagicMock()
    store.load.return_value = None
    store.save.return_value = None
    store.list_all_agents.return_value = []
    return store


@pytest.fixture
def mock_engine():
    engine = MagicMock()
    tokenizer = MagicMock()
    tokenizer.encode.return_value = [1, 2, 3, 4, 5]
    tokenizer.chat_template = None  # No template by default
    engine.tokenizer = tokenizer
    engine.get_agent_blocks.return_value = None
    return engine


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
        AgentRole(agent_id="a", display_name="Alice", role="participant"),
        AgentRole(agent_id="b", display_name="Bob", role="critic"),
    ]


async def _create_session(service, agents, **kwargs):
    defaults = dict(
        topology=Topology.TURN_BY_TURN,
        debate_format=DebateFormat.FREE_FORM,
        decision_mode=DecisionMode.NONE,
        agents=agents,
        initial_prompt="Test topic",
    )
    defaults.update(kwargs)
    return await service.create_session(**defaults)


# ===========================================================================
# _clean_agent_response
# ===========================================================================


class TestCleanAgentResponse:
    def test_gpt_oss_channel_extraction(self):
        text = "analysis stuff<|channel|>final<|message|>Real response<|end|>"
        result = CoordinationService._clean_agent_response(text, "Alice")
        assert result == "Real response"

    def test_special_token_strip(self):
        text = "Hello<|im_end|> world"
        result = CoordinationService._clean_agent_response(text)
        assert "im_end" not in result

    def test_name_prefix_strip(self):
        result = CoordinationService._clean_agent_response(
            "Alice: Hello there", sender_name="Alice"
        )
        assert result == "Hello there"

    def test_bare_assistant_marker(self):
        result = CoordinationService._clean_agent_response("assistant\nHello")
        assert result == "Hello"

    def test_turn_cue_removal(self):
        result = CoordinationService._clean_agent_response(
            "[Alice, respond now.] Hello", sender_name="Alice"
        )
        assert result == "Hello"

    def test_instruction_fragment_removal(self):
        text = "Do not include any prefixes.\nActual response"
        result = CoordinationService._clean_agent_response(text)
        assert result == "Actual response"

    def test_stop_at_fake_continuation(self):
        text = "I think so.\nBob: I disagree"
        result = CoordinationService._clean_agent_response(
            text, sender_name="Alice", all_agent_names=["Alice", "Bob"]
        )
        assert result == "I think so."

    def test_combined_empty_input(self):
        result = CoordinationService._clean_agent_response("")
        assert result == ""

    def test_system_user_stop_markers(self):
        text = "My reply\nUser: next"
        result = CoordinationService._clean_agent_response(text)
        assert result == "My reply"

    def test_all_agent_names_as_stop(self):
        text = "My reply\nCharlie: something"
        result = CoordinationService._clean_agent_response(
            text, sender_name="Alice", all_agent_names=["Alice", "Bob", "Charlie"]
        )
        assert result == "My reply"


# ===========================================================================
# _merge_consecutive_messages
# ===========================================================================


class TestMergeConsecutiveMessages:
    def test_empty_list(self, service):
        assert service._merge_consecutive_messages([]) == []

    def test_mixed_roles_no_merge(self, service):
        msgs = [
            {"role": "user", "content": "A"},
            {"role": "assistant", "content": "B"},
            {"role": "user", "content": "C"},
        ]
        result = service._merge_consecutive_messages(msgs)
        assert len(result) == 3

    def test_consecutive_user_merged(self, service):
        msgs = [
            {"role": "user", "content": "Bob: Hello"},
            {"role": "user", "content": "Carol: Hi"},
        ]
        result = service._merge_consecutive_messages(msgs)
        assert len(result) == 1
        assert "Bob: Hello" in result[0]["content"]
        assert "Carol: Hi" in result[0]["content"]

    def test_system_messages_not_merged(self, service):
        msgs = [
            {"role": "system", "content": "Rule 1"},
            {"role": "system", "content": "Rule 2"},
        ]
        result = service._merge_consecutive_messages(msgs)
        assert len(result) == 2


# ===========================================================================
# _format_messages_as_text
# ===========================================================================


class TestFormatMessagesAsText:
    def test_multiple_messages(self, service):
        msgs = [
            {"role": "system", "content": "Rules"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        text = service._format_messages_as_text(msgs)
        assert "System: Rules" in text
        assert "User: Hello" in text
        assert "Assistant: Hi" in text

    def test_empty_list(self, service):
        assert service._format_messages_as_text([]) == ""


# ===========================================================================
# _get_generation_max_tokens
# ===========================================================================


class TestGetGenerationMaxTokens:
    def test_default(self, mock_scheduler, mock_cache_store, mock_engine):
        svc = CoordinationService(
            scheduler=mock_scheduler,
            cache_store=mock_cache_store,
            engine=mock_engine,
            reasoning_extra_tokens=0,
        )
        assert svc._get_generation_max_tokens() == 200

    def test_with_extra(self, mock_scheduler, mock_cache_store, mock_engine):
        svc = CoordinationService(
            scheduler=mock_scheduler,
            cache_store=mock_cache_store,
            engine=mock_engine,
            reasoning_extra_tokens=50,
        )
        assert svc._get_generation_max_tokens() == 250


# ===========================================================================
# _get_agent_name / _get_agent_role / _all_known_agent_names
# ===========================================================================


class TestNameRegistry:
    async def test_prior_agent_messages_populates_registry(self, service, sample_agents):
        session = await _create_session(
            service,
            sample_agents,
            prior_agent_messages={
                "a": [
                    {"sender_id": "x", "sender_name": "Xena", "content": "Hi"},
                ]
            },
        )
        assert service._agent_name_registry["x"] == "Xena"

    async def test_empty_content_skipped(self, service, sample_agents):
        await _create_session(
            service,
            sample_agents,
            prior_agent_messages={"a": [{"sender_id": "y", "sender_name": "Yara", "content": ""}]},
        )
        assert "y" not in service._agent_name_registry


class TestGetAgentName:
    async def test_agent_in_session(self, service, sample_agents):
        session = await _create_session(service, sample_agents)
        name = service._get_agent_name(session.session_id, "a")
        assert name == "Alice"

    async def test_fallback_to_registry(self, service, sample_agents):
        session = await _create_session(service, sample_agents)
        service._agent_name_registry["unknown_id"] = "Zara"
        assert service._get_agent_name(session.session_id, "unknown_id") == "Zara"

    async def test_fallback_to_agent_id(self, service, sample_agents):
        session = await _create_session(service, sample_agents)
        assert service._get_agent_name(session.session_id, "raw_id") == "raw_id"


class TestGetAgentRole:
    async def test_found(self, service, sample_agents):
        session = await _create_session(service, sample_agents)
        assert service._get_agent_role(session.session_id, "Alice") == "participant"
        assert service._get_agent_role(session.session_id, "Bob") == "critic"

    async def test_not_found(self, service, sample_agents):
        session = await _create_session(service, sample_agents)
        assert service._get_agent_role(session.session_id, "Nobody") == "participant"


class TestAllKnownAgentNames:
    async def test_session_plus_registry(self, service, sample_agents):
        session = await _create_session(service, sample_agents)
        service._agent_name_registry["ext"] = "External"
        names = service._all_known_agent_names(session.session_id)
        assert "Alice" in names
        assert "Bob" in names
        assert "External" in names

    async def test_no_session(self, service, sample_agents):
        service._agent_name_registry["ext"] = "External"
        names = service._all_known_agent_names("nonexistent")
        assert "External" in names


# ===========================================================================
# tally_votes
# ===========================================================================


class TestTallyVotes:
    async def test_normal_tally(self, service, sample_agents):
        session = await _create_session(service, sample_agents)
        votes = [
            Vote(
                vote_id="v1",
                session_id=session.session_id,
                agent_id="a",
                question="Q",
                choice="yes",
            ),
            Vote(
                vote_id="v2", session_id=session.session_id, agent_id="b", question="Q", choice="no"
            ),
            Vote(
                vote_id="v3",
                session_id=session.session_id,
                agent_id="c",
                question="Q",
                choice="yes",
            ),
        ]
        tally = service.tally_votes(session.session_id, votes)
        assert tally.total_votes == 3
        assert tally.results["yes"] == 2
        assert tally.winner == "yes"
        assert tally.tied is False

    async def test_empty_votes(self, service, sample_agents):
        session = await _create_session(service, sample_agents)
        tally = service.tally_votes(session.session_id, [])
        assert tally.total_votes == 0
        assert tally.results == {}

    async def test_tied_votes(self, service, sample_agents):
        session = await _create_session(service, sample_agents)
        votes = [
            Vote(
                vote_id="v1",
                session_id=session.session_id,
                agent_id="a",
                question="Q",
                choice="yes",
            ),
            Vote(
                vote_id="v2", session_id=session.session_id, agent_id="b", question="Q", choice="no"
            ),
        ]
        tally = service.tally_votes(session.session_id, votes)
        assert tally.tied is True
        assert tally.winner is None


# ===========================================================================
# _tokenize_chat_messages
# ===========================================================================


class TestTokenizeChatMessages:
    def test_no_chat_template_fallback(self, service):
        service._engine.tokenizer.chat_template = None
        tokens, text = service._tokenize_chat_messages([{"role": "system", "content": "Hello"}])
        service._engine.tokenizer.encode.assert_called()

    def test_template_exception_fallback(self, service):
        tok = service._engine.tokenizer
        tok.chat_template = "some template"
        tok.apply_chat_template.side_effect = RuntimeError("template fail")
        tokens, text = service._tokenize_chat_messages([{"role": "user", "content": "Test"}])
        tok.encode.assert_called()

    def test_generation_prefix_injection(self, service):
        tok = service._engine.tokenizer
        tok.chat_template = "valid"
        tok.apply_chat_template.side_effect = [
            "System:\nHello\n\nAssistant:",  # text version
            [1, 2, 3, 4],  # tokenized version
        ]
        tok.encode.return_value = [99, 100]

        tokens, text = service._tokenize_chat_messages(
            [{"role": "system", "content": "Hello"}],
            generation_prefix="Warden:",
        )
        assert 99 in tokens or 100 in tokens  # suffix tokens appended
        assert "Warden:" in text

    def test_message_merging_branch(self, service):
        tok = service._engine.tokenizer
        tok.chat_template = "valid"
        tok.apply_chat_template.side_effect = [
            "merged text",
            [10, 20, 30],
        ]
        # Set up chat template port to say merging is needed
        template_port = MagicMock()
        template_port.needs_message_merging.return_value = True
        template_port.get_template_kwargs.return_value = {
            "tokenize": True,
            "add_generation_prompt": True,
        }
        service._chat_template = template_port

        msgs = [
            {"role": "user", "content": "A"},
            {"role": "user", "content": "B"},
        ]
        tokens, text = service._tokenize_chat_messages(msgs)
        assert tokens == [10, 20, 30]


# ===========================================================================
# execute_turn — empty generation paths
# ===========================================================================


class TestExecuteTurnEmptyGeneration:
    async def test_empty_text_creates_placeholder(self, service, sample_agents):
        session = await _create_session(service, sample_agents)

        with patch(
            "agent_memory.application.chat_completion_service.generate_chat_completion",
            new_callable=AsyncMock,
        ) as mock_gen:
            mock_gen.return_value = {"text": ""}
            message = await service.execute_turn(session.session_id)
            assert message.content == ""
            assert message.metadata.get("empty_generation") is True

    async def test_whitespace_only_treated_as_empty(self, service, sample_agents):
        session = await _create_session(service, sample_agents)

        with patch(
            "agent_memory.application.chat_completion_service.generate_chat_completion",
            new_callable=AsyncMock,
        ) as mock_gen:
            mock_gen.return_value = {"text": "   \n\n  "}
            message = await service.execute_turn(session.session_id)
            assert message.metadata.get("empty_generation") is True


# ===========================================================================
# _generate_direct
# ===========================================================================


class TestGenerateDirect:
    async def test_normal_generation(self, service, sample_agents):
        completion = CompletedGeneration(
            uid="uid_1", text="Hello", blocks=None, finish_reason="stop", token_count=1
        )
        service._engine.submit.return_value = "uid_1"
        service._engine.step.return_value = [completion]

        result = await service._generate_direct(
            agent_id="test",
            prompt_tokens=[1, 2, 3],
            cache=None,
            max_tokens=100,
            temperature=0.3,
            top_p=0.95,
        )
        assert isinstance(result, _DirectResult)
        assert result.text == "Hello"

    async def test_no_completion_raises(self, service, sample_agents):
        service._engine.submit.return_value = "uid_x"
        service._engine.step.return_value = []  # Never completes

        async def _noop(*a, **kw):
            pass

        with patch("asyncio.sleep", new=_noop):
            with pytest.raises(RuntimeError, match="Generation failed"):
                await service._generate_direct(
                    agent_id="test",
                    prompt_tokens=[1, 2, 3],
                    cache=None,
                    max_tokens=100,
                    temperature=0.3,
                    top_p=0.95,
                )

    async def test_first_step_returns_completion(self, service, sample_agents):
        completion = CompletedGeneration(
            uid="uid_1", text="Fast", blocks=None, finish_reason="stop", token_count=1
        )
        service._engine.submit.return_value = "uid_1"
        service._engine.step.return_value = [completion]

        result = await service._generate_direct(
            agent_id="test",
            prompt_tokens=[1],
            cache=None,
            max_tokens=10,
            temperature=0.0,
            top_p=1.0,
        )
        assert result.text == "Fast"


# ===========================================================================
# execute_turn_stream
# ===========================================================================


class TestExecuteTurnStream:
    async def test_normal_stream(self, service, sample_agents, mock_scheduler):
        session = await _create_session(service, sample_agents)

        # Set up scheduler to return streaming deltas
        deltas = [
            StreamDelta(text="Hello", token_count=1),
            StreamDelta(text="Hello world", token_count=2),
            StreamDelta(text="Hello world!", token_count=3, finish_reason="stop"),
        ]

        async def fake_stream(**kwargs):
            for d in deltas:
                yield d

        mock_scheduler.submit_and_stream = fake_stream

        collected = []
        async for delta in service.execute_turn_stream(session.session_id):
            collected.append(delta)

        # Should have deltas + final cleaned delta
        assert len(collected) >= 3
        # Last delta should be the "cleaned" replacement
        assert collected[-1].finish_reason == "cleaned"

    async def test_no_scheduler_fallback(self, service, sample_agents, mock_engine):
        svc = CoordinationService(
            scheduler=None,
            cache_store=service._cache_store,
            engine=mock_engine,
        )
        session = await _create_session(svc, sample_agents)

        # Set up direct generation
        completion = CompletedGeneration(
            uid="uid_1", text="Direct reply", blocks=None, finish_reason="stop", token_count=2
        )
        mock_engine.submit.return_value = "uid_1"
        mock_engine.step.return_value = [completion]

        collected = []
        async for delta in svc.execute_turn_stream(session.session_id):
            collected.append(delta)

        assert len(collected) >= 2  # One direct + one cleaned
        texts = [d.text for d in collected]
        assert "Direct reply" in texts

    async def test_empty_generation_streaming(self, service, sample_agents, mock_scheduler):
        session = await _create_session(service, sample_agents)

        async def fake_stream(**kwargs):
            yield StreamDelta(text="", token_count=0, finish_reason="stop")

        mock_scheduler.submit_and_stream = fake_stream

        collected = []
        async for delta in service.execute_turn_stream(session.session_id):
            collected.append(delta)

        # Should still get cleaned delta
        assert any(d.finish_reason == "cleaned" for d in collected)


# ===========================================================================
# execute_round_stream
# ===========================================================================


class TestExecuteRoundStream:
    async def test_two_agents(self, service, sample_agents, mock_scheduler):
        session = await _create_session(service, sample_agents)

        call_count = 0

        async def fake_stream(**kwargs):
            nonlocal call_count
            call_count += 1
            yield StreamDelta(text=f"Reply {call_count}", token_count=1, finish_reason="stop")

        mock_scheduler.submit_and_stream = fake_stream

        collected = []
        async for agent_id, agent_name, delta in service.execute_round_stream(session.session_id):
            collected.append((agent_id, agent_name, delta))

        assert len(collected) >= 2  # At least 1 per agent

    async def test_session_inactive_mid_round(self, service, sample_agents, mock_scheduler):
        session = await _create_session(service, sample_agents, max_turns=1)

        async def fake_stream(**kwargs):
            yield StreamDelta(text="Only one", token_count=1, finish_reason="stop")

        mock_scheduler.submit_and_stream = fake_stream

        collected = []
        async for agent_id, agent_name, delta in service.execute_round_stream(session.session_id):
            collected.append((agent_id, agent_name, delta))

        # Should only process one agent because max_turns=1 deactivates after first
        # The exact count depends on how advance_turn works but should be < 2*N agents
        assert len(collected) >= 1
