# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Yakov Shkolnikov and contributors
"""Unit tests for CoordinationService.

Tests coordination service logic with mocked scheduler, cache_store, and engine.
Verifies prompt construction, turn routing, message recording, and vote tallying.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_memory.application.coordination_service import CoordinationService
from agent_memory.domain.coordination import (
    AgentLifecycle,
    AgentRole,
    DebateFormat,
    DecisionMode,
    Topology,
    Vote,
)
from agent_memory.domain.errors import InvalidTurnError, SessionNotFoundError

pytestmark = pytest.mark.unit


@pytest.fixture
def mock_scheduler():
    """Mock ConcurrentScheduler."""
    scheduler = AsyncMock()
    return scheduler


@pytest.fixture
def mock_cache_store():
    """Mock AgentCacheStore."""
    store = MagicMock()
    store.load.return_value = None  # No cached blocks initially
    store.save.return_value = None
    return store


@pytest.fixture
def mock_engine():
    """Mock BlockPoolBatchEngine."""
    engine = MagicMock()
    tokenizer = MagicMock()
    tokenizer.encode.return_value = [1, 2, 3, 4, 5]  # Mock token IDs
    engine.tokenizer = tokenizer
    return engine


@pytest.fixture
def service(mock_scheduler, mock_cache_store, mock_engine):
    """CoordinationService with mocked dependencies."""
    return CoordinationService(
        scheduler=mock_scheduler,
        cache_store=mock_cache_store,
        engine=mock_engine,
    )


@pytest.fixture
def sample_agents():
    """Sample agent roles for testing."""
    return [
        AgentRole(agent_id="a", display_name="Alice", role="participant"),
        AgentRole(agent_id="b", display_name="Bob", role="critic"),
        AgentRole(agent_id="c", display_name="Charlie", role="moderator"),
    ]


class TestCreateSession:
    """Tests for create_session()."""

    async def test_create_basic_session(self, service: CoordinationService, sample_agents) -> None:
        session = await service.create_session(
            topology=Topology.TURN_BY_TURN,
            debate_format=DebateFormat.FREE_FORM,
            decision_mode=DecisionMode.NONE,
            agents=sample_agents,
            initial_prompt="Test topic",
            max_turns=10,
        )

        assert session.session_id.startswith("coord_")
        assert session.topology == Topology.TURN_BY_TURN
        assert session.debate_format == DebateFormat.FREE_FORM
        assert session.decision_mode == DecisionMode.NONE
        assert len(session.agents) == 3
        assert session.turn_order == ["a", "b", "c"]
        assert session.max_turns == 10
        assert session.is_active

    async def test_create_session_adds_initial_prompt(
        self, service: CoordinationService, sample_agents
    ) -> None:
        session = await service.create_session(
            topology=Topology.TURN_BY_TURN,
            debate_format=DebateFormat.FREE_FORM,
            decision_mode=DecisionMode.NONE,
            agents=sample_agents,
            initial_prompt="Should AI be open source?",
        )

        # Check that public channel has system message
        public_channel = next(c for c in session.channels.values() if c.channel_type == "public")
        assert len(public_channel.messages) == 1
        assert public_channel.messages[0].sender_id == "system"
        assert "Should AI be open source?" in public_channel.messages[0].content

    async def test_create_session_stores_in_service(
        self, service: CoordinationService, sample_agents
    ) -> None:
        session = await service.create_session(
            topology=Topology.TURN_BY_TURN,
            debate_format=DebateFormat.FREE_FORM,
            decision_mode=DecisionMode.NONE,
            agents=sample_agents,
        )

        # Session should be retrievable
        retrieved = service.get_session(session.session_id)
        assert retrieved.session_id == session.session_id

    async def test_create_session_per_agent_prompts(
        self, service: CoordinationService, sample_agents
    ) -> None:
        session = await service.create_session(
            topology=Topology.TURN_BY_TURN,
            debate_format=DebateFormat.FREE_FORM,
            decision_mode=DecisionMode.NONE,
            agents=sample_agents,
            initial_prompt="Shared context",
            per_agent_prompts={
                "Alice": "Alice's private memories",
                "Bob": "Bob's private memories",
            },
        )

        public_channel = next(c for c in session.channels.values() if c.channel_type == "public")
        # 1 shared + 2 per-agent = 3 system messages
        system_msgs = [m for m in public_channel.messages if m.sender_id == "system"]
        assert len(system_msgs) == 3

        # Shared message is public (empty visible_to)
        shared = system_msgs[0]
        assert shared.visible_to == frozenset()
        assert "Shared context" in shared.content

        # Per-agent messages have restricted visibility
        alice_msg = next(m for m in system_msgs if "Alice" in m.content and m.visible_to)
        assert alice_msg.visible_to == frozenset(["a"])  # Only Alice can see

        bob_msg = next(m for m in system_msgs if "Bob" in m.content and m.visible_to)
        assert bob_msg.visible_to == frozenset(["b"])  # Only Bob can see


class TestGetSession:
    """Tests for get_session()."""

    async def test_get_existing_session(self, service: CoordinationService, sample_agents) -> None:
        created = await service.create_session(
            topology=Topology.TURN_BY_TURN,
            debate_format=DebateFormat.FREE_FORM,
            decision_mode=DecisionMode.NONE,
            agents=sample_agents,
        )

        retrieved = service.get_session(created.session_id)
        assert retrieved.session_id == created.session_id

    def test_get_nonexistent_session_raises(self, service: CoordinationService) -> None:
        with pytest.raises(SessionNotFoundError):
            service.get_session("nonexistent_id")


class TestDeleteSession:
    """Tests for delete_session()."""

    async def test_delete_existing_session(
        self,
        service: CoordinationService,
        sample_agents,
    ) -> None:
        session = await service.create_session(
            topology=Topology.TURN_BY_TURN,
            debate_format=DebateFormat.FREE_FORM,
            decision_mode=DecisionMode.NONE,
            agents=sample_agents,
        )

        await service.delete_session(session.session_id)

        with pytest.raises(SessionNotFoundError):
            service.get_session(session.session_id)

    async def test_delete_nonexistent_session_raises(self, service: CoordinationService) -> None:
        with pytest.raises(SessionNotFoundError):
            await service.delete_session("nonexistent_id")


class TestGetNextTurn:
    """Tests for get_next_turn()."""

    async def test_get_first_turn(self, service: CoordinationService, sample_agents) -> None:
        session = await service.create_session(
            topology=Topology.TURN_BY_TURN,
            debate_format=DebateFormat.FREE_FORM,
            decision_mode=DecisionMode.NONE,
            agents=sample_agents,
        )

        directive = service.get_next_turn(session.session_id)

        assert directive.session_id == session.session_id
        assert directive.agent_id == "a"  # First in turn_order
        assert directive.turn_number == 0
        assert directive.system_instruction  # Should have system prompt

    async def test_get_turn_inactive_session_raises(
        self, service: CoordinationService, sample_agents
    ) -> None:
        session = await service.create_session(
            topology=Topology.TURN_BY_TURN,
            debate_format=DebateFormat.FREE_FORM,
            decision_mode=DecisionMode.NONE,
            agents=sample_agents,
        )
        session.is_active = False

        with pytest.raises(InvalidTurnError):
            service.get_next_turn(session.session_id)

    async def test_visible_messages_filtered(
        self, service: CoordinationService, sample_agents
    ) -> None:
        session = await service.create_session(
            topology=Topology.TURN_BY_TURN,
            debate_format=DebateFormat.FREE_FORM,
            decision_mode=DecisionMode.NONE,
            agents=sample_agents,
            initial_prompt="Test",
        )

        directive = service.get_next_turn(session.session_id)

        # Should see initial prompt
        assert len(directive.visible_messages) == 1
        assert directive.visible_messages[0].sender_id == "system"


class TestBuildAgentPrompt:
    """Tests for build_agent_prompt()."""

    async def test_build_prompt_includes_system_instruction(
        self, service: CoordinationService, sample_agents
    ) -> None:
        session = await service.create_session(
            topology=Topology.TURN_BY_TURN,
            debate_format=DebateFormat.FREE_FORM,
            decision_mode=DecisionMode.NONE,
            agents=sample_agents,
        )

        directive = service.get_next_turn(session.session_id)
        agent_role = session.agents[directive.agent_id]
        messages = service.build_agent_prompt(directive, agent_role)

        assert messages[0]["role"] == "system"
        assert "Alice" in messages[0]["content"]

    async def test_build_prompt_formats_other_agents_as_user(
        self, service: CoordinationService, sample_agents
    ) -> None:
        session = await service.create_session(
            topology=Topology.TURN_BY_TURN,
            debate_format=DebateFormat.FREE_FORM,
            decision_mode=DecisionMode.NONE,
            agents=sample_agents,
            initial_prompt="Test",
        )

        public_channel = next(c for c in session.channels.values() if c.channel_type == "public")
        public_channel.add_message(sender_id="b", content="Hello from Bob", turn_number=1)

        directive = service.get_next_turn(session.session_id)
        agent_role = session.agents["a"]
        messages = service.build_agent_prompt(directive, agent_role)

        bob_messages = [
            m for m in messages
            if m["role"] == "user" and "Hello from Bob" in m.get("content", "")
        ]
        assert len(bob_messages) == 1
        assert "Bob: Hello from Bob" in bob_messages[0]["content"]

    async def test_per_agent_prompt_visible_only_to_target(
        self,
        service: CoordinationService,
        sample_agents,
    ) -> None:
        session = await service.create_session(
            topology=Topology.TURN_BY_TURN,
            debate_format=DebateFormat.FREE_FORM,
            decision_mode=DecisionMode.NONE,
            agents=sample_agents,
            initial_prompt="Shared topic",
            per_agent_prompts={"Alice": "Alice secret context"},
        )

        # Agent a (Alice) should see both shared + her private context
        directive_a = service.get_next_turn(session.session_id)
        assert directive_a.agent_id == "a"
        visible_a = [m.content for m in directive_a.visible_messages]
        assert any("Shared topic" in c for c in visible_a)
        assert any("Alice secret context" in c for c in visible_a)

        # Simulate advancing to agent b's turn
        session.advance_turn()
        directive_b = service.get_next_turn(session.session_id)
        assert directive_b.agent_id == "b"
        visible_b = [m.content for m in directive_b.visible_messages]
        assert any("Shared topic" in c for c in visible_b)
        assert not any("Alice secret context" in c for c in visible_b)


class TestFilterVisibleMessages:
    """Tests for _filter_visible_messages()."""

    def test_public_messages_visible_to_all(self, service: CoordinationService) -> None:
        from agent_memory.domain.coordination import ChannelMessage

        messages = [
            ChannelMessage(
                message_id="m1",
                channel_id="public",
                sender_id="a",
                content="Public message",
                turn_number=1,
                visible_to=frozenset(),  # Empty = public
            ),
        ]

        # Agent b should see agent a's public message
        visible = service._filter_visible_messages(messages, "b")
        assert len(visible) == 1

    def test_private_messages_filtered(self, service: CoordinationService) -> None:
        from agent_memory.domain.coordination import ChannelMessage

        messages = [
            ChannelMessage(
                message_id="m1",
                channel_id="whisper",
                sender_id="a",
                content="Private to b",
                turn_number=1,
                visible_to=frozenset(["a", "b"]),
            ),
        ]

        # Agent c should NOT see message visible only to a and b
        visible_c = service._filter_visible_messages(messages, "c")
        assert len(visible_c) == 0

        # Agent b SHOULD see it
        visible_b = service._filter_visible_messages(messages, "b")
        assert len(visible_b) == 1


class TestBuildSystemInstruction:
    """Tests for _build_system_instruction()."""

    def test_custom_system_prompt_used(self, service: CoordinationService) -> None:
        agent = AgentRole(
            agent_id="a",
            display_name="Alice",
            role="critic",
            system_prompt="You are a harsh critic. Challenge everything.",
        )

        instruction = service._build_system_instruction(agent, DebateFormat.FREE_FORM)

        assert "You are a harsh critic" in instruction

    def test_default_prompt_generated(self, service: CoordinationService) -> None:
        agent = AgentRole(
            agent_id="a",
            display_name="Alice",
            role="participant",
        )

        instruction = service._build_system_instruction(agent, DebateFormat.FREE_FORM)

        assert "Alice" in instruction
        assert "participant" in instruction

    def test_debate_format_adds_instructions(self, service: CoordinationService) -> None:
        agent = AgentRole(agent_id="a", display_name="Alice", role="participant")

        structured = service._build_system_instruction(agent, DebateFormat.STRUCTURED)
        assert "evidence" in structured.lower()

        socratic = service._build_system_instruction(agent, DebateFormat.SOCRATIC)
        assert "question" in socratic.lower()


class TestTallyVotes:
    """Tests for tally_votes()."""

    def test_tally_simple_majority(self, service: CoordinationService) -> None:
        votes = [
            Vote("v1", "s1", "a", "Proceed?", "yes"),
            Vote("v2", "s1", "b", "Proceed?", "yes"),
            Vote("v3", "s1", "c", "Proceed?", "no"),
        ]

        tally = service.tally_votes("s1", votes)

        assert tally.total_votes == 3
        assert tally.results == {"yes": 2, "no": 1}
        assert tally.winner == "yes"
        assert not tally.tied

    def test_tally_tie(self, service: CoordinationService) -> None:
        votes = [
            Vote("v1", "s1", "a", "Choose", "A"),
            Vote("v2", "s1", "b", "Choose", "B"),
        ]

        tally = service.tally_votes("s1", votes)

        assert tally.total_votes == 2
        assert tally.results == {"A": 1, "B": 1}
        assert tally.winner is None
        assert tally.tied

    def test_tally_empty_votes(self, service: CoordinationService) -> None:
        tally = service.tally_votes("s1", [])

        assert tally.total_votes == 0
        assert tally.results == {}


class TestStreamCleanedDelta:
    """Tests for execute_turn_stream cleaned delta protocol."""

    def test_stream_delta_cleaned_is_replacement(self) -> None:
        """StreamDelta with finish_reason='cleaned' should be treated as replacement text."""
        from agent_memory.domain.value_objects import StreamDelta

        # Simulate the stream protocol: raw deltas followed by cleaned
        raw_deltas = [
            StreamDelta(text="Hello", token_count=1, finish_reason=None),
            StreamDelta(text="Hello world", token_count=2, finish_reason=None),
            StreamDelta(text="Hello world! Extra junk Bob: ...", token_count=5, finish_reason="stop"),
        ]
        cleaned_delta = StreamDelta(
            text="Hello world!",  # Cleaned version (junk stripped)
            token_count=2,
            finish_reason="cleaned",
        )

        # Consumer should use cleaned delta text as authoritative, not concatenate
        all_deltas = raw_deltas + [cleaned_delta]
        final_delta = [d for d in all_deltas if d.finish_reason == "cleaned"]

        assert len(final_delta) == 1
        assert final_delta[0].text == "Hello world!"

        # Verify it's different from raw accumulated text (the whole point)
        raw_final = [d for d in all_deltas if d.finish_reason == "stop"]
        assert raw_final[0].text != final_delta[0].text

    def test_stream_no_duplicate_text(self) -> None:
        """Summing only non-cleaned deltas gives raw text; cleaned is separate."""
        from agent_memory.domain.value_objects import StreamDelta

        deltas = [
            StreamDelta(text="A", token_count=1, finish_reason=None),
            StreamDelta(text="AB", token_count=2, finish_reason=None),
            StreamDelta(text="ABC", token_count=3, finish_reason="stop"),
            StreamDelta(text="AB", token_count=2, finish_reason="cleaned"),  # Cleaned = trimmed
        ]

        # Accumulate raw tokens (skip cleaned)
        raw_text = ""
        clean_text = ""
        for d in deltas:
            if d.finish_reason == "cleaned":
                clean_text = d.text
            else:
                raw_text = d.text  # Each delta has full accumulated text

        # Raw and clean are intentionally different
        assert raw_text == "ABC"
        assert clean_text == "AB"
        # No duplication: clean_text is not appended to raw_text
        assert clean_text != raw_text


class TestFormatMessagesAsText:
    """Tests for _format_messages_as_text()."""

    def test_format_mixed_roles(self, service: CoordinationService) -> None:
        messages = [
            {"role": "system", "content": "You are an agent."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]

        text = service._format_messages_as_text(messages)

        assert "System: You are an agent." in text
        assert "User: Hello" in text
        assert "Assistant: Hi there" in text


# ── _clean_agent_response() ─────────────────────────────────────────


class TestCleanAgentResponse:
    """Tests for the static _clean_agent_response() method."""

    def test_plain_text_unchanged(self) -> None:
        result = CoordinationService._clean_agent_response("Hello world")
        assert result == "Hello world"

    def test_empty_input(self) -> None:
        assert CoordinationService._clean_agent_response("") == ""

    def test_whitespace_only(self) -> None:
        assert CoordinationService._clean_agent_response("   \n\n  ") == ""

    # GPT-OSS channel extraction
    def test_gpt_oss_final_channel(self) -> None:
        text = (
            "<|channel|>reasoning<|message|>thinking..."
            "<|channel|>final<|message|>The answer is 42.<|end|>"
        )
        result = CoordinationService._clean_agent_response(text)
        assert result == "The answer is 42."

    def test_gpt_oss_final_no_end_marker(self) -> None:
        text = "<|channel|>final<|message|>Just the answer"
        result = CoordinationService._clean_agent_response(text)
        assert result == "Just the answer"

    def test_gpt_oss_multiple_final_uses_last(self) -> None:
        text = (
            "<|channel|>final<|message|>first<|end|>"
            "<|channel|>final<|message|>second<|end|>"
        )
        result = CoordinationService._clean_agent_response(text)
        assert result == "second"

    # Special token stripping
    def test_strip_special_tokens(self) -> None:
        result = CoordinationService._clean_agent_response(
            "<|endoftext|>Hello<|pad|>"
        )
        assert result == "Hello"

    def test_strip_bare_assistant_marker(self) -> None:
        result = CoordinationService._clean_agent_response("assistant\nHello world")
        assert result == "Hello world"

    def test_strip_assistant_before_capital(self) -> None:
        result = CoordinationService._clean_agent_response("assistantHello there")
        assert result == "Hello there"

    # Echoed name/role prefix
    def test_strip_sender_name_prefix(self) -> None:
        result = CoordinationService._clean_agent_response(
            "Alice: Hello there", sender_name="Alice"
        )
        assert result == "Hello there"

    def test_strip_sender_name_said_prefix(self) -> None:
        result = CoordinationService._clean_agent_response(
            "Alice said: I agree", sender_name="Alice"
        )
        assert result == "I agree"

    def test_strip_assistant_role_prefix(self) -> None:
        result = CoordinationService._clean_agent_response("Assistant: I think so")
        assert result == "I think so"

    def test_strip_user_role_prefix(self) -> None:
        result = CoordinationService._clean_agent_response("User: echoed prompt")
        assert result == "echoed prompt"

    def test_strip_all_agent_name_prefix(self) -> None:
        result = CoordinationService._clean_agent_response(
            "Bob: I think yes", all_agent_names=["Alice", "Bob"]
        )
        assert result == "I think yes"

    # Turn cue stripping
    def test_strip_respond_now_cue(self) -> None:
        result = CoordinationService._clean_agent_response(
            "[Danny, respond now.] I think so"
        )
        assert result == "I think so"

    def test_strip_what_do_you_say_cue(self) -> None:
        result = CoordinationService._clean_agent_response(
            "Danny, what do you say? I agree"
        )
        assert result == "I agree"

    # Instruction fragment stripping
    def test_strip_instruction_echoes(self) -> None:
        result = CoordinationService._clean_agent_response(
            "Do not include other speakers' dialogue.\nActual response here"
        )
        assert result == "Actual response here"

    def test_strip_debate_format_echo(self) -> None:
        result = CoordinationService._clean_agent_response(
            "FREE-FORM DISCUSSION: I think we should..."
        )
        assert "I think we should" in result

    # Truncation at fake continuation
    def test_truncate_at_user_continuation(self) -> None:
        result = CoordinationService._clean_agent_response(
            "I agree with the proposal.\nUser: But what about costs?"
        )
        assert result == "I agree with the proposal."

    def test_truncate_at_system_continuation(self) -> None:
        result = CoordinationService._clean_agent_response(
            "Good point.\nSystem: new instruction"
        )
        assert result == "Good point."

    def test_truncate_at_agent_name(self) -> None:
        result = CoordinationService._clean_agent_response(
            "I think so.\nBob: I disagree.", all_agent_names=["Alice", "Bob"]
        )
        assert result == "I think so."

    def test_truncate_finds_earliest_marker(self) -> None:
        result = CoordinationService._clean_agent_response(
            "OK.\nBob: nope\nUser: more",
            all_agent_names=["Bob"],
        )
        assert result == "OK."

    def test_no_truncate_at_position_zero(self) -> None:
        """Markers at position 0 (no leading newline) are not stop markers."""
        result = CoordinationService._clean_agent_response(
            "User: this is the actual content"
        )
        assert result == "this is the actual content"

    def test_truncate_at_gemma_markers(self) -> None:
        result = CoordinationService._clean_agent_response(
            "My answer.\n<start_of_turn>user\nNew question"
        )
        assert result == "My answer."

    # Whitespace cleanup
    def test_collapse_multiple_newlines(self) -> None:
        result = CoordinationService._clean_agent_response(
            "Line one.\n\n\n\nLine two."
        )
        assert result == "Line one.\n\nLine two."

    # Combined artifacts
    def test_combined_prefix_and_continuation(self) -> None:
        result = CoordinationService._clean_agent_response(
            "Alice: I agree with the plan.\nBob: What about costs?",
            sender_name="Alice",
            all_agent_names=["Alice", "Bob"],
        )
        assert result == "I agree with the plan."

    def test_combined_special_tokens_and_name(self) -> None:
        result = CoordinationService._clean_agent_response(
            "<|endoftext|>Alice: The answer is yes.",
            sender_name="Alice",
        )
        assert result == "The answer is yes."


# ── _merge_consecutive_messages() ───────────────────────────────────


class TestMergeConsecutiveMessages:
    """Tests for _merge_consecutive_messages()."""

    def test_empty_list(self, service: CoordinationService) -> None:
        assert service._merge_consecutive_messages([]) == []

    def test_single_message(self, service: CoordinationService) -> None:
        msgs = [{"role": "user", "content": "Hello"}]
        result = service._merge_consecutive_messages(msgs)
        assert len(result) == 1
        assert result[0]["content"] == "Hello"

    def test_alternating_roles_unchanged(self, service: CoordinationService) -> None:
        msgs = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
            {"role": "user", "content": "Bye"},
        ]
        result = service._merge_consecutive_messages(msgs)
        assert len(result) == 3

    def test_consecutive_user_merged(self, service: CoordinationService) -> None:
        msgs = [
            {"role": "user", "content": "Alice: Hi"},
            {"role": "user", "content": "Bob: Hello"},
            {"role": "user", "content": "[Carol, respond now.]"},
        ]
        result = service._merge_consecutive_messages(msgs)
        assert len(result) == 1
        assert "Alice: Hi\nBob: Hello\n[Carol, respond now.]" == result[0]["content"]

    def test_consecutive_assistant_merged(self, service: CoordinationService) -> None:
        msgs = [
            {"role": "assistant", "content": "Part 1"},
            {"role": "assistant", "content": "Part 2"},
        ]
        result = service._merge_consecutive_messages(msgs)
        assert len(result) == 1
        assert result[0]["content"] == "Part 1\nPart 2"

    def test_system_never_merged(self, service: CoordinationService) -> None:
        msgs = [
            {"role": "system", "content": "Rule 1"},
            {"role": "system", "content": "Rule 2"},
        ]
        result = service._merge_consecutive_messages(msgs)
        assert len(result) == 2

    def test_does_not_mutate_input(self, service: CoordinationService) -> None:
        msgs = [
            {"role": "user", "content": "A"},
            {"role": "user", "content": "B"},
        ]
        original_first = msgs[0]["content"]
        service._merge_consecutive_messages(msgs)
        assert msgs[0]["content"] == original_first  # not modified


# ── _resolve_cache_key() ────────────────────────────────────────────


class TestResolveCacheKey:
    """Tests for _resolve_cache_key()."""

    async def test_no_prefix_returns_session_key(
        self, service: CoordinationService, sample_agents
    ) -> None:
        session = await service.create_session(
            topology=Topology.TURN_BY_TURN,
            debate_format=DebateFormat.FREE_FORM,
            decision_mode=DecisionMode.NONE,
            agents=sample_agents,
        )
        key = service._resolve_cache_key(session.session_id, "a")
        assert key == f"coord_{session.session_id}_a"

    async def test_persistent_prefix_with_permanent_agent(
        self, service: CoordinationService,
    ) -> None:
        agents = [
            AgentRole(
                agent_id="warden",
                display_name="Warden",
                role="moderator",
                lifecycle=AgentLifecycle.PERMANENT,
            ),
        ]
        session = await service.create_session(
            topology=Topology.TURN_BY_TURN,
            debate_format=DebateFormat.FREE_FORM,
            decision_mode=DecisionMode.NONE,
            agents=agents,
            persistent_cache_prefix="pd_scenario",
        )
        key = service._resolve_cache_key(session.session_id, "warden")
        assert key == "persist_pd_scenario_warden"

    async def test_persistent_prefix_with_ephemeral_agent(
        self, service: CoordinationService, sample_agents
    ) -> None:
        """Ephemeral agents ignore persistent_cache_prefix."""
        session = await service.create_session(
            topology=Topology.TURN_BY_TURN,
            debate_format=DebateFormat.FREE_FORM,
            decision_mode=DecisionMode.NONE,
            agents=sample_agents,
            persistent_cache_prefix="pd_scenario",
        )
        key = service._resolve_cache_key(session.session_id, "a")
        assert key.startswith("coord_")
        assert "persist_" not in key


# ── delete_persistent_caches() ──────────────────────────────────────


class TestDeletePersistentCaches:
    """Tests for delete_persistent_caches()."""

    def test_deletes_matching_prefix(self, service: CoordinationService, mock_cache_store) -> None:
        mock_cache_store.list_all_agents.return_value = [
            {"agent_id": "persist_pd_game_warden"},
            {"agent_id": "persist_pd_game_prisoner1"},
            {"agent_id": "coord_session123_other"},
        ]
        count = service.delete_persistent_caches("pd_game")
        assert count == 2
        assert mock_cache_store.delete.call_count == 2

    def test_no_match_returns_zero(self, service: CoordinationService, mock_cache_store) -> None:
        mock_cache_store.list_all_agents.return_value = [
            {"agent_id": "coord_session123_other"},
        ]
        count = service.delete_persistent_caches("pd_game")
        assert count == 0
        mock_cache_store.delete.assert_not_called()

    def test_empty_agent_list(self, service: CoordinationService, mock_cache_store) -> None:
        mock_cache_store.list_all_agents.return_value = []
        count = service.delete_persistent_caches("anything")
        assert count == 0


# ── add_whisper() ───────────────────────────────────────────────────


class TestAddWhisper:
    """Tests for add_whisper()."""

    async def test_creates_whisper_channel(
        self, service: CoordinationService, sample_agents
    ) -> None:
        session = await service.create_session(
            topology=Topology.TURN_BY_TURN,
            debate_format=DebateFormat.FREE_FORM,
            decision_mode=DecisionMode.NONE,
            agents=sample_agents,
        )

        msg = service.add_whisper(session.session_id, "a", "b", "psst")

        assert msg.content == "psst"
        assert msg.sender_id == "a"
        assert msg.visible_to == frozenset(["a", "b"])

    async def test_reuses_existing_channel(
        self, service: CoordinationService, sample_agents
    ) -> None:
        session = await service.create_session(
            topology=Topology.TURN_BY_TURN,
            debate_format=DebateFormat.FREE_FORM,
            decision_mode=DecisionMode.NONE,
            agents=sample_agents,
        )

        service.add_whisper(session.session_id, "a", "b", "first")
        service.add_whisper(session.session_id, "b", "a", "reply")

        # Same channel (a < b alphabetically → "a_b")
        whisper_channels = [
            c for c in session.channels.values() if c.channel_type == "whisper"
        ]
        assert len(whisper_channels) == 1
        assert len(whisper_channels[0].messages) == 2

    async def test_whisper_visibility_correct(
        self, service: CoordinationService, sample_agents
    ) -> None:
        session = await service.create_session(
            topology=Topology.TURN_BY_TURN,
            debate_format=DebateFormat.FREE_FORM,
            decision_mode=DecisionMode.NONE,
            agents=sample_agents,
        )

        msg = service.add_whisper(session.session_id, "a", "c", "secret")
        assert "a" in msg.visible_to
        assert "c" in msg.visible_to
        assert "b" not in msg.visible_to

    async def test_whisper_nonexistent_session_raises(
        self, service: CoordinationService
    ) -> None:
        with pytest.raises(SessionNotFoundError):
            service.add_whisper("nonexistent", "a", "b", "hello")


# ── _get_generation_max_tokens() ────────────────────────────────────


class TestGetGenerationMaxTokens:

    def test_default_returns_200(self, mock_scheduler, mock_cache_store, mock_engine) -> None:
        svc = CoordinationService(
            scheduler=mock_scheduler,
            cache_store=mock_cache_store,
            engine=mock_engine,
            reasoning_extra_tokens=0,
        )
        assert svc._get_generation_max_tokens() == 200

    def test_with_extra_tokens(self, mock_scheduler, mock_cache_store, mock_engine) -> None:
        svc = CoordinationService(
            scheduler=mock_scheduler,
            cache_store=mock_cache_store,
            engine=mock_engine,
            reasoning_extra_tokens=100,
        )
        assert svc._get_generation_max_tokens() == 300


# ── DeepSeek-specific paths ─────────────────────────────────────────


class TestDeepSeekPaths:
    """Tests for DeepSeek-specific behavior in system instruction and prompts."""

    @pytest.fixture
    def deepseek_service(self, mock_scheduler, mock_cache_store, mock_engine):
        """Service with a chat_template that detects DeepSeek."""
        from agent_memory.adapters.outbound.chat_template_adapter import ChatTemplateAdapter

        adapter = ChatTemplateAdapter()
        # Mock tokenizer to have a DeepSeek chat template
        mock_engine.tokenizer.chat_template = (
            "template with 'User: ' and 'Assistant: ' labels"
        )
        return CoordinationService(
            scheduler=mock_scheduler,
            cache_store=mock_cache_store,
            engine=mock_engine,
            chat_template=adapter,
        )

    def test_build_system_instruction_deepseek(self, deepseek_service) -> None:
        agent = AgentRole(agent_id="a", display_name="Alice", role="participant")
        instruction = deepseek_service._build_system_instruction(agent, DebateFormat.FREE_FORM)
        assert "English only" in instruction
        assert "Chinese" in instruction
        assert "Alice" in instruction
        # Should NOT have standard "RULES:" prefix
        assert "RULES:" not in instruction

    def test_build_system_instruction_non_deepseek(self, service) -> None:
        agent = AgentRole(agent_id="a", display_name="Alice", role="participant")
        instruction = service._build_system_instruction(agent, DebateFormat.FREE_FORM)
        assert "RULES:" in instruction
        assert "English" not in instruction

    async def test_build_prompt_deepseek_first_turn_has_primer(
        self, deepseek_service, sample_agents
    ) -> None:
        session = await deepseek_service.create_session(
            topology=Topology.TURN_BY_TURN,
            debate_format=DebateFormat.FREE_FORM,
            decision_mode=DecisionMode.NONE,
            agents=sample_agents,
        )
        directive = deepseek_service.get_next_turn(session.session_id)
        agent_role = session.agents[directive.agent_id]
        messages = deepseek_service.build_agent_prompt(directive, agent_role)

        # Should have identity primer as assistant message
        assistant_msgs = [m for m in messages if m["role"] == "assistant"]
        assert len(assistant_msgs) >= 1
        assert "I'm Alice" in assistant_msgs[0]["content"]

        # Should NOT have "[Name, respond now.]" cue
        user_msgs = [m for m in messages if m["role"] == "user"]
        assert not any("respond now" in m["content"] for m in user_msgs)

    async def test_build_prompt_deepseek_subsequent_turn_no_primer(
        self, deepseek_service, sample_agents
    ) -> None:
        session = await deepseek_service.create_session(
            topology=Topology.TURN_BY_TURN,
            debate_format=DebateFormat.FREE_FORM,
            decision_mode=DecisionMode.NONE,
            agents=sample_agents,
            initial_prompt="Topic",
        )
        # Add a prior message from Alice
        public_channel = next(
            c for c in session.channels.values() if c.channel_type == "public"
        )
        public_channel.add_message(sender_id="a", content="I think so", turn_number=1)

        directive = deepseek_service.get_next_turn(session.session_id)
        agent_role = session.agents["a"]
        messages = deepseek_service.build_agent_prompt(directive, agent_role)

        # Agent already has prior messages → no identity primer
        assistant_msgs = [m for m in messages if m["role"] == "assistant"]
        primer_msgs = [m for m in assistant_msgs if "I'm Alice" in m["content"]]
        assert len(primer_msgs) == 0

    async def test_build_prompt_standard_has_respond_cue(
        self, service, sample_agents
    ) -> None:
        session = await service.create_session(
            topology=Topology.TURN_BY_TURN,
            debate_format=DebateFormat.FREE_FORM,
            decision_mode=DecisionMode.NONE,
            agents=sample_agents,
        )
        directive = service.get_next_turn(session.session_id)
        agent_role = session.agents[directive.agent_id]
        messages = service.build_agent_prompt(directive, agent_role)

        # Standard models get "[Name, respond now.]"
        user_msgs = [m for m in messages if m["role"] == "user"]
        assert any("respond now" in m["content"] for m in user_msgs)


# ── update_engine() ─────────────────────────────────────────────────


class TestUpdateEngine:

    def test_updates_engine_reference(
        self, mock_cache_store, mock_engine
    ) -> None:
        scheduler = MagicMock()  # Sync mock — update_engine is not async
        svc = CoordinationService(
            scheduler=scheduler,
            cache_store=mock_cache_store,
            engine=mock_engine,
        )
        new_engine = MagicMock()
        svc.update_engine(new_engine)
        assert svc._engine is new_engine
        scheduler.update_engine.assert_called_once_with(new_engine)

    def test_updates_engine_no_scheduler(
        self, mock_cache_store, mock_engine
    ) -> None:
        svc = CoordinationService(
            scheduler=None,
            cache_store=mock_cache_store,
            engine=mock_engine,
        )
        new_engine = MagicMock()
        svc.update_engine(new_engine)
        assert svc._engine is new_engine


# ── list_sessions() ─────────────────────────────────────────────────


class TestListSessions:

    async def test_empty_initially(self, service: CoordinationService) -> None:
        assert service.list_sessions() == []

    async def test_returns_all_sessions(
        self, service: CoordinationService, sample_agents
    ) -> None:
        await service.create_session(
            topology=Topology.TURN_BY_TURN,
            debate_format=DebateFormat.FREE_FORM,
            decision_mode=DecisionMode.NONE,
            agents=sample_agents,
        )
        await service.create_session(
            topology=Topology.TURN_BY_TURN,
            debate_format=DebateFormat.FREE_FORM,
            decision_mode=DecisionMode.NONE,
            agents=sample_agents,
        )
        assert len(service.list_sessions()) == 2


# ── Debate style hints ──────────────────────────────────────────────


class TestDebateStyleHint:

    def test_devils_advocate_critic(self) -> None:
        hint = CoordinationService._debate_style_hint(
            DebateFormat.DEVILS_ADVOCATE, "critic"
        )
        assert "challenge" in hint.lower() or "weakness" in hint.lower()

    def test_devils_advocate_non_critic(self) -> None:
        hint = CoordinationService._debate_style_hint(
            DebateFormat.DEVILS_ADVOCATE, "participant"
        )
        assert hint == ""

    def test_parliamentary(self) -> None:
        hint = CoordinationService._debate_style_hint(
            DebateFormat.PARLIAMENTARY, "participant"
        )
        assert "formal" in hint.lower()

    def test_free_form_returns_empty(self) -> None:
        hint = CoordinationService._debate_style_hint(
            DebateFormat.FREE_FORM, "participant"
        )
        assert hint == ""


# ── _tokenize_chat_messages() ───────────────────────────────────────


class TestTokenizeChatMessages:
    """Tests for _tokenize_chat_messages() on the coordination service."""

    def test_no_template_uses_fallback(self, service: CoordinationService) -> None:
        """Without chat template, falls back to raw encode."""
        service._engine.tokenizer.chat_template = None
        messages = [
            {"role": "system", "content": "Rules"},
            {"role": "user", "content": "Hello"},
        ]
        tokens, text = service._tokenize_chat_messages(messages)
        service._engine.tokenizer.encode.assert_called()

    def test_with_template_returns_tokens_and_text(
        self, mock_scheduler, mock_cache_store, mock_engine
    ) -> None:
        from agent_memory.adapters.outbound.chat_template_adapter import ChatTemplateAdapter

        adapter = ChatTemplateAdapter()
        mock_engine.tokenizer.chat_template = "simple {{message}} template"

        def apply_side_effect(msgs, **kwargs):
            if kwargs.get("tokenize", True):
                return [10, 20, 30]
            else:
                return "formatted text"

        mock_engine.tokenizer.apply_chat_template.side_effect = apply_side_effect

        svc = CoordinationService(
            scheduler=mock_scheduler,
            cache_store=mock_cache_store,
            engine=mock_engine,
            chat_template=adapter,
        )
        tokens, text = svc._tokenize_chat_messages(
            [{"role": "user", "content": "Hi"}]
        )
        assert tokens == [10, 20, 30]
        assert text == "formatted text"

    def test_generation_prefix_injected(
        self, mock_scheduler, mock_cache_store, mock_engine
    ) -> None:
        from agent_memory.adapters.outbound.chat_template_adapter import ChatTemplateAdapter

        adapter = ChatTemplateAdapter()
        mock_engine.tokenizer.chat_template = "simple template"

        def apply_side_effect(msgs, **kwargs):
            if kwargs.get("tokenize", True):
                return [10, 20]
            else:
                return "text ending with Assistant:"

        mock_engine.tokenizer.apply_chat_template.side_effect = apply_side_effect
        mock_engine.tokenizer.encode.return_value = [99, 98]

        svc = CoordinationService(
            scheduler=mock_scheduler,
            cache_store=mock_cache_store,
            engine=mock_engine,
            chat_template=adapter,
        )
        tokens, text = svc._tokenize_chat_messages(
            [{"role": "user", "content": "Hi"}],
            generation_prefix="Warden:",
        )
        # Suffix tokens appended
        assert tokens == [10, 20, 99, 98]
        assert "Warden:" in text

    def test_template_exception_falls_back(
        self, mock_scheduler, mock_cache_store, mock_engine
    ) -> None:
        from agent_memory.adapters.outbound.chat_template_adapter import ChatTemplateAdapter

        adapter = ChatTemplateAdapter()
        mock_engine.tokenizer.chat_template = "valid template"
        mock_engine.tokenizer.apply_chat_template.side_effect = RuntimeError("boom")
        mock_engine.tokenizer.encode.return_value = [1, 2]

        svc = CoordinationService(
            scheduler=mock_scheduler,
            cache_store=mock_cache_store,
            engine=mock_engine,
            chat_template=adapter,
        )
        tokens, text = svc._tokenize_chat_messages(
            [{"role": "user", "content": "Hello"}]
        )
        assert tokens == [1, 2]


# ── execute_turn_stream() ───────────────────────────────────────────


class TestExecuteTurnStream:
    """Tests for the streaming execute_turn_stream path."""

    async def test_stream_yields_deltas(
        self, mock_cache_store, mock_engine
    ) -> None:
        from agent_memory.domain.value_objects import StreamDelta

        mock_engine.tokenizer.chat_template = None
        mock_engine.get_agent_blocks.return_value = MagicMock(total_tokens=50)

        # Use MagicMock for scheduler so we can set submit_and_stream to
        # an async generator function (AsyncMock returns coroutines, not iterables)
        scheduler = MagicMock()

        async def mock_stream(*args, **kwargs):
            yield StreamDelta(text="Hello", token_count=1, finish_reason=None)
            yield StreamDelta(text="Hello world", token_count=2, finish_reason="stop")

        scheduler.submit_and_stream = mock_stream

        svc = CoordinationService(
            scheduler=scheduler,
            cache_store=mock_cache_store,
            engine=mock_engine,
        )
        agents = [
            AgentRole(agent_id="a", display_name="Alice", role="participant"),
        ]
        session = await svc.create_session(
            topology=Topology.TURN_BY_TURN,
            debate_format=DebateFormat.FREE_FORM,
            decision_mode=DecisionMode.NONE,
            agents=agents,
            initial_prompt="Topic",
        )

        deltas = []
        async for delta in svc.execute_turn_stream(session.session_id):
            deltas.append(delta)

        # Should have raw deltas + cleaned delta
        assert len(deltas) >= 2
        # Last delta should have finish_reason "cleaned"
        assert deltas[-1].finish_reason == "cleaned"
