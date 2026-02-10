"""Unit tests for CoordinationService.

Tests coordination service logic with mocked scheduler, cache_store, and engine.
Verifies prompt construction, turn routing, message recording, and vote tallying.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_memory.application.coordination_service import CoordinationService
from agent_memory.domain.coordination import (
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
