"""Unit tests for coordination domain entities.

Tests CoordinationSession, Channel, ChannelMessage, AgentRole, TurnDirective,
and related value objects. All tests verify domain logic without external
dependencies.
"""

import pytest

from agent_memory.domain.coordination import (
    AgentRole,
    Channel,
    ChannelMessage,
    CoordinationSession,
    DebateFormat,
    DecisionMode,
    Topology,
    TurnDirective,
    Vote,
    VoteTally,
)

pytestmark = pytest.mark.unit


class TestAgentRole:
    """AgentRole value object tests."""

    def test_create_with_all_fields(self) -> None:
        role = AgentRole(
            agent_id="agent_1",
            display_name="Alice",
            role="moderator",
            system_prompt="You are a moderator.",
        )
        assert role.agent_id == "agent_1"
        assert role.display_name == "Alice"
        assert role.role == "moderator"
        assert role.system_prompt == "You are a moderator."

    def test_create_with_defaults(self) -> None:
        role = AgentRole(
            agent_id="agent_2",
            display_name="Bob",
        )
        assert role.role == "participant"
        assert role.system_prompt == ""

    def test_immutable(self) -> None:
        """AgentRole is frozen, cannot be modified."""
        role = AgentRole(agent_id="a", display_name="Test")
        with pytest.raises(AttributeError):
            role.display_name = "Modified"  # type: ignore


class TestChannelMessage:
    """ChannelMessage value object tests."""

    def test_create_public_message(self) -> None:
        """Public message has empty visible_to set."""
        msg = ChannelMessage(
            message_id="msg_1",
            channel_id="public",
            sender_id="agent_1",
            content="Hello everyone",
            turn_number=1,
        )
        assert msg.visible_to == frozenset()
        assert not msg.is_interrupted

    def test_create_private_message(self) -> None:
        """Private message has specific visible_to set."""
        msg = ChannelMessage(
            message_id="msg_2",
            channel_id="whisper_a_b",
            sender_id="agent_a",
            content="Secret message",
            turn_number=2,
            visible_to=frozenset(["agent_a", "agent_b"]),
        )
        assert msg.visible_to == frozenset(["agent_a", "agent_b"])

    def test_interrupted_message(self) -> None:
        msg = ChannelMessage(
            message_id="msg_3",
            channel_id="public",
            sender_id="agent_1",
            content="I was saying...",
            turn_number=3,
            is_interrupted=True,
        )
        assert msg.is_interrupted


class TestChannel:
    """Channel entity tests."""

    def test_create_public_channel(self) -> None:
        channel = Channel(
            channel_id="public",
            channel_type="public",
            participant_ids=frozenset(["a", "b", "c"]),
        )
        assert channel.channel_id == "public"
        assert channel.channel_type == "public"
        assert len(channel.participant_ids) == 3
        assert channel.messages == []

    def test_add_message(self) -> None:
        channel = Channel(
            channel_id="public",
            channel_type="public",
            participant_ids=frozenset(["a", "b"]),
        )

        msg = channel.add_message(
            sender_id="a",
            content="Hello",
            turn_number=1,
        )

        assert len(channel.messages) == 1
        assert msg.sender_id == "a"
        assert msg.content == "Hello"
        assert msg.channel_id == "public"
        assert msg.visible_to == frozenset()  # Default = public

    def test_add_private_message(self) -> None:
        channel = Channel(
            channel_id="whisper",
            channel_type="whisper",
            participant_ids=frozenset(["a", "b"]),
        )

        msg = channel.add_message(
            sender_id="a",
            content="Secret",
            turn_number=1,
            visible_to=frozenset(["a", "b"]),
        )

        assert msg.visible_to == frozenset(["a", "b"])

    def test_multiple_messages(self) -> None:
        channel = Channel(
            channel_id="public",
            channel_type="public",
            participant_ids=frozenset(["a", "b"]),
        )

        channel.add_message("a", "First", 1)
        channel.add_message("b", "Second", 2)
        channel.add_message("a", "Third", 3)

        assert len(channel.messages) == 3
        assert channel.messages[0].content == "First"
        assert channel.messages[1].content == "Second"
        assert channel.messages[2].content == "Third"


class TestCoordinationSession:
    """CoordinationSession entity tests."""

    @pytest.fixture
    def agents(self) -> list[AgentRole]:
        return [
            AgentRole(agent_id="a", display_name="Alice", role="participant"),
            AgentRole(agent_id="b", display_name="Bob", role="participant"),
            AgentRole(agent_id="c", display_name="Charlie", role="moderator"),
        ]

    @pytest.fixture
    def public_channel(self, agents: list[AgentRole]) -> Channel:
        return Channel(
            channel_id="public",
            channel_type="public",
            participant_ids=frozenset(a.agent_id for a in agents),
        )

    def test_create_session(self, agents: list[AgentRole], public_channel: Channel) -> None:
        session = CoordinationSession(
            session_id="session_1",
            topology=Topology.TURN_BY_TURN,
            debate_format=DebateFormat.FREE_FORM,
            decision_mode=DecisionMode.NONE,
            agents={a.agent_id: a for a in agents},
            channels={"public": public_channel},
            turn_order=[a.agent_id for a in agents],
        )

        assert session.session_id == "session_1"
        assert session.topology == Topology.TURN_BY_TURN
        assert session.current_turn == 0
        assert session.is_active
        assert len(session.agents) == 3
        assert len(session.turn_order) == 3

    def test_get_next_speaker_turn_by_turn(
        self, agents: list[AgentRole], public_channel: Channel
    ) -> None:
        session = CoordinationSession(
            session_id="s1",
            topology=Topology.TURN_BY_TURN,
            debate_format=DebateFormat.FREE_FORM,
            decision_mode=DecisionMode.NONE,
            agents={a.agent_id: a for a in agents},
            channels={"public": public_channel},
            turn_order=["a", "b", "c"],
        )

        # Turn 0 -> a
        assert session.get_next_speaker() == "a"

        # Advance turn
        session.current_turn = 1
        assert session.get_next_speaker() == "b"

        # Turn 2 -> c
        session.current_turn = 2
        assert session.get_next_speaker() == "c"

        # Turn 3 -> cycle back to a
        session.current_turn = 3
        assert session.get_next_speaker() == "a"

    def test_get_next_speaker_round_robin(
        self, agents: list[AgentRole], public_channel: Channel
    ) -> None:
        """Round robin is same as turn_by_turn (cycles through turn_order)."""
        session = CoordinationSession(
            session_id="s1",
            topology=Topology.ROUND_ROBIN,
            debate_format=DebateFormat.FREE_FORM,
            decision_mode=DecisionMode.NONE,
            agents={a.agent_id: a for a in agents},
            channels={"public": public_channel},
            turn_order=["a", "b", "c"],
        )

        assert session.get_next_speaker() == "a"
        session.current_turn = 1
        assert session.get_next_speaker() == "b"

    def test_get_next_speaker_inactive(
        self, agents: list[AgentRole], public_channel: Channel
    ) -> None:
        session = CoordinationSession(
            session_id="s1",
            topology=Topology.TURN_BY_TURN,
            debate_format=DebateFormat.FREE_FORM,
            decision_mode=DecisionMode.NONE,
            agents={a.agent_id: a for a in agents},
            channels={"public": public_channel},
            turn_order=["a", "b", "c"],
            is_active=False,
        )

        assert session.get_next_speaker() is None

    def test_advance_turn(
        self, agents: list[AgentRole], public_channel: Channel
    ) -> None:
        session = CoordinationSession(
            session_id="s1",
            topology=Topology.TURN_BY_TURN,
            debate_format=DebateFormat.FREE_FORM,
            decision_mode=DecisionMode.NONE,
            agents={a.agent_id: a for a in agents},
            channels={"public": public_channel},
            turn_order=["a", "b", "c"],
            max_turns=0,  # Unlimited
        )

        assert session.current_turn == 0
        session.advance_turn()
        assert session.current_turn == 1
        assert session.is_active  # Still active (no max_turns)

    def test_advance_turn_reaches_max(
        self, agents: list[AgentRole], public_channel: Channel
    ) -> None:
        session = CoordinationSession(
            session_id="s1",
            topology=Topology.TURN_BY_TURN,
            debate_format=DebateFormat.FREE_FORM,
            decision_mode=DecisionMode.NONE,
            agents={a.agent_id: a for a in agents},
            channels={"public": public_channel},
            turn_order=["a", "b", "c"],
            max_turns=3,
        )

        assert session.is_active
        session.advance_turn()  # Turn 1
        assert session.is_active
        session.advance_turn()  # Turn 2
        assert session.is_active
        session.advance_turn()  # Turn 3 -> reaches max
        assert not session.is_active


class TestTurnDirective:
    """TurnDirective value object tests."""

    def test_create_directive(self) -> None:
        messages = [
            ChannelMessage(
                message_id="m1",
                channel_id="public",
                sender_id="system",
                content="Welcome",
                turn_number=0,
            ),
        ]

        directive = TurnDirective(
            session_id="s1",
            agent_id="agent_a",
            turn_number=1,
            visible_messages=messages,
            system_instruction="You are a participant.",
        )

        assert directive.session_id == "s1"
        assert directive.agent_id == "agent_a"
        assert directive.turn_number == 1
        assert len(directive.visible_messages) == 1
        assert directive.system_instruction == "You are a participant."


class TestVote:
    """Vote entity tests."""

    def test_create_vote(self) -> None:
        vote = Vote(
            vote_id="v1",
            session_id="s1",
            agent_id="a",
            question="Should we proceed?",
            choice="yes",
        )

        assert vote.vote_id == "v1"
        assert vote.agent_id == "a"
        assert vote.choice == "yes"
        assert vote.ranking == []

    def test_create_ranked_vote(self) -> None:
        vote = Vote(
            vote_id="v2",
            session_id="s1",
            agent_id="b",
            question="Rank options",
            choice="option_a",
            ranking=["option_a", "option_b", "option_c"],
        )

        assert vote.ranking == ["option_a", "option_b", "option_c"]


class TestVoteTally:
    """VoteTally entity tests."""

    def test_create_tally(self) -> None:
        tally = VoteTally(
            question="Proceed?",
            total_votes=3,
            results={"yes": 2, "no": 1},
            winner="yes",
            tied=False,
        )

        assert tally.question == "Proceed?"
        assert tally.total_votes == 3
        assert tally.winner == "yes"
        assert not tally.tied

    def test_tied_tally(self) -> None:
        tally = VoteTally(
            question="Choose",
            total_votes=4,
            results={"A": 2, "B": 2},
            winner=None,
            tied=True,
        )

        assert tally.tied
        assert tally.winner is None


class TestTopologyEnum:
    """Topology enum tests."""

    def test_enum_values(self) -> None:
        assert Topology.TURN_BY_TURN.value == "turn_by_turn"
        assert Topology.BROADCAST.value == "broadcast"
        assert Topology.WHISPER.value == "whisper"
        assert Topology.ROUND_ROBIN.value == "round_robin"
        assert Topology.INTERRUPT.value == "interrupt"

    def test_enum_from_string(self) -> None:
        assert Topology("turn_by_turn") == Topology.TURN_BY_TURN
        assert Topology("broadcast") == Topology.BROADCAST


class TestDebateFormatEnum:
    """DebateFormat enum tests."""

    def test_enum_values(self) -> None:
        assert DebateFormat.FREE_FORM.value == "free_form"
        assert DebateFormat.STRUCTURED.value == "structured"
        assert DebateFormat.SOCRATIC.value == "socratic"
        assert DebateFormat.DEVILS_ADVOCATE.value == "devils_advocate"
        assert DebateFormat.PARLIAMENTARY.value == "parliamentary"


class TestDecisionModeEnum:
    """DecisionMode enum tests."""

    def test_enum_values(self) -> None:
        assert DecisionMode.NONE.value == "none"
        assert DecisionMode.MAJORITY_VOTE.value == "majority_vote"
        assert DecisionMode.RANKED_CHOICE.value == "ranked_choice"
        assert DecisionMode.COORDINATOR_DECIDES.value == "coordinator_decides"
        assert DecisionMode.CONSENSUS.value == "consensus"
