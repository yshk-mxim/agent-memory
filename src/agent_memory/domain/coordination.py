# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Yakov Shkolnikov and contributors
"""Domain entities for multi-agent coordination.

Coordination sessions enable multiple agents to participate in structured
conversations with various topologies (turn-by-turn, broadcast, whisper),
debate formats (free-form, structured, Socratic), and decision-making modes
(voting, consensus, coordinator).

All entities in this module have NO external dependencies - only Python
stdlib and typing imports.
"""

from dataclasses import dataclass, field
from enum import Enum
from uuid import uuid4


class Topology(Enum):
    """Communication topology for multi-agent coordination."""

    TURN_BY_TURN = "turn_by_turn"
    BROADCAST = "broadcast"
    WHISPER = "whisper"
    ROUND_ROBIN = "round_robin"
    INTERRUPT = "interrupt"


class DebateFormat(Enum):
    """Structured debate formats."""

    FREE_FORM = "free_form"
    STRUCTURED = "structured"
    SOCRATIC = "socratic"
    DEVILS_ADVOCATE = "devils_advocate"
    PARLIAMENTARY = "parliamentary"


class DecisionMode(Enum):
    """Decision-making modes for coordination sessions."""

    NONE = "none"
    MAJORITY_VOTE = "majority_vote"
    RANKED_CHOICE = "ranked_choice"
    COORDINATOR_DECIDES = "coordinator_decides"
    CONSENSUS = "consensus"


class AgentLifecycle(Enum):
    """Agent memory lifecycle strategy.

    EPHEMERAL: Session-only memory. Cache is deleted when session ends.
               Used for temporary agents or testing.

    PERMANENT: Persistent memory across sessions (future LTM support).
               For now, behaves same as ephemeral (STM only).
               Future: Will maintain long-term memory via summarization.
    """

    EPHEMERAL = "ephemeral"
    PERMANENT = "permanent"

    @classmethod
    def _missing_(cls, value: object) -> "AgentLifecycle":
        """Safe default for unknown values."""
        return cls.EPHEMERAL


@dataclass(frozen=True)
class AgentRole:
    """Role assignment for an agent in a coordination session.

    Attributes:
        agent_id: Unique identifier for this agent (matches AgentCacheStore IDs).
        display_name: Human-readable name shown in UI.
        role: Role type ("moderator", "critic", "advocate", "participant").
        system_prompt: Role-specific system prompt injected into context.
        lifecycle: Memory lifecycle strategy (ephemeral or permanent).
    """

    agent_id: str
    display_name: str
    role: str = "participant"
    system_prompt: str = ""
    lifecycle: AgentLifecycle = AgentLifecycle.EPHEMERAL


@dataclass(frozen=True)
class ChannelMessage:
    """Single message in a coordination channel.

    Attributes:
        message_id: Unique message identifier.
        channel_id: Channel this message belongs to.
        sender_id: Agent ID of the sender.
        content: Message text content.
        turn_number: Turn count when message was sent.
        visible_to: Set of agent IDs that can see this message (empty = public).
        is_interrupted: True if this message was cut off mid-generation.
        metadata: Optional metadata for debugging/analysis.
    """

    message_id: str
    channel_id: str
    sender_id: str
    content: str
    turn_number: int
    visible_to: frozenset[str] = field(default_factory=frozenset)
    is_interrupted: bool = False
    metadata: dict = field(default_factory=dict)


@dataclass
class Channel:
    """Communication channel within a coordination session.

    Attributes:
        channel_id: Unique channel identifier.
        channel_type: Channel type ("public", "whisper_{a}_{b}", etc.).
        participant_ids: Set of agent IDs that can access this channel.
        messages: Chronological list of messages in this channel.
    """

    channel_id: str
    channel_type: str
    participant_ids: frozenset[str]
    messages: list[ChannelMessage] = field(default_factory=list)

    def add_message(
        self,
        sender_id: str,
        content: str,
        turn_number: int,
        visible_to: frozenset[str] | None = None,
        is_interrupted: bool = False,
    ) -> ChannelMessage:
        """Add a new message to this channel.

        Args:
            sender_id: Agent ID of the sender.
            content: Message text.
            turn_number: Current turn count.
            visible_to: Specific agents that can see this (None = all participants).
            is_interrupted: Whether this message was interrupted mid-generation.

        Returns:
            The created ChannelMessage.
        """
        if visible_to is None:
            visible_to = frozenset()  # Empty = public (all can see)

        message = ChannelMessage(
            message_id=uuid4().hex[:12],
            channel_id=self.channel_id,
            sender_id=sender_id,
            content=content,
            turn_number=turn_number,
            visible_to=visible_to,
            is_interrupted=is_interrupted,
        )
        self.messages.append(message)
        return message


@dataclass
class CoordinationSession:
    """Multi-agent coordination session.

    Attributes:
        session_id: Unique session identifier.
        topology: Communication topology (turn_by_turn, broadcast, etc.).
        debate_format: Debate structure (free_form, structured, etc.).
        decision_mode: Decision-making mode (voting, consensus, etc.).
        agents: Map of agent_id to AgentRole.
        channels: Map of channel_id to Channel.
        current_turn: Current turn counter.
        turn_order: Ordered list of agent_ids for turn sequencing.
        is_active: Whether the session is active (can accept new messages).
        max_turns: Maximum turns before auto-termination (0 = unlimited).
    """

    session_id: str
    topology: Topology
    debate_format: DebateFormat
    decision_mode: DecisionMode
    agents: dict[str, AgentRole]
    channels: dict[str, Channel]
    current_turn: int = 0
    turn_order: list[str] = field(default_factory=list)
    is_active: bool = True
    max_turns: int = 0
    persistent_cache_prefix: str = ""

    def get_next_speaker(self) -> str | None:
        """Get the agent_id of the next speaker based on topology.

        Returns:
            agent_id of next speaker, or None if session is not active.
        """
        if not self.is_active or not self.turn_order:
            return None

        if self.topology in (Topology.TURN_BY_TURN, Topology.ROUND_ROBIN):
            # Cycle through turn_order
            idx = self.current_turn % len(self.turn_order)
            return self.turn_order[idx]

        # For broadcast/whisper/interrupt, caller determines speaker
        return None

    def advance_turn(self) -> None:
        """Increment turn counter and check max_turns."""
        self.current_turn += 1
        if self.max_turns > 0 and self.current_turn >= self.max_turns:
            self.is_active = False


@dataclass(frozen=True)
class TurnDirective:
    """Instruction for which agent should speak next and what they should see.

    Attributes:
        session_id: Session this directive belongs to.
        agent_id: Agent that should generate the next message.
        turn_number: Current turn number.
        visible_messages: Messages this agent can see in its context.
        system_instruction: System prompt with role/debate format instructions.
    """

    session_id: str
    agent_id: str
    turn_number: int
    visible_messages: list[ChannelMessage]
    system_instruction: str


@dataclass
class Vote:
    """A single vote cast by an agent.

    Attributes:
        vote_id: Unique vote identifier.
        session_id: Session this vote belongs to.
        agent_id: Agent that cast this vote.
        question: Question being voted on.
        choice: The agent's choice (option label).
        ranking: For ranked-choice voting, ordered list of preferences.
    """

    vote_id: str
    session_id: str
    agent_id: str
    question: str
    choice: str
    ranking: list[str] = field(default_factory=list)


@dataclass
class VoteTally:
    """Results of a vote.

    Attributes:
        question: Question that was voted on.
        total_votes: Total number of votes cast.
        results: Map of choice to vote count.
        winner: Winning choice (if any).
        tied: Whether the result was a tie.
    """

    question: str
    total_votes: int
    results: dict[str, int]
    winner: str | None = None
    tied: bool = False
