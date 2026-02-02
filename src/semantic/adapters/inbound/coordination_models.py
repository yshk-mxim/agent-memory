"""Request and response models for coordination API endpoints.

All models use Pydantic for validation and OpenAPI schema generation.
"""

from pydantic import BaseModel, Field


class AgentRoleConfig(BaseModel):
    """Configuration for an agent's role in a coordination session."""

    agent_id: str | None = Field(
        default=None,
        description="Agent identifier (auto-generated if not provided)",
    )
    display_name: str = Field(
        ...,
        description="Human-readable name for this agent",
        min_length=1,
        max_length=50,
    )
    role: str = Field(
        default="participant",
        description="Role type: moderator, critic, advocate, participant",
    )
    system_prompt: str = Field(
        default="",
        description="Optional role-specific system prompt",
        max_length=2000,
    )
    lifecycle: str = Field(
        default="ephemeral",
        description="Memory lifecycle: ephemeral (session-only) or permanent (long-term memory)",
        pattern="^(ephemeral|permanent)$",
    )


class CreateSessionRequest(BaseModel):
    """Request to create a new coordination session."""

    topology: str = Field(
        default="turn_by_turn",
        description="Communication topology: turn_by_turn, broadcast, whisper, round_robin, interrupt",
    )
    debate_format: str = Field(
        default="free_form",
        description="Debate format: free_form, structured, socratic, devils_advocate, parliamentary",
    )
    decision_mode: str = Field(
        default="none",
        description="Decision mode: none, majority_vote, ranked_choice, coordinator_decides, consensus",
    )
    agents: list[AgentRoleConfig] = Field(
        ...,
        description="List of agents participating in this session",
        min_length=2,
        max_length=10,
    )
    initial_prompt: str = Field(
        default="",
        description="Optional initial prompt/topic for discussion",
        max_length=50000,
    )
    max_turns: int = Field(
        default=0,
        description="Maximum turns before auto-termination (0 = unlimited)",
        ge=0,
        le=1000,
    )


class CreateSessionResponse(BaseModel):
    """Response after creating a coordination session."""

    session_id: str = Field(..., description="Unique session identifier")
    agents: list[AgentRoleConfig] = Field(..., description="Configured agents")
    topology: str = Field(..., description="Communication topology")
    debate_format: str = Field(..., description="Debate format")
    decision_mode: str = Field(..., description="Decision mode")
    status: str = Field(..., description="Session status (active, completed, etc.)")


class ChannelMessageResponse(BaseModel):
    """A single message in a coordination channel."""

    message_id: str = Field(..., description="Unique message identifier")
    sender_id: str = Field(..., description="Agent ID of sender")
    sender_name: str = Field(..., description="Display name of sender")
    content: str = Field(..., description="Message text content")
    turn_number: int = Field(..., description="Turn when message was sent")
    channel_type: str = Field(..., description="Channel type (public, whisper)")
    is_interrupted: bool = Field(
        default=False, description="Whether message was interrupted mid-generation"
    )


class AgentStateResponse(BaseModel):
    """Current state of an agent in the session."""

    agent_id: str = Field(..., description="Agent identifier")
    display_name: str = Field(..., description="Display name")
    role: str = Field(..., description="Role type")
    message_count: int = Field(..., description="Number of messages sent")
    lifecycle: str = Field(..., description="Memory lifecycle (ephemeral or permanent)")


class SessionStatusResponse(BaseModel):
    """Current status of a coordination session."""

    session_id: str = Field(..., description="Session identifier")
    current_turn: int = Field(..., description="Current turn number")
    is_active: bool = Field(..., description="Whether session is active")
    next_speaker: str | None = Field(..., description="Agent ID of next speaker (if any)")
    agent_states: list[AgentStateResponse] = Field(
        ..., description="State of all agents in session"
    )


class ExecuteTurnResponse(BaseModel):
    """Response after executing a single turn."""

    message: ChannelMessageResponse = Field(..., description="The generated message")
    session_status: SessionStatusResponse = Field(..., description="Updated session status")


class ExecuteRoundResponse(BaseModel):
    """Response after executing a full round (all agents speak once)."""

    messages: list[ChannelMessageResponse] = Field(
        ..., description="All messages generated in this round"
    )
    session_status: SessionStatusResponse = Field(..., description="Updated session status")


class WhisperRequest(BaseModel):
    """Request to send a whisper (private message) between agents."""

    from_agent_id: str = Field(..., description="Sender agent ID")
    to_agent_id: str = Field(..., description="Recipient agent ID")
    content: str = Field(
        ...,
        description="Message content",
        min_length=1,
        max_length=5000,
    )


class WhisperResponse(BaseModel):
    """Response after sending a whisper message."""

    message: ChannelMessageResponse = Field(..., description="The whisper message")


class VoteRequest(BaseModel):
    """Request to submit a vote."""

    agent_id: str = Field(..., description="Agent casting the vote")
    question: str = Field(..., description="Question being voted on")
    choice: str = Field(..., description="Agent's choice")
    ranking: list[str] = Field(
        default_factory=list,
        description="For ranked-choice voting, ordered preferences",
    )


class VoteResponse(BaseModel):
    """Response after submitting a vote."""

    vote_id: str = Field(..., description="Unique vote identifier")
    agent_id: str = Field(..., description="Agent that voted")
    choice: str = Field(..., description="Chosen option")


class TallyResponse(BaseModel):
    """Vote tally results."""

    question: str = Field(..., description="Question that was voted on")
    total_votes: int = Field(..., description="Total votes cast")
    results: dict[str, int] = Field(..., description="Map of choice to vote count")
    winner: str | None = Field(..., description="Winning choice (if any)")
    tied: bool = Field(..., description="Whether result was a tie")


class SessionListResponse(BaseModel):
    """List of active sessions."""

    sessions: list[SessionStatusResponse] = Field(..., description="Active sessions")


class MessageListResponse(BaseModel):
    """List of messages in a session."""

    session_id: str = Field(..., description="Session identifier")
    messages: list[ChannelMessageResponse] = Field(..., description="All messages")
