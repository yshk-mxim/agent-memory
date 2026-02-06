"""Coordination API adapter (FastAPI router).

Implements REST endpoints for multi-agent coordination sessions at
/v1/coordination/*.

Routes:
  POST   /v1/coordination/sessions          - Create session
  GET    /v1/coordination/sessions          - List sessions
  GET    /v1/coordination/sessions/{id}     - Get session status
  DELETE /v1/coordination/sessions/{id}     - Delete session
  POST   /v1/coordination/sessions/{id}/turn     - Execute next turn
  POST   /v1/coordination/sessions/{id}/round    - Execute full round
  POST   /v1/coordination/sessions/{id}/whisper  - Send whisper message
  POST   /v1/coordination/sessions/{id}/vote     - Submit vote
  GET    /v1/coordination/sessions/{id}/messages - Get all messages
"""

import json
import logging
from collections.abc import AsyncIterator
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request, status
from sse_starlette.sse import EventSourceResponse

from semantic.adapters.inbound.adapter_helpers import get_coordination_service
from semantic.adapters.inbound.coordination_models import (
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
    VoteRequest,
    VoteResponse,
    WhisperRequest,
    WhisperResponse,
)
from semantic.domain.coordination import (
    AgentLifecycle,
    AgentRole,
    DebateFormat,
    DecisionMode,
    Topology,
)
from semantic.domain.errors import CoordinationError, SessionNotFoundError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/coordination", tags=["coordination"])


@router.post("/sessions", status_code=status.HTTP_201_CREATED)
async def create_session(
    request: Request,
    body: CreateSessionRequest,
) -> CreateSessionResponse:
    """Create a new multi-agent coordination session.

    Args:
        request: FastAPI request (for accessing app state).
        body: Session creation request.

    Returns:
        CreateSessionResponse with session ID and configuration.
    """
    service = get_coordination_service(request)

    # Convert topology/format/mode strings to enums
    try:
        topology = Topology(body.topology)
        debate_format = DebateFormat(body.debate_format)
        decision_mode = DecisionMode(body.decision_mode)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid topology/format/mode: {e}",
        ) from e

    # Convert agent configs to AgentRole domain objects
    agents = []
    for agent_config in body.agents:
        agent_id = agent_config.agent_id or f"agent_{uuid4().hex[:8]}"
        agents.append(
            AgentRole(
                agent_id=agent_id,
                display_name=agent_config.display_name,
                role=agent_config.role,
                system_prompt=agent_config.system_prompt,
                lifecycle=AgentLifecycle(agent_config.lifecycle),
            )
        )

    # Create session
    session = await service.create_session(
        topology=topology,
        debate_format=debate_format,
        decision_mode=decision_mode,
        agents=agents,
        initial_prompt=body.initial_prompt,
        per_agent_prompts=body.per_agent_prompts or None,
        max_turns=body.max_turns,
        persistent_cache_prefix=body.persistent_cache_prefix,
        prior_agent_messages=body.prior_agent_messages or None,
    )

    # Build response
    agent_configs = [
        AgentRoleConfig(
            agent_id=a.agent_id,
            display_name=a.display_name,
            role=a.role,
            system_prompt=a.system_prompt,
            lifecycle=a.lifecycle.value,
        )
        for a in agents
    ]

    return CreateSessionResponse(
        session_id=session.session_id,
        agents=agent_configs,
        topology=session.topology.value,
        debate_format=session.debate_format.value,
        decision_mode=session.decision_mode.value,
        status="active" if session.is_active else "completed",
    )


@router.get("/sessions")
async def list_sessions(request: Request) -> SessionListResponse:
    """List all active coordination sessions.

    Args:
        request: FastAPI request.

    Returns:
        SessionListResponse with list of sessions.
    """
    service = get_coordination_service(request)

    # Get all sessions via public service method
    sessions = []
    for session in service.list_sessions():
        # Build agent states
        agent_states = []
        public_channel = next(
            (c for c in session.channels.values() if c.channel_type == "public"), None
        )
        for agent in session.agents.values():
            message_count = 0
            if public_channel:
                message_count = sum(
                    1 for msg in public_channel.messages if msg.sender_id == agent.agent_id
                )
            agent_states.append(
                AgentStateResponse(
                    agent_id=agent.agent_id,
                    display_name=agent.display_name,
                    role=agent.role,
                    message_count=message_count,
                    lifecycle=agent.lifecycle.value,
                )
            )

        # Determine next speaker
        try:
            next_speaker = session.get_next_speaker()
        except Exception:
            next_speaker = None

        sessions.append(
            SessionStatusResponse(
                session_id=session.session_id,
                current_turn=session.current_turn,
                is_active=session.is_active,
                next_speaker=next_speaker,
                agent_states=agent_states,
            )
        )

    return SessionListResponse(sessions=sessions)


@router.get("/sessions/{session_id}")
async def get_session_status(
    request: Request,
    session_id: str,
) -> SessionStatusResponse:
    """Get status of a coordination session.

    Args:
        request: FastAPI request.
        session_id: Session identifier.

    Returns:
        SessionStatusResponse with current state.
    """
    service = get_coordination_service(request)

    try:
        session = service.get_session(session_id)
    except SessionNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        ) from e

    # Build agent states
    agent_states = []
    public_channel = next(
        (c for c in session.channels.values() if c.channel_type == "public"), None
    )
    for agent in session.agents.values():
        message_count = 0
        if public_channel:
            message_count = sum(
                1 for msg in public_channel.messages if msg.sender_id == agent.agent_id
            )
        agent_states.append(
            AgentStateResponse(
                agent_id=agent.agent_id,
                display_name=agent.display_name,
                role=agent.role,
                message_count=message_count,
                lifecycle=agent.lifecycle.value,
            )
        )

    # Determine next speaker
    try:
        next_speaker = session.get_next_speaker()
    except Exception:
        next_speaker = None

    return SessionStatusResponse(
        session_id=session.session_id,
        current_turn=session.current_turn,
        is_active=session.is_active,
        next_speaker=next_speaker,
        agent_states=agent_states,
    )


@router.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_session(
    request: Request,
    session_id: str,
) -> None:
    """Delete a coordination session.

    Args:
        request: FastAPI request.
        session_id: Session identifier.
    """
    service = get_coordination_service(request)

    try:
        await service.delete_session(session_id)
    except SessionNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        ) from e


@router.post("/sessions/{session_id}/turn")
async def execute_turn(
    request: Request,
    session_id: str,
) -> ExecuteTurnResponse:
    """Execute the next turn in a coordination session.

    Args:
        request: FastAPI request.
        session_id: Session identifier.

    Returns:
        ExecuteTurnResponse with generated message and updated status.
    """
    service = get_coordination_service(request)

    try:
        # Execute turn
        message = await service.execute_turn(session_id)

        # Get updated session status
        session = service.get_session(session_id)
        sender_agent = session.agents[message.sender_id]

        # Build response
        message_response = ChannelMessageResponse(
            message_id=message.message_id,
            sender_id=message.sender_id,
            sender_name=sender_agent.display_name,
            content=message.content,
            turn_number=message.turn_number,
            channel_type="public",
            is_interrupted=message.is_interrupted,
        )

        # Build agent states
        agent_states = []
        public_channel = next(c for c in session.channels.values() if c.channel_type == "public")
        for agent in session.agents.values():
            message_count = sum(
                1 for msg in public_channel.messages if msg.sender_id == agent.agent_id
            )
            agent_states.append(
                AgentStateResponse(
                    agent_id=agent.agent_id,
                    display_name=agent.display_name,
                    role=agent.role,
                    message_count=message_count,
                    lifecycle=agent.lifecycle.value,
                )
            )

        # Determine next speaker
        try:
            next_speaker = session.get_next_speaker()
        except Exception:
            next_speaker = None

        status_response = SessionStatusResponse(
            session_id=session.session_id,
            current_turn=session.current_turn,
            is_active=session.is_active,
            next_speaker=next_speaker,
            agent_states=agent_states,
        )

        return ExecuteTurnResponse(
            message=message_response,
            session_status=status_response,
        )

    except CoordinationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e


@router.post("/sessions/{session_id}/round")
async def execute_round(
    request: Request,
    session_id: str,
) -> ExecuteRoundResponse:
    """Execute a full round (all agents get one turn).

    Args:
        request: FastAPI request.
        session_id: Session identifier.

    Returns:
        ExecuteRoundResponse with all generated messages and updated status.
    """
    service = get_coordination_service(request)

    try:
        # Execute round
        messages = await service.execute_round(session_id)

        # Get updated session status
        session = service.get_session(session_id)

        # Build message responses
        message_responses = []
        for msg in messages:
            sender_agent = session.agents[msg.sender_id]
            message_responses.append(
                ChannelMessageResponse(
                    message_id=msg.message_id,
                    sender_id=msg.sender_id,
                    sender_name=sender_agent.display_name,
                    content=msg.content,
                    turn_number=msg.turn_number,
                    channel_type="public",
                    is_interrupted=msg.is_interrupted,
                )
            )

        # Build agent states
        agent_states = []
        public_channel = next(c for c in session.channels.values() if c.channel_type == "public")
        for agent in session.agents.values():
            message_count = sum(
                1 for msg in public_channel.messages if msg.sender_id == agent.agent_id
            )
            agent_states.append(
                AgentStateResponse(
                    agent_id=agent.agent_id,
                    display_name=agent.display_name,
                    role=agent.role,
                    message_count=message_count,
                    lifecycle=agent.lifecycle.value,
                )
            )

        # Determine next speaker
        try:
            next_speaker = session.get_next_speaker()
        except Exception:
            next_speaker = None

        status_response = SessionStatusResponse(
            session_id=session.session_id,
            current_turn=session.current_turn,
            is_active=session.is_active,
            next_speaker=next_speaker,
            agent_states=agent_states,
        )

        return ExecuteRoundResponse(
            messages=message_responses,
            session_status=status_response,
        )

    except CoordinationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e


async def stream_turn_events(service, session_id: str) -> AsyncIterator[dict[str, str]]:
    """Generate SSE events for streaming a single turn.

    Yields:
        SSE event dicts with 'event' and 'data' keys.
    """
    # Get next speaker info before streaming
    try:
        directive = service.get_next_turn(session_id)
        session = service.get_session(session_id)
        agent_role = session.agents[directive.agent_id]

        # Send turn_start event
        yield {
            "event": "turn_start",
            "data": json.dumps(
                {
                    "agent_id": directive.agent_id,
                    "agent_name": agent_role.display_name,
                    "turn": session.current_turn + 1,
                }
            ),
        }

        accumulated_text = ""
        clean_text = ""

        # Stream tokens
        async for delta in service.execute_turn_stream(session_id):
            if delta.finish_reason == "cleaned":
                # Final delta contains the cleaned text for turn_complete
                clean_text = delta.text
                continue

            new_text = delta.text[len(accumulated_text) :]
            accumulated_text = delta.text

            if new_text:
                yield {
                    "event": "token",
                    "data": json.dumps(
                        {
                            "text": new_text,
                            "accumulated": accumulated_text,
                        }
                    ),
                }

        # Send turn_complete event with cleaned content
        yield {
            "event": "turn_complete",
            "data": json.dumps(
                {
                    "agent_id": directive.agent_id,
                    "agent_name": agent_role.display_name,
                    "content": clean_text or accumulated_text,
                    "turn": directive.turn_number,
                }
            ),
        }

    except CoordinationError as e:
        yield {
            "event": "error",
            "data": json.dumps({"error": str(e)}),
        }


async def stream_round_events(service, session_id: str) -> AsyncIterator[dict[str, str]]:
    """Generate SSE events for streaming a full round.

    Yields:
        SSE event dicts with 'event' and 'data' keys.
    """
    try:
        async for agent_id, agent_name, delta in service.execute_round_stream(session_id):
            # These events come from execute_turn_stream within execute_round_stream
            yield {
                "event": "token",
                "data": json.dumps(
                    {
                        "agent_id": agent_id,
                        "agent_name": agent_name,
                        "text": delta.text,
                        "token_count": delta.token_count,
                    }
                ),
            }

        yield {
            "event": "round_complete",
            "data": json.dumps({"status": "complete"}),
        }

    except CoordinationError as e:
        yield {
            "event": "error",
            "data": json.dumps({"error": str(e)}),
        }


@router.post("/sessions/{session_id}/turn/stream")
async def execute_turn_stream(
    request: Request,
    session_id: str,
):
    """Execute the next turn with Server-Sent Events streaming.

    Args:
        request: FastAPI request.
        session_id: Session identifier.

    Returns:
        EventSourceResponse with streaming tokens.
    """
    service = get_coordination_service(request)
    return EventSourceResponse(stream_turn_events(service, session_id))


@router.post("/sessions/{session_id}/round/stream")
async def execute_round_stream(
    request: Request,
    session_id: str,
):
    """Execute a full round with Server-Sent Events streaming.

    Args:
        request: FastAPI request.
        session_id: Session identifier.

    Returns:
        EventSourceResponse with streaming tokens for each agent.
    """
    service = get_coordination_service(request)
    return EventSourceResponse(stream_round_events(service, session_id))


@router.post("/sessions/{session_id}/whisper")
async def send_whisper(
    request: Request,
    session_id: str,
    body: WhisperRequest,
) -> WhisperResponse:
    """Send a whisper (private message) between two agents.

    Args:
        request: FastAPI request.
        session_id: Session identifier.
        body: Whisper request with sender, recipient, and content.

    Returns:
        WhisperResponse with the created message.
    """
    service = get_coordination_service(request)

    try:
        message = service.add_whisper(
            session_id=session_id,
            from_id=body.from_agent_id,
            to_id=body.to_agent_id,
            content=body.content,
        )

        session = service.get_session(session_id)
        sender_agent = session.agents[message.sender_id]

        message_response = ChannelMessageResponse(
            message_id=message.message_id,
            sender_id=message.sender_id,
            sender_name=sender_agent.display_name,
            content=message.content,
            turn_number=message.turn_number,
            channel_type="whisper",
            is_interrupted=False,
        )

        return WhisperResponse(message=message_response)

    except CoordinationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e


@router.post("/sessions/{session_id}/vote")
async def submit_vote(
    request: Request,  # noqa: ARG001
    session_id: str,  # noqa: ARG001
    body: VoteRequest,
) -> VoteResponse:
    """Submit a vote from an agent."""
    vote_id = uuid4().hex[:12]

    return VoteResponse(
        vote_id=vote_id,
        agent_id=body.agent_id,
        choice=body.choice,
    )


@router.delete("/caches/{prefix}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_persistent_caches(
    request: Request,
    prefix: str,
) -> None:
    """Delete all persistent caches matching a prefix.

    Used by scenario "Reset All" to clear cross-phase agent memory.

    Args:
        request: FastAPI request.
        prefix: The persistent_cache_prefix to match.
    """
    service = get_coordination_service(request)
    service.delete_persistent_caches(prefix)


@router.get("/sessions/{session_id}/messages")
async def get_messages(
    request: Request,
    session_id: str,
) -> MessageListResponse:
    """Get all messages in a coordination session.

    Args:
        request: FastAPI request.
        session_id: Session identifier.

    Returns:
        MessageListResponse with all messages.
    """
    service = get_coordination_service(request)

    try:
        session = service.get_session(session_id)

        # Get public channel messages
        public_channel = next(
            (c for c in session.channels.values() if c.channel_type == "public"), None
        )
        if not public_channel:
            return MessageListResponse(session_id=session_id, messages=[])

        # Return only public messages (empty visible_to).
        # Private messages (prior-phase context for KV cache prefix matching,
        # per-agent prompts) are internal and not part of the phase transcript.
        message_responses = []
        for msg in public_channel.messages:
            if msg.visible_to:
                continue
            sender_name = session.agents.get(
                msg.sender_id, type("obj", (), {"display_name": "System"})()
            ).display_name
            message_responses.append(
                ChannelMessageResponse(
                    message_id=msg.message_id,
                    sender_id=msg.sender_id,
                    sender_name=sender_name,
                    content=msg.content,
                    turn_number=msg.turn_number,
                    channel_type="public",
                    is_interrupted=msg.is_interrupted,
                )
            )

        return MessageListResponse(
            session_id=session_id,
            messages=message_responses,
        )

    except SessionNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        ) from e
