"""CoordinationService: orchestrates multi-agent coordination sessions.

The coordination service manages structured multi-agent conversations by:
1. Creating and tracking coordination sessions
2. Building prompts that include inter-agent context
3. Routing generation requests through the scheduler
4. Managing turn sequencing based on topology

Architecture layer: application service.
No MLX / infrastructure imports — interacts with adapters through ports.
"""

import structlog
from uuid import uuid4

from semantic.domain.coordination import (
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
from semantic.domain.errors import InvalidTurnError, SessionNotFoundError

logger = structlog.get_logger(__name__)


class CoordinationService:
    """Orchestrates multi-agent coordination sessions.

    Dependencies (injected):
      - scheduler: ConcurrentScheduler (for inference)
      - cache_store: AgentCacheStore (for per-agent KV caches)
      - engine: BlockPoolBatchEngine (for tokenization)
    """

    def __init__(self, scheduler, cache_store, engine) -> None:
        self._scheduler = scheduler
        self._cache_store = cache_store
        self._engine = engine
        self._sessions: dict[str, CoordinationSession] = {}

    def create_session(
        self,
        topology: Topology,
        debate_format: DebateFormat,
        decision_mode: DecisionMode,
        agents: list[AgentRole],
        initial_prompt: str = "",
        max_turns: int = 0,
    ) -> CoordinationSession:
        """Create a new coordination session with agents and topology.

        Args:
            topology: Communication topology (turn_by_turn, broadcast, etc.).
            debate_format: Debate structure (free_form, structured, etc.).
            decision_mode: Decision-making mode (voting, consensus, etc.).
            agents: List of AgentRole objects defining participants.
            initial_prompt: Optional initial prompt/topic for discussion.
            max_turns: Maximum turns before auto-termination (0 = unlimited).

        Returns:
            The created CoordinationSession.
        """
        session_id = f"coord_{uuid4().hex[:12]}"

        # Create public channel
        public_channel = Channel(
            channel_id=f"{session_id}_public",
            channel_type="public",
            participant_ids=frozenset(a.agent_id for a in agents),
        )

        # Add initial prompt as system message if provided
        if initial_prompt:
            public_channel.add_message(
                sender_id="system",
                content=initial_prompt,
                turn_number=0,
                visible_to=frozenset(),  # Empty = public
            )

        session = CoordinationSession(
            session_id=session_id,
            topology=topology,
            debate_format=debate_format,
            decision_mode=decision_mode,
            agents={a.agent_id: a for a in agents},
            channels={public_channel.channel_id: public_channel},
            turn_order=[a.agent_id for a in agents],
            max_turns=max_turns,
        )

        self._sessions[session_id] = session
        logger.info(
            "coordination_session_created",
            session_id=session_id,
            topology=topology.value,
            num_agents=len(agents),
        )
        return session

    def get_session(self, session_id: str) -> CoordinationSession:
        """Get a coordination session by ID.

        Args:
            session_id: Session identifier.

        Returns:
            The CoordinationSession.

        Raises:
            SessionNotFoundError: If session does not exist.
        """
        if session_id not in self._sessions:
            raise SessionNotFoundError(f"Session {session_id} not found")
        return self._sessions[session_id]

    def delete_session(self, session_id: str) -> None:
        """Delete a coordination session.

        Args:
            session_id: Session identifier.

        Raises:
            SessionNotFoundError: If session does not exist.
        """
        if session_id not in self._sessions:
            raise SessionNotFoundError(f"Session {session_id} not found")
        del self._sessions[session_id]
        logger.info("coordination_session_deleted", session_id=session_id)

    def get_next_turn(self, session_id: str) -> TurnDirective:
        """Determine which agent speaks next based on topology and turn state.

        Args:
            session_id: Session identifier.

        Returns:
            TurnDirective with next speaker and visible context.

        Raises:
            SessionNotFoundError: If session does not exist.
            InvalidTurnError: If session is not active or has no valid next speaker.
        """
        session = self.get_session(session_id)

        if not session.is_active:
            raise InvalidTurnError(f"Session {session_id} is not active")

        next_speaker = session.get_next_speaker()
        if next_speaker is None:
            raise InvalidTurnError(f"No valid next speaker for session {session_id}")

        # Get agent role
        agent_role = session.agents[next_speaker]

        # Get public channel
        public_channel = next(
            (c for c in session.channels.values() if c.channel_type == "public"), None
        )
        if public_channel is None:
            raise InvalidTurnError(f"No public channel found for session {session_id}")

        # Filter messages visible to this agent
        visible_messages = self._filter_visible_messages(
            public_channel.messages, next_speaker
        )

        # Build system instruction based on role and debate format
        system_instruction = self._build_system_instruction(agent_role, session.debate_format)

        return TurnDirective(
            session_id=session_id,
            agent_id=next_speaker,
            turn_number=session.current_turn,
            visible_messages=visible_messages,
            system_instruction=system_instruction,
        )

    def _filter_visible_messages(
        self, messages: list[ChannelMessage], agent_id: str
    ) -> list[ChannelMessage]:
        """Filter messages to those visible to a specific agent.

        Args:
            messages: All messages in the channel.
            agent_id: Agent ID to filter for.

        Returns:
            List of messages this agent can see.
        """
        visible = []
        for msg in messages:
            # Empty visible_to = public (all can see)
            if not msg.visible_to or agent_id in msg.visible_to:
                visible.append(msg)
        return visible

    def _build_system_instruction(
        self, agent_role: AgentRole, debate_format: DebateFormat
    ) -> str:
        """Build system instruction prompt based on role and debate format.

        Args:
            agent_role: The agent's role in the coordination.
            debate_format: The debate format being used.

        Returns:
            System instruction string.
        """
        instructions = []

        # Role-specific instruction
        if agent_role.system_prompt:
            instructions.append(agent_role.system_prompt)
        else:
            instructions.append(f"You are {agent_role.display_name}, a {agent_role.role}.")

        # Debate format instruction
        if debate_format == DebateFormat.STRUCTURED:
            instructions.append(
                "This is a structured debate. Present arguments clearly with evidence. "
                "Respond to points made by other participants."
            )
        elif debate_format == DebateFormat.SOCRATIC:
            instructions.append(
                "Use the Socratic method: ask probing questions to explore ideas deeply. "
                "Question assumptions and guide the discussion through inquiry."
            )
        elif debate_format == DebateFormat.DEVILS_ADVOCATE:
            if agent_role.role == "critic":
                instructions.append(
                    "Play devil's advocate: challenge proposals and arguments. "
                    "Find weaknesses and alternative perspectives."
                )
        elif debate_format == DebateFormat.PARLIAMENTARY:
            instructions.append(
                "Follow parliamentary procedure. Use formal language and structure. "
                "Address the moderator when speaking."
            )

        return "\n\n".join(instructions)

    def build_agent_prompt(
        self, directive: TurnDirective, agent_role: AgentRole
    ) -> list[dict]:
        """Build the full prompt for an agent including visible messages and role instruction.

        Args:
            directive: Turn directive with context.
            agent_role: The agent's role.

        Returns:
            List of message dicts in OpenAI format [{"role": ..., "content": ...}].
        """
        messages = []

        # System message with role instruction
        if directive.system_instruction:
            messages.append({"role": "system", "content": directive.system_instruction})

        # Visible channel messages formatted as conversation
        for msg in directive.visible_messages:
            # System messages stay as system role
            if msg.sender_id == "system":
                messages.append({"role": "system", "content": msg.content})
            # Agent's own messages as assistant role
            elif msg.sender_id == agent_role.agent_id:
                messages.append({"role": "assistant", "content": msg.content})
            # Other agents' messages as user role (with name prefix)
            else:
                sender_name = self._get_agent_name(directive.session_id, msg.sender_id)
                content = f"[{sender_name}]: {msg.content}"
                messages.append({"role": "user", "content": content})

        return messages

    def _get_agent_name(self, session_id: str, agent_id: str) -> str:
        """Get display name for an agent.

        Args:
            session_id: Session identifier.
            agent_id: Agent identifier.

        Returns:
            Display name, or agent_id if not found.
        """
        session = self._sessions.get(session_id)
        if session and agent_id in session.agents:
            return session.agents[agent_id].display_name
        return agent_id

    async def execute_turn(self, session_id: str) -> ChannelMessage:
        """Execute the next turn: determine speaker, build prompt, generate, record message.

        Args:
            session_id: Session identifier.

        Returns:
            The ChannelMessage that was generated.

        Raises:
            SessionNotFoundError: If session does not exist.
            InvalidTurnError: If no valid next turn.
        """
        session = self.get_session(session_id)
        directive = self.get_next_turn(session_id)
        agent_role = session.agents[directive.agent_id]

        # Build prompt
        prompt_messages = self.build_agent_prompt(directive, agent_role)

        # Tokenize prompt (engine provides tokenizer)
        prompt_text = self._format_messages_as_text(prompt_messages)
        prompt_tokens = self._engine.tokenizer.encode(prompt_text)

        # Load agent's cache (if any)
        namespaced_agent_id = f"coord_{session_id}_{directive.agent_id}"
        cached_blocks = self._cache_store.load(namespaced_agent_id)

        # Submit to scheduler (or fall back to direct engine if no scheduler)
        if self._scheduler is not None:
            result = await self._scheduler.submit_and_wait(
                agent_id=namespaced_agent_id,
                prompt_tokens=prompt_tokens,
                cache=cached_blocks,
                max_tokens=512,
                prompt_text=prompt_text,
                temperature=0.7,
                top_p=1.0,
            )
        else:
            # Direct engine path (no scheduler available)
            result = await self._generate_direct(
                agent_id=namespaced_agent_id,
                prompt_tokens=prompt_tokens,
                cache=cached_blocks,
                max_tokens=512,
                temperature=0.7,
                top_p=1.0,
            )

        # Save updated cache
        if result.cache:
            self._cache_store.save(namespaced_agent_id, result.cache)

        # Add message to public channel
        public_channel = next(
            c for c in session.channels.values() if c.channel_type == "public"
        )
        message = public_channel.add_message(
            sender_id=directive.agent_id,
            content=result.text,
            turn_number=session.current_turn,
        )

        # Advance turn
        session.advance_turn()

        logger.info(
            "coordination_turn_executed",
            session_id=session_id,
            agent_id=directive.agent_id,
            turn=directive.turn_number,
            text_length=len(result.text),
        )

        return message

    async def execute_round(self, session_id: str) -> list[ChannelMessage]:
        """Execute a full round (all agents get one turn each).

        Args:
            session_id: Session identifier.

        Returns:
            List of ChannelMessages generated in this round.

        Raises:
            SessionNotFoundError: If session does not exist.
        """
        session = self.get_session(session_id)
        messages = []

        for _ in range(len(session.agents)):
            if not session.is_active:
                break
            message = await self.execute_turn(session_id)
            messages.append(message)

        logger.info(
            "coordination_round_complete",
            session_id=session_id,
            messages_generated=len(messages),
        )
        return messages

    def add_whisper(
        self, session_id: str, from_id: str, to_id: str, content: str
    ) -> ChannelMessage:
        """Add a private message between two agents (whisper channel).

        Args:
            session_id: Session identifier.
            from_id: Sender agent ID.
            to_id: Recipient agent ID.
            content: Message content.

        Returns:
            The created ChannelMessage.

        Raises:
            SessionNotFoundError: If session does not exist.
        """
        session = self.get_session(session_id)

        # Find or create whisper channel
        channel_id = f"{session_id}_whisper_{min(from_id, to_id)}_{max(from_id, to_id)}"
        if channel_id not in session.channels:
            session.channels[channel_id] = Channel(
                channel_id=channel_id,
                channel_type="whisper",
                participant_ids=frozenset([from_id, to_id]),
            )

        whisper_channel = session.channels[channel_id]
        message = whisper_channel.add_message(
            sender_id=from_id,
            content=content,
            turn_number=session.current_turn,
            visible_to=frozenset([from_id, to_id]),
        )

        logger.info(
            "whisper_message_sent",
            session_id=session_id,
            from_id=from_id,
            to_id=to_id,
        )
        return message

    def tally_votes(self, session_id: str, votes: list[Vote]) -> VoteTally:
        """Tally votes and determine winner.

        Args:
            session_id: Session identifier.
            votes: List of Vote objects.

        Returns:
            VoteTally with results.
        """
        if not votes:
            return VoteTally(
                question="",
                total_votes=0,
                results={},
            )

        question = votes[0].question
        results: dict[str, int] = {}

        # Count votes
        for vote in votes:
            choice = vote.choice
            results[choice] = results.get(choice, 0) + 1

        # Determine winner
        if results:
            max_votes = max(results.values())
            winners = [choice for choice, count in results.items() if count == max_votes]
            tied = len(winners) > 1
            winner = None if tied else winners[0]
        else:
            tied = False
            winner = None

        tally = VoteTally(
            question=question,
            total_votes=len(votes),
            results=results,
            winner=winner,
            tied=tied,
        )

        logger.info(
            "votes_tallied",
            session_id=session_id,
            total_votes=len(votes),
            winner=winner,
            tied=tied,
        )
        return tally

    def _format_messages_as_text(self, messages: list[dict]) -> str:
        """Format message list as plain text for tokenization.

        Args:
            messages: List of message dicts.

        Returns:
            Formatted text string.
        """
        lines = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                lines.append(f"System: {content}")
            elif role == "user":
                lines.append(f"User: {content}")
            elif role == "assistant":
                lines.append(f"Assistant: {content}")
        return "\n\n".join(lines)

    async def _generate_direct(
        self,
        agent_id: str,
        prompt_tokens: list[int],
        cache,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ):
        """Generate text using batch engine directly (fallback when no scheduler).

        Args:
            agent_id: Agent identifier.
            prompt_tokens: Tokenized prompt.
            cache: Cached blocks from previous turns.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Top-p sampling parameter.

        Returns:
            Generation result with text and cache.
        """
        import asyncio
        from dataclasses import dataclass

        # Submit request to batch engine
        uid = self._engine.submit(
            agent_id=agent_id,
            prompt="",  # Empty string since we provide prompt_tokens
            cache=cache,
            max_tokens=max_tokens,
            prompt_tokens=prompt_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        # Poll engine until result is ready
        result = None
        max_steps = 10000  # Safety limit
        for _ in range(max_steps):
            for completion in self._engine.step():
                if completion.uid == uid:
                    result = completion
                    break
            if result:
                break
            await asyncio.sleep(0.01)  # Small delay to avoid busy-waiting

        if result is None:
            raise RuntimeError(f"Generation failed for agent {agent_id}")

        # Build result object compatible with scheduler result
        @dataclass
        class DirectResult:
            text: str
            cache: any

        return DirectResult(text=result.text, cache=result.blocks)
