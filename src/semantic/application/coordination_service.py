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
from typing import AsyncIterator
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
from semantic.domain.value_objects import StreamDelta

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

        # Enhanced debate format instruction with concrete examples
        if debate_format == DebateFormat.STRUCTURED:
            instructions.append("""
STRUCTURED DEBATE FORMAT - Required Structure:

Your response must:
1. CITE other agents: "As [Agent] said in Turn X..."
2. PROVIDE evidence: Support claims with clear reasoning
3. ADDRESS counterarguments: "While [Agent] raises a valid concern about X..."
4. BUILD logical chains: "Building on [Agent]'s point..."

Example structure:
"Alice argued in Turn 2 that [X]. I agree/disagree because [reasoning with evidence].
Bob's point in Turn 4 about [Y] is valid, but we must also consider [Z].
Therefore, my position is [clear conclusion]."
""")
        elif debate_format == DebateFormat.SOCRATIC:
            instructions.append("""
SOCRATIC METHOD - Question-Driven Discussion:

Your response must be PRIMARILY QUESTIONS that probe others' reasoning:

Required elements:
1. QUESTION premises: "You stated X. What exactly do you mean by X?"
2. EXPLORE assumptions: "Why do you assume Y?"
3. TEST implications: "If X is true, wouldn't that mean Z?"
4. SEEK clarity: "Help me understand your reasoning about..."

Example:
"Alice, you argued for option X in Turn 2. What specific problem does X solve?
If we choose X, what trade-offs are we accepting?
Bob disagreed in Turn 3, citing concern Y. How would you respond to Bob's concern?"
""")
        elif debate_format == DebateFormat.DEVILS_ADVOCATE:
            if agent_role.role == "critic":
                instructions.append("""
DEVIL'S ADVOCATE ROLE - Critical Challenge:

Your task is to challenge proposals and find weaknesses:

1. IDENTIFY assumptions: What is being taken for granted?
2. FIND edge cases: Where might this fail?
3. PRESENT alternatives: What other approaches exist?
4. STRESS TEST arguments: What happens in worst-case scenarios?

Always reference specific claims from other agents and explain WHY they might be problematic.
""")
        elif debate_format == DebateFormat.PARLIAMENTARY:
            instructions.append("""
PARLIAMENTARY PROCEDURE:

Use formal structure:
1. Address the moderator: "Madam/Mr. Moderator..."
2. Reference previous speakers: "The honorable [Agent] from [Role] stated..."
3. Use formal language: "I propose...", "I submit that...", "With respect to..."
4. Structure your argument: Opening statement, evidence, conclusion

Follow debate etiquette and maintain formal tone throughout.
""")
        elif debate_format == DebateFormat.FREE_FORM:
            instructions.append("""
FREE-FORM DISCUSSION:

While this is a free-form discussion, please:
- Reference other agents' points by name and turn number
- Build on or respond to specific arguments made
- Maintain a constructive and collaborative tone
- Contribute unique insights rather than merely agreeing
""")

        return "\n\n".join(instructions)

    def build_agent_prompt(
        self, directive: TurnDirective, agent_role: AgentRole
    ) -> list[dict]:
        """Build multi-turn prompt with proper role attribution.

        Uses standard chat format where:
        - System message: identity, rules, metacognitive prompts
        - Conversation history: alternating user/assistant turns
        - Agent's own messages: "assistant" role
        - Others' messages: "user" role

        This leverages the model's natural understanding of conversation roles.

        Args:
            directive: Turn directive with context.
            agent_role: The agent's role.

        Returns:
            List of message dicts in OpenAI format.
        """
        # Build system message with identity and instructions
        system_content = f"You are {agent_role.display_name}, a {agent_role.role}.\n\n"

        # Add metacognitive prompts
        system_content += """ANALYSIS TASK - Before responding, consider:
1. What position has each agent taken?
2. What are the key arguments from each agent?
3. Where do you agree or disagree with each agent?
4. What specific points should you address?
5. What is your unique perspective that adds to this discussion?

"""

        # Add debate format instructions
        if directive.system_instruction:
            system_content += f"{directive.system_instruction}\n\n"

        # Add response instructions
        system_content += """YOUR RESPONSE:
Please share your perspective, making sure to:
- Reference specific agents by name (e.g., "As Alice said in Turn 2...")
- Address the arguments made by other agents
- Build on or counter specific points raised
- Explain your reasoning clearly
"""

        messages = [{"role": "system", "content": system_content}]

        # Replay conversation history as alternating user/assistant turns
        # Keep only recent messages if conversation is very long
        MAX_CONTEXT_MESSAGES = 40  # Total across all speakers
        visible_messages = directive.visible_messages
        if len(visible_messages) > MAX_CONTEXT_MESSAGES:
            omitted_count = len(visible_messages) - MAX_CONTEXT_MESSAGES
            visible_messages = visible_messages[-MAX_CONTEXT_MESSAGES:]
            messages.append({
                "role": "user",
                "content": f"[{omitted_count} earlier messages omitted for brevity]"
            })

        for msg in visible_messages:
            if msg.sender_id == "system":
                # System messages about the topic appear as user turns
                messages.append({
                    "role": "user",
                    "content": f"[Turn {msg.turn_number}] Discussion topic: {msg.content}"
                })
            elif msg.sender_id == agent_role.agent_id:
                # Agent's own messages appear as assistant turns
                messages.append({
                    "role": "assistant",
                    "content": f"[Turn {msg.turn_number}] {msg.content}"
                })
            else:
                # Other agents' messages appear as user turns with attribution
                sender_name = self._get_agent_name(directive.session_id, msg.sender_id)
                sender_role = self._get_agent_role(directive.session_id, sender_name)
                messages.append({
                    "role": "user",
                    "content": f"[Turn {msg.turn_number}] {sender_name} ({sender_role}): {msg.content}"
                })

        # Final user turn prompting response
        messages.append({
            "role": "user",
            "content": "Please share your analysis and response."
        })

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

    def _get_agent_role(self, session_id: str, agent_name: str) -> str:
        """Get role for an agent by display name.

        Args:
            session_id: Session identifier.
            agent_name: Agent display name.

        Returns:
            Agent role (participant, moderator, critic, etc).
        """
        session = self._sessions.get(session_id)
        if session:
            for agent in session.agents.values():
                if agent.display_name == agent_name:
                    return agent.role
        return "participant"

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
        if result.blocks:
            self._cache_store.save(namespaced_agent_id, result.blocks)

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

    async def execute_turn_stream(self, session_id: str) -> AsyncIterator[StreamDelta]:
        """Execute the next turn with token-by-token streaming.

        Streams tokens as they're generated using the scheduler's streaming API.
        After streaming completes, saves the cache and records the message.

        Args:
            session_id: Session identifier.

        Yields:
            StreamDelta objects with accumulated text and token count.

        Raises:
            SessionNotFoundError: If session does not exist.
            InvalidTurnError: If no valid next turn.
        """
        session = self.get_session(session_id)
        directive = self.get_next_turn(session_id)
        agent_role = session.agents[directive.agent_id]

        # Build prompt
        prompt_messages = self.build_agent_prompt(directive, agent_role)

        # Tokenize prompt
        prompt_text = self._format_messages_as_text(prompt_messages)
        prompt_tokens = self._engine.tokenizer.encode(prompt_text)

        # Load agent's cache
        namespaced_agent_id = f"coord_{session_id}_{directive.agent_id}"
        cached_blocks = self._cache_store.load(namespaced_agent_id)

        # Log cache reuse for debugging
        if cached_blocks:
            logger.debug(
                "cache_reuse",
                agent_id=namespaced_agent_id,
                cached_tokens=cached_blocks.total_tokens,
                prompt_tokens=len(prompt_tokens),
            )

        accumulated_text = ""

        # Stream tokens via scheduler
        if self._scheduler is not None:
            async for delta in self._scheduler.submit_and_stream(
                agent_id=namespaced_agent_id,
                prompt_tokens=prompt_tokens,
                cache=cached_blocks,
                max_tokens=512,
                prompt_text=prompt_text,
                temperature=0.7,
                top_p=1.0,
            ):
                accumulated_text = delta.text
                yield delta
        else:
            # Fallback: non-streaming direct generation
            result = await self._generate_direct(
                agent_id=namespaced_agent_id,
                prompt_tokens=prompt_tokens,
                cache=cached_blocks,
                max_tokens=512,
                temperature=0.7,
                top_p=1.0,
            )
            accumulated_text = result.text
            # Yield single delta with final text
            yield StreamDelta(text=result.text, token_count=len(result.text.split()), finish_reason="stop")

        # Post-stream: save cache using engine's stored blocks
        updated_blocks = self._engine.get_agent_blocks(namespaced_agent_id)
        if updated_blocks:
            self._cache_store.save(namespaced_agent_id, updated_blocks)

        # Record message in public channel
        public_channel = next(
            c for c in session.channels.values() if c.channel_type == "public"
        )
        public_channel.add_message(
            sender_id=directive.agent_id,
            content=accumulated_text,
            turn_number=session.current_turn,
        )

        # Advance turn
        session.advance_turn()

        logger.info(
            "coordination_turn_streamed",
            session_id=session_id,
            agent_id=directive.agent_id,
            turn=directive.turn_number,
            text_length=len(accumulated_text),
        )

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

    async def execute_round_stream(
        self, session_id: str
    ) -> AsyncIterator[tuple[str, str, StreamDelta]]:
        """Execute a full round with streaming for each agent.

        Yields tuples of (agent_id, agent_name, StreamDelta) for each token.

        Args:
            session_id: Session identifier.

        Yields:
            Tuples of (agent_id, agent_name, StreamDelta) as tokens are generated.

        Raises:
            SessionNotFoundError: If session does not exist.
        """
        session = self.get_session(session_id)

        for _ in range(len(session.agents)):
            if not session.is_active:
                break

            # Get next speaker info before streaming
            directive = self.get_next_turn(session_id)
            agent_role = session.agents[directive.agent_id]

            # Stream this agent's turn
            async for delta in self.execute_turn_stream(session_id):
                yield (directive.agent_id, agent_role.display_name, delta)

        logger.info("coordination_round_streamed", session_id=session_id)

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
            blocks: any

        return DirectResult(text=result.text, blocks=result.blocks)
