"""CoordinationService: orchestrates multi-agent coordination sessions.

The coordination service manages structured multi-agent conversations by:
1. Creating and tracking coordination sessions
2. Building prompts that include inter-agent context
3. Routing generation requests through the scheduler
4. Managing turn sequencing based on topology

Architecture layer: application service.
No MLX / infrastructure imports — interacts with adapters through ports.
"""

import asyncio
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass
from uuid import uuid4

import structlog

from semantic.domain.coordination import (
    AgentLifecycle,
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

MAX_CONTEXT_MESSAGES = 40  # Total across all speakers


@dataclass
class _DirectResult:
    """Result from direct engine generation (scheduler bypass)."""

    text: str
    blocks: object


class CoordinationService:
    """Orchestrates multi-agent coordination sessions.

    Dependencies (injected):
      - scheduler: ConcurrentScheduler (for inference)
      - cache_store: AgentCacheStore (for per-agent KV caches)
      - engine: BlockPoolBatchEngine (for tokenization)
    """

    def __init__(self, scheduler, cache_store, engine) -> None:
        """Initialize with injected scheduler, cache store, and engine."""
        self._scheduler = scheduler
        self._cache_store = cache_store
        self._engine = engine
        self._sessions: dict[str, CoordinationSession] = {}
        self._agent_name_registry: dict[str, str] = {}
        self._lock = asyncio.Lock()

    @staticmethod
    def _agent_cache_key(session_id: str, agent_id: str) -> str:
        """Construct namespaced cache key for coordination agent.

        Args:
            session_id: Session identifier.
            agent_id: Agent identifier.

        Returns:
            Namespaced cache key (e.g., "coord_<session_id>_<agent_id>").
        """
        return f"coord_{session_id}_{agent_id}"

    def _resolve_cache_key(self, session_id: str, agent_id: str) -> str:
        """Resolve cache key, using persistent key for permanent agents.

        Permanent agents in sessions with a persistent_cache_prefix get
        identity-based cache keys that survive session deletion, enabling
        KV cache reuse across phases.
        """
        session = self._sessions.get(session_id)
        if session and session.persistent_cache_prefix:
            agent_role = session.agents.get(agent_id)
            if agent_role and agent_role.lifecycle == AgentLifecycle.PERMANENT:
                key = f"persist_{session.persistent_cache_prefix}_{agent_id}"
                logger.debug("persistent_cache_key agent=%s key=%s", agent_id, key)
                return key
        return self._agent_cache_key(session_id, agent_id)

    async def create_session(
        self,
        topology: Topology,
        debate_format: DebateFormat,
        decision_mode: DecisionMode,
        agents: list[AgentRole],
        initial_prompt: str = "",
        per_agent_prompts: dict[str, str] | None = None,
        max_turns: int = 0,
        persistent_cache_prefix: str = "",
        prior_agent_messages: dict[str, list[dict[str, str]]] | None = None,
    ) -> CoordinationSession:
        """Create a new coordination session with agents and topology.

        Args:
            topology: Communication topology (turn_by_turn, broadcast, etc.).
            debate_format: Debate structure (free_form, structured, etc.).
            decision_mode: Decision-making mode (voting, consensus, etc.).
            agents: List of AgentRole objects defining participants.
            initial_prompt: Optional initial prompt/topic for discussion.
            per_agent_prompts: Per-agent private context keyed by display_name.
                Each entry becomes a system message visible only to that agent.
            max_turns: Maximum turns before auto-termination (0 = unlimited).
            persistent_cache_prefix: If set, permanent agents use identity-based
                cache keys that persist across sessions (for cross-phase memory).
            prior_agent_messages: Per-agent messages from prior phases, keyed by
                agent_id. Injected into the channel (private, filtered from API)
                so that build_agent_prompt produces token sequences that extend
                the agent's persistent KV cache via prefix matching.

        Returns:
            The created CoordinationSession.
        """
        async with self._lock:
            session_id = f"coord_{uuid4().hex[:12]}"

            # Create public channel
            public_channel = Channel(
                channel_id=f"{session_id}_public",
                channel_type="public",
                participant_ids=frozenset(a.agent_id for a in agents),
            )

            # Register agent names (for cross-phase name resolution)
            for a in agents:
                self._agent_name_registry[a.agent_id] = a.display_name

            # Inject prior phase messages (private to each agent).
            # The API filters these out (non-empty visible_to), so transcripts
            # stay clean. But build_agent_prompt includes them, making the token
            # sequence extend the agent's persistent KV cache for prefix matching.
            if prior_agent_messages:
                for target_agent_id, messages in prior_agent_messages.items():
                    for msg in messages:
                        sender_id = msg.get("sender_id", "system")
                        sender_name = msg.get("sender_name", "")
                        content = msg.get("content", "")
                        if not content:
                            continue
                        if sender_name and sender_id != "system":
                            self._agent_name_registry[sender_id] = sender_name
                        public_channel.add_message(
                            sender_id=sender_id,
                            content=content,
                            turn_number=0,
                            visible_to=frozenset([target_agent_id]),
                        )

            # Add shared initial prompt as public system message
            if initial_prompt:
                public_channel.add_message(
                    sender_id="system",
                    content=initial_prompt,
                    turn_number=0,
                    visible_to=frozenset(),  # Empty = public
                )

            # Add per-agent private context (visible only to each agent)
            if per_agent_prompts:
                name_to_id = {a.display_name: a.agent_id for a in agents}
                for display_name, prompt in per_agent_prompts.items():
                    agent_id = name_to_id.get(display_name)
                    if agent_id and prompt.strip():
                        public_channel.add_message(
                            sender_id="system",
                            content=prompt,
                            turn_number=0,
                            visible_to=frozenset([agent_id]),
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
                persistent_cache_prefix=persistent_cache_prefix,
            )

            self._sessions[session_id] = session
            logger.info(
                "coordination_session_created",
                session_id=session_id,
                topology=topology.value,
                num_agents=len(agents),
                persistent_cache=bool(persistent_cache_prefix),
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

    async def delete_session(self, session_id: str) -> None:
        """Delete a coordination session and clean up ephemeral agent caches.

        Persistent caches (for permanent agents with persistent_cache_prefix)
        are preserved — use delete_persistent_caches() to clear them.

        Args:
            session_id: Session identifier.

        Raises:
            SessionNotFoundError: If session does not exist.
        """
        async with self._lock:
            if session_id not in self._sessions:
                raise SessionNotFoundError(f"Session {session_id} not found")
            session = self._sessions[session_id]

            for agent_id in session.agents:
                resolved_key = self._resolve_cache_key(session_id, agent_id)
                session_key = self._agent_cache_key(session_id, agent_id)
                if resolved_key != session_key:
                    # Persistent cache — don't delete (survives session lifecycle)
                    continue
                self._cache_store.delete(session_key)

            del self._sessions[session_id]
            logger.info("coordination_session_deleted", session_id=session_id)

    def delete_persistent_caches(self, prefix: str) -> int:
        """Delete all persistent caches matching a prefix.

        Used by "Reset All" to clear cross-phase agent memory.

        Args:
            prefix: The persistent_cache_prefix used when creating sessions.

        Returns:
            Number of caches deleted.
        """
        target_prefix = f"persist_{prefix}_"
        deleted = 0
        all_agents = self._cache_store.list_all_agents()
        for entry in all_agents:
            agent_id = entry.get("agent_id", "")
            if agent_id.startswith(target_prefix):
                self._cache_store.delete(agent_id)
                deleted += 1
        if deleted:
            logger.info("persistent_caches_deleted", prefix=prefix, count=deleted)
        return deleted

    def list_sessions(self) -> list[CoordinationSession]:
        """List all active coordination sessions.

        Returns:
            List of CoordinationSession objects.
        """
        return list(self._sessions.values())

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
        visible_messages = self._filter_visible_messages(public_channel.messages, next_speaker)

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

    def _build_system_instruction(self, agent_role: AgentRole, debate_format: DebateFormat) -> str:
        """Build system instruction with clear identity and anti-drift rules.

        Args:
            agent_role: The agent's role in the coordination.
            debate_format: The debate format being used.

        Returns:
            System instruction string.
        """
        name = agent_role.display_name

        if agent_role.system_prompt:
            identity = f"Your name is {name}. {agent_role.system_prompt}"
        else:
            identity = f"Your name is {name}. You are a {agent_role.role}."

        rules = (
            "RULES: "
            f"You are {name} and nobody else. "
            "Respond in first person as yourself. "
            "Never generate dialogue for other characters. "
            "Never prefix your response with any name or label. "
            "Give one short reply only."
        )

        style = self._debate_style_hint(debate_format, agent_role.role)

        parts = [identity, rules]
        if style:
            parts.append(style)
        return "\n\n".join(parts)

    @staticmethod
    def _debate_style_hint(debate_format: DebateFormat, role: str) -> str:
        """Return a short behavioral hint for the debate format."""
        if debate_format == DebateFormat.STRUCTURED:
            return "Cite others' points, provide evidence, and state your position with reasoning."
        if debate_format == DebateFormat.SOCRATIC:
            return "Respond primarily with questions that probe others' reasoning."
        if debate_format == DebateFormat.DEVILS_ADVOCATE and role == "critic":
            return "Challenge proposals and find weaknesses."
        if debate_format == DebateFormat.PARLIAMENTARY:
            return "Use formal structure: address the moderator and reference speakers formally."
        return ""

    def build_agent_prompt(self, directive: TurnDirective, agent_role: AgentRole) -> list[dict]:
        """Build multi-turn prompt using chat roles as identity signals.

        Identity design (extends naturally to N participants):
          system  → "You are Alice." — establishes who THIS agent is
          assistant → agent's own past messages (plain text, no name prefix)
          user    → everyone else, prefixed with "Name: " to distinguish speakers

        The chat API's role system is the identity mechanism:
          - assistant = "me" (model's own voice, never prefixed with a name)
          - user = "not me" (Name: prefix tells the model who is speaking)
          - system = rules and context

        Args:
            directive: Turn directive with context.
            agent_role: The agent's role.

        Returns:
            List of message dicts in chat format.
        """
        system_content = directive.system_instruction or ""

        messages = [{"role": "system", "content": system_content}]

        visible_messages = directive.visible_messages
        if len(visible_messages) > MAX_CONTEXT_MESSAGES:
            visible_messages = visible_messages[-MAX_CONTEXT_MESSAGES:]

        for msg in visible_messages:
            if msg.sender_id == "system":
                messages.append({"role": "user", "content": msg.content})
            elif msg.sender_id == agent_role.agent_id:
                messages.append({"role": "assistant", "content": msg.content})
            else:
                sender_name = self._get_agent_name(
                    directive.session_id,
                    msg.sender_id,
                )
                messages.append(
                    {"role": "user", "content": f"{sender_name}: {msg.content}"},
                )

        # Short cue to elicit next response without confusing role labels
        messages.append({
            "role": "user",
            "content": f"[{agent_role.display_name}, respond now.]",
        })

        return messages

    def _get_agent_name(self, session_id: str, agent_id: str) -> str:
        """Get display name for an agent.

        Checks session agents first, then falls back to a global registry
        populated from prior-phase message injection. This ensures agents
        from prior phases are referenced by display name (not raw ID),
        which is critical for cross-phase KV cache prefix matching.
        """
        session = self._sessions.get(session_id)
        if session and agent_id in session.agents:
            return session.agents[agent_id].display_name
        return self._agent_name_registry.get(agent_id, agent_id)

    def _all_known_agent_names(self, session_id: str) -> list[str]:
        """Get ALL known agent display names: session agents + name registry.

        Used for stop markers so model-generated text from agents in
        prior phases (e.g., Warden in The Yard) is properly truncated.
        """
        names: set[str] = set()
        session = self._sessions.get(session_id)
        if session:
            for a in session.agents.values():
                names.add(a.display_name)
        for name in self._agent_name_registry.values():
            names.add(name)
        names.discard("")
        return list(names)

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

        # Tokenize using model's chat template for proper turn boundaries
        prompt_tokens, prompt_text = self._tokenize_chat_messages(prompt_messages)

        # Load agent's cache (persistent key for permanent agents, session-scoped otherwise)
        namespaced_agent_id = self._resolve_cache_key(session_id, directive.agent_id)
        cached_blocks = self._cache_store.load(namespaced_agent_id)

        if cached_blocks:
            logger.info(
                "cache_reuse agent_id=%s cached_tokens=%d prompt_tokens=%d",
                namespaced_agent_id,
                cached_blocks.total_tokens,
                len(prompt_tokens),
            )

        # Submit to scheduler (or fall back to direct engine if no scheduler)
        if self._scheduler is not None:
            result = await self._scheduler.submit_and_wait(
                agent_id=namespaced_agent_id,
                prompt_tokens=prompt_tokens,
                cache=cached_blocks,
                max_tokens=200,
                prompt_text=prompt_text,
                temperature=1.0,
                top_p=0.95,
                top_k=64,
            )
        else:
            # Direct engine path (no scheduler available)
            result = await self._generate_direct(
                agent_id=namespaced_agent_id,
                prompt_tokens=prompt_tokens,
                cache=cached_blocks,
                max_tokens=200,
                temperature=1.0,
                top_p=0.95,
                top_k=64,
            )

        # Cache invariant: result.blocks contains KV state for prompt + generated
        # tokens. The blocks.token_sequence stores ONLY the prompt tokens (used for
        # prefix matching on the next turn). The generated text is added to the
        # channel as a message, and build_agent_prompt() will include it as an
        # assistant-role message next turn — the engine's prefix matcher reuses
        # the cached prompt KV and only recomputes the new tokens.
        if result.blocks:
            self._cache_store.save(namespaced_agent_id, result.blocks)

        # Strip runaway continuation (model generating fake turns)
        all_names = self._all_known_agent_names(session_id)
        clean_text = self._clean_agent_response(
            result.text,
            sender_name=agent_role.display_name,
            all_agent_names=all_names,
        )

        # Add message to public channel
        public_channel = next(c for c in session.channels.values() if c.channel_type == "public")
        message = public_channel.add_message(
            sender_id=directive.agent_id,
            content=clean_text,
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

        # Tokenize using model's chat template for proper turn boundaries
        prompt_tokens, prompt_text = self._tokenize_chat_messages(prompt_messages)

        # Load agent's cache (persistent key for permanent agents)
        namespaced_agent_id = self._resolve_cache_key(session_id, directive.agent_id)
        cached_blocks = self._cache_store.load(namespaced_agent_id)

        # Log cache reuse for debugging
        if cached_blocks:
            logger.info(
                "cache_reuse agent_id=%s cached_tokens=%d prompt_tokens=%d",
                namespaced_agent_id,
                cached_blocks.total_tokens,
                len(prompt_tokens),
            )

        accumulated_text = ""

        # Stream tokens via scheduler
        if self._scheduler is not None:
            async for delta in self._scheduler.submit_and_stream(
                agent_id=namespaced_agent_id,
                prompt_tokens=prompt_tokens,
                cache=cached_blocks,
                max_tokens=200,
                prompt_text=prompt_text,
                temperature=1.0,
                top_p=0.95,
                top_k=64,
            ):
                accumulated_text = delta.text
                yield delta
        else:
            # Fallback: non-streaming direct generation
            result = await self._generate_direct(
                agent_id=namespaced_agent_id,
                prompt_tokens=prompt_tokens,
                cache=cached_blocks,
                max_tokens=200,
                temperature=1.0,
                top_p=0.95,
                top_k=64,
            )
            accumulated_text = result.text
            # Yield single delta with final text
            yield StreamDelta(
                text=result.text, token_count=len(result.text.split()), finish_reason="stop"
            )

        # Cache invariant: same as execute_turn() — blocks contain prompt + generated
        # KV state, but token_sequence is prompt-only for prefix matching.
        updated_blocks = self._engine.get_agent_blocks(namespaced_agent_id)
        if updated_blocks:
            self._cache_store.save(namespaced_agent_id, updated_blocks)

        # Strip runaway continuation (model generating fake turns)
        all_names = self._all_known_agent_names(session_id)
        clean_text = self._clean_agent_response(
            accumulated_text,
            sender_name=agent_role.display_name,
            all_agent_names=all_names,
        )

        # Yield final delta with cleaned text so adapter can use it in turn_complete
        yield StreamDelta(
            text=clean_text,
            token_count=len(clean_text.split()),
            finish_reason="cleaned",
        )

        # Record message in public channel
        public_channel = next(c for c in session.channels.values() if c.channel_type == "public")
        public_channel.add_message(
            sender_id=directive.agent_id,
            content=clean_text,
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

    @staticmethod
    def _clean_agent_response(
        text: str,
        sender_name: str = "",
        all_agent_names: list[str] | None = None,
    ) -> str:
        """Strip runaway generation and internal model markers from responses.

        Handles three classes of artifacts:
        1. Gemma3-style: model continues generating fake turns (User:/System:/etc.)
        2. GPT-OSS-style: model emits internal channel/reasoning markers
        3. Name/role echoing: model prefixes response with its own name or role
        """
        # GPT-OSS: extract final channel content if present
        final_marker = "<|channel|>final<|message|>"
        if final_marker in text:
            last_idx = text.rfind(final_marker)
            text = text[last_idx + len(final_marker):]
            end_idx = text.find("<|end|>")
            if end_idx > 0:
                text = text[:end_idx]

        # Strip special tokens and bare role markers
        text = re.sub(r"<\|[a-z_]+\|>", "", text)
        text = re.sub(r"(?m)^assistant$", "", text)
        text = re.sub(r"\bassistant(?=[A-Z])", "", text)

        # Normalize before pattern matching
        text = text.strip()

        # Strip echoed role/name prefix at start of response
        prefixes = ["Assistant", "System", "User", "You"]
        if sender_name:
            prefixes.append(re.escape(sender_name))
        for name in (all_agent_names or []):
            prefixes.append(re.escape(name))
        prefix_pattern = "|".join(dict.fromkeys(prefixes))  # dedupe, preserve order
        text = re.sub(rf"^(?:{prefix_pattern}):\s?", "", text)

        # Strip echoed turn cue (e.g., "[Danny, respond now.]")
        text = re.sub(r"^\[.+?, respond now\.\]\s*", "", text)

        # Strip echoed instruction/rule fragments (Gemma3 artifact)
        text = re.sub(
            r"^(?:(?:Do not|Don'?t) (?:include|incorporate|respond|present|narrate|write|talk)"
            r"[^\n.]*[.\n]?\s*)+",
            "",
            text,
        )
        text = re.sub(r"(?:FREE-FORM DISCUSSION|STRUCTURED DEBATE|SOCRATIC METHOD):?\s*", "", text)

        # Re-strip and re-check prefix after instruction removal may expose new prefix
        text = text.strip()
        text = re.sub(rf"^(?:{prefix_pattern}):\s?", "", text)

        # Truncate at first sign of fake turn continuation
        stop_markers = [
            "\nUser:", "\nuser:",
            "\nYou:", "\nyou:",
            "\nSystem:", "\nsystem:",
            "\nAssistant:", "\nassistant:",
            "\n<start_of_turn>", "\n<end_of_turn>",
        ]
        # Add ALL agent names as stop markers (catches cross-agent generation)
        agent_names = set(all_agent_names or [])
        if sender_name:
            agent_names.add(sender_name)
        for name in agent_names:
            stop_markers.append(f"\n{name}:")
        for marker in stop_markers:
            idx = text.find(marker)
            if idx > 0:
                text = text[:idx]

        # Clean up extra whitespace from marker removal
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _format_messages_as_text(self, messages: list[dict]) -> str:
        """Format message list as plain text for logging.

        Args:
            messages: List of message dicts.

        Returns:
            Human-readable text string (for logging, not tokenization).
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

    def _tokenize_chat_messages(self, messages: list[dict]) -> tuple[list[int], str]:
        """Tokenize using model's chat template for proper turn boundaries.

        Models like Gemma3 require special turn markers (e.g. <start_of_turn>)
        to maintain identity in multi-agent conversations. Falls back to raw
        text tokenization if no chat template is available.

        Returns:
            Tuple of (token_ids, prompt_text_for_logging).
        """
        prompt_text = self._format_messages_as_text(messages)

        tokenizer = self._engine.tokenizer
        if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
            try:
                tokens = tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                )
                if isinstance(tokens, list):
                    return tokens, prompt_text
            except Exception:
                logger.debug("Chat template failed, falling back to raw text")

        return tokenizer.encode(prompt_text), prompt_text

    async def _generate_direct(
        self,
        agent_id: str,
        prompt_tokens: list[int],
        cache,
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int = 0,
    ):
        """Generate text using batch engine directly (fallback when no scheduler).

        Args:
            agent_id: Agent identifier.
            prompt_tokens: Tokenized prompt.
            cache: Cached blocks from previous turns.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Top-p sampling parameter.
            top_k: Top-k sampling parameter.

        Returns:
            Generation result with text and cache.
        """
        # Submit request to batch engine
        uid = self._engine.submit(
            agent_id=agent_id,
            prompt="",  # Empty string since we provide prompt_tokens
            cache=cache,
            max_tokens=max_tokens,
            prompt_tokens=prompt_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
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

        return _DirectResult(text=result.text, blocks=result.blocks)
