"""CoordinationService: orchestrates multi-agent coordination sessions.

Manages structured multi-agent conversations: session lifecycle, prompt
building with inter-agent context, generation routing, and turn sequencing.

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

from semantic.application.ports import ChatTemplatePort

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

    def __init__(
        self,
        scheduler,
        cache_store,
        engine,
        reasoning_extra_tokens: int = 300,
        chat_template: ChatTemplatePort | None = None,
    ) -> None:
        self._scheduler = scheduler
        self._cache_store = cache_store
        self._engine = engine
        self._reasoning_extra_tokens = reasoning_extra_tokens
        self._chat_template = chat_template
        self._sessions: dict[str, CoordinationSession] = {}
        self._agent_name_registry: dict[str, str] = {}
        self._lock = asyncio.Lock()

    def update_engine(self, new_engine) -> None:
        """Update engine reference after model hot-swap.

        Called by admin API after model swap to ensure CoordinationService
        uses the new BatchEngine with the newly loaded model.
        """
        self._engine = new_engine
        if self._scheduler is not None:
            self._scheduler.update_engine(new_engine)
        logger.info("CoordinationService engine updated after model swap")

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

        # Check if DeepSeek model (needs special anti-echo instructions)
        is_deepseek = (
            self._chat_template.is_deepseek(self._engine._tokenizer)
            if self._chat_template
            else False
        )

        if is_deepseek:
            # DeepSeek-specific: Simple rules (assistant priming does the heavy lifting)
            # CRITICAL: DeepSeek is bilingual Chinese/English - must enforce English explicitly
            rules = (
                f"IMPORTANT: You must respond in English only. Do not use Chinese. "
                f"Speak as {name} in first person. "
                f"Do not repeat what others say. "
                f"Do not prefix your response. "
                f"Give one short reply only."
            )
        else:
            # Standard models (Gemma, etc.)
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

        # Detect DeepSeek model for special handling
        is_deepseek = (
            self._chat_template.is_deepseek(self._engine._tokenizer)
            if self._chat_template
            else False
        )

        visible_messages = directive.visible_messages

        # CRITICAL: DeepSeek needs assistant-role priming to establish identity
        # Without this initial assistant message, DeepSeek echoes other speakers
        # or narrates in third person instead of responding as the agent.
        # IMPORTANT: Only add priming on FIRST turn when agent has no prior messages.
        # Adding it on subsequent turns creates consecutive assistant messages which breaks.
        if is_deepseek:
            # Check if this agent has any prior assistant messages in visible history
            has_prior_messages = any(
                msg.sender_id == agent_role.agent_id
                for msg in visible_messages
            )
            if not has_prior_messages:
                # First turn: Add identity-priming assistant message
                # This establishes the agent's voice in first person before any dialogue
                # CRITICAL: Keep it simple - complex priming causes DeepSeek to echo/repeat it
                prime_message = f"I'm {agent_role.display_name}."
                messages.append({"role": "assistant", "content": prime_message})
                logger.debug(f"DeepSeek: Added identity primer for {agent_role.display_name}")
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
                # Prefix with speaker name only (no "said:" to avoid narrative mode)
                messages.append(
                    {"role": "user", "content": f"{sender_name}: {msg.content}"},
                )

        # Short cue to elicit next response without confusing role labels
        if not is_deepseek:
            # Standard models: use explicit prompt
            prompt_text = f"[{agent_role.display_name}, respond now.]"
            messages.append({
                "role": "user",
                "content": prompt_text,
            })
        # DeepSeek: No final cue message.
        # The DeepSeek template closes EVERY assistant message with <EOS>, so
        # an assistant message like "Name:" becomes a completed empty turn:
        #   Assistant: Name:<EOS>Assistant:
        # The name cue is wasted — the model starts fresh from "Assistant:".
        # Instead, rely on: (1) system prompt identity ("Your name is X"),
        # (2) first-turn priming ("I'm X."), and (3) conversation history
        # where the agent's own prior messages appear as assistant role.
        # add_generation_prompt=True appends "Assistant:" for generation.

        # DEBUG: Log the full prompt being sent
        logger.info(
            f"[PROMPT_DEBUG] agent={agent_role.agent_id} messages_count={len(messages)}"
        )
        for i, msg in enumerate(messages):
            content_preview = msg["content"][:200] if len(msg["content"]) > 200 else msg["content"]
            logger.info(
                f"  [{i}] role={msg['role']} content={repr(content_preview)}"
            )

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

        Uses shared chat completion logic (same as OpenAI API) to ensure consistent
        output quality with no space stripping or other tokenization artifacts.

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

        # Build prompt messages (standard chat format)
        prompt_messages = self.build_agent_prompt(directive, agent_role)

        # Generate using SHARED logic (same as OpenAI API)
        # This ensures no space stripping and consistent output quality
        from semantic.application.chat_completion_service import generate_chat_completion

        namespaced_agent_id = self._resolve_cache_key(session_id, directive.agent_id)
        gen_max_tokens = self._get_generation_max_tokens()

        is_deepseek = (
            self._chat_template.is_deepseek(self._engine._tokenizer)
            if self._chat_template
            else False
        )

        # DeepSeek sampling: T=0 causes deterministic echo loops (commit 0acecd4).
        # T>=0.3 causes spacing corruption (concatenated words).
        # T=0.2 balances variety vs coherence: less echo than 0.1, no spacing issues.
        # (T=0.1 was original sweet spot but caused heavy echo in multi-turn dialogue.)
        temperature = 0.3
        top_p = 0.95
        top_k = 64

        # DeepSeek: inject agent name into the token stream after "Assistant:"
        # so the model knows which character to generate as (see option 3 in
        # chat_completion_service.py for details on the EOS closure problem).
        gen_prefix = f"{agent_role.display_name}:" if is_deepseek else None

        result = await generate_chat_completion(
            messages=prompt_messages,
            batch_engine=self._engine,
            cache_store=self._cache_store,
            scheduler=self._scheduler,
            agent_id=namespaced_agent_id,
            max_tokens=gen_max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            generation_prefix=gen_prefix,
        )

        # Strip runaway continuation (model generating fake turns)
        all_names = self._all_known_agent_names(session_id)
        clean_text = self._clean_agent_response(
            result["text"],  # Clean text from shared generation logic
            sender_name=agent_role.display_name,
            all_agent_names=all_names,
        )

        # Add message to public channel
        # Skip empty messages to prevent corrupting conversation history
        public_channel = next(c for c in session.channels.values() if c.channel_type == "public")
        if not clean_text.strip():
            logger.warning(
                "empty_generation agent_id=%s turn=%d - skipping history update",
                directive.agent_id,
                session.current_turn,
            )
            # Create a placeholder message for API response, but don't add to channel
            from semantic.domain.coordination import ChannelMessage
            message = ChannelMessage(
                message_id="",
                channel_id=public_channel.channel_id,
                sender_id=directive.agent_id,
                content="",
                turn_number=session.current_turn,
                metadata={"empty_generation": True},
            )
        else:
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
            text_length=len(result["text"]),
        )

        return message

    async def execute_turn_stream(self, session_id: str) -> AsyncIterator[StreamDelta]:
        """Execute the next turn with token-by-token streaming.

        Streams tokens as they're generated using the scheduler's streaming API.
        After streaming completes, saves the cache and records the message.

        Yields a final StreamDelta with finish_reason="cleaned" containing the
        post-processed text (runaway continuation stripped). This delta is a
        **replacement** — consumers must use it as the authoritative final text,
        not append it to previously accumulated raw text.

        Args:
            session_id: Session identifier.

        Yields:
            StreamDelta objects with accumulated text and token count.
            The last yielded delta has finish_reason="cleaned" and its text
            field contains the cleaned replacement text.

        Raises:
            SessionNotFoundError: If session does not exist.
            InvalidTurnError: If no valid next turn.
        """
        session = self.get_session(session_id)
        directive = self.get_next_turn(session_id)
        agent_role = session.agents[directive.agent_id]

        # Build prompt
        prompt_messages = self.build_agent_prompt(directive, agent_role)

        # DeepSeek: inject agent name into token stream (see option 3 comments)
        is_deepseek = (
            self._chat_template.is_deepseek(self._engine._tokenizer)
            if self._chat_template
            else False
        )
        gen_prefix = f"{agent_role.display_name}:" if is_deepseek else None

        # Tokenize using model's chat template for proper turn boundaries
        logger.info("before_tokenize_call", num_messages=len(prompt_messages))
        prompt_tokens, prompt_text = self._tokenize_chat_messages(
            prompt_messages, generation_prefix=gen_prefix,
        )
        logger.info("after_tokenize_call", num_tokens=len(prompt_tokens), prompt_text_len=len(prompt_text))

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

        # Sampling: same for all models (see execute_turn for rationale)
        temperature = 0.3
        top_p = 0.95
        top_k = 64

        # Stream tokens via scheduler
        # GPT-OSS needs more tokens for analysis channel before final
        gen_max_tokens = self._get_generation_max_tokens()
        if self._scheduler is not None:
            async for delta in self._scheduler.submit_and_stream(
                agent_id=namespaced_agent_id,
                prompt_tokens=prompt_tokens,
                cache=cached_blocks,
                max_tokens=gen_max_tokens,
                prompt_text=prompt_text,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            ):
                accumulated_text = delta.text
                yield delta
        else:
            # Fallback: non-streaming direct generation
            result = await self._generate_direct(
                agent_id=namespaced_agent_id,
                prompt_tokens=prompt_tokens,
                cache=cached_blocks,
                max_tokens=gen_max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
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

        # Yield final delta with cleaned text so adapter can use it in turn_complete.
        # PROTOCOL: This delta is a REPLACEMENT, not an append. Consumers must
        # treat finish_reason="cleaned" as the authoritative final text and
        # discard previously accumulated raw text.
        yield StreamDelta(
            text=clean_text,
            token_count=len(clean_text.split()),
            finish_reason="cleaned",
        )

        # Record message in public channel
        # Skip empty messages to prevent corrupting conversation history
        public_channel = next(c for c in session.channels.values() if c.channel_type == "public")
        if not clean_text.strip():
            logger.warning(
                "empty_generation_streaming agent_id=%s turn=%d - skipping history update",
                directive.agent_id,
                session.current_turn,
            )
            # Don't add empty message to channel history
        else:
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

        Handles four classes of artifacts:
        1. Gemma3-style: model continues generating fake turns (User:/System:/etc.)
        2. GPT-OSS-style: model emits internal channel/reasoning markers
        3. Name/role echoing: model prefixes response with its own name or role
        4. DeepSeek-style: meta-narration, inline cross-agent gen, spacing loss
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

        # Strip echoed role/name prefix at start of response (handles both "Name:" and "Name said:")
        prefixes = ["Assistant", "System", "User", "You"]
        if sender_name:
            prefixes.append(re.escape(sender_name))
        for name in (all_agent_names or []):
            prefixes.append(re.escape(name))
        prefix_pattern = "|".join(dict.fromkeys(prefixes))  # dedupe, preserve order
        text = re.sub(rf"^(?:{prefix_pattern})(?:\s+said)?:\s?", "", text)

        # Strip echoed turn cue (e.g., "[Danny, respond now.]" or "Danny, what do you say?")
        text = re.sub(r"^\[.+?, respond now\.\]\s*", "", text)
        text = re.sub(r"^.+?, what do you say\?\s*", "", text, flags=re.IGNORECASE)

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
        # Find the EARLIEST marker occurrence (not first in list)
        min_idx = len(text)
        for marker in stop_markers:
            idx = text.find(marker)
            if 0 < idx < min_idx:
                min_idx = idx
        if min_idx < len(text):
            text = text[:min_idx]

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

    def _merge_consecutive_messages(self, messages: list[dict]) -> list[dict]:
        """Merge consecutive messages with the same role.

        Llama 3.1 and other models expect strict user/assistant alternation.
        In multi-agent scenarios, we may have:
          user: "Bob: Hello"
          user: "Carol: Hi"
          user: "[Alice, respond now.]"

        This merges them into a single user message to maintain model compatibility.
        System messages are always kept separate at the start.

        Args:
            messages: List of chat messages with role/content.

        Returns:
            List with consecutive same-role messages merged.
        """
        if not messages:
            return []

        merged: list[dict] = []
        for msg in messages:
            if not merged:
                merged.append(dict(msg))
                continue

            prev = merged[-1]
            # Merge consecutive same-role messages (except system)
            # Use single newline - chat template adds paragraph breaks
            if msg["role"] == prev["role"] and msg["role"] != "system":
                prev["content"] = prev["content"] + "\n" + msg["content"]
            else:
                merged.append(dict(msg))

        return merged

    def _get_generation_max_tokens(self) -> int:
        """Get max_tokens including reasoning headroom."""
        base_tokens = 200
        return base_tokens + self._reasoning_extra_tokens

    def _tokenize_chat_messages(
        self, messages: list[dict], generation_prefix: str | None = None,
    ) -> tuple[list[int], str]:
        """Tokenize using model's chat template for proper turn boundaries.

        Models like Gemma3 require special turn markers (e.g. <start_of_turn>)
        to maintain identity in multi-agent conversations. Falls back to raw
        text tokenization if no chat template is available.

        Message merging and template kwargs are delegated to the injected
        ChatTemplatePort, keeping model-specific logic in adapters.

        Args:
            messages: Chat messages to tokenize.
            generation_prefix: Optional prefix to inject after "Assistant:"
                (e.g. "Warden:" for DeepSeek identity signaling).

        Returns:
            Tuple of (token_ids, prompt_text_for_logging).
        """
        logger.info("_tokenize_chat_messages_called", num_messages=len(messages))
        prompt_text = self._format_messages_as_text(messages)

        tokenizer = self._engine.tokenizer

        has_template = hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None) is not None
        logger.info("tokenize_debug_start", has_template=has_template)

        if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
            try:
                needs_merge = (
                    self._chat_template.needs_message_merging(tokenizer)
                    if self._chat_template
                    else False
                )
                if needs_merge:
                    messages_for_template = self._merge_consecutive_messages(messages)
                    logger.info(
                        "message_merge original=%d merged=%d",
                        len(messages),
                        len(messages_for_template),
                    )
                    for m in messages_for_template:
                        if m["role"] == "user":
                            content_preview = m["content"][:200].replace("\n", "\\n")
                            logger.info("merged_user_content preview=%s...", content_preview)
                            break
                else:
                    messages_for_template = messages

                template_kwargs = (
                    self._chat_template.get_template_kwargs(tokenizer)
                    if self._chat_template
                    else {"tokenize": True, "add_generation_prompt": True}
                )

                # CRITICAL FIX: Get the ACTUAL templated text that will be tokenized
                # This ensures prompt_text matches what's in the cache for prefix matching
                template_kwargs_text = template_kwargs.copy()
                template_kwargs_text["tokenize"] = False
                logger.info("get_templated_text", kwargs=template_kwargs_text)
                templated_text = tokenizer.apply_chat_template(
                    messages_for_template,
                    **template_kwargs_text,
                )
                logger.info("got_templated_text",
                           text_type=type(templated_text).__name__,
                           text_len=len(templated_text) if hasattr(templated_text, '__len__') else 0)

                # Now tokenize the templated text
                logger.info("before_tokenize", kwargs=template_kwargs)
                tokens = tokenizer.apply_chat_template(
                    messages_for_template,
                    **template_kwargs,
                )
                logger.info("after_tokenize",
                           tokens_type=type(tokens).__name__,
                           is_list=isinstance(tokens, list),
                           tokens_len=len(tokens) if hasattr(tokens, '__len__') else 0)
                if isinstance(tokens, list):
                    # Inject generation prefix (DeepSeek identity, see option 3)
                    if (
                        generation_prefix
                        and isinstance(templated_text, str)
                        and templated_text.rstrip().endswith("Assistant:")
                    ):
                        suffix = " " + generation_prefix
                        templated_text = templated_text + suffix
                        suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)
                        if hasattr(suffix_tokens, "ids"):
                            suffix_tokens = list(suffix_tokens.ids)
                        tokens = list(tokens) + suffix_tokens
                        logger.info(
                            "injected_generation_prefix",
                            prefix=generation_prefix,
                            extra_tokens=len(suffix_tokens),
                        )

                    logger.info("template_fix_applied",
                        templated_text_len=len(templated_text) if isinstance(templated_text, str) else 0,
                        tokens_count=len(tokens),
                        preview=templated_text[:100] if isinstance(templated_text, str) else str(type(templated_text))
                    )
                    # Return tokens AND the actual templated text (not raw message text)
                    return tokens, templated_text
            except Exception as e:
                logger.debug("Chat template failed: %s, falling back to raw text", e)

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
