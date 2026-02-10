# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Yakov Shkolnikov and contributors
"""Integration tests for ChatTemplateAdapter wired into CoordinationService.

Verifies that model-specific template behavior (DeepSeek priming,
Gemma message merging) flows correctly through the service layer.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_memory.adapters.outbound.chat_template_adapter import ChatTemplateAdapter
from agent_memory.application.coordination_service import CoordinationService
from agent_memory.domain.coordination import (
    AgentRole,
    DebateFormat,
    DecisionMode,
    Topology,
)


def _make_deepseek_tokenizer() -> MagicMock:
    """Create a mock tokenizer that mimics DeepSeek's chat template."""
    tok = MagicMock()
    tok.chat_template = "{% for msg in messages %}'User: '{{msg.content}}'Assistant: '{% endfor %}"
    tok.encode.return_value = [1, 2, 3]
    tok.model_max_length = 32768
    tok.eos_token_id = 2
    return tok


def _make_gemma_tokenizer() -> MagicMock:
    """Create a mock tokenizer that mimics Gemma's chat template."""
    tok = MagicMock()
    tok.chat_template = "<start_of_turn>user\n{{message}}<end_of_turn>\n<start_of_turn>model\n"
    tok.encode.return_value = [10, 20, 30]
    tok.model_max_length = 32768
    tok.eos_token_id = 1
    return tok


@pytest.fixture
def mock_cache_store():
    store = MagicMock()
    store.load.return_value = None
    store.save.return_value = None
    return store


@pytest.fixture
def mock_scheduler():
    scheduler = AsyncMock()
    completion = MagicMock()
    completion.text = "Response text"
    completion.token_count = 5
    completion.finish_reason = "stop"
    completion.blocks = MagicMock()
    scheduler.submit_and_wait.return_value = completion
    return scheduler


# ── DeepSeek template integration ──────────────────────────────────


class TestDeepSeekIntegration:

    @pytest.mark.asyncio
    async def test_deepseek_identity_primer_in_prompt(
        self, mock_cache_store, mock_scheduler
    ) -> None:
        """DeepSeek tokenizer → identity primer appears in agent prompt."""
        tok = _make_deepseek_tokenizer()
        engine = MagicMock()
        engine.tokenizer = tok
        engine.get_agent_blocks.return_value = MagicMock(total_tokens=50)

        adapter = ChatTemplateAdapter()
        service = CoordinationService(
            scheduler=mock_scheduler,
            cache_store=mock_cache_store,
            engine=engine,
            chat_template=adapter,
        )

        agents = [
            AgentRole(agent_id="a", display_name="Warden", role="moderator"),
            AgentRole(agent_id="b", display_name="Prisoner", role="participant"),
        ]

        session = await service.create_session(
            topology=Topology.TURN_BY_TURN,
            debate_format=DebateFormat.FREE_FORM,
            decision_mode=DecisionMode.NONE,
            agents=agents,
            initial_prompt="Begin the interrogation.",
        )

        directive = service.get_next_turn(session.session_id)
        agent_role = session.agents[directive.agent_id]
        messages = service.build_agent_prompt(directive, agent_role)

        # DeepSeek first turn → identity primer
        assistant_msgs = [m for m in messages if m["role"] == "assistant"]
        assert len(assistant_msgs) >= 1
        assert "I'm Warden" in assistant_msgs[0]["content"]

        # System instruction should have English-only rule
        system_msgs = [m for m in messages if m["role"] == "system"]
        assert any("English" in m["content"] for m in system_msgs)

        # No "[Name, respond now.]" cue for DeepSeek
        user_msgs = [m for m in messages if m["role"] == "user"]
        assert not any("respond now" in m["content"] for m in user_msgs)

    @pytest.mark.asyncio
    async def test_deepseek_system_instruction_format(
        self, mock_cache_store, mock_scheduler
    ) -> None:
        """DeepSeek system instruction uses simple rules, not RULES: block."""
        tok = _make_deepseek_tokenizer()
        engine = MagicMock()
        engine.tokenizer = tok

        adapter = ChatTemplateAdapter()
        service = CoordinationService(
            scheduler=mock_scheduler,
            cache_store=mock_cache_store,
            engine=engine,
            chat_template=adapter,
        )

        agent = AgentRole(agent_id="a", display_name="Warden", role="moderator")
        instruction = service._build_system_instruction(agent, DebateFormat.FREE_FORM)

        assert "English only" in instruction
        assert "Chinese" in instruction
        assert "RULES:" not in instruction


# ── Gemma template integration ──────────────────────────────────────


class TestGemmaIntegration:

    @pytest.mark.asyncio
    async def test_gemma_standard_prompt_format(
        self, mock_cache_store, mock_scheduler
    ) -> None:
        """Gemma tokenizer → standard prompt with respond-now cue."""
        tok = _make_gemma_tokenizer()
        engine = MagicMock()
        engine.tokenizer = tok
        engine.get_agent_blocks.return_value = MagicMock(total_tokens=50)

        adapter = ChatTemplateAdapter()
        service = CoordinationService(
            scheduler=mock_scheduler,
            cache_store=mock_cache_store,
            engine=engine,
            chat_template=adapter,
        )

        agents = [
            AgentRole(agent_id="a", display_name="Alice", role="participant"),
            AgentRole(agent_id="b", display_name="Bob", role="critic"),
        ]

        session = await service.create_session(
            topology=Topology.TURN_BY_TURN,
            debate_format=DebateFormat.FREE_FORM,
            decision_mode=DecisionMode.NONE,
            agents=agents,
            initial_prompt="Discuss AI safety.",
        )

        directive = service.get_next_turn(session.session_id)
        agent_role = session.agents[directive.agent_id]
        messages = service.build_agent_prompt(directive, agent_role)

        # Standard models have RULES: block
        system_msgs = [m for m in messages if m["role"] == "system"]
        assert any("RULES:" in m["content"] for m in system_msgs)

        # Standard models have respond-now cue
        user_msgs = [m for m in messages if m["role"] == "user"]
        assert any("respond now" in m["content"] for m in user_msgs)

        # No identity primer for Gemma
        assistant_msgs = [m for m in messages if m["role"] == "assistant"]
        primer_msgs = [m for m in assistant_msgs if "I'm Alice" in m.get("content", "")]
        assert len(primer_msgs) == 0

    @pytest.mark.asyncio
    async def test_gemma_message_merging_detected(
        self, mock_cache_store, mock_scheduler
    ) -> None:
        """ChatTemplateAdapter correctly detects Gemma needs merging."""
        tok = _make_gemma_tokenizer()
        adapter = ChatTemplateAdapter()
        assert adapter.needs_message_merging(tok) is True
        assert adapter.is_deepseek(tok) is False


# ── No adapter (default) ───────────────────────────────────────────


class TestNoAdapter:

    @pytest.mark.asyncio
    async def test_no_adapter_defaults_to_standard(
        self, mock_cache_store, mock_scheduler
    ) -> None:
        """Without chat_template adapter, service uses standard (non-DeepSeek) behavior."""
        engine = MagicMock()
        engine.tokenizer = MagicMock()
        engine.tokenizer.chat_template = None

        service = CoordinationService(
            scheduler=mock_scheduler,
            cache_store=mock_cache_store,
            engine=engine,
            chat_template=None,  # No adapter
        )

        agents = [AgentRole(agent_id="a", display_name="Alice", role="participant")]
        session = await service.create_session(
            topology=Topology.TURN_BY_TURN,
            debate_format=DebateFormat.FREE_FORM,
            decision_mode=DecisionMode.NONE,
            agents=agents,
        )

        directive = service.get_next_turn(session.session_id)
        agent_role = session.agents[directive.agent_id]
        messages = service.build_agent_prompt(directive, agent_role)

        # Should use standard RULES: format
        system_content = messages[0]["content"]
        assert "RULES:" in system_content

        # Should have respond-now cue
        user_msgs = [m for m in messages if m["role"] == "user"]
        assert any("respond now" in m["content"] for m in user_msgs)
