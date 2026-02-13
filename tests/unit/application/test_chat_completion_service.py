# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Yakov Shkolnikov and contributors
"""Unit tests for chat_completion_service.

Verifies that generate_chat_completion does not prematurely invalidate the hot cache.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_memory.application.chat_completion_service import generate_chat_completion


@pytest.fixture
def mock_batch_engine():
    """Create a mock batch engine."""
    engine = MagicMock()
    tokenizer = MagicMock()
    tokenizer.encode.return_value = [1, 2, 3, 4, 5]
    engine.tokenizer = tokenizer
    engine.get_agent_blocks.return_value = MagicMock()
    return engine


@pytest.fixture
def mock_cache_store():
    """Create a mock cache store."""
    store = MagicMock()
    store.load.return_value = MagicMock(total_tokens=100)  # Cache hit
    store.save.return_value = None
    return store


@pytest.fixture
def mock_scheduler():
    """Create a mock scheduler that returns a completion."""
    scheduler = AsyncMock()
    completion = MagicMock()
    completion.text = "Generated response"
    completion.token_count = 10
    completion.finish_reason = "stop"
    scheduler.submit_and_wait.return_value = completion
    return scheduler


class TestNoInvalidateHot:
    """Verify invalidate_hot is never called during generate_chat_completion."""

    @pytest.mark.asyncio
    async def test_no_premature_invalidation(
        self, mock_batch_engine, mock_cache_store, mock_scheduler
    ) -> None:
        """invalidate_hot should never be called by generate_chat_completion."""
        with patch(
            "agent_memory.application.chat_completion_service.tokenize_with_chat_template",
            return_value=([1, 2, 3], "templated text"),
        ):
            await generate_chat_completion(
                messages=[{"role": "user", "content": "Hello"}],
                batch_engine=mock_batch_engine,
                cache_store=mock_cache_store,
                scheduler=mock_scheduler,
                agent_id="test_agent",
            )

        mock_cache_store.invalidate_hot.assert_not_called()

    @pytest.mark.asyncio
    async def test_save_called_after_generation(
        self, mock_batch_engine, mock_cache_store, mock_scheduler
    ) -> None:
        """save() should be called after generation completes."""
        with patch(
            "agent_memory.application.chat_completion_service.tokenize_with_chat_template",
            return_value=([1, 2, 3], "templated text"),
        ):
            result = await generate_chat_completion(
                messages=[{"role": "user", "content": "Hello"}],
                batch_engine=mock_batch_engine,
                cache_store=mock_cache_store,
                scheduler=mock_scheduler,
                agent_id="test_agent",
            )

        mock_cache_store.save.assert_called_once()
        assert result["text"] == "Generated response"


# ── Generation prefix injection ─────────────────────────────────────


class TestGenerationPrefix:
    """Tests for generation_prefix token stream injection."""

    @pytest.mark.asyncio
    async def test_prefix_injected_when_ends_with_assistant(
        self, mock_batch_engine, mock_cache_store, mock_scheduler
    ) -> None:
        """generation_prefix appends tokens when templated text ends with 'Assistant:'."""
        mock_batch_engine.tokenizer.encode.return_value = [42, 43]

        with patch(
            "agent_memory.application.chat_completion_service.tokenize_with_chat_template",
            return_value=([1, 2, 3], "some text\nAssistant:"),
        ):
            result = await generate_chat_completion(
                messages=[{"role": "user", "content": "Hello"}],
                batch_engine=mock_batch_engine,
                cache_store=mock_cache_store,
                scheduler=mock_scheduler,
                agent_id="test_agent",
                generation_prefix="Warden:",
            )

        # Scheduler should have received extended tokens (original + suffix)
        call_kwargs = mock_scheduler.submit_and_wait.call_args[1]
        prompt_tokens = call_kwargs["prompt_tokens"]
        # Original [1,2,3] + suffix tokens [42,43]
        assert prompt_tokens == [1, 2, 3, 42, 43]
        # Templated text should include the prefix
        assert "Warden:" in call_kwargs["prompt_text"]

    @pytest.mark.asyncio
    async def test_prefix_not_injected_when_no_assistant_ending(
        self, mock_batch_engine, mock_cache_store, mock_scheduler
    ) -> None:
        """generation_prefix is a no-op if text doesn't end with 'Assistant:'."""
        with patch(
            "agent_memory.application.chat_completion_service.tokenize_with_chat_template",
            return_value=([1, 2, 3], "some text without assistant ending"),
        ):
            await generate_chat_completion(
                messages=[{"role": "user", "content": "Hello"}],
                batch_engine=mock_batch_engine,
                cache_store=mock_cache_store,
                scheduler=mock_scheduler,
                agent_id="test_agent",
                generation_prefix="Warden:",
            )

        call_kwargs = mock_scheduler.submit_and_wait.call_args[1]
        # Tokens should be unmodified
        assert call_kwargs["prompt_tokens"] == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_prefix_handles_encoding_object(
        self, mock_batch_engine, mock_cache_store, mock_scheduler
    ) -> None:
        """Handles fast tokenizer Encoding objects (have .ids attr)."""
        # Simulate fast tokenizer returning Encoding-like object
        encoding = MagicMock()
        encoding.ids = [42, 43]
        mock_batch_engine.tokenizer.encode.return_value = encoding

        with patch(
            "agent_memory.application.chat_completion_service.tokenize_with_chat_template",
            return_value=([1, 2, 3], "text\nAssistant:"),
        ):
            await generate_chat_completion(
                messages=[{"role": "user", "content": "Hello"}],
                batch_engine=mock_batch_engine,
                cache_store=mock_cache_store,
                scheduler=mock_scheduler,
                agent_id="test_agent",
                generation_prefix="Name:",
            )

        call_kwargs = mock_scheduler.submit_and_wait.call_args[1]
        assert call_kwargs["prompt_tokens"] == [1, 2, 3, 42, 43]


# ── Direct engine path (no scheduler) ──────────────────────────────


class TestDirectEnginePath:
    """Tests for the scheduler=None fallback path."""

    @pytest.mark.asyncio
    async def test_direct_path_uses_engine(self, mock_batch_engine, mock_cache_store) -> None:
        """When scheduler is None, falls back to batch_engine.submit + step loop."""
        completion = MagicMock()
        completion.text = "Direct response"
        completion.token_count = 5
        completion.finish_reason = "stop"
        completion.uid = "uid_1"

        mock_batch_engine.submit.return_value = "uid_1"
        mock_batch_engine.step.return_value = [completion]

        with patch(
            "agent_memory.application.chat_completion_service.tokenize_with_chat_template",
            return_value=([1, 2, 3], "templated text"),
        ):
            result = await generate_chat_completion(
                messages=[{"role": "user", "content": "Hello"}],
                batch_engine=mock_batch_engine,
                cache_store=mock_cache_store,
                scheduler=None,  # No scheduler
                agent_id="test_agent",
            )

        assert result["text"] == "Direct response"
        mock_batch_engine.submit.assert_called_once()
        mock_batch_engine.step.assert_called()

    @pytest.mark.asyncio
    async def test_none_completion_raises(
        self, mock_batch_engine, mock_cache_store, mock_scheduler
    ) -> None:
        """If completion is None, raises RuntimeError."""
        mock_scheduler.submit_and_wait.return_value = None

        with patch(
            "agent_memory.application.chat_completion_service.tokenize_with_chat_template",
            return_value=([1, 2, 3], "templated text"),
        ):
            with pytest.raises(RuntimeError, match="Generation failed"):
                await generate_chat_completion(
                    messages=[{"role": "user", "content": "Hello"}],
                    batch_engine=mock_batch_engine,
                    cache_store=mock_cache_store,
                    scheduler=mock_scheduler,
                    agent_id="test_agent",
                )


# ── Cache behavior ──────────────────────────────────────────────────


class TestCacheBehavior:
    """Tests for cache load/save flow."""

    @pytest.mark.asyncio
    async def test_cache_miss_proceeds(self, mock_batch_engine, mock_scheduler) -> None:
        """Cache miss (load returns None) still generates successfully."""
        store = MagicMock()
        store.load.return_value = None  # Cache miss
        store.save.return_value = None

        with patch(
            "agent_memory.application.chat_completion_service.tokenize_with_chat_template",
            return_value=([1, 2], "text"),
        ):
            result = await generate_chat_completion(
                messages=[{"role": "user", "content": "Hi"}],
                batch_engine=mock_batch_engine,
                cache_store=store,
                scheduler=mock_scheduler,
                agent_id="agent_x",
            )

        assert result["text"] == "Generated response"
        # Cache should still be saved even on miss
        store.save.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_blocks_skips_save(
        self, mock_batch_engine, mock_cache_store, mock_scheduler
    ) -> None:
        """If get_agent_blocks returns None, save is not called."""
        mock_batch_engine.get_agent_blocks.return_value = None

        with patch(
            "agent_memory.application.chat_completion_service.tokenize_with_chat_template",
            return_value=([1, 2], "text"),
        ):
            await generate_chat_completion(
                messages=[{"role": "user", "content": "Hi"}],
                batch_engine=mock_batch_engine,
                cache_store=mock_cache_store,
                scheduler=mock_scheduler,
                agent_id="agent_x",
            )

        mock_cache_store.save.assert_not_called()
