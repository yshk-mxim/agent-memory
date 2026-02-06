"""Unit tests for chat_completion_service.

Verifies that generate_chat_completion does not prematurely invalidate the hot cache.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from semantic.application.chat_completion_service import generate_chat_completion


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
            "semantic.application.chat_completion_service.tokenize_with_chat_template",
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
            "semantic.application.chat_completion_service.tokenize_with_chat_template",
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
