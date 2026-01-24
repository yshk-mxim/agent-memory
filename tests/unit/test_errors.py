"""Unit tests for domain error hierarchy.

This module validates the domain exception hierarchy and ensures
all custom exceptions are properly defined and can be raised/caught.
"""

import pytest

from semantic.domain.errors import (
    AgentNotFoundError,
    CacheCorruptionError,
    CachePersistenceError,
    IncompatibleCacheError,
    InvalidRequestError,
    ModelNotFoundError,
    ModelSwapError,
    PoolExhaustedError,
    SemanticError,
)


@pytest.mark.unit
class TestSemanticErrorHierarchy:
    """Test the domain exception hierarchy."""

    def test_semantic_error_base_class(self) -> None:
        """Test that SemanticError is the base exception."""
        error = SemanticError("test error")
        assert isinstance(error, Exception)
        assert str(error) == "test error"

    def test_agent_not_found_error(self) -> None:
        """Test AgentNotFoundError inherits from SemanticError."""
        error = AgentNotFoundError("agent_123 not found")
        assert isinstance(error, SemanticError)
        assert isinstance(error, Exception)
        assert str(error) == "agent_123 not found"

    def test_pool_exhausted_error(self) -> None:
        """Test PoolExhaustedError inherits from SemanticError."""
        error = PoolExhaustedError("No blocks available")
        assert isinstance(error, SemanticError)

    def test_cache_corruption_error(self) -> None:
        """Test CacheCorruptionError inherits from SemanticError."""
        error = CacheCorruptionError("Checksum mismatch")
        assert isinstance(error, SemanticError)

    def test_model_swap_error(self) -> None:
        """Test ModelSwapError inherits from SemanticError."""
        error = ModelSwapError("Failed to load model")
        assert isinstance(error, SemanticError)

    def test_cache_persistence_error(self) -> None:
        """Test CachePersistenceError inherits from SemanticError."""
        error = CachePersistenceError("Disk full")
        assert isinstance(error, SemanticError)

    def test_invalid_request_error(self) -> None:
        """Test InvalidRequestError inherits from SemanticError."""
        error = InvalidRequestError("Empty prompt")
        assert isinstance(error, SemanticError)

    def test_model_not_found_error(self) -> None:
        """Test ModelNotFoundError inherits from SemanticError."""
        error = ModelNotFoundError("model_id not found")
        assert isinstance(error, SemanticError)

    def test_incompatible_cache_error(self) -> None:
        """Test IncompatibleCacheError inherits from SemanticError."""
        error = IncompatibleCacheError("Cache model mismatch")
        assert isinstance(error, SemanticError)

    def test_can_catch_with_base_class(self) -> None:
        """Test that all domain errors can be caught with SemanticError."""
        with pytest.raises(SemanticError):
            raise AgentNotFoundError("test")

        with pytest.raises(SemanticError):
            raise PoolExhaustedError("test")

        with pytest.raises(SemanticError):
            raise CacheCorruptionError("test")

    def test_can_catch_specific_error(self) -> None:
        """Test that specific errors can be caught individually."""
        with pytest.raises(AgentNotFoundError) as exc_info:
            raise AgentNotFoundError("specific agent error")

        assert "specific agent error" in str(exc_info.value)

    def test_error_message_preserved(self) -> None:
        """Test that error messages are preserved correctly."""
        message = "This is a detailed error message with context"
        error = PoolExhaustedError(message)
        assert str(error) == message
