"""Domain exception hierarchy.

All domain-level errors inherit from SemanticError.
This allows clean exception handling at adapter boundaries.
"""


class SemanticError(Exception):
    """Base exception for all domain errors."""


class AgentNotFoundError(SemanticError):
    """Requested agent does not exist in the cache store."""


class PoolExhaustedError(SemanticError):
    """Block pool has no available blocks for allocation."""


class CacheCorruptionError(SemanticError):
    """Cached data failed integrity check (checksum mismatch, invalid format)."""


class ModelSwapError(SemanticError):
    """Model hot-swap failed (load error, incompatible spec)."""


class CachePersistenceError(SemanticError):
    """Failed to save or load cache from disk."""


class InvalidRequestError(SemanticError):
    """Request validation failed (invalid agent_id, empty prompt, etc)."""


class ModelNotFoundError(SemanticError):
    """Requested model not available in registry."""


class IncompatibleCacheError(SemanticError):
    """Cache model tag does not match current model."""
