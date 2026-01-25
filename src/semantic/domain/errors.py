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


# NEW-5: Domain validation errors (Sprint 3.5)


class BlockValidationError(SemanticError):
    """Block entity validation failed (block_id, layer_id, token_count out of range)."""


class AgentBlocksValidationError(SemanticError):
    """AgentBlocks validation failed (total_tokens mismatch, invalid structure)."""


class ModelSpecValidationError(SemanticError):
    """ModelCacheSpec validation failed (missing attributes, invalid configuration)."""


class PoolConfigurationError(SemanticError):
    """BlockPool configuration error (invalid total_blocks, invalid spec, reconfiguration failed)."""


class BlockOperationError(SemanticError):
    """Block operation failed (block not found, double free, invalid layer_id, wrong owner)."""


class GenerationError(SemanticError):
    """Generation operation failed (prefill failed, no active generator, request not found)."""
