"""Domain value objects (immutable data structures).

Value objects are immutable data structures that represent concepts
from the domain model. They have no identity - two instances with
the same values are considered equal.
"""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GenerationResult:
    """Result of a text generation operation.

    Attributes:
        text: Generated text string.
        tokens: List of generated token IDs.
        cache: Updated KV cache after generation (per-layer).
    """

    text: str
    tokens: list[int]
    cache: list[Any]


@dataclass(frozen=True)
class ModelCacheSpec:
    """Specification for a model's KV cache geometry.

    This value object describes the structure of a model's KV cache,
    including layer count, attention pattern, and block requirements.

    Attributes:
        n_layers: Total number of transformer layers.
        n_kv_heads: Number of key-value attention heads per layer.
        head_dim: Dimension of each attention head.
        block_tokens: Number of tokens per cache block (universal: 256).
        layer_types: Type of each layer ("global" or "sliding_window").
        sliding_window_size: Window size for sliding window layers (e.g., 1024).
    """

    n_layers: int
    n_kv_heads: int
    head_dim: int
    block_tokens: int
    layer_types: list[str]
    sliding_window_size: int | None = None


@dataclass(frozen=True)
class CacheKey:
    """Unique identifier for a cache entry.

    Attributes:
        agent_id: Unique agent identifier.
        model_id: Model identifier (for cache invalidation on model swap).
        prefix_hash: Hash of token prefix (for prefix matching).
    """

    agent_id: str
    model_id: str
    prefix_hash: str
