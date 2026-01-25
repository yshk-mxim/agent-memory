"""Domain value objects (immutable data structures)."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from semantic.domain.errors import ModelSpecValidationError

if TYPE_CHECKING:
    from semantic.domain.entities import AgentBlocks


@dataclass(frozen=True)
class GenerationResult:
    """Result of a text generation operation."""

    text: str
    tokens: list[int]
    cache: list[tuple[Any, Any]]


@dataclass(frozen=True)
class ModelCacheSpec:
    """Specification for a model's KV cache geometry.

    Attributes:
        n_layers: Total number of transformer layers.
        n_kv_heads: Number of key-value attention heads per layer.
        head_dim: Dimension of each attention head.
        block_tokens: Number of tokens per cache block.
        layer_types: Type of each layer ("global" or "sliding_window").
        sliding_window_size: Window size for sliding window layers.
    """

    n_layers: int
    n_kv_heads: int
    head_dim: int
    block_tokens: int
    layer_types: list[str]
    sliding_window_size: int | None = None

    def __post_init__(self) -> None:
        """Validate cache spec invariants."""
        if len(self.layer_types) != self.n_layers:
            raise ModelSpecValidationError(
                f"layer_types length ({len(self.layer_types)}) must equal "
                f"n_layers ({self.n_layers})"
            )

    def bytes_per_block_per_layer(self) -> int:
        """Calculate memory bytes required for one block at one layer."""
        return (
            self.n_kv_heads
            * self.head_dim
            * 2  # K and V
            * 2  # float16 = 2 bytes
            * self.block_tokens
        )

    def max_blocks_for_layer(self, layer_type: str) -> int | None:
        """Calculate maximum blocks needed for a layer type."""
        if layer_type == "global":
            return None

        if layer_type == "sliding_window" and self.sliding_window_size is not None:
            return (self.sliding_window_size + self.block_tokens - 1) // self.block_tokens

        return None


@dataclass(frozen=True)
class CacheKey:
    """Unique identifier for a cache entry."""

    agent_id: str
    model_id: str
    prefix_hash: str


@dataclass(frozen=True)
class CompletedGeneration:
    """Result of a completed async generation request."""

    uid: str
    text: str
    blocks: "AgentBlocks"
    finish_reason: str
    token_count: int
