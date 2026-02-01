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
        kv_bits: KV cache quantization bits (4 or 8, None = FP16).
        kv_group_size: Quantization group size.
    """

    n_layers: int
    n_kv_heads: int
    head_dim: int
    block_tokens: int
    layer_types: list[str]
    sliding_window_size: int | None = None
    kv_bits: int | None = 4
    kv_group_size: int = 64

    def __post_init__(self) -> None:
        """Validate cache spec invariants."""
        if len(self.layer_types) != self.n_layers:
            raise ModelSpecValidationError(
                f"layer_types length ({len(self.layer_types)}) must equal "
                f"n_layers ({self.n_layers})"
            )

        # Validate kv_bits is a supported quantization level
        valid_kv_bits = {None, 4, 8, 16}
        if self.kv_bits not in valid_kv_bits:
            raise ModelSpecValidationError(
                f"kv_bits must be one of {valid_kv_bits}, got {self.kv_bits}"
            )

        # Validate kv_group_size is positive
        if self.kv_group_size <= 0:
            raise ModelSpecValidationError(f"kv_group_size must be > 0, got {self.kv_group_size}")

    def bytes_per_block_per_layer(self) -> int:
        """Calculate memory bytes required for one block at one layer.

        Accounts for quantization if kv_bits is set:
        - FP16 (kv_bits=None or 16): 2 bytes per element
        - Q8 (kv_bits=8): ~1.0625 bytes (1 + scales/group_size)
        - Q4 (kv_bits=4): ~0.5625 bytes (0.5 + scales/group_size)

        Quantization adds scales+biases overhead of ~2 bytes per group element.
        """
        elements_per_kv = self.n_kv_heads * self.head_dim * self.block_tokens
        total_elements = elements_per_kv * 2  # K and V

        if self.kv_bits is None or self.kv_bits == 16:
            # FP16: 2 bytes per element
            return total_elements * 2

        # Quantized: main weights + scales + biases overhead
        # Weights: kv_bits / 8 bytes per element
        weight_bytes = (total_elements * self.kv_bits) // 8

        # Scales and biases: 2 bytes each per group (float16)
        # Number of groups = elements / group_size (per K and per V)
        groups_per_kv = (elements_per_kv + self.kv_group_size - 1) // self.kv_group_size
        total_groups = groups_per_kv * 2  # K and V
        scales_bytes = total_groups * 2  # float16 scales
        biases_bytes = total_groups * 2  # float16 biases

        return weight_bytes + scales_bytes + biases_bytes

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


@dataclass(frozen=True)
class StreamDelta:
    """Single token delta pushed to streaming clients."""

    text: str  # Full accumulated text so far
    token_count: int
    finish_reason: str | None = None


@dataclass
class StepOneResult:
    """Result from a single decode step for one sequence."""

    uid: str
    text: str
    token_count: int
    finish_reason: str | None = None
    completion: CompletedGeneration | None = None
