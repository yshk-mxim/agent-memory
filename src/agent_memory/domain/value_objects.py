# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Yakov Shkolnikov and contributors
"""Domain value objects (immutable data structures)."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from agent_memory.domain.errors import ModelSpecValidationError

if TYPE_CHECKING:
    from agent_memory.domain.entities import AgentBlocks


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
        head_dim: Dimension of each K attention head (also V when symmetric).
        block_tokens: Number of tokens per cache block.
        layer_types: Type of each layer ("global" or "sliding_window").
        sliding_window_size: Window size for sliding window layers.
        kv_bits: KV cache quantization bits (4 or 8, None = FP16).
        kv_group_size: Quantization group size.
        v_head_dim: Dimension of each V attention head. None means same as
            head_dim (symmetric K/V, the common case). Set explicitly for
            architectures like DeepSeek V2 MLA where K and V have different
            dimensions (e.g. K=192, V=128).
    """

    n_layers: int
    n_kv_heads: int
    head_dim: int
    block_tokens: int
    layer_types: list[str]
    sliding_window_size: int | None = None
    kv_bits: int | None = 4
    kv_group_size: int = 64
    v_head_dim: int | None = None

    @property
    def effective_v_head_dim(self) -> int:
        """V head dimension, defaulting to head_dim when symmetric."""
        return self.v_head_dim if self.v_head_dim is not None else self.head_dim

    def __post_init__(self) -> None:
        """Validate cache spec invariants."""
        if self.n_layers <= 0:
            raise ModelSpecValidationError(f"n_layers must be > 0, got {self.n_layers}")
        if self.n_kv_heads <= 0:
            raise ModelSpecValidationError(f"n_kv_heads must be > 0, got {self.n_kv_heads}")
        if self.head_dim <= 0:
            raise ModelSpecValidationError(f"head_dim must be > 0, got {self.head_dim}")
        if self.v_head_dim is not None and self.v_head_dim <= 0:
            raise ModelSpecValidationError(f"v_head_dim must be > 0, got {self.v_head_dim}")
        if self.block_tokens <= 0:
            raise ModelSpecValidationError(f"block_tokens must be > 0, got {self.block_tokens}")

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

        # Validate kv_group_size is positive and power of 2
        if self.kv_group_size <= 0:
            raise ModelSpecValidationError(f"kv_group_size must be > 0, got {self.kv_group_size}")
        if self.kv_group_size & (self.kv_group_size - 1) != 0:
            raise ModelSpecValidationError(
                f"kv_group_size must be a power of 2, got {self.kv_group_size}"
            )

    def bytes_per_block_per_layer(self) -> int:
        """Calculate memory bytes required for one block at one layer.

        Accounts for quantization if kv_bits is set:
        - FP16 (kv_bits=None or 16): 2 bytes per element
        - Q8 (kv_bits=8): ~1.0625 bytes (1 + scales/group_size)
        - Q4 (kv_bits=4): ~0.5625 bytes (0.5 + scales/group_size)

        Quantization adds scales+biases overhead of ~4 bytes per group (2 scales + 2 biases).

        Supports asymmetric K/V dimensions (e.g. DeepSeek V2 MLA: K=192, V=128).
        """
        k_elements = self.n_kv_heads * self.head_dim * self.block_tokens
        v_elements = self.n_kv_heads * self.effective_v_head_dim * self.block_tokens
        total_elements = k_elements + v_elements

        if self.kv_bits is None or self.kv_bits == 16:
            # FP16: 2 bytes per element
            return total_elements * 2

        # Quantized: main weights + scales + biases overhead
        # Weights: kv_bits / 8 bytes per element
        weight_bytes = (total_elements * self.kv_bits) // 8

        # Scales and biases: 2 bytes each per group (float16)
        k_groups = (k_elements + self.kv_group_size - 1) // self.kv_group_size
        v_groups = (v_elements + self.kv_group_size - 1) // self.kv_group_size
        total_groups = k_groups + v_groups
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
    """Single token delta pushed to streaming clients.

    Streaming protocol:
        1. During generation, deltas have finish_reason=None and text contains
           the full accumulated raw text so far. Consumers extract new content
           via ``delta.text[len(previous_text):]``.

        2. When generation completes, a delta with finish_reason="stop" or
           "length" is yielded.

        3. After the final generation delta, execute_turn_stream yields one
           extra delta with finish_reason="cleaned". This delta's text is a
           **replacement** (not an append) containing the post-processed text
           with runaway continuation stripped. Consumers MUST treat this as
           the authoritative final text and NOT concatenate it with previously
           accumulated raw text.
    """

    text: str  # Full accumulated text so far (or replacement text if finish_reason="cleaned")
    token_count: int
    finish_reason: str | None = None


@dataclass(frozen=True)
class StepOneResult:
    """Result from a single decode step for one sequence."""

    uid: str
    text: str
    token_count: int
    finish_reason: str | None = None
    completion: CompletedGeneration | None = None
