"""Domain entities for block-pool memory management.

Entities represent objects with identity and lifecycle in the domain.
Unlike value objects, entities are mutable and can change over time.

All entities in this module have NO external dependencies - only Python
stdlib and typing imports.
"""

from dataclasses import dataclass, field
from typing import Any

from semantic.domain.errors import AgentBlocksValidationError, BlockValidationError

BLOCK_SIZE_TOKENS = 256


@dataclass
class KVBlock:
    """Single 256-token cache block for one layer.

    A block stores KV cache data for exactly 256 tokens at one layer of the model.
    Blocks are allocated from a shared pool and returned when no longer needed.

    Attributes:
        block_id: Unique identifier for this block within the pool.
        layer_id: Which transformer layer this block belongs to (0-indexed).
        token_count: Number of tokens currently stored (0-256).
        layer_data: Actual KV cache tensors for this layer. Format depends on
            the backend adapter (MLX, PyTorch, etc.). Domain layer treats this
            as opaque data.
        metadata: Optional metadata for debugging (allocation time, owner, etc.).

    Example:
        >>> block = KVBlock(
        ...     block_id=42,
        ...     layer_id=0,
        ...     token_count=256,
        ...     layer_data={"k": ..., "v": ...},  # Actual tensors from adapter
        ... )
        >>> block.is_full()
        True
        >>> block.is_empty()
        False
    """

    block_id: int
    layer_id: int
    token_count: int
    layer_data: Any  # Opaque - actual type depends on backend adapter
    metadata: dict[str, Any] = field(default_factory=dict)

    def is_full(self) -> bool:
        """Check if block is at maximum capacity.

        Returns:
            True if block contains BLOCK_SIZE_TOKENS, False otherwise.
        """
        return self.token_count == BLOCK_SIZE_TOKENS

    def is_empty(self) -> bool:
        """Check if block contains no tokens.

        Returns:
            True if block contains 0 tokens, False otherwise.
        """
        return self.token_count == 0

    def __post_init__(self) -> None:
        """Validate block invariants after construction."""
        if self.block_id < 0:
            raise BlockValidationError(f"block_id must be >= 0, got {self.block_id}")
        if self.layer_id < 0:
            raise BlockValidationError(f"layer_id must be >= 0, got {self.layer_id}")
        if not 0 <= self.token_count <= BLOCK_SIZE_TOKENS:
            raise BlockValidationError(
                f"token_count must be 0-{BLOCK_SIZE_TOKENS}, got {self.token_count}"
            )


@dataclass
class AgentBlocks:
    """Collection of blocks allocated to a single agent.

    An agent may have blocks across multiple layers (e.g., 48 layers for Gemma 3).
    Each layer may have multiple blocks if the agent's context exceeds 256 tokens.

    **Immutability**: After construction, AgentBlocks should be treated as immutable.
    To update an agent's blocks, create a new AgentBlocks instance.

    Attributes:
        agent_id: Unique identifier for the agent owning these blocks.
        blocks: List of blocks allocated to this agent, organized by layer.
            Format: blocks[layer_id] = [block1, block2, ...]
        total_tokens: Total number of tokens stored across all blocks.
        metadata: Optional metadata for debugging (creation time, model, etc.).

    Example:
        >>> agent = AgentBlocks(
        ...     agent_id="agent_123",
        ...     blocks={
        ...         0: [KVBlock(block_id=1, layer_id=0, token_count=256, layer_data={})],
        ...         1: [KVBlock(block_id=2, layer_id=1, token_count=128, layer_data={})],
        ...     },
        ...     total_tokens=384,
        ... )
        >>> agent.num_blocks()
        2
        >>> agent.blocks_for_layer(0)
        [KVBlock(...)]
    """

    agent_id: str
    blocks: dict[int, list[KVBlock]]
    total_tokens: int
    token_sequence: list[int] = field(default_factory=list)  # Actual tokens for prefix matching
    metadata: dict[str, Any] = field(default_factory=dict)

    def num_blocks(self) -> int:
        """Count total number of blocks allocated to this agent.

        Returns:
            Total count of blocks across all layers.
        """
        return sum(len(layer_blocks) for layer_blocks in self.blocks.values())

    def num_layers(self) -> int:
        """Count number of layers with allocated blocks.

        Returns:
            Number of unique layer IDs with blocks.
        """
        return len(self.blocks)

    def blocks_for_layer(self, layer_id: int) -> list[KVBlock]:
        """Get all blocks for a specific layer.

        Args:
            layer_id: The layer to query (0-indexed).

        Returns:
            List of blocks for this layer (empty list if none allocated).
        """
        return self.blocks.get(layer_id, [])

    def common_prefix_length(self, query_tokens: list[int]) -> int:
        """Find length of common prefix between cached tokens and query.

        This is the key method for prefix cache hit detection. Returns the
        number of tokens that match between the cached sequence and the query.

        Args:
            query_tokens: The new token sequence to compare against.

        Returns:
            Number of matching tokens from the start (0 if no match).

        Example:
            >>> # Cache has: [1, 2, 3, 4, 5]
            >>> # Query is:  [1, 2, 3, 6, 7]
            >>> agent.common_prefix_length([1, 2, 3, 6, 7])
            3  # First 3 tokens match
        """
        if not self.token_sequence or not query_tokens:
            return 0

        prefix_len = 0
        for cached_tok, query_tok in zip(self.token_sequence, query_tokens):
            if cached_tok != query_tok:
                break
            prefix_len += 1
        return prefix_len

    def __post_init__(self) -> None:
        """Validate agent blocks invariants after construction."""
        if not self.agent_id:
            raise AgentBlocksValidationError("agent_id cannot be empty")
        if self.total_tokens < 0:
            raise AgentBlocksValidationError(f"total_tokens must be >= 0, got {self.total_tokens}")

        # Validate consistency between total_tokens and blocks
        if not self.blocks:
            # Empty blocks dict: total_tokens must be 0
            if self.total_tokens > 0:
                raise AgentBlocksValidationError(
                    f"total_tokens is {self.total_tokens} but blocks dict is empty"
                )
            return  # Valid empty state

        # Find first non-empty layer (skip layers with empty block lists)
        first_layer_blocks: list[KVBlock] | None = None
        for layer_blocks in self.blocks.values():
            if layer_blocks:  # Skip empty lists
                first_layer_blocks = layer_blocks
                break

        if first_layer_blocks is None:
            # All layers have empty block lists
            if self.total_tokens > 0:
                raise AgentBlocksValidationError(
                    f"total_tokens is {self.total_tokens} but all layer block lists are empty"
                )
            return  # Valid empty state (all layers exist but have no blocks)

        # Validate that total_tokens matches token count in first non-empty layer
        # (all layers store the SAME sequence, so we don't sum across layers)
        computed_total = sum(block.token_count for block in first_layer_blocks)

        if self.total_tokens != computed_total:
            raise AgentBlocksValidationError(
                f"total_tokens ({self.total_tokens}) doesn't match "
                f"sum of block tokens in first layer ({computed_total})"
            )

        # Validate all non-empty layers have the same token count
        for layer_id, layer_blocks in self.blocks.items():
            if not layer_blocks:
                continue  # Skip empty layers
            layer_total = sum(block.token_count for block in layer_blocks)
            if layer_total != self.total_tokens:
                raise AgentBlocksValidationError(
                    f"Layer {layer_id} has {layer_total} tokens, "
                    f"but total_tokens is {self.total_tokens}"
                )
