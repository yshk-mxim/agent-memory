"""Domain services for business logic that doesn't belong to entities.

Services encapsulate domain logic that involves multiple entities or
requires complex coordination. Services are stateless - they operate
on entities passed as parameters.

BlockPool is an exception: it maintains state (the pool of free blocks)
but has no identity - there is only one pool per server instance.
"""

import threading

from semantic.domain.entities import KVBlock
from semantic.domain.errors import (
    BlockOperationError,
    PoolConfigurationError,
    PoolExhaustedError,
)
from semantic.domain.value_objects import ModelCacheSpec


class BlockPool:
    """Manages allocation and deallocation of KV cache blocks.

    The BlockPool maintains a fixed-size pool of blocks that are allocated
    to agents as needed and returned when no longer required. This implements
    block-level memory management similar to vLLM's paged attention.

    Key features:
    - Pre-allocated blocks (avoid runtime allocation overhead)
    - Free list management (O(1) allocate/free)
    - Per-layer-group tracking (global vs sliding window)
    - Memory budgeting (used + available = total)
    - Model hot-swap support (reconfigure without restart)

    Thread safety:
    - THREAD-SAFE: All operations protected by internal lock.
    - Multiple threads can safely call allocate/free concurrently.
    - Lock is acquired for the duration of each operation.

    Attributes:
        spec: Model cache specification (defines layer types, dimensions).
        total_blocks: Total number of blocks in the pool.
        block_tokens: Tokens per block (always 256, per ADR-002).
        free_list: Stack of available block IDs (O(1) pop/push).
        allocated_blocks: Map of block_id -> KVBlock for tracking.
        agent_allocations: Map of agent_id -> set of block_ids.

    Example:
        >>> spec = ModelCacheSpec(
        ...     n_layers=48,
        ...     n_kv_heads=8,
        ...     head_dim=256,
        ...     block_tokens=256,
        ...     layer_types=["global"] * 8 + ["sliding_window"] * 40,
        ...     sliding_window_size=512,
        ... )
        >>> pool = BlockPool(spec=spec, total_blocks=1000)
        >>> pool.available_blocks()
        1000
        >>> blocks = pool.allocate(n_blocks=4, layer_id=0, agent_id="agent_1")
        >>> len(blocks)
        4
        >>> pool.available_blocks()
        996
        >>> pool.free(blocks, agent_id="agent_1")
        >>> pool.available_blocks()
        1000
    """

    def __init__(self, spec: ModelCacheSpec, total_blocks: int) -> None:
        """Initialize the block pool.

        Args:
            spec: Model cache specification.
            total_blocks: Total number of blocks to pre-allocate.

        Raises:
            PoolConfigurationError: If total_blocks <= 0 or spec is invalid.
        """
        if total_blocks <= 0:
            raise PoolConfigurationError(f"total_blocks must be > 0, got {total_blocks}")

        if spec.n_layers <= 0:
            raise PoolConfigurationError(f"spec.n_layers must be > 0, got {spec.n_layers}")

        self.spec = spec
        self.total_blocks = total_blocks
        self.block_tokens = spec.block_tokens

        self._lock = threading.Lock()

        # Free list: stack of available block IDs (LIFO for cache locality)
        self.free_list: list[int] = list(range(total_blocks))

        # Allocated blocks: track all in-use blocks
        self.allocated_blocks: dict[int, KVBlock] = {}

        # Agent allocations: track which blocks belong to which agent
        self.agent_allocations: dict[str, set[int]] = {}

    def allocate(
        self,
        n_blocks: int,
        layer_id: int,
        agent_id: str,
    ) -> list[KVBlock]:
        """Allocate blocks for an agent at a specific layer.

        Thread-safe: Multiple threads can call concurrently.

        Args:
            n_blocks: Number of blocks to allocate.
            layer_id: Which layer these blocks belong to (0-indexed).
            agent_id: Unique agent identifier.

        Returns:
            List of newly allocated blocks.

        Raises:
            PoolExhaustedError: If insufficient blocks available.
            ValueError: If n_blocks <= 0 or layer_id invalid.

        Example:
            >>> pool = BlockPool(spec, total_blocks=100)
            >>> blocks = pool.allocate(n_blocks=2, layer_id=0, agent_id="agent_1")
            >>> len(blocks)
            2
            >>> blocks[0].layer_id
            0
            >>> blocks[0].token_count
            0
        """
        with self._lock:
            if n_blocks <= 0:
                raise BlockOperationError(f"n_blocks must be > 0, got {n_blocks}")

            if layer_id < 0 or layer_id >= self.spec.n_layers:
                raise BlockOperationError(
                    f"layer_id must be 0-{self.spec.n_layers - 1}, got {layer_id}"
                )

            if len(self.free_list) < n_blocks:
                raise PoolExhaustedError(
                    f"Requested {n_blocks} blocks but only {len(self.free_list)} available. "
                    f"Total pool size: {self.total_blocks}, "
                    f"allocated: {len(self.allocated_blocks)}"
                )

            # Allocate from free list
            allocated: list[KVBlock] = []
            for _ in range(n_blocks):
                block_id = self.free_list.pop()
                block = KVBlock(
                    block_id=block_id,
                    layer_id=layer_id,
                    token_count=0,  # Empty block
                    layer_data=None,  # Will be filled by adapter
                    metadata={"agent_id": agent_id},
                )
                allocated.append(block)
                self.allocated_blocks[block_id] = block

            # Track agent allocation
            if agent_id not in self.agent_allocations:
                self.agent_allocations[agent_id] = set()
            self.agent_allocations[agent_id].update(block.block_id for block in allocated)

            return allocated

    def free(self, blocks: list[KVBlock], agent_id: str) -> None:
        """Return blocks to the free list.

        Thread-safe: Multiple threads can call concurrently.

        Args:
            blocks: List of blocks to free.
            agent_id: Agent that owns these blocks.

        Raises:
            BlockOperationError: If block is not allocated or doesn't belong to agent.

        Example:
            >>> pool = BlockPool(spec, total_blocks=100)
            >>> blocks = pool.allocate(n_blocks=2, layer_id=0, agent_id="agent_1")
            >>> pool.available_blocks()
            98
            >>> pool.free(blocks, agent_id="agent_1")
            >>> pool.available_blocks()
            100
        """
        with self._lock:
            for block in blocks:
                # Validate block is allocated
                if block.block_id not in self.allocated_blocks:
                    raise BlockOperationError(
                        f"Block {block.block_id} is not allocated (double-free?)"
                    )

                # Validate block belongs to agent
                if (
                    agent_id not in self.agent_allocations
                    or block.block_id not in self.agent_allocations[agent_id]
                ):
                    raise BlockOperationError(
                        f"Block {block.block_id} does not belong to agent {agent_id}"
                    )

                # Clear layer_data to free memory
                if hasattr(block, 'layer_data'):
                    block.layer_data = None

                # Return to free list
                self.free_list.append(block.block_id)
                del self.allocated_blocks[block.block_id]
                self.agent_allocations[agent_id].discard(block.block_id)

            # Clean up empty agent entries
            if agent_id in self.agent_allocations and not self.agent_allocations[agent_id]:
                del self.agent_allocations[agent_id]

    def free_agent_blocks(self, agent_id: str) -> int:
        """Free all blocks belonging to an agent.

        Thread-safe: Multiple threads can call concurrently.

        Args:
            agent_id: Agent whose blocks should be freed.

        Returns:
            Number of blocks freed.

        Example:
            >>> pool = BlockPool(spec, total_blocks=100)
            >>> pool.allocate(n_blocks=5, layer_id=0, agent_id="agent_1")
            [...]
            >>> freed = pool.free_agent_blocks("agent_1")
            >>> freed
            5
            >>> pool.available_blocks()
            100
        """
        with self._lock:
            if agent_id not in self.agent_allocations:
                return 0

            block_ids = list(self.agent_allocations[agent_id])
            freed_count = 0

            for block_id in block_ids:
                if block_id in self.allocated_blocks:
                    block = self.allocated_blocks[block_id]
                    # Clear layer_data to free memory
                    if hasattr(block, 'layer_data'):
                        block.layer_data = None
                    self.free_list.append(block_id)
                    del self.allocated_blocks[block_id]
                    freed_count += 1

            del self.agent_allocations[agent_id]
            return freed_count

    def used_memory(self) -> int:
        """Calculate total memory used by allocated blocks (bytes).

        Thread-safe: Acquires lock to ensure consistent read.

        Returns:
            Memory in bytes.

        Example:
            >>> spec = ModelCacheSpec(
            ...     n_layers=48,
            ...     n_kv_heads=8,
            ...     head_dim=256,
            ...     block_tokens=256,
            ...     layer_types=["global"] * 48,
            ... )
            >>> pool = BlockPool(spec, total_blocks=100)
            >>> pool.allocate(n_blocks=10, layer_id=0, agent_id="agent_1")
            [...]
            >>> pool.used_memory()
            20971520  # 10 blocks x 2 MB/block
        """
        with self._lock:
            bytes_per_block = self.spec.bytes_per_block_per_layer()
            return len(self.allocated_blocks) * bytes_per_block

    def available_memory(self) -> int:
        """Calculate total memory available (bytes).

        Thread-safe: Acquires lock to ensure consistent read.

        Returns:
            Memory in bytes.
        """
        with self._lock:
            bytes_per_block = self.spec.bytes_per_block_per_layer()
            return len(self.free_list) * bytes_per_block

    def total_memory(self) -> int:
        """Calculate total pool memory capacity (bytes).

        Returns:
            Memory in bytes.

        Note:
            Invariant: used_memory() + available_memory() == total_memory()
        """
        bytes_per_block = self.spec.bytes_per_block_per_layer()
        return self.total_blocks * bytes_per_block

    def available_blocks(self) -> int:
        """Get count of available blocks.

        Thread-safe: Acquires lock to ensure consistent read.

        Returns:
            Number of blocks in free list.
        """
        with self._lock:
            return len(self.free_list)

    def allocated_block_count(self) -> int:
        """Get count of allocated blocks.

        Thread-safe: Acquires lock to ensure consistent read.

        Returns:
            Number of blocks currently in use.
        """
        with self._lock:
            return len(self.allocated_blocks)

    def reconfigure(self, new_spec: ModelCacheSpec) -> None:
        """Reconfigure pool for a different model (hot-swap support).

        This clears all allocations and updates the spec. Used when
        swapping models at runtime.

        Thread-safe: Caller must ensure no concurrent operations during
        reconfiguration. All agents must be drained before calling.

        Args:
            new_spec: New model cache specification.

        Raises:
            RuntimeError: If there are active allocations. All agents must call
                free_agent_blocks() before reconfiguring the pool.

        Example:
            >>> pool = BlockPool(spec1, total_blocks=100)
            >>> # Must drain all agents first
            >>> pool.free_agent_blocks("agent_1")
            >>> pool.reconfigure(spec2)  # Hot-swap to new model
            >>> pool.spec == spec2
            True
            >>> pool.available_blocks()
            100
        """
        with self._lock:
            if self.allocated_blocks:
                raise PoolConfigurationError(
                    f"Cannot reconfigure pool with {len(self.allocated_blocks)} "
                    "active allocations. Drain all agents first."
                )

            # Update spec
            self.spec = new_spec
            self.block_tokens = new_spec.block_tokens

            # Reset free list (all blocks available)
            self.free_list = list(range(self.total_blocks))

            # Clear tracking structures
            self.allocated_blocks.clear()
            self.agent_allocations.clear()

    def max_batch_size(self, tokens_per_agent: int = 256) -> int:
        """Calculate maximum number of agents that can fit in the pool.

        This is a rough estimate assuming uniform token distribution.

        Args:
            tokens_per_agent: Estimated tokens per agent (default: 1 block).

        Returns:
            Maximum concurrent agents.

        Example:
            >>> pool = BlockPool(spec, total_blocks=1000)
            >>> pool.max_batch_size(tokens_per_agent=512)
            500  # 1000 blocks / 2 blocks per agent
        """
        blocks_per_agent = (tokens_per_agent + self.block_tokens - 1) // self.block_tokens
        if blocks_per_agent == 0:
            return self.total_blocks  # Edge case: very small contexts

        # Account for multi-layer allocation (each agent needs blocks at all layers)
        blocks_per_agent *= self.spec.n_layers

        return self.total_blocks // blocks_per_agent if blocks_per_agent > 0 else 0
