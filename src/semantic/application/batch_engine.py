"""Block-pool batch inference engine.

This module implements BlockPoolBatchEngine, a batched inference engine
that wraps mlx_lm's BatchGenerator with block-based KV cache allocation.

The engine provides an async submit/step API for variable-length batch
processing with block-pool memory management.
"""

from typing import TYPE_CHECKING, Any, Iterator
from uuid import uuid4

if TYPE_CHECKING:
    import mlx.core as mx
    from mlx_lm import BatchGenerator  # type: ignore[attr-defined]

from semantic.domain.entities import AgentBlocks
from semantic.domain.errors import InvalidRequestError, ModelNotFoundError, PoolExhaustedError
from semantic.domain.services import BlockPool
from semantic.domain.value_objects import CompletedGeneration, ModelCacheSpec


class BlockPoolBatchEngine:
    """Batched inference engine with block-pool memory management.

    Implements GenerationEnginePort for async submit/step pattern.
    Wraps mlx_lm BatchGenerator with block-based KV cache allocation.

    Responsibilities:
    - Allocate blocks for new sequences
    - Reconstruct KVCache from blocks (one-time at restore)
    - Submit requests to batch queue
    - Execute decode steps in batches
    - Extract updated cache back to blocks
    - Free blocks when sequences complete

    Example:
        >>> engine = BlockPoolBatchEngine(model, tokenizer, pool, spec)
        >>> uid = engine.submit("agent_1", "Hello", max_tokens=50)
        >>> for completion in engine.step():
        ...     print(f"{completion.uid}: {completion.text}")
    """

    def __init__(
        self,
        model: Any,  # MLX model
        tokenizer: Any,  # MLX tokenizer
        pool: BlockPool,
        spec: ModelCacheSpec,
    ) -> None:
        """Initialize batch engine.

        Args:
            model: Loaded MLX model (from mlx_lm.load).
            tokenizer: Loaded MLX tokenizer (from mlx_lm.load).
            pool: BlockPool for cache memory management.
            spec: ModelCacheSpec describing model cache geometry.

        Raises:
            ModelNotFoundError: If model or tokenizer is None.
            InvalidRequestError: If pool or spec is None.
        """
        # 1. Validate inputs
        if model is None:
            raise ModelNotFoundError("Model must be loaded before creating engine")
        if tokenizer is None:
            raise ModelNotFoundError("Tokenizer must be loaded before creating engine")
        if pool is None:
            raise InvalidRequestError("BlockPool is required")
        if spec is None:
            raise InvalidRequestError("ModelCacheSpec is required")

        # 2. Store dependencies
        self._model = model
        self._tokenizer = tokenizer
        self._pool = pool
        self._spec = spec

        # 3. Initialize batch generator (lazy - created on first submit)
        self._batch_gen: "BatchGenerator | None" = None

        # 4. Track active requests (UID → agent_id)
        self._active_requests: dict[str, str] = {}

        # 5. Track agent blocks (agent_id → AgentBlocks)
        self._agent_blocks: dict[str, AgentBlocks] = {}

    def submit(
        self,
        agent_id: str,
        prompt: str,
        cache: Any | None = None,  # AgentBlocks (avoid circular import)
        max_tokens: int = 256,
    ) -> str:
        """Submit a generation request to the batch queue.

        Args:
            agent_id: Unique identifier for the agent.
            prompt: Input text to continue.
            cache: Optional pre-built cache (AgentBlocks from previous generation).
            max_tokens: Maximum tokens to generate.

        Returns:
            Request UID for tracking this generation.

        Raises:
            PoolExhaustedError: If no blocks available for allocation.
            InvalidRequestError: If prompt is empty or parameters invalid.
            ModelNotFoundError: If no model is loaded.

        Notes:
            - Non-blocking: returns immediately with UID
            - Actual generation happens during step() calls
            - Multiple submit() calls can be batched together
        """
        # 1. Validate inputs
        if not prompt:
            raise InvalidRequestError("Prompt cannot be empty")
        if max_tokens <= 0:
            raise InvalidRequestError(f"max_tokens must be positive, got {max_tokens}")
        if not agent_id:
            raise InvalidRequestError("agent_id cannot be empty")

        # 2. Tokenize prompt
        prompt_tokens = self._tokenizer.encode(prompt)

        # 3. Handle cache (reconstruct or allocate)
        kv_cache: "list[tuple[mx.array, mx.array]] | None" = None

        if cache is not None:
            # Cache provided - reconstruct from blocks (Day 7 implementation)
            # TODO: Implement _reconstruct_cache()
            # kv_cache = self._reconstruct_cache(cache)
            raise NotImplementedError(
                "Cache reconstruction not yet implemented (Day 7)"
            )
        else:
            # No cache - allocate blocks for prompt
            # Calculate blocks needed for prompt
            n_blocks_needed = (len(prompt_tokens) + self._spec.block_tokens - 1) // self._spec.block_tokens

            # Allocate blocks for first layer (global layer - layer 0)
            # Note: We allocate for layer 0 as placeholder; actual layer-specific
            # allocation happens during decode when we know the cache structure
            try:
                blocks = self._pool.allocate(
                    n_blocks=n_blocks_needed,
                    layer_id=0,
                    agent_id=agent_id,
                )
            except PoolExhaustedError as e:
                raise PoolExhaustedError(
                    f"Failed to allocate {n_blocks_needed} blocks for agent {agent_id}"
                ) from e

            # Create AgentBlocks to track allocation
            agent_blocks = AgentBlocks(
                agent_id=agent_id,
                blocks={},  # Will be populated as blocks are added
                total_tokens=len(prompt_tokens),
            )
            for block in blocks:
                agent_blocks.add_block(block)

            # Store agent blocks
            self._agent_blocks[agent_id] = agent_blocks

        # 4. Create BatchGenerator lazily
        if self._batch_gen is None:
            # Import at runtime (avoid import error in unit tests)
            from mlx_lm import BatchGenerator  # type: ignore[attr-defined]

            self._batch_gen = BatchGenerator(
                model=self._model,
                tokenizer=self._tokenizer,
            )

        # 5. Insert into batch
        # Generate unique UID for this request
        uid = str(uuid4())

        # Insert prompt into batch
        # Note: If kv_cache is None, batch will create fresh KV cache
        try:
            uids = self._batch_gen.insert(
                prompts=[prompt_tokens],
                max_tokens=max_tokens,
                caches=[kv_cache] if kv_cache is not None else None,
            )
        except Exception as e:
            # If insertion fails, free allocated blocks
            if cache is None and agent_id in self._agent_blocks:
                blocks = self._agent_blocks[agent_id].blocks_for_layer(0)
                self._pool.free(blocks, agent_id)
                del self._agent_blocks[agent_id]
            raise InvalidRequestError(f"Failed to insert into batch: {e}") from e

        # Use the UID from BatchGenerator
        actual_uid: str = uids[0]  # BatchGenerator returns list of string UIDs

        # 6. Track UID → agent_id mapping
        self._active_requests[actual_uid] = agent_id

        # 7. Return UID
        return actual_uid

    def step(self) -> Iterator[CompletedGeneration]:
        """Execute one batch decode step and yield completed generations.

        Yields:
            CompletedGeneration for each sequence that finished this step.
            Sequences finish when:
            - EOS token generated (finish_reason="stop")
            - max_tokens limit reached (finish_reason="length")
            - Error occurred (finish_reason="error")

        Notes:
            - Call repeatedly until all in-flight requests complete
            - Non-blocking: returns empty iterator if no completions this step
            - Single-threaded: only one caller should invoke step()
            - Batching window: Waits briefly to collect concurrent submits

        Example:
            >>> engine = BlockPoolBatchEngine(...)
            >>> uid1 = engine.submit("agent_a", "Hello", max_tokens=50)
            >>> uid2 = engine.submit("agent_b", "World", max_tokens=50)
            >>> for completion in engine.step():
            ...     print(f"{completion.uid}: {completion.text[:20]}...")
            ...     if completion.finish_reason == "stop":
            ...         print(f"Completed with {completion.token_count} tokens")
        """
        # 1. Guard: if no active batch, return immediately
        if self._batch_gen is None:
            return iter([])  # Return empty iterator

        # 2. Execute one decode step
        # TODO: Day 8 implementation
        # - Call batch_gen.next() or equivalent
        # - Check for finished sequences
        # - Extract cache and blocks for finished sequences
        # - Yield CompletedGeneration objects
        # - Clean up tracking
        raise NotImplementedError("step() not yet implemented (Day 8)")

    def _reconstruct_cache(self, agent_blocks: AgentBlocks) -> "list[tuple[mx.array, mx.array]]":
        """Reconstruct KVCache from blocks (one-time at restore).

        Args:
            agent_blocks: AgentBlocks containing allocated blocks.

        Returns:
            List of (K, V) tensor tuples, one per layer.

        Notes:
            - Performance target: p95 < 5ms for 32 blocks × 48 layers (EXP-006)
            - One-time cost at cache restore, not per-step overhead
            - Uses mx.concatenate along sequence length axis (axis=2)
            - Forces mx.eval() to ensure immediate execution

        Performance:
            Predicted p95 ~4ms for 8K context based on:
            - 32 blocks × 48 layers = 1,536 concatenate operations
            - Each concat: ~2-3 μs
            - Total: ~3-5ms
        """
        # TODO: Day 7 implementation
        # Algorithm (7 steps from design doc):
        # 1. Initialize cache list
        # 2. For each layer_id in range(n_layers):
        #    - Get layer blocks
        #    - Extract K tensors
        #    - Extract V tensors
        #    - Concatenate K (axis=2)
        #    - Concatenate V (axis=2)
        #    - Force mx.eval()
        #    - Append to cache
        # 3. Return cache
        raise NotImplementedError("_reconstruct_cache() not yet implemented (Day 7)")

    def _extract_cache(self, uid: str) -> AgentBlocks:
        """Extract updated cache from batch and convert to blocks.

        Args:
            uid: Request UID for the finished sequence.

        Returns:
            AgentBlocks with updated cache data.

        Notes:
            - Called after sequence completes
            - Converts KVCache → blocks for persistence
            - Handles partial blocks (last block may not be full)
        """
        # TODO: Day 8 implementation
        # Algorithm (6 steps from design doc):
        # 1. Call batch_gen.extract_cache(uid)
        # 2. Create AgentBlocks container
        # 3. For each layer in cache:
        #    - Get K and V tensors
        #    - Split into chunks (block_tokens size)
        #    - For each chunk:
        #      - Create KVBlock
        #      - Add to AgentBlocks
        # 4. Return AgentBlocks
        raise NotImplementedError("_extract_cache() not yet implemented (Day 8)")
