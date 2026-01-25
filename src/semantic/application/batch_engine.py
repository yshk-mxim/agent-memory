"""Block-pool batch inference engine.

This module implements BlockPoolBatchEngine, a batched inference engine
that wraps mlx_lm's BatchGenerator with block-based KV cache allocation.

The engine provides an async submit/step API for variable-length batch
processing with block-pool memory management.
"""

from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mlx_lm import BatchGenerator  # type: ignore[attr-defined]

    # Sprint 2.5 fix: Import AgentBlocks for type checking
    from semantic.domain.entities import AgentBlocks as AgentBlocksType

from semantic.domain.entities import AgentBlocks, KVBlock
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
        batch_gen_factory: Callable[[Any, Any], Any] | None = None,  # For testing
    ) -> None:
        """Initialize batch engine.

        Args:
            model: Loaded MLX model (from mlx_lm.load).
            tokenizer: Loaded MLX tokenizer (from mlx_lm.load).
            pool: BlockPool for cache memory management.
            spec: ModelCacheSpec describing model cache geometry.
            batch_gen_factory: Optional factory function for creating BatchGenerator.
                Used for testing to inject FakeBatchGenerator. If None, uses mlx_lm.BatchGenerator.

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
        self._batch_gen_factory = batch_gen_factory

        # 3. Initialize batch generator (lazy - created on first submit)
        self._batch_gen: BatchGenerator | None = None

        # 4. Track active requests (UID → agent_id)
        self._active_requests: dict[str, str] = {}

        # 5. Track agent blocks (agent_id → AgentBlocks)
        self._agent_blocks: dict[str, AgentBlocks] = {}

    def submit(
        self,
        agent_id: str,
        prompt: str,
        cache: "AgentBlocksType | None" = None,  # Sprint 2.5 fix: Proper type annotation
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
        kv_cache: Any | None = None  # list[tuple[mx.array, mx.array]] | None

        if cache is not None:
            # Cache provided - reconstruct from blocks
            kv_cache = self._reconstruct_cache(cache)
        else:
            # No cache - allocate blocks for prompt
            # Calculate blocks needed for prompt
            n_blocks_needed = (
                (len(prompt_tokens) + self._spec.block_tokens - 1) // self._spec.block_tokens
            )

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
            # Pre-populate blocks dict to satisfy validation
            # Note: blocks start with token_count=0 (empty), will be filled during generation
            blocks_dict = {0: blocks}  # layer_id 0 (will expand for all layers later)
            agent_blocks = AgentBlocks(
                agent_id=agent_id,
                blocks=blocks_dict,
                total_tokens=0,  # Blocks are empty initially; will update during generation
            )

            # Store agent blocks
            self._agent_blocks[agent_id] = agent_blocks

        # 4. Create BatchGenerator lazily
        if self._batch_gen is None:
            if self._batch_gen_factory is not None:
                # Use injected factory (for testing)
                self._batch_gen = self._batch_gen_factory(self._model, self._tokenizer)
            else:
                # Import and use real mlx_lm BatchGenerator
                from mlx_lm import BatchGenerator  # type: ignore[attr-defined]  # noqa: PLC0415

                self._batch_gen = BatchGenerator(
                    model=self._model,
                    tokenizer=self._tokenizer,
                )

        # 5. Insert into batch
        # Insert prompt into batch
        # Note: If kv_cache is None, batch will create fresh KV cache
        try:
            uids = self._batch_gen.insert(
                prompts=[prompt_tokens],
                max_tokens=max_tokens,
                caches=[kv_cache] if kv_cache is not None else None,
            )
        except Exception as e:
            # Sprint 2.5 fix: If insertion fails, free ALL allocated blocks (not just layer 0)
            if cache is None and agent_id in self._agent_blocks:
                agent_blocks = self._agent_blocks[agent_id]
                # Free all layers to prevent resource leak
                for layer_blocks in agent_blocks.blocks.values():
                    self._pool.free(layer_blocks, agent_id)
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
            return  # Generator returns empty

        # 2. Execute one decode step
        try:
            batch_response = self._batch_gen.next()
        except StopIteration:
            # No more sequences in batch
            return  # Generator returns empty

        # 3. Process finished sequences
        finished_sequences = batch_response.finished

        # 4. Yield CompletedGeneration for each finished sequence
        for finished in finished_sequences:
            uid = finished.uid

            # Get agent_id from tracking
            if uid not in self._active_requests:
                # Sprint 2.5 fix: Log error and attempt cleanup to prevent memory leak
                import logging  # noqa: PLC0415

                logging.error(
                    f"Untracked UID {uid} in batch - possible memory leak. "
                    f"Active UIDs: {list(self._active_requests.keys())}"
                )
                # Try to extract cache anyway to prevent leak in BatchGenerator
                try:
                    if self._batch_gen is not None:
                        self._batch_gen.extract_cache(uid)
                except Exception as e:
                    logging.warning(f"Failed to clean up untracked UID {uid}: {e}")
                continue

            agent_id = self._active_requests[uid]

            # Extract cache and convert to blocks
            blocks = self._extract_cache(uid)

            # Free old prefill blocks (if any)
            if agent_id in self._agent_blocks:
                old_blocks = self._agent_blocks[agent_id]
                # Free all blocks from all layers
                for layer_blocks in old_blocks.blocks.values():
                    # Sprint 2.5 fix: Clear layer_data BEFORE freeing to prevent memory leak
                    for block in layer_blocks:
                        block.layer_data = None  # Force immediate tensor release
                    self._pool.free(layer_blocks, agent_id)

            # Store new blocks
            self._agent_blocks[agent_id] = blocks

            # Create CompletedGeneration
            completion = CompletedGeneration(
                uid=uid,
                text=finished.text,
                blocks=blocks,
                finish_reason=finished.finish_reason,
                token_count=finished.token_count,
            )

            # Clean up tracking
            del self._active_requests[uid]

            # Yield completion
            yield completion

    def _reconstruct_cache(self, agent_blocks: AgentBlocks) -> Any:
        """Reconstruct KVCache from blocks (one-time at restore).

        Return type: list[tuple[mx.array, mx.array]]

        Args:
            agent_blocks: AgentBlocks containing allocated blocks.

        Returns:
            List of (K, V) tensor tuples, one per layer.
            Shape: [(k_0, v_0), (k_1, v_1), ...] where k/v are mx.array
            Each k/v has shape: (n_kv_heads, head_dim, total_seq_len)

        Raises:
            ValueError: If blocks have no layer_data (corrupted cache).

        Notes:
            - Performance target: p95 < 5ms for 32 blocks x 48 layers (EXP-006)
            - One-time cost at cache restore, not per-step overhead
            - Uses mx.concatenate along sequence length axis (axis=2)
            - Forces mx.eval() to ensure immediate execution

        Performance:
            Predicted p95 ~4ms for 8K context based on:
            - 32 blocks x 48 layers = 1,536 concatenate operations
            - Each concat: ~2-3 μs
            - Total: ~3-5ms
        """
        # Import MLX at runtime (avoid import error in unit tests)
        import mlx.core as mx  # noqa: PLC0415

        # 1. Initialize cache list
        cache: list[tuple[Any, Any]] = []

        # 2. For each layer in the model
        for layer_id in range(self._spec.n_layers):
            # Get all blocks for this layer
            layer_blocks = agent_blocks.blocks_for_layer(layer_id)

            # Handle empty layers (shouldn't happen, but be defensive)
            if not layer_blocks:
                cache.append((None, None))
                continue

            # Extract K and V tensors from all blocks
            k_tensors = []
            v_tensors = []
            for block in layer_blocks:
                if block.layer_data is None or "k" not in block.layer_data:
                    raise ValueError(
                        f"Block {block.block_id} for layer {layer_id} has no K/V data"
                    )
                k_tensors.append(block.layer_data["k"])
                v_tensors.append(block.layer_data["v"])

            # Concatenate K and V tensors along sequence length axis (axis=2)
            # Shape: (n_kv_heads, head_dim, total_seq_len)
            k_full = mx.concatenate(k_tensors, axis=2)
            v_full = mx.concatenate(v_tensors, axis=2)

            # Force evaluation (MLX lazy evaluation)
            mx.eval(k_full, v_full)

            # Append to cache
            cache.append((k_full, v_full))

        return cache

    def _extract_cache(self, uid: str) -> AgentBlocks:
        """Extract updated cache from batch and convert to blocks.

        Args:
            uid: Request UID for the finished sequence.

        Returns:
            AgentBlocks with updated cache data (KV cache split into 256-token blocks).

        Raises:
            PoolExhaustedError: If not enough blocks available for extraction.
            ValueError: If cache format invalid.

        Notes:
            - Called after sequence completes
            - Converts KVCache → blocks for persistence
            - Handles partial blocks (last block may not be full)
            - Inverse of _reconstruct_cache()
        """
        # 1. Get agent_id from UID tracking
        if uid not in self._active_requests:
            raise ValueError(f"UID {uid} not found in active requests")
        agent_id = self._active_requests[uid]

        # 2. Extract cache from BatchGenerator
        if self._batch_gen is None:
            raise ValueError("No active batch generator")
        cache = self._batch_gen.extract_cache(uid)

        # 3. Handle empty cache (before importing MLX to avoid crash in tests)
        if not cache or len(cache) == 0 or cache[0][0] is None:
            # Empty cache - return empty AgentBlocks
            return AgentBlocks(agent_id=agent_id, blocks={}, total_tokens=0)

        # Import MLX at runtime (after empty check to avoid crash in tests)
        import mlx.core as mx  # noqa: PLC0415, F401

        # 4. Get total tokens from first layer K tensor shape
        first_k = cache[0][0]  # Shape: [n_kv_heads, head_dim, total_seq_len]
        total_tokens = first_k.shape[2]

        # 5. Calculate blocks needed
        n_blocks = (total_tokens + self._spec.block_tokens - 1) // self._spec.block_tokens

        # Check pool availability before allocating
        total_blocks_needed = n_blocks * self._spec.n_layers
        if self._pool.available_blocks() < total_blocks_needed:
            raise PoolExhaustedError(
                f"Need {total_blocks_needed} blocks for extraction, "
                f"only {self._pool.available_blocks()} available"
            )

        # 6. Create blocks dictionary
        blocks_dict: dict[int, list[KVBlock]] = {}

        # 7. For each layer, split cache into blocks
        for layer_id, (k, v) in enumerate(cache):
            if k is None:
                continue  # Skip empty layers (sliding window)

            layer_blocks = []

            # Split K, V into 256-token chunks
            for block_idx in range(n_blocks):
                start_token = block_idx * self._spec.block_tokens
                end_token = min(start_token + self._spec.block_tokens, total_tokens)

                # Slice tensors [start:end] along seq_len axis (axis=2)
                k_chunk = k[:, :, start_token:end_token]
                v_chunk = v[:, :, start_token:end_token]

                # Allocate block from pool
                allocated_blocks = self._pool.allocate(1, layer_id, agent_id)
                block_id = allocated_blocks[0].block_id

                # Create KVBlock with cache data
                block = KVBlock(
                    block_id=block_id,
                    layer_id=layer_id,
                    token_count=end_token - start_token,
                    layer_data={"k": k_chunk, "v": v_chunk},
                )

                layer_blocks.append(block)

            blocks_dict[layer_id] = layer_blocks

        # 8. Return AgentBlocks with total_tokens
        return AgentBlocks(
            agent_id=agent_id,
            blocks=blocks_dict,
            total_tokens=total_tokens,
        )
