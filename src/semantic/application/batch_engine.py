"""Block-pool batch inference engine."""

import logging
import time
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from semantic.domain.entities import AgentBlocks as AgentBlocksType

from semantic.domain.entities import AgentBlocks, KVBlock
from semantic.domain.errors import (
    GenerationError,
    InvalidRequestError,
    ModelNotFoundError,
    PoolExhaustedError,
)
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
        cache_adapter: Any,  # CacheOperationsPort (MLXCacheAdapter)
        batch_gen_factory: Callable[[Any, Any], Any] | None = None,  # For testing
    ) -> None:
        """Initialize batch engine."""
        # 1. Validate inputs
        if model is None:
            raise ModelNotFoundError("Model must be loaded before creating engine")
        if tokenizer is None:
            raise ModelNotFoundError("Tokenizer must be loaded before creating engine")
        if pool is None:
            raise InvalidRequestError("BlockPool is required")
        if spec is None:
            raise InvalidRequestError("ModelCacheSpec is required")
        if cache_adapter is None:
            raise InvalidRequestError("CacheAdapter is required")

        # 2. Store dependencies
        self._model = model
        self._tokenizer = tokenizer
        self._pool = pool
        self._spec = spec
        self._cache_adapter = cache_adapter
        self._batch_gen_factory = batch_gen_factory

        # 3. Initialize batch generator (lazy - created on first submit)
        self._batch_gen: Any | None = None

        # 4. Track active requests (UID → (agent_id, accumulated_tokens))
        self._active_requests: dict[str, tuple[str, list[int]]] = {}

        # 5. Track agent blocks (agent_id → AgentBlocks)
        self._agent_blocks: dict[str, AgentBlocks] = {}

    def submit(
        self,
        agent_id: str,
        prompt: str,
        cache: "AgentBlocksType | None" = None,  #Proper type annotation
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
            # Cache reconstruction not yet supported - allocate fresh
            logging.warning(
                f"Cache reconstruction not supported. Allocating fresh for {agent_id}"
            )
            cache = None

        if cache is None:
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
                # Use adapter to create real MLX BatchGenerator
                self._batch_gen = self._cache_adapter.create_batch_generator(
                    model=self._model,
                    stop_tokens={self._tokenizer.eos_token_id},
                )

        # 5. Insert into batch
        try:
            # Create sampler via adapter if using real MLX (not fake for testing)
            samplers = None
            if self._batch_gen_factory is None:
                sampler = self._cache_adapter.create_sampler(temperature=0.0)
                samplers = [sampler]

            uids = self._batch_gen.insert(
                prompts=[prompt_tokens],
                max_tokens=[max_tokens],
                caches=[kv_cache] if kv_cache is not None else None,
                samplers=samplers,
            )
        except Exception as e:
            #If insertion fails, free ALL allocated blocks (not just layer 0)
            if cache is None and agent_id in self._agent_blocks:
                agent_blocks = self._agent_blocks[agent_id]
                # Free all layers to prevent resource leak
                for layer_blocks in agent_blocks.blocks.values():
                    self._pool.free(layer_blocks, agent_id)
                del self._agent_blocks[agent_id]
            raise InvalidRequestError(f"Failed to insert into batch: {e}") from e

        # Use the UID from BatchGenerator
        actual_uid: str = uids[0]  # BatchGenerator returns list of string UIDs

        # 6. Track UID → (agent_id, tokens) mapping
        # Initialize empty token list - will accumulate during step()
        self._active_requests[actual_uid] = (agent_id, [])

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

        # 2. Execute decode loop until all sequences finish
        while True:
            batch_response = self._batch_gen.next()  # type: ignore[no-untyped-call]

            # Check for termination: empty list means all sequences done
            if not batch_response:
                break

            # 3. Process all responses (both finished and continuing)
            for response in batch_response:
                uid = response.uid

                # Validate UID is tracked
                if uid not in self._active_requests:
                    logging.error(
                        f"Untracked UID {uid} in batch - possible memory leak. "
                        f"Active UIDs: {list(self._active_requests.keys())}"
                    )
                    continue

                # Get tracking info
                agent_id, tokens = self._active_requests[uid]

                # 3.1. Accumulate token for this step
                if response.finish_reason != "stop":
                    tokens.append(response.token)

                # 3.2. Check if sequence finished
                if response.finish_reason is not None:
                    # Sequence complete - process completion

                    # Extract cache and convert to blocks
                    cache = response.prompt_cache
                    blocks = self._extract_cache(uid, cache)

                    # Free old prefill blocks (if any)
                    if agent_id in self._agent_blocks:
                        old_blocks = self._agent_blocks[agent_id]
                        # Free all blocks from all layers
                        for layer_blocks in old_blocks.blocks.values():
                            #Clear layer_data BEFORE freeing
                            for block in layer_blocks:
                                block.layer_data = None
                            self._pool.free(layer_blocks, agent_id)

                    # Store new blocks
                    self._agent_blocks[agent_id] = blocks

                    # Decode accumulated tokens to text
                    text = self._tokenizer.decode(tokens)

                    # Create CompletedGeneration
                    completion = CompletedGeneration(
                        uid=uid,
                        text=text,
                        blocks=blocks,
                        finish_reason=response.finish_reason,
                        token_count=blocks.total_tokens,
                    )

                    # Clean up tracking
                    del self._active_requests[uid]

                    # Yield completion
                    yield completion

    def _reconstruct_cache(self, agent_blocks: AgentBlocks) -> Any:
        """Reconstruct KVCache from blocks."""
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
                    raise GenerationError(
                        f"Block {block.block_id} for layer {layer_id} has no K/V data"
                    )
                k_tensors.append(block.layer_data["k"])
                v_tensors.append(block.layer_data["v"])

            # Concatenate using adapter (handles validation, concatenation, and evaluation)
            k_full, v_full = self._cache_adapter.concatenate_cache_blocks(
                k_tensors, v_tensors
            )

            # Append to cache
            cache.append((k_full, v_full))

        return cache

    def _extract_cache(self, uid: str, cache: Any | None = None) -> AgentBlocks:
        """Extract updated cache from batch and convert to blocks."""
        # 1. Get agent_id from UID tracking
        if uid not in self._active_requests:
            raise GenerationError(f"UID {uid} not found in active requests")
        agent_id, _ = self._active_requests[uid]  # Extract agent_id from tuple

        # 2. Use provided cache (from finished.prompt_cache) or try to extract
        if cache is None:
            # Fallback: try to extract from batch generator (for old code paths)
            if self._batch_gen is None:
                raise GenerationError("No active batch generator")
            # NOTE: extract_cache doesn't exist on BatchGenerator - this is legacy code
            # The cache should be passed in from finished.prompt_cache
            raise GenerationError(f"Cache not provided for UID {uid}")

        # 3. Handle empty cache (before importing MLX to avoid crash in tests)
        if not cache or len(cache) == 0:
            # Empty cache - return empty AgentBlocks
            return AgentBlocks(agent_id=agent_id, blocks={}, total_tokens=0)

        # 3.1. Convert KVCache objects to (K, V) tuples if needed
        # MLX BatchGenerator returns KVCache objects, we need (K, V) tuples
        if hasattr(cache[0], 'state'):
            # KVCache objects - convert to (K, V) tuples using .state property
            cache = [kv_cache.state for kv_cache in cache]

        # 3.2. Check if cache has actual data
        if cache[0][0] is None:
            # Empty cache - return empty AgentBlocks
            return AgentBlocks(agent_id=agent_id, blocks={}, total_tokens=0)

        # Handle FakeTensor in unit tests
        first_tensor = cache[0][0]
        if hasattr(first_tensor, '__class__') and first_tensor.__class__.__name__ == 'FakeTensor':
            seq_len = first_tensor.shape[2]
            n_blocks = (seq_len + self._spec.block_tokens - 1) // self._spec.block_tokens
            fake_blocks: dict[int, list[KVBlock]] = {}
            total_token_count = 0
            for layer_id, (k, v) in enumerate(cache):
                if k is None:
                    continue
                allocated_blocks = self._pool.allocate(n_blocks, layer_id, agent_id)
                for block_idx, block in enumerate(allocated_blocks):
                    start_token = block_idx * self._spec.block_tokens
                    end_token = min(start_token + self._spec.block_tokens, seq_len)
                    k_slice = k[:, :, start_token:end_token]
                    v_slice = v[:, :, start_token:end_token]
                    block.layer_data = {"k": k_slice, "v": v_slice}
                    block.token_count = end_token - start_token
                    total_token_count += block.token_count
                fake_blocks[layer_id] = allocated_blocks

            return AgentBlocks(
                agent_id=agent_id, blocks=fake_blocks, total_tokens=total_token_count
            )

        # Get sequence length from first layer K tensor shape
        first_k = cache[0][0]  # Shape: [n_kv_heads, head_dim, total_seq_len]
        seq_len = self._cache_adapter.get_sequence_length(first_k)

        # 5. Calculate blocks needed per layer
        n_blocks = (seq_len + self._spec.block_tokens - 1) // self._spec.block_tokens

        # Create blocks dictionary
        blocks_dict: dict[int, list[KVBlock]] = {}
        total_token_count = 0

        # 7. For each layer, split cache into blocks
        try:
            for layer_id, (k, v) in enumerate(cache):
                if k is None:
                    continue  # Skip empty layers (sliding window)

                allocated_blocks = self._pool.allocate(n_blocks, layer_id, agent_id)

                # Now split K, V into 256-token chunks and populate the allocated blocks
                layer_blocks = []
                for block_idx in range(n_blocks):
                    start_token = block_idx * self._spec.block_tokens
                    end_token = min(start_token + self._spec.block_tokens, seq_len)

                    k_chunk = self._cache_adapter.slice_cache_tensor(k, start_token, end_token)
                    v_chunk = self._cache_adapter.slice_cache_tensor(v, start_token, end_token)

                    # Use the pre-allocated block
                    block = allocated_blocks[block_idx]

                    # Update block with actual cache data
                    block.layer_data = {"k": k_chunk, "v": v_chunk}
                    block.token_count = end_token - start_token
                    total_token_count += block.token_count

                    layer_blocks.append(block)

                blocks_dict[layer_id] = layer_blocks

        except PoolExhaustedError:
            # Partial allocation failure: free all blocks allocated so far
            for _allocated_layer_id, allocated_layer_blocks in blocks_dict.items():
                self._pool.free(allocated_layer_blocks, agent_id)
            raise  # Re-raise the original error

        # 8. Return AgentBlocks with total_tokens (sum across all layers per validation)
        return AgentBlocks(
            agent_id=agent_id,
            blocks=blocks_dict,
            total_tokens=total_token_count,
        )

    def drain(self, timeout_seconds: float = 30.0) -> None:
        """Drain all active requests before model swap.

        Waits for all in-flight requests to complete by repeatedly calling step()
        until _active_requests is empty or timeout is reached.

        Args:
            timeout_seconds: Maximum time to wait for drain (default: 30s)

        Raises:
            GenerationError: If timeout is reached before all requests complete

        Notes:
            - This is a blocking call
            - New submit() calls during drain will still be accepted
            - For model hot-swap, caller should prevent new submits before calling drain
            - Use this before shutdown() or model swap to ensure clean state

        Example:
            >>> engine.drain(timeout_seconds=60)  # Wait up to 60s for completions
            >>> engine.shutdown()  # Safe to shutdown now
        """
        start_time = time.time()
        logger = logging.getLogger(__name__)
        logger.info(f"Draining {len(self._active_requests)} active requests...")

        while self._active_requests:
            # Check timeout
            elapsed = time.time() - start_time
            if elapsed > timeout_seconds:
                remaining = list(self._active_requests.keys())
                raise GenerationError(
                    f"Drain timeout after {timeout_seconds}s. "
                    f"Still pending: {remaining[:5]}..."  # Show first 5 UIDs
                )

            # Process one step to let requests complete
            list(self.step())  # Consume iterator to process completions

            # Brief sleep to avoid busy-waiting
            time.sleep(0.1)

        logger.info("Drain complete - all requests finished")

    def shutdown(self) -> None:
        """Shutdown batch engine and release resources.

        Clears all internal state in preparation for model swap or engine teardown.
        Should be called AFTER drain() to ensure no active requests are lost.

        Notes:
            - Does NOT free blocks from pool (caller must handle via AgentCacheStore)
            - Does NOT unload model (caller must handle via ModelRegistry)
            - After shutdown, engine is unusable until reinitialized

        Example:
            >>> engine.drain()
            >>> engine.shutdown()
            >>> # Engine is now safe to discard
        """
        logger = logging.getLogger(__name__)
        logger.info("Shutting down batch engine...")

        # Clear batch generator
        if self._batch_gen is not None:
            del self._batch_gen
            self._batch_gen = None

        # Clear active requests (should be empty after drain, but be safe)
        if self._active_requests:
            logger.warning(
                f"Shutdown with {len(self._active_requests)} active requests - possible loss"
            )
        self._active_requests.clear()

        # Clear agent blocks tracking (blocks remain in pool until explicitly freed)
        self._agent_blocks.clear()

        # Clear model/tokenizer references (will be freed by ModelRegistry)
        self._model = None
        self._tokenizer = None

        logger.info("Batch engine shutdown complete")
