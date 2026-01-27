"""Block-pool batch inference engine."""

import asyncio
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

# Import Q4 extensions to enable QuantizedKVCache.merge() for batching
from semantic.adapters.outbound import mlx_quantized_extensions  # noqa: F401

logger = logging.getLogger(__name__)


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

        # 6. Draining state (prevents new requests during graceful shutdown)
        self._draining: bool = False

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
        # 1. Check if draining (reject new requests during graceful shutdown)
        if self._draining:
            raise PoolExhaustedError(
                "Engine is draining - not accepting new requests. "
                "Server is shutting down gracefully."
            )

        # 2. Validate inputs
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
            # CRITICAL: Check if cache is too large to reconstruct safely
            # Large caches (>10K tokens) when dequantized to FP16 can exceed GPU memory
            MAX_SAFE_CACHE_TOKENS = 10000  # ~512MB FP16 per 10K tokens (safe threshold)

            if cache.total_tokens > MAX_SAFE_CACHE_TOKENS:
                # Skip reconstruction for large caches - treat as cache miss
                logging.warning(
                    f"[CACHE SKIP] Cache too large ({cache.total_tokens} tokens > {MAX_SAFE_CACHE_TOKENS} threshold). "
                    f"FP16 reconstruction would require ~{cache.total_tokens * 0.052:.0f}MB which risks OOM. "
                    f"Treating as cache miss - will regenerate from scratch."
                )
                cache = None  # Force cache miss path
                kv_cache = None
            else:
                # Cache is safe to reconstruct
                try:
                    import mlx.core as mx

                    # Memory tracking BEFORE reconstruction
                    mem_before = mx.metal.get_active_memory() / (1024**3)
                    cache_before = mx.metal.get_cache_memory() / (1024**3)
                    peak_before = mx.metal.get_peak_memory() / (1024**3)
                    logging.info(f"[MEMORY BEFORE RECONSTRUCT] Active: {mem_before:.2f}GB, Cache: {cache_before:.2f}GB, Peak: {peak_before:.2f}GB")
                    logging.info(f"[CACHE] Agent: {agent_id}, Tokens: {cache.total_tokens}, Blocks: {sum(len(blocks) for blocks in cache.blocks.values())}")

                    # Calculate Q4 memory footprint
                    bytes_per_token = (self._spec.n_kv_heads * self._spec.head_dim * 2 * 0.5)  # K+V, 4-bit = 0.5 bytes
                    q4_memory_mb = (cache.total_tokens * bytes_per_token * self._spec.n_layers) / (1024**2)
                    logging.info(f"[Q4 BLOCKS] Estimated Q4 memory: {q4_memory_mb:.1f}MB for {cache.total_tokens} tokens")

                    # FIX: Reconstruct cache without dequantization
                    kv_cache = self._reconstruct_cache(cache)

                    # Memory tracking AFTER reconstruction (should be SAME as before - no dequant!)
                    mem_after_reconstruct = mx.metal.get_active_memory() / (1024**3)
                    cache_after_reconstruct = mx.metal.get_cache_memory() / (1024**3)
                    peak_after_reconstruct = mx.metal.get_peak_memory() / (1024**3)
                    logging.info(f"[MEMORY AFTER RECONSTRUCT] Active: {mem_after_reconstruct:.2f}GB, Cache: {cache_after_reconstruct:.2f}GB, Peak: {peak_after_reconstruct:.2f}GB")
                    logging.info(f"[MEMORY DELTA RECONSTRUCT] Active: +{(mem_after_reconstruct - mem_before):.2f}GB (should be ~0 with Q4!), Cache: +{(cache_after_reconstruct - cache_before):.2f}GB")

                    cache_len = len(kv_cache) if kv_cache else 0
                    logging.info(f"[RECONSTRUCT] Created {cache_len} layer caches, type: {type(kv_cache[0]) if kv_cache else None}")

                    # CRITICAL: Free Q4 memory immediately to avoid OOM
                    # The hot_cache entry will be replaced with NEW blocks after generation.
                    # Clearing these blocks is safe - they're about to be replaced anyway.
                    logging.info(f"[RECONSTRUCT COMPLETE] Loaded {cache.total_tokens} tokens across {len(cache.blocks)} layers, now in FP16 KVCache")

                    # Free Q4 memory by clearing layer_data
                    blocks_cleared = 0
                    for layer_id, layer_blocks in cache.blocks.items():
                        for block in layer_blocks:
                            block.layer_data = None  # Free Q4 tensors (weights, scales, biases)
                            blocks_cleared += 1

                    # Clear the blocks dict to allow GC
                    cache.blocks.clear()

                    # Force garbage collection
                    import gc
                    gc.collect()

                    logging.info(f"[Q4 FREED] Cleared {blocks_cleared} blocks to free Q4 memory")

                    # Memory tracking AFTER freeing Q4
                    mem_after_free = mx.metal.get_active_memory() / (1024**3)
                    cache_after_free = mx.metal.get_cache_memory() / (1024**3)
                    peak_after_free = mx.metal.get_peak_memory() / (1024**3)
                    logging.info(f"[MEMORY POST-FREE] Active: {mem_after_free:.2f}GB, Cache: {cache_after_free:.2f}GB, Peak: {peak_after_free:.2f}GB")
                    logging.info(f"[MEMORY DELTA] Active: +{(mem_after_free - mem_before):.2f}GB (Q4 freed, FP16 remains)")

                except Exception as e:
                    logging.error(f"Cache reconstruction failed for {agent_id}: {e}", exc_info=True)
                    cache = None
                    kv_cache = None

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
                # Use adapter to create real MLX BatchGenerator with QuantizedKVCache
                self._batch_gen = self._cache_adapter.create_batch_generator(
                    model=self._model,
                    stop_tokens={self._tokenizer.eos_token_id},
                    kv_bits=self._spec.kv_bits if hasattr(self._spec, 'kv_bits') else 4,
                    kv_group_size=self._spec.kv_group_size if hasattr(self._spec, 'kv_group_size') else 64,
                )

        # 5. Insert into batch
        try:
            # Create sampler via adapter if using real MLX (not fake for testing)
            samplers = None
            if self._batch_gen_factory is None:
                sampler = self._cache_adapter.create_sampler(temperature=0.0)
                samplers = [sampler]

            # Build insert kwargs
            insert_kwargs = {
                "prompts": [prompt_tokens],
                "max_tokens": [max_tokens],
            }
            if samplers is not None:
                insert_kwargs["samplers"] = samplers
            if kv_cache is not None:
                insert_kwargs["caches"] = [kv_cache]

            uids = self._batch_gen.insert(**insert_kwargs)
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

        # 6. Track UID → (agent_id, tokens, detokenizer) mapping
        # Initialize empty token list and detokenizer - will accumulate during step()
        detokenizer = self._tokenizer.detokenizer
        detokenizer.reset()
        self._active_requests[actual_uid] = (agent_id, [], detokenizer)

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
                agent_id, tokens, detokenizer = self._active_requests[uid]

                # 3.1. Accumulate token for this step
                if response.finish_reason != "stop":
                    tokens.append(response.token)
                    # Add token to streaming detokenizer for proper space handling
                    detokenizer.add_token(response.token)
                    if len(tokens) <= 5:  # Log first 5 tokens only
                        logger.info(f"Token {len(tokens)}: {response.token} -> '{detokenizer.text}'")

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

                    # CRITICAL FIX: Use tokenizer.decode() for proper spacing
                    # The streaming detokenizer.text doesn't handle spacing correctly
                    # when the model generates tokens without space prefixes
                    text = self._tokenizer.decode(tokens)

                    # Post-processing to remove any remaining BPE markers
                    # (DeepSeek tokenizer sometimes leaves these in)
                    text = text.replace('Ġ', ' ')  # BPE space marker (U+0120)
                    text = text.replace('Ċ', '\n')  # BPE newline marker (U+010A)
                    text = text.replace('ċ', '\n')  # Lowercase variant
                    text = text.replace('▁', ' ')  # SentencePiece space marker

                    # Clean up excessive spacing
                    import re
                    text = re.sub(r' {3,}', '  ', text)  # Preserve double space (indentation)
                    text = re.sub(r'\n\n\n+', '\n\n', text)  # Max 2 newlines

                    logger.info(f"Detokenizer finalized for {uid}: '{text}' ({len(tokens)} tokens)")

                    # Create CompletedGeneration
                    completion = CompletedGeneration(
                        uid=uid,
                        text=text,
                        blocks=blocks,
                        finish_reason=response.finish_reason,
                        token_count=len(tokens),  # Generated tokens only
                    )

                    # Clean up tracking
                    del self._active_requests[uid]

                    # Yield completion
                    yield completion

    def _reconstruct_cache(self, agent_blocks: AgentBlocks) -> list[Any]:
        """Reconstruct MLX cache objects from blocks.

        Returns:
            List[KVCache]: List of cache objects (one per layer) in production
            or List[tuple]: List of (k, v) tuples in test mode
        """
        # Detect test mode: if batch_gen_factory is set, we're using fakes
        use_kv_cache = self._batch_gen_factory is None

        if use_kv_cache:
            # Production mode: use real MLX KVCache objects
            import mlx.core as mx  # noqa: PLC0415
            from mlx_lm.models.cache import KVCache, QuantizedKVCache  # noqa: PLC0415
        else:
            # Test mode: use tuples (FakeBatchGenerator expects tuples)
            mx = None  # type: ignore[assignment]
            KVCache = None  # type: ignore[assignment]  # noqa: N806
            QuantizedKVCache = None  # type: ignore[assignment]  # noqa: N806

        cache: list[Any] = []

        # For each layer in the model
        for layer_id in range(self._spec.n_layers):
            # Get all blocks for this layer
            layer_blocks = agent_blocks.blocks_for_layer(layer_id)

            # Handle empty layers
            if not layer_blocks:
                if use_kv_cache:
                    empty_cache = KVCache()
                    cache.append(empty_cache)
                else:
                    # Test environment: use tuples
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

                k_data = block.layer_data["k"]
                v_data = block.layer_data["v"]

                # CRITICAL: Keep quantized format - don't dequantize yet!
                # We'll concatenate quantized components and only dequantize
                # right before feeding to model. This saves 100-500ms per cache hit!
                if isinstance(k_data, tuple) and len(k_data) == 3:
                    # Quantized format: (weights, scales, biases)
                    # Keep as-is - concatenate_cache_blocks will handle it
                    if not use_kv_cache:
                        # Test mode: extract weights only (ignore scales/biases)
                        k_data, _, _ = k_data
                        v_data, _, _ = v_data
                    # else: Production mode - keep quantized tuple as-is

                # Convert numpy → mx.array if needed (handle quantized tuples)
                if use_kv_cache and mx is not None:
                    if isinstance(k_data, tuple) and len(k_data) == 3:
                        # Quantized tuple - convert components if needed
                        k_weights, k_scales, k_biases = k_data
                        v_weights, v_scales, v_biases = v_data

                        if not isinstance(k_weights, mx.array):
                            k_weights = mx.array(k_weights)
                            k_scales = mx.array(k_scales)
                            k_biases = mx.array(k_biases) if k_biases is not None else None
                            k_data = (k_weights, k_scales, k_biases)

                        if not isinstance(v_weights, mx.array):
                            v_weights = mx.array(v_weights)
                            v_scales = mx.array(v_scales)
                            v_biases = mx.array(v_biases) if v_biases is not None else None
                            v_data = (v_weights, v_scales, v_biases)
                    else:
                        # Float tensor - convert normally
                        if not isinstance(k_data, mx.array):
                            k_data = mx.array(k_data)
                        if not isinstance(v_data, mx.array):
                            v_data = mx.array(v_data)

                k_tensors.append(k_data)
                v_tensors.append(v_data)

            # Concatenate using adapter (keeps quantized if input is quantized)
            k_full, v_full = self._cache_adapter.concatenate_cache_blocks(
                k_tensors, v_tensors
            )

            if use_kv_cache:
                # Q4 DIRECT INJECTION: Keep Q4 format end-to-end (75% memory savings)
                # We inject QuantizedKVCache directly - MLX routes to quantized attention
                if isinstance(k_full, tuple) and len(k_full) == 3:
                    from mlx_lm.models.cache import QuantizedKVCache

                    k_weights, k_scales, k_biases = k_full
                    v_weights, v_scales, v_biases = v_full

                    kv_bits = self._spec.kv_bits if hasattr(self._spec, 'kv_bits') and self._spec.kv_bits else 4
                    kv_group_size = self._spec.kv_group_size if hasattr(self._spec, 'kv_group_size') else 64

                    # Calculate Q4 size
                    q4_size_mb = (k_weights.nbytes + k_scales.nbytes + (k_biases.nbytes if k_biases is not None else 0) +
                                  v_weights.nbytes + v_scales.nbytes + (v_biases.nbytes if v_biases is not None else 0)) / (1024**2)

                    # Create QuantizedKVCache directly (NO dequantization!)
                    kv_cache = QuantizedKVCache(group_size=kv_group_size, bits=kv_bits)
                    kv_cache.keys = (k_weights, k_scales, k_biases)
                    kv_cache.values = (v_weights, v_scales, v_biases)
                    kv_cache.offset = k_weights.shape[2]

                    logger = logging.getLogger(__name__)
                    seq_len = self._cache_adapter.get_sequence_length(k_full)
                    logger.info(
                        f"[Q4 INJECT L{layer_id}] Q4: {q4_size_mb:.1f}MB (NO dequantization!) | {seq_len} tokens"
                    )
                else:
                    # Float format - use regular KVCache
                    kv_cache = KVCache()
                    kv_cache.state = (k_full, v_full)

                cache.append(kv_cache)
            else:
                # Test environment: Use tuple format
                cache.append((k_full, v_full))

        return cache

    def _extract_cache(self, uid: str, cache: Any | None = None) -> AgentBlocks:
        """Extract updated cache from batch and convert to blocks."""
        # 1. Get agent_id from UID tracking
        if uid not in self._active_requests:
            raise GenerationError(f"UID {uid} not found in active requests")
        agent_id, _, _ = self._active_requests[uid]  # Extract agent_id from tuple (agent_id, tokens, detokenizer)

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

            # Log cache format for debugging quantization
            if cache and len(cache) > 0 and cache[0][0] is not None:
                k_sample = cache[0][0]
                logger.debug(
                    f"Cache format check: type={type(k_sample)}, "
                    f"is_tuple={isinstance(k_sample, tuple)}, "
                    f"dtype={getattr(k_sample, 'dtype', None)}"
                )

        # 3.2. Quantize float cache to save GPU memory (CRITICAL for large contexts!)
        # MLX returns float cache - quantize it immediately to reduce memory 4x
        if cache and len(cache) > 0 and cache[0][0] is not None:
            k_sample = cache[0][0]
            # Check if float (not already quantized)
            if not isinstance(k_sample, tuple):
                import mlx.core as mx

                # Get quantization settings from spec
                kv_bits = self._spec.kv_bits if hasattr(self._spec, 'kv_bits') and self._spec.kv_bits else 4
                kv_group_size = self._spec.kv_group_size if hasattr(self._spec, 'kv_group_size') else 64

                # Quantize each layer's K and V tensors
                quantized_cache = []
                for layer_id, (k, v) in enumerate(cache):
                    if k is None:
                        quantized_cache.append((None, None))
                        continue

                    # Quantize K and V tensors
                    # mx.quantize returns list [weights, scales, biases], convert to tuple
                    k_quant = tuple(mx.quantize(k, group_size=kv_group_size, bits=kv_bits))
                    v_quant = tuple(mx.quantize(v, group_size=kv_group_size, bits=kv_bits))

                    quantized_cache.append((k_quant, v_quant))

                cache = quantized_cache

                logger.info(
                    f"Quantized cache for {agent_id}: {len(cache)} layers, "
                    f"bits={kv_bits}, group_size={kv_group_size} "
                    f"(75% memory savings!)"
                )

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
                fake_blocks[layer_id] = allocated_blocks

            return AgentBlocks(
                agent_id=agent_id, blocks=fake_blocks, total_tokens=seq_len
            )

        # Get sequence length from first layer K tensor shape
        first_k = cache[0][0]  # Shape: [n_kv_heads, head_dim, total_seq_len]
        seq_len = self._cache_adapter.get_sequence_length(first_k)

        # 5. Calculate blocks needed per layer
        n_blocks = (seq_len + self._spec.block_tokens - 1) // self._spec.block_tokens

        # Create blocks dictionary
        blocks_dict: dict[int, list[KVBlock]] = {}

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

                    layer_blocks.append(block)

                blocks_dict[layer_id] = layer_blocks

        except PoolExhaustedError:
            # Partial allocation failure: free all blocks allocated so far
            for _allocated_layer_id, allocated_layer_blocks in blocks_dict.items():
                self._pool.free(allocated_layer_blocks, agent_id)
            raise  # Re-raise the original error

        # 8. Return AgentBlocks with total_tokens (seq_len, not sum across layers)
        # CRITICAL FIX: All layers store the SAME sequence of tokens, so total_tokens
        # should be seq_len, not accumulated across layers (that was counting 27x too many!)
        return AgentBlocks(
            agent_id=agent_id,
            blocks=blocks_dict,
            total_tokens=seq_len,
        )

    async def drain(self, timeout_seconds: float = 30.0) -> int:
        """Drain all active requests before shutdown or model swap.

        Sets draining flag to prevent new requests, then waits for all in-flight
        requests to complete by repeatedly calling step() until _active_requests
        is empty or timeout is reached.

        Args:
            timeout_seconds: Maximum time to wait for drain (default: 30s)

        Returns:
            Number of requests drained successfully

        Raises:
            GenerationError: If timeout is reached before all requests complete

        Notes:
            - This is an async blocking call
            - New submit() calls during drain will be REJECTED (PoolExhaustedError)
            - Use this before shutdown() or model swap to ensure clean state
            - Draining flag is set immediately to prevent new requests

        Example:
            >>> drained = await engine.drain(timeout_seconds=60)
            >>> engine.shutdown()  # Safe to shutdown now
        """
        # Set draining flag IMMEDIATELY to prevent new requests
        self._draining = True

        start_time = time.time()
        logger = logging.getLogger(__name__)
        initial_count = len(self._active_requests)
        logger.info(f"Draining {initial_count} active requests...")

        drained_count = 0

        while self._active_requests:
            # Check timeout
            elapsed = time.time() - start_time
            if elapsed > timeout_seconds:
                remaining = list(self._active_requests.keys())
                error_msg = (
                    f"Drain timeout after {timeout_seconds}s. "
                    f"Still pending: {len(remaining)} requests ({remaining[:5]}...)"
                )
                logger.error(error_msg)
                raise GenerationError(error_msg)

            # Process one step to let requests complete
            for _completion in self.step():
                drained_count += 1

            # Brief sleep to avoid busy-waiting (async-friendly)
            await asyncio.sleep(0.1)

        logger.info(f"Drain complete - {drained_count}/{initial_count} requests finished")
        return drained_count

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
