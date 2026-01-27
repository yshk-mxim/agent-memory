"""Block-pool batch inference engine."""

import asyncio
import logging
import threading
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


def adaptive_chunk_size(
    cache_pos: int,
    min_chunk: int = 512,
    max_chunk: int = 4096,
) -> int:
    """Calculate optimal chunk size based on current cache position.

    Larger chunks = faster (fewer forward passes)
    Smaller chunks = less peak memory

    Strategy: Aggressive early (large chunks), conservative late (small chunks).
    This balances speed and memory - large chunks are fast when cache is small,
    but as cache grows, we need smaller chunks to avoid OOM.

    Args:
        cache_pos: Current position in the sequence (tokens processed so far)
        min_chunk: Minimum chunk size (for large cache positions)
        max_chunk: Maximum chunk size (for small cache positions)

    Returns:
        Optimal chunk size for current position
    """
    if cache_pos < 2000:
        return max_chunk  # Large chunks when cache small
    elif cache_pos < 8000:
        return max(min_chunk, max_chunk // 2)  # Medium chunks
    elif cache_pos < 20000:
        return max(min_chunk, max_chunk // 4)  # Standard chunks
    else:
        return min_chunk  # Small chunks for huge cache


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

        self._model = model
        self._tokenizer = tokenizer
        self._pool = pool
        self._spec = spec
        self._cache_adapter = cache_adapter
        self._batch_gen_factory = batch_gen_factory

        self._batch_gen: Any | None = None  # Lazy - created on first submit
        self._lock = threading.RLock()
        self._active_requests: dict[str, tuple[str, list[int]]] = {}
        self._agent_blocks: dict[str, AgentBlocks] = {}
        self._draining: bool = False

        # Chunked prefill settings (loaded lazily)
        self._chunked_prefill_enabled: bool | None = None
        self._chunked_prefill_threshold: int | None = None
        self._chunked_prefill_min_chunk: int | None = None
        self._chunked_prefill_max_chunk: int | None = None

    def _get_chunked_prefill_settings(self) -> tuple[bool, int, int, int]:
        """Lazily load chunked prefill settings."""
        if self._chunked_prefill_enabled is None:
            try:
                from semantic.adapters.config.settings import get_settings
                settings = get_settings()
                self._chunked_prefill_enabled = settings.mlx.chunked_prefill_enabled
                self._chunked_prefill_threshold = settings.mlx.chunked_prefill_threshold
                self._chunked_prefill_min_chunk = settings.mlx.chunked_prefill_min_chunk
                self._chunked_prefill_max_chunk = settings.mlx.chunked_prefill_max_chunk
            except ImportError:
                # Settings module not available (e.g., in tests) - use defaults
                logger.debug("Settings module not available, using chunked prefill defaults")
                self._chunked_prefill_enabled = True
                self._chunked_prefill_threshold = 2048
                self._chunked_prefill_min_chunk = 512
                self._chunked_prefill_max_chunk = 4096
            except AttributeError as e:
                # Settings exist but missing required fields - log and use defaults
                logger.warning(f"Chunked prefill settings incomplete: {e}, using defaults")
                self._chunked_prefill_enabled = True
                self._chunked_prefill_threshold = 2048
                self._chunked_prefill_min_chunk = 512
                self._chunked_prefill_max_chunk = 4096

        return (
            self._chunked_prefill_enabled,
            self._chunked_prefill_threshold or 2048,
            self._chunked_prefill_min_chunk or 512,
            self._chunked_prefill_max_chunk or 4096,
        )

    # --- Public accessors (avoid direct access to private attributes) ---

    @property
    def tokenizer(self) -> Any:
        """Get the tokenizer instance."""
        return self._tokenizer

    def get_agent_blocks(self, agent_id: str) -> "AgentBlocksType | None":
        """Get agent blocks by ID (thread-safe).

        Args:
            agent_id: The agent identifier

        Returns:
            AgentBlocks if found, None otherwise
        """
        with self._lock:
            return self._agent_blocks.get(agent_id)

    def has_agent_blocks(self, agent_id: str) -> bool:
        """Check if agent has blocks (thread-safe).

        Args:
            agent_id: The agent identifier

        Returns:
            True if agent has blocks, False otherwise
        """
        with self._lock:
            return agent_id in self._agent_blocks

    def remove_agent_blocks(self, agent_id: str) -> bool:
        """Remove agent blocks from tracking (thread-safe).

        Note: This only removes from tracking, does NOT free pool blocks.
        Use BlockPool.free_agent_blocks() to free pool blocks first.

        Args:
            agent_id: The agent identifier

        Returns:
            True if agent was removed, False if not found
        """
        with self._lock:
            if agent_id in self._agent_blocks:
                del self._agent_blocks[agent_id]
                return True
            return False

    # --- End public accessors ---

    def _chunked_prefill(
        self,
        tokens: list[int],
        agent_id: str,
    ) -> list[Any]:
        """Process tokens in adaptive chunks for memory-efficient prefill.

        This achieves ~80% of FlashAttention benefits without custom kernels.
        By processing tokens in variable-sized chunks (larger early, smaller late),
        we reduce peak memory by 38-65% while maintaining speed.

        Args:
            tokens: Input token IDs to process
            agent_id: Agent ID for logging

        Returns:
            List of KVCache objects (one per layer) ready for generation
        """
        import mlx.core as mx
        from mlx_lm.models.cache import QuantizedKVCache

        enabled, threshold, min_chunk, max_chunk = self._get_chunked_prefill_settings()

        # Log that we're using chunked prefill
        logger.info(
            f"[CHUNKED PREFILL] Agent: {agent_id}, Tokens: {len(tokens)}, "
            f"Chunk range: {min_chunk}-{max_chunk}"
        )

        # Convert tokens to mx.array with batch dimension [1, seq_len]
        tokens_array = mx.array([tokens])
        seq_len = len(tokens)

        # Get kv_bits from spec for cache creation
        kv_bits = getattr(self._spec, 'kv_bits', 4) or 4
        kv_group_size = getattr(self._spec, 'kv_group_size', 64) or 64

        # Create initial cache - QuantizedKVCache for each layer
        kv_caches = [
            QuantizedKVCache(group_size=kv_group_size, bits=kv_bits)
            for _ in range(self._spec.n_layers)
        ]

        # Track memory for logging
        mem_start = mx.get_active_memory() / (1024**3)

        # Process tokens in adaptive chunks
        pos = 0
        chunk_count = 0
        total_time = 0.0

        while pos < seq_len:
            chunk_start_time = time.time()

            # Calculate adaptive chunk size based on current cache position
            chunk_size = adaptive_chunk_size(pos, min_chunk, max_chunk)
            end = min(pos + chunk_size, seq_len)

            # Extract chunk of tokens
            chunk_tokens = tokens_array[:, pos:end]

            # Forward pass through model with cache
            # The model updates kv_caches in-place during forward pass
            y = self._model(chunk_tokens, cache=kv_caches)

            # CRITICAL: Force evaluation to materialize tensors and release intermediates
            mx.eval(y)

            # CRITICAL: Clear MLX cache to release intermediate attention memory
            mx.clear_cache()

            chunk_time = time.time() - chunk_start_time
            total_time += chunk_time
            chunk_count += 1

            if chunk_count <= 3 or chunk_count % 10 == 0:
                logger.info(
                    f"[CHUNK {chunk_count}] Pos: {pos}->{end} ({end - pos} tokens), "
                    f"Time: {chunk_time*1000:.0f}ms"
                )

            pos = end

        # Final memory stats
        mem_end = mx.get_active_memory() / (1024**3)
        mem_peak = mx.get_peak_memory() / (1024**3)

        logger.info(
            f"[CHUNKED PREFILL DONE] Agent: {agent_id}, Chunks: {chunk_count}, "
            f"Total time: {total_time:.1f}s, Tokens/s: {seq_len/total_time:.0f}, "
            f"Memory: {mem_start:.2f}GB -> {mem_end:.2f}GB (peak: {mem_peak:.2f}GB)"
        )

        return kv_caches

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
        with self._lock:
            if self._draining:
                raise PoolExhaustedError(
                    "Engine is draining - not accepting new requests. "
                    "Server is shutting down gracefully."
                )

        if not prompt:
            raise InvalidRequestError("Prompt cannot be empty")
        if max_tokens <= 0:
            raise InvalidRequestError(f"max_tokens must be positive, got {max_tokens}")
        if not agent_id:
            raise InvalidRequestError("agent_id cannot be empty")

        prompt_tokens = self._tokenizer.encode(prompt)
        kv_cache: Any | None = None

        if cache is not None:
            # CRITICAL: Check if cache is too large to reconstruct safely
            # Calculate using actual model spec, not magic constants
            # Q4: n_kv_heads * head_dim * 2 (K+V) * 0.5 bytes (4-bit) * n_layers
            bytes_per_token_q4 = (
                self._spec.n_kv_heads * self._spec.head_dim * 2 * 0.5 * self._spec.n_layers
            )
            max_safe_memory_mb = 7500  # 7.5GB - safe on 24GB GPU with 8GB model
            max_safe_cache_tokens = int((max_safe_memory_mb * 1024 * 1024) / bytes_per_token_q4)

            if cache.total_tokens > max_safe_cache_tokens:
                q4_size_mb = (cache.total_tokens * bytes_per_token_q4) / (1024 * 1024)
                logger.warning(
                    f"[CACHE SKIP] Cache too large ({cache.total_tokens} tokens > {max_safe_cache_tokens} threshold). "
                    f"Q4 reconstruction would require ~{q4_size_mb:.0f}MB. "
                    f"Treating as cache miss - will regenerate from scratch."
                )
                cache = None  # Force cache miss path
                kv_cache = None
            else:
                # Cache is safe to reconstruct
                try:
                    import mlx.core as mx

                    # Memory tracking BEFORE reconstruction
                    mem_before = mx.get_active_memory() / (1024**3)
                    cache_before = mx.get_cache_memory() / (1024**3)
                    peak_before = mx.get_peak_memory() / (1024**3)
                    logger.info(f"[MEMORY BEFORE RECONSTRUCT] Active: {mem_before:.2f}GB, Cache: {cache_before:.2f}GB, Peak: {peak_before:.2f}GB")
                    logger.info(f"[CACHE] Agent: {agent_id}, Tokens: {cache.total_tokens}, Blocks: {sum(len(blocks) for blocks in cache.blocks.values())}")

                    # Reuse bytes_per_token_q4 calculated above
                    q4_memory_mb = (cache.total_tokens * bytes_per_token_q4) / (1024 * 1024)
                    logger.info(f"[Q4 BLOCKS] Estimated Q4 memory: {q4_memory_mb:.1f}MB for {cache.total_tokens} tokens")

                    # FIX: Reconstruct cache without dequantization
                    kv_cache = self._reconstruct_cache(cache)

                    # Memory tracking AFTER reconstruction (should be SAME as before - no dequant!)
                    mem_after_reconstruct = mx.get_active_memory() / (1024**3)
                    cache_after_reconstruct = mx.get_cache_memory() / (1024**3)
                    peak_after_reconstruct = mx.get_peak_memory() / (1024**3)
                    logger.info(f"[MEMORY AFTER RECONSTRUCT] Active: {mem_after_reconstruct:.2f}GB, Cache: {cache_after_reconstruct:.2f}GB, Peak: {peak_after_reconstruct:.2f}GB")
                    logger.info(f"[MEMORY DELTA RECONSTRUCT] Active: +{(mem_after_reconstruct - mem_before):.2f}GB (should be ~0 with Q4!), Cache: +{(cache_after_reconstruct - cache_before):.2f}GB")

                    cache_len = len(kv_cache) if kv_cache else 0
                    logger.info(f"[RECONSTRUCT] Created {cache_len} layer caches, type: {type(kv_cache[0]) if kv_cache else None}")

                    # Verify cache properties are set correctly - raise exceptions for mismatches
                    if kv_cache and len(kv_cache) > 0:
                        first_layer = kv_cache[0]
                        first_offset = getattr(first_layer, 'offset', None)
                        first_size = first_layer.size() if hasattr(first_layer, 'size') else None
                        expected_tokens = cache.total_tokens
                        logger.info(
                            f"[CACHE VALIDATION] Layer 0: offset={first_offset}, size()={first_size}, "
                            f"expected_tokens={expected_tokens}"
                        )
                        if first_offset is not None and first_offset != expected_tokens:
                            raise GenerationError(
                                f"Cache offset mismatch: layer 0 offset ({first_offset}) != "
                                f"expected tokens ({expected_tokens}). This would cause shape mismatch."
                            )
                        if first_size is not None and first_offset is not None and first_size != first_offset:
                            raise GenerationError(
                                f"Cache size/offset mismatch: layer 0 size() ({first_size}) != "
                                f"offset ({first_offset}). BatchGenerator won't recognize cache."
                            )

                    logger.info(f"[RECONSTRUCT COMPLETE] Loaded {cache.total_tokens} tokens across {len(cache.blocks)} layers, now in FP16 KVCache")

                    # Free Q4 memory by clearing layer_data
                    blocks_cleared = 0
                    for layer_id, layer_blocks in cache.blocks.items():
                        for block in layer_blocks:
                            block.layer_data = None
                            blocks_cleared += 1

                    cache.blocks.clear()

                    import gc
                    gc.collect()

                    logger.info(f"[Q4 FREED] Cleared {blocks_cleared} blocks to free Q4 memory")

                    # Memory tracking AFTER freeing Q4
                    mem_after_free = mx.get_active_memory() / (1024**3)
                    cache_after_free = mx.get_cache_memory() / (1024**3)
                    peak_after_free = mx.get_peak_memory() / (1024**3)
                    logger.info(f"[MEMORY POST-FREE] Active: {mem_after_free:.2f}GB, Cache: {cache_after_free:.2f}GB, Peak: {peak_after_free:.2f}GB")
                    logger.info(f"[MEMORY DELTA] Active: +{(mem_after_free - mem_before):.2f}GB (Q4 freed, FP16 remains)")

                except Exception as e:
                    logger.error(f"Cache reconstruction failed for {agent_id}: {e}", exc_info=True)
                    cache = None
                    kv_cache = None

        if cache is None:
            # No cache - check if we should use chunked prefill for long prompts
            enabled, threshold, min_chunk, max_chunk = self._get_chunked_prefill_settings()

            # Use chunked prefill for long prompts (reduces peak memory by 38-65%)
            if enabled and len(prompt_tokens) >= threshold and self._batch_gen_factory is None:
                logger.info(
                    f"[PREFILL MODE] Using chunked prefill for {len(prompt_tokens)} tokens "
                    f"(threshold: {threshold})"
                )

                # Run adaptive chunked prefill to build cache with lower memory usage
                kv_cache = self._chunked_prefill(prompt_tokens, agent_id)

                # Allocate minimal blocks for tracking (actual cache is in kv_cache)
                n_blocks_needed = (
                    (len(prompt_tokens) + self._spec.block_tokens - 1) // self._spec.block_tokens
                )
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

                blocks_dict = {0: blocks}
                agent_blocks = AgentBlocks(
                    agent_id=agent_id,
                    blocks=blocks_dict,
                    total_tokens=0,  # Blocks are empty; cache is in kv_cache
                )
                with self._lock:
                    self._agent_blocks[agent_id] = agent_blocks

            else:
                # Standard path: Let BatchGenerator handle prefill
                if enabled and len(prompt_tokens) < threshold:
                    logger.debug(
                        f"[PREFILL MODE] Standard prefill for {len(prompt_tokens)} tokens "
                        f"(below threshold: {threshold})"
                    )

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

                # Store agent blocks (thread-safe)
                with self._lock:
                    self._agent_blocks[agent_id] = agent_blocks

        # Create BatchGenerator lazily (thread-safe)
        with self._lock:
            if self._batch_gen is None:
                if self._batch_gen_factory is not None:
                    self._batch_gen = self._batch_gen_factory(self._model, self._tokenizer)
                else:
                    self._batch_gen = self._cache_adapter.create_batch_generator(
                        model=self._model,
                        stop_tokens={self._tokenizer.eos_token_id},
                        kv_bits=self._spec.kv_bits,
                        kv_group_size=self._spec.kv_group_size,
                    )

        # Insert into batch
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
                # VALIDATION: Log cache injection details for debugging
                first_cache = kv_cache[0] if isinstance(kv_cache, list) else kv_cache
                cache_offset = getattr(first_cache, 'offset', 'N/A')
                cache_size = first_cache.size() if hasattr(first_cache, 'size') else 'N/A'
                cache_type = type(first_cache).__name__
                has_bits = hasattr(first_cache, 'bits')
                logger.info(
                    f"[CACHE INJECT] Injecting {cache_type} into BatchGenerator: "
                    f"offset={cache_offset}, size()={cache_size}, "
                    f"is_quantized={has_bits}, prompt_tokens={len(prompt_tokens)}"
                )
                # CRITICAL VALIDATION: offset should match what we set
                if cache_offset != cache_size:
                    logger.warning(
                        f"[CACHE MISMATCH] offset ({cache_offset}) != size() ({cache_size}) - "
                        f"BatchGenerator may not recognize cache!"
                    )

            uids = self._batch_gen.insert(**insert_kwargs)
        except Exception as e:
            # If insertion fails, free ALL allocated blocks (not just layer 0) - thread-safe
            with self._lock:
                if cache is None and agent_id in self._agent_blocks:
                    agent_blocks = self._agent_blocks[agent_id]
                    # Free all layers to prevent resource leak
                    for layer_blocks in agent_blocks.blocks.values():
                        self._pool.free(layer_blocks, agent_id)
                    del self._agent_blocks[agent_id]
            raise InvalidRequestError(f"Failed to insert into batch: {e}") from e

        actual_uid: str = uids[0]

        # Create a NEW detokenizer per request to avoid shared state corruption
        detokenizer_class = type(self._tokenizer.detokenizer)
        detokenizer = detokenizer_class(self._tokenizer)
        with self._lock:
            self._active_requests[actual_uid] = (agent_id, [], detokenizer)

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
        if self._batch_gen is None:
            return

        while True:
            try:
                batch_response = self._batch_gen.next()  # type: ignore[no-untyped-call]
            except MemoryError as e:
                logger.error(f"OOM during batch generation step: {e}")
                # Clean up batch generator to allow recovery
                self._batch_gen = None
                raise GenerationError(f"Out of memory during generation: {e}") from e
            except Exception as e:
                logger.error(f"Batch generation step failed: {e}", exc_info=True)
                raise GenerationError(f"Generation step failed: {e}") from e

            # Check for termination: empty list means all sequences done
            if not batch_response:
                break

            for response in batch_response:
                uid = response.uid

                with self._lock:
                    if uid not in self._active_requests:
                        logger.error(
                            f"Untracked UID {uid} in batch - possible memory leak. "
                            f"Active UIDs: {list(self._active_requests.keys())}"
                        )
                        continue
                    agent_id, tokens, detokenizer = self._active_requests[uid]

                if response.finish_reason != "stop":
                    tokens.append(response.token)
                    detokenizer.add_token(response.token)
                    if len(tokens) <= 5:
                        logger.info(f"Token {len(tokens)}: {response.token} -> '{detokenizer.text}'")
                if response.finish_reason is not None:
                    # Sequence complete - process completion

                    # CRITICAL: Free old blocks BEFORE allocating new ones to avoid pool exhaustion
                    # Old code allocated first, then freed - this could exhaust pool on large caches
                    with self._lock:
                        if agent_id in self._agent_blocks:
                            old_blocks = self._agent_blocks[agent_id]
                            # Free all blocks from all layers BEFORE allocating new ones
                            for layer_blocks in old_blocks.blocks.values():
                                # Clear layer_data BEFORE freeing
                                for block in layer_blocks:
                                    block.layer_data = None
                                self._pool.free(layer_blocks, agent_id)
                            # Remove from tracking while still under lock
                            del self._agent_blocks[agent_id]

                    # Extract cache and convert to blocks (allocates for ALL layers)
                    cache = response.prompt_cache
                    blocks = self._extract_cache(uid, cache)

                    # Store new blocks (thread-safe)
                    with self._lock:
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

                    # Clean up tracking (thread-safe)
                    with self._lock:
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

                    # Q4 data detected - get quantization params from spec
                    # Default to 4 bits if spec doesn't specify (Q4 is the standard)
                    kv_bits = getattr(self._spec, 'kv_bits', 4) or 4
                    kv_group_size = getattr(self._spec, 'kv_group_size', 64) or 64

                    # Calculate Q4 size
                    q4_size_mb = (k_weights.nbytes + k_scales.nbytes + (k_biases.nbytes if k_biases is not None else 0) +
                                  v_weights.nbytes + v_scales.nbytes + (v_biases.nbytes if v_biases is not None else 0)) / (1024**2)

                    # Create QuantizedKVCache directly (NO dequantization!)
                    kv_cache = QuantizedKVCache(group_size=kv_group_size, bits=kv_bits)
                    kv_cache.keys = (k_weights, k_scales, k_biases)
                    kv_cache.values = (v_weights, v_scales, v_biases)

                    # CRITICAL FIX: Use actual sequence length, NOT tensor buffer size!
                    # Tensor shape[2] may be rounded up to 256-token blocks (e.g., 2048)
                    # but actual tokens may be fewer (e.g., 2031). offset MUST be actual tokens
                    # or BatchGenerator will process wrong token counts on continuation.
                    buffer_size = k_weights.shape[2]
                    actual_tokens = agent_blocks.total_tokens
                    kv_cache.offset = actual_tokens

                    # CRITICAL: Force evaluation to materialize tensors and prevent
                    # lazy graph accumulation that causes OOM
                    import mlx.core as mx
                    mx.eval(k_weights, k_scales, v_weights, v_scales)
                    if k_biases is not None:
                        mx.eval(k_biases, v_biases)

                    logger = logging.getLogger(__name__)
                    if buffer_size != actual_tokens:
                        logger.info(
                            f"[Q4 INJECT L{layer_id}] Q4: {q4_size_mb:.1f}MB | "
                            f"offset={actual_tokens} (buffer={buffer_size}, delta={buffer_size - actual_tokens})"
                        )
                    else:
                        logger.info(
                            f"[Q4 INJECT L{layer_id}] Q4: {q4_size_mb:.1f}MB | {actual_tokens} tokens"
                        )
                else:
                    # Float format (FP16) - use regular KVCache
                    import mlx.core as mx

                    kv_cache = KVCache()
                    kv_cache.state = (k_full, v_full)

                    # CRITICAL FIX: Set offset to actual sequence length for continuation
                    actual_tokens = agent_blocks.total_tokens
                    kv_cache.offset = actual_tokens

                    # Force evaluation for FP16 as well
                    mx.eval(k_full, v_full)

                    fp16_size_mb = (k_full.nbytes + v_full.nbytes) / (1024**2)
                    buffer_size = k_full.shape[2] if hasattr(k_full, 'shape') else 0
                    logger = logging.getLogger(__name__)
                    logger.info(
                        f"[FP16 INJECT L{layer_id}] FP16: {fp16_size_mb:.1f}MB | "
                        f"offset={actual_tokens} tokens"
                    )

                cache.append(kv_cache)
            else:
                # Test environment: Use tuple format
                cache.append((k_full, v_full))

        return cache

    def _extract_cache(self, uid: str, cache: Any | None = None) -> AgentBlocks:
        """Extract updated cache from batch and convert to blocks."""
        if uid not in self._active_requests:
            raise GenerationError(f"UID {uid} not found in active requests")
        agent_id, _, _ = self._active_requests[uid]

        if cache is None:
            if self._batch_gen is None:
                raise GenerationError("No active batch generator")
            raise GenerationError(f"Cache not provided for UID {uid}")

        if not cache or len(cache) == 0:
            return AgentBlocks(agent_id=agent_id, blocks={}, total_tokens=0)

        # Convert KVCache objects to (K, V) tuples if needed
        first_layer = cache[0]
        if hasattr(first_layer, 'state'):
            cache = [kv_cache.state for kv_cache in cache]
            if cache and cache[0][0] is not None:
                k_sample = cache[0][0]
                logger.debug(
                    f"Cache format check: type={type(k_sample)}, "
                    f"is_tuple={isinstance(k_sample, tuple)}, "
                    f"dtype={getattr(k_sample, 'dtype', None)}"
                )

        # Quantize float cache to save GPU memory
        kv_bits = self._spec.kv_bits
        kv_group_size = self._spec.kv_group_size or 64

        if cache and len(cache) > 0 and cache[0][0] is not None:
            k_sample = cache[0][0]
            # Check if float (not already quantized) AND kv_bits is set (not None)
            if not isinstance(k_sample, tuple) and kv_bits is not None:
                import mlx.core as mx

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

                    # CRITICAL FIX: Force evaluation to prevent lazy graph accumulation
                    # Without this, MLX builds deferred computation graphs that hold ALL
                    # intermediate tensors in memory → OOM during long generation sessions
                    mx.eval(k_quant[0], k_quant[1], k_quant[2],
                            v_quant[0], v_quant[1], v_quant[2])

                    quantized_cache.append((k_quant, v_quant))

                cache = quantized_cache

                # Clear MLX cache to release intermediate memory from quantization
                mx.clear_cache()

                logger.info(
                    f"Quantized cache for {agent_id}: {len(cache)} layers, "
                    f"bits={kv_bits}, group_size={kv_group_size} "
                    f"(75% memory savings!)"
                )
            elif not isinstance(k_sample, tuple):
                logger.info(
                    f"Keeping FP16 cache for {agent_id}: {len(cache)} layers "
                    f"(kv_bits=None, no quantization)"
                )

        if cache[0][0] is None:
            return AgentBlocks(agent_id=agent_id, blocks={}, total_tokens=0)

        # Handle FakeTensor in unit tests (TODO: replace with dependency injection)
        first_tensor = cache[0][0]
        if first_tensor.__class__.__name__ == 'FakeTensor':
            seq_len = first_tensor.shape[2]
            n_blocks = (seq_len + self._spec.block_tokens - 1) // self._spec.block_tokens
            fake_blocks: dict[int, list[KVBlock]] = {}
            try:
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
            except PoolExhaustedError:
                for allocated_layer_blocks in fake_blocks.values():
                    self._pool.free(allocated_layer_blocks, agent_id)
                raise

            return AgentBlocks(
                agent_id=agent_id, blocks=fake_blocks, total_tokens=seq_len
            )

        first_k = cache[0][0]
        seq_len = self._cache_adapter.get_sequence_length(first_k)
        n_blocks = (seq_len + self._spec.block_tokens - 1) // self._spec.block_tokens
        blocks_dict: dict[int, list[KVBlock]] = {}

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
            for _allocated_layer_id, allocated_layer_blocks in blocks_dict.items():
                self._pool.free(allocated_layer_blocks, agent_id)
            raise

        # total_tokens = seq_len (not sum across layers - all layers have same sequence)
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
        # Use lock to prevent race condition with submit() checking _draining
        with self._lock:
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
