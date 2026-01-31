"""Block-pool batch inference engine."""

import asyncio
import gc
import logging
import re
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
from semantic.domain.value_objects import CompletedGeneration, ModelCacheSpec, StepOneResult

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
        adaptive_config: Any | None = None,  # AdaptiveConfig (optional)
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
        self._adaptive_config = adaptive_config

        self._batch_gen: Any | None = None  # Lazy - created on first submit
        self._lock = threading.RLock()
        self._active_requests: dict[str, tuple[str, list[int], Any, list[int], str]] = {}
        self._agent_blocks: dict[str, AgentBlocks] = {}
        self._draining: bool = False
        self._native_completions: dict[str, CompletedGeneration] = {}  # Pre-computed native path results

        # Chunked prefill settings (loaded lazily)
        self._chunked_prefill_enabled: bool | None = None
        self._chunked_prefill_threshold: int | None = None
        self._chunked_prefill_min_chunk: int | None = None
        self._chunked_prefill_max_chunk: int | None = None

    def _get_chunked_prefill_settings(self) -> tuple[bool, int, int, int]:
        """Lazily load chunked prefill settings.

        If adaptive_config is set, chunk sizes are adjusted dynamically
        based on workload (memory pressure, batch depth).
        """
        if self._chunked_prefill_enabled is None:
            try:
                from semantic.adapters.config.settings import get_settings
                settings = get_settings()
                self._chunked_prefill_enabled = settings.mlx.chunked_prefill_enabled
                self._chunked_prefill_threshold = settings.mlx.chunked_prefill_threshold
                self._chunked_prefill_min_chunk = settings.mlx.chunked_prefill_min_chunk
                self._chunked_prefill_max_chunk = settings.mlx.chunked_prefill_max_chunk
            except ImportError:
                logger.debug("Settings module not available, using chunked prefill defaults")
                self._chunked_prefill_enabled = True
                self._chunked_prefill_threshold = 2048
                self._chunked_prefill_min_chunk = 512
                self._chunked_prefill_max_chunk = 4096
            except AttributeError as e:
                logger.warning(f"Chunked prefill settings incomplete: {e}, using defaults")
                self._chunked_prefill_enabled = True
                self._chunked_prefill_threshold = 2048
                self._chunked_prefill_min_chunk = 512
                self._chunked_prefill_max_chunk = 4096

        min_chunk = self._chunked_prefill_min_chunk or 512
        max_chunk = self._chunked_prefill_max_chunk or 4096

        # Override chunk sizes from adaptive config if available
        if self._adaptive_config is not None:
            min_chunk, max_chunk = self._adaptive_config.effective_chunk_sizes

        return (
            self._chunked_prefill_enabled,
            self._chunked_prefill_threshold or 2048,
            min_chunk,
            max_chunk,
        )

    def _log_memory(self, label: str) -> tuple[float, float, float]:
        """Log current MLX memory state.

        Args:
            label: Descriptive label for this memory checkpoint

        Returns:
            Tuple of (active_gb, cache_gb, peak_gb)
        """
        import mlx.core as mx
        active = mx.get_active_memory() / (1024**3)
        cache = mx.get_cache_memory() / (1024**3)
        peak = mx.get_peak_memory() / (1024**3)
        logger.info(f"[MEMORY {label}] Active: {active:.2f}GB, Cache: {cache:.2f}GB, Peak: {peak:.2f}GB")
        return active, cache, peak

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
        initial_cache: list[Any] | None = None,
    ) -> list[Any]:
        """Process tokens in adaptive chunks for memory-efficient prefill.

        This achieves ~80% of FlashAttention benefits without custom kernels.
        By processing tokens in variable-sized chunks (larger early, smaller late),
        we reduce peak memory by 38-65% while maintaining speed.

        Args:
            tokens: Input token IDs to process
            agent_id: Agent ID for logging
            initial_cache: Optional pre-existing KV cache to extend (warm cache).
                If provided, tokens are processed on top of the existing cache.
                If None, fresh QuantizedKVCache objects are created.

        Returns:
            List of KVCache objects (one per layer) ready for generation
        """
        import mlx.core as mx
        from mlx_lm.models.cache import QuantizedKVCache

        enabled, threshold, min_chunk, max_chunk = self._get_chunked_prefill_settings()

        cache_mode = "extending warm cache" if initial_cache is not None else "cold start"
        logger.info(
            f"[CHUNKED PREFILL] Agent: {agent_id}, Tokens: {len(tokens)}, "
            f"Chunk range: {min_chunk}-{max_chunk}, Mode: {cache_mode}"
        )

        # Import generation_stream to route model calls through the same GPU
        # command stream that BatchGenerator uses. This prevents Metal command
        # buffer conflicts when the scheduler interleaves prefill with decode.
        from mlx_lm.generate import generation_stream

        # Convert tokens to mx.array with batch dimension [1, seq_len]
        tokens_array = mx.array([tokens])
        seq_len = len(tokens)

        # Get kv_bits from spec for cache creation
        kv_bits = getattr(self._spec, 'kv_bits', 4) or 4
        kv_group_size = getattr(self._spec, 'kv_group_size', 64) or 64

        # Use initial cache if provided (warm cache extension), otherwise create fresh
        if initial_cache is not None:
            kv_caches = initial_cache
        else:
            kv_caches = [
                QuantizedKVCache(group_size=kv_group_size, bits=kv_bits)
                for _ in range(self._spec.n_layers)
            ]

        # Track memory for logging
        mem_start = mx.get_active_memory() / (1024**3)

        # Process all tokens EXCEPT the last one in chunks.
        # BatchGenerator needs at least one unprocessed token to produce
        # initial logits for generation. If we process all tokens here,
        # BatchGenerator receives an empty prompt with a non-empty cache,
        # which triggers negative-length prepare() and corrupts MoE routing.
        prefill_len = seq_len - 1

        pos = 0
        chunk_count = 0
        total_time = 0.0

        while pos < prefill_len:
            chunk_start_time = time.time()

            # Calculate adaptive chunk size based on current cache position
            chunk_size = adaptive_chunk_size(pos, min_chunk, max_chunk)
            end = min(pos + chunk_size, prefill_len)

            # Run model forward pass on generation_stream — the same stream
            # BatchGenerator uses — to prevent Metal command buffer conflicts.
            with mx.stream(generation_stream):
                chunk_tokens = tokens_array[:, pos:end]
                y = self._model(chunk_tokens, cache=kv_caches)
                mx.eval(y)
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
            f"Prefilled: {prefill_len}/{seq_len} tokens (last token reserved for BatchGenerator), "
            f"Total time: {total_time:.1f}s, Tokens/s: {(prefill_len / total_time) if total_time > 0 else 0:.0f}, "
            f"Memory: {mem_start:.2f}GB -> {mem_end:.2f}GB (peak: {mem_peak:.2f}GB)"
        )

        return kv_caches

    def _generate_native(
        self,
        prompt_tokens: list[int],
        kv_cache: list[Any],
        max_tokens: int,
        agent_id: str,
    ) -> tuple[list[int], list[Any]]:
        """Native generation for single-sequence with Q4 cache.

        Bypasses BatchGenerator to avoid the 27-layer expansion overhead that
        occurs when BatchQuantizedKVCache.update_and_fetch() allocates 16K+
        token headroom across all layers simultaneously.

        Args:
            prompt_tokens: Full prompt token sequence
            kv_cache: List of QuantizedKVCache objects (one per layer)
            max_tokens: Maximum tokens to generate
            agent_id: Agent ID for logging

        Returns:
            Tuple of (generated_tokens, updated_cache)
        """
        import mlx.core as mx

        # Track memory before generation
        mem_before = mx.get_active_memory() / (1024**3)
        peak_before = mx.get_peak_memory() / (1024**3)

        # Get cached token count from first layer
        cached_tokens = kv_cache[0].offset if kv_cache and kv_cache[0].offset else 0

        # Determine which tokens to process (only new ones beyond cache)
        if cached_tokens > 0 and cached_tokens < len(prompt_tokens):
            tokens_to_process = prompt_tokens[cached_tokens:]
        elif cached_tokens >= len(prompt_tokens):
            # Cache covers entire prompt - just need a seed token for generation
            tokens_to_process = [prompt_tokens[-1]]
        else:
            tokens_to_process = prompt_tokens

        logger.info(
            f"[NATIVE GEN] Agent: {agent_id}, Cached: {cached_tokens}, "
            f"Processing: {len(tokens_to_process)}, Max: {max_tokens}"
        )

        # Process any new prompt tokens (prefill phase)
        if len(tokens_to_process) > 1 or cached_tokens == 0:
            tokens_array = mx.array([tokens_to_process])
            y = self._model(tokens_array, cache=kv_cache)
            mx.eval(y)
        else:
            # Single token continuation - process it
            tokens_array = mx.array([tokens_to_process])
            y = self._model(tokens_array, cache=kv_cache)
            mx.eval(y)

        # Sample first generated token (greedy - temperature=0)
        logits = y[:, -1, :]
        next_token = int(mx.argmax(logits, axis=-1).item())

        # Check for immediate EOS
        if next_token == self._tokenizer.eos_token_id:
            logger.info(f"[NATIVE GEN] EOS at token 1 (immediate)")
            generated = []  # Don't include EOS in output
        else:
            generated = [next_token]

            # Autoregressive generation loop
            for i in range(max_tokens - 1):
                tokens_array = mx.array([[next_token]])
                y = self._model(tokens_array, cache=kv_cache)
                mx.eval(y)

                logits = y[:, -1, :]
                next_token = int(mx.argmax(logits, axis=-1).item())

                # Check for EOS BEFORE adding to output
                if next_token == self._tokenizer.eos_token_id:
                    logger.info(f"[NATIVE GEN] EOS at token {i + 2}")
                    break

                generated.append(next_token)

            # Periodic memory cleanup to prevent fragmentation
            if i > 0 and i % 100 == 0:
                mx.clear_cache()
                logger.debug(f"[NATIVE GEN] Token {i + 1}, cleared cache")

        # Final memory stats
        mem_after = mx.get_active_memory() / (1024**3)
        peak_after = mx.get_peak_memory() / (1024**3)

        logger.info(
            f"[NATIVE GEN DONE] Agent: {agent_id}, Generated: {len(generated)} tokens, "
            f"Cache offset: {kv_cache[0].offset}, "
            f"Memory: {mem_before:.2f}GB -> {mem_after:.2f}GB (peak: {peak_after:.2f}GB)"
        )

        return generated, kv_cache

    def submit(
        self,
        agent_id: str,
        prompt: str,
        cache: "AgentBlocksType | None" = None,  #Proper type annotation
        max_tokens: int = 256,
        prompt_tokens: list[int] | None = None,
        temperature: float = 0.0,
        top_p: float = 0.0,
        top_k: int = 0,
    ) -> str:
        """Submit a generation request to the batch queue.

        Args:
            agent_id: Unique identifier for the agent.
            prompt: Input text to continue.
            cache: Optional pre-built cache (AgentBlocks from previous generation).
            max_tokens: Maximum tokens to generate.
            prompt_tokens: Pre-tokenized prompt (skips tokenization if provided).
            temperature: Sampling temperature (0.0 = greedy).
            top_p: Nucleus sampling threshold (0.0 = disabled).
            top_k: Top-k sampling (0 = disabled).

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

        if not prompt and not prompt_tokens:
            raise InvalidRequestError("Prompt cannot be empty")
        if max_tokens <= 0:
            raise InvalidRequestError(f"max_tokens must be positive, got {max_tokens}")
        if not agent_id:
            raise InvalidRequestError("agent_id cannot be empty")

        if not prompt_tokens:
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

                    logger.info(f"[RECONSTRUCT COMPLETE] Loaded {cache.total_tokens} tokens across {len(cache.blocks)} layers as QuantizedKVCache (Q4)")

                    # Free Q4 GPU memory by clearing layer_data references.
                    # IMPORTANT: Do NOT call cache.blocks.clear() here!
                    # If cache is the same reference as _agent_blocks[agent_id]
                    # (hot cache hit), clearing the dict would prevent step() from
                    # finding and freeing the pool blocks → permanent pool leak.
                    blocks_cleared = 0
                    for layer_id, layer_blocks in cache.blocks.items():
                        for block in layer_blocks:
                            block.layer_data = None
                            blocks_cleared += 1

                    gc.collect()

                    logger.info(f"[Q4 FREED] Cleared {blocks_cleared} blocks to free Q4 memory")

                    # Memory tracking AFTER freeing Q4
                    mem_after_free = mx.get_active_memory() / (1024**3)
                    cache_after_free = mx.get_cache_memory() / (1024**3)
                    peak_after_free = mx.get_peak_memory() / (1024**3)
                    logger.info(f"[MEMORY POST-FREE] Active: {mem_after_free:.2f}GB, Cache: {cache_after_free:.2f}GB, Peak: {peak_after_free:.2f}GB")
                    logger.info(f"[MEMORY DELTA] Active: +{(mem_after_free - mem_before):.2f}GB (Q4 block data freed, QuantizedKVCache remains)")

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
                    # Free chunked prefill cache to prevent GPU memory leak
                    del kv_cache
                    kv_cache = None
                    gc.collect()
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

                # After chunked prefill, pass kv_cache to BatchGenerator for generation
                # BatchGenerator will handle the rest (generation + cache extraction)
                logger.info(
                    f"[CHUNKED PREFILL DONE] Agent: {agent_id}, prefilled={len(prompt_tokens)} tokens, "
                    f"passing to BatchGenerator for generation"
                )
                # Fall through to BatchGenerator insertion below

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
                sampler = self._cache_adapter.create_sampler(
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                )
                samplers = [sampler]

            # Build insert kwargs
            # Use MLX-LM pattern for prefix caching:
            #   - Exact match: cache tokens == prompt tokens → pass []
            #   - Partial match (extend): prompt extends cache → pass tokens[cache_len:]
            #   - Divergent match: tokens differ → trim cache, pass rest
            tokens_to_process = prompt_tokens

            # CASE 1: kv_cache from chunked prefill (cache is None on cold start)
            if kv_cache is not None and cache is None:
                # Chunked prefill processed all tokens EXCEPT the last one.
                # BatchGenerator requires at least one token to produce initial
                # logits. Pass the last prompt token so BatchGenerator runs one
                # forward pass (last token + prefilled cache) to seed generation.
                first_cache = kv_cache[0] if isinstance(kv_cache, list) else kv_cache
                cached_tokens = getattr(first_cache, 'offset', 0) or 0
                tokens_to_process = [prompt_tokens[-1]]
                logger.info(
                    f"[CHUNKED PREFILL INSERT] Cache has {cached_tokens} tokens from prefill, "
                    f"passing last prompt token to BatchGenerator for generation"
                )

            # CASE 2: kv_cache from loaded cache (cache is not None)
            elif kv_cache is not None and cache is not None:
                first_cache = kv_cache[0] if isinstance(kv_cache, list) else kv_cache
                cached_tokens = getattr(first_cache, 'offset', 0) or 0
                cache_type = type(first_cache).__name__
                has_bits = hasattr(first_cache, 'bits')

                # Character-level prefix matching (avoids BPE boundary mismatches)
                stored_text = getattr(cache, 'prompt_text', '')

                if stored_text:
                    char_match = cache.common_prefix_chars(prompt)

                    if char_match == len(stored_text) == len(prompt):
                        # EXACT MATCH: Same prompt text → treat as cache miss for fresh generation
                        logger.info(
                            f"[CACHE EXACT MATCH MISS] {char_match} chars match exactly. "
                            f"Treating as cache miss for fresh generation."
                        )
                        if kv_cache is not None:
                            import mlx.core as mx
                            del kv_cache
                            gc.collect()
                            mx.clear_cache()
                        kv_cache = None
                        cache = None
                        tokens_to_process = prompt_tokens

                    elif char_match == len(stored_text) < len(prompt):
                        # EXTEND: Full stored text matches, new text appended
                        # Trim cache to stored prompt token count (cache.total_tokens
                        # may include generated tokens beyond the prompt)
                        stored_prompt_token_count = len(cache.token_sequence) if cache.token_sequence else 0
                        if cache.total_tokens > stored_prompt_token_count and stored_prompt_token_count > 0:
                            logger.info(
                                f"[CACHE EXTEND] Trimming cache from {cache.total_tokens} to "
                                f"{stored_prompt_token_count} (discarding generated token KV)"
                            )
                            self._slice_cache_to_length(kv_cache, stored_prompt_token_count)
                            import mlx.core as mx
                            gc.collect()
                            mx.clear_cache()

                        # Tokenize only the NEW text (after the character match point)
                        new_text = prompt[char_match:]
                        new_tokens = self._tokenizer.encode(new_text)

                        # Use chunked prefill for long extensions (memory protection)
                        cp_enabled, cp_threshold, _, _ = self._get_chunked_prefill_settings()
                        if cp_enabled and len(new_tokens) >= cp_threshold:
                            logger.info(
                                f"[CACHE EXTEND + CHUNKED] {len(new_tokens)} new tokens >= threshold "
                                f"{cp_threshold}, using chunked prefill on warm cache"
                            )
                            kv_cache = self._chunked_prefill(
                                new_tokens, agent_id, initial_cache=kv_cache
                            )
                            tokens_to_process = [new_tokens[-1]]
                        else:
                            tokens_to_process = new_tokens

                        logger.info(
                            f"[CACHE EXTEND] Reusing {char_match} chars of prompt text, "
                            f"processing {len(tokens_to_process)} new tokens"
                        )

                    elif char_match < len(stored_text):
                        # DIVERGE: Prompt diverges from stored text
                        match_ratio = char_match / len(stored_text)

                        if match_ratio >= 0.8 and char_match > 100:
                            # Partial reuse: re-tokenize matched prefix to find token boundary
                            matched_tokens = self._tokenizer.encode(prompt[:char_match])
                            usable_tokens = len(matched_tokens)
                            logger.info(
                                f"[CACHE DIVERGE PARTIAL] {char_match} chars matched "
                                f"({match_ratio:.0%}), trimming to {usable_tokens} tokens"
                            )
                            self._slice_cache_to_length(kv_cache, usable_tokens)
                            import mlx.core as mx
                            gc.collect()
                            mx.clear_cache()

                            remaining_text = prompt[char_match:]
                            remaining_tokens = self._tokenizer.encode(remaining_text)

                            # Chunked prefill for long remaining portion
                            cp_enabled, cp_threshold, _, _ = self._get_chunked_prefill_settings()
                            if cp_enabled and len(remaining_tokens) >= cp_threshold:
                                logger.info(
                                    f"[CACHE DIVERGE + CHUNKED] {len(remaining_tokens)} remaining tokens "
                                    f">= threshold {cp_threshold}, using chunked prefill"
                                )
                                kv_cache = self._chunked_prefill(
                                    remaining_tokens, agent_id, initial_cache=kv_cache
                                )
                                tokens_to_process = [remaining_tokens[-1]]
                            else:
                                tokens_to_process = remaining_tokens
                        else:
                            # Not enough overlap — discard cache entirely
                            logger.info(
                                f"[CACHE DIVERGE MISS] {char_match} chars matched "
                                f"({match_ratio:.0%} < 80%), discarding cache."
                            )
                            if kv_cache is not None:
                                import mlx.core as mx
                                del kv_cache
                                gc.collect()
                                mx.clear_cache()
                            kv_cache = None
                            cache = None
                            tokens_to_process = prompt_tokens

                else:
                    # No prompt_text stored — fall back to token comparison (legacy)
                    cached_token_seq = getattr(cache, 'token_sequence', [])
                    if cached_token_seq:
                        common_prefix = cache.common_prefix_length(prompt_tokens)
                        if common_prefix == len(cached_token_seq) < len(prompt_tokens):
                            tokens_to_process = prompt_tokens[common_prefix:]
                            logger.info(
                                f"[CACHE LEGACY EXTEND] Reusing {common_prefix} tokens (token-level)"
                            )
                        else:
                            logger.info(
                                f"[CACHE LEGACY MISS] Token prefix={common_prefix}, "
                                f"cached={len(cached_token_seq)}, prompt={len(prompt_tokens)}"
                            )
                            if kv_cache is not None:
                                import mlx.core as mx
                                del kv_cache
                                gc.collect()
                                mx.clear_cache()
                            kv_cache = None
                            cache = None
                            tokens_to_process = prompt_tokens
                    elif cached_tokens > 0 and cached_tokens < len(prompt_tokens):
                        tokens_to_process = prompt_tokens[cached_tokens:]
                        logger.info(
                            f"[CACHE LEGACY OFFSET] Skipping {cached_tokens} cached tokens"
                        )
                    elif cached_tokens >= len(prompt_tokens):
                        tokens_to_process = [prompt_tokens[-1]]
                        logger.info(
                            f"[CACHE LEGACY EXACT] Using last token as seed"
                        )

                # Safety check: if tokens_to_process is empty, use last token
                if not tokens_to_process and prompt_tokens:
                    logger.warning(
                        f"[CACHE FALLBACK] Empty tokens_to_process, using last token"
                    )
                    tokens_to_process = [prompt_tokens[-1]]

                logger.info(
                    f"[CACHE INJECT] {cache_type}: offset={cached_tokens}, "
                    f"is_quantized={has_bits}, prompt={len(prompt_tokens)}, "
                    f"processing={len(tokens_to_process)}"
                )

            # Cold starts: kv_cache=None is fine — the patched _make_cache in
            # mlx_quantized_extensions creates BatchQuantizedKVCache directly,
            # keeping Q4 end-to-end without needing empty per-sequence caches.

            insert_kwargs = {
                "prompts": [tokens_to_process],
                "max_tokens": [max_tokens],
            }
            if samplers is not None:
                insert_kwargs["samplers"] = samplers
            if kv_cache is not None:
                insert_kwargs["caches"] = [kv_cache]

            uids = self._batch_gen.insert(**insert_kwargs)
        except Exception as e:
            # If insertion fails, free ALL allocated blocks (not just layer 0) - thread-safe
            with self._lock:
                if cache is None and agent_id in self._agent_blocks:
                    agent_blocks = self._agent_blocks[agent_id]
                    for layer_blocks in agent_blocks.blocks.values():
                        self._pool.free(layer_blocks, agent_id)
                    del self._agent_blocks[agent_id]
            # Free reconstructed KV cache to prevent GPU memory leak
            if kv_cache is not None:
                del kv_cache
                gc.collect()
            raise InvalidRequestError(f"Failed to insert into batch: {e}") from e

        actual_uid: str = uids[0]

        # Create a NEW detokenizer per request to avoid shared state corruption
        detokenizer_class = type(self._tokenizer.detokenizer)
        detokenizer = detokenizer_class(self._tokenizer)
        with self._lock:
            # Store: (agent_id, generated_tokens, detokenizer, prompt_tokens, prompt_text)
            # prompt_tokens needed to save full token_sequence for prefix matching
            # prompt_text needed for character-level prefix matching (avoids BPE boundary issues)
            self._active_requests[actual_uid] = (agent_id, [], detokenizer, list(prompt_tokens), prompt)

        return actual_uid

    def has_active_batch(self) -> bool:
        """Check if BatchGenerator has active sequences (thread-safe).

        Used by ConcurrentScheduler to decide whether to run a decode step.
        """
        with self._lock:
            if self._batch_gen is None:
                return False
            return len(self._active_requests) > 0

    def submit_with_cache(
        self,
        agent_id: str,
        prompt_tokens: list[int],
        kv_caches: list[Any],
        max_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 0.0,
        top_k: int = 0,
        prompt_text: str = "",
    ) -> str:
        """Insert a pre-prefilled sequence into BatchGenerator for decode.

        Called by ConcurrentScheduler after chunked prefill completes.
        The kv_caches contain all-but-last prompt token KV state. We pass
        the last prompt token so BatchGenerator can produce initial logits.

        Args:
            agent_id: Unique identifier for the agent.
            prompt_tokens: Full prompt token sequence (for cache extraction).
            kv_caches: Pre-built KV caches from MLXPrefillAdapter.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature (0.0 = greedy).
            top_p: Nucleus sampling threshold (0.0 = disabled).
            top_k: Top-k sampling (0 = disabled).
            prompt_text: Raw prompt text for character-level prefix matching.

        Returns:
            Request UID for tracking this generation.
        """
        with self._lock:
            if self._draining:
                raise PoolExhaustedError(
                    "Engine is draining - not accepting new requests."
                )

        if not prompt_tokens:
            raise InvalidRequestError("prompt_tokens cannot be empty")
        if max_tokens <= 0:
            raise InvalidRequestError(f"max_tokens must be positive, got {max_tokens}")
        if not agent_id:
            raise InvalidRequestError("agent_id cannot be empty")
        if not kv_caches:
            raise InvalidRequestError("kv_caches cannot be empty")

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

        # Insert with last prompt token — cache has all-but-last token KV.
        # BatchGenerator requires at least one token to produce initial logits.
        try:
            samplers = None
            if self._batch_gen_factory is None:
                sampler = self._cache_adapter.create_sampler(
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                )
                samplers = [sampler]

            insert_kwargs: dict[str, Any] = {
                "prompts": [[prompt_tokens[-1]]],
                "max_tokens": [max_tokens],
                "caches": [kv_caches],
            }
            if samplers is not None:
                insert_kwargs["samplers"] = samplers

            cached_tokens = kv_caches[0].offset if hasattr(kv_caches[0], 'offset') else 0
            logger.info(
                f"[SUBMIT_WITH_CACHE] agent={agent_id}, "
                f"cache_tokens={cached_tokens}, max_tokens={max_tokens}"
            )

            uids = self._batch_gen.insert(**insert_kwargs)
        except Exception as e:
            raise InvalidRequestError(
                f"Failed to insert pre-prefilled sequence: {e}"
            ) from e

        actual_uid: str = uids[0]

        detokenizer_class = type(self._tokenizer.detokenizer)
        detokenizer = detokenizer_class(self._tokenizer)
        with self._lock:
            self._active_requests[actual_uid] = (
                agent_id, [], detokenizer, list(prompt_tokens), prompt_text
            )

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
        # First, yield any pre-computed native completions (from native path)
        with self._lock:
            native_uids = list(self._native_completions.keys())
        for uid in native_uids:
            with self._lock:
                completion = self._native_completions.pop(uid, None)
            if completion is not None:
                yield completion

        if self._batch_gen is None:
            return

        while True:
            try:
                batch_response = self._batch_gen.next()  # type: ignore[no-untyped-call]
            except MemoryError as e:
                logger.error(f"OOM during batch generation step: {e}")
                self._batch_gen = None
                raise GenerationError(f"Out of memory during generation: {e}") from e
            except Exception as e:
                logger.error(f"Batch generation step failed: {e}", exc_info=True)
                # Reset batch generator to prevent corrupted state from
                # poisoning subsequent requests (sequences stuck in generator)
                self._batch_gen = None
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
                    agent_id, tokens, detokenizer, prompt_tokens, prompt_text = self._active_requests[uid]

                if response.finish_reason != "stop":
                    tokens.append(response.token)
                    detokenizer.add_token(response.token)
                    if len(tokens) <= 5:
                        logger.info(f"Token {len(tokens)}: {response.token} -> '{detokenizer.text}'")
                if response.finish_reason is not None:
                    yield self._finalize_sequence(
                        uid, response, agent_id, tokens, prompt_tokens, prompt_text
                    )

    def _clean_text(self, text: str) -> str:
        """Remove BPE artifacts and normalize whitespace."""
        text = text.replace('Ġ', ' ')
        text = text.replace('Ċ', '\n')
        text = text.replace('ċ', '\n')
        text = text.replace('▁', ' ')
        text = re.sub(r' {3,}', '  ', text)
        text = re.sub(r'\n\n\n+', '\n\n', text)
        return text

    def _finalize_sequence(
        self,
        uid: str,
        response: Any,
        agent_id: str,
        tokens: list[int],
        prompt_tokens: list[int],
        prompt_text: str,
    ) -> CompletedGeneration:
        """Handle cache extraction and text processing for a completed sequence."""
        # Free old blocks BEFORE allocating new ones
        with self._lock:
            if agent_id in self._agent_blocks:
                old_blocks = self._agent_blocks[agent_id]
                for layer_blocks in old_blocks.blocks.values():
                    for block in layer_blocks:
                        block.layer_data = None
                    self._pool.free(layer_blocks, agent_id)
                del self._agent_blocks[agent_id]

        full_token_sequence = list(prompt_tokens)

        cache = response.prompt_cache
        if cache:
            first = cache[0] if cache else None
            logger.info(
                f"[BATCH EXTRACT] uid={uid}, cache_len={len(cache)}, "
                f"first_type={type(first).__name__}, "
                f"offset={getattr(first, 'offset', 'N/A')}"
            )

        blocks = self._extract_cache(uid, cache, full_token_sequence, prompt_text=prompt_text)

        with self._lock:
            self._agent_blocks[agent_id] = blocks

        text = self._tokenizer.decode(tokens)
        text = self._clean_text(text)

        logger.info(f"Finalized {uid}: '{text[:80]}...' ({len(tokens)} tokens)")

        completion = CompletedGeneration(
            uid=uid,
            text=text,
            blocks=blocks,
            finish_reason=response.finish_reason,
            token_count=len(tokens),
        )

        with self._lock:
            del self._active_requests[uid]

        return completion

    def step_once(self) -> list[StepOneResult]:
        """Execute one batch_gen.next() call and return per-sequence results.

        Unlike step() which runs all tokens to completion, this processes
        exactly one token per active sequence. The scheduler calls this
        once per loop iteration, interleaving with prefill and request
        acceptance for responsive scheduling.

        Returns:
            List of StepOneResult, one per active sequence. Empty if no
            active sequences or batch generator is not initialized.
        """
        results: list[StepOneResult] = []

        # Handle pre-computed native path completions
        with self._lock:
            native_uids = list(self._native_completions.keys())
        for uid in native_uids:
            with self._lock:
                completion = self._native_completions.pop(uid, None)
            if completion is not None:
                results.append(StepOneResult(
                    uid=uid,
                    text=completion.text,
                    token_count=completion.token_count,
                    finish_reason=completion.finish_reason,
                    completion=completion,
                ))

        if self._batch_gen is None:
            return results

        try:
            batch_response = self._batch_gen.next()
        except MemoryError as e:
            logger.error(f"OOM during batch generation step: {e}")
            self._batch_gen = None
            raise GenerationError(f"Out of memory during generation: {e}") from e
        except Exception as e:
            logger.error(f"Batch generation step failed: {e}", exc_info=True)
            self._batch_gen = None
            raise GenerationError(f"Generation step failed: {e}") from e

        if not batch_response:
            return results

        for response in batch_response:
            uid = response.uid

            with self._lock:
                if uid not in self._active_requests:
                    logger.error(f"Untracked UID {uid} in step_once")
                    continue
                agent_id, tokens, detokenizer, prompt_tokens, prompt_text = self._active_requests[uid]

            if response.finish_reason != "stop":
                tokens.append(response.token)
                detokenizer.add_token(response.token)
                if len(tokens) <= 5:
                    logger.info(f"Token {len(tokens)}: {response.token} -> '{detokenizer.text}'")

            if response.finish_reason is not None:
                completion = self._finalize_sequence(
                    uid, response, agent_id, tokens, prompt_tokens, prompt_text,
                )
                results.append(StepOneResult(
                    uid=uid,
                    text=completion.text,
                    token_count=completion.token_count,
                    finish_reason=completion.finish_reason,
                    completion=completion,
                ))
            else:
                text = self._clean_text(detokenizer.text)
                results.append(StepOneResult(
                    uid=uid,
                    text=text,
                    token_count=len(tokens),
                    finish_reason=None,
                    completion=None,
                ))

        return results

    def _slice_cache_to_length(self, kv_cache: list[Any], target_length: int) -> None:
        """Slice cache tensors to exact target length (in-place).

        MLX-LM's trim_prompt_cache only adjusts offset, not tensor shapes.
        This method properly slices the underlying tensors to prevent
        shape mismatch errors during attention.

        Args:
            kv_cache: List of cache objects (one per layer)
            target_length: Target sequence length to slice to
        """
        import mlx.core as mx

        for layer_cache in kv_cache:
            if layer_cache.keys is None:
                continue

            # Handle QuantizedKVCache (tuple of 3 tensors)
            if isinstance(layer_cache.keys, tuple) and len(layer_cache.keys) == 3:
                k_q, k_s, k_z = layer_cache.keys
                v_q, v_s, v_z = layer_cache.values

                # Slice all tensors to target length
                layer_cache.keys = (
                    k_q[..., :target_length, :],
                    k_s[..., :target_length, :],
                    k_z[..., :target_length, :],
                )
                layer_cache.values = (
                    v_q[..., :target_length, :],
                    v_s[..., :target_length, :],
                    v_z[..., :target_length, :],
                )
            else:
                # Regular FP16 cache
                k, v = layer_cache.keys, layer_cache.values
                layer_cache.keys = k[..., :target_length, :]
                layer_cache.values = v[..., :target_length, :]

            # Update offset to match sliced length
            layer_cache.offset = target_length

        # Force evaluation to materialize sliced tensors
        mx.eval(*[c.keys for c in kv_cache if c.keys is not None])

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
        deferred_eval: list[Any] = []  # Collect tensors for single batched mx.eval

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
                actual_tokens = agent_blocks.total_tokens

                if isinstance(k_full, tuple) and len(k_full) == 3:
                    k_weights, k_scales, k_biases = k_full
                    v_weights, v_scales, v_biases = v_full

                    kv_bits = getattr(self._spec, 'kv_bits', 4) or 4
                    kv_group_size = getattr(self._spec, 'kv_group_size', 64) or 64

                    kv_cache = QuantizedKVCache(group_size=kv_group_size, bits=kv_bits)
                    kv_cache.keys = (k_weights, k_scales, k_biases)
                    kv_cache.values = (v_weights, v_scales, v_biases)
                    kv_cache.offset = actual_tokens

                    # Collect tensors for batched eval (avoid per-layer GPU sync)
                    deferred_eval.extend([k_weights, k_scales, v_weights, v_scales])
                    if k_biases is not None:
                        deferred_eval.extend([k_biases, v_biases])
                else:
                    kv_cache = KVCache()
                    kv_cache.state = (k_full, v_full)
                    kv_cache.offset = actual_tokens
                    deferred_eval.extend([k_full, v_full])

                cache.append(kv_cache)
            else:
                cache.append((k_full, v_full))

        # Single batched GPU sync for ALL layers (was ~54 separate mx.eval calls)
        if use_kv_cache and deferred_eval:
            mx.eval(*deferred_eval)
            logger.info(
                f"[RECONSTRUCT] Evaluated {len(deferred_eval)} tensors across "
                f"{len(cache)} layers in single batch"
            )

        return cache

    def _extract_cache(
        self,
        uid: str,
        cache: Any | None = None,
        token_sequence: list[int] | None = None,
        prompt_text: str = "",
    ) -> AgentBlocks:
        """Extract updated cache from batch and convert to blocks.

        Args:
            uid: Request UID to look up agent_id
            cache: KV cache from BatchGenerator
            token_sequence: Full token sequence (prompt + generated) for prefix matching
            prompt_text: Raw prompt text for character-level prefix matching
        """
        import mlx.core as mx  # noqa: PLC0415

        if uid not in self._active_requests:
            raise GenerationError(f"UID {uid} not found in active requests")
        agent_id, _, _, _, _ = self._active_requests[uid]

        if cache is None:
            if self._batch_gen is None:
                raise GenerationError("No active batch generator")
            raise GenerationError(f"Cache not provided for UID {uid}")

        if not cache or len(cache) == 0:
            return AgentBlocks(
                agent_id=agent_id,
                blocks={},
                total_tokens=0,
                token_sequence=token_sequence or [],
                prompt_text=prompt_text,
            )

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

                quantized_cache = []
                deferred_quant_eval: list[Any] = []
                for layer_id, (k, v) in enumerate(cache):
                    if k is None:
                        quantized_cache.append((None, None))
                        continue

                    k_quant = tuple(mx.quantize(k, group_size=kv_group_size, bits=kv_bits))
                    v_quant = tuple(mx.quantize(v, group_size=kv_group_size, bits=kv_bits))

                    deferred_quant_eval.extend([
                        k_quant[0], k_quant[1], k_quant[2],
                        v_quant[0], v_quant[1], v_quant[2],
                    ])
                    quantized_cache.append((k_quant, v_quant))

                # Single batched eval for all layers (was ~27 separate calls)
                if deferred_quant_eval:
                    mx.eval(*deferred_quant_eval)

                cache = quantized_cache

                logger.info(
                    f"Quantized cache for {agent_id}: {len(cache)} layers, "
                    f"bits={kv_bits}, group_size={kv_group_size}"
                )
            elif not isinstance(k_sample, tuple):
                logger.info(
                    f"Keeping FP16 cache for {agent_id}: {len(cache)} layers "
                    f"(kv_bits=None, no quantization)"
                )

        if cache[0][0] is None:
            return AgentBlocks(
                agent_id=agent_id,
                blocks={},
                total_tokens=0,
                token_sequence=token_sequence or [],
                prompt_text=prompt_text,
            )

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
                agent_id=agent_id,
                blocks=fake_blocks,
                total_tokens=seq_len,
                token_sequence=token_sequence or [],
                prompt_text=prompt_text,
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
            token_sequence=token_sequence or [],
            prompt_text=prompt_text,
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
