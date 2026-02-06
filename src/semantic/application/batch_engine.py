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

logger = logging.getLogger(__name__)


# Adaptive chunk size thresholds (token counts)
_CHUNK_TIER_SMALL_CACHE = 2000
_CHUNK_TIER_MEDIUM_CACHE = 8000
_CHUNK_TIER_LARGE_CACHE = 20000


def adaptive_chunk_size(
    cache_pos: int,
    min_chunk: int = 512,
    max_chunk: int = 4096,
) -> int:
    """Calculate optimal chunk size based on current cache position.

    Aggressive early (large chunks for speed), conservative late
    (small chunks to avoid OOM as cache grows).
    """
    if cache_pos < _CHUNK_TIER_SMALL_CACHE:
        return max_chunk
    elif cache_pos < _CHUNK_TIER_MEDIUM_CACHE:
        return max(min_chunk, max_chunk // 2)
    elif cache_pos < _CHUNK_TIER_LARGE_CACHE:
        return max(min_chunk, max_chunk // 4)
    else:
        return min_chunk


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

    # Pre-compiled regex for text cleaning (avoid re-compilation per call)
    _RE_MULTI_SPACES = re.compile(r" {3,}")
    _RE_MULTI_NEWLINES = re.compile(r"\n\n\n+")

    # Default chunked prefill settings
    DEFAULT_CHUNKED_PREFILL_ENABLED = True
    DEFAULT_CHUNKED_PREFILL_THRESHOLD = 2048
    DEFAULT_CHUNKED_PREFILL_MIN_CHUNK = 512
    DEFAULT_CHUNKED_PREFILL_MAX_CHUNK = 4096
    MAX_SAFE_CACHE_MEMORY_MB = 7500  # Safe on 24GB GPU with ~8GB model

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        pool: BlockPool,
        spec: ModelCacheSpec,
        cache_adapter: Any,
        batch_gen_factory: Callable[[Any, Any], Any] | None = None,
        adaptive_config: Any | None = None,
        chunked_prefill_enabled: bool = DEFAULT_CHUNKED_PREFILL_ENABLED,
        chunked_prefill_threshold: int = DEFAULT_CHUNKED_PREFILL_THRESHOLD,
        chunked_prefill_min_chunk: int = DEFAULT_CHUNKED_PREFILL_MIN_CHUNK,
        chunked_prefill_max_chunk: int = DEFAULT_CHUNKED_PREFILL_MAX_CHUNK,
    ) -> None:
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

        self._batch_gen: Any | None = None
        self._lock = threading.RLock()
        self._active_requests: dict[str, tuple[str, list[int], Any, list[int], str]] = {}
        self._agent_blocks: dict[str, AgentBlocks] = {}
        self._draining: bool = False
        self._native_completions: dict[str, CompletedGeneration] = {}

        self._chunked_prefill_enabled: bool | None = chunked_prefill_enabled
        self._chunked_prefill_threshold: int | None = chunked_prefill_threshold
        self._chunked_prefill_min_chunk: int | None = chunked_prefill_min_chunk
        self._chunked_prefill_max_chunk: int | None = chunked_prefill_max_chunk

    def _get_chunked_prefill_settings(self) -> tuple[bool, int, int, int]:
        """Get chunked prefill settings, applying adaptive adjustments if configured."""
        min_chunk = self._chunked_prefill_min_chunk or self.DEFAULT_CHUNKED_PREFILL_MIN_CHUNK
        max_chunk = self._chunked_prefill_max_chunk or self.DEFAULT_CHUNKED_PREFILL_MAX_CHUNK

        # Override chunk sizes from adaptive config if available
        if self._adaptive_config is not None:
            min_chunk, max_chunk = self._adaptive_config.effective_chunk_sizes

        return (
            self._chunked_prefill_enabled,
            self._chunked_prefill_threshold or 2048,
            min_chunk,
            max_chunk,
        )

    def _is_gpt_oss_model(self) -> bool:
        """Check if current model uses GPT-OSS Harmony format.

        GPT-OSS uses <|channel|> markers and <|end|> is a delimiter, not EOS.
        Only <|return|> should be a stop token for GPT-OSS.
        """
        chat_template = getattr(self._tokenizer, "chat_template", "") or ""
        return "<|channel|>" in chat_template and "<|start|>" in chat_template

    def _build_stop_tokens(self) -> set[int]:
        """Build set of stop tokens for current model.

        Different models use different special tokens for end-of-turn:
        - Standard: eos_token_id
        - Llama 3.1: <|eot_id|> (128009) for end-of-turn
        - GPT-OSS: <|return|> only (NOT <|end|>, which is a message delimiter)
        - DeepSeek: <｜end▁of▁sentence｜> (special unicode)

        Returns:
            Set of token IDs that should stop generation.
        """
        stop_tokens: set[int] = set()

        # Always include standard EOS
        if self._tokenizer.eos_token_id is not None:
            stop_tokens.add(self._tokenizer.eos_token_id)

        # Check if this is a GPT-OSS model (uses Harmony format)
        is_gpt_oss = self._is_gpt_oss_model()

        # Model-specific stop tokens
        special_stop_tokens = [
            # Llama 3.1 end-of-turn
            "<|eot_id|>",
            # GPT-OSS return (end of full response)
            "<|return|>",
            # DeepSeek (special unicode)
            "<｜end▁of▁sentence｜>",
            # Common alternatives
            "<|endoftext|>",
            "<|im_end|>",
        ]

        # For non-GPT-OSS models, also include <|end|> as stop token
        # GPT-OSS uses <|end|> as message delimiter, not EOS
        if not is_gpt_oss:
            special_stop_tokens.append("<|end|>")
        else:
            logger.info("[STOP_TOKENS] GPT-OSS detected, excluding <|end|> from stop tokens")

        for token_str in special_stop_tokens:
            try:
                token_id = self._tokenizer.convert_tokens_to_ids(token_str)
                # Only add if it's a valid token (not UNK)
                if token_id != self._tokenizer.unk_token_id and token_id is not None:
                    stop_tokens.add(token_id)
            except (KeyError, AttributeError):
                # Token doesn't exist in this tokenizer's vocabulary
                pass

        logger.debug(
            "[STOP_TOKENS] Built stop set: %s (eos=%s, gpt_oss=%s)",
            stop_tokens,
            self._tokenizer.eos_token_id,
            is_gpt_oss,
        )

        return stop_tokens

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
        logger.info(
            f"[MEMORY {label}] Active: {active:.2f}GB, Cache: {cache:.2f}GB, Peak: {peak:.2f}GB"
        )
        return active, cache, peak

    # --- Public accessors (avoid direct access to private attributes) ---

    @property
    def tokenizer(self) -> Any:
        """Get the tokenizer instance."""
        return self._tokenizer

    def get_agent_blocks(self, agent_id: str) -> "AgentBlocksType | None":
        """Get agent blocks by ID (thread-safe).

        Returns a SNAPSHOT of the agent's blocks to prevent race conditions
        where the caller saves blocks while another thread mutates them.

        Args:
            agent_id: The agent identifier

        Returns:
            AgentBlocks snapshot if found, None otherwise
        """
        from semantic.domain.entities import AgentBlocks, KVBlock

        with self._lock:
            original = self._agent_blocks.get(agent_id)
            if original is None:
                return None

            # Create a DEEP COPY of blocks to prevent mutation races
            # Without this, concurrent requests can corrupt the cache being saved
            blocks_copy: dict[int, list[KVBlock]] = {}
            for layer_id, layer_blocks in original.blocks.items():
                # Only copy blocks with actual data
                valid_blocks = []
                for block in layer_blocks:
                    if block.layer_data is not None:
                        # Shallow copy of KVBlock (layer_data references are OK)
                        block_copy = KVBlock(
                            block_id=block.block_id,
                            layer_id=block.layer_id,
                            token_count=block.token_count,
                            layer_data=block.layer_data,  # Reference is fine
                            metadata=block.metadata.copy() if block.metadata else None,
                        )
                        valid_blocks.append(block_copy)
                blocks_copy[layer_id] = valid_blocks

            # Create snapshot AgentBlocks
            return AgentBlocks(
                agent_id=original.agent_id,
                blocks=blocks_copy,
                total_tokens=original.total_tokens,
                token_sequence=original.token_sequence.copy() if original.token_sequence else [],
                prompt_text=original.prompt_text,
            )

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
        kv_bits = getattr(self._spec, "kv_bits", 4) or 4
        kv_group_size = getattr(self._spec, "kv_group_size", 64) or 64

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
            # Note: Do NOT call mx.clear_cache() here. Per-chunk clearing
            # destroys the warmed Metal buffer pool, forcing expensive
            # reallocation on every chunk. The intermediate y tensor is
            # naturally freed when Python rebinds the variable next iteration.

            chunk_time = time.time() - chunk_start_time
            total_time += chunk_time
            chunk_count += 1

            if chunk_count <= 3 or chunk_count % 10 == 0:
                logger.info(
                    f"[CHUNK {chunk_count}] Pos: {pos}->{end} ({end - pos} tokens), "
                    f"Time: {chunk_time * 1000:.0f}ms"
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

    def submit(
        self,
        agent_id: str,
        prompt: str,
        cache: "AgentBlocksType | None" = None,  # Proper type annotation
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
            max_safe_memory_mb = self.MAX_SAFE_CACHE_MEMORY_MB
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
                    logger.debug(
                        "[MEMORY BEFORE RECONSTRUCT] Active: %.2fGB, Cache: %.2fGB, Peak: %.2fGB",
                        mem_before,
                        cache_before,
                        peak_before,
                    )
                    logger.debug(
                        "[CACHE] Agent: %s, Tokens: %d, Blocks: %d",
                        agent_id,
                        cache.total_tokens,
                        sum(len(blocks) for blocks in cache.blocks.values()),
                    )

                    # Early EXACT match detection: if the stored prompt matches
                    # the current prompt, only reconstruct prompt-token blocks.
                    # Skips loading generated-token blocks that would be trimmed
                    # anyway (saves 20-50% reconstruction at high output counts).
                    stored_text = getattr(cache, "prompt_text", "")
                    n_prompt = len(prompt_tokens)
                    max_reconstruct: int | None = None
                    if stored_text and stored_text == prompt:
                        max_reconstruct = n_prompt
                        n_skipped = cache.total_tokens - n_prompt
                        logger.debug(
                            "[EARLY EXACT] Limiting reconstruction to %d prompt tokens "
                            "(skipping %d generated tokens)",
                            n_prompt,
                            n_skipped,
                        )
                        # Free generated-token GPU memory BEFORE reconstruction
                        # to reduce memory pressure. gc.collect() releases Python
                        # wrappers so MLX can reclaim GPU memory. Do NOT call
                        # mx.clear_cache() — that would destroy the warmed Metal
                        # memory pool and degrade all subsequent allocations.
                        if n_skipped > 0:
                            freed_blocks = 0
                            for layer_blocks in cache.blocks.values():
                                tokens_seen = 0
                                for block in layer_blocks:
                                    block_start = tokens_seen  # Start position of this block
                                    tokens_seen += block.token_count
                                    # Only free blocks that START after the prompt
                                    # (contain ONLY generated tokens, not prompt tokens)
                                    if block_start >= n_prompt and block.layer_data is not None:
                                        block.layer_data = None
                                        freed_blocks += 1
                            if freed_blocks > 0:
                                gc.collect()
                                logger.debug(
                                    "[EARLY EXACT FREE] Freed %d generated-token blocks before reconstruction",
                                    freed_blocks,
                                )

                    reconstruct_tokens = max_reconstruct or cache.total_tokens
                    q4_memory_mb = (reconstruct_tokens * bytes_per_token_q4) / (1024 * 1024)
                    logger.debug(
                        "[Q4 BLOCKS] Estimated Q4 memory: %.1fMB for %d tokens",
                        q4_memory_mb,
                        reconstruct_tokens,
                    )

                    import time as _time

                    t_reconstruct_start = _time.perf_counter()
                    kv_cache = self._reconstruct_cache(cache, max_tokens=max_reconstruct)
                    t_reconstruct_end = _time.perf_counter()
                    logger.info(
                        "[TIMING] reconstruct=%.3fs tokens=%d",
                        t_reconstruct_end - t_reconstruct_start,
                        reconstruct_tokens,
                    )

                    # Memory tracking AFTER reconstruction (should be SAME as before - no dequant!)
                    mem_after_reconstruct = mx.get_active_memory() / (1024**3)
                    cache_after_reconstruct = mx.get_cache_memory() / (1024**3)
                    peak_after_reconstruct = mx.get_peak_memory() / (1024**3)
                    logger.debug(
                        "[MEMORY AFTER RECONSTRUCT] Active: %.2fGB, Cache: %.2fGB, Peak: %.2fGB",
                        mem_after_reconstruct,
                        cache_after_reconstruct,
                        peak_after_reconstruct,
                    )
                    logger.debug(
                        "[MEMORY DELTA RECONSTRUCT] Active: +%.2fGB, Cache: +%.2fGB",
                        mem_after_reconstruct - mem_before,
                        cache_after_reconstruct - cache_before,
                    )

                    cache_len = len(kv_cache) if kv_cache else 0
                    logger.debug(
                        "[RECONSTRUCT] Created %d layer caches, type: %s",
                        cache_len,
                        type(kv_cache[0]) if kv_cache else None,
                    )

                    # Verify cache properties are set correctly - raise exceptions for mismatches
                    if kv_cache and len(kv_cache) > 0:
                        first_layer = kv_cache[0]
                        first_offset = getattr(first_layer, "offset", None)
                        first_size = first_layer.size() if hasattr(first_layer, "size") else None
                        logger.debug(
                            "[CACHE VALIDATION] Layer 0: offset=%s, size()=%s, total=%s, max_reconstruct=%s",
                            first_offset,
                            first_size,
                            cache.total_tokens,
                            max_reconstruct,
                        )
                        if max_reconstruct is not None:
                            # Partial reconstruction (EARLY EXACT): offset may exceed
                            # max_reconstruct due to block alignment (256-token blocks)
                            # but must be <= total_tokens and >= max_reconstruct.
                            if first_offset is not None and first_offset < max_reconstruct:
                                raise GenerationError(
                                    f"Cache offset too small: layer 0 offset ({first_offset}) < "
                                    f"requested reconstruction ({max_reconstruct})."
                                )
                        else:
                            # Full reconstruction: offset must exactly match total_tokens.
                            if first_offset is not None and first_offset != cache.total_tokens:
                                raise GenerationError(
                                    f"Cache offset mismatch: layer 0 offset ({first_offset}) != "
                                    f"expected tokens ({cache.total_tokens}). This would cause shape mismatch."
                                )
                        if (
                            first_size is not None
                            and first_offset is not None
                            and first_size != first_offset
                        ):
                            raise GenerationError(
                                f"Cache size/offset mismatch: layer 0 size() ({first_size}) != "
                                f"offset ({first_offset}). BatchGenerator won't recognize cache."
                            )

                    loaded_desc = (
                        f"{first_offset} (partial)"
                        if max_reconstruct is not None
                        else str(cache.total_tokens)
                    )
                    logger.debug(
                        "[RECONSTRUCT COMPLETE] Loaded %s tokens across %d layers as QuantizedKVCache (Q4)",
                        loaded_desc,
                        len(cache.blocks),
                    )

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

                    logger.debug("[Q4 FREED] Cleared %d blocks to free Q4 memory", blocks_cleared)

                    # Memory tracking AFTER freeing Q4
                    mem_after_free = mx.get_active_memory() / (1024**3)
                    cache_after_free = mx.get_cache_memory() / (1024**3)
                    peak_after_free = mx.get_peak_memory() / (1024**3)
                    logger.debug(
                        "[MEMORY POST-FREE] Active: %.2fGB, Cache: %.2fGB, Peak: %.2fGB",
                        mem_after_free,
                        cache_after_free,
                        peak_after_free,
                    )
                    logger.debug(
                        "[MEMORY DELTA] Active: +%.2fGB",
                        mem_after_free - mem_before,
                    )

                except Exception as e:
                    logger.error(f"Cache reconstruction failed for {agent_id}: {e}", exc_info=True)
                    cache = None
                    kv_cache = None

        if cache is None:
            # No cache - check if we should use chunked prefill for long prompts
            enabled, threshold, min_chunk, max_chunk = self._get_chunked_prefill_settings()

            # Use chunked prefill for long prompts (reduces peak memory by 38-65%)
            if enabled and len(prompt_tokens) >= threshold and self._batch_gen_factory is None:
                logger.debug(
                    "[PREFILL MODE] Chunked prefill for %d tokens (threshold: %d)",
                    len(prompt_tokens),
                    threshold,
                )

                # Run adaptive chunked prefill to build cache with lower memory usage
                kv_cache = self._chunked_prefill(prompt_tokens, agent_id)

                # Allocate minimal blocks for tracking (actual cache is in kv_cache)
                n_blocks_needed = (
                    len(prompt_tokens) + self._spec.block_tokens - 1
                ) // self._spec.block_tokens
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

                logger.debug(
                    "[CHUNKED PREFILL DONE] Agent: %s, prefilled=%d tokens",
                    agent_id,
                    len(prompt_tokens),
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
                    len(prompt_tokens) + self._spec.block_tokens - 1
                ) // self._spec.block_tokens

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
                    # Build model-specific stop tokens
                    stop_tokens = self._build_stop_tokens()

                    self._batch_gen = self._cache_adapter.create_batch_generator(
                        model=self._model,
                        stop_tokens=stop_tokens,
                        kv_bits=self._spec.kv_bits,
                        kv_group_size=self._spec.kv_group_size,
                    )

        # Insert into batch
        try:
            # Create sampler via adapter if using real MLX (not fake for testing)
            samplers = None
            logits_processors_list = None

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
                cached_tokens = getattr(first_cache, "offset", 0) or 0
                tokens_to_process = [prompt_tokens[-1]]
                logger.debug(
                    "[CHUNKED PREFILL INSERT] Cache has %d tokens from prefill",
                    cached_tokens,
                )

            # CASE 2: kv_cache from loaded cache (cache is not None)
            elif kv_cache is not None and cache is not None:
                first_cache = kv_cache[0] if isinstance(kv_cache, list) else kv_cache
                cached_tokens = getattr(first_cache, "offset", 0) or 0
                cache_type = type(first_cache).__name__
                has_bits = hasattr(first_cache, "bits")

                # Character-level prefix matching (avoids BPE boundary mismatches)
                stored_text = getattr(cache, "prompt_text", "")

                t_match_start = _time.perf_counter()
                if stored_text:
                    char_match = cache.common_prefix_chars(prompt)
                    logger.info(
                        "[CACHE PREFIX] agent=%s char_match=%d stored=%d prompt=%d ratio=%.1f%%",
                        agent_id,
                        char_match,
                        len(stored_text),
                        len(prompt),
                        (char_match / len(stored_text) * 100) if stored_text else 0,
                    )
                    if char_match < len(stored_text) and char_match < len(prompt):
                        ctx = 40
                        stored_at = stored_text[char_match:char_match + ctx]
                        prompt_at = prompt[char_match:char_match + ctx]
                        logger.debug(
                            "[CACHE DIVERGE AT] stored='%s' vs prompt='%s'",
                            stored_at.replace("\n", "\\n"),
                            prompt_at.replace("\n", "\\n"),
                        )
                        # When char_match=0, log the start of both strings to understand mismatch
                        if char_match == 0:
                            logger.info(
                                "[CACHE ZERO MATCH] stored[:80]=%r prompt[:80]=%r",
                                stored_text[:80],
                                prompt[:80],
                            )
                        # Log divergence point for debugging
                        if char_match > 0 and char_match < len(stored_text):
                            ctx = 30
                            logger.info(
                                "[CACHE DIVERGE POINT] at char %d: stored='...%s' vs prompt='...%s'",
                                char_match,
                                stored_text[char_match:char_match+ctx].replace('\n', '\\n'),
                                prompt[char_match:char_match+ctx].replace('\n', '\\n'),
                            )

                    if char_match == len(stored_text) == len(prompt):
                        # EXACT MATCH: Reuse prompt portion of cache only.
                        # With early EXACT detection, reconstruction already
                        # limited to prompt tokens (trim is just 1 token).
                        # Without it, cached_tokens includes generated tokens
                        # and this trim removes the generated portion.
                        n_prompt = len(prompt_tokens)
                        trim_to = max(n_prompt - 1, 0)
                        logger.debug(
                            "[CACHE EXACT HIT] %d chars match, trimming %d→%d "
                            "(prompt=%d, cached=%d)",
                            char_match,
                            cached_tokens,
                            trim_to,
                            n_prompt,
                            cached_tokens,
                        )
                        self._slice_cache_to_length(kv_cache, trim_to)
                        tokens_to_process = [prompt_tokens[-1]]

                    elif char_match == len(stored_text) < len(prompt):
                        # EXTEND: Full stored text matches, new text appended
                        stored_prompt_token_count = (
                            len(cache.token_sequence) if cache.token_sequence else 0
                        )
                        if (
                            cache.total_tokens > stored_prompt_token_count
                            and stored_prompt_token_count > 0
                        ):
                            logger.debug(
                                "[CACHE EXTEND] Trimming from %d to %d tokens",
                                cache.total_tokens,
                                stored_prompt_token_count,
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
                            logger.debug(
                                "[CACHE EXTEND + CHUNKED] %d new tokens >= threshold %d",
                                len(new_tokens),
                                cp_threshold,
                            )
                            kv_cache = self._chunked_prefill(
                                new_tokens, agent_id, initial_cache=kv_cache
                            )
                            tokens_to_process = [new_tokens[-1]]
                        else:
                            tokens_to_process = new_tokens

                        logger.debug(
                            "[CACHE EXTEND] Reusing %d chars, processing %d new tokens",
                            char_match,
                            len(tokens_to_process),
                        )

                    elif char_match < len(stored_text):
                        # DIVERGE: Prompt diverges from stored text
                        match_ratio = char_match / len(stored_text)

                        if match_ratio >= 0.5 and char_match > 100:
                            # FIX: Tokenize FULL prompt first, then find token boundary
                            # This avoids BPE boundary issues from split tokenization
                            full_tokens = self._tokenizer.encode(prompt)

                            # Find token boundary: decode tokens until we exceed char_match
                            # This gives us the exact token count for the matched prefix
                            usable_tokens = 0
                            decoded_len = 0
                            for i, tok in enumerate(full_tokens):
                                tok_text = self._tokenizer.decode([tok])
                                decoded_len += len(tok_text)
                                if decoded_len >= char_match:
                                    usable_tokens = i  # Stop BEFORE exceeding char_match
                                    break

                            # If we couldn't find boundary, fall back to prefix tokenization
                            if usable_tokens == 0:
                                matched_tokens = self._tokenizer.encode(prompt[:char_match])
                                usable_tokens = len(matched_tokens)
                                remaining_tokens = self._tokenizer.encode(prompt[char_match:])
                            else:
                                # Use exact tokens from full tokenization
                                remaining_tokens = full_tokens[usable_tokens:]

                            logger.info(
                                "[CACHE DIVERGE PARTIAL] chars=%d (%.0f%%), "
                                "cache_tokens=%d, new_tokens=%d, full_tokens=%d",
                                char_match,
                                match_ratio * 100,
                                usable_tokens,
                                len(remaining_tokens),
                                len(full_tokens),
                            )
                            self._slice_cache_to_length(kv_cache, usable_tokens)
                            import mlx.core as mx

                            gc.collect()
                            mx.clear_cache()

                            # Chunked prefill for long remaining portion
                            cp_enabled, cp_threshold, _, _ = self._get_chunked_prefill_settings()
                            if cp_enabled and len(remaining_tokens) >= cp_threshold:
                                logger.debug(
                                    "[CACHE DIVERGE + CHUNKED] %d remaining tokens >= threshold %d",
                                    len(remaining_tokens),
                                    cp_threshold,
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
                                "[CACHE DIVERGE MISS] %d chars matched (%.0f%% < 50%%)",
                                char_match,
                                match_ratio * 100,
                            )
                            if kv_cache is not None:
                                import mlx.core as mx

                                del kv_cache
                                gc.collect()
                                mx.clear_cache()
                            kv_cache = None
                            cache = None
                            tokens_to_process = prompt_tokens

                    t_match_end = _time.perf_counter()
                    logger.info(
                        "[TIMING] prefix_match=%.3fs tokens_to_process=%d",
                        t_match_end - t_match_start,
                        len(tokens_to_process),
                    )
                else:
                    # No prompt_text stored — fall back to token comparison (legacy)
                    logger.info(
                        "[CACHE NO PROMPT_TEXT] agent=%s cached_tokens=%d — using legacy path",
                        agent_id,
                        cached_tokens,
                    )
                    cached_token_seq = getattr(cache, "token_sequence", [])
                    if cached_token_seq:
                        common_prefix = cache.common_prefix_length(prompt_tokens)
                        if common_prefix == len(cached_token_seq) < len(prompt_tokens):
                            tokens_to_process = prompt_tokens[common_prefix:]
                            logger.debug(
                                "[CACHE LEGACY EXTEND] Reusing %d tokens",
                                common_prefix,
                            )
                        else:
                            logger.debug(
                                "[CACHE LEGACY MISS] prefix=%d, cached=%d, prompt=%d",
                                common_prefix,
                                len(cached_token_seq),
                                len(prompt_tokens),
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
                        logger.debug(
                            "[CACHE LEGACY OFFSET] Skipping %d cached tokens", cached_tokens
                        )
                    elif cached_tokens >= len(prompt_tokens):
                        tokens_to_process = [prompt_tokens[-1]]
                        logger.debug("[CACHE LEGACY EXACT] Using last token as seed")

                # Safety check: if tokens_to_process is empty, use last token
                if not tokens_to_process and prompt_tokens:
                    logger.warning("[CACHE FALLBACK] Empty tokens_to_process, using last token")
                    tokens_to_process = [prompt_tokens[-1]]

                logger.info(
                    "[CACHE INJECT] %s: offset=%d, quantized=%s, prompt=%d, processing=%d",
                    cache_type,
                    cached_tokens,
                    has_bits,
                    len(prompt_tokens),
                    len(tokens_to_process),
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
            if logits_processors_list is not None:
                insert_kwargs["logits_processors"] = logits_processors_list
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
            self._active_requests[actual_uid] = (
                agent_id,
                [],
                detokenizer,
                list(prompt_tokens),
                prompt,
            )

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
                raise PoolExhaustedError("Engine is draining - not accepting new requests.")

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
                    # Build model-specific stop tokens
                    stop_tokens = self._build_stop_tokens()

                    self._batch_gen = self._cache_adapter.create_batch_generator(
                        model=self._model,
                        stop_tokens=stop_tokens,
                        kv_bits=self._spec.kv_bits,
                        kv_group_size=self._spec.kv_group_size,
                    )

        # Insert with last prompt token — cache has all-but-last token KV.
        # BatchGenerator requires at least one token to produce initial logits.
        try:
            samplers = None
            logits_processors_list = None

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
            if logits_processors_list is not None:
                insert_kwargs["logits_processors"] = logits_processors_list

            cached_tokens = kv_caches[0].offset if hasattr(kv_caches[0], "offset") else 0
            logger.debug(
                "[SUBMIT_WITH_CACHE] agent=%s, cache_tokens=%d, max_tokens=%d",
                agent_id,
                cached_tokens,
                max_tokens,
            )

            uids = self._batch_gen.insert(**insert_kwargs)
        except Exception as e:
            raise InvalidRequestError(f"Failed to insert pre-prefilled sequence: {e}") from e

        actual_uid: str = uids[0]

        detokenizer_class = type(self._tokenizer.detokenizer)
        detokenizer = detokenizer_class(self._tokenizer)
        with self._lock:
            self._active_requests[actual_uid] = (
                agent_id,
                [],
                detokenizer,
                list(prompt_tokens),
                prompt_text,
            )

        return actual_uid

    def step(self, max_iterations: int = 100_000) -> Iterator[CompletedGeneration]:
        """Execute one batch decode step and yield completed generations.

        Yields:
            CompletedGeneration for each sequence that finished this step.
            Sequences finish when:
            - EOS token generated (finish_reason="stop")
            - max_tokens limit reached (finish_reason="length")
            - Error occurred (finish_reason="error")

        Args:
            max_iterations: Safety limit on loop iterations to prevent infinite loops.
                If BatchGenerator stalls (no sequence reaches finish_reason), this
                ensures step() eventually returns.

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

        iterations = 0
        while True:
            if iterations >= max_iterations:
                logger.error("step() exceeded %d iterations — breaking to prevent infinite loop", max_iterations)
                break
            iterations += 1
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
                    agent_id, tokens, detokenizer, prompt_tokens, prompt_text = (
                        self._active_requests[uid]
                    )

                if response.finish_reason != "stop":
                    tokens.append(response.token)
                    detokenizer.add_token(response.token)
                    if len(tokens) <= 5:
                        logger.debug("Token %d: %d", len(tokens), response.token)
                if response.finish_reason is not None:
                    yield self._finalize_sequence(
                        uid, response, agent_id, tokens, prompt_tokens, prompt_text
                    )

    def _clean_text(self, text: str) -> str:
        """Remove BPE artifacts, model-specific tokens, and normalize whitespace.

        For GPT-OSS models that output analysis before final response, extracts
        only the final channel content.
        """
        # GPT-OSS: Extract final channel content
        # Normal: <|channel|>analysis<|message|>...<|end|><|start|>assistant<|channel|>final<|message|>...
        # Malformed: <|channel|>final<|channel|>commentary<|message|>... (missing <|message|> after final)
        if "<|channel|>final" in text:
            # Find the last occurrence of final channel marker
            final_idx = text.rfind("<|channel|>final")
            text_after_final = text[final_idx:]

            # Look for <|message|> after the final channel marker
            msg_idx = text_after_final.find("<|message|>")
            if msg_idx != -1:
                text = text_after_final[msg_idx + len("<|message|>"):]
                # Remove trailing markers
                for marker in ("<|end|>", "<|return|>"):
                    if marker in text:
                        text = text.split(marker)[0]
            else:
                # No <|message|> found - malformed, return empty
                logger.warning("[GPT-OSS] Malformed final channel (no message marker)")
                return ""
        elif "<|channel|>analysis<|message|>" in text and "<|channel|>final" not in text:
            # Pure analysis mode - don't leak internal reasoning
            logger.warning("[GPT-OSS] Model stuck in analysis mode, returning empty response")
            return ""
        else:
            # No channel markers - clean up any stray markers
            text = text.replace("<|channel|>commentary<|message|>", "")
            text = text.replace("<|channel|>final<|message|>", "")
            text = text.replace("<|end|>", "")
            text = text.replace("<|return|>", "")

        # Remove any remaining start markers (from incomplete generation)
        if "<|start|>assistant" in text:
            text = text.split("<|start|>assistant")[0]

        # Remove any remaining <|...|> markers (GPT-OSS artifacts)
        text = re.sub(r"<\|[^|]+\|>", "", text)

        # BPE artifacts
        text = text.replace("Ġ", " ")
        text = text.replace("Ċ", "\n")
        text = text.replace("ċ", "\n")
        text = text.replace("▁", " ")
        text = self._RE_MULTI_SPACES.sub("  ", text)
        text = self._RE_MULTI_NEWLINES.sub("\n\n", text)
        return text.strip()

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
            logger.debug(
                "[BATCH EXTRACT] uid=%s, cache_len=%d, type=%s, offset=%s",
                uid,
                len(cache),
                type(first).__name__,
                getattr(first, "offset", "N/A"),
            )

        blocks = self._extract_cache(uid, cache, full_token_sequence, prompt_text=prompt_text)

        with self._lock:
            self._agent_blocks[agent_id] = blocks

        # DEBUG: Log tokens being decoded
        logger.info(
            "[DECODE] uid=%s agent=%s token_count=%d first_10_tokens=%s",
            uid,
            agent_id,
            len(tokens),
            tokens[:10] if tokens else [],
        )

        text = self._tokenizer.decode(tokens)

        # DEBUG: Log raw decoded text
        logger.info(
            "[RAW_DECODED] uid=%s length=%d text=%s",
            uid,
            len(text),
            repr(text[:200]) if text else "(empty)",
        )

        # Debug: log raw text before cleaning to diagnose GPT-OSS channel issues
        if "<|" in text or "channel" in text.lower():
            logger.info("[RAW OUTPUT] %s: %s", uid, text[:500])

        text_before_clean = text
        text = self._clean_text(text)

        # DEBUG: Log what cleaning did
        if text != text_before_clean:
            logger.info(
                "[CLEANED] uid=%s before_len=%d after_len=%d removed=%d chars",
                uid,
                len(text_before_clean),
                len(text),
                len(text_before_clean) - len(text),
            )

        logger.debug("Finalized %s: %d tokens", uid, len(tokens))

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
                results.append(
                    StepOneResult(
                        uid=uid,
                        text=completion.text,
                        token_count=completion.token_count,
                        finish_reason=completion.finish_reason,
                        completion=completion,
                    )
                )

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

        # Batch=1 fast path: skip for-loop, containment check, debug logging
        if len(batch_response) == 1:
            response = batch_response[0]
            uid = response.uid
            with self._lock:
                req = self._active_requests.get(uid)
            if req is None:
                return results
            agent_id, tokens, detokenizer, prompt_tokens, prompt_text = req

            if response.finish_reason != "stop":
                tokens.append(response.token)
                detokenizer.add_token(response.token)

            if response.finish_reason is not None:
                completion = self._finalize_sequence(
                    uid,
                    response,
                    agent_id,
                    tokens,
                    prompt_tokens,
                    prompt_text,
                )
                results.append(
                    StepOneResult(
                        uid=uid,
                        text=completion.text,
                        token_count=completion.token_count,
                        finish_reason=completion.finish_reason,
                        completion=completion,
                    )
                )
            else:
                results.append(
                    StepOneResult(
                        uid=uid,
                        text=detokenizer.text,
                        token_count=len(tokens),
                        finish_reason=None,
                        completion=None,
                    )
                )
            return results

        # Multi-sequence path (batch > 1)
        for response in batch_response:
            uid = response.uid

            with self._lock:
                if uid not in self._active_requests:
                    logger.error("Untracked UID %s in step_once", uid)
                    continue
                agent_id, tokens, detokenizer, prompt_tokens, prompt_text = self._active_requests[
                    uid
                ]

            if response.finish_reason != "stop":
                tokens.append(response.token)
                detokenizer.add_token(response.token)
                if len(tokens) <= 5:
                    logger.debug("Token %d: %d", len(tokens), response.token)

            if response.finish_reason is not None:
                completion = self._finalize_sequence(
                    uid,
                    response,
                    agent_id,
                    tokens,
                    prompt_tokens,
                    prompt_text,
                )
                results.append(
                    StepOneResult(
                        uid=uid,
                        text=completion.text,
                        token_count=completion.token_count,
                        finish_reason=completion.finish_reason,
                        completion=completion,
                    )
                )
            else:
                results.append(
                    StepOneResult(
                        uid=uid,
                        text=detokenizer.text,
                        token_count=len(tokens),
                        finish_reason=None,
                        completion=None,
                    )
                )

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

        # Force evaluation to materialize ALL sliced tensors (keys AND values).
        # Both must be concrete before BatchGenerator uses them for attention.
        tensors_to_eval: list[Any] = []
        for c in kv_cache:
            if c.keys is not None:
                if isinstance(c.keys, tuple):
                    tensors_to_eval.extend(t for t in c.keys if t is not None)
                    tensors_to_eval.extend(t for t in c.values if t is not None)
                else:
                    tensors_to_eval.extend([c.keys, c.values])
        if tensors_to_eval:
            mx.eval(*tensors_to_eval)

    def _reconstruct_cache(
        self,
        agent_blocks: AgentBlocks,
        max_tokens: int | None = None,
    ) -> list[Any]:
        """Reconstruct MLX cache objects from blocks.

        Args:
            agent_blocks: Blocks to reconstruct from.
            max_tokens: If set, only load blocks up to this token count.
                Skips remaining blocks to avoid reconstructing generated
                tokens that will be trimmed anyway (EXACT match optimization).

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

            # Extract K and V tensors from blocks (respecting max_tokens limit)
            k_tensors = []
            v_tensors = []
            tokens_loaded = 0
            for block in layer_blocks:
                if max_tokens is not None and tokens_loaded >= max_tokens:
                    break  # Skip generated-token blocks (EXACT match optimization)

                if block.layer_data is None or "k" not in block.layer_data:
                    raise GenerationError(
                        f"Block {block.block_id} for layer {layer_id} has no K/V data"
                    )
                tokens_loaded += block.token_count

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
            k_full, v_full = self._cache_adapter.concatenate_cache_blocks(k_tensors, v_tensors)

            if use_kv_cache:
                actual_tokens = (
                    tokens_loaded if max_tokens is not None else agent_blocks.total_tokens
                )

                if isinstance(k_full, tuple) and len(k_full) == 3:
                    k_weights, k_scales, k_biases = k_full
                    v_weights, v_scales, v_biases = v_full

                    kv_bits = getattr(self._spec, "kv_bits", 4) or 4
                    kv_group_size = getattr(self._spec, "kv_group_size", 64) or 64

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
        if hasattr(first_layer, "state"):
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

                    deferred_quant_eval.extend(
                        [
                            k_quant[0],
                            k_quant[1],
                            k_quant[2],
                            v_quant[0],
                            v_quant[1],
                            v_quant[2],
                        ]
                    )
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
        if first_tensor.__class__.__name__ == "FakeTensor":
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

            # Use step_once() instead of step() to avoid blocking the event loop.
            # step() runs a while-True loop that only exits when all sequences
            # finish. step_once() processes exactly one batch_gen.next() call,
            # allowing the timeout check and asyncio.sleep() to execute between
            # iterations.
            results = self.step_once()
            for result in results:
                if result.finish_reason is not None:
                    drained_count += 1

            # Yield to event loop frequently to keep timeout checks responsive
            await asyncio.sleep(0.01)

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
