"""MLX QuantizedKVCache extensions for batching support.

Adds merge() method to QuantizedKVCache to enable direct Q4 cache injection.
This allows us to keep Q4 format end-to-end (storage → injection → generation)
without dequantizing to FP16, maintaining 75% memory savings.
"""

import mlx.core as mx
from typing import Any
import logging

# Import MLX's base cache class so model recognizes our cache
try:
    from mlx_lm.models.cache import _BaseCache
except ImportError:
    _BaseCache = object  # Fallback if not available

logger = logging.getLogger(__name__)


class BatchQuantizedKVCache(_BaseCache):
    """Batched quantized KV cache for multiple sequences.

    This class provides a batched version of QuantizedKVCache that can be
    used with BatchGenerator. It maintains the Q4 quantized format to
    save 75% memory compared to FP16.

    Implements the full interface expected by BatchGenerator:
    - prepare(): Configure cache before generation
    - update_and_fetch(): Update with new tokens and return for attention
    - finalize(): Post-generation cleanup
    - filter(): Batch filtering
    - empty(): Check if initialized
    - make_mask(): Generate attention masks
    """

    step = 256

    def __init__(self, padding: list[int] | None = None, group_size: int = 64, bits: int = 4):
        """Initialize batched quantized cache.

        Args:
            padding: List of padding lengths for each sequence (optional)
            group_size: Quantization group size (default: 64)
            bits: Number of bits for quantization (default: 4)
        """
        self.keys: tuple[Any, Any, Any] | None = None
        self.values: tuple[Any, Any, Any] | None = None
        self.offset = 0
        self._idx = 0
        self._left_padding: list[int] = []
        self._right_padding: list[int] = padding or []
        self._lengths: list[int] = []
        self.group_size = group_size
        self.bits = bits

    @classmethod
    def merge(cls, caches: list[Any]) -> "BatchQuantizedKVCache":
        """Merge multiple QuantizedKVCache instances into batched cache.

        This is the key method that enables Q4 direct injection. Instead of
        dequantizing Q4→FP16 (10GB spike), we keep Q4 format throughout.

        Args:
            caches: List of QuantizedKVCache instances to merge

        Returns:
            BatchQuantizedKVCache with merged Q4 data

        Raises:
            ValueError: If caches list is empty
        """
        if not caches:
            raise ValueError("Cannot merge empty cache list")

        first_cache = caches[0]
        group_size = first_cache.group_size
        bits = first_cache.bits

        lengths = [c.offset for c in caches]
        max_length = max(lengths)
        left_padding = [max_length - l for l in lengths]
        B = len(caches)

        # Get dimensions from first non-empty cache
        sample_cache = next(c for c in caches if c.keys is not None)
        k_quant, k_scales, k_zeros = sample_cache.keys
        v_quant, v_scales, v_zeros = sample_cache.values

        H = k_quant.shape[1]
        Dk_packed = k_quant.shape[-1]
        Dv_packed = v_quant.shape[-1]
        Dk_scales = k_scales.shape[-1]
        Dv_scales = v_scales.shape[-1]

        dt = k_scales.dtype

        # Allocate batched tensors (stays Q4!)
        keys_quant = mx.zeros((B, H, max_length, Dk_packed), dtype=mx.uint32)
        keys_scales = mx.zeros((B, H, max_length, Dk_scales), dtype=dt)
        keys_zeros = mx.zeros((B, H, max_length, Dk_scales), dtype=dt)

        values_quant = mx.zeros((B, H, max_length, Dv_packed), dtype=mx.uint32)
        values_scales = mx.zeros((B, H, max_length, Dv_scales), dtype=dt)
        values_zeros = mx.zeros((B, H, max_length, Dv_scales), dtype=dt)

        # Copy each cache with left padding (stays Q4!)
        for i, (p, c) in enumerate(zip(left_padding, caches)):
            if c.keys is None:
                continue

            k_q, k_s, k_z = c.keys
            v_q, v_s, v_z = c.values

            # Validate source tensor bounds before slicing
            source_len = k_q.shape[-2] if len(k_q.shape) >= 3 else k_q.shape[-1]
            if c.offset > source_len:
                logger.warning(
                    f"[Q4 MERGE] Cache {i} offset ({c.offset}) > tensor size ({source_len}), "
                    f"clamping to tensor size"
                )
                actual_offset = source_len
            else:
                actual_offset = c.offset

            # Validate destination bounds
            dest_end = p + actual_offset
            if dest_end > max_length:
                raise ValueError(
                    f"Merge bounds error: cache {i} would write to position {dest_end} "
                    f"but max_length is {max_length}"
                )

            keys_quant[i:i+1, :, p:dest_end, :] = k_q[..., :actual_offset, :]
            values_quant[i:i+1, :, p:dest_end, :] = v_q[..., :actual_offset, :]

            keys_scales[i:i+1, :, p:dest_end, :] = k_s[..., :actual_offset, :]
            values_scales[i:i+1, :, p:dest_end, :] = v_s[..., :actual_offset, :]

            keys_zeros[i:i+1, :, p:dest_end, :] = k_z[..., :actual_offset, :]
            values_zeros[i:i+1, :, p:dest_end, :] = v_z[..., :actual_offset, :]

        # CRITICAL: Force evaluation to prevent lazy graph accumulation
        # Without this, MLX builds a deferred computation graph that
        # holds ALL intermediate tensors in memory → OOM
        mx.eval(keys_quant, keys_scales, keys_zeros,
                values_quant, values_scales, values_zeros)

        batch_cache = cls(group_size=group_size, bits=bits)
        batch_cache.keys = (keys_quant, keys_scales, keys_zeros)
        batch_cache.values = (values_quant, values_scales, values_zeros)
        batch_cache.offset = max_length
        batch_cache._idx = max_length
        batch_cache._left_padding = left_padding
        batch_cache._lengths = lengths

        logger.info(
            f"[Q4 MERGE] Merged {B} caches: max_len={max_length}, "
            f"group_size={group_size}, bits={bits} (NO DEQUANTIZATION!)"
        )

        return batch_cache

    def prepare(
        self,
        *,
        left_padding: list[int] | None = None,
        lengths: list[int] | None = None,
        right_padding: list[int] | None = None,
    ) -> None:
        """Prepare cache for batch generation.

        Called by BatchGenerator._process_prompts() before processing
        continuation tokens. The ``lengths`` parameter describes how many
        NEW tokens remain to process (prompt_len - 1), NOT the total cache
        size. We must NOT overwrite self.offset, self._idx, or
        self._lengths which were set correctly by merge().

        Args:
            left_padding: Left padding for each sequence (optional)
            lengths: New token counts to process (NOT total cache size)
            right_padding: Right padding for each batch element (optional)
        """
        if left_padding is not None:
            self._left_padding = left_padding
        if right_padding is not None:
            self._right_padding = right_padding

    def update_and_fetch(self, keys: Any, values: Any) -> tuple[tuple[Any, Any, Any], tuple[Any, Any, Any]]:
        """Update cache with new keys/values and return full cache for attention.

        CRITICAL: This must quantize new K/V tokens and append them to the cache.
        Without this, new tokens during generation are ignored!

        Args:
            keys: New keys tensor (for current token, FP16) shape [B, n_kv_heads, num_steps, head_dim]
            values: New values tensor (for current token, FP16)

        Returns:
            Tuple of (keys_tuple, values_tuple) in Q4 format for quantized attention
        """
        import time
        start_time = time.time()

        B, n_kv_heads, num_steps, k_head_dim = keys.shape
        v_head_dim = values.shape[-1]
        prev = self.offset

        expanded = False
        # Check if we need to expand cache capacity
        # Add +1 to account for potential rounding issues
        required_capacity = prev + num_steps + 1
        if self.keys is None or required_capacity > self.keys[0].shape[-2]:
            expanded = True
            el_per_int = 8 * mx.uint32.size // self.bits
            # Conservative headroom to avoid massive memory spikes
            # Old value (16K tokens) caused 162 tensor allocations across 27 layers
            # which triggered OOM during single-sequence cache hits
            # Cap at 1024 tokens max to prevent OOM even with large prefill chunks
            headroom = min(max(num_steps + 256, 512), 1024)  # Capped at 1K tokens
            new_capacity = ((prev + headroom + self.step - 1) // self.step) * self.step
            new_steps = new_capacity - prev if self.keys is not None else new_capacity
            shape = (B, n_kv_heads, new_steps)
            logger.info(f"[Q4 EXPAND] Allocating {new_steps} new slots (headroom={headroom})")

            def init_quant(dim):
                return (
                    mx.zeros((*shape, dim // el_per_int), dtype=mx.uint32),
                    mx.zeros((*shape, dim // self.group_size), dtype=keys.dtype),
                    mx.zeros((*shape, dim // self.group_size), dtype=keys.dtype),
                )

            def expand_quant(x):
                new_x = mx.zeros((*shape, x.shape[-1]), dtype=x.dtype)
                return mx.concatenate([x, new_x], axis=-2)

            if self.keys is not None:
                # Trim and expand existing cache
                if prev % self.step != 0:
                    from mlx.utils import tree_map
                    self.keys, self.values = tree_map(
                        lambda x: x[..., :prev, :], (self.keys, self.values)
                    )
                from mlx.utils import tree_map
                self.keys, self.values = tree_map(
                    expand_quant, (self.keys, self.values)
                )
                # CRITICAL: Force evaluation after expansion to prevent lazy graph accumulation
                mx.eval(self.keys[0], self.keys[1], self.keys[2],
                       self.values[0], self.values[1], self.values[2])
            else:
                self.keys, self.values = init_quant(k_head_dim), init_quant(v_head_dim)

        # Quantize new keys/values and append to cache
        q_keys = mx.quantize(keys, group_size=self.group_size, bits=self.bits)
        q_values = mx.quantize(values, group_size=self.group_size, bits=self.bits)

        # Use num_steps directly - quantization preserves token dimension
        # Calculate available space in cache and clamp to avoid overflow
        cache_seq_len = self.keys[0].shape[-2]
        available_space = cache_seq_len - prev
        n_tokens = min(num_steps, available_space)

        # Update offset
        self.offset = prev + n_tokens

        # Keep per-element lengths in sync with offset so extract() works
        if self._lengths is not None:
            self._lengths = [l + n_tokens for l in self._lengths]

        # Update cache with quantized data - slice both to exact size
        for i in range(len(self.keys)):
            self.keys[i][..., prev:prev + n_tokens, :] = q_keys[i][..., :n_tokens, :]
            self.values[i][..., prev:prev + n_tokens, :] = q_values[i][..., :n_tokens, :]

        # Return trimmed cache up to current offset
        from mlx.utils import tree_map
        result = tree_map(lambda x: x[..., :self.offset, :], (self.keys, self.values))

        elapsed = (time.time() - start_time) * 1000
        if elapsed > 100 or expanded:  # Log if slow or if we expanded
            logger.info(
                f"[Q4 UPDATE] offset: {prev}->{self.offset}, expanded={expanded}, "
                f"time={elapsed:.0f}ms"
            )

        return result

    def finalize(self) -> None:
        """Finalize cache after generation completes.

        Called by BatchGenerator for cleanup after generation is done.
        """
        # No cleanup needed for quantized cache
        pass

    def filter(self, batch_indices: Any) -> None:
        """Filter cache to keep only specified batch indices (in-place).

        Called when sequences complete to remove their cache data.
        CRITICAL: Must properly release memory for filtered-out sequences.

        Args:
            batch_indices: Indices of batch elements to keep (mx.array or list)
        """
        if self.keys is None:
            return

        k_quant, k_scales, k_zeros = self.keys
        v_quant, v_scales, v_zeros = self.values

        # Create new tensors with only kept indices
        self.keys = (
            k_quant[batch_indices],
            k_scales[batch_indices],
            k_zeros[batch_indices],
        )
        self.values = (
            v_quant[batch_indices],
            v_scales[batch_indices],
            v_zeros[batch_indices],
        )

        # Force evaluation to materialize new tensors and allow old ones to be freed
        mx.eval(self.keys[0], self.keys[1], self.keys[2],
                self.values[0], self.values[1], self.values[2])

        # Convert mx.array to list for Python list indexing
        if hasattr(batch_indices, 'tolist'):
            indices_list = batch_indices.tolist()
        else:
            indices_list = list(batch_indices)

        self._left_padding = [self._left_padding[i] for i in indices_list]
        self._lengths = [self._lengths[i] for i in indices_list]

    def extend(self, other: "BatchQuantizedKVCache") -> None:
        """Extend this cache with another cache (in-place merge).

        Args:
            other: Another BatchQuantizedKVCache to merge into this one
        """
        if other.keys is None:
            return

        if self.keys is None:
            self.keys = other.keys
            self.values = other.values
            self._left_padding = other._left_padding
            self._lengths = other._lengths
            self.offset = other.offset
            return

        # Concatenate along batch dimension
        sk_q, sk_s, sk_z = self.keys
        ok_q, ok_s, ok_z = other.keys
        sv_q, sv_s, sv_z = self.values
        ov_q, ov_s, ov_z = other.values

        self.keys = (
            mx.concatenate([sk_q, ok_q], axis=0),
            mx.concatenate([sk_s, ok_s], axis=0),
            mx.concatenate([sk_z, ok_z], axis=0),
        )
        self.values = (
            mx.concatenate([sv_q, ov_q], axis=0),
            mx.concatenate([sv_s, ov_s], axis=0),
            mx.concatenate([sv_z, ov_z], axis=0),
        )

        self._left_padding.extend(other._left_padding)
        self._lengths.extend(other._lengths)

    def extract(self, idx: int) -> Any:
        """Extract a single batch entry as a new QuantizedKVCache.

        Args:
            idx: Index of batch element to extract

        Returns:
            QuantizedKVCache for the single sequence
        """
        from mlx_lm.models.cache import QuantizedKVCache

        cache = QuantizedKVCache(group_size=self.group_size, bits=self.bits)

        if self.keys is not None:
            k_quant, k_scales, k_zeros = self.keys
            v_quant, v_scales, v_zeros = self.values

            # Extract single batch element
            cache.keys = (
                k_quant[idx:idx+1],
                k_scales[idx:idx+1],
                k_zeros[idx:idx+1],
            )
            cache.values = (
                v_quant[idx:idx+1],
                v_scales[idx:idx+1],
                v_zeros[idx:idx+1],
            )
            cache.offset = self._lengths[idx] if idx < len(self._lengths) else self.offset

        return cache

    def make_mask(
        self,
        n: int | None = None,
        *,
        return_array: bool = False,
        window_size: int | None = None,
    ) -> Any | None:
        """Generate attention mask for the batched cache.

        Args:
            n: Number of new tokens (optional)
            return_array: If True, always return an array (not None)
            window_size: Sliding window size for attention (optional)

        Returns:
            Attention mask tensor or None if no masking needed
        """
        if not self._left_padding or all(p == 0 for p in self._left_padding):
            if return_array:
                # Return a mask of all True (attend to everything)
                B = len(self._left_padding) if self._left_padding else 1
                seq_len = self.offset
                return mx.ones((B, 1, seq_len), dtype=mx.bool_)
            return None

        # Create mask where padded positions are masked out
        B = len(self._left_padding)
        seq_len = self.offset

        # Create causal mask with left padding masked out
        # Shape: (B, 1, seq_len) for broadcasting with attention
        mask = mx.ones((B, 1, seq_len), dtype=mx.bool_)

        # Mask out left padding for each sequence
        for i, pad in enumerate(self._left_padding):
            if pad > 0:
                mask[i, 0, :pad] = False

        # Apply window size if specified (sliding window attention)
        if window_size is not None and n is not None:
            # For sliding window, only attend to last window_size positions
            start_pos = max(0, seq_len - window_size)
            window_mask = mx.zeros((B, 1, seq_len), dtype=mx.bool_)
            window_mask[:, :, start_pos:] = True
            mask = mx.logical_and(mask, window_mask)

        return mask

    def empty(self) -> bool:
        """Check if cache is empty/uninitialized.

        Returns:
            True if cache has no data, False otherwise
        """
        return self.keys is None

    def size(self) -> int:
        """Return current sequence length in the cache.

        CRITICAL: BatchGenerator uses this to determine cache hit vs miss.
        If this returns 0, BatchGenerator treats it as empty and passes
        the FULL prompt through, causing shape mismatch errors.

        Returns:
            Number of tokens currently in cache (self.offset)
        """
        return self.offset

    @property
    def state(self) -> tuple[Any, Any]:
        """Get cache state as (keys, values) tuple."""
        return (self.keys, self.values)

    @state.setter
    def state(self, new_state: tuple[Any, Any]) -> None:
        """Set cache state from (keys, values) tuple."""
        self.keys, self.values = new_state


def add_quantized_merge_method():
    """Add merge() to QuantizedKVCache via monkey-patching.

    This patches the mlx_lm QuantizedKVCache class to add a merge() method,
    which enables us to use Q4 caches with BatchGenerator.
    """
    try:
        from mlx_lm.models.cache import QuantizedKVCache

        if hasattr(QuantizedKVCache, 'merge'):
            logger.debug("[Q4 EXT] QuantizedKVCache.merge() already exists")
            return

        @classmethod
        def merge(cls, caches):
            return BatchQuantizedKVCache.merge(caches)

        QuantizedKVCache.merge = merge

        logger.info("[Q4 EXT] Added QuantizedKVCache.merge() successfully")

        # CRITICAL FIX: Add size() method to QuantizedKVCache
        # MLX's QuantizedKVCache inherits from _BaseCache which returns 0 for size()
        # This breaks BatchGenerator's cache continuation logic - it can't tell
        # the difference between a 19K token cache hit and a cold start!
        def size(self) -> int:
            """Return actual sequence length (offset), not buffer capacity."""
            return self.offset

        # Only patch if size() is missing or returns 0 for a non-empty cache
        test_cache = QuantizedKVCache()
        test_cache.offset = 100
        if not hasattr(QuantizedKVCache, 'size') or test_cache.size() == 0:
            QuantizedKVCache.size = size
            logger.info("[Q4 EXT] Added QuantizedKVCache.size() method (returns offset)")
        else:
            logger.debug("[Q4 EXT] QuantizedKVCache.size() already works correctly")

    except ImportError as e:
        logger.warning(f"[Q4 EXT] Could not import QuantizedKVCache: {e}")
    except Exception as e:
        logger.error(f"[Q4 EXT] Failed to patch QuantizedKVCache: {e}")


def patch_batch_kv_cache_merge():
    """Patch BatchKVCache.merge() to handle QuantizedKVCache objects.

    The BatchGenerator calls BatchKVCache.merge(caches) which doesn't know
    how to handle QuantizedKVCache. This patch intercepts that call and
    delegates to BatchQuantizedKVCache.merge() when needed.
    """
    try:
        from mlx_lm.models.cache import BatchKVCache, QuantizedKVCache

        # Store original merge method
        original_merge = BatchKVCache.merge

        @classmethod
        def patched_merge(cls, caches):
            """Merge caches, delegating to Q4 merge if caches are quantized."""
            if not caches:
                return original_merge.__func__(cls, caches)

            # Check if first cache is QuantizedKVCache
            first_cache = caches[0]
            if isinstance(first_cache, QuantizedKVCache):
                logger.debug(f"[BatchKVCache.merge] Delegating to Q4 merge for {len(caches)} caches")
                return BatchQuantizedKVCache.merge(caches)

            # Check if first cache's keys are tuples (quantized format)
            if hasattr(first_cache, 'keys') and first_cache.keys is not None:
                if isinstance(first_cache.keys, tuple) and len(first_cache.keys) == 3:
                    logger.debug(f"[BatchKVCache.merge] Detected Q4 tuples, delegating to Q4 merge")
                    return BatchQuantizedKVCache.merge(caches)

            # Otherwise use original merge
            return original_merge.__func__(cls, caches)

        BatchKVCache.merge = patched_merge

        logger.info("[Q4 EXT] Patched BatchKVCache.merge() to handle QuantizedKVCache")

    except ImportError as e:
        logger.warning(f"[Q4 EXT] Could not import BatchKVCache: {e}")
    except Exception as e:
        logger.error(f"[Q4 EXT] Failed to patch BatchKVCache.merge(): {e}")


# Auto-patch on module import
add_quantized_merge_method()
patch_batch_kv_cache_merge()
