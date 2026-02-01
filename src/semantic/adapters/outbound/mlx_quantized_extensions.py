"""MLX QuantizedKVCache extensions for batching support.

Adds merge() method to QuantizedKVCache to enable direct Q4 cache injection.
This allows us to keep Q4 format end-to-end (storage -> injection -> generation)
without dequantizing to FP16, maintaining 75% memory savings.

Patches applied on import:
  1. QuantizedKVCache.merge()  -> delegates to BatchQuantizedKVCache.merge()
  2. QuantizedKVCache.size()   -> returns offset (not 0)
  3. BatchKVCache.merge()      -> delegates to Q4 merge for QuantizedKVCache inputs
  4. _make_cache()             -> creates BatchQuantizedKVCache for cold starts
"""

import logging
from typing import Any

import mlx.core as mx
from mlx.utils import tree_map

try:
    from mlx_lm.models.cache import _BaseCache
except ImportError:
    _BaseCache = object

logger = logging.getLogger(__name__)


class BatchQuantizedKVCache(_BaseCache):
    """Batched quantized KV cache for multiple sequences.

    Drop-in replacement for BatchKVCache that keeps data in Q4 format.
    Used by BatchGenerator for both cold starts (via patched _make_cache)
    and warm starts (via patched merge).

    Matches upstream BatchKVCache convention:
      - offset: per-batch mx.array tracking logical token count per sequence.
                Starts negative for left-padded seqs. Used by model for RoPE.
      - _idx:   scalar int tracking shared buffer write position.
                Used for mask generation and buffer management.
      - left_padding: mx.array tracking left-padding amount per sequence.
                Used by make_mask for attention masking.
    """

    step = 256

    def __init__(
        self,
        left_padding: list[int] | None = None,
        group_size: int = 64,
        bits: int = 4,
    ):
        self.keys: tuple[Any, Any, Any] | None = None
        self.values: tuple[Any, Any, Any] | None = None
        self.group_size = group_size
        self.bits = bits
        self._right_padding: mx.array | None = None

        if left_padding:
            self.left_padding = mx.array(left_padding)
            self.offset = mx.array([-l for l in left_padding])
        else:
            self.left_padding = mx.array([0])
            self.offset = mx.array([0])
        self._idx = 0

    # ------------------------------------------------------------------
    # merge: combine per-sequence QuantizedKVCache -> batched Q4
    # ------------------------------------------------------------------
    @classmethod
    def merge(cls, caches: list[Any]) -> "BatchQuantizedKVCache":
        if not caches:
            raise ValueError("Cannot merge empty cache list")

        # Find Q4 params from any quantized cache in the list.
        # In mixed warm+cold batches, some entries may be plain KVCache
        # (no bits/group_size) while others are QuantizedKVCache.
        q4_source = next(
            (c for c in caches if hasattr(c, "bits") and hasattr(c, "group_size")),
            None,
        )
        if q4_source is None:
            raise ValueError("No QuantizedKVCache found in merge list")
        group_size = q4_source.group_size
        bits = q4_source.bits

        lengths = [c.offset for c in caches]
        max_length = max(lengths)
        lp = [max_length - length for length in lengths]
        B = len(caches)

        if max_length == 0:
            batch_cache = cls(group_size=group_size, bits=bits)
            batch_cache.left_padding = mx.array([0] * B)
            batch_cache.offset = mx.array([0] * B)
            logger.info(f"[Q4 MERGE] Cold start batch of {B} empty caches")
            return batch_cache

        sample_cache = next(c for c in caches if c.keys is not None)
        k_quant, k_scales, k_zeros = sample_cache.keys
        v_quant, v_scales, v_zeros = sample_cache.values

        H = k_quant.shape[1]
        Dk_packed = k_quant.shape[-1]
        Dv_packed = v_quant.shape[-1]
        Dk_scales = k_scales.shape[-1]
        Dv_scales = v_scales.shape[-1]
        dt = k_scales.dtype

        keys_quant = mx.zeros((B, H, max_length, Dk_packed), dtype=mx.uint32)
        keys_scales = mx.zeros((B, H, max_length, Dk_scales), dtype=dt)
        keys_zeros = mx.zeros((B, H, max_length, Dk_scales), dtype=dt)
        values_quant = mx.zeros((B, H, max_length, Dv_packed), dtype=mx.uint32)
        values_scales = mx.zeros((B, H, max_length, Dv_scales), dtype=dt)
        values_zeros = mx.zeros((B, H, max_length, Dv_scales), dtype=dt)

        for i, (p, c) in enumerate(zip(lp, caches)):
            if c.keys is None:
                continue
            k_q, k_s, k_z = c.keys
            v_q, v_s, v_z = c.values

            source_len = k_q.shape[-2] if len(k_q.shape) >= 3 else k_q.shape[-1]
            actual_offset = min(c.offset, source_len)
            dest_end = p + actual_offset
            if dest_end > max_length:
                raise ValueError(
                    f"Merge bounds error: cache {i} writes to {dest_end} "
                    f"but max_length={max_length}"
                )

            keys_quant[i : i + 1, :, p:dest_end, :] = k_q[..., :actual_offset, :]
            keys_scales[i : i + 1, :, p:dest_end, :] = k_s[..., :actual_offset, :]
            keys_zeros[i : i + 1, :, p:dest_end, :] = k_z[..., :actual_offset, :]
            values_quant[i : i + 1, :, p:dest_end, :] = v_q[..., :actual_offset, :]
            values_scales[i : i + 1, :, p:dest_end, :] = v_s[..., :actual_offset, :]
            values_zeros[i : i + 1, :, p:dest_end, :] = v_z[..., :actual_offset, :]

        mx.eval(keys_quant, keys_scales, keys_zeros, values_quant, values_scales, values_zeros)

        batch_cache = cls(group_size=group_size, bits=bits)
        batch_cache.keys = (keys_quant, keys_scales, keys_zeros)
        batch_cache.values = (values_quant, values_scales, values_zeros)
        batch_cache.offset = mx.array(lengths)
        batch_cache.left_padding = mx.array(lp)
        batch_cache._idx = max_length

        logger.info(
            f"[Q4 MERGE] Merged {B} caches: lengths={lengths}, "
            f"padding={lp}, gs={group_size}, bits={bits}"
        )
        return batch_cache

    # ------------------------------------------------------------------
    # prepare / finalize — match upstream BatchKVCache interface
    # ------------------------------------------------------------------
    def prepare(
        self,
        *,
        left_padding: list[int] | None = None,
        lengths: list[int] | None = None,
        right_padding: list[int] | None = None,
    ) -> None:
        if left_padding is not None:
            if self.keys is not None:
                raise ValueError("Left padding can only be added to an empty BatchQuantizedKVCache")
            lp = mx.array(left_padding)
            self.left_padding = self.left_padding + lp
            self.offset = self.offset - lp

        if right_padding is not None and max(right_padding) > 0:
            self._right_padding = mx.array(right_padding)

    def finalize(self) -> None:
        if self._right_padding is not None:
            padding = self._right_padding
            if self.keys is not None:
                self.keys = _q4_dynamic_roll(self.keys, padding[:, None])
                self.values = _q4_dynamic_roll(self.values, padding[:, None])
            self.offset = self.offset - padding
            self.left_padding = self.left_padding + padding
            self._right_padding = None

    # ------------------------------------------------------------------
    # update_and_fetch: quantize new tokens and append to buffer
    # ------------------------------------------------------------------
    def update_and_fetch(
        self,
        keys: Any,
        values: Any,
    ) -> tuple[tuple[Any, Any, Any], tuple[Any, Any, Any]]:
        B, n_kv_heads, num_steps, k_head_dim = keys.shape
        v_head_dim = values.shape[-1]
        prev = self._idx

        expanded = False
        required_capacity = prev + num_steps + 1
        if self.keys is None or required_capacity > self.keys[0].shape[-2]:
            expanded = True
            el_per_int = 8 * mx.uint32.size // self.bits
            extra = min(256, 1024)
            new_capacity = (required_capacity + extra + self.step - 1) // self.step * self.step
            new_steps = new_capacity - prev if self.keys is not None else new_capacity
            shape = (B, n_kv_heads, new_steps)

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
                if prev % self.step != 0:
                    self.keys, self.values = tree_map(
                        lambda x: x[..., :prev, :], (self.keys, self.values)
                    )
                self.keys, self.values = tree_map(expand_quant, (self.keys, self.values))
                mx.eval(*self.keys, *self.values)
            else:
                self.keys = init_quant(k_head_dim)
                self.values = init_quant(v_head_dim)

        q_keys = mx.quantize(keys, group_size=self.group_size, bits=self.bits)
        q_values = mx.quantize(values, group_size=self.group_size, bits=self.bits)

        cache_seq_len = self.keys[0].shape[-2]
        available_space = cache_seq_len - prev
        n_tokens = min(num_steps, available_space)

        self._idx += n_tokens
        self.offset = self.offset + n_tokens

        for i in range(len(self.keys)):
            self.keys[i][..., prev : prev + n_tokens, :] = q_keys[i][..., :n_tokens, :]
            self.values[i][..., prev : prev + n_tokens, :] = q_values[i][..., :n_tokens, :]

        result = tree_map(lambda x: x[..., : self._idx, :], (self.keys, self.values))

        if expanded:
            logger.info(f"[Q4 UPDATE] _idx: {prev}->{self._idx}, expanded={expanded}")
        return result

    # ------------------------------------------------------------------
    # filter: keep only specified batch indices
    # ------------------------------------------------------------------
    def filter(self, batch_indices: Any) -> None:
        if self.keys is None:
            return

        self.keys = tuple(k[batch_indices] for k in self.keys)
        self.values = tuple(v[batch_indices] for v in self.values)
        mx.eval(*self.keys, *self.values)

        self.offset = self.offset[batch_indices]
        self.left_padding = self.left_padding[batch_indices]

        min_pad = self.left_padding.min().item()
        if min_pad > 0:
            self.keys = tuple(k[..., min_pad:, :] for k in self.keys)
            self.values = tuple(v[..., min_pad:, :] for v in self.values)
            self._idx -= min_pad
            self.left_padding = self.left_padding - min_pad

    # ------------------------------------------------------------------
    # extend: merge another batch cache into this one (staggered arrival)
    # ------------------------------------------------------------------
    def extend(self, other: "BatchQuantizedKVCache") -> None:
        if other.keys is None:
            self.offset = mx.concatenate([self.offset, other.offset])
            self.left_padding = mx.concatenate([self.left_padding, other.left_padding])
            return

        if self.keys is None:
            self.keys = other.keys
            self.values = other.values
            self.offset = mx.concatenate([self.offset, other.offset])
            self.left_padding = mx.concatenate([self.left_padding, other.left_padding])
            self._idx = other._idx
            return

        max_idx = max(self._idx, other._idx)

        def _pad_cache(cache: "BatchQuantizedKVCache"):
            """Trim to _idx and left-pad to max_idx."""
            k_q, k_s, k_z = cache.keys
            v_q, v_s, v_z = cache.values

            idx = cache._idx
            k_q = k_q[..., :idx, :]
            k_s = k_s[..., :idx, :]
            k_z = k_z[..., :idx, :]
            v_q = v_q[..., :idx, :]
            v_s = v_s[..., :idx, :]
            v_z = v_z[..., :idx, :]

            left = max_idx - idx
            if left > 0:
                B_loc, H_loc = k_q.shape[0], k_q.shape[1]
                pad_kq = mx.zeros((B_loc, H_loc, left, k_q.shape[-1]), dtype=k_q.dtype)
                pad_ks = mx.zeros((B_loc, H_loc, left, k_s.shape[-1]), dtype=k_s.dtype)
                pad_kz = mx.zeros((B_loc, H_loc, left, k_z.shape[-1]), dtype=k_z.dtype)
                pad_vq = mx.zeros((B_loc, H_loc, left, v_q.shape[-1]), dtype=v_q.dtype)
                pad_vs = mx.zeros((B_loc, H_loc, left, v_s.shape[-1]), dtype=v_s.dtype)
                pad_vz = mx.zeros((B_loc, H_loc, left, v_z.shape[-1]), dtype=v_z.dtype)

                k_q = mx.concatenate([pad_kq, k_q], axis=-2)
                k_s = mx.concatenate([pad_ks, k_s], axis=-2)
                k_z = mx.concatenate([pad_kz, k_z], axis=-2)
                v_q = mx.concatenate([pad_vq, v_q], axis=-2)
                v_s = mx.concatenate([pad_vs, v_s], axis=-2)
                v_z = mx.concatenate([pad_vz, v_z], axis=-2)

            new_lp = cache.left_padding + left
            return (k_q, k_s, k_z), (v_q, v_s, v_z), cache.offset, new_lp

        self_k, self_v, self_off, self_lp = _pad_cache(self)
        other_k, other_v, other_off, other_lp = _pad_cache(other)

        self.keys = tuple(mx.concatenate([sk, ok], axis=0) for sk, ok in zip(self_k, other_k))
        self.values = tuple(mx.concatenate([sv, ov], axis=0) for sv, ov in zip(self_v, other_v))
        mx.eval(*self.keys, *self.values)

        self.offset = mx.concatenate([self_off, other_off])
        self.left_padding = mx.concatenate([self_lp, other_lp])
        self._idx = max_idx

        logger.info(
            f"[Q4 EXTEND] Merged batches: "
            f"offset={self.offset.tolist()}, "
            f"padding={self.left_padding.tolist()}, _idx={self._idx}"
        )

    # ------------------------------------------------------------------
    # extract: pull out a single sequence as QuantizedKVCache
    # ------------------------------------------------------------------
    def extract(self, idx: int) -> Any:
        from mlx_lm.models.cache import QuantizedKVCache

        cache = QuantizedKVCache(group_size=self.group_size, bits=self.bits)

        if self.keys is not None:
            k_quant, k_scales, k_zeros = self.keys
            v_quant, v_scales, v_zeros = self.values

            pad = self.left_padding[idx].item()
            end = self._idx

            cache.keys = (
                mx.contiguous(k_quant[idx : idx + 1, :, pad:end, :]),
                mx.contiguous(k_scales[idx : idx + 1, :, pad:end, :]),
                mx.contiguous(k_zeros[idx : idx + 1, :, pad:end, :]),
            )
            cache.values = (
                mx.contiguous(v_quant[idx : idx + 1, :, pad:end, :]),
                mx.contiguous(v_scales[idx : idx + 1, :, pad:end, :]),
                mx.contiguous(v_zeros[idx : idx + 1, :, pad:end, :]),
            )
            # Force eval so extracted data is independent of batch tensors.
            # Without this, lazy contiguous ops can reference batch cache
            # tensors that get freed when active_batch = None in _next().
            mx.eval(*cache.keys, *cache.values)
            cache.offset = end - pad

        return cache

    # ------------------------------------------------------------------
    # make_mask: delegate to MLX's create_causal_mask
    # ------------------------------------------------------------------
    def make_mask(
        self,
        N: int = 1,
        return_array: bool = False,
        **kwargs: Any,
    ) -> Any | None:
        from mlx_lm.models.base import create_causal_mask

        return create_causal_mask(N, offset=self._idx, left_padding=self.left_padding, **kwargs)

    # ------------------------------------------------------------------
    # empty / size / state / trim
    # ------------------------------------------------------------------
    def empty(self) -> bool:
        return self.keys is None

    def size(self) -> int:
        return self._idx

    def is_trimmable(self) -> bool:
        return True

    def trim(self, n: int) -> int:
        n = min(self._idx, n)
        self._idx -= n
        self.offset = self.offset - n
        return n

    @property
    def state(self) -> tuple[Any, ...]:
        k, v = self.keys, self.values
        if self.keys is not None and self._idx < self.keys[0].shape[-2]:
            k, v = tree_map(lambda x: x[..., : self._idx, :], (k, v))
        return k, v, self.offset, self.left_padding

    @state.setter
    def state(self, new_state: tuple[Any, ...]) -> None:
        self.keys, self.values, self.offset, self.left_padding = new_state
        if self.keys is not None:
            self._idx = self.keys[0].shape[-2]
        else:
            self._idx = 0


# ======================================================================
# Helper: Q4 dynamic roll for finalize (right-pad -> left-pad conversion)
# ======================================================================


def _q4_dynamic_roll(q4_tuple: tuple[Any, Any, Any], shifts: Any) -> tuple:
    """Roll each batch element in a Q4 tuple by its shift amount.

    Used by finalize() to convert right-padded data back to left-padded
    format after processing continuation tokens in _process_prompts.
    """
    try:
        from mlx_lm.models.cache import dynamic_roll

        return tuple(dynamic_roll(t, shifts, axis=2) for t in q4_tuple)
    except ImportError:
        rolled = []
        for t in q4_tuple:
            B = t.shape[0]
            parts = []
            for b in range(B):
                s = shifts[b, 0].item()
                if s > 0:
                    row = mx.concatenate(
                        [t[b : b + 1, :, -s:, :], t[b : b + 1, :, :-s, :]],
                        axis=2,
                    )
                else:
                    row = t[b : b + 1]
                parts.append(row)
            rolled.append(mx.concatenate(parts, axis=0))
        return tuple(rolled)


# ======================================================================
# Monkey-patches applied at import time
# ======================================================================


def add_quantized_merge_method():
    """Override merge() and size() on QuantizedKVCache."""
    try:
        from mlx_lm.models.cache import QuantizedKVCache

        # Always override merge to route directly to Q4 batch merge.
        # The upstream QuantizedKVCache.merge delegates to BatchKVCache.merge
        # which doesn't handle Q4 tuples. We replace it unconditionally.
        @classmethod  # type: ignore[misc]
        def merge(cls, caches):  # type: ignore[no-redef]
            return BatchQuantizedKVCache.merge(caches)

        QuantizedKVCache.merge = merge
        logger.info("[Q4 EXT] Overrode QuantizedKVCache.merge()")

        test_cache = QuantizedKVCache()
        test_cache.offset = 100
        if test_cache.size() == 0:

            def size(self) -> int:
                return self.offset

            QuantizedKVCache.size = size
            logger.info("[Q4 EXT] Added QuantizedKVCache.size()")

    except ImportError as e:
        logger.warning(f"[Q4 EXT] Could not import QuantizedKVCache: {e}")
    except Exception as e:
        logger.error(f"[Q4 EXT] Failed to patch QuantizedKVCache: {e}")


def patch_batch_kv_cache_merge():
    """Patch BatchKVCache.merge() to delegate to Q4 merge for quantized caches."""
    try:
        from mlx_lm.models.cache import BatchKVCache, QuantizedKVCache

        original_merge = BatchKVCache.merge

        @classmethod
        def patched_merge(cls, caches):
            if not caches:
                return original_merge.__func__(cls, caches)

            first = caches[0]
            if isinstance(first, QuantizedKVCache):
                return BatchQuantizedKVCache.merge(caches)
            # Fallback: detect Q4 cache by structural check (bits attr or tuple keys)
            if hasattr(first, "bits") and hasattr(first, "group_size"):
                return BatchQuantizedKVCache.merge(caches)
            if hasattr(first, "keys") and isinstance(getattr(first, "keys", None), tuple):
                if len(first.keys) == 3:
                    return BatchQuantizedKVCache.merge(caches)

            return original_merge.__func__(cls, caches)

        BatchKVCache.merge = patched_merge
        logger.info("[Q4 EXT] Patched BatchKVCache.merge()")

    except ImportError as e:
        logger.warning(f"[Q4 EXT] Could not import BatchKVCache: {e}")
    except Exception as e:
        logger.error(f"[Q4 EXT] Failed to patch BatchKVCache.merge(): {e}")


def patch_make_cache_for_q4(group_size: int = 64, bits: int = 4):
    """Patch _make_cache and _merge_caches in mlx_lm.generate for Q4."""
    try:
        import importlib

        gen_module = importlib.import_module("mlx_lm.generate")
        from mlx_lm.models.cache import ArraysCache, CacheList

        original_make_cache = gen_module._make_cache

        def patched_make_cache(model, left_padding):
            if hasattr(model, "make_cache"):
                sample_caches = model.make_cache()
                for c in sample_caches:
                    if isinstance(c, (CacheList, ArraysCache)):
                        return original_make_cache(model, left_padding)
                n_layers = len(sample_caches)
            else:
                n_layers = len(model.layers)

            logger.info(
                f"[Q4 _make_cache] Creating {n_layers} Q4 batch caches "
                f"(gs={group_size}, bits={bits}, padding={left_padding})"
            )
            return [
                BatchQuantizedKVCache(left_padding, group_size=group_size, bits=bits)
                for _ in range(n_layers)
            ]

        gen_module._make_cache = patched_make_cache

        # Patch _merge_caches to route Q4 caches to our merge.
        # The upstream _merge_caches calls QuantizedKVCache.merge which
        # delegates to BatchKVCache.merge — but the isinstance check in our
        # patched BatchKVCache.merge can fail. Patching _merge_caches directly
        # is the most reliable interception point.
        original_merge_caches = gen_module._merge_caches

        def patched_merge_caches(caches):
            if not caches or not caches[0]:
                return original_merge_caches(caches)
            # Check ANY prompt's first-layer cache for Q4 format.
            # In warm+cold batches the cold prompt may have a plain
            # KVCache (no bits attr) while the warm prompt has Q4.
            has_q4 = any(
                hasattr(c[0], "bits") and hasattr(c[0], "group_size")
                for c in caches
                if c and len(c) > 0
            )
            if has_q4:
                batch_cache = []
                for i in range(len(caches[0])):
                    layer_caches = [c[i] for c in caches]
                    batch_cache.append(BatchQuantizedKVCache.merge(layer_caches))
                return batch_cache
            return original_merge_caches(caches)

        gen_module._merge_caches = patched_merge_caches
        logger.info(
            f"[Q4 EXT] Patched _make_cache and _merge_caches for Q4 (gs={group_size}, bits={bits})"
        )

    except ImportError as e:
        logger.warning(f"[Q4 EXT] Could not patch _make_cache: {e}")
    except Exception as e:
        logger.error(f"[Q4 EXT] Failed to patch _make_cache: {e}")


# Apply all patches on import
add_quantized_merge_method()
patch_batch_kv_cache_merge()
patch_make_cache_for_q4()
