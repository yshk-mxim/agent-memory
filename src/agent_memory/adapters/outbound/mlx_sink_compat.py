# mypy: disable-error-code="attr-defined,arg-type"
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Yakov Shkolnikov and contributors
"""Compatibility patch: quantized KV cache with attention sinks.

GPT-OSS and similar models use attention sinks, but MLX's quantized SDPA
kernel doesn't support sinks (raises ValueError). This module patches
mlx_lm.models.base.scaled_dot_product_attention to fall back: dequantize
KV → FP16, then call the standard SDPA path which supports sinks.

Trade-off: Q4 storage savings preserved, transient FP16 during attention
compute only. Dequantize is one fast memcpy-like op per forward pass.

Applied on import. Import before first model inference.
"""

import logging

logger = logging.getLogger(__name__)

_patched = False


def _apply_patch() -> None:
    global _patched
    if _patched:
        return

    try:
        import mlx.core as mx
        import mlx_lm.models.base as base_module
    except ImportError:
        logger.debug("mlx_lm not available, skipping sink compat patch")
        return

    def _patched_sdpa(
        queries: mx.array,
        keys: mx.array | tuple,
        values: mx.array | tuple,
        cache: object,
        scale: float,
        mask: mx.array | None,
        sinks: mx.array | None = None,
    ) -> mx.array:
        if hasattr(cache, "bits"):
            if sinks is not None:
                # Dequantize Q4 tuples → FP16 arrays for sink-aware attention
                k_fp = mx.dequantize(
                    keys[0],
                    scales=keys[1],
                    biases=keys[2],
                    group_size=cache.group_size,
                    bits=cache.bits,
                )
                v_fp = mx.dequantize(
                    values[0],
                    scales=values[1],
                    biases=values[2],
                    group_size=cache.group_size,
                    bits=cache.bits,
                )
                return mx.fast.scaled_dot_product_attention(
                    queries,
                    k_fp,
                    v_fp,
                    scale=scale,
                    mask=mask,
                    sinks=sinks,
                )
            # Dynamic lookup allows later patches (e.g. fused Q4 attention)
            # to intercept the quantized path
            qsdpa = base_module.quantized_scaled_dot_product_attention
            return qsdpa(
                queries,
                keys,
                values,
                scale=scale,
                mask=mask,
                group_size=cache.group_size,
                bits=cache.bits,
            )
        return mx.fast.scaled_dot_product_attention(
            queries,
            keys,
            values,
            scale=scale,
            mask=mask,
            sinks=sinks,
        )

    base_module.scaled_dot_product_attention = _patched_sdpa
    _patched = True
    logger.info("Patched scaled_dot_product_attention for Q4 + attention sinks")


_apply_patch()
