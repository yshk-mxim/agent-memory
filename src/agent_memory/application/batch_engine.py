# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Yakov Shkolnikov and contributors
"""Backward-compatible re-export of BlockPoolBatchEngine.

The canonical location is now adapters/outbound/mlx_batch_engine.py
(hexagonal architecture: MLX-specific code belongs in the adapter layer).
"""

from agent_memory.adapters.outbound.mlx_batch_engine import BlockPoolBatchEngine

__all__ = ["BlockPoolBatchEngine"]
