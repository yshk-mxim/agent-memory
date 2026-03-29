# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Yakov Shkolnikov and contributors
"""TRT model spec extractor.

Queries the ``llm_inference`` subprocess for model geometry and returns
a ``ModelCacheSpec`` with ``kv_format='fp'``.
"""

from agent_memory.adapters.outbound.trt_subprocess_adapter import TRTSubprocessAdapter
from agent_memory.domain.value_objects import ModelCacheSpec


class TRTSpecExtractor:
    """Extract ModelCacheSpec from a running TRT subprocess."""

    def __init__(self, subprocess_adapter: TRTSubprocessAdapter) -> None:
        """Initialize with a running subprocess adapter."""
        self._subprocess = subprocess_adapter

    def extract(self) -> ModelCacheSpec:
        """Query subprocess for model spec.

        Returns:
            ModelCacheSpec with kv_format='fp' and kv_bits=None (FP16 on GPU).
        """
        return self._subprocess.extract_model_spec()
