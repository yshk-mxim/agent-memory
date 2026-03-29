# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Yakov Shkolnikov and contributors
"""TRT model spec extractor.

Queries the ``llm_inference`` subprocess for model geometry and returns
a ``ModelCacheSpec`` with ``kv_format='fp'``.
"""

from agent_memory.domain.value_objects import ModelCacheSpec
from agent_memory.ports.outbound import ModelBackendPort


class TRTSpecExtractor:
    """Extract ModelCacheSpec from a running TRT backend."""

    def __init__(self, backend: ModelBackendPort) -> None:
        """Initialize with a backend that can report its model spec."""
        self._backend = backend

    def extract(self) -> ModelCacheSpec:
        """Query subprocess for model spec.

        Returns:
            ModelCacheSpec with kv_format='fp' and kv_bits=None (FP16 on GPU).
        """
        return self._backend.extract_model_spec()
