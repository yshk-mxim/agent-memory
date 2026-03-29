# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Yakov Shkolnikov and contributors
"""Unit tests for TRT spec extractor."""

from unittest.mock import Mock

import pytest

from agent_memory.adapters.outbound.trt_spec_extractor import TRTSpecExtractor
from agent_memory.domain.value_objects import ModelCacheSpec

pytestmark = pytest.mark.unit


class TestTRTSpecExtractor:
    def test_extract_delegates_to_backend_port(self) -> None:
        """Extractor should delegate to ModelBackendPort."""
        expected_spec = ModelCacheSpec(
            n_layers=64,
            n_kv_heads=8,
            head_dim=128,
            block_tokens=256,
            layer_types=["global"] * 64,
            kv_format="fp",
            kv_bits=None,
        )

        mock_backend = Mock()
        mock_backend.extract_model_spec.return_value = expected_spec

        extractor = TRTSpecExtractor(mock_backend)
        spec = extractor.extract()

        assert spec is expected_spec
        assert spec.kv_format == "fp"
        assert spec.n_layers == 64
        mock_backend.extract_model_spec.assert_called_once()
