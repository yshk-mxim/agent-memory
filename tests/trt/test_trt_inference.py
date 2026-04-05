# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Yakov Shkolnikov and contributors
"""TRT inference integration tests.

Tests the full subprocess lifecycle: start -> generate -> extract spec -> stop.
Uses fake_llm_inference.py on Mac/CI, real binary on Thor.
"""

import pytest

from agent_memory.domain.value_objects import ModelCacheSpec
from agent_memory.ports.outbound import ModelBackendPort

from .conftest import SMOLLM2_HEAD_DIM, SMOLLM2_N_KV_HEADS, SMOLLM2_N_LAYERS

pytestmark = pytest.mark.integration


class TestSubprocessLifecycle:
    """Test full subprocess start -> use -> stop lifecycle via ModelBackendPort."""

    def test_start_and_extract_spec(self, fake_trt_subprocess: ModelBackendPort) -> None:
        """Backend should report model geometry via port interface."""
        spec = fake_trt_subprocess.extract_model_spec()

        assert isinstance(spec, ModelCacheSpec)
        assert spec.n_layers == SMOLLM2_N_LAYERS
        assert spec.n_kv_heads == SMOLLM2_N_KV_HEADS
        assert spec.head_dim == SMOLLM2_HEAD_DIM
        assert spec.kv_format == "fp"
        assert spec.kv_bits is None

    def test_generate_returns_text(self, fake_trt_subprocess: ModelBackendPort) -> None:
        """Generate should return text and tokens via port interface."""
        result = fake_trt_subprocess.generate(
            prompt_tokens=[1, 2, 3, 4, 5],
            max_tokens=10,
            temperature=0.7,
        )

        assert result.text  # Non-empty
        assert len(result.tokens) == 10
        assert result.cache  # Should have cache layers

    def test_generate_with_cache_injection(
        self,
        fake_trt_subprocess: ModelBackendPort,
        fake_fp16_cache: list,
    ) -> None:
        """Generate with injected KV cache should work."""
        result = fake_trt_subprocess.generate(
            prompt_tokens=[10, 20, 30],
            cache=fake_fp16_cache,
            max_tokens=5,
        )

        assert result.text
        assert result.tokens

    def test_multiple_generations(self, fake_trt_subprocess: ModelBackendPort) -> None:
        """Multiple sequential generations should work (persistent process)."""
        for i in range(3):
            result = fake_trt_subprocess.generate(
                prompt_tokens=list(range(i * 10, i * 10 + 5)),
                max_tokens=5,
            )
            assert result.text


class TestSpecExtractor:
    """Test TRT spec extraction via port interface."""

    def test_extract_via_extractor(self, fake_trt_subprocess: ModelBackendPort) -> None:
        from agent_memory.adapters.outbound.trt_spec_extractor import TRTSpecExtractor

        extractor = TRTSpecExtractor(fake_trt_subprocess)
        spec = extractor.extract_spec()

        assert spec.kv_format == "fp"
        assert spec.n_layers == SMOLLM2_N_LAYERS
        assert spec.layer_types == ["global"] * SMOLLM2_N_LAYERS
