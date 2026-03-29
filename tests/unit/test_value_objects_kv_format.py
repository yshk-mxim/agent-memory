# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Yakov Shkolnikov and contributors
"""Unit tests for kv_format field in ModelCacheSpec and TRT error classes."""

import pytest

from agent_memory.domain.errors import (
    ModelSpecValidationError,
    TRTEngineError,
    TRTLayoutError,
    TRTSubprocessError,
)
from agent_memory.domain.value_objects import ModelCacheSpec

pytestmark = pytest.mark.unit


def _make_spec(**overrides: object) -> ModelCacheSpec:
    """Create a ModelCacheSpec with sensible defaults, overriding as needed."""
    defaults: dict[str, object] = {
        "n_layers": 32,
        "n_kv_heads": 8,
        "head_dim": 128,
        "block_tokens": 256,
        "layer_types": ["global"] * 32,
        "kv_bits": None,
        "kv_group_size": 64,
        "kv_format": "quantized",
    }
    defaults.update(overrides)
    return ModelCacheSpec(**defaults)  # type: ignore[arg-type]


class TestKvFormatValidation:
    """Validation of the kv_format field."""

    def test_default_is_quantized(self) -> None:
        """Default kv_format should be 'quantized' for backward compatibility."""
        spec = _make_spec()
        assert spec.kv_format == "quantized"

    def test_accept_quantized(self) -> None:
        spec = _make_spec(kv_format="quantized")
        assert spec.kv_format == "quantized"

    def test_accept_fp(self) -> None:
        spec = _make_spec(kv_format="fp")
        assert spec.kv_format == "fp"

    def test_reject_invalid_format(self) -> None:
        with pytest.raises(ModelSpecValidationError, match="kv_format must be one of"):
            _make_spec(kv_format="gguf")

    def test_reject_empty_string(self) -> None:
        with pytest.raises(ModelSpecValidationError, match="kv_format must be one of"):
            _make_spec(kv_format="")


class TestFpFormatByteCalculation:
    """bytes_per_block_per_layer() for kv_format='fp' (native floating point)."""

    def test_fp16_same_as_quantized_fp16(self) -> None:
        """FP16 byte count should be identical regardless of kv_format."""
        spec_q = _make_spec(kv_format="quantized", kv_bits=None)
        spec_f = _make_spec(kv_format="fp", kv_bits=None)
        assert spec_q.bytes_per_block_per_layer() == spec_f.bytes_per_block_per_layer()

    def test_fp16_explicit_bits_16(self) -> None:
        """kv_bits=16 with fp format should equal kv_bits=None."""
        spec_none = _make_spec(kv_format="fp", kv_bits=None)
        spec_16 = _make_spec(kv_format="fp", kv_bits=16)
        assert spec_none.bytes_per_block_per_layer() == spec_16.bytes_per_block_per_layer()

    def test_fp8_byte_calculation(self) -> None:
        """FP8: 1 byte per element, no scales/biases overhead."""
        spec = _make_spec(kv_format="fp", kv_bits=8)
        # K: 8 * 128 * 256 = 262,144 elements
        # V: 8 * 128 * 256 = 262,144 elements
        # Total: 524,288 elements * 1 byte = 524,288
        assert spec.bytes_per_block_per_layer() == 524_288

    def test_fp8_is_half_of_fp16(self) -> None:
        """FP8 should be exactly 50% of FP16 memory (no overhead)."""
        spec_fp16 = _make_spec(kv_format="fp", kv_bits=None)
        spec_fp8 = _make_spec(kv_format="fp", kv_bits=8)
        assert spec_fp8.bytes_per_block_per_layer() * 2 == spec_fp16.bytes_per_block_per_layer()

    def test_fp8_less_than_quantized_q8(self) -> None:
        """Native FP8 should use less memory than software Q8 (no scales/biases)."""
        spec_fp8 = _make_spec(kv_format="fp", kv_bits=8)
        spec_q8 = _make_spec(kv_format="quantized", kv_bits=8)
        assert spec_fp8.bytes_per_block_per_layer() < spec_q8.bytes_per_block_per_layer()

    def test_fp_format_ignores_group_size(self) -> None:
        """FP format byte count should not depend on kv_group_size."""
        spec_g32 = _make_spec(kv_format="fp", kv_bits=8, kv_group_size=32)
        spec_g128 = _make_spec(kv_format="fp", kv_bits=8, kv_group_size=128)
        assert spec_g32.bytes_per_block_per_layer() == spec_g128.bytes_per_block_per_layer()

    def test_fp8_asymmetric_kv(self) -> None:
        """FP8 with asymmetric K/V dimensions (DeepSeek MLA style)."""
        spec = _make_spec(
            kv_format="fp",
            kv_bits=8,
            n_kv_heads=16,
            head_dim=192,
            block_tokens=128,
            n_layers=27,
            layer_types=["global"] * 27,
            v_head_dim=128,
        )
        # K: 16 * 192 * 128 = 393,216 elements
        # V: 16 * 128 * 128 = 262,144 elements
        # Total: 655,360 * 1 byte = 655,360
        assert spec.bytes_per_block_per_layer() == 655_360

    def test_fp4_byte_calculation(self) -> None:
        """FP4 (kv_bits=4) with fp format: raw bits, no overhead."""
        spec = _make_spec(kv_format="fp", kv_bits=4)
        # Total elements: 524,288
        # 4 bits each = 524,288 * 4 / 8 = 262,144 bytes
        assert spec.bytes_per_block_per_layer() == 262_144


class TestQuantizedFormatBackwardCompat:
    """Ensure kv_format='quantized' produces identical results to pre-kv_format behavior."""

    def test_q4_unchanged(self) -> None:
        spec = _make_spec(kv_format="quantized", kv_bits=4, kv_group_size=64)
        # Same expected value as TestMemoryBudgetFormulas.test_q4_formula_step_by_step
        # but with n_kv_heads=8, head_dim=128
        k_elements = 8 * 128 * 256
        v_elements = 8 * 128 * 256
        total_elements = k_elements + v_elements
        weight_bytes = (total_elements * 4) // 8
        groups = (k_elements + 64 - 1) // 64 + (v_elements + 64 - 1) // 64
        expected = weight_bytes + groups * 2 + groups * 2
        assert spec.bytes_per_block_per_layer() == expected

    def test_q8_unchanged(self) -> None:
        spec = _make_spec(kv_format="quantized", kv_bits=8, kv_group_size=64)
        k_elements = 8 * 128 * 256
        v_elements = 8 * 128 * 256
        total_elements = k_elements + v_elements
        weight_bytes = (total_elements * 8) // 8
        groups = (k_elements + 64 - 1) // 64 + (v_elements + 64 - 1) // 64
        expected = weight_bytes + groups * 2 + groups * 2
        assert spec.bytes_per_block_per_layer() == expected

    def test_fp16_unchanged(self) -> None:
        spec = _make_spec(kv_format="quantized", kv_bits=None)
        total_elements = 8 * 128 * 256 * 2
        assert spec.bytes_per_block_per_layer() == total_elements * 2


class TestTRTSpecPattern:
    """Test the pattern TRT spec extractors will use."""

    def test_trt_spec_fp16(self) -> None:
        """TRT engines typically report FP16 KV cache with no quantization."""
        spec = _make_spec(kv_format="fp", kv_bits=None)
        assert spec.kv_format == "fp"
        assert spec.kv_bits is None
        total_elements = 8 * 128 * 256 * 2
        assert spec.bytes_per_block_per_layer() == total_elements * 2

    def test_trt_spec_fp8(self) -> None:
        """TRT FP8 quantization is native — 1 byte per element."""
        spec = _make_spec(kv_format="fp", kv_bits=8)
        assert spec.kv_format == "fp"
        assert spec.kv_bits == 8

    def test_trt_spec_equality(self) -> None:
        """Two identical TRT specs should be equal."""
        spec1 = _make_spec(kv_format="fp", kv_bits=8)
        spec2 = _make_spec(kv_format="fp", kv_bits=8)
        assert spec1 == spec2

    def test_trt_spec_inequality_with_mlx(self) -> None:
        """TRT fp spec should differ from MLX quantized spec even with same kv_bits."""
        spec_trt = _make_spec(kv_format="fp", kv_bits=8)
        spec_mlx = _make_spec(kv_format="quantized", kv_bits=8)
        assert spec_trt != spec_mlx


class TestTRTErrors:
    """Test TRT-specific error classes."""

    def test_trt_subprocess_error_is_semantic(self) -> None:
        from agent_memory.domain.errors import SemanticError

        err = TRTSubprocessError("llm_inference timed out after 30s")
        assert isinstance(err, SemanticError)
        assert str(err) == "llm_inference timed out after 30s"

    def test_trt_engine_error_is_semantic(self) -> None:
        from agent_memory.domain.errors import SemanticError

        err = TRTEngineError("failed to load engine: out of GPU memory")
        assert isinstance(err, SemanticError)
        assert str(err) == "failed to load engine: out of GPU memory"

    def test_trt_layout_error_is_semantic(self) -> None:
        from agent_memory.domain.errors import SemanticError

        err = TRTLayoutError("expected [L,2,H,S,D] got [L,H,2,S,D]")
        assert isinstance(err, SemanticError)
        assert str(err) == "expected [L,2,H,S,D] got [L,H,2,S,D]"

    def test_trt_errors_are_catchable_as_semantic(self) -> None:
        """All TRT errors should be catchable via SemanticError."""
        from agent_memory.domain.errors import SemanticError

        for err_cls in [TRTSubprocessError, TRTEngineError, TRTLayoutError]:
            with pytest.raises(SemanticError):
                raise err_cls("test")
