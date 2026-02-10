# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Yakov Shkolnikov and contributors
"""Unit tests for domain value objects."""

import pytest

from agent_memory.domain.errors import ModelSpecValidationError
from agent_memory.domain.value_objects import (
    CacheKey,
    GenerationResult,
    ModelCacheSpec,
)

pytestmark = pytest.mark.unit


class TestGenerationResult:
    """Test suite for GenerationResult value object."""

    def test_create_valid_result(self) -> None:
        """Should create result with all fields."""
        result = GenerationResult(
            text="Hello, world!",
            tokens=[1, 2, 3, 4, 5],
            cache=[("k_tensor", "v_tensor")],
        )

        assert result.text == "Hello, world!"
        assert result.tokens == [1, 2, 3, 4, 5]
        assert result.cache == [("k_tensor", "v_tensor")]

    def test_is_immutable(self) -> None:
        """Should be immutable (frozen dataclass)."""
        result = GenerationResult(text="test", tokens=[1], cache=[])

        with pytest.raises(AttributeError):
            result.text = "modified"  # type: ignore[misc]

    def test_equality(self) -> None:
        """Should be equal if all values match (value object semantics)."""
        result1 = GenerationResult(text="test", tokens=[1, 2], cache=[])
        result2 = GenerationResult(text="test", tokens=[1, 2], cache=[])

        assert result1 == result2

    def test_inequality(self) -> None:
        """Should be unequal if any value differs."""
        result1 = GenerationResult(text="test1", tokens=[1], cache=[])
        result2 = GenerationResult(text="test2", tokens=[1], cache=[])

        assert result1 != result2


class TestCacheKey:
    """Test suite for CacheKey value object."""

    def test_create_valid_key(self) -> None:
        """Should create key with all fields."""
        key = CacheKey(
            agent_id="agent_123",
            model_id="gemma-3-12b",
            prefix_hash="abc123",
        )

        assert key.agent_id == "agent_123"
        assert key.model_id == "gemma-3-12b"
        assert key.prefix_hash == "abc123"

    def test_is_immutable(self) -> None:
        """Should be immutable (frozen dataclass)."""
        key = CacheKey(agent_id="agent_1", model_id="model", prefix_hash="hash")

        with pytest.raises(AttributeError):
            key.agent_id = "modified"  # type: ignore[misc]

    def test_equality(self) -> None:
        """Should be equal if all values match."""
        key1 = CacheKey(agent_id="agent_1", model_id="model", prefix_hash="hash")
        key2 = CacheKey(agent_id="agent_1", model_id="model", prefix_hash="hash")

        assert key1 == key2

    def test_inequality_different_agent(self) -> None:
        """Should be unequal if agent_id differs."""
        key1 = CacheKey(agent_id="agent_1", model_id="model", prefix_hash="hash")
        key2 = CacheKey(agent_id="agent_2", model_id="model", prefix_hash="hash")

        assert key1 != key2

    def test_can_use_as_dict_key(self) -> None:
        """Should be hashable (can be used as dict key)."""
        key = CacheKey(agent_id="agent_1", model_id="model", prefix_hash="hash")
        cache_dict = {key: "cached_data"}

        assert cache_dict[key] == "cached_data"


class TestModelCacheSpec:
    """Test suite for ModelCacheSpec value object."""

    def test_create_valid_spec(self) -> None:
        """Should create spec with all required fields."""
        spec = ModelCacheSpec(
            n_layers=48,
            n_kv_heads=8,
            head_dim=256,
            block_tokens=256,
            layer_types=["global"] * 48,
        )

        assert spec.n_layers == 48
        assert spec.n_kv_heads == 8
        assert spec.head_dim == 256
        assert spec.block_tokens == 256
        assert len(spec.layer_types) == 48
        assert spec.sliding_window_size is None

    def test_create_with_sliding_window(self) -> None:
        """Should create spec with sliding window configuration."""
        spec = ModelCacheSpec(
            n_layers=48,
            n_kv_heads=8,
            head_dim=256,
            block_tokens=256,
            layer_types=["global"] * 8 + ["sliding_window"] * 40,
            sliding_window_size=1024,
        )

        assert spec.sliding_window_size == 1024
        assert spec.layer_types[0] == "global"
        assert spec.layer_types[8] == "sliding_window"

    def test_is_immutable(self) -> None:
        """Should be immutable (frozen dataclass)."""
        spec = ModelCacheSpec(
            n_layers=48,
            n_kv_heads=8,
            head_dim=256,
            block_tokens=256,
            layer_types=["global"] * 48,
        )

        with pytest.raises(AttributeError):
            spec.n_layers = 32  # type: ignore[misc]

    def test_reject_layer_types_length_mismatch(self) -> None:
        """Should raise when len(layer_types) != n_layers."""
        with pytest.raises(
            ModelSpecValidationError, match=r"layer_types length .* must equal n_layers"
        ):
            ModelCacheSpec(
                n_layers=48,
                n_kv_heads=8,
                head_dim=256,
                block_tokens=256,
                layer_types=["global"] * 40,
            )

    def test_bytes_per_block_per_layer_computation(self) -> None:
        """Should compute bytes per block correctly for FP16."""
        spec = ModelCacheSpec(
            n_layers=48,
            n_kv_heads=8,
            head_dim=256,
            block_tokens=256,
            layer_types=["global"] * 48,
            kv_bits=None,  # FP16
        )

        # Formula: n_kv_heads * head_dim * 2 (K+V) * 2 (float16) * block_tokens
        expected = 8 * 256 * 2 * 2 * 256
        assert spec.bytes_per_block_per_layer() == expected
        assert spec.bytes_per_block_per_layer() == 2_097_152

    def test_max_blocks_for_layer_global(self) -> None:
        """Should return None for global attention layers (no limit)."""
        spec = ModelCacheSpec(
            n_layers=48,
            n_kv_heads=8,
            head_dim=256,
            block_tokens=256,
            layer_types=["global"] * 48,
        )

        assert spec.max_blocks_for_layer("global") is None

    def test_max_blocks_for_layer_sliding_window(self) -> None:
        """Should compute max blocks for sliding window layers."""
        spec = ModelCacheSpec(
            n_layers=48,
            n_kv_heads=8,
            head_dim=256,
            block_tokens=256,
            layer_types=["sliding_window"] * 48,
            sliding_window_size=512,
        )

        assert spec.max_blocks_for_layer("sliding_window") == 2

    def test_max_blocks_for_layer_sliding_window_partial_block(self) -> None:
        """Should round up when window size doesn't align with block size."""
        spec = ModelCacheSpec(
            n_layers=48,
            n_kv_heads=8,
            head_dim=256,
            block_tokens=256,
            layer_types=["sliding_window"] * 48,
            sliding_window_size=1024,
        )

        assert spec.max_blocks_for_layer("sliding_window") == 4

    def test_max_blocks_for_layer_exact_multiple(self) -> None:
        """Should handle exact multiples of block size."""
        spec = ModelCacheSpec(
            n_layers=48,
            n_kv_heads=8,
            head_dim=256,
            block_tokens=256,
            layer_types=["sliding_window"] * 48,
            sliding_window_size=768,
        )

        assert spec.max_blocks_for_layer("sliding_window") == 3

    def test_equality(self) -> None:
        """Should be equal if all values match."""
        spec1 = ModelCacheSpec(
            n_layers=48,
            n_kv_heads=8,
            head_dim=256,
            block_tokens=256,
            layer_types=["global"] * 48,
        )
        spec2 = ModelCacheSpec(
            n_layers=48,
            n_kv_heads=8,
            head_dim=256,
            block_tokens=256,
            layer_types=["global"] * 48,
        )

        assert spec1 == spec2

    def test_inequality(self) -> None:
        """Should be unequal if any value differs."""
        spec1 = ModelCacheSpec(
            n_layers=48,
            n_kv_heads=8,
            head_dim=256,
            block_tokens=256,
            layer_types=["global"] * 48,
        )
        spec2 = ModelCacheSpec(
            n_layers=32,
            n_kv_heads=8,
            head_dim=256,
            block_tokens=256,
            layer_types=["global"] * 32,
        )

        assert spec1 != spec2

    def test_moe_alternating_layer_pattern(self) -> None:
        """Should handle MoE layer patterns."""
        n_layers = 24
        layer_types = ["global"] * n_layers

        spec = ModelCacheSpec(
            n_layers=n_layers,
            n_kv_heads=8,
            head_dim=128,
            block_tokens=256,
            layer_types=layer_types,
            sliding_window_size=None,
        )

        assert spec.n_layers == 24
        assert all(lt == "global" for lt in spec.layer_types)
        for layer_id in range(spec.n_layers):
            assert spec.max_blocks_for_layer(spec.layer_types[layer_id]) is None

    def test_hybrid_model_with_sliding_window(self) -> None:
        """Should handle hybrid sliding window patterns."""
        layer_types = ["global"] * 8 + ["sliding_window"] * 24

        spec = ModelCacheSpec(
            n_layers=32,
            n_kv_heads=8,
            head_dim=256,
            block_tokens=256,
            layer_types=layer_types,
            sliding_window_size=1024,
        )

        for layer_id in range(8):
            assert spec.max_blocks_for_layer(spec.layer_types[layer_id]) is None

        for layer_id in range(8, 32):
            assert spec.max_blocks_for_layer(spec.layer_types[layer_id]) == 4


class TestMemoryBudgetFormulas:
    """Verify memory formulas step-by-step against analytical computation."""

    def test_fp16_formula_step_by_step(self) -> None:
        """FP16: n_kv_heads * head_dim * block_tokens * 2(K+V) * 2(bytes)."""
        n_kv_heads = 8
        head_dim = 256
        block_tokens = 256

        spec = ModelCacheSpec(
            n_layers=48,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            block_tokens=block_tokens,
            layer_types=["global"] * 48,
            kv_bits=None,
        )

        elements_per_kv = n_kv_heads * head_dim * block_tokens  # 524,288
        total_elements = elements_per_kv * 2  # K and V = 1,048,576
        expected_bytes = total_elements * 2  # float16 = 2 bytes = 2,097,152

        assert elements_per_kv == 524_288
        assert total_elements == 1_048_576
        assert expected_bytes == 2_097_152
        assert spec.bytes_per_block_per_layer() == expected_bytes

    def test_q4_formula_step_by_step(self) -> None:
        """Q4: weight_bytes + scales_bytes + biases_bytes."""
        n_kv_heads = 8
        head_dim = 256
        block_tokens = 256
        kv_group_size = 64

        spec = ModelCacheSpec(
            n_layers=48,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            block_tokens=block_tokens,
            layer_types=["global"] * 48,
            kv_bits=4,
            kv_group_size=kv_group_size,
        )

        elements_per_kv = n_kv_heads * head_dim * block_tokens  # 524,288
        total_elements = elements_per_kv * 2  # 1,048,576

        weight_bytes = (total_elements * 4) // 8  # 4 bits per element = 524,288
        groups_per_kv = (elements_per_kv + kv_group_size - 1) // kv_group_size  # 8,192
        total_groups = groups_per_kv * 2  # K and V = 16,384
        scales_bytes = total_groups * 2  # float16 = 32,768
        biases_bytes = total_groups * 2  # float16 = 32,768
        expected = weight_bytes + scales_bytes + biases_bytes  # 589,824

        assert weight_bytes == 524_288
        assert groups_per_kv == 8_192
        assert total_groups == 16_384
        assert scales_bytes == 32_768
        assert biases_bytes == 32_768
        assert expected == 589_824
        assert spec.bytes_per_block_per_layer() == expected

    def test_q8_formula_step_by_step(self) -> None:
        """Q8: same structure as Q4 but 8 bits per element."""
        n_kv_heads = 8
        head_dim = 256
        block_tokens = 256
        kv_group_size = 64

        spec = ModelCacheSpec(
            n_layers=48,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            block_tokens=block_tokens,
            layer_types=["global"] * 48,
            kv_bits=8,
            kv_group_size=kv_group_size,
        )

        elements_per_kv = n_kv_heads * head_dim * block_tokens
        total_elements = elements_per_kv * 2

        weight_bytes = (total_elements * 8) // 8  # 8 bits = 1 byte each = 1,048,576
        groups_per_kv = (elements_per_kv + kv_group_size - 1) // kv_group_size
        total_groups = groups_per_kv * 2
        scales_bytes = total_groups * 2
        biases_bytes = total_groups * 2
        expected = weight_bytes + scales_bytes + biases_bytes  # 1,114,112

        assert weight_bytes == 1_048_576
        assert expected == 1_114_112
        assert spec.bytes_per_block_per_layer() == expected

    def test_q4_is_roughly_28_percent_of_fp16(self) -> None:
        """Q4 should be ~25-28% of FP16 memory per block."""
        spec_fp16 = ModelCacheSpec(
            n_layers=48,
            n_kv_heads=8,
            head_dim=256,
            block_tokens=256,
            layer_types=["global"] * 48,
            kv_bits=None,
        )
        spec_q4 = ModelCacheSpec(
            n_layers=48,
            n_kv_heads=8,
            head_dim=256,
            block_tokens=256,
            layer_types=["global"] * 48,
            kv_bits=4,
            kv_group_size=64,
        )

        fp16_bytes = spec_fp16.bytes_per_block_per_layer()
        q4_bytes = spec_q4.bytes_per_block_per_layer()
        ratio = q4_bytes / fp16_bytes

        # Q4 should be ~28.1% of FP16
        assert 0.25 <= ratio <= 0.30, f"Q4/FP16 ratio {ratio:.3f} outside 25-30% range"

    def test_q8_is_roughly_53_percent_of_fp16(self) -> None:
        """Q8 should be ~53% of FP16 memory."""
        spec_fp16 = ModelCacheSpec(
            n_layers=48,
            n_kv_heads=8,
            head_dim=256,
            block_tokens=256,
            layer_types=["global"] * 48,
            kv_bits=None,
        )
        spec_q8 = ModelCacheSpec(
            n_layers=48,
            n_kv_heads=8,
            head_dim=256,
            block_tokens=256,
            layer_types=["global"] * 48,
            kv_bits=8,
            kv_group_size=64,
        )

        fp16_bytes = spec_fp16.bytes_per_block_per_layer()
        q8_bytes = spec_q8.bytes_per_block_per_layer()
        ratio = q8_bytes / fp16_bytes

        assert 0.50 <= ratio <= 0.55, f"Q8/FP16 ratio {ratio:.3f} outside 50-55% range"

    def test_kv_bits_16_equals_none(self) -> None:
        """kv_bits=16 should produce same result as kv_bits=None (both FP16)."""
        spec_none = ModelCacheSpec(
            n_layers=12,
            n_kv_heads=4,
            head_dim=64,
            block_tokens=256,
            layer_types=["global"] * 12,
            kv_bits=None,
        )
        spec_16 = ModelCacheSpec(
            n_layers=12,
            n_kv_heads=4,
            head_dim=64,
            block_tokens=256,
            layer_types=["global"] * 12,
            kv_bits=16,
        )

        assert spec_none.bytes_per_block_per_layer() == spec_16.bytes_per_block_per_layer()


class TestModelCacheSpecNewValidations:
    """Test new validation rules added for positive values and power-of-2 group size."""

    def test_reject_zero_n_layers(self) -> None:
        with pytest.raises(ModelSpecValidationError, match="n_layers must be > 0"):
            ModelCacheSpec(
                n_layers=0, n_kv_heads=4, head_dim=64, block_tokens=256,
                layer_types=[],
            )

    def test_reject_negative_n_layers(self) -> None:
        with pytest.raises(ModelSpecValidationError, match="n_layers must be > 0"):
            ModelCacheSpec(
                n_layers=-1, n_kv_heads=4, head_dim=64, block_tokens=256,
                layer_types=[],
            )

    def test_reject_zero_n_kv_heads(self) -> None:
        with pytest.raises(ModelSpecValidationError, match="n_kv_heads must be > 0"):
            ModelCacheSpec(
                n_layers=12, n_kv_heads=0, head_dim=64, block_tokens=256,
                layer_types=["global"] * 12,
            )

    def test_reject_zero_head_dim(self) -> None:
        with pytest.raises(ModelSpecValidationError, match="head_dim must be > 0"):
            ModelCacheSpec(
                n_layers=12, n_kv_heads=4, head_dim=0, block_tokens=256,
                layer_types=["global"] * 12,
            )

    def test_reject_zero_block_tokens(self) -> None:
        with pytest.raises(ModelSpecValidationError, match="block_tokens must be > 0"):
            ModelCacheSpec(
                n_layers=12, n_kv_heads=4, head_dim=64, block_tokens=0,
                layer_types=["global"] * 12,
            )

    def test_reject_non_power_of_2_group_size(self) -> None:
        with pytest.raises(ModelSpecValidationError, match="power of 2"):
            ModelCacheSpec(
                n_layers=12, n_kv_heads=4, head_dim=64, block_tokens=256,
                layer_types=["global"] * 12,
                kv_bits=4,
                kv_group_size=100,
            )

    def test_accept_power_of_2_group_sizes(self) -> None:
        for gs in [1, 2, 4, 8, 16, 32, 64, 128]:
            spec = ModelCacheSpec(
                n_layers=12, n_kv_heads=4, head_dim=64, block_tokens=256,
                layer_types=["global"] * 12,
                kv_bits=4,
                kv_group_size=gs,
            )
            assert spec.kv_group_size == gs

    def test_reject_invalid_kv_bits(self) -> None:
        with pytest.raises(ModelSpecValidationError, match="kv_bits must be one of"):
            ModelCacheSpec(
                n_layers=12, n_kv_heads=4, head_dim=64, block_tokens=256,
                layer_types=["global"] * 12,
                kv_bits=6,
            )

    # --- MLA (asymmetric K/V) tests ---

    def test_v_head_dim_defaults_to_head_dim(self) -> None:
        """v_head_dim=None means symmetric K=V=head_dim."""
        spec = ModelCacheSpec(
            n_layers=27, n_kv_heads=16, head_dim=192,
            block_tokens=128, layer_types=["global"] * 27,
        )
        assert spec.v_head_dim is None
        assert spec.effective_v_head_dim == 192

    def test_asymmetric_kv_dims(self) -> None:
        """DeepSeek V2 MLA: K=192, V=128."""
        spec = ModelCacheSpec(
            n_layers=27, n_kv_heads=16, head_dim=192,
            block_tokens=128, layer_types=["global"] * 27,
            v_head_dim=128,
        )
        assert spec.head_dim == 192
        assert spec.v_head_dim == 128
        assert spec.effective_v_head_dim == 128

    def test_bytes_per_block_asymmetric_fp16(self) -> None:
        """Asymmetric K/V should compute correct FP16 memory."""
        spec = ModelCacheSpec(
            n_layers=27, n_kv_heads=16, head_dim=192,
            block_tokens=128, layer_types=["global"] * 27,
            kv_bits=None, v_head_dim=128,
        )
        # K: 16*192*128 = 393,216 elements
        # V: 16*128*128 = 262,144 elements
        # Total: 655,360 * 2 bytes = 1,310,720
        assert spec.bytes_per_block_per_layer() == 1_310_720

    def test_bytes_per_block_asymmetric_q4(self) -> None:
        """Asymmetric K/V should compute correct Q4 memory."""
        spec = ModelCacheSpec(
            n_layers=27, n_kv_heads=16, head_dim=192,
            block_tokens=128, layer_types=["global"] * 27,
            kv_bits=4, kv_group_size=64, v_head_dim=128,
        )
        # K elements: 16*192*128 = 393,216
        # V elements: 16*128*128 = 262,144
        # Total elements: 655,360
        # Weight bytes: 655,360 * 4 / 8 = 327,680
        # K groups: 393,216/64 = 6,144 ; V groups: 262,144/64 = 4,096
        # Scales: 10,240 * 2 = 20,480 ; Biases: 10,240 * 2 = 20,480
        expected = 327_680 + 20_480 + 20_480
        assert spec.bytes_per_block_per_layer() == expected

    def test_bytes_per_block_symmetric_unchanged(self) -> None:
        """Symmetric K=V should give same result as before (no v_head_dim)."""
        spec_old = ModelCacheSpec(
            n_layers=27, n_kv_heads=16, head_dim=128,
            block_tokens=128, layer_types=["global"] * 27,
            kv_bits=4, kv_group_size=64,
        )
        spec_explicit = ModelCacheSpec(
            n_layers=27, n_kv_heads=16, head_dim=128,
            block_tokens=128, layer_types=["global"] * 27,
            kv_bits=4, kv_group_size=64, v_head_dim=128,
        )
        assert spec_old.bytes_per_block_per_layer() == spec_explicit.bytes_per_block_per_layer()

    def test_reject_invalid_v_head_dim(self) -> None:
        with pytest.raises(ModelSpecValidationError, match="v_head_dim must be > 0"):
            ModelCacheSpec(
                n_layers=12, n_kv_heads=4, head_dim=128,
                block_tokens=128, layer_types=["global"] * 12,
                v_head_dim=0,
            )
