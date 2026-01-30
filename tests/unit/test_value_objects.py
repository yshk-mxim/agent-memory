"""Unit tests for domain value objects."""

import pytest

from semantic.domain.errors import ModelSpecValidationError
from semantic.domain.value_objects import (
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
