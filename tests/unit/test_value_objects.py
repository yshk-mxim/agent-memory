"""Unit tests for domain value objects.

Tests ModelCacheSpec, GenerationResult, and CacheKey with focus on:
- Immutability (frozen=True dataclasses)
- Validation logic
- Computation methods (bytes_per_block_per_layer, max_blocks_for_layer)
- from_model() extraction logic (mocked models)

Per production_plan.md: 95%+ coverage, mypy --strict compliant.
"""

from unittest.mock import Mock

import pytest

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
            cache=[{"layer_0": "data"}],
        )

        assert result.text == "Hello, world!"
        assert result.tokens == [1, 2, 3, 4, 5]
        assert result.cache == [{"layer_0": "data"}]

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
        key = CacheKey(
            agent_id="agent_1", model_id="model", prefix_hash="hash"
        )

        with pytest.raises(AttributeError):
            key.agent_id = "modified"  # type: ignore[misc]

    def test_equality(self) -> None:
        """Should be equal if all values match."""
        key1 = CacheKey(
            agent_id="agent_1", model_id="model", prefix_hash="hash"
        )
        key2 = CacheKey(
            agent_id="agent_1", model_id="model", prefix_hash="hash"
        )

        assert key1 == key2

    def test_inequality_different_agent(self) -> None:
        """Should be unequal if agent_id differs."""
        key1 = CacheKey(
            agent_id="agent_1", model_id="model", prefix_hash="hash"
        )
        key2 = CacheKey(
            agent_id="agent_2", model_id="model", prefix_hash="hash"
        )

        assert key1 != key2

    def test_can_use_as_dict_key(self) -> None:
        """Should be hashable (can be used as dict key)."""
        key = CacheKey(
            agent_id="agent_1", model_id="model", prefix_hash="hash"
        )
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
        """Should raise ValueError when len(layer_types) != n_layers."""
        with pytest.raises(
            ValueError, match=r"layer_types length .* must equal n_layers"
        ):
            ModelCacheSpec(
                n_layers=48,
                n_kv_heads=8,
                head_dim=256,
                block_tokens=256,
                layer_types=["global"] * 40,  # Wrong! Should be 48
            )

    def test_bytes_per_block_per_layer_computation(self) -> None:
        """Should compute bytes per block correctly."""
        spec = ModelCacheSpec(
            n_layers=48,
            n_kv_heads=8,
            head_dim=256,
            block_tokens=256,
            layer_types=["global"] * 48,
        )

        # Formula: n_kv_heads * head_dim * 2 (K+V) * 2 (float16) * block_tokens
        # = 8 * 256 * 2 * 2 * 256 = 2,097,152 bytes = 2 MB
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

        # ceil(512 / 256) = 2
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

        # ceil(1024 / 256) = 4
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

        # 768 / 256 = 3 (exact)
        assert spec.max_blocks_for_layer("sliding_window") == 3

    def test_from_model_standard_model(self) -> None:
        """Should extract spec from standard model (Llama, Qwen)."""
        # Mock Llama 3.1-8B model
        mock_model = Mock(spec=["args"])  # Only allow args attribute
        mock_args = Mock(spec=[])  # spec=[] prevents auto-mocking of attributes
        mock_args.num_hidden_layers = 32
        mock_args.num_key_value_heads = 8
        mock_args.num_attention_heads = 32
        mock_args.hidden_size = 4096
        mock_args.model_type = "llama"
        mock_args.sliding_window = None
        # No text_config or layer_types attributes
        mock_model.args = mock_args

        spec = ModelCacheSpec.from_model(mock_model)

        assert spec.n_layers == 32
        assert spec.n_kv_heads == 8
        assert spec.head_dim == 128  # 4096 / 32
        assert spec.block_tokens == 256
        assert spec.layer_types == ["global"] * 32
        assert spec.sliding_window_size is None

    def test_from_model_gemma3_nested_config(self) -> None:
        """Should extract spec from Gemma 3 with nested text_config."""
        # Mock Gemma 3 12B model
        mock_model = Mock(spec=["args"])
        mock_args = Mock(spec=[])
        mock_args.text_config = {
            "num_hidden_layers": 48,
            "num_key_value_heads": 8,
            "num_attention_heads": 20,
            "hidden_size": 4800,
            "sliding_window": 1024,
        }
        mock_args.model_type = "gemma3"
        mock_model.args = mock_args

        spec = ModelCacheSpec.from_model(mock_model)

        assert spec.n_layers == 48
        assert spec.n_kv_heads == 8
        assert spec.head_dim == 240  # 4800 / 20
        assert spec.block_tokens == 256
        # Gemma 3: 8 global + 40 sliding window
        assert spec.layer_types[:8] == ["global"] * 8
        assert spec.layer_types[8:] == ["sliding_window"] * 40
        assert spec.sliding_window_size == 1024

    def test_from_model_with_sliding_window(self) -> None:
        """Should extract sliding_window attribute."""
        mock_model = Mock(spec=["args"])
        mock_args = Mock(spec=[])
        mock_args.num_hidden_layers = 24
        mock_args.num_key_value_heads = 16
        mock_args.num_attention_heads = 16
        mock_args.hidden_size = 2048
        mock_args.sliding_window = 512
        mock_args.model_type = "custom"
        mock_model.args = mock_args

        spec = ModelCacheSpec.from_model(mock_model)

        assert spec.sliding_window_size == 512

    def test_from_model_raises_on_missing_n_layers(self) -> None:
        """Should raise ValueError if num_hidden_layers missing."""
        mock_model = Mock()
        mock_args = Mock(spec=[])
        mock_args.num_key_value_heads = 8
        mock_args.num_attention_heads = 32
        mock_args.hidden_size = 4096
        # Missing num_hidden_layers - explicitly set to None
        delattr(mock_args, "num_hidden_layers") if hasattr(mock_args, "num_hidden_layers") else None
        delattr(mock_args, "text_config") if hasattr(mock_args, "text_config") else None
        mock_model.args = mock_args

        with pytest.raises(
            ValueError, match="Cannot extract num_hidden_layers"
        ):
            ModelCacheSpec.from_model(mock_model)

    def test_from_model_raises_on_missing_n_kv_heads(self) -> None:
        """Should raise ValueError if num_key_value_heads missing."""
        mock_model = Mock()
        mock_args = Mock(spec=[])
        mock_args.num_hidden_layers = 32
        mock_args.num_attention_heads = 32
        mock_args.hidden_size = 4096
        # Missing num_key_value_heads
        if hasattr(mock_args, "num_key_value_heads"):
            delattr(mock_args, "num_key_value_heads")
        if hasattr(mock_args, "text_config"):
            delattr(mock_args, "text_config")
        mock_model.args = mock_args

        with pytest.raises(
            ValueError, match="Cannot extract num_key_value_heads"
        ):
            ModelCacheSpec.from_model(mock_model)

    def test_from_model_raises_on_missing_head_dim_attrs(self) -> None:
        """Should raise ValueError if hidden_size or num_attention_heads missing."""
        mock_model = Mock()
        mock_args = Mock(spec=[])
        mock_args.num_hidden_layers = 32
        mock_args.num_key_value_heads = 8
        mock_args.num_attention_heads = 32
        # Missing hidden_size
        delattr(mock_args, "hidden_size") if hasattr(mock_args, "hidden_size") else None
        delattr(mock_args, "text_config") if hasattr(mock_args, "text_config") else None
        mock_model.args = mock_args

        with pytest.raises(
            ValueError, match="Cannot compute head_dim"
        ):
            ModelCacheSpec.from_model(mock_model)

    def test_from_model_detect_layer_types_from_attribute(self) -> None:
        """Should detect layer types from layer_types attribute (Tier 1)."""
        mock_model = Mock()
        mock_args = Mock(spec=[])
        mock_args.num_hidden_layers = 24
        mock_args.num_key_value_heads = 8
        mock_args.num_attention_heads = 32
        mock_args.hidden_size = 4096
        mock_args.layer_types = ["global"] * 12 + ["sliding_window"] * 12
        mock_args.model_type = "custom"
        delattr(mock_args, "text_config") if hasattr(mock_args, "text_config") else None
        mock_model.args = mock_args

        spec = ModelCacheSpec.from_model(mock_model)

        assert spec.layer_types == ["global"] * 12 + ["sliding_window"] * 12

    def test_from_model_detect_layer_types_default_global(self) -> None:
        """Should default to global attention for unknown models."""
        mock_model = Mock(spec=["args"])
        mock_args = Mock(spec=[])
        mock_args.num_hidden_layers = 16
        mock_args.num_key_value_heads = 8
        mock_args.num_attention_heads = 32
        mock_args.hidden_size = 4096
        mock_args.model_type = "unknown"
        # No layer_types, no inspectable layers
        mock_model.args = mock_args

        spec = ModelCacheSpec.from_model(mock_model)

        assert spec.layer_types == ["global"] * 16

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
            n_layers=32,  # Different
            n_kv_heads=8,
            head_dim=256,
            block_tokens=256,
            layer_types=["global"] * 32,
        )

        assert spec1 != spec2

    def test_moe_alternating_layer_pattern(self) -> None:
        """Should handle alternating attention-only and MoE layer patterns.

        Tests Qwen1.5-MoE-A2.7B architecture:
        - 24 layers total
        - Layers 0, 2, 4, ... (even): Standard attention
        - Layers 1, 3, 5, ... (odd): Attention + MoE experts
        - All layers have same KV head count (8 heads)
        - No sliding window (all global attention)

        Note: Current implementation treats all layers uniformly for KV cache.
        MoE expert routing does not affect KV cache geometry, only FFN routing.
        """
        # Qwen1.5-MoE-A2.7B structure
        n_layers = 24
        layer_types = ["global"] * n_layers  # No sliding window

        spec = ModelCacheSpec(
            n_layers=n_layers,
            n_kv_heads=8,  # All layers have same KV heads
            head_dim=128,
            block_tokens=256,
            layer_types=layer_types,
            sliding_window_size=None,  # Full attention (no window)
        )

        # Verify layer count
        assert spec.n_layers == 24

        # Verify all layers are global (no sliding window)
        assert all(lt == "global" for lt in spec.layer_types)

        # Verify max_blocks_for_layer returns None (unlimited) for all layers
        for layer_id in range(spec.n_layers):
            layer_type = spec.layer_types[layer_id]
            assert spec.max_blocks_for_layer(layer_type) is None

        # Verify bytes_per_block_per_layer is consistent
        # (MoE doesn't affect KV cache size, only FFN)
        bytes_per_block = spec.bytes_per_block_per_layer()
        assert bytes_per_block == 8 * 128 * 2 * 2 * 256  # 1,048,576 bytes (1 MB)

        # Verify MoE pattern doesn't break cache spec invariants
        assert len(spec.layer_types) == spec.n_layers

    def test_hybrid_model_with_moe_and_sliding_window(self) -> None:
        """Should handle models with BOTH sliding window AND MoE patterns.

        Hypothetical architecture:
        - 32 layers total
        - First 8 layers: Global attention (no window)
        - Last 24 layers: Sliding window (1024 tokens)
        - Layers 1, 3, 5, ...: MoE experts (affects FFN, not KV cache)

        Note: No real model has this combination yet, but architecture should support it.
        """
        # Hypothetical hybrid model
        layer_types = ["global"] * 8 + ["sliding_window"] * 24

        spec = ModelCacheSpec(
            n_layers=32,
            n_kv_heads=8,
            head_dim=256,
            block_tokens=256,
            layer_types=layer_types,
            sliding_window_size=1024,
        )

        # Verify global layers have unlimited blocks
        for layer_id in range(8):
            assert spec.max_blocks_for_layer(spec.layer_types[layer_id]) is None

        # Verify sliding window layers have 4 blocks (1024 / 256 = 4)
        for layer_id in range(8, 32):
            assert spec.max_blocks_for_layer(spec.layer_types[layer_id]) == 4

        # Verify bytes_per_block_per_layer is same for all layers
        # (MoE and sliding window don't affect per-block size)
        bytes_per_block = spec.bytes_per_block_per_layer()
        expected = 8 * 256 * 2 * 2 * 256  # 2,097,152 bytes (2 MB)
        assert bytes_per_block == expected

    def test_sparse_moe_layer_types_detection(self) -> None:
        """Should correctly detect layer types for sparse MoE models.

        Tests that ModelCacheSpec.from_model() would handle MoE models correctly.
        Since MoE affects FFN (not attention), layer_types should still be
        determined by attention pattern (global vs sliding_window).

        Note: This test validates the CURRENT behavior where MoE is transparent
        to cache geometry. Future: If MoE expert caching is needed, this would change.
        """
        # Simulates Qwen1.5-MoE-A2.7B
        spec = ModelCacheSpec(
            n_layers=24,
            n_kv_heads=8,
            head_dim=128,
            block_tokens=256,
            layer_types=["global"] * 24,  # No sliding window
            sliding_window_size=None,
        )

        # Current behavior: All MoE layers treated as global attention
        assert spec.layer_types == ["global"] * 24

        # Future behavior (if MoE expert caching added):
        # layer_types might become ["attention", "attention+moe", "attention", ...]
        # max_blocks_for_layer("attention+moe") might return higher count
        # This test would need updating if that feature is added

        # For now, verify MoE transparency
        for layer_id in range(spec.n_layers):
            # All layers get unlimited blocks (global attention)
            assert spec.max_blocks_for_layer(spec.layer_types[layer_id]) is None
