# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Yakov Shkolnikov and contributors
"""Unit tests for MLX model spec extractor adapter."""

from unittest.mock import Mock

import pytest

from agent_memory.adapters.outbound.mlx_spec_extractor import (
    Gemma3DetectionStrategy,
    MLXModelSpecExtractor,
    UniformAttentionDetectionStrategy,
    get_extractor,
)
from agent_memory.domain.errors import ModelSpecValidationError

pytestmark = pytest.mark.unit


class TestMLXModelSpecExtractor:
    """Test suite for MLXModelSpecExtractor."""

    def test_extract_spec_standard_model(self) -> None:
        """Should extract spec from standard model (Llama, Qwen)."""
        mock_model = Mock(spec=["args"])
        mock_args = Mock(spec=[])
        mock_args.num_hidden_layers = 32
        mock_args.num_key_value_heads = 8
        mock_args.num_attention_heads = 32
        mock_args.hidden_size = 4096
        mock_args.model_type = "llama"
        mock_args.sliding_window = None
        mock_model.args = mock_args

        extractor = MLXModelSpecExtractor()
        spec = extractor.extract_spec(mock_model)

        assert spec.n_layers == 32
        assert spec.n_kv_heads == 8
        assert spec.head_dim == 128  # 4096 / 32
        assert spec.block_tokens == 256
        assert spec.layer_types == ["global"] * 32
        assert spec.sliding_window_size is None

    def test_extract_spec_gemma3_nested_config(self) -> None:
        """Should extract spec from Gemma 3 with nested text_config."""
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

        extractor = MLXModelSpecExtractor()
        spec = extractor.extract_spec(mock_model)

        assert spec.n_layers == 48
        assert spec.n_kv_heads == 8
        assert spec.head_dim == 240  # 4800 / 20
        assert spec.block_tokens == 256
        assert spec.layer_types[:8] == ["global"] * 8
        assert spec.layer_types[8:] == ["sliding_window"] * 40
        assert spec.sliding_window_size == 1024

    def test_extract_spec_with_sliding_window(self) -> None:
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

        extractor = MLXModelSpecExtractor()
        spec = extractor.extract_spec(mock_model)

        assert spec.sliding_window_size == 512

    def test_raises_on_missing_n_layers(self) -> None:
        """Should raise if num_hidden_layers missing."""
        mock_model = Mock()
        mock_args = Mock(spec=[])
        mock_args.num_key_value_heads = 8
        mock_args.num_attention_heads = 32
        mock_args.hidden_size = 4096
        mock_model.args = mock_args

        extractor = MLXModelSpecExtractor()
        with pytest.raises(ModelSpecValidationError, match="num_hidden_layers"):
            extractor.extract_spec(mock_model)

    def test_raises_on_missing_n_kv_heads(self) -> None:
        """Should raise if num_key_value_heads missing."""
        mock_model = Mock()
        mock_args = Mock(spec=[])
        mock_args.num_hidden_layers = 32
        mock_args.num_attention_heads = 32
        mock_args.hidden_size = 4096
        mock_model.args = mock_args

        extractor = MLXModelSpecExtractor()
        with pytest.raises(ModelSpecValidationError, match="num_key_value_heads"):
            extractor.extract_spec(mock_model)

    def test_raises_on_missing_head_dim_attrs(self) -> None:
        """Should raise if hidden_size or num_attention_heads missing."""
        mock_model = Mock()
        mock_args = Mock(spec=[])
        mock_args.num_hidden_layers = 32
        mock_args.num_key_value_heads = 8
        mock_args.num_attention_heads = 32
        mock_model.args = mock_args

        extractor = MLXModelSpecExtractor()
        with pytest.raises(ModelSpecValidationError, match="head_dim"):
            extractor.extract_spec(mock_model)

    def test_detect_layer_types_from_attribute(self) -> None:
        """Should detect layer types from layer_types attribute (Tier 1)."""
        mock_model = Mock()
        mock_args = Mock(spec=[])
        mock_args.num_hidden_layers = 24
        mock_args.num_key_value_heads = 8
        mock_args.num_attention_heads = 32
        mock_args.hidden_size = 4096
        mock_args.layer_types = ["global"] * 12 + ["sliding_window"] * 12
        mock_args.model_type = "custom"
        mock_model.args = mock_args

        extractor = MLXModelSpecExtractor()
        spec = extractor.extract_spec(mock_model)

        assert spec.layer_types == ["global"] * 12 + ["sliding_window"] * 12

    def test_detect_layer_types_default_global(self) -> None:
        """Should default to global attention for unknown models."""
        mock_model = Mock(spec=["args"])
        mock_args = Mock(spec=[])
        mock_args.num_hidden_layers = 16
        mock_args.num_key_value_heads = 8
        mock_args.num_attention_heads = 32
        mock_args.hidden_size = 4096
        mock_args.model_type = "unknown"
        mock_model.args = mock_args

        extractor = MLXModelSpecExtractor()
        spec = extractor.extract_spec(mock_model)

        assert spec.layer_types == ["global"] * 16


class TestGemma3DetectionStrategy:
    """Test suite for Gemma3DetectionStrategy."""

    def test_detects_gemma3_model(self) -> None:
        """Should detect Gemma 3 hybrid pattern."""
        mock_args = Mock()
        mock_args.model_type = "gemma3"

        strategy = Gemma3DetectionStrategy()
        result = strategy.detect_layer_types(None, mock_args, 48)

        assert result == ["global"] * 8 + ["sliding_window"] * 40

    def test_returns_none_for_other_models(self) -> None:
        """Should return None for non-Gemma 3 models."""
        mock_args = Mock()
        mock_args.model_type = "llama"

        strategy = Gemma3DetectionStrategy()
        result = strategy.detect_layer_types(None, mock_args, 32)

        assert result is None


class TestUniformAttentionDetectionStrategy:
    """Test suite for UniformAttentionDetectionStrategy."""

    def test_returns_all_global(self) -> None:
        """Should return all global layer types."""
        strategy = UniformAttentionDetectionStrategy()
        result = strategy.detect_layer_types(None, None, 32)

        assert result == ["global"] * 32


class TestGetExtractor:
    """Test suite for get_extractor singleton."""

    def test_returns_extractor(self) -> None:
        """Should return an MLXModelSpecExtractor instance."""
        extractor = get_extractor()
        assert isinstance(extractor, MLXModelSpecExtractor)

    def test_returns_same_instance(self) -> None:
        """Should return the same instance on subsequent calls."""
        extractor1 = get_extractor()
        extractor2 = get_extractor()
        assert extractor1 is extractor2
