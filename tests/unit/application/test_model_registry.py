# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Yakov Shkolnikov and contributors
"""Unit tests for ModelRegistry lifecycle management."""

import sys
from unittest.mock import MagicMock, Mock, patch

import pytest

# Mock MLX modules before importing ModelRegistry
sys.modules["mlx"] = MagicMock()
sys.modules["mlx.core"] = MagicMock()
sys.modules["mlx.utils"] = MagicMock()
sys.modules["mlx_lm"] = MagicMock()

from agent_memory.application.model_registry import ModelRegistry
from agent_memory.domain.errors import ModelNotFoundError
from agent_memory.domain.value_objects import ModelCacheSpec


@pytest.fixture
def mock_loader():
    loader = Mock()
    loader.load_model = Mock()
    loader.get_active_memory = Mock()
    loader.clear_cache = Mock()
    return loader


@pytest.fixture
def mock_extractor():
    return Mock()


def _make_registry(mock_loader, mock_extractor):
    return ModelRegistry(model_loader=mock_loader, spec_extractor=mock_extractor)


class TestModelRegistryLifecycle:

    def test_init_creates_empty_registry(self, mock_loader, mock_extractor):
        registry = _make_registry(mock_loader, mock_extractor)

        assert registry._model is None
        assert registry._tokenizer is None
        assert registry._spec is None
        assert registry._current_model_id is None
        assert not registry.is_loaded()

    def test_load_model_success(self, mock_loader, mock_extractor):
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_loader.load_model.return_value = (mock_model, mock_tokenizer)

        mock_spec = ModelCacheSpec(
            n_layers=24, n_kv_heads=8, head_dim=128, block_tokens=16, layer_types=["global"] * 24
        )
        mock_extractor.extract_spec.return_value = mock_spec

        registry = _make_registry(mock_loader, mock_extractor)
        model, tokenizer = registry.load_model("mlx-community/SmolLM2-135M-Instruct")

        assert model is mock_model
        assert tokenizer is mock_tokenizer
        assert registry._model is mock_model
        assert registry._spec == mock_spec
        assert registry._current_model_id == "mlx-community/SmolLM2-135M-Instruct"
        assert registry.is_loaded()
        mock_loader.load_model.assert_called_once_with("mlx-community/SmolLM2-135M-Instruct")

    def test_load_model_not_found(self, mock_loader, mock_extractor):
        mock_loader.load_model.side_effect = Exception("Model not found on HuggingFace")

        registry = _make_registry(mock_loader, mock_extractor)

        with pytest.raises(ModelNotFoundError) as exc_info:
            registry.load_model("nonexistent/model")

        assert "Failed to load model nonexistent/model" in str(exc_info.value)
        assert not registry.is_loaded()

    @patch("agent_memory.application.model_registry.gc.collect")
    def test_unload_model_reclaims_memory(self, mock_gc_collect, mock_loader, mock_extractor):
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_loader.load_model.return_value = (mock_model, mock_tokenizer)

        mock_spec = ModelCacheSpec(
            n_layers=24, n_kv_heads=8, head_dim=128, block_tokens=16, layer_types=["global"] * 24
        )
        mock_extractor.extract_spec.return_value = mock_spec
        mock_loader.get_active_memory.side_effect = [100 * 1024 * 1024, 0]

        registry = _make_registry(mock_loader, mock_extractor)
        registry.load_model("mlx-community/SmolLM2-135M-Instruct")
        assert registry.is_loaded()

        registry.unload_model()

        assert registry._model is None
        assert registry._tokenizer is None
        assert registry._spec is None
        assert registry._current_model_id is None
        assert not registry.is_loaded()
        mock_gc_collect.assert_called_once()
        mock_loader.clear_cache.assert_called_once()

    def test_unload_when_no_model_loaded(self, mock_loader, mock_extractor):
        registry = _make_registry(mock_loader, mock_extractor)
        registry.unload_model()
        assert not registry.is_loaded()

    def test_get_current_returns_model_and_tokenizer(self, mock_loader, mock_extractor):
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_loader.load_model.return_value = (mock_model, mock_tokenizer)

        mock_spec = ModelCacheSpec(
            n_layers=24, n_kv_heads=8, head_dim=128, block_tokens=16, layer_types=["global"] * 24
        )
        mock_extractor.extract_spec.return_value = mock_spec

        registry = _make_registry(mock_loader, mock_extractor)
        assert registry.get_current() is None

        registry.load_model("mlx-community/SmolLM2-135M-Instruct")
        current = registry.get_current()
        assert current == (mock_model, mock_tokenizer)

    def test_get_current_spec_returns_spec(self, mock_loader, mock_extractor):
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_loader.load_model.return_value = (mock_model, mock_tokenizer)

        mock_spec = ModelCacheSpec(
            n_layers=24, n_kv_heads=8, head_dim=128, block_tokens=16, layer_types=["global"] * 24
        )
        mock_extractor.extract_spec.return_value = mock_spec

        registry = _make_registry(mock_loader, mock_extractor)
        assert registry.get_current_spec() is None

        registry.load_model("mlx-community/SmolLM2-135M-Instruct")
        assert registry.get_current_spec() == mock_spec

    def test_get_current_id_returns_model_id(self, mock_loader, mock_extractor):
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_loader.load_model.return_value = (mock_model, mock_tokenizer)

        mock_spec = ModelCacheSpec(
            n_layers=24, n_kv_heads=8, head_dim=128, block_tokens=16, layer_types=["global"] * 24
        )
        mock_extractor.extract_spec.return_value = mock_spec

        registry = _make_registry(mock_loader, mock_extractor)
        assert registry.get_current_id() is None

        registry.load_model("mlx-community/SmolLM2-135M-Instruct")
        assert registry.get_current_id() == "mlx-community/SmolLM2-135M-Instruct"

    @patch("agent_memory.application.model_registry.gc.collect")
    def test_swap_models_lifecycle(self, mock_gc_collect, mock_loader, mock_extractor):
        mock_model_1 = MagicMock()
        mock_tokenizer_1 = MagicMock()
        mock_spec_1 = ModelCacheSpec(
            n_layers=24, n_kv_heads=8, head_dim=128, block_tokens=16, layer_types=["global"] * 24
        )

        mock_model_2 = MagicMock()
        mock_tokenizer_2 = MagicMock()
        mock_spec_2 = ModelCacheSpec(
            n_layers=32, n_kv_heads=16, head_dim=128, block_tokens=16, layer_types=["global"] * 32
        )

        mock_loader.load_model.side_effect = [
            (mock_model_1, mock_tokenizer_1),
            (mock_model_2, mock_tokenizer_2),
        ]
        mock_extractor.extract_spec.side_effect = [mock_spec_1, mock_spec_2]
        mock_loader.get_active_memory.side_effect = [100 * 1024 * 1024, 0]

        registry = _make_registry(mock_loader, mock_extractor)

        registry.load_model("mlx-community/SmolLM2-135M-Instruct")
        assert registry.get_current_id() == "mlx-community/SmolLM2-135M-Instruct"
        assert registry.get_current_spec() == mock_spec_1

        registry.unload_model()
        assert not registry.is_loaded()

        registry.load_model("mlx-community/Qwen2.5-0.5B-Instruct-4bit")
        assert registry.get_current_id() == "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
        assert registry.get_current_spec() == mock_spec_2

        mock_gc_collect.assert_called_once()
        mock_loader.clear_cache.assert_called_once()
