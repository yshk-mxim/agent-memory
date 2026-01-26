"""Unit tests for ModelRegistry lifecycle management."""

import gc
import sys
from unittest.mock import MagicMock, Mock, patch

import pytest

# Mock MLX modules before importing ModelRegistry
sys.modules["mlx"] = MagicMock()
sys.modules["mlx.core"] = MagicMock()
sys.modules["mlx_lm"] = MagicMock()

from semantic.application.model_registry import ModelRegistry
from semantic.domain.errors import ModelNotFoundError
from semantic.domain.value_objects import ModelCacheSpec


@pytest.fixture
def mock_loader():
    """Create mock ModelLoaderPort for testing (CR-3 architecture fix)."""
    loader = Mock()
    loader.load_model = Mock()
    loader.get_active_memory = Mock()
    loader.clear_cache = Mock()
    return loader


@pytest.fixture
def mock_extractor():
    """Create mock spec extractor."""
    extractor = Mock()
    return extractor


class TestModelRegistryLifecycle:
    """Test ModelRegistry load/unload/tracking methods."""

    def test_init_creates_empty_registry(self, mock_loader):
        """Registry starts with no model loaded."""
        registry = ModelRegistry(model_loader=mock_loader)

        assert registry._model is None
        assert registry._tokenizer is None
        assert registry._spec is None
        assert registry._current_model_id is None
        assert not registry.is_loaded()

    @patch("semantic.application.model_registry.get_extractor")
    def test_load_model_success(self, mock_get_extractor, mock_loader):
        """Loading a model stores state correctly."""
        # Setup mocks
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_loader.load_model.return_value = (mock_model, mock_tokenizer)

        mock_spec = ModelCacheSpec(
            n_layers=24,
            n_kv_heads=8,
            head_dim=128,
            block_tokens=16,
            layer_types=["global"] * 24,
        )
        mock_extractor = Mock()
        mock_extractor.extract_spec.return_value = mock_spec
        mock_get_extractor.return_value = mock_extractor

        # Execute
        registry = ModelRegistry(model_loader=mock_loader)
        model, tokenizer = registry.load_model("mlx-community/SmolLM2-135M-Instruct")

        # Verify
        assert model is mock_model
        assert tokenizer is mock_tokenizer
        assert registry._model is mock_model
        assert registry._tokenizer is mock_tokenizer
        assert registry._spec == mock_spec
        assert registry._current_model_id == "mlx-community/SmolLM2-135M-Instruct"
        assert registry.is_loaded()

        # Verify loader was called
        mock_loader.load_model.assert_called_once_with("mlx-community/SmolLM2-135M-Instruct")

    def test_load_model_not_found(self, mock_loader):
        """Loading a non-existent model raises ModelNotFoundError."""
        mock_loader.load_model.side_effect = Exception("Model not found on HuggingFace")

        registry = ModelRegistry(model_loader=mock_loader)

        with pytest.raises(ModelNotFoundError) as exc_info:
            registry.load_model("nonexistent/model")

        assert "Failed to load model nonexistent/model" in str(exc_info.value)
        assert not registry.is_loaded()

    @patch("semantic.application.model_registry.get_extractor")
    @patch("semantic.application.model_registry.gc.collect")
    def test_unload_model_reclaims_memory(self, mock_gc_collect, mock_get_extractor, mock_loader):
        """Unloading a model clears state and reclaims memory."""
        # Setup mocks
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_loader.load_model.return_value = (mock_model, mock_tokenizer)

        mock_spec = ModelCacheSpec(
            n_layers=24, n_kv_heads=8, head_dim=128, block_tokens=16, layer_types=["global"] * 24
        )
        mock_extractor = Mock()
        mock_extractor.extract_spec.return_value = mock_spec
        mock_get_extractor.return_value = mock_extractor

        # Memory measurements: 100 MB before, 0 MB after
        mock_loader.get_active_memory.side_effect = [100 * 1024 * 1024, 0]

        # Load model
        registry = ModelRegistry(model_loader=mock_loader)
        registry.load_model("mlx-community/SmolLM2-135M-Instruct")

        assert registry.is_loaded()

        # Unload model
        registry.unload_model()

        # Verify state cleared
        assert registry._model is None
        assert registry._tokenizer is None
        assert registry._spec is None
        assert registry._current_model_id is None
        assert not registry.is_loaded()

        # Verify memory cleanup called via loader (CR-3 fix)
        mock_gc_collect.assert_called_once()
        mock_loader.clear_cache.assert_called_once()

    def test_unload_when_no_model_loaded(self, mock_loader):
        """Unloading when no model loaded is a no-op."""
        registry = ModelRegistry(model_loader=mock_loader)

        # Should not raise
        registry.unload_model()

        assert not registry.is_loaded()

    @patch("semantic.application.model_registry.get_extractor")
    def test_get_current_returns_model_and_tokenizer(self, mock_get_extractor, mock_loader):
        """get_current() returns loaded model and tokenizer."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_loader.load_model.return_value = (mock_model, mock_tokenizer)

        mock_spec = ModelCacheSpec(
            n_layers=24, n_kv_heads=8, head_dim=128, block_tokens=16, layer_types=["global"] * 24
        )
        mock_extractor = Mock()
        mock_extractor.extract_spec.return_value = mock_spec
        mock_get_extractor.return_value = mock_extractor

        registry = ModelRegistry(model_loader=mock_loader)

        # Before load
        assert registry.get_current() is None

        # After load
        registry.load_model("mlx-community/SmolLM2-135M-Instruct")
        current = registry.get_current()

        assert current is not None
        assert current[0] is mock_model
        assert current[1] is mock_tokenizer

    @patch("semantic.application.model_registry.get_extractor")
    def test_get_current_spec_returns_spec(self, mock_get_extractor, mock_loader):
        """get_current_spec() returns loaded model's spec."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_loader.load_model.return_value = (mock_model, mock_tokenizer)

        mock_spec = ModelCacheSpec(
            n_layers=24, n_kv_heads=8, head_dim=128, block_tokens=16, layer_types=["global"] * 24
        )
        mock_extractor = Mock()
        mock_extractor.extract_spec.return_value = mock_spec
        mock_get_extractor.return_value = mock_extractor

        registry = ModelRegistry(model_loader=mock_loader)

        # Before load
        assert registry.get_current_spec() is None

        # After load
        registry.load_model("mlx-community/SmolLM2-135M-Instruct")
        spec = registry.get_current_spec()

        assert spec == mock_spec

    @patch("semantic.application.model_registry.get_extractor")
    def test_get_current_id_returns_model_id(self, mock_get_extractor, mock_loader):
        """get_current_id() returns loaded model's HuggingFace ID."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_loader.load_model.return_value = (mock_model, mock_tokenizer)

        mock_spec = ModelCacheSpec(
            n_layers=24, n_kv_heads=8, head_dim=128, block_tokens=16, layer_types=["global"] * 24
        )
        mock_extractor = Mock()
        mock_extractor.extract_spec.return_value = mock_spec
        mock_get_extractor.return_value = mock_extractor

        registry = ModelRegistry(model_loader=mock_loader)

        # Before load
        assert registry.get_current_id() is None

        # After load
        registry.load_model("mlx-community/SmolLM2-135M-Instruct")
        model_id = registry.get_current_id()

        assert model_id == "mlx-community/SmolLM2-135M-Instruct"

    @patch("semantic.application.model_registry.get_extractor")
    @patch("semantic.application.model_registry.gc.collect")
    def test_swap_models_lifecycle(self, mock_gc_collect, mock_get_extractor, mock_loader):
        """Loading a second model after unloading first works correctly."""
        # Setup mocks for first model
        mock_model_1 = MagicMock()
        mock_tokenizer_1 = MagicMock()
        mock_spec_1 = ModelCacheSpec(
            n_layers=24, n_kv_heads=8, head_dim=128, block_tokens=16, layer_types=["global"] * 24
        )

        # Setup mocks for second model
        mock_model_2 = MagicMock()
        mock_tokenizer_2 = MagicMock()
        mock_spec_2 = ModelCacheSpec(
            n_layers=32, n_kv_heads=16, head_dim=128, block_tokens=16, layer_types=["global"] * 32
        )

        # Mock loader to return different models
        mock_loader.load_model.side_effect = [
            (mock_model_1, mock_tokenizer_1),
            (mock_model_2, mock_tokenizer_2),
        ]

        # Mock extractor to return different specs
        mock_extractor = Mock()
        mock_extractor.extract_spec.side_effect = [mock_spec_1, mock_spec_2]
        mock_get_extractor.return_value = mock_extractor

        # Memory measurements
        mock_loader.get_active_memory.side_effect = [100 * 1024 * 1024, 0]

        registry = ModelRegistry(model_loader=mock_loader)

        # Load first model
        registry.load_model("mlx-community/SmolLM2-135M-Instruct")
        assert registry.get_current_id() == "mlx-community/SmolLM2-135M-Instruct"
        assert registry.get_current_spec() == mock_spec_1

        # Unload first model
        registry.unload_model()
        assert not registry.is_loaded()

        # Load second model
        registry.load_model("mlx-community/Qwen2.5-0.5B-Instruct-4bit")
        assert registry.get_current_id() == "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
        assert registry.get_current_spec() == mock_spec_2

        # Verify cleanup was called for first model via loader (CR-3 fix)
        mock_gc_collect.assert_called_once()
        mock_loader.clear_cache.assert_called_once()
