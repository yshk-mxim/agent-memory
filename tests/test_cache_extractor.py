"""
Tests for MLXCacheExtractor

Tests cache extraction, reuse, and metadata functions.
Uses real KVCache objects but mocks generation.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
import mlx.core as mx
from mlx_lm.models.cache import KVCache

from src.mlx_cache_extractor import MLXCacheExtractor


@pytest.fixture
def mock_model():
    """Mock MLX model."""
    model = Mock()
    model.layers = [Mock() for _ in range(4)]  # 4 layer model
    return model


@pytest.fixture
def mock_tokenizer():
    """Mock tokenizer."""
    tokenizer = Mock()
    tokenizer.eos_token_ids = [1, 2]
    return tokenizer


@pytest.fixture
def extractor(mock_model, mock_tokenizer):
    """Create MLXCacheExtractor with mocked model."""
    return MLXCacheExtractor(mock_model, mock_tokenizer)


@pytest.fixture
def sample_cache():
    """Create a real KVCache with sample data."""
    cache = []
    for _ in range(4):  # 4 layers
        layer_cache = KVCache()
        # Simulate some cached tokens
        # Shape: (B=1, n_kv_heads=8, seq_len=10, head_dim=64)
        keys = mx.zeros((1, 8, 10, 64))
        values = mx.zeros((1, 8, 10, 64))
        layer_cache.keys = keys
        layer_cache.values = values
        layer_cache.offset = 10
        cache.append(layer_cache)
    return cache


def test_init(extractor, mock_model, mock_tokenizer):
    """Test MLXCacheExtractor initialization."""
    assert extractor.model == mock_model
    assert extractor.tokenizer == mock_tokenizer


@patch('src.mlx_cache_extractor.stream_generate')
@patch('src.mlx_cache_extractor.make_prompt_cache')
def test_generate_with_cache_returns_text_and_cache(
    mock_make_cache,
    mock_stream_gen,
    extractor,
    sample_cache
):
    """Test generate_with_cache returns both text and cache."""
    # Setup mocks
    mock_make_cache.return_value = sample_cache

    # Mock stream_generate to yield responses
    mock_response1 = Mock()
    mock_response1.text = "Hello "
    mock_response2 = Mock()
    mock_response2.text = "world"

    mock_stream_gen.return_value = [mock_response1, mock_response2]

    # Generate
    text, cache = extractor.generate_with_cache(
        prompt="Test prompt",
        max_tokens=10,
        temperature=0.5
    )

    # Assertions
    assert text == "Hello world"
    assert cache == sample_cache
    mock_make_cache.assert_called_once_with(extractor.model)
    mock_stream_gen.assert_called_once()


@patch('src.mlx_cache_extractor.stream_generate')
def test_generate_with_existing_cache(
    mock_stream_gen,
    extractor,
    sample_cache
):
    """Test generate_with_cache reuses existing cache."""
    # Mock stream_generate
    mock_response = Mock()
    mock_response.text = "response"
    mock_stream_gen.return_value = [mock_response]

    # Generate with existing cache
    text, cache = extractor.generate_with_cache(
        prompt="Test",
        existing_cache=sample_cache,
        max_tokens=5
    )

    # Cache should be reused, not created
    assert cache == sample_cache
    assert text == "response"


@patch('src.mlx_cache_extractor.stream_generate')
@patch('src.mlx_cache_extractor.make_prompt_cache')
def test_process_prompt_creates_cache(
    mock_make_cache,
    mock_stream_gen,
    extractor,
    sample_cache
):
    """Test process_prompt creates cache without generating."""
    # Setup mocks
    mock_make_cache.return_value = sample_cache
    mock_stream_gen.return_value = []  # No tokens generated

    # Process prompt
    cache = extractor.process_prompt("System prompt")

    # Assertions
    assert cache == sample_cache
    # Should call stream_generate with max_tokens=0
    call_args = mock_stream_gen.call_args
    assert call_args[1]['max_tokens'] == 0


def test_get_cache_info_structure(sample_cache):
    """Test get_cache_info returns correct structure."""
    extractor = MLXCacheExtractor(Mock(), Mock())

    info = extractor.get_cache_info(sample_cache)

    assert 'num_layers' in info
    assert 'total_tokens' in info
    assert 'memory_bytes' in info

    assert info['num_layers'] == 4
    assert info['total_tokens'] == 10  # From offset


def test_get_cache_info_empty_cache():
    """Test get_cache_info with empty cache."""
    extractor = MLXCacheExtractor(Mock(), Mock())

    info = extractor.get_cache_info([])

    assert info['num_layers'] == 0
    assert info['total_tokens'] == 0
    assert info['memory_bytes'] == 0


def test_get_cache_memory_bytes(sample_cache):
    """Test get_cache_memory_bytes calculates memory."""
    extractor = MLXCacheExtractor(Mock(), Mock())

    memory = extractor.get_cache_memory_bytes(sample_cache)

    # Each layer: 2 arrays (keys, values)
    # Shape: (1, 8, 10, 64)
    # Total elements per array: 1 * 8 * 10 * 64 = 5120
    # Default dtype is float32 (4 bytes)
    # Per array: 5120 * 4 = 20480 bytes
    # Per layer: 2 * 20480 = 40960 bytes
    # 4 layers: 4 * 40960 = 163840 bytes

    assert memory > 0
    assert isinstance(memory, int)


def test_get_cache_memory_bytes_empty():
    """Test get_cache_memory_bytes with empty cache."""
    extractor = MLXCacheExtractor(Mock(), Mock())

    memory = extractor.get_cache_memory_bytes([])

    assert memory == 0
