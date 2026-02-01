"""Pytest configuration and shared fixtures.

This module defines:
- Test markers (unit, integration, smoke, e2e)
- Shared fixtures for fake port implementations
- Platform-specific skip conditions
"""

import platform
import sys
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Constants derived from real model specs (config/models/*.toml)
# ---------------------------------------------------------------------------

# Gemma 3 12B IT 4-bit (primary production model)
GEMMA3_N_LAYERS = 48
GEMMA3_N_KV_HEADS = 4  # from config.json: num_key_value_heads
GEMMA3_HEAD_DIM = 256  # hidden_size(3072) / num_attention_heads(12)
GEMMA3_BLOCK_TOKENS = 256  # universal block size
GEMMA3_GLOBAL_LAYERS = 8  # first 8 layers use global attention
GEMMA3_SLIDING_WINDOW = 1024
GEMMA3_LAYER_TYPES = ["global"] * GEMMA3_GLOBAL_LAYERS + ["sliding_window"] * (
    GEMMA3_N_LAYERS - GEMMA3_GLOBAL_LAYERS
)


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "unit: Fast unit tests with mocked boundaries (no MLX dependencies)",
    )
    config.addinivalue_line(
        "markers",
        "integration: Tests with real MLX and disk I/O (Apple Silicon only)",
    )
    config.addinivalue_line(
        "markers",
        "smoke: Basic server lifecycle tests",
    )
    config.addinivalue_line(
        "markers",
        "e2e: Full-stack multi-agent tests (slow, Apple Silicon only)",
    )


# Platform detection for conditional test skipping
def is_apple_silicon() -> bool:
    """Check if running on Apple Silicon (M1/M2/M3/M4)."""
    if sys.platform != "darwin":
        return False
    try:
        return platform.machine() == "arm64"
    except Exception:
        return False


# Fixtures for fake port implementations (used in unit tests)


@pytest.fixture
def fake_model_backend() -> MagicMock:
    """Fake ModelBackendPort for unit testing.

    Returns a mock that implements the ModelBackendPort protocol
    without requiring MLX or actual model loading.
    """
    mock = MagicMock()
    mock.generate.return_value = MagicMock(
        text="Generated text",
        tokens=[1, 2, 3, 4, 5],
        cache=[MagicMock() for _ in range(GEMMA3_N_LAYERS)],
    )
    return mock


@pytest.fixture
def fake_cache_persistence() -> MagicMock:
    """Fake CachePersistencePort for unit testing.

    Returns a mock that implements the CachePersistencePort protocol
    without requiring disk I/O or safetensors.
    """
    mock = MagicMock()
    mock.save.return_value = None
    mock.load.return_value = [MagicMock() for _ in range(GEMMA3_N_LAYERS)]
    mock.exists.return_value = False
    return mock


@pytest.fixture
def fake_tokenizer() -> MagicMock:
    """Fake TokenizerPort for unit testing.

    Returns a mock that implements the TokenizerPort protocol
    without requiring transformers or HuggingFace.
    """
    mock = MagicMock()
    mock.encode.return_value = [1, 2, 3, 4, 5]
    mock.decode.return_value = "decoded text"
    mock.eos_token_id = 2
    return mock


@pytest.fixture
def fake_model_cache_spec() -> MagicMock:
    """Fake ModelCacheSpec for unit testing.

    Returns a minimal ModelCacheSpec for Gemma 3 12B hybrid architecture.
    """
    spec = MagicMock()
    spec.n_layers = GEMMA3_N_LAYERS
    spec.n_kv_heads = GEMMA3_N_KV_HEADS
    spec.head_dim = GEMMA3_HEAD_DIM
    spec.block_tokens = GEMMA3_BLOCK_TOKENS
    spec.layer_types = GEMMA3_LAYER_TYPES
    spec.sliding_window_size = GEMMA3_SLIDING_WINDOW
    return spec


# Skip conditions for platform-specific tests


skip_if_not_apple_silicon = pytest.mark.skipif(
    not is_apple_silicon(),
    reason="Requires Apple Silicon (M1/M2/M3/M4) for MLX Metal GPU support",
)
