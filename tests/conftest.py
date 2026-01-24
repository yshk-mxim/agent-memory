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
        cache=[MagicMock() for _ in range(48)],  # 48 layers
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
    mock.load.return_value = [MagicMock() for _ in range(48)]  # 48 layers
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
    spec.n_layers = 48
    spec.n_kv_heads = 8
    spec.head_dim = 256
    spec.block_tokens = 256
    spec.layer_types = ["global"] * 8 + ["sliding_window"] * 40
    spec.sliding_window_size = 1024
    return spec


# Skip conditions for platform-specific tests


skip_if_not_apple_silicon = pytest.mark.skipif(
    not is_apple_silicon(),
    reason="Requires Apple Silicon (M1/M2/M3/M4) for MLX Metal GPU support",
)
