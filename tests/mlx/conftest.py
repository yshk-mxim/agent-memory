"""Real MLX test configuration.

NO MOCKING. This conftest loads actual MLX models and runs real GPU operations.
Tests in this directory require Apple Silicon and take 30-120 seconds each.

Run with: pytest tests/mlx/ -v -x --timeout=120
MUST use dangerouslyDisableSandbox: true for Metal GPU access.
"""

import tempfile
from pathlib import Path

import pytest

# Model used for real integration tests (small, fast to load)
TEST_MODEL_ID = "mlx-community/SmolLM2-135M-Instruct"


@pytest.fixture(scope="session")
def real_model_and_tokenizer():
    """Load real SmolLM2-135M model. Session-scoped to avoid reloading."""
    from mlx_lm import load

    model, tokenizer = load(TEST_MODEL_ID)
    return model, tokenizer


@pytest.fixture(scope="session")
def real_spec(real_model_and_tokenizer):
    """Extract real ModelCacheSpec from loaded model."""
    from agent_memory.adapters.outbound.mlx_spec_extractor import get_extractor

    model, _ = real_model_and_tokenizer
    return get_extractor().extract_spec(model)


@pytest.fixture
def cache_dir():
    """Temporary directory for cache persistence tests."""
    with tempfile.TemporaryDirectory(prefix="mlx_test_cache_") as tmpdir:
        yield Path(tmpdir)
