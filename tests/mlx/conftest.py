# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Yakov Shkolnikov and contributors
"""Real MLX test configuration.

NO MOCKING. This conftest loads actual MLX models and runs real GPU operations.
Tests in this directory require Apple Silicon and take 30-120 seconds each.

Run with: pytest tests/mlx/ -v -x --timeout=120
MUST use dangerouslyDisableSandbox: true for Metal GPU access.
"""

import contextlib
import importlib
import sys
import tempfile
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# CRITICAL: Remove MagicMock MLX modules BEFORE test file collection.
#
# Multiple unit/integration test files install MagicMock into sys.modules at
# module level (e.g. `sys.modules["mlx.core"] = MagicMock()`). These run
# during pytest collection and poison the module cache. When mlx test files
# do `import mlx.core as mx` at module level, they capture the mock.
#
# This code runs when conftest.py is imported (during collection), BEFORE
# test module imports, so it restores real modules in time.
# ---------------------------------------------------------------------------
_MOCK_KEYS = [
    "mlx",
    "mlx.core",
    "mlx.utils",
    "mlx_lm",
    "mlx_lm.models",
    "mlx_lm.models.cache",
    "mlx_lm.server",
    "mlx_lm.sample_utils",
]

for _key in _MOCK_KEYS:
    _mod = sys.modules.get(_key)
    if _mod is not None and type(_mod).__name__ == "MagicMock":
        del sys.modules[_key]

# Force re-import of real C extensions and all MLX submodules that the
# integration conftest mocks. We need them all in sys.modules so they can
# be saved to _SAVED_REAL_MODULES and restored during test execution.
for _key in _MOCK_KEYS:
    with contextlib.suppress(ImportError):
        importlib.import_module(_key)

# ---------------------------------------------------------------------------
# Save references to real modules so they survive GC when the integration
# conftest's _reinstall_mlx_mocks replaces them with MagicMock in sys.modules.
# Without these saved refs, the C extension modules get garbage collected and
# re-importing them from scratch causes SIGABRT.
# ---------------------------------------------------------------------------
_SAVED_REAL_MODULES: dict[str, object] = {}
for _key in _MOCK_KEYS:
    _mod = sys.modules.get(_key)
    if _mod is not None and type(_mod).__name__ != "MagicMock":
        _SAVED_REAL_MODULES[_key] = _mod

# ---------------------------------------------------------------------------
# Also remove agent_memory modules that may have captured mock mlx references
# during unit test file collection. When test_admin_api.py or test_batch_engine.py
# do `sys.modules["mlx.core"] = MagicMock()` then import agent_memory modules,
# those modules capture mock `mx` references at module level. Removing them
# from sys.modules forces re-import with real mlx when mlx test files import them.
# ---------------------------------------------------------------------------
_am_keys = [k for k in list(sys.modules) if k.startswith("agent_memory")]
for _key in _am_keys:
    del sys.modules[_key]

# Model used for real integration tests (small, fast to load)
TEST_MODEL_ID = "mlx-community/SmolLM2-135M-Instruct"


@pytest.fixture(scope="session", autouse=True)
def _restore_real_mlx_for_gpu_tests():
    """Restore real MLX modules in sys.modules for GPU test execution.

    The integration conftest's _reinstall_mlx_mocks session fixture installs
    MagicMock MLX modules into sys.modules. That fixture is set up during the
    first integration test and persists for the entire session. This fixture
    undoes that installation by restoring the saved real module references,
    so imports inside mlx test fixtures (like `from mlx_lm import load`)
    get real modules, not mocks.

    This runs lazily when the first mlx test executes — after all integration
    tests are done but before any GPU test needs real MLX.
    """
    # Restore real modules from saved references (avoids re-importing C extensions
    # which can SIGABRT if the originals were GC'd).
    for key, real_mod in _SAVED_REAL_MODULES.items():
        sys.modules[key] = real_mod

    # Reload mlx_lm.generate to undo stale _make_cache patch from unit test collection
    if "mlx_lm.generate" in sys.modules:
        importlib.reload(sys.modules["mlx_lm.generate"])
    # Remove agent_memory modules so they get freshly imported with real mlx refs
    for _am_key in [k for k in list(sys.modules) if k.startswith("agent_memory")]:
        del sys.modules[_am_key]

    yield


@pytest.fixture(autouse=True, scope="module")
def _ensure_real_mlx_refs(request):
    """Replace MagicMock mlx references in test modules with real modules.

    During pytest collection, unit test files install MagicMock into sys.modules
    at module level. Despite cleanup in this conftest's module-level code,
    sys.modules can be re-polluted during the collection of later test files
    (via import chains). This means module-level ``import mlx.core as mx`` in
    mlx test files may capture a MagicMock instead of the real C extension.

    This fixture runs at test EXECUTION time (after the session fixture has
    restored real modules) and patches the test module's globals so that ``mx``,
    ``generate``, ``load``, and ``make_sampler`` point to real objects.

    Module-scoped so it runs BEFORE module-scoped fixtures in test files
    (e.g. model_and_tokenizer) that depend on these references being real.
    """
    import mlx.core as real_mx

    mod = request.module
    _saved: dict[str, object] = {}

    # Replace mx if it's a MagicMock
    if hasattr(mod, "mx") and type(mod.mx).__name__ == "MagicMock":
        _saved["mx"] = mod.mx
        mod.mx = real_mx

    # Replace mlx_lm imports (generate, load, make_sampler) if they're mocks
    try:
        from mlx_lm import generate as real_generate
        from mlx_lm import load as real_load
        from mlx_lm.sample_utils import make_sampler as real_sampler

        for name, real_obj in [
            ("generate", real_generate),
            ("load", real_load),
            ("make_sampler", real_sampler),
        ]:
            if hasattr(mod, name) and type(getattr(mod, name)).__name__ == "MagicMock":
                _saved[name] = getattr(mod, name)
                setattr(mod, name, real_obj)
    except ImportError:
        pass

    yield

    # Restore originals after all tests in the module complete
    for name, orig in _saved.items():
        setattr(mod, name, orig)


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
