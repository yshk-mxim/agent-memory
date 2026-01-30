"""Integration test configuration.

Mocks MLX modules at module level to prevent C-level crashes from Metal GPU
initialization during test collection. Provides a FakeBatchGenerator that
implements the insert/next protocol so the engine pipeline works end-to-end
without real MLX inference.
"""

import os
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

# =============================================================================
# Module-level: Mock MLX modules before any test imports api_server.
# The real MLX library crashes at the C level during Metal GPU initialization
# in the test environment. These mocks allow api_server.py to be imported safely.
# =============================================================================

# --- mlx.core with sensible return values ---
_mock_mx = MagicMock()
_mock_mx.get_active_memory.return_value = 100 * 1024 * 1024
_mock_mx.get_cache_memory.return_value = 0
_mock_mx.get_peak_memory.return_value = 200 * 1024 * 1024
_mock_mx.eval = MagicMock()
_mock_mx.clear_cache = MagicMock()
_mock_mx.quantize.return_value = [MagicMock(), MagicMock(), MagicMock()]

_mock_mlx = MagicMock()
_mock_mlx.core = _mock_mx  # Wire parent.core → child mock
sys.modules["mlx"] = _mock_mlx
sys.modules["mlx.core"] = _mock_mx

# --- Mock model with attributes for spec extraction ---
_mock_model = MagicMock()
_mock_model.args = SimpleNamespace(
    num_hidden_layers=12,
    num_key_value_heads=4,
    num_attention_heads=12,
    hidden_size=768,
    model_type="llama",
    sliding_window=None,
)

# --- FakeDetokenizer: real class to avoid MagicMock spec issues in Python 3.12+ ---
class _FakeDetokenizer:
    """Minimal detokenizer for testing. batch_engine.py does type(tok.detokenizer)(tok)."""

    def __init__(self, tokenizer=None):
        self._tokens: list[int] = []

    def add_token(self, token: int) -> None:
        self._tokens.append(token)

    @property
    def text(self) -> str:
        return "Hello world"


# --- Mock tokenizer ---
_mock_tokenizer = MagicMock()
_mock_tokenizer.model_max_length = 32768
_mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]
_mock_tokenizer.decode.return_value = "Hello world"
_mock_tokenizer.eos_token_id = 2
_mock_tokenizer.detokenizer = _FakeDetokenizer()

# --- Configure mlx_lm.load() to return proper (model, tokenizer) tuple ---
_mock_mlx_lm = MagicMock()
_mock_mlx_lm.load.return_value = (_mock_model, _mock_tokenizer)
_mock_mlx_lm_models = MagicMock()
_mock_mlx_lm_models_cache = MagicMock()
_mock_mlx_lm_models.cache = _mock_mlx_lm_models_cache
_mock_mlx_lm.models = _mock_mlx_lm_models
sys.modules["mlx_lm"] = _mock_mlx_lm
sys.modules["mlx_lm.models"] = _mock_mlx_lm_models
sys.modules["mlx_lm.models.cache"] = _mock_mlx_lm_models_cache


# =============================================================================
# FakeBatchGenerator: test-compatible replacement for mlx_lm.server.BatchGenerator
# =============================================================================

class _FakeResponse:
    """Mimics the response object from BatchGenerator.next()."""

    def __init__(self, uid, token, finish_reason, prompt_cache=None):
        self.uid = uid
        self.token = token
        self.finish_reason = finish_reason
        self.prompt_cache = prompt_cache or []


class FakeBatchGenerator:
    """Test-compatible replacement for mlx_lm.server.BatchGenerator.

    Implements the insert/next protocol that BlockPoolBatchEngine expects.
    Generates a few fake tokens then returns finish_reason="stop".
    """

    def __init__(self, model=None, stop_tokens=None, **kwargs):
        self._sequences: dict[str, dict] = {}
        self._next_uid = 0

    def insert(self, prompts=None, max_tokens=None, caches=None, samplers=None):
        uids = []
        n = len(prompts) if prompts else 0
        for i in range(n):
            uid = f"fuid_{self._next_uid}"
            self._next_uid += 1
            max_t = max_tokens[i] if max_tokens else 256
            self._sequences[uid] = {"max_tokens": max_t, "generated": 0}
            uids.append(uid)
        return uids

    def next(self):
        if not self._sequences:
            return []

        responses = []
        done_uids = []
        for uid, seq in list(self._sequences.items()):
            seq["generated"] += 1
            if seq["generated"] >= min(3, seq["max_tokens"]):
                # Done: return finish_reason and fake prompt_cache (12 layers)
                fake_caches = [MagicMock() for _ in range(12)]
                for fc in fake_caches:
                    fc.state = (MagicMock(), MagicMock())
                    fc.offset = 5
                responses.append(_FakeResponse(
                    uid=uid,
                    token=42,
                    finish_reason="stop",
                    prompt_cache=fake_caches,
                ))
                done_uids.append(uid)
            else:
                responses.append(_FakeResponse(
                    uid=uid,
                    token=42,
                    finish_reason=None,
                ))

        for uid in done_uids:
            del self._sequences[uid]

        return responses


# Wire FakeBatchGenerator into the mock module hierarchy
_mock_server = MagicMock()
_mock_server.BatchGenerator = FakeBatchGenerator
_mock_sample_utils = MagicMock()
_mock_mlx_lm.server = _mock_server
_mock_mlx_lm.sample_utils = _mock_sample_utils
sys.modules["mlx_lm.server"] = _mock_server
sys.modules["mlx_lm.sample_utils"] = _mock_sample_utils


# =============================================================================
# Pytest fixtures
# =============================================================================

@pytest.fixture(autouse=True, scope="session")
def _reinstall_mlx_mocks():
    """Re-install MLX mocks that may be clobbered by unit test module-level code.

    Session-scoped so it runs before module-scoped fixtures like model_and_tokenizer.
    """
    sys.modules["mlx"] = _mock_mlx
    sys.modules["mlx.core"] = _mock_mx
    sys.modules["mlx_lm"] = _mock_mlx_lm
    sys.modules["mlx_lm.models"] = _mock_mlx_lm_models
    sys.modules["mlx_lm.models.cache"] = _mock_mlx_lm_models_cache
    sys.modules["mlx_lm.server"] = _mock_server
    sys.modules["mlx_lm.sample_utils"] = _mock_sample_utils
    yield


@pytest.fixture(autouse=True)
def _patch_for_integration(monkeypatch):
    """Patches applied to every integration test for mock-MLX compatibility.

    - Resets settings singleton so each test gets fresh config
    - Patches _extract_cache to skip deep MLX tensor operations
    - Resets spec extractor singleton
    """
    # Re-install configured mocks into sys.modules — unit test files may
    # clobber these with unconfigured MagicMock() at module level.
    # Direct assignment (not monkeypatch) because this must take effect before
    # module-scoped fixtures run, and the values are idempotent.
    sys.modules["mlx"] = _mock_mlx
    sys.modules["mlx.core"] = _mock_mx
    sys.modules["mlx_lm"] = _mock_mlx_lm
    sys.modules["mlx_lm.models"] = _mock_mlx_lm_models
    sys.modules["mlx_lm.models.cache"] = _mock_mlx_lm_models_cache
    sys.modules["mlx_lm.server"] = _mock_server
    sys.modules["mlx_lm.sample_utils"] = _mock_sample_utils

    # Fix stale module-level references: modules that did `from mlx_lm import load`
    # at import time may hold a reference to an unconfigured MagicMock's .load if
    # unit test files clobbered sys.modules["mlx_lm"] before these modules loaded.
    _stale_load_modules = [
        "semantic.entrypoints.api_server",
        "semantic.adapters.outbound.mlx_model_loader",
    ]
    for _mod_name in _stale_load_modules:
        _mod = sys.modules.get(_mod_name)
        if _mod is not None and hasattr(_mod, "load"):
            monkeypatch.setattr(_mod, "load", _mock_mlx_lm.load)

    # Point cache_dir to a writable tmp directory (sandbox blocks ~/.semantic)
    test_cache_dir = tempfile.mkdtemp(prefix="semantic_test_")
    monkeypatch.setenv("SEMANTIC_AGENT_CACHE_DIR", test_cache_dir)

    # Reset settings singleton so each test gets a fresh config load
    from semantic.adapters.config import settings as settings_module
    monkeypatch.setattr(settings_module, "_settings", None)

    # Reset spec extractor singleton
    from semantic.adapters.outbound import mlx_spec_extractor
    monkeypatch.setattr(mlx_spec_extractor, "_extractor", None)

    # Patch _extract_cache to skip MLX tensor operations
    from semantic.application.batch_engine import BlockPoolBatchEngine
    from semantic.domain.entities import AgentBlocks
    from semantic.domain.errors import GenerationError

    def _fake_extract_cache(self, uid, cache=None, token_sequence=None):
        if uid not in self._active_requests:
            raise GenerationError(f"UID {uid} not found")
        agent_id, _, _, _ = self._active_requests[uid]
        tok_seq = token_sequence or [1, 2, 3, 4, 5]
        n_tokens = len(tok_seq)
        # Allocate real blocks from the pool so free() works correctly
        n_layers = self._spec.n_layers
        blocks: dict[int, list] = {}
        for layer_id in range(n_layers):
            allocated = self._pool.allocate(1, layer_id, agent_id)
            for b in allocated:
                b.token_count = n_tokens
                b.layer_data = {"fake": True}  # Non-None so load() sees data
            blocks[layer_id] = allocated
        return AgentBlocks(
            agent_id=agent_id,
            blocks=blocks,
            total_tokens=n_tokens,
            token_sequence=tok_seq,
        )

    monkeypatch.setattr(BlockPoolBatchEngine, "_extract_cache", _fake_extract_cache)

    # Patch SafetensorsCacheAdapter.save/load — fake blocks have opaque layer_data
    # that safetensors can't serialize. Mock the adapter to store in memory instead.
    from semantic.adapters.outbound.safetensors_cache_adapter import SafetensorsCacheAdapter
    _disk_cache: dict[str, tuple] = {}

    def _fake_adapter_save(self, agent_id, blocks, metadata):
        import pathlib
        _disk_cache[agent_id] = (blocks, metadata)
        return pathlib.Path(test_cache_dir) / f"{agent_id}.safetensors"

    def _fake_adapter_load(self, path):
        import pathlib
        agent_id = pathlib.Path(path).stem
        if agent_id in _disk_cache:
            blocks, metadata = _disk_cache[agent_id]
            return blocks.blocks if hasattr(blocks, "blocks") else {}, metadata
        return {}, {}

    monkeypatch.setattr(SafetensorsCacheAdapter, "save", _fake_adapter_save)
    monkeypatch.setattr(SafetensorsCacheAdapter, "load", _fake_adapter_load)
