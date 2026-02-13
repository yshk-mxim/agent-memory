# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Yakov Shkolnikov and contributors
"""Unit tests for fused attention guard and dispatch logic.

Tests environment variable guards, idempotency, import failure handling,
and dispatch fallbacks WITHOUT requiring real MLX/Metal hardware.
"""

import importlib
import sys
from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _reset_module_state(monkeypatch):
    """Reset module-level globals before each test.

    The fused attention module caches _patched and _metal_decode_disabled
    at module level. We need a fresh import for each test.
    """
    # Remove cached module so next import is fresh
    mod_name = "agent_memory.adapters.outbound.mlx_fused_attention"
    saved = sys.modules.pop(mod_name, None)
    yield
    # Restore or remove
    if saved is not None:
        sys.modules[mod_name] = saved
    else:
        sys.modules.pop(mod_name, None)


def _import_fused():
    """Fresh import of the fused attention module."""
    return importlib.import_module("agent_memory.adapters.outbound.mlx_fused_attention")


def _setup_mlx_mocks(monkeypatch):
    """Create properly-chained mock MLX modules.

    `import mlx_lm.models.base as base_module` resolves via attribute chain:
    sys.modules['mlx_lm'].models.base, NOT sys.modules['mlx_lm.models.base'].
    So we must wire the chain: mock_mlx_lm.models = mock_models, mock_models.base = mock_base.
    """
    mock_mx = MagicMock()
    mock_mx.eval = MagicMock(name="sync_eval")
    mock_mx.async_eval = MagicMock(name="async_eval")

    mock_base = MagicMock()
    original_fn = MagicMock(name="original_q4_sdpa")
    mock_base.quantized_scaled_dot_product_attention = original_fn

    mock_gemma = MagicMock()
    original_clip = MagicMock(name="original_clip_residual")
    mock_gemma.clip_residual = original_clip

    # Build chain: mlx_lm -> models -> base
    mock_models = MagicMock()
    mock_models.base = mock_base
    mock_models.gemma3_text = mock_gemma

    mock_mlx = MagicMock()
    mock_mlx.core = mock_mx

    mock_mlx_lm = MagicMock()
    mock_mlx_lm.models = mock_models

    # Install in sys.modules (both chain AND direct keys for completeness)
    monkeypatch.setitem(sys.modules, "mlx", mock_mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", mock_mx)
    monkeypatch.setitem(sys.modules, "mlx_lm", mock_mlx_lm)
    monkeypatch.setitem(sys.modules, "mlx_lm.models", mock_models)
    monkeypatch.setitem(sys.modules, "mlx_lm.models.base", mock_base)
    monkeypatch.setitem(sys.modules, "mlx_lm.models.gemma3_text", mock_gemma)

    return {
        "mx": mock_mx,
        "base": mock_base,
        "gemma": mock_gemma,
        "original_sdpa": original_fn,
        "original_clip": original_clip,
    }


# ── Guard tests ─────────────────────────────────────────────────────


class TestApplyFusedAttentionPatch:
    def test_disabled_by_env_var(self, monkeypatch) -> None:
        """SEMANTIC_DISABLE_FUSED_ATTN=1 → returns False, sets _patched."""
        monkeypatch.setenv("SEMANTIC_DISABLE_FUSED_ATTN", "1")
        mod = _import_fused()
        result = mod.apply_fused_attention_patch()
        assert result is False
        # Second call returns True (idempotent, _patched=True)
        assert mod.apply_fused_attention_patch() is True

    def test_import_failure_returns_false(self, monkeypatch) -> None:
        """If mlx/mlx_lm can't be imported, returns False."""
        monkeypatch.delenv("SEMANTIC_DISABLE_FUSED_ATTN", raising=False)

        # Remove any cached mlx modules so the import inside the function fails
        for key in list(sys.modules):
            if key.startswith("mlx"):
                monkeypatch.delitem(sys.modules, key, raising=False)

        mod = _import_fused()

        # Patch the import to fail
        original_import = (
            __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__
        )

        def failing_import(name, *args, **kwargs):
            if name in ("mlx.core", "mlx"):
                raise ImportError("no mlx")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=failing_import):
            result = mod.apply_fused_attention_patch()

        assert result is False

    def test_idempotent_after_success(self, monkeypatch) -> None:
        """Second call returns True immediately without re-patching."""
        monkeypatch.delenv("SEMANTIC_DISABLE_FUSED_ATTN", raising=False)
        _setup_mlx_mocks(monkeypatch)

        mod = _import_fused()
        first = mod.apply_fused_attention_patch()
        assert first is True

        second = mod.apply_fused_attention_patch()
        assert second is True

    def test_patches_sdpa_on_base_module(self, monkeypatch) -> None:
        """After patching, base_module.quantized_scaled_dot_product_attention is replaced."""
        monkeypatch.delenv("SEMANTIC_DISABLE_FUSED_ATTN", raising=False)
        mocks = _setup_mlx_mocks(monkeypatch)

        mod = _import_fused()
        mod.apply_fused_attention_patch()

        # The SDPA should now be replaced (attribute chain resolves correctly)
        assert mocks["base"].quantized_scaled_dot_product_attention is not mocks["original_sdpa"]

    def test_async_eval_replaced_with_eval(self, monkeypatch) -> None:
        """After patching, mx.async_eval should point to mx.eval."""
        monkeypatch.delenv("SEMANTIC_DISABLE_FUSED_ATTN", raising=False)
        mocks = _setup_mlx_mocks(monkeypatch)

        mod = _import_fused()
        mod.apply_fused_attention_patch()

        # async_eval should now be the same object as eval
        assert mocks["mx"].async_eval is mocks["mx"].eval

    def test_clip_residual_patched(self, monkeypatch) -> None:
        """After patching, gemma3_text.clip_residual should be replaced."""
        monkeypatch.delenv("SEMANTIC_DISABLE_FUSED_ATTN", raising=False)
        mocks = _setup_mlx_mocks(monkeypatch)

        mod = _import_fused()
        mod.apply_fused_attention_patch()

        assert mocks["gemma"].clip_residual is not mocks["original_clip"]

    def test_clip_residual_import_failure_handled(self, monkeypatch) -> None:
        """If gemma3_text can't be imported, patching still succeeds."""
        monkeypatch.delenv("SEMANTIC_DISABLE_FUSED_ATTN", raising=False)
        mocks = _setup_mlx_mocks(monkeypatch)

        # Make gemma3_text import raise ImportError
        # The function does: import mlx_lm.models.gemma3_text as gemma3_text_module
        # which resolves to: sys.modules['mlx_lm'].models.gemma3_text
        # We set it to a property that raises, but simpler: delete the attribute
        # and let the try/except in the source handle it.
        del mocks["base"]  # not needed for this test
        mock_models = sys.modules["mlx_lm"].models
        type(mock_models).gemma3_text = property(
            lambda self: (_ for _ in ()).throw(ImportError("no gemma3"))
        )

        mod = _import_fused()
        result = mod.apply_fused_attention_patch()
        assert result is True

        # Restore to avoid affecting other tests
        del type(mock_models).gemma3_text

    def test_fallback_to_original_for_non_q4(self, monkeypatch) -> None:
        """Non-Q4 group_size or bits falls through to original SDPA."""
        monkeypatch.delenv("SEMANTIC_DISABLE_FUSED_ATTN", raising=False)
        mocks = _setup_mlx_mocks(monkeypatch)

        mod = _import_fused()
        mod.apply_fused_attention_patch()

        # Get the patched function from the attribute chain
        patched_fn = mocks["base"].quantized_scaled_dot_product_attention
        assert patched_fn is not mocks["original_sdpa"]

        # Create mock tensor inputs
        queries = MagicMock()
        queries.shape = (1, 16, 1, 256)
        q_keys = (MagicMock(), MagicMock(), MagicMock())
        q_keys[0].shape = (1, 8, 100, 32)
        q_values = (MagicMock(), MagicMock(), MagicMock())

        # Call with non-standard group_size — should delegate to original
        patched_fn(
            queries,
            q_keys,
            q_values,
            scale=0.1,
            mask=None,
            group_size=128,
            bits=4,
        )
        mocks["original_sdpa"].assert_called_once()


# ── Metal decode control ────────────────────────────────────────────


class TestMetalDecodeControl:
    def test_metal_decode_disabled_by_default(self, monkeypatch) -> None:
        """Without SEMANTIC_ENABLE_METAL_DECODE, _metal_decode_disabled is True."""
        monkeypatch.delenv("SEMANTIC_ENABLE_METAL_DECODE", raising=False)
        mod = _import_fused()
        assert mod._metal_decode_disabled is True

    def test_metal_decode_enabled_by_env(self, monkeypatch) -> None:
        """SEMANTIC_ENABLE_METAL_DECODE=1 → _metal_decode_disabled is False."""
        monkeypatch.setenv("SEMANTIC_ENABLE_METAL_DECODE", "1")
        mod = _import_fused()
        assert mod._metal_decode_disabled is False
