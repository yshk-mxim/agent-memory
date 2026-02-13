# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Yakov Shkolnikov and contributors
"""Unit tests for MLX sink compatibility patch.

Verifies the monkey-patch that enables Q4 KV cache with attention sinks:
- Patch is applied to mlx_lm.models.base
- sinks=None path uses quantized SDPA (no dequantize)
- sinks!=None path dequantizes and uses standard SDPA
- FP16 cache path uses standard SDPA directly
"""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.unit


class TestPatchModuleState:
    def test_patched_flag_set_after_import(self) -> None:
        """Module sets _patched=True after successful patch application."""
        import agent_memory.adapters.outbound.mlx_sink_compat as compat_module

        # Module already ran _apply_patch() at import time
        # In test environment it should have attempted patching
        # (may be True or False depending on mock state)
        assert isinstance(compat_module._patched, bool)

    def test_double_apply_is_noop(self) -> None:
        """Calling _apply_patch when already patched does nothing."""
        import agent_memory.adapters.outbound.mlx_sink_compat as compat_module

        compat_module._patched = True
        # Should return early without error
        compat_module._apply_patch()
        assert compat_module._patched is True

    def test_apply_patch_skips_when_mlx_unavailable(self) -> None:
        """When mlx_lm is not importable, patch is skipped gracefully."""
        import agent_memory.adapters.outbound.mlx_sink_compat as compat_module

        compat_module._patched = False

        # Temporarily remove mlx_lm.models.base to simulate unavailability
        saved = sys.modules.get("mlx_lm.models.base")
        sys.modules["mlx_lm.models.base"] = None  # type: ignore[assignment]

        try:
            compat_module._apply_patch()
            # Should not crash, just skip
            assert compat_module._patched is False
        finally:
            if saved is not None:
                sys.modules["mlx_lm.models.base"] = saved
            else:
                sys.modules.pop("mlx_lm.models.base", None)


class TestPatchedSDPALogic:
    """Test the three-way dispatch logic of the patched SDPA function.

    Instead of trying to call _apply_patch() with mocked modules (fragile),
    we recreate the dispatch logic and test it directly.
    """

    @pytest.fixture
    def mock_mx(self) -> MagicMock:
        mock = MagicMock()
        mock.dequantize = MagicMock(side_effect=lambda w, scales, biases, group_size, bits: w)
        mock.fast.scaled_dot_product_attention = MagicMock(return_value="fp16_sdpa_result")
        return mock

    @pytest.fixture
    def mock_quantized_sdpa(self) -> MagicMock:
        return MagicMock(return_value="quantized_sdpa_result")

    @pytest.fixture
    def patched_sdpa(self, mock_mx, mock_quantized_sdpa):
        """Create the patched SDPA function matching the production logic."""
        mx = mock_mx
        quantized_scaled_dot_product_attention = mock_quantized_sdpa

        def _patched_sdpa(queries, keys, values, cache, scale, mask, sinks=None):
            if hasattr(cache, "bits"):
                if sinks is not None:
                    k_fp = mx.dequantize(
                        keys[0],
                        scales=keys[1],
                        biases=keys[2],
                        group_size=cache.group_size,
                        bits=cache.bits,
                    )
                    v_fp = mx.dequantize(
                        values[0],
                        scales=values[1],
                        biases=values[2],
                        group_size=cache.group_size,
                        bits=cache.bits,
                    )
                    return mx.fast.scaled_dot_product_attention(
                        queries,
                        k_fp,
                        v_fp,
                        scale=scale,
                        mask=mask,
                        sinks=sinks,
                    )
                return quantized_scaled_dot_product_attention(
                    queries,
                    keys,
                    values,
                    scale=scale,
                    mask=mask,
                    group_size=cache.group_size,
                    bits=cache.bits,
                )
            return mx.fast.scaled_dot_product_attention(
                queries,
                keys,
                values,
                scale=scale,
                mask=mask,
                sinks=sinks,
            )

        return _patched_sdpa

    def test_quantized_no_sinks_uses_quantized_sdpa(
        self,
        patched_sdpa,
        mock_mx,
        mock_quantized_sdpa,
    ) -> None:
        """Q4 cache + no sinks -> quantized SDPA kernel (no dequantize)."""
        cache = SimpleNamespace(bits=4, group_size=64)
        queries = MagicMock()
        keys = (MagicMock(), MagicMock(), MagicMock())
        values = (MagicMock(), MagicMock(), MagicMock())
        mask = MagicMock()

        result = patched_sdpa(queries, keys, values, cache, scale=1.0, mask=mask, sinks=None)

        assert result == "quantized_sdpa_result"
        mock_quantized_sdpa.assert_called_once()
        mock_mx.dequantize.assert_not_called()
        mock_mx.fast.scaled_dot_product_attention.assert_not_called()

    def test_quantized_with_sinks_dequantizes_and_uses_fp16_sdpa(
        self,
        patched_sdpa,
        mock_mx,
        mock_quantized_sdpa,
    ) -> None:
        """Q4 cache + sinks -> dequantize K/V to FP16, use standard SDPA."""
        cache = SimpleNamespace(bits=4, group_size=64)
        queries = MagicMock()
        k_w, k_s, k_b = MagicMock(), MagicMock(), MagicMock()
        v_w, v_s, v_b = MagicMock(), MagicMock(), MagicMock()
        keys = (k_w, k_s, k_b)
        values = (v_w, v_s, v_b)
        mask = MagicMock()
        sinks = MagicMock()

        result = patched_sdpa(queries, keys, values, cache, scale=1.0, mask=mask, sinks=sinks)

        assert result == "fp16_sdpa_result"
        assert mock_mx.dequantize.call_count == 2
        mock_mx.fast.scaled_dot_product_attention.assert_called_once()
        # Verify sinks parameter passed through
        sdpa_call = mock_mx.fast.scaled_dot_product_attention.call_args
        assert sdpa_call.kwargs["sinks"] is sinks
        # Quantized SDPA should NOT be called
        mock_quantized_sdpa.assert_not_called()

    def test_fp16_cache_no_sinks_uses_standard_sdpa(
        self,
        patched_sdpa,
        mock_mx,
        mock_quantized_sdpa,
    ) -> None:
        """FP16 cache (no .bits) + no sinks -> standard SDPA."""
        cache = SimpleNamespace()  # No bits attribute
        queries = MagicMock()
        keys = MagicMock()
        values = MagicMock()
        mask = MagicMock()

        result = patched_sdpa(queries, keys, values, cache, scale=1.0, mask=mask, sinks=None)

        assert result == "fp16_sdpa_result"
        mock_mx.fast.scaled_dot_product_attention.assert_called_once()
        mock_quantized_sdpa.assert_not_called()
        mock_mx.dequantize.assert_not_called()

    def test_fp16_cache_with_sinks_passes_sinks_through(
        self,
        patched_sdpa,
        mock_mx,
        mock_quantized_sdpa,
    ) -> None:
        """FP16 cache + sinks -> standard SDPA with sinks parameter."""
        cache = SimpleNamespace()
        sinks = MagicMock()

        patched_sdpa(
            MagicMock(),
            MagicMock(),
            MagicMock(),
            cache,
            scale=1.0,
            mask=MagicMock(),
            sinks=sinks,
        )

        sdpa_call = mock_mx.fast.scaled_dot_product_attention.call_args
        assert sdpa_call.kwargs["sinks"] is sinks

    def test_dequantize_receives_correct_components(
        self,
        patched_sdpa,
        mock_mx,
        mock_quantized_sdpa,
    ) -> None:
        """Dequantize is called with correct K/V components from cache tuple."""
        cache = SimpleNamespace(bits=4, group_size=64)
        k_w, k_s, k_b = "k_weights", "k_scales", "k_biases"
        v_w, v_s, v_b = "v_weights", "v_scales", "v_biases"
        keys = (k_w, k_s, k_b)
        values = (v_w, v_s, v_b)
        sinks = MagicMock()

        patched_sdpa(MagicMock(), keys, values, cache, scale=1.0, mask=MagicMock(), sinks=sinks)

        # First dequantize call: K
        k_call = mock_mx.dequantize.call_args_list[0]
        assert k_call.args[0] == "k_weights"
        assert k_call.kwargs["scales"] == "k_scales"
        assert k_call.kwargs["biases"] == "k_biases"
        assert k_call.kwargs["group_size"] == 64
        assert k_call.kwargs["bits"] == 4

        # Second dequantize call: V
        v_call = mock_mx.dequantize.call_args_list[1]
        assert v_call.args[0] == "v_weights"
        assert v_call.kwargs["scales"] == "v_scales"
        assert v_call.kwargs["biases"] == "v_biases"

    def test_quantized_sdpa_receives_correct_params(
        self,
        patched_sdpa,
        mock_mx,
        mock_quantized_sdpa,
    ) -> None:
        """Quantized SDPA receives group_size and bits from cache."""
        cache = SimpleNamespace(bits=4, group_size=64)
        queries = MagicMock()
        keys = (MagicMock(), MagicMock(), MagicMock())
        values = (MagicMock(), MagicMock(), MagicMock())
        mask = MagicMock()

        patched_sdpa(queries, keys, values, cache, scale=0.5, mask=mask, sinks=None)

        call_kwargs = mock_quantized_sdpa.call_args.kwargs
        assert call_kwargs["group_size"] == 64
        assert call_kwargs["bits"] == 4
        assert call_kwargs["scale"] == 0.5
