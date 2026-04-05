# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Yakov Shkolnikov and contributors
"""Unit tests for TRT cache adapter — numpy array ops and layout translation."""

import numpy as np
import pytest

from agent_memory.adapters.outbound.trt_cache_adapter import TRTCacheAdapter
from agent_memory.domain.errors import TRTLayoutError

pytestmark = pytest.mark.unit

N_KV_HEADS = 8
HEAD_DIM = 128
SEQ_LEN = 256
N_LAYERS = 4


@pytest.fixture
def adapter() -> TRTCacheAdapter:
    return TRTCacheAdapter()


def _make_kv(seq_len: int = SEQ_LEN) -> tuple[np.ndarray, np.ndarray]:
    """Create random FP16 K/V tensors."""
    rng = np.random.default_rng(42)
    k = rng.standard_normal((N_KV_HEADS, seq_len, HEAD_DIM)).astype(np.float16)
    v = rng.standard_normal((N_KV_HEADS, seq_len, HEAD_DIM)).astype(np.float16)
    return k, v


class TestConcatenate:
    def test_single_block(self, adapter: TRTCacheAdapter) -> None:
        k, v = _make_kv(128)
        k_cat, v_cat = adapter.concatenate_cache_blocks([k], [v])
        assert k_cat.shape == (N_KV_HEADS, 128, HEAD_DIM)
        np.testing.assert_array_equal(k_cat, k)

    def test_two_blocks(self, adapter: TRTCacheAdapter) -> None:
        k1, v1 = _make_kv(128)
        k2, v2 = _make_kv(64)
        k_cat, v_cat = adapter.concatenate_cache_blocks([k1, k2], [v1, v2])
        assert k_cat.shape == (N_KV_HEADS, 192, HEAD_DIM)
        assert v_cat.shape == (N_KV_HEADS, 192, HEAD_DIM)

    def test_empty_raises(self, adapter: TRTCacheAdapter) -> None:
        with pytest.raises(TRTLayoutError, match="empty"):
            adapter.concatenate_cache_blocks([], [])


class TestSequenceLength:
    def test_get_seq_len(self, adapter: TRTCacheAdapter) -> None:
        k, _ = _make_kv(512)
        assert adapter.get_sequence_length(k) == 512


class TestSlice:
    def test_slice_middle(self, adapter: TRTCacheAdapter) -> None:
        k, _ = _make_kv(256)
        sliced = adapter.slice_cache_tensor(k, 100, 200)
        assert sliced.shape == (N_KV_HEADS, 100, HEAD_DIM)
        np.testing.assert_array_equal(sliced, k[:, 100:200, :])


class TestLayoutConversion:
    def test_round_trip(self, adapter: TRTCacheAdapter) -> None:
        """per_layer -> stacked -> per_layer should be identity."""
        pairs = [_make_kv(SEQ_LEN) for _ in range(N_LAYERS)]
        stacked = adapter.per_layer_to_stacked(pairs)

        assert stacked.shape == (N_LAYERS, 2, N_KV_HEADS, SEQ_LEN, HEAD_DIM)
        assert stacked.dtype == np.float16

        recovered = adapter.stacked_to_per_layer(stacked)
        assert len(recovered) == N_LAYERS

        for (k_orig, v_orig), (k_rec, v_rec) in zip(pairs, recovered):
            np.testing.assert_array_equal(k_orig, k_rec)
            np.testing.assert_array_equal(v_orig, v_rec)

    def test_stacked_shape(self, adapter: TRTCacheAdapter) -> None:
        pairs = [_make_kv(64) for _ in range(2)]
        stacked = adapter.per_layer_to_stacked(pairs)
        assert stacked.shape == (2, 2, N_KV_HEADS, 64, HEAD_DIM)

    def test_empty_layers_raises(self, adapter: TRTCacheAdapter) -> None:
        with pytest.raises(TRTLayoutError, match="empty"):
            adapter.per_layer_to_stacked([])

    def test_wrong_ndim_raises(self, adapter: TRTCacheAdapter) -> None:
        bad = np.zeros((4, 3, 8, 64, 128), dtype=np.float16)
        # ndim is 5 but axis 1 is 3, not 2
        with pytest.raises(TRTLayoutError, match="axis 1 size 2"):
            adapter.stacked_to_per_layer(bad)

    def test_4d_raises(self, adapter: TRTCacheAdapter) -> None:
        bad = np.zeros((4, 8, 64, 128), dtype=np.float16)
        with pytest.raises(TRTLayoutError, match="5D"):
            adapter.stacked_to_per_layer(bad)
