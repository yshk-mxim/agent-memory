# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Yakov Shkolnikov and contributors
"""Unit tests for TRT prefill adapter — init caches, chunk sizing, prefill processing."""

from unittest.mock import Mock

import numpy as np
import pytest

from agent_memory.adapters.outbound.trt_prefill_adapter import (
    _LARGE_CACHE_THRESHOLD,
    _MAX_CHUNK,
    _MIN_CHUNK,
    TRTPrefillAdapter,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def mock_backend() -> Mock:
    return Mock()


@pytest.fixture
def adapter(mock_backend: Mock) -> TRTPrefillAdapter:
    return TRTPrefillAdapter(backend=mock_backend)


class TestInitPrefillCaches:
    def test_returns_empty_arrays_for_each_layer(self, adapter: TRTPrefillAdapter) -> None:
        caches = adapter.init_prefill_caches(n_layers=4)

        assert len(caches) == 4
        for k, v in caches:
            assert isinstance(k, np.ndarray)
            assert isinstance(v, np.ndarray)
            assert k.shape == (0,)
            assert v.shape == (0,)
            assert k.dtype == np.float16
            assert v.dtype == np.float16

    def test_zero_layers_returns_empty_list(self, adapter: TRTPrefillAdapter) -> None:
        caches = adapter.init_prefill_caches(n_layers=0)
        assert caches == []

    def test_single_layer(self, adapter: TRTPrefillAdapter) -> None:
        caches = adapter.init_prefill_caches(n_layers=1)
        assert len(caches) == 1
        k, v = caches[0]
        assert k.shape == (0,)
        assert v.shape == (0,)

    def test_many_layers(self, adapter: TRTPrefillAdapter) -> None:
        caches = adapter.init_prefill_caches(n_layers=48)
        assert len(caches) == 48


class TestChunkSizeForPosition:
    def test_below_threshold_returns_max_chunk(self, adapter: TRTPrefillAdapter) -> None:
        assert adapter.chunk_size_for_position(0) == _MAX_CHUNK
        assert adapter.chunk_size_for_position(100) == _MAX_CHUNK
        assert adapter.chunk_size_for_position(_LARGE_CACHE_THRESHOLD - 1) == _MAX_CHUNK

    def test_at_threshold_returns_max_chunk(self, adapter: TRTPrefillAdapter) -> None:
        # Exactly at threshold: condition is >, not >=
        assert adapter.chunk_size_for_position(_LARGE_CACHE_THRESHOLD) == _MAX_CHUNK

    def test_above_threshold_returns_min_chunk(self, adapter: TRTPrefillAdapter) -> None:
        assert adapter.chunk_size_for_position(_LARGE_CACHE_THRESHOLD + 1) == _MIN_CHUNK
        assert adapter.chunk_size_for_position(_LARGE_CACHE_THRESHOLD + 1000) == _MIN_CHUNK
        assert adapter.chunk_size_for_position(100_000) == _MIN_CHUNK

    def test_custom_chunk_bounds(self, mock_backend: Mock) -> None:
        custom = TRTPrefillAdapter(backend=mock_backend, min_chunk=128, max_chunk=4096)
        assert custom.chunk_size_for_position(0) == 4096
        assert custom.chunk_size_for_position(_LARGE_CACHE_THRESHOLD + 1) == 128

    def test_default_constants(self) -> None:
        """Verify module-level constants are sensible."""
        assert _MIN_CHUNK == 512
        assert _MAX_CHUNK == 2048
        assert _LARGE_CACHE_THRESHOLD == 8192
        assert _MIN_CHUNK < _MAX_CHUNK


class TestProcessPrefillChunk:
    def test_first_chunk_passes_no_cache(
        self, adapter: TRTPrefillAdapter, mock_backend: Mock
    ) -> None:
        mock_result = Mock()
        mock_result.cache = None
        mock_backend.generate.return_value = mock_result

        tokens = [1, 2, 3, 4, 5]
        kv_caches = [(np.zeros((0,)), np.zeros((0,)))]

        adapter.process_prefill_chunk(tokens, start=0, end=3, kv_caches=kv_caches)

        mock_backend.generate.assert_called_once_with(
            prompt_tokens=[1, 2, 3],
            cache=None,  # start == 0 => no cache passed
            max_tokens=0,
            temperature=0.0,
        )

    def test_subsequent_chunk_passes_cache(
        self, adapter: TRTPrefillAdapter, mock_backend: Mock
    ) -> None:
        mock_result = Mock()
        mock_result.cache = None
        mock_backend.generate.return_value = mock_result

        tokens = [1, 2, 3, 4, 5]
        kv_caches = [(np.zeros((0,)), np.zeros((0,)))]

        adapter.process_prefill_chunk(tokens, start=3, end=5, kv_caches=kv_caches)

        mock_backend.generate.assert_called_once_with(
            prompt_tokens=[4, 5],
            cache=kv_caches,  # start > 0 => cache passed
            max_tokens=0,
            temperature=0.0,
        )

    def test_result_cache_updates_kv_in_place(
        self, adapter: TRTPrefillAdapter, mock_backend: Mock
    ) -> None:
        new_k = np.ones((8, 3, 128), dtype=np.float16)
        new_v = np.ones((8, 3, 128), dtype=np.float16) * 2

        mock_result = Mock()
        mock_result.cache = [(new_k, new_v)]
        mock_backend.generate.return_value = mock_result

        kv_caches = [(np.zeros((0,)), np.zeros((0,)))]
        adapter.process_prefill_chunk([1, 2, 3], start=0, end=3, kv_caches=kv_caches)

        k_out, v_out = kv_caches[0]
        np.testing.assert_array_equal(k_out, new_k)
        np.testing.assert_array_equal(v_out, new_v)

    def test_no_cache_in_result_leaves_kv_unchanged(
        self, adapter: TRTPrefillAdapter, mock_backend: Mock
    ) -> None:
        mock_result = Mock()
        mock_result.cache = None
        mock_backend.generate.return_value = mock_result

        original_k = np.zeros((0,), dtype=np.float16)
        original_v = np.zeros((0,), dtype=np.float16)
        kv_caches = [(original_k, original_v)]

        adapter.process_prefill_chunk([1, 2], start=0, end=2, kv_caches=kv_caches)

        # Caches unchanged
        assert kv_caches[0][0] is original_k
        assert kv_caches[0][1] is original_v
