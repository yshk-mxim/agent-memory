# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Yakov Shkolnikov and contributors
"""Unit tests for SharedPrefixCache."""

from agent_memory.application.shared_prefix_cache import SharedPrefixCache


class TestSharedPrefixCacheBasic:
    def test_empty_cache_returns_none(self) -> None:
        cache = SharedPrefixCache()
        assert cache.get("nonexistent") is None

    def test_put_and_get(self) -> None:
        cache = SharedPrefixCache()
        cache.put("abc123", kv_caches=["fake_kv"], n_tokens=100, token_sequence=[1, 2, 3])

        entry = cache.get("abc123")
        assert entry.n_tokens == 100
        assert entry.kv_caches == ["fake_kv"]
        assert entry.token_sequence == [1, 2, 3]

    def test_get_increments_hit_count(self) -> None:
        cache = SharedPrefixCache()
        cache.put("abc123", kv_caches=[], n_tokens=10, token_sequence=[])

        cache.get("abc123")
        cache.get("abc123")
        cache.get("abc123")

        entry = cache.get("abc123")
        assert entry.hit_count == 4  # 3 + 1 from final get

    def test_put_duplicate_is_noop(self) -> None:
        cache = SharedPrefixCache()
        cache.put("abc", kv_caches=["first"], n_tokens=10, token_sequence=[1])
        cache.put("abc", kv_caches=["second"], n_tokens=20, token_sequence=[2])

        entry = cache.get("abc")
        assert entry.kv_caches == ["first"]  # First value kept
        assert entry.n_tokens == 10

    def test_size_property(self) -> None:
        cache = SharedPrefixCache()
        assert cache.size == 0

        cache.put("a", kv_caches=[], n_tokens=1, token_sequence=[])
        assert cache.size == 1

        cache.put("b", kv_caches=[], n_tokens=2, token_sequence=[])
        assert cache.size == 2

    def test_clear(self) -> None:
        cache = SharedPrefixCache()
        cache.put("a", kv_caches=[], n_tokens=1, token_sequence=[])
        cache.put("b", kv_caches=[], n_tokens=2, token_sequence=[])
        assert cache.size == 2

        cache.clear()
        assert cache.size == 0
        assert cache.get("a") is None


class TestSharedPrefixCacheEviction:
    def test_evicts_when_full(self) -> None:
        cache = SharedPrefixCache(max_entries=2)
        cache.put("a", kv_caches=[], n_tokens=1, token_sequence=[])
        cache.put("b", kv_caches=[], n_tokens=2, token_sequence=[])
        assert cache.size == 2

        # Adding third should evict one (LFU)
        cache.put("c", kv_caches=[], n_tokens=3, token_sequence=[])
        assert cache.size == 2

    def test_evicts_least_hit(self) -> None:
        cache = SharedPrefixCache(max_entries=2)
        cache.put("popular", kv_caches=[], n_tokens=1, token_sequence=[])
        cache.put("unpopular", kv_caches=[], n_tokens=2, token_sequence=[])

        # Make "popular" have more hits
        cache.get("popular")
        cache.get("popular")
        cache.get("popular")

        # Add third — should evict "unpopular" (fewer hits)
        cache.put("new", kv_caches=[], n_tokens=3, token_sequence=[])

        assert cache.get("popular").n_tokens == 1  # popular survived eviction
        assert cache.get("new").n_tokens == 3  # new was just inserted
        assert cache.get("unpopular") is None  # evicted (fewest hits)


class TestSharedPrefixCacheHash:
    def test_compute_hash_deterministic(self) -> None:
        h1 = SharedPrefixCache.compute_hash("system prompt", "tool defs")
        h2 = SharedPrefixCache.compute_hash("system prompt", "tool defs")
        assert h1 == h2

    def test_compute_hash_different_for_different_input(self) -> None:
        h1 = SharedPrefixCache.compute_hash("prompt A", "tools")
        h2 = SharedPrefixCache.compute_hash("prompt B", "tools")
        assert h1 != h2

    def test_compute_hash_distinguishes_system_from_tools(self) -> None:
        h1 = SharedPrefixCache.compute_hash("abc", "def")
        h2 = SharedPrefixCache.compute_hash("abcdef", "")
        assert h1 != h2  # Separator prevents collision
