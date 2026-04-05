# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Yakov Shkolnikov and contributors
"""Unit tests for cache eviction policies (LRU, LFU, hybrid LRU-LFU)."""

import time

import pytest

from agent_memory.application.agent_cache_store import CacheEntry

pytestmark = pytest.mark.unit


def _make_entry(
    agent_id: str,
    last_accessed: float = 0.0,
    access_count: int = 0,
    pinned: bool = False,
) -> CacheEntry:
    """Create a CacheEntry with specified eviction parameters."""
    entry = CacheEntry(
        agent_id=agent_id,
        blocks=None,
        model_tag=None,  # type: ignore[arg-type]
        last_accessed=last_accessed,
        access_count=access_count,
        pinned=pinned,
    )
    return entry


class TestEvictionScore:
    """Test CacheEntry.eviction_score() for different policies."""

    def test_lru_score_based_on_recency(self) -> None:
        """LRU: older entries have lower score (evicted first)."""
        old = _make_entry("old", last_accessed=100.0, access_count=50)
        new = _make_entry("new", last_accessed=200.0, access_count=1)
        assert old.eviction_score("lru") < new.eviction_score("lru")

    def test_lfu_score_based_on_frequency(self) -> None:
        """LFU: less-accessed entries have lower score (evicted first)."""
        rare = _make_entry("rare", last_accessed=200.0, access_count=1)
        frequent = _make_entry("freq", last_accessed=100.0, access_count=50)
        assert rare.eviction_score("lfu") < frequent.eviction_score("lfu")

    def test_hybrid_keeps_frequent_and_recent(self) -> None:
        """Hybrid: frequently-used system prompt beats rarely-used recent cache."""
        # System prompt: accessed 100 times, last accessed 1 hour ago
        sys_prompt = _make_entry(
            "sysprompt",
            last_accessed=time.time() - 3600,
            access_count=100,
        )
        # Agent cache: accessed 2 times, last accessed just now
        agent = _make_entry(
            "agent",
            last_accessed=time.time(),
            access_count=2,
        )
        # System prompt should score higher (kept longer)
        assert sys_prompt.eviction_score("lru-lfu") > agent.eviction_score("lru-lfu")

    def test_hybrid_evicts_stale_infrequent(self) -> None:
        """Hybrid: stale + infrequent entry should be evicted first."""
        stale = _make_entry(
            "stale",
            last_accessed=time.time() - 86400,
            access_count=1,
        )
        active = _make_entry(
            "active",
            last_accessed=time.time(),
            access_count=10,
        )
        assert stale.eviction_score("lru-lfu") < active.eviction_score("lru-lfu")

    def test_pinned_never_evicted(self) -> None:
        """Pinned entries always return infinity score."""
        pinned = _make_entry("pinned", last_accessed=0.0, access_count=0, pinned=True)
        assert pinned.eviction_score("lru") == float("inf")
        assert pinned.eviction_score("lfu") == float("inf")
        assert pinned.eviction_score("lru-lfu") == float("inf")


class TestNemoClawPattern:
    """Test eviction with NemoClaw-like workload pattern.

    NemoClaw: N agents share 1-2 system prompts + tools prefix.
    System prompt cache is most valuable but accessed "long ago"
    (at conversation start). Pure LRU would evict it.
    """

    def test_lru_evicts_system_prompt_wrongly(self) -> None:
        """Pure LRU evicts the system prompt (wrong for NemoClaw)."""
        sys_prompt = _make_entry(
            "sysprompt_tools",
            last_accessed=time.time() - 300,  # 5 min ago
            access_count=50,
        )
        recent_agent = _make_entry(
            "agent_turn_5",
            last_accessed=time.time(),
            access_count=1,
        )
        # LRU: system prompt has lower score (evicted first) — BAD
        assert sys_prompt.eviction_score("lru") < recent_agent.eviction_score("lru")

    def test_hybrid_keeps_system_prompt(self) -> None:
        """Hybrid policy keeps system prompt (correct for NemoClaw)."""
        sys_prompt = _make_entry(
            "sysprompt_tools",
            last_accessed=time.time() - 300,
            access_count=50,
        )
        recent_agent = _make_entry(
            "agent_turn_5",
            last_accessed=time.time(),
            access_count=1,
        )
        # Hybrid: system prompt has higher score (kept) — CORRECT
        assert sys_prompt.eviction_score("lru-lfu") > recent_agent.eviction_score("lru-lfu")

    def test_pinned_system_prompt_immune(self) -> None:
        """Pinned system prompt is immune to all eviction policies."""
        sys_prompt = _make_entry(
            "sysprompt_tools",
            last_accessed=0.0,  # Very old
            access_count=0,  # Never accessed (worst case)
            pinned=True,
        )
        assert sys_prompt.eviction_score("lru") == float("inf")
        assert sys_prompt.eviction_score("lru-lfu") == float("inf")


class TestClaudeCodePattern:
    """Test eviction with Claude Code-like workload pattern.

    Claude Code: long system prompts (~18K tokens) with tool definitions.
    Each agentic loop iteration appends conversation context.
    X-Session-ID keeps agent_id stable across turns.
    """

    def test_hybrid_keeps_long_session(self) -> None:
        """Hybrid keeps a frequently-used session over a stale one-shot."""
        long_session = _make_entry(
            "sess_claude_code_main",
            last_accessed=time.time() - 60,  # 1 min ago
            access_count=20,  # 20 turns in agentic loop
        )
        one_shot = _make_entry(
            "msg_random_hash",
            last_accessed=time.time() - 10,  # 10 sec ago
            access_count=1,
        )
        assert long_session.eviction_score("lru-lfu") > one_shot.eviction_score("lru-lfu")
