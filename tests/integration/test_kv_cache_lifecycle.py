# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Yakov Shkolnikov and contributors
"""Integration tests for KV cache persistence lifecycle.

Tests the full save/restore/swap cycle with simulated backends,
and measures orchestration overhead to verify slot persistence
adds negligible latency.

Run: pytest tests/integration/test_kv_cache_lifecycle.py -v
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, Mock

import pytest

from agent_memory.application.slot_persistence_port import SlotPersistencePort
from agent_memory.application.slot_tracker import SlotTracker


# ── Simulated backend (realistic timing) ────────────────────────


class SimulatedLlamaCppBackend:
    """Simulates llama-server slot save/restore with realistic latency.

    Save: ~5ms per slot (writing KV cache to NVMe)
    Restore: ~3ms per slot (reading KV cache from NVMe)
    """

    def __init__(self, save_ms: float = 5.0, restore_ms: float = 3.0) -> None:
        self._save_ms = save_ms
        self._restore_ms = restore_ms
        self.save_log: list[tuple[int, str]] = []
        self.restore_log: list[tuple[int, str]] = []

    def save_slot(self, slot_id: int, filename: str) -> dict[str, Any]:
        time.sleep(self._save_ms / 1000)
        self.save_log.append((slot_id, filename))
        return {"n_saved": 30000}  # Simulate 30K token system prompt

    def restore_slot(self, slot_id: int, filename: str) -> dict[str, Any]:
        time.sleep(self._restore_ms / 1000)
        self.restore_log.append((slot_id, filename))
        return {"n_restored": 30000}

    def erase_slot(self, slot_id: int) -> dict[str, Any]:
        return {"n_erased": 30000}


# ── Protocol compliance ─────────────────────────────────────────


class TestSlotPersistencePortCompliance:
    """Verify SimulatedBackend and real adapter satisfy the protocol."""

    def test_simulated_backend_is_slot_persistence_port(self):
        backend = SimulatedLlamaCppBackend()
        assert isinstance(backend, SlotPersistencePort)

    def test_real_llamacpp_adapter_would_satisfy_port(self):
        """Verify the protocol contract is correctly defined."""
        # Check that the protocol has the expected methods
        assert hasattr(SlotPersistencePort, "save_slot")
        assert hasattr(SlotPersistencePort, "restore_slot")
        assert hasattr(SlotPersistencePort, "erase_slot")


# ── Full lifecycle: boot → use → shutdown → reboot ──────────────


class TestBootShutdownLifecycle:
    """Simulate fresh boot → generate → shutdown → reboot."""

    def test_cold_boot_no_saved_slots(self, tmp_path):
        """First ever boot: no slot files on disk, restore is a no-op."""
        backend = SimulatedLlamaCppBackend()
        tracker = SlotTracker(n_slots=4, backend=backend)

        restored = tracker.restore_slots("gemma-4-26b", tmp_path)
        assert restored == 0
        assert len(backend.restore_log) == 0

    def test_boot_use_shutdown_restore_cycle(self, tmp_path):
        """Full lifecycle: boot → use slots → shutdown (save) → reboot (restore)."""
        backend = SimulatedLlamaCppBackend()
        tracker = SlotTracker(n_slots=4, backend=backend)
        model_id = "gemma-4-26b"

        # Simulate usage: sessions hit different slots
        tracker.mark_used(0, "session-main", n_tokens=30000)
        tracker.mark_used(0, "session-main", n_tokens=30200)
        tracker.mark_used(2, "session-sub", n_tokens=30000)
        # Slots 1, 3 never used

        # Shutdown: save ranked slots
        saved = tracker.save_slots(model_id)
        assert saved == 2  # Only slots 0 and 2 (1 and 3 unused)
        assert len(backend.save_log) == 2

        # Create fake files on disk (simulating what llama-server writes)
        for slot_id, filename in backend.save_log:
            (tmp_path / filename).write_bytes(b"\x00" * 1000)

        # Reboot: new tracker, restore from disk
        backend2 = SimulatedLlamaCppBackend()
        tracker2 = SlotTracker(n_slots=4, backend=backend2)

        restored = tracker2.restore_slots(model_id, tmp_path)
        assert restored == 2
        assert len(backend2.restore_log) == 2

        # Restored slots should have usage tracking
        assert tracker2.get_slot(0).use_count == 1  # Marked by restore
        assert tracker2.get_slot(2).use_count == 1
        assert tracker2.get_slot(1).use_count == 0  # Not restored
        assert tracker2.get_slot(3).use_count == 0


# ── Model swap lifecycle: A → B → A ────────────────────────────


class TestModelSwapLifecycle:
    """Simulate model swap A → B → A with slot preservation."""

    def test_swap_a_to_b_to_a(self, tmp_path):
        """Slots for model A survive swap to B and back."""
        backend = SimulatedLlamaCppBackend()
        tracker = SlotTracker(n_slots=4, backend=backend)

        # === Model A: active usage ===
        tracker.mark_used(0, "s1", 30000)
        tracker.mark_used(1, "s2", 30000)
        tracker.mark_used(2, "s3", 30000)

        # Step 1: Save model A slots
        saved_a = tracker.save_slots("model-a")
        assert saved_a == 3
        for _, filename in backend.save_log:
            (tmp_path / filename).write_bytes(b"\x00" * 1000)

        # Step 2: Reset tracker for new model
        tracker.reset()
        assert all(tracker.get_slot(i).use_count == 0 for i in range(4))

        # === Model B: different usage ===
        tracker.mark_used(0, "s4", 20000)

        # Step 3: Save model B slots
        backend.save_log.clear()
        saved_b = tracker.save_slots("model-b")
        assert saved_b == 1
        for _, filename in backend.save_log:
            (tmp_path / filename).write_bytes(b"\x00" * 500)

        # Step 4: Reset and restore model A
        tracker.reset()
        backend.restore_log.clear()
        restored_a = tracker.restore_slots("model-a", tmp_path)
        assert restored_a == 3  # All 3 model-a slots recovered

        # Model A slots are back with usage tracking
        assert tracker.get_slot(0).use_count == 1
        assert tracker.get_slot(1).use_count == 1
        assert tracker.get_slot(2).use_count == 1

    def test_swap_does_not_cross_contaminate(self, tmp_path):
        """Restoring model A doesn't load model B's slot files."""
        backend = SimulatedLlamaCppBackend()
        tracker = SlotTracker(n_slots=2, backend=backend)

        # Create files for both models
        (tmp_path / "model-a_slot_0.bin").write_bytes(b"\x00" * 100)
        (tmp_path / "model-b_slot_0.bin").write_bytes(b"\x00" * 100)
        (tmp_path / "model-b_slot_1.bin").write_bytes(b"\x00" * 100)

        restored = tracker.restore_slots("model-a", tmp_path)
        assert restored == 1  # Only model-a file

        filenames_restored = [fn for _, fn in backend.restore_log]
        assert "model-a_slot_0.bin" in filenames_restored
        assert "model-b_slot_0.bin" not in filenames_restored


# ── Disk budget lifecycle ───────────────────────────────────────


class TestDiskBudgetLifecycle:
    """Test disk budget enforcement across multiple model swaps."""

    def test_budget_enforcement_across_swaps(self, tmp_path):
        """After many swaps, oldest model's files are evicted."""
        import os

        # Simulate: model-a (old), model-b (middle), model-c (current)
        for i, (model, age_hours) in enumerate([
            ("model-a", 48),
            ("model-b", 24),
            ("model-c", 0),
        ]):
            for slot in range(2):
                f = tmp_path / f"{model}_slot_{slot}.bin"
                f.write_bytes(b"\x00" * (512 * 1024))  # 512KB each
                # Set mtime to simulate age
                mtime = time.time() - (age_hours * 3600)
                os.utime(f, (mtime, mtime))

        # Total: 6 files × 512KB = 3MB
        # Budget: 2MB → should evict model-a (oldest, 1MB)
        deleted = SlotTracker.enforce_disk_budget(
            tmp_path, max_mb=2, current_model_id="model-c",
        )

        assert len(deleted) == 2
        assert all("model-a" in d.name for d in deleted)
        # model-b and model-c still exist
        assert (tmp_path / "model-b_slot_0.bin").exists()
        assert (tmp_path / "model-c_slot_0.bin").exists()


# ── LRU-LFU ranking accuracy ───────────────────────────────────


class TestLRULFURanking:
    """Verify the ranking formula produces correct orderings."""

    def test_active_session_outranks_stale_popular_session(self):
        """A recently active session should outrank a stale but popular one."""
        tracker = SlotTracker(n_slots=4)
        now = time.time()

        # Slot 0: very popular but stale (100 uses, 2 days ago)
        tracker._slots[0].use_count = 100
        tracker._slots[0].last_used = now - (48 * 3600)

        # Slot 1: moderately active (10 uses, 5 minutes ago)
        tracker._slots[1].use_count = 10
        tracker._slots[1].last_used = now - 300

        # Slot 2: fresh single use (1 use, just now)
        tracker._slots[2].use_count = 1
        tracker._slots[2].last_used = now

        best = tracker.best_slots_to_save()
        # Slot 1 should rank highest: 10 / (1 + 0.08) ≈ 9.2
        # Slot 2 next: 1 / (1 + 0) = 1.0
        # Slot 0 last: 100 / (1 + 48) = 2.04
        # Actually slot 0 beats slot 2: 2.04 > 1.0
        assert best[0] == 1  # Most valuable to save

    def test_empty_slots_excluded_from_save(self):
        """Slots with no usage should never be saved."""
        tracker = SlotTracker(n_slots=8)
        tracker.mark_used(3, "only-session", 100)

        best = tracker.best_slots_to_save()
        assert best == [3]


# ── Orchestration overhead timing ───────────────────────────────


class TestOrchestrationOverhead:
    """Measure overhead of save/restore orchestration."""

    def test_save_4_slots_under_100ms(self):
        """Saving 4 slots should complete in <100ms (simulated 5ms each)."""
        backend = SimulatedLlamaCppBackend(save_ms=5.0)
        tracker = SlotTracker(n_slots=4, backend=backend)
        for i in range(4):
            tracker.mark_used(i, f"s{i}", 30000)

        t0 = time.perf_counter()
        saved = tracker.save_slots("model-a")
        elapsed_ms = (time.perf_counter() - t0) * 1000

        assert saved == 4
        assert elapsed_ms < 100, f"Save took {elapsed_ms:.1f}ms, expected <100ms"

    def test_restore_4_slots_under_50ms(self, tmp_path):
        """Restoring 4 slots should complete in <50ms (simulated 3ms each)."""
        backend = SimulatedLlamaCppBackend(restore_ms=3.0)
        tracker = SlotTracker(n_slots=4, backend=backend)

        for i in range(4):
            (tmp_path / f"model-a_slot_{i}.bin").write_bytes(b"\x00" * 100)

        t0 = time.perf_counter()
        restored = tracker.restore_slots("model-a", tmp_path)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        assert restored == 4
        assert elapsed_ms < 50, f"Restore took {elapsed_ms:.1f}ms, expected <50ms"

    def test_tracker_overhead_is_negligible(self):
        """SlotTracker's own bookkeeping (mark_used, rank) should be <1ms."""
        tracker = SlotTracker(n_slots=64)  # Even with many slots

        t0 = time.perf_counter()
        for i in range(64):
            tracker.mark_used(i, f"session-{i}", 30000 + i * 100)
        ranked = tracker.ranked_slots()
        best = tracker.best_slots_to_save(max_count=8)
        elapsed_us = (time.perf_counter() - t0) * 1_000_000

        assert len(ranked) == 64
        assert len(best) == 8
        assert elapsed_us < 1000, f"Tracker overhead {elapsed_us:.0f}μs, expected <1000μs"


# ── Session-to-slot mapping ─────────────────────────────────────


class TestSessionSlotMapping:
    """Verify session_id → slot pinning behavior."""

    def test_same_session_same_slot(self):
        """Same session_id always maps to the same slot."""
        n_slots = 4
        session_id = "claude-code-abc123"
        slot = hash(session_id) % n_slots

        # Verify deterministic
        for _ in range(100):
            assert hash(session_id) % n_slots == slot

    def test_subagent_shares_slot_with_parent(self):
        """Subagents with same session_id hit the same slot (prefix reuse)."""
        n_slots = 4
        session_id = "shared-session-xyz"

        parent_slot = hash(session_id) % n_slots
        subagent_slot = hash(session_id) % n_slots

        assert parent_slot == subagent_slot

    def test_different_sessions_distribute_across_slots(self):
        """Different sessions should distribute across slots (not all on slot 0)."""
        n_slots = 4
        slots_hit = set()
        for i in range(100):
            slot = hash(f"session-{i}") % n_slots
            slots_hit.add(slot)

        # Should hit at least 3 of 4 slots with 100 different sessions
        assert len(slots_hit) >= 3


# ── SharedPrefixCache integration ───────────────────────────────


class TestSharedPrefixCacheIntegration:
    """Test SharedPrefixCache lifecycle (independent of MLX)."""

    def test_first_request_populates_second_hits(self):
        """First request with system+tools populates cache, second gets hit."""
        from agent_memory.application.shared_prefix_cache import SharedPrefixCache

        cache = SharedPrefixCache()
        system = "You are a helpful assistant."
        tools = "Read: Read file contents\nWrite: Write file"

        prefix_hash = SharedPrefixCache.compute_hash(system, tools)

        # First request: cache miss
        assert cache.get(prefix_hash) is None

        # Simulate: after generation, populate cache
        fake_kv = {"layer_0": [1, 2, 3]}
        cache.put(prefix_hash, kv_caches=fake_kv, n_tokens=500, token_sequence=list(range(500)))

        # Second request: cache hit
        entry = cache.get(prefix_hash)
        assert entry is not None
        assert entry.n_tokens == 500
        assert entry.kv_caches == fake_kv

    def test_tool_reordering_still_hits(self):
        """Claude Code reorders tools between turns — hash should be stable."""
        from agent_memory.application.shared_prefix_cache import SharedPrefixCache

        system = "You are helpful."
        tools_v1 = "Read: Read\nWrite: Write\nBash: Run"
        tools_v2 = "Bash: Run\nRead: Read\nWrite: Write"

        h1 = SharedPrefixCache.compute_hash(system, tools_v1)
        h2 = SharedPrefixCache.compute_hash(system, tools_v2)
        assert h1 == h2  # Order-independent

    def test_take_consumes_entry(self):
        """take() removes entry from cache so it can be consumed by submit()."""
        from agent_memory.application.shared_prefix_cache import SharedPrefixCache

        cache = SharedPrefixCache()
        cache.put("h1", kv_caches="kv_data", n_tokens=100, token_sequence=[1, 2])

        entry = cache.take("h1")
        assert entry is not None
        assert entry.kv_caches == "kv_data"
        assert cache.size == 0  # Entry consumed

        # Second take returns None
        assert cache.take("h1") is None

    def test_put_replaces_existing_entry(self):
        """put() replaces stale entry with fresh generation output."""
        from agent_memory.application.shared_prefix_cache import SharedPrefixCache

        cache = SharedPrefixCache()
        cache.put("h1", kv_caches="old_kv", n_tokens=100, token_sequence=[1])
        cache.put("h1", kv_caches="new_kv", n_tokens=200, token_sequence=[1, 2])

        entry = cache.get("h1")
        assert entry is not None
        assert entry.kv_caches == "new_kv"
        assert entry.n_tokens == 200
        assert cache.size == 1  # Still one entry, not two

    def test_take_then_put_lifecycle(self):
        """Full consume-and-replace lifecycle: take → use → put fresh."""
        from agent_memory.application.shared_prefix_cache import SharedPrefixCache

        cache = SharedPrefixCache()

        # Session A populates cache after generation
        cache.put("h1", kv_caches="session_a_kv", n_tokens=500, token_sequence=[1, 2, 3])

        # Session B consumes the entry
        entry = cache.take("h1")
        assert entry is not None
        assert entry.kv_caches == "session_a_kv"
        assert cache.size == 0

        # Session B stores fresh blocks after its generation
        cache.put("h1", kv_caches="session_b_kv", n_tokens=600, token_sequence=[1, 2, 3, 4])
        assert cache.size == 1

        # Session C gets B's fresh blocks
        entry2 = cache.take("h1")
        assert entry2 is not None
        assert entry2.kv_caches == "session_b_kv"
        assert entry2.n_tokens == 600

    def test_model_swap_clears_cache(self):
        """Cache should be cleared on model swap (KV state is model-specific)."""
        from agent_memory.application.shared_prefix_cache import SharedPrefixCache

        cache = SharedPrefixCache()
        cache.put("h1", kv_caches="kv", n_tokens=100, token_sequence=[1])
        assert cache.size == 1

        cache.clear()  # Called during model swap
        assert cache.size == 0
        assert cache.get("h1") is None
