# SPDX-License-Identifier: MIT
"""Tests for SlotTracker — LRU-LFU ranking, persistence, disk budget."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from agent_memory.application.slot_tracker import SlotTracker, SlotUsage


# ── Helpers ──────────────────────────────────────────────────────


class FakeBackend:
    """Minimal SlotPersistencePort stub."""

    def __init__(self) -> None:
        self.saved: list[tuple[int, str]] = []
        self.restored: list[tuple[int, str]] = []
        self.erased: list[int] = []

    def save_slot(self, slot_id: int, filename: str) -> dict[str, Any]:
        self.saved.append((slot_id, filename))
        return {"n_saved": 1000}

    def restore_slot(self, slot_id: int, filename: str) -> dict[str, Any]:
        self.restored.append((slot_id, filename))
        return {"n_restored": 1000}

    def erase_slot(self, slot_id: int) -> dict[str, Any]:
        self.erased.append(slot_id)
        return {"n_erased": 500}


class FailingBackend:
    """Backend that raises on all slot operations."""

    def save_slot(self, slot_id: int, filename: str) -> dict[str, Any]:
        raise RuntimeError("save failed")

    def restore_slot(self, slot_id: int, filename: str) -> dict[str, Any]:
        raise RuntimeError("restore failed")

    def erase_slot(self, slot_id: int) -> dict[str, Any]:
        raise RuntimeError("erase failed")


# ── SlotUsage dataclass ─────────────────────────────────────────


class TestSlotUsage:
    def test_defaults(self):
        s = SlotUsage(slot_id=0)
        assert s.session_id is None
        assert s.last_used == 0.0
        assert s.use_count == 0
        assert s.n_tokens == 0


# ── mark_used ────────────────────────────────────────────────────


class TestMarkUsed:
    def test_basic_tracking(self):
        tracker = SlotTracker(n_slots=4)
        tracker.mark_used(0, session_id="abc", n_tokens=30000)

        s = tracker.get_slot(0)
        assert s is not None
        assert s.session_id == "abc"
        assert s.use_count == 1
        assert s.n_tokens == 30000
        assert s.last_used > 0

    def test_increments_use_count(self):
        tracker = SlotTracker(n_slots=2)
        tracker.mark_used(0, "s1", 100)
        tracker.mark_used(0, "s1", 200)
        tracker.mark_used(0, "s1", 300)

        assert tracker.get_slot(0).use_count == 3
        assert tracker.get_slot(0).n_tokens == 300

    def test_updates_session_id(self):
        tracker = SlotTracker(n_slots=2)
        tracker.mark_used(0, "session_a", 100)
        tracker.mark_used(0, "session_b", 200)

        assert tracker.get_slot(0).session_id == "session_b"

    def test_ignores_out_of_range(self):
        tracker = SlotTracker(n_slots=2)
        tracker.mark_used(99, "s1", 100)  # should not raise
        assert tracker.get_slot(99) is None

    def test_zero_tokens_preserves_existing(self):
        tracker = SlotTracker(n_slots=2)
        tracker.mark_used(0, "s1", 500)
        tracker.mark_used(0, "s1", 0)  # n_tokens=0 should not overwrite

        assert tracker.get_slot(0).n_tokens == 500
        assert tracker.get_slot(0).use_count == 2


# ── LRU-LFU ranking ─────────────────────────────────────────────


class TestRanking:
    def test_unused_slots_score_zero(self):
        tracker = SlotTracker(n_slots=4)
        ranked = tracker.ranked_slots()
        assert all(tracker._score(s) == 0.0 for s in ranked)

    def test_more_frequent_ranks_higher(self):
        tracker = SlotTracker(n_slots=4)
        now = time.time()

        # Slot 0: used 10 times, just now
        tracker._slots[0].use_count = 10
        tracker._slots[0].last_used = now

        # Slot 1: used 2 times, just now
        tracker._slots[1].use_count = 2
        tracker._slots[1].last_used = now

        ranked = tracker.ranked_slots()
        assert ranked[0].slot_id == 0
        assert ranked[1].slot_id == 1

    def test_recency_beats_stale_frequency(self):
        tracker = SlotTracker(n_slots=4)
        now = time.time()

        # Slot 0: used 100 times, 48 hours ago
        tracker._slots[0].use_count = 100
        tracker._slots[0].last_used = now - (48 * 3600)

        # Slot 1: used 5 times, just now
        tracker._slots[1].use_count = 5
        tracker._slots[1].last_used = now

        # Score slot 0: 100 / (1 + 48) = 2.04
        # Score slot 1: 5 / (1 + 0) = 5.0
        ranked = tracker.ranked_slots()
        assert ranked[0].slot_id == 1  # recent beats stale

    def test_best_slots_excludes_empty(self):
        tracker = SlotTracker(n_slots=4)
        tracker.mark_used(0, "s1", 100)
        tracker.mark_used(2, "s2", 200)

        best = tracker.best_slots_to_save()
        assert 1 not in best
        assert 3 not in best
        assert 0 in best
        assert 2 in best

    def test_best_slots_respects_max_count(self):
        tracker = SlotTracker(n_slots=4)
        for i in range(4):
            tracker.mark_used(i, f"s{i}", 100)

        best = tracker.best_slots_to_save(max_count=2)
        assert len(best) == 2


# ── save_slots ───────────────────────────────────────────────────


class TestSaveSlots:
    def test_saves_non_empty_slots(self):
        backend = FakeBackend()
        tracker = SlotTracker(n_slots=4, backend=backend)
        tracker.mark_used(0, "s1", 100)
        tracker.mark_used(2, "s2", 200)

        saved = tracker.save_slots("gemma-4-26b")
        assert saved == 2
        filenames = [fn for _, fn in backend.saved]
        assert "gemma-4-26b_slot_0.bin" in filenames
        assert "gemma-4-26b_slot_2.bin" in filenames

    def test_skips_empty_slots(self):
        backend = FakeBackend()
        tracker = SlotTracker(n_slots=4, backend=backend)
        tracker.mark_used(1, "s1", 100)

        tracker.save_slots("model-a")
        assert len(backend.saved) == 1
        assert backend.saved[0] == (1, "model-a_slot_1.bin")

    def test_save_with_max_count(self):
        backend = FakeBackend()
        tracker = SlotTracker(n_slots=4, backend=backend)
        for i in range(4):
            tracker.mark_used(i, f"s{i}", 100)

        saved = tracker.save_slots("model-a", max_count=2)
        assert saved == 2
        assert len(backend.saved) == 2

    def test_save_handles_backend_failure(self):
        backend = FailingBackend()
        tracker = SlotTracker(n_slots=4, backend=backend)
        tracker.mark_used(0, "s1", 100)

        saved = tracker.save_slots("model-a")
        assert saved == 0  # graceful failure

    def test_save_without_backend_returns_zero(self):
        tracker = SlotTracker(n_slots=4, backend=None)
        tracker.mark_used(0, "s1", 100)

        assert tracker.save_slots("model-a") == 0


# ── restore_slots ────────────────────────────────────────────────


class TestRestoreSlots:
    def test_restores_existing_files(self, tmp_path):
        backend = FakeBackend()
        tracker = SlotTracker(n_slots=4, backend=backend)

        # Create slot files for slots 0 and 2
        (tmp_path / "model-a_slot_0.bin").write_bytes(b"\x00" * 100)
        (tmp_path / "model-a_slot_2.bin").write_bytes(b"\x00" * 100)

        restored = tracker.restore_slots("model-a", tmp_path)
        assert restored == 2
        assert (0, "model-a_slot_0.bin") in backend.restored
        assert (2, "model-a_slot_2.bin") in backend.restored

    def test_skips_missing_files(self, tmp_path):
        backend = FakeBackend()
        tracker = SlotTracker(n_slots=4, backend=backend)

        # Only slot 1 has a file
        (tmp_path / "model-a_slot_1.bin").write_bytes(b"\x00" * 100)

        restored = tracker.restore_slots("model-a", tmp_path)
        assert restored == 1
        assert len(backend.restored) == 1

    def test_restore_marks_used(self, tmp_path):
        backend = FakeBackend()
        tracker = SlotTracker(n_slots=4, backend=backend)
        (tmp_path / "model-a_slot_0.bin").write_bytes(b"\x00" * 100)

        tracker.restore_slots("model-a", tmp_path)

        s = tracker.get_slot(0)
        assert s.use_count == 1
        assert s.n_tokens == 1000  # from FakeBackend.restore_slot

    def test_restore_handles_backend_failure(self, tmp_path):
        backend = FailingBackend()
        tracker = SlotTracker(n_slots=4, backend=backend)
        (tmp_path / "model-a_slot_0.bin").write_bytes(b"\x00" * 100)

        restored = tracker.restore_slots("model-a", tmp_path)
        assert restored == 0

    def test_restore_without_backend_returns_zero(self, tmp_path):
        tracker = SlotTracker(n_slots=4, backend=None)
        (tmp_path / "model-a_slot_0.bin").write_bytes(b"\x00" * 100)

        assert tracker.restore_slots("model-a", tmp_path) == 0


# ── Disk budget enforcement ──────────────────────────────────────


class TestDiskBudget:
    def _create_slot_files(
        self, tmp_path: Path, model_id: str, n_slots: int, size_bytes: int,
    ) -> list[Path]:
        """Create fake slot files and return their paths."""
        files = []
        for i in range(n_slots):
            f = tmp_path / f"{model_id}_slot_{i}.bin"
            f.write_bytes(b"\x00" * size_bytes)
            files.append(f)
        return files

    def test_no_limit_does_nothing(self, tmp_path):
        self._create_slot_files(tmp_path, "model-a", 4, 1024 * 1024)
        deleted = SlotTracker.enforce_disk_budget(tmp_path, max_mb=0)
        assert deleted == []

    def test_under_budget_does_nothing(self, tmp_path):
        # 4 files × 1MB = 4MB, budget = 10MB
        self._create_slot_files(tmp_path, "model-a", 4, 1024 * 1024)
        deleted = SlotTracker.enforce_disk_budget(tmp_path, max_mb=10)
        assert deleted == []

    def test_evicts_oldest_model_first(self, tmp_path):
        # model-a: 2MB (older)
        old_files = self._create_slot_files(tmp_path, "model-a", 2, 1024 * 1024)
        # Touch with old mtime
        import os
        for f in old_files:
            os.utime(f, (time.time() - 3600, time.time() - 3600))

        # model-b: 2MB (newer, current)
        self._create_slot_files(tmp_path, "model-b", 2, 1024 * 1024)

        # Budget = 3MB → need to evict model-a (oldest)
        deleted = SlotTracker.enforce_disk_budget(
            tmp_path, max_mb=3, current_model_id="model-b",
        )
        assert len(deleted) == 2
        assert all("model-a" in d.name for d in deleted)
        # model-b files still exist
        assert (tmp_path / "model-b_slot_0.bin").exists()
        assert (tmp_path / "model-b_slot_1.bin").exists()

    def test_never_evicts_current_model(self, tmp_path):
        # Only current model's files, over budget
        self._create_slot_files(tmp_path, "model-a", 4, 1024 * 1024)

        deleted = SlotTracker.enforce_disk_budget(
            tmp_path, max_mb=1, current_model_id="model-a",
        )
        assert deleted == []  # can't evict protected model

    def test_evicts_multiple_models_if_needed(self, tmp_path):
        import os

        # model-a: 2MB (oldest)
        a_files = self._create_slot_files(tmp_path, "model-a", 2, 1024 * 1024)
        for f in a_files:
            os.utime(f, (time.time() - 7200, time.time() - 7200))

        # model-b: 2MB (middle)
        b_files = self._create_slot_files(tmp_path, "model-b", 2, 1024 * 1024)
        for f in b_files:
            os.utime(f, (time.time() - 3600, time.time() - 3600))

        # model-c: 2MB (current)
        self._create_slot_files(tmp_path, "model-c", 2, 1024 * 1024)

        # Budget = 3MB → evict model-a and model-b
        deleted = SlotTracker.enforce_disk_budget(
            tmp_path, max_mb=3, current_model_id="model-c",
        )
        assert len(deleted) == 4
        model_ids = {d.name.rsplit("_slot_", 1)[0] for d in deleted}
        assert model_ids == {"model-a", "model-b"}

    def test_nonexistent_dir_returns_empty(self):
        deleted = SlotTracker.enforce_disk_budget(
            "/nonexistent/path", max_mb=1,
        )
        assert deleted == []

    def test_ignores_non_slot_files(self, tmp_path):
        # Create a non-slot file (shouldn't be grouped/evicted)
        (tmp_path / "config.json").write_bytes(b"\x00" * (2 * 1024 * 1024))
        self._create_slot_files(tmp_path, "model-a", 1, 1024)

        deleted = SlotTracker.enforce_disk_budget(tmp_path, max_mb=1)
        # Non-slot files are excluded from accounting — slot files are tiny
        assert deleted == []


# ── reset ────────────────────────────────────────────────────────


class TestReset:
    def test_clears_all_usage(self):
        tracker = SlotTracker(n_slots=4)
        for i in range(4):
            tracker.mark_used(i, f"s{i}", 1000)

        tracker.reset()
        for i in range(4):
            s = tracker.get_slot(i)
            assert s.use_count == 0
            assert s.n_tokens == 0
            assert s.session_id is None
            assert s.last_used == 0.0


# ── Filename convention ──────────────────────────────────────────


class TestFilename:
    def test_format(self):
        assert SlotTracker._slot_filename("gemma-4-26b", 0) == "gemma-4-26b_slot_0.bin"
        assert SlotTracker._slot_filename("qwen3-coder", 3) == "qwen3-coder_slot_3.bin"

    def test_model_id_with_special_chars(self):
        fn = SlotTracker._slot_filename("gemma-4-26b-a4b", 1)
        assert fn == "gemma-4-26b-a4b_slot_1.bin"
