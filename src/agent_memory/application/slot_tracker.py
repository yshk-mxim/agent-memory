# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Yakov Shkolnikov and contributors
"""Slot-level KV cache tracker with LRU-LFU–aware persistence.

Tracks per-slot usage (last-used timestamp, frequency, token count) and
orchestrates save/restore through any ``SlotPersistencePort`` implementation.
The application layer (swap orchestrator, api_server shutdown) calls
``SlotTracker`` — never a concrete adapter — to persist or recover slots.

Architecture layer: application (service).

Depends on:
- SlotPersistencePort (application port — injected, never imported from adapters)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

from agent_memory.application.slot_persistence_port import SlotPersistencePort

logger = logging.getLogger(__name__)


# ── Per-slot usage record ───────────────────────────────────────


@dataclass
class SlotUsage:
    """Tracks a single slot's usage statistics."""

    slot_id: int
    session_id: str | None = None
    last_used: float = 0.0
    use_count: int = 0
    n_tokens: int = 0


# ── SlotTracker service ────────────────────────────────────────


class SlotTracker:
    """LRU-LFU–aware slot tracker with backend-agnostic persistence.

    Scoring formula (higher = more valuable to keep):
        score = use_count / (1 + hours_since_last_use)

    A slot that was used 100 times but 24 hours ago scores:
        100 / (1 + 24) = 4.0

    A slot used 5 times but 1 minute ago scores:
        5 / (1 + 0.017) ≈ 4.9

    This hybrid naturally balances frequency and recency.
    """

    def __init__(
        self,
        n_slots: int,
        backend: SlotPersistencePort | None = None,
    ) -> None:
        self._n_slots = n_slots
        self._backend = backend
        self._slots: dict[int, SlotUsage] = {i: SlotUsage(slot_id=i) for i in range(n_slots)}

    # ── Usage tracking ──────────────────────────────────────────

    def mark_used(
        self,
        slot_id: int,
        session_id: str | None = None,
        n_tokens: int = 0,
    ) -> None:
        """Record that a slot was used (call after each generate())."""
        if slot_id not in self._slots:
            return
        s = self._slots[slot_id]
        s.session_id = session_id
        s.last_used = time.time()
        s.use_count += 1
        if n_tokens > 0:
            s.n_tokens = n_tokens

    def get_slot(self, slot_id: int) -> SlotUsage | None:
        """Return usage record for a slot, or None if out of range."""
        return self._slots.get(slot_id)

    # ── LRU-LFU ranking ────────────────────────────────────────

    def _score(self, s: SlotUsage) -> float:
        """Compute LRU-LFU hybrid score for a slot.

        Higher = more valuable to persist.
        Slots with 0 use_count always score 0 (empty / never used).
        """
        if s.use_count == 0:
            return 0.0
        hours_since = (time.time() - s.last_used) / 3600.0
        return s.use_count / (1.0 + hours_since)

    def ranked_slots(self) -> list[SlotUsage]:
        """Return slots ranked by value (highest score first)."""
        return sorted(self._slots.values(), key=self._score, reverse=True)

    def best_slots_to_save(self, max_count: int | None = None) -> list[int]:
        """Return slot IDs worth persisting, ranked by score.

        Excludes slots with 0 use_count (empty / never used).
        """
        if max_count is None:
            max_count = self._n_slots
        ranked = [s for s in self.ranked_slots() if s.use_count > 0]
        return [s.slot_id for s in ranked[:max_count]]

    # ── Persistence orchestration ───────────────────────────────

    def save_slots(
        self,
        model_id: str,
        max_count: int | None = None,
    ) -> int:
        """Save top-ranked slots to disk via the backend port.

        Args:
            model_id: Current model ID (used in filename).
            max_count: Max slots to save (default: all non-empty).

        Returns:
            Number of slots successfully saved.
        """
        if self._backend is None:
            logger.warning("save_slots called but no backend attached")
            return 0

        slot_ids = self.best_slots_to_save(max_count)
        saved = 0
        for slot_id in slot_ids:
            filename = self._slot_filename(model_id, slot_id)
            try:
                result = self._backend.save_slot(slot_id, filename)
                n_saved = result.get("n_saved", 0)
                if n_saved > 0:
                    saved += 1
                    logger.debug(
                        "saved slot %d → %s (%d tokens)",
                        slot_id,
                        filename,
                        n_saved,
                    )
            except Exception:
                logger.warning("failed to save slot %d", slot_id, exc_info=True)
        logger.info(
            "saved %d/%d slot caches for %s",
            saved,
            len(slot_ids),
            model_id,
        )
        return saved

    def restore_slots(
        self,
        model_id: str,
        slot_save_path: str | Path,
    ) -> int:
        """Restore slot caches from disk for the given model.

        Only attempts restore for slots whose files exist on disk.

        Args:
            model_id: Model ID (determines filename pattern).
            slot_save_path: Directory where slot files are stored.

        Returns:
            Number of slots successfully restored.
        """
        if self._backend is None:
            logger.warning("restore_slots called but no backend attached")
            return 0

        save_dir = Path(slot_save_path).expanduser()
        restored = 0
        for slot_id in range(self._n_slots):
            filename = self._slot_filename(model_id, slot_id)
            slot_path = save_dir / filename
            if not slot_path.exists():
                continue
            try:
                result = self._backend.restore_slot(slot_id, filename)
                n_restored = result.get("n_restored", 0)
                if n_restored > 0:
                    restored += 1
                    # Mark as used so restored slots have non-zero score
                    self.mark_used(slot_id, n_tokens=n_restored)
                    logger.debug(
                        "restored slot %d ← %s (%d tokens)",
                        slot_id,
                        filename,
                        n_restored,
                    )
            except Exception:
                logger.warning(
                    "failed to restore slot %d from %s",
                    slot_id,
                    filename,
                    exc_info=True,
                )
        logger.info(
            "restored %d/%d slot caches for %s",
            restored,
            self._n_slots,
            model_id,
        )
        return restored

    # ── Disk budget enforcement ─────────────────────────────────

    @staticmethod
    def enforce_disk_budget(
        slot_save_path: str | Path,
        max_mb: int,
        current_model_id: str | None = None,
    ) -> list[Path]:
        """Delete oldest model's slot files when over disk budget.

        Groups ``*.bin`` files in *slot_save_path* by model ID prefix,
        sorts groups by oldest mtime, and deletes whole groups until the
        total size is within budget.  Never deletes the *current_model_id*
        group.

        Args:
            slot_save_path: Directory containing slot cache files.
            max_mb: Maximum allowed disk usage in MB (0 = no limit).
            current_model_id: Model to protect from eviction.

        Returns:
            List of deleted file paths.
        """
        if max_mb <= 0:
            return []

        save_dir = Path(slot_save_path).expanduser()
        if not save_dir.exists():
            return []

        # Group files by model_id prefix: {model_id}_slot_{n}.bin
        model_groups: dict[str, list[Path]] = {}
        for f in save_dir.glob("*_slot_*.bin"):
            # Extract model_id from filename (everything before _slot_)
            parts = f.name.rsplit("_slot_", 1)
            if len(parts) != 2:
                continue
            mid = parts[0]
            model_groups.setdefault(mid, []).append(f)

        # Calculate total size
        total_bytes = sum(f.stat().st_size for files in model_groups.values() for f in files)
        max_bytes = max_mb * 1024 * 1024

        if total_bytes <= max_bytes:
            return []

        # Sort evictable groups by oldest file mtime (oldest first)
        evictable = [(mid, files) for mid, files in model_groups.items() if mid != current_model_id]
        evictable.sort(key=lambda g: min(f.stat().st_mtime for f in g[1]))

        deleted: list[Path] = []
        for mid, files in evictable:
            if total_bytes <= max_bytes:
                break
            group_bytes = sum(f.stat().st_size for f in files)
            for f in files:
                f.unlink()
                deleted.append(f)
                logger.debug("evicted slot file: %s", f)
            total_bytes -= group_bytes
            logger.info(
                "evicted %d slot files for model %s (freed %.1f MB)",
                len(files),
                mid,
                group_bytes / (1024 * 1024),
            )

        return deleted

    # ── Reset ───────────────────────────────────────────────────

    def reset(self) -> None:
        """Reset all usage tracking (e.g. after model swap)."""
        for s in self._slots.values():
            s.session_id = None
            s.last_used = 0.0
            s.use_count = 0
            s.n_tokens = 0

    def resize(self, n_slots: int) -> None:
        """Resize to a different number of slots and reset all usage.

        Called after model swap when the new model has a different
        slot count (e.g. 26B-A4B has 4 slots, 31B with spec decode has 1).
        """
        self._n_slots = n_slots
        self._slots = {i: SlotUsage(slot_id=i) for i in range(n_slots)}

    # ── Internals ───────────────────────────────────────────────

    @staticmethod
    def _slot_filename(model_id: str, slot_id: int) -> str:
        """Filename convention: ``{model_id}_slot_{slot_id}.bin``."""
        return f"{model_id}_slot_{slot_id}.bin"
