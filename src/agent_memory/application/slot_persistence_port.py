# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Yakov Shkolnikov and contributors
"""Slot-level KV cache persistence port.

Defines the protocol that any backend adapter must implement to support
slot-level KV cache save/restore.  The application layer (SlotTracker,
swap orchestrator) depends on this protocol — never on a concrete adapter.

Architecture layer: application (port / driven interface).

Implementors:
- LlamaCppBackendAdapter (via HTTP /slots/ API)
- (future) VllmAdapter, TrtAdapter, etc.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class SlotPersistencePort(Protocol):
    """Backend-agnostic slot-level KV cache persistence."""

    def save_slot(self, slot_id: int, filename: str) -> dict[str, Any]:
        """Persist a slot's KV cache to disk.

        Args:
            slot_id: Slot index (0-based).
            filename: Filename relative to the backend's slot save directory.

        Returns:
            Response dict with at least ``n_saved`` (tokens saved).
            Backend-specific fields (timings, bytes written) may also appear.
        """
        ...

    def restore_slot(self, slot_id: int, filename: str) -> dict[str, Any]:
        """Restore a slot's KV cache from disk.

        Args:
            slot_id: Slot index (0-based).
            filename: Filename relative to the backend's slot save directory.

        Returns:
            Response dict with at least ``n_restored`` (tokens restored).
        """
        ...

    def erase_slot(self, slot_id: int) -> dict[str, Any]:
        """Clear a slot's KV cache from memory.

        Args:
            slot_id: Slot index (0-based).

        Returns:
            Response dict with at least ``n_erased`` (tokens erased).
        """
        ...
