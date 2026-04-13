# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Yakov Shkolnikov and contributors
"""llama.cpp model swap orchestrator.

Coordinates model hot-swap for llama-server backends:
1. Save active slot KV caches via SlotTracker (LRU-LFU ranked)
2. Evict agent-memory caches to disk
3. Stop llama-server (via ModelRegistry → LlamaCppModelLoader)
4. Start llama-server with new model GGUF
4.5. Restore slot KV caches for new model (if saved from previous swap)
5. Update cache store model tag
6. Return new LlamaCppBackendAdapter

Follows the same pattern as MLX ModelSwapOrchestrator but adapted for
external process management. No BatchEngine — llama-server handles
its own batching via parallel slots.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from agent_memory.application.agent_cache_store import AgentCacheStore, ModelTag
from agent_memory.application.model_registry import ModelRegistry

if TYPE_CHECKING:
    from agent_memory.application.slot_tracker import SlotTracker

logger = logging.getLogger(__name__)


class LlamaCppSwapOrchestrator:
    """Orchestrates model swap for llama.cpp backends.

    Unlike the MLX orchestrator, there is no BatchEngine to drain.
    llama-server slots are saved via HTTP before killing the process.

    Thread Safety:
        Not thread-safe. Caller must hold the admin API swap lock.
    """

    def __init__(
        self,
        model_registry: ModelRegistry,
        cache_store: AgentCacheStore,
        model_loader: Any,  # LlamaCppModelLoader
        slot_tracker: SlotTracker | None = None,
        slot_save_path: str = "~/.agent_memory/llamacpp_slots",
    ) -> None:
        self._registry = model_registry
        self._cache_store = cache_store
        self._loader = model_loader  # For slot save/restore + current_n_slots
        self._slot_tracker = slot_tracker
        self._slot_save_path = slot_save_path

    def swap_model_sync(
        self,
        new_model_id: str,
        timeout_seconds: float = 60.0,
    ) -> Any:
        """Synchronous swap — used by auto-swap in TRTInferenceService.

        All operations in the swap sequence are synchronous (subprocess
        start/stop, HTTP health checks, cache eviction). The async wrapper
        exists for the admin API endpoint.
        """
        return self._do_swap(new_model_id, timeout_seconds)

    async def swap_model(
        self,
        new_model_id: str,
        timeout_seconds: float = 60.0,
    ) -> Any:
        """Async swap — used by admin API endpoint."""
        return self._do_swap(new_model_id, timeout_seconds)

    def _do_swap(
        self,
        new_model_id: str,
        timeout_seconds: float = 60.0,
    ) -> Any:
        """Execute swap sequence. Returns new (adapter, tokenizer).

        Args:
            new_model_id: Model ID matching a config/models/*.toml
            timeout_seconds: Max time for server startup health check

        Returns:
            Tuple of (LlamaCppBackendAdapter, tokenizer)

        Raises:
            ModelNotFoundError: If model config or GGUF not found
            GenerationError: If server fails to start
        """
        old_model_id = self._registry.get_current_id()

        if old_model_id == new_model_id:
            logger.info("model %s already loaded, skipping swap", new_model_id)
            current = self._registry.get_current()
            if current:
                return current
            # Shouldn't happen, but load if state is inconsistent
            return self._registry.load_model(new_model_id)

        logger.info("llamacpp swap: %s -> %s", old_model_id, new_model_id)
        old_spec = self._registry.get_current_spec()

        try:
            # Step 1: Save slot KV caches (LRU-LFU ranked via SlotTracker)
            # Use loader's current_n_slots — reflects the OLD model's TOML config
            old_n_slots = self._loader.current_n_slots
            if self._loader.is_running and old_model_id:
                logger.info("Step 1/6: Saving slot caches (%d slots)...", old_n_slots)
                if self._slot_tracker is not None:
                    saved = self._slot_tracker.save_slots(old_model_id)
                    logger.info("saved %d ranked slots for %s", saved, old_model_id)
                else:
                    self._loader.save_all_slots(old_n_slots, old_model_id)

            # Step 2: Evict agent-memory caches to disk
            logger.info("Step 2/6: Evicting agent caches to disk...")
            evicted = self._cache_store.evict_all_to_disk()
            logger.info("evicted %d caches to disk", evicted)

            # Step 3: Unload old model (kills llama-server)
            if old_model_id is not None:
                logger.info("Step 3/6: Stopping llama-server (%s)...", old_model_id)
                self._registry.unload_model()

            # Step 4: Load new model (starts llama-server with new GGUF)
            logger.info("Step 4/6: Starting llama-server (%s)...", new_model_id)
            adapter, tokenizer = self._registry.load_model(new_model_id)
            new_spec = self._registry.get_current_spec()

            # After load, loader.current_n_slots reflects the NEW model's TOML
            new_n_slots = self._loader.current_n_slots

            # Step 4.5: Resize slot tracker + restore caches for new model
            logger.info(
                "Step 5/6: Restoring slot caches for %s (%d slots)...",
                new_model_id,
                new_n_slots,
            )
            if self._slot_tracker is not None:
                # Resize tracker to match new model's slot count
                self._slot_tracker.resize(new_n_slots)
                restored = self._slot_tracker.restore_slots(
                    new_model_id,
                    self._slot_save_path,
                )
                logger.info("restored %d slots for %s", restored, new_model_id)
            else:
                self._loader.restore_all_slots(new_n_slots, new_model_id)

            # Step 6: Update cache store model tag
            if new_spec:
                logger.info("Step 6/6: Updating cache store model tag...")
                new_tag = ModelTag.from_spec(new_model_id, new_spec)
                self._cache_store.update_model_tag(new_tag)

            logger.info("llamacpp swap complete: %s -> %s", old_model_id, new_model_id)
            return adapter, tokenizer

        except Exception as e:
            logger.error("llamacpp swap failed: %s", e)

            # Attempt rollback
            if old_model_id and old_spec:
                logger.warning("attempting rollback to %s", old_model_id)
                try:
                    self._registry.load_model(old_model_id)
                    logger.info("rollback successful")
                except Exception as rollback_err:
                    logger.critical("rollback failed: %s — system degraded", rollback_err)

            raise
