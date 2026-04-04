# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Yakov Shkolnikov and contributors
"""Unit tests for llama.cpp swap orchestrator — 5-step sequence, rollback."""

from unittest.mock import MagicMock, Mock, call, patch

import pytest

from agent_memory.application.llamacpp_swap_orchestrator import LlamaCppSwapOrchestrator
from agent_memory.domain.errors import GenerationError, ModelNotFoundError
from agent_memory.domain.value_objects import ModelCacheSpec

pytestmark = pytest.mark.unit


def _make_spec(n_layers: int = 24) -> ModelCacheSpec:
    return ModelCacheSpec(
        n_layers=n_layers,
        n_kv_heads=8,
        head_dim=128,
        block_tokens=16,
        layer_types=["global"] * n_layers,
    )


def _make_orchestrator(
    current_model_id: str | None = "old-model",
    current_spec: ModelCacheSpec | None = None,
    loader_running: bool = True,
):
    """Build orchestrator with standard mocks."""
    mock_registry = Mock()
    mock_registry.get_current_id.return_value = current_model_id
    mock_registry.get_current_spec.return_value = current_spec or _make_spec(24)
    mock_registry.load_model.return_value = (MagicMock(name="adapter"), MagicMock(name="tokenizer"))

    mock_cache_store = Mock()
    mock_cache_store.evict_all_to_disk.return_value = 5

    mock_loader = Mock()
    mock_loader.is_running = loader_running
    mock_loader.save_all_slots.return_value = 3

    orchestrator = LlamaCppSwapOrchestrator(
        model_registry=mock_registry,
        cache_store=mock_cache_store,
        model_loader=mock_loader,
    )
    return orchestrator, mock_registry, mock_cache_store, mock_loader


# ── swap_model 5-step sequence ─────────────────────────────────


class TestSwapModelSequence:
    """swap_model executes 5-step sequence in order."""

    async def test_full_swap_sequence(self) -> None:
        old_spec = _make_spec(24)
        new_spec = _make_spec(32)

        orchestrator, registry, cache_store, loader = _make_orchestrator(
            current_model_id="old-model",
            current_spec=old_spec,
        )
        # get_current_spec returns old_spec first (for rollback), then new_spec (after load)
        registry.get_current_spec.side_effect = [old_spec, new_spec]

        result = await orchestrator.swap_model("new-model")

        # Step 1: Save slots
        loader.save_all_slots.assert_called_once_with(4)

        # Step 2: Evict caches
        cache_store.evict_all_to_disk.assert_called_once()

        # Step 3: Unload old model
        registry.unload_model.assert_called_once()

        # Step 4: Load new model
        registry.load_model.assert_called_once_with("new-model")

        # Step 5: Update cache store model tag
        cache_store.update_model_tag.assert_called_once()

        # Returns (adapter, tokenizer)
        assert result is not None
        assert len(result) == 2

    async def test_skips_slot_save_when_server_not_running(self) -> None:
        orchestrator, registry, cache_store, loader = _make_orchestrator(
            current_model_id="old-model",
            loader_running=False,
        )
        new_spec = _make_spec(32)
        registry.get_current_spec.side_effect = [_make_spec(24), new_spec]

        await orchestrator.swap_model("new-model")

        # Slot save should be skipped
        loader.save_all_slots.assert_not_called()

        # But the rest should still run
        cache_store.evict_all_to_disk.assert_called_once()
        registry.unload_model.assert_called_once()
        registry.load_model.assert_called_once_with("new-model")

    async def test_skips_unload_when_no_old_model(self) -> None:
        orchestrator, registry, cache_store, loader = _make_orchestrator(
            current_model_id=None,
            loader_running=False,
        )
        new_spec = _make_spec(32)
        registry.get_current_spec.return_value = new_spec

        await orchestrator.swap_model("new-model")

        registry.unload_model.assert_not_called()
        registry.load_model.assert_called_once_with("new-model")


# ── swap_model same model returns early ────────────────────────


class TestSwapModelSameModel:
    """swap_model with same model returns early without restarting."""

    async def test_same_model_returns_current(self) -> None:
        orchestrator, registry, cache_store, loader = _make_orchestrator(
            current_model_id="same-model",
        )
        current = (MagicMock(name="current_adapter"), MagicMock(name="current_tokenizer"))
        registry.get_current.return_value = current

        result = await orchestrator.swap_model("same-model")

        # Should return current model without any swap steps
        assert result is current
        loader.save_all_slots.assert_not_called()
        cache_store.evict_all_to_disk.assert_not_called()
        registry.unload_model.assert_not_called()
        registry.load_model.assert_not_called()

    async def test_same_model_loads_if_state_inconsistent(self) -> None:
        """If current model matches but get_current() returns None, load anyway."""
        orchestrator, registry, cache_store, loader = _make_orchestrator(
            current_model_id="same-model",
        )
        registry.get_current.return_value = None

        result = await orchestrator.swap_model("same-model")

        # Falls through to load_model since get_current returned None
        registry.load_model.assert_called_once_with("same-model")


# ── swap_model rollback on failure ─────────────────────────────


class TestSwapModelRollback:
    """swap_model rolls back to old model on failure."""

    async def test_rollback_on_load_failure(self) -> None:
        old_spec = _make_spec(24)
        orchestrator, registry, cache_store, loader = _make_orchestrator(
            current_model_id="old-model",
            current_spec=old_spec,
        )

        registry.get_current_spec.return_value = old_spec
        registry.load_model.side_effect = [
            ModelNotFoundError("new-model GGUF not found"),
            (MagicMock(), MagicMock()),  # rollback load succeeds
        ]

        with pytest.raises(ModelNotFoundError, match="GGUF not found"):
            await orchestrator.swap_model("new-model")

        # Should have attempted rollback
        assert registry.load_model.call_count == 2
        registry.load_model.assert_any_call("old-model")

    async def test_rollback_failure_still_raises_original_error(self) -> None:
        old_spec = _make_spec(24)
        orchestrator, registry, cache_store, loader = _make_orchestrator(
            current_model_id="old-model",
            current_spec=old_spec,
        )

        registry.get_current_spec.return_value = old_spec
        registry.load_model.side_effect = [
            GenerationError("server crashed"),
            GenerationError("rollback also failed"),
        ]

        with pytest.raises(GenerationError, match="server crashed"):
            await orchestrator.swap_model("new-model")

        assert registry.load_model.call_count == 2

    async def test_no_rollback_when_no_old_model(self) -> None:
        orchestrator, registry, cache_store, loader = _make_orchestrator(
            current_model_id=None,
            loader_running=False,
        )
        registry.get_current_spec.return_value = None
        registry.load_model.side_effect = ModelNotFoundError("not found")

        with pytest.raises(ModelNotFoundError):
            await orchestrator.swap_model("new-model")

        # Only one load attempt, no rollback
        assert registry.load_model.call_count == 1


# ── swap_model updates cache store model tag ───────────────────


class TestSwapModelUpdatesTag:
    """swap_model updates the cache store model tag after loading."""

    async def test_update_model_tag_called_with_new_spec(self) -> None:
        old_spec = _make_spec(24)
        new_spec = _make_spec(32)

        orchestrator, registry, cache_store, loader = _make_orchestrator(
            current_model_id="old-model",
            current_spec=old_spec,
        )
        registry.get_current_spec.side_effect = [old_spec, new_spec]

        await orchestrator.swap_model("new-model")

        cache_store.update_model_tag.assert_called_once()
        tag_arg = cache_store.update_model_tag.call_args[0][0]
        assert tag_arg.model_id == "new-model"
        assert tag_arg.n_layers == 32

    async def test_tag_not_updated_when_spec_is_none(self) -> None:
        old_spec = _make_spec(24)
        orchestrator, registry, cache_store, loader = _make_orchestrator(
            current_model_id="old-model",
            current_spec=old_spec,
        )
        # After load, get_current_spec returns None (unusual but handled)
        registry.get_current_spec.side_effect = [old_spec, None]

        await orchestrator.swap_model("new-model")

        cache_store.update_model_tag.assert_not_called()
