"""Admin API for model management and server control.

Provides HTTP endpoints for:
- Model hot-swapping
- Model status queries
- Server health checks

Authentication: Requires SEMANTIC_ADMIN_KEY header matching env var.
"""

import asyncio
import logging
import os
import secrets
from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
from pydantic import BaseModel, Field

from semantic.application.model_registry import ModelRegistry
from semantic.application.model_swap_orchestrator import ModelSwapOrchestrator

logger = logging.getLogger(__name__)

# Global lock to prevent concurrent model swaps (CR-2 fix)
# CRITICAL: On M4 Pro 24GB, only ONE model fits in memory at a time.
# Concurrent swaps would load multiple models → OOM crash
_swap_lock = asyncio.Lock()

# Router for all admin endpoints
router = APIRouter(prefix="/admin", tags=["admin"])


# --- Request/Response Models ---


class SwapModelRequest(BaseModel):
    """Request to swap to a different model."""

    model_id: str = Field(
        ...,
        description="HuggingFace model ID (e.g., 'mlx-community/Qwen2.5-14B-Instruct-4bit')",
        examples=["mlx-community/Qwen2.5-14B-Instruct-4bit"],
    )
    timeout_seconds: float = Field(
        default=30.0,
        description="Maximum time to wait for active requests to drain",
        ge=1.0,
        le=300.0,
    )


class SwapModelResponse(BaseModel):
    """Response from model swap operation."""

    status: str = Field(..., description="Swap status: 'success' or 'failed'")
    old_model_id: str | None = Field(..., description="Previously loaded model ID")
    new_model_id: str = Field(..., description="Newly loaded model ID")
    message: str = Field(..., description="Human-readable status message")


class CurrentModelResponse(BaseModel):
    """Response with currently loaded model info."""

    model_id: str | None = Field(..., description="Currently loaded model ID (null if none)")
    n_layers: int | None = Field(..., description="Number of transformer layers")
    n_kv_heads: int | None = Field(..., description="Number of KV attention heads")
    head_dim: int | None = Field(..., description="Dimension of each attention head")
    block_tokens: int | None = Field(..., description="Tokens per cache block")


class AvailableModelsResponse(BaseModel):
    """Response with list of available models."""

    models: list[str] = Field(
        ...,
        description="List of supported model IDs",
        examples=[
            [
                "mlx-community/Qwen2.5-14B-Instruct-4bit",
                "mlx-community/Llama-3.1-8B-Instruct-4bit",
                "mlx-community/gemma-3-12b-it-4bit",
            ]
        ],
    )


# --- Authentication ---


def verify_admin_key(
    x_admin_key: str | None = Header(None, alias="X-Admin-Key"),
) -> None:
    """Verify admin authentication key.

    Args:
        x_admin_key: Admin key from X-Admin-Key header

    Raises:
        HTTPException: 401 if key missing or invalid
    """
    expected_key = os.getenv("SEMANTIC_ADMIN_KEY")

    if expected_key is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Admin authentication not configured (SEMANTIC_ADMIN_KEY missing)",
        )

    if x_admin_key is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Admin authentication required (X-Admin-Key header missing)",
        )

    if not secrets.compare_digest(x_admin_key, expected_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid admin key",
        )


# --- Endpoints ---


def get_orchestrator() -> ModelSwapOrchestrator:
    """Dependency to get ModelSwapOrchestrator (override in tests)."""
    raise NotImplementedError("ModelSwapOrchestrator must be provided via dependency override")


def get_old_engine() -> Any:
    """Dependency to get current BatchEngine (override in tests)."""
    raise NotImplementedError("Old engine must be provided via dependency override")


@router.post("/models/swap", response_model=SwapModelResponse)
async def swap_model(
    swap_request: SwapModelRequest,
    request: Request,
    orchestrator: ModelSwapOrchestrator = Depends(get_orchestrator),  # noqa: B008
    old_engine: Any = Depends(get_old_engine),  # noqa: B008
    _auth: None = Depends(verify_admin_key),
) -> SwapModelResponse:
    """Swap to a different model while preserving agent caches.

    Executes hot-swap sequence:
    1. Drain active requests
    2. Evict caches to disk
    3. Shutdown old BatchEngine
    4. Unload old model
    5. Load new model
    6. Reconfigure BlockPool
    7. Update cache store model tag
    8. Create new BatchEngine

    On failure, attempts rollback to previous model.

    Thread Safety: This endpoint uses a global lock to ensure only ONE swap
    executes at a time. Concurrent swap requests will block until the current
    swap completes. This prevents memory exhaustion on M4 Pro 24GB.

    Args:
        swap_request: Swap request with model_id and timeout
        request: FastAPI Request (for accessing app.state)
        orchestrator: ModelSwapOrchestrator (injected)
        old_engine: Current BatchEngine (injected from app.state)
        _auth: Admin authentication (dependency)

    Returns:
        SwapModelResponse with status and details

    Raises:
        HTTPException: 500 if swap fails (rollback attempted)
        HTTPException: 401 if authentication fails
    """
    # CRITICAL: Acquire lock to prevent concurrent swaps (CR-2 fix)
    # Without this, two simultaneous swaps could load multiple models → OOM crash
    async with _swap_lock:
        try:
            logger.info(f"Admin API: Swap request to {swap_request.model_id} (lock acquired)")

            # Get old model ID for response
            old_model_id = None
            if hasattr(orchestrator, "_registry"):
                old_model_id = orchestrator._registry.get_current_id()

            # Execute swap (async to properly await drain)
            new_engine = await orchestrator.swap_model(
                old_engine=old_engine,
                new_model_id=swap_request.model_id,
                timeout_seconds=swap_request.timeout_seconds,
            )

            # CRITICAL: Update app.state with new engine (CR-1 fix)
            request.app.state.semantic.batch_engine = new_engine

            # Update CoordinationService's engine reference
            if hasattr(request.app.state, "coordination_service"):
                request.app.state.coordination_service.update_engine(new_engine)

            logger.info("Admin API: App state updated with new batch engine")

            return SwapModelResponse(
                status="success",
                old_model_id=old_model_id,
                new_model_id=swap_request.model_id,
                message=f"Model swapped successfully to {swap_request.model_id}",
            )

        except Exception as e:
            logger.error(f"Admin API: Swap failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Model swap failed: {e!s}",
            ) from e


def get_registry() -> ModelRegistry:
    """Dependency to get ModelRegistry (override in tests)."""
    raise NotImplementedError("ModelRegistry must be provided via dependency override")


@router.get("/models/current", response_model=CurrentModelResponse)
async def get_current_model(
    registry: ModelRegistry = Depends(get_registry),  # noqa: B008
    _auth: None = Depends(verify_admin_key),
) -> CurrentModelResponse:
    """Get information about the currently loaded model.

    Args:
        registry: ModelRegistry (injected)
        _auth: Admin authentication (dependency)

    Returns:
        CurrentModelResponse with model details (null if no model loaded)
    """
    model_id = registry.get_current_id()
    spec = registry.get_current_spec()

    if model_id is None or spec is None:
        return CurrentModelResponse(
            model_id=None,
            n_layers=None,
            n_kv_heads=None,
            head_dim=None,
            block_tokens=None,
        )

    return CurrentModelResponse(
        model_id=model_id,
        n_layers=spec.n_layers,
        n_kv_heads=spec.n_kv_heads,
        head_dim=spec.head_dim,
        block_tokens=spec.block_tokens,
    )


@router.get("/models/available", response_model=AvailableModelsResponse)
async def get_available_models(
    _auth: None = Depends(verify_admin_key),
) -> AvailableModelsResponse:
    """Get list of available models for swapping.

    Args:
        _auth: Admin authentication (dependency)

    Returns:
        AvailableModelsResponse with list of supported model IDs

    Notes:
        - This is a static list of recommended 4-bit quantized models
        - Any HuggingFace model ID can be used, but these are validated
        - All models optimized for M4 Pro 24GB memory
    """
    # Recommended models (all validated on M4 Pro 24GB)
    supported_models = [
        "mlx-community/gemma-3-12b-it-4bit",
        "mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx",
        "mlx-community/Qwen2.5-14B-Instruct-4bit",
        "mlx-community/Llama-3.1-8B-Instruct-4bit",
        "mlx-community/gpt-oss-20b-MXFP4-Q4",
        "mlx-community/SmolLM2-135M-Instruct",
    ]

    return AvailableModelsResponse(models=supported_models)


class OffloadModelResponse(BaseModel):
    """Response from model offload operation."""

    status: str = Field(..., description="Offload status: 'success' or 'failed'")
    model_id: str | None = Field(..., description="Model that was offloaded")
    message: str = Field(..., description="Human-readable status message")


@router.post("/models/offload", response_model=OffloadModelResponse)
async def offload_model(
    request: Request,
    registry: ModelRegistry = Depends(get_registry),  # noqa: B008
    _auth: None = Depends(verify_admin_key),
) -> OffloadModelResponse:
    """Offload current model to free memory before loading a new one.

    This endpoint:
    1. Drains active requests
    2. Evicts all caches to disk
    3. Unloads model from memory
    4. Clears GPU/Metal cache

    After offload, no model is loaded. Call /models/swap to load a new model.

    Args:
        request: FastAPI Request (for accessing app.state)
        registry: ModelRegistry (injected)
        _auth: Admin authentication (dependency)

    Returns:
        OffloadModelResponse with status
    """
    import gc

    try:
        model_id = registry.get_current_id()
        if model_id is None:
            return OffloadModelResponse(
                status="success",
                model_id=None,
                message="No model currently loaded",
            )

        logger.info(f"Admin API: Offloading model {model_id}")

        engine = request.app.state.semantic.batch_engine
        if engine:
            logger.info("Draining active requests...")
            await engine.drain(timeout_seconds=30.0)

        cache_store = request.app.state.semantic.cache_store
        if cache_store:
            logger.info("Evicting caches to disk...")
            evicted = cache_store.evict_all_to_disk()
            logger.info(f"Evicted {evicted} caches")

        block_pool = request.app.state.semantic.block_pool
        if block_pool:
            logger.info("Clearing block pool allocations...")
            cleared = block_pool.force_clear_all_allocations()
            logger.info(f"Cleared {cleared} agent allocations from pool")

        if engine:
            logger.info("Shutting down batch engine...")
            engine.shutdown()
            request.app.state.semantic.batch_engine = None

        logger.info("Unloading model...")
        registry.unload_model()

        gc.collect()

        logger.info(f"Model {model_id} offloaded successfully")
        return OffloadModelResponse(
            status="success",
            model_id=model_id,
            message=f"Model {model_id} offloaded. Memory freed.",
        )

    except Exception as e:
        logger.error(f"Admin API: Offload failed: {e}")
        return OffloadModelResponse(
            status="failed",
            model_id=registry.get_current_id(),
            message=f"Offload failed: {e!s}",
        )


class ClearCachesResponse(BaseModel):
    """Response from cache clear operation."""

    status: str = Field(..., description="Clear status: 'success' or 'failed'")
    hot_cleared: int = Field(..., description="Number of hot caches cleared")
    disk_cleared: int = Field(..., description="Number of disk caches cleared")
    pool_cleared: int = Field(..., description="Number of pool allocations cleared")
    message: str = Field(..., description="Human-readable status message")


@router.delete("/caches", response_model=ClearCachesResponse)
async def clear_all_caches(
    request: Request,
    _auth: None = Depends(verify_admin_key),
) -> ClearCachesResponse:
    """Clear ALL caches from memory and disk.

    This endpoint:
    1. Clears all hot-tier caches (memory)
    2. Deletes all disk caches
    3. Clears BlockPool allocations

    Use with caution - all agent context will be permanently lost.

    Args:
        request: FastAPI Request (for accessing app.state)
        _auth: Admin authentication (dependency)

    Returns:
        ClearCachesResponse with counts of cleared items
    """
    import shutil

    try:
        hot_cleared = 0
        disk_cleared = 0
        pool_cleared = 0

        semantic = request.app.state.semantic

        cache_store = semantic.cache_store
        if cache_store:
            hot_cleared = len(cache_store._hot_cache)
            cache_store._hot_cache.clear()
            logger.info(f"Cleared {hot_cleared} hot caches")

        cache_dir = cache_store.cache_dir if cache_store else None
        if cache_dir and cache_dir.exists():
            for cache_file in cache_dir.glob("*.safetensors"):
                cache_file.unlink()
                disk_cleared += 1
            logger.info(f"Deleted {disk_cleared} disk cache files")

        block_pool = semantic.block_pool
        if block_pool:
            pool_cleared = block_pool.force_clear_all_allocations()
            logger.info(f"Cleared {pool_cleared} pool allocations")

        total = hot_cleared + disk_cleared + pool_cleared
        return ClearCachesResponse(
            status="success",
            hot_cleared=hot_cleared,
            disk_cleared=disk_cleared,
            pool_cleared=pool_cleared,
            message=f"Cleared {total} total items (hot: {hot_cleared}, disk: {disk_cleared}, pool: {pool_cleared})",
        )

    except Exception as e:
        logger.error(f"Admin API: Clear caches failed: {e}")
        return ClearCachesResponse(
            status="failed",
            hot_cleared=0,
            disk_cleared=0,
            pool_cleared=0,
            message=f"Clear failed: {e!s}",
        )
