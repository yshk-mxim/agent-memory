"""Direct Agent API adapter (POST /v1/agents/*).

Provides low-level CRUD operations for agent cache management:
- Create agents with explicit IDs
- Generate text with direct agent access
- Query agent cache status
- Delete agents
"""

import asyncio
import gc
import logging
import uuid

from fastapi import APIRouter, HTTPException, Request, status

from agent_memory.adapters.inbound.adapter_helpers import (
    get_semantic_state,
    run_step_for_uid,
)
from agent_memory.adapters.inbound.request_models import (
    AgentResponse,
    CreateAgentRequest,
    GenerateRequest,
    GenerateResponse,
)
from agent_memory.application.agent_cache_store import AgentCacheStore
from agent_memory.application.batch_engine import BlockPoolBatchEngine
from agent_memory.domain.errors import PoolExhaustedError, SemanticError

try:
    import mlx.core as mx
except ImportError:
    mx = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/agents", tags=["direct"])


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_agent(request_body: CreateAgentRequest, request: Request) -> AgentResponse:
    """Create a new agent (POST /v1/agents).

    Creates an agent with optional explicit ID. If no ID provided, generates one.
    Initializes empty cache for the agent.

    Args:
        request_body: CreateAgentRequest with optional agent_id
        request: FastAPI request (for accessing app state)

    Returns:
        AgentResponse with agent_id and status

    Raises:
        HTTPException: On creation errors
    """
    # Get app dependencies (with null check)
    semantic_state = get_semantic_state(request)
    cache_store: AgentCacheStore = semantic_state.cache_store

    try:
        # Generate or use provided agent ID
        agent_id = request_body.agent_id or f"agent_{uuid.uuid4().hex[:16]}"
        logger.info(f"POST /v1/agents: Creating agent {agent_id}")

        # Check if agent already exists
        existing_cache = cache_store.load(agent_id)
        if existing_cache:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Agent {agent_id} already exists",
            )

        # Create empty cache entry (will be populated on first generation)
        # For now, just return success - cache will be created on first generate
        logger.info(f"Agent created: {agent_id}")

        return AgentResponse(
            agent_id=agent_id,
            status="active",
            cache_size_tokens=0,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Agent creation error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create agent: {e!s}",
        ) from e


@router.get("/list", status_code=status.HTTP_200_OK)
async def list_agents(request: Request) -> dict:
    """List all cached agents across memory tiers (GET /v1/agents/list).

    Returns union of hot (in-memory) and warm (on-disk) agents with metadata.

    Args:
        request: FastAPI request (for accessing app state)

    Returns:
        Dict with agents list and total count.
    """
    try:
        semantic_state = get_semantic_state(request)
        cache_store: AgentCacheStore = semantic_state.cache_store

        agents = cache_store.list_all_agents()

        return {
            "agents": agents,
            "total": len(agents),
        }

    except Exception as e:
        logger.error(f"Failed to list agents: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list agents: {e!s}",
        ) from e


@router.get("/stats", status_code=status.HTTP_200_OK)
async def get_agent_stats(request: Request) -> dict:
    """Get aggregate agent cache statistics (GET /v1/agents/stats).

    Returns tier counts, pool utilization, and total cache size.

    Args:
        request: FastAPI request (for accessing app state)

    Returns:
        Dict with hot_count, warm_count, total_count, dirty_count,
        pool_utilization_pct, total_cache_size_mb.
    """
    try:
        semantic_state = get_semantic_state(request)
        cache_store: AgentCacheStore = semantic_state.cache_store
        block_pool = semantic_state.block_pool

        # Get all agents
        agents = cache_store.list_all_agents()

        # Count by tier
        hot_count = sum(1 for a in agents if a["tier"] == "hot")
        warm_count = sum(1 for a in agents if a["tier"] == "warm")
        dirty_count = sum(1 for a in agents if a.get("dirty", False))

        # Calculate total cache size
        total_size_bytes = sum(a["file_size_bytes"] for a in agents)
        total_size_mb = total_size_bytes / (1024 * 1024)

        # Pool utilization
        pool_utilization = 0.0
        if block_pool.total_blocks > 0:
            used_blocks = block_pool.total_blocks - len(block_pool.free_list)
            pool_utilization = (used_blocks / block_pool.total_blocks) * 100

        return {
            "hot_count": hot_count,
            "warm_count": warm_count,
            "total_count": len(agents),
            "dirty_count": dirty_count,
            "pool_utilization_pct": round(pool_utilization, 2),
            "total_cache_size_mb": round(total_size_mb, 2),
        }

    except Exception as e:
        logger.error(f"Failed to get agent stats: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get agent stats: {e!s}",
        ) from e


@router.get("/{agent_id}", status_code=status.HTTP_200_OK)
async def get_agent(agent_id: str, request: Request) -> AgentResponse:
    """Get agent info (GET /v1/agents/{agent_id}).

    Returns agent status and cache size.

    Args:
        agent_id: Agent identifier
        request: FastAPI request (for accessing app state)

    Returns:
        AgentResponse with agent status

    Raises:
        HTTPException: If agent not found or error occurs
    """
    logger.info(f"GET /v1/agents/{agent_id}")

    # Get app dependencies (with null check)
    semantic_state = get_semantic_state(request)
    cache_store: AgentCacheStore = semantic_state.cache_store

    try:
        # Load agent cache
        cached_blocks = cache_store.load(agent_id)

        if not cached_blocks:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Agent {agent_id} not found",
            )

        return AgentResponse(
            agent_id=agent_id,
            status="active",
            cache_size_tokens=cached_blocks.total_tokens,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Agent retrieval error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve agent: {e!s}",
        ) from e


@router.post("/{agent_id}/generate", status_code=status.HTTP_200_OK)
async def generate(
    agent_id: str, request_body: GenerateRequest, request: Request
) -> GenerateResponse:
    """Generate text for agent (POST /v1/agents/{agent_id}/generate).

    Generates text using agent's cache. Creates agent if doesn't exist.

    Args:
        agent_id: Agent identifier
        request_body: GenerateRequest with prompt and parameters
        request: FastAPI request (for accessing app state)

    Returns:
        GenerateResponse with generated text

    Raises:
        HTTPException: On generation errors
    """
    logger.info(f"POST /v1/agents/{agent_id}/generate: {len(request_body.prompt)} chars")

    # Get app dependencies (with null check)
    semantic_state = get_semantic_state(request)
    batch_engine: BlockPoolBatchEngine = semantic_state.batch_engine
    cache_store: AgentCacheStore = semantic_state.cache_store

    try:
        # Load agent cache (may be empty for new agent)
        cached_blocks = cache_store.load(agent_id)
        if cached_blocks:
            logger.info(f"Cache hit: {agent_id} ({cached_blocks.total_tokens} tokens)")
        else:
            logger.info(f"New agent or cache miss: {agent_id}")

        # Submit to batch engine (run in executor to avoid blocking)
        uid = await asyncio.to_thread(
            batch_engine.submit,
            agent_id=agent_id,
            prompt=request_body.prompt,
            cache=cached_blocks,
            max_tokens=request_body.max_tokens,
        )
        logger.debug(f"Submitted generation: uid={uid}")

        # Invalidate hot cache entry if we passed in a cache
        # batch_engine clears Q4 blocks after reconstruction
        if cached_blocks is not None:
            cache_store.invalidate_hot(agent_id)

        # Execute generation (run in executor to avoid blocking)
        completion = await asyncio.to_thread(run_step_for_uid, batch_engine, uid)
        if completion:
            logger.debug(
                f"Generation complete: {completion.token_count} tokens, "
                f"finish_reason={completion.finish_reason}"
            )

        if completion is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Generation failed - no completion returned",
            )

        # Save updated cache
        updated_blocks = batch_engine.get_agent_blocks(agent_id)
        if updated_blocks:
            cache_store.save(agent_id, updated_blocks)
            logger.debug(f"Saved cache: {agent_id} ({updated_blocks.total_tokens} tokens)")

        # Return response (reuse updated_blocks to avoid redundant lookup)
        response = GenerateResponse(
            text=completion.text,
            tokens_generated=completion.token_count,
            finish_reason=completion.finish_reason,
            cache_size_tokens=updated_blocks.total_tokens if updated_blocks else 0,
        )

        logger.info(f"Response: {response.tokens_generated} tokens generated")
        return response

    except PoolExhaustedError as e:
        logger.error(f"Pool exhausted: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Server capacity exceeded: {e!s}",
        ) from e
    except SemanticError as e:
        logger.error(f"Domain error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e
    except TimeoutError as e:
        logger.error(f"Generation timeout: {e}")
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=f"Generation timed out: {e!s}",
        ) from e
    except Exception as e:
        logger.error(f"Generation error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        ) from e


@router.delete("/{agent_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_agent(
    agent_id: str,
    request: Request,
    evict_only: bool = False,
):
    """Delete agent (DELETE /v1/agents/{agent_id}?evict_only=true).

    Removes agent cache from memory and optionally disk.

    Args:
        agent_id: Agent identifier
        request: FastAPI request (for accessing app state)
        evict_only: If True, evict from hot tier but keep disk file.
                    Used for testing warm cache reload.

    Returns:
        No content (204)

    Raises:
        HTTPException: If agent not found or error occurs
    """
    mode = "evict" if evict_only else "delete"
    logger.info(f"DELETE /v1/agents/{agent_id} (mode={mode})")

    # Get app dependencies (with null check)
    semantic_state = get_semantic_state(request)
    cache_store: AgentCacheStore = semantic_state.cache_store
    batch_engine: BlockPoolBatchEngine = semantic_state.batch_engine

    try:
        # Check if agent exists
        cached_blocks = cache_store.load(agent_id)
        if not cached_blocks:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Agent {agent_id} not found",
            )

        # CRITICAL: Free pool blocks BEFORE removing from tracking
        # Old code just deleted from dict, leaking pool blocks
        block_pool = semantic_state.block_pool
        freed_count = block_pool.free_agent_blocks(agent_id)
        if freed_count > 0:
            logger.info(f"Freed {freed_count} pool blocks for agent {agent_id}")

        # Remove from batch engine's active agents
        if batch_engine.remove_agent_blocks(agent_id):
            logger.debug(f"Removed {agent_id} from batch engine")

        # Delete from cache store (memory and optionally disk)
        cache_store.delete(agent_id, keep_disk=evict_only)

        # Explicitly free GPU memory held by cached tensors
        del cached_blocks
        gc.collect()
        if mx is not None:
            mx.clear_cache()

        if evict_only:
            logger.info(f"Agent evicted to disk: {agent_id}")
        else:
            logger.info(f"Agent deleted: {agent_id}")

        return  # 204 No Content

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Agent deletion error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete agent: {e!s}",
        ) from e
