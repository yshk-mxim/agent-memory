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

from semantic.adapters.inbound.adapter_helpers import (
    get_semantic_state,
    run_step_for_uid,
)
from semantic.adapters.inbound.request_models import (
    AgentResponse,
    CreateAgentRequest,
    GenerateRequest,
    GenerateResponse,
)
from semantic.application.agent_cache_store import AgentCacheStore
from semantic.application.batch_engine import BlockPoolBatchEngine
from semantic.domain.errors import PoolExhaustedError, SemanticError

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

        # Return response
        agent_blocks_for_size = batch_engine.get_agent_blocks(agent_id)
        response = GenerateResponse(
            text=completion.text,
            tokens_generated=completion.token_count,
            finish_reason=completion.finish_reason,
            cache_size_tokens=agent_blocks_for_size.total_tokens if agent_blocks_for_size else 0,
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
async def delete_agent(agent_id: str, request: Request):
    """Delete agent (DELETE /v1/agents/{agent_id}).

    Removes agent cache from memory and disk.

    Args:
        agent_id: Agent identifier
        request: FastAPI request (for accessing app state)

    Returns:
        No content (204)

    Raises:
        HTTPException: If agent not found or error occurs
    """
    logger.info(f"DELETE /v1/agents/{agent_id}")

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

        # Delete from cache store (memory and disk)
        cache_store.delete(agent_id)

        # Explicitly free GPU memory held by cached tensors
        del cached_blocks
        gc.collect()
        if mx is not None:
            mx.clear_cache()

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
