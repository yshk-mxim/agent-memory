"""Direct Agent API adapter (POST /v1/agents/*).

Provides low-level CRUD operations for agent cache management:
- Create agents with explicit IDs
- Generate text with direct agent access
- Query agent cache status
- Delete agents
"""

import logging
import uuid

from fastapi import APIRouter, HTTPException, Request, status

from semantic.adapters.inbound.request_models import (
    AgentResponse,
    CreateAgentRequest,
    GenerateRequest,
    GenerateResponse,
)
from semantic.application.agent_cache_store import AgentCacheStore
from semantic.application.batch_engine import BlockPoolBatchEngine
from semantic.domain.errors import PoolExhaustedError, SemanticError

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
    # Get app dependencies
    cache_store: AgentCacheStore = request.app.state.semantic.cache_store

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
        )


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

    # Get app dependencies
    cache_store: AgentCacheStore = request.app.state.semantic.cache_store

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
        )


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

    # Get app dependencies
    batch_engine: BlockPoolBatchEngine = request.app.state.semantic.batch_engine
    cache_store: AgentCacheStore = request.app.state.semantic.cache_store

    try:
        # Load agent cache (may be empty for new agent)
        cached_blocks = cache_store.load(agent_id)
        if cached_blocks:
            logger.info(f"Cache hit: {agent_id} ({cached_blocks.total_tokens} tokens)")
        else:
            logger.info(f"New agent or cache miss: {agent_id}")

        # Submit to batch engine
        uid = batch_engine.submit(
            agent_id=agent_id,
            prompt=request_body.prompt,
            cache=cached_blocks,
            max_tokens=request_body.max_tokens,
        )
        logger.debug(f"Submitted generation: uid={uid}")

        # Execute generation
        completion = None
        for result in batch_engine.step():
            if result.uid == uid:
                completion = result
                logger.debug(
                    f"Generation complete: {result.token_count} tokens, "
                    f"finish_reason={result.finish_reason}"
                )
                break

        if completion is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Generation failed - no completion returned",
            )

        # Save updated cache
        if agent_id in batch_engine._agent_blocks:
            updated_blocks = batch_engine._agent_blocks[agent_id]
            cache_store.save(agent_id, updated_blocks)
            logger.debug(f"Saved cache: {agent_id} ({updated_blocks.total_tokens} tokens)")

        # Return response
        response = GenerateResponse(
            text=completion.text,
            tokens_generated=completion.token_count,
            finish_reason=completion.finish_reason,
            cache_size_tokens=(
                batch_engine._agent_blocks[agent_id].total_tokens
                if agent_id in batch_engine._agent_blocks
                else 0
            ),
        )

        logger.info(f"Response: {response.tokens_generated} tokens generated")
        return response

    except PoolExhaustedError as e:
        logger.error(f"Pool exhausted: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Server capacity exceeded: {e!s}",
        )
    except SemanticError as e:
        logger.error(f"Domain error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Generation error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        )


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

    # Get app dependencies
    cache_store: AgentCacheStore = request.app.state.semantic.cache_store
    batch_engine: BlockPoolBatchEngine = request.app.state.semantic.batch_engine

    try:
        # Check if agent exists
        cached_blocks = cache_store.load(agent_id)
        if not cached_blocks:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Agent {agent_id} not found",
            )

        # Remove from batch engine's active agents
        if agent_id in batch_engine._agent_blocks:
            del batch_engine._agent_blocks[agent_id]
            logger.debug(f"Removed {agent_id} from batch engine")

        # Delete from cache store (memory and disk)
        cache_store.delete(agent_id)
        logger.info(f"Agent deleted: {agent_id}")

        return  # 204 No Content

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Agent deletion error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete agent: {e!s}",
        )
