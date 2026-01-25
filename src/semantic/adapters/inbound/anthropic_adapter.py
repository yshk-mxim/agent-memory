"""Anthropic Messages API adapter (POST /v1/messages).

Implements the Anthropic Messages API with:
- Non-streaming generation
- SSE streaming (Day 4)
- Tool use support
- Extended thinking
- Prompt caching
"""

import hashlib
import logging
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, Request, status

from semantic.adapters.inbound.request_models import (
    Message,
    MessagesRequest,
    MessagesResponse,
    TextContentBlock,
    Usage,
)
from semantic.application.agent_cache_store import AgentCacheStore
from semantic.application.batch_engine import BlockPoolBatchEngine
from semantic.domain.errors import PoolExhaustedError, SemanticError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["anthropic"])


def generate_agent_id_from_tokens(tokens: list[int]) -> str:
    """Generate agent ID from token prefix for cache lookup.

    Uses first 100 tokens for stability (prefix matching).

    Args:
        tokens: Full token sequence

    Returns:
        Agent ID in format "msg_{hash}"
    """
    prefix = tokens[:100]
    hash_val = hashlib.sha256(str(prefix).encode()).hexdigest()[:16]
    return f"msg_{hash_val}"


def messages_to_prompt(messages: list[Message], system: str | list[Any] = "") -> str:
    """Convert Anthropic messages to prompt string.

    Args:
        messages: List of user/assistant messages
        system: System prompt (string or blocks)

    Returns:
        Formatted prompt string for tokenization
    """
    lines = []

    # Add system prompt if present
    if system:
        if isinstance(system, str):
            lines.append(f"System: {system}\n")
        else:
            # System blocks
            for block in system:
                if hasattr(block, "text"):
                    lines.append(f"System: {block.text}\n")

    # Add conversation messages
    for msg in messages:
        if isinstance(msg.content, str):
            lines.append(f"{msg.role.capitalize()}: {msg.content}")
        else:
            # Content blocks
            for block in msg.content:
                if hasattr(block, "text"):
                    lines.append(f"{msg.role.capitalize()}: {block.text}")
                elif hasattr(block, "thinking"):
                    lines.append(f"{msg.role.capitalize()} (thinking): {block.thinking}")

    # Add assistant prefix for continuation
    lines.append("Assistant:")

    return "\n".join(lines)


@router.post("/messages", status_code=status.HTTP_200_OK)
async def create_message(request_body: MessagesRequest, request: Request) -> MessagesResponse:
    """Create a message (POST /v1/messages).

    Non-streaming endpoint that generates a complete response.

    Args:
        request_body: Validated MessagesRequest
        request: FastAPI request (for accessing app state)

    Returns:
        MessagesResponse with generated content

    Raises:
        HTTPException: On generation errors
    """
    logger.info(f"POST /v1/messages: model={request_body.model}, stream={request_body.stream}")

    # Get app dependencies
    batch_engine: BlockPoolBatchEngine = request.app.state.semantic.batch_engine
    cache_store: AgentCacheStore = request.app.state.semantic.cache_store

    try:
        # 1. Convert messages to prompt
        prompt = messages_to_prompt(request_body.messages, request_body.system)
        logger.debug(f"Prompt length: {len(prompt)} chars")

        # 2. Tokenize to get agent ID
        tokenizer = batch_engine._tokenizer
        tokens = tokenizer.encode(prompt)
        agent_id = generate_agent_id_from_tokens(tokens)
        logger.debug(f"Agent ID: {agent_id}, tokens: {len(tokens)}")

        # 3. Check cache store for existing cache
        cached_blocks = cache_store.load(agent_id)
        if cached_blocks:
            logger.info(f"Cache hit: {agent_id} ({cached_blocks.total_tokens} tokens)")
        else:
            logger.info(f"Cache miss: {agent_id}")

        # 4. Submit to batch engine
        uid = batch_engine.submit(
            agent_id=agent_id,
            prompt=prompt,
            cache=cached_blocks,
            max_tokens=request_body.max_tokens,
        )
        logger.debug(f"Submitted generation: uid={uid}")

        # 5. Execute generation (step until complete)
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

        # 6. Save updated cache
        if agent_id in batch_engine._agent_blocks:
            updated_blocks = batch_engine._agent_blocks[agent_id]
            cache_store.save(agent_id, updated_blocks)
            logger.debug(f"Saved cache: {agent_id} ({updated_blocks.total_tokens} tokens)")

        # 7. Format response
        response = MessagesResponse(
            id=f"msg_{uuid.uuid4().hex[:24]}",
            content=[TextContentBlock(text=completion.text)],
            model=request_body.model,
            stop_reason=(
                "end_turn" if completion.finish_reason == "stop" else "max_tokens"
            ),
            usage=Usage(
                input_tokens=len(tokens),
                output_tokens=completion.token_count,
                cache_creation_input_tokens=0 if cached_blocks else len(tokens),
                cache_read_input_tokens=len(tokens) if cached_blocks else 0,
            ),
        )

        logger.info(
            f"Response: {len(response.content)} blocks, {response.usage.output_tokens} output tokens"
        )
        return response

    except PoolExhaustedError as e:
        logger.error(f"Pool exhausted: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Server capacity exceeded: {str(e)}",
        )
    except SemanticError as e:
        logger.error(f"Domain error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        )
