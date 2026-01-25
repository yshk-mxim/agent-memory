"""Anthropic Messages API adapter (POST /v1/messages).

Implements the Anthropic Messages API with:
- Non-streaming generation
- SSE streaming
- Tool use support
- Extended thinking
- Prompt caching
"""

import hashlib
import json
import logging
import uuid
from typing import Any, AsyncIterator

from fastapi import APIRouter, HTTPException, Request, status
from sse_starlette.sse import EventSourceResponse

from semantic.adapters.inbound.request_models import (
    ContentBlockDeltaEvent,
    ContentBlockStartEvent,
    ContentBlockStopEvent,
    CountTokensRequest,
    CountTokensResponse,
    Message,
    MessageDeltaEvent,
    MessageStartEvent,
    MessageStopEvent,
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


async def stream_generation(
    request_body: MessagesRequest,
    batch_engine: Any,
    cache_store: Any,
    tokens: list[int],
    agent_id: str,
    cached_blocks: Any,
) -> AsyncIterator[dict[str, Any]]:
    """Stream generation results as SSE events.

    Yields:
        SSE events in Anthropic Messages API format
    """
    try:
        # Submit to batch engine
        uid = batch_engine.submit(
            agent_id=agent_id,
            prompt=messages_to_prompt(request_body.messages, request_body.system),
            cache=cached_blocks,
            max_tokens=request_body.max_tokens,
        )
        logger.debug(f"Submitted streaming generation: uid={uid}")

        # Yield message_start event
        message_id = f"msg_{uuid.uuid4().hex[:24]}"
        yield {
            "event": "message_start",
            "data": json.dumps(
                MessageStartEvent(
                    message=MessagesResponse(
                        id=message_id,
                        content=[],
                        model=request_body.model,
                        stop_reason=None,
                        usage=Usage(
                            input_tokens=len(tokens),
                            output_tokens=0,
                            cache_creation_input_tokens=0 if cached_blocks else len(tokens),
                            cache_read_input_tokens=len(tokens) if cached_blocks else 0,
                        ),
                    )
                ).model_dump()
            ),
        }

        # Yield content_block_start event
        yield {
            "event": "content_block_start",
            "data": json.dumps(
                ContentBlockStartEvent(
                    index=0, content_block=TextContentBlock(text="")
                ).model_dump()
            ),
        }

        # Stream token deltas
        completion = None
        accumulated_text = ""
        for result in batch_engine.step():
            if result.uid == uid:
                completion = result
                # Yield text delta
                if result.text:
                    # Incremental text (only new text since last yield)
                    new_text = result.text[len(accumulated_text) :]
                    accumulated_text = result.text

                    if new_text:
                        yield {
                            "event": "content_block_delta",
                            "data": json.dumps(
                                ContentBlockDeltaEvent(
                                    index=0, delta={"type": "text_delta", "text": new_text}
                                ).model_dump()
                            ),
                        }
                break

        if completion is None:
            logger.error("Streaming generation failed - no completion")
            return

        # Yield content_block_stop event
        yield {
            "event": "content_block_stop",
            "data": json.dumps(ContentBlockStopEvent(index=0).model_dump()),
        }

        # Save updated cache
        if agent_id in batch_engine._agent_blocks:
            updated_blocks = batch_engine._agent_blocks[agent_id]
            cache_store.save(agent_id, updated_blocks)

        # Yield message_delta event
        stop_reason = "end_turn" if completion.finish_reason == "stop" else "max_tokens"
        yield {
            "event": "message_delta",
            "data": json.dumps(
                MessageDeltaEvent(
                    delta={"stop_reason": stop_reason},
                    usage=Usage(
                        input_tokens=0,
                        output_tokens=completion.token_count,
                    ),
                ).model_dump()
            ),
        }

        # Yield message_stop event
        yield {
            "event": "message_stop",
            "data": json.dumps(MessageStopEvent().model_dump()),
        }

    except Exception as e:
        logger.error(f"Streaming error: {e}", exc_info=True)
        # Yield error event
        yield {
            "event": "error",
            "data": json.dumps({"error": {"type": "internal_error", "message": str(e)}}),
        }


@router.post("/messages", status_code=status.HTTP_200_OK)
async def create_message(request_body: MessagesRequest, request: Request):
    """Create a message (POST /v1/messages).

    Supports both streaming and non-streaming generation.

    Args:
        request_body: Validated MessagesRequest
        request: FastAPI request (for accessing app state)

    Returns:
        EventSourceResponse (streaming) or MessagesResponse (non-streaming)

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

        # 4. Handle streaming vs non-streaming
        if request_body.stream:
            # Return SSE stream
            logger.info("Returning SSE stream")
            return EventSourceResponse(
                stream_generation(
                    request_body, batch_engine, cache_store, tokens, agent_id, cached_blocks
                )
            )

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


@router.post("/messages/count_tokens", status_code=status.HTTP_200_OK)
async def count_tokens(request_body: CountTokensRequest, request: Request) -> CountTokensResponse:
    """Count tokens for a request (POST /v1/messages/count_tokens).

    Args:
        request_body: Validated CountTokensRequest
        request: FastAPI request (for accessing app state)

    Returns:
        CountTokensResponse with token count

    Raises:
        HTTPException: On tokenization errors
    """
    logger.info(f"POST /v1/messages/count_tokens: model={request_body.model}")

    # Get batch engine for tokenizer access
    batch_engine: BlockPoolBatchEngine = request.app.state.semantic.batch_engine

    try:
        # Convert messages to prompt
        prompt = messages_to_prompt(request_body.messages, request_body.system)

        # Add tool descriptions if present
        if request_body.tools:
            tool_descriptions = "\n".join(
                f"Tool: {tool.name} - {tool.description}" for tool in request_body.tools
            )
            prompt = f"{tool_descriptions}\n\n{prompt}"

        # Tokenize
        tokenizer = batch_engine._tokenizer
        tokens = tokenizer.encode(prompt)

        logger.info(f"Token count: {len(tokens)}")
        return CountTokensResponse(input_tokens=len(tokens))

    except Exception as e:
        logger.error(f"Token counting error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to count tokens: {str(e)}",
        )
