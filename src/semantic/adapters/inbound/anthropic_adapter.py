"""Anthropic Messages API adapter (POST /v1/messages).

Implements the Anthropic Messages API with:
- Non-streaming generation
- SSE streaming
- Tool use support
- Extended thinking
- Prompt caching
"""

import asyncio
import hashlib
import json
import logging
import uuid
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, HTTPException, Request, status
from sse_starlette.sse import EventSourceResponse

from semantic.adapters.inbound.adapter_helpers import (
    get_semantic_state,
    run_step_for_uid,
    try_parse_json_at,
)
from semantic.adapters.inbound.request_models import (
    ContentBlockDeltaEvent,
    ContentBlockStartEvent,
    ContentBlockStopEvent,
    CountTokensRequest,
    CountTokensResponse,
    Message,
    MessageDeltaEvent,
    MessagesRequest,
    MessagesResponse,
    MessageStartEvent,
    MessageStopEvent,
    TextContentBlock,
    ToolUseContentBlock,
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


def parse_tool_calls(text: str) -> tuple[str, list[dict[str, Any]]]:
    """Parse tool calls from model output.

    Looks for JSON patterns like:
    {"tool_use": {"name": "read_file", "input": {"path": "test.py"}}}

    Uses proper JSON parsing instead of regex to handle nested objects.

    Args:
        text: Model generated text

    Returns:
        Tuple of (remaining_text, list of tool call dicts)
        Tool call dict contains: {"name": str, "input": dict}
    """
    tool_calls = []
    found_ranges = []  # Track (start, end) ranges to remove

    # Find all potential JSON start positions with "tool_use" key
    search_pattern = '{"tool_use"'
    pos = 0

    while True:
        start = text.find(search_pattern, pos)
        if start == -1:
            break

        parsed, end = try_parse_json_at(text, start)
        if parsed and "tool_use" in parsed:
            tool_data = parsed["tool_use"]
            if isinstance(tool_data, dict) and "name" in tool_data and "input" in tool_data:
                tool_calls.append({
                    "name": tool_data["name"],
                    "input": tool_data["input"],
                })
                found_ranges.append((start, end))
                pos = end  # Continue searching after this match
            else:
                pos = start + 1
        else:
            pos = start + 1

    # Remove found tool calls from text (in reverse order to preserve indices)
    remaining_text = text
    for start, end in sorted(found_ranges, reverse=True):
        remaining_text = remaining_text[:start] + remaining_text[end:]

    return remaining_text.strip(), tool_calls


def messages_to_prompt(  # noqa: PLR0912, C901
    messages: list[Message],
    system: str | list[Any] = "",
    tools: list[Any] | None = None,
) -> str:
    """Convert Anthropic messages to prompt string.

    Args:
        messages: List of user/assistant messages
        system: System prompt (string or blocks)
        tools: Optional list of tool definitions

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

    # Add tool definitions if present
    if tools:
        lines.append("\nAvailable Tools:")
        for tool in tools:
            tool_def = {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.input_schema,
            }
            lines.append(json.dumps(tool_def, indent=2))
        lines.append(
            '\nTo use a tool, output JSON: {"tool_use": {"name": "<tool_name>", '
            '"input": {<parameters>}}}\n'
        )

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
                elif hasattr(block, "tool_use_id"):
                    # ToolResultContentBlock
                    result_content = (
                        block.content if isinstance(block.content, str)
                        else json.dumps(block.content)
                    )
                    status = "ERROR" if block.is_error else "SUCCESS"
                    lines.append(
                        f"{msg.role.capitalize()} [Tool Result - {status}]: "
                        f"{result_content}"
                    )
                elif hasattr(block, "name") and hasattr(block, "input"):
                    # ToolUseContentBlock (in assistant messages)
                    tool_call = {"name": block.name, "input": block.input}
                    lines.append(
                        f"{msg.role.capitalize()} [Tool Call]: "
                        f'{json.dumps(tool_call)}'
                    )

    # Add assistant prefix for continuation
    lines.append("Assistant:")

    return "\n".join(lines)


async def stream_generation(  # noqa: C901, PLR0912
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
            prompt=messages_to_prompt(
                request_body.messages,
                request_body.system,
                request_body.tools if request_body.tools else None,
            ),
            cache=cached_blocks,
            max_tokens=request_body.max_tokens,
        )
        logger.debug(f"Submitted streaming generation: uid={uid}")

        # Invalidate hot cache entry if we passed in a cache
        # batch_engine clears the Q4 blocks after reconstruction, so the
        # shared reference in hot_cache is now stale. Invalidating prevents
        # unnecessary has_data checks on future loads (disk backup remains valid).
        if cached_blocks is not None:
            cache_store.invalidate_hot(agent_id)

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

        # Yield content_block_stop event for text
        yield {
            "event": "content_block_stop",
            "data": json.dumps(ContentBlockStopEvent(index=0).model_dump()),
        }

        # Parse for tool calls
        _remaining_text, tool_calls = parse_tool_calls(accumulated_text)

        # Yield tool_use content blocks if any
        content_block_index = 1  # Text block is index 0
        for tool_call in tool_calls:
            tool_use_block = ToolUseContentBlock(
                id=f"toolu_{uuid.uuid4().hex[:24]}",
                name=tool_call["name"],
                input=tool_call["input"],
            )

            # Yield content_block_start for tool
            yield {
                "event": "content_block_start",
                "data": json.dumps(
                    ContentBlockStartEvent(
                        index=content_block_index,
                        content_block=tool_use_block,
                    ).model_dump()
                ),
            }

            # Yield content_block_stop for tool
            yield {
                "event": "content_block_stop",
                "data": json.dumps(
                    ContentBlockStopEvent(index=content_block_index).model_dump()
                ),
            }

            content_block_index += 1

        # Save updated cache
        updated_blocks = batch_engine.get_agent_blocks(agent_id)
        if updated_blocks:
            cache_store.save(agent_id, updated_blocks)

        # Determine stop_reason
        if tool_calls:
            stop_reason = "tool_use"
        elif completion.finish_reason == "stop":
            stop_reason = "end_turn"
        else:
            stop_reason = "max_tokens"

        # Yield message_delta event
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

    except asyncio.CancelledError:
        # Client disconnected mid-stream - clean up gracefully
        logger.info(f"Streaming cancelled for agent {agent_id} (client disconnect)")
        # Don't yield anything - client is gone
        raise  # Re-raise to properly cancel the coroutine
    except Exception as e:
        logger.error(f"Streaming error: {e}", exc_info=True)
        # Yield error event
        yield {
            "event": "error",
            "data": json.dumps({"error": {"type": "internal_error", "message": str(e)}}),
        }


@router.post("/messages", status_code=status.HTTP_200_OK)
async def create_message(request_body: MessagesRequest, request: Request):  # noqa: C901, PLR0912, PLR0915
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
    logger.debug(f"Messages: {request_body.messages}")

    # Get app dependencies (with null check)
    semantic_state = get_semantic_state(request)
    batch_engine: BlockPoolBatchEngine = semantic_state.batch_engine
    cache_store: AgentCacheStore = semantic_state.cache_store

    try:
        # 1. Convert messages to prompt (including tools if present)
        prompt = messages_to_prompt(
            request_body.messages,
            request_body.system,
            request_body.tools if request_body.tools else None,
        )
        logger.debug(f"Prompt length: {len(prompt)} chars")
        logger.debug(f"Full prompt:\n{prompt}")

        # 2. Tokenize (run in executor to avoid blocking)
        tokenizer = batch_engine.tokenizer
        tokens = await asyncio.to_thread(tokenizer.encode, prompt)

        # 3. Determine agent_id: use X-Session-ID for session-based caching,
        # fall back to token-based ID for stateless requests.
        # Session-based lookup enables prefix caching across conversation turns.
        session_id = request.headers.get("X-Session-ID")
        if session_id:
            agent_id = f"sess_{session_id}"
            logger.debug(f"Session-based agent ID: {agent_id}, tokens: {len(tokens)}")
        else:
            agent_id = generate_agent_id_from_tokens(tokens)
            logger.debug(f"Token-based agent ID: {agent_id}, tokens: {len(tokens)}")

        # 4. Check cache store for existing cache
        cached_blocks = cache_store.load(agent_id)
        if cached_blocks:
            logger.info(f"Cache hit: {agent_id} ({cached_blocks.total_tokens} tokens)")
        else:
            logger.info(f"Cache miss: {agent_id}")

        # 5. Handle streaming vs non-streaming
        if request_body.stream:
            # Return SSE stream
            logger.info("Returning SSE stream")
            return EventSourceResponse(
                stream_generation(
                    request_body, batch_engine, cache_store, tokens, agent_id, cached_blocks
                )
            )

        # 6. Submit to batch engine (run in executor to avoid blocking)
        uid = await asyncio.to_thread(
            batch_engine.submit,
            agent_id=agent_id,
            prompt=prompt,
            cache=cached_blocks,
            max_tokens=request_body.max_tokens,
        )
        logger.debug(f"Submitted generation: uid={uid}")

        # Invalidate hot cache entry if we passed in a cache
        # batch_engine clears Q4 blocks after reconstruction
        if cached_blocks is not None:
            cache_store.invalidate_hot(agent_id)

        # 7. Execute generation (step until complete - run in executor)
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

        # 8. Save updated cache
        updated_blocks = batch_engine.get_agent_blocks(agent_id)
        if updated_blocks:
            cache_store.save(agent_id, updated_blocks)
            logger.debug(f"Saved cache: {agent_id} ({updated_blocks.total_tokens} tokens)")

        # 9. Parse for tool calls
        remaining_text, tool_calls = parse_tool_calls(completion.text)

        # 10. Format response
        content_blocks = []

        # Add text block if there's remaining text
        if remaining_text.strip():
            content_blocks.append(TextContentBlock(text=remaining_text))

        # Add tool_use blocks
        for tool_call in tool_calls:
            tool_use_block = ToolUseContentBlock(
                id=f"toolu_{uuid.uuid4().hex[:24]}",
                name=tool_call["name"],
                input=tool_call["input"],
            )
            content_blocks.append(tool_use_block)

        # Determine stop_reason
        if tool_calls:
            stop_reason = "tool_use"
        elif completion.finish_reason == "stop":
            stop_reason = "end_turn"
        else:
            stop_reason = "max_tokens"

        response = MessagesResponse(
            id=f"msg_{uuid.uuid4().hex[:24]}",
            content=content_blocks,
            model=request_body.model,
            stop_reason=stop_reason,
            usage=Usage(
                input_tokens=len(tokens),
                output_tokens=completion.token_count,
                cache_creation_input_tokens=0 if cached_blocks else len(tokens),
                cache_read_input_tokens=len(tokens) if cached_blocks else 0,
            ),
        )

        logger.info(
            f"Response: {len(response.content)} blocks, "
            f"{response.usage.output_tokens} output tokens"
        )
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
        logger.error(f"Unexpected error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        ) from e


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

    # Get batch engine for tokenizer access (with null check)
    semantic_state = get_semantic_state(request)
    batch_engine: BlockPoolBatchEngine = semantic_state.batch_engine

    try:
        # Convert messages to prompt
        prompt = messages_to_prompt(request_body.messages, request_body.system)

        # Add tool descriptions if present
        if request_body.tools:
            tool_descriptions = "\n".join(
                f"Tool: {tool.name} - {tool.description}" for tool in request_body.tools
            )
            prompt = f"{tool_descriptions}\n\n{prompt}"

        # Tokenize (run in executor to avoid blocking)
        tokenizer = batch_engine.tokenizer
        tokens = await asyncio.to_thread(tokenizer.encode, prompt)

        logger.info(f"Token count: {len(tokens)}")
        return CountTokensResponse(input_tokens=len(tokens))

    except Exception as e:
        logger.error(f"Token counting error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to count tokens: {e!s}",
        ) from e
