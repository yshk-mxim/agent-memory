r"""OpenAI Chat Completions API adapter (POST /v1/chat/completions).

Implements the OpenAI Chat Completions API with:
- Non-streaming generation
- Streaming (SSE format: data: {...}\ndata: [DONE])
- Session ID extension for persistent caching
"""

import asyncio
import contextlib
import hashlib
import json
import logging
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, HTTPException, Request, status
from sse_starlette.sse import EventSourceResponse

from semantic.adapters.config.settings import get_settings
from semantic.adapters.inbound.adapter_helpers import (
    get_semantic_state,
    run_step_for_uid,
    tokenize_with_chat_template,
    try_parse_json_at,
)
from semantic.adapters.inbound.request_models import (
    ChatCompletionsRequest,
    ChatCompletionsResponse,
    OpenAIChatChoice,
    OpenAIChatCompletionUsage,
    OpenAIChatMessage,
)
from semantic.application.agent_cache_store import AgentCacheStore
from semantic.application.batch_engine import BlockPoolBatchEngine
from semantic.domain.errors import PoolExhaustedError, SemanticError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["openai"])


def generate_agent_id_openai(session_id: str | None, tokens: list[int]) -> str:
    """Generate agent ID for OpenAI requests.

    Args:
        session_id: Optional explicit session ID from X-Session-ID header or request
        tokens: Full token sequence

    Returns:
        Agent ID in format "oai_{session_id}" or "oai_{hash}"
    """
    if session_id:
        return f"oai_{session_id}"

    # No session ID - use token prefix hash (like Anthropic)
    prefix = tokens[:100]
    hash_val = hashlib.sha256(str(prefix).encode()).hexdigest()[:16]
    return f"oai_{hash_val}"


def parse_function_calls(text: str) -> tuple[str, list[dict[str, Any]]]:
    """Parse function calls from model output.

    Looks for JSON patterns like:
    {"function_call": {"name": "get_weather", "arguments": {"location": "Paris"}}}

    Uses proper JSON parsing instead of regex to handle nested objects.

    Args:
        text: Model generated text

    Returns:
        Tuple of (remaining_text, list of function call dicts)
        Function call dict contains: {"name": str, "arguments": dict or str}
    """
    function_calls = []
    found_ranges = []  # Track (start, end) ranges to remove

    # Find all potential JSON start positions with "function_call" key
    search_pattern = '{"function_call"'
    pos = 0

    while True:
        start = text.find(search_pattern, pos)
        if start == -1:
            break

        parsed, end = try_parse_json_at(text, start)
        if parsed and "function_call" in parsed:
            func_data = parsed["function_call"]
            if isinstance(func_data, dict) and "name" in func_data and "arguments" in func_data:
                # Arguments might be a dict or a JSON string
                arguments = func_data["arguments"]
                if isinstance(arguments, str):
                    # Try to parse as JSON, leave as string if fails
                    with contextlib.suppress(json.JSONDecodeError):
                        arguments = json.loads(arguments)

                function_calls.append(
                    {
                        "name": func_data["name"],
                        "arguments": arguments,
                    }
                )
                found_ranges.append((start, end))
                pos = end  # Continue searching after this match
            else:
                pos = start + 1
        else:
            pos = start + 1

    # Remove found function calls from text (in reverse order to preserve indices)
    remaining_text = text
    for start, end in sorted(found_ranges, reverse=True):
        remaining_text = remaining_text[:start] + remaining_text[end:]

    return remaining_text.strip(), function_calls


def openai_messages_to_prompt(  # noqa: C901, PLR0912
    messages: list[OpenAIChatMessage],
    tools: list[Any] | None = None,
) -> str:
    """Convert OpenAI messages to prompt string.

    Args:
        messages: List of OpenAI chat messages
        tools: Optional list of tool definitions

    Returns:
        Formatted prompt string for tokenization
    """
    lines = []

    # Add tool definitions if present
    if tools:
        lines.append("\nAvailable Functions:")
        for tool in tools:
            if tool.type == "function":
                func_def = {
                    "name": tool.function.get("name"),
                    "description": tool.function.get("description"),
                    "parameters": tool.function.get("parameters"),
                }
                lines.append(json.dumps(func_def, indent=2))
        lines.append(
            '\nTo call a function, output JSON: {"function_call": {"name": "<function_name>", '
            '"arguments": {<parameters>}}}\n'
        )

    for msg in messages:
        if msg.role == "system":
            lines.append(f"System: {msg.content}")
        elif msg.role == "user":
            lines.append(f"User: {msg.content}")
        elif msg.role == "assistant":
            # Handle tool_calls in assistant messages
            if msg.tool_calls:
                for tool_call in msg.tool_calls:
                    func_call = {
                        "name": tool_call.get("function", {}).get("name"),
                        "arguments": tool_call.get("function", {}).get("arguments"),
                    }
                    lines.append(f"Assistant [Function Call]: {json.dumps(func_call)}")
            elif msg.content:
                lines.append(f"Assistant: {msg.content}")
        elif msg.role == "tool" and msg.content:
            # Tool result
            lines.append(f"Tool [Result]: {msg.content}")

    # Add assistant prefix for continuation
    lines.append("Assistant:")

    return "\n".join(lines)


def openai_messages_to_chat_dicts(  # noqa: C901, PLR0912
    messages: list[OpenAIChatMessage],
    tools: list[Any] | None = None,
) -> list[dict[str, str]]:
    """Convert OpenAI messages to simple chat dicts for chat template.

    Args:
        messages: List of OpenAI chat messages
        tools: Optional list of tool definitions

    Returns:
        List of {"role": ..., "content": ...} dicts
    """
    result: list[dict[str, str]] = []

    for msg in messages:
        if msg.role == "system":
            content = msg.content or ""
            if tools and not result:
                # Append tool definitions to first system message
                tool_lines = [content, "\nAvailable Functions:"]
                for tool in tools:
                    if tool.type == "function":
                        func_def = {
                            "name": tool.function.get("name"),
                            "description": tool.function.get("description"),
                            "parameters": tool.function.get("parameters"),
                        }
                        tool_lines.append(json.dumps(func_def, indent=2))
                tool_lines.append(
                    "\nTo call a function, output JSON: "
                    '{"function_call": {"name": "<name>", "arguments": {<params>}}}'
                )
                content = "\n".join(tool_lines)
                tools = None  # Don't add again
            result.append({"role": "system", "content": content})
        elif msg.role == "user":
            result.append({"role": "user", "content": msg.content or ""})
        elif msg.role == "assistant":
            if msg.tool_calls:
                parts = []
                for tc in msg.tool_calls:
                    fc = {
                        "name": tc.get("function", {}).get("name"),
                        "arguments": tc.get("function", {}).get("arguments"),
                    }
                    parts.append(f"[Function Call]: {json.dumps(fc)}")
                result.append({"role": "assistant", "content": "\n".join(parts)})
            elif msg.content:
                result.append({"role": "assistant", "content": msg.content})
        elif msg.role == "tool" and msg.content:
            result.append({"role": "user", "content": f"[Tool Result]: {msg.content}"})

    return result


async def _stream_via_scheduler(  # noqa: C901, PLR0912
    request_body: ChatCompletionsRequest,
    batch_engine: Any,
    cache_store: Any,
    tokens: list[int],
    agent_id: str,
    cached_blocks: Any,
    prompt: str,
    max_tokens: int,
    scheduler: Any,
) -> AsyncIterator[dict[str, Any]]:
    """Stream via ConcurrentScheduler for true per-token streaming.

    Uses scheduler.submit_and_stream() which yields StreamDelta objects
    as each token is decoded, enabling interleaved prefill/decode.
    """
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())
    model = request_body.model

    yield {
        "data": json.dumps(
            {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": ""},
                        "finish_reason": None,
                    }
                ],
            }
        )
    }

    accumulated_text = ""
    token_count = 0
    finish_reason_raw = None

    async for delta in scheduler.submit_and_stream(
        agent_id=agent_id,
        prompt_tokens=tokens,
        cache=cached_blocks,
        max_tokens=max_tokens,
        prompt_text=prompt,
        temperature=request_body.temperature,
        top_p=request_body.top_p,
    ):
        new_text = delta.text[len(accumulated_text) :]
        accumulated_text = delta.text
        token_count = delta.token_count
        if new_text:
            yield {
                "data": json.dumps(
                    {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [
                            {"index": 0, "delta": {"content": new_text}, "finish_reason": None}
                        ],
                    }
                )
            }
        if delta.finish_reason is not None:
            finish_reason_raw = delta.finish_reason

    # Parse function calls from accumulated output
    _remaining_text, function_calls = parse_function_calls(accumulated_text)

    if function_calls:
        for func_call in function_calls:
            tool_call_id = f"call_{uuid.uuid4().hex[:24]}"
            arguments = func_call["arguments"]
            if isinstance(arguments, dict):
                arguments = json.dumps(arguments)
            yield {
                "data": json.dumps(
                    {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": 0,
                                            "id": tool_call_id,
                                            "type": "function",
                                            "function": {
                                                "name": func_call["name"],
                                                "arguments": arguments,
                                            },
                                        }
                                    ],
                                },
                                "finish_reason": None,
                            }
                        ],
                    }
                )
            }

    # Save updated cache
    updated_blocks = batch_engine.get_agent_blocks(agent_id)
    if updated_blocks:
        cache_store.save(agent_id, updated_blocks)
        # CRITICAL: Flush to disk immediately before batch_engine clears layer_data
        cache_store.flush_dirty()
        logger.debug(f"Saved and flushed cache: {agent_id} ({updated_blocks.total_tokens} tokens)")

    if function_calls:
        finish_reason = "tool_calls"
    elif finish_reason_raw == "stop":
        finish_reason = "stop"
    else:
        finish_reason = "length"

    yield {
        "data": json.dumps(
            {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
            }
        )
    }
    yield {"data": "[DONE]"}
    logger.info(
        f"Scheduler streaming complete: {token_count} tokens, finish_reason={finish_reason}"
    )


async def stream_chat_completion(  # noqa: C901, PLR0912, PLR0915
    request_body: ChatCompletionsRequest,
    batch_engine: Any,
    cache_store: Any,
    _tokens: list[int],
    agent_id: str,
    cached_blocks: Any,
    prompt: str,
    scheduler: Any | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """Stream chat completion results as SSE events (OpenAI format).

    OpenAI SSE format:
    - data: {"id": "...", "choices": [{"delta": {"content": "..."}, ...}], ...}
    - data: [DONE]

    Yields:
        SSE events in OpenAI Chat Completions format
    """
    try:
        settings = get_settings()
        base_max_tokens = request_body.max_tokens or 256
        max_tokens = base_max_tokens + settings.mlx.reasoning_extra_tokens

        if cached_blocks is not None:
            cache_store.invalidate_hot(agent_id)

        # --- Scheduler path: true per-token streaming ---
        if scheduler is not None:
            async for event in _stream_via_scheduler(
                request_body,
                batch_engine,
                cache_store,
                _tokens,
                agent_id,
                cached_blocks,
                prompt,
                max_tokens,
                scheduler,
            ):
                yield event
            return

        # --- Direct batch engine path (no scheduler) — unsafe for concurrent requests ---
        logger.warning(
            "Using direct batch_engine streaming (no scheduler) — concurrent requests unsafe"
        )
        uid = batch_engine.submit(
            agent_id=agent_id,
            prompt=prompt,
            cache=cached_blocks,
            max_tokens=max_tokens,
            temperature=request_body.temperature,
            top_p=request_body.top_p,
        )
        logger.debug(f"Submitted streaming generation: uid={uid}")

        completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created_timestamp = int(time.time())

        # Yield initial chunk with role
        yield {
            "data": json.dumps(
                {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created_timestamp,
                    "model": request_body.model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant", "content": ""},
                            "finish_reason": None,
                        }
                    ],
                }
            )
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
                            "data": json.dumps(
                                {
                                    "id": completion_id,
                                    "object": "chat.completion.chunk",
                                    "created": created_timestamp,
                                    "model": request_body.model,
                                    "choices": [
                                        {
                                            "index": 0,
                                            "delta": {"content": new_text},
                                            "finish_reason": None,
                                        }
                                    ],
                                }
                            )
                        }
                break

        if completion is None:
            logger.error("Streaming generation failed - no completion")
            return

        # Parse for function calls
        _remaining_text, function_calls = parse_function_calls(accumulated_text)

        # Yield tool_calls if any
        if function_calls:
            for func_call in function_calls:
                tool_call_id = f"call_{uuid.uuid4().hex[:24]}"
                # Ensure arguments is a JSON string
                arguments = func_call["arguments"]
                if isinstance(arguments, dict):
                    arguments = json.dumps(arguments)

                # Yield tool call delta
                yield {
                    "data": json.dumps(
                        {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created_timestamp,
                            "model": request_body.model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {
                                        "tool_calls": [
                                            {
                                                "index": 0,
                                                "id": tool_call_id,
                                                "type": "function",
                                                "function": {
                                                    "name": func_call["name"],
                                                    "arguments": arguments,
                                                },
                                            }
                                        ],
                                    },
                                    "finish_reason": None,
                                }
                            ],
                        }
                    )
                }

        # Save updated cache
        updated_blocks = batch_engine.get_agent_blocks(agent_id)
        if updated_blocks:
            cache_store.save(agent_id, updated_blocks)
            logger.debug(f"Saved cache: {agent_id} ({updated_blocks.total_tokens} tokens)")

        # Determine finish_reason
        if function_calls:
            finish_reason = "tool_calls"
        elif completion.finish_reason == "stop":
            finish_reason = "stop"
        else:
            finish_reason = "length"

        # Yield final chunk with finish_reason
        yield {
            "data": json.dumps(
                {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created_timestamp,
                    "model": request_body.model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": finish_reason,
                        }
                    ],
                }
            )
        }

        # Yield [DONE] marker
        yield {"data": "[DONE]"}

        logger.info(
            f"Streaming complete: {completion.token_count} tokens, finish_reason={finish_reason}"
        )

    except asyncio.CancelledError:
        # Client disconnected mid-stream - clean up gracefully
        logger.info(f"Streaming cancelled for agent {agent_id} (client disconnect)")
        # Don't yield anything - client is gone
        raise  # Re-raise to properly cancel the coroutine
    except Exception as e:
        logger.error(f"Streaming error: {e}", exc_info=True)
        # Yield error event with proper SSE format
        # OpenAI format: errors are JSON objects in data field
        yield {
            "event": "error",
            "data": json.dumps(
                {
                    "error": {
                        "message": str(e),
                        "type": "server_error",
                        "code": "internal_error",
                    }
                }
            ),
        }


@router.post("/chat/completions", status_code=status.HTTP_200_OK)
async def create_chat_completion(  # noqa: C901, PLR0912, PLR0915
    request_body: ChatCompletionsRequest, request: Request
) -> ChatCompletionsResponse:
    """Create a chat completion (POST /v1/chat/completions).

    OpenAI-compatible endpoint with session_id extension.

    Args:
        request_body: Validated ChatCompletionsRequest
        request: FastAPI request (for accessing app state)

    Returns:
        ChatCompletionsResponse with generated content

    Raises:
        HTTPException: On generation errors
    """
    logger.info(
        f"POST /v1/chat/completions: model={request_body.model}, stream={request_body.stream}"
    )
    logger.debug(f"Messages: {request_body.messages}")

    # Get app dependencies (with null check)
    semantic_state = get_semantic_state(request)
    batch_engine: BlockPoolBatchEngine = semantic_state.batch_engine
    cache_store: AgentCacheStore = semantic_state.cache_store

    try:
        tools_arg = request_body.tools if request_body.tools else None
        prompt = openai_messages_to_prompt(request_body.messages, tools_arg)
        logger.debug(f"Prompt length: {len(prompt)} chars")
        logger.debug(f"Full prompt:\n{prompt}")

        tokenizer = batch_engine.tokenizer
        chat_dicts = openai_messages_to_chat_dicts(request_body.messages, tools_arg)
        tokens = await asyncio.to_thread(
            tokenize_with_chat_template,
            tokenizer,
            chat_dicts,
            prompt,
        )

        # Check for session_id in request body or X-Session-ID header
        session_id = request_body.session_id or request.headers.get("X-Session-ID")
        agent_id = generate_agent_id_openai(session_id, tokens)
        logger.debug(f"Agent ID: {agent_id}, tokens: {len(tokens)}, session_id={session_id}")

        cached_blocks = cache_store.load(agent_id)
        if cached_blocks:
            logger.info(f"Cache hit: {agent_id} ({cached_blocks.total_tokens} tokens)")
        else:
            logger.info(f"Cache miss: {agent_id}")

        # Streaming vs non-streaming
        if request_body.stream:
            # Return SSE stream
            logger.info("Returning OpenAI SSE stream")
            return EventSourceResponse(
                stream_chat_completion(
                    request_body,
                    batch_engine,
                    cache_store,
                    tokens,
                    agent_id,
                    cached_blocks,
                    prompt,
                    scheduler=semantic_state.scheduler,
                )
            )

        # Generate (scheduler or direct batch engine)
        settings = get_settings()
        base_max_tokens = request_body.max_tokens or 256
        max_tokens = base_max_tokens + settings.mlx.reasoning_extra_tokens
        scheduler = semantic_state.scheduler

        if cached_blocks is not None:
            cache_store.invalidate_hot(agent_id)

        if scheduler is not None:
            completion = await scheduler.submit_and_wait(
                agent_id=agent_id,
                prompt_tokens=tokens,
                cache=cached_blocks,
                max_tokens=max_tokens,
                prompt_text=prompt,
                temperature=request_body.temperature,
                top_p=request_body.top_p,
            )
        else:
            # Legacy direct path: no concurrency protection — unsafe for
            # simultaneous requests. Enable SEMANTIC_MLX_SCHEDULER_ENABLED=true.
            logger.warning(
                "Using direct batch_engine path (no scheduler) — concurrent requests unsafe"
            )
            uid = await asyncio.to_thread(
                batch_engine.submit,
                agent_id=agent_id,
                prompt=prompt,
                cache=cached_blocks,
                max_tokens=max_tokens,
                temperature=request_body.temperature,
                top_p=request_body.top_p,
            )
            logger.debug(f"Submitted generation: uid={uid}")
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

        # Parse for function calls
        remaining_text, function_calls = parse_function_calls(completion.text)

        # Format OpenAI response
        # Build tool_calls array if function calls detected
        tool_calls_array = None
        if function_calls:
            tool_calls_array = []
            for func_call in function_calls:
                tool_call_id = f"call_{uuid.uuid4().hex[:24]}"
                # Ensure arguments is a JSON string
                arguments = func_call["arguments"]
                if isinstance(arguments, dict):
                    arguments = json.dumps(arguments)

                tool_calls_array.append(
                    {
                        "id": tool_call_id,
                        "type": "function",
                        "function": {
                            "name": func_call["name"],
                            "arguments": arguments,
                        },
                    }
                )

        # Determine finish_reason
        if function_calls:
            finish_reason = "tool_calls"
        elif completion.finish_reason == "stop":
            finish_reason = "stop"
        else:
            finish_reason = "length"

        # Build message (content can be None if only tool calls)
        message_content = remaining_text.strip() if remaining_text.strip() else None

        response = ChatCompletionsResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:24]}",
            created=int(time.time()),
            model=request_body.model,
            choices=[
                OpenAIChatChoice(
                    index=0,
                    message=OpenAIChatMessage(
                        role="assistant",
                        content=message_content,
                        tool_calls=tool_calls_array,
                    ),
                    finish_reason=finish_reason,
                )
            ],
            usage=OpenAIChatCompletionUsage(
                prompt_tokens=len(tokens),
                completion_tokens=completion.token_count,
                total_tokens=len(tokens) + completion.token_count,
            ),
        )

        logger.info(f"Response: {response.usage.completion_tokens} completion tokens")
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
