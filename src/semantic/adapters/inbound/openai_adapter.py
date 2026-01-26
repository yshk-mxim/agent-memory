r"""OpenAI Chat Completions API adapter (POST /v1/chat/completions).

Implements the OpenAI Chat Completions API with:
- Non-streaming generation
- Streaming (SSE format: data: {...}\ndata: [DONE])
- Session ID extension for persistent caching
"""

import contextlib
import hashlib
import json
import logging
import re
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, HTTPException, Request, status
from sse_starlette.sse import EventSourceResponse

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

    Args:
        text: Model generated text

    Returns:
        Tuple of (remaining_text, list of function call dicts)
        Function call dict contains: {"name": str, "arguments": dict or str}
    """
    function_calls = []
    remaining_text = text

    # Look for {"function_call": ...} pattern
    pattern = r'\{"function_call":\s*\{[^}]+\}\s*\}'
    matches = re.finditer(pattern, text, re.DOTALL)

    for match in matches:
        try:
            func_json = json.loads(match.group())
            if "function_call" in func_json:
                func_data = func_json["function_call"]
                if "name" in func_data and "arguments" in func_data:
                    # Arguments might be a dict or a JSON string
                    arguments = func_data["arguments"]
                    if isinstance(arguments, str):
                        # Try to parse as JSON, leave as string if fails
                        with contextlib.suppress(json.JSONDecodeError):
                            arguments = json.loads(arguments)

                    function_calls.append({
                        "name": func_data["name"],
                        "arguments": arguments,
                    })
                    # Remove function call from text
                    remaining_text = remaining_text.replace(match.group(), "").strip()
        except json.JSONDecodeError:
            continue

    return remaining_text, function_calls


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


async def stream_chat_completion(  # noqa: C901, PLR0912
    request_body: ChatCompletionsRequest,
    batch_engine: Any,
    cache_store: Any,
    _tokens: list[int],
    agent_id: str,
    cached_blocks: Any,
    prompt: str,
) -> AsyncIterator[dict[str, Any]]:
    """Stream chat completion results as SSE events (OpenAI format).

    OpenAI SSE format:
    - data: {"id": "...", "choices": [{"delta": {"content": "..."}, ...}], ...}
    - data: [DONE]

    Yields:
        SSE events in OpenAI Chat Completions format
    """
    try:
        # Submit to batch engine
        max_tokens = request_body.max_tokens or 256
        uid = batch_engine.submit(
            agent_id=agent_id,
            prompt=prompt,
            cache=cached_blocks,
            max_tokens=max_tokens,
        )
        logger.debug(f"Submitted streaming generation: uid={uid}")

        completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created_timestamp = int(time.time())

        # Yield initial chunk with role
        yield {
            "data": json.dumps({
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created_timestamp,
                "model": request_body.model,
                "choices": [{
                    "index": 0,
                    "delta": {"role": "assistant", "content": ""},
                    "finish_reason": None,
                }],
            })
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
                    new_text = result.text[len(accumulated_text):]
                    accumulated_text = result.text

                    if new_text:
                        yield {
                            "data": json.dumps({
                                "id": completion_id,
                                "object": "chat.completion.chunk",
                                "created": created_timestamp,
                                "model": request_body.model,
                                "choices": [{
                                    "index": 0,
                                    "delta": {"content": new_text},
                                    "finish_reason": None,
                                }],
                            })
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
                    "data": json.dumps({
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created_timestamp,
                        "model": request_body.model,
                        "choices": [{
                            "index": 0,
                            "delta": {
                                "tool_calls": [{
                                    "index": 0,
                                    "id": tool_call_id,
                                    "type": "function",
                                    "function": {
                                        "name": func_call["name"],
                                        "arguments": arguments,
                                    },
                                }],
                            },
                            "finish_reason": None,
                        }],
                    })
                }

        # Save updated cache
        if agent_id in batch_engine._agent_blocks:
            updated_blocks = batch_engine._agent_blocks[agent_id]
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
            "data": json.dumps({
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created_timestamp,
                "model": request_body.model,
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": finish_reason,
                }],
            })
        }

        # Yield [DONE] marker
        yield {"data": "[DONE]"}

        logger.info(
            f"Streaming complete: {completion.token_count} tokens, "
            f"finish_reason={finish_reason}"
        )

    except Exception as e:
        logger.error(f"Streaming error: {e}", exc_info=True)
        # Yield error event
        yield {
            "data": json.dumps({
                "error": {
                    "message": str(e),
                    "type": "server_error",
                }
            })
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

    # Get app dependencies
    batch_engine: BlockPoolBatchEngine = request.app.state.semantic.batch_engine
    cache_store: AgentCacheStore = request.app.state.semantic.cache_store

    try:
        # 1. Convert OpenAI messages to prompt (including tools if present)
        prompt = openai_messages_to_prompt(
            request_body.messages,
            request_body.tools if request_body.tools else None,
        )
        logger.debug(f"Prompt length: {len(prompt)} chars")
        logger.debug(f"Full prompt:\n{prompt}")

        # 2. Tokenize to get agent ID
        tokenizer = batch_engine._tokenizer
        tokens = tokenizer.encode(prompt)

        # Check for session_id in request body or X-Session-ID header
        session_id = request_body.session_id or request.headers.get("X-Session-ID")
        agent_id = generate_agent_id_openai(session_id, tokens)
        logger.debug(f"Agent ID: {agent_id}, tokens: {len(tokens)}, session_id={session_id}")

        # 3. Check cache store for existing cache
        cached_blocks = cache_store.load(agent_id)
        if cached_blocks:
            logger.info(f"Cache hit: {agent_id} ({cached_blocks.total_tokens} tokens)")
        else:
            logger.info(f"Cache miss: {agent_id}")

        # 4. Handle streaming vs non-streaming
        if request_body.stream:
            # Return SSE stream
            logger.info("Returning OpenAI SSE stream")
            return EventSourceResponse(
                stream_chat_completion(
                    request_body, batch_engine, cache_store, tokens, agent_id, cached_blocks, prompt
                )
            )

        # 5. Submit to batch engine (non-streaming)
        max_tokens = request_body.max_tokens or 256  # Default if not specified
        uid = batch_engine.submit(
            agent_id=agent_id,
            prompt=prompt,
            cache=cached_blocks,
            max_tokens=max_tokens,
        )
        logger.debug(f"Submitted generation: uid={uid}")

        # 6. Execute generation (step until complete)
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

        # 7. Save updated cache
        if agent_id in batch_engine._agent_blocks:
            updated_blocks = batch_engine._agent_blocks[agent_id]
            cache_store.save(agent_id, updated_blocks)
            logger.debug(f"Saved cache: {agent_id} ({updated_blocks.total_tokens} tokens)")

        # 8. Parse for function calls
        remaining_text, function_calls = parse_function_calls(completion.text)

        # 9. Format OpenAI response
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

                tool_calls_array.append({
                    "id": tool_call_id,
                    "type": "function",
                    "function": {
                        "name": func_call["name"],
                        "arguments": arguments,
                    },
                })

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

        logger.info(
            f"Response: {response.usage.completion_tokens} completion tokens"
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
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        ) from e
