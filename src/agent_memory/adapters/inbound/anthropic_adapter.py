# mypy: disable-error-code="union-attr,arg-type,attr-defined,no-untyped-def"
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Yakov Shkolnikov and contributors
"""Anthropic Messages API adapter (POST /v1/messages).

Implements the Anthropic Messages API with:
- Non-streaming generation via ConcurrentScheduler
- SSE streaming via ConcurrentScheduler
- Tool use support
- Extended thinking
- Prompt caching
"""

import asyncio
import copy
import hashlib
import json
import logging
import uuid
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, HTTPException, Request, status
from sse_starlette.sse import EventSourceResponse

from agent_memory.adapters.inbound.adapter_helpers import (
    extract_session_id,
    extract_system_text,
    get_semantic_state,
    run_step_for_uid,
    strip_thinking_tags,
    tokenize_with_chat_template,
)
from agent_memory.adapters.inbound.request_models import (
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
from agent_memory.application.agent_cache_store import AgentCacheStore
from agent_memory.application.batch_engine import BlockPoolBatchEngine
from agent_memory.application.generation_guardrails import detect_tool_retry_loop
from agent_memory.application.generation_request import GenerationRequest
from agent_memory.application.shared_prefix_cache import SharedPrefixCache
from agent_memory.domain.errors import PoolExhaustedError, SemanticError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["anthropic"])


def _anthropic_to_openai_messages(
    anthropic_messages: list,
    system_text: str | None = None,
) -> list[dict]:
    """Convert Anthropic Messages API messages to OpenAI chat format.

    Handles all content block types:
    - text/thinking → plain string content
    - tool_use (assistant) → tool_calls array
    - tool_result (user) → role=tool message
    System prompt is prepended as role=system if provided.
    """
    result: list[dict] = []

    # Inject current date/time so models know what day it is.
    # For Gemma 4 (custom template), strftime_now() handles this at the
    # template level. For other models (Qwen etc.) this is the only source.
    # Duplicates are harmless — better than the model thinking it's 2024.
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).strftime("%A, %d %B %Y, %H:%M UTC")
    date_line = f"Current date and time: {now}"

    if system_text:
        system_text = f"{date_line}\n\n{system_text}"
    else:
        system_text = date_line
    result.append({"role": "system", "content": system_text})

    for msg in anthropic_messages:
        role = msg.role if hasattr(msg, "role") else msg.get("role", "user")
        content = msg.content if hasattr(msg, "content") else msg.get("content", "")

        if isinstance(content, str):
            result.append({"role": role, "content": content})
            continue

        # content is a list of typed blocks
        text_parts: list[str] = []
        tool_calls: list[dict] = []
        tool_results: list[dict] = []

        for block in content:
            btype = block.type if hasattr(block, "type") else block.get("type", "")

            if btype in ("text", "thinking"):
                text = block.text if hasattr(block, "text") else block.get("text", "")
                if text:
                    text_parts.append(text)

            elif btype == "tool_use":
                tid = block.id if hasattr(block, "id") else block.get("id", "")
                name = block.name if hasattr(block, "name") else block.get("name", "")
                inp = block.input if hasattr(block, "input") else block.get("input", {})
                tool_calls.append(
                    {
                        "id": tid,
                        "type": "function",
                        "function": {"name": name, "arguments": json.dumps(inp)},
                    }
                )

            elif btype == "tool_result":
                tool_use_id = (
                    block.tool_use_id
                    if hasattr(block, "tool_use_id")
                    else block.get("tool_use_id", "")
                )
                rc = block.content if hasattr(block, "content") else block.get("content", "")
                if isinstance(rc, list):
                    rc = "\n".join(
                        (b.text if hasattr(b, "text") else b.get("text", str(b))) for b in rc
                    )
                is_error = (
                    block.is_error
                    if hasattr(block, "is_error")
                    else block.get("is_error", False)
                )
                if is_error and rc:
                    rc = f"[ERROR] {rc}"
                tool_results.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_use_id,
                        "content": rc or "",
                    }
                )

        if tool_calls:
            result.append(
                {
                    "role": "assistant",
                    "content": "\n".join(text_parts) or None,
                    "tool_calls": tool_calls,
                }
            )
        elif tool_results:
            result.extend(tool_results)
        else:
            result.append({"role": role, "content": "\n".join(text_parts)})

    return result


# Tool description hints for local models.
# Small/MoE models need explicit instructions to correctly use tools that
# require exact string matching (like Edit). Claude is trained on these
# tool schemas; local models are not.
_TOOL_DESCRIPTION_HINTS: dict[str, str] = {
    "Edit": (
        "\n\nCRITICAL: old_string must be copied character-for-character "
        "from the most recent Read output. Do NOT paraphrase, reword, or "
        "use similar text — it must be an EXACT substring of the file. "
        "Do NOT insert HTML tags unless they exist in the original file."
    ),
    "TaskUpdate": (
        " When all tasks are finished, mark every task as completed. "
        "Do not leave tasks in_progress after the work is done."
    ),
}

# Property description overrides for local models.
# Claude Code's default "The text to replace" is ambiguous for models not
# trained on this specific schema. Gemma 4's native template sorts properties
# alphabetically (new_string before old_string via dictsort), reversing the
# natural reasoning flow. Clear descriptions help the model despite ordering.
_PROPERTY_DESCRIPTION_OVERRIDES: dict[str, dict[str, str]] = {
    "Edit": {
        "old_string": (
            "The EXACT text currently in the file that you want to change. "
            "Must match the file content character-for-character."
        ),
        "new_string": (
            "The replacement text to substitute for old_string."
        ),
    },
}


def _anthropic_to_openai_tools(tools: list) -> list[dict]:
    """Convert Anthropic tool definitions to OpenAI function calling format.

    Applies tool description hints and property description overrides to
    help local models that weren't trained on Claude Code's specific tool
    schemas.
    """
    result = []
    for tool in tools:
        t = tool.model_dump() if hasattr(tool, "model_dump") else tool
        name = t.get("name", "")
        description = t.get("description", "")
        hint = _TOOL_DESCRIPTION_HINTS.get(name)
        if hint:
            description += hint

        # Deep-copy schema so we don't mutate the original request
        schema = copy.deepcopy(t.get("input_schema", {}))

        # Override ambiguous property descriptions for local models
        overrides = _PROPERTY_DESCRIPTION_OVERRIDES.get(name, {})
        if overrides and "properties" in schema:
            for prop_name, prop_desc in overrides.items():
                if prop_name in schema["properties"]:
                    schema["properties"][prop_name]["description"] = prop_desc

        result.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": schema,
                },
            }
        )
    return result


# Anthropic API defaults — when these arrive, the client didn't set them.
_ANTHROPIC_DEFAULT_TEMPERATURE = 1.0
_ANTHROPIC_DEFAULT_TOP_P = 1.0
_ANTHROPIC_DEFAULT_TOP_K = 0


def _resolve_sampling_params(
    request_body: MessagesRequest,
) -> tuple[float, float, int]:
    """Resolve sampling params: model profile > client > hardcoded fallback.

    The Anthropic API defaults (temperature=1.0, top_p=1.0, top_k=0) are
    meaningless for open models. When the client sends these defaults, we
    substitute values from the model's TOML profile instead.

    Returns:
        (temperature, top_p, top_k)
    """
    from agent_memory.adapters.config.settings import load_model_profile

    profile = load_model_profile(model_id=request_body.model)
    inference = profile.get("inference", {})

    # Model profile values (authoritative for the model)
    prof_temp = inference.get("temperature", 0.7)
    prof_top_p = inference.get("top_p", 0.95)
    prof_top_k = inference.get("top_k", 40)

    # Use client value if explicitly set (differs from Anthropic defaults),
    # otherwise use model profile value
    temperature = (
        request_body.temperature
        if request_body.temperature != _ANTHROPIC_DEFAULT_TEMPERATURE
        else prof_temp
    )
    top_p = request_body.top_p if request_body.top_p != _ANTHROPIC_DEFAULT_TOP_P else prof_top_p
    top_k = request_body.top_k if request_body.top_k != _ANTHROPIC_DEFAULT_TOP_K else prof_top_k

    return temperature, top_p, top_k


def generate_agent_id_from_tokens(tokens: list[int]) -> str:
    """Generate agent ID from token prefix for cache lookup.

    Uses first 100 tokens for stability (prefix matching).

    Args:
        tokens: Full token sequence

    Returns:
        Agent ID in format "msg_{hash}"
    """
    prefix = tokens[:100]
    # Use JSON serialization for deterministic, platform-independent hashing
    hash_val = hashlib.sha256(json.dumps(prefix).encode()).hexdigest()[:16]
    return f"msg_{hash_val}"


def parse_tool_calls(text: str) -> tuple[str, list[dict[str, Any]]]:
    """Parse tool calls from model output using the canonical parser chain.

    Delegates to ``JsonLinesToolCallParser`` which handles all known formats:
    ReAct (action/action_input), name/parameters, name/arguments, tool_use,
    function_call — including JSON inside markdown code blocks.

    Strips thinking/channel tags first, then parses. Parameter names are
    normalized via ``tool_param_normalization`` (e.g. instructions→prompt).

    Args:
        text: Model generated text (may contain thinking/channel tags)

    Returns:
        Tuple of (remaining_text, list of tool call dicts)
        Tool call dict contains: {"name": str, "input": dict}
    """
    text = strip_thinking_tags(text)

    from agent_memory.adapters.outbound.tool_call_parsers.json_lines import (
        JsonLinesToolCallParser,
    )

    parser = JsonLinesToolCallParser()
    remaining, parsed = parser.parse(text)
    return remaining, [{"name": tc.name, "input": tc.input} for tc in parsed]


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

    # Add tool definitions if present (compressed for local model efficiency)
    if tools:
        from agent_memory.adapters.inbound.tool_compression import compress_tool_definitions

        tool_dicts = [
            {"name": t.name, "description": t.description, "input_schema": t.input_schema}
            for t in tools
        ]
        lines.append("\n" + compress_tool_definitions(tool_dicts) + "\n")

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
                        block.content
                        if isinstance(block.content, str)
                        else json.dumps(block.content)
                    )
                    status = "ERROR" if block.is_error else "SUCCESS"
                    lines.append(
                        f"{msg.role.capitalize()} [Tool Result - {status}]: {result_content}"
                    )
                elif hasattr(block, "name") and hasattr(block, "input"):
                    # ToolUseContentBlock (in assistant messages)
                    tool_call = {"name": block.name, "input": block.input}
                    lines.append(f"{msg.role.capitalize()} [Tool Call]: {json.dumps(tool_call)}")

    # Add assistant prefix for continuation
    lines.append("Assistant:")

    return "\n".join(lines)


def messages_to_chat_dicts(  # noqa: C901, PLR0912
    messages: list[Message],
    system: str | list[Any] = "",
    tools: list[Any] | None = None,
) -> list[dict[str, str]]:
    """Convert Anthropic messages to simple chat dicts for chat template.

    Args:
        messages: List of user/assistant messages
        system: System prompt (string or blocks)
        tools: Optional list of tool definitions

    Returns:
        List of {"role": ..., "content": ...} dicts
    """
    result: list[dict[str, str]] = []

    # Build system content
    system_parts: list[str] = []
    if system:
        if isinstance(system, str):
            system_parts.append(system)
        else:
            for block in system:
                if hasattr(block, "text"):
                    system_parts.append(block.text)

    if tools:
        from agent_memory.adapters.inbound.tool_compression import compress_tool_definitions

        tool_dicts = [
            {"name": t.name, "description": t.description, "input_schema": t.input_schema}
            for t in tools
        ]
        system_parts.append(compress_tool_definitions(tool_dicts))

    if system_parts:
        result.append({"role": "system", "content": "\n\n".join(system_parts)})

    # Convert messages
    for msg in messages:
        parts: list[str] = []
        if isinstance(msg.content, str):
            parts.append(msg.content)
        else:
            for block in msg.content:
                if hasattr(block, "text"):
                    parts.append(block.text)
                elif hasattr(block, "thinking"):
                    parts.append(f"(thinking): {block.thinking}")
                elif hasattr(block, "tool_use_id"):
                    rc = (
                        block.content
                        if isinstance(block.content, str)
                        else json.dumps(block.content)
                    )
                    st = "ERROR" if block.is_error else "SUCCESS"
                    parts.append(f"[Tool Result - {st}]: {rc}")
                elif hasattr(block, "name") and hasattr(block, "input"):
                    tc = {"name": block.name, "input": block.input}
                    parts.append(f"[Tool Call]: {json.dumps(tc)}")
        result.append({"role": msg.role, "content": "\n".join(parts)})

    return result


async def _stream_trt_response(
    trt_inference: Any,
    agent_id: str,
    prompt: str,
    request_body: MessagesRequest,
    messages: list[dict[str, str]],
    tokens: list[int],
    openai_tools: list[dict] | None = None,
    session_id: str | None = None,
    server_tool_executor: Any | None = None,
    sampling_params: tuple[float, float, int] | None = None,
) -> AsyncIterator[dict[str, str]]:
    """Generate full TRT response then yield as Anthropic SSE events.

    This is "chunked streaming" — the model generates everything at once
    (TRT subprocess is synchronous) but the response is sent to the client
    as SSE events matching the Anthropic streaming protocol.
    """
    msg_id = f"msg_{uuid.uuid4().hex[:24]}"

    # thinking.type == "disabled" → suppress thinking
    thinking = request_body.thinking
    disable_thinking = thinking is None or thinking.type == "disabled"

    # Use pre-resolved sampling params from caller (DRY)
    if sampling_params is not None:
        temperature, top_p, top_k = sampling_params
    else:
        temperature, top_p, top_k = _resolve_sampling_params(request_body)

    # Generate full response (all sampling params forwarded)
    result = await asyncio.to_thread(
        trt_inference.generate,
        agent_id=agent_id,
        prompt=prompt,
        max_tokens=request_body.max_tokens,
        temperature=temperature,
        messages=messages,
        top_p=top_p,
        top_k=top_k,
        stop_sequences=request_body.stop_sequences or None,
        openai_tools=openai_tools,
        disable_thinking=disable_thinking,
        model=request_body.model or None,
        session_id=session_id,
    )

    logger.info(
        "generate result: text=%r tool_calls=%r tokens=%d",
        result.text[:300] if result.text else "",
        result.tool_calls,
        len(result.tokens),
    )

    # Prefer structured tool_calls from backend (llamacpp/vllm native function calling).
    # Fall back to text-based parsing for backends that encode tool calls in output text.
    if result.tool_calls is not None:
        remaining_text = result.text
        tool_calls = result.tool_calls
    else:
        remaining_text, tool_calls = parse_tool_calls(result.text)

    # Server-side tool execution loop (WebSearch, WebFetch)
    from agent_memory.application.server_tool_port import MAX_TOOL_ROUNDS, split_tool_calls

    for _round in range(MAX_TOOL_ROUNDS):
        server_calls, client_calls = split_tool_calls(tool_calls, server_tool_executor)
        if not server_calls:
            break  # No server-side tools — proceed to response

        # Execute server-side tools
        logger.info("executing %d server-side tools (round %d)", len(server_calls), _round + 1)
        # Build assistant message with tool calls
        assistant_content = remaining_text or ""
        # Build tool call entries for the conversation
        tc_entries = []
        for tc in server_calls:
            tc_entries.append(
                {
                    "id": tc.get("id", f"call_{uuid.uuid4().hex[:8]}"),
                    "type": "function",
                    "function": {"name": tc["name"], "arguments": json.dumps(tc.get("input", {}))},
                }
            )
        messages = list(messages)  # Copy to avoid mutating caller's list
        messages.append(
            {"role": "assistant", "content": assistant_content, "tool_calls": tc_entries}
        )

        # Add tool results
        for tc, entry in zip(server_calls, tc_entries):
            tool_result = await asyncio.to_thread(
                server_tool_executor.execute, tc["name"], tc.get("input", {})
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": entry["id"],
                    "content": tool_result,
                }
            )

        # Re-generate with tool results in context
        result = await asyncio.to_thread(
            trt_inference.generate,
            agent_id=agent_id,
            prompt=prompt,
            max_tokens=request_body.max_tokens,
            temperature=temperature,
            messages=messages,
            top_p=top_p,
            top_k=top_k,
            stop_sequences=request_body.stop_sequences or None,
            openai_tools=openai_tools,
            disable_thinking=disable_thinking,
            model=request_body.model or None,
            session_id=session_id,
        )
        logger.info(
            "re-generate result (round %d): text=%r tool_calls=%r",
            _round + 1,
            result.text[:200] if result.text else "",
            result.tool_calls,
        )
        if result.tool_calls is not None:
            remaining_text = result.text
            tool_calls = result.tool_calls + client_calls
        else:
            remaining_text, new_tools = parse_tool_calls(result.text)
            tool_calls = new_tools + client_calls
        client_calls = []  # Already merged

    # message_start
    yield {
        "event": "message_start",
        "data": json.dumps(
            MessageStartEvent(
                message=MessagesResponse(
                    id=msg_id,
                    model=request_body.model or "trt",
                    content=[],
                    stop_reason=None,
                    usage=Usage(
                        input_tokens=len(tokens),
                        output_tokens=0,
                    ),
                ),
            ).model_dump()
        ),
    }

    block_idx = 0

    # Text content block — stream in word-sized chunks for realistic SSE
    if remaining_text:
        yield {
            "event": "content_block_start",
            "data": json.dumps(
                ContentBlockStartEvent(
                    index=block_idx,
                    content_block=TextContentBlock(text=""),
                ).model_dump()
            ),
        }

        # Chunk text into words/tokens for progressive streaming
        # This simulates per-token output from a streaming model
        words = remaining_text.split(" ")
        for i, word in enumerate(words):
            chunk = word if i == 0 else " " + word
            yield {
                "event": "content_block_delta",
                "data": json.dumps(
                    ContentBlockDeltaEvent(
                        index=block_idx,
                        delta={"type": "text_delta", "text": chunk},
                    ).model_dump()
                ),
            }

        yield {
            "event": "content_block_stop",
            "data": json.dumps(ContentBlockStopEvent(index=block_idx).model_dump()),
        }
        block_idx += 1

    # Tool use blocks
    for tc in tool_calls:
        tool_id = f"toolu_{uuid.uuid4().hex[:24]}"
        yield {
            "event": "content_block_start",
            "data": json.dumps(
                ContentBlockStartEvent(
                    index=block_idx,
                    content_block=ToolUseContentBlock(
                        id=tool_id,
                        name=tc["name"],
                        input={},
                    ),
                ).model_dump()
            ),
        }
        yield {
            "event": "content_block_delta",
            "data": json.dumps(
                ContentBlockDeltaEvent(
                    index=block_idx,
                    delta={"type": "input_json_delta", "partial_json": json.dumps(tc["input"])},
                ).model_dump()
            ),
        }
        yield {
            "event": "content_block_stop",
            "data": json.dumps(ContentBlockStopEvent(index=block_idx).model_dump()),
        }
        block_idx += 1

    stop_reason = "tool_use" if tool_calls else "end_turn"
    yield {
        "event": "message_delta",
        "data": json.dumps(
            MessageDeltaEvent(
                delta={"stop_reason": stop_reason},
                usage=Usage(input_tokens=0, output_tokens=len(result.tokens)),
            ).model_dump()
        ),
    }

    yield {
        "event": "message_stop",
        "data": json.dumps({"type": "message_stop"}),
    }


async def stream_generation(  # noqa: C901, PLR0912
    request_body: MessagesRequest,
    batch_engine: Any,
    cache_store: Any,
    tokens: list[int],
    agent_id: str,
    cached_blocks: Any,
    prefix_cache: Any = None,
    prefix_hash: str | None = None,
    system_prefix_len: int = 0,
) -> AsyncIterator[dict[str, Any]]:
    """Stream generation results as SSE events.

    Yields:
        SSE events in Anthropic Messages API format
    """
    temperature, top_p, top_k = _resolve_sampling_params(request_body)
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
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
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
                "data": json.dumps(ContentBlockStopEvent(index=content_block_index).model_dump()),
            }

            content_block_index += 1

        # Save updated cache
        updated_blocks = batch_engine.get_agent_blocks(agent_id)
        if updated_blocks:
            cache_store.save(agent_id, updated_blocks)

            # Store only the system prefix KV state — user question and
            # response are session-specific and must not be cached.
            if prefix_cache is not None and prefix_hash is not None and system_prefix_len > 0:
                trimmed = updated_blocks.trim_to_prefix(system_prefix_len)
                detached = trimmed.detach_for_prefix_cache()
                prefix_cache.put(
                    prefix_hash=prefix_hash,
                    kv_caches=detached,
                    n_tokens=system_prefix_len,
                    token_sequence=list(tokens[:system_prefix_len]),
                )

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


async def stream_generation_via_scheduler(
    request_body: MessagesRequest,
    scheduler: Any,
    cache_store: Any,
    batch_engine: Any,
    tokens: list[int],
    prompt: str,
    agent_id: str,
    cached_blocks: Any,
    prefix_cache: Any = None,
    prefix_hash: str | None = None,
    system_prefix_len: int = 0,
) -> AsyncIterator[dict[str, Any]]:
    """Stream generation via scheduler (supports batch=2).

    Uses scheduler.submit_and_stream() for per-token streaming
    through the scheduler's interleaved decode loop.
    """
    try:
        # Invalidate hot cache before streaming
        if cached_blocks is not None:
            cache_store.invalidate_hot(agent_id)

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

        yield {
            "event": "content_block_start",
            "data": json.dumps(
                ContentBlockStartEvent(
                    index=0, content_block=TextContentBlock(text="")
                ).model_dump()
            ),
        }

        accumulated_text = ""
        final_text = ""
        final_token_count = 0
        final_finish_reason = "end_turn"

        temperature, top_p, top_k = _resolve_sampling_params(request_body)
        async for delta in scheduler.submit_and_stream(
            agent_id=agent_id,
            prompt_tokens=tokens,
            cache=cached_blocks,
            max_tokens=request_body.max_tokens,
            prompt_text=prompt,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        ):
            new_text = delta.text[len(accumulated_text) :]
            accumulated_text = delta.text

            if new_text:
                yield {
                    "event": "content_block_delta",
                    "data": json.dumps(
                        ContentBlockDeltaEvent(
                            index=0, delta={"type": "text_delta", "text": new_text}
                        ).model_dump()
                    ),
                }

            if delta.finish_reason is not None:
                final_text = delta.text
                final_token_count = delta.token_count
                final_finish_reason = "end_turn" if delta.finish_reason == "stop" else "max_tokens"

        yield {
            "event": "content_block_stop",
            "data": json.dumps(ContentBlockStopEvent(index=0).model_dump()),
        }

        # Parse for tool calls
        _remaining_text, tool_calls = parse_tool_calls(final_text)

        content_block_index = 1
        for tool_call in tool_calls:
            tool_use_block = ToolUseContentBlock(
                id=f"toolu_{uuid.uuid4().hex[:24]}",
                name=tool_call["name"],
                input=tool_call["input"],
            )
            yield {
                "event": "content_block_start",
                "data": json.dumps(
                    ContentBlockStartEvent(
                        index=content_block_index,
                        content_block=tool_use_block,
                    ).model_dump()
                ),
            }
            yield {
                "event": "content_block_stop",
                "data": json.dumps(ContentBlockStopEvent(index=content_block_index).model_dump()),
            }
            content_block_index += 1

        # Save cache
        updated_blocks = batch_engine.get_agent_blocks(agent_id)
        if updated_blocks:
            cache_store.save(agent_id, updated_blocks)

            # Store only the system prefix KV state
            if prefix_cache is not None and prefix_hash is not None and system_prefix_len > 0:
                trimmed = updated_blocks.trim_to_prefix(system_prefix_len)
                detached = trimmed.detach_for_prefix_cache()
                prefix_cache.put(
                    prefix_hash=prefix_hash,
                    kv_caches=detached,
                    n_tokens=system_prefix_len,
                    token_sequence=list(tokens[:system_prefix_len]),
                )

        if tool_calls:
            final_finish_reason = "tool_use"

        yield {
            "event": "message_delta",
            "data": json.dumps(
                MessageDeltaEvent(
                    delta={"stop_reason": final_finish_reason},
                    usage=Usage(
                        input_tokens=0,
                        output_tokens=final_token_count,
                    ),
                ).model_dump()
            ),
        }

        yield {
            "event": "message_stop",
            "data": json.dumps(MessageStopEvent().model_dump()),
        }

    except asyncio.CancelledError:
        logger.info(f"Streaming cancelled for agent {agent_id} (client disconnect)")
        raise
    except Exception as e:
        logger.error(f"Scheduler streaming error: {e}", exc_info=True)
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
    logger.info(
        f"POST /v1/messages: model={request_body.model}, stream={request_body.stream}, max_tokens={request_body.max_tokens}"
    )
    logger.debug(f"Messages: {request_body.messages}")

    # Get app dependencies (with null check)
    semantic_state = get_semantic_state(request)
    batch_engine = semantic_state.batch_engine  # None for TRT backend
    cache_store: AgentCacheStore = semantic_state.cache_store
    scheduler = getattr(semantic_state, "scheduler", None)
    prefix_cache: SharedPrefixCache | None = getattr(semantic_state, "prefix_cache", None)
    trt_inference = getattr(semantic_state, "trt_inference", None)

    try:
        # Filter out Anthropic server-side tools (e.g. web_search_20250305) — these
        # have type like "web_search_20250305" but no description/input_schema.
        # Our custom WebSearch/WebFetch are regular function tools and pass through.
        tools_arg = [
            t for t in request_body.tools
            if t.description or t.input_schema
        ] if request_body.tools else None
        tools_arg = tools_arg or None  # empty list → None
        prompt = messages_to_prompt(
            request_body.messages,
            request_body.system,
            tools_arg,
        )
        logger.debug(f"Prompt length: {len(prompt)} chars")
        logger.debug(f"Full prompt:\n{prompt}")

        tokenizer = getattr(semantic_state, "tokenizer", None) or batch_engine.tokenizer
        chat_dicts = messages_to_chat_dicts(
            request_body.messages,
            request_body.system,
            tools_arg,
        )
        tokens, templated_prompt = await asyncio.to_thread(
            tokenize_with_chat_template,
            tokenizer,
            chat_dicts,
            prompt,
        )

        # Session-based lookup enables prefix caching across conversation turns
        session_id = extract_session_id(request)
        if session_id:
            agent_id = f"sess_{session_id}"
            logger.debug(f"Session-based agent ID: {agent_id}, tokens: {len(tokens)}")
        else:
            agent_id = generate_agent_id_from_tokens(tokens)
            logger.debug(f"Token-based agent ID: {agent_id}, tokens: {len(tokens)}")

        cached_blocks = cache_store.load(agent_id)
        prefix_hash: str | None = None
        system_prefix_len: int = 0
        if cached_blocks:
            logger.info(f"Cache hit: {agent_id} ({cached_blocks.total_tokens} tokens)")
        else:
            logger.info(f"Cache miss: {agent_id}")

            # Compute shared prefix hash for system+tools reuse
            if prefix_cache is not None:
                system_text = ""
                if request_body.system:
                    system_text = extract_system_text(request_body.system)
                tools_text = ""
                if request_body.tools:
                    tools_text = json.dumps(
                        [{"name": t.name, "description": t.description} for t in request_body.tools]
                    )
                if system_text or tools_text:
                    prefix_hash = SharedPrefixCache.compute_hash(system_text, tools_text)
                    # Tokenize system-only to know where the reusable prefix ends.
                    # We store only these tokens in the prefix cache — user question
                    # and response are session-specific and must not be cached.
                    system_only_dicts = messages_to_chat_dicts(
                        [],
                        request_body.system,
                        tools_arg,
                    )
                    _sys_tokens, _ = tokenize_with_chat_template(
                        tokenizer,
                        system_only_dicts,
                        "",
                    )
                    system_prefix_len = len(_sys_tokens)
                    prefix_entry = prefix_cache.take(prefix_hash)
                    if prefix_entry is not None:
                        logger.info(
                            f"Prefix cache hit: hash={prefix_hash[:8]}, "
                            f"tokens={prefix_entry.n_tokens}"
                        )
                        # Consume the cached blocks directly — submit() will
                        # clear layer_data after reconstruction, which is fine
                        # since we popped the entry.  After generation we store
                        # fresh blocks back into the cache.
                        cached_blocks = prefix_entry.kv_caches

        # TRT backend: generation via TRTInferenceService (handles cache persistence)
        if trt_inference is not None and batch_engine is None:
            # Guardrail: detect repeated tool failures and inject corrective hint
            retry_hint = detect_tool_retry_loop(request_body.messages)
            if retry_hint:
                logger.warning("tool_retry_loop_detected, injecting hint")
                request_body.messages.append(
                    Message(role="user", content=retry_hint)
                )

            system_text = extract_system_text(request_body.system) if request_body.system else None
            messages = _anthropic_to_openai_messages(request_body.messages, system_text)
            openai_tools = _anthropic_to_openai_tools(tools_arg) if tools_arg else None

            # thinking.type == "disabled" → suppress thinking; "enabled"/"adaptive" → allow it
            thinking = request_body.thinking
            disable_thinking = thinking is None or thinking.type == "disabled"

            # Resolve sampling params from model profile (not Anthropic API defaults)
            temperature, top_p, top_k = _resolve_sampling_params(request_body)

            gen_req = GenerationRequest(
                agent_id=agent_id,
                messages=messages,
                prompt=templated_prompt,
                max_tokens=request_body.max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                stop_sequences=request_body.stop_sequences or [],
                stream=request_body.stream,
                model=request_body.model or "trt",
                openai_tools=openai_tools,
                disable_thinking=disable_thinking,
                session_id=session_id,
            )

            # Streaming: generate full response, then yield as SSE events
            if request_body.stream:
                server_tool_executor = getattr(semantic_state, "server_tool_executor", None)
                return EventSourceResponse(
                    _stream_trt_response(
                        trt_inference,
                        agent_id,
                        templated_prompt,
                        request_body,
                        messages,
                        tokens,
                        openai_tools=openai_tools,
                        session_id=session_id,
                        server_tool_executor=server_tool_executor,
                        sampling_params=(temperature, top_p, top_k),
                    )
                )

            result = trt_inference.generate_from_request(gen_req)

            # Prefer structured tool_calls from backend; fall back to text parsing.
            if result.tool_calls is not None:
                remaining_text = result.text
                tool_calls = result.tool_calls
            else:
                remaining_text, tool_calls = parse_tool_calls(result.text)

            content_blocks = []
            if remaining_text:
                content_blocks.append(TextContentBlock(text=remaining_text))
            for tc in tool_calls:
                content_blocks.append(
                    ToolUseContentBlock(
                        id=f"toolu_{uuid.uuid4().hex[:24]}",
                        name=tc["name"],
                        input=tc["input"],
                    )
                )

            return MessagesResponse(
                id=f"msg_{uuid.uuid4().hex[:24]}",
                model=request_body.model or "trt",
                content=content_blocks,
                stop_reason="end_turn" if not tool_calls else "tool_use",
                usage=Usage(
                    input_tokens=len(tokens),
                    output_tokens=len(result.tokens),
                ),
            )

        # Streaming vs non-streaming
        if request_body.stream:
            if scheduler is not None:
                # Batched streaming via scheduler (supports batch=2)
                logger.info("Returning SSE stream via scheduler")
                return EventSourceResponse(
                    stream_generation_via_scheduler(
                        request_body,
                        scheduler,
                        cache_store,
                        batch_engine,
                        tokens,
                        templated_prompt,
                        agent_id,
                        cached_blocks,
                        prefix_cache=prefix_cache,
                        prefix_hash=prefix_hash,
                        system_prefix_len=system_prefix_len,
                    )
                )
            # Legacy direct streaming (no scheduler) — unsafe for concurrent requests
            logger.warning(
                "Returning SSE stream (direct, no scheduler) — concurrent requests unsafe"
            )
            return EventSourceResponse(
                stream_generation(
                    request_body,
                    batch_engine,
                    cache_store,
                    tokens,
                    agent_id,
                    cached_blocks,
                    prefix_cache=prefix_cache,
                    prefix_hash=prefix_hash,
                    system_prefix_len=system_prefix_len,
                )
            )

        # Resolve sampling params from model profile (not Anthropic API defaults)
        temperature, top_p, top_k = _resolve_sampling_params(request_body)

        # Route through scheduler or direct engine path
        if scheduler is not None:
            # Scheduler path: interleaved prefill + decode
            logger.info(f"Routing through scheduler: agent={agent_id}, tokens={len(tokens)}")
            completion = await scheduler.submit_and_wait(
                agent_id=agent_id,
                prompt_tokens=tokens,
                cache=cached_blocks,
                max_tokens=request_body.max_tokens,
                prompt_text=templated_prompt,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )

            # Invalidate hot cache if we passed one in
            if cached_blocks is not None:
                cache_store.invalidate_hot(agent_id)
        else:
            # Legacy direct path: no concurrency protection — unsafe for
            # simultaneous requests. Enable SEMANTIC_MLX_SCHEDULER_ENABLED=true.
            logger.warning(
                "Using direct batch_engine path (no scheduler) — concurrent requests unsafe"
            )
            uid = await asyncio.to_thread(
                batch_engine.submit,
                agent_id=agent_id,
                prompt=templated_prompt,
                cache=cached_blocks,
                max_tokens=request_body.max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )
            logger.debug(f"Submitted generation: uid={uid}")

            # Invalidate hot cache entry if we passed in a cache
            # batch_engine clears Q4 blocks after reconstruction
            if cached_blocks is not None:
                cache_store.invalidate_hot(agent_id)

            # Execute generation (step until complete)
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

            # Store only the system prefix KV state — the reusable portion.
            # User question and response are session-specific; causal attention
            # guarantees the system prefix KV data is identical regardless.
            if prefix_cache is not None and prefix_hash is not None and system_prefix_len > 0:
                trimmed = updated_blocks.trim_to_prefix(system_prefix_len)
                detached = trimmed.detach_for_prefix_cache()
                prefix_cache.put(
                    prefix_hash=prefix_hash,
                    kv_caches=detached,
                    n_tokens=system_prefix_len,
                    token_sequence=list(tokens[:system_prefix_len]),
                )

        # Parse for tool calls
        remaining_text, tool_calls = parse_tool_calls(completion.text)

        # Format response
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
        tokenizer = getattr(semantic_state, "tokenizer", None) or batch_engine.tokenizer
        tokens = await asyncio.to_thread(tokenizer.encode, prompt)

        logger.info(f"Token count: {len(tokens)}")
        return CountTokensResponse(input_tokens=len(tokens))

    except Exception as e:
        logger.error(f"Token counting error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to count tokens: {e!s}",
        ) from e
