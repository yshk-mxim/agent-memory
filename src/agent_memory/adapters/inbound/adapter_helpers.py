"""Shared helper functions for inbound adapters.

Contains common functionality used by multiple API adapters to avoid code duplication.
"""

import json
import logging
import time
from typing import Any

from fastapi import HTTPException, Request, status

logger = logging.getLogger(__name__)

# Default timeout for batch engine step operations (5 minutes)
STEP_TIMEOUT_SECONDS = 300


def get_semantic_state(request: Request) -> Any:
    """Safely get semantic state from request, raising clear error if not initialized.

    Args:
        request: FastAPI request object

    Returns:
        The semantic state object

    Raises:
        HTTPException: If semantic state is not initialized
    """
    if not hasattr(request.app.state, "agent_memory") or request.app.state.agent_memory is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Server is still initializing. Please retry in a few seconds.",
        )
    return request.app.state.agent_memory


def get_coordination_service(request: Request) -> Any:
    """Safely get coordination service from request, raising clear error if not initialized.

    Args:
        request: FastAPI request object

    Returns:
        The coordination service object

    Raises:
        HTTPException: If coordination service is not initialized
    """
    if (
        not hasattr(request.app.state, "coordination_service")
        or request.app.state.coordination_service is None
    ):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Coordination service is not available. Server may still be initializing.",
        )
    return request.app.state.coordination_service


def run_step_for_uid(
    batch_engine: Any,
    uid: str,
    timeout_seconds: float = STEP_TIMEOUT_SECONDS,
) -> Any:
    """Run batch engine step until we get result for uid, with timeout.

    This is a blocking helper meant to run in a thread executor.

    Args:
        batch_engine: The batch engine instance
        uid: The unique ID to wait for
        timeout_seconds: Maximum time to wait (default: 5 minutes)

    Returns:
        The completion result for the given uid, or None if not found/timeout

    Raises:
        TimeoutError: If generation exceeds timeout
    """
    start_time = time.monotonic()

    for result in batch_engine.step():
        if result.uid == uid:
            return result

        elapsed = time.monotonic() - start_time
        if elapsed > timeout_seconds:
            logger.error(f"Generation timeout after {elapsed:.1f}s for uid={uid}")
            raise TimeoutError(f"Generation timed out after {timeout_seconds}s")

    return None


def _is_gpt_oss_tokenizer(tokenizer: Any) -> bool:
    """Check if tokenizer uses GPT-OSS Harmony format.

    GPT-OSS models have <|channel|> markers in their chat template.
    """
    chat_template = getattr(tokenizer, "chat_template", "") or ""
    return "<|channel|>" in chat_template and "<|start|>" in chat_template


def tokenize_with_chat_template(
    tokenizer: Any,
    chat_messages: list[dict[str, str]],
    fallback_text: str,
) -> tuple[list[int], str]:
    """Tokenize using model's native chat template when available.

    Models like Gemma3 use special turn markers (<start_of_turn>user, etc.)
    that are critical for proper identity handling in multi-turn conversations.
    Falls back to raw text tokenization if no chat template is available.

    For GPT-OSS models, uses reasoning_effort="low" to minimize analysis
    channel output and prevent the model from getting stuck in analysis mode.

    CRITICAL: Merges consecutive same-role messages for DeepSeek/Llama models
    that require strict user/assistant alternation. Without this, consecutive
    "User:" labels confuse the model and cause broken tokenization.

    Args:
        tokenizer: HuggingFace-compatible tokenizer.
        chat_messages: List of {"role": ..., "content": ...} dicts.
        fallback_text: Pre-formatted text to use if no chat template.

    Returns:
        Tuple of (token_ids, prompt_text) where prompt_text is the ACTUAL
        templated text that was tokenized (for cache prefix matching).
    """
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        try:
            # CRITICAL: Merge consecutive USER messages for models requiring alternation
            # DeepSeek and Gemma templates require strict user/assistant alternation.
            # Without merging, consecutive "user" messages cause template errors.
            # IMPORTANT: Only merge "user" role, NOT "assistant" - assistant messages
            # are the agent's own voice and must remain separate.
            chat_template = getattr(tokenizer, "chat_template", "") or ""
            is_deepseek = ("'User: '" in chat_template and "'Assistant: '" in chat_template)
            is_gemma = ("<start_of_turn>" in chat_template and "<end_of_turn>" in chat_template)

            messages_for_template = chat_messages
            if (is_deepseek or is_gemma) and chat_messages:
                merged = []
                for msg in chat_messages:
                    if not merged:
                        merged.append(dict(msg))
                        continue
                    prev = merged[-1]
                    # For Gemma: system role doesn't exist, convert to user
                    # Merge system into first user message, or convert system→user
                    if is_gemma and prev["role"] == "system" and msg["role"] == "user":
                        # Convert system to user and merge with following user
                        prev["role"] = "user"
                        prev["content"] = prev["content"] + "\n" + msg["content"]
                    elif is_gemma and prev["role"] == "system":
                        # Convert standalone system to user (no following user to merge)
                        prev["role"] = "user"
                        merged.append(dict(msg))
                    elif msg["role"] == "user" and prev["role"] == "user":
                        # Merge consecutive user messages
                        prev["content"] = prev["content"] + "\n" + msg["content"]
                    else:
                        merged.append(dict(msg))
                messages_for_template = merged
                model_type = "DeepSeek" if is_deepseek else "Gemma" if is_gemma else "Unknown"
                logger.info(
                    f"{model_type} tokenization: merged {len(chat_messages)} messages → "
                    f"{len(merged)} (collapsed consecutive user messages)"
                )

            # GPT-OSS: Use low reasoning to prevent analysis mode loops
            template_kwargs: dict[str, Any] = {
                "tokenize": True,
                "add_generation_prompt": True,
            }
            if _is_gpt_oss_tokenizer(tokenizer):
                template_kwargs["reasoning_effort"] = "low"
                logger.debug("GPT-OSS detected, using reasoning_effort=low")

            # CRITICAL FIX: Get the ACTUAL templated text that will be tokenized
            # This ensures prompt_text matches tokens for cache prefix matching
            template_kwargs_text = template_kwargs.copy()
            template_kwargs_text["tokenize"] = False
            templated_text = tokenizer.apply_chat_template(messages_for_template, **template_kwargs_text)

            # Now tokenize the templated text
            tokens = tokenizer.apply_chat_template(messages_for_template, **template_kwargs)
            logger.info(f"Template result: tokens_type={type(tokens).__name__}, tokens_is_list={isinstance(tokens, list)}, text_type={type(templated_text).__name__}, text_is_str={isinstance(templated_text, str)}")
            if isinstance(tokens, list) and isinstance(templated_text, str):
                logger.info(f"Template applied: {len(tokens)} tokens, {len(templated_text)} chars")
                return tokens, templated_text
            else:
                logger.warning(f"Template check failed: returning fallback (tokens={type(tokens).__name__}, text={type(templated_text).__name__})")
        except Exception as e:
            logger.warning(f"Chat template exception: {e}, falling back to raw text tokenization")
    return tokenizer.encode(fallback_text), fallback_text


def try_parse_json_at(text: str, start: int) -> tuple[dict[str, Any] | None, int]:
    """Try to parse a JSON object starting at the given position.

    Uses a bracket-counting approach to find the end of the JSON object,
    then validates with json.loads().

    Args:
        text: The full text
        start: Starting position (should be at '{')

    Returns:
        Tuple of (parsed_dict or None, end_position)
    """
    if start >= len(text) or text[start] != "{":
        return None, start

    depth = 0
    in_string = False
    escape_next = False
    end = start

    for i, char in enumerate(text[start:], start):
        if escape_next:
            escape_next = False
            continue

        if char == "\\" and in_string:
            escape_next = True
            continue

        if char == '"' and not escape_next:
            in_string = not in_string
            continue

        if in_string:
            continue

        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    else:
        return None, start

    try:
        json_str = text[start:end]
        return json.loads(json_str), end
    except json.JSONDecodeError:
        return None, start
