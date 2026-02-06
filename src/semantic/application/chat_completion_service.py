"""Shared chat completion logic used by OpenAI API and Coordination."""

import asyncio
import structlog
from typing import Any

from semantic.adapters.inbound.adapter_helpers import tokenize_with_chat_template

logger = structlog.get_logger(__name__)


async def generate_chat_completion(
    messages: list[dict[str, str]],
    batch_engine: Any,
    cache_store: Any,
    scheduler: Any,
    agent_id: str,
    max_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.95,
    top_k: int | None = None,
) -> dict[str, Any]:
    """Generate a chat completion using shared logic.

    This is the core generation logic used by both:
    - OpenAI API endpoint (/v1/chat/completions)
    - Coordination service (multi-agent conversations)

    Args:
        messages: List of {"role": "...", "content": "..."} dicts
        batch_engine: BatchEngine for inference
        cache_store: AgentCacheStore for KV cache persistence
        scheduler: ConcurrentScheduler (if available) for concurrent requests
        agent_id: Unique agent identifier for cache lookup
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        top_k: Top-k sampling parameter (optional)

    Returns:
        dict with:
            - "text": Generated text (clean, no space stripping)
            - "token_count": Number of tokens generated
            - "finish_reason": "stop" or "length"
            - "blocks": Updated cache blocks (for saving)

    Raises:
        Exception: If generation fails
    """
    # Convert to prompt text for fallback
    # Simple conversion - just concatenate role/content
    prompt_parts = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "system":
            prompt_parts.append(content)
        elif role == "user":
            prompt_parts.append(f"User: {content}")
        elif role == "assistant":
            prompt_parts.append(f"Assistant: {content}")
    prompt = "\n\n".join(prompt_parts)

    # Tokenize using chat template (handles model-specific formatting)
    # Note: tokenize_with_chat_template() handles message merging for DeepSeek
    # CRITICAL: Now returns (tokens, templated_text) tuple for cache matching
    tokenizer = batch_engine.tokenizer
    tokens, templated_text = await asyncio.to_thread(
        tokenize_with_chat_template,
        tokenizer,
        messages,
        prompt,
    )

    logger.debug(f"Tokenized {len(messages)} messages → {len(tokens)} tokens")
    logger.info(f"Using templated text: {len(templated_text)} chars vs fallback {len(prompt)} chars")

    # Load cached blocks
    cached_blocks = cache_store.load(agent_id)
    if cached_blocks:
        logger.info(f"Cache hit: {agent_id} ({cached_blocks.total_tokens} tokens)")
    else:
        logger.info(f"Cache miss: {agent_id}")

    # Note: We intentionally do NOT call cache_store.invalidate_hot() here.
    # Premature invalidation would remove the hot entry between load() and save(),
    # causing concurrent requests for the same agent to miss the hot cache and
    # fall through to disk. save() at the end replaces the hot entry with
    # updated blocks, and load()'s has_data check handles cleared layer_data
    # by falling through to disk automatically.

    # Generate using scheduler (preferred) or direct engine
    if scheduler is not None:
        completion = await scheduler.submit_and_wait(
            agent_id=agent_id,
            prompt_tokens=tokens,
            cache=cached_blocks,
            max_tokens=max_tokens,
            prompt_text=templated_text,  # Use actual templated text, not fallback
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
    else:
        # Direct path (legacy, no concurrency protection)
        logger.warning("Using direct batch_engine path (no scheduler)")

        def run_step_for_uid(engine, uid):
            """Run generation steps until completion."""
            while True:
                completions = engine.step()
                for c in completions:
                    if c.uid == uid:
                        return c

        uid = await asyncio.to_thread(
            batch_engine.submit,
            agent_id=agent_id,
            prompt=prompt,
            cache=cached_blocks,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        logger.debug(f"Submitted generation: uid={uid}")
        completion = await asyncio.to_thread(run_step_for_uid, batch_engine, uid)

    if completion is None:
        raise RuntimeError("Generation failed - no completion returned")

    logger.debug(
        f"Generation complete: {completion.token_count} tokens, "
        f"finish_reason={completion.finish_reason}"
    )

    # Get updated cache blocks
    updated_blocks = batch_engine.get_agent_blocks(agent_id)
    if updated_blocks:
        cache_store.save(agent_id, updated_blocks)
        logger.debug(f"Saved cache: {agent_id} ({updated_blocks.total_tokens} tokens)")

    return {
        "text": completion.text,
        "token_count": completion.token_count,
        "finish_reason": completion.finish_reason,
        "blocks": updated_blocks,
    }
