# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Yakov Shkolnikov and contributors
"""Common generation request model.

Protocol-agnostic request that both Anthropic and OpenAI adapters
transform their API-specific requests into. The TRTInferenceService
(and future MLX inference service) consumes this.
"""

from dataclasses import dataclass, field


@dataclass
class GenerationRequest:
    """Unified generation request for all backends and API protocols.

    Inbound adapters (Anthropic, OpenAI) transform protocol-specific
    requests into this common model. The application service handles
    generation without knowing which API the request came from.

    Attributes:
        agent_id: Unique identifier for cache persistence (from session or hash).
        messages: Chat messages with role and content. System prompt is the
            first message with role='system'. Tools are appended to system.
        prompt: Full templated prompt string (for tokenization/fallback).
        max_tokens: Maximum tokens to generate.
        temperature: Sampling temperature.
        top_p: Nucleus sampling parameter.
        top_k: Top-k sampling parameter.
        stop_sequences: Stop strings to truncate output at.
        stream: Whether to stream the response (handled by adapter, not service).
        model: Model name (informational, for response).
    """

    agent_id: str
    messages: list[dict[str, str]]
    prompt: str = ""
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 40
    stop_sequences: list[str] = field(default_factory=list)
    stream: bool = False
    model: str = ""

    # FIM (fill-in-the-middle) for code completion
    fim_prefix: str | None = None  # Code before cursor
    fim_suffix: str | None = None  # Code after cursor
    fim_mode: bool = False  # Enable FIM template instead of chat

    # Repetition control
    repetition_penalty: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0

    # System prompt pinning hint
    pin_system_cache: bool = False  # Pin system prompt KV cache in memory

    # OpenAI-format tool definitions (for backends that support native function calling)
    openai_tools: list[dict] | None = None
