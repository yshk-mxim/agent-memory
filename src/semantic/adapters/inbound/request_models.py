"""Pydantic request and response models for inbound API adapters.

Defines models for:
- Anthropic Messages API (/v1/messages)
- OpenAI Chat Completions API (/v1/chat/completions)
- Content blocks (text, thinking, tool_use, tool_result)
- Request validation and response serialization
"""

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


# ============================================================================
# Anthropic Messages API Models
# ============================================================================


class TextContentBlock(BaseModel):
    """Text content block."""

    type: Literal["text"] = "text"
    text: str


class ThinkingContentBlock(BaseModel):
    """Thinking content block (extended thinking)."""

    type: Literal["thinking"] = "thinking"
    thinking: str


class ToolUseContentBlock(BaseModel):
    """Tool use content block."""

    type: Literal["tool_use"] = "tool_use"
    id: str
    name: str
    input: dict[str, Any]


class ToolResultContentBlock(BaseModel):
    """Tool result content block."""

    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str
    content: str | list[dict[str, Any]]
    is_error: bool = False


# Union of all content block types
ContentBlock = (
    TextContentBlock | ThinkingContentBlock | ToolUseContentBlock | ToolResultContentBlock
)


class Message(BaseModel):
    """Message in conversation (user or assistant)."""

    role: Literal["user", "assistant"]
    content: str | list[ContentBlock]


class SystemBlock(BaseModel):
    """System prompt block with optional cache control."""

    type: Literal["text"] = "text"
    text: str
    cache_control: dict[str, str] | None = None


class Tool(BaseModel):
    """Tool definition."""

    name: str
    description: str
    input_schema: dict[str, Any]


class ThinkingConfig(BaseModel):
    """Extended thinking configuration."""

    type: Literal["enabled", "disabled"] = "enabled"
    budget_tokens: int = Field(default=1000, ge=0, le=10000)


class MessagesRequest(BaseModel):
    """Request to Anthropic Messages API (POST /v1/messages).

    Example:
        {
          "model": "claude-sonnet-4-5-20250929",
          "max_tokens": 1024,
          "messages": [
            {"role": "user", "content": "Hello!"}
          ]
        }
    """

    model: str
    messages: list[Message]
    max_tokens: int = Field(ge=1, le=8192)
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    top_k: int = Field(default=0, ge=0)
    stream: bool = False
    stop_sequences: list[str] = Field(default_factory=list)
    system: str | list[SystemBlock] = ""
    tools: list[Tool] = Field(default_factory=list)
    thinking: ThinkingConfig | None = None

    @field_validator("messages")
    @classmethod
    def validate_message_alternation(cls, messages: list[Message]) -> list[Message]:
        """Validate that messages alternate between user and assistant."""
        if not messages:
            raise ValueError("At least one message is required")

        # First message must be user
        if messages[0].role != "user":
            raise ValueError("First message must have role 'user'")

        # Check alternation
        for i in range(1, len(messages)):
            if messages[i].role == messages[i - 1].role:
                raise ValueError(
                    f"Messages must alternate between user and assistant "
                    f"(found consecutive {messages[i].role} at index {i})"
                )

        return messages


class Usage(BaseModel):
    """Token usage statistics."""

    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


class MessagesResponse(BaseModel):
    """Response from Anthropic Messages API.

    Example:
        {
          "id": "msg_01...",
          "type": "message",
          "role": "assistant",
          "content": [{"type": "text", "text": "Hello!"}],
          "model": "claude-sonnet-4-5-20250929",
          "stop_reason": "end_turn",
          "usage": {...}
        }
    """

    id: str
    type: Literal["message"] = "message"
    role: Literal["assistant"] = "assistant"
    content: list[ContentBlock]
    model: str
    stop_reason: Literal["end_turn", "max_tokens", "stop_sequence", "tool_use"] | None
    stop_sequence: str | None = None
    usage: Usage


# ============================================================================
# Anthropic SSE Streaming Events
# ============================================================================


class MessageStartEvent(BaseModel):
    """SSE event: message_start."""

    type: Literal["message_start"] = "message_start"
    message: MessagesResponse


class ContentBlockStartEvent(BaseModel):
    """SSE event: content_block_start."""

    type: Literal["content_block_start"] = "content_block_start"
    index: int
    content_block: ContentBlock


class ContentBlockDeltaEvent(BaseModel):
    """SSE event: content_block_delta."""

    type: Literal["content_block_delta"] = "content_block_delta"
    index: int
    delta: dict[str, Any]  # {"type": "text_delta", "text": "..."}


class ContentBlockStopEvent(BaseModel):
    """SSE event: content_block_stop."""

    type: Literal["content_block_stop"] = "content_block_stop"
    index: int


class MessageDeltaEvent(BaseModel):
    """SSE event: message_delta."""

    type: Literal["message_delta"] = "message_delta"
    delta: dict[str, Any]  # {"stop_reason": "end_turn"}
    usage: Usage


class MessageStopEvent(BaseModel):
    """SSE event: message_stop."""

    type: Literal["message_stop"] = "message_stop"


# ============================================================================
# OpenAI Chat Completions API Models
# ============================================================================


class OpenAIChatMessage(BaseModel):
    """Message in OpenAI chat format."""

    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None
    name: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None


class OpenAITool(BaseModel):
    """Tool definition in OpenAI format."""

    type: Literal["function"] = "function"
    function: dict[str, Any]


class ChatCompletionsRequest(BaseModel):
    """Request to OpenAI Chat Completions API.

    Example:
        {
          "model": "gpt-4",
          "messages": [
            {"role": "user", "content": "Hello!"}
          ]
        }
    """

    model: str
    messages: list[OpenAIChatMessage]
    max_tokens: int | None = None
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    stream: bool = False
    stop: str | list[str] | None = None
    tools: list[OpenAITool] | None = None
    tool_choice: str | dict[str, Any] | None = None

    # Extension: session ID for persistent cache
    session_id: str | None = Field(default=None, description="Optional session ID for cache persistence")

    @field_validator("messages")
    @classmethod
    def validate_messages_not_empty(cls, messages: list[OpenAIChatMessage]) -> list[OpenAIChatMessage]:
        """Validate that messages list is not empty."""
        if not messages:
            raise ValueError("At least one message is required")
        return messages


class OpenAIChatChoice(BaseModel):
    """Choice in OpenAI chat response."""

    index: int
    message: OpenAIChatMessage
    finish_reason: str | None


class OpenAIChatCompletionUsage(BaseModel):
    """Token usage in OpenAI format."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionsResponse(BaseModel):
    """Response from OpenAI Chat Completions API.

    Example:
        {
          "id": "chatcmpl-...",
          "object": "chat.completion",
          "created": 1234567890,
          "model": "gpt-4",
          "choices": [{...}],
          "usage": {...}
        }
    """

    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[OpenAIChatChoice]
    usage: OpenAIChatCompletionUsage


# ============================================================================
# Direct Agent API Models
# ============================================================================


class CreateAgentRequest(BaseModel):
    """Request to create a new agent."""

    agent_id: str | None = None


class CreateAgentResponse(BaseModel):
    """Response from creating an agent."""

    agent_id: str
    created_at: str


class GenerateRequest(BaseModel):
    """Request to generate text for an agent."""

    prompt: str
    max_tokens: int = Field(default=256, ge=1, le=8192)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)


class GenerateResponse(BaseModel):
    """Response from generation."""

    text: str
    tokens: list[int]
    usage: Usage


class AgentInfoResponse(BaseModel):
    """Information about an agent."""

    agent_id: str
    total_tokens: int
    num_blocks: int
    created_at: str


# ============================================================================
# Token Counting Models
# ============================================================================


class CountTokensRequest(BaseModel):
    """Request to count tokens (POST /v1/messages/count_tokens)."""

    model: str
    messages: list[Message]
    system: str | list[SystemBlock] = ""
    tools: list[Tool] = Field(default_factory=list)


class CountTokensResponse(BaseModel):
    """Response from token counting."""

    input_tokens: int
