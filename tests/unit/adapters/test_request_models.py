"""Unit tests for request and response models.

Tests validation, serialization, and parsing for:
- Anthropic Messages API models
- OpenAI Chat Completions API models
- Content blocks
- Message alternation
"""

import pytest
from pydantic import ValidationError

from agent_memory.adapters.inbound.request_models import (
    ChatCompletionsRequest,
    ContentBlockDeltaEvent,
    CountTokensRequest,
    CreateAgentRequest,
    GenerateRequest,
    Message,
    MessagesRequest,
    MessagesResponse,
    OpenAIChatMessage,
    SystemBlock,
    TextContentBlock,
    ThinkingConfig,
    ThinkingContentBlock,
    Tool,
    ToolResultContentBlock,
    ToolUseContentBlock,
    Usage,
)


@pytest.mark.unit
class TestAnthropicContentBlocks:
    """Test Anthropic content block models."""

    def test_text_content_block(self):
        """Text content block should serialize correctly."""
        block = TextContentBlock(text="Hello world")

        assert block.type == "text"
        assert block.text == "Hello world"
        assert block.model_dump() == {"type": "text", "text": "Hello world"}

    def test_thinking_content_block(self):
        """Thinking content block should serialize correctly."""
        block = ThinkingContentBlock(thinking="Let me think...")

        assert block.type == "thinking"
        assert block.thinking == "Let me think..."
        assert block.model_dump() == {"type": "thinking", "thinking": "Let me think..."}

    def test_tool_use_content_block(self):
        """Tool use content block should serialize correctly."""
        block = ToolUseContentBlock(id="tool_1", name="search", input={"query": "test"})

        assert block.type == "tool_use"
        assert block.id == "tool_1"
        assert block.name == "search"
        assert block.input == {"query": "test"}

    def test_tool_result_content_block(self):
        """Tool result content block should serialize correctly."""
        block = ToolResultContentBlock(tool_use_id="tool_1", content="Result", is_error=False)

        assert block.type == "tool_result"
        assert block.tool_use_id == "tool_1"
        assert block.content == "Result"
        assert block.is_error is False


@pytest.mark.unit
class TestMessagesRequest:
    """Test Anthropic MessagesRequest validation."""

    def test_valid_simple_request(self):
        """Simple valid request should parse correctly."""
        request = MessagesRequest(
            model="claude-sonnet-4-5-20250929",
            max_tokens=1024,
            messages=[Message(role="user", content="Hello!")],
        )

        assert request.model == "claude-sonnet-4-5-20250929"
        assert request.max_tokens == 1024
        assert len(request.messages) == 1
        assert request.messages[0].role == "user"
        assert request.messages[0].content == "Hello!"

    def test_valid_alternating_messages(self):
        """Alternating user/assistant messages should be valid."""
        request = MessagesRequest(
            model="claude-sonnet-4-5-20250929",
            max_tokens=1024,
            messages=[
                Message(role="user", content="Hello!"),
                Message(role="assistant", content="Hi there!"),
                Message(role="user", content="How are you?"),
            ],
        )

        assert len(request.messages) == 3

    def test_reject_empty_messages(self):
        """Empty messages list should be rejected."""
        with pytest.raises(ValidationError, match="At least one message is required"):
            MessagesRequest(
                model="claude-sonnet-4-5-20250929",
                max_tokens=1024,
                messages=[],
            )

    def test_reject_first_message_not_user(self):
        """First message must be from user."""
        with pytest.raises(ValidationError, match="First message must have role 'user'"):
            MessagesRequest(
                model="claude-sonnet-4-5-20250929",
                max_tokens=1024,
                messages=[Message(role="assistant", content="Hello!")],
            )

    def test_reject_consecutive_same_role(self):
        """Consecutive messages with same role should be rejected."""
        with pytest.raises(ValidationError, match="must alternate"):
            MessagesRequest(
                model="claude-sonnet-4-5-20250929",
                max_tokens=1024,
                messages=[
                    Message(role="user", content="Hello!"),
                    Message(role="user", content="Again!"),
                ],
            )

    def test_system_prompt_string(self):
        """System prompt as string should be valid."""
        request = MessagesRequest(
            model="claude-sonnet-4-5-20250929",
            max_tokens=1024,
            messages=[Message(role="user", content="Hello!")],
            system="You are a helpful assistant.",
        )

        assert request.system == "You are a helpful assistant."

    def test_system_prompt_blocks(self):
        """System prompt as blocks should be valid."""
        request = MessagesRequest(
            model="claude-sonnet-4-5-20250929",
            max_tokens=1024,
            messages=[Message(role="user", content="Hello!")],
            system=[
                SystemBlock(text="You are a helpful assistant."),
                SystemBlock(text="You are an expert.", cache_control={"type": "ephemeral"}),
            ],
        )

        assert len(request.system) == 2
        assert request.system[1].cache_control == {"type": "ephemeral"}

    def test_tools_definition(self):
        """Tools should be parsed correctly."""
        request = MessagesRequest(
            model="claude-sonnet-4-5-20250929",
            max_tokens=1024,
            messages=[Message(role="user", content="Search for cats")],
            tools=[
                Tool(
                    name="search",
                    description="Search the web",
                    input_schema={"type": "object", "properties": {"query": {"type": "string"}}},
                )
            ],
        )

        assert len(request.tools) == 1
        assert request.tools[0].name == "search"

    def test_thinking_config(self):
        """Thinking configuration should be parsed correctly."""
        request = MessagesRequest(
            model="claude-sonnet-4-5-20250929",
            max_tokens=1024,
            messages=[Message(role="user", content="Think carefully")],
            thinking=ThinkingConfig(type="enabled", budget_tokens=2000),
        )

        assert request.thinking.type == "enabled"
        assert request.thinking.budget_tokens == 2000

    def test_temperature_validation(self):
        """Temperature should be validated to 0-2 range."""
        # Valid
        MessagesRequest(
            model="claude-sonnet-4-5-20250929",
            max_tokens=1024,
            messages=[Message(role="user", content="Hello")],
            temperature=0.0,
        )
        MessagesRequest(
            model="claude-sonnet-4-5-20250929",
            max_tokens=1024,
            messages=[Message(role="user", content="Hello")],
            temperature=2.0,
        )

        # Invalid
        with pytest.raises(ValidationError):
            MessagesRequest(
                model="claude-sonnet-4-5-20250929",
                max_tokens=1024,
                messages=[Message(role="user", content="Hello")],
                temperature=-0.1,
            )
        with pytest.raises(ValidationError):
            MessagesRequest(
                model="claude-sonnet-4-5-20250929",
                max_tokens=1024,
                messages=[Message(role="user", content="Hello")],
                temperature=2.1,
            )


@pytest.mark.unit
class TestMessagesResponse:
    """Test Anthropic MessagesResponse serialization."""

    def test_serialize_simple_response(self):
        """Simple response should serialize correctly."""
        response = MessagesResponse(
            id="msg_01ABC",
            content=[TextContentBlock(text="Hello!")],
            model="claude-sonnet-4-5-20250929",
            stop_reason="end_turn",
            usage=Usage(input_tokens=10, output_tokens=5),
        )

        data = response.model_dump()
        assert data["id"] == "msg_01ABC"
        assert data["type"] == "message"
        assert data["role"] == "assistant"
        assert len(data["content"]) == 1
        assert data["content"][0]["type"] == "text"
        assert data["content"][0]["text"] == "Hello!"


@pytest.mark.unit
class TestOpenAIModels:
    """Test OpenAI Chat Completions models."""

    def test_valid_chat_request(self):
        """Valid OpenAI chat request should parse correctly."""
        request = ChatCompletionsRequest(
            model="gpt-4",
            messages=[OpenAIChatMessage(role="user", content="Hello!")],
        )

        assert request.model == "gpt-4"
        assert len(request.messages) == 1
        assert request.messages[0].role == "user"

    def test_session_id_extension(self):
        """Session ID extension should be parsed."""
        request = ChatCompletionsRequest(
            model="gpt-4",
            messages=[OpenAIChatMessage(role="user", content="Hello!")],
            session_id="session_123",
        )

        assert request.session_id == "session_123"

    def test_optional_fields(self):
        """Optional fields should have defaults."""
        request = ChatCompletionsRequest(
            model="gpt-4",
            messages=[OpenAIChatMessage(role="user", content="Hello!")],
        )

        assert request.max_tokens is None
        assert request.temperature == 1.0
        assert request.stream is False


@pytest.mark.unit
class TestDirectAgentModels:
    """Test Direct Agent API models."""

    def test_create_agent_without_id(self):
        """CreateAgentRequest without ID should be valid."""
        request = CreateAgentRequest()
        assert request.agent_id is None

    def test_create_agent_with_id(self):
        """CreateAgentRequest with ID should be valid."""
        request = CreateAgentRequest(agent_id="agent_123")
        assert request.agent_id == "agent_123"

    def test_generate_request(self):
        """GenerateRequest should parse correctly."""
        request = GenerateRequest(prompt="Hello", max_tokens=100, temperature=0.5)

        assert request.prompt == "Hello"
        assert request.max_tokens == 100
        assert request.temperature == 0.5


@pytest.mark.unit
class TestCountTokensModels:
    """Test token counting models."""

    def test_count_tokens_request(self):
        """CountTokensRequest should parse correctly."""
        request = CountTokensRequest(
            model="claude-sonnet-4-5-20250929",
            messages=[Message(role="user", content="Count these tokens")],
        )

        assert request.model == "claude-sonnet-4-5-20250929"
        assert len(request.messages) == 1


@pytest.mark.unit
class TestSSEEvents:
    """Test SSE streaming event models."""

    def test_content_block_delta_event(self):
        """ContentBlockDeltaEvent should serialize correctly."""
        event = ContentBlockDeltaEvent(index=0, delta={"type": "text_delta", "text": "Hello"})

        assert event.type == "content_block_delta"
        assert event.index == 0
        assert event.delta["text"] == "Hello"
