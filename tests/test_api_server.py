"""
Tests for Anthropic-compatible API Server

Tests both streaming and non-streaming endpoints, agent persistence,
and proper API response formatting.
"""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from fastapi.testclient import TestClient

from src.api_server import (
    app,
    APIServer,
    Message,
    MessagesRequest,
    get_server,
    _server_instance
)


@pytest.fixture
def mock_manager():
    """Mock PersistentAgentManager for testing."""
    manager = Mock()
    manager.agents = {}
    manager.tokenizer = Mock()
    manager.tokenizer.encode = Mock(return_value=[1, 2, 3, 4, 5])  # 5 tokens
    manager.create_agent = Mock()
    manager.load_agent = Mock(side_effect=ValueError("Not found"))
    manager.generate = Mock(return_value="This is a test response")
    return manager


@pytest.fixture
def api_server(mock_manager):
    """Create APIServer with mocked manager."""
    with patch('src.api_server.PersistentAgentManager', return_value=mock_manager):
        server = APIServer()
        server.manager = mock_manager
        return server


@pytest.fixture
def client(api_server):
    """FastAPI test client."""
    # Reset global server instance
    import src.api_server
    src.api_server._server_instance = api_server

    return TestClient(app)


def test_system_to_agent_id(api_server):
    """Test system prompt hashing to agent ID."""
    # Same prompt = same ID
    id1 = api_server._system_to_agent_id("You are helpful")
    id2 = api_server._system_to_agent_id("You are helpful")
    assert id1 == id2
    assert len(id1) == 12

    # Different prompt = different ID
    id3 = api_server._system_to_agent_id("You are different")
    assert id3 != id1

    # None prompt = default
    id4 = api_server._system_to_agent_id(None)
    assert len(id4) == 12


def test_count_tokens(api_server):
    """Test token counting."""
    count = api_server._count_tokens("test text")
    assert count == 5  # Mock returns 5 tokens


def test_build_prompt_from_messages(api_server):
    """Test message list to prompt conversion."""
    messages = [
        Message(role="user", content="Hello"),
        Message(role="assistant", content="Hi there"),
        Message(role="user", content="How are you?")
    ]

    prompt = api_server._build_prompt_from_messages(messages)

    assert "User: Hello" in prompt
    assert "Assistant: Hi there" in prompt
    assert "User: How are you?" in prompt
    assert "Assistant:" in prompt  # Final prefix


def test_ensure_agent_creates_new(api_server):
    """Test _ensure_agent creates new agent if not found."""
    agent_id = "test123"
    system_prompt = "Test prompt"

    api_server._ensure_agent(agent_id, system_prompt)

    # Should try to load first
    api_server.manager.load_agent.assert_called_once_with(agent_id)

    # Then create when load fails
    api_server.manager.create_agent.assert_called_once_with(
        agent_id=agent_id,
        agent_type="api_agent",
        system_prompt=system_prompt
    )


def test_ensure_agent_loads_existing(api_server):
    """Test _ensure_agent loads from disk if exists."""
    agent_id = "test123"
    system_prompt = "Test prompt"

    # Mock successful load
    api_server.manager.load_agent = Mock(return_value=Mock())

    api_server._ensure_agent(agent_id, system_prompt)

    # Should load
    api_server.manager.load_agent.assert_called_once_with(agent_id)

    # Should NOT create
    api_server.manager.create_agent.assert_not_called()


def test_ensure_agent_already_in_memory(api_server):
    """Test _ensure_agent skips if agent in memory."""
    agent_id = "test123"
    system_prompt = "Test prompt"

    # Agent already in memory
    api_server.manager.agents[agent_id] = Mock()

    api_server._ensure_agent(agent_id, system_prompt)

    # Should not try to load or create
    api_server.manager.load_agent.assert_not_called()
    api_server.manager.create_agent.assert_not_called()


@pytest.mark.asyncio
async def test_handle_messages_non_streaming(api_server):
    """Test non-streaming message handling."""
    request = MessagesRequest(
        model="test-model",
        messages=[
            Message(role="user", content="Hello")
        ],
        max_tokens=100,
        temperature=0.5,
        system="You are helpful",
        stream=False
    )

    response = await api_server.handle_messages(request)

    # Check response format
    assert response.type == "message"
    assert response.role == "assistant"
    assert len(response.content) == 1
    assert response.content[0].text == "This is a test response"
    assert response.stop_reason == "end_turn"
    assert response.usage.input_tokens == 5
    assert response.usage.output_tokens == 5


@pytest.mark.asyncio
async def test_handle_messages_stream(api_server):
    """Test streaming message handling."""
    request = MessagesRequest(
        model="test-model",
        messages=[
            Message(role="user", content="Hello")
        ],
        max_tokens=100,
        temperature=0.5,
        system="You are helpful",
        stream=True
    )

    events = []
    async for event in api_server.handle_messages_stream(request):
        events.append(event)

    # Should have all event types
    event_types = [e.split('\n')[0].split(': ')[1] for e in events]

    assert "message_start" in event_types
    assert "content_block_start" in event_types
    assert "content_block_delta" in event_types
    assert "content_block_stop" in event_types
    assert "message_delta" in event_types
    assert "message_stop" in event_types


def test_post_messages_non_streaming(client):
    """Test POST /v1/messages non-streaming."""
    response = client.post(
        "/v1/messages",
        json={
            "model": "test-model",
            "messages": [
                {"role": "user", "content": "Hello"}
            ],
            "max_tokens": 100,
            "stream": False
        }
    )

    assert response.status_code == 200
    data = response.json()

    assert data["type"] == "message"
    assert data["role"] == "assistant"
    assert len(data["content"]) == 1
    assert "text" in data["content"][0]
    assert "usage" in data
    assert "input_tokens" in data["usage"]
    assert "output_tokens" in data["usage"]


def test_post_messages_streaming(client):
    """Test POST /v1/messages streaming."""
    response = client.post(
        "/v1/messages",
        json={
            "model": "test-model",
            "messages": [
                {"role": "user", "content": "Hello"}
            ],
            "max_tokens": 100,
            "stream": True
        }
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "text/event-stream; charset=utf-8"

    # Parse SSE events
    events = response.text.split("\n\n")
    event_types = []

    for event in events:
        if event.strip():
            lines = event.split("\n")
            for line in lines:
                if line.startswith("event: "):
                    event_types.append(line.split(": ")[1])

    assert "message_start" in event_types
    assert "content_block_start" in event_types
    assert "message_stop" in event_types


def test_health_endpoint(client):
    """Test GET /health."""
    response = client.get("/health")

    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "healthy"
    assert "model" in data
    assert "agents_in_memory" in data
