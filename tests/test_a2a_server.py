"""
Tests for A2A Protocol Integration

Tests agent card, task execution, and multi-agent delegation.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from starlette.testclient import TestClient

from src.a2a_server import (
    PersistentA2AExecutor,
    create_agent_card,
    create_a2a_app,
    get_executor
)


@pytest.fixture
def mock_manager():
    """Mock PersistentAgentManager for testing."""
    manager = Mock()
    manager.agents = {
        "a2a_tech": Mock(cache_tokens=100),
        "a2a_biz": Mock(cache_tokens=100),
        "a2a_coord": Mock(cache_tokens=100)
    }
    manager.create_agent = Mock()
    manager.load_agent = Mock(side_effect=ValueError("Not found"))
    manager.save_agent = Mock()
    manager.generate = Mock(return_value="This is a test A2A response")
    return manager


@pytest.fixture
def a2a_executor(mock_manager):
    """Create PersistentA2AExecutor with mocked manager."""
    with patch('src.a2a_server.PersistentAgentManager', return_value=mock_manager):
        executor = PersistentA2AExecutor()
        executor.manager = mock_manager
        return executor


@pytest.fixture
def client(a2a_executor):
    """Starlette test client for A2A app."""
    # Reset global executor instance
    import src.a2a_server
    src.a2a_server._executor_instance = a2a_executor

    app = create_a2a_app()
    return TestClient(app)


def test_agent_card_structure():
    """Test agent card has correct structure."""
    card = create_agent_card()

    assert "name" in card
    assert "description" in card
    assert "version" in card
    assert "skills" in card
    assert "capabilities" in card

    # Check skills
    assert len(card["skills"]) == 3
    skill_ids = [s["id"] for s in card["skills"]]
    assert "technical_analysis" in skill_ids
    assert "business_analysis" in skill_ids
    assert "coordination" in skill_ids

    # Check capabilities
    assert card["capabilities"]["streaming"] is True
    assert card["capabilities"]["persistent_cache"] is True
    assert card["capabilities"]["multi_agent"] is True


def test_executor_initialization(a2a_executor):
    """Test executor initializes with skill-agent mapping."""
    assert "technical_analysis" in a2a_executor.skill_agents
    assert "business_analysis" in a2a_executor.skill_agents
    assert "coordination" in a2a_executor.skill_agents

    assert a2a_executor.skill_agents["technical_analysis"] == "a2a_tech"
    assert a2a_executor.skill_agents["business_analysis"] == "a2a_biz"
    assert a2a_executor.skill_agents["coordination"] == "a2a_coord"


def test_execute_task_technical(a2a_executor):
    """Test executing technical analysis task."""
    task = {
        "skill": "technical_analysis",
        "message": "Analyze this system"
    }

    result = a2a_executor.execute_task(task)

    assert "result" in result
    assert "agent_id" in result
    assert "cache_tokens" in result

    assert result["result"] == "This is a test A2A response"
    assert result["agent_id"] == "a2a_tech"
    assert result["cache_tokens"] == 100


def test_execute_task_business(a2a_executor):
    """Test executing business analysis task."""
    task = {
        "skill": "business_analysis",
        "message": "What's the ROI?"
    }

    result = a2a_executor.execute_task(task)

    assert result["agent_id"] == "a2a_biz"
    assert result["cache_tokens"] == 100


def test_execute_task_coordination(a2a_executor):
    """Test executing coordination task."""
    task = {
        "skill": "coordination",
        "message": "Synthesize these inputs"
    }

    result = a2a_executor.execute_task(task)

    assert result["agent_id"] == "a2a_coord"
    assert result["cache_tokens"] == 100


def test_execute_task_default_skill(a2a_executor):
    """Test task with unknown skill defaults to coordination."""
    task = {
        "skill": "unknown_skill",
        "message": "Test message"
    }

    result = a2a_executor.execute_task(task)

    # Should default to coordinator
    assert result["agent_id"] == "a2a_coord"


def test_get_agent_card_endpoint(client):
    """Test GET /.well-known/agent.json endpoint."""
    response = client.get("/.well-known/agent.json")

    assert response.status_code == 200
    card = response.json()

    assert "name" in card
    assert "skills" in card
    assert len(card["skills"]) == 3


def test_post_task_endpoint(client):
    """Test POST /tasks endpoint."""
    response = client.post(
        "/tasks",
        json={
            "skill": "technical_analysis",
            "message": "Test message"
        }
    )

    assert response.status_code == 200
    result = response.json()

    assert "result" in result
    assert "agent_id" in result
    assert result["agent_id"] == "a2a_tech"


def test_health_endpoint(client):
    """Test GET /health endpoint."""
    response = client.get("/health")

    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "healthy"
    assert "agents_in_memory" in data
    assert "skills" in data
    assert len(data["skills"]) == 3


def test_multi_agent_workflow(a2a_executor):
    """Test multi-agent task delegation workflow."""
    # Task 1: Technical analysis
    tech_task = {
        "skill": "technical_analysis",
        "message": "Analyze architecture"
    }
    tech_result = a2a_executor.execute_task(tech_task)
    assert tech_result["agent_id"] == "a2a_tech"

    # Task 2: Business analysis
    biz_task = {
        "skill": "business_analysis",
        "message": "Analyze ROI"
    }
    biz_result = a2a_executor.execute_task(biz_task)
    assert biz_result["agent_id"] == "a2a_biz"

    # Task 3: Coordination synthesizes
    coord_task = {
        "skill": "coordination",
        "message": "Synthesize tech and biz analysis"
    }
    coord_result = a2a_executor.execute_task(coord_task)
    assert coord_result["agent_id"] == "a2a_coord"

    # All should return results
    assert all(r["result"] for r in [tech_result, biz_result, coord_result])
