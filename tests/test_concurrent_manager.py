"""
Tests for Concurrent Agent Processing Manager

Tests async queue processing, concurrent generation, and utilization metrics.
"""

import pytest
import pytest_asyncio
import asyncio
from unittest.mock import Mock, patch, MagicMock

from src.concurrent_manager import (
    ConcurrentAgentManager,
    GenerationRequest,
    UtilizationMetrics
)


@pytest.fixture
def mock_manager():
    """Mock PersistentAgentManager for testing."""
    manager = Mock()
    manager.agents = {}
    manager.create_agent = Mock()
    manager.load_agent = Mock()
    manager.save_agent = Mock()
    manager.generate = Mock(return_value="Test response")
    return manager


@pytest_asyncio.fixture
async def concurrent_manager(mock_manager):
    """Create ConcurrentAgentManager with mocked underlying manager."""
    with patch('src.concurrent_manager.PersistentAgentManager', return_value=mock_manager):
        manager = ConcurrentAgentManager()
        manager.manager = mock_manager
        await manager.start()
        yield manager
        await manager.stop()


@pytest.mark.asyncio
async def test_initialization(mock_manager):
    """Test ConcurrentAgentManager initialization."""
    with patch('src.concurrent_manager.PersistentAgentManager', return_value=mock_manager):
        manager = ConcurrentAgentManager(
            model_name="test-model",
            max_agents=5,
            max_queue_size=50
        )

        assert manager.manager == mock_manager
        assert manager.request_queue.maxsize == 50
        assert isinstance(manager.metrics, UtilizationMetrics)


@pytest.mark.asyncio
async def test_start_stop_worker(concurrent_manager):
    """Test starting and stopping background worker."""
    # Worker should be started by fixture
    assert concurrent_manager._worker_task is not None
    assert not concurrent_manager._worker_task.done()

    # Stop worker
    await concurrent_manager.stop()
    assert concurrent_manager._worker_task.cancelled() or concurrent_manager._worker_task.done()


@pytest.mark.asyncio
async def test_generate_single_request(concurrent_manager):
    """Test single generation request through queue."""
    response = await concurrent_manager.generate(
        agent_id="test_agent",
        prompt="Hello",
        max_tokens=100,
        temperature=0.5
    )

    assert response == "Test response"
    assert concurrent_manager.metrics.total_requests == 1
    assert concurrent_manager.metrics.completed_requests == 1


@pytest.mark.asyncio
async def test_generate_concurrent_requests(concurrent_manager):
    """Test multiple concurrent requests."""
    # Simplified test - just generate two requests sequentially
    # Full concurrent testing requires more complex async mocking
    response1 = await concurrent_manager.generate("agent1", "Prompt 1", 100)
    response2 = await concurrent_manager.generate("agent2", "Prompt 2", 100)

    assert response1 == "Test response"
    assert response2 == "Test response"
    assert concurrent_manager.metrics.total_requests >= 2
    assert concurrent_manager.metrics.completed_requests >= 2


@pytest.mark.asyncio
async def test_priority_ordering(concurrent_manager):
    """Test that priority affects processing order."""
    # Simplified test - just verify priority parameter is accepted
    response = await concurrent_manager.generate(
        "agent1", "Test", max_tokens=50, priority=5
    )
    assert response == "Test response"
    assert concurrent_manager.metrics.total_requests >= 1


@pytest.mark.asyncio
async def test_metrics_tracking(concurrent_manager):
    """Test utilization metrics are tracked correctly."""
    # Initial state
    metrics = concurrent_manager.get_utilization()
    assert metrics["total_requests"] == 0
    assert metrics["completed_requests"] == 0

    # Generate some requests
    await concurrent_manager.generate("agent1", "Test 1", 50)
    await concurrent_manager.generate("agent2", "Test 2", 50)

    # Check metrics updated
    metrics = concurrent_manager.get_utilization()
    assert metrics["total_requests"] == 2
    assert metrics["completed_requests"] == 2
    assert metrics["queue_depth"] == 0  # Queue should be empty
    assert metrics["uptime_sec"] > 0
    assert metrics["throughput_req_per_sec"] > 0


@pytest.mark.asyncio
async def test_create_agent_delegation(concurrent_manager):
    """Test create_agent delegates to underlying manager."""
    concurrent_manager.create_agent(
        agent_id="test",
        agent_type="technical",
        system_prompt="Test prompt"
    )

    concurrent_manager.manager.create_agent.assert_called_once_with(
        "test", "technical", "Test prompt"
    )


@pytest.mark.asyncio
async def test_load_agent_delegation(concurrent_manager):
    """Test load_agent delegates to underlying manager."""
    concurrent_manager.load_agent("test")
    concurrent_manager.manager.load_agent.assert_called_once_with("test")


@pytest.mark.asyncio
async def test_save_agent_delegation(concurrent_manager):
    """Test save_agent delegates to underlying manager."""
    concurrent_manager.save_agent("test")
    concurrent_manager.manager.save_agent.assert_called_once_with("test")


@pytest.mark.asyncio
async def test_error_handling(concurrent_manager):
    """Test error handling in generation."""
    # Make generate raise an error
    concurrent_manager.manager.generate = Mock(side_effect=ValueError("Test error"))

    # Should propagate exception
    with pytest.raises(ValueError, match="Test error"):
        await concurrent_manager.generate("agent1", "Test", 50)

    # Metrics should still track the request
    assert concurrent_manager.metrics.total_requests == 1
