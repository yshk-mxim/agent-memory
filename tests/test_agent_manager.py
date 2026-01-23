"""
Tests for PersistentAgentManager

Tests multi-agent orchestration with mocked model loading
but real cache operations.
"""

import pytest
import tempfile
import shutil
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime, timedelta
import mlx.core as mx
from mlx_lm.models.cache import KVCache

from src.agent_manager import PersistentAgentManager, AgentContext


@pytest.fixture
def temp_cache_dir():
    """Create temporary cache directory."""
    temp_dir = tempfile.mkdtemp(prefix="test_agent_cache_")
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_cache():
    """Create sample KVCache."""
    cache = []
    for _ in range(2):
        layer_cache = KVCache()
        keys = mx.random.uniform(shape=(1, 4, 5, 32))
        values = mx.random.uniform(shape=(1, 4, 5, 32))
        layer_cache.keys = keys
        layer_cache.values = values
        layer_cache.offset = 5
        cache.append(layer_cache)
    return cache


@pytest.fixture
def mock_model_load(sample_cache):
    """Mock model loading and generation."""
    with patch('src.agent_manager.MLXModelLoader.load_model') as mock_load, \
         patch('src.agent_manager.MLXCacheExtractor') as mock_extractor_class:

        # Mock model and tokenizer
        mock_model = Mock()
        mock_model.layers = [Mock() for _ in range(2)]
        mock_tokenizer = Mock()
        mock_load.return_value = (mock_model, mock_tokenizer)

        # Mock extractor
        mock_extractor = Mock()
        mock_extractor.process_prompt.return_value = sample_cache
        mock_extractor.generate_with_cache.return_value = ("Generated text", sample_cache)
        mock_extractor.get_cache_info.return_value = {
            'num_layers': 2,
            'total_tokens': 5,
            'memory_bytes': 1024
        }
        mock_extractor.get_cache_memory_bytes.return_value = 1024
        mock_extractor_class.return_value = mock_extractor

        yield {
            'load': mock_load,
            'extractor_class': mock_extractor_class,
            'extractor': mock_extractor,
            'model': mock_model,
            'tokenizer': mock_tokenizer
        }


@pytest.fixture
def manager(mock_model_load, temp_cache_dir):
    """Create PersistentAgentManager with mocks."""
    return PersistentAgentManager(
        model_name="test-model",
        max_agents=2,
        cache_dir=temp_cache_dir
    )


def test_init(manager, mock_model_load):
    """Test manager initialization."""
    assert manager.model == mock_model_load['model']
    assert manager.tokenizer == mock_model_load['tokenizer']
    assert manager.max_agents == 2
    assert len(manager.agents) == 0


def test_create_agent(manager):
    """Test agent creation."""
    agent = manager.create_agent(
        agent_id="tech_001",
        agent_type="technical",
        system_prompt="You are a technical expert."
    )

    assert agent.agent_id == "tech_001"
    assert agent.agent_type == "technical"
    assert agent.system_prompt == "You are a technical expert."
    assert agent.cache is not None
    assert agent.cache_tokens == 5

    # Should be in memory
    assert "tech_001" in manager.agents


def test_create_agent_duplicate_returns_existing(manager):
    """Test creating duplicate agent returns existing."""
    agent1 = manager.create_agent("agent_1", "type1", "prompt1")
    agent2 = manager.create_agent("agent_1", "type2", "prompt2")

    assert agent1 is agent2
    assert agent1.agent_type == "type1"  # Original not overwritten


def test_create_agent_triggers_eviction(manager):
    """Test creating agent when max_agents exceeded triggers eviction."""
    # Create max_agents (2)
    manager.create_agent("agent_1", "type1", "prompt1")
    manager.create_agent("agent_2", "type2", "prompt2")

    assert len(manager.agents) == 2

    # Create 3rd agent (should evict LRU)
    manager.create_agent("agent_3", "type3", "prompt3")

    # Should still have only 2 agents
    assert len(manager.agents) == 2
    # agent_1 should be evicted (least recently created)
    assert "agent_1" not in manager.agents
    assert "agent_2" in manager.agents
    assert "agent_3" in manager.agents


def test_lru_eviction_order(manager, temp_cache_dir):
    """Test LRU eviction selects least recently accessed."""
    # Create 2 agents
    agent1 = manager.create_agent("agent_1", "type1", "prompt1")
    agent2 = manager.create_agent("agent_2", "type2", "prompt2")

    # Update agent1's access time (make it more recent)
    agent1.update_access()

    # Create 3rd agent
    manager.create_agent("agent_3", "type3", "prompt3")

    # agent_2 should be evicted (oldest access)
    assert "agent_1" in manager.agents
    assert "agent_2" not in manager.agents
    assert "agent_3" in manager.agents


def test_generate_updates_cache(manager, mock_model_load):
    """Test generate updates agent's cache."""
    agent = manager.create_agent("test_agent", "type", "prompt")
    initial_cache = agent.cache

    response = manager.generate(
        agent_id="test_agent",
        user_input="Tell me about Python",
        max_tokens=50
    )

    assert response == "Generated text"
    assert len(agent.conversation_history) == 2  # user + assistant


def test_generate_nonexistent_agent_raises(manager):
    """Test generating for nonexistent agent raises."""
    with pytest.raises(ValueError, match="not in memory"):
        manager.generate("nonexistent", "test input")


def test_save_and_load_agent(manager, temp_cache_dir):
    """Test save and load agent with real persistence."""
    # Create and save agent
    agent = manager.create_agent("save_test", "technical", "System prompt")
    manager.save_agent("save_test")

    # Remove from memory
    del manager.agents["save_test"]

    # Load back
    loaded_agent = manager.load_agent("save_test")

    assert loaded_agent.agent_id == "save_test"
    assert loaded_agent.agent_type == "technical"
    assert loaded_agent.cache_tokens == 5


def test_load_agent_already_in_memory(manager):
    """Test loading agent already in memory returns existing."""
    agent = manager.create_agent("mem_test", "type", "prompt")
    original_access = agent.last_access

    # Wait a bit
    import time
    time.sleep(0.01)

    # Load (should return existing and update access)
    loaded = manager.load_agent("mem_test")

    assert loaded is agent
    assert loaded.last_access > original_access


def test_load_agent_not_found_raises(manager):
    """Test loading nonexistent agent raises."""
    with pytest.raises(ValueError, match="not found"):
        manager.load_agent("nonexistent")


def test_save_all(manager, temp_cache_dir):
    """Test save_all saves all agents."""
    manager.create_agent("agent_1", "type1", "prompt1")
    manager.create_agent("agent_2", "type2", "prompt2")

    manager.save_all()

    # Both should be saved
    assert manager.persistence.agent_cache_exists("agent_1")
    assert manager.persistence.agent_cache_exists("agent_2")


def test_memory_usage_reporting(manager):
    """Test get_memory_usage returns correct structure."""
    manager.create_agent("agent_1", "type1", "prompt1")
    manager.create_agent("agent_2", "type2", "prompt2")

    usage = manager.get_memory_usage()

    assert 'model_memory_gb' in usage
    assert 'agents' in usage
    assert 'total_cache_mb' in usage
    assert 'total_gb' in usage

    assert len(usage['agents']) == 2
    assert 'agent_1' in usage['agents']
    assert 'agent_2' in usage['agents']


def test_multi_agent_workflow(manager):
    """Test complete multi-agent workflow."""
    # Create multiple agents
    tech = manager.create_agent(
        "tech_specialist",
        "technical",
        "You are a technical expert."
    )

    biz = manager.create_agent(
        "biz_analyst",
        "business",
        "You are a business analyst."
    )

    # Generate from both
    tech_response = manager.generate("tech_specialist", "Analyze the API")
    biz_response = manager.generate("biz_analyst", "What's the ROI?")

    assert tech_response == "Generated text"
    assert biz_response == "Generated text"

    # Save all
    manager.save_all()

    # Verify saved
    assert manager.persistence.agent_cache_exists("tech_specialist")
    assert manager.persistence.agent_cache_exists("biz_analyst")

    # Clear memory
    manager.agents.clear()

    # Load back
    tech_loaded = manager.load_agent("tech_specialist")
    biz_loaded = manager.load_agent("biz_analyst")

    assert tech_loaded.agent_id == "tech_specialist"
    assert biz_loaded.agent_id == "biz_analyst"
