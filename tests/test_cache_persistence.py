"""
Tests for CachePersistence

Tests cache save/load to disk using real safetensors format.
Uses temp directories and real KVCache objects.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import mlx.core as mx
from mlx_lm.models.cache import KVCache

from src.cache_persistence import CachePersistence


@pytest.fixture
def temp_cache_dir():
    """Create temporary cache directory."""
    temp_dir = tempfile.mkdtemp(prefix="test_cache_")
    yield temp_dir
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def persistence(temp_cache_dir):
    """Create CachePersistence with temp directory."""
    return CachePersistence(cache_dir=temp_cache_dir)


@pytest.fixture
def sample_cache():
    """Create a real KVCache with sample data."""
    cache = []
    for _ in range(2):  # 2 layers for faster tests
        layer_cache = KVCache()
        # Shape: (B=1, n_kv_heads=4, seq_len=5, head_dim=32)
        keys = mx.random.uniform(shape=(1, 4, 5, 32))
        values = mx.random.uniform(shape=(1, 4, 5, 32))
        layer_cache.keys = keys
        layer_cache.values = values
        layer_cache.offset = 5
        cache.append(layer_cache)
    return cache


def test_init_creates_directory(temp_cache_dir):
    """Test CachePersistence creates cache directory."""
    # Remove the directory to test creation
    shutil.rmtree(temp_cache_dir)

    # Create persistence (should create directory)
    persistence = CachePersistence(cache_dir=temp_cache_dir)

    assert Path(temp_cache_dir).exists()
    assert Path(temp_cache_dir).is_dir()


def test_save_and_load_roundtrip(persistence, sample_cache):
    """Test save and load roundtrip preserves cache."""
    agent_id = "test_agent_001"
    metadata = {
        'agent_type': 'technical',
        'model': 'test-model',
        'cache_tokens': 5
    }

    # Save
    persistence.save_agent_cache(agent_id, sample_cache, metadata)

    # Load
    loaded_cache, loaded_metadata = persistence.load_agent_cache(agent_id)

    # Verify cache structure
    assert len(loaded_cache) == len(sample_cache)
    assert loaded_cache[0].offset == sample_cache[0].offset

    # Verify metadata
    assert loaded_metadata['agent_id'] == agent_id
    assert loaded_metadata['agent_type'] == metadata['agent_type']
    assert 'timestamp' in loaded_metadata


def test_agent_cache_exists(persistence, sample_cache):
    """Test agent_cache_exists detection."""
    agent_id = "test_exists"

    # Should not exist initially
    assert not persistence.agent_cache_exists(agent_id)

    # Save cache
    persistence.save_agent_cache(agent_id, sample_cache)

    # Should exist now
    assert persistence.agent_cache_exists(agent_id)


def test_load_nonexistent_raises(persistence):
    """Test loading nonexistent cache raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        persistence.load_agent_cache("nonexistent_agent")


def test_list_cached_agents(persistence, sample_cache):
    """Test listing all cached agents."""
    # Save multiple agents
    agent_ids = ["agent_1", "agent_2", "agent_3"]
    for agent_id in agent_ids:
        persistence.save_agent_cache(agent_id, sample_cache, {'test': 'data'})

    # List agents
    agents = persistence.list_cached_agents()

    # Verify
    assert len(agents) == 3
    listed_ids = [a['agent_id'] for a in agents]
    assert set(listed_ids) == set(agent_ids)

    # Check structure
    for agent in agents:
        assert 'agent_id' in agent
        assert 'file_path' in agent
        assert 'file_size' in agent
        assert 'modified_time' in agent
        assert 'metadata' in agent


def test_delete_agent_cache(persistence, sample_cache):
    """Test deleting agent cache."""
    agent_id = "test_delete"

    # Save
    persistence.save_agent_cache(agent_id, sample_cache)
    assert persistence.agent_cache_exists(agent_id)

    # Delete
    result = persistence.delete_agent_cache(agent_id)
    assert result is True
    assert not persistence.agent_cache_exists(agent_id)

    # Delete non-existent
    result = persistence.delete_agent_cache(agent_id)
    assert result is False


def test_get_cache_disk_usage(persistence, sample_cache):
    """Test disk usage reporting."""
    # Save multiple agents
    agents = ["agent_a", "agent_b", "agent_c"]
    for agent_id in agents:
        persistence.save_agent_cache(agent_id, sample_cache)

    # Get disk usage
    usage = persistence.get_cache_disk_usage()

    # Verify structure
    assert 'total_bytes' in usage
    assert 'total_mb' in usage
    assert 'num_agents' in usage
    assert 'per_agent' in usage

    # Verify values
    assert usage['num_agents'] == 3
    assert usage['total_bytes'] > 0
    assert usage['total_mb'] > 0
    assert len(usage['per_agent']) == 3


def test_cache_dir_creation_with_tilde():
    """Test cache directory creation with ~ expansion."""
    # Create in temp location to avoid polluting home dir
    with tempfile.TemporaryDirectory() as temp_dir:
        cache_path = Path(temp_dir) / ".test_agent_caches"

        persistence = CachePersistence(cache_dir=str(cache_path))

        assert cache_path.exists()
        assert cache_path.is_dir()


def test_save_updates_metadata(persistence, sample_cache):
    """Test that save automatically adds metadata fields."""
    agent_id = "test_metadata"

    # Save with minimal metadata
    persistence.save_agent_cache(agent_id, sample_cache, {'custom': 'value'})

    # Load and check
    _, metadata = persistence.load_agent_cache(agent_id)

    # Should have auto-added fields
    assert 'agent_id' in metadata
    assert 'timestamp' in metadata
    assert 'cache_tokens' in metadata
    assert 'custom' in metadata

    assert metadata['agent_id'] == agent_id
    assert metadata['custom'] == 'value'
