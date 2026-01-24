# Installation & Usage Guide

Complete guide to installing, configuring, and using the Persistent Multi-Agent Memory system.

---

## Prerequisites

### Hardware Requirements

- **Mac with Apple Silicon** (M1, M2, M3, M4)
- **Minimum 16GB RAM** (24GB+ recommended)
- **15GB free disk space**:
  - ~7GB for model weights (Gemma 3 12B 4-bit)
  - ~5GB for MLX framework and dependencies
  - ~1-3GB for agent caches

### Software Requirements

- **macOS 12.0+** (Monterey or later)
- **Python 3.10+**
- **Xcode Command Line Tools** (for MLX Metal support)

### Verify Metal Support

Check that Metal GPU acceleration is available:

```bash
# Check Metal availability
python3 -c "import mlx.core as mx; print(mx.metal.is_available())"
# Expected output: True
```

If Metal is not available, install Xcode Command Line Tools:

```bash
xcode-select --install
```

---

## Installation

### Step 1: Clone Repository

```bash
git clone https://github.com/yshk-mxim/rdic.git
cd rdic
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
# Install all dependencies
pip install -r requirements.txt
```

**Expected installation time**: 5-10 minutes (depending on network speed)

**Key dependencies installed**:
- `mlx>=0.30.0` - Apple Silicon ML framework
- `mlx-lm>=0.30.0` - Language model utilities for MLX
- `safetensors>=0.7.0` - Secure tensor serialization
- `pytest>=7.4.0` - Testing framework

### Step 4: Verify Installation

```bash
# Run tests to verify installation
pytest tests/ -v

# Expected: 30 tests passed
```

---

## Quick Start

### Run Demo Script

The easiest way to see the system in action is to run the demo script:

#### Session 1: Create Agents

```bash
python demo_persistent_agents.py --session 1
```

**What happens**:
1. Loads Gemma 3 12B model (~7-10 seconds)
2. Creates 3 agents (tech_specialist, biz_analyst, coordinator)
3. Processes user query about an API bug
4. Generates responses from all 3 agents
5. Saves agents to `~/.agent_caches/`

**Output**:
```
==================================================================
  Session 1: Creating Agents
==================================================================

Loading model...
✅ Model loaded (8.2s)

──────────────────────────────────────────────────────────────
  Creating 3 Agents
──────────────────────────────────────────────────────────────

  ✅ tech_specialist: 142 tokens cached (system prompt)
  ✅ biz_analyst: 137 tokens cached (system prompt)
  ✅ coordinator: 151 tokens cached (system prompt)

...
```

#### Session 2: Resume with Cached Context

```bash
python demo_persistent_agents.py --session 2
```

**What happens**:
1. Loads Gemma 3 12B model (~7-10 seconds)
2. **Loads agents from disk** (<500ms per agent)
3. Continues conversation with follow-up query
4. **Generates responses faster** (40-60% speedup from cached context)

**Output**:
```
──────────────────────────────────────────────────────────────
  📊 Performance Comparison
──────────────────────────────────────────────────────────────

Session 1 (no cache):  ~5-8s generation time
Session 2 (with cache): 3.2s generation time
Speedup: ~50% faster! ⚡
```

---

## Basic Usage (Python API)

### Import the Library

```python
from src.agent_manager import PersistentAgentManager
```

### Initialize Manager

```python
manager = PersistentAgentManager(
    model_name="mlx-community/gemma-3-12b-it-4bit",
    max_agents=3,
    cache_dir="~/.agent_caches"  # Optional, default: ~/.agent_caches
)
```

**Parameters**:
- `model_name`: MLX model to load (any model from mlx-community)
- `max_agents`: Maximum agents in memory (default: 3, triggers LRU eviction)
- `cache_dir`: Directory for cache files (default: `~/.agent_caches`)

### Create an Agent

```python
agent = manager.create_agent(
    agent_id="my_assistant",
    agent_type="technical",
    system_prompt="You are a helpful technical assistant with expertise in Python and MLX."
)

print(f"Agent created with {agent.cache_tokens} tokens cached")
# Output: Agent created with 142 tokens cached
```

**Parameters**:
- `agent_id`: Unique identifier for the agent (used for save/load)
- `agent_type`: Agent type tag (for organization)
- `system_prompt`: System prompt to prefill into cache

### Generate Response

```python
response = manager.generate(
    agent_id="my_assistant",
    user_input="How do I optimize MLX inference on Mac?",
    max_tokens=200,
    temperature=0.7
)

print(response)
```

**Parameters**:
- `agent_id`: Agent to generate with
- `user_input`: User query
- `max_tokens`: Maximum tokens to generate (default: 256)
- `temperature`: Sampling temperature (default: 0.0)

**Returns**: Generated text as string

### Save Agent to Disk

```python
manager.save_agent("my_assistant")
# Agent cache saved to: ~/.agent_caches/my_assistant.safetensors
```

### Load Agent from Disk

```python
# Later, in a new session:
manager = PersistentAgentManager(model_name="mlx-community/gemma-3-12b-it-4bit")

agent = manager.load_agent("my_assistant")
print(f"Agent loaded with {agent.cache_tokens} tokens cached")

# Continue conversation with cached context
response = manager.generate("my_assistant", "What about memory optimization?", max_tokens=200)
```

### List Saved Agents

```python
saved_agents = manager.list_saved_agents()
print(f"Found {len(saved_agents)} saved agents: {saved_agents}")
# Output: Found 3 saved agents: ['my_assistant', 'tech_specialist', 'biz_analyst']
```

### Save All Agents

```python
manager.save_all()
# Saves all agents currently in memory
```

### Delete Agent

```python
manager.delete_agent("my_assistant")
# Removes from memory and deletes cache file from disk
```

---

## Advanced Usage

### Multi-Agent Workflow

```python
manager = PersistentAgentManager(max_agents=3)

# Create specialized agents
manager.create_agent("tech", "technical", "You are a technical specialist...")
manager.create_agent("creative", "creative", "You are a creative writer...")
manager.create_agent("analyst", "business", "You are a business analyst...")

# Get responses from all agents
user_query = "How should we approach this product launch?"

tech_view = manager.generate("tech", user_query, max_tokens=150)
creative_view = manager.generate("creative", user_query, max_tokens=150)
analyst_view = manager.generate("analyst", user_query, max_tokens=150)

# Save all for next session
manager.save_all()
```

### LRU Eviction in Action

```python
manager = PersistentAgentManager(max_agents=2)  # Only 2 agents in memory

manager.create_agent("agent_1", "type1", "Prompt 1")
manager.create_agent("agent_2", "type2", "Prompt 2")

# Creating agent_3 will evict agent_1 (LRU)
manager.create_agent("agent_3", "type3", "Prompt 3")

print(manager.agents.keys())
# Output: dict_keys(['agent_2', 'agent_3'])
# agent_1 was evicted (saved to disk automatically)

# Load agent_1 back (will evict agent_2)
manager.load_agent("agent_1")
print(manager.agents.keys())
# Output: dict_keys(['agent_3', 'agent_1'])
```

### Check Memory Usage

```python
memory_info = manager.get_memory_usage()

print(f"Model memory: {memory_info['model_memory_gb']:.2f} GB")
print(f"Cache memory: {memory_info['total_cache_mb']:.1f} MB")
print(f"Total: {memory_info['total_gb']:.2f} GB")

# Output:
# Model memory: 7.12 GB
# Cache memory: 384.2 MB
# Total: 7.50 GB
```

### Check Disk Usage

```python
disk_info = manager.persistence.get_cache_disk_usage()

print(f"Total cache disk usage: {disk_info['total_mb']:.1f} MB")
print(f"Number of agents: {disk_info['num_agents']}")

for agent_id, size_bytes in disk_info['per_agent']:
    size_mb = size_bytes / (1024 * 1024)
    print(f"  {agent_id}: {size_mb:.1f} MB")
```

---

## Configuration

### Custom Cache Directory

```python
# Use custom cache directory
manager = PersistentAgentManager(
    model_name="mlx-community/gemma-3-12b-it-4bit",
    cache_dir="/path/to/my/caches"
)
```

### Adjust Max Agents

```python
# Allow more agents in memory (requires more RAM)
manager = PersistentAgentManager(
    model_name="mlx-community/gemma-3-12b-it-4bit",
    max_agents=5  # Up to 5 agents in memory
)
```

**Memory budget**:
- Model: ~7GB (Gemma 3 12B 4-bit)
- Per agent cache: ~100-200MB (varies with conversation length)
- Total: 7GB + (max_agents × 150MB)

**Recommended**:
- 16GB RAM: `max_agents=2`
- 24GB RAM: `max_agents=3` (default)
- 32GB RAM: `max_agents=5`
- 64GB RAM: `max_agents=10`

### Use Different Models

```python
# Use a different MLX model
manager = PersistentAgentManager(
    model_name="mlx-community/Llama-3.2-11B-Vision-Instruct-4bit"
)
```

**Available MLX models**: https://huggingface.co/mlx-community

---

## Benchmarking

### Run Performance Benchmarks

```bash
# Full benchmark suite
python benchmarks/benchmark_suite.py

# Quick benchmark (shorter prompts, fewer tokens)
python benchmarks/benchmark_suite.py --quick

# Save results to JSON
python benchmarks/benchmark_suite.py --output results.json
```

**Output**:
```
==================================================================
  Persistent Multi-Agent Memory - Performance Benchmarks
==================================================================

📊 Benchmark 1: Model Loading Time
   Result: 8.12s

📊 Benchmark 2: Agent Creation (System Prompt Prefill)
   tech_1: 0.142s
   biz_1: 0.138s
   coord_1: 0.151s
   Average: 0.144s per agent

📊 Benchmark 3: Cache Save Time
   tech_1: 0.087s (87ms)
   biz_1: 0.091s (91ms)
   coord_1: 0.093s (93ms)
   Average: 0.090s per agent (90ms)

...
```

---

## Troubleshooting

### Issue: `mlx.metal.is_available()` returns False

**Solution**: Install Xcode Command Line Tools

```bash
xcode-select --install
```

Then reinstall MLX:

```bash
pip uninstall mlx mlx-lm
pip install mlx>=0.30.0 mlx-lm>=0.30.0
```

### Issue: Out of Memory Error

**Solution**: Reduce `max_agents` or use a smaller model

```python
# Option 1: Reduce max_agents
manager = PersistentAgentManager(max_agents=1)

# Option 2: Use smaller model (e.g., Gemma 3 7B instead of 12B)
manager = PersistentAgentManager(model_name="mlx-community/gemma-3-7b-it-4bit")
```

### Issue: Cache Load Failed (FileNotFoundError)

**Solution**: Agent cache file doesn't exist

```python
# Check which agents are saved
saved_agents = manager.list_saved_agents()
print(saved_agents)

# Create agent if it doesn't exist
if "my_agent" not in saved_agents:
    manager.create_agent("my_agent", "general", "System prompt here...")
```

### Issue: Slow Generation

**Expected behavior**:
- First generation (no cache): 8-10 seconds
- Session resume (with cache): 3-5 seconds

**If slower than expected**:
- Check CPU/GPU usage (should see high GPU utilization)
- Verify Metal is enabled: `mlx.metal.is_available()`
- Close other applications to free RAM
- Use smaller `max_tokens` for faster generation

### Issue: Disk Space Running Out

**Solution**: Clean old caches

```bash
# Check cache directory size
du -sh ~/.agent_caches

# List cache files
ls -lh ~/.agent_caches

# Delete old caches
rm ~/.agent_caches/old_agent_*.safetensors
```

Or programmatically:

```python
# Delete specific agent
manager.delete_agent("old_agent_id")

# Or manually clean cache directory
import shutil
from pathlib import Path

cache_dir = Path.home() / ".agent_caches"
for cache_file in cache_dir.glob("*.safetensors"):
    if cache_file.stem.startswith("old_"):
        cache_file.unlink()
        print(f"Deleted {cache_file.name}")
```

---

## Testing

### Run All Tests

```bash
# Run all tests with verbose output
pytest tests/ -v

# Expected: 30 tests passed
```

### Run Specific Test Module

```bash
# Test cache extractor only
pytest tests/test_cache_extractor.py -v

# Test agent manager only
pytest tests/test_agent_manager.py -v
```

### Run with Coverage

```bash
# Generate coverage report
pytest tests/ --cov=src --cov-report=html

# Open coverage report
open htmlcov/index.html
```

---

## Best Practices

### 1. Regular Saves

Save agents after important conversations:

```python
# After key interaction
response = manager.generate("my_agent", user_input, max_tokens=200)
manager.save_agent("my_agent")  # Save immediately
```

### 2. Descriptive Agent IDs

Use clear, descriptive agent IDs:

```python
# Good
manager.create_agent("python_expert_2024", "technical", "...")
manager.create_agent("creative_writer_fiction", "creative", "...")

# Bad
manager.create_agent("agent1", "type1", "...")
manager.create_agent("a", "b", "...")
```

### 3. Monitor Memory Usage

Periodically check memory usage, especially with many agents:

```python
memory_info = manager.get_memory_usage()
if memory_info['total_gb'] > 20:  # 20GB threshold
    print("Warning: High memory usage, consider saving and evicting agents")
    manager.save_all()
```

### 4. Clean Up Old Agents

Delete agents you no longer need:

```python
# List all saved agents
saved = manager.list_saved_agents()

# Delete old experiments
for agent_id in saved:
    if agent_id.startswith("experiment_"):
        manager.delete_agent(agent_id)
```

---

## Uninstallation

### Remove Cache Files

```bash
rm -rf ~/.agent_caches
```

### Uninstall Dependencies

```bash
# Deactivate virtual environment
deactivate

# Remove virtual environment
rm -rf venv
```

### Remove Repository

```bash
cd ..
rm -rf rdic
```

---

## Support

For issues and questions:

- **GitHub Issues**: https://github.com/yshk-mxim/rdic/issues
- **MLX Documentation**: https://ml-explore.github.io/mlx/
- **MLX Community**: https://github.com/ml-explore/mlx/discussions

---

**Last Updated**: 2026-01-23 | **Version**: 0.1.0
