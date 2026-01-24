# Sprint 2: Agent Manager & Integration (Week 2)

**Duration**: 5 days
**Goal**: Implement multi-agent manager with persistent memory
**Status**: Pending Sprint 1 completion

---

## Objectives

- [ ] Implement PersistentAgentManager class
- [ ] Support 3 concurrent agents with isolated contexts
- [ ] Implement LRU eviction when max agents exceeded
- [ ] Memory usage monitoring and reporting
- [ ] End-to-end integration testing

---

## Daily Breakdown

### Monday: Agent Manager - Core Structure

**Morning (3h)**:
- [ ] Create `src/agent_manager.py`
- [ ] Define AgentContext dataclass
  ```python
  @dataclass
  class AgentContext:
      agent_id: str
      agent_type: str  # technical, business, coordinator
      system_prompt: str
      kv_cache: Optional[List[Tuple[mx.array, mx.array]]]
      cache_size: int
      last_access: datetime
      conversation_history: List[dict]
  ```

**Afternoon (2h)**:
- [ ] Implement PersistentAgentManager skeleton
  ```python
  class PersistentAgentManager:
      def __init__(self, model_name, max_agents=3):
          self.model, self.tokenizer = MLXModelLoader.load_model(model_name)
          self.cache_extractor = MLXCacheExtractor(self.model, self.tokenizer)
          self.persistence = CachePersistence()
          self.agents = {}  # {agent_id: AgentContext}
          self.max_agents = max_agents
  ```

**Deliverable**: `src/agent_manager.py` (skeleton)

---

### Tuesday: Agent Creation & Loading

**Morning (3h)**:
- [ ] Implement create_agent()
  ```python
  def create_agent(self, agent_id: str, agent_type: str, system_prompt: str):
      """Create new agent with empty cache"""
      # 1. Create AgentContext
      # 2. Add to self.agents
      # 3. Check max_agents limit, evict if needed
  ```

**Afternoon (2h)**:
- [ ] Implement load_agent()
  ```python
  def load_agent(self, agent_id: str):
      """Load agent from disk if exists, else create new"""
      # 1. Check if agent in memory
      # 2. If not, try load from disk
      # 3. If not on disk, create new
      # 4. Update last_access
  ```

**Deliverable**: Agent creation and loading working

---

### Wednesday: Agent Generation & Cache Management

**Morning (3h)**:
- [ ] Implement generate()
  ```python
  def generate(self, agent_id: str, user_input: str, max_tokens=300):
      """Generate response using agent's cached context"""
      # 1. Get agent from memory
      # 2. Build context: system_prompt + conversation_history + user_input
      # 3. Generate with existing cache
      # 4. Update agent's cache and history
      # 5. Update last_access
      # 6. Return response
  ```

**Afternoon (2h)**:
- [ ] Implement cache update logic
  - Merge new cache with existing cache
  - Track cache growth
  - Validate cache integrity

**Deliverable**: Agent generation with cache accumulation

---

### Thursday: LRU Eviction & Memory Management

**Morning (3h)**:
- [ ] Implement LRU eviction policy
  ```python
  def evict_lru(self):
      """Evict least recently used agent when max_agents exceeded"""
      # 1. Find agent with oldest last_access
      # 2. Save agent to disk
      # 3. Remove from memory
      # 4. Log eviction
  ```

**Afternoon (2h)**:
- [ ] Implement memory monitoring
  ```python
  def get_memory_usage(self) -> dict:
      """Report memory usage per agent"""
      # Return:
      # {
      #   'model': 7.2GB,
      #   'agents': {
      #     'tech': {'cache_tokens': 450, 'cache_mb': 120},
      #     'biz': {'cache_tokens': 380, 'cache_mb': 100}
      #   },
      #   'total': 7.5GB
      # }
  ```

**Deliverable**: LRU eviction + memory monitoring

---

### Friday: Integration Testing & Polishing

**Morning (3h)**:
- [ ] Write comprehensive integration tests
  ```python
  def test_multi_agent_workflow():
      # 1. Create 3 agents
      # 2. Generate responses from each
      # 3. Save all agents
      # 4. Clear memory
      # 5. Load agents back
      # 6. Generate again (verify cache reuse)
  ```

**Afternoon (2h)**:
- [ ] Write unit tests for agent_manager
  - Test agent creation
  - Test LRU eviction
  - Test memory monitoring
  - Test save/load

**Evening (1h)**:
- [ ] Code review and documentation
  - Add docstrings to all methods
  - Add usage examples
  - Update README

**Deliverable**: `tests/test_agent_manager.py`, all tests passing

---

## Success Criteria

- ✅ Can create 3 agents with distinct roles
- ✅ Each agent maintains isolated KV cache
- ✅ Agents can generate responses using cached context
- ✅ Cache accumulates across multiple turns
- ✅ LRU eviction works when 4th agent created
- ✅ Memory monitoring reports accurate usage
- ✅ All unit and integration tests passing

---

## Technical Details

### Agent Lifecycle

```
Create Agent
    ↓
┌─────────────────┐
│ In Memory       │ ← generate() updates cache
│ (active)        │
└─────────────────┘
    ↓ (save_agent or evict_lru)
┌─────────────────┐
│ On Disk         │
│ (persistent)    │
└─────────────────┘
    ↓ (load_agent)
┌─────────────────┐
│ In Memory       │
│ (active again)  │
└─────────────────┘
```

### Cache Accumulation Example

```python
# Turn 1
user: "Analyze API bug"
system_prompt: "You are technical expert..." (100 tokens)
user_input: "Analyze API bug" (10 tokens)
cache_size: 110 tokens

# Turn 2
user: "What's the root cause?"
# Cache already contains: system_prompt + turn1_history (110 + 50 = 160 tokens)
new_input: "What's the root cause?" (15 tokens)
cache_size: 175 tokens (accumulated)

# Result: No re-prefill of system_prompt or turn 1!
```

### LRU Eviction Logic

```python
# Scenario: 3 agents in memory (max_agents=3)
agents = {
    'tech': last_access=10:00,
    'biz': last_access=10:05,
    'coord': last_access=10:10
}

# User creates 4th agent at 10:15
# LRU eviction triggered:
# - 'tech' has oldest last_access (10:00)
# - Save 'tech' to disk
# - Remove 'tech' from memory
# - Add new agent to memory

# User asks for 'tech' again at 10:20
# - load_agent('tech') from disk
# - Evict 'biz' (now oldest)
```

---

## Code Structure

### File Organization

```
src/
├── mlx_utils.py (existing)
├── mlx_cache_extractor.py (Sprint 1)
├── cache_persistence.py (Sprint 1)
└── agent_manager.py (THIS SPRINT)
    ├── AgentContext (dataclass)
    └── PersistentAgentManager (class)
        ├── __init__()
        ├── create_agent()
        ├── load_agent()
        ├── generate()
        ├── save_agent()
        ├── save_all()
        ├── evict_lru()
        └── get_memory_usage()
```

### Dependencies

```
agent_manager.py depends on:
- mlx_utils.MLXModelLoader
- mlx_cache_extractor.MLXCacheExtractor
- cache_persistence.CachePersistence
```

---

## Example Usage

```python
# Initialize manager
manager = PersistentAgentManager(
    model_name="mlx-community/gemma-3-12b-it-4bit",
    max_agents=3
)

# Create agents
manager.create_agent(
    "tech_specialist",
    "technical",
    "You are a technical expert..."
)

manager.create_agent(
    "biz_analyst",
    "business",
    "You are a business analyst..."
)

# Generate responses
tech_response = manager.generate(
    "tech_specialist",
    "Analyze this API bug: ..."
)

biz_response = manager.generate(
    "biz_analyst",
    "What's the business impact?"
)

# Save all agents
manager.save_all()

# Later: Load and continue
manager.load_agent("tech_specialist")
follow_up = manager.generate(
    "tech_specialist",
    "More details on that bug?"
)  # Uses cached context from previous session!
```

---

## Risks & Mitigation

**Risk**: Cache merging is complex
- **Mitigation**: Keep it simple - append new cache to existing
- **Fallback**: Replace cache entirely (less efficient but works)

**Risk**: Memory usage exceeds available RAM
- **Mitigation**: Monitor usage, implement aggressive eviction
- **Fallback**: Reduce max_agents to 2

**Risk**: LRU eviction too frequent
- **Mitigation**: Set reasonable max_agents (3 should fit in 24GB)
- **Fallback**: Increase max_agents if memory allows

---

## Deliverables

- [ ] `src/agent_manager.py` - Multi-agent orchestration (~400 lines)
- [ ] `tests/test_agent_manager.py` - Unit tests (~300 lines)
- [ ] Integration tests demonstrating full workflow

---

## Performance Targets

- Agent creation: <2s (includes model load)
- Agent loading from disk: <500ms
- Generate with cache: 3-5s (no re-prefill)
- Generate without cache: 8-10s (includes prefill)
- **Target speedup**: 40-60% faster with cache

---

## Next Sprint

**Sprint 3**: Documentation & Demonstration (Week 3)
- User-facing demo script
- Performance benchmarks
- Blog post / write-up
- Comparison vs existing tools

---

**Created**: January 23, 2026
**Status**: Pending Sprint 1
**Blockers**: Requires MLXCacheExtractor and CachePersistence from Sprint 1
**Estimated Effort**: 25-30 hours over 5 days
