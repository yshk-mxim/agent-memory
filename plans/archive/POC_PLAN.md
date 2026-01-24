# POC Plan: Persistent Multi-Agent Memory for Mac

**Created**: January 23, 2026
**Status**: Active Development
**Goal**: Demonstrate persistent multi-agent system with KV cache memory on Mac
**Timeline**: 2-3 weeks
**Positioning**: Fill gap that LM Studio, Ollama, and llama.cpp don't provide

---

## Executive Summary

**What We're Building**: A local multi-agent system on Mac that maintains **persistent KV cache memory** across sessions, enabling agents to efficiently "remember" previous conversations without reprocessing.

**Why This Matters**:
- ❌ **LM Studio**: Saves conversation text only, NOT KV cache
- ❌ **Ollama**: No session persistence at all
- ⚠️ **llama.cpp**: Has KV cache persistence API but NOT exposed in user-facing tools
- ✅ **This POC**: User-facing multi-agent tool with persistent KV cache + UMA optimization

**Deliverable**: Working demonstration showcasing your technical capabilities to potential users/clients.

---

## Research Foundation

### Novelty Analysis Completed

**Key Documents**:
1. `/Users/dev_user/semantic/novelty/EDGE_KV_CACHE_NOVELTY_REVIEW.md`
   - Comprehensive survey of academic work (KVCOMM, KVFlow, Continuum, etc.)
   - Validated: Mac is compute-bound for prefill (avoiding re-prefill is critical)
   - Identified gap: UMA-aware multi-agent cache management

2. `/Users/dev_user/semantic/novelty/EXISTING_TOOLS_COMPARISON.md`
   - Survey of LM Studio, Ollama, llama.cpp capabilities
   - **Critical finding**: NONE provide persistent agent KV cache
   - llama.cpp has API but not in WebUI (GitHub issue #17107)

### What's Novel

**NOT Novel** (already exists):
- ❌ Per-agent cache isolation (vLLM, SafeKV)
- ❌ LRU eviction policies (widely used)
- ❌ Application-layer agent memory (Mem0, Zep, A-MEM)

**Novel/Incremental**:
- ✅ User-facing tool with persistent agent KV cache
- ✅ Multi-agent system with cross-session memory
- ✅ UMA-optimized for Mac (exploit zero-copy memory)
- ✅ Fills gap in popular local LLM tools

---

## POC Scope (Limited for Demonstration)

### Core Capabilities

**3 Persistent Agents**:
1. **Technical Specialist** - Technical analysis and debugging
2. **Business Analyst** - Business impact and strategy
3. **Coordinator** - Synthesis and coordination

**Persistent Memory**:
- Each agent maintains isolated KV cache
- Cache persisted to disk on exit
- Cache loaded on startup (avoids re-prefill)
- Cross-session continuity

**UMA Optimization** (Mac-specific):
- Exploit unified memory architecture
- Zero-copy cache access (CPU + GPU)
- Memory wiring for active agents (macOS 15+)

### What We're NOT Building

❌ Production multi-tenant server
❌ Advanced eviction policies (just simple LRU)
❌ Agent orchestration framework
❌ Web UI or complex interface
❌ Publication-quality research

**Focus**: Clean, working demonstration in 2-3 weeks.

---

## Technical Architecture

### Component Overview

```
┌─────────────────────────────────────────────┐
│  User Interface (CLI demo script)          │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│  PersistentAgentManager                     │
│  - Create/load/save agents                  │
│  - LRU eviction (max 3 agents)              │
│  - Session management                       │
└─────────────────────────────────────────────┘
        ↓                          ↓
┌──────────────────┐     ┌─────────────────────┐
│ MLXCacheExtractor│     │ CachePersistence    │
│ - Expose KV cache│     │ - Save to disk      │
│ - Cache metadata │     │ - Load from disk    │
└──────────────────┘     └─────────────────────┘
        ↓                          ↓
┌─────────────────────────────────────────────┐
│  MLX Framework (Apple Silicon)              │
│  - Gemma 3 12B (4-bit quantization)         │
│  - Unified Memory Architecture              │
└─────────────────────────────────────────────┘
```

### Key Components

#### 1. **MLXCacheExtractor** (`src/mlx_cache_extractor.py`)

**Purpose**: Expose KV cache from mlx_lm.generate()

**Current Gap**: mlx_lm.generate() hides KV cache internally

**Implementation**:
```python
class MLXCacheExtractor:
    """Wrapper around mlx_lm to expose internal KV cache"""

    def generate_with_cache(self, prompt, existing_cache=None):
        """
        Generate text and return both output and KV cache

        Returns: (output_text: str, kv_cache: List[Tuple[mx.array, mx.array]])
        """

    def get_cache_size(self, cache) -> int:
        """Get token count in cache"""

    def merge_caches(self, cache1, cache2):
        """Combine two KV caches (for multi-turn)"""
```

**MLX Details**:
- Cache format: List of (key, value) tuples, one per layer
- Key/value shapes: [batch_size, num_heads, seq_len, head_dim]
- Sequence length dimension (index 2) indicates token count

#### 2. **CachePersistence** (`src/cache_persistence.py`)

**Purpose**: Save/load KV cache to disk

**Storage Format**: safetensors (efficient, secure)

**Implementation**:
```python
class CachePersistence:
    """Persist agent KV caches to disk"""

    def __init__(self, cache_dir="~/.agent_caches"):
        self.cache_dir = Path(cache_dir).expanduser()

    def save_agent_cache(self, agent_id: str, cache, metadata: dict):
        """
        Save cache to: ~/.agent_caches/tech_specialist_001.safetensors

        metadata:
          - agent_id
          - agent_type (technical, business, coordinator)
          - cache_size (token count)
          - timestamp
          - model_name
          - system_prompt_hash
        """

    def load_agent_cache(self, agent_id: str):
        """Load cache from disk, return (cache, metadata)"""

    def list_cached_agents(self) -> List[str]:
        """List all cached agent IDs"""

    def delete_agent_cache(self, agent_id: str):
        """Remove cache file"""
```

#### 3. **PersistentAgentManager** (`src/agent_manager.py`)

**Purpose**: Manage multiple agents with persistent state

**Implementation**:
```python
class AgentContext:
    """Single agent's state"""
    agent_id: str
    agent_type: str
    system_prompt: str
    kv_cache: Optional[List[Tuple[mx.array, mx.array]]]
    cache_size: int
    last_access: datetime
    conversation_history: List[dict]

class PersistentAgentManager:
    """Manage multiple persistent agents"""

    def __init__(self, model_name="mlx-community/gemma-3-12b-it-4bit", max_agents=3):
        self.model, self.tokenizer = load(model_name)
        self.cache_extractor = MLXCacheExtractor(self.model, self.tokenizer)
        self.persistence = CachePersistence()
        self.agents = {}  # {agent_id: AgentContext}
        self.max_agents = max_agents

    def create_agent(self, agent_id: str, agent_type: str, system_prompt: str):
        """Create new agent with empty cache"""

    def load_agent(self, agent_id: str):
        """Load agent from disk if exists, else create new"""

    def generate(self, agent_id: str, user_input: str, max_tokens=300):
        """
        Generate response using agent's cached context
        Updates agent's KV cache
        """

    def save_agent(self, agent_id: str):
        """Persist single agent to disk"""

    def save_all(self):
        """Save all agents to disk"""

    def evict_lru(self):
        """Evict least recently used agent when max_agents exceeded"""

    def get_memory_usage(self) -> dict:
        """Report memory usage per agent"""
```

#### 4. **Demo Script** (`demo_persistent_agents.py`)

**Purpose**: User-facing demonstration

**Implementation**:
```python
#!/usr/bin/env python3
"""
Persistent Multi-Agent Memory Demo

Demonstrates:
1. Creating agents with persistent KV cache
2. Cross-session memory (agents remember past conversations)
3. Efficient cache reuse (no re-prefill needed)
"""

def demo_session_1():
    """First session: Create agents and have conversation"""
    print("=== Session 1: Creating Agents ===\n")

    manager = PersistentAgentManager(max_agents=3)

    # Create 3 agents
    manager.create_agent(
        "tech_specialist",
        "technical",
        "You are a technical expert specializing in API debugging and system architecture."
    )
    manager.create_agent(
        "biz_analyst",
        "business",
        "You are a business analyst focused on ROI, market impact, and strategy."
    )
    manager.create_agent(
        "coordinator",
        "synthesis",
        "You are a coordinator who synthesizes technical and business perspectives."
    )

    # User query
    user_input = "We have a critical API bug affecting payment processing. Analyze technical issues and business impact."

    print(f"User: {user_input}\n")

    # Technical analysis
    tech_response = manager.generate("tech_specialist", user_input)
    print(f"Technical Specialist:\n{tech_response}\n")

    # Business analysis
    biz_response = manager.generate("biz_analyst", user_input)
    print(f"Business Analyst:\n{biz_response}\n")

    # Coordination
    coord_input = f"Technical analysis: {tech_response}\n\nBusiness analysis: {biz_response}\n\nProvide coordinated action plan."
    coord_response = manager.generate("coordinator", coord_input)
    print(f"Coordinator:\n{coord_response}\n")

    # Save all agents
    manager.save_all()
    print("✅ All agents saved with KV cache to disk\n")

    return manager.get_memory_usage()

def demo_session_2():
    """Second session: Load agents and continue conversation"""
    print("\n=== Session 2: Loading Agents (simulating new session) ===\n")

    manager = PersistentAgentManager(max_agents=3)

    # Load existing agents (with cached KV memory)
    manager.load_agent("tech_specialist")
    manager.load_agent("biz_analyst")
    manager.load_agent("coordinator")

    print("✅ Agents loaded from disk with KV cache\n")

    # Continue conversation
    user_input = "What are the detailed steps to fix that API bug from our previous discussion?"
    print(f"User: {user_input}\n")

    # Technical specialist already has context from Session 1!
    tech_response = manager.generate("tech_specialist", user_input)
    print(f"Technical Specialist (using cached context):\n{tech_response}\n")

    # Show memory usage
    print(f"\n📊 Memory Usage:\n{manager.get_memory_usage()}\n")

if __name__ == "__main__":
    # Run demo
    session1_memory = demo_session_1()
    print(f"\nSession 1 final memory: {session1_memory}")

    input("\n[Press Enter to simulate Session 2 - user returning later]")

    demo_session_2()

    print("\n✅ Demo complete!")
```

---

## Code Refactoring Plan

### Current State Analysis

**Existing Code** (`/Users/dev_user/semantic/src/`):
1. `semantic_isolation_mlx.py` (439 lines) - MLX-based semantic isolation tester
2. `semantic_isolation.py` (825 lines) - HuggingFace-based semantic isolation
3. Various test scripts in root directory

**Decision**:
- ✅ Keep: Core MLX utilities that are reusable
- ❌ Archive: Semantic isolation specific code (no longer needed)
- ♻️ Refactor: Extract reusable MLX components

### Refactoring Steps

#### Step 1: Archive Old Code
```
archive/semantic_isolation/
├── README.md (explaining what was archived)
├── semantic_isolation_mlx.py (original implementation)
├── semantic_isolation.py (HuggingFace version)
└── validation_001_isolation_test_mlx.json (test results)
```

#### Step 2: Extract Reusable Components

From `semantic_isolation_mlx.py`, extract:
- Model loading utilities → `src/mlx_utils.py`
- Basic generation wrapper → Foundation for `MLXCacheExtractor`

#### Step 3: New Code Structure
```
src/
├── mlx_cache_extractor.py      # NEW - Component 1
├── cache_persistence.py         # NEW - Component 2
├── agent_manager.py             # NEW - Component 3
└── mlx_utils.py                 # REFACTORED - Extracted utilities

tests/
├── test_cache_extractor.py     # NEW
├── test_cache_persistence.py   # NEW
└── test_agent_manager.py       # NEW

demo_persistent_agents.py        # NEW - User-facing demo

archive/
└── semantic_isolation/
    ├── README.md
    ├── semantic_isolation_mlx.py
    ├── semantic_isolation.py
    └── tests/
```

---

## Implementation Timeline

### Week 1: Core Infrastructure

**Monday-Tuesday** (2 days):
- [ ] Archive old semantic isolation code
- [ ] Extract reusable MLX utilities
- [ ] Implement `MLXCacheExtractor` (expose KV cache)
- [ ] Write unit tests for cache extraction

**Wednesday-Thursday** (2 days):
- [ ] Implement `CachePersistence` (save/load to disk)
- [ ] Test safetensors serialization
- [ ] Validate cache reload works correctly
- [ ] Write unit tests

**Friday** (1 day):
- [ ] Begin `PersistentAgentManager` implementation
- [ ] Basic agent creation/loading
- [ ] Test with 1 agent

### Week 2: Agent Management & Integration

**Monday-Tuesday** (2 days):
- [ ] Complete `PersistentAgentManager`
- [ ] Multi-agent support (3 agents)
- [ ] LRU eviction policy
- [ ] Memory usage reporting

**Wednesday-Thursday** (2 days):
- [ ] Integration testing
- [ ] End-to-end flow testing
- [ ] Performance benchmarking
- [ ] Bug fixes

**Friday** (1 day):
- [ ] Create demo script
- [ ] Polish user experience
- [ ] Add helpful output messages

### Week 3: Documentation & Demonstration

**Monday-Tuesday** (2 days):
- [ ] Write comprehensive README
- [ ] Create comparison table (vs LM Studio/Ollama/llama.cpp)
- [ ] Document architecture
- [ ] Add code comments

**Wednesday** (1 day):
- [ ] Performance metrics collection
- [ ] Create benchmarks showing cache reuse benefits

**Thursday-Friday** (2 days):
- [ ] Blog post / write-up
- [ ] Demo video/screenshots
- [ ] Final polish
- [ ] Prepare for showcase

---

## Success Criteria

### Functional Requirements
- ✅ 3 agents can be created with distinct roles
- ✅ Agents maintain isolated KV caches
- ✅ Caches persist to disk on save
- ✅ Caches load correctly on startup
- ✅ Agent responses use cached context (no re-prefill)
- ✅ LRU eviction works when max agents exceeded

### Performance Targets
- ✅ Session 2 startup: <1s (vs 5-10s re-prefill)
- ✅ Cache load time: <500ms per agent
- ✅ Memory usage: <15GB total (model + 3 agent caches)
- ✅ Cache reuse: Demonstrable speed improvement

### Demonstration Goals
- ✅ Clear showcase of cross-session memory
- ✅ Comparison showing gap filled vs existing tools
- ✅ Clean, professional code quality
- ✅ Comprehensive documentation

---

## Deliverables

### Code
1. `src/mlx_cache_extractor.py` - KV cache extraction
2. `src/cache_persistence.py` - Disk persistence
3. `src/agent_manager.py` - Multi-agent orchestration
4. `demo_persistent_agents.py` - User-facing demo
5. Comprehensive test suite

### Documentation
1. `README.md` - Project overview and setup
2. `ARCHITECTURE.md` - Technical design
3. `COMPARISON.md` - vs LM Studio/Ollama/llama.cpp
4. Code documentation (docstrings)

### Demonstration Materials
1. Demo script with clear output
2. Performance benchmarks
3. Blog post / technical write-up
4. Comparison table showing gap filled

---

## Risk Mitigation

### Risk 1: MLX KV Cache Extraction Difficulty
**Likelihood**: Medium
**Impact**: High
**Mitigation**:
- Start with cache extraction in Week 1 Day 1
- If blocked, use mlx_lm source code inspection
- Fallback: Contact MLX community for guidance

### Risk 2: Cache Serialization Issues
**Likelihood**: Low
**Impact**: Medium
**Mitigation**:
- Use proven safetensors format
- Test serialization early (Week 1)
- Fallback: Use pickle if safetensors problematic

### Risk 3: Memory Constraints
**Likelihood**: Low
**Impact**: Low
**Mitigation**:
- Target Mac with 24GB RAM (ample headroom)
- Monitor memory usage throughout
- Implement aggressive LRU if needed

### Risk 4: Demo Not Compelling
**Likelihood**: Low
**Impact**: Medium
**Mitigation**:
- Focus on clear before/after comparison
- Show time savings quantitatively
- Emphasize gap vs popular tools

---

## Post-POC Opportunities

### If Successful
1. **Open source release** - GitHub repo with demo
2. **Technical blog post** - Medium/dev.to article
3. **Integration** - Package for easy installation
4. **Extensions**:
   - Web UI for agent management
   - More sophisticated eviction policies
   - Integration with agent frameworks (AutoGen, CrewAI)

### Future Enhancements (Out of Scope for POC)
- Multi-user support
- Advanced orchestration
- Cloud backup/sync
- Mobile companion app
- Commercial licensing

---

## Notes

**Focus**: This is a **capability demonstration**, not a research publication.

**Target Audience**: Potential users/clients who want to see your technical skills.

**Positioning**: Fills a real gap that popular tools (LM Studio, Ollama, llama.cpp) don't provide.

**Timeline**: 2-3 weeks is realistic for a clean, working demonstration.

---

**Created**: January 23, 2026
**Status**: Ready for implementation
**Next**: Convert to detailed sprint breakdown
