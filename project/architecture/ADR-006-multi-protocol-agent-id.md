# ADR-006: Multi-Protocol Agent Identification Strategy

**Status**: Draft (In Progress - Day 0)
**Date**: 2026-01-25
**Deciders**: Sprint 4 Team, Technical Fellows
**Sprint**: Sprint 4 - Multi-Protocol API Adapter

---

## Context

Sprint 4 implements multiple API protocols (Anthropic, OpenAI, Direct) that external clients use to interact with the semantic caching system. A key question is: **How do we identify agents across requests to enable cache reuse?**

### Problem Statement

Different API protocols have different conventions for session management:

1. **Anthropic Messages API** (`/v1/messages`):
   - Each request is independent
   - Full conversation history sent every time
   - No server-side session concept
   - Cache reuse via prompt caching (`cache_control` blocks)

2. **OpenAI Chat Completions API** (`/v1/chat/completions`):
   - Standard API has no session concept
   - Full conversation history sent every time
   - Our extension: `X-Session-ID` header for explicit session tracking

3. **Direct Agent API** (`/v1/agents/{agent_id}`):
   - Explicit agent_id in URL path
   - Server maintains agent state
   - Multiple generations per agent

### Requirements

1. **Cache Reuse**: Agents with matching conversation prefixes should share cache blocks
2. **Protocol Independence**: Each protocol should feel natural to its users
3. **Performance**: Agent identification must be fast (< 1ms)
4. **Simplicity**: Avoid complex hash-based ID generation
5. **Compatibility**: Must work with Claude Code CLI (client-side sessions)

---

## Decision

### Simplified Approach: Use Existing Trie-Based Prefix Matching

**We will NOT implement complex content-based agent identification.** Instead, we leverage the existing `AgentCacheStore.find_prefix()` method which already implements trie-based token prefix matching.

### Agent Identification by Protocol

#### 1. Anthropic Messages API (`/v1/messages`)

**Strategy**: Each request is independent, rely on trie-based cache lookup.

**Implementation**:
```python
# In anthropic_adapter.py
async def handle_messages(request: MessagesRequest):
    # 1. Tokenize full conversation
    tokens = tokenize_conversation(request.system, request.messages)

    # 2. Find longest prefix match in cache
    cached_blocks = cache_store.find_prefix(tokens)

    # 3. Generate continuation from cache point
    result = batch_engine.generate(
        prompt_tokens=tokens,
        cache=cached_blocks,
        ...
    )

    # 4. Save updated cache (auto-generated agent_id)
    cache_store.save(agent_id=f"msg_{hash(tokens[:100])}", blocks=result.cache)
```

**Rationale**:
- Matches Anthropic's actual API behavior (no sessions)
- Claude Code CLI sends full history every request (client-side sessions)
- Trie matching handles prefix reuse automatically
- No server-side session management needed

**Trade-offs**:
- ✅ Simple implementation
- ✅ Natural for Claude Code CLI
- ✅ Leverages existing trie infrastructure
- ⚠️ No explicit session persistence (matches real Anthropic API)
- ⚠️ Cache eviction follows LRU (inactive caches removed)

#### 2. OpenAI Chat Completions API (`/v1/chat/completions`)

**Strategy**: Support optional `X-Session-ID` header for explicit session tracking.

**Implementation**:
```python
# In openai_adapter.py
async def handle_chat_completions(request: ChatCompletionsRequest):
    # 1. Check for explicit session ID
    session_id = request.headers.get("X-Session-ID")

    if session_id:
        # Explicit session: load agent cache by ID
        agent_id = f"oai_{session_id}"
        cached_blocks = cache_store.load(agent_id)
    else:
        # No session: use trie-based prefix matching (like Anthropic)
        tokens = tokenize_chat(request.messages)
        cached_blocks = cache_store.find_prefix(tokens)
        agent_id = f"oai_{hash(tokens[:100])}"

    # 2. Generate continuation
    result = batch_engine.generate(prompt_tokens=tokens, cache=cached_blocks, ...)

    # 3. Save updated cache
    cache_store.save(agent_id=agent_id, blocks=result.cache)
```

**Rationale**:
- Backwards compatible (no session ID = stateless like Anthropic)
- Opt-in explicit sessions for users who need it
- Natural extension to OpenAI API

**Trade-offs**:
- ✅ Flexibility (both stateless and stateful)
- ✅ Familiar to OpenAI API users
- ⚠️ Two code paths to maintain

#### 3. Direct Agent API (`/v1/agents/{agent_id}`)

**Strategy**: Explicit agent_id in URL path, full CRUD operations.

**Implementation**:
```python
# In direct_agent_adapter.py

@app.post("/v1/agents")
async def create_agent(request: CreateAgentRequest):
    """Create new agent with explicit ID."""
    agent_id = request.agent_id or generate_uuid()
    # Initialize empty cache entry
    cache_store.initialize(agent_id)
    return {"agent_id": agent_id, "created_at": now()}

@app.post("/v1/agents/{agent_id}/generate")
async def generate(agent_id: str, request: GenerateRequest):
    """Generate using agent's cache."""
    # 1. Load agent cache by ID
    cached_blocks = cache_store.load(agent_id)
    if cached_blocks is None:
        raise AgentNotFoundError(f"Agent {agent_id} not found")

    # 2. Generate continuation
    result = batch_engine.generate(prompt_tokens=request.tokens, cache=cached_blocks, ...)

    # 3. Save updated cache
    cache_store.save(agent_id=agent_id, blocks=result.cache)

    return {"text": result.text, "tokens": result.tokens}

@app.delete("/v1/agents/{agent_id}")
async def delete_agent(agent_id: str):
    """Permanently delete agent cache."""
    cache_store.delete(agent_id)
    return {"deleted": agent_id}
```

**Rationale**:
- Most explicit and predictable
- Full control over agent lifecycle
- Useful for debugging and testing

**Trade-offs**:
- ✅ Explicit and clear
- ✅ Full CRUD control
- ⚠️ Client must track agent IDs

---

## Alternatives Considered

### Alternative 1: Content-Based Hash Agent IDs

**Approach**: Generate agent_id from hash of conversation content.

```python
def get_agent_id(messages: list[Message]) -> str:
    content = "".join(msg.content for msg in messages)
    return f"agent_{hashlib.sha256(content.encode()).hexdigest()[:16]}"
```

**Rejected Because**:
- ❌ Complex implementation
- ❌ Hash collisions possible
- ❌ Doesn't leverage existing trie infrastructure
- ❌ Doesn't match how Claude Code CLI works (client-side sessions)
- ❌ Requires rehashing on every request

### Alternative 2: Server-Side Session Store

**Approach**: Maintain session_id → agent_id mapping in memory.

**Rejected Because**:
- ❌ Doesn't match Anthropic's stateless API design
- ❌ Requires session expiration logic
- ❌ Doesn't work with Claude Code CLI (no session concept)
- ❌ Adds complexity for minimal benefit

### Alternative 3: Database-Backed Session Management

**Approach**: Store sessions in SQLite/PostgreSQL.

**Rejected Because**:
- ❌ Massive overkill for Sprint 4
- ❌ Introduces dependency on database
- ❌ Performance overhead (disk I/O)
- ❌ Doesn't align with real Anthropic API behavior

---

## Implementation Details

### Cache Store Interface (Already Exists)

The existing `AgentCacheStore` (from `agent_cache_store.py`) already provides:

```python
class AgentCacheStore:
    def save(self, agent_id: str, blocks: AgentBlocks) -> None:
        """Save agent cache to hot tier."""
        # Adds to hot tier, triggers LRU eviction if needed
        ...

    def load(self, agent_id: str) -> AgentBlocks | None:
        """Load agent cache from hot or warm tier."""
        # Checks hot tier first, falls back to disk
        ...

    def find_prefix(self, tokens: list[int]) -> AgentBlocks | None:
        """Find longest prefix match in cache."""
        # Trie-based prefix matching (already implemented)
        ...
```

**No changes needed** - existing methods are sufficient!

### Agent ID Generation Utilities

```python
# In src/semantic/adapters/inbound/agent_id_utils.py (NEW)

import hashlib
from typing import Any

def generate_anthropic_agent_id(tokens: list[int]) -> str:
    """Generate agent ID for Anthropic API requests.

    Uses first 100 tokens for stability (prefix matching).
    """
    prefix = tokens[:100]
    hash_val = hashlib.sha256(str(prefix).encode()).hexdigest()[:16]
    return f"msg_{hash_val}"

def generate_openai_agent_id(session_id: str | None, tokens: list[int]) -> str:
    """Generate agent ID for OpenAI API requests.

    Args:
        session_id: Optional explicit session ID from X-Session-ID header
        tokens: Tokenized conversation

    Returns:
        Agent ID (explicit session or hash-based)
    """
    if session_id:
        return f"oai_{session_id}"

    prefix = tokens[:100]
    hash_val = hashlib.sha256(str(prefix).encode()).hexdigest()[:16]
    return f"oai_{hash_val}"

def generate_direct_agent_id() -> str:
    """Generate UUID for direct agent API."""
    import uuid
    return f"agent_{uuid.uuid4().hex[:16]}"
```

---

## Consequences

### Positive

1. **Simplicity**: Leverages existing trie-based cache infrastructure
2. **Performance**: No complex hashing or database lookups
3. **Compatibility**: Works naturally with Claude Code CLI (client-side sessions)
4. **Flexibility**: Multiple protocols, each with appropriate semantics
5. **Maintainability**: Minimal new code, reuses tested components

### Negative

1. **No persistent sessions** for Anthropic API (matches real API)
2. **Cache eviction** follows LRU (inactive agents removed)
3. **Two code paths** for OpenAI API (with/without session ID)

### Neutral

1. **Agent ID visibility**: IDs are opaque hashes (not human-readable)
2. **Cache reuse**: Automatic via trie matching (no explicit control)

---

## Validation

### Tests (Day 2-6)

1. **Unit tests**:
   - Agent ID generation utilities
   - Trie-based prefix matching (already tested)

2. **Integration tests**:
   - Anthropic API: 3-turn conversation, verify cache reuse
   - OpenAI API: With/without X-Session-ID header
   - Direct API: CRUD operations, agent lifecycle

### Experiments (Day 4, Day 6)

1. **EXP-009**: Validate SSE format against real Anthropic API
2. **EXP-010**: Test Claude Code CLI end-to-end (3-turn conversation)

### Success Criteria

- ✅ Claude Code CLI works with `ANTHROPIC_BASE_URL` override
- ✅ Cache reuse verified across 3-turn conversation
- ✅ OpenAI session persistence works
- ✅ Direct API CRUD operations functional
- ✅ Performance: Agent lookup < 1ms

---

## Related ADRs

- **ADR-002**: Block-Level KV Cache Management (block allocation strategy)
- **ADR-003**: Three-Tier Cache Architecture (hot/warm/cold tiers)
- **ADR-005**: Model Hot-Swap Support (cache reconfiguration)

---

## Open Questions

1. **Session expiration**: Should OpenAI sessions expire after inactivity?
   - **Answer**: Defer to Sprint 5 - rely on LRU eviction for now

2. **Agent ID collision**: What if two conversations hash to same ID?
   - **Answer**: Unlikely (SHA-256 first 16 chars = 64 bits), accept risk

3. **Cache metrics**: How to track cache hit rates per protocol?
   - **Answer**: Defer to Sprint 5 (observability work)

---

## Status

- **Draft**: Day 0 (2026-01-25) - Skeleton created
- **In Review**: TBD (after Day 2 implementation)
- **Approved**: TBD (after Fellows review on Day 8)

---

**Document Owner**: Sprint 4 Team
**Reviewers**: Technical Fellows (SE, ML, QE, HW)
**Last Updated**: 2026-01-25
