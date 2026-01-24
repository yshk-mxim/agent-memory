# Continuous Batching + KV Cache Quantization

## Goal
Enable **continuous batching** for multi-agent inference using mlx_lm's built-in `BatchGenerator`, integrated with the Anthropic-compatible API server for Claude Code CLI compatibility.

Two features:
1. **Continuous Batching**: Use mlx_lm's `BatchGenerator` + `BatchKVCache`/`BatchRotatingKVCache` for concurrent multi-agent generation
2. **KV Cache Quantization**: Use MLX's existing `QuantizedKVCache` to halve memory/disk usage

## Key Discovery
`mlx_lm` already provides everything needed for continuous batching:
- `BatchGenerator` class (generate.py:920) - full continuous batching with insert/remove/next loop
- `BatchKVCache` (cache.py:803) - left-padded batched KV cache with filter/extend/extract
- `BatchRotatingKVCache` (cache.py:992) - rotating variant for Gemma 3's sliding window layers
- `_make_cache(model, left_padding)` - creates batched caches from model spec
- `Batch.extract_cache()` - extract per-sequence KVCache for persistence
- Model's `__call__` natively supports `[B, L]` input shape

## Current System
- Model: Gemma 3 12B 4-bit (48 layers, 16 attn heads, 8 KV heads, head_dim=256)
- Cache: `KVCache` (8 global layers, every 6th) + `RotatingKVCache(max_size=512)` (40 sliding window layers)
- Persistence: `save_prompt_cache`/`load_prompt_cache` → safetensors
- Generation: `mlx_lm.stream_generate()` with `prompt_cache` (single-sequence only)
- API Server: Anthropic Messages API compatible, SSE streaming (uses sync PersistentAgentManager)
- Concurrent Manager: Async queue but processes requests ONE at a time

## Architecture

```
    Claude Code CLI (ANTHROPIC_BASE_URL=http://localhost:8000)
              │
              ▼
    ┌─────────────────────────────────────┐
    │  API Server (Anthropic Messages API) │
    │  POST /v1/messages (stream/non-stream)│
    └──────────────────┬──────────────────┘
                       │ async
                       ▼
    ┌─────────────────────────────────────┐
    │    ConcurrentAgentManager            │
    │  (async queue → BatchGenerator)      │
    └──────────────────┬──────────────────┘
                       │ batch requests
                       ▼
    ┌─────────────────────────────────────┐
    │  BatchedGenerationEngine             │
    │  (wraps mlx_lm BatchGenerator)       │
    │  - insert(agent_id, prompt, cache)   │
    │  - next() → yields tokens per agent  │
    │  - extract_cache(agent_id) → persist │
    └──────────────────┬──────────────────┘
                       │
              ┌────────┴────────┐
              ▼                 ▼
    ┌──────────────┐   ┌─────────────────┐
    │ BatchKVCache  │   │BatchRotatingKV  │
    │ (8 global)    │   │(40 sliding win) │
    └──────────────┘   └─────────────────┘
              │
              ▼
    ┌─────────────────────────────────────┐
    │       CachePersistence               │
    │  extract → save_prompt_cache         │
    │  load_prompt_cache → merge into batch│
    └─────────────────────────────────────┘
```

---

## Phase 0: Documentation

### Step 0.1: Save Plan to Project

- Copy this plan to `plans/continuous_batching.md` in the project
- Create `novelty/continuous_batching.md` documenting the novel features:
  - Continuous batching with persistent per-agent KV caches
  - Per-agent sequential / cross-agent parallel semantics
  - Integration with Anthropic Messages API for Claude Code CLI compatibility
  - Leveraging mlx_lm's BatchGenerator (not reimplementing)

---

## Phase 1: Wire API Server to ConcurrentAgentManager

Currently the API server uses synchronous `PersistentAgentManager` directly. This blocks the event loop and prevents concurrent request handling.

### Step 1.1: Make API Server Use ConcurrentAgentManager

**File**: `src/api_server.py` (modify)

Changes:
- Replace `PersistentAgentManager` with `ConcurrentAgentManager`
- Use `await manager.generate(agent_id, prompt, max_tokens)` instead of sync call
- Add startup/shutdown lifecycle to start/stop the concurrent manager worker
- Handle streaming: generate full response async, then stream SSE chunks

### Step 1.2: Per-Agent Sequential, Cross-Agent Parallel

**File**: `src/concurrent_manager.py` (modify)

**Concurrency semantics**:
- **Cross-agent**: Parallel. Agent A and Agent B can be in the same batch, generating simultaneously.
- **Per-agent**: Sequential. If Agent A has two requests, the second WAITS for the first to complete and inherits the updated KV cache.

**Why**: Each agent's KV cache represents its conversation state. Running two requests for the same agent in parallel would corrupt the cache (both would start from the same state, then one would overwrite the other). Sequential per-agent ensures each request sees the complete conversation history.

**Implementation**:
```python
class ConcurrentAgentManager:
    def __init__(self, ...):
        self._agent_locks: Dict[str, asyncio.Lock] = {}  # per-agent serialization

    async def generate(self, agent_id: str, prompt: str, max_tokens: int) -> str:
        # Get or create per-agent lock
        if agent_id not in self._agent_locks:
            self._agent_locks[agent_id] = asyncio.Lock()

        async with self._agent_locks[agent_id]:
            # This ensures sequential per-agent:
            # 1. Acquire lock (waits if agent is in a running batch)
            # 2. Load agent's CURRENT cache (includes updates from previous request)
            # 3. Submit to batch engine
            # 4. Wait for batch completion
            # 5. Update agent's cache with new state
            # 6. Release lock → next request for this agent can proceed
            cache = self.manager.get_agent_cache(agent_id)
            uid = self.engine.submit(agent_id, prompt, cache, max_tokens)
            result = await self._wait_for_completion(uid)
            self.manager.update_agent_cache(agent_id, result.cache)
            return result.text
```

**Flow example**:
```
Time 0: Agent A req 1 + Agent B req 1 arrive
  → Both submitted to batch (parallel, different agents)
  → Batch processes: A₁ and B₁ generating simultaneously

Time 1: Agent A req 2 arrives (while batch running)
  → Blocked on agent_a_lock (A₁ still in batch)

Time 2: Batch completes
  → A's cache updated with req 1 results
  → B's cache updated with req 1 results
  → agent_a_lock released

Time 3: Agent A req 2 proceeds
  → Loads A's UPDATED cache (includes context from req 1)
  → Submitted to next batch
```

### Step 1.3: Test API Server with Concurrent Requests

**File**: `tests/test_api_server.py` (extend)

- Test 3 concurrent POST /v1/messages with different system prompts
- Verify all responses complete correctly
- Verify streaming works concurrently

**Commit**: `feat: Wire API server to async ConcurrentAgentManager`

---

## Phase 2: BatchedGenerationEngine (using mlx_lm's BatchGenerator)

Replace the sequential single-sequence generation with mlx_lm's built-in `BatchGenerator` for true continuous batching.

### Step 2.1: Create BatchedGenerationEngine

**File**: `src/batched_engine.py` (CREATE, ~250 lines)

```python
from mlx_lm.generate import BatchGenerator, batch_generate

class BatchedGenerationEngine:
    """Wraps mlx_lm's BatchGenerator for multi-agent continuous batching."""

    def __init__(self, model, tokenizer, max_batch_size: int = 5):
        self.model = model
        self.tokenizer = tokenizer
        self.max_batch_size = max_batch_size
        self._batch_gen: Optional[BatchGenerator] = None
        self._active_requests: Dict[int, AgentRequest] = {}  # uid → request
        self._agent_caches: Dict[str, List] = {}  # agent_id → per-sequence cache

    def start(self):
        """Initialize the BatchGenerator."""
        self._batch_gen = BatchGenerator(
            model=self.model,
            tokenizer=self.tokenizer,
            stop_strings=[self.tokenizer.eos_token],
        )

    def submit(self, agent_id: str, prompt: str,
               existing_cache: Optional[List] = None,
               max_tokens: int = 256, temperature: float = 0.7) -> int:
        """Submit a generation request. Returns UID for tracking."""
        # Insert into BatchGenerator with optional pre-computed cache
        uids = self._batch_gen.insert(
            prompts=[prompt],
            max_tokens=[max_tokens],
            prompt_cache=[existing_cache] if existing_cache else None,
            samplers=[make_sampler(temperature)],
        )
        uid = uids[0]
        self._active_requests[uid] = AgentRequest(agent_id=agent_id, uid=uid)
        return uid

    def step(self) -> List[CompletedGeneration]:
        """Run one decode step. Returns list of completed generations."""
        results = self._batch_gen.next()
        completed = []
        for uid, text, metadata in results:
            if metadata.get("done"):
                # Extract per-sequence cache for persistence
                cache = self._batch_gen.batch.extract_cache(uid)
                completed.append(CompletedGeneration(
                    agent_id=self._active_requests[uid].agent_id,
                    text=text,
                    cache=cache,
                ))
                del self._active_requests[uid]
        return completed

    def step_until_done(self) -> List[CompletedGeneration]:
        """Run decode loop until all active requests complete."""
        all_completed = []
        while self._active_requests:
            all_completed.extend(self.step())
        return all_completed
```

### Step 2.2: Integrate with Agent Cache Persistence

**File**: `src/batched_engine.py` (continued)

The key innovation: when a generation completes, extract the per-sequence cache from the batch and save it. When resuming, merge the loaded cache back into the batch.

```python
    def submit_with_cached_agent(self, agent_id: str, prompt: str,
                                  persistence: CachePersistence,
                                  max_tokens: int = 256) -> int:
        """Submit request, loading agent's cache from disk if available."""
        existing_cache = None
        if persistence.agent_cache_exists(agent_id):
            existing_cache, metadata = persistence.load_agent_cache(agent_id)
        return self.submit(agent_id, prompt, existing_cache, max_tokens)
```

### Step 2.3: Tests

**File**: `tests/test_batched_engine.py` (CREATE, ~150 lines)

- Single agent generation through batch engine
- 3 agents concurrently, verify all get correct responses
- Cache extraction after generation (verify can be saved/loaded)
- Insert new request while batch is in progress (continuous batching)
- Max batch size enforcement

**Commit**: `feat: Add BatchedGenerationEngine using mlx_lm BatchGenerator`

---

## Phase 3: Wire ConcurrentAgentManager to BatchedGenerationEngine

Replace `run_in_executor(manager.generate)` with batched generation.

### Step 3.1: Update ConcurrentAgentManager

**File**: `src/concurrent_manager.py` (modify)

The architecture has two layers:
1. **Per-agent locks** (from Phase 1.2): Ensure sequential per-agent, allow cross-agent parallelism
2. **Batch decode worker**: Central loop that runs `engine.step()` and resolves futures

```python
class ConcurrentAgentManager:
    def __init__(self, model_name, max_agents=5, max_batch_size=5):
        self.manager = PersistentAgentManager(model_name, max_agents)
        self.engine = BatchedGenerationEngine(
            self.manager.model, self.manager.tokenizer,
            max_batch_size=max_batch_size
        )
        self._agent_locks: Dict[str, asyncio.Lock] = {}
        self._pending_futures: Dict[int, asyncio.Future] = {}  # uid → future
        self._submit_event = asyncio.Event()  # signals new work available

    async def generate(self, agent_id: str, prompt: str, max_tokens: int = 256) -> str:
        """Per-agent sequential, cross-agent parallel."""
        if agent_id not in self._agent_locks:
            self._agent_locks[agent_id] = asyncio.Lock()

        async with self._agent_locks[agent_id]:
            # Load current cache (updated by previous request if any)
            cache = self.manager.get_agent_cache(agent_id)

            # Submit to batch engine (non-blocking)
            uid = self.engine.submit(agent_id, prompt, cache, max_tokens)
            future = asyncio.get_event_loop().create_future()
            self._pending_futures[uid] = future
            self._submit_event.set()  # wake up batch worker

            # Wait for this specific generation to complete
            result = await future

            # Update agent's cache with new state
            self.manager.update_agent_cache(agent_id, result.cache)
            return result.text

    async def _batch_worker(self):
        """Central decode loop - processes all active requests in batches."""
        while True:
            # Wait for work
            await self._submit_event.wait()
            self._submit_event.clear()

            # Optional: small delay to collect more requests for batching
            await asyncio.sleep(0.01)  # 10ms batching window

            # Run decode steps until all current requests complete
            while self.engine.has_active_requests():
                completed = await asyncio.get_event_loop().run_in_executor(
                    None, self.engine.step
                )
                for result in completed:
                    uid = result.uid
                    if uid in self._pending_futures:
                        self._pending_futures[uid].set_result(result)
                        del self._pending_futures[uid]
```

**How batching happens naturally**:
- Agent A calls `generate()` → submits to engine → signals worker
- Agent B calls `generate()` → submits to engine (while worker doing 10ms wait)
- Worker wakes up → engine now has both A and B → processes as batch
- Both futures resolve → both `generate()` calls return

### Step 3.2: Add Agent Cache Access Methods to PersistentAgentManager

**File**: `src/agent_manager.py` (modify)

Add methods for the batch engine to access/update caches:
- `get_agent_cache(agent_id) -> Optional[List]`: Get current cache (load from disk if needed)
- `update_agent_cache(agent_id, cache)`: Update cache after batch generation
- `build_prompt(agent_id, messages) -> str`: Build prompt from messages with history

### Step 3.3: True Streaming from Batch Engine

**File**: `src/api_server.py` (modify)

For streaming responses, yield tokens as they're generated per-agent:
- Each `step()` call may yield partial text for a specific agent
- The SSE handler yields `content_block_delta` events as tokens arrive
- Use an asyncio.Queue per request to bridge batch engine → SSE stream

### Step 3.4: Integration Tests

**File**: `tests/test_concurrent_batched.py` (CREATE)

- 3 agents submit requests simultaneously → all complete in one batch
- Sequential requests to same agent → cache persists between calls
- Streaming with concurrent requests → correct per-agent token delivery
- API server end-to-end with concurrent curl-style requests

**Commit**: `feat: Wire ConcurrentAgentManager to BatchedGenerationEngine`

---

## Phase 4: KV Cache Quantization

### Step 4.1: Add Quantization to MLXCacheExtractor

**File**: `src/mlx_cache_extractor.py` (modify)

- Add `kv_bits: Optional[int] = None` and `kv_group_size: int = 64` params
- Pass to `stream_generate` kwargs (mlx_lm supports `kv_bits`/`kv_group_size` natively)
- Update `get_cache_memory_bytes` to handle quantized caches

### Step 4.2: Update AgentManager and BatchedEngine

**File**: `src/agent_manager.py` + `src/batched_engine.py` (modify)

- Add `kv_bits` constructor param, pass through to generation
- Quantize before save (reduces disk usage ~50%)
- Load quantized caches and merge into batch (BatchKVCache supports this)

### Step 4.3: Tests

**File**: `tests/test_quantized_cache.py` (CREATE)

- Quantized roundtrip: generate → quantize → save → load → resume
- Memory comparison: 8-bit vs float16
- Batch generation with quantized caches

**Commit**: `feat: Add KV cache quantization (8-bit, 50% memory reduction)`

---

## Phase 5: Demo + Benchmarks

### Step 5.1: Update Full-Stack Demo

**File**: `demo_full_stack.py` (modify)

Add demo showing continuous batching with Anthropic API:
```python
async def demo_continuous_batching():
    """Demo: 3 concurrent Claude Code-style requests."""
    # Start server with batched engine
    # Send 3 requests in parallel (different system prompts = different agents)
    # Show all 3 completing simultaneously with per-agent streaming
    # Show cache persistence: repeat calls are faster (cache loaded from disk)
```

### Step 5.2: Benchmark: Sequential vs Batched

**File**: `benchmarks/batched_benchmark.py` (CREATE)

Compare:
- Sequential: 5 agents × 1 request each, processed one at a time
- Batched: 5 agents × 1 request each, processed in one batch
- Measure: total time, throughput (tokens/sec), per-agent latency

### Step 5.3: Claude Code CLI Integration Demo

Document in demo/README:
```bash
# Start the server with continuous batching
python -m src.api_server --batch-size 5

# In terminal 1: Claude Code session (agent A)
ANTHROPIC_BASE_URL=http://localhost:8000 claude

# In terminal 2: Another Claude Code session (agent B)
ANTHROPIC_BASE_URL=http://localhost:8000 claude

# Both sessions run simultaneously with shared GPU via batching!
```

**Commit**: `feat: Add continuous batching demo and benchmarks`

---

## Key Files

| File | Action | Purpose |
|------|--------|---------|
| `src/batched_engine.py` | CREATE | Wraps mlx_lm BatchGenerator for multi-agent use |
| `src/api_server.py` | MODIFY | Use ConcurrentAgentManager, async generation |
| `src/concurrent_manager.py` | MODIFY | Batch-aware worker using BatchedGenerationEngine |
| `src/agent_manager.py` | MODIFY | Cache access methods, quantization params |
| `src/mlx_cache_extractor.py` | MODIFY | Add kv_bits/kv_group_size params |
| `demo_full_stack.py` | MODIFY | Add continuous batching demo |
| `benchmarks/batched_benchmark.py` | CREATE | Sequential vs batched comparison |
| `tests/test_batched_engine.py` | CREATE | BatchedGenerationEngine tests |
| `tests/test_concurrent_batched.py` | CREATE | Integration tests for batch + concurrent |
| `tests/test_quantized_cache.py` | CREATE | Quantization roundtrip tests |

## Memory Budget (M4 Pro, 24GB)

| Config | Per Agent (4K context) | Max Concurrent Agents |
|--------|------------------------|----------------------|
| Current (float16, sequential) | ~130MB | 1 active (others saved to disk) |
| Batched (float16) | ~130MB each | ~5 agents in batch |
| Batched + quantized (8-bit) | ~65MB each | ~10 agents in batch |

Note: Sliding window layers (40×512 tokens) use ~160MB shared across batch.

## Verification Plan

1. **Phase 1**: `pytest tests/test_api_server.py` - concurrent API calls work
2. **Phase 2**: `pytest tests/test_batched_engine.py` - batch generation produces correct output
3. **Phase 3**: `pytest tests/test_concurrent_batched.py` - full pipeline works
4. **Phase 4**: `pytest tests/test_quantized_cache.py` - quantization roundtrip works
5. **Phase 5**: Run demo with 2+ Claude Code CLI sessions simultaneously
6. **End-to-end**: `ANTHROPIC_BASE_URL=http://localhost:8000 claude` works with persistent agents

## Commit Strategy

0. `docs: Add continuous batching plan and novelty documentation`
1. `feat: Wire API server to async ConcurrentAgentManager`
2. `feat: Add BatchedGenerationEngine using mlx_lm BatchGenerator`
3. `feat: Wire ConcurrentAgentManager to BatchedGenerationEngine`
4. `feat: Add KV cache quantization (8-bit, 50% memory reduction)`
5. `feat: Add continuous batching demo and benchmarks`
