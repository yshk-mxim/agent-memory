# Sprint 2: Block-Pool Batch Engine

**Duration**: 2 weeks (10 days)
**Start Date**: January 24, 2026
**Status**: 🚀 IN PROGRESS (Day 1)

---

## Goal

Deliver a Block-Pool Batch Engine that generates correct text using block-pool allocation with variable-length batching.

**Exit Criteria**:
- Engine generates text matching reference output (greedy, temperature=0)
- No memory leaks (pool size stable across 10+ generations)
- < 20% throughput regression vs POC baseline
- Integration tests pass on Apple Silicon with real MLX models

---

## Sprint 1 Carryover (MANDATORY - Week 1)

| Item | Owner | Due | Status |
|------|-------|-----|--------|
| ADR-001: Hexagonal Architecture | SE | Day 2 | ✅ COMPLETE (Day 1) |
| ADR-002: Block Size = 256 Tokens | ML | Day 2 | ✅ COMPLETE (Day 1) |
| Mock MoE test (alternating layers) | QE | Day 3 | ⏳ PENDING |
| EXP-001: Validate model.args (4 models) | ML | Day 5 | ⏳ PENDING |
| EXP-002: Allocation overhead < 1ms | ML | Day 5 | ⏳ PENDING |

---

## Critical Experiments (BLOCKING)

### EXP-003: Cache Injection Validation ⚠️ BLOCKING
**Owner**: ML, QE
**Duration**: Day 3-5 (1.5 days)
**Status**: ⏳ PENDING

**Objective**: Prove that caches can be injected into BatchGenerator via `caches` parameter.

**Setup**:
1. Load SmolLM2-135M model
2. Generate reference output: "The quick brown fox" → full completion
3. Save cache from reference generation
4. Create NEW BatchGenerator instance
5. Insert same prompt WITH cache via `caches=[loaded_cache]`
6. Generate output with temperature=0 (greedy)

**Success Criteria**:
- Output from cached generation EXACTLY matches reference
- No errors during cache injection
- Cache loading time < 500ms

**Failure Plan**:
- If fails → Trigger **Plan B** discussion (sequential processing)
- Document failure mode in experiment report
- Escalate to PM immediately

**Report**: `/project/experiments/EXP-003-cache-injection.md`

---

### EXP-004: Cache Extraction Validation ⚠️ BLOCKING
**Owner**: ML, QE
**Duration**: Day 4-5 (1.5 days)
**Status**: ⏳ PENDING

**Objective**: Prove that per-sequence caches can be extracted via `Response.prompt_cache()`.

**Setup**:
1. Load SmolLM2-135M model
2. Create BatchGenerator with 3 sequences:
   - Seq A: "Hello world"
   - Seq B: "The quick brown fox"
   - Seq C: "Once upon a time"
3. Run batch to completion
4. Extract cache for each sequence via `response.prompt_cache()`
5. Save caches with `save_prompt_cache()`
6. Load caches and re-inject into NEW batch
7. Continue generation

**Success Criteria**:
- All 3 caches extract successfully
- Roundtrip: save → load → re-inject works
- Continued generation is correct (matches fresh generation)

**Failure Plan**:
- If extraction fails → Explore alternative extraction methods
- If roundtrip fails → Validate safetensors serialization
- Document findings and escalate

**Report**: `/project/experiments/EXP-004-cache-extraction.md`

---

## Week 1 Tasks (Days 1-5)

### Day 1-2: Foundation

#### SE Tasks
- ✅ **ADR-001**: Hexagonal Architecture (4 hours)
  - Document ports/adapters pattern
  - Justify Protocol-based dependency inversion
  - Diagram showing dependency flow
  - Publish to `/project/architecture/ADR-001-hexagonal-architecture.md`

- 🔄 **Design Typed Ports** (6 hours)
  - `GenerationEnginePort` Protocol:
    ```python
    class GenerationEnginePort(Protocol):
        def submit(self, prompt: str, cache: Any | None, max_tokens: int) -> str: ...
        def step(self) -> Iterator[CompletedGeneration]: ...
    ```
  - `CacheStorePort` Protocol:
    ```python
    class CacheStorePort(Protocol):
        def get(self, cache_key: CacheKey) -> AgentBlocks | None: ...
        def put(self, cache_key: CacheKey, blocks: AgentBlocks) -> None: ...
    ```
  - `ModelProviderPort` Protocol:
    ```python
    class ModelProviderPort(Protocol):
        def load_model(self, model_id: str) -> tuple[Any, Any]: ...  # (model, tokenizer)
        def extract_spec(self, model: Any) -> ModelCacheSpec: ...
    ```
  - Publish to `/src/semantic/ports/inbound.py` and `/src/semantic/ports/outbound.py`

#### ML Tasks
- ✅ **ADR-002**: Block Size = 256 Tokens (4 hours)
  - Document rationale (matches MLX KVCache.step)
  - Analyze memory efficiency
  - Diagram showing block alignment
  - Publish to `/project/architecture/ADR-002-block-size-256.md`

- 🔄 **mlx_lm Source Code Review** (2 hours)
  - Read `mlx_lm/generate.py` BatchGenerator class
  - Document `insert()` method signature
  - Document `next()` method behavior
  - Document `Response.prompt_cache()` method
  - Create cheat sheet: `/project/reference/mlx_lm_api_v0.30.4.md`

- 🔄 **Experiment Framework Setup** (2 hours)
  - Create `/project/experiments/` template
  - Download SmolLM2-135M for testing
  - Create fixture for EXP-003/004

#### QE Tasks
- 🔄 **Mock MoE Test** (4 hours)
  - Create test with alternating layer types
  - Validate ModelCacheSpec handles pattern
  - Add to `/tests/unit/test_value_objects.py`

- 🔄 **EXP-003/004 Test Strategy** (2 hours)
  - Define "output matches" criteria
  - Create test data generation script
  - Set up validation framework

---

### Day 3-5: Critical Experiments

#### Day 3-4: EXP-003 Execution
- ML: Implement cache injection test
- QE: Validate outputs match reference
- PM: Monitor progress (GO/NO-GO checkpoint Day 5)

#### Day 4-5: EXP-004 Execution
- ML: Implement cache extraction test
- QE: Validate roundtrip correctness
- PM: Prepare Plan B materials if needed

#### Day 5: Sprint 1 Experiments (Parallel)
- **EXP-001**: ML loads 4 real models, validates model.args
  - Gemma 3 12B: n_kv_heads=8 ✅
  - Llama 3.1 8B: n_kv_heads=4 ✅
  - Qwen 2.5 7B: n_kv_heads=4 ✅
  - Qwen1.5-MoE-A2.7B: Validate MoE structure ✅

- **EXP-002**: ML benchmarks block allocation
  - 1000 allocate/free cycles
  - Target: < 1ms per operation
  - Report: `/project/benchmarks/block_allocation_overhead.md`

---

## Week 2 Tasks (Days 6-10)

### Day 6-7: BlockPoolBatchEngine.submit()

**Owner**: SE, ML
**Duration**: 1.5 days

**Implementation**:
```python
class BlockPoolBatchEngine:
    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        pool: BlockPool,
        spec: ModelCacheSpec,
    ):
        self._model = model
        self._tokenizer = tokenizer
        self._pool = pool
        self._spec = spec
        self._batch_gen = BatchGenerator(model, tokenizer.eos_token_ids)

    def submit(
        self,
        agent_id: str,
        prompt: str,
        cache: AgentBlocks | None = None,
        max_tokens: int = 256,
    ) -> str:
        """Submit generation request with optional cached blocks."""
        # 1. Allocate blocks for new tokens
        tokens_needed = len(self._tokenizer.encode(prompt))
        blocks_needed = (tokens_needed + 255) // 256  # Ceiling division

        # 2. Reconstruct KVCache from blocks (if cache provided)
        kv_cache = None
        if cache:
            kv_cache = self._reconstruct_cache(cache)

        # 3. Insert into BatchGenerator
        uid = self._batch_gen.insert([prompt], max_tokens, caches=[kv_cache])

        return uid[0]
```

**Tests**:
- Unit test: submit() allocates correct number of blocks
- Unit test: submit() with cache reconstructs correctly
- Unit test: submit() without cache works

---

### Day 7-8: Block-to-Cache Reconstruction

**Owner**: ML
**Duration**: 1 day

**Implementation**:
```python
def _reconstruct_cache(self, agent_blocks: AgentBlocks) -> Any:
    """Reconstruct KVCache from blocks (one-time at restore)."""
    import mlx.core as mx

    cache = []
    for layer_id in range(self._spec.n_layers):
        layer_blocks = agent_blocks.blocks_for_layer(layer_id)

        # Gather all K tensors for this layer
        k_tensors = [block.layer_data["k"] for block in layer_blocks]
        v_tensors = [block.layer_data["v"] for block in layer_blocks]

        # Concatenate (one-time cost)
        k_full = mx.concatenate(k_tensors, axis=2)  # [n_heads, head_dim, seq_len]
        v_full = mx.concatenate(v_tensors, axis=2)

        mx.eval(k_full, v_full)  # Force evaluation

        cache.append((k_full, v_full))

    return cache
```

**EXP-006**: Benchmark gather overhead
- 32 blocks × 48 layers for 8K context
- Target: < 5ms total
- If > 5ms: Document in ADR-004, accept trade-off

**Tests**:
- Unit test: _reconstruct_cache() produces correct shape
- Unit test: mx.concatenate happens only once
- Integration test: Reconstructed cache works in generation

---

### Day 8-9: BlockPoolBatchEngine.step() + Cache Extraction

**Owner**: ML
**Duration**: 1 day

**Implementation**:
```python
def step(self) -> Iterator[CompletedGeneration]:
    """Yield completed generations with extracted caches."""
    responses = self._batch_gen.next()

    for response in responses:
        if response.finish_reason is not None:
            # Extract cache for this sequence
            cache_func = response.prompt_cache()
            raw_cache = cache_func()  # Call to get actual cache

            # Convert cache to blocks
            blocks = self._cache_to_blocks(raw_cache, response.uid)

            yield CompletedGeneration(
                uid=response.uid,
                text=response.text,
                blocks=blocks,
                finish_reason=response.finish_reason,
            )
```

**Block Extension During Decode**:
```python
def _check_extension(self, uid: str, current_tokens: int):
    """Allocate new block every 256 tokens."""
    if current_tokens % 256 == 0:
        new_block = self._pool.allocate(
            n_blocks=1,
            layer_id=0,  # Will allocate for all layers
            agent_id=uid,
        )
```

**Tests**:
- Unit test: step() yields completions correctly
- Unit test: Cache extraction works
- Unit test: Block extension triggers at 256, 512, 768 tokens
- Integration test: Extended blocks are freed on completion

---

### Day 9-10: Integration Tests + Failure Modes

**Owner**: QE, ML
**Duration**: 2 days

#### Integration Test 1: Single Agent
```python
@pytest.mark.integration
def test_single_agent_output_matches_reference():
    """Single agent output should match reference (greedy)."""
    # 1. Load SmolLM2-135M
    model, tokenizer = load("mlx-community/SmolLM2-135M-Instruct")
    spec = ModelCacheSpec.from_model(model)
    pool = BlockPool(spec, total_blocks=100)

    # 2. Generate reference (no cache)
    prompt = "The quick brown fox"
    reference = generate_reference(model, tokenizer, prompt, max_tokens=50)

    # 3. Generate with engine
    engine = BlockPoolBatchEngine(model, tokenizer, pool, spec)
    uid = engine.submit("agent_1", prompt, cache=None, max_tokens=50)

    result = None
    for completion in engine.step():
        if completion.uid == uid:
            result = completion.text
            break

    # 4. Validate
    assert result == reference  # Byte-identical (greedy)
```

#### Integration Test 2: 3 Agents Variable Length
```python
@pytest.mark.integration
def test_three_agents_variable_length():
    """3 agents with different lengths should all complete correctly."""
    agents = [
        ("agent_a", "Hello world", 500),
        ("agent_b", "The quick brown fox", 4000),
        ("agent_c", "Once upon a time", 1200),
    ]

    # Submit all 3
    for agent_id, prompt, max_tokens in agents:
        engine.submit(agent_id, prompt, max_tokens=max_tokens)

    # Collect completions
    completions = {}
    for completion in engine.step():
        completions[completion.uid] = completion
        if len(completions) == 3:
            break

    # Validate all completed
    assert len(completions) == 3
    assert all(c.finish_reason == "stop" for c in completions.values())
```

#### Failure-Mode Tests
```python
@pytest.mark.integration
def test_pool_exhaustion_mid_decode():
    """Should fail gracefully when pool exhausted."""
    # Allocate all blocks
    pool = BlockPool(spec, total_blocks=10)
    for i in range(10):
        pool.allocate(n_blocks=1, layer_id=0, agent_id=f"blocker_{i}")

    # Try to submit new request
    with pytest.raises(ValueError, match="Cannot allocate.*blocks"):
        engine.submit("agent_new", "Hello", max_tokens=256)

@pytest.mark.integration
def test_lock_starvation():
    """Concurrent requests to same agent should not deadlock."""
    # This test would require threading, may defer to Sprint 4
    pass
```

---

## ADRs to Write

### ADR-004: Block Gather Strategy
**Owner**: ML
**Due**: Day 10
**Content**:
- Decision: One-time gather at restore, not per-step
- Benchmark: EXP-006 results (overhead measurement)
- Trade-off: Higher restore latency vs lower per-step cost
- Alternatives considered: Per-step gather, pre-allocated buffers

### ADR-005: Composition Pivot
**Owner**: SE
**Due**: Day 10
**Content**:
- Decision: Custom BlockPoolBatchEngine wraps BatchGenerator
- Rationale: Need block lifecycle management, cache extraction
- Alternatives: Subclass BatchGenerator (rejected - tight coupling)
- Trade-off: More code vs cleaner abstraction

---

## Exit Gate Checklist

### Functional Criteria
- [ ] BlockPoolBatchEngine.submit() works
- [ ] BlockPoolBatchEngine.step() works
- [ ] Block-to-cache reconstruction works
- [ ] Cache-to-block extraction works
- [ ] Block extension during decode works
- [ ] Single-agent output matches reference
- [ ] 3-agent variable-length test passes
- [ ] Failure-mode tests pass

### Quality Criteria
- [ ] No memory leaks (profiling)
- [ ] < 20% throughput regression
- [ ] Unit tests pass
- [ ] Integration tests pass (Apple Silicon)
- [ ] All experiments documented

### Documentation Criteria
- [ ] ADR-004: Block Gather Strategy
- [ ] ADR-005: Composition Pivot
- [ ] EXP-003 report
- [ ] EXP-004 report
- [ ] EXP-006 report

### Sprint 1 Carryover
- [x] ADR-001: Hexagonal Architecture (Day 1 ✅)
- [x] ADR-002: Block Size = 256 Tokens (Day 1 ✅)
- [ ] Mock MoE test
- [ ] EXP-001: model.args validated
- [ ] EXP-002: Allocation < 1ms

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| EXP-003/004 fail | MEDIUM | CRITICAL | Execute early (Day 3-5); Plan B ready |
| mlx_lm API undocumented | MEDIUM | HIGH | Source code review; extensive testing |
| Block gather > 5ms | MEDIUM | HIGH | Benchmark early; accept trade-off if needed |
| Integration tests fail | LOW | HIGH | Start early (Day 9); leave buffer for fixes |
| Sprint 1 carryover delays Sprint 2 | LOW | MEDIUM | Parallel execution; clear deadlines |

---

## Daily Standup Log

### Day 1 (January 24, 2026)
**Attendees**: SE, ML, QE, PM

**Completed**:
- ✅ Sprint 2 standup meeting conducted
- ✅ Sprint 2 planning document created
- ✅ .gitignore updated (site/, .hypothesis/, .ruff_cache/)
- ✅ Sprint 1 committed (a98cfca)
- ✅ ADR-001: Hexagonal Architecture (SE) - 200+ lines, comprehensive
- ✅ ADR-002: Block Size = 256 Tokens (ML) - 400+ lines, comprehensive
- ✅ Day 1 review conducted - CONDITIONAL GO for Day 2
- ✅ Duplicate ADR-002 file deleted

**Pending (carry to Day 2)**:
- ⏳ Typed ports design (SE) - clarify vs Sprint 1 ports
- ⏳ mlx_lm source code review (ML)
- ⏳ Experiment framework setup (ML)
- ⏳ Mock MoE test (QE)
- ⏳ EXP-003/004 test strategy (QE)

**Blockers**: None

**Next**:
- Day 2: Execute 4-hour critical path (mlx_lm review, experiment setup, MoE test)
- Day 2: Clarify port design strategy (SE)
- Day 3: Start EXP-003 (cache injection validation)

---

## Notes

- EXP-003/004 are CRITICAL PATH - all Week 2 work depends on them
- Plan B (sequential processing) ready if experiments fail
- Integration tests require Apple Silicon (macos-14 runner)
- All carryover items MUST complete by end of Week 1
