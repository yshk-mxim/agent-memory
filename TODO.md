# TODO — TRT Backend Integration & Full Audit

Branch `feat/trt-backend`. Audit date: 2026-03-29.

---

## CRITICAL — Blocks merge

### C1. No unified InferencePort — backend branching in inbound adapters
**Hex violation V13+V14.** The Anthropic adapter has `if trt_subprocess is not None
and batch_engine is None:` with a completely separate code path for TRT. The OpenAI
adapter likely has the same issue. Inbound adapters should call ONE application
service; the backend choice must be invisible to them.

**Fix**: Define `InferencePort` (or `GenerationServicePort`) with
`generate(messages, max_tokens, temperature, cache?) → GenerationResult`.
Both `BlockPoolBatchEngine` and `TRTSubprocessAdapter` implement it (or a thin
application-layer wrapper does). The adapter calls `inference.generate()`.

**Files**: `ports/outbound.py`, `anthropic_adapter.py`, `openai_adapter.py`,
`api_server.py`

### C2. TRT path missing KV cache persistence in Anthropic adapter
The TRT code path in `anthropic_adapter.py:692-732` does NOT:
- Look up `agent_id` from `X-Session-ID` header
- Load cached blocks from `cache_store`
- Inject prior KV cache before generation
- Save updated KV cache to `cache_store`

**Breaks NemoClaw/OpenClaw multi-turn conversations** — the core value prop.

**Fix**: Wire through `agent_id → cache_store.load() → inject → generate →
extract → cache_store.save()`. Ideally this happens inside the unified
InferencePort (C1) so the adapter doesn't know about caching either.

### C3. BatchEngine is in application layer but hardcoded to MLX
**Hex violation V2.** `application/batch_engine.py` has 14+ deferred `import mlx.core`
calls. It cannot be used without MLX. TRT bypasses it entirely (`batch_engine=None`).

**Fix**: Move to `adapters/outbound/mlx_batch_engine.py`, or extract MLX-specific
ops to injected ports. Blocked on C1 (need InferencePort first).

### C4. SafetensorsCacheAdapter requires MLX for all backends
**Hex violation V5.** `save()` calls `mx.save_safetensors()`, `load()` calls
`mx.load()`. TRT backend cache persistence silently depends on MLX being installed.

**Fix**: Split into `MLXSafetensorsCacheAdapter` (uses mlx I/O) and
`NumpySafetensorsCacheAdapter` (uses safetensors.numpy). Both implement
`CachePersistencePort`. Wired by backend in `api_server.py`.

### C5. README.md and docs/ not updated
Zero mention of TRT, Jetson, `SEMANTIC_BACKEND`, dual-backend, or NemoClaw.
- `README.md` — tagline, requirements, supported models, config
- `docs/configuration.md` — `SEMANTIC_TRT_*` variables
- `docs/architecture.md` — dual-backend diagram
- `docs/deployment.md` — Jetson deployment
- New: `docs/trt-backend.md` — setup + build guide

### C6. Startup health probe broken for TRT
**Hex violation V10.** `/health/startup` checks `batch_engine is not None` and
returns 503 for TRT backend (where `batch_engine=None`). Fix: also check
`trt_subprocess is not None`.

---

## HIGH — Should fix before merge

### H1. TRTSpecExtractor doesn't conform to SpecExtractorPort
**V6.** Port: `extract_spec(model) → ModelCacheSpec`. Adapter: `extract() → ModelCacheSpec`.
Different name, different arity.

### H2. Model hot-swap does not work for TRT
`ModelSwapOrchestrator` is MLX-centric. TRT needs subprocess stop/start.

### H3. /debug/memory endpoint crashes on TRT
**V7.** `import mlx.core as mx` inside `/debug/memory` — registered unconditionally.
Will crash on non-Apple platforms.

### H4. SBOM stale
Missing `numpy` dependency, vendored `TensorRT-Edge-LLM` (Apache-2.0).

### H5. Missing SPDX header on vendor/patches/add_debug.py

### H6. Thread safety: TRTSubprocessAdapter has no lock
`_send_command`/`_read_response` are not thread-safe. Two concurrent `generate()`
calls would interleave NDJSON on stdin.

### H7. OpenAI adapter not wired for TRT
`/v1/chat/completions` uses `batch_engine` directly. Same issue as C1/C2.

### H8. `mlx_io_lock` in domain layer named for specific backend
**V1.** `domain/services.py:20` has `mlx_io_lock`. Rename to `backend_io_lock`.

---

### H9. Evaluate LRU vs LFU eviction for NemoClaw workload
NemoClaw/OpenClaw agents have long-lived system prompts that are shared across
conversations. With LRU, a system prompt cache is evicted when newer agent caches
push it out — even though it's the most frequently reused cache. LFU (or
frequency-weighted LRU) would keep hot system prompt caches in memory.

Current: `AgentCacheStore` uses pure LRU (`min(last_accessed)`).

**Analysis needed**:
- NemoClaw pattern: N agents share 1-2 system prompts + tools prefix. Each
  conversation turn creates an agent-specific cache. System prompt KV cache
  (~2K tokens) is the most reusable, but LRU evicts it when max_agents is hit
  because it was accessed "long ago" (at conversation start).
- LFU would keep system prompt caches warm (high access count) while evicting
  stale per-agent caches.
- Hybrid LFU+LRU (frequency-weighted recency) may be best: score =
  `access_count / (now - last_accessed)`.
- The `SharedPrefixCache` partially solves this for the MLX path (caches system
  prompt computation), but TRT doesn't use it.
- `TRTSystemPromptCache` uses `genAndSaveSystemPromptKVCache()` which stores
  the system prompt KV on disk — this is effectively LFU-infinite (never evicted).
  But it's not wired into `AgentCacheStore`.

**Claude Code CLI pattern**: Similar but different — Claude Code sends long system
prompts with tool definitions (~18K tokens observed) that change rarely. Each
agentic loop iteration appends to the conversation. LRU would evict the system
prompt cache between loops. The `X-Session-ID` header keeps the agent_id stable
across turns, but the system+tools prefix KV cache (which is the expensive part
to recompute) gets evicted when other sessions push it out.

**Options**:
1. Add `eviction_policy: Literal["lru", "lfu", "lru-lfu"]` to `AgentSettings`
2. Exempt system prompt caches from eviction (pin them)
3. Use TRT's native system prompt caching for TRT backend, SharedPrefixCache for MLX
4. Frequency-weighted scoring: `score = access_count * recency_weight` where
   `recency_weight = 1 / (1 + hours_since_last_access)`. This keeps both
   frequently-used (system prompts) and recently-used (active conversations)
   caches warm.

**Files**: `application/agent_cache_store.py`, `adapters/config/settings.py`

---

## MEDIUM — Fix soon after merge

### M1. No `top_p`/`top_k` in TRTSettings
### M2. No `EDGELLM_PLUGIN_PATH` in TRTSettings
### M3. NemoClaw default model (Qwen3-Coder-Next) not built/verified on Thor
### M4. Remove debug fprintf from attention plugin
### M5. Magic number 256 in trt_subprocess_adapter.py
### M6. `settings.mlx.reasoning_extra_tokens` used for TRT coordination service (V11)
### M7. `/v1/models` endpoint references `settings.mlx.max_context_length` for all backends (V12)
### M8. `cli.py` sets `MLX_METAL_FAST_SYNCH` unconditionally (V9)
### M9. structlog (third-party) in application layer (V3) — consider using stdlib logging

---

## LOW — Nice to have

### L1. Performance benchmarks on Thor
### L2. CITATION.cff version stale (says 1.0.1)
### L3. Streaming support for TRT backend
### L4. System prompt caching for TRT (`genAndSaveSystemPromptKVCache`)
### L5. Decompose `swap_model()` (117 lines) per 50-line guideline

---

## Test Coverage Gaps

### Untested files (zero tests):
- `trt_model_loader.py` — tokenizer error handling, subprocess creation
- `trt_prefill_adapter.py` — chunk sizing, prefill pipeline
- `trt_system_prompt_cache.py` — cache-hit, cache-miss, save/load

### Untested error paths:
- Cache inject failure in `TRTSubprocessAdapter.generate()`
- `OSError` on subprocess start
- `kill()` after `TimeoutExpired` in `stop()`
- Tokenizer load failure in `TRTModelLoader`
- System prompt prefill returning no cache

### Missing test scenarios:
- [ ] Multi-turn (5+) with cache persistence on TRT
- [ ] Cache persistence across server restart
- [ ] Concurrent requests on TRT (thread safety)
- [ ] Memory budget enforcement under load
- [ ] Disk budget enforcement under load
- [ ] Large context (>4K tokens) prefill chunking
- [ ] Cache corruption handling for TRT FP16 files
- [ ] Server graceful shutdown with subprocess cleanup
- [ ] `messages` parameter in fake_llm_inference.py
- [ ] `max_tokens=0` (prefill-only) in fake

### Test infrastructure:
- [ ] CI job for TRT integration tests (fake binary, ubuntu-latest)
- [ ] Convert `test_e2e_server.sh` to pytest
- [ ] Error simulation mode in fake_llm_inference.py
- [ ] Property-based (hypothesis) tests for quantization
- [ ] Replace `os.environ` mutation in conftest with `monkeypatch`

---

## PR Status (from plan)

| PR | Scope | Status |
|----|-------|--------|
| 1 | Domain: kv_format + TRT errors | Done |
| 2 | Port: CacheQuantizationPort | Done |
| 3 | Settings: TRTSettings, backend selector | Done |
| 4 | Adapters: trt_cache, trt_quantization | Done |
| 5 | Refactor: safetensors uses CacheQuantizationPort | Done |
| 6 | Adapters: subprocess, loader, extractor, prefill, sysprompt | Done |
| 7 | Fix: Anthropic adapter + unit tests | Done |
| 8 | Wiring: api_server.py backend selection | Done |
| 9 | Thor: integration tests (SmolLM2-135M) | Done |
| — | C++ interactive binary with KV cache | Done |
| — | End-to-end Anthropic API on Thor | Done |
| — | **InferencePort abstraction (C1+C3)** | **TODO** |
| — | **TRT KV cache persistence in adapter (C2)** | **TODO** |
| — | **Split SafetensorsCacheAdapter (C4)** | **TODO** |
| — | **Docs update (C5)** | **TODO** |
| — | **NemoClaw integration + Qwen3 engine (M3)** | **TODO** |

---

## Verification Items (from plan)

- [x] Unit tests pass everywhere: `make test-unit` — 1211 pass
- [x] Lint + type check: `make ci` — clean
- [x] On Thor: `pytest tests/trt/ -v` — 4 real tests pass
- [x] End-to-end: `SEMANTIC_BACKEND=trt` + curl — "2 + 2 = 4"
- [ ] Claude Code integration: `ANTHROPIC_BASE_URL=localhost:8000 claude`
- [ ] OpenClaw/NemoClaw: persistent KV cache across conversation turns
- [ ] Cache persistence: generate → save → restart → load → continue
