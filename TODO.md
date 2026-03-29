# TODO — TRT Backend Integration & Full Audit

Branch `feat/trt-backend`. Audit date: 2026-03-29. Last updated: 2026-03-29.

---

## CRITICAL — Blocks merge

### ~~C1. InferencePort~~ — DONE
TRTInferenceService wraps backend + cache persistence. Anthropic/OpenAI adapters
call `trt_inference.generate(agent_id, prompt, messages)`.

### ~~C2. TRT KV cache persistence~~ — DONE
TRTInferenceService._save_agent_cache builds AgentBlocks from per-layer KV tuples,
calls TRTSafetensorsCacheAdapter.save(). _load_agent_cache reverses.

### C3. BatchEngine is in application layer but hardcoded to MLX
**Hex violation V2.** `application/batch_engine.py` has 14+ deferred `import mlx.core`
calls. Cannot be used without MLX.

**Fix**: Move to `adapters/outbound/mlx_batch_engine.py`. Update all imports.

### ~~C4. SafetensorsCacheAdapter requires MLX~~ — DONE
TRTSafetensorsCacheAdapter uses safetensors.numpy. No MLX dependency. Each backend
has its own cache I/O adapter. Wired by backend in api_server.py.

### C5. README.md and docs/ not updated
Zero mention of TRT, Jetson, `SEMANTIC_BACKEND`, or NemoClaw.

### ~~C6. Startup health probe + debug endpoint~~ — DONE
Probe checks both batch_engine and trt_subprocess. Debug endpoint gracefully
handles missing MLX.

---

## HIGH — Should fix before merge

### ~~H1. TRTSpecExtractor port conformance~~ — DONE
Now uses `extract_spec(model)` matching SpecExtractorPort.

### H2. Model hot-swap does not work for TRT
### ~~H3. /debug/memory endpoint~~ — DONE (folded into C6)
### H4. SBOM stale
### H5. Missing SPDX header on vendor/patches/add_debug.py
### ~~H6. Thread safety~~ — DONE (threading.Lock in generate)
### ~~H7. OpenAI adapter TRT path~~ — DONE
### ~~H8. Domain naming~~ — DONE (backend_io_lock)

### H9. LRU vs LFU eviction for NemoClaw/Claude Code
See analysis in previous version of this file. Options:
1. `eviction_policy` setting (lru/lfu/hybrid)
2. Pin system prompt caches
3. Frequency-weighted scoring: `access_count / (1 + hours_since_last_access)`

---

## MEDIUM — Fix soon after merge

- [ ] M1. Add `top_p`/`top_k` to TRTSettings
- [ ] M2. Add `EDGELLM_PLUGIN_PATH` to TRTSettings
- [ ] M3. NemoClaw default model (Qwen3-Coder-Next) engine build on Thor
- [ ] M4. Remove debug fprintf from attention plugin
- [ ] M5. Magic number 256 → BLOCK_SIZE_TOKENS in trt_subprocess_adapter.py
- [ ] M6. Fix `settings.mlx.reasoning_extra_tokens` for TRT coordination (V11)
- [ ] M7. Fix `/v1/models` endpoint to use backend-specific settings (V12)
- [ ] M8. Gate `MLX_METAL_FAST_SYNCH` env var on backend==mlx in cli.py (V9)
- [ ] M9. Consider stdlib logging in application layer (V3)

---

## LOW — Nice to have

- [ ] L1. Performance benchmarks on Thor
- [ ] L2. CITATION.cff version update
- [ ] L3. Streaming support for TRT backend
- [ ] L4. System prompt caching for TRT
- [ ] L5. Decompose swap_model() per 50-line guideline

---

## Test Coverage Gaps

### Untested files:
- [ ] `trt_model_loader.py`
- [ ] `trt_prefill_adapter.py`
- [ ] `trt_system_prompt_cache.py`
- [ ] `trt_safetensors_cache_adapter.py`

### Missing test scenarios:
- [ ] Multi-turn (5+) with cache persistence on TRT
- [ ] Cache persistence across server restart
- [ ] Concurrent requests on TRT (thread safety)
- [ ] Memory/disk budget enforcement under load
- [ ] Server graceful shutdown with subprocess cleanup
- [ ] `messages` parameter + `max_tokens=0` in fake_llm_inference.py
- [ ] Property-based (hypothesis) tests for quantization

### Test infrastructure:
- [ ] CI job for TRT integration tests
- [ ] Convert test_e2e_server.sh to pytest
- [ ] Error simulation in fake_llm_inference.py

---

## Verification Items (from plan)

- [x] Unit tests pass everywhere: 1211 pass
- [x] Lint + type check: clean
- [x] On Thor: 4 real tests pass
- [x] End-to-end: Anthropic API → "2 + 2 = 4"
- [ ] Claude Code integration
- [ ] OpenClaw/NemoClaw multi-turn with persistent KV cache
- [ ] Cache persistence across server restart
