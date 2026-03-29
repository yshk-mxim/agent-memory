# TODO — TRT Backend Integration

Branch `feat/trt-backend`. Last updated: 2026-03-29.

---

## Completed

- [x] C1. InferencePort / TRTInferenceService
- [x] C2. TRT KV cache persistence (full AgentBlocks save/load/dequantize)
- [x] C3. BatchEngine moved to adapter layer
- [x] C4. TRTSafetensorsCacheAdapter (no MLX dependency)
- [x] C5. Startup probe + debug endpoint for TRT (was C6)
- [x] H1. TRTSpecExtractor port conformance
- [x] H6. Thread safety (Lock in subprocess adapter)
- [x] H7. OpenAI adapter TRT path
- [x] H8. Domain naming (backend_io_lock)
- [x] M1-M8. Settings, magic numbers, backend-specific config
- [x] Multi-turn KV persistence verified on Thor
- [x] SSE streaming for TRT (enables Claude Code)
- [x] Model-agnostic special token cleanup

---

## In Progress — Current Sprint

### 1. Convert e2e tests to pytest + functional test suite
Replace `test_e2e_server.sh` with pytest-based e2e tests. Add:
- Server lifecycle (start with TRT, health check, shutdown)
- Non-streaming Anthropic API request
- Streaming Anthropic API request
- Multi-turn conversation with X-Session-ID + cache persistence
- OpenAI API request
- Cache file exists on disk after generation

### 2. Multiple models support on TRT
- TRT hot-swap via subprocess stop/restart with new engine path
- Or: graceful "not supported" error with restart instructions
- Model registry awareness of TRT backend

### 3. Full LLM ops pipeline
Unified in TRTInferenceService:
- Chat templates (ChatML, Llama, Qwen) via ChatTemplatePort
- Completion/FIM templates for code completion
- Stop sequence handling (model-specific EOS tokens)
- Sampling parameters (temperature, top_p, top_k, repetition_penalty)
- Accurate token counting (prompt + completion)

### 4. Complete test coverage for all above

---

## Deferred — Later

### Claude Code integration test
Test with `ANTHROPIC_BASE_URL=http://localhost:8199 CLAUDE_CODE_ATTRIBUTION_HEADER=0 claude`.
Requires: streaming works (done), tool_use works (done), agentic loop with
multi-turn persistence. Cannot be automated — requires interactive Claude Code session.

### NemoClaw production deployment
- Build Qwen3-Coder-Next engine on Thor (same pipeline as SmolLM2)
- Configure NemoClaw to use `http://thor:8199` as LLM endpoint
- Verify multi-turn agent conversations with persistent KV cache
- Performance benchmarks (tok/s, TTFT, cache restore latency)

### Other deferred items
- C5. README.md and docs/ update for dual-backend
- H2. Model hot-swap orchestrator for TRT
- H4. SBOM update (numpy, TensorRT-Edge-LLM)
- H9. LRU vs LFU eviction analysis for NemoClaw/Claude Code workloads
- L1. Performance benchmarks on Thor
- L2. CITATION.cff version update
- L4. System prompt caching for TRT (genAndSaveSystemPromptKVCache)
