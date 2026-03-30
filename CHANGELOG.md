# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.1.0-alpha] — TRT Backend + MLX 0.31 Upgrade (Unreleased)

> **Status: Alpha.** TRT backend and mlx-lm 0.31 cache pipeline are functional
> but not production-hardened. Interactive testing with Claude Code CLI and
> NemoClaw/OpenClaw pending. Cache persistence across restarts verified on
> Mac (MLX) and Thor (TRT) but not load-tested. API surface may change.

### Added

- **TensorRT Edge-LLM backend** for NVIDIA Jetson AGX Thor (aarch64, sm_110).
  Set `SEMANTIC_BACKEND=trt` to activate. Manages `llm_inference` binary via
  subprocess with NDJSON protocol and safetensors KV cache transfer over `/dev/shm`.
- `kv_format` field on `ModelCacheSpec` — distinguishes software-quantized caches
  (`"quantized"`, MLX Q4/Q8 with per-group scales/biases) from native floating-point
  caches (`"fp"`, TRT FP16/FP8). `bytes_per_block_per_layer()` now handles FP8
  (1 byte/element, no overhead). Default is `"quantized"` (backward compatible).
- `CacheQuantizationPort` protocol in `ports/outbound.py` — pluggable quantization
  interface for external quantizers like [turboquant-mlx](https://github.com/arozanov/turboquant-mlx).
- `TRTSettings` configuration (`SEMANTIC_TRT_*` env vars) for engine path, binary
  path, subprocess timeout, shared memory directory, FP8 KV bits, and disk Q4 format.
- `backend` field on root `Settings` (`SEMANTIC_BACKEND` env var) — selects `"mlx"`
  or `"trt"` inference backend.
- `TRTSubprocessAdapter` — persistent subprocess management with NDJSON control
  protocol, safetensors KV cache injection/extraction via shared memory.
- `TRTCacheAdapter` — numpy-based KV cache tensor operations with TRT 5D layout
  translation (`[numLayers, 2, numKVHeads, seqLen, headDim]`).
- `TRTQuantizationAdapter` — asymmetric min-max Q4/Q8 quantization compatible with
  MLX's `mx.quantize()` disk format. Supports external quantizer plugins via
  `CacheQuantizationPort`.
- `TRTSpecExtractor`, `TRTPrefillAdapter`, `TRTModelLoader`, `TRTSystemPromptCache`.
- `Qwen/Qwen3-Coder-Next-nvfp4` model profile (`config/models/qwen3-coder-next-nvfp4.toml`).
- TRT-specific domain errors: `TRTSubprocessError`, `TRTEngineError`, `TRTLayoutError`.
- Cross-platform memory/disk budget settings: `SEMANTIC_AGENT_MAX_MEMORY_MB` and
  `SEMANTIC_AGENT_MAX_DISK_MB` in `AgentSettings`.
- Unit tests for Anthropic adapter pure functions (`parse_tool_calls`,
  `generate_agent_id_from_tokens`).
- `fake_llm_inference.py` test fixture — standalone NDJSON subprocess mock for
  TRT adapter unit tests without CUDA/TRT.
- TRT integration tests (`tests/trt/`) — subprocess lifecycle, KV cache layout
  round-trip, Q4 quantization round-trip through safetensors. Runs on Mac via
  fake binary with SmolLM2-135M geometry.
- `llm_inference_wrapper.py` — Python NDJSON wrapper bridging `TRTSubprocessAdapter`
  to the stock `llm_inference` binary's JSON file I/O. No custom C++ needed.
- Real TRT inference tests on Thor (`test_trt_real_inference.py`) — SmolLM2-135M
  generating coherent text through the full adapter -> wrapper -> engine pipeline.
- TRT Edge-LLM build pipeline (`vendor/`) — build scripts, Dockerfile, sm_110
  FMHA patch, and detailed `BUILD_LOG.md` documenting every Thor-specific gotcha.
- `vendor/patches/sm110_fmha_fix.py` — patches Edge-LLM 0.6.0 context FMHA runner
  to remap sm_110 -> sm_101 cubins (matching NVIDIA's own `applyThorSMRenumberWAR`
  in the attention plugin, which was missing from the context attention path).
- `vendor/patches/add_engine_accessor.py` — adds `getEngineRunner()` public accessor
  to `LLMInferenceRuntime` for KV cache access via `LinearKVCache` API.
- `vendor/llm_inference_interactive.cpp` — C++ interactive binary with KV cache
  inject/extract. Commands: `generate` (with optional `extract_cache`),
  `extract_cache`, `inject_cache`, `get_model_spec`, `shutdown`. KV cache
  serialized as safetensors via Edge-LLM's native `saveSafetensors()`.

- `AgentCacheStore` enforces `max_memory_mb` (evicts LRU hot caches when exceeded)
  and `max_disk_mb` (deletes oldest warm cache files when exceeded). Adds
  `hot_memory_bytes`, `disk_usage_bytes`, and `cache_location` properties.
- End-to-end Anthropic Messages API serving on Thor via TRT backend:
  `SEMANTIC_BACKEND=trt` + uvicorn + SmolLM2-135M = `"2 + 2 = 4"`.
- Lazy MLX imports in `api_server.py` — MLX modules only imported when
  `SEMANTIC_BACKEND=mlx`, enabling TRT-only operation on non-Apple platforms.

- `BlockPoolBatchEngine` moved from `application/` to `adapters/outbound/mlx_batch_engine.py`
  (hexagonal architecture: MLX-specific code in adapter layer). Backward-compatible
  re-export from old location.
- `TRTSettings` now includes `default_top_p`, `default_top_k`, `edgellm_plugin_path`,
  and `reasoning_extra_tokens` for full parity with MLX settings.
- `/v1/models` endpoint and `CoordinationService` now use backend-specific settings
  instead of always referencing `settings.mlx`.
- CLI `MLX_METAL_FAST_SYNCH` env var only set when `backend == "mlx"`.
- `GenerationRequest` common model in `application/generation_request.py` —
  both Anthropic and OpenAI adapters transform their protocol-specific requests
  into this unified model. Supports chat, FIM (fill-in-the-middle), stop sequences,
  repetition/frequency/presence penalties, and system prompt cache pinning.
- System prompt (`system` field) and tools now properly forwarded to TRT backend.
  Previously the TRT path dropped the system prompt entirely (critical bug).
- All Anthropic API fields survive to TRT: temperature, top_p, top_k, stop_sequences.
- README.md updated: dual-backend tagline, Jetson requirements, backend selection,
  Claude Code/NemoClaw client integration examples.
- docs/configuration.md: TRT settings table, backend selection, agent settings.
- docs/architecture/overview.md: dual-backend description.
- docs/deployment.md: Jetson Thor deployment path.
- Upgraded to mlx-lm 0.31+ / mlx 0.31+ / transformers 5.4+. Native Q4 KV
  cache, Qwen3.5 hybrid architecture (KVCache + ArraysCache/Mamba layers).
- MLX cache adapter handles both KVCache (4D: batch,heads,seq,dim) and
  ArraysCache (3D: batch,heads,state_dim) tensor layouts. Sequence length
  detection skips SSM state tensors. Slice operations only apply to KV layers.
- Batch engine uses `trim()` and `update_and_fetch()` APIs instead of direct
  `.keys`/`.values` attribute access. Cache eval via adapter method.
- Q4 monkeypatches conditionally skipped for mlx-lm >= 0.31 (native support).
- Dependencies unpinned: `mlx>=0.31.0`, `mlx-lm>=0.31.0`, `transformers>=5.4.0`.
- Verified: Qwen3.5-9B-MLX-4bit generates correctly on Apple Silicon.
- Configurable eviction policy: `SEMANTIC_AGENT_EVICTION_POLICY` (`lru`, `lfu`,
  `lru-lfu`). Default `lru-lfu` hybrid keeps both frequently-used system prompts
  (NemoClaw/Claude Code) and recently-used conversation caches warm.
- `pin_system_prompt_caches` setting auto-pins `sysprompt_*` cache entries
  (never evicted). Manual `pin()`/`unpin()` API for arbitrary entries.
- `CacheEntry.eviction_score(policy)` computes per-entry eviction priority.
  Hybrid scoring: `access_count / (1 + hours_since_last_access)`.
- Admin `/models/swap` on TRT backend: offloads all caches to SSD (Q4 safetensors),
  stops the subprocess, returns `"offloaded"` status. Old model caches preserved on
  disk tagged with original model_id for rollback. New engine requires server restart
  with updated `SEMANTIC_TRT_ENGINE_PATH` (TRT engines are pre-built, not loadable
  at runtime). This replaces the previous 501 response.
- MLX `/models/swap` now offloads all caches to SSD before swap sequence.
  Old model caches preserved on disk for rollback. Both backends use the same
  pattern: offload → stop → start fresh. Caches from old model are not reusable
  (different geometry/weights) but preserved tagged with original model_id.

- Configurable eviction policy with `SEMANTIC_AGENT_EVICTION_POLICY` setting.
- `TRTSafetensorsCacheAdapter` — numpy-based cache I/O, no MLX dependency.
  Each backend has its own cache adapter wired by `api_server.py`.
- `TRTInferenceService` — application service wrapping TRT backend with cache
  persistence. Handles load/inject/generate/extract/save transparently.
  Includes FIM prompt construction and model-agnostic special token stripping.
- SSE streaming for TRT + OpenAI streaming for TRT (word-level chunked).
- `extract_session_id()` common helper — supports both `X-Session-ID` and
  `X-Claude-Code-Session-Id` headers for Claude Code CLI compatibility.
- Pytest e2e test suite (`test_e2e_trt_server.py`) — 8 tests covering health,
  Anthropic API, streaming, multi-turn, OpenAI API.
- 83 new unit tests across 5 files: trt_model_loader, trt_prefill_adapter,
  trt_safetensors_cache_adapter, trt_inference_service, generation_request.
- `docs/mlx_example.md` — quick start guide for Qwen3.5-9B on Apple Silicon
  with Claude Code CLI and NemoClaw integration instructions.
- `config/models/qwen3.5-9b-4bit.toml` — model profile for Qwen3.5-9B.
- SBOM updated: numpy, TensorRT-Edge-LLM, nlohmann-json (17 components).
- CITATION.cff updated to 2.0.0-rc1 with nvidia-jetson keywords.
- Removed stale `semantic-server` package dependency (project unified as
  `agent-memory`).

### Fixed

- **Cache pipeline for mlx-lm 0.31**: Complete fix for hybrid cache architecture.
  KVCache (4D: batch,heads,seq,dim) and ArraysCache (3D: SSM state) handled
  correctly in extract, split, reconstruct, and slice operations. Warm cache
  reuse after server restart verified working (cache_read tokens reported).
- **System prompt dropped on TRT path**: Anthropic adapter now prepends `system`
  field and tools to the messages list before forwarding to TRT backend.
- **X-Claude-Code-Session-Id header not recognized**: Session ID extraction
  moved to common helper supporting both header names.
- **TRT streaming dropped sampling params**: `_stream_trt_response()` now
  forwards top_p, top_k, stop_sequences (were falling back to defaults).
- **OpenAI TRT streaming**: Added SSE chunk streaming for `/v1/chat/completions`.
- **Startup health probe**: Checks both `batch_engine` (MLX) and
  `trt_subprocess` (TRT). `/debug/memory` gracefully handles missing MLX.
- Batch engine `_reconstruct_cache`: Uses direct `.keys`/`.values` assignment
  for QuantizedKVCache (not `update_and_fetch` which expects float input).
- Batch engine `_extract_cache`: Safely handles ArraysCache layers in hybrid
  models without destructuring crash.
- Batch engine `_split_cache_to_blocks`: Finds first KVCache layer for
  sequence length, skips ArraysCache/SSM layers.

### Removed

- Q4 monkeypatches disabled for mlx-lm >= 0.31 (native Q4 KV cache support).
- `semantic-server` package dependency (project unified as `agent-memory`).
- Pinned dependency versions: now uses `>=` ranges for mlx, mlx-lm, transformers.

---

## [1.0.0] - 2026-02-10

- Admin API model-swap tests (`TestSwapModelEndpoint`) returned 422 instead of
  expected status codes. Root cause: `Mock` class passed directly as FastAPI
  dependency override — FastAPI introspected `Mock.__init__(**kwargs)` and tried
  to resolve `kwargs` as a query parameter. Fixed by wrapping in `lambda: Mock()`.
- Anthropic adapter `generate_agent_id_from_tokens()` used `str(prefix)` for
  hashing, which is not guaranteed deterministic across Python implementations.
  Switched to `json.dumps(prefix)` for stable serialization.
- `SafetensorsCacheAdapter` hardcoded `bits=4, group_size=64` at quantization
  fallback path. Now uses configurable `kv_bits`/`kv_group_size` from constructor,
  with `quantizer` parameter properly typed as `CacheQuantizationPort | None`.
- TRT adapters had cross-adapter imports (`trt_prefill_adapter`, `trt_spec_extractor`,
  `trt_system_prompt_cache` imported `TRTSubprocessAdapter` directly). Refactored to
  depend on `ModelBackendPort` protocol, preserving hexagonal architecture.
- `TRTSubprocessAdapter.stop()` leaked file handles (stdin/stdout/stderr not closed),
  causing `ResourceWarning` in tests. Now explicitly closes all subprocess pipes
  and supports pluggable `CacheQuantizationPort`.

---

## [1.0.0] - 2026-02-10

Initial open-source release.

### Features

- **Persistent KV cache** for multi-agent LLM inference on Apple Silicon
  - 3-tier cache hierarchy: hot (GPU memory), warm (metadata), cold (disk safetensors)
  - Q4-quantized KV cache persistence with ~3% perplexity impact
  - 8-31x faster time-to-first-token on cache hits (scales with context length)
- **OpenAI-compatible API** at `/v1/chat/completions` — works with LangChain, OpenAI SDK, curl
- **Concurrent batch inference** with interleaved chunked prefill and token-by-token decode
- **Block-pool memory management** with LRU eviction under memory pressure
- **Concurrent scheduler** supporting batch=2 with automatic prefill/decode interleaving
- **Multi-agent coordination** service with session management, whisper channels, and voting
- **Streamlit demos**: multi-agent chat, coordination, cache inspector, gossip scenario, prisoner's dilemma

### Supported Models

| Model | Size | Notes |
|-------|------|-------|
| Gemma 3 12B IT | Q4, ~6.5 GB | Hybrid attention: 8 global + 40 sliding window layers (default) |
| DeepSeek-Coder-V2-Lite | Q4, ~8 GB | MLA with asymmetric K=192/V=128 dims |

### Architecture

- Hexagonal architecture (ports and adapters) with domain-driven design
- MLX-native safetensors I/O (no numpy intermediary)
- Fused Q4 attention with GQA mask broadcast fix for batch>1
- Thread-safe design: single MLX inference thread + `mlx_io_lock` for cross-thread I/O

### Infrastructure

- Structured logging via structlog (JSON in production, console in development)
- Prometheus metrics endpoint (`/metrics`)
- 3-tier health probes (`/health/live`, `/health/ready`, `/health/startup`)
- Graceful shutdown with 6-stage cleanup (scheduler, drain, persist, engine, model, GPU memory)
- Admin API for cache management and model operations
- `scripts/setup.sh` for first-time setup and `scripts/launch.sh` for server + demo

### Testing

- 1,100+ unit tests, 170+ integration tests, 110+ GPU tests
- Ruff linting (bandit security rules, complexity checks, import sorting)
- mypy strict type checking
- All dependencies pinned to exact versions, all permissive licenses (MIT, BSD-3, Apache-2.0)

### Requirements

- Apple Silicon Mac (M1/M2/M3/M4)
- macOS 13+ (Ventura)
- Python 3.11+
- 16 GB RAM minimum (24 GB recommended for 12B models)

---

**Maintainer**: Yakov Shkolnikov and contributors
**License**: MIT
**Repository**: https://github.com/yshk-mxim/agent-memory
