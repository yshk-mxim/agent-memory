# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

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

### Fixed

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
