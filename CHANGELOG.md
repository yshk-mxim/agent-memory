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

### Fixed

- Admin API model-swap tests (`TestSwapModelEndpoint`) returned 422 instead of
  expected status codes. Root cause: `Mock` class passed directly as FastAPI
  dependency override — FastAPI introspected `Mock.__init__(**kwargs)` and tried
  to resolve `kwargs` as a query parameter. Fixed by wrapping in `lambda: Mock()`.
- Anthropic adapter `generate_agent_id_from_tokens()` used `str(prefix)` for
  hashing, which is not guaranteed deterministic across Python implementations.
  Switched to `json.dumps(prefix)` for stable serialization.
- `SafetensorsCacheAdapter` hardcoded `bits=4, group_size=64` at quantization
  fallback path. Now uses configurable `kv_bits`/`kv_group_size` from constructor
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
