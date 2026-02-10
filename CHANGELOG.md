# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

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
