# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Yakov Shkolnikov and contributors
"""agent-memory: Persistent KV cache for multi-agent LLM systems on Apple Silicon.

Block-pool memory management with hexagonal architecture.

This package provides a production-quality multi-agent LLM inference server
with persistent KV cache, continuous batching, and block-pool memory management
for Apple Silicon (MLX).

Architecture: Hexagonal (Ports & Adapters)
- Domain core: Pure business logic (no external dependencies)
- Ports: Protocol-based interfaces
- Adapters: Infrastructure bindings (MLX, FastAPI, safetensors)

For details, see docs/architecture/hexagonal.md.
"""

__version__ = "1.0.0"

__all__ = ["__version__"]
