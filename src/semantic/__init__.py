"""Semantic: Production-quality multi-agent LLM inference server.

Block-pool memory management with hexagonal architecture.

This package provides a production-quality multi-agent LLM inference server
with persistent KV cache, continuous batching, and block-pool memory management
for Apple Silicon (MLX).

Architecture: Hexagonal (Ports & Adapters)
- Domain core: Pure business logic (no external dependencies)
- Ports: Protocol-based interfaces
- Adapters: Infrastructure bindings (MLX, FastAPI, safetensors)

For details, see project/architecture/ADR-001-hexagonal-architecture.md.
"""

__version__ = "0.1.0-alpha"

__all__ = ["__version__"]
