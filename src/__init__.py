"""
Persistent Multi-Agent Memory System

A demonstration of persistent agent memory using KV cache
persistence on Mac with unified memory architecture.

Fills gap that LM Studio, Ollama, and llama.cpp don't provide:
- Persistent KV cache across sessions
- Multi-agent context isolation
- Cross-session agent memory
"""

__version__ = "0.1.0"
__author__ = "dev_user"

from .mlx_utils import MLXModelLoader

__all__ = ['MLXModelLoader']
