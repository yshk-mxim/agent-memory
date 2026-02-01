# Architecture

Semantic Cache API is a local LLM inference server for Apple Silicon that persists per-agent KV caches to disk and reuses them across turns, eliminating redundant prefill computation.

## The Problem

Local LLM tools (LM Studio, Ollama, llama.cpp) waste 40-90% of compute re-processing unchanged context on every request. On Apple Silicon, where prefill is compute-bound (not memory-bound like datacenter GPUs), this overhead is severe: a 2,000-token prompt takes ~1.6 seconds to prefill, and a 32,000-token prompt takes ~48 seconds.

Multi-agent workflows multiply this cost: each agent re-prefills its entire conversation history on every turn.

## Solution

Persistent per-agent KV caches stored in quantized (Q4) format, with character-level prefix matching to maximize cache reuse and adaptive chunked prefill for memory-efficient long contexts.

## System Architecture

```
                          Hexagonal Architecture
  ┌─────────────────────────────────────────────────────────────┐
  │                     INBOUND ADAPTERS                        │
  │  ┌──────────────┐  ┌───────────────┐  ┌──────────────────┐ │
  │  │ OpenAI API   │  │ Anthropic API │  │ Direct Agent API │ │
  │  │ /v1/chat/*   │  │ /v1/messages  │  │ /v1/agents/*     │ │
  │  └──────┬───────┘  └───────┬───────┘  └────────┬─────────┘ │
  │         │                  │                    │           │
  │  ┌──────▼──────────────────▼────────────────────▼─────────┐ │
  │  │              APPLICATION LAYER                          │ │
  │  │  ┌─────────────────┐  ┌────────────────────────┐       │ │
  │  │  │  BatchEngine    │  │  AgentCacheStore       │       │ │
  │  │  │  (orchestrate)  │  │  (LRU, persist, load)  │       │ │
  │  │  └────────┬────────┘  └────────────┬───────────┘       │ │
  │  │           │                        │                    │ │
  │  │  ┌────────▼────────────────────────▼───────────┐       │ │
  │  │  │             DOMAIN CORE                      │       │ │
  │  │  │  BlockPool · KVBlock · AgentBlocks           │       │ │
  │  │  │  (pure Python, no external dependencies)     │       │ │
  │  │  └─────────────────────────────────────────────┘       │ │
  │  └────────────────────────────────────────────────────────┘ │
  │         │                  │                    │           │
  │  ┌──────▼───────┐  ┌──────▼───────┐  ┌────────▼─────────┐ │
  │  │ MLX Backend  │  │ safetensors  │  │ SharedPrefix     │ │
  │  │ (inference)  │  │ (disk I/O)   │  │ (prefix cache)   │ │
  │  └──────────────┘  └──────────────┘  └──────────────────┘ │
  │                     OUTBOUND ADAPTERS                       │
  └─────────────────────────────────────────────────────────────┘
```

**Dependency rule**: All dependencies point inward. Domain has zero external imports.

## Request Flow

1. **API adapter** receives chat completion request with `X-Session-ID` header
2. **BatchEngine** looks up agent's cached blocks in `AgentCacheStore`
3. **Character-level prefix matching** compares stored prompt text vs new prompt
4. Three paths:
   - **EXACT match**: Reuse prompt cache, generate new tokens only
   - **EXTEND**: Previous prompt is a prefix of new prompt — only process new tokens
   - **MISMATCH**: Cache invalidated, full re-prefill
5. **Q4 cache reconstruction**: Load quantized blocks from disk, inject directly into MLX without dequantization
6. **Adaptive chunked prefill**: Process new tokens in memory-safe chunks
7. **Decode**: Generate output tokens with full cache context
8. **Persist**: Save updated cache blocks to disk (safetensors)

## Five Co-Designed Techniques

Each technique is documented in detail in the `novelty/` directory.

### 1. Q4 End-to-End Cache Pipeline

Keeps KV cache in 4-bit quantized format through the entire lifecycle: storage, disk persistence, reconstruction, and injection into MLX. Eliminates the dequantize-requantize round trip that would consume 4x memory.

- 75% memory reduction vs FP16
- Direct injection via `QuantizedKVCache.merge()`
- See: [`novelty/q4_direct_injection.md`](novelty/q4_direct_injection.md)

### 2. Adaptive Chunked Prefill

Splits long prefill into memory-safe chunks that adapt to available GPU memory. Achieves ~80% of FlashAttention benefits without custom Metal kernels, extending practical context from ~20K to 80K+ tokens on 24GB systems.

- Dynamic chunk sizing based on available memory
- Progressive sequence extension
- See: [`novelty/adaptive_chunked_prefill.md`](novelty/adaptive_chunked_prefill.md)

### 3. Character-Level Prefix Matching

Replaces token-level prefix matching with character-level text comparison, fixing BPE tokenization boundary mismatches. Only the genuinely new portion of a conversation is tokenized and processed.

- Eliminates false cache misses from tokenizer non-compositionality
- Works across EXACT, EXTEND, and MISMATCH paths
- See: [`novelty/character_level_prefix_matching.md`](novelty/character_level_prefix_matching.md)

### 4. Continuous Batching with Persistent Caches

Combines continuous batching (interleaved prefill/decode) with persistent per-agent KV caches. Each agent maintains isolated cache state while sharing GPU time.

- Per-agent session isolation via `X-Session-ID`
- LRU eviction to disk when memory pressure exceeds threshold
- See: [`novelty/continuous_batching.md`](novelty/continuous_batching.md)

### 5. MLX/UMA Memory Discipline

Exploits Apple Silicon's Unified Memory Architecture for zero-copy cache operations. Three-step memory reclamation (reference clearing, gc.collect, lazy MLX evaluation) prevents memory fragmentation.

- Zero-copy disk-to-GPU via memory-mapped safetensors
- Lazy tensor evaluation batched across layers (27 fewer GPU sync fences)
- See: [`novelty/mlx_uma_system_novelty.md`](novelty/mlx_uma_system_novelty.md)

## Module Map

| Module | Layer | Purpose |
|--------|-------|---------|
| `domain/entities.py` | Domain | KVBlock, AgentBlocks data structures |
| `domain/services.py` | Domain | BlockPool allocation/deallocation |
| `domain/value_objects.py` | Domain | ModelCacheSpec (model parameters) |
| `application/batch_engine.py` | Application | Core orchestration: cache lookup, prefill, decode |
| `application/agent_cache_store.py` | Application | Agent lifecycle, LRU eviction, disk persistence |
| `application/scheduler.py` | Application | Concurrent request scheduling |
| `application/shared_prefix_cache.py` | Application | Cross-agent system prompt sharing |
| `adapters/inbound/openai_adapter.py` | Adapter | OpenAI API compatibility layer |
| `adapters/inbound/anthropic_adapter.py` | Adapter | Anthropic Messages API compatibility |
| `adapters/inbound/direct_agent_adapter.py` | Adapter | Low-level agent CRUD API |
| `adapters/outbound/mlx_cache_adapter.py` | Adapter | Q4 cache extraction and reconstruction |
| `adapters/outbound/mlx_model_loader.py` | Adapter | Model loading and spec extraction |
| `adapters/outbound/mlx_prefill_adapter.py` | Adapter | Adaptive chunked prefill |
| `adapters/outbound/safetensors_cache_adapter.py` | Adapter | Disk persistence (safetensors) |
| `entrypoints/api_server.py` | Entrypoint | FastAPI server wiring |
| `entrypoints/cli.py` | Entrypoint | CLI (`semantic serve`, `semantic bench`) |

## Performance Summary

| Metric | Value |
|--------|-------|
| Multi-turn speedup (Turn 1 to Turn 2) | 2.0x |
| Prefix sharing speedup | 1.6x |
| Max practical context | 50K+ tokens |
| Q4 memory savings | 75% vs FP16 |
| Disk persistence | Yes (survives restart) |
| Multi-agent isolation | Per-session KV caches |

Full results: [`benchmarks/BENCHMARK_RESULTS.md`](benchmarks/BENCHMARK_RESULTS.md)

## Comparison with Existing Tools

As of January 2026, no major local LLM tool (LM Studio, Ollama, llama.cpp) provides native multi-agent KV cache persistence with disk-backed storage and Q4 quantized format.

Full comparison: [`novelty/EXISTING_TOOLS_COMPARISON.md`](novelty/EXISTING_TOOLS_COMPARISON.md)
