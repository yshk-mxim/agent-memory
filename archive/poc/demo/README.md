# Demo Guide

This directory contains demonstrations of the continuous batching system with persistent KV cache.

## Claude Code CLI Integration

The API server is compatible with [Claude Code CLI](https://github.com/anthropics/claude-code), enabling multiple Claude Code sessions to share GPU resources via continuous batching.

### Setup

1. **Start the API server with continuous batching:**

```bash
python -m src.api_server
```

The server will start on `http://localhost:8000` with:
- Max batch size: 5 concurrent agents
- Persistent KV cache for each unique system prompt
- Optional 8-bit KV cache quantization (50% memory reduction)

2. **Configure Claude Code to use local server:**

```bash
export ANTHROPIC_BASE_URL=http://localhost:8000
```

3. **Start multiple Claude Code sessions:**

**Terminal 1:**
```bash
ANTHROPIC_BASE_URL=http://localhost:8000 claude
```

**Terminal 2:**
```bash
ANTHROPIC_BASE_URL=http://localhost:8000 claude
```

**Terminal 3:**
```bash
ANTHROPIC_BASE_URL=http://localhost:8000 claude
```

Each session will:
- Get its own persistent agent (based on system prompt hash)
- Share GPU resources via continuous batching
- Maintain conversation context across requests
- Resume instantly from cached state (97.9% faster at 3500 tokens)

### How It Works

```
Claude Code Session A → API Server → Agent A (cached)
                             ↓
Claude Code Session B → BatchEngine → GPU (parallel)
                             ↓
Claude Code Session C → Agent C (cached)
```

**Per-Agent Sequential Semantics:**
- Multiple requests from the same Claude Code session execute sequentially
- Each request inherits the updated KV cache from the previous request
- Ensures conversation coherence and cache consistency

**Cross-Agent Parallel Batching:**
- Requests from different Claude Code sessions batch together on GPU
- 2-4× throughput improvement with 5 concurrent agents
- Shared model weights, isolated per-agent KV caches

### Advanced Configuration

**Enable KV cache quantization (8-bit, 50% memory reduction):**

```python
# In src/api_server.py
if __name__ == "__main__":
    import uvicorn

    # Configure server with quantization
    _server_instance = APIServer(
        model_name="mlx-community/gemma-3-12b-it-4bit",
        max_agents=5,
        max_batch_size=5,
        kv_bits=8,           # 8-bit quantization
        kv_group_size=64     # Quantization group size
    )

    uvicorn.run(app, host="0.0.0.0", port=8000)
```

**Adjust batch size:**

```python
_server_instance = APIServer(
    max_batch_size=10  # Support up to 10 concurrent agents
)
```

## Demos

### Full-Stack Demo

Run all integrated features:

```bash
python demo_full_stack.py
```

Demonstrates:
1. **API Server**: Anthropic Messages API with persistent cache
2. **Streaming**: SSE events for real-time responses
3. **Continuous Batching**: Multiple agents processed simultaneously
4. **Per-Agent Sequential**: Cache consistency per agent
5. **A2A Protocol**: Multi-agent delegation (if a2a_server running)
6. **Concurrent Processing**: Async queue utilization
7. **Session Resume**: Cache persistence across restarts
8. **KV Cache Quantization**: Optional 8-bit compression

### Batched Benchmark

Compare sequential vs batched performance:

```bash
python benchmarks/batched_benchmark.py
```

Expected results (M4 Pro, 24GB):
- **Sequential**: 5 agents × 50 tokens = ~8s total
- **Batched**: 5 agents × 50 tokens = ~2-3s total
- **Speedup**: 2.5-4× faster

## Performance Characteristics

### Cache Resume Speed

| Context Size | LM Studio | This System | Speedup |
|--------------|-----------|-------------|---------|
| 50 tokens    | 1.58s     | 1.1ms       | 1418×   |
| 3500 tokens  | 18.89s    | 0.40s       | **47×** |

### Batching Throughput

| Scenario | Sequential | Batched | Improvement |
|----------|-----------|---------|-------------|
| 5 agents × 50 tokens | ~8s | ~2-3s | 2.5-4× |
| Throughput (tokens/sec) | ~31 | ~83-125 | 2.5-4× |

### Memory Usage (M4 Pro, 24GB)

| Configuration | Per Agent (4K context) | Max Concurrent |
|---------------|------------------------|----------------|
| Float16       | ~130MB                 | ~5 agents      |
| 8-bit quantized | ~65MB                | ~10 agents     |

## Architecture

```
Claude Code CLI (ANTHROPIC_BASE_URL=http://localhost:8000)
        │
        ▼
┌────────────────────────────────────┐
│  APIServer (Anthropic Messages API) │
│  - System prompt → agent_id hash    │
│  - SSE streaming support            │
└───────────┬────────────────────────┘
            │ await manager.generate()
            ▼
┌────────────────────────────────────┐
│   ConcurrentAgentManager            │
│  - Per-agent asyncio.Lock           │
│  - Batch worker (10ms window)       │
└───────────┬────────────────────────┘
            │ submit(agent_id, prompt, cache)
            ▼
┌────────────────────────────────────┐
│   BatchedGenerationEngine           │
│  - Wraps mlx_lm.BatchGenerator      │
│  - Tracks uid → agent_id            │
│  - extract_cache(uid) after gen     │
└───────────┬────────────────────────┘
            │
      ┌─────┴─────┐
      ▼           ▼
┌──────────┐ ┌──────────────┐
│BatchKV   │ │BatchRotatingKV│ (Gemma 3: 8 global + 40 sliding)
│(8 layers)│ │(40 layers)    │
└──────────┘ └──────────────┘
      │
      ▼
┌────────────────────────────────────┐
│   CachePersistence (safetensors)    │
│  - extract_cache → save             │
│  - load → merge into batch          │
└────────────────────────────────────┘
```

## Novel Features

### 1. Persistent Batching
- Standard continuous batching: processes concurrent requests, discards KV cache
- **Our approach**: processes concurrent requests, **extracts and persists KV cache per agent**
- Agents resume from saved cache across server restarts (1ms cache load vs 18s re-processing)

### 2. Per-Agent Sequential / Cross-Agent Parallel
- **Per-agent**: `asyncio.Lock` per `agent_id` → sequential requests, cache inheritance
- **Cross-agent**: Different agents batch together on GPU → parallel processing
- **Result**: Cache consistency per agent + batching efficiency across agents

### 3. Composition Over Reimplementation
- **Did NOT**: Build custom paged attention from scratch
- **DID**: Wrapped mlx_lm's existing `BatchGenerator` infrastructure
- **Why**: `BatchKVCache`, `BatchRotatingKVCache`, and batch management already exist in mlx_lm

### 4. Anthropic Messages API Integration
- `ANTHROPIC_BASE_URL=http://localhost:8000` → Claude Code CLI uses local server
- System prompt hash → persistent `agent_id` → cache persistence
- Multiple Claude Code sessions share GPU via batching

## Troubleshooting

**Server won't start:**
- Check port 8000 is available: `lsof -i :8000`
- Kill existing process if needed: `kill -9 <PID>`

**Claude Code can't connect:**
- Verify `ANTHROPIC_BASE_URL` is set: `echo $ANTHROPIC_BASE_URL`
- Check server health: `curl http://localhost:8000/health`
- Review server logs for errors

**Low batching efficiency:**
- Increase batch size: Set `max_batch_size=10` in APIServer init
- Reduce batching window if latency is high: Adjust `await asyncio.sleep(0.01)` in concurrent_manager.py

**Memory issues:**
- Enable KV cache quantization: `kv_bits=8`
- Reduce max_agents: `max_agents=3`
- Monitor memory: `top -pid <server_pid>`

## References

- [Continuous Batching Plan](../plans/continuous_batching.md)
- [Novel Contributions](../novelty/continuous_batching.md)
- [mlx_lm BatchGenerator](https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/generate.py)
- [Claude Code CLI](https://github.com/anthropics/claude-code)
- [Anthropic Messages API](https://docs.anthropic.com/claude/reference/messages_post)
