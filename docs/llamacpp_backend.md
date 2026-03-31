# llama.cpp Backend — Run Any Model on Thor

> **v1.1.0-alpha** — functional but not production-hardened.

Run Qwen3-Coder-Next (70.6% SWE-bench), Qwen3.5-27B-Opus-Distilled, or any
GGUF model on NVIDIA Jetson Thor through agent-memory with persistent KV cache.

## Architecture

```
Claude Code CLI ──→ agent-memory (:8000) ──→ llama-server (:8001) ──→ Thor GPU
                         │                         │
                   Anthropic Messages API    OpenAI-compat API
                   Session management        GGUF Q4_K_M weights
                   Cache persistence         Slot-level KV save/restore
                   Tool call translation     Native CUDA on sm_110
                   Thinking tag stripping    DeltaNet/MoE/MLA — all handled
```

agent-memory translates Anthropic Messages API (Claude Code) to OpenAI Chat
Completions API (llama-server), manages sessions, and handles cache persistence.
llama-server manages the actual model weights and KV cache in GPU memory.

## Why llama.cpp?

| Engine | DeltaNet (Qwen3.5) | MoE | MLA (DeepSeek) | sm_110 | FP4 native |
|--------|-------------------|-----|----------------|--------|-----------|
| **llama.cpp** | **Yes** | **Yes** | **Yes** | **Yes** | No (Q4 dequant) |
| vLLM | Yes (fla) | Yes | Yes | Partial (rebuild needed) | No (Marlin dequant) |
| Edge-LLM v0.6.0 | No | Qwen3 only | No | Yes | Yes |
| TensorRT-LLM | N/A | N/A | N/A | **Not supported on Thor** | N/A |

llama.cpp handles **every architecture** via GGUF — no precompiled CUDA kernels,
no architecture-specific gaps.  It just works.

## Supported Models

| Model | Active Params | SWE-bench | GGUF Q4_K_M Size | Tool Calling | Recommended |
|-------|--------------|-----------|-----------------|-------------|-------------|
| **Qwen3-Coder-Next** | 3B (of 80B MoE) | **70.6%** | ~46 GB | Yes | **Best for coding** |
| **Qwen3.5-27B-Opus-Distilled** | 27B (dense) | Good | ~16.5 GB | Yes (stable) | **Best for reasoning** |
| Qwen3.5-35B-A3B | 3B (of 35B MoE) | 76.4% (27B) | ~20 GB | Yes | Good all-round |
| Qwen2.5-Coder-32B | 32B (dense) | ~55% | ~20 GB | Yes | Stable fallback |
| Nemotron-3-Super-120B | 12B (of 120B) | 60.5% | ~80 GB | Yes | Too large |

## 1. Build llama.cpp for Thor (sm_110)

```bash
# On Thor (main4.local)
source ~/vllm-env/bin/activate  # or any env with cmake

git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
cmake -B build \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="110" \
    -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j$(nproc)
```

Verify: `./build/bin/llama-server --help | head -5`

See `thor_compile_fix.md` for Triton/ptxas environment setup if using
torch.compile alongside llama.cpp.

## 2. Download a Model

```bash
# Qwen3-Coder-Next (best coding model — 70.6% SWE-bench)
pip install huggingface-hub
huggingface-cli download unsloth/Qwen3-Coder-Next-GGUF \
    --include "*Q4_K_M*" \
    --local-dir ~/models/qwen3-coder-next

# OR: Qwen3.5-27B-Opus-Distilled (best reasoning + stable tool calling)
huggingface-cli download Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-GGUF \
    --include "*Q4_K_M*" \
    --local-dir ~/models/qwen35-opus-distilled
```

## 3. Start llama-server

```bash
mkdir -p ~/.agent_memory/llamacpp_slots

./build/bin/llama-server \
    -m ~/models/qwen3-coder-next/Qwen3-Coder-Next-Q4_K_M.gguf \
    --port 8001 \
    --host 0.0.0.0 \
    -ngl 999 \
    --ctx-size 65536 \
    -np 4 \
    --slot-save-path ~/.agent_memory/llamacpp_slots \
    --cache-type-k q4_0 \
    --cache-type-v q4_0 \
    --cache-prompt
```

| Flag | Purpose |
|------|---------|
| `-ngl 999` | Offload all layers to GPU |
| `--ctx-size 65536` | Context window (divided among slots) |
| `-np 4` | 4 parallel slots (4 concurrent sessions) |
| `--slot-save-path` | Enable KV cache save/restore to disk |
| `--cache-type-k q4_0` | Q4 quantized KV cache (saves memory) |
| `--cache-prompt` | Reuse KV cache for shared prompt prefixes |
| `-b 2048` | Prefill batch size (chunked prefill, default 2048) |
| `-ub 512` | Micro-batch size for prefill (default 512) |

Verify:
```bash
curl http://localhost:8001/health
# {"status":"ok"}
```

## 4. Start agent-memory

```bash
cd ~/agent-memory

SEMANTIC_BACKEND=llamacpp \
SEMANTIC_LLAMACPP_BASE_URL=http://localhost:8001 \
SEMANTIC_LLAMACPP_MODEL_ID=unsloth/Qwen3-Coder-Next-GGUF \
SEMANTIC_LLAMACPP_MAX_CONTEXT_LENGTH=65536 \
SEMANTIC_LLAMACPP_N_SLOTS=4 \
python -m uvicorn agent_memory.entrypoints.api_server:create_app \
    --factory --host 0.0.0.0 --port 8000
```

Verify:
```bash
curl http://localhost:8000/health/live
# {"status":"alive"}

curl -s http://localhost:8000/v1/messages \
    -H "Content-Type: application/json" \
    -d '{
        "model": "qwen3-coder-next",
        "max_tokens": 32,
        "messages": [{"role": "user", "content": "What is 2+2?"}]
    }' | python3 -m json.tool
```

## 5. Connect Claude Code CLI

Add to `~/.claude/settings.json`:

```json
{
    "env": {
        "ANTHROPIC_BASE_URL": "http://localhost:8000",
        "ANTHROPIC_AUTH_TOKEN": "local",
        "ANTHROPIC_MODEL": "qwen3-coder-next",
        "CLAUDE_CODE_ATTRIBUTION_HEADER": "0",
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        "MAX_THINKING_TOKENS": "0"
    },
    "permissions": {
        "allow": [
            "Bash(npm*)", "Bash(node*)", "Bash(python*)", "Bash(pip*)",
            "Bash(git*)", "Bash(ls*)", "Bash(cat*)", "Bash(find*)",
            "Bash(grep*)", "Bash(rg*)", "Bash(mkdir*)", "Bash(touch*)",
            "Read", "Write", "Edit", "Glob", "Grep"
        ]
    }
}
```

**`CLAUDE_CODE_ATTRIBUTION_HEADER=0`** prevents a header that invalidates
the KV cache with local models — critical for performance.

Headless mode:
```bash
ANTHROPIC_BASE_URL=http://localhost:8000 \
ANTHROPIC_AUTH_TOKEN=local \
MAX_THINKING_TOKENS=0 \
claude --bare -p "What files are in this directory?" \
    --output-format json \
    --max-turns 3
```

## 6. KV Cache Persistence

llama-server provides slot-level KV cache save/restore:

```bash
# Save slot 0's KV cache to disk
curl -X POST "http://localhost:8001/slots/0?action=save" \
    -H "Content-Type: application/json" \
    -d '{"filename": "session-abc.bin"}'

# Restore into slot 1
curl -X POST "http://localhost:8001/slots/1?action=restore" \
    -H "Content-Type: application/json" \
    -d '{"filename": "session-abc.bin"}'

# Clear a slot
curl -X POST "http://localhost:8001/slots/0?action=erase"
```

agent-memory's `LlamaCppBackendAdapter` maps sessions to slots using
`hash(session_id) % n_slots`.  The `--cache-prompt` flag automatically
reuses KV cache when requests share a prefix (e.g., system prompt).

Cache files are stored in `--slot-save-path` (default:
`~/.agent_memory/llamacpp_slots/`).  They survive server restarts.

## Configuration Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `SEMANTIC_BACKEND` | `mlx` | Set to `llamacpp` |
| `SEMANTIC_LLAMACPP_BASE_URL` | `http://localhost:8001` | llama-server URL |
| `SEMANTIC_LLAMACPP_MODEL_ID` | `qwen3-coder-next` | HuggingFace model ID (for tokenizer) |
| `SEMANTIC_LLAMACPP_TIMEOUT_S` | `120.0` | HTTP timeout |
| `SEMANTIC_LLAMACPP_MAX_CONTEXT_LENGTH` | `65536` | Context window |
| `SEMANTIC_LLAMACPP_SLOT_SAVE_PATH` | `~/.agent_memory/llamacpp_slots` | Slot cache directory |
| `SEMANTIC_LLAMACPP_N_SLOTS` | `4` | Parallel slots (mirrors `-np`) |
| `SEMANTIC_LLAMACPP_CACHE_TYPE_K` | `q4_0` | Key cache quantization |
| `SEMANTIC_LLAMACPP_CACHE_TYPE_V` | `q4_0` | Value cache quantization |

## Memory Budget (Thor 128 GB)

| Setup | Model | KV Cache | OS/System | Free |
|-------|-------|----------|-----------|------|
| Qwen3-Coder-Next Q4_K_M | ~46 GB | ~12 GB (64K ctx, Q4 KV) | ~10 GB | **~60 GB** |
| Qwen3.5-27B-Opus Q4_K_M | ~16.5 GB | ~12 GB | ~10 GB | **~89 GB** |
| Qwen3.5-35B-A3B Q4_K_M | ~20 GB | ~12 GB | ~10 GB | **~86 GB** |

## Performance Expectations (Thor)

Based on llama.cpp benchmarks for similar MoE models on Thor:

| Model | Quantization | Prefill (tok/s) | Generate (tok/s) |
|-------|-------------|-----------------|-----------------|
| Qwen3-30B-A3B | Q8_0 | ~1,533 | ~42.7 |
| Qwen3-Coder-Next (80B/3B active) | Q4_K_M | ~1,000 (est.) | ~15-20 (est.) |
| Qwen3.5-27B-Opus (dense) | Q4_K_M | ~500 (est.) | ~25-35 (est.) |

## Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `Connection refused` | llama-server not running | Start llama-server first |
| `{"status":"loading"}` | Model still loading | Wait for `{"status":"ok"}` |
| `slot save failed` | `--slot-save-path` not set | Add flag to llama-server command |
| `OOM killed` | Model too large for Thor | Use smaller quantization or model |
| `no choices` | Context overflow or bad prompt | Check `--ctx-size` vs prompt length |
| Slow first request | Triton/CUDA kernel compilation | Normal on first run; cached after |
| `ptxas fatal: sm_110a` | Triton ptxas too old | Set `TRITON_PTXAS_BLACKWELL_PATH=/usr/local/cuda/bin/ptxas` |

## Prefill Chunking

llama.cpp handles prefill chunking internally — agent-memory does **not** need
to orchestrate it (unlike the MLX and TRT backends).

| Mechanism | How it works |
|-----------|-------------|
| **Chunked prefill** (`-b 2048`) | Long prompts are processed in batches of 2048 tokens, preventing OOM |
| **Micro-batching** (`-ub 512`) | Each batch is further split into 512-token micro-batches for GPU efficiency |
| **Prefix caching** (`--cache-prompt`) | When the same system prompt is sent across turns, llama.cpp skips re-processing the matching prefix — equivalent to agent-memory's system prompt cache |
| **Slot persistence** (`--slot-save-path`) | KV cache saved to disk survives server restarts — equivalent to agent-memory's warm cache tier |

For Claude Code's agentic loop (10-40K system prompt repeated every turn),
`--cache-prompt` is critical — it avoids re-computing the system prompt KV
state on every request. The first request is slow (full prefill), subsequent
requests in the same slot skip the shared prefix entirely.

Tune prefill batch size based on available memory:
- **Thor (128 GB):** `-b 4096 -ub 1024` (larger batches, more GPU utilization)
- **16 GB Mac:** `-b 1024 -ub 256` (smaller to avoid OOM)

## Notes

- **No native FP4 compute:** llama.cpp uses Q4_K_M dequantization, not Thor's
  native FP4 tensor cores.  When TensorRT Edge-LLM adds DeltaNet support,
  native FP4 will deliver ~2-4x higher throughput.
- **Thinking mode:** Qwen3.5 models generate `<think>` tags by default.
  agent-memory strips these automatically.  Use `MAX_THINKING_TOKENS=0` in
  Claude Code settings to minimize wasted tokens.
- **Tool calling:** Qwen3-Coder-Next and Qwen3.5 models support native tool
  calling via `<tool_call>` tags.  agent-memory's anthropic adapter handles
  Anthropic ↔ Qwen tool format translation.
