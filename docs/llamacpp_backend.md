# llama.cpp Backend — Multi-Model on Thor

> **v1.1.0** — Managed multi-model with hot-swap.

Run Gemma 4 26B-A4B (fast MoE), Gemma 4 31B (deep dense), or Qwen3-Coder-Next
(coding specialist) on NVIDIA Jetson Thor through agent-memory. Models swap
automatically when you change the model name in your API request — one model
loaded at a time maximizes context window per model.

## Architecture

```
Claude Code CLI ──→ agent-memory (:8000) ──→ llama-server (:8001) ──→ Thor GPU
                         │                         │
                   Anthropic Messages API    OpenAI-compat API
                   Model swap orchestrator   GGUF Q4_K_M weights
                   Session management        Slot-level KV cache
                   Cache persistence         Native CUDA on sm_110
                   Tool call translation     MoE/Dense — all handled
                   Thinking tag stripping
```

### Managed Mode (recommended)

agent-memory manages the llama-server process lifecycle:

```
API request (model=gemma-4-31b)
  │
  ├── Model already loaded? → generate directly
  │
  └── Different model? → LlamaCppSwapOrchestrator:
        1. Save slot KV caches (HTTP /slots/{id}?action=save)
        2. Evict agent-memory caches to disk
        3. Stop llama-server (SIGTERM → SIGKILL)
        4. Start llama-server with new GGUF
        5. Update cache store model tag
        6. Generate response
```

Swap time: ~10-15 seconds (NVMe → GPU). KV caches survive swaps via disk.

### External Mode (advanced)

Start llama-server yourself, point agent-memory at it. No swap support.

## Why llama.cpp?

| Engine | MoE (Gemma 4) | Dense (Gemma 4) | DeltaNet (Qwen3.5) | sm_110 |
|--------|--------------|-----------------|-------------------|--------|
| **llama.cpp** | **Yes** | **Yes** | **Yes** | **Yes** |
| vLLM | Yes | Yes | Yes (fla) | Partial (rebuild, cuBLAS blocked) |
| Edge-LLM v0.6.0 | Qwen3 only | No Gemma | No | Yes |
| TensorRT-LLM | N/A | N/A | N/A | **Not supported** |

llama.cpp handles **every architecture** via GGUF — no precompiled CUDA
kernels, no architecture-specific gaps.

## Supported Models

| Model | Type | Active Params | GGUF Q4_K_M | Best For |
|-------|------|--------------|-------------|----------|
| **Gemma 4 26B-A4B** | MoE | 3.8B (of 26B) | ~14 GB | Fast interactive coding, research, triage |
| **Gemma 4 31B** | Dense | 31B | ~18 GB | Deep reasoning, architecture, security audit |
| **Qwen3-Coder-Next** | Hybrid | 3B (of 80B) | ~46 GB | Coding specialist (SWE-bench 70.6%) |

## Performance (Measured)

### Thor (Jetson AGX, 273 GB/s, sm_110)

| Model | Quantization | Prefill (tok/s) | Generate (tok/s) | Context | Slots |
|-------|-------------|-----------------|------------------|---------|-------|
| **Gemma 4 26B-A4B** | Q4_K_M + Q8 KV | **1,681** | **51** | 262K | 4 |
| **Gemma 4 31B** | Q4_K_M + Q8 KV | **361** | **10** | 131K | 2 |
| Qwen3-Coder-Next | Q4_K_M + Q8 KV | ~1,000 (est.) | ~15-20 (est.) | 131K | 2 |

### Cross-Platform Comparison

| Platform | Gemma 4 26B-A4B gen (tok/s) | Gemma 4 31B gen (tok/s) | Bandwidth |
|----------|---------------------------|------------------------|-----------|
| **Thor** (sm_110) | **51** | **10** | 273 GB/s |
| M5 Max (128 GB) | 81 | ~15 (est.) | 546 GB/s |
| M3 Ultra (192 GB) | ~60 (est.) | ~11 (est.) | 409 GB/s |
| RTX 5090 (32 GB) | 55-60 | N/A (32 GB) | 1792 GB/s |

> RTX 5090 bandwidth is 6.5x Thor but gen speed only ~1.1x for MoE —
> generation is bottlenecked by active parameter bandwidth (3.8B), not total.
> Dense 31B won't fit in 32 GB VRAM at Q4_K_M + Q8 KV cache.

### End-to-End Latency Analysis (Thor)

For Claude Code agentic workloads (system prompt reuse, ~30K prefill + ~500 gen):

| Model | First turn (30K pp) | Subsequent turns (cached pp) | 500 tok gen |
|-------|--------------------|-----------------------------|-------------|
| **Gemma 4 26B-A4B** | 17.8s pp + 9.8s gen = **27.6s** | ~0s (cached) + 9.8s = **9.8s** |
| **Gemma 4 31B** | 83s pp + 50s gen = **133s** | ~0s (cached) + 50s = **50s** |

The MoE model is **5x faster** end-to-end for interactive coding. Use Dense 31B
only when you need deeper reasoning (architecture reviews, security audits).

### Custom FP4 Kernel (tcgen05)

Thor has native FP4 tensor cores (tcgen05 MXFP4). A custom kernel was built and
benchmarked but is **not recommended** for production:

| Kernel | Gemma 4 26B-A4B pp (tok/s) | Gemma 4 26B-A4B gen (tok/s) |
|--------|---------------------------|----------------------------|
| Stock Q4_K_M (dequant) | **1,681** | **51** |
| Custom MXFP4 (tcgen05) | **2,200** (est. at scale) | **3.3** (broken pipelining) |

Custom FP4 wins on prefill-heavy workloads (>15:1 prefill:gen ratio) but the
pipelining issue makes it impractical for interactive use. Stock Q4_K_M is the
production choice.

## Memory Budget (Thor 128 GB, single model loaded)

| Model | Model Size | KV Cache (Q8) | Context | Slots | Free |
|-------|-----------|---------------|---------|-------|------|
| **Gemma 4 26B-A4B** | 16 GB | 102 GB | 262K total (65K/slot) | 4 | ~10 GB |
| **Gemma 4 31B** | 17 GB | 101 GB | 131K total (65K/slot) | 2 | ~10 GB |
| Qwen3-Coder-Next | 46 GB | 48 GB | 131K total (65K/slot) | 2 | ~34 GB |

> Single model loaded at a time. Full 128 GB available per model.
> MoE at 262K context = 4 concurrent Claude Code sessions at 65K each.

## 1. Build llama.cpp for Thor (sm_110)

**Minimum version: b8665** (April 4, 2026) — required for native Gemma 4 tool
calling (PR #21418), `final_logit_softcapping` (PR #21390), and tokenizer fixes.

```bash
# On Thor (main4.local)
git clone https://github.com/ggml-org/llama.cpp ~/llama.cpp-build
cd ~/llama.cpp-build
git checkout b8665    # or later
export PATH=$HOME/.local/bin:$PATH  # cmake from pip
cmake -B build \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="110" \
    -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j$(nproc) --target llama-server
```

> **Only sm_110.** Do not add other architectures — wastes compile time and
> the fat binary won't run faster. See `thor_compile_fix.md` for ptxas setup.

Verify: `./build/bin/llama-server --version`

## 2. Download Models

```bash
pip install huggingface-hub

# Gemma 4 26B-A4B (MoE, fast — 51 t/s gen, 262K context)
huggingface-cli download ggml-org/gemma-4-26B-A4B-it-GGUF \
    --include "*Q4_K_M*" \
    --local-dir ~/models/gemma4-26b-a4b

# Gemma 4 31B (Dense, deep — 10 t/s gen, 131K context)
huggingface-cli download ggml-org/gemma-4-31B-it-GGUF \
    --include "*Q4_K_M*" \
    --local-dir ~/models/gemma4-31b

# Qwen3-Coder-Next (SWE-bench 70.6%, 131K context)
huggingface-cli download unsloth/Qwen3-Coder-Next-GGUF \
    --include "*Q4_K_M*" \
    --local-dir ~/models/qwen3-coder-next
```

## 3. Configure Model Profiles

Each model has a TOML config in `config/models/`:

```toml
# config/models/gemma-4-26b-a4b.toml
[llamacpp]
gguf_path = "~/models/gemma4-26b-a4b/gemma-4-26B-A4B-it-Q4_K_M.gguf"
tokenizer_id = "google/gemma-4-26B-A4B-it"
ctx_size = 262144
n_slots = 4
n_gpu_layers = 99
extra_args = ["--jinja", "--reasoning", "off"]
# No chat_template_file — b8665 auto-detects Gemma 4 template from GGUF
```

The managed mode loader reads these profiles when swapping models.

## 4. Start (Managed Mode)

Use the helper script — it starts agent-memory which manages llama-server:

```bash
# Default: starts with MoE (fastest)
~/agent-memory/scripts/thor/start.sh

# Or start with a specific model
~/agent-memory/scripts/thor/start.sh gemma-4-31b
~/agent-memory/scripts/thor/start.sh qwen3-coder-next
```

This sets:
- `SEMANTIC_BACKEND=llamacpp`
- `SEMANTIC_LLAMACPP_SERVER_BINARY=~/llama.cpp-build/build/bin/llama-server`
- `SEMANTIC_LLAMACPP_DEFAULT_MODEL=gemma-4-26b-a4b`
- `SEMANTIC_LLAMACPP_CACHE_TYPE_K=q8_0`
- `SEMANTIC_LLAMACPP_CACHE_TYPE_V=q8_0`
- `SEMANTIC_LLAMACPP_AUTO_SWAP=true`

## 5. Swap Models

### Automatic (recommended)

Just change the model name in your API request:

```bash
# This auto-swaps to gemma-4-31b if gemma-4-26b-a4b was loaded
curl -s http://localhost:8000/v1/messages \
    -H "Content-Type: application/json" \
    -d '{
        "model": "gemma-4-31b",
        "max_tokens": 256,
        "messages": [{"role": "user", "content": "Review this architecture"}]
    }'
```

### Explicit

```bash
# Via admin API
curl -X POST http://localhost:8000/admin/models/swap \
    -H "X-Admin-Key: $SEMANTIC_ADMIN_KEY" \
    -H "Content-Type: application/json" \
    -d '{"model_id": "gemma-4-31b"}'

# Via script
~/agent-memory/scripts/thor/swap_model.sh gemma-4-31b
```

### Stop

```bash
~/agent-memory/scripts/thor/stop.sh
```

## 6. Connect Claude Code CLI

Add to `~/.claude/settings.json`:

```json
{
    "env": {
        "ANTHROPIC_BASE_URL": "http://localhost:8000",
        "ANTHROPIC_AUTH_TOKEN": "local",
        "ANTHROPIC_MODEL": "gemma-4-26b-a4b",
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

Switch models from Claude Code by changing `ANTHROPIC_MODEL` or by letting
the agent-memory auto-swap handle it when different model names appear in
API requests.

Headless mode:
```bash
ANTHROPIC_BASE_URL=http://localhost:8000 \
ANTHROPIC_AUTH_TOKEN=local \
MAX_THINKING_TOKENS=0 \
claude --bare -p "What files are in this directory?" \
    --output-format json \
    --max-turns 3
```

## 7. KV Cache Persistence

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
```

During model swaps, the orchestrator automatically saves all slot caches
before stopping the server. The `--cache-prompt` flag reuses KV cache when
requests share a prefix (e.g., system prompt) — critical for Claude Code's
agentic loop (10-40K system prompt repeated every turn).

## Configuration Reference

### Managed Mode (recommended)

| Variable | Default | Description |
|----------|---------|-------------|
| `SEMANTIC_BACKEND` | `mlx` | Set to `llamacpp` |
| `SEMANTIC_LLAMACPP_SERVER_BINARY` | `llama-server` | Path to llama-server binary |
| `SEMANTIC_LLAMACPP_DEFAULT_MODEL` | *(empty)* | Model ID to load on startup (enables managed mode) |
| `SEMANTIC_LLAMACPP_AUTO_SWAP` | `true` | Auto-swap when request model differs from loaded |
| `SEMANTIC_LLAMACPP_BASE_URL` | `http://localhost:8001` | llama-server URL (port for managed process) |
| `SEMANTIC_LLAMACPP_CACHE_TYPE_K` | `q8_0` | Key cache quantization |
| `SEMANTIC_LLAMACPP_CACHE_TYPE_V` | `q8_0` | Value cache quantization |
| `SEMANTIC_LLAMACPP_TIMEOUT_S` | `120.0` | HTTP timeout |

### External Mode

| Variable | Default | Description |
|----------|---------|-------------|
| `SEMANTIC_LLAMACPP_BASE_URL` | `http://localhost:8001` | llama-server URL |
| `SEMANTIC_LLAMACPP_MODEL_ID` | `qwen3-coder-next` | Model name for API requests |
| `SEMANTIC_LLAMACPP_TOKENIZER_ID` | *(same as MODEL_ID)* | HuggingFace tokenizer repo |
| `SEMANTIC_LLAMACPP_MAX_CONTEXT_LENGTH` | `65536` | Context window |
| `SEMANTIC_LLAMACPP_N_SLOTS` | `4` | Parallel slots |

## Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `Connection refused` | llama-server not running | Check start script logs |
| `{"status":"loading"}` | Model still loading | Wait for `{"status":"ok"}` |
| `slot save failed` | `--slot-save-path` not set | Managed mode sets this automatically |
| `OOM killed` | Model too large | Use smaller quantization or fewer slots |
| `no choices` | Context overflow | Check ctx_size in model TOML |
| Swap timeout | GGUF too large for 60s | Increase `timeout_seconds` in swap call |
| Slow first request | CUDA kernel compilation | Normal on first run; cached after |
| `ptxas fatal: sm_110a` | Triton ptxas too old | Set `TRITON_PTXAS_BLACKWELL_PATH=/usr/local/cuda/bin/ptxas` |

## Why Q8 KV Cache?

Q4 saves more memory but adds +0.2 perplexity and causes context-tracking
errors on long conversations. Q8 is near-lossless (+0.002–0.05 perplexity)
and still halves KV memory vs FP16. For coding workloads where precision
matters, Q8 is the right tradeoff.

## Prefill Chunking

llama.cpp handles prefill chunking internally — agent-memory does **not**
need to orchestrate it (unlike the MLX and TRT backends).

| Mechanism | How it works |
|-----------|-------------|
| **Chunked prefill** (`-b 2048`) | Long prompts processed in 2048-token batches |
| **Micro-batching** (`-ub 512`) | Each batch split into 512-token micro-batches |
| **Prefix caching** (`--cache-prompt`) | Shared system prompt KV reused across turns |
| **Slot persistence** (`--slot-save-path`) | KV cache saved to disk survives swaps/restarts |

Tune for Thor: `-b 4096 -ub 1024` (larger batches, more GPU utilization).

## Notes

- **No native FP4 compute:** llama.cpp uses Q4_K_M dequantization, not Thor's
  native FP4 tensor cores. Custom tcgen05 kernel exists but pipelining issues
  make stock Q4_K_M faster for interactive workloads.
- **Thinking mode:** Qwen3.5 models generate `<think>` tags by default.
  agent-memory strips these automatically. Use `MAX_THINKING_TOKENS=0`.
- **Tool calling:** Full Anthropic ↔ OpenAI tool format translation pipeline
  in agent-memory's anthropic adapter.
- **Gemma 4 thinking control:** The server runs with `--reasoning auto`, which
  enables thinking by default. To suppress thinking per-request, callers **must**
  pass `chat_template_kwargs: {"enable_thinking": false}` in the API payload.
  Without this flag, the model will generate reasoning tokens and consume the
  token budget on thinking. agent-memory's adapter handles this automatically
  via the `disable_thinking` parameter, but direct llama-server API callers
  must include it explicitly. Example:
  ```json
  {
    "messages": [...],
    "max_tokens": 256,
    "chat_template_kwargs": {"enable_thinking": false}
  }
  ```
