# vLLM + Qwen3-Coder-Next-NVFP4 on Thor

> Replacement for llama.cpp when prefill speed matters.
> Uses FP4 weights on Blackwell tensor cores + flash-linear-attention
> for the DeltaNet layers + chunked prefill + prefix caching.

## Architecture

```
Mac (Claude Code CLI)
  │
  │  Anthropic Messages API (HTTP, private LAN)
  ▼
Thor — agent-memory :8000
  ├── /v1/messages ────────► vLLM :8001 ──► GPU (Qwen3-Coder-Next-NVFP4)
  ├── /search?q=... ───────► SearXNG :8080
  └── (all local, no cloud)

vLLM internals:
  ├── Triton/FLA GDN prefill kernel (DeltaNet linear attention layers)
  ├── FLASHINFER_CUTLASS NVFP4 GEMM (FP4 Blackwell tensor cores)
  ├── Flash Attention v2 (full attention layers, 1-in-4 layer pattern)
  ├── Prefix caching (system prompt cached across turns)
  └── Chunked prefill (batched prefill for long contexts)
```

## Why vLLM over llama.cpp

| | llama.cpp Q4_K_M | vLLM NVFP4 |
|---|---|---|
| Prefill (23K sys prompt) | ~60s cold | ~15s cold |
| FP4 tensor cores | No | Yes (Blackwell SM110) |
| Continuous batching | No | Yes |
| Parallel agents | Sequential prefill | Batched prefill |
| KV cache on restart | Persists to disk | Lost (in-memory only) |
| Prefix caching | Per-slot (cache_prompt) | APC (system prompt shared) |
| Architecture support | Universal (GGUF) | Explicit per-model |

## Prerequisites

- vLLM 0.18+ built for Thor (see [`thor_vllm_build.md`](thor_vllm_build.md))
- `~/vllm-env` with working CUDA 13.0 + torch
- 68+ GB free GPU memory (model: 47.6 GB + KV cache + overhead)

## Model

```
RedHatAI/Qwen3-Coder-Next-NVFP4
```

- 80B total / 3B active params (MoE, 512 experts, 10 active)
- Hybrid architecture: 3 linear attention + 1 full attention, repeating
- NVFP4 quantized (compressed-tensors format), 47.6 GB
- SWE-bench Verified: 52% (105% recovery vs base model's 49.3%)
- Apache 2.0 license

First download: ~47.6 GB from HuggingFace (cached after first run).

## Start vLLM

```bash
# Stop llama-server if running
pkill -f llama-server

# Start vLLM
bash ~/start_vllm_coder.sh
```

Or use the switcher:
```bash
bash ~/switch_backend.sh vllm
```

Wait for `Uvicorn running on http://0.0.0.0:8001` in `~/vllm-coder-serve.log`.

## Start agent-memory

```bash
bash ~/start_agent_memory_vllm.sh
```

Verify:
```bash
# From Mac
python3 -c "import urllib.request; print(urllib.request.urlopen('http://192.168.184.150:8000/health').read().decode())"
```

## Revert to llama.cpp

```bash
bash ~/switch_backend.sh llamacpp
```

This kills vLLM, starts llama-server + agent-memory with the llama.cpp config.
All four scripts are independent and idempotent:

| Script | Purpose |
|--------|---------|
| `~/start_vllm_coder.sh` | Start vLLM on :8001 |
| `~/start_llamacpp.sh` | Start llama-server on :8001 |
| `~/start_agent_memory_vllm.sh` | Start agent-memory (vLLM backend) on :8000 |
| `~/start_agent_memory.sh` | Start agent-memory (llamacpp backend) on :8000 |
| `~/switch_backend.sh {vllm\|llamacpp}` | Full stack switch |

## vLLM Flags Explained

```
--model RedHatAI/Qwen3-Coder-Next-NVFP4    # NVFP4 quantized weights
--gpu-memory-utilization 0.50               # ~61 GB for model + KV cache
--max-model-len 65536                       # 64K context (tunable)
--max-num-seqs 4                            # Max concurrent requests
--enable-chunked-prefill                    # Break long prefills into batches
--enable-prefix-caching                     # APC — cache common prefixes
--enforce-eager                             # No torch.compile (faster startup)
--tool-call-parser qwen3_coder              # Native tool calling support
--kv-cache-dtype auto                       # Let vLLM choose optimal KV dtype
--trust-remote-code                         # Required for qwen3_next architecture
```

## Cache Management

vLLM's KV cache is in GPU memory only — no disk persistence.

- **Prefix cache**: System prompt cached in GPU memory across turns (lost on restart)
- **Warmup**: First request after startup prefills the system prompt (~15s)
- **Clear**: Restart vLLM (`pkill -f vllm && bash ~/start_vllm_coder.sh`)

agent-memory's disk cache (`~/.agent_memory/caches/`) is still available but not used
by the vLLM backend — vLLM manages its own KV cache internally.

## Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `ValueError: Free memory ... less than desired` | llama-server still running | `pkill -f llama-server` first |
| `OOM` during inference | Context too long for available memory | Reduce `--max-model-len` or `--gpu-memory-utilization` |
| `Engine core initialization failed` | Various — check full log | `tail -50 ~/vllm-coder-serve.log` |
| Slow first request | Prefix cache cold | Expected — ~15s for 23K system prompt, fast after |
| Model downloading | First run of NVFP4 model | Wait — 47.6 GB download, cached after |

## See Also

- [`llamacpp_thor_example.md`](llamacpp_thor_example.md) — llama.cpp setup (fallback)
- [`privacy_enhanced_example.md`](privacy_enhanced_example.md) — full private stack overview
- [`thor_vllm_build.md`](thor_vllm_build.md) — building vLLM from source on Thor
