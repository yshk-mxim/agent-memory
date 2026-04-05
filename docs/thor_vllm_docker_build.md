# Building vLLM for Thor (SM110) — Docker Container

> Complete, reproducible guide for running vLLM with Qwen3-Coder-Next-NVFP4
> on NVIDIA Jetson AGX Thor. Builds a single Docker image with all patches
> baked in.

## Why Docker?

Thor ships CUDA 13.0 but only the `sbsa-linux` (server-class Arm) cuBLAS
libraries. PyTorch pip wheels (2.10+) call cuBLAS in a way that is
**incompatible with SM110** — every `torch.mm()` fails with
`CUBLAS_STATUS_INVALID_VALUE`. NVIDIA's own PyTorch fork (2.8.x) in their
NGC container calls cuBLAS correctly. The container is therefore mandatory,
not optional.

## Prerequisites

| Component | Value |
|-----------|-------|
| Platform | Jetson AGX Thor (aarch64, sm_110) |
| JetPack | 7.1 |
| CUDA | 13.0 (driver 580.00) |
| Docker | With `--runtime=nvidia` support |
| RAM | 128 GB unified (GPU/CPU shared) |
| Disk | ~100 GB free (images + model + caches) |
| SSH | Assumes `<user>@<thor-hostname>` |

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

Docker image: vllm-thor (based on nvcr.io/nvidia/pytorch:25.08-py3)
  ├── PyTorch 2.8 (NVIDIA fork, working cuBLAS for SM110)
  ├── vLLM v0.18.1 (source, C extensions built for SM110 only)
  ├── FlashInfer 0.6.7 (CUTLASS MoE kernels JIT-compiled on first run)
  ├── torch.accelerator shim (backports torch 2.10 API to 2.8)
  └── Fake vLLM dist-info (source install has no pip metadata)
```

## Overview of Issues Solved

| Issue | Root Cause | Fix |
|-------|-----------|-----|
| cuBLAS `INVALID_VALUE` on all GEMMs | torch 2.10 pip wheel's cuBLAS calling code incompatible with SM110 | Use NVIDIA container's torch 2.8 |
| `torch.accelerator.empty_cache` missing | torch 2.8 lacks functions vLLM 0.18 expects | Append shim functions delegating to `torch.cuda.*` |
| `No package metadata for vllm` | Source install via PYTHONPATH has no pip metadata | Create fake dist-info directory |
| `_C.abi3.so: undefined symbol` | C extensions compiled against wrong torch ABI | Rebuild inside container against torch 2.8 |
| Triton allocator RuntimeError | FLA DeltaNet kernels need `triton.set_allocator()` | Patch `qwen3_next.py` to call `set_triton_allocator()` |
| FlashInfer MoE build OOM | 16 parallel nvcc processes + 44 GB model | Set `MAX_JOBS=2` |
| FlashInfer build timeout | 266 CUTLASS kernels take ~2h on ARM | Set `VLLM_ENGINE_ITERATION_TIMEOUT_S=7200` |
| FlashInfer cache lost on restart | Cache was inside ephemeral container | Mount `~/.cache/flashinfer` to host |
| vLLM source lost on restart | `/tmp/vllm-build` cleaned by OS | Use `~/vllm-build` (persistent) |

## Step 1: Pull the Base Image

```bash
docker pull nvcr.io/nvidia/pytorch:25.08-py3
```

This is ~15 GB. Contains PyTorch 2.8.0a0+34c6371 with **working cuBLAS
for SM110**.

## Step 2: Build the vLLM Docker Image

Create `~/Dockerfile.vllm-thor`:

```dockerfile
FROM nvcr.io/nvidia/pytorch:25.08-py3

# ── Step 2a: Install vLLM Python dependencies (without replacing torch) ──
# Two-pass install: first try --no-deps for packages that pull torch,
# then install remaining deps that are safe.
RUN pip install --no-deps \
    uvloop uvicorn fastapi starlette sse-starlette msgspec \
    compressed-tensors openai anthropic tiktoken tokenizers transformers \
    huggingface-hub hf-xet pydantic pydantic-settings \
    prometheus-fastapi-instrumentator py-cpuinfo outlines_core interegular \
    lm-format-enforcer partial-json-parser depyf xgrammar blake3 gguf \
    flashinfer-python opentelemetry-api opentelemetry-sdk \
    opentelemetry-exporter-otlp opentelemetry-exporter-otlp-proto-common \
    opentelemetry-exporter-otlp-proto-grpc opentelemetry-exporter-otlp-proto-http \
    opentelemetry-proto opentelemetry-semantic-conventions \
    mistral_common sentencepiece quack-kernels nvidia-cutlass-dsl \
    nvidia-cutlass-dsl-libs-base watchfiles httptools python-multipart \
    setproctitle diskcache cbor2 pybase64 torch-c-dlpack-ext cuda-tile \
    fastar loguru distro docstring-parser pycountry annotated-doc jiter \
    astor apache-tvm-ffi typing-inspection pydantic-extra-types \
    python-dotenv httpx-sse 2>/dev/null; \
    pip install \
    uvloop uvicorn fastapi starlette sse-starlette msgspec \
    compressed-tensors openai anthropic tiktoken tokenizers transformers \
    huggingface-hub hf-xet pydantic pydantic-settings \
    prometheus-fastapi-instrumentator py-cpuinfo outlines_core interegular \
    lm-format-enforcer partial-json-parser depyf xgrammar blake3 gguf \
    flashinfer-python opentelemetry-api opentelemetry-sdk \
    opentelemetry-exporter-otlp mistral_common sentencepiece quack-kernels \
    nvidia-cutlass-dsl nvidia-cutlass-dsl-libs-base watchfiles httptools \
    python-multipart setproctitle diskcache cbor2 pybase64 cuda-tile \
    loguru distro jiter pycountry annotated-doc typing-inspection \
    pydantic-extra-types python-dotenv httpx-sse 2>/dev/null || true

# Additional deps that may have been missed
RUN pip install openai-harmony llguidance rignore \
    model-hosting-container-standards sentry-sdk setuptools-scm \
    2>/dev/null || true

# ── Step 2b: Patch torch.accelerator for torch 2.8 compatibility ──
# vLLM 0.18 calls torch.accelerator.empty_cache() etc. which don't exist
# in torch 2.8. Add shim functions that delegate to torch.cuda.*.
RUN echo -e "\n\
def empty_cache():\n\
    import torch; torch.cuda.empty_cache()\n\
def memory_stats(device=None):\n\
    import torch; return torch.cuda.memory_stats(device)\n\
def memory_reserved(device=None):\n\
    import torch; return torch.cuda.memory_reserved(device)\n\
def max_memory_allocated(device=None):\n\
    import torch; return torch.cuda.max_memory_allocated(device)\n\
def reset_peak_memory_stats(device=None):\n\
    import torch; torch.cuda.reset_peak_memory_stats(device)" \
    >> /usr/local/lib/python3.12/dist-packages/torch/accelerator/__init__.py

# ── Step 2c: Create fake vLLM dist-info ──
# vLLM source install via PYTHONPATH has no pip metadata, causing
# platform detection to fail. Create minimal dist-info.
RUN mkdir -p /usr/local/lib/python3.12/dist-packages/vllm-0.18.1.dist-info && \
    printf "Metadata-Version: 2.1\nName: vllm\nVersion: 0.18.1\n" \
    > /usr/local/lib/python3.12/dist-packages/vllm-0.18.1.dist-info/METADATA && \
    touch /usr/local/lib/python3.12/dist-packages/vllm-0.18.1.dist-info/RECORD && \
    echo pip > /usr/local/lib/python3.12/dist-packages/vllm-0.18.1.dist-info/INSTALLER

# ── Step 2d: Verify shim works ──
RUN python3 -c "import torch; torch.accelerator.empty_cache(); print('torch.accelerator shim OK')"

WORKDIR /workspace
```

Build:

```bash
docker build -f ~/Dockerfile.vllm-thor -t vllm-thor .
```

## Step 3: Clone and Patch vLLM Source

```bash
cd ~
git clone --depth 1 --branch v0.18.1 \
    https://github.com/vllm-project/vllm.git vllm-build
```

### Patch: Triton allocator for FLA DeltaNet kernels

Qwen3-Coder-Next uses FLA (Flash Linear Attention) DeltaNet layers with
Triton kernels. These kernels require a runtime memory allocator that
vLLM only sets for `olmo_hybrid` models. Without this patch, the GDN
prefill kernel fails with:
```
RuntimeError: Kernel requires a runtime memory allocation, but no allocator was set.
```

Apply the patch:

```bash
cd ~/vllm-build

# Add import
sed -i '/^from vllm.triton_utils import tl, triton$/a from vllm.triton_utils.allocation import set_triton_allocator' \
    vllm/model_executor/models/qwen3_next.py

# Add allocator call in Qwen3NextForCausalLM.__init__
# (after super().__init__(), before self.config = config)
python3 -c "
import pathlib
p = pathlib.Path('vllm/model_executor/models/qwen3_next.py')
code = p.read_text()
old = '''        super().__init__()
        self.config = config
        self.scheduler_config = scheduler_config
        self.model = Qwen3NextModel'''
new = '''        super().__init__()
        # Set Triton allocator for FLA DeltaNet kernels
        from vllm.platforms import current_platform
        set_triton_allocator(current_platform.current_device())
        self.config = config
        self.scheduler_config = scheduler_config
        self.model = Qwen3NextModel'''
assert old in code, 'Patch target not found — check vLLM version'
code = code.replace(old, new, 1)
p.write_text(code)
print('Patched qwen3_next.py')
"
```

Verify:
```bash
grep -n 'set_triton_allocator' vllm/model_executor/models/qwen3_next.py
# Should show import line and usage line
```

## Step 4: Build vLLM C Extensions for SM110

This compiles the CUDA kernels (`_C`, `_moe_C`, `_vllm_fa2_C`,
`cumem_allocator`) against the container's torch 2.8 ABI, targeting
**only SM110**.

```bash
# Start a build container
docker run -d --runtime=nvidia --name vllm-build-ctr \
    --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    -v ~/vllm-build:/workspace/vllm \
    -e TORCH_CUDA_ARCH_LIST="11.0" \
    -e VLLM_TARGET_DEVICE=cuda \
    -e MAX_JOBS=4 \
    vllm-thor sleep 7200

# Build C extensions (takes ~45 min on Thor ARM64)
docker exec \
    -e VLLM_TARGET_DEVICE=cuda \
    -e TORCH_CUDA_ARCH_LIST="11.0" \
    -e MAX_JOBS=4 \
    vllm-build-ctr \
    bash -c 'cd /workspace/vllm && python3 setup.py build_ext --inplace'
```

**Note:** The build compiles targets including `_vllm_fa3_C` (Flash
Attention 3). FA3 has 451 template instantiation files and takes 9+ hours
on ARM. It is **not needed** for Qwen3-Coder-Next (which uses FA2). If
the build hangs on FA3, kill it and manually build only the needed targets:

```bash
# Build only essential targets (skip FA3)
docker exec vllm-build-ctr bash -c '
    cd /workspace/vllm/build/temp.linux-aarch64-cpython-312 && \
    cmake --build . -j=4 \
        --target _C \
        --target _moe_C \
        --target cumem_allocator \
        --target _vllm_fa2_C
'
```

Copy the built `.so` files into the source tree:

```bash
docker exec vllm-build-ctr bash -c '
    BUILD=/workspace/vllm/build/temp.linux-aarch64-cpython-312
    cp $BUILD/_C.abi3.so /workspace/vllm/vllm/_C.abi3.so
    cp $BUILD/_moe_C.abi3.so /workspace/vllm/vllm/_moe_C.abi3.so
    cp $BUILD/cumem_allocator.abi3.so /workspace/vllm/vllm/cumem_allocator.abi3.so
    cp $BUILD/vllm-flash-attn/_vllm_fa2_C.abi3.so \
       /workspace/vllm/vllm/vllm_flash_attn/_vllm_fa2_C.abi3.so
    echo "Copied .so files:"
    ls -lh /workspace/vllm/vllm/_C.abi3.so \
           /workspace/vllm/vllm/_moe_C.abi3.so \
           /workspace/vllm/vllm/cumem_allocator.abi3.so \
           /workspace/vllm/vllm/vllm_flash_attn/_vllm_fa2_C.abi3.so
'
```

Expected sizes:
- `_C.abi3.so` — ~300 MB
- `_vllm_fa2_C.abi3.so` — ~225 MB
- `_moe_C.abi3.so` — ~80 MB
- `cumem_allocator.abi3.so` — ~150 KB

Clean up the build container:

```bash
docker stop vllm-build-ctr && docker rm vllm-build-ctr
```

## Step 5: Download the Model

```bash
pip install huggingface-hub  # on host
huggingface-cli download RedHatAI/Qwen3-Coder-Next-NVFP4
```

This downloads ~47.6 GB to `~/.cache/huggingface/`. Only needed once.

## Step 6: Create Persistent Cache Directories

```bash
mkdir -p ~/.cache/flashinfer
```

FlashInfer JIT-compiles 266 CUTLASS MoE kernel variants for SM110 on first
startup (~2 hours on ARM). The compiled kernels are cached at
`~/.cache/flashinfer/0.6.7/110a/`. This directory **must** be mounted into
the container to survive restarts.

## Step 7: Start vLLM

```bash
docker run -d --runtime=nvidia --name vllm-serve \
    --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    -p 8001:8001 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -v ~/.cache/flashinfer:/root/.cache/flashinfer \
    -v ~/vllm-build:/workspace/vllm \
    -e TRITON_PTXAS_BLACKWELL_PATH=/usr/local/cuda/bin/ptxas \
    -e CUDA_MODULE_LOADING=LAZY \
    -e PYTHONPATH=/workspace/vllm \
    -e VLLM_ENGINE_ITERATION_TIMEOUT_S=7200 \
    -e MAX_JOBS=2 \
    vllm-thor python3 -m vllm.entrypoints.openai.api_server \
    --model RedHatAI/Qwen3-Coder-Next-NVFP4 \
    --port 8001 --host 0.0.0.0 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.45 \
    --max-model-len 65536 \
    --max-num-seqs 4 \
    --enable-chunked-prefill \
    --enable-prefix-caching \
    --kv-cache-dtype auto \
    --trust-remote-code \
    --enforce-eager \
    --tool-call-parser qwen3_coder
```

### Key environment variables explained

| Variable | Value | Why |
|----------|-------|-----|
| `PYTHONPATH=/workspace/vllm` | Source install path | vLLM source + compiled .so files |
| `VLLM_ENGINE_ITERATION_TIMEOUT_S=7200` | 2 hours | FlashInfer JIT compilation takes ~2h on first run |
| `MAX_JOBS=2` | 2 parallel nvcc | Prevents OOM during FlashInfer JIT (model uses 44 GB, 16 nvcc would OOM) |
| `TRITON_PTXAS_BLACKWELL_PATH` | System ptxas | Container's Triton bundles incompatible ptxas for SM110 |
| `CUDA_MODULE_LOADING=LAZY` | Lazy | Reduces startup memory footprint |

### Key vLLM flags explained

| Flag | Value | Why |
|------|-------|-----|
| `--gpu-memory-utilization 0.45` | 45% of 122.8 GB = ~55 GB | Model is 44.3 GB, leaves ~11 GB for KV cache + overhead |
| `--max-model-len 65536` | 64K tokens | Maximum context window |
| `--max-num-seqs 4` | 4 concurrent | Limit concurrent requests to fit in memory |
| `--enable-chunked-prefill` | | Break long prefills into batches |
| `--enable-prefix-caching` | | Cache system prompt across turns |
| `--enforce-eager` | | No torch.compile (faster startup, avoids Triton issues) |
| `--trust-remote-code` | | Required for qwen3_next architecture |
| `--tool-call-parser qwen3_coder` | | Native tool calling support |

### Volume mounts

| Host Path | Container Path | Purpose |
|-----------|---------------|---------|
| `~/.cache/huggingface` | `/root/.cache/huggingface` | Model weights (47.6 GB) |
| `~/.cache/flashinfer` | `/root/.cache/flashinfer` | JIT-compiled CUTLASS kernels (survives restarts) |
| `~/vllm-build` | `/workspace/vllm` | vLLM source + compiled C extensions |

## Step 8: Wait for First Startup

First startup has three phases:

1. **Model weight loading** (~80 seconds)
   ```
   Loading safetensors checkpoint shards: 100% | 10/10
   Model loading took 44.31 GiB memory and 79.xx seconds
   ```

2. **FlashInfer CUTLASS MoE kernel JIT compilation** (~2 hours, first run only)
   - 266 kernel variants compiled by nvcc for SM110
   - Cached at `~/.cache/flashinfer/0.6.7/110a/cached_ops/`
   - Subsequent startups skip this entirely

3. **Triton FLA kernel autotuning** (~5-10 minutes, first run only)
   - DeltaNet attention kernels autotuned via Triton
   - Cached in container's Triton cache

Monitor progress:

```bash
# Watch logs
docker logs -f vllm-serve

# Check compilation progress
docker exec vllm-serve find /root/.cache/flashinfer/0.6.7/110a/cached_ops/fused_moe_100/ \
    -name '*.o' | wc -l
# Target: 266 object files

# Check compile processes
docker exec vllm-serve ps aux | grep -c 'nvcc\|ptxas'
# Should be 2 (MAX_JOBS=2)

# Check if ready
docker exec vllm-serve python3 -c \
    "import urllib.request; print(urllib.request.urlopen('http://localhost:8001/health', timeout=3).read().decode())"
```

Ready when you see:
```
INFO: Uvicorn running on http://0.0.0.0:8001
```

**Subsequent startups** (with cached FlashInfer kernels): ~90 seconds.

## Step 9: Test

```bash
# From Thor
docker exec vllm-serve python3 -c "
import urllib.request, json
req = urllib.request.Request(
    'http://localhost:8001/v1/chat/completions',
    data=json.dumps({
        'model': 'RedHatAI/Qwen3-Coder-Next-NVFP4',
        'max_tokens': 32,
        'messages': [{'role': 'user', 'content': 'Hello!'}]
    }).encode(),
    headers={'Content-Type': 'application/json'}
)
resp = json.loads(urllib.request.urlopen(req, timeout=60).read())
print(resp['choices'][0]['message']['content'])
"

# From Mac (via LAN)
curl -s http://main4.local:8001/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"RedHatAI/Qwen3-Coder-Next-NVFP4","max_tokens":32,"messages":[{"role":"user","content":"Hello!"}]}' \
    | python3 -m json.tool
```

## Step 10: Start agent-memory (vLLM backend)

```bash
# On Thor (outside Docker)
cd ~/agent-memory
source .venv/bin/activate

SEMANTIC_BACKEND=vllm \
SEMANTIC_VLLM_BASE_URL=http://localhost:8001 \
SEMANTIC_VLLM_MODEL_ID=RedHatAI/Qwen3-Coder-Next-NVFP4 \
python -m uvicorn agent_memory.entrypoints.api_server:create_app \
    --factory --host 0.0.0.0 --port 8000
```

## Revert to llama.cpp

```bash
docker stop vllm-serve
bash ~/switch_backend.sh llamacpp
```

## Memory Budget (128 GB Unified)

| Component | Size | Notes |
|-----------|------|-------|
| OS + Docker + CUDA context | ~15 GB | Always reserved |
| Model weights (NVFP4) | 44.3 GB | 80B params, MoE, compressed-tensors |
| KV cache | ~11 GB | At 0.45 util (~55 GB total - 44.3 GB model) |
| FlashInfer JIT compilation | ~2 GB peak | Only during first startup |
| **Total** | **~72 GB** | ~56 GB free for OS/other |

## Model Details

```
RedHatAI/Qwen3-Coder-Next-NVFP4
├── Parameters: 80B total / 3B active (MoE, 512 experts, 10 active)
├── Architecture: Hybrid — 3 DeltaNet (linear attention) + 1 full attention, repeating
├── Layers: 48
├── KV heads: 2
├── Head dim: 256
├── Quantization: NVFP4 (compressed-tensors), 47.6 GB
├── SWE-bench Verified: 52%
└── License: Apache 2.0
```

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `CUBLAS_STATUS_INVALID_VALUE` on `torch.mm()` | torch pip wheel (2.10+) cuBLAS calling code broken on SM110 | Must use NVIDIA container torch 2.8 |
| `torch.accelerator.empty_cache` AttributeError | torch 2.8 missing functions vLLM expects | Check torch.accelerator shim (Step 2b) |
| `No package metadata for vllm` | Source install has no pip metadata | Check fake dist-info (Step 2c) |
| `_C.abi3.so: undefined symbol` | C extensions built against wrong torch | Rebuild inside container (Step 4) |
| `RuntimeError: no allocator was set` | FLA Triton kernels need `set_triton_allocator` | Check qwen3_next.py patch (Step 3) |
| `NameError: set_triton_allocator` | Import line missing from patch | Verify both import and usage lines exist |
| `Killed` during FlashInfer JIT | OOM from too many parallel nvcc | Set `MAX_JOBS=2` |
| Container exits during FlashInfer JIT | Startup timeout | Set `VLLM_ENGINE_ITERATION_TIMEOUT_S=7200` |
| FlashInfer cache lost on restart | Not mounted to host | Mount `~/.cache/flashinfer` (Step 6) |
| vLLM source not found after restart | Source was in `/tmp` (cleaned) | Use `~/vllm-build` (persistent) |
| `Free memory less than desired` | Zombie GPU processes or too high utilization | Kill zombie containers, lower `--gpu-memory-utilization` |
| `--gpus all` not supported | Thor Docker setup | Use `--runtime=nvidia` instead |
| FA3 build takes 9+ hours | 451 template files on ARM | Skip FA3 — build only essential targets (Step 4) |
| `ptxas fatal: sm_110a not defined` | Container Triton's ptxas too old | Set `TRITON_PTXAS_BLACKWELL_PATH=/usr/local/cuda/bin/ptxas` |
| `No module named 'triton.language.target_info'` | Container Triton 3.3.1 incompatible | Warning only, non-fatal — vLLM falls back |
| Startup takes 2+ hours | FlashInfer JIT (first run) | Normal — cached after first run, ~90s thereafter |

## File Inventory

After setup, these files exist on Thor:

```
~/vllm-build/                          # vLLM v0.18.1 source (persistent)
  vllm/_C.abi3.so                      # Compiled for SM110
  vllm/_moe_C.abi3.so
  vllm/cumem_allocator.abi3.so
  vllm/vllm_flash_attn/_vllm_fa2_C.abi3.so
  vllm/model_executor/models/qwen3_next.py  # Patched (Triton allocator)

~/.cache/huggingface/                  # HuggingFace model cache
  hub/models--RedHatAI--Qwen3-Coder-Next-NVFP4/  # 47.6 GB

~/.cache/flashinfer/                   # FlashInfer JIT cache (persistent)
  0.6.7/110a/cached_ops/
    fp4_gemm_cutlass/fp4_gemm_cutlass.so
    fused_moe_100/fused_moe_100.so     # ~266 kernel variants

~/Dockerfile.vllm-thor                 # Dockerfile (this guide)
```

## See Also

- [`vllm_coder_next_thor.md`](vllm_coder_next_thor.md) — vLLM backend overview and comparison with llama.cpp
- [`llamacpp_backend.md`](llamacpp_backend.md) — llama.cpp fallback (always works, no cuBLAS dependency)
- [`thor_vllm_build.md`](thor_vllm_build.md) — Containerless vLLM build (for reference, not recommended)
