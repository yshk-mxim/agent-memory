# Building vLLM for NVIDIA Thor (sm_110) — Containerless Setup

> **Note:** For DeltaNet/MoE models (Qwen3-Coder-Next, Qwen3.5), **llama.cpp
> is the recommended path** — see [llamacpp_backend.md](llamacpp_backend.md).
> vLLM on Thor has unresolved issues with MIXED_PRECISION quantization and
> FlashInfer kernel gaps.  This doc is retained for reference.

> Reproduces vLLM 0.18.0 with sm_110 CUDA kernels on Jetson AGX Thor.
> No NGC container required — runs in a Python venv with native CUDA 13.0.

## Prerequisites

| Component | Value | Notes |
|-----------|-------|-------|
| Platform | aarch64 (Linux 6.8.12-tegra) | Jetson AGX Thor |
| GPU | NVIDIA Thor (sm_110 / CC 11.0) | |
| Driver | 580.00 | |
| System CUDA | 13.0 (V13.0.48) at `/usr/local/cuda` | |
| Python | 3.12.3 (system) | 3.14 also works |
| System packages | `python3-venv` not installed (no sudo) | Workaround below |

## Why Not Just `pip install vllm`?

Three issues with the default pip wheel:

1. **Default PyPI wheel (`vllm-0.18.0`)** — compiled against CUDA 12, links `libcudart.so.12`. Thor has CUDA 13 only → `ImportError: libcudart.so.12 not found`.

2. **cu130 wheel (`vllm-0.18.0+cu130`)** from GitHub releases — correct CUDA runtime, but compiled for `sm_87, sm_89, sm_90, sm_100, sm_120`. Missing `sm_110` (Thor) → `cudaErrorNoKernelImageForDevice`.

3. **NGC container (26.02, vLLM 0.15.1)** — has sm_110 kernels but doesn't support `MIXED_PRECISION` quant_algo needed by Nemotron 3 Super NVFP4.

**Solution:** Install the cu130 wheel for all Python deps, then rebuild vLLM's CUDA extensions from source targeting sm_110 only.

## Step 1: Create venv (without ensurepip)

`python3-venv` isn't installed and no sudo is available. Create venv without pip, then bootstrap it.

```bash
python3 -m venv --without-pip ~/vllm-env
wget -qO- https://bootstrap.pypa.io/get-pip.py | ~/vllm-env/bin/python3
```

## Step 2: Patch activate script with Thor env vars

```bash
cat >> ~/vllm-env/bin/activate << 'PATCH'

# --- Thor environment ---
export TRITON_PTXAS_BLACKWELL_PATH=/usr/local/cuda/bin/ptxas
export CUDA_MODULE_LOADING=LAZY
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
PATCH
```

## Step 3: Install PyTorch cu130

```bash
source ~/vllm-env/bin/activate
pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 \
    --index-url https://download.pytorch.org/whl/cu130
```

Verify sm_110 support:

```bash
python3 -c "import torch; print(torch.cuda.get_arch_list()); print(torch.cuda.get_device_capability(0))"
# ['sm_80', 'sm_90', 'sm_100', 'sm_110', 'sm_120', 'compute_120']
# (11, 0)
```

## Step 4: Install vLLM cu130 wheel (for Python deps)

```bash
pip install "https://github.com/vllm-project/vllm/releases/download/v0.18.0/vllm-0.18.0%2Bcu130-cp38-abi3-manylinux_2_35_aarch64.whl"
```

This installs all vLLM Python dependencies. The C extensions won't work yet (no sm_110).

**Reinstall torch cu130** (vLLM pulls in CPU torch as a dependency):

```bash
pip install --force-reinstall torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 \
    --index-url https://download.pytorch.org/whl/cu130
```

## Step 5: Rebuild vLLM CUDA extensions for sm_110

```bash
pip install cmake setuptools-scm "setuptools>=77.0.3,<81"

cd /tmp
git clone --depth 1 --branch v0.18.0 https://github.com/vllm-project/vllm.git vllm-build
cd vllm-build

# Fix pyproject.toml license format (setuptools compatibility)
sed -i 's/^license = "Apache-2.0"/license = {text = "Apache-2.0"}/' pyproject.toml
sed -i '/^license-files/d' pyproject.toml

# Build only for sm_110 (much faster than all architectures)
export TORCH_CUDA_ARCH_LIST="11.0"
export VLLM_TARGET_DEVICE=cuda
export MAX_JOBS=4
export PATH=/usr/local/cuda/bin:$PATH

pip install -e . --no-build-isolation
```

Build takes ~30-60 minutes on Thor with 4 jobs. Compiles: `_C`, `_moe_C`, `_vllm_fa2_C`, `_vllm_fa3_C`, `_vllm_fa4_cutedsl_C`, `_flashmla_C`, `_flashmla_extension_C`, `cumem_allocator`, `triton_kernels`.

## Step 6: Model setup

### Download Nemotron 3 Super 120B-A12B NVFP4

```bash
pip install huggingface-hub
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4',
                  local_dir='/home/yshkolni/models/nemotron-3-super-120b')
"
```

Or if already downloaded to HF cache:

```bash
export HF_HUB_CACHE=/home/yshkolni/.cache/huggingface
```

### Note on MIXED_PRECISION quant_algo

The NVFP4 model uses `"quant_algo": "MIXED_PRECISION"` (FP8 for Mamba layers, NVFP4 for MoE experts). vLLM 0.18.0 supports this natively — no config patching needed.

If using **NGC vLLM 0.15.1** (container), you must patch `config.json` to change `MIXED_PRECISION` → `NVFP4` since 0.15.1 doesn't support mixed precision.

## Step 7: Serve

```bash
cat > ~/start_vllm.sh << "EOF"
#!/bin/bash
source ~/vllm-env/bin/activate
export TRITON_PTXAS_BLACKWELL_PATH=/usr/local/cuda/bin/ptxas
export CUDA_MODULE_LOADING=LAZY
export PATH=/usr/local/cuda/bin:$PATH
export HF_HUB_CACHE=/home/yshkolni/.cache/huggingface

pkill -f "vllm serve" 2>/dev/null
sleep 1

nohup vllm serve nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 \
    --port 5000 --dtype bfloat16 --kv-cache-dtype fp8 \
    --tensor-parallel-size 1 --gpu-memory-utilization 0.40 \
    --max-num-seqs 8 --enable-chunked-prefill \
    --host 0.0.0.0 --trust-remote-code --enforce-eager \
    > ~/vllm-serve.log 2>&1 &
echo "vLLM PID: $!"
EOF
chmod +x ~/start_vllm.sh
bash ~/start_vllm.sh
```

### Memory budget (128 GB unified)

| Component | Estimated | Notes |
|-----------|-----------|-------|
| OS + driver | ~5 GB | Always reserved |
| Model weights (NVFP4 + FP8) | ~62-71 GB | NVFP4 + scale factors + FP8 layers |
| KV cache (fp8) | ~40-49 GB | At 0.40 util: ~49 GiB |
| CUDA context | ~2 GB | Kernels, allocator |

`--gpu-memory-utilization` is fraction of **total VRAM** (122.82 GiB visible), not free VRAM. After model loads (~71 GB), only ~52 GiB remains. Set to 0.40 (= 49 GiB) to fit.

## Step 8: Test

```bash
curl http://localhost:5000/v1/models

curl -s http://localhost:5000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4",
        "max_tokens": 32,
        "messages": [{"role": "user", "content": "Hello!"}]
    }' | python3 -m json.tool
```

## Cleanup

Remove containers and images no longer needed:

```bash
# Remove NGC vLLM container and image
docker rm -f vllm-nemotron 2>/dev/null
docker rmi nvcr.io/nvidia/vllm:26.02-py3

# Remove dangling images
docker image prune -f

# Remove incomplete hub download (root-owned, use Docker)
docker run --rm -v ~/.cache/huggingface:/hf alpine \
    rm -rf /hf/hub/models--nvidia--NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 /hf/hub/.locks

# Clean build artifacts
rm -rf /tmp/vllm-build
```

## Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `libcudart.so.12 not found` | Default PyPI wheel is CUDA 12 | Use cu130 wheel from GitHub |
| `cudaErrorNoKernelImageForDevice` | No sm_110 cubins in wheel | Rebuild from source (Step 5) |
| `MIXED_PRECISION not supported` | vLLM < 0.17.0 | Use vLLM >= 0.17.0 or patch config |
| `PermissionError: .locks/` | HF cache owned by root (Docker) | `sudo chown -R $USER ~/.cache/huggingface/hub/` or cleanup via Docker |
| `Free memory < gpu-memory-utilization` | Model too large for memory budget | Lower `--gpu-memory-utilization` |
| `ptxas fatal: sm_110a not defined` | Triton bundled ptxas too old | Set `TRITON_PTXAS_BLACKWELL_PATH=/usr/local/cuda/bin/ptxas` |
