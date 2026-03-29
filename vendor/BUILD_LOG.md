# TRT Edge-LLM Build Log — SmolLM2-135M on Thor

Date: 2026-03-29
Platform: Jetson AGX Thor (sm_110, JetPack 7.1, CUDA 13.0, TensorRT 10.13)
Container: `repnet/pytorch-triton:latest` (based on `nvcr.io/nvidia/pytorch:25.08-py3`)

## Step 1: Container Setup

```bash
docker run -d --name triton_build \
    --runtime nvidia --gpus all \
    -v ~/agent-memory:/workspace/agent-memory \
    -v ~/repnet:/workspace/repnet \
    --network host \
    -e TRITON_PTXAS_BLACKWELL_PATH=/usr/local/cuda/bin/ptxas \
    -e CUDA_MODULE_LOADING=LAZY \
    -e TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1 \
    repnet/pytorch-triton:latest sleep infinity
```

Container has: PyTorch 2.12.0.dev+cu130, TensorRT 10.13.2.6, cmake 3.31, nvcc 13.0.

## Step 2: Install Edge-LLM Python Deps

**CRITICAL: Use `--no-deps` for Edge-LLM to avoid overwriting CUDA torch.**

The NGC container ships torch 2.12.0.dev+cu130 built from source. Edge-LLM's
`pyproject.toml` declares `torch` as a dependency, and pip resolves it to a
CPU-only `torch==2.10.0` from PyPI, breaking CUDA. There is no way to restore
the NGC torch via pip (it was built from source with custom CUDA arch flags).

```bash
docker exec triton_build bash -c "
    cd /workspace/agent-memory/vendor/TensorRT-Edge-LLM
    pip install --no-deps -e .
    pip install --no-deps nvidia-modelopt==0.39.0
    pip install --no-deps onnx onnxruntime onnx-graphsurgeon accelerate einops coloredlogs humanfriendly
    pip install --no-deps huggingface-hub hf-xet tokenizers peft datasets multiprocess dill xxhash
    pip install --no-deps transformers
    pip install 'regex>=2025.10.22' ml-dtypes
"
```

### Dep version notes:
- `nvidia-modelopt==0.39.0` must match `nvidia-modelopt-core==0.33.0` pre-installed in NGC
  - 0.27.1 crashes with `ModuleNotFoundError: modelopt.torch.quantization.backends`
- `transformers>=5.4.0` requires `regex>=2025.10.22` (NGC ships 2025.7.34)
- `ml-dtypes` is needed by ONNX export but not declared as a dep
- `transformer_engine` plugin warnings (`sym_ne` symbol) are harmless — ABI mismatch
  between NGC's TE and the newer modelopt, does not affect LLM export

### Verification:
```bash
docker exec triton_build python3 -c "
import torch; print(torch.__version__, torch.cuda.is_available())
# Should print: 2.12.0.dev20260325+cu130 True
"
```

## Step 3: Build C++ Runtime

```bash
docker exec triton_build bash -c "
    cd /workspace/agent-memory/vendor/TensorRT-Edge-LLM
    git submodule update --init
    mkdir -p build && cd build
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DCUDA_CTK_VERSION=13.0 \
        -DTRT_PACKAGE_DIR=/usr \
        -DBUILD_TESTS=OFF \
        -DCMAKE_CUDA_ARCHITECTURES='80;100;110;120'
    make -j\$(nproc) llm_inference llm_build NvInfer_edgellm_plugin
"
```

Produces:
- `build/examples/llm/llm_inference` — stock batch-mode binary
- `build/examples/llm/llm_build` — engine builder
- `build/libNvInfer_edgellm_plugin.so` — TRT plugin (MUST be loaded via `EDGELLM_PLUGIN_PATH`)

## Step 4: ONNX Export

SmolLM2-135M uses LlamaForCausalLM architecture. Export works but the config
export step fails because SmolLM2 doesn't have `rope_theta` in its HF config
(it uses the default 10000.0).

```bash
docker exec triton_build bash -c "
    export PYTHONPATH=/workspace/agent-memory/vendor/TensorRT-Edge-LLM:\$PYTHONPATH
    python3 /workspace/agent-memory/vendor/TensorRT-Edge-LLM/tensorrt_edgellm/scripts/export_llm.py \
        --model_dir HuggingFaceTB/SmolLM2-135M-Instruct \
        --output_dir /workspace/agent-memory/vendor/engines/SmolLM2-135M/onnx
"
```

**Result**: ONNX model exported successfully (514MB), but `config.json` not
generated due to missing `rope_theta`. The ONNX export completes first, then
the config export fails — so the ONNX file is usable.

### Manual config.json creation

The config must use Edge-LLM's field names (different from HuggingFace):
- `num_hidden_layers` not `num_decoder_layers`
- `num_key_value_heads` not `num_kv_heads`
- `trt_native_ops` not `use_trt_native_ops`

```json
{
  "model_type": "llm",
  "num_hidden_layers": 30,
  "num_attention_heads": 9,
  "num_key_value_heads": 3,
  "head_dim": 64,
  "hidden_size": 576,
  "intermediate_size": 1536,
  "vocab_size": 49152,
  "max_position_embeddings": 8192,
  "rope_theta": 10000.0,
  "rope_type": "default",
  "rotary_dim": 64,
  "rms_norm_eps": 1e-05,
  "trt_native_ops": false,
  "use_context_dependent_rope": false
}
```

### Embedding table extraction

The ONNX export fails at the config stage (missing `rope_theta`), which also
skips embedding table extraction. Must extract manually:

```bash
docker exec triton_build python3 -c "
import torch
from transformers import AutoModelForCausalLM
from safetensors.torch import save_file
model = AutoModelForCausalLM.from_pretrained('HuggingFaceTB/SmolLM2-135M-Instruct', torch_dtype=torch.float16)
save_file({'embedding': model.model.embed_tokens.weight.data}, '<engine_dir>/embedding.safetensors')
"
```

### Chat template

```bash
docker exec triton_build bash -c "
    export PYTHONPATH=/workspace/agent-memory/vendor/TensorRT-Edge-LLM:\$PYTHONPATH
    python3 -c \"
from tensorrt_edgellm.chat_templates.chat_template import process_chat_template
process_chat_template('HuggingFaceTB/SmolLM2-135M-Instruct', '<engine_dir>')
\"
"
```

### CRITICAL: sm_110 missing from CUDA architectures

Edge-LLM's CMakeLists.txt sets `CMAKE_CUDA_ARCHITECTURES` to `80;86;89;100;120`
but does NOT include `110` (Thor). The FMHA kernels compile for sm_100 (Blackwell
datacenter) but the kernel lookup at runtime uses the device's actual SM version
(sm_110), which has no matching kernel, causing the "There must be one kernel to
implement the MHA" crash.

**Fix**: Two changes required:

1. Pass `-DAARCH64_BUILD=1 -DCMAKE_CUDA_ARCHITECTURES=110` to cmake.
   Without `AARCH64_BUILD`, the top-level CMakeLists.txt overrides
   `CMAKE_CUDA_ARCHITECTURES` to `80;86;89;100;120` before the FMHA
   exclusion logic runs. The cmake FMHA logic already maps sm_110->sm_101
   cubins, but only if 110 is in the architecture list.

2. Apply `patches/sm110_fmha_fix.py` to `contextFMHARunner.cpp`.
   The attention plugin has `applyThorSMRenumberWAR` (sm_110->sm_101)
   but the context FMHA runner does not. The patch adds the same remap
   to `getFMHAKernelList()`, the constructor, and the `isSm10x` check.

This is likely a bug in Edge-LLM 0.6.0 — Thor (sm_110) is a supported Jetson
platform but its SM architecture is not in the default build configuration.

### Custom attention plugin vs TRT native ops

Edge-LLM supports two attention paths:
- **Custom plugin** (`trt_native_ops: false`): Uses `libNvInfer_edgellm_plugin.so` with
  custom FMHA kernels. Faster but requires plugin .so at runtime.
- **TRT native ops** (`trt_native_ops: true`): Uses TensorRT's built-in attention.
  More portable, no plugin needed at runtime.

**SmolLM2-135M crashes with custom plugin** ("There must be one kernel to implement the MHA")
because the FMHA kernel selection doesn't find a matching kernel for SmolLM2's
geometry (9 heads, head_dim=64, 3 KV heads). The TRT native ops path works.

**Always export AND build with matching trt_native_ops setting.**

## Step 5: Engine Build

```bash
docker exec triton_build bash -c "
    export EDGELLM_PLUGIN_PATH=/workspace/agent-memory/vendor/TensorRT-Edge-LLM/build/libNvInfer_edgellm_plugin.so
    /workspace/agent-memory/vendor/TensorRT-Edge-LLM/build/examples/llm/llm_build \
        --onnxDir /workspace/agent-memory/vendor/engines/SmolLM2-135M/onnx \
        --engineDir /workspace/agent-memory/vendor/engines/SmolLM2-135M/engine \
        --maxBatchSize 1 \
        --maxInputLen 2048 \
        --maxKVCacheCapacity 4096
"
```

**CRITICAL**: `EDGELLM_PLUGIN_PATH` must point to the built plugin .so, otherwise
the builder crashes with "Cannot open plugin library".

## Step 6: Test Inference (pending engine build)

```bash
docker exec triton_build bash -c "
    export EDGELLM_PLUGIN_PATH=/workspace/agent-memory/vendor/TensorRT-Edge-LLM/build/libNvInfer_edgellm_plugin.so
    cat > /tmp/test_input.json << 'EOF'
{
    \"batch_size\": 1,
    \"temperature\": 0.7,
    \"max_generate_length\": 32,
    \"requests\": [{\"messages\": [{\"role\": \"user\", \"content\": \"Hello!\"}]}]
}
EOF
    /workspace/agent-memory/vendor/TensorRT-Edge-LLM/build/examples/llm/llm_inference \
        --engineDir /workspace/agent-memory/vendor/engines/SmolLM2-135M/engine \
        --inputFile /tmp/test_input.json \
        --dumpOutput
"
```

## Gotchas Summary

| Issue | Root Cause | Fix |
|-------|-----------|-----|
| `pip install tensorrt-edgellm` breaks CUDA torch | Unpinned torch dep pulls CPU wheel | Use `--no-deps` |
| modelopt version mismatch | NGC has core 0.33, pip gets 0.27 | Pin `nvidia-modelopt==0.39.0` |
| `regex>=2025.10.22` required | transformers 5.4.0 needs newer regex | `pip install regex>=2025.10.22` |
| `ml_dtypes` missing | Not declared as dep by onnx export | `pip install ml-dtypes` |
| `rope_theta` not in SmolLM2 config | HF config omits default values | Create config.json manually |
| Config field names differ | Edge-LLM uses `num_hidden_layers`, HF uses `num_decoder_layers` | Map manually |
| Plugin not found | `EDGELLM_PLUGIN_PATH` not set | Export env var |
| `tensorrt-edgellm-export-llm` not found | `--no-deps` skips entry point install | Use `python3 script.py` directly |
| Missing embedding.safetensors | Config error aborts before embedding export | Extract manually from HF model |
| Missing processed_chat_template.json | Same — config error aborts early | Generate via `process_chat_template()` |
| "There must be one kernel for MHA" | sm_110 missing from CMAKE_CUDA_ARCHITECTURES | Add `-DCMAKE_CUDA_ARCHITECTURES="80;100;110;120"` |
| `--outputFile` is required | Binary requires it despite docs saying "optional" | Always pass `--outputFile` |
