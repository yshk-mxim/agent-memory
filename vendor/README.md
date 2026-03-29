# vendor/ — TRT Edge-LLM Interactive Wrapper

Custom NDJSON wrapper around [TensorRT Edge-LLM](https://github.com/NVIDIA/TensorRT-Edge-LLM)
for agent-memory's persistent KV cache system.

## Why

TensorRT-LLM does not support Jetson AGX Thor (sm_110). TensorRT Edge-LLM is
the supported path, but its stock `llm_inference` binary is batch-mode
(JSON file I/O). agent-memory needs an interactive subprocess with NDJSON
over stdin/stdout and KV cache exchange via safetensors in `/dev/shm`.

## Files

| File | Purpose |
|------|---------|
| `llm_inference_interactive.cpp` | C++ NDJSON wrapper with KV cache inject/extract |
| `llm_inference_wrapper.py` | Python NDJSON wrapper (uses stock binary, no C++ needed) |
| `CMakeLists.txt` | Build config documentation |
| `build_in_container.sh` | Reproducible build script for Docker container |
| `Dockerfile` | Full build from NGC base image |
| `BUILD_LOG.md` | Detailed build log with every gotcha discovered |
| `.gitignore` | Ignores cloned SDK and build artifacts |
| `patches/sm110_fmha_fix.py` | Fix missing sm_110 FMHA support in Edge-LLM |
| `patches/add_engine_accessor.py` | Add public getEngineRunner() for KV cache access |
| `patches/add_debug.py` | Debug fprintf (dev only, removed in production) |
| `patches/remove_debug_prints.py` | Removes debug prints before production build |

## Build (on Thor)

```bash
# 1. Clone Edge-LLM SDK into vendor/
cd ~/agent-memory/vendor
git clone --depth 1 https://github.com/NVIDIA/TensorRT-Edge-LLM.git
cd TensorRT-Edge-LLM && git submodule update --init && cd ..

# 2. Build inside Docker (has cmake + nvcc + TensorRT)
docker exec triton_build bash -c "
    cd /workspace/agent-memory/vendor/TensorRT-Edge-LLM
    mkdir -p build && cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release -DCUDA_CTK_VERSION=13.0 -DTRT_PACKAGE_DIR=/usr -DBUILD_TESTS=OFF
    make -j\$(nproc) llm_inference
"

# Binary at: vendor/TensorRT-Edge-LLM/build/examples/llm/llm_inference
```

## NDJSON Protocol

```
→ stdin:  {"cmd": "get_model_spec"}\n
← stdout: {"n_layers": 30, "n_kv_heads": 3, "head_dim": 64, ...}\n

→ stdin:  {"cmd": "generate", "tokens": [1,2,3], "max_tokens": 10}\n
← stdout: {"text": "...", "tokens": [...], "finish_reason": "stop"}\n

→ stdin:  {"cmd": "shutdown"}\n
← stdout: {"status": "shutdown"}\n
```

## Full Pipeline (SmolLM2-135M on Thor)

```bash
# 1. Start fresh NGC container with Thor env vars
docker run -d --name triton_build \
    --runtime nvidia --gpus all \
    -v ~/agent-memory:/workspace/agent-memory \
    --network host \
    -e TRITON_PTXAS_BLACKWELL_PATH=/usr/local/cuda/bin/ptxas \
    -e CUDA_MODULE_LOADING=LAZY \
    -e TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1 \
    nvcr.io/nvidia/pytorch:25.08-py3 sleep infinity

# 2. Install Edge-LLM Python deps WITHOUT overwriting CUDA torch
docker exec triton_build bash -c "
    pip install --no-deps nvidia-modelopt==0.39.0
    pip install --no-deps onnx onnxruntime onnx-graphsurgeon accelerate einops coloredlogs humanfriendly
    pip install --no-deps huggingface-hub hf-xet tokenizers peft datasets multiprocess dill xxhash
    pip install --no-deps transformers
    pip install 'regex>=2025.10.22' ml-dtypes
"

# 3. Clone and build Edge-LLM C++ runtime
docker exec triton_build bash /workspace/agent-memory/vendor/build_in_container.sh

# 4. Export SmolLM2-135M to ONNX (FP16, no quantization)
docker exec triton_build bash -c "
    export PYTHONPATH=/workspace/agent-memory/vendor/TensorRT-Edge-LLM:\$PYTHONPATH
    tensorrt-edgellm-export-llm \
        --model_dir HuggingFaceTB/SmolLM2-135M-Instruct \
        --output_dir /workspace/agent-memory/vendor/engines/SmolLM2-135M/onnx
"

# 5. Build TRT engine
docker exec triton_build bash -c "
    /workspace/agent-memory/vendor/TensorRT-Edge-LLM/build/examples/llm/llm_build \
        --onnxDir /workspace/agent-memory/vendor/engines/SmolLM2-135M/onnx \
        --engineDir /workspace/agent-memory/vendor/engines/SmolLM2-135M/engine \
        --maxBatchSize 1 \
        --maxInputLen 2048 \
        --maxKVCacheCapacity 4096
"

# 6. Test inference
docker exec triton_build bash -c "
    cat > /tmp/test_input.json << 'TESTEOF'
    {\"batch_size\": 1, \"temperature\": 0.7, \"max_generate_length\": 32,
     \"requests\": [{\"messages\": [{\"role\": \"user\", \"content\": \"Hello!\"}]}]}
    TESTEOF
    /workspace/agent-memory/vendor/TensorRT-Edge-LLM/build/examples/llm/llm_inference \
        --engineDir /workspace/agent-memory/vendor/engines/SmolLM2-135M/engine \
        --inputFile /tmp/test_input.json \
        --dumpOutput
"
```

## Known Issues

**`pip install tensorrt-edgellm` overwrites CUDA PyTorch with CPU-only version.**
The Edge-LLM package declares `torch` as a dependency without version pins,
so pip may replace the NGC container's cu130 torch with a CPU-only wheel from PyPI.
Fix: use `pip install --no-deps -e .` and install deps individually (see pipeline above).

**`nvidia-modelopt` version must match `nvidia-modelopt-core` in NGC container.**
The NGC `pytorch:25.08-py3` image ships `modelopt-core==0.33.0`. Use
`nvidia-modelopt==0.39.0` (compatible). Older versions (0.27.x) will crash.

**`transformers>=5.4.0` requires `regex>=2025.10.22`.**
The NGC container ships an older regex. Install explicitly: `pip install 'regex>=2025.10.22'`

**`transformer_engine` plugin warnings are harmless.**
The `_ZNK3c106SymInt6sym_neERKS0_` symbol mismatch is between the NGC container's
transformer_engine (built for nightly torch) and the stable modelopt. It does not
affect LLM export or inference.

## License

- `llm_inference_interactive.cpp`: Apache-2.0 (based on Edge-LLM example code)
- TensorRT Edge-LLM SDK: Apache-2.0 (NVIDIA)
- nlohmann/json: MIT
