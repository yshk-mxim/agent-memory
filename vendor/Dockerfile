# SPDX-License-Identifier: Apache-2.0
# Dockerfile for building llm_inference_interactive on Jetson AGX Thor
#
# Base: NGC PyTorch container (has cmake, nvcc, TensorRT dev libs)
# Target: aarch64 / JetPack 7.1 / CUDA 13.0 / TensorRT 10.13
#
# Usage:
#   docker build -t agent-memory/trt-edgellm -f vendor/Dockerfile .
#   docker run --rm --runtime nvidia --gpus all \
#       -v $(pwd):/workspace/agent-memory \
#       agent-memory/trt-edgellm \
#       /workspace/agent-memory/vendor/TensorRT-Edge-LLM/build/examples/llm/llm_inference --help
#
# Or build inside an existing container:
#   docker exec <container> bash /workspace/agent-memory/vendor/build_in_container.sh

FROM nvcr.io/nvidia/pytorch:25.08-py3

LABEL maintainer="Yakov Shkolnikov"
LABEL description="TensorRT Edge-LLM interactive wrapper for agent-memory"
LABEL org.opencontainers.image.source="https://github.com/yshk-mxim/agent-memory"
LABEL org.opencontainers.image.licenses="Apache-2.0"

WORKDIR /workspace

# Clone and build TensorRT Edge-LLM
RUN git clone --depth 1 https://github.com/NVIDIA/TensorRT-Edge-LLM.git && \
    cd TensorRT-Edge-LLM && \
    git submodule update --init

# Copy our interactive wrapper source
COPY vendor/llm_inference_interactive.cpp /workspace/TensorRT-Edge-LLM/examples/llm/
COPY vendor/CMakeLists_edgellm.txt /workspace/TensorRT-Edge-LLM/examples/llm/CMakeLists_interactive.txt

# Build stock llm_inference + our interactive wrapper
RUN cd /workspace/TensorRT-Edge-LLM && \
    mkdir -p build && cd build && \
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DCUDA_CTK_VERSION=13.0 \
        -DTRT_PACKAGE_DIR=/usr \
        -DBUILD_TESTS=OFF && \
    make -j$(nproc) llm_inference && \
    echo "Stock llm_inference built successfully"

# Install Python deps for agent-memory
RUN pip install --no-cache-dir \
    safetensors numpy pydantic pydantic-settings \
    fastapi uvicorn transformers structlog

ENV PATH="/workspace/TensorRT-Edge-LLM/build/examples/llm:${PATH}"
