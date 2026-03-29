#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Build llm_inference_interactive inside Docker container on Thor.
#
# Prerequisites:
#   - Docker container 'triton_build' running (nvcr.io/nvidia/pytorch:25.08-py3)
#   - TensorRT-Edge-LLM cloned to /workspace/repnet/TensorRT-Edge-LLM
#   - This script and llm_inference_interactive.cpp in /workspace/repnet/TensorRT-Edge-LLM/
#
# Usage:
#   ssh yshkolni@main4.local
#   bash ~/agent-memory/vendor/build.sh
#
# Output:
#   ~/repnet/TensorRT-Edge-LLM/build_interactive/llm_inference_interactive

set -euo pipefail

CONTAINER="triton_build"
EDGELLM_DIR="/workspace/repnet/TensorRT-Edge-LLM"
BUILD_DIR="${EDGELLM_DIR}/build_interactive"
SRC_DIR="${EDGELLM_DIR}"

echo "=== Copying wrapper source into container workspace ==="
# Files are already in ~/repnet/ which is mounted at /workspace/repnet/

echo "=== Building inside container ==="
docker exec "${CONTAINER}" bash -c "
    set -euo pipefail
    cd ${EDGELLM_DIR}

    # Copy wrapper files into Edge-LLM tree
    mkdir -p ${BUILD_DIR}
    cd ${BUILD_DIR}

    cmake ${EDGELLM_DIR} \
        -DCMAKE_BUILD_TYPE=Release \
        -DCUDA_CTK_VERSION=13.0 \
        -DBUILD_TESTS=OFF

    make -j\$(nproc) llm_inference 2>&1 | tail -20

    echo '=== Build complete ==='
    ls -la examples/llm/llm_inference
"

echo ""
echo "Binary at: ~/repnet/TensorRT-Edge-LLM/build_interactive/examples/llm/llm_inference"
echo "To test: docker exec ${CONTAINER} ${BUILD_DIR}/examples/llm/llm_inference --help"
