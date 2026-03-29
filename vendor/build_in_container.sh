#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Build TensorRT Edge-LLM inside a Docker container on Jetson AGX Thor.
#
# Prerequisites:
#   - Container with cmake >= 3.20, nvcc (CUDA 13.0), TensorRT dev libs
#   - agent-memory repo mounted at /workspace/agent-memory
#
# Usage:
#   docker exec triton_build bash /workspace/agent-memory/vendor/build_in_container.sh
#
# Output:
#   /workspace/agent-memory/vendor/TensorRT-Edge-LLM/build/examples/llm/llm_inference

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENDOR_DIR="${SCRIPT_DIR}"
EDGELLM_DIR="${VENDOR_DIR}/TensorRT-Edge-LLM"
BUILD_DIR="${EDGELLM_DIR}/build"

echo "=== TensorRT Edge-LLM Build ==="
echo "Vendor dir: ${VENDOR_DIR}"
echo "Edge-LLM dir: ${EDGELLM_DIR}"

# Clone if not present
if [ ! -d "${EDGELLM_DIR}" ]; then
    echo "=== Cloning TensorRT Edge-LLM ==="
    cd "${VENDOR_DIR}"
    git clone --depth 1 https://github.com/NVIDIA/TensorRT-Edge-LLM.git
    cd "${EDGELLM_DIR}"
    git submodule update --init
fi

# Configure
echo "=== Configuring ==="
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"
# Apply sm_110 FMHA fix (Thor context attention kernel remap)
if [ -f "${VENDOR_DIR}/patches/sm110_fmha_fix.py" ]; then
    echo "=== Applying sm_110 FMHA patch ==="
    python3 "${VENDOR_DIR}/patches/sm110_fmha_fix.py" \
        "${EDGELLM_DIR}/cpp/kernels/contextAttentionKernels/contextFMHARunner.cpp"
fi

cmake "${EDGELLM_DIR}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCUDA_CTK_VERSION=13.0 \
    -DTRT_PACKAGE_DIR=/usr \
    -DBUILD_TESTS=OFF \
    -DAARCH64_BUILD=1 \
    -DCMAKE_CUDA_ARCHITECTURES=110

# Copy interactive wrapper source into Edge-LLM examples
echo "=== Copying llm_inference_interactive.cpp ==="
cp "${VENDOR_DIR}/llm_inference_interactive.cpp" "${EDGELLM_DIR}/examples/llm/"

# Append build target if not present
EXAMPLE_CMAKE="${EDGELLM_DIR}/examples/llm/CMakeLists.txt"
if ! grep -q "llm_inference_interactive" "${EXAMPLE_CMAKE}"; then
    cat >> "${EXAMPLE_CMAKE}" << 'CMAKEEOF'

# agent-memory interactive wrapper with KV cache inject/extract
add_executable(llm_inference_interactive llm_inference_interactive.cpp)
target_include_directories(llm_inference_interactive PRIVATE ${COMMON_INCLUDE_DIRS})
target_link_libraries(llm_inference_interactive PRIVATE edgellmCore ${CUDA_DRIVER_LIB} ${CUDART_LIB})
add_cross_build_link_options(llm_inference_interactive)
CMAKEEOF
    echo "Added llm_inference_interactive target to CMakeLists.txt"
fi

# Build stock + interactive
# Remove debug fprintf from attention plugin (if present)
if [ -f "${VENDOR_DIR}/patches/remove_debug_prints.py" ]; then
    echo "=== Removing debug prints ==="
    python3 "${VENDOR_DIR}/patches/remove_debug_prints.py" \
        "${EDGELLM_DIR}/cpp/plugins/attentionPlugin/attentionPlugin.cpp" 2>/dev/null || true
fi

echo "=== Building llm_inference + llm_inference_interactive ==="
make -j"$(nproc)" llm_inference llm_inference_interactive

echo ""
echo "=== Build complete ==="
echo "Binary: ${BUILD_DIR}/examples/llm/llm_inference"
ls -la "${BUILD_DIR}/examples/llm/llm_inference"

# Verify
echo ""
echo "=== Verify ==="
"${BUILD_DIR}/examples/llm/llm_inference" --help 2>&1 | head -3
