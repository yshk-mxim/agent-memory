# SPDX-License-Identifier: Apache-2.0
# Build llm_inference_interactive against TensorRT Edge-LLM SDK
#
# Usage (inside Docker container with TensorRT + CUDA):
#   mkdir build && cd build
#   cmake .. -DEDGELLM_DIR=/path/to/TensorRT-Edge-LLM
#   make -j$(nproc)

cmake_minimum_required(VERSION 3.20)
project(llm_inference_interactive LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Edge-LLM SDK root
if(NOT DEFINED EDGELLM_DIR)
    set(EDGELLM_DIR "${CMAKE_SOURCE_DIR}/TensorRT-Edge-LLM")
endif()

# Build Edge-LLM as subdirectory
add_subdirectory(${EDGELLM_DIR} edgellm_build)

# Our interactive wrapper
add_executable(llm_inference_interactive llm_inference_interactive.cpp)

target_include_directories(llm_inference_interactive PRIVATE
    ${EDGELLM_DIR}/cpp
    ${EDGELLM_DIR}/3rdParty/nlohmannJson/include
)

target_link_libraries(llm_inference_interactive PRIVATE
    trt_edgellm_runtime
    trt_edgellm_tokenizer
    nvinfer
    cudart
)
