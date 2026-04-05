#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Patch llmInferenceRuntime.h to add public accessor for LLMEngineRunner.

Needed by llm_inference_interactive to access LinearKVCache for
KV cache inject/extract operations.
"""

import sys
from pathlib import Path


def patch(filepath: str) -> None:
    path = Path(filepath)
    src = path.read_text()

    if "getEngineRunner" in src:
        print(f"Already patched: {filepath}")
        return

    accessor = (
        "    /*! \\brief Get reference to the LLM engine runner (for KV cache access)\n"
        "     *  \\return Reference to the LLM engine runner\n"
        "     */\n"
        "    rt::LLMEngineRunner& getEngineRunner() noexcept { return *mLLMEngineRunner; }\n\n"
    )

    # Insert after the last public method (captureDecodingCUDAGraph)
    marker = "    bool captureDecodingCUDAGraph(cudaStream_t stream);"
    if marker in src:
        src = src.replace(marker, marker + "\n\n" + accessor)
        path.write_text(src)
        print(f"Patched: added getEngineRunner() after captureDecodingCUDAGraph in {filepath}")
    else:
        print(f"Marker not found in {filepath}")


if __name__ == "__main__":
    patch(sys.argv[1])
