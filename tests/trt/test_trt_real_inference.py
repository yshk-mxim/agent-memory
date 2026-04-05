# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Yakov Shkolnikov and contributors
"""TRT real inference tests — runs on Jetson Thor with actual TRT engine.

Requires:
    REAL_ENGINE_DIR: Path to SmolLM2-135M engine directory
    REAL_WRAPPER_PATH: Path to llm_inference_wrapper.py
    EDGELLM_PLUGIN_PATH: Path to libNvInfer_edgellm_plugin.so

Example:
    REAL_ENGINE_DIR=~/agent-memory/vendor/engines/SmolLM2-135M/engine2 \
    REAL_WRAPPER_PATH=~/agent-memory/vendor/llm_inference_wrapper.py \
    EDGELLM_PLUGIN_PATH=~/agent-memory/vendor/TensorRT-Edge-LLM/build/libNvInfer_edgellm_plugin.so \
    pytest tests/trt/test_trt_real_inference.py -v
"""

import os
import sys
from pathlib import Path

import pytest

from agent_memory.domain.value_objects import ModelCacheSpec
from agent_memory.ports.outbound import ModelBackendPort

from .conftest import skip_if_not_jetson

ENGINE_DIR = os.environ.get("REAL_ENGINE_DIR", "")
WRAPPER_PATH = os.environ.get("REAL_WRAPPER_PATH", "")
INTERACTIVE_BIN = os.environ.get("REAL_INTERACTIVE_BIN", "")

skip_if_no_engine = pytest.mark.skipif(
    not ENGINE_DIR or not Path(ENGINE_DIR).is_dir(),
    reason="REAL_ENGINE_DIR not set or not a directory",
)


@pytest.fixture
def real_trt_subprocess(tmp_path: Path) -> ModelBackendPort:
    """TRT subprocess using interactive binary or Python wrapper.

    Prefers REAL_INTERACTIVE_BIN (C++ with KV cache support).
    Falls back to REAL_WRAPPER_PATH (Python, no KV cache).
    """
    from agent_memory.adapters.outbound.trt_subprocess_adapter import TRTSubprocessAdapter

    shm = tmp_path / "shm"
    shm.mkdir()

    interactive = Path(INTERACTIVE_BIN) if INTERACTIVE_BIN else None
    if interactive and interactive.is_file():
        script = shm / "run_real.sh"
        script.write_text(f'#!/bin/sh\nexec {interactive} --engineDir {ENGINE_DIR} "$@"\n')
        script.chmod(0o755)
    else:
        wrapper = Path(WRAPPER_PATH) if WRAPPER_PATH else None
        if not wrapper or not wrapper.is_file():
            wrapper = Path(__file__).resolve().parents[2] / "vendor" / "llm_inference_wrapper.py"
        script = shm / "run_real.sh"
        script.write_text(
            f'#!/bin/sh\nexec {sys.executable} {wrapper} --engineDir {ENGINE_DIR} "$@"\n'
        )
        script.chmod(0o755)

    adapter = TRTSubprocessAdapter(
        llm_inference_bin=str(script),
        engine_path=ENGINE_DIR,
        timeout_s=60.0,
        shm_dir=str(shm),
    )
    adapter.start()
    yield adapter
    adapter.stop()


@skip_if_not_jetson
@skip_if_no_engine
class TestRealInference:
    """Tests using actual TRT engine on Thor."""

    def test_get_model_spec(self, real_trt_subprocess: ModelBackendPort) -> None:
        spec = real_trt_subprocess.extract_model_spec()
        assert isinstance(spec, ModelCacheSpec)
        assert spec.n_layers == 30
        assert spec.n_kv_heads == 3
        assert spec.head_dim == 64
        assert spec.kv_format == "fp"

    def test_generate(self, real_trt_subprocess: ModelBackendPort) -> None:
        result = real_trt_subprocess.generate(
            prompt_tokens=[1, 2, 3],
            max_tokens=16,
            temperature=0.7,
        )
        assert result.text
        assert len(result.text) > 0

    def test_generate_returns_cache(self, real_trt_subprocess: ModelBackendPort) -> None:
        """Generate should return extractable KV cache."""
        result = real_trt_subprocess.generate(
            prompt_tokens=[1],
            max_tokens=8,
        )
        assert result.text
        assert result.cache  # Should have per-layer KV pairs
        assert len(result.cache) > 0

    def test_cache_round_trip(self, real_trt_subprocess: ModelBackendPort) -> None:
        """Generate -> extract cache -> inject cache -> generate continuation."""
        # Turn 1: generate and get cache
        result1 = real_trt_subprocess.generate(
            prompt_tokens=[1],
            max_tokens=8,
        )
        assert result1.text
        assert result1.cache

        # Turn 2: inject turn 1 cache and generate more
        result2 = real_trt_subprocess.generate(
            prompt_tokens=[1],
            cache=result1.cache,
            max_tokens=8,
        )
        assert result2.text
        assert result2.cache
