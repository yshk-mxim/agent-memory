# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Yakov Shkolnikov and contributors
"""TRT integration test fixtures.

Follows the hexagonal architecture: tests use fake port implementations
(mocks satisfying protocols) by default.  When REAL_LLM_INFERENCE_BIN is
set, the real subprocess binary is used for hardware validation on Thor.

Default (Mac / CI):
    - Fake ModelBackendPort via TRTSubprocessAdapter + fake_llm_inference.py
    - Real HuggingFace tokenizer (SmolLM2-135M-Instruct)
    - Real numpy-based TRTCacheAdapter and TRTQuantizationAdapter

Thor hardware:
    - Real llm_inference binary via TRTSubprocessAdapter
    - Real HuggingFace tokenizer (Qwen3-Coder-Next)
"""

import os
import platform
import sys
from pathlib import Path

import numpy as np
import pytest

from agent_memory.domain.value_objects import ModelCacheSpec
from agent_memory.ports.outbound import ModelBackendPort

# SmolLM2-135M geometry
SMOLLM2_N_LAYERS = 30
SMOLLM2_N_KV_HEADS = 3
SMOLLM2_HEAD_DIM = 64
SMOLLM2_BLOCK_TOKENS = 256


def is_jetson() -> bool:
    """Check if running on NVIDIA Jetson (aarch64 Linux)."""
    return sys.platform == "linux" and platform.machine() == "aarch64"


skip_if_not_jetson = pytest.mark.skipif(
    not is_jetson(),
    reason="Requires NVIDIA Jetson (aarch64 Linux) for real TRT inference",
)


@pytest.fixture
def trt_model_spec() -> ModelCacheSpec:
    """ModelCacheSpec for TRT backend with SmolLM2-135M geometry."""
    return ModelCacheSpec(
        n_layers=SMOLLM2_N_LAYERS,
        n_kv_heads=SMOLLM2_N_KV_HEADS,
        head_dim=SMOLLM2_HEAD_DIM,
        block_tokens=SMOLLM2_BLOCK_TOKENS,
        layer_types=["global"] * SMOLLM2_N_LAYERS,
        kv_format="fp",
        kv_bits=None,  # FP16 on GPU
    )


@pytest.fixture
def trt_cache_adapter():
    """Real TRT cache adapter (numpy, no CUDA needed)."""
    from agent_memory.adapters.outbound.trt_cache_adapter import TRTCacheAdapter

    return TRTCacheAdapter()


@pytest.fixture
def trt_quantizer():
    """Real TRT quantization adapter (numpy, no CUDA needed)."""
    from agent_memory.adapters.outbound.trt_quantization_adapter import TRTQuantizationAdapter

    return TRTQuantizationAdapter()


@pytest.fixture
def fake_fp16_cache() -> list[tuple[np.ndarray, np.ndarray]]:
    """Fake FP16 KV cache matching SmolLM2-135M geometry.

    Returns per-layer (K, V) tuples with shape [n_kv_heads, seq_len, head_dim].
    """
    rng = np.random.default_rng(42)
    seq_len = 64
    cache = []
    for _ in range(SMOLLM2_N_LAYERS):
        k = rng.standard_normal((SMOLLM2_N_KV_HEADS, seq_len, SMOLLM2_HEAD_DIM)).astype(np.float16)
        v = rng.standard_normal((SMOLLM2_N_KV_HEADS, seq_len, SMOLLM2_HEAD_DIM)).astype(np.float16)
        cache.append((k, v))
    return cache


@pytest.fixture
def cache_dir(tmp_path: Path) -> Path:
    """Temporary cache directory for disk persistence."""
    d = tmp_path / "caches"
    d.mkdir()
    return d


@pytest.fixture
def shm_dir(tmp_path: Path) -> Path:
    """Temporary shared memory directory."""
    d = tmp_path / "shm"
    d.mkdir()
    return d


@pytest.fixture
def fake_trt_subprocess(shm_dir: Path) -> ModelBackendPort:
    """TRT subprocess adapter using fake_llm_inference.py.

    Test wiring point: creates concrete adapter, returns it typed as
    ModelBackendPort. The adapter implements the port protocol.
    """
    from agent_memory.adapters.outbound.trt_subprocess_adapter import TRTSubprocessAdapter

    fake_bin = Path(__file__).resolve().parents[1] / "fixtures" / "fake_llm_inference.py"

    # Create wrapper shell script that invokes python with the fake
    wrapper = shm_dir / "run_fake.sh"
    wrapper.write_text(f"#!/bin/sh\nexec {sys.executable} {fake_bin}\n")
    wrapper.chmod(0o755)

    # Configure fake to use SmolLM2 geometry
    env_patch = {
        "FAKE_N_LAYERS": str(SMOLLM2_N_LAYERS),
        "FAKE_N_KV_HEADS": str(SMOLLM2_N_KV_HEADS),
        "FAKE_HEAD_DIM": str(SMOLLM2_HEAD_DIM),
    }
    original_env = {}
    for k, v in env_patch.items():
        original_env[k] = os.environ.get(k)
        os.environ[k] = v

    adapter = TRTSubprocessAdapter(
        llm_inference_bin=str(wrapper),
        engine_path="/fake/engine",
        timeout_s=10.0,
        shm_dir=str(shm_dir),
    )
    adapter.start()

    yield adapter

    adapter.stop()

    for k, v in original_env.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
