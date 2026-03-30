# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Yakov Shkolnikov and contributors
"""Unit tests for TRTSettings and backend selector."""

import pytest

from agent_memory.adapters.config.settings import Settings, TRTSettings

pytestmark = pytest.mark.unit


class TestTRTSettingsDefaults:
    """Test TRTSettings default values."""

    def test_default_values(self) -> None:
        settings = TRTSettings()
        assert settings.engine_path == "/opt/trt-edge-llm/engines/qwen3-coder-next"
        assert settings.llm_inference_bin == "/opt/trt-edge-llm/bin/llm_inference"
        assert settings.model_id == "Qwen/Qwen3-Coder-Next-nvfp4"
        assert settings.max_context_length == 65536
        assert settings.max_batch_size == 1
        assert settings.kv_bits is None  # FP16 on GPU
        assert settings.kv_group_size == 64
        assert settings.disk_kv_bits == 4
        assert settings.subprocess_timeout_s == 30.0
        assert settings.shm_dir == "/dev/shm"
        assert settings.block_tokens == 256
        assert settings.cache_budget_mb == 16384
        assert settings.default_max_tokens == 256
        assert settings.default_temperature == 0.7


class TestTRTSettingsEnvVars:
    """Test TRTSettings loading from environment variables."""

    def test_load_from_env(self, monkeypatch) -> None:
        monkeypatch.setenv("SEMANTIC_TRT_ENGINE_PATH", "/custom/path")
        monkeypatch.setenv("SEMANTIC_TRT_MODEL_ID", "custom-model")
        monkeypatch.setenv("SEMANTIC_TRT_MAX_BATCH_SIZE", "4")
        monkeypatch.setenv("SEMANTIC_TRT_SUBPROCESS_TIMEOUT_S", "60.0")

        settings = TRTSettings()
        assert settings.engine_path == "/custom/path"
        assert settings.model_id == "custom-model"
        assert settings.max_batch_size == 4
        assert settings.subprocess_timeout_s == 60.0

    def test_kv_bits_env_none(self, monkeypatch) -> None:
        """'none' string maps to None (FP16)."""
        monkeypatch.setenv("SEMANTIC_TRT_KV_BITS", "none")
        settings = TRTSettings()
        assert settings.kv_bits is None

    def test_kv_bits_env_fp8(self, monkeypatch) -> None:
        """'8' maps to FP8."""
        monkeypatch.setenv("SEMANTIC_TRT_KV_BITS", "8")
        settings = TRTSettings()
        assert settings.kv_bits == 8

    def test_kv_bits_env_16_maps_to_none(self, monkeypatch) -> None:
        """'16' maps to None (FP16 = no quantization)."""
        monkeypatch.setenv("SEMANTIC_TRT_KV_BITS", "16")
        settings = TRTSettings()
        assert settings.kv_bits is None


class TestTRTSettingsValidation:
    """Test TRTSettings validation rules."""

    def test_reject_kv_bits_4(self) -> None:
        """TRT doesn't support Q4 on GPU — only FP16 or FP8."""
        with pytest.raises(ValueError, match="TRT kv_bits must be None"):
            TRTSettings(kv_bits=4)

    def test_reject_kv_bits_arbitrary(self) -> None:
        with pytest.raises(ValueError, match="TRT kv_bits must be None"):
            TRTSettings(kv_bits=6)

    def test_reject_non_power_of_2_group_size(self) -> None:
        with pytest.raises(ValueError, match="power of 2"):
            TRTSettings(kv_group_size=100)

    def test_accept_power_of_2_group_sizes(self) -> None:
        for gs in [16, 32, 64, 128, 256]:
            settings = TRTSettings(kv_group_size=gs)
            assert settings.kv_group_size == gs

    def test_reject_batch_size_too_large(self) -> None:
        with pytest.raises(Exception):
            TRTSettings(max_batch_size=100)

    def test_reject_timeout_too_small(self) -> None:
        with pytest.raises(Exception):
            TRTSettings(subprocess_timeout_s=0.1)


class TestBackendSelector:
    """Test backend field in root Settings."""

    def test_default_backend_is_mlx(self) -> None:
        settings = Settings()
        assert settings.backend == "mlx"

    def test_backend_trt_from_env(self, monkeypatch) -> None:
        monkeypatch.setenv("SEMANTIC_BACKEND", "trt")
        settings = Settings()
        assert settings.backend == "trt"

    def test_backend_mlx_from_env(self, monkeypatch) -> None:
        monkeypatch.setenv("SEMANTIC_BACKEND", "mlx")
        settings = Settings()
        assert settings.backend == "mlx"

    def test_reject_invalid_backend(self, monkeypatch) -> None:
        monkeypatch.setenv("SEMANTIC_BACKEND", "invalid_backend")
        with pytest.raises(Exception):
            Settings()

    def test_backend_vllm_from_env(self, monkeypatch) -> None:
        monkeypatch.setenv("SEMANTIC_BACKEND", "vllm")
        settings = Settings()
        assert settings.backend == "vllm"

    def test_trt_settings_accessible(self) -> None:
        settings = Settings()
        assert settings.trt is not None
        assert isinstance(settings.trt, TRTSettings)

    def test_trt_settings_independent_of_mlx(self, monkeypatch) -> None:
        """TRT and MLX settings don't interfere with each other."""
        monkeypatch.setenv("SEMANTIC_TRT_MODEL_ID", "trt-model")
        monkeypatch.setenv("SEMANTIC_MLX_MODEL_ID", "mlx-model")
        settings = Settings()
        assert settings.trt.model_id == "trt-model"
        assert settings.mlx.model_id == "mlx-model"
