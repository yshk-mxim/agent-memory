"""Unit tests for Pydantic Settings configuration.

Tests configuration loading from environment variables, .env files,
and defaults with proper precedence and validation.
"""

import pytest

from semantic.adapters.config.settings import (
    AgentSettings,
    MLXSettings,
    ServerSettings,
    Settings,
    get_settings,
    reload_settings,
)


class TestMLXSettings:
    """Tests for MLX inference configuration."""

    def test_default_values(self) -> None:
        """Should load with documented default values."""
        settings = MLXSettings()

        assert settings.model_id == "mlx-community/gemma-3-12b-it-4bit"
        assert settings.max_batch_size == 2
        assert settings.prefill_step_size == 256
        assert settings.kv_bits == 4
        assert settings.block_tokens == 256
        assert settings.cache_budget_mb == 8192
        assert settings.default_max_tokens == 256
        assert settings.default_temperature == 0.7

    def test_load_from_env_vars(self, monkeypatch) -> None:
        """Should load from SEMANTIC_MLX_* environment variables."""
        monkeypatch.setenv("SEMANTIC_MLX_MODEL_ID", "test-model")
        monkeypatch.setenv("SEMANTIC_MLX_MAX_BATCH_SIZE", "10")
        monkeypatch.setenv("SEMANTIC_MLX_KV_BITS", "8")

        settings = MLXSettings()

        assert settings.model_id == "test-model"
        assert settings.max_batch_size == 10
        assert settings.kv_bits == 8

    def test_validation_max_batch_size_ge_1(self) -> None:
        """Should reject max_batch_size < 1."""
        with pytest.raises(ValueError, match="greater than or equal to 1"):
            MLXSettings(max_batch_size=0)

    def test_validation_max_batch_size_le_20(self) -> None:
        """Should reject max_batch_size > 20."""
        with pytest.raises(ValueError, match="less than or equal to 20"):
            MLXSettings(max_batch_size=21)

    def test_validation_kv_bits_range(self) -> None:
        """Should only allow kv_bits in {4, 8} or None."""
        # Valid
        assert MLXSettings(kv_bits=4).kv_bits == 4
        assert MLXSettings(kv_bits=8).kv_bits == 8
        assert MLXSettings(kv_bits=None).kv_bits is None

        # Invalid
        with pytest.raises(ValueError, match="kv_bits must be 4, 8, or None"):
            MLXSettings(kv_bits=3)

        with pytest.raises(ValueError, match="kv_bits must be 4, 8, or None"):
            MLXSettings(kv_bits=16)


class TestMLXSettingsKvBitsEdgeCases:
    """Edge cases for kv_bits string parsing from environment variables."""

    def test_empty_string_becomes_none(self, monkeypatch) -> None:
        """Empty env var resolves to None (FP16 mode)."""
        monkeypatch.setenv("SEMANTIC_MLX_KV_BITS", "")
        settings = MLXSettings()
        assert settings.kv_bits is None

    def test_string_none_becomes_none(self, monkeypatch) -> None:
        """String 'none' resolves to None (FP16 mode)."""
        monkeypatch.setenv("SEMANTIC_MLX_KV_BITS", "none")
        settings = MLXSettings()
        assert settings.kv_bits is None

    def test_string_null_becomes_none(self, monkeypatch) -> None:
        """String 'null' resolves to None (FP16 mode)."""
        monkeypatch.setenv("SEMANTIC_MLX_KV_BITS", "null")
        settings = MLXSettings()
        assert settings.kv_bits is None

    def test_string_zero_becomes_none(self, monkeypatch) -> None:
        """String '0' resolves to None (FP16 mode)."""
        monkeypatch.setenv("SEMANTIC_MLX_KV_BITS", "0")
        settings = MLXSettings()
        assert settings.kv_bits is None

    def test_integer_zero_becomes_none(self) -> None:
        """Integer 0 resolves to None (FP16 mode)."""
        settings = MLXSettings(kv_bits=0)
        assert settings.kv_bits is None

    def test_string_4_parses_correctly(self, monkeypatch) -> None:
        """String '4' from env var parses to int 4."""
        monkeypatch.setenv("SEMANTIC_MLX_KV_BITS", "4")
        settings = MLXSettings()
        assert settings.kv_bits == 4

    def test_string_8_parses_correctly(self, monkeypatch) -> None:
        """String '8' from env var parses to int 8."""
        monkeypatch.setenv("SEMANTIC_MLX_KV_BITS", "8")
        settings = MLXSettings()
        assert settings.kv_bits == 8

    def test_invalid_string_raises(self, monkeypatch) -> None:
        """Non-numeric, non-special string raises ValueError."""
        monkeypatch.setenv("SEMANTIC_MLX_KV_BITS", "invalid")
        with pytest.raises(ValueError):
            MLXSettings()


class TestMLXSettingsKvGroupSize:
    """Validation tests for kv_group_size (must be power of 2)."""

    def test_valid_power_of_2_values(self) -> None:
        """Powers of 2 within range are accepted."""
        for size in (16, 32, 64, 128, 256):
            settings = MLXSettings(kv_group_size=size)
            assert settings.kv_group_size == size

    def test_below_minimum_rejected(self) -> None:
        """Values below minimum (16) are rejected by range check."""
        for size in (3, 5, 8, 15):
            with pytest.raises(ValueError, match="greater than or equal to 16"):
                MLXSettings(kv_group_size=size)

    def test_non_power_of_2_in_range_rejected(self) -> None:
        """Non-power-of-2 values within [16, 256] are rejected."""
        for size in (17, 48, 63, 65, 100, 200):
            with pytest.raises(ValueError, match="power of 2"):
                MLXSettings(kv_group_size=size)

    def test_from_env_var(self, monkeypatch) -> None:
        """kv_group_size loaded from env var."""
        monkeypatch.setenv("SEMANTIC_MLX_KV_GROUP_SIZE", "128")
        settings = MLXSettings()
        assert settings.kv_group_size == 128

    def test_invalid_from_env_var(self, monkeypatch) -> None:
        """Non-power-of-2 from env var is rejected."""
        monkeypatch.setenv("SEMANTIC_MLX_KV_GROUP_SIZE", "100")
        with pytest.raises(ValueError, match="power of 2"):
            MLXSettings()


class TestServerSettingsPortEdgeCases:
    """Edge cases for port validation."""

    def test_port_99999_rejected(self) -> None:
        """Port 99999 exceeds max (65535)."""
        with pytest.raises(ValueError, match="less than or equal to 65535"):
            ServerSettings(port=99999)

    def test_port_from_env_var(self, monkeypatch) -> None:
        """Port loaded from env var and validated."""
        monkeypatch.setenv("SEMANTIC_SERVER_PORT", "9000")
        settings = ServerSettings()
        assert settings.port == 9000

    def test_port_0_rejected(self) -> None:
        """Port 0 is below minimum (1024)."""
        with pytest.raises(ValueError, match="greater than or equal to 1024"):
            ServerSettings(port=0)

    def test_port_boundary_values(self) -> None:
        """Ports 1023 and 65536 are just outside valid range."""
        with pytest.raises(ValueError, match="greater than or equal to 1024"):
            ServerSettings(port=1023)
        with pytest.raises(ValueError, match="less than or equal to 65535"):
            ServerSettings(port=65536)


class TestAgentSettings:
    """Tests for agent cache management configuration."""

    def test_default_values(self) -> None:
        """Should load with documented default values."""
        settings = AgentSettings()

        assert settings.max_agents_in_memory == 12
        assert settings.cache_dir == "~/.semantic/caches"
        assert settings.batch_window_ms == 10
        assert settings.lru_eviction_enabled is True
        assert settings.evict_to_disk is True
        assert settings.validate_model_tag is True

    def test_load_from_env_vars(self, monkeypatch) -> None:
        """Should load from SEMANTIC_AGENT_* environment variables."""
        monkeypatch.setenv("SEMANTIC_AGENT_MAX_AGENTS_IN_MEMORY", "10")
        monkeypatch.setenv("SEMANTIC_AGENT_CACHE_DIR", "/tmp/caches")
        monkeypatch.setenv("SEMANTIC_AGENT_BATCH_WINDOW_MS", "50")

        settings = AgentSettings()

        assert settings.max_agents_in_memory == 10
        assert settings.cache_dir == "/tmp/caches"
        assert settings.batch_window_ms == 50

    def test_validation_max_agents_ge_1(self) -> None:
        """Should reject max_agents < 1."""
        with pytest.raises(ValueError, match="greater than or equal to 1"):
            AgentSettings(max_agents_in_memory=0)

    def test_validation_batch_window_range(self) -> None:
        """Should enforce batch_window_ms range [1, 1000]."""
        # Valid
        assert AgentSettings(batch_window_ms=1).batch_window_ms == 1
        assert AgentSettings(batch_window_ms=1000).batch_window_ms == 1000

        # Invalid
        with pytest.raises(ValueError, match="greater than or equal to 1"):
            AgentSettings(batch_window_ms=0)

        with pytest.raises(ValueError, match="less than or equal to 1000"):
            AgentSettings(batch_window_ms=1001)


class TestServerSettings:
    """Tests for HTTP server configuration."""

    def test_default_values(self) -> None:
        """Should load with documented default values."""
        settings = ServerSettings()

        assert settings.host == "0.0.0.0"  # noqa: S104
        assert settings.port == 8000
        assert settings.workers == 1
        assert settings.log_level == "INFO"

    def test_load_from_env_vars(self, monkeypatch) -> None:
        """Should load from SEMANTIC_SERVER_* environment variables."""
        monkeypatch.setenv("SEMANTIC_SERVER_HOST", "127.0.0.1")
        monkeypatch.setenv("SEMANTIC_SERVER_PORT", "9000")
        monkeypatch.setenv("SEMANTIC_SERVER_LOG_LEVEL", "DEBUG")

        settings = ServerSettings()

        assert settings.host == "127.0.0.1"
        assert settings.port == 9000
        assert settings.log_level == "DEBUG"

    def test_validation_port_range(self) -> None:
        """Should enforce port range [1024, 65535]."""
        # Valid
        assert ServerSettings(port=1024).port == 1024
        assert ServerSettings(port=65535).port == 65535

        # Invalid
        with pytest.raises(ValueError, match="greater than or equal to 1024"):
            ServerSettings(port=80)

        with pytest.raises(ValueError, match="less than or equal to 65535"):
            ServerSettings(port=70000)


class TestRootSettings:
    """Tests for root Settings container."""

    def test_default_aggregation(self) -> None:
        """Should aggregate all subsettings with defaults."""
        settings = Settings()

        # Verify all subsettings present
        assert isinstance(settings.mlx, MLXSettings)
        assert isinstance(settings.agent, AgentSettings)
        assert isinstance(settings.server, ServerSettings)

        # Verify defaults propagate
        assert settings.mlx.model_id == "mlx-community/gemma-3-12b-it-4bit"
        assert settings.agent.max_agents_in_memory == 12
        assert settings.server.port == 8000

    def test_load_mixed_env_vars(self, monkeypatch) -> None:
        """Should load env vars for different subsettings simultaneously."""
        monkeypatch.setenv("SEMANTIC_MLX_MODEL_ID", "test-model")
        monkeypatch.setenv("SEMANTIC_AGENT_CACHE_DIR", "/custom/cache")
        monkeypatch.setenv("SEMANTIC_SERVER_PORT", "9000")

        settings = Settings()

        assert settings.mlx.model_id == "test-model"
        assert settings.agent.cache_dir == "/custom/cache"
        assert settings.server.port == 9000

    def test_get_settings_singleton(self) -> None:
        """get_settings() should return same instance."""
        settings1 = get_settings()
        settings2 = get_settings()

        assert settings1 is settings2

    def test_reload_settings_creates_new_instance(self, monkeypatch) -> None:
        """reload_settings() should create fresh instance."""
        settings1 = get_settings()

        # Change environment
        monkeypatch.setenv("SEMANTIC_MLX_MODEL_ID", "new-model")

        # Reload
        settings2 = reload_settings()

        assert settings2 is not settings1
        assert settings2.mlx.model_id == "new-model"


class TestConfigurationPrecedence:
    """Tests for configuration precedence rules."""

    def test_env_var_overrides_default(self, monkeypatch) -> None:
        """Environment variables should override defaults."""
        monkeypatch.setenv("SEMANTIC_MLX_BLOCK_TOKENS", "512")

        settings = Settings()

        # ENV var wins over default (256)
        assert settings.mlx.block_tokens == 512

    def test_explicit_param_overrides_env(self, monkeypatch) -> None:
        """Explicit constructor params should override env vars."""
        monkeypatch.setenv("SEMANTIC_MLX_BLOCK_TOKENS", "512")

        # Direct instantiation with explicit param
        mlx = MLXSettings(block_tokens=128)

        # Explicit param wins
        assert mlx.block_tokens == 128


class TestDotEnvLoading:
    """Tests for .env file loading."""

    def test_loads_from_dotenv_if_exists(self, tmp_path, monkeypatch) -> None:
        """Should load configuration from .env file."""
        # Create temporary .env file
        env_file = tmp_path / ".env"
        env_file.write_text(
            "SEMANTIC_MLX_MODEL_ID=dotenv-model\nSEMANTIC_AGENT_MAX_AGENTS_IN_MEMORY=15\n"
        )

        # Change to temp directory
        monkeypatch.chdir(tmp_path)

        # Reload settings to pick up .env
        settings = Settings()

        assert settings.mlx.model_id == "dotenv-model"
        assert settings.agent.max_agents_in_memory == 15

    def test_env_var_overrides_dotenv(self, tmp_path, monkeypatch) -> None:
        """Environment variables should override .env file."""
        # Create .env with value
        env_file = tmp_path / ".env"
        env_file.write_text("SEMANTIC_MLX_MODEL_ID=dotenv-model\n")

        monkeypatch.chdir(tmp_path)

        # Set env var to different value
        monkeypatch.setenv("SEMANTIC_MLX_MODEL_ID", "env-model")

        settings = Settings()

        # ENV var wins over .env file
        assert settings.mlx.model_id == "env-model"
