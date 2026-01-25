"""Configuration management using Pydantic Settings."""


from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from semantic.domain.entities import BLOCK_SIZE_TOKENS


class MLXSettings(BaseSettings):
    """MLX inference engine configuration.

    Controls model loading, batch processing, and cache parameters.
    """

    model_config = SettingsConfigDict(
        env_prefix="SEMANTIC_MLX_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Model configuration
    model_id: str = Field(
        default="mlx-community/gemma-3-12b-it-4bit",
        description="HuggingFace model ID or local path",
    )

    max_batch_size: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of concurrent sequences in batch",
    )

    prefill_step_size: int = Field(
        default=512,
        ge=128,
        le=2048,
        description="Tokens per prefill step (larger = faster prefill, more memory)",
    )

    kv_bits: int | None = Field(
        default=None,
        ge=4,
        le=8,
        description="KV cache quantization (4 or 8 bits, None = FP16)",
    )

    block_tokens: int = Field(
        default=BLOCK_SIZE_TOKENS,
        ge=64,
        le=512,
        description="Tokens per cache block (must match BlockPool)",
    )

    cache_budget_mb: int = Field(
        default=4096,
        ge=512,
        le=16384,
        description="Maximum cache memory budget in MB",
    )

    # Generation defaults
    default_max_tokens: int = Field(
        default=256,
        ge=1,
        le=8192,
        description="Default max tokens for generation if not specified",
    )

    default_temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Default sampling temperature",
    )


class AgentSettings(BaseSettings):
    """Agent cache management configuration.

    Controls cache lifecycle, eviction, and persistence.
    """

    model_config = SettingsConfigDict(
        env_prefix="SEMANTIC_AGENT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    max_agents_in_memory: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Maximum agents with hot caches in memory",
    )

    cache_dir: str = Field(
        default="~/.semantic/caches",
        description="Directory for persistent cache storage",
    )

    batch_window_ms: int = Field(
        default=10,
        ge=1,
        le=1000,
        description="Batch collection window in milliseconds",
    )

    # Cache eviction policy
    lru_eviction_enabled: bool = Field(
        default=True,
        description="Enable LRU eviction when max_agents exceeded",
    )

    evict_to_disk: bool = Field(
        default=True,
        description="Persist evicted caches to disk (warm tier)",
    )

    # Cache validation
    validate_model_tag: bool = Field(
        default=True,
        description="Validate cache compatibility with current model",
    )


class ServerSettings(BaseSettings):
    """HTTP server configuration."""

    model_config = SettingsConfigDict(
        env_prefix="SEMANTIC_SERVER_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    host: str = Field(
        default="0.0.0.0",
        description="Server bind address",
    )

    port: int = Field(
        default=8000,
        ge=1024,
        le=65535,
        description="Server port",
    )

    workers: int = Field(
        default=1,
        ge=1,
        le=4,
        description="Number of worker processes (MLX limits concurrency)",
    )

    rate_limit_per_agent: int = Field(
        default=60,
        ge=1,
        le=1000,
        description="Maximum requests per agent per minute",
    )

    rate_limit_global: int = Field(
        default=1000,
        ge=1,
        le=10000,
        description="Maximum global requests per minute",
    )

    log_level: str = Field(
        default="INFO",
        description="Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )


class SecretsSettings(BaseSettings):
    """Sensitive configuration (API keys, tokens).

    Loaded from environment variables only (never from TOML files).
    """

    model_config = SettingsConfigDict(
        env_prefix="SEMANTIC_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    api_key: SecretStr = Field(
        default=SecretStr(""),
        description="Optional API key for authentication",
    )


class Settings(BaseSettings):
    """Root settings container.

    Aggregates all subsettings into a single object.

    Example:
        >>> settings = Settings()
        >>> settings.mlx.model_id
        'mlx-community/gemma-3-12b-it-4bit'
        >>> settings.agent.cache_dir
        '~/.semantic/caches'
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    mlx: MLXSettings = Field(default_factory=MLXSettings)
    agent: AgentSettings = Field(default_factory=AgentSettings)
    server: ServerSettings = Field(default_factory=ServerSettings)
    secrets: SecretsSettings = Field(default_factory=SecretsSettings)


# Singleton instance
_settings: Settings | None = None


def get_settings() -> Settings:
    """Get or create settings singleton.

    Loads configuration from environment variables and .env file.

    Returns:
        Settings instance.

    Example:
        >>> settings = get_settings()
        >>> print(settings.mlx.model_id)
        mlx-community/gemma-3-12b-it-4bit
    """
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reload_settings() -> Settings:
    """Reload settings (for testing).

    Forces reload of configuration from environment.

    Returns:
        Fresh Settings instance.
    """
    global _settings
    _settings = Settings()
    return _settings
