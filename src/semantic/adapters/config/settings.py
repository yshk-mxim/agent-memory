"""Configuration management using Pydantic Settings."""

import logging
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from semantic.domain.entities import BLOCK_SIZE_TOKENS

_logger = logging.getLogger(__name__)


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
        default="mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx",
        description="HuggingFace model ID or local path",
    )

    max_context_length: int = Field(
        default=100000,
        ge=1024,
        le=163840,
        description="Maximum context length in tokens (tokenizer override)",
    )

    max_batch_size: int = Field(
        default=1,
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

    # Adaptive chunked prefill settings (memory-efficient long context)
    chunked_prefill_enabled: bool = Field(
        default=True,
        description="Enable adaptive chunked prefill for memory efficiency",
    )

    chunked_prefill_threshold: int = Field(
        default=2048,
        ge=512,
        le=16384,
        description="Minimum tokens to trigger chunked prefill (shorter prompts use standard prefill)",
    )

    chunked_prefill_min_chunk: int = Field(
        default=512,
        ge=256,
        le=2048,
        description="Minimum chunk size for chunked prefill (used for large cache positions)",
    )

    chunked_prefill_max_chunk: int = Field(
        default=2048,
        ge=1024,
        le=8192,
        description="Maximum chunk size for chunked prefill (used for small cache positions)",
    )

    kv_bits: int | None = Field(
        default=4,
        description="KV cache quantization (4 or 8 bits, None = FP16)",
    )

    kv_group_size: int = Field(
        default=64,
        ge=16,
        le=256,
        description="KV cache quantization group size (must be power of 2)",
    )

    @field_validator("kv_bits")
    @classmethod
    def validate_kv_bits(cls, v: int | None) -> int | None:
        """Validate kv_bits is 4, 8, or None."""
        if v is not None and v not in (4, 8):
            raise ValueError("kv_bits must be 4, 8, or None (FP16)")
        return v

    @field_validator("kv_group_size")
    @classmethod
    def validate_kv_group_size(cls, v: int) -> int:
        """Validate kv_group_size is a power of 2."""
        if v & (v - 1) != 0:
            raise ValueError("kv_group_size must be a power of 2")
        return v

    block_tokens: int = Field(
        default=BLOCK_SIZE_TOKENS,
        ge=64,
        le=512,
        description="Tokens per cache block (must match BlockPool)",
    )

    cache_budget_mb: int = Field(
        default=8192,
        ge=512,
        le=16384,
        description="Maximum cache memory budget in MB",
    )

    # Scheduler settings (interleaved prefill + decode)
    scheduler_enabled: bool = Field(
        default=False,
        description="Enable ConcurrentScheduler for interleaved prefill/decode",
    )

    scheduler_interleave_threshold: int = Field(
        default=2048,
        ge=256,
        le=32768,
        description="Min prompt tokens to use chunked interleaved prefill",
    )

    # Generation defaults
    default_max_tokens: int = Field(
        default=256,
        ge=1,
        le=65536,
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

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )

    cors_origins: str = Field(
        default="http://localhost:3000",
        description=(
            "Comma-separated list of allowed CORS origins "
            "(* for all, not recommended for production)"
        ),
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


def _find_model_profile_path(model_id: str) -> Path | None:
    """Locate model profile TOML by model_id.

    Searches config/models/ for a TOML file whose filename matches
    a slug derived from the model_id.
    """
    # settings.py is at src/semantic/adapters/config/ → parents[4] = project root
    config_dir = Path(__file__).resolve().parents[4] / "config" / "models"
    if not config_dir.is_dir():
        return None

    # Slug: last part of model_id, lowercased
    slug = model_id.rsplit("/", 1)[-1].lower()
    slug_parts = set(slug.split("-"))

    best_match: Path | None = None
    best_score = 0

    for toml_file in config_dir.glob("*.toml"):
        stem = toml_file.stem.lower()
        if slug == stem or slug in stem or stem in slug:
            return toml_file
        # Score by number of matching dash-separated parts
        stem_parts = set(stem.split("-"))
        overlap = len(slug_parts & stem_parts)
        if overlap > best_score and overlap >= 3:
            best_score = overlap
            best_match = toml_file

    return best_match


def load_model_profile(
    model_id: str | None = None,
    profile_path: str | None = None,
) -> dict[str, Any]:
    """Load a per-model configuration profile from TOML.

    Args:
        model_id: HuggingFace model ID (used to auto-discover profile).
        profile_path: Explicit path to a TOML profile file.

    Returns:
        Dict with 'model', 'optimal', 'thresholds', 'memory' sections.
        Empty dict if no profile found.
    """
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib  # type: ignore[no-redef]

    path: Path | None = None

    if profile_path:
        path = Path(profile_path)
    elif model_id:
        path = _find_model_profile_path(model_id)

    if path is None or not path.exists():
        _logger.debug(f"No model profile found for {model_id}")
        return {}

    with open(path, "rb") as f:
        profile = tomllib.load(f)

    _logger.info(f"Loaded model profile from {path}")
    return profile
