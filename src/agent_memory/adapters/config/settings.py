# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Yakov Shkolnikov and contributors
"""Configuration management using Pydantic Settings."""

import logging
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from agent_memory.domain.entities import BLOCK_SIZE_TOKENS

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
        default="mlx-community/gemma-3-12b-it-4bit",
        description="HuggingFace model ID or local path (override with SEMANTIC_MLX_MODEL_ID)",
    )

    max_context_length: int = Field(
        default=100000,
        ge=1024,
        le=163840,
        description="Maximum context length in tokens (tokenizer override)",
    )

    max_batch_size: int = Field(
        default=2,
        ge=1,
        le=20,
        description="Maximum number of concurrent sequences in batch",
    )

    prefill_step_size: int = Field(
        default=256,
        ge=128,
        le=2048,
        description="Tokens per prefill step (smaller = lower TTFT for queued requests)",
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
        description=(
            "Minimum tokens to trigger chunked prefill (shorter prompts use standard prefill)"
        ),
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

    reasoning_extra_tokens: int = Field(
        default=0,
        ge=0,
        le=1000,
        description=(
            "Extra tokens for reasoning models that generate "
            "chain-of-thought before final response. Default 0 — only set "
            "explicitly for models that support structured reasoning."
        ),
    )

    @field_validator("kv_bits", mode="before")
    @classmethod
    def validate_kv_bits(cls, v: int | str | None) -> int | None:
        """Validate kv_bits is 4, 8, or None (FP16).

        Accepts "none" or "0" from environment variables to disable quantization.
        """
        if isinstance(v, str):
            v = v.strip().lower()
            if v in ("none", "null", "", "0"):
                return None
            return int(v)
        if v == 0:
            return None
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
        default=True,
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


class TRTSettings(BaseSettings):
    """TensorRT Edge-LLM inference engine configuration.

    Controls TRT subprocess management, model paths, and KV cache format
    for NVIDIA Jetson AGX Thor (aarch64, sm_110).
    """

    model_config = SettingsConfigDict(
        env_prefix="SEMANTIC_TRT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    engine_path: str = Field(
        default="/opt/trt-edge-llm/engines/qwen3-coder-next",
        description="Path to the TRT engine directory",
    )

    llm_inference_bin: str = Field(
        default="/opt/trt-edge-llm/bin/llm_inference",
        description="Path to the llm_inference binary",
    )

    model_id: str = Field(
        default="Qwen/Qwen3-Coder-Next-nvfp4",
        description="HuggingFace model ID (for tokenizer loading)",
    )

    max_context_length: int = Field(
        default=65536,
        ge=1024,
        le=262144,
        description="Maximum context length in tokens",
    )

    max_batch_size: int = Field(
        default=1,
        ge=1,
        le=8,
        description="Maximum batch size (TRT subprocess handles 1 at a time)",
    )

    kv_bits: int | None = Field(
        default=None,
        description="KV cache bits on GPU (None=FP16, 8=FP8). TRT uses native FP.",
    )

    kv_group_size: int = Field(
        default=64,
        ge=16,
        le=256,
        description="Quantization group size for disk format (Q4/Q8 safetensors)",
    )

    disk_kv_bits: int = Field(
        default=4,
        ge=4,
        le=8,
        description="KV cache quantization bits for disk storage (Q4 saves 72%% vs FP16)",
    )

    subprocess_timeout_s: float = Field(
        default=30.0,
        ge=1.0,
        le=300.0,
        description="Timeout in seconds for subprocess commands",
    )

    shm_dir: str = Field(
        default="/dev/shm",  # noqa: S108
        description="Shared memory directory for KV cache temp files",
    )

    block_tokens: int = Field(
        default=BLOCK_SIZE_TOKENS,
        ge=64,
        le=512,
        description="Tokens per cache block (must match BlockPool)",
    )

    cache_budget_mb: int = Field(
        default=16384,
        ge=512,
        le=65536,
        description="Maximum cache memory budget in MB (Thor has 128GB unified)",
    )

    default_max_tokens: int = Field(
        default=256,
        ge=1,
        le=65536,
        description="Default max tokens for generation",
    )

    default_temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Default sampling temperature",
    )

    default_top_p: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="Default top-p (nucleus) sampling parameter",
    )

    default_top_k: int = Field(
        default=40,
        ge=1,
        le=1000,
        description="Default top-k sampling parameter",
    )

    edgellm_plugin_path: str = Field(
        default="",
        description="Path to libNvInfer_edgellm_plugin.so (set via EDGELLM_PLUGIN_PATH env)",
    )

    reasoning_extra_tokens: int = Field(
        default=0,
        ge=0,
        le=1000,
        description="Extra tokens for reasoning models (same as MLX setting)",
    )

    @field_validator("kv_bits", mode="before")
    @classmethod
    def validate_kv_bits(cls, v: int | str | None) -> int | None:
        """Validate kv_bits for TRT (None = FP16, 8 = FP8)."""
        if isinstance(v, str):
            v = v.strip().lower()
            if v in ("none", "null", "", "0", "16"):
                return None
            return int(v)
        if v in {0, 16}:
            return None
        if v is not None and v not in {8}:
            raise ValueError("TRT kv_bits must be None (FP16) or 8 (FP8)")
        return v

    @field_validator("kv_group_size")
    @classmethod
    def validate_kv_group_size(cls, v: int) -> int:
        """Validate kv_group_size is a power of 2."""
        if v & (v - 1) != 0:
            raise ValueError("kv_group_size must be a power of 2")
        return v


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
        default=12,
        ge=1,
        le=50,
        description="Maximum agents with hot caches in memory",
    )

    cache_dir: str = Field(
        default="~/.agent_memory/caches",
        description="Directory for persistent cache storage",
    )

    batch_window_ms: int = Field(
        default=10,
        ge=1,
        le=1000,
        description="Batch collection window in milliseconds",
    )

    # Memory and disk budgets (cross-platform)
    max_memory_mb: int = Field(
        default=0,
        ge=0,
        le=524288,
        description=(
            "Maximum memory (RAM/VRAM/unified) for all caches in MB. "
            "0 = no limit (use backend cache_budget_mb instead). "
            "On unified memory systems (Apple Silicon, Jetson), this is the "
            "combined GPU+CPU budget."
        ),
    )

    max_disk_mb: int = Field(
        default=0,
        ge=0,
        le=10485760,
        description=(
            "Maximum disk usage for warm/cold cache files in MB. "
            "0 = no limit. Eviction deletes oldest caches when exceeded."
        ),
    )

    # Cache eviction policy
    eviction_policy: Literal["lru", "lfu", "lru-lfu"] = Field(
        default="lru-lfu",
        description=(
            "Cache eviction policy: 'lru' (least recently used), "
            "'lfu' (least frequently used), or 'lru-lfu' (hybrid — "
            "keeps both frequently-used system prompts and recently-used "
            "conversation caches warm). Hybrid recommended for NemoClaw/Claude Code."
        ),
    )

    pin_system_prompt_caches: bool = Field(
        default=True,
        description=(
            "Pin system prompt KV caches in memory (never evict). "
            "System prompts are expensive to recompute (~2-18K tokens) "
            "and shared across conversations."
        ),
    )

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

    searxng_url: str = Field(
        default="",
        description=(
            "SearXNG base URL for /search proxy endpoint. "
            "E.g. http://localhost:8080. Leave empty to disable."
        ),
    )

    jina_reader_url: str = Field(
        default="",
        description=(
            "Jina Reader base URL for /fetch proxy endpoint. "
            "E.g. http://localhost:3000. Converts web pages to clean markdown. "
            "Leave empty to disable."
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


class VLLMSettings(BaseSettings):
    """vLLM backend configuration.

    Connects to an externally-running vLLM server via OpenAI-compatible API.
    """

    model_config = SettingsConfigDict(
        env_prefix="SEMANTIC_VLLM_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    base_url: str = Field(
        default="http://localhost:5000",
        description="vLLM server URL",
    )

    model_id: str = Field(
        default="nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4",
        description="Model name for API requests",
    )

    timeout_s: float = Field(
        default=120.0,
        ge=5.0,
        le=600.0,
        description="HTTP request timeout in seconds",
    )

    max_context_length: int = Field(
        default=262144,
        ge=1024,
        le=1048576,
        description="Maximum context length in tokens",
    )


class LlamaCppSettings(BaseSettings):
    """llama.cpp backend configuration.

    Connects to an externally-running llama-server via OpenAI-compatible API.
    Adds slot-level KV cache save/restore for session persistence.
    """

    model_config = SettingsConfigDict(
        env_prefix="SEMANTIC_LLAMACPP_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    base_url: str = Field(
        default="http://localhost:8001",
        description="llama-server URL",
    )

    model_id: str = Field(
        default="qwen3-coder-next",
        description="Model name sent in API requests (can be any string llama-server accepts)",
    )

    tokenizer_id: str = Field(
        default="",
        description=(
            "HuggingFace model ID to load the tokenizer from. "
            "Defaults to model_id when empty. "
            "Set this when model_id is a GGUF-only repo (e.g. unsloth/*-GGUF) "
            "that has no HuggingFace tokenizer files — point it at the base model instead, "
            "e.g. SEMANTIC_LLAMACPP_TOKENIZER_ID=Qwen/Qwen2.5-Coder-32B-Instruct."
        ),
    )

    timeout_s: float = Field(
        default=600.0,
        ge=5.0,
        le=1200.0,
        description="HTTP request timeout in seconds (600s needed for dense 31B at 10 t/s)",
    )

    max_context_length: int = Field(
        default=65536,
        ge=1024,
        le=1048576,
        description="Maximum context length in tokens",
    )

    slot_save_path: str = Field(
        default="~/.agent_memory/llamacpp_slots",
        description="Directory for slot KV cache files (mirrors --slot-save-path)",
    )

    n_slots: int = Field(
        default=4,
        ge=1,
        le=64,
        description="Number of parallel slots (mirrors --parallel)",
    )

    max_slot_disk_mb: int = Field(
        default=0,
        ge=0,
        description="Max disk for slot cache files in MB (0 = no limit).",
    )

    cache_type_k: str = Field(
        default="q8_0",
        description="KV cache quantization for keys (mirrors --cache-type-k)",
    )

    cache_type_v: str = Field(
        default="q8_0",
        description="KV cache quantization for values (mirrors --cache-type-v)",
    )

    # Server process management (agent-memory starts/stops llama-server)
    server_binary: str = Field(
        default="llama-server",
        description=(
            "Path to llama-server binary. "
            "When set, agent-memory manages the process lifecycle for model swapping. "
            "Leave as 'llama-server' to use PATH, or set absolute path."
        ),
    )

    default_model: str = Field(
        default="",
        description=(
            "Model ID to load on startup (must match a [llamacpp] section in config/models/*.toml). "
            "Empty = use model_id field as a static single-model config."
        ),
    )

    auto_swap: bool = Field(
        default=True,
        description=(
            "Automatically swap models when a request specifies a different model "
            "than the one currently loaded. Set to false to require explicit "
            "/admin/models/swap calls."
        ),
    )

    capture_traffic: str = Field(
        default="",
        description=(
            "Path to JSONL file for raw traffic capture. "
            "Records messages + raw model output before any processing. "
            "Use for building parser regression test fixtures. "
            "Empty = disabled."
        ),
    )


class Settings(BaseSettings):
    """Root settings container.

    Aggregates all subsettings into a single object.
    Set ``SEMANTIC_BACKEND`` to ``"mlx"``, ``"trt"``, ``"vllm"``, or
    ``"llamacpp"`` to select the inference backend.

    Example:
        >>> settings = Settings()
        >>> settings.mlx.model_id
        'mlx-community/gemma-3-12b-it-4bit'
        >>> settings.agent.cache_dir
        '~/.agent_memory/caches'
    """

    model_config = SettingsConfigDict(
        env_prefix="SEMANTIC_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    backend: Literal["mlx", "trt", "vllm", "llamacpp"] = Field(
        default="mlx",
        description=(
            "Inference backend: 'mlx' (Apple Silicon), 'trt' (TensorRT Edge-LLM), "
            "'vllm' (external vLLM server), or 'llamacpp' (llama-server)"
        ),
    )

    mlx: MLXSettings = Field(default_factory=MLXSettings)
    trt: TRTSettings = Field(default_factory=TRTSettings)
    vllm: VLLMSettings = Field(default_factory=VLLMSettings)
    llamacpp: LlamaCppSettings = Field(default_factory=LlamaCppSettings)
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
    # settings.py is at src/agent_memory/adapters/config/ → parents[4] = project root
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
        min_overlap = 3
        if overlap > best_score and overlap >= min_overlap:
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

    with path.open("rb") as f:
        profile = tomllib.load(f)

    _logger.info(f"Loaded model profile from {path}")
    return profile
