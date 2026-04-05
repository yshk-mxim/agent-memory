# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Yakov Shkolnikov and contributors
"""FastAPI application factory and server setup.

This module provides the main FastAPI application with dependency injection,
middleware, error handlers, and route registration.
"""

from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any

import structlog
from fastapi import FastAPI, Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import generate_latest

from agent_memory.adapters.config.logging import configure_logging
from agent_memory.adapters.config.settings import get_settings
from agent_memory.adapters.inbound.admin_api import (
    get_old_engine,
    get_orchestrator,
    get_registry,
)
from agent_memory.adapters.inbound.admin_api import (
    router as admin_router,
)
from agent_memory.adapters.inbound.anthropic_adapter import router as anthropic_router
from agent_memory.adapters.inbound.auth_middleware import AuthenticationMiddleware
from agent_memory.adapters.inbound.coordination_adapter import router as coordination_router
from agent_memory.adapters.inbound.direct_agent_adapter import router as direct_router
from agent_memory.adapters.inbound.metrics import agents_active, pool_utilization_ratio, registry
from agent_memory.adapters.inbound.metrics_middleware import RequestMetricsMiddleware
from agent_memory.adapters.inbound.openai_adapter import router as openai_router
from agent_memory.adapters.inbound.rate_limiter import RateLimiter
from agent_memory.adapters.inbound.request_id_middleware import RequestIDMiddleware
from agent_memory.adapters.inbound.request_logging_middleware import RequestLoggingMiddleware
from agent_memory.adapters.outbound.chat_template_adapter import ChatTemplateAdapter
from agent_memory.adapters.outbound.safetensors_cache_adapter import SafetensorsCacheAdapter
from agent_memory.application.agent_cache_store import AgentCacheStore, ModelTag
from agent_memory.application.coordination_service import CoordinationService
from agent_memory.application.shared_prefix_cache import SharedPrefixCache
from agent_memory.domain.errors import (
    AgentNotFoundError,
    CacheCorruptionError,
    CachePersistenceError,
    GenerationError,
    IncompatibleCacheError,
    InvalidRequestError,
    PoolExhaustedError,
    SemanticError,
)
from agent_memory.domain.services import BlockPool
from agent_memory.domain.value_objects import ModelCacheSpec

# Health check thresholds
POOL_UTILIZATION_THRESHOLD = 0.9  # 90% utilization triggers degraded state


class AppState:
    """Application state container for dependency injection."""

    def __init__(self) -> None:
        """Initialize empty state (populated during startup)."""
        self.block_pool: BlockPool | None = None
        self.batch_engine: Any = None
        self.cache_store: AgentCacheStore | None = None
        self.mlx_adapter: Any = None
        self.cache_adapter: SafetensorsCacheAdapter | None = None
        self.scheduler: Any = None
        self.prefix_cache: SharedPrefixCache | None = None
        self.coordination_service: CoordinationService | None = None
        self.model_registry: Any = None
        self.model_swap_orchestrator: Any = None
        self.trt_subprocess: Any = None  # TRT backend subprocess adapter
        self.trt_inference: Any = None  # TRT inference service (with cache)
        self.tokenizer: Any = None  # Tokenizer (shared, used by adapters)
        self.llamacpp_swap_orchestrator: Any = None  # llama.cpp model swap
        self.server_tool_executor: Any = None  # Server-side WebSearch/WebFetch


class _LlamaCppSpecExtractor:
    """SpecExtractorPort for llama.cpp — returns advisory spec.

    llama.cpp manages its own KV cache, so the spec is advisory
    (used only for cache store model tagging, not block allocation).
    The 'model' here is actually a LlamaCppBackendAdapter.
    """

    def extract_spec(self, model: Any) -> ModelCacheSpec:
        """Extract spec from a LlamaCppBackendAdapter."""
        if hasattr(model, "extract_model_spec"):
            return model.extract_model_spec()
        return ModelCacheSpec(
            n_layers=48,
            n_kv_heads=16,
            head_dim=128,
            block_tokens=256,
            layer_types=["global"] * 48,
            kv_format="fp",
            kv_bits=None,
        )


def _load_trt_model_and_extract_spec(settings):
    """Load TRT engine via subprocess and extract cache spec.

    Args:
        settings: Application settings (uses settings.trt)

    Returns:
        Tuple of (subprocess_adapter, tokenizer, model_spec)
    """
    logger = structlog.get_logger(__name__)
    trt = settings.trt
    logger.info("loading_trt_model", engine_path=trt.engine_path, model_id=trt.model_id)

    from agent_memory.adapters.outbound.trt_model_loader import TRTModelLoader

    loader = TRTModelLoader()
    subprocess_adapter, tokenizer = loader.load(
        model_id=trt.model_id,
        engine_path=trt.engine_path,
        llm_inference_bin=trt.llm_inference_bin,
        timeout_s=trt.subprocess_timeout_s,
        shm_dir=trt.shm_dir,
    )

    # Override tokenizer max length
    tokenizer.model_max_length = trt.max_context_length

    # Extract spec from running engine
    model_spec = subprocess_adapter.extract_model_spec()
    model_spec = replace(
        model_spec,
        kv_bits=trt.kv_bits,
        kv_format="fp",
    )

    logger.info(
        "trt_model_loaded",
        n_layers=model_spec.n_layers,
        n_kv_heads=model_spec.n_kv_heads,
        head_dim=model_spec.head_dim,
        kv_format=model_spec.kv_format,
    )

    return subprocess_adapter, tokenizer, model_spec


def _load_model_and_extract_spec(settings):
    """Load MLX model and extract cache spec.

    CRITICAL: Override tokenizer model_max_length to support long context
    required by Claude Code CLI (18K+ tokens observed).

    Args:
        settings: Application settings

    Returns:
        Tuple of (model, tokenizer, model_spec)
    """
    logger = structlog.get_logger(__name__)
    logger.info("loading_model", model_id=settings.mlx.model_id)

    # CRITICAL: Override tokenizer max length for long context support
    tokenizer_config = {
        "model_max_length": settings.mlx.max_context_length,
        "truncation_side": "left",  # Keep recent tokens if needed
        "trust_remote_code": True,
    }

    from mlx_lm import load  # Runtime import — MLX backend only

    model, tokenizer = load(
        settings.mlx.model_id,
        tokenizer_config=tokenizer_config,
    )

    # Verify tokenizer configuration applied
    actual_max = tokenizer.model_max_length
    expected_max = settings.mlx.max_context_length
    logger.info("tokenizer_configured", max_length=actual_max, expected=expected_max)

    if actual_max < expected_max:
        logger.warning(
            "tokenizer_limit_warning",
            actual=actual_max,
            target=expected_max,
            message="Tokenizer max length less than target, requests may be truncated",
        )

    from agent_memory.adapters.outbound.mlx_spec_extractor import get_extractor

    spec_extractor = get_extractor()
    base_spec: ModelCacheSpec = spec_extractor.extract_spec(model)

    model_spec = replace(
        base_spec,
        kv_bits=settings.mlx.kv_bits,
        kv_group_size=settings.mlx.kv_group_size,
    )

    logger.info(
        "model_loaded",
        n_layers=model_spec.n_layers,
        n_kv_heads=model_spec.n_kv_heads,
        head_dim=model_spec.head_dim,
        kv_bits=model_spec.kv_bits,
        kv_group_size=model_spec.kv_group_size,
    )

    return model, tokenizer, model_spec


def _initialize_block_pool(settings, model_spec):
    """Initialize BlockPool with cache budget.

    Args:
        settings: Application settings
        model_spec: Model cache specification

    Returns:
        Configured BlockPool instance
    """
    logger = structlog.get_logger(__name__)

    bytes_per_block = model_spec.bytes_per_block_per_layer()
    backend_settings = settings.trt if settings.backend == "trt" else settings.mlx
    total_blocks = (backend_settings.cache_budget_mb * 1024 * 1024) // bytes_per_block
    mb_per_block = bytes_per_block / 1024 / 1024
    logger.info(
        "block_budget_calculated", total_blocks=total_blocks, mb_per_block=round(mb_per_block, 2)
    )

    block_pool = BlockPool(spec=model_spec, total_blocks=total_blocks)
    logger.info("block_pool_initialized", total_blocks=total_blocks)

    return block_pool


def _initialize_cache_store(settings, model_spec):
    """Initialize cache store and persistence adapter.

    Args:
        settings: Application settings
        model_spec: Model cache specification

    Returns:
        Tuple of (cache_store, cache_adapter)
    """
    logger = structlog.get_logger(__name__)

    cache_dir = Path(settings.agent.cache_dir).expanduser()

    if settings.backend == "trt":
        # TRT: numpy-based I/O, no MLX dependency
        from agent_memory.adapters.outbound.trt_quantization_adapter import TRTQuantizationAdapter
        from agent_memory.adapters.outbound.trt_safetensors_cache_adapter import (
            TRTSafetensorsCacheAdapter,
        )

        quantizer = TRTQuantizationAdapter()
        cache_adapter = TRTSafetensorsCacheAdapter(
            cache_dir=cache_dir,
            kv_bits=settings.trt.disk_kv_bits,
            kv_group_size=settings.trt.kv_group_size,
            quantizer=quantizer,
        )
    else:
        # MLX: uses mx.save/mx.load for native MLX tensor I/O
        cache_adapter = SafetensorsCacheAdapter(
            cache_dir=cache_dir,
            kv_bits=settings.mlx.kv_bits or 4,
            kv_group_size=settings.mlx.kv_group_size,
        )
    logger.info("cache_persistence_configured", cache_dir=str(cache_dir), backend=settings.backend)

    model_id = settings.trt.model_id if settings.backend == "trt" else settings.mlx.model_id
    model_tag = ModelTag.from_spec(model_id, model_spec)
    cache_store = AgentCacheStore(
        cache_dir=cache_dir,
        max_hot_agents=settings.agent.max_agents_in_memory,
        model_tag=model_tag,
        cache_adapter=cache_adapter,
    )
    logger.info("cache_store_initialized", max_hot_agents=settings.agent.max_agents_in_memory)

    return cache_store, cache_adapter


def _initialize_batch_engine(model, tokenizer, block_pool, model_spec, settings):
    """Initialize batch engine for inference.

    Args:
        model: MLX model
        tokenizer: Model tokenizer
        block_pool: BlockPool instance
        model_spec: Model cache specification
        settings: Application settings

    Returns:
        Configured BlockPoolBatchEngine instance
    """
    logger = structlog.get_logger(__name__)

    from agent_memory.adapters.outbound.mlx_cache_adapter import MLXCacheAdapter
    from agent_memory.application.batch_engine import BlockPoolBatchEngine

    mlx_adapter = MLXCacheAdapter()
    batch_engine = BlockPoolBatchEngine(
        model=model,
        tokenizer=tokenizer,
        pool=block_pool,
        spec=model_spec,
        cache_adapter=mlx_adapter,
        chunked_prefill_enabled=settings.mlx.chunked_prefill_enabled,
        chunked_prefill_threshold=settings.mlx.chunked_prefill_threshold,
        chunked_prefill_min_chunk=settings.mlx.chunked_prefill_min_chunk,
        chunked_prefill_max_chunk=settings.mlx.chunked_prefill_max_chunk,
    )
    logger.info(
        "batch_engine_initialized",
        max_batch_size=settings.mlx.max_batch_size,
        prefill_step_size=settings.mlx.prefill_step_size,
    )

    return batch_engine, mlx_adapter


async def _drain_and_persist(batch_engine, cache_store):
    """Drain pending requests and persist caches during shutdown.

    Args:
        batch_engine: Batch engine to drain
        cache_store: Cache store to persist
    """
    logger = structlog.get_logger(__name__)

    logger.info("draining_requests")
    if batch_engine:
        try:
            drained = await batch_engine.drain(timeout_seconds=30)
            logger.info("requests_drained", count=drained)
        except Exception as e:
            logger.error("drain_error", error=str(e), exc_info=True)

    logger.info("persisting_caches")
    if cache_store:
        try:
            saved_count = cache_store.evict_all_to_disk()
            logger.info("caches_persisted", count=saved_count)
        except Exception as e:
            logger.error("persist_error", error=str(e), exc_info=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager (startup/shutdown).

    Initializes:
    - MLX model and backend
    - BlockPool with cache budget
    - BatchEngine for inference
    - AgentCacheStore for cache management
    - Persistence adapters

    Args:
        app: FastAPI application instance

    Yields:
        Control to application during its lifetime
    """
    logger = structlog.get_logger(__name__)
    logger.info("server_starting")
    settings = get_settings()

    # Track TRT subprocess for cleanup
    trt_subprocess = None

    try:
        if settings.backend == "vllm":
            # --- vLLM backend path ---
            from transformers import AutoTokenizer

            from agent_memory.adapters.outbound.vllm_backend_adapter import VLLMBackendAdapter

            vllm_adapter = VLLMBackendAdapter(
                base_url=settings.vllm.base_url,
                model_id=settings.vllm.model_id,
                timeout_s=settings.vllm.timeout_s,
            )
            tokenizer = AutoTokenizer.from_pretrained(settings.vllm.model_id)
            tokenizer.model_max_length = settings.vllm.max_context_length
            model_spec = vllm_adapter.extract_model_spec()
            trt_subprocess = vllm_adapter  # Reuse TRT inference service path
            model = None
            logger.info(
                "vllm_backend_configured",
                base_url=settings.vllm.base_url,
                model_id=settings.vllm.model_id,
            )

        elif settings.backend == "llamacpp":
            # --- llama.cpp backend path ---
            # Two modes:
            # 1. Managed mode (server_binary set, default_model set):
            #    agent-memory starts/stops llama-server, supports model swapping.
            # 2. External mode (base_url only):
            #    llama-server runs independently, single model, no swap.

            if settings.llamacpp.default_model and settings.llamacpp.server_binary:
                # -- Managed mode: agent-memory manages llama-server lifecycle --
                from agent_memory.adapters.outbound.llamacpp_model_loader import (
                    LlamaCppModelLoader,
                )
                from agent_memory.application.llamacpp_swap_orchestrator import (
                    LlamaCppSwapOrchestrator,
                )
                from agent_memory.application.model_registry import ModelRegistry

                llamacpp_loader = LlamaCppModelLoader(
                    server_binary=settings.llamacpp.server_binary,
                    port=int(settings.llamacpp.base_url.rsplit(":", 1)[-1]),
                    cache_type_k=settings.llamacpp.cache_type_k,
                    cache_type_v=settings.llamacpp.cache_type_v,
                    timeout_s=settings.llamacpp.timeout_s,
                    slot_save_path=settings.llamacpp.slot_save_path,
                )

                # LlamaCppModelLoader implements ModelLoaderPort — reuse ModelRegistry
                llamacpp_registry = ModelRegistry(
                    model_loader=llamacpp_loader,
                    spec_extractor=_LlamaCppSpecExtractor(),
                )

                # Load default model (starts llama-server)
                llamacpp_adapter, tokenizer = llamacpp_registry.load_model(
                    settings.llamacpp.default_model,
                )
                model_spec = llamacpp_registry.get_current_spec()

            else:
                # -- External mode: llama-server managed externally --
                from transformers import AutoTokenizer

                from agent_memory.adapters.outbound.llamacpp_backend_adapter import (
                    LlamaCppBackendAdapter,
                )

                llamacpp_adapter = LlamaCppBackendAdapter(
                    base_url=settings.llamacpp.base_url,
                    model_id=settings.llamacpp.model_id,
                    timeout_s=settings.llamacpp.timeout_s,
                    n_slots=settings.llamacpp.n_slots,
                )
                tokenizer_id = settings.llamacpp.tokenizer_id or settings.llamacpp.model_id
                tokenizer = AutoTokenizer.from_pretrained(
                    tokenizer_id,
                    trust_remote_code=True,
                )
                tokenizer.model_max_length = settings.llamacpp.max_context_length
                model_spec = llamacpp_adapter.extract_model_spec()
                llamacpp_loader = None
                llamacpp_registry = None

            trt_subprocess = llamacpp_adapter
            model = None
            logger.info(
                "llamacpp_backend_configured",
                base_url=settings.llamacpp.base_url,
                model_id=(settings.llamacpp.default_model or settings.llamacpp.model_id),
                managed=bool(settings.llamacpp.default_model),
            )

        elif settings.backend == "trt":
            # --- TRT backend path ---
            trt_subprocess, tokenizer, model_spec = _load_trt_model_and_extract_spec(settings)
            model = None  # No MLX model
        else:
            # --- MLX backend path ---
            # Apply fused Q4 attention patch for mlx-lm < 0.31 (native Q4 support
            # was added in 0.31, making these patches unnecessary).
            try:
                import mlx_lm

                mlx_lm_version = tuple(int(x) for x in mlx_lm.__version__.split(".")[:2])
                if mlx_lm_version < (0, 31):
                    from agent_memory.adapters.outbound.mlx_fused_attention import (
                        apply_fused_attention_patch,
                    )

                    apply_fused_attention_patch()
                    logger.info("applied_q4_patches", mlx_lm_version=mlx_lm.__version__)
                else:
                    logger.info(
                        "skipping_q4_patches",
                        mlx_lm_version=mlx_lm.__version__,
                        reason="native Q4 KV cache in mlx-lm >= 0.31",
                    )
            except (ImportError, ValueError):
                logger.warning("mlx_lm_version_check_failed")

            model, tokenizer, model_spec = _load_model_and_extract_spec(settings)

        # Initialize components (backend-agnostic)
        block_pool = _initialize_block_pool(settings, model_spec)
        cache_store, cache_adapter = _initialize_cache_store(settings, model_spec)

        if settings.backend in ("trt", "vllm", "llamacpp"):
            from agent_memory.application.trt_inference_service import TRTInferenceService

            mlx_adapter = None
            batch_engine = None  # External backends use subprocess/HTTP, not batch engine
            from agent_memory.adapters.outbound.trt_quantization_adapter import (
                TRTQuantizationAdapter as TRTQuant,
            )

            trt_inference = TRTInferenceService(
                backend=trt_subprocess,
                tokenizer=tokenizer,
                cache_adapter=cache_adapter,
                quantizer=TRTQuant(),
            )
        else:
            batch_engine, mlx_adapter = _initialize_batch_engine(
                model, tokenizer, block_pool, model_spec, settings
            )

        if settings.backend == "mlx":
            # Initialize model registry and swap orchestrator (MLX only)
            from agent_memory.adapters.outbound.mlx_model_loader import MLXModelLoader
            from agent_memory.adapters.outbound.mlx_spec_extractor import (
                get_extractor as get_mlx_extractor,
            )
            from agent_memory.application.model_registry import ModelRegistry
            from agent_memory.application.model_swap_orchestrator import (
                ModelSwapOrchestrator,
            )

            model_loader = MLXModelLoader()
            spec_extractor = get_mlx_extractor()
            model_registry = ModelRegistry(
                model_loader=model_loader,
                spec_extractor=spec_extractor,
            )
            model_registry.set_loaded_model(
                model=model,
                tokenizer=tokenizer,
                spec=model_spec,
                model_id=settings.mlx.model_id,
            )

            model_swap_orchestrator = ModelSwapOrchestrator(
                model_registry=model_registry,
                block_pool=block_pool,
                cache_store=cache_store,
                cache_adapter=mlx_adapter,
            )
            llamacpp_swap_orchestrator = None

        elif settings.backend == "llamacpp" and locals().get("llamacpp_registry") is not None:
            # Managed llama.cpp: wire swap orchestrator + slot tracker
            from agent_memory.application.llamacpp_swap_orchestrator import (
                LlamaCppSwapOrchestrator,
            )
            from agent_memory.application.slot_tracker import SlotTracker

            slot_tracker = SlotTracker(
                n_slots=settings.llamacpp.n_slots,
                backend=llamacpp_adapter,  # type: ignore[name-defined]  # Implements SlotPersistencePort
            )
            # Wire slot tracker into adapter so generate() tracks usage
            llamacpp_adapter.slot_tracker = slot_tracker  # type: ignore[name-defined]
            # Enable traffic capture for parser regression tests
            if settings.llamacpp.capture_traffic:
                llamacpp_adapter.enable_capture(settings.llamacpp.capture_traffic)  # type: ignore[name-defined]
            # Restore slot caches from previous session (if any exist on disk)
            current_model = llamacpp_registry.get_current_id()  # type: ignore[union-attr]
            if current_model:
                restored = slot_tracker.restore_slots(
                    current_model,
                    settings.llamacpp.slot_save_path,
                )
                if restored:
                    logger.info("startup_slot_restore", count=restored, model_id=current_model)

            model_registry = llamacpp_registry  # type: ignore[assignment]
            model_swap_orchestrator = None  # MLX-specific, not used
            llamacpp_swap_orchestrator = LlamaCppSwapOrchestrator(
                model_registry=llamacpp_registry,
                cache_store=cache_store,
                model_loader=llamacpp_loader,  # type: ignore[name-defined]
                slot_tracker=slot_tracker,
                n_slots=settings.llamacpp.n_slots,
                slot_save_path=settings.llamacpp.slot_save_path,
            )
            # Wire swap orchestrator into TRTInferenceService for auto-swap
            # (can't pass at construction — orchestrator depends on cache_store
            # which is created between TRTInferenceService and orchestrator)
            if settings.llamacpp.auto_swap:
                trt_inference._swap_orchestrator = llamacpp_swap_orchestrator  # type: ignore[union-attr]
                trt_inference._model_registry = llamacpp_registry  # type: ignore[union-attr]
        else:
            model_registry = None
            model_swap_orchestrator = None
            llamacpp_swap_orchestrator = None

        # Store in app state
        app.state.agent_memory = AppState()
        app.state.agent_memory.block_pool = block_pool
        app.state.agent_memory.batch_engine = batch_engine
        app.state.agent_memory.cache_store = cache_store
        app.state.agent_memory.mlx_adapter = mlx_adapter
        app.state.agent_memory.cache_adapter = cache_adapter
        app.state.agent_memory.model_registry = model_registry
        app.state.agent_memory.model_swap_orchestrator = model_swap_orchestrator
        app.state.agent_memory.llamacpp_swap_orchestrator = llamacpp_swap_orchestrator
        app.state.agent_memory.trt_subprocess = trt_subprocess
        app.state.agent_memory.trt_inference = (
            trt_inference if settings.backend in ("trt", "vllm", "llamacpp") else None
        )
        app.state.agent_memory.tokenizer = tokenizer
        app.state.shutting_down = False

        # Shared prefix cache (always enabled)
        prefix_cache = SharedPrefixCache()
        app.state.agent_memory.prefix_cache = prefix_cache
        logger.info("prefix_cache_initialized")

        # Server-side tool executor (WebSearch via SearXNG, WebFetch via Jina Reader)
        if settings.server.searxng_url or settings.server.jina_reader_url:
            from agent_memory.adapters.outbound.server_tool_adapter import ServerToolAdapter

            app.state.agent_memory.server_tool_executor = ServerToolAdapter(
                searxng_url=settings.server.searxng_url,
                jina_reader_url=settings.server.jina_reader_url,
            )
            logger.info(
                "server_tool_executor_initialized",
                searxng=bool(settings.server.searxng_url),
                jina=bool(settings.server.jina_reader_url),
            )

        # Start ConcurrentScheduler — serializes all engine access through
        # a single worker thread, preventing concurrent Metal GPU crashes.
        # (MLX only — TRT subprocess handles its own serialization)
        scheduler = None
        if settings.backend == "mlx" and settings.mlx.scheduler_enabled:
            try:
                from agent_memory.adapters.outbound.mlx_prefill_adapter import MLXPrefillAdapter

                prefill_adapter = MLXPrefillAdapter(
                    model=model,
                    kv_bits=settings.mlx.kv_bits or 4,
                    kv_group_size=settings.mlx.kv_group_size,
                    min_chunk=settings.mlx.chunked_prefill_min_chunk,
                    max_chunk=settings.mlx.chunked_prefill_max_chunk,
                )
                from agent_memory.application.scheduler import ConcurrentScheduler

                scheduler = ConcurrentScheduler(
                    engine=batch_engine,
                    prefill_adapter=prefill_adapter,
                    n_layers=model_spec.n_layers,
                    interleave_threshold=settings.mlx.scheduler_interleave_threshold,
                    max_batch_size=settings.mlx.max_batch_size,
                )
                scheduler.start()
                app.state.agent_memory.scheduler = scheduler
                logger.info(
                    "scheduler_started",
                    interleave_threshold=settings.mlx.scheduler_interleave_threshold,
                )
            except ImportError:
                logger.warning(
                    "scheduler_unavailable",
                    reason="MLXPrefillAdapter not importable — falling back to direct engine path",
                )

        # Initialize CoordinationService (can work with or without scheduler)
        chat_template_adapter = ChatTemplateAdapter()
        coordination_service = CoordinationService(
            scheduler=scheduler,
            cache_store=cache_store,
            engine=batch_engine,
            reasoning_extra_tokens=(
                settings.trt.reasoning_extra_tokens
                if settings.backend == "trt"
                else settings.mlx.reasoning_extra_tokens
            ),
            chat_template=chat_template_adapter,
        )
        app.state.coordination_service = coordination_service
        logger.info(
            "coordination_service_initialized",
            scheduler_enabled=(scheduler is not None),
        )

        # Validate Q4 pipeline patches (MLX < 0.31 only)
        if settings.backend == "mlx" and mlx_lm_version < (0, 31):  # type: ignore[possibly-undefined]
            from agent_memory.adapters.outbound.mlx_quantized_extensions import (
                validate_q4_pipeline,
            )

            if not validate_q4_pipeline():
                logger.error(
                    "q4_validation_failed",
                    message="Q4 cache patches may not be applied correctly. "
                    "KV caches may fall back to FP16, causing higher memory usage. "
                    "Check mlx-lm version compatibility.",
                )

        logger.info("server_ready")

        yield

        # Shutdown: cleanup resources
        logger.info("server_shutting_down")
        app.state.shutting_down = True

        if scheduler is not None:
            scheduler.stop()
            logger.info("scheduler_stopped")

        await _drain_and_persist(batch_engine, cache_store)

        # Release backend resources
        if settings.backend == "trt" and trt_subprocess is not None:
            logger.info("stopping_trt_subprocess")
            trt_subprocess.stop()
            logger.info("trt_subprocess_stopped")
        elif settings.backend == "llamacpp" and llamacpp_swap_orchestrator is not None:
            # Managed mode: save slot caches, then stop llama-server
            logger.info("saving_slot_caches_before_shutdown")
            try:
                model_id = llamacpp_registry.get_current_id()  # type: ignore[union-attr]
                if model_id and locals().get("slot_tracker"):
                    saved = slot_tracker.save_slots(model_id)  # type: ignore[union-attr]
                    logger.info("slot_caches_saved", count=saved, model_id=model_id)
                    # Enforce disk budget after save
                    if settings.llamacpp.max_slot_disk_mb > 0:
                        from agent_memory.application.slot_tracker import SlotTracker as ST

                        ST.enforce_disk_budget(
                            settings.llamacpp.slot_save_path,
                            settings.llamacpp.max_slot_disk_mb,
                            current_model_id=model_id,
                        )
            except Exception as e:
                logger.warning("slot_cache_save_error", error=str(e))

            logger.info("stopping_llama_server")
            try:
                llamacpp_loader.clear_cache()  # type: ignore[union-attr]
                logger.info("llama_server_stopped")
            except Exception as e:
                logger.warning("llama_server_stop_error", error=str(e))
        elif settings.backend == "mlx":
            # Explicitly release model and GPU memory to prevent wired memory
            # accumulation across server restarts.
            logger.info("releasing_gpu_memory")
            try:
                import gc

                import mlx.core as mx

                if batch_engine is not None:
                    batch_engine.shutdown()
                if model_registry is not None:
                    model_registry.unload_model()
                if block_pool is not None:
                    block_pool.force_clear_all_allocations()

                gc.collect()
                gc.collect()
                mx.clear_cache()
                logger.info(
                    "gpu_memory_released",
                    active_mb=round(mx.get_active_memory() / 1024**2),
                    cache_mb=round(mx.get_cache_memory() / 1024**2),
                )
            except Exception as e:
                logger.warning("gpu_memory_release_error", error=str(e))

        # Clear app.state references
        if hasattr(app.state, "agent_memory"):
            for attr in list(vars(app.state.agent_memory).keys()):
                setattr(app.state.agent_memory, attr, None)

        logger.info("server_shutdown_complete")
    except Exception as e:
        logger.error("lifespan_error", error=str(e), exc_info=True)
        raise


def _register_middleware(app: FastAPI, settings):
    """Register all middleware in correct order.

    Args:
        app: FastAPI application
        settings: Application settings
    """
    logger = structlog.get_logger(__name__)

    # Request ID middleware (FIRST - sets up context)
    app.add_middleware(RequestIDMiddleware)
    logger.info("middleware_registered", middleware="RequestIDMiddleware")

    # Request logging middleware
    app.add_middleware(
        RequestLoggingMiddleware, skip_paths={"/health/live", "/health/ready", "/health/startup"}
    )
    logger.info("middleware_registered", middleware="RequestLoggingMiddleware")

    # Metrics middleware
    app.add_middleware(RequestMetricsMiddleware, skip_paths={"/metrics"})
    logger.info("middleware_registered", middleware="RequestMetricsMiddleware")

    # CORS middleware
    cors_origins_str = settings.server.cors_origins
    cors_origins = (
        ["*"]
        if cors_origins_str == "*"
        else [origin.strip() for origin in cors_origins_str.split(",")]
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Authentication middleware
    app.add_middleware(AuthenticationMiddleware)
    logger.info("middleware_registered", middleware="AuthenticationMiddleware")

    # Rate limiting middleware
    app.add_middleware(
        RateLimiter,
        requests_per_minute_per_agent=settings.server.rate_limit_per_agent,
        requests_per_minute_global=settings.server.rate_limit_global,
    )
    logger.info("middleware_registered", middleware="RateLimiter")


def _register_health_endpoints(app: FastAPI):
    """Register 3-tier health check endpoints.

    Args:
        app: FastAPI application
    """

    @app.get("/health")
    async def health():
        """Basic health check - alias for /health/live."""
        return {"status": "ok"}

    @app.get("/health/live")
    async def health_live():
        """Liveness probe - process is alive."""
        return {"status": "alive"}

    @app.get("/health/ready")
    async def health_ready(response: Response):
        """Readiness probe - ready to accept requests."""
        # Check pool utilization
        pool = app.state.agent_memory.block_pool if hasattr(app.state, "agent_memory") else None

        if not pool:
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
            return {"status": "not_ready", "reason": "pool_not_initialized"}

        # Check if shutting down
        if getattr(app.state, "shutting_down", False):
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
            return {"status": "not_ready", "reason": "shutting_down"}

        # Check pool exhaustion
        used_blocks = pool.total_blocks - pool.available_blocks()
        total_blocks = pool.total_blocks
        utilization = (used_blocks / total_blocks) if total_blocks > 0 else 0

        # Update metrics
        pool_utilization_ratio.set(utilization)
        cache_store = (
            app.state.agent_memory.cache_store if hasattr(app.state, "agent_memory") else None
        )
        if cache_store:
            agents_active.set(len(cache_store._hot_cache))

        if utilization > POOL_UTILIZATION_THRESHOLD:
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
            return {
                "status": "not_ready",
                "reason": "pool_near_exhaustion",
                "pool_utilization": round(utilization * 100, 1),
            }

        return {"status": "ready", "pool_utilization": round(utilization * 100, 1)}

    @app.get("/health/startup")
    async def health_startup(response: Response):
        """Startup probe - initialization complete."""
        if not hasattr(app.state, "agent_memory"):
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
            return {"status": "starting", "reason": "initializing"}
        state = app.state.agent_memory
        # Either batch_engine (MLX) or trt_subprocess (TRT) must be ready
        if not state.batch_engine and not state.trt_subprocess:
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
            return {"status": "starting", "reason": "model_loading"}
        return {"status": "started"}


def _register_search_proxy(app: FastAPI, searxng_url: str):
    """Register /search proxy endpoint that forwards to SearXNG.

    Allows Claude Code (and other clients) to use web search via the same
    port as agent-memory (8000) without needing direct access to SearXNG (8080).
    """
    import json as _json
    from urllib.error import URLError as _URLError
    from urllib.parse import quote_plus as _qp
    from urllib.request import urlopen as _urlopen

    @app.get("/search")
    async def search_proxy(q: str, format: str = "json", engines: str = "", num: int = 10):
        """Proxy web search to SearXNG."""
        params = f"q={_qp(q)}&format=json&pageno=1"
        if engines:
            params += f"&engines={_qp(engines)}"
        url = f"{searxng_url}/search?{params}"
        try:
            with _urlopen(url, timeout=30) as resp:  # noqa: S310
                data = _json.loads(resp.read())
            results = data.get("results", [])[:num]
            return {"query": q, "results": results, "n": len(results)}
        except _URLError as e:
            return JSONResponse(status_code=502, content={"error": f"SearXNG unavailable: {e}"})


def _register_fetch_proxy(app: FastAPI, jina_reader_url: str):
    """Register /fetch proxy endpoint that converts URLs to clean markdown via Jina Reader.

    Allows Claude Code to fetch web pages as markdown through agent-memory,
    avoiding raw HTML (token-heavy) and private-IP restrictions on WebFetch.
    """
    from urllib.error import URLError as _URLError
    from urllib.request import Request as _Req
    from urllib.request import urlopen as _urlopen

    @app.get("/fetch")
    async def fetch_proxy(url: str, timeout: int = 15):
        """Fetch a URL via Jina Reader, returning clean markdown."""
        reader_url = f"{jina_reader_url}/{url}"
        try:
            req = _Req(reader_url)  # noqa: S310
            with _urlopen(req, timeout=timeout) as resp:  # noqa: S310
                content = resp.read().decode("utf-8", errors="replace")
            return Response(content=content, media_type="text/plain; charset=utf-8")
        except _URLError as e:
            return JSONResponse(
                status_code=502,
                content={"error": f"Jina Reader unavailable: {e}"},
            )
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={"error": f"Fetch failed: {e}"},
            )


def _register_metrics_endpoint(app: FastAPI):
    """Register Prometheus metrics endpoint.

    Args:
        app: FastAPI application
    """

    @app.get("/metrics")
    async def metrics():
        """Prometheus metrics endpoint."""
        return Response(content=generate_latest(registry), media_type="text/plain; version=0.0.4")


def _register_debug_endpoints(app: FastAPI):
    """Register debug endpoints for benchmarking and diagnostics.

    Args:
        app: FastAPI application
    """

    @app.get("/debug/memory")
    async def debug_memory():
        """Memory statistics for benchmarking (MLX or generic)."""
        semantic = getattr(app.state, "agent_memory", None)
        pool = semantic.block_pool if semantic else None

        result: dict[str, Any] = {
            "pool_used_blocks": (pool.total_blocks - pool.available_blocks()) if pool else 0,
            "pool_total_blocks": pool.total_blocks if pool else 0,
        }

        try:
            import mlx.core as mx

            result["active_memory_mb"] = round(mx.get_active_memory() / (1024**2), 1)
            result["peak_memory_mb"] = round(mx.get_peak_memory() / (1024**2), 1)
            result["cache_memory_mb"] = round(mx.get_cache_memory() / (1024**2), 1)
        except ImportError:
            result["backend"] = "trt"
            if semantic and semantic.cache_store:
                result["hot_memory_bytes"] = semantic.cache_store.hot_memory_bytes
                result["disk_usage_bytes"] = semantic.cache_store.disk_usage_bytes

        return result


def _is_openai_request(request: Request) -> bool:
    """Check if request is to OpenAI-style endpoint."""
    return "/chat/completions" in request.url.path


def _is_anthropic_request(request: Request) -> bool:
    """Check if request is to Anthropic-style endpoint."""
    path = request.url.path
    return "/messages" in path and "/chat/completions" not in path


def _format_error_response(
    request: Request,
    status_code: int,
    error_type: str,
    message: str,
) -> JSONResponse:
    """Format error response according to API type (OpenAI/Anthropic/default).

    Args:
        request: The HTTP request
        status_code: HTTP status code
        error_type: Error type string
        message: Error message

    Returns:
        JSONResponse with properly formatted error
    """
    if _is_openai_request(request):
        # OpenAI format: {"error": {"message": ..., "type": ..., "param": null, "code": null}}
        content = {
            "error": {
                "message": message,
                "type": error_type,
                "param": None,
                "code": None,
            }
        }
    elif _is_anthropic_request(request):
        # Anthropic format: {"type": "error", "error": {"type": ..., "message": ...}}
        content = {
            "type": "error",
            "error": {
                "type": error_type,
                "message": message,
            },
        }
    else:
        # Default format for other endpoints
        content = {
            "error": {
                "type": error_type,
                "message": message,
            }
        }

    return JSONResponse(status_code=status_code, content=content)


def _get_semantic_error_details(exc: SemanticError) -> tuple[int, str]:
    """Get HTTP status code and error type for SemanticError subclasses.

    Args:
        exc: The SemanticError instance

    Returns:
        Tuple of (status_code, error_type)
    """
    # Map specific error types to appropriate HTTP status codes
    if isinstance(exc, PoolExhaustedError):
        return status.HTTP_503_SERVICE_UNAVAILABLE, "overloaded_error"
    if isinstance(exc, AgentNotFoundError):
        return status.HTTP_404_NOT_FOUND, "not_found_error"
    if isinstance(exc, InvalidRequestError):
        return status.HTTP_400_BAD_REQUEST, "invalid_request_error"
    if isinstance(exc, (CacheCorruptionError, CachePersistenceError)):
        return status.HTTP_500_INTERNAL_SERVER_ERROR, "api_error"
    if isinstance(exc, IncompatibleCacheError):
        return status.HTTP_409_CONFLICT, "invalid_request_error"
    if isinstance(exc, GenerationError):
        return status.HTTP_500_INTERNAL_SERVER_ERROR, "api_error"
    # Default for unknown SemanticError subclasses
    return status.HTTP_400_BAD_REQUEST, "invalid_request_error"


def _register_error_handlers(app: FastAPI):
    """Register error handlers for exceptions.

    Args:
        app: FastAPI application
    """
    logger = structlog.get_logger(__name__)

    @app.exception_handler(SemanticError)
    async def semantic_error_handler(request: Request, exc: SemanticError):
        """Handle domain errors with appropriate status codes and API format."""
        status_code, error_type = _get_semantic_error_details(exc)
        logger.error(
            "domain_error",
            error_type=exc.__class__.__name__,
            http_status=status_code,
            message=str(exc),
            exc_info=True,
        )
        return _format_error_response(request, status_code, error_type, str(exc))

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, exc: RequestValidationError):
        """Handle request validation errors."""
        logger.warning("validation_error", error=str(exc))
        # Format validation errors into a readable message
        error_messages = []
        for error in exc.errors():
            loc = ".".join(str(part) for part in error["loc"])
            error_messages.append(f"{loc}: {error['msg']}")
        message = "; ".join(error_messages)
        return _format_error_response(
            request,
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "invalid_request_error",
            message,
        )

    @app.exception_handler(Exception)
    async def general_error_handler(request: Request, exc: Exception):
        """Handle unexpected errors."""
        logger.error(
            "unexpected_error", error_type=type(exc).__name__, message=str(exc), exc_info=True
        )
        return _format_error_response(
            request,
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "api_error",
            "An internal error occurred",
        )


def _register_routes(app: FastAPI):
    """Register API route handlers.

    Args:
        app: FastAPI application
    """
    logger = structlog.get_logger(__name__)

    @app.post("/api/event_logging/batch", status_code=status.HTTP_200_OK)
    async def event_logging_stub():
        """Stub endpoint for Claude Code CLI event logging (no-op)."""
        return {"status": "ok"}

    @app.get("/", status_code=status.HTTP_200_OK)
    async def root():
        """Root endpoint with API info."""
        return {
            "name": "agent-memory",
            "version": "0.2.0",
            "endpoints": {
                "health": "/health",
                "metrics": "/metrics",
                "anthropic": "/v1/messages",
                "openai": "/v1/chat/completions",
                "agents": "/v1/agents",
            },
        }

    @app.get("/v1/models", status_code=status.HTTP_200_OK)
    async def list_models():
        """OpenAI-compatible models endpoint — returns available models.

        For llama.cpp managed mode, returns all models from config/models/*.toml.
        The currently loaded model is marked with 'active: true'.
        """
        semantic = getattr(app.state, "agent_memory", None)
        engine = semantic.batch_engine if semantic else None
        registry = semantic.model_registry if semantic else None
        settings = get_settings()

        models = []

        if settings.backend == "llamacpp" and settings.llamacpp.default_model:
            # Managed mode: list all available models from TOML configs
            from pathlib import Path

            config_dir = Path(__file__).resolve().parents[3] / "config" / "models"
            current_id = registry.get_current_id() if registry else None

            if config_dir.is_dir():
                try:
                    import tomllib
                except ImportError:
                    import tomli as tomllib  # type: ignore[no-redef]

                for toml_file in sorted(config_dir.glob("*.toml")):
                    try:
                        with toml_file.open("rb") as f:
                            profile = tomllib.load(f)
                        if "llamacpp" not in profile:
                            continue
                        mid = profile.get("model", {}).get("model_id", toml_file.stem)
                        entry = {
                            "id": mid,
                            "object": "model",
                            "owned_by": "local",
                            "active": mid == current_id,
                        }
                        models.append(entry)
                    except Exception:
                        pass
        else:
            # Single model mode (MLX, TRT, vLLM, external llamacpp)
            model_id = registry.get_current_id() if registry else None
            if model_id:
                model_entry = {
                    "id": model_id,
                    "object": "model",
                    "owned_by": "local",
                }
                if engine:
                    spec = engine._spec
                    model_entry["spec"] = {
                        "n_layers": spec.n_layers,
                        "n_kv_heads": spec.n_kv_heads,
                        "head_dim": spec.head_dim,
                        "block_tokens": spec.block_tokens,
                        "kv_bits": spec.kv_bits,
                        "max_context_length": (
                            settings.trt.max_context_length
                            if settings.backend == "trt"
                            else settings.mlx.max_context_length
                        ),
                    }
                models.append(model_entry)

        return {"object": "list", "data": models}

    app.include_router(anthropic_router)
    logger.info("routes_registered", router="anthropic", path="/v1/messages")

    app.include_router(openai_router)
    logger.info("routes_registered", router="openai", path="/v1/chat/completions")

    app.include_router(direct_router)
    logger.info("routes_registered", router="direct_agent", path="/v1/agents")

    app.include_router(coordination_router)
    logger.info("routes_registered", router="coordination", path="/v1/coordination")

    # Admin API for model management (requires SEMANTIC_ADMIN_KEY)
    app.include_router(admin_router)
    logger.info("routes_registered", router="admin", path="/admin")


def create_app() -> FastAPI:
    """Create and configure FastAPI application.

    Returns:
        Configured FastAPI application instance.
    """
    settings = get_settings()

    # Initialize structured logging
    # Use JSON output for non-DEBUG levels (production-like environments)
    json_output = settings.server.log_level not in ("DEBUG",)
    configure_logging(log_level=settings.server.log_level, json_output=json_output)

    logger = structlog.get_logger(__name__)
    logger.info("creating_fastapi_app", version="0.2.0")

    # Create FastAPI app
    app = FastAPI(
        title="agent-memory",
        description="Multi-protocol API for agent-memory KV cache management",
        version="0.2.0",
        lifespan=lifespan,
    )

    # Register components
    _register_middleware(app, settings)
    _register_health_endpoints(app)
    _register_metrics_endpoint(app)
    _register_debug_endpoints(app)
    _register_error_handlers(app)
    _register_routes(app)
    if settings.server.searxng_url:
        _register_search_proxy(app, settings.server.searxng_url)
        logger.info("search_proxy_registered", searxng_url=settings.server.searxng_url)
    if settings.server.jina_reader_url:
        _register_fetch_proxy(app, settings.server.jina_reader_url)
        logger.info("fetch_proxy_registered", jina_reader_url=settings.server.jina_reader_url)

    # Set up dependency overrides for admin API
    def _get_orchestrator():
        return app.state.agent_memory.model_swap_orchestrator

    def _get_old_engine():
        return app.state.agent_memory.batch_engine

    def _get_registry():
        return app.state.agent_memory.model_registry

    app.dependency_overrides[get_orchestrator] = _get_orchestrator
    app.dependency_overrides[get_old_engine] = _get_old_engine
    app.dependency_overrides[get_registry] = _get_registry

    logger.info("fastapi_app_created", log_level=settings.server.log_level)
    return app
