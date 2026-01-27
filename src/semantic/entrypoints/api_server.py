"""FastAPI application factory and server setup.

This module provides the main FastAPI application with dependency injection,
middleware, error handlers, and route registration.
"""

from contextlib import asynccontextmanager
from pathlib import Path

import structlog
from fastapi import FastAPI, Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from mlx_lm import load
from prometheus_client import generate_latest

from semantic.adapters.config.logging import configure_logging
from semantic.adapters.config.settings import get_settings
from semantic.adapters.inbound.anthropic_adapter import router as anthropic_router
from semantic.adapters.inbound.auth_middleware import AuthenticationMiddleware
from semantic.adapters.inbound.direct_agent_adapter import router as direct_router
from semantic.adapters.inbound.metrics import agents_active, pool_utilization_ratio, registry
from semantic.adapters.inbound.metrics_middleware import RequestMetricsMiddleware
from semantic.adapters.inbound.openai_adapter import router as openai_router
from semantic.adapters.inbound.rate_limiter import RateLimiter
from semantic.adapters.inbound.request_id_middleware import RequestIDMiddleware
from semantic.adapters.inbound.request_logging_middleware import RequestLoggingMiddleware
from semantic.adapters.outbound.mlx_cache_adapter import MLXCacheAdapter
from semantic.adapters.outbound.mlx_spec_extractor import get_extractor
from semantic.adapters.outbound.safetensors_cache_adapter import SafetensorsCacheAdapter
from semantic.application.agent_cache_store import AgentCacheStore, ModelTag
from semantic.application.batch_engine import BlockPoolBatchEngine
from semantic.domain.errors import (
    AgentNotFoundError,
    CacheCorruptionError,
    CachePersistenceError,
    GenerationError,
    IncompatibleCacheError,
    InvalidRequestError,
    PoolExhaustedError,
    SemanticError,
)
from semantic.domain.services import BlockPool
from semantic.domain.value_objects import ModelCacheSpec

# Health check thresholds
POOL_UTILIZATION_THRESHOLD = 0.9  # 90% utilization triggers degraded state


class AppState:
    """Application state container for dependency injection."""

    def __init__(self) -> None:
        """Initialize empty state (populated during startup)."""
        self.block_pool: BlockPool | None = None
        self.batch_engine: BlockPoolBatchEngine | None = None
        self.cache_store: AgentCacheStore | None = None
        self.mlx_adapter: MLXCacheAdapter | None = None
        self.cache_adapter: SafetensorsCacheAdapter | None = None


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
        "truncation_side": "left",   # Keep recent tokens if needed
        "trust_remote_code": True,
    }

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
            message="Tokenizer max length less than target, requests may be truncated"
        )

    spec_extractor = get_extractor()
    base_spec: ModelCacheSpec = spec_extractor.extract_spec(model)

    # Add quantization settings from config
    from dataclasses import replace
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
        kv_group_size=model_spec.kv_group_size
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
    total_blocks = (settings.mlx.cache_budget_mb * 1024 * 1024) // bytes_per_block
    mb_per_block = bytes_per_block / 1024 / 1024
    logger.info(
        "block_budget_calculated",
        total_blocks=total_blocks,
        mb_per_block=round(mb_per_block, 2)
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
    cache_adapter = SafetensorsCacheAdapter(cache_dir=cache_dir)
    logger.info("cache_persistence_configured", cache_dir=str(cache_dir))

    model_tag = ModelTag.from_spec(settings.mlx.model_id, model_spec)
    cache_store = AgentCacheStore(
        cache_dir=cache_dir,
        max_hot_agents=settings.agent.max_agents_in_memory,
        model_tag=model_tag,
        cache_adapter=cache_adapter,
    )
    logger.info(
        "cache_store_initialized",
        max_hot_agents=settings.agent.max_agents_in_memory
    )

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

    mlx_adapter = MLXCacheAdapter()
    batch_engine = BlockPoolBatchEngine(
        model=model,
        tokenizer=tokenizer,
        pool=block_pool,
        spec=model_spec,
        cache_adapter=mlx_adapter,
    )
    logger.info(
        "batch_engine_initialized",
        max_batch_size=settings.mlx.max_batch_size,
        prefill_step_size=settings.mlx.prefill_step_size
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

    try:
        # Load model and extract spec
        model, tokenizer, model_spec = _load_model_and_extract_spec(settings)

        # Initialize components
        block_pool = _initialize_block_pool(settings, model_spec)
        cache_store, cache_adapter = _initialize_cache_store(settings, model_spec)
        batch_engine, mlx_adapter = _initialize_batch_engine(
            model, tokenizer, block_pool, model_spec, settings
        )

        # Store in app state
        app.state.semantic = AppState()
        app.state.semantic.block_pool = block_pool
        app.state.semantic.batch_engine = batch_engine
        app.state.semantic.cache_store = cache_store
        app.state.semantic.mlx_adapter = mlx_adapter
        app.state.semantic.cache_adapter = cache_adapter
        app.state.shutting_down = False

        logger.info("server_ready")

        yield

        # Shutdown: cleanup resources
        logger.info("server_shutting_down")
        app.state.shutting_down = True

        await _drain_and_persist(batch_engine, cache_store)
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
        RequestLoggingMiddleware,
        skip_paths={"/health/live", "/health/ready", "/health/startup"}
    )
    logger.info("middleware_registered", middleware="RequestLoggingMiddleware")

    # Metrics middleware
    app.add_middleware(RequestMetricsMiddleware, skip_paths={"/metrics"})
    logger.info("middleware_registered", middleware="RequestMetricsMiddleware")

    # CORS middleware
    cors_origins_str = settings.server.cors_origins
    cors_origins = ["*"] if cors_origins_str == "*" else [
        origin.strip() for origin in cors_origins_str.split(",")
    ]
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
        pool = app.state.semantic.block_pool if hasattr(app.state, "semantic") else None

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
        cache_store = app.state.semantic.cache_store if hasattr(app.state, "semantic") else None
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
        if not hasattr(app.state, "semantic") or not app.state.semantic.batch_engine:
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
            return {"status": "starting", "reason": "model_loading"}
        return {"status": "started"}


def _register_metrics_endpoint(app: FastAPI):
    """Register Prometheus metrics endpoint.

    Args:
        app: FastAPI application
    """
    @app.get("/metrics")
    async def metrics():
        """Prometheus metrics endpoint."""
        return Response(
            content=generate_latest(registry),
            media_type="text/plain; version=0.0.4"
        )


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
            }
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
    elif isinstance(exc, AgentNotFoundError):
        return status.HTTP_404_NOT_FOUND, "not_found_error"
    elif isinstance(exc, InvalidRequestError):
        return status.HTTP_400_BAD_REQUEST, "invalid_request_error"
    elif isinstance(exc, CacheCorruptionError):
        return status.HTTP_500_INTERNAL_SERVER_ERROR, "api_error"
    elif isinstance(exc, CachePersistenceError):
        return status.HTTP_500_INTERNAL_SERVER_ERROR, "api_error"
    elif isinstance(exc, IncompatibleCacheError):
        return status.HTTP_409_CONFLICT, "invalid_request_error"
    elif isinstance(exc, GenerationError):
        return status.HTTP_500_INTERNAL_SERVER_ERROR, "api_error"
    else:
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
            exc_info=True
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
            "unexpected_error",
            error_type=type(exc).__name__,
            message=str(exc),
            exc_info=True
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
            "name": "Semantic Caching API",
            "version": "0.2.0",
            "endpoints": {
                "health": "/health",
                "metrics": "/metrics",
                "anthropic": "/v1/messages",
                "openai": "/v1/chat/completions",
                "agents": "/v1/agents",
            },
        }

    app.include_router(anthropic_router)
    logger.info("routes_registered", router="anthropic", path="/v1/messages")

    app.include_router(openai_router)
    logger.info("routes_registered", router="openai", path="/v1/chat/completions")

    app.include_router(direct_router)
    logger.info("routes_registered", router="direct_agent", path="/v1/agents")


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
        title="Semantic Caching API",
        description="Multi-protocol API for semantic KV cache management",
        version="0.2.0",
        lifespan=lifespan,
    )

    # Register components
    _register_middleware(app, settings)
    _register_health_endpoints(app)
    _register_metrics_endpoint(app)
    _register_error_handlers(app)
    _register_routes(app)

    logger.info("fastapi_app_created", log_level=settings.server.log_level)
    return app
