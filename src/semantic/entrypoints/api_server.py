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

from semantic.adapters.config.logging import configure_logging
from semantic.adapters.config.settings import get_settings
from semantic.adapters.outbound.mlx_cache_adapter import MLXCacheAdapter
from semantic.adapters.outbound.mlx_spec_extractor import get_extractor
from semantic.adapters.outbound.safetensors_cache_adapter import SafetensorsCacheAdapter
from semantic.application.agent_cache_store import AgentCacheStore, ModelTag
from semantic.application.batch_engine import BlockPoolBatchEngine
from semantic.domain.errors import SemanticError
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

    # Load MLX model and extract spec
    logger.info("loading_model", model_id=settings.mlx.model_id)
    from mlx_lm import load

    model, tokenizer = load(
        settings.mlx.model_id,
        tokenizer_config={"trust_remote_code": True},
    )

    # Extract model cache spec
    spec_extractor = get_extractor()
    model_spec: ModelCacheSpec = spec_extractor.extract_spec(model)
    logger.info(
        "model_loaded",
        n_layers=model_spec.n_layers,
        n_kv_heads=model_spec.n_kv_heads,
        head_dim=model_spec.head_dim
    )

    # Calculate block budget
    bytes_per_block = model_spec.bytes_per_block_per_layer()
    total_blocks = (settings.mlx.cache_budget_mb * 1024 * 1024) // bytes_per_block
    mb_per_block = bytes_per_block / 1024 / 1024
    logger.info(
        "block_budget_calculated",
        total_blocks=total_blocks,
        mb_per_block=round(mb_per_block, 2)
    )

    # Initialize BlockPool
    block_pool = BlockPool(spec=model_spec, total_blocks=total_blocks)
    logger.info("block_pool_initialized", total_blocks=total_blocks)

    # Initialize MLX adapter (stateless, no arguments needed)
    mlx_adapter = MLXCacheAdapter()

    # Initialize safetensors persistence adapter
    cache_dir = Path(settings.agent.cache_dir).expanduser()
    cache_adapter = SafetensorsCacheAdapter(cache_dir=cache_dir)
    logger.info("cache_persistence_configured", cache_dir=str(cache_dir))

    # Initialize AgentCacheStore
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

    # Initialize BatchEngine
    batch_engine = BlockPoolBatchEngine(
        model=model,
        tokenizer=tokenizer,
        pool=block_pool,  # Fixed: parameter is 'pool', not 'block_pool'
        spec=model_spec,
        cache_adapter=mlx_adapter,
    )
    logger.info(
        "batch_engine_initialized",
        max_batch_size=settings.mlx.max_batch_size,
        prefill_step_size=settings.mlx.prefill_step_size
    )

    # Store in app state
    app.state.semantic = AppState()
    app.state.semantic.block_pool = block_pool
    app.state.semantic.batch_engine = batch_engine
    app.state.semantic.cache_store = cache_store
    app.state.semantic.mlx_adapter = mlx_adapter
    app.state.semantic.cache_adapter = cache_adapter

    logger.info("server_ready")

    # Set shutdown flag to false (ready to serve)
    app.state.shutting_down = False

    yield

    # Shutdown: cleanup resources
    logger.info("server_shutting_down")

    # Set shutdown flag IMMEDIATELY (health checks will return 503)
    app.state.shutting_down = True

    # Graceful shutdown: drain pending requests and persist caches
    logger.info("draining_requests")
    if batch_engine:
        try:
            drained = await batch_engine.drain(timeout_seconds=30)
            logger.info("requests_drained", count=drained)
        except Exception as e:
            logger.error("drain_error", error=str(e), exc_info=True)

    logger.info("persisting_caches")
    # Save all hot agent caches to disk
    if cache_store:
        try:
            saved_count = cache_store.evict_all_to_disk()
            logger.info("caches_persisted", count=saved_count)
        except Exception as e:
            logger.error("persist_error", error=str(e), exc_info=True)

    logger.info("server_shutdown_complete")


def create_app() -> FastAPI:
    """Create and configure FastAPI application.

    Returns:
        Configured FastAPI application instance.

    Example:
        >>> app = create_app()
        >>> # Use with uvicorn: uvicorn.run(app, host="0.0.0.0", port=8000)
    """
    settings = get_settings()

    # Initialize structured logging (BEFORE creating FastAPI app)
    json_output = settings.server.log_level == "PRODUCTION"
    configure_logging(
        log_level=settings.server.log_level,
        json_output=json_output
    )

    # Get structlog logger
    logger = structlog.get_logger(__name__)
    logger.info("creating_fastapi_app", version="0.1.0")

    app = FastAPI(
        title="Semantic Caching API",
        description="Multi-protocol API for semantic KV cache management",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Request ID middleware (FIRST - executed last, but sets up context)
    from semantic.adapters.inbound.request_id_middleware import RequestIDMiddleware

    app.add_middleware(RequestIDMiddleware)
    logger.info("middleware_registered", middleware="RequestIDMiddleware")

    # Request logging middleware (skip health checks to avoid spam)
    from semantic.adapters.inbound.request_logging_middleware import RequestLoggingMiddleware

    app.add_middleware(
        RequestLoggingMiddleware,
        skip_paths={"/health/live", "/health/ready", "/health/startup"}
    )
    logger.info("middleware_registered", middleware="RequestLoggingMiddleware")

    # Metrics middleware (skip /metrics endpoint itself)
    from semantic.adapters.inbound.metrics_middleware import RequestMetricsMiddleware

    app.add_middleware(
        RequestMetricsMiddleware,
        skip_paths={"/metrics"}
    )
    logger.info("middleware_registered", middleware="RequestMetricsMiddleware")

    # CORS middleware (production-ready configuration)
    cors_origins_str = settings.server.cors_origins
    # Parse comma-separated origins
    if cors_origins_str == "*":
        cors_origins = ["*"]
    else:
        cors_origins = [origin.strip() for origin in cors_origins_str.split(",")]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Authentication middleware
    from semantic.adapters.inbound.auth_middleware import AuthenticationMiddleware

    app.add_middleware(AuthenticationMiddleware)
    logger.info("middleware_registered", middleware="AuthenticationMiddleware")

    # Rate limiting middleware
    from semantic.adapters.inbound.rate_limiter import RateLimiter

    app.add_middleware(
        RateLimiter,
        requests_per_minute_per_agent=settings.server.rate_limit_per_agent,
        requests_per_minute_global=settings.server.rate_limit_global,
    )
    logger.info("middleware_registered", middleware="RateLimiter")

    # 3-Tier Health Check Endpoints (Kubernetes-compatible)

    @app.get("/health/live")
    async def health_live():
        """Liveness probe - process is alive.

        Returns 200 if server process is running.
        Kubernetes liveness probe should use this endpoint.

        Returns:
            Status dict indicating the process is alive.

        Example:
            $ curl http://localhost:8000/health/live
            {"status":"alive"}
        """
        return {"status": "alive"}

    @app.get("/health/ready")
    async def health_ready(response: Response):
        """Readiness probe - ready to accept requests.

        Returns:
            200 if ready, 503 if not ready

        Not ready states:
        - Pool utilization >90%
        - Server is shutting down
        - Pool not initialized

        Kubernetes readiness probe should use this endpoint.

        Also updates Prometheus metrics:
        - pool_utilization_ratio gauge
        - agents_active gauge

        Example:
            $ curl http://localhost:8000/health/ready
            {"status":"ready","pool_utilization":15.2}
        """
        from semantic.adapters.inbound.metrics import agents_active, pool_utilization_ratio

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

        # Update pool utilization metric
        pool_utilization_ratio.set(utilization)

        # Update active agents metric
        cache_store = app.state.semantic.cache_store if hasattr(app.state, "semantic") else None
        if cache_store and hasattr(cache_store, '_hot_agents'):
            hot_count = len(cache_store._hot_agents)
            agents_active.set(hot_count)

        if utilization > POOL_UTILIZATION_THRESHOLD:
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
            return {
                "status": "not_ready",
                "reason": "pool_near_exhaustion",
                "pool_utilization": round(utilization * 100, 1),
            }

        # Ready
        return {
            "status": "ready",
            "pool_utilization": round(utilization * 100, 1),
        }

    @app.get("/health/startup")
    async def health_startup(response: Response):
        """Startup probe - initialization complete.

        Returns:
            200 if startup complete, 503 if still initializing

        Kubernetes startup probe should use this endpoint.

        Example:
            $ curl http://localhost:8000/health/startup
            {"status":"started"}
        """
        # Check if model loaded (semantic state initialized)
        if not hasattr(app.state, "semantic") or not app.state.semantic.batch_engine:
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
            return {"status": "starting", "reason": "model_loading"}

        return {"status": "started"}

    # Metrics endpoint (Prometheus)
    @app.get("/metrics")
    async def metrics():
        """Prometheus metrics endpoint.

        Returns metrics in Prometheus exposition format.

        Returns:
            Plain text response with Prometheus metrics.

        Example:
            $ curl http://localhost:8000/metrics
            # HELP semantic_request_total Total number of HTTP requests
            # TYPE semantic_request_total counter
            semantic_request_total{method="GET",path="/",status_code="200"} 1.0
        """
        from prometheus_client import generate_latest

        from semantic.adapters.inbound.metrics import registry

        return Response(
            content=generate_latest(registry),
            media_type="text/plain; version=0.0.4"
        )

    # Root endpoint
    @app.get("/", status_code=status.HTTP_200_OK)
    async def root():
        """Root endpoint with API info.

        Returns:
            API info dict.
        """
        return {
            "name": "Semantic Caching API",
            "version": "0.1.0",
            "endpoints": {
                "health": "/health",
                "metrics": "/metrics",
                "anthropic": "/v1/messages",
                "openai": "/v1/chat/completions",
                "agents": "/v1/agents",
            },
        }

    # Error handlers
    @app.exception_handler(SemanticError)
    async def semantic_error_handler(request: Request, exc: SemanticError):
        """Handle domain errors."""
        logger.error("domain_error", error_type=exc.__class__.__name__, message=str(exc), exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"error": exc.__class__.__name__, "message": str(exc)},
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, exc: RequestValidationError):
        """Handle request validation errors."""
        logger.warning("validation_error", error=str(exc))
        # Convert errors to JSON-serializable format
        errors = []
        for error in exc.errors():
            errors.append({
                "loc": error["loc"],
                "msg": error["msg"],
                "type": error["type"],
            })
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"error": "ValidationError", "details": errors},
        )

    @app.exception_handler(Exception)
    async def general_error_handler(request: Request, exc: Exception):
        """Handle unexpected errors."""
        logger.error("unexpected_error", error_type=type(exc).__name__, message=str(exc), exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": "InternalServerError", "message": "An unexpected error occurred"},
        )

    # Register route handlers
    from semantic.adapters.inbound.anthropic_adapter import router as anthropic_router
    from semantic.adapters.inbound.direct_agent_adapter import router as direct_router
    from semantic.adapters.inbound.openai_adapter import router as openai_router

    app.include_router(anthropic_router)
    logger.info("routes_registered", router="anthropic", path="/v1/messages")

    app.include_router(openai_router)
    logger.info("routes_registered", router="openai", path="/v1/chat/completions")

    app.include_router(direct_router)
    logger.info("routes_registered", router="direct_agent", path="/v1/agents")

    logger.info("fastapi_app_created", log_level=settings.server.log_level)
    return app
