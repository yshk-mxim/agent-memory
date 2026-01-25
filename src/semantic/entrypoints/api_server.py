"""FastAPI application factory and server setup.

This module provides the main FastAPI application with dependency injection,
middleware, error handlers, and route registration.
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from semantic.adapters.config.settings import get_settings
from semantic.adapters.outbound.mlx_cache_adapter import MLXCacheAdapter
from semantic.adapters.outbound.mlx_spec_extractor import get_extractor
from semantic.adapters.outbound.safetensors_cache_adapter import SafetensorsCacheAdapter
from semantic.application.agent_cache_store import AgentCacheStore, ModelTag
from semantic.application.batch_engine import BlockPoolBatchEngine
from semantic.domain.errors import SemanticError
from semantic.domain.services import BlockPool
from semantic.domain.value_objects import ModelCacheSpec

logger = logging.getLogger(__name__)


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
    logger.info("🚀 Starting semantic caching server...")
    settings = get_settings()

    # Load MLX model and extract spec
    logger.info(f"Loading model: {settings.mlx.model_id}")
    from mlx_lm import load

    model, tokenizer = load(
        settings.mlx.model_id,
        tokenizer_config={"trust_remote_code": True},
    )

    # Extract model cache spec
    spec_extractor = get_extractor()
    model_spec: ModelCacheSpec = spec_extractor.extract_spec(model)
    logger.info(
        f"Model spec: {model_spec.n_layers} layers, "
        f"{model_spec.n_kv_heads} KV heads, "
        f"{model_spec.head_dim} head dim"
    )

    # Calculate block budget
    bytes_per_block = model_spec.bytes_per_block_per_layer()
    total_blocks = (settings.mlx.cache_budget_mb * 1024 * 1024) // bytes_per_block
    logger.info(
        f"Block budget: {total_blocks} blocks "
        f"({bytes_per_block / 1024 / 1024:.2f} MB per block)"
    )

    # Initialize BlockPool
    block_pool = BlockPool(spec=model_spec, total_blocks=total_blocks)
    logger.info(f"BlockPool initialized: {total_blocks} blocks available")

    # Initialize MLX adapter
    mlx_adapter = MLXCacheAdapter(model=model, spec=model_spec)

    # Initialize safetensors persistence adapter
    cache_dir = Path(settings.agent.cache_dir).expanduser()
    cache_adapter = SafetensorsCacheAdapter(cache_dir=cache_dir)
    logger.info(f"Cache persistence: {cache_dir}")

    # Initialize AgentCacheStore
    model_tag = ModelTag.from_spec(settings.mlx.model_id, model_spec)
    cache_store = AgentCacheStore(
        cache_dir=cache_dir,
        max_hot_agents=settings.agent.max_agents_in_memory,
        model_tag=model_tag,
        cache_adapter=cache_adapter,
    )
    logger.info(
        f"AgentCacheStore initialized: max {settings.agent.max_agents_in_memory} hot agents"
    )

    # Initialize BatchEngine
    batch_engine = BlockPoolBatchEngine(
        model=model,
        tokenizer=tokenizer,
        cache_adapter=mlx_adapter,
        block_pool=block_pool,
        batch_window_ms=settings.agent.batch_window_ms,
        max_batch_size=settings.mlx.max_batch_size,
        prefill_step_size=settings.mlx.prefill_step_size,
        spec=model_spec,
    )
    logger.info(
        f"BatchEngine initialized: max_batch_size={settings.mlx.max_batch_size}, "
        f"prefill_step_size={settings.mlx.prefill_step_size}"
    )

    # Store in app state
    app.state.semantic = AppState()
    app.state.semantic.block_pool = block_pool
    app.state.semantic.batch_engine = batch_engine
    app.state.semantic.cache_store = cache_store
    app.state.semantic.mlx_adapter = mlx_adapter
    app.state.semantic.cache_adapter = cache_adapter

    logger.info("✅ Server ready to accept requests")

    yield

    # Shutdown: cleanup resources
    logger.info("🛑 Shutting down server...")
    # TODO: Drain pending requests, persist caches
    logger.info("✅ Server shutdown complete")


def create_app() -> FastAPI:
    """Create and configure FastAPI application.

    Returns:
        Configured FastAPI application instance.

    Example:
        >>> app = create_app()
        >>> # Use with uvicorn: uvicorn.run(app, host="0.0.0.0", port=8000)
    """
    settings = get_settings()

    app = FastAPI(
        title="Semantic Caching API",
        description="Multi-protocol API for semantic KV cache management",
        version="0.1.0",
        lifespan=lifespan,
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # TODO: Configure in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Authentication middleware
    from semantic.adapters.inbound.auth_middleware import AuthenticationMiddleware

    app.add_middleware(AuthenticationMiddleware)
    logger.info("Authentication middleware enabled")

    # Health check endpoint
    @app.get("/health", status_code=status.HTTP_200_OK)
    async def health_check():
        """Health check endpoint.

        Returns:
            Status dict with "ok" status.

        Example:
            $ curl http://localhost:8000/health
            {"status":"ok"}
        """
        return {"status": "ok"}

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
                "anthropic": "/v1/messages",
                "openai": "/v1/chat/completions",
                "agents": "/v1/agents",
            },
        }

    # Error handlers
    @app.exception_handler(SemanticError)
    async def semantic_error_handler(request: Request, exc: SemanticError):
        """Handle domain errors."""
        logger.error(f"Domain error: {exc}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"error": exc.__class__.__name__, "message": str(exc)},
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, exc: RequestValidationError):
        """Handle request validation errors."""
        logger.warning(f"Validation error: {exc}")
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
        logger.error(f"Unexpected error: {exc}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": "InternalServerError", "message": "An unexpected error occurred"},
        )

    # Register route handlers
    from semantic.adapters.inbound.anthropic_adapter import router as anthropic_router
    from semantic.adapters.inbound.direct_agent_adapter import router as direct_router
    from semantic.adapters.inbound.openai_adapter import router as openai_router

    app.include_router(anthropic_router)
    logger.info("Registered Anthropic Messages API routes (/v1/messages)")

    app.include_router(openai_router)
    logger.info("Registered OpenAI Chat Completions API routes (/v1/chat/completions)")

    app.include_router(direct_router)
    logger.info("Registered Direct Agent API routes (/v1/agents)")

    logger.info(f"FastAPI application created (log_level={settings.server.log_level})")
    return app
