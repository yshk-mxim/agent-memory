"""CLI entrypoint for semantic caching server.

Usage:
    python -m semantic.entrypoints.cli serve
    python -m semantic.entrypoints.cli serve --host 0.0.0.0 --port 8080
"""

import logging
import sys

import typer
import uvicorn

from semantic import __version__
from semantic.adapters.config.settings import get_settings
from semantic.entrypoints.api_server import create_app

app = typer.Typer(
    name="semantic",
    help="Semantic caching server for MLX inference",
    add_completion=False,
)


def setup_logging(log_level: str) -> None:
    """Configure structured logging.

    Args:
        log_level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )


@app.command()
def serve(
    host: str = typer.Option(
        None,
        "--host",
        "-h",
        help="Server bind address (default: from settings)",
    ),
    port: int = typer.Option(
        None,
        "--port",
        "-p",
        help="Server port (default: from settings)",
    ),
    model: str = typer.Option(
        None,
        "--model",
        "-m",
        help="Model ID (default: from settings)",
    ),
    workers: int = typer.Option(
        None,
        "--workers",
        "-w",
        help="Number of worker processes (default: from settings)",
    ),
    log_level: str = typer.Option(
        None,
        "--log-level",
        "-l",
        help="Log level (default: from settings)",
    ),
    reload: bool = typer.Option(
        False,
        "--reload",
        help="Enable auto-reload on code changes (dev only)",
    ),
) -> None:
    """Start the semantic caching server.

    Example:
        $ semantic serve
        $ semantic serve --host 0.0.0.0 --port 8080
        $ semantic serve --model mlx-community/gpt-oss-20b-MXFP4-Q4
        $ semantic serve --reload  # Dev mode
    """
    settings = get_settings()

    # Use CLI args or fall back to settings
    final_host = host or settings.server.host
    final_port = port or settings.server.port
    final_model = model or settings.mlx.model_id
    final_workers = workers or settings.server.workers
    final_log_level = log_level or settings.server.log_level

    # Override model in settings if provided
    if model:
        settings.mlx.model_id = model

    # Setup logging
    setup_logging(final_log_level)
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("Semantic Caching Server")
    logger.info("=" * 60)
    logger.info(f"Host: {final_host}")
    logger.info(f"Port: {final_port}")
    logger.info(f"Workers: {final_workers}")
    logger.info(f"Log level: {final_log_level}")
    logger.info(f"Reload: {reload}")
    logger.info(f"Model: {final_model}")
    logger.info(f"Cache budget: {settings.mlx.cache_budget_mb} MB")
    logger.info(f"Max agents: {settings.agent.max_agents_in_memory}")
    logger.info("=" * 60)

    # Create FastAPI app
    fastapi_app = create_app()

    # Run server with uvicorn
    uvicorn.run(
        fastapi_app,
        host=final_host,
        port=final_port,
        workers=final_workers if not reload else 1,  # Reload requires single worker
        log_level=final_log_level.lower(),
        reload=reload,
        access_log=True,
    )


@app.command()
def version() -> None:
    """Show version information."""
    typer.echo(f"Semantic Caching Server v{__version__}")
    typer.echo("Sprint 8: Production Release - Tool Calling + Multi-Model Support")


@app.command()
def config() -> None:
    """Show current configuration."""
    settings = get_settings()

    typer.echo("=" * 60)
    typer.echo("Semantic Caching Server - Configuration")
    typer.echo("=" * 60)
    typer.echo()
    typer.echo("[Server]")
    typer.echo(f"  Host: {settings.server.host}")
    typer.echo(f"  Port: {settings.server.port}")
    typer.echo(f"  Workers: {settings.server.workers}")
    typer.echo(f"  Log level: {settings.server.log_level}")
    typer.echo(f"  CORS origins: {settings.server.cors_origins}")
    typer.echo()
    typer.echo("[MLX]")
    typer.echo(f"  Model ID: {settings.mlx.model_id}")
    typer.echo(f"  Cache budget: {settings.mlx.cache_budget_mb} MB")
    typer.echo(f"  Max batch size: {settings.mlx.max_batch_size}")
    typer.echo(f"  Prefill step size: {settings.mlx.prefill_step_size}")
    typer.echo()
    typer.echo("[Agent]")
    typer.echo(f"  Cache dir: {settings.agent.cache_dir}")
    typer.echo(f"  Max agents in memory: {settings.agent.max_agents_in_memory}")
    typer.echo()
    typer.echo("[Rate Limiting]")
    typer.echo(f"  Per agent: {settings.server.rate_limit_per_agent}/min")
    typer.echo(f"  Global: {settings.server.rate_limit_global}/min")
    typer.echo("=" * 60)


def main() -> None:
    """Main CLI entry point."""
    app()


if __name__ == "__main__":
    main()
