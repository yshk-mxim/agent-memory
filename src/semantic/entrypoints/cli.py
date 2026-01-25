"""CLI entrypoint for semantic caching server.

Usage:
    python -m semantic.entrypoints.cli serve
    python -m semantic.entrypoints.cli serve --host 0.0.0.0 --port 8080
"""

import logging
import sys

import typer
import uvicorn

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
        $ python -m semantic.entrypoints.cli serve
        $ python -m semantic.entrypoints.cli serve --host 0.0.0.0 --port 8080
        $ python -m semantic.entrypoints.cli serve --reload  # Dev mode
    """
    settings = get_settings()

    # Use CLI args or fall back to settings
    final_host = host or settings.server.host
    final_port = port or settings.server.port
    final_workers = workers or settings.server.workers
    final_log_level = log_level or settings.server.log_level

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
    logger.info(f"Model: {settings.mlx.model_id}")
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
    typer.echo("Semantic Caching Server v0.1.0")
    typer.echo("Sprint 4: Multi-Protocol API Adapter")


def main() -> None:
    """Main CLI entry point."""
    app()


if __name__ == "__main__":
    main()
