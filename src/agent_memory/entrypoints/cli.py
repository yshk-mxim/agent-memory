# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Yakov Shkolnikov and contributors
"""CLI entrypoint for agent-memory server.

Usage:
    python -m agent_memory.entrypoints.cli serve
    python -m agent_memory.entrypoints.cli serve --host 0.0.0.0 --port 8080
"""

import faulthandler
import logging
import signal
import sys
from typing import Any

# Print Python traceback on SIGSEGV/SIGABRT/SIGBUS (Metal crashes).
faulthandler.enable()

import typer
import uvicorn

from agent_memory import __version__
from agent_memory.adapters.config.settings import get_settings
from agent_memory.entrypoints.api_server import create_app

app = typer.Typer(
    name="agent-memory",
    help="Persistent KV cache server for multi-agent LLM inference on Apple Silicon",
    add_completion=False,
)


def setup_logging(log_level: str) -> None:
    """Configure structured logging.

    Args:
        log_level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    # Validate log level to avoid AttributeError
    valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
    normalized_level = log_level.upper()
    if normalized_level not in valid_levels:
        logging.warning(f"Invalid log level '{log_level}', defaulting to INFO")
        normalized_level = "INFO"

    logging.basicConfig(
        level=getattr(logging, normalized_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )


def setup_signal_handlers(logger: Any) -> None:
    """Setup signal handlers for graceful shutdown.

    Args:
        logger: Logger instance for shutdown messages
    """

    def handle_shutdown(signum: int, frame: Any) -> None:
        """Handle shutdown signals gracefully."""
        sig_name = signal.Signals(signum).name
        logger.info(f"Received {sig_name}, initiating graceful shutdown...")
        # Uvicorn handles SIGINT/SIGTERM internally, but we log for visibility
        # The actual cleanup happens in FastAPI's lifespan context manager
        sys.exit(0)

    # Register handlers for common shutdown signals
    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    # SIGHUP for config reload (Unix only)
    if hasattr(signal, "SIGHUP"):

        def handle_reload(signum: int, frame: Any) -> None:
            logger.info("Received SIGHUP - config reload not implemented yet")

        signal.signal(signal.SIGHUP, handle_reload)


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
    """Start the agent-memory server.

    Example:
        $ agent-memory serve
        $ agent-memory serve --host 0.0.0.0 --port 8080
        $ agent-memory serve --model mlx-community/gpt-oss-20b-MXFP4-Q4
        $ agent-memory serve --reload  # Dev mode
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

    # Setup signal handlers for graceful shutdown
    setup_signal_handlers(logger)

    logger.info("=" * 60)
    logger.info("agent-memory")
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

    # Enable faster GPU↔CPU synchronization for Metal.
    # Reduces per-token sync fence overhead during decode.
    import os
    os.environ.setdefault("MLX_METAL_FAST_SYNCH", "1")

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
    typer.echo(f"agent-memory v{__version__}")
    typer.echo("MLX inference server with persistent KV cache for multi-agent workflows")


@app.command()
def tune(
    quick: bool = typer.Option(
        False,
        "--quick",
        help="Quick mode: fewer scenarios, faster results (~5 min)",
    ),
    output: str = typer.Option(
        None,
        "--output",
        "-o",
        help="Output TOML path (default: config/models/tuned.toml)",
    ),
    port: int = typer.Option(
        8399,
        "--port",
        "-p",
        help="Server port for benchmarking (default: 8399)",
    ),
) -> None:
    """Auto-tune inference parameters for your hardware.

    Runs profiling experiments to find optimal prefill step size,
    batch size, and other parameters. Writes a per-model TOML
    config profile.

    Example:
        $ agent-memory tune --quick
        $ agent-memory tune --output config/models/my-machine.toml
    """
    from agent_memory.entrypoints.tune import _run_tune

    _run_tune(quick=quick, output=output, port=port)


@app.command()
def config() -> None:
    """Show current configuration."""
    settings = get_settings()

    typer.echo("=" * 60)
    typer.echo("agent-memory - Configuration")
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
